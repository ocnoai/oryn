import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from oryn_engine import isnet_tasks as binding
from oryn_engine import shared_tasks

FIXTURES = Path(__file__).parent / "fixtures/isnet"
TASK = "image-segmentation"


def metadata(name="config.json"):
    return json.loads((FIXTURES / name).read_text())


class TinyNN:
    def __getattr__(self, name):
        return getattr(torch.nn, name)

    @staticmethod
    def Conv2d(in_channels, out_channels, *args, **kwargs):
        return torch.nn.Conv2d(
            in_channels // 16 if in_channels >= 16 else in_channels,
            out_channels // 16 if out_channels >= 16 else out_channels,
            *args,
            **kwargs,
        )

    @staticmethod
    def BatchNorm2d(channels, *args, **kwargs):
        return torch.nn.BatchNorm2d(channels // 16 if channels >= 16 else channels, *args, **kwargs)


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    from oryn_engine.native_isnet import model as native

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    # Test-only channel shrinking keeps every native block and all six heads.
    # Production has no width setting and requires the complete original graph.
    monkeypatch.setattr(native, "nn", TinyNN())
    monkeypatch.setattr(binding, "IMAGE_SIZE", 64)
    vectors = metadata("reference.json")
    torch.manual_seed(vectors["seed"])
    reference = binding.native_model().eval()
    state = reference.state_dict()
    assert {
        k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in state.items()
    } == vectors["stateSha256"]
    save_file(state, tmp_path / "model.safetensors")
    for name in ["config.json", "preprocessor_config.json"]:
        (tmp_path / name).write_bytes((FIXTURES / name).read_bytes())
    (tmp_path / "MyPipe.py").write_text("raise AssertionError('repository pipeline is forbidden')")
    yield tmp_path, binding.normalized_config(metadata(), TASK), reference, vectors
    (tmp_path / "model.safetensors").unlink(missing_ok=True)
    torch.set_num_threads(previous)


def fixture_image(data):
    c = {"RGB": 3, "L": 1, "RGBA": 4}[data["mode"]]
    w, h = data["width"], data["height"]
    array = ((np.arange(w * h * c) * 19 + 13 + data["offset"] * 29) % 256).astype(np.uint8)
    return Image.fromarray(
        array.reshape((h, w, c) if c > 1 else (h, w)), mode=data["mode"]
    ).convert("RGB")


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("case", range(4))
def test_all_six_outputs_and_declared_pipeline_match_original(tiny, backend, case):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    root, config, reference, vectors = tiny
    model, processor, selected = shared_tasks.load_task(root, TASK, backend)
    assert processor is None and selected == config
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name].cpu(), expected, atol=0, rtol=0)
    data = vectors["cases"][case]
    image = fixture_image(data)
    pixels = binding.preprocess(image)
    assert hashlib.sha256(pixels.numpy().tobytes()).hexdigest() == data["pixelSha256"]
    with torch.inference_mode():
        outputs, features = model(pixels.to(backend))
        mask = binding.segmentation(model, image, backend)
    assert len(outputs) == len(features) == 6
    for actual, expected in zip(outputs, data["outputs"], strict=True):
        np.testing.assert_allclose(actual.float().cpu().flatten(), expected, atol=2e-6, rtol=2e-5)
    assert (
        hashlib.sha256(mask.numpy().astype(np.uint16).tobytes()).hexdigest() == data["maskSha256"]
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "conv_in.",
        "stage1.",
        "stage6.",
        "stage1d.",
        "stage5d.",
        "side1.",
        "side2.",
        "side3.",
        "side4.",
        "side5.",
        "side6.",
    ],
)
def test_every_encoder_decoder_and_side_head_is_required(tiny, prefix):
    root, config, _, _ = tiny
    state = load_file(root / "model.safetensors")
    del state[next(k for k in state if k.startswith(prefix))]
    save_file(state, root / "model.safetensors")
    with pytest.raises(ValueError, match="ISNET_WEIGHTS_INCOMPLETE"):
        binding.load(root, config)


@pytest.mark.parametrize(
    "damage", ["extra", "shape", "nan", "overflow", "integer", "counter", "variance"]
)
def test_invalid_numeric_state_rejected(tiny, damage):
    root, config, _, _ = tiny
    state = load_file(root / "model.safetensors")
    key = "side1.weight"
    if damage == "extra":
        state["extra"] = torch.zeros(1)
    elif damage == "shape":
        state[key] = state[key][:, :-1]
    elif damage == "nan":
        state[key].flatten()[0] = float("nan")
    elif damage == "overflow":
        state[key] = state[key].double()
        state[key].flatten()[0] = 1e300
    elif damage == "integer":
        state[key] = state[key].long()
    elif damage == "counter":
        state[next(k for k in state if k.endswith("num_batches_tracked"))].fill_(-1)
    else:
        state[next(k for k in state if k.endswith("running_var"))].fill_(-1)
    save_file(state, root / "model.safetensors")
    with pytest.raises(ValueError, match="ISNET_"):
        binding.load(root, config)


@pytest.mark.parametrize(
    "change",
    [
        {"in_ch": 4},
        {"out_ch": True},
        {"out_ch": 2},
        {"model_type": "segformer"},
        {"auto_map": {"AutoModel": "remote.Model"}},
        {"custom_pipelines": {"image-segmentation": {"impl": "remote.Pipe"}}},
        {"id2label": {"0": "cat", "1": "dog"}},
        {"torch_dtype": "int8"},
        {"backbone": "remote"},
    ],
)
def test_closed_config_rejects_graph_or_pipeline_overrides(change):
    with pytest.raises(ValueError):
        binding.normalized_config(metadata() | change, TASK)


@pytest.mark.parametrize(
    "change",
    [
        {"size": {"height": 512, "width": 512}},
        {"do_resize": False},
        {"do_rescale": False},
        {"do_normalize": False},
        {"do_pad": True},
        {"resample": 3},
        {"image_mean": [0, 0, 0]},
        {"image_std": [1, 0, 1]},
        {"feature_extractor_type": "RemoteProcessor"},
        {"rescale_factor": 1},
    ],
)
def test_declared_processor_semantics_are_closed(change):
    with pytest.raises(ValueError):
        binding.validate_processor(metadata("preprocessor_config.json") | change)


def test_original_whole_graph_shapes():
    shapes = {k: tuple(v) for k, v in metadata("original-shapes.json").items()}
    model = binding.validate_weight_shapes(binding.normalized_config(metadata(), TASK), shapes)
    assert len(model.state_dict()) == 798


@pytest.mark.parametrize("damage", [None, "weights", "processor", "bad-processor", "remote"])
def test_original_discovery_uses_shared_foreground_contract(tmp_path, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore, ModelStoreError
    from oryn_engine.runtime_packs.execution import PackRuntime

    class Hub(FakeHub):
        def read_json(self, repository, revision, name):
            result = metadata(name)
            return result | {"do_pad": True} if damage == "bad-processor" else result

    config = metadata()
    if damage == "remote":
        config["auto_map"] = {"AutoModel": "remote.Model"}
    files = [("config.json", 100, "a" * 64)]
    if damage != "weights":
        files.append(("model.safetensors", 100, "b" * 64))
    if damage != "processor":
        files.append(("preprocessor_config.json", 100, "c" * 64))
    hub = Hub(config=config, files=files, pipeline_tag=TASK)
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    request = {"provider": "huggingface", "repository": "test/isnet", "revision": "main"}
    if damage == "remote":
        with pytest.raises(ModelStoreError):
            store.inspect(request)
    else:
        result = store.inspect(request)
        assert result["executable"] == (damage is None), result
        if damage is None:
            assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
    assert not hub.downloaded


def test_discovery_is_metadata_only():
    script = f"""
import sys,json
from pathlib import Path
from oryn_engine.generated_capabilities import normalized_transformers_config
from oryn_engine.model_layouts import assess_layout
root=Path({str(FIXTURES)!r})
config=normalized_transformers_config(json.loads((root/'config.json').read_text()),'image-segmentation')
layout=assess_layout(['config.json','model.safetensors','preprocessor_config.json','MyPipe.py'],config,'image-segmentation')
assert not layout.artifact_reasons and not layout.runtime_reasons
assert 'torch' not in sys.modules and 'oryn_engine.native_isnet.model' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_shared_artifact_native_mask_and_original_coordinates(tiny):
    root, _, _, vectors = tiny
    case = vectors["cases"][0]
    source, output = root / "source.png", root / "mask.png"
    fixture_image(case).save(source)
    result = shared_tasks.execute_task(
        root,
        TASK,
        {"imageArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa"},
        "cpu",
        source,
        output,
        lambda *_: None,
    )
    assert (result["width"], result["height"]) == (case["width"], case["height"])
    assert {s["label"] for s in result["metadata"]["segments"]} == {"background", "foreground"}
    assert (
        sum(s["pixelCount"] for s in result["metadata"]["segments"])
        == case["width"] * case["height"]
    )
    with Image.open(output) as mask:
        assert (
            hashlib.sha256(np.asarray(mask).astype(np.uint16).tobytes()).hexdigest()
            == case["maskSha256"]
        )


@pytest.mark.parametrize("damage", ["nan", "infinite", "shape", "range", "constant"])
def test_invalid_or_undefined_native_output_cannot_publish(tiny, damage):
    output = torch.linspace(0, 1, 64 * 64).reshape(1, 1, 64, 64)
    if damage == "shape":
        output = output[:, :, :-1]
    elif damage == "range":
        output[0, 0, 0, 0] = -0.1
    elif damage == "constant":
        output.fill_(0.5)
    else:
        output[0, 0, 0, 0] = float("nan" if damage == "nan" else "inf")
    with pytest.raises(ValueError, match="ISNET_OUTPUT"):
        binding.segmentation(lambda _: ([output], []), Image.new("RGB", (17, 19)), "cpu")


def test_apache_source_and_provenance_are_preserved():
    from oryn_engine import native_isnet

    root = Path(native_isnet.__file__).parent
    for name, digest in json.loads((root / "provenance.json").read_text())["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    assert "Apache License" in (root / "LICENSE").read_text()
    assert "No repository model code" in (root / "NOTICE.md").read_text()
