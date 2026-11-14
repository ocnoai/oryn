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

from oryn_engine import birefnet_tasks as binding
from oryn_engine import shared_tasks

FIXTURES = Path(__file__).parent / "fixtures/birefnet"
TASK = "image-segmentation"


def original(name="BiRefNet"):
    return json.loads((FIXTURES / (name + "-config.json")).read_text())


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    from oryn_engine.native_birefnet import model as native

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    parent = native.Config

    class TinyConfig(parent):
        def __init__(self, *args):
            super().__init__()
            self.bb = "swin_v1_t"
            self.lateral_channels_in_collection = [64, 32, 16, 8]
            self.cxt = [8, 16, 32]

    monkeypatch.setattr(native, "Config", TinyConfig)
    monkeypatch.setattr(
        native,
        "swin_v1_t",
        lambda: native.SwinTransformer(
            embed_dim=4, depths=[2, 2, 2, 2], num_heads=[1, 2, 4, 8], window_size=4
        ),
    )

    # Test-only shrinking of the fixed native graph; production selects only the
    # two original shapes, tested separately against complete original headers.
    def variant(shapes):
        if shapes.get("bb.patch_embed.proj.weight") != (4, 3, 4, 4):
            raise ValueError("BIREFNET_BACKBONE")
        return "swin_v1_t"

    monkeypatch.setattr(binding, "backbone_variant", variant)
    monkeypatch.setattr(binding, "IMAGE_SIZE", 64)
    vectors = json.loads((FIXTURES / "reference.json").read_text())
    torch.manual_seed(vectors["seed"])
    reference = binding.native_model("swin_v1_t").eval()
    with torch.no_grad():
        for name, value in reference.named_parameters():
            if ".offset_conv." in name or ".modulator_conv." in name:
                value.copy_(torch.randn_like(value) * 0.01)
    with torch.no_grad():
        reference.decoder.conv_out1[0].bias.sub_(0.14)
    state = reference.state_dict()
    assert {
        k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in state.items()
    } == vectors["stateSha256"]
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(original()))
    (tmp_path / "birefnet.py").write_text("raise AssertionError('repository code must stay inert')")
    yield tmp_path, binding.normalized_config(original(), TASK), reference, vectors
    (tmp_path / "model.safetensors").unlink(missing_ok=True)
    torch.set_num_threads(previous)


def fixture_image(data):
    channels = {"RGB": 3, "L": 1, "RGBA": 4}[data["mode"]]
    w, h = data["width"], data["height"]
    pixels = ((np.arange(w * h * channels) * 13 + 27 + data["offset"] * 31) % 256).astype(np.uint8)
    return Image.fromarray(
        pixels.reshape((h, w, channels) if channels > 1 else (h, w)), mode=data["mode"]
    ).convert("RGB")


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("case", range(4))
def test_complete_graph_matches_pinned_original_numerics(tiny, backend, case):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    root, config, reference, vectors = tiny
    loaded, processor, selected = shared_tasks.load_task(root, TASK, backend)
    assert processor is None and selected == config
    for key, expected in reference.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key].cpu(), expected, atol=0, rtol=0)
    data = vectors["cases"][case]
    image = fixture_image(data)
    pixels = binding.preprocess(image)
    assert hashlib.sha256(pixels.numpy().tobytes()).hexdigest() == data["pixelSha256"]
    with torch.inference_mode():
        logits = loaded(pixels[None].to(backend))[-1]
        mask = binding.segmentation(loaded, image, backend)
    np.testing.assert_allclose(logits.float().cpu().flatten(), data["logits"], atol=3e-5, rtol=3e-4)
    assert (
        hashlib.sha256(mask.numpy().astype(np.uint16).tobytes()).hexdigest() == data["maskSha256"]
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "bb.layers.3.",
        "squeeze_module.",
        "decoder.ipt_blk5.",
        "decoder.conv_out1.",
        "decoder.conv_ms_spvn_4.",
        "decoder.gdt_convs_pred_4.",
        "decoder.gdt_convs_attn_4.",
        "decoder.decoder_block1.dec_att.",
    ],
)
def test_every_native_branch_is_required(tiny, prefix):
    root, config, _, _ = tiny
    state = load_file(root / "model.safetensors")
    del state[next(k for k in state if k.startswith(prefix))]
    save_file(state, root / "model.safetensors")
    with pytest.raises(ValueError, match="BIREFNET_WEIGHTS_INCOMPLETE"):
        binding.load(root, config)


@pytest.mark.parametrize(
    "damage", ["extra", "shape", "nan", "overflow", "integer", "counter", "variance", "position"]
)
def test_corrupt_graph_and_native_statistics_rejected(tiny, damage):
    root, config, _, _ = tiny
    state = load_file(root / "model.safetensors")
    key = "decoder.conv_out1.0.weight"
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
    elif damage == "variance":
        state[next(k for k in state if k.endswith("running_var"))].fill_(-1)
    else:
        state[next(k for k in state if k.endswith("relative_position_index"))].flatten()[0] += 1
    save_file(state, root / "model.safetensors")
    with pytest.raises(ValueError, match="BIREFNET_"):
        binding.load(root, config)


@pytest.mark.parametrize(
    "change",
    [
        {"bb_pretrained": True},
        {"bb_pretrained": 0},
        {"bb": "remote"},
        {"auto_map": {"AutoModel": "remote.Model"}},
        {"custom_pipelines": {"image-segmentation": {"impl": "remote.Pipe"}}},
        {"model_type": "segformer"},
        {"id2label": {"0": "cat", "1": "dog"}},
        {"quantization_config": {"bits": 4}},
        {"size": 2048},
    ],
)
def test_closed_metadata_rejects_implementation_overrides(change):
    with pytest.raises(ValueError):
        binding.normalized_config(original() | change, TASK)


@pytest.mark.parametrize(
    "name,count,variant", [("BiRefNet", 754, "swin_v1_l"), ("BiRefNet_lite", 586, "swin_v1_t")]
)
def test_original_full_header_matches_owned_graph(name, count, variant):
    shapes = {
        k: tuple(v) for k, v in json.loads((FIXTURES / (name + "-shapes.json")).read_text()).items()
    }
    assert binding.backbone_variant(shapes) == variant
    model = binding.validate_weight_shapes(binding.normalized_config(original(name), TASK), shapes)
    assert len(model.state_dict()) == count


@pytest.mark.parametrize("shape", [(128, 3, 4, 4), (192, 4, 4, 4), (96, 3, 8, 8), None])
def test_unreviewed_backbone_is_not_guessed(shape):
    with pytest.raises(ValueError, match="BIREFNET_BACKBONE"):
        binding.backbone_variant({"bb.patch_embed.proj.weight": shape})


@pytest.mark.parametrize("name", ["BiRefNet", "BiRefNet_lite"])
@pytest.mark.parametrize("damage", [None, "missing", "pretrained"])
def test_metadata_discovery_uses_shared_segmentation(tmp_path, name, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore, ModelStoreError
    from oryn_engine.runtime_packs.execution import PackRuntime

    config = original(name)
    if damage == "pretrained":
        config["bb_pretrained"] = True
    files = [("config.json", 100, "a" * 64)]
    if damage != "missing":
        files.append(("model.safetensors", 100, "b" * 64))
    hub = FakeHub(config=config, files=files, pipeline_tag=TASK)
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    request = {"provider": "huggingface", "repository": "test/foreground", "revision": "main"}
    if damage == "pretrained":
        with pytest.raises(ModelStoreError):
            store.inspect(request)
    else:
        result = store.inspect(request)
        assert result["executable"] == (damage is None), result
        if damage is None:
            assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
    assert not hub.downloaded


def test_discovery_does_not_import_native_runtime():
    script = f"""
import sys,json
from pathlib import Path
from oryn_engine.generated_capabilities import normalized_transformers_config
from oryn_engine.model_layouts import assess_layout
raw=json.loads(Path({str(FIXTURES / "BiRefNet-config.json")!r}).read_text())
config=normalized_transformers_config(raw,'image-segmentation')
layout=assess_layout(['config.json','model.safetensors','birefnet.py'],config,'image-segmentation')
assert not layout.artifact_reasons and not layout.runtime_reasons
assert 'torch' not in sys.modules and 'oryn_engine.native_birefnet.model' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_shared_artifact_preserves_original_coordinates_and_labels(tiny):
    root, _, _, vectors = tiny
    data = vectors["cases"][0]
    source, output = root / "source.png", root / "mask.png"
    fixture_image(data).save(source)
    result = shared_tasks.execute_task(
        root,
        TASK,
        {"imageArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa"},
        "cpu",
        source,
        output,
        lambda *_: None,
    )
    assert (result["width"], result["height"]) == (data["width"], data["height"])
    assert (
        sum(s["pixelCount"] for s in result["metadata"]["segments"])
        == data["width"] * data["height"]
    )
    assert {s["label"] for s in result["metadata"]["segments"]} == {"background", "foreground"}
    with Image.open(output) as mask:
        assert (
            hashlib.sha256(np.asarray(mask).astype(np.uint16).tobytes()).hexdigest()
            == data["maskSha256"]
        )


def test_native_license_and_source_provenance_are_packaged():
    from oryn_engine import native_birefnet

    root = Path(native_birefnet.__file__).parent
    provenance = json.loads((root / "provenance.json").read_text())
    for name, digest in provenance["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    assert "MIT License" in (root / "LICENSE").read_text()
    assert "Copyright (c) 2021 Microsoft" in (root / "model.py").read_text()


@pytest.mark.parametrize("damage", ["nan", "infinite", "shape"])
def test_invalid_output_cannot_publish_a_mask(monkeypatch, damage):
    monkeypatch.setattr(binding, "IMAGE_SIZE", 32)
    output = torch.zeros(1, 1, 32, 32)
    if damage == "shape":
        output = output[:, :, :-1]
    else:
        output[0, 0, 0, 0] = float("nan" if damage == "nan" else "inf")
    with pytest.raises(ValueError, match="BIREFNET_OUTPUT"):
        binding.segmentation(lambda _: [output], Image.new("RGB", (17, 19)), "cpu")
