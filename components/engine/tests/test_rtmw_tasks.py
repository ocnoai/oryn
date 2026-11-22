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

from oryn_engine import rtmw_tasks as binding

FIXTURES = Path(__file__).parent / "fixtures/rtmw"


def document(name):
    return json.loads((FIXTURES / name).read_text())


def normalized(name="tiny-config.json"):
    return binding.normalized_config(document(name), binding.TASK)


@pytest.fixture
def checkpoint(tmp_path):
    config = document("tiny-config.json")
    state = binding.native_model(normalized()).state_dict()
    generator = torch.Generator().manual_seed(61280)
    for name, value in sorted(state.items()):
        if name.endswith(".num_batches_tracked"):
            value.fill_(3)
            continue
        noise = torch.randn(value.shape, generator=generator)
        if value.ndim >= 2 and name.endswith(".weight"):
            value.copy_(noise * (2 / (value.numel() / value.shape[0])) ** 0.5)
        else:
            value.copy_(noise * 0.05)
        if name.endswith(".running_var"):
            value.fill_(1)
        if name.endswith(".running_mean") or name.endswith(".bn.bias"):
            value.zero_()
        if name.endswith(".bn.weight") or name.endswith(".g") or name == "head.gau.res_scale.scale":
            value.fill_(1)
    assert {
        k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in state.items()
    } == document("reference.json")["tensorHashes"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    processor = document("preprocessor_config.json") | {"size": {"width": 64, "height": 96}}
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(processor))
    (tmp_path / "modeling_rtmw.py").write_text("raise AssertionError('repository code forbidden')")
    save_file(state, tmp_path / "model.safetensors")
    yield tmp_path
    for path in tmp_path.glob("*.safetensors"):
        path.unlink()


def image(case):
    mode, width, height = (case[k] for k in ("mode", "width", "height"))
    channels = {"RGB": 3, "L": 1, "RGBA": 4}[mode]
    values = ((np.arange(width * height * channels, dtype=np.uint32) * 37 + 13) % 256).astype(
        np.uint8
    )
    return Image.fromarray(
        values.reshape((height, width, channels) if channels > 1 else (height, width)), mode
    )


def expected_points(case, threshold=0, count=20):
    order = sorted(range(len(case["scores"])), key=lambda i: (-case["scores"][i], i))
    return [
        {"x": case["points"][i][0], "y": case["points"][i][1], "score": case["scores"][i]}
        for i in order
        if case["scores"][i] > 0 and case["scores"][i] >= threshold
    ][:count]


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("case", document("reference.json")["cases"])
def test_complete_graph_pixels_confidence_and_original_coordinates(checkpoint, backend, case):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    config = normalized()
    pixels = binding.preprocess(image(case), config["input_size"])
    assert hashlib.sha256(pixels.numpy().tobytes()).hexdigest() == case["pixelsSha256"]
    model = binding.load(checkpoint, config).to(backend)
    with np.load(FIXTURES / "logits.npz", allow_pickle=False) as gold, torch.inference_mode():
        x, y = model(pixels.to(backend))
        torch.testing.assert_close(
            x.cpu(), torch.from_numpy(gold[f"x{case['index']}"]), atol=3e-5, rtol=5e-5
        )
        torch.testing.assert_close(
            y.cpu(), torch.from_numpy(gold[f"y{case['index']}"]), atol=3e-5, rtol=5e-5
        )
        result = binding.execute(
            model, config, image(case), {"threshold": 0, "maxKeypoints": 20}, backend
        )
    assert (result["width"], result["height"]) == (case["width"], case["height"])
    expected = expected_points(case)
    assert len(result["keypoints"]) == len(expected) > 0
    np.testing.assert_allclose(
        [[p[k] for k in ("x", "y", "score")] for p in result["keypoints"]],
        [[p[k] for k in ("x", "y", "score")] for p in expected],
        atol=3e-5,
        rtol=5e-5,
    )


def test_original_full_graph_and_original_shapes():
    shapes = {k: tuple(v) for k, v in document("original-shapes.json").items()}
    model = binding.validate_weight_shapes(normalized("config.json"), shapes)
    assert len(shapes) == len(model.state_dict()) == 728
    assert all(v.device.type == "meta" for v in model.parameters())


@pytest.mark.parametrize(
    "missing",
    [
        "backbone.stem.0.conv.weight",
        "backbone.stage4.0.conv.weight",
        "neck.reduce_layers.0.conv.weight",
        "neck.top_down_blocks.0.main_conv.conv.weight",
        "neck.downsamples.1.conv.weight",
        "neck.bottom_up_blocks.1.final_conv.conv.weight",
        "head.conv_dec.conv.weight",
        "head.final_layer.conv.weight",
        "head.final_layer2.conv.weight",
        "head.mlp.1.weight",
        "head.mlp2.1.weight",
        "head.gau.uv.weight",
        "head.gau.gamma",
        "head.gau.res_scale.scale",
        "head.cls_x.weight",
        "head.cls_y.weight",
    ],
)
def test_every_native_branch_is_required(checkpoint, missing):
    path = checkpoint / "model.safetensors"
    state = load_file(path)
    del state[missing]
    save_file(state, path)
    with pytest.raises(ValueError, match="RTMW_WEIGHTS_INCOMPLETE"):
        binding.load(checkpoint, normalized())


@pytest.mark.parametrize(
    "damage",
    [
        "extra",
        "shape",
        "nan",
        "integer",
        "overflow",
        "negative-variance",
        "negative-counter",
        "float-counter",
    ],
)
def test_invalid_numeric_states_fail_closed(checkpoint, damage):
    path = checkpoint / "model.safetensors"
    state = load_file(path)
    key = "head.cls_x.weight"
    if damage == "extra":
        state["unexpected"] = torch.tensor(1.0)
    elif damage == "shape":
        state[key] = state[key][:1]
    elif damage == "nan":
        state[key][0, 0] = float("nan")
    elif damage == "integer":
        state[key] = state[key].long()
    elif damage == "overflow":
        state[key] = torch.full(state[key].shape, 1e300, dtype=torch.float64)
    elif damage == "negative-variance":
        state["backbone.stem.0.bn.running_var"][0] = -1
    elif damage == "negative-counter":
        state["backbone.stem.0.bn.num_batches_tracked"] = torch.tensor(-1)
    else:
        state["backbone.stem.0.bn.num_batches_tracked"] = torch.tensor(1.0)
    save_file(state, path)
    with pytest.raises(ValueError, match="RTMW_(WEIGHT|BATCH)"):
        binding.load(checkpoint, normalized())


@pytest.mark.parametrize(
    "change",
    [
        {"backbone_arch": "P6"},
        {"backbone_widen_factor": float("nan")},
        {"backbone_widen_factor": 0.07},
        {"backbone_deepen_factor": 0.5},
        {"input_size": [192, 257]},
        {"input_size": [4096, 4096]},
        {"head_in_channels": 768},
        {"neck_in_channels": [16.0, 32, 64]},
        {"head_in_featuremap_size": [2.0, 3]},
        {"gau_hidden_dims": 33},
        {"gau_s": 256},
        {"gau_use_rel_bias": True},
        {"gau_pos_enc": True},
        {"decoder_normalize": True},
        {"decoder_use_dark": True},
        {"num_keypoints": 17},
        {"score_max": 10.0},
        {"auto_map": {"AutoModel": "other.Code"}},
    ],
)
def test_unreviewed_graphs_are_rejected(change):
    with pytest.raises(ValueError):
        binding.validate_config(normalized() | change)


@pytest.mark.parametrize(
    "change",
    [
        {"do_rescale": True},
        {"do_resize": False},
        {"size": {"height": 192, "width": 256}},
        {"image_mean": [0.485, 0.456, 0.406]},
        {"do_pad": True},
        {"auto_map": {"AutoImageProcessor": "other.Code"}},
    ],
)
def test_processor_declaration_is_closed(change):
    with pytest.raises(ValueError):
        binding.validate_processor(
            document("preprocessor_config.json") | change, normalized("config.json")
        )


@pytest.mark.parametrize(
    "damage", [None, "processor", "size", "missing-processor", "missing-weights", "custom"]
)
def test_metadata_inspection_and_shared_route(tmp_path, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore, ModelStoreError
    from oryn_engine.runtime_packs.execution import PackRuntime

    config, processor = document("config.json"), document("preprocessor_config.json")
    if damage == "processor":
        processor["do_rescale"] = True
    elif damage == "size":
        processor["size"]["width"] = 256
    elif damage == "custom":
        config["auto_map"]["AutoModel"] = "other.Code"

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            assert filename == "preprocessor_config.json"
            return processor

    names = {"config.json", "preprocessor_config.json", "model.safetensors", "modeling_rtmw.py"}
    missing = {
        "missing-processor": "preprocessor_config.json",
        "missing-weights": "model.safetensors",
    }.get(damage)
    if missing:
        names.remove(missing)
    hub = Hub(config=config, pipeline_tag=binding.TASK, files=[(n, 100, "b" * 64) for n in names])
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    request = {"provider": "huggingface", "repository": "test/rtmw", "revision": "main"}
    if damage == "custom":
        with pytest.raises(ModelStoreError, match="reviewed binding"):
            store.inspect(request)
    else:
        result = store.inspect(request)
        if damage:
            assert not result["executable"], result
        else:
            assert result["executable"], result
            assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
            assert result["taskProfile"]["contract"]["id"] == "keypoint-detection-v1"
            assert {x["name"] for x in result["selectedFiles"]} == names - {"modeling_rtmw.py"}
    assert not hub.downloaded


def test_public_path_preserves_full_image_coordinates(checkpoint):
    from oryn_engine.shared_tasks import execute_task

    case = document("reference.json")["cases"][-1]
    path = checkpoint / "input.png"
    image(case).save(path)
    result = execute_task(
        checkpoint,
        binding.TASK,
        {"imageArtifactId": "artifact_" + "a" * 24, "threshold": 0, "maxKeypoints": 20},
        "cpu",
        path,
        None,
        lambda *_: None,
    )
    expected = expected_points(case)
    np.testing.assert_allclose(
        [[p[k] for k in ("x", "y", "score")] for p in result["keypoints"]],
        [[p[k] for k in ("x", "y", "score")] for p in expected],
        atol=3e-5,
        rtol=5e-5,
    )
    assert result["width"] == 1031


def test_decode_filters_invalid_points_and_applies_original_confidence_policy():
    from oryn_engine.native_rtmw.model import SimCCCodec

    config = normalized()
    x = torch.zeros(1, 133, 128)
    y = torch.zeros(1, 133, 192)
    for index, score in enumerate([0.1, 0.8, 2.0, 0.8]):
        x[0, index, 7 + index] = score
        y[0, index, 13 + index] = score / 2

    class Model:
        head = SimpleNamespace(decoder=SimCCCodec([64, 96], simcc_split_ratio=2))

        def __call__(self, _):
            return x, y

    result = binding.execute(
        Model(), config, Image.new("RGB", (320, 192)), {"threshold": 0.4, "maxKeypoints": 3}, "cpu"
    )
    assert result["keypoints"] == [
        {"x": 22.5, "y": 15.0, "score": 1.0},
        {"x": 20.0, "y": 14.0, "score": pytest.approx(0.4)},
        {"x": 25.0, "y": 16.0, "score": pytest.approx(0.4)},
    ]


@pytest.mark.parametrize("damage", ["nan", "width", "keypoints", "rank"])
def test_invalid_head_outputs_are_rejected(damage):
    x, y = torch.zeros(1, 133, 128), torch.zeros(1, 133, 192)
    if damage == "nan":
        x[0, 0, 0] = float("nan")
    elif damage == "width":
        x = x[:, :, :127]
    elif damage == "keypoints":
        y = y[:, :17]
    else:
        y = y[0]
    with pytest.raises(ValueError, match="RTMW_OUTPUT_INVALID"):
        binding.execute(
            lambda _: (x, y),
            normalized(),
            Image.new("RGB", (32, 32)),
            {"threshold": 0, "maxKeypoints": 20},
            "cpu",
        )


def test_metadata_validation_does_not_import_runtime():
    script = """
import json, sys
from pathlib import Path
from oryn_engine.generated_capabilities import normalized_transformers_config
from oryn_engine.shared_tasks import validate_task_config
from oryn_engine.rtmw_tasks import validate_processor
p = Path(sys.argv[1])
c = normalized_transformers_config(
    json.loads((p / 'config.json').read_text()), 'keypoint-detection'
)
validate_task_config('keypoint-detection', c)
validate_processor(json.loads((p / 'preprocessor_config.json').read_text()), c)
assert not {'torch', 'transformers'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", script, str(FIXTURES.resolve())], check=True)


def test_native_numerics_attribution_and_source_are_bound():
    from oryn_engine.generated_capabilities import generated_artifact

    root = Path(binding.__file__).parent
    sources = generated_artifact()["buildInputs"]["orynNativeSources"]
    for name in [
        "rtmw_tasks.py",
        *[
            "native_rtmw/" + n
            for n in ("__init__.py", "model.py", "LICENSE", "NOTICE.md", "provenance.json")
        ],
    ]:
        assert sources[name] == hashlib.sha256((root / name).read_bytes()).hexdigest()
