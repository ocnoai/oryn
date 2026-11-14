import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from oryn_engine import pixai_tasks as binding

FIXTURES = Path(__file__).parent / "fixtures/pixai"


def document(name):
    return json.loads((FIXTURES / name).read_text())


def normalized(name="tiny-config.json"):
    return binding.normalized_config(document(name), binding.TASK)


@pytest.fixture
def checkpoint(tmp_path):
    config = document("tiny-config.json")
    model = binding.native_model(normalized())
    shapes = binding.original_shapes(model)
    generator = torch.Generator().manual_seed(61278)
    state = {}
    for name, shape in sorted(shapes.items()):
        value = torch.randn(shape, generator=generator) * 0.05
        if name.endswith(".weight") and (".norm" in name or name.startswith("ln_pre.")):
            value += 1
        state[name] = value
    expected = document("reference.json")["tensorHashes"]
    assert {
        k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in state.items()
    } == expected
    (tmp_path / "config.json").write_text(json.dumps(config))
    processor = document("preprocessor_config.json") | {"size": config["img_size"]}
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(processor))
    save_file(state, tmp_path / "model.safetensors")
    yield tmp_path
    for path in tmp_path.glob("*.safetensors"):
        path.unlink()


def image(case):
    mode, width, height = (case[k] for k in ("mode", "width", "height"))
    channels = {"RGB": 3, "L": 1, "RGBA": 4}[mode]
    data = ((np.arange(width * height * channels, dtype=np.uint32) * 37 + 13) % 256).astype(
        np.uint8
    )
    return Image.fromarray(
        data.reshape((height, width, channels) if channels > 1 else (height, width)), mode
    )


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("case", document("reference.json")["cases"])
def test_complete_native_backbone_and_pool_match_publisher(checkpoint, backend, case):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    config = normalized()
    pixels = binding.preprocess(image(case), config["img_size"])
    assert hashlib.sha256(pixels.numpy().tobytes()).hexdigest() == case["pixelsSha256"]
    model = binding.load(checkpoint, config).to(backend)
    with torch.inference_mode():
        logits = model(pixels.to(backend)).cpu()[0]
        result = binding.execute(model, config, image(case), {"topK": 3}, backend)
    torch.testing.assert_close(logits, torch.tensor(case["logits"]), atol=2e-6, rtol=2e-5)
    assert [x["labelId"] for x in result["predictions"]] == case["labelOrder"][:3]
    np.testing.assert_allclose(
        [x["score"] for x in result["predictions"]],
        [case["scores"][i] for i in case["labelOrder"][:3]],
        atol=2e-6,
        rtol=2e-5,
    )
    # Both original positional pieces remain loaded, including the unused row.
    original = load_file(checkpoint / "model.safetensors")["pos_embed"]
    torch.testing.assert_close(model.unused_class_position.cpu(), original[:, :1])
    assert sum(x["score"] for x in result["predictions"]) > 1


def test_all_original_tensor_shapes_bind_to_native_graph():
    shapes = {k: tuple(v) for k, v in document("original-shapes.json").items()}
    model = binding.validate_weight_shapes(normalized("config.json"), shapes)
    assert len(shapes) == 398
    assert len(model.state_dict()) == 527
    assert binding.original_shapes(model) == shapes


@pytest.mark.parametrize(
    "missing",
    [
        "pos_embed",
        "patch_embed.proj.weight",
        "ln_pre.bias",
        "blocks.0.attn.qkv.weight",
        "blocks.1.attn.proj.bias",
        "blocks.2.mlp.fc2.weight",
        "blocks.3.norm2.weight",
        "head_pool.q",
        "head_pool.pos_embed",
        "head_pool.kv.weight",
        "head_pool.proj.bias",
        "head.weight",
    ],
)
def test_each_learned_branch_is_required(checkpoint, missing):
    path = checkpoint / "model.safetensors"
    state = load_file(path)
    del state[missing]
    save_file(state, path)
    with pytest.raises(ValueError, match="PIXAI_WEIGHTS_INCOMPLETE"):
        binding.load(checkpoint, normalized())


@pytest.mark.parametrize(
    "damage", ["extra", "shape", "nan", "integer", "overflow", "unused-cls-nan"]
)
def test_invalid_original_weights_fail_closed(checkpoint, damage):
    path = checkpoint / "model.safetensors"
    state = load_file(path)
    if damage == "extra":
        state["extra"] = torch.zeros(1)
    elif damage == "shape":
        state["head.weight"] = state["head.weight"][:1]
    elif damage == "integer":
        state["head.bias"] = state["head.bias"].long()
    elif damage == "overflow":
        state["head.bias"] = torch.full_like(state["head.bias"], 1e100, dtype=torch.float64)
    elif damage == "unused-cls-nan":
        state["pos_embed"][0, 0, 0] = float("nan")
    else:
        state["head.bias"][0] = float("nan")
    save_file(state, path)
    with pytest.raises(ValueError, match="PIXAI_"):
        binding.load(checkpoint, normalized())


@pytest.mark.parametrize(
    "change",
    [
        {"use_rope": False},
        {"retain_cls_token": True},
        {"ln_post": True},
        {"use_act_checkpoint": True},
        {"mlp_ratio": 0},
        {"num_heads": 3},
        {"global_att_blocks": [1]},
        {"global_att_blocks": [3, 1]},
        {"tags": ["duplicate"] * 8},
        {"tags_split": [["general", 7]]},
        {"category_best_threshold": {"general": float("nan")}},
        {"auto_map": {"AutoModel": "unreviewed.Code"}},
    ],
)
def test_unimplemented_graph_and_label_semantics_rejected(change):
    with pytest.raises(ValueError):
        binding.validate_config(normalized() | change)


@pytest.mark.parametrize(
    "change",
    [
        {"size": 24},
        {"do_resize": True},
        {"image_mean": [0, 0, 0]},
        {"auto_map": {"AutoImageProcessor": "unreviewed.Code"}},
        {"rescale_factor": 0.5},
    ],
)
def test_processor_must_match_declared_native_pipeline(change):
    metadata = document("preprocessor_config.json") | {"size": 20} | change
    with pytest.raises(ValueError):
        binding.validate_processor(metadata, normalized())


@pytest.mark.parametrize("damage", [None, "processor", "size", "missing-weights", "custom"])
def test_owned_metadata_inspection_and_task_routing(tmp_path, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    config = document("config.json")
    processor = document("preprocessor_config.json")
    if damage == "processor":
        processor["do_resize"] = True
    elif damage == "size":
        processor["size"] = 512
    elif damage == "custom":
        config["auto_map"]["AutoModel"] = "unreviewed.Code"

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            assert filename == "preprocessor_config.json"
            return processor

    names = ["config.json", "preprocessor_config.json", "model.safetensors", "tagger_pipeline.py"]
    if damage == "missing-weights":
        names.remove("model.safetensors")
    hub = Hub(config=config, pipeline_tag=binding.TASK, files=[(n, 100, "b" * 64) for n in names])
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    if damage == "custom":
        from oryn_engine.models import ModelStoreError

        with pytest.raises(ModelStoreError, match="reviewed binding"):
            store.inspect(
                {"provider": "huggingface", "repository": "test/pixai", "revision": "main"}
            )
        assert not hub.downloaded
        return
    result = store.inspect(
        {"provider": "huggingface", "repository": "test/pixai", "revision": "main"}
    )
    if damage:
        assert not result["executable"]
    else:
        assert result["executable"], result
        assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
        assert result["taskProfile"]["contract"]["id"] == "image-classification-v1"
        assert {x["name"] for x in result["selectedFiles"]} == set(names) - {"tagger_pipeline.py"}
    assert not hub.downloaded


def test_public_path_preserves_transparency_and_original_coordinates(checkpoint):
    from oryn_engine.shared_tasks import execute_task

    case = document("reference.json")["cases"][-1]
    path = checkpoint / "input.png"
    image(case).save(path)
    result = execute_task(
        checkpoint,
        binding.TASK,
        {"imageArtifactId": "artifact_" + "a" * 24, "topK": 3},
        "cpu",
        path,
        None,
        lambda *_: None,
    )
    assert [x["labelId"] for x in result["predictions"]] == case["labelOrder"][:3]
    np.testing.assert_allclose(
        [x["score"] for x in result["predictions"]],
        [case["scores"][i] for i in case["labelOrder"][:3]],
        atol=2e-6,
        rtol=2e-5,
    )


@pytest.mark.parametrize("damage", ["nan", "width", "rank"])
def test_invalid_predictions_are_rejected(damage):
    values = torch.zeros((1, 8))
    if damage == "nan":
        values[0, 0] = float("nan")
    elif damage == "width":
        values = values[:, :7]
    else:
        values = values[0]
    with pytest.raises(ValueError, match="PIXAI_OUTPUT_INVALID"):
        binding.execute(
            lambda _: values, normalized(), image(document("reference.json")["cases"][0]), {}, "cpu"
        )


def test_native_sam3_dependency_source_bytes_are_bound():
    from transformers.models.auto import modeling_auto

    from oryn_engine.generated_capabilities import generated_artifact

    sources = generated_artifact()["buildInputs"]["transformersVisionSources"]
    assert {"models/sam3/modeling_sam3.py", "models/sam3/configuration_sam3.py"} <= set(sources)
    for name, digest in sources.items():
        assert (
            hashlib.sha256(
                (Path(modeling_auto.__file__).parents[2] / name).read_bytes()
            ).hexdigest()
            == digest
        )
