"""Complete DPT/BiT numeric graphs, original processor metadata and language aliases."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import transformers as tf
from PIL import Image

from oryn_engine import numeric_archive_tasks as archive
from oryn_engine import numeric_dpt as family
from oryn_engine import numeric_seq2seq, shared_tasks

TASK = "depth-estimation"
ARTIFACT = "artifact_" + "a" * 24


def config(hybrid):
    options = dict(
        image_size=32,
        patch_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        backbone_out_indices=[0, 1, 2, 3],
        neck_hidden_sizes=[16, 32, 32, 32],
        fusion_hidden_size=16,
        is_hybrid=hybrid,
    )
    if hybrid:
        options.update(
            backbone_config=dict(
                model_type="bit",
                embedding_size=8,
                hidden_sizes=[16, 32, 64],
                depths=[1, 1, 1],
                num_groups=2,
                layer_type="bottleneck",
                out_features=["stage1", "stage2", "stage3"],
                embedding_dynamic_padding=True,
                global_padding="SAME",
            ),
            backbone_featmap_shape=[1, 64, 2, 2],
            neck_ignore_stages=[0, 1],
            reassemble_factors=[1, 1, 1, 0.5],
        )
    c = tf.DPTConfig(**options)
    c.architectures = ["DPTForDepthEstimation"]
    return c


def pixels(width=45, height=31):
    return Image.fromarray(
        ((np.arange(width * height * 3) * 37 + 19) % 256).astype(np.uint8).reshape(height, width, 3)
    )


def fixture(root, hybrid, dtype=torch.float32):
    torch.manual_seed(6129)
    c = config(hybrid)
    c._attn_implementation = "eager"
    native = tf.DPTForDepthEstimation(c).eval()
    with torch.no_grad():
        native.head.head[4].bias.fill_(0.2)
    state = {k: v.to(dtype) for k, v in native.state_dict().items()}
    torch.save(state, root / "pytorch_model.bin")
    native.load_state_dict(state, strict=True)
    c.save_pretrained(root)
    metadata = dict(
        image_processor_type="DPTImageProcessor",
        size={"height": 32, "width": 32},
        resample=2,
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
        rescale_factor=1 / 255,
        ensure_multiple_of=1,
        keep_aspect_ratio=False,
    )
    (root / "preprocessor_config.json").write_text(json.dumps(metadata))
    return native, json.loads((root / "config.json").read_text()), metadata


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_depth_graph_matches_native_reference(tmp_path, hybrid, dtype, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    native, c, metadata = fixture(tmp_path, hybrid, dtype)
    model, processor, actual_config = shared_tasks.load_task(tmp_path, TASK, backend)
    assert family.expected_shapes(c) == {k: list(v.shape) for k, v in native.state_dict().items()}
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), native.state_dict()[name], atol=0, rtol=0)
    image = pixels()
    reference_processor = tf.DPTImageProcessor(**metadata)
    inputs = reference_processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        reference = native(**inputs).predicted_depth
        actual = model(pixel_values=inputs["pixel_values"].to(backend)).predicted_depth.cpu()
        torch.testing.assert_close(actual, reference, atol=1e-7, rtol=3e-4)
        result = archive.execute(
            model, processor, actual_config, TASK, {"imageArtifactId": ARTIFACT}, backend, image
        )
    assert result["maximum"] > 0 and (result["width"], result["height"]) == (32, 32)
    assert result["mean"] == pytest.approx(reference.mean().item(), abs=1e-7, rel=3e-4)
    shared_tasks.validate_task_output(TASK, result)
    assert TASK in shared_tasks.IMAGE_TASKS


@pytest.mark.parametrize("resample", [2, 3])
@pytest.mark.parametrize("width,height", [(45, 31), (31, 45)])
def test_pixels_preserve_explicit_interpolation_and_rgb_normalization(
    tmp_path, resample, width, height
):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as tvf

    _, c, m = fixture(tmp_path, True)
    m["resample"] = resample
    image = pixels(width, height)
    processor = tf.DPTImageProcessor(**family.processor_options(c, m))
    actual = processor(images=image, return_tensors="pt")["pixel_values"]
    source = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    resized = tvf.resize(
        source,
        [32, 32],
        interpolation={2: InterpolationMode.BILINEAR, 3: InterpolationMode.BICUBIC}[resample],
        antialias=True,
    )
    expected = ((resized.double() / 255 - 0.5) / 0.5).float().unsqueeze(0)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=0)


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("part", ["embedding", "transformer", "neck", "head", "extra", "shape"])
def test_missing_or_incompatible_graph_is_rejected(tmp_path, hybrid, part):
    native, _, _ = fixture(tmp_path, hybrid)
    state = native.state_dict()
    fragments = {
        "embedding": "dpt.embeddings.cls_token",
        "transformer": "dpt.encoder.layer.0.attention.attention.query.weight",
        "neck": "neck.convs.0.weight",
        "head": "head.head.4.weight",
        "shape": "head.head.4.weight",
    }
    if part == "extra":
        state["custom.head"] = torch.zeros(1)
    elif part == "shape":
        state[fragments[part]] = torch.zeros(2, 3)
    else:
        del state[fragments[part]]
    torch.save(state, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError):
        archive.load_components(tmp_path, TASK, "cpu")


@pytest.mark.parametrize(
    "damage", ["size", "aspect", "resize", "resample", "mean", "std", "scale", "custom"]
)
def test_processor_rejects_unbound_geometry_or_semantics(tmp_path, damage):
    _, c, m = fixture(tmp_path, True)
    change = {
        "size": {"size": {"height": 32, "width": 64}},
        "aspect": {"keep_aspect_ratio": True},
        "resize": {"do_resize": False},
        "resample": {"resample": 0},
        "mean": {"image_mean": [0, 0, 0]},
        "std": {"image_std": [1, 1, 1]},
        "scale": {"rescale_factor": 1},
        "custom": {"auto_map": {"AutoImageProcessor": "repo.Custom"}},
    }[damage]
    with pytest.raises(ValueError):
        family.processor_options(c, m | change)


def test_complete_original_414_tensor_graph_and_bilinear_processor():
    from oryn_engine.numeric_state_dict import parse

    root = Path(__file__).parent / "fixtures/numeric_archive"
    c = json.loads((root / "dpt.config.json").read_text())
    m = json.loads((root / "dpt.preprocessor.json").read_text())
    tensors = parse((root / "dpt.metadata").read_bytes())
    metadata = {
        "format": "torch-numeric-zip-v1",
        "byteOrder": "little",
        "tensors": {
            k: {"shape": list(v.shape), "dtype": v.storage.dtype} for k, v in tensors.items()
        },
    }
    shapes, legacy = archive.validate_archive(c, TASK, metadata)
    assert len(shapes) == 414 and legacy is None
    assert family.processor_options(c, m)["resample"] == 2


@pytest.mark.parametrize("hybrid", [False, True])
def test_repeated_offline_worker(tmp_path, hybrid):
    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    fixture(tmp_path, hybrid)
    image = tmp_path / "input.png"
    pixels().save(image)
    pack = builtin_registry().active_pack("oryn.transformers-pytorch")
    request = dict(
        abi=pack.abi,
        packId=pack.pack_id,
        version=pack.version,
        command="generate",
        payload=dict(
            modelRoot=str(tmp_path),
            task=TASK,
            backend="cpu",
            input={"imageArtifactId": ARTIFACT},
            inputPath=str(image),
            outputPath=None,
        ),
    )
    results = []
    for _ in range(2):
        child = subprocess.run(
            [sys.executable, "-m", "oryn_engine.runtime_packs.transformers_worker"],
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            timeout=60,
            env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        assert child.returncode == 0, child.stderr
        result = next(
            json.loads(line)
            for line in child.stdout.splitlines()
            if json.loads(line)["type"] == "result"
        )
        assert result["ok"], (result, child.stderr)
        results.append(result["result"])
    assert results[0] == results[1]


@pytest.mark.parametrize(
    "language,expected",
    [
        ("fr", "fra_Latn"),
        ("fra", "fra_Latn"),
        ("eng", "eng_Latn"),
        ("rus", "rus_Cyrl"),
        ("jpn", "jpn_Jpan"),
        ("zho", None),
        ("ace", None),
        ("xxx", None),
    ],
)
def test_nllb_iso_three_letter_codes_preserve_scripts(language, expected):
    if expected:
        assert numeric_seq2seq.target_language(language) == expected
    else:
        with pytest.raises(ValueError):
            numeric_seq2seq.target_language(language)


def test_nllb_iso_alias_and_native_code_generate_identical_translation(tmp_path):
    from test_numeric_seq2seq import fixture as sequence_fixture

    model, tokenizer, c = sequence_fixture(tmp_path, "translation", "m2m_100")
    with torch.inference_mode():
        values = [
            numeric_seq2seq.execute(
                model,
                tokenizer,
                c,
                "translation",
                {"text": "hello world", "maxTokens": 4, "targetLanguage": lang},
                "cpu",
            )
            for lang in ["fr", "fra"]
        ]
    assert values[0] == values[1]


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize(
    "damage", [None, "missing-processor", "processor", "missing-head", "extra"]
)
def test_discovery_checks_image_graph_without_tokenizer(tmp_path, monkeypatch, hybrid, damage):
    from types import SimpleNamespace

    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    _, c, m = fixture(tmp_path, hybrid)
    metadata = {
        "format": "torch-numeric-zip-v1",
        "byteOrder": "little",
        "tensors": {k: {"shape": v, "dtype": "F32"} for k, v in family.expected_shapes(c).items()},
    }
    if damage == "processor":
        m["resample"] = 0
    if damage == "missing-head":
        del metadata["tensors"]["head.head.4.weight"]
    if damage == "extra":
        metadata["tensors"]["custom.head"] = {"shape": [1], "dtype": "F32"}

    class Hub(FakeHub):
        def read_numeric_archive(self, *args):
            return metadata

        def read_json(self, *args):
            assert args[-1] == "preprocessor_config.json"
            return m

        def download(self, *args, **kwargs):
            raise AssertionError("Original weight download forbidden")

    files = [("config.json", 1000, "a" * 64), ("pytorch_model.bin", 100000, "b" * 64)]
    if damage != "missing-processor":
        files.append(("preprocessor_config.json", 1000, "c" * 64))
    hub = Hub(config=c, pipeline_tag=TASK, files=files)

    def forbidden(*args, **kwargs):
        raise AssertionError("Model construction forbidden during metadata inspection")

    monkeypatch.setattr(tf.AutoConfig, "for_model", forbidden)
    store = ModelStore(tmp_path / "store", hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "test/numeric-dpt", "revision": "main"}
    )
    assert result["executable"] == (damage is None), result.get("blockingReasons")
    if damage is None:
        assert {x["name"] for x in result["selectedFiles"]} == {
            "config.json",
            "pytorch_model.bin",
            "preprocessor_config.json",
        }
        assert result["taskProfile"]["contract"]["id"] == "depth-estimation-v1"


@pytest.mark.parametrize(
    "change",
    [
        {"patch_size": 16.0},
        {"num_channels": 3.0},
        {"head_in_index": -1.0},
        {"backbone": ""},
        {"backbone_kwargs": {"out_indices": [0]}},
        {"reassemble_factors": [4.0, 2, 1, 0.5]},
        {"use_pretrained_backbone": True},
        {"use_timm_backbone": True},
        {"use_batch_norm_in_fusion_residual": True},
        {"add_projection": True},
        {"auto_map": {"AutoModel": "repo.Custom"}},
    ],
)
def test_unbound_native_options_are_rejected_before_model_construction(change):
    c = json.loads(config(False).to_json_string())
    with pytest.raises(ValueError):
        family.validate_config(c | change)


@pytest.mark.parametrize(
    "change",
    [
        {"num_channels": 3.0},
        {"out_indices": [0]},
        {"stage_names": ["stem"]},
        {"num_groups": 3},
        {"depths": [1, 1, 1, 1]},
        {"layer_type": "preactivation"},
    ],
)
def test_hybrid_backbone_geometry_matches_the_bound_graph(change):
    c = json.loads(config(True).to_json_string())
    c["backbone_config"].update(change)
    with pytest.raises(ValueError):
        family.validate_config(c)


def test_hybrid_feature_map_dimensions_are_integers():
    c = json.loads(config(True).to_json_string())
    c["backbone_featmap_shape"] = [1, 64.0, 2, 2]
    with pytest.raises(ValueError):
        family.validate_config(c)


@pytest.mark.parametrize("shared", [False, True])
def test_numeric_binding_does_not_hijack_existing_safetensors_depth_loader(
    tmp_path, monkeypatch, shared
):
    from types import SimpleNamespace

    from oryn_engine.runners import RunnerRegistry, SharedTaskPackHandle, TransformersTaskHandle
    from oryn_engine.runtime_packs.execution import PackRuntime

    native, c, _ = fixture(tmp_path, True)
    native.save_pretrained(tmp_path)
    record = {
        "runner": "transformers",
        "model": {"repository": str(tmp_path), "revision": "a" * 40},
        "cacheRoot": str(tmp_path / "cache"),
        "technicalCapabilities": {"task": TASK, "architectures": c["architectures"]},
        "_runtimeCandidate": {"backend": "cpu"},
    }
    registry = RunnerRegistry()
    if shared:
        record["runner"] = "runtime-pack"
        record["runtimePack"] = {
            "packId": "oryn.transformers-pytorch",
            "runnerId": "transformers.shared-tasks",
        }
        monkeypatch.setattr(
            registry,
            "_pack_locator",
            SimpleNamespace(resolve=lambda _: PackRuntime(Path(sys.executable), None)),
        )
    handle = registry._load_single(record)
    assert isinstance(handle, SharedTaskPackHandle if shared else TransformersTaskHandle)
    if not shared:
        for name, tensor in native.state_dict().items():
            torch.testing.assert_close(handle.model.state_dict()[name], tensor, atol=0, rtol=0)
