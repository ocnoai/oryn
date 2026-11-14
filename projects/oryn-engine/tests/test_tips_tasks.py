import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from oryn_engine import tips_tasks as binding

FIXTURES = Path(__file__).parent / "fixtures/tips"


def document(name):
    return json.loads((FIXTURES / name).read_text())


def normalized(name="tiny-config.json"):
    return binding.normalized_config(document(name), binding.TASK)


@pytest.fixture
def checkpoint(tmp_path):
    config = document("tiny-config.json")
    model = binding.native_model(normalized())
    generator = torch.Generator().manual_seed(61279)
    state = {}
    for name, shape in sorted(binding.original_shapes(model).items()):
        value = torch.randn(shape, generator=generator) * 0.05
        if name.endswith(".weight") and (".norm" in name or ".ln_" in name):
            value += 1
        if name.endswith(".gamma"):
            value += 1
        state[name] = value
    assert {
        k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in state.items()
    } == document("reference.json")["tensorHashes"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    for name in ("tokenizer_config.json", "tokenizer.model"):
        shutil.copyfile(FIXTURES / name, tmp_path / name)
    metadata = document("processor_config.json")
    metadata["image_processor"]["size"] = {"height": 16, "width": 16}
    (tmp_path / "processor_config.json").write_text(json.dumps(metadata))
    (tmp_path / "modeling_tips.py").write_text(
        "raise AssertionError('repository code is forbidden')"
    )
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
def test_full_original_encoders_preprocess_and_scores_match_publisher(checkpoint, backend, case):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    config = normalized()
    pixels = binding.preprocess(image(case), 16)
    assert hashlib.sha256(pixels.numpy().tobytes()).hexdigest() == case["pixelsSha256"]
    model, tokenizer, _ = binding.load_components(checkpoint, config, backend)
    vectors = document("reference.json")
    ids, mask = binding.tokenize(tokenizer, vectors["labels"])
    assert ids.tolist() == vectors["inputIds"]
    assert mask.tolist() == vectors["attentionMask"]
    with torch.inference_mode():
        output = model(
            pixel_values=pixels.to(backend),
            input_ids=ids.to(backend),
            attention_mask=mask.to(backend),
        )
        result = binding.execute(
            model, tokenizer, config, image(case), {"labels": vectors["labels"]}, backend
        )
    torch.testing.assert_close(
        output.vision_model_output.pooler_output.cpu()[0],
        torch.tensor(case["imageEmbedding"]),
        atol=3e-6,
        rtol=3e-5,
    )
    torch.testing.assert_close(
        output.text_model_output.pooler_output.cpu(),
        torch.tensor(vectors["textEmbeddings"]),
        atol=3e-6,
        rtol=3e-5,
    )
    torch.testing.assert_close(
        output.logits_per_image.cpu()[0], torch.tensor(case["logits"]), atol=2e-4, rtol=3e-5
    )
    assert [x["label"] for x in result["predictions"]] == [
        vectors["labels"][i] for i in case["order"]
    ]
    np.testing.assert_allclose(
        [x["score"] for x in result["predictions"]],
        [case["scores"][i] for i in case["order"]],
        atol=2e-5,
        rtol=3e-5,
    )
    assert sum(x["score"] for x in result["predictions"]) == pytest.approx(1)
    assert model.temperature.item() == pytest.approx(config["temperature_init_value"])


def test_all_original_learned_tensors_bind_to_complete_native_graph():
    shapes = {k: tuple(v) for k, v in document("original-shapes.json").items()}
    model = binding.validate_weight_shapes(normalized("config.json"), shapes)
    assert len(shapes) == 715
    assert len(model.state_dict()) == 1004  # Transformers 5.20 split fused projections
    assert binding.original_shapes(model) == shapes


@pytest.mark.parametrize(
    "missing",
    [
        "vision_encoder.cls_token",
        "vision_encoder.mask_token",
        "vision_encoder.register_tokens",
        "vision_encoder.pos_embed",
        "vision_encoder.patch_embed.proj.bias",
        "vision_encoder.blocks.0.attn.qkv.weight",
        "vision_encoder.blocks.1.ls2.gamma",
        "vision_encoder.blocks.1.mlp.w12.weight",
        "vision_encoder.norm.bias",
        "text_encoder.token_embedding.weight",
        "text_encoder.transformer.resblocks.0.attn.in_proj_bias",
        "text_encoder.transformer.resblocks.1.mlp.c_fc.weight",
        "text_encoder.ln_final.bias",
    ],
)
def test_every_learned_branch_is_required(checkpoint, missing):
    path = checkpoint / "model.safetensors"
    state = load_file(path)
    del state[missing]
    save_file(state, path)
    with pytest.raises(ValueError, match="TIPS_WEIGHTS_INCOMPLETE"):
        binding.load_components(checkpoint, normalized(), "cpu")


@pytest.mark.parametrize("damage", ["extra", "shape", "nan", "integer", "overflow"])
def test_invalid_weights_fail_closed(checkpoint, damage):
    path = checkpoint / "model.safetensors"
    state = load_file(path)
    key = "text_encoder.ln_final.weight"
    if damage == "extra":
        state["temperature"] = torch.tensor(0.01)
    elif damage == "shape":
        state[key] = state[key][:1]
    elif damage == "nan":
        state[key][0] = float("nan")
    elif damage == "integer":
        state[key] = state[key].long()
    else:
        state[key] = torch.full(state[key].shape, 1e300, dtype=torch.float64)
    save_file(state, path)
    with pytest.raises(ValueError, match="TIPS_(WEIGHT_VALUES|WEIGHTS_INCOMPLETE)"):
        binding.load_components(checkpoint, normalized(), "cpu")


@pytest.mark.parametrize(
    "section, key, value",
    [
        (None, "temperature_init_value", 0),
        (None, "temperature_init_value", float("nan")),
        (None, "temperature_init_value", True),
        (None, "auto_map", {"AutoModel": "other.Code"}),
        (None, "architectures", ["OtherModel"]),
        (None, "quantization_config", {"bits": 4}),
        ("text_config", "hidden_size", 48),
        ("text_config", "hidden_act", "gelu"),
        ("text_config", "max_position_embeddings", 128),
        ("text_config", "scale_sqrt_depth", False),
        ("vision_config", "num_register_tokens", 2),
        ("vision_config", "use_swiglu_ffn", False),
        ("vision_config", "qkv_bias", False),
        ("vision_config", "out_indices", [0]),
        ("vision_config", "image_size", 17),
        ("vision_config", "drop_path_rate", 0.1),
    ],
)
def test_unreviewed_graph_variants_are_rejected(section, key, value):
    config = normalized()
    (config if section is None else config[section])[key] = value
    with pytest.raises(ValueError):
        binding.validate_config(config)


@pytest.mark.parametrize(
    "name, change",
    [
        ("tokenizer_config.json", {"do_lower_case": False}),
        ("tokenizer_config.json", {"bos_token": "<s>"}),
        ("tokenizer_config.json", {"model_max_length": 77}),
        ("tokenizer_config.json", {"auto_map": {"AutoTokenizer": "other.Code"}}),
        ("processor_config.json", {"processor_class": "OtherProcessor"}),
        ("processor_config.json", {"extra": True}),
    ],
)
def test_metadata_semantics_are_closed(name, change):
    with pytest.raises(ValueError):
        binding.validate_metadata(name, document(name) | change, normalized("config.json"))


@pytest.mark.parametrize(
    "damage", ["missing", "symlink", "oversize", "corrupt", "vocab", "special"]
)
def test_invalid_sentencepiece_artifacts_fail_closed(checkpoint, damage):
    from sentencepiece import sentencepiece_model_pb2

    path = checkpoint / "tokenizer.model"
    config = normalized()
    if damage == "missing":
        path.unlink()
    elif damage == "symlink":
        path.unlink()
        path.symlink_to((FIXTURES / "tokenizer.model").resolve())
    elif damage == "oversize":
        path.write_bytes(b"x" * (2 * 1024**2 + 1))
    elif damage == "corrupt":
        path.write_bytes(b"invalid protobuf")
    elif damage == "vocab":
        config["text_config"]["vocab_size"] -= 1
    else:
        proto = sentencepiece_model_pb2.ModelProto()
        proto.ParseFromString(path.read_bytes())
        proto.trainer_spec.pad_id = -1
        proto.pieces[0].type = 1
        path.write_bytes(proto.SerializeToString())
    with pytest.raises(ValueError, match="TIPS_TOKENIZER"):
        binding.load_tokenizer(checkpoint, config)


def test_original_tokenizer_case_unicode_padding_truncation_and_empty_labels():
    tokenizer = binding.load_tokenizer(FIXTURES, normalized("config.json"))
    vectors = document("reference.json")
    ids, mask = binding.tokenize(tokenizer, vectors["labels"])
    assert ids.tolist() == vectors["inputIds"]
    assert mask.tolist() == vectors["attentionMask"]
    assert mask[2].sum() == 64
    assert ids[0, 0] not in (tokenizer.bos_id(), tokenizer.eos_id())
    torch.testing.assert_close(
        ids[:1], binding.tokenize(tokenizer, [vectors["labels"][0].lower()])[0]
    )
    for label in ("", "  \n\t"):
        with pytest.raises(ValueError, match="TIPS_EMPTY_LABEL"):
            binding.tokenize(tokenizer, [label])


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "processor",
        "tokenizer",
        "missing-tokenizer",
        "missing-processor",
        "missing-weights",
        "custom",
    ],
)
def test_metadata_inspection_routes_to_owned_shared_task(tmp_path, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore, ModelStoreError
    from oryn_engine.runtime_packs.execution import PackRuntime

    config = document("config.json")
    metadata = {name: document(name) for name in binding.METADATA_NAMES}
    if damage == "processor":
        metadata["processor_config.json"]["image_processor"]["size"]["height"] = 448
    elif damage == "tokenizer":
        metadata["tokenizer_config.json"]["do_lower_case"] = False
    elif damage == "custom":
        config["auto_map"]["AutoModel"] = "other.Code"

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return metadata[filename]

    names = {
        "config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "model.safetensors",
        "modeling_tips.py",
    }
    missing = {
        "missing-tokenizer": "tokenizer.model",
        "missing-processor": "processor_config.json",
        "missing-weights": "model.safetensors",
    }.get(damage)
    if missing:
        names.remove(missing)
    hub = Hub(config=config, pipeline_tag=binding.TASK, files=[(n, 100, "b" * 64) for n in names])
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    request = {"provider": "huggingface", "repository": "test/tips", "revision": "main"}
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
            assert result["taskProfile"]["contract"]["id"] == "zero-shot-image-classification-v1"
            assert {x["name"] for x in result["selectedFiles"]} == names - {"modeling_tips.py"}
    assert not hub.downloaded


def test_public_task_preserves_original_image_resize_and_labels(checkpoint):
    from oryn_engine.shared_tasks import execute_task

    vectors = document("reference.json")
    case = vectors["cases"][-1]
    path = checkpoint / "input.png"
    image(case).save(path)
    result = execute_task(
        checkpoint,
        binding.TASK,
        {"imageArtifactId": "artifact_" + "a" * 24, "labels": vectors["labels"]},
        "cpu",
        path,
        None,
        lambda *_: None,
    )
    assert [x["label"] for x in result["predictions"]] == [
        vectors["labels"][i] for i in case["order"]
    ]
    np.testing.assert_allclose(
        [x["score"] for x in result["predictions"]],
        [case["scores"][i] for i in case["order"]],
        atol=2e-5,
        rtol=3e-5,
    )


@pytest.mark.parametrize("damage", ["nan", "width", "rank"])
def test_invalid_outputs_are_rejected(damage):
    logits = torch.zeros(1, 2)
    if damage == "nan":
        logits[0, 0] = float("nan")
    elif damage == "width":
        logits = logits[:, :1]
    else:
        logits = logits[0]

    def model(**_):
        return SimpleNamespace(logits_per_image=logits)

    tokenizer = binding.load_tokenizer(FIXTURES, normalized())
    with pytest.raises(ValueError, match="TIPS_OUTPUT_INVALID"):
        binding.execute(
            model,
            tokenizer,
            normalized(),
            image(document("reference.json")["cases"][0]),
            {"labels": ["bird", "tree"]},
            "cpu",
        )


def test_metadata_path_does_not_import_numerical_runtime():
    script = """
import json, sys
from pathlib import Path
from oryn_engine.generated_capabilities import normalized_transformers_config
from oryn_engine.shared_tasks import validate_task_config
from oryn_engine.tips_tasks import validate_metadata
p = Path(sys.argv[1])
c = normalized_transformers_config(
    json.loads((p / 'config.json').read_text()), 'zero-shot-image-classification'
)
validate_task_config('zero-shot-image-classification', c)
for name in ('tokenizer_config.json', 'processor_config.json'):
    validate_metadata(name, json.loads((p / name).read_text()), c)
assert not {'torch', 'transformers', 'sentencepiece'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", script, str(FIXTURES.resolve())], check=True)


def test_native_tips_source_bytes_are_in_signed_closure():
    from transformers.models.auto import modeling_auto

    from oryn_engine.generated_capabilities import generated_artifact

    inputs = generated_artifact()["buildInputs"]
    for name in ("models/tipsv2/modeling_tipsv2.py", "models/tipsv2/configuration_tipsv2.py"):
        assert (
            inputs["transformersVisionSources"][name]
            == hashlib.sha256(
                (Path(modeling_auto.__file__).parents[2] / name).read_bytes()
            ).hexdigest()
        )
    assert (
        inputs["orynNativeSources"]["tips_tasks.py"]
        == hashlib.sha256(Path(binding.__file__).read_bytes()).hexdigest()
    )
