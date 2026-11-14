"""Closed MessagePack rejection boundaries and independent upstream graph parity."""

import io
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from oryn_engine import flax_weights
from oryn_engine.model_layouts import assess_layout

FIXTURES = Path(__file__).parent / "fixtures/flax"


def string(value):
    raw = value.encode("ascii")
    return bytes([0xA0 + len(raw)]) + raw


def tensor(shape=(1,), dtype="float32", data=b"\0\0\0\0", extension=1):
    body = b"\x93" + bytes([0x90 + len(shape), *shape]) + string(dtype)
    body += b"\xc4" + bytes([len(data)]) + data
    return b"\xc7" + bytes([len(body), extension]) + body


def checkpoint(value):
    return b"\x81" + string("weight") + value


def test_parser_accepts_only_complete_numeric_leaves(tmp_path):
    path = tmp_path / "flax_model.msgpack"
    path.write_bytes(checkpoint(tensor()))
    spec = flax_weights.inspect(path)["weight"]
    assert spec["shape"] == (1,) and spec["dtype"] == "float32"
    assert path.read_bytes()[spec["offset"] :] == b"\0" * 4
    raw = path.read_bytes()
    for length in range(len(raw)):
        path.write_bytes(raw[:length])
        with pytest.raises(ValueError):
            flax_weights.inspect(path)
    path.write_bytes(raw + b"\0")
    with pytest.raises(ValueError, match="TRAILING"):
        flax_weights.inspect(path)


@pytest.mark.parametrize(
    "value",
    [
        b"\xc0",
        b"\xc2",
        b"\x01",
        b"\x80",
        b"\x90",
        b"\xc4\x01x",
        tensor(extension=2),
        tensor(extension=3),
        tensor(extension=255),
        tensor(shape=()),
        tensor(shape=(0,)),
        tensor(shape=(1,) * 7),
        tensor(dtype="object"),
        tensor(dtype="complex64"),
        tensor(dtype="int32"),
        tensor(shape=(2,)),
        tensor(data=b"\0" * 3),
        b"\x82" + string("a") + tensor() + string("a") + tensor(),
        b"\x81" + string("..") + tensor(),
        b"\xdf\xff\xff\xff\xff",
    ],
)
def test_parser_rejects_objects_extensions_aliases_and_wrong_sizes(tmp_path, value):
    path = tmp_path / "flax_model.msgpack"
    path.write_bytes(checkpoint(value))
    with pytest.raises(ValueError):
        flax_weights.inspect(path)


def test_parser_rejects_depth_file_size_and_symlink(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes((b"\x81" + string("a")) * 26 + tensor())
    with pytest.raises(ValueError, match="TREE_LIMIT"):
        flax_weights.inspect(path)
    path.write_bytes(checkpoint(tensor()))
    link = tmp_path / "flax_model.msgpack"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="FILE_SCOPE"):
        flax_weights.inspect(link)
    for size in [0, 4 * 1024**3 + 1]:
        with pytest.raises(ValueError, match="FILE_LIMIT"):
            flax_weights.Reader(io.BytesIO(b""), size)


@pytest.mark.parametrize("family", ["bart", "t5", "mt5", "pegasus"])
def test_original_upstream_flax_parameters_match_native_logits(family):
    root = FIXTURES / family
    expected = json.loads((root / "forward.json").read_text())
    model = flax_weights.load_seq2seq(root, "summarization")
    inputs = torch.tensor(expected["input_ids"])
    with torch.inference_mode():
        actual = model(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            decoder_input_ids=torch.tensor(expected["decoder_input_ids"]),
        ).logits
    np.testing.assert_allclose(actual.numpy(), expected["logits"], rtol=2e-5, atol=2e-5)
    assert not any(p.is_meta for p in model.parameters())


def test_original_upstream_flax_bert_matches_native_classifier_logits():
    root = FIXTURES / "bert"
    expected = json.loads((root / "forward.json").read_text())
    model = flax_weights.load_native(root, "text-classification")
    inputs = torch.tensor(expected["input_ids"])
    with torch.inference_mode():
        logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits
    np.testing.assert_allclose(logits.numpy(), expected["logits"], rtol=2e-5, atol=2e-5)
    assert not any(p.is_meta for p in model.parameters())


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_original_flax_wav2vec_ctc_matches_native_forward(backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    root = FIXTURES / "wav2vec2"
    expected = json.loads((root / "forward.json").read_text())
    model = flax_weights.load_native(root, "automatic-speech-recognition").to(backend)
    inputs = torch.tensor(expected["input_values"], device=backend)
    with torch.inference_mode():
        logits = model(
            input_values=inputs, attention_mask=torch.ones_like(inputs, dtype=torch.long)
        ).logits
    np.testing.assert_allclose(logits.cpu().numpy(), expected["logits"], rtol=3e-5, atol=3e-5)


def save_ctc_processor(root):
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "A": 5, "B": 6, "C": 7}
    (root / "vocab.json").write_text(json.dumps(vocab))
    Wav2Vec2CTCTokenizer(str(root / "vocab.json")).save_pretrained(root)
    Wav2Vec2FeatureExtractor(
        sampling_rate=16000, feature_size=1, do_normalize=True, return_attention_mask=True
    ).save_pretrained(root)


def test_flax_ctc_and_safetensors_share_transcription_and_preserve_original_weights(tmp_path):
    import wave

    from oryn_engine import shared_tasks

    shutil.copytree(FIXTURES / "wav2vec2", tmp_path, dirs_exist_ok=True)
    save_ctc_processor(tmp_path)
    model, processor, config = shared_tasks.load_task(
        tmp_path, "automatic-speech-recognition", "cpu"
    )
    path = tmp_path / "input.wav"
    pcm = (np.sin(np.arange(800) * 0.2) * 10000).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm.tobytes())
    value = {"audioArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa"}
    events = []

    def execute():
        return shared_tasks.execute_task(
            tmp_path,
            "automatic-speech-recognition",
            value,
            "cpu",
            path,
            None,
            lambda name, data: events.append((name, data)),
        )

    result = execute()
    shared_tasks.validate_task_output("automatic-speech-recognition", result)
    assert isinstance(result["text"], str) and result["language"] is None
    assert any(name == "job.state" and data["state"] == "running" for name, data in events)
    model.save_pretrained(tmp_path)
    # Presence of the alternative Flax file must not change a valid Safetensors route.
    (tmp_path / "flax_model.msgpack").write_bytes(b"not numeric weights")
    assert execute() == result


@pytest.mark.parametrize(
    "change",
    [
        {"do_stable_layer_norm": False},
        {"feat_extract_norm": "group"},
        {"add_adapter": True},
        {"conv_dim": [8]},
        {"conv_stride": [0, 2]},
        {"hidden_size": 15},
        {"num_conv_pos_embedding_groups": 3},
        {"mask_time_prob": 0},
        {"pad_token_id": 8},
    ],
)
def test_flax_ctc_rejects_unreviewed_native_graphs(change):
    config = json.loads((FIXTURES / "wav2vec2/config.json").read_text())
    with pytest.raises(ValueError):
        flax_weights.validate_config(config | change, "automatic-speech-recognition")


@pytest.mark.parametrize("mutation", ["head", "positional", "convolution", "shape"])
def test_flax_ctc_requires_complete_original_tensors(tmp_path, mutation):
    shutil.copytree(FIXTURES / "wav2vec2", tmp_path, dirs_exist_ok=True)
    path = tmp_path / "flax_model.msgpack"
    raw = path.read_bytes()
    if mutation == "shape":
        config = json.loads((tmp_path / "config.json").read_text())
        config["vocab_size"] += 1
        (tmp_path / "config.json").write_text(json.dumps(config))
    else:
        name = {"head": b"lm_head", "positional": b"weight_v", "convolution": b"conv_layers"}[
            mutation
        ]
        raw = raw.replace(name, b"x" * len(name))
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="GRAPH_MISMATCH|KERNEL_UNSUPPORTED"):
        flax_weights.load_native(tmp_path, "automatic-speech-recognition")


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_original_upstream_flax_clip_matches_both_embeddings_and_logits(backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    root = FIXTURES / "clip"
    expected = json.loads((root / "forward.json").read_text())
    model = flax_weights.load_native(root, "zero-shot-image-classification").to(backend)
    inputs = torch.tensor(expected["input_ids"], device=backend)
    with torch.inference_mode():
        output = model(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            pixel_values=torch.tensor(expected["pixel_values"], device=backend),
        )
    for name in ("logits_per_image", "image_embeds", "text_embeds"):
        np.testing.assert_allclose(
            getattr(output, name).cpu().numpy(), expected[name], rtol=3e-5, atol=3e-5
        )
    assert not any(p.is_meta for p in model.parameters())
    assert not torch.allclose(output.image_embeds[0], output.image_embeds[1])


@pytest.mark.parametrize(
    "mutation", ["missing", "vision", "text", "projection", "geometry", "nonfinite"]
)
def test_flax_clip_rejects_incomplete_or_mismatched_original_graph(tmp_path, mutation):
    shutil.copytree(FIXTURES / "clip", tmp_path, dirs_exist_ok=True)
    path = tmp_path / "flax_model.msgpack"
    raw = path.read_bytes()
    if mutation == "missing":
        raw = checkpoint(tensor())
    elif mutation in {"vision", "text", "projection"}:
        name = {
            "vision": b"vision_model",
            "text": b"text_model",
            "projection": b"visual_projection",
        }[mutation]
        raw = raw.replace(name, b"x" * len(name))
    elif mutation == "geometry":
        config = json.loads((tmp_path / "config.json").read_text())
        config["vision_config"]["image_size"] = 12
        (tmp_path / "config.json").write_text(json.dumps(config))
    else:
        spec = flax_weights.inspect(path)["logit_scale"]
        assert spec["shape"] == ()
        raw = raw[: spec["offset"]] + struct.pack("<f", float("nan")) + raw[spec["offset"] + 4 :]
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="GRAPH_MISMATCH|NONFINITE"):
        flax_weights.load_native(tmp_path, "zero-shot-image-classification")


def save_clip_processor(root):
    from transformers import CLIPImageProcessor, CLIPProcessor, CLIPTokenizer

    tokenizer = CLIPTokenizer(
        vocab={"<|startoftext|>": 0, "<pad>": 1, "<|endoftext|>": 2, "a</w>": 3, "b</w>": 4},
        merges=[],
        pad_token="<pad>",
    )
    image_processor = CLIPImageProcessor(
        size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8}
    )
    CLIPProcessor(tokenizer=tokenizer, image_processor=image_processor).save_pretrained(root)
    image_processor.save_pretrained(root)


def test_flax_clip_uses_native_processor_and_existing_result_contract(tmp_path):
    from PIL import Image

    from oryn_engine import shared_tasks

    shutil.copytree(FIXTURES / "clip", tmp_path, dirs_exist_ok=True)
    save_clip_processor(tmp_path)
    image = Image.new("RGB", (12, 9), (20, 100, 250))
    model, processor, config = shared_tasks.load_task(
        tmp_path, "zero-shot-image-classification", "cpu"
    )
    value = shared_tasks.validate_task_input(
        "zero-shot-image-classification",
        {"labels": ["a", "b"], "imageArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa"},
    )
    with torch.inference_mode():
        result = flax_weights.execute(model, processor, value, "cpu", config, image)
    shared_tasks.validate_task_output("zero-shot-image-classification", result)
    assert {p["label"] for p in result["predictions"]} == {"a", "b"}
    assert sum(p["score"] for p in result["predictions"]) == pytest.approx(1)
    image_path = tmp_path / "input.png"
    image.save(image_path)
    events = []
    executed = shared_tasks.execute_task(
        tmp_path,
        "zero-shot-image-classification",
        value,
        "cpu",
        image_path,
        None,
        lambda name, payload: events.append((name, payload)),
    )
    assert executed == result
    assert ("job.state", {"state": "running"}) in events
    with pytest.raises(ValueError, match="TOKEN_LIMIT"):
        flax_weights.execute(
            model, processor, {**value, "labels": ["a " * 30, "b"]}, "cpu", config, image
        )


def test_existing_safetensors_clip_keeps_its_native_runner(tmp_path):
    from types import SimpleNamespace

    from oryn_engine.runners import RunnerRegistry, TransformersTaskHandle

    model = flax_weights.load_native(FIXTURES / "clip", "zero-shot-image-classification")
    model.save_pretrained(tmp_path)
    save_clip_processor(tmp_path)
    registry = RunnerRegistry()
    registry._pack_locator = SimpleNamespace(resolve=lambda _: pytest.fail("Legacy route changed"))
    handle = registry._load_single(
        {
            "model": {"provider": "huggingface", "repository": str(tmp_path), "revision": "a" * 40},
            "runner": "transformers",
            "cacheRoot": str(tmp_path),
            "technicalCapabilities": {
                "task": "zero-shot-image-classification",
                "architectures": ["CLIPModel"],
            },
            "_runtimeCandidate": {"backend": "cpu"},
        }
    )
    assert isinstance(handle, TransformersTaskHandle)
    assert handle.task == "zero-shot-image-classification"
    expected = json.loads((FIXTURES / "clip/forward.json").read_text())
    ids, pixels = torch.tensor(expected["input_ids"]), torch.tensor(expected["pixel_values"])
    torch.testing.assert_close(
        handle.model(input_ids=ids, pixel_values=pixels).logits_per_image,
        model(input_ids=ids, pixel_values=pixels).logits_per_image,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"do_resize": False},
        {"do_center_crop": False},
        {"do_rescale": False},
        {"image_std": [1, 0, 1]},
        {"image_mean": [1, 2]},
        {"size": 1000000},
        {"crop_size": 16},
        {"rescale_factor": float("nan")},
        {"resample": 2},
        {"image_processor_type": "RepositoryImageProcessor"},
    ],
)
def test_flax_clip_processor_rejects_unreviewed_geometry_or_operations(change):
    with pytest.raises(ValueError):
        flax_weights.clip_processor_options({"crop_size": 8, **change}, 8)


def test_flax_clip_legacy_alias_cannot_override_the_validated_graph(tmp_path):
    shutil.copytree(FIXTURES / "clip", tmp_path, dirs_exist_ok=True)
    path = tmp_path / "config.json"
    config = json.loads(path.read_text())
    config["vision_config_dict"] = dict(config["vision_config"])
    path.write_text(json.dumps(config))
    model = flax_weights.load_native(tmp_path, "zero-shot-image-classification")
    assert model.config.vision_config.hidden_size == 16
    for legacy in (
        {"patch_size": 4},
        {**config["vision_config"], "hidden_act": "gelu"},
        {"patch_size": 8},
    ):
        with pytest.raises(ValueError, match="CONFIGURATION"):
            flax_weights.validate_config(
                config | {"vision_config_dict": legacy}, "zero-shot-image-classification"
            )


def test_clip_metadata_defaults_match_pinned_native_and_do_not_import_backend():
    import subprocess
    import sys

    from transformers import CLIPTextConfig, CLIPVisionConfig

    for name, native in (("text_config", CLIPTextConfig()), ("vision_config", CLIPVisionConfig())):
        assert all(
            getattr(native, key) == value for key, value in flax_weights.CLIP_DEFAULTS[name].items()
        )
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; from oryn_engine.flax_weights import validate_config; "
                "config=json.load(open(sys.argv[1])); "
                "config['vision_config_dict']=config['vision_config']; "
                "validate_config(config, 'zero-shot-image-classification'); "
                "assert 'transformers' not in sys.modules and 'torch' not in sys.modules"
            ),
            str(FIXTURES / "clip/config.json"),
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "part,key,value",
    [
        ("vision_config", "num_channels", 4),
        ("vision_config", "patch_size", 3),
        ("vision_config", "image_size", 4096),
        ("vision_config", "hidden_act", "custom"),
        ("text_config", "num_attention_heads", 3),
        ("text_config", "max_position_embeddings", 10000),
        ("text_config", "eos_token_id", 40),
        ("text_config", "is_decoder", True),
    ],
)
def test_flax_clip_config_rejects_unreviewed_geometry(part, key, value):
    config = json.loads((FIXTURES / "clip/config.json").read_text())
    config[part][key] = value
    with pytest.raises(ValueError):
        flax_weights.validate_config(config, "zero-shot-image-classification")


def test_flax_clip_layout_requires_both_tokenizer_and_exact_image_processor():
    config = json.loads((FIXTURES / "clip/config.json").read_text())
    files = [
        "config.json",
        "flax_model.msgpack",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
    ]
    valid = assess_layout(
        files, config, "zero-shot-image-classification", source_kind="transformers"
    )
    assert not valid.artifact_reasons and not valid.runtime_reasons
    for missing in ("vocab.json", "preprocessor_config.json"):
        selected = [name for name in files if name != missing]
        layout = assess_layout(
            selected + ["processor_config.json"],
            config,
            "zero-shot-image-classification",
            source_kind="transformers",
        )
        assert layout.artifact_reasons


def test_flax_bert_loads_local_tokenizer_and_reuses_classification_contract(tmp_path):
    from transformers import BertTokenizer

    from oryn_engine import shared_tasks

    shutil.copytree(FIXTURES / "bert", tmp_path, dirs_exist_ok=True)
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "this", "is", "good"]
    tokenizer = BertTokenizer(vocab={word: i for i, word in enumerate(words)})
    tokenizer.save_pretrained(tmp_path)
    value = shared_tasks.validate_task_input("text-classification", {"text": "this is good"})
    assert value == {"text": "this is good", "topK": 5}
    model, processor, config = shared_tasks.load_task(tmp_path, "text-classification", "cpu")
    with torch.inference_mode():
        result = flax_weights.execute(model, processor, value, "cpu", config)
    assert len(result["predictions"]) == 3
    assert {p["label"] for p in result["predictions"]} == {"negative", "neutral", "positive"}
    assert sum(p["score"] for p in result["predictions"]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="TOKEN_LIMIT"):
        flax_weights.execute(model, processor, {"text": "good " * 40, "topK": 1}, "cpu", config)


def test_existing_safetensors_classifier_keeps_its_native_runner(tmp_path):
    from types import SimpleNamespace

    from transformers import BertTokenizer

    from oryn_engine.runners import RunnerRegistry, TransformersTaskHandle

    model = flax_weights.load_native(FIXTURES / "bert", "text-classification")
    model.save_pretrained(tmp_path)
    BertTokenizer(vocab={"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}).save_pretrained(tmp_path)
    registry = RunnerRegistry()
    registry._pack_locator = SimpleNamespace(resolve=lambda _: pytest.fail("Legacy route changed"))
    handle = registry._load_single(
        {
            "model": {"provider": "huggingface", "repository": str(tmp_path), "revision": "a" * 40},
            "runner": "transformers",
            "cacheRoot": str(tmp_path),
            "technicalCapabilities": {
                "task": "text-classification",
                "architectures": ["BertForSequenceClassification"],
            },
            "_runtimeCandidate": {"backend": "cpu"},
        }
    )
    assert isinstance(handle, TransformersTaskHandle)
    assert handle.task == "text-classification" and handle.backend == "cpu"
    expected = model(torch.tensor([[4, 5, 1]])).logits
    actual = handle.model(torch.tensor([[4, 5, 1]])).logits
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("mutation", ["head", "missing", "shape", "labels", "family"])
def test_flax_classifier_requires_original_complete_graph_and_label_semantics(tmp_path, mutation):
    shutil.copytree(FIXTURES / "bert", tmp_path, dirs_exist_ok=True)
    path = tmp_path / "flax_model.msgpack"
    config = json.loads((tmp_path / "config.json").read_text())
    if mutation == "head":
        path.write_bytes(path.read_bytes().replace(b"classifier", b"wrong_head"))
    elif mutation == "missing":
        path.write_bytes(checkpoint(tensor()))
    elif mutation == "shape":
        config["vocab_size"] += 1
    elif mutation == "labels":
        config["id2label"] = {"0": "", "1": "neutral", "2": "positive"}
    else:
        config["architectures"] = ["BertModel"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="GRAPH_MISMATCH|CONFIGURATION"):
        flax_weights.load_native(tmp_path, "text-classification")


@pytest.mark.parametrize(
    "change",
    [
        {"num_labels": 4},
        {"pad_token_id": 32},
        {"num_attention_heads": 3},
        {"hidden_size": 100000},
        {"is_decoder": True},
        {"add_cross_attention": True},
        {"position_embedding_type": "relative_key"},
    ],
)
def test_flax_classifier_rejects_unreviewed_geometry(change):
    config = json.loads((FIXTURES / "bert/config.json").read_text())
    with pytest.raises(ValueError, match="CONFIGURATION|dimensions exceed"):
        flax_weights.validate_config(config | change, "text-classification")


@pytest.mark.parametrize("mutation", ["nan", "missing", "unknown", "shape"])
@pytest.mark.parametrize("family", ["bart", "pegasus"])
def test_native_loader_rejects_nonfinite_and_mismatched_graph(tmp_path, mutation, family):
    shutil.copyfile(FIXTURES / family / "config.json", tmp_path / "config.json")
    path = tmp_path / "flax_model.msgpack"
    raw = bytearray((FIXTURES / family / "flax_model.msgpack").read_bytes())
    if mutation == "nan":
        path.write_bytes(raw)
        item = next(iter(flax_weights.inspect(path).values()))
        raw[item["offset"] : item["offset"] + 4] = struct.pack("<f", float("nan"))
    elif mutation == "missing":
        raw = checkpoint(tensor())
    elif mutation == "unknown":
        raw = raw.replace(b"final_logits_bias", b"other_logits_bias")
    else:
        config = json.loads((tmp_path / "config.json").read_text())
        config["vocab_size"] += 1
        (tmp_path / "config.json").write_text(json.dumps(config))
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="NONFINITE|GRAPH_MISMATCH"):
        flax_weights.load_seq2seq(tmp_path, "summarization")


@pytest.mark.parametrize(
    "change",
    [
        {"model_type": "unreviewed"},
        {"d_model": 2**20},
        {"num_beams": 9},
        {"num_layers": True},
        {"min_length": 10000},
        {"tie_word_embeddings": "yes"},
        {"eos_token_id": -1},
        {"forced_bos_token_id": 32},
        {"auto_map": {"AutoModel": "repo.Code"}},
        {"quantization_config": {"bits": 4}},
    ],
)
def test_config_rejects_unreviewed_families_options_and_dimensions(change):
    config = json.loads((FIXTURES / "bart/config.json").read_text())
    with pytest.raises(ValueError):
        flax_weights.validate_config(config | change, "summarization")


def test_layout_never_borrows_flax_to_complete_broken_safetensors():
    config = json.loads((FIXTURES / "bart/config.json").read_text())
    names = ["config.json", "tokenizer.json", "flax_model.msgpack", "pytorch_model.bin", "model.py"]
    good = assess_layout(names, config, "summarization")
    assert not good.artifact_reasons and not good.runtime_reasons
    assert set(good.selected_files) == set(names[:3])
    bad = assess_layout(names + ["model-00001-of-00002.safetensors"], config, "summarization")
    assert "MODEL_SUBSET_INCOMPLETE" in bad.artifact_reasons
    assert "flax_model.msgpack" not in bad.selected_files
    bad = assess_layout(names, config, "translation")
    assert "MODEL_SUBSET_INCOMPLETE" in bad.artifact_reasons
    legacy = assess_layout(names + ["model.safetensors"], config, "summarization")
    assert "flax_model.msgpack" not in legacy.selected_files


def test_flax_inspection_crosses_protocol_and_preserves_legacy_seq2seq(tmp_path):
    from types import SimpleNamespace

    from oryn_protocol import ProtocolSchemaStore

    from oryn_engine.model_support import transformers_workload
    from oryn_engine.models import ModelFile, ModelStore
    from oryn_engine.task_profiles import profile_for_execution

    config = json.loads((FIXTURES / "bart/config.json").read_text())
    hub = SimpleNamespace(read_json=lambda *args: {})
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    profile = store._inspect_generated_pack(
        "org/bart",
        "a" * 40,
        [
            ModelFile(name, 100, "1" * 64)
            for name in ["config.json", "flax_model.msgpack", "tokenizer.json"]
        ],
        config,
        "summarization",
        True,
        source_kind="transformers",
    )
    profile["taskProfile"] = profile_for_execution(profile, config)
    ProtocolSchemaStore().validator("candidateExecutionProfile").validate(profile)
    assert profile["taskProfile"]["parameters"]["maxTokens"]["default"] == 128
    assert profile["runtimePack"]["runnerId"] == "transformers.shared-tasks"
    assert transformers_workload(["BartForConditionalGeneration"], "summarization") == "seq2seq"


def test_legacy_t5_base_uses_complete_tied_head_and_upstream_logits(tmp_path):
    from oryn_engine.model_support import transformers_workload

    shutil.copyfile(FIXTURES / "t5/flax_model.msgpack", tmp_path / "flax_model.msgpack")
    config = json.loads((FIXTURES / "t5/config.json").read_text())
    config["architectures"] = ["T5Model"]
    config["task_specific_params"] = {
        "summarization": {
            "prefix": "function documentation generation python: ",
            "num_beams": 4,
            "max_length": 512,
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    model = flax_weights.load_seq2seq(tmp_path, "summarization")
    expected = json.loads((FIXTURES / "t5/forward.json").read_text())
    inputs = torch.tensor(expected["input_ids"])
    with torch.inference_mode():
        actual = model(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            decoder_input_ids=torch.tensor(expected["decoder_input_ids"]),
        ).logits
    np.testing.assert_allclose(actual.numpy(), expected["logits"], rtol=2e-5, atol=2e-5)
    assert model.lm_head.weight is model.shared.weight
    assert model.generation_config.num_beams == 4
    assert transformers_workload(["T5Model"], "summarization") is None
    with pytest.raises(ValueError):
        flax_weights.validate_config(config | {"tie_word_embeddings": False}, "summarization")


def test_summary_prefix_is_literal_and_included_in_input_budget():
    from unittest.mock import Mock

    tokenizer = Mock(return_value={"input_ids": torch.ones((1, 4), dtype=torch.long)})
    tokenizer.decode.return_value = "An explanation."
    model = Mock()
    model.generate.return_value = torch.tensor([[1, 2]])
    config = {"task_specific_params": {"summarization": {"prefix": "describe: ", "max_length": 12}}}
    assert flax_weights.execute(model, tokenizer, {"text": "def f(): pass"}, "cpu", config) == {
        "text": "An explanation."
    }
    tokenizer.assert_called_once_with(
        "describe: def f(): pass", return_tensors="pt", truncation=False
    )
    assert model.generate.call_args.kwargs["max_new_tokens"] == 11
    flax_weights.execute(model, tokenizer, {"text": "code", "maxTokens": 20}, "cpu", config)
    assert model.generate.call_args.kwargs["max_new_tokens"] == 20
    tokenizer.return_value = {"input_ids": torch.ones((1, 513), dtype=torch.long)}
    with pytest.raises(ValueError, match="TOKEN_LIMIT"):
        flax_weights.execute(model, tokenizer, {"text": "code"}, "cpu", config)


@pytest.mark.parametrize(
    "options",
    [
        {"prefix": ["code"]},
        {"prefix": "x" * 513},
        {"prefix": "x\0"},
        {"num_beams": 100},
        {"max_length": 100000},
        {"min_length": True},
        {"length_penalty": float("nan")},
        {"early_stopping": "never"},
        {"custom_generate": "repo.py"},
        {"do_sample": True},
    ],
)
def test_summary_task_settings_cannot_escape_closed_generation(options):
    with pytest.raises(ValueError):
        flax_weights.decoding_options({"task_specific_params": {"summarization": options}})
