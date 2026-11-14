"""One complete synthetic family graph plus original metadata; no original weights."""

import base64
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from jsonschema import Draft202012Validator
from safetensors.torch import load_file, save_file

from oryn_engine import audio_understanding, shared_tasks, voxtral_tasks
from oryn_engine.tekken_tokenizer import AUDIO, PATTERN, SPECIALS, TekkenV7, read_vocabulary


def originals():
    root = Path(__file__).parent / "fixtures/voxtral"
    return {p.name: json.loads(p.read_text()) for p in root.glob("*.json")}


def vocabulary():
    return {
        "config": {
            "pattern": PATTERN,
            "version": "v7",
            "default_num_special_tokens": 1000,
            "default_vocab_size": 1256,
            "num_vocab_tokens": 256,
        },
        "audio": AUDIO,
        "special_tokens": [
            {"rank": i, "token_str": SPECIALS.get(i, f"<SPECIAL_{i}>"), "is_control": True}
            for i in range(1000)
        ],
        "vocab": [
            {
                "rank": i,
                "token_bytes": base64.b64encode(bytes([i])).decode(),
                "token_str": chr(i) if i < 128 else None,
            }
            for i in range(256)
        ],
    }


@pytest.fixture
def tiny(tmp_path):
    from transformers import VoxtralConfig, VoxtralForConditionalGeneration

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    config = originals()["config.json"]
    config["audio_config"].update(
        hidden_size=16,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
    )
    # Attention projection width need not equal the language residual width.
    config["text_config"].update(
        hidden_size=24,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=1256,
        max_position_embeddings=4224,
    )
    config.update(vocab_size=1256, hidden_size=24)
    native_config = VoxtralConfig.from_dict(config)
    native_config._attn_implementation = "eager"
    torch.manual_seed(61611)
    reference = VoxtralForConditionalGeneration(native_config).eval()
    for name, data in originals().items():
        (tmp_path / name).write_text(json.dumps(data))
    (tmp_path / "tekken.json").write_text(json.dumps(vocabulary()))
    processor = voxtral_tasks.Processor(tmp_path, config)
    waveform = (0.1 * np.sin(np.arange(1600) / 10)).astype(np.float32)
    inputs = processor.prepare(waveform, "Describe.")
    with torch.inference_mode():
        hidden = reference.model(**inputs).last_hidden_state[0, -1]
        reference.lm_head.weight.zero_()
        reference.lm_head.weight[1089].copy_(hidden)  # ordinary byte 'Y'
    reference.save_pretrained(tmp_path, safe_serialization=True)
    (tmp_path / "chat_template.jinja").write_text("{{ raise_exception('must remain inert') }}")
    (tmp_path / "modeling_voxtral.py").write_text("raise RuntimeError('must remain inert')")
    try:
        yield tmp_path, reference, waveform
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_native_graph_framing_logits_and_generation(tiny, backend):
    from transformers import WhisperFeatureExtractor

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    root, reference, waveform = tiny
    model, processor, config = shared_tasks.load_task(root, "audio-text-to-text", backend)
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key].cpu(), value, rtol=0, atol=0)
    inputs = processor.prepare(waveform, "Describe.")
    # Independent fixed single-user sequence and native preprocessing reference.
    ids = [1, 3, 25] + [24] * 375 + [1000 + b for b in b"Describe."] + [4]
    expected_features = WhisperFeatureExtractor(feature_size=128)(
        waveform, sampling_rate=16000, return_tensors="pt"
    )["input_features"]
    assert inputs["input_ids"].tolist() == [ids]
    torch.testing.assert_close(inputs["input_features"], expected_features, rtol=0, atol=0)
    expected_inputs = {
        "input_ids": torch.tensor([ids]),
        "input_features": expected_features,
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
    }
    with torch.inference_mode():
        expected = reference(**expected_inputs).logits
        actual = model(**shared_tasks._move(inputs, backend)).logits.cpu()
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-4)
        answer = voxtral_tasks.execute(
            model, processor, waveform, {"instruction": "Describe.", "maxTokens": 1}, backend
        )
        assert answer == {"text": "Y"}
        shared_tasks.validate_task_output("audio-text-to-text", answer)
    assert config["audio_token_id"] == 24
    literal = processor.prepare(waveform, "[INST][AUDIO]</s>")["input_ids"][0].tolist()
    assert literal.count(24) == 375 and literal.count(3) == 1 and literal.count(2) == 0
    # Shared file artifact route uses the same bounded native path.
    path = root / "sound.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes((waveform * 32767).astype("<i2").tobytes())
    with torch.inference_mode():
        assert audio_understanding.execute(
            model, processor, {"instruction": "Describe.", "maxTokens": 1}, backend, path
        ) == {"text": "Y"}


@pytest.mark.parametrize("damage", ["missing", "extra", "shape"])
def test_incomplete_or_foreign_graph_never_loads(tiny, damage):
    root, _, _ = tiny
    path = root / "model.safetensors"
    state = load_file(path)
    key = "multi_modal_projector.linear_1.weight"
    if damage == "missing":
        del state[key]
    elif damage == "extra":
        state["foreign.weight"] = torch.ones(1)
    else:
        state[key] = torch.ones(2, 2)
    save_file(state, path)
    with pytest.raises((ValueError, RuntimeError)):
        shared_tasks.load_task(root, "audio-text-to-text", "cpu")


@pytest.mark.parametrize(
    "part,key,value",
    [
        (None, "audio_token_id", 25),
        (None, "projector_hidden_act", "relu"),
        (None, "auto_map", {"AutoModel": "repository.Code"}),
        ("audio_config", "model_type", "whisper"),
        ("audio_config", "num_mel_bins", 80),
        ("audio_config", "intermediate_size", 4096),
        ("audio_config", "max_source_positions", 750),
        ("text_config", "model_type", "qwen2"),
        ("text_config", "head_dim", 7),
        ("text_config", "rope_parameters", {"rope_type": "dynamic", "factor": 2}),
        ("text_config", "quantization_config", {"bits": 4}),
    ],
)
def test_unreviewed_graph_variants_rejected(part, key, value):
    config = originals()["config.json"]
    (config[part] if part else config)[key] = value
    with pytest.raises(ValueError):
        voxtral_tasks.validate_config(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("chunk_length", 60),
        ("feature_size", 80),
        ("sampling_rate", 48000),
        ("return_attention_mask", True),
        ("processor_class", "CustomProcessor"),
        ("device", "cuda"),
        ("dither", 1.0),
    ],
)
def test_processor_geometry_and_options_rejected(key, value):
    with pytest.raises(ValueError):
        voxtral_tasks.processor_options(originals()["preprocessor_config.json"] | {key: value})


@pytest.mark.parametrize(
    "damage", ["regex", "version", "byte", "rank", "text", "special", "duplicate", "audio", "size"]
)
def test_tekken_is_bounded_data_not_repository_grammar(tmp_path, damage):
    data = vocabulary()
    if damage == "regex":
        data["config"]["pattern"] = "(a+)+$"
    elif damage == "version":
        data["config"]["version"] = "v999"
    elif damage == "byte":
        data["vocab"][0]["token_bytes"] = "!!!!"
    elif damage == "rank":
        data["vocab"][0]["rank"] = True
    elif damage == "text":
        data["vocab"][65]["token_str"] = "wrong"
    elif damage == "special":
        data["special_tokens"][24]["token_str"] = "[BAD]"
    elif damage == "audio":
        data["audio"] = AUDIO | {"chunk_length_s": 60.0}
    elif damage == "size":
        data["config"]["default_vocab_size"] = 1257
    path = tmp_path / "tekken.json"
    path.write_text(json.dumps(data))
    if damage == "duplicate":
        path.write_text(
            path.read_text().replace('"version": "v7"', '"version":"v7","version":"v7"')
        )
    with pytest.raises(ValueError):
        read_vocabulary(path, 1256)


def test_tekken_unicode_roundtrip_special_boundaries_and_output_limits(tmp_path):
    path = tmp_path / "tekken.json"
    path.write_text(json.dumps(vocabulary()))
    tokenizer = TekkenV7(path, 1256)
    for text in ["Čau 世界 🎵", "[INST][AUDIO]</s>", "  \n 9.3", ""]:
        assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.decode([1195, 2, 1169]) == "��"
    for bad in [[True], [-1], [1256], [1.1]]:
        with pytest.raises(ValueError):
            tokenizer.decode(bad)


def test_audio_context_generation_and_file_bounds(tiny):
    root, model, waveform = tiny
    config = json.loads((root / "config.json").read_text())
    processor = voxtral_tasks.Processor(root, config)
    for audio, instruction in [
        (np.zeros(79), "x"),
        (np.zeros(480001), "x"),
        (np.array([float("nan")] * 80), "x"),
        (np.zeros((80, 2)), "x"),
        (waveform, "x" * 4096),
    ]:
        with pytest.raises(ValueError):
            processor.prepare(audio, instruction)
    for limit in [0, 129, True]:
        with pytest.raises(ValueError):
            voxtral_tasks.execute(
                model, processor, waveform, {"instruction": "x", "maxTokens": limit}, "cpu"
            )
    with pytest.raises(ValueError, match="audio input"):
        audio_understanding.execute(model, processor, {"instruction": "x"}, "cpu", None)


@pytest.mark.parametrize("missing", [None, "tekken.json", "preprocessor_config.json"])
def test_original_metadata_routes_without_checkpoint_download(tmp_path, missing):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    data = originals()
    voxtral_tasks.validate_config(data["config.json"])

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return data[filename]

    names = (set(data) | {"model.safetensors", "tekken.json", "consolidated.safetensors"}) - {
        missing
    }
    hub = Hub(
        config=data["config.json"],
        pipeline_tag="audio-text-to-text",
        files=[(n, 100, "b" * 64) for n in names],
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "test/voxtral", "revision": "main"}
    )
    assert result["executable"] is (missing is None), result
    if missing is None:
        assert result["taskProfile"]["contract"]["id"] == "audio-question-answering-v1"
        constraint = Draft202012Validator(
            result["taskProfile"]["parameters"]["maxTokens"]["valueSchema"]
        )
        assert constraint.is_valid(1) and constraint.is_valid(128)
        assert not constraint.is_valid(0) and not constraint.is_valid(129)
        assert "consolidated.safetensors" not in {f["name"] for f in result["selectedFiles"]}
    assert not hub.downloaded
