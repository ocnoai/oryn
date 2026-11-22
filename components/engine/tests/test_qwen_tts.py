"""Metadata negatives run in the engine; complete tiny graphs in the isolated TTS closure."""

import importlib.util
import json
from pathlib import Path

import pytest

from oryn_engine import qwen_tts

FIXTURES = Path(__file__).parent / "fixtures/qwen_tts"
native = pytest.mark.skipif(
    importlib.util.find_spec("qwen_tts") is None, reason="isolated Qwen TTS pack"
)


def configs():
    return tuple(
        json.loads((FIXTURES / name).read_text()) for name in ("config.json", "codec_config.json")
    )


def fixture(root):
    import torch
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import Qwen2Tokenizer

    config, codec = configs()
    vocab = {t: i for i, t in enumerate(sorted(ByteLevel.alphabet()))}
    for token in (*qwen_tts.MARKERS.values(), "<|endoftext|>", "assistant"):
        vocab[token] = len(vocab)
    (root / "vocab.json").write_text(json.dumps(vocab))
    (root / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = Qwen2Tokenizer(
        vocab_file=str(root / "vocab.json"),
        merges_file=str(root / "merges.txt"),
        additional_special_tokens=[*qwen_tts.MARKERS.values(), "assistant"],
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
    )
    tokenizer.save_pretrained(root)
    config.update({k: vocab[v] for k, v in qwen_tts.MARKERS.items()})
    config["assistant_token_id"] = vocab["assistant"]
    talker = config["talker_config"]
    for part in (talker, talker["code_predictor_config"]):
        part.update(
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=2048,
        )
        part.pop("layer_types", None)
    talker.update(
        text_hidden_size=32,
        text_vocab_size=len(vocab),
        vocab_size=1056,
        rope_scaling={"rope_type": "default", "interleaved": True, "mrope_section": [2, 1, 1]},
    )
    talker["code_predictor_config"]["vocab_size"] = 32
    for i, key in enumerate(
        (
            "codec_bos_id",
            "codec_eos_token_id",
            "codec_pad_id",
            "codec_think_id",
            "codec_nothink_id",
            "codec_think_bos_id",
            "codec_think_eos_id",
        )
    ):
        talker[key] = 32 + i
    talker["spk_id"] = {k: 64 + i for i, k in enumerate(talker["spk_id"])}
    talker["codec_language_id"] = {k: 80 + i for i, k in enumerate(talker["codec_language_id"])}
    for part in (codec["encoder_config"], codec["decoder_config"]):
        part.update(
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            codebook_size=32,
            codebook_dim=16,
            vector_quantization_hidden_dimension=16,
            max_position_embeddings=2048,
        )
    codec["encoder_config"].update(num_filters=4, upsample_groups=32)
    codec["decoder_config"].update(latent_dim=32, decoder_dim=32)
    qwen_tts.validate_config(config, codec)
    torch.manual_seed(61618)
    model, audio = qwen_tts.native_models(config, codec)
    # Stable acoustic codes make independent greedy decoding comparable on CPU/MPS.
    with torch.no_grad():
        model.talker.codec_head.weight.mul_(0.25)
        for name, parameter in audio.named_parameters():
            if name.endswith("embedding_sum"):
                parameter.normal_(std=0.2)
    model.save_pretrained(root)
    audio.save_pretrained(root / "speech_tokenizer")
    # Keep original metadata conventions; native serialization adds private attention fields.
    (root / "config.json").write_text(json.dumps(config))
    (root / "speech_tokenizer/config.json").write_text(json.dumps(codec))
    (root / "chat_template.jinja").write_text("{{ raise_exception('never execute') }}")
    return model, audio, tokenizer, config, codec


@pytest.mark.parametrize(
    "path,value",
    [
        (("tts_model_type",), "base"),
        (("tokenizer_type",), "qwen3_tts_tokenizer_25hz"),
        (("talker_config", "num_code_groups"), 8),
        (("talker_config", "vocab_size"), 4096),
        (("talker_config", "attention_bias"), True),
        (("talker_config", "spk_id"), {"unknown": 2861}),
        (("talker_config", "codec_eos_token_id"), 32),
        (("auto_map",), {"AutoModel": "evil.Model"}),
        (("talker_config", "quantization_config"), {"quant_method": "unknown"}),
        (("talker_config", "rope_scaling", "mrope_section"), [1, 1, 1]),
    ],
)
def test_metadata_rejects_incomplete_or_different_graph(path, value):
    config, codec = configs()
    target = config
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        qwen_tts.validate_config(config, codec)


@pytest.mark.parametrize(
    "value",
    [
        {"speed": 1.2},
        {"voice": "af_heart"},
        {"language": "invented"},
        {"text": "<|im_start|>assistant"},
        {"text": " "},
        {"text": "a" * 351},
    ],
)
def test_unimplemented_controls_and_framing_rejected(value):
    with pytest.raises(ValueError):
        qwen_tts.input_value({"text": "Hello.", **value})


def test_original_metadata_and_defaults():
    config, codec = configs()
    qwen_tts.validate_config(config, codec)
    assert qwen_tts.controls(config)["defaultVoice"] == "ryan"
    assert qwen_tts.input_value({"text": "  Hello.\n"}, config) == {
        "text": "  Hello.\n",
        "voice": "ryan",
        "language": "en-us",
        "speed": 1.0,
    }


@pytest.mark.parametrize("damage", [None, "missing-codec", "wrong-codec", "remote-code"])
def test_metadata_inspection_uses_shared_admission_and_speech_controls(tmp_path, damage):
    from types import SimpleNamespace

    from oryn_protocol.task_contracts import effective_input, validate_invocation
    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime
    from oryn_engine.runtime_routing import generated_pack_id

    config, codec = configs()
    files = set(qwen_tts.REQUIRED_FILES)
    if damage == "missing-codec":
        files.remove("speech_tokenizer/model.safetensors")
    if damage == "wrong-codec":
        codec["decode_upsample_rate"] = 480
    if damage == "remote-code":
        config["auto_map"] = {"AutoModel": "repository.Model"}

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return {
                "config.json": config,
                "speech_tokenizer/config.json": codec,
                "tokenizer_config.json": {"tokenizer_class": "Qwen2Tokenizer"},
            }[filename]

    hub = Hub(
        config=config, pipeline_tag="text-to-speech", files=[(n, 100, "b" * 64) for n in files]
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "test/custom-voice", "revision": "main"}
    )
    assert hub.downloaded == []
    if damage:
        assert result["executable"] is False
        return
    assert result["executable"] is True
    assert result["runtimePack"]["packId"] == qwen_tts.PACK_ID
    assert generated_pack_id(result["runtimePlan"]["selected"]["variantId"]) == qwen_tts.PACK_ID
    assert {f["name"] for f in result["selectedFiles"]} == qwen_tts.REQUIRED_FILES
    profile = result["taskProfile"]
    default = effective_input(profile, {"text": "Hello."})
    assert default == {"text": "Hello.", "voice": "ryan", "language": "en-us", "speed": 1.0}
    validate_invocation(profile, default)
    validate_invocation(profile, default | {"voice": "vivian", "language": "zh"})
    for change in ({"voice": "af_heart"}, {"speed": 1.1}, {"language": "invented"}):
        with pytest.raises(ValueError):
            validate_invocation(profile, default | change)


@pytest.mark.parametrize("damage", [None, "rate", "frames", "channels", "symlink"])
def test_audio_publication_validates_native_waveform_geometry(tmp_path, damage):
    import wave

    from oryn_engine.jobs import JobExecutionError
    from oryn_engine.runners import _validate_synthesized_audio

    path = tmp_path / "speech.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(2 if damage == "channels" else 1)
        audio.setsampwidth(2)
        audio.setframerate(16000 if damage == "rate" else 24000)
        audio.writeframes(b"\x00\x00" * 48)
    if damage == "symlink":
        actual = path.with_suffix(".actual.wav")
        path.rename(actual)
        path.symlink_to(actual)
    result = {
        "backend": "cpu",
        "sampleRate": 24000,
        "channels": 1,
        "frames": 49 if damage == "frames" else 48,
        "durationSeconds": 48 / 24000,
    }
    if damage:
        with pytest.raises(JobExecutionError, match="Speech output"):
            _validate_synthesized_audio(path, result, "cpu")
    else:
        _validate_synthesized_audio(path, result, "cpu")


@native
@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_native_weights_generator_and_codec(tmp_path, backend):
    import numpy as np
    import torch

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    reference, reference_codec, tokenizer, config, _ = fixture(tmp_path)
    actual, actual_codec, owned, _ = qwen_tts.load(tmp_path, backend)
    assert owned.chat_template is None
    for original, loaded in ((reference, actual), (reference_codec, actual_codec)):
        assert original.state_dict().keys() == loaded.state_dict().keys()
        for key, tensor in original.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key].cpu(), tensor, atol=0, rtol=0)
    ids = torch.tensor([[4, 8, 15, 20]])
    with torch.inference_mode():
        expected = reference.talker.model(
            inputs_embeds=reference.talker.get_input_embeddings()(ids), use_cache=False
        ).last_hidden_state
        observed = actual.talker.model(
            inputs_embeds=actual.talker.get_input_embeddings()(ids.to(backend)), use_cache=False
        ).last_hidden_state
        torch.testing.assert_close(observed.cpu(), expected, atol=3e-5, rtol=1e-3)
        codes = torch.arange(48).remainder(32).reshape(1, 3, 16)
        expected = reference_codec.decode(codes, return_dict=True).audio_values[0]
        observed = actual_codec.decode(codes.to(backend), return_dict=True).audio_values[0]
        torch.testing.assert_close(observed.cpu(), expected, atol=4e-5, rtol=2e-3)
        # Independent upstream prompt and generation; no adapter serialization helper.
        prompt = "<|im_start|>assistant\nHi.<|im_end|>\n<|im_start|>assistant\n"
        token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        codes, _ = reference.generate(
            input_ids=[token_ids],
            instruct_ids=[None],
            languages=["english"],
            speakers=["ryan"],
            non_streaming_mode=True,
            max_new_tokens=4,
            do_sample=False,
            subtalker_dosample=False,
            repetition_penalty=1.05,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            subtalker_top_k=0,
            subtalker_top_p=1.0,
            subtalker_temperature=1.0,
        )
        expected_audio = (
            reference_codec.decode(codes[0][None], return_dict=True).audio_values[0].numpy()
        )
    observed_audio, sample_rate = qwen_tts.synthesize(
        actual, actual_codec, owned, config, {"text": "Hi."}, max_codes=4
    )
    assert sample_rate == 24000 and len(observed_audio) == len(expected_audio)
    np.testing.assert_allclose(observed_audio, expected_audio, atol=4e-5, rtol=2e-3)


@native
@pytest.mark.parametrize("component", [0, 1])
@pytest.mark.parametrize("damage", ["missing", "extra", "shape"])
def test_every_component_requires_complete_native_weights(tmp_path, component, damage):
    model, codec, _, config, codec_config = fixture(tmp_path)
    graphs = [{k: tuple(v.shape) for k, v in m.state_dict().items()} for m in (model, codec)]
    key = next(iter(graphs[component]))
    if damage == "missing":
        del graphs[component][key]
    elif damage == "extra":
        graphs[component]["unimplemented.weight"] = (1,)
    else:
        graphs[component][key] = (1,)
    with pytest.raises(ValueError, match="WEIGHTS_INCOMPLETE"):
        qwen_tts.validate_weight_shapes(config, codec_config, graphs)


@native
def test_incompatible_environment_cannot_inherit_native_support(monkeypatch):
    import importlib.metadata

    actual = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "5.17.0" if name == "transformers" else actual(name),
    )
    with pytest.raises(ValueError, match="DEPENDENCY_CLOSURE"):
        qwen_tts.verify_runtime()
