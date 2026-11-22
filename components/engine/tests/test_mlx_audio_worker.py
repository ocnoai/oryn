"""Reusable MLX-Audio request, local-loader and native-waveform boundaries."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from oryn_engine.runtime_packs import mlx_audio_worker as worker

ARTIFACT = "artifact_" + "a" * 24


@pytest.fixture(autouse=True)
def no_protocol_events(monkeypatch):
    monkeypatch.setattr(worker, "event", lambda *_args: None)


def sample(audio=None, rate=44100):
    return SimpleNamespace(
        audio=np.array([0, 0.2, -0.2], dtype=np.float32) if audio is None else audio,
        sample_rate=rate,
    )


def payload(tmp_path, task, value, input_path=None):
    root = tmp_path / "checkpoint"
    root.mkdir(exist_ok=True)
    return {
        "modelRoot": str(root),
        "task": task,
        "backend": "mlx",
        "input": value,
        "inputPath": str(input_path) if input_path else None,
        "outputPath": str(tmp_path / "result.wav") if task == "text-to-speech" else None,
    }


@pytest.mark.parametrize(
    "value",
    [
        {"text": " "},
        {"text": "a" * 351},
        {"text": "Hello", "speed": True},
        {"text": "Hello", "speed": 1.5},
        {"text": "Hello", "temperature": 0.7},
        {"text": "Hello", "voice": "reference"},
        {"text": "Hello", "audioArtifactId": ARTIFACT},
        {"text": "Hello", "referenceText": "A transcript without reference audio."},
    ],
)
def test_input_rejects_unnegotiated_controls(value):
    with pytest.raises(ValueError, match="MODEL_INPUT_INVALID"):
        worker.input_value("text-to-speech", value)


def test_native_speech_chunks_keep_their_sample_rate_and_join(tmp_path):
    class Model:
        def generate(self, text, voice=None, lang_code="auto"):
            assert (text, voice, lang_code) == ("Hello", "Vivian", "English")
            yield sample()
            yield sample()

    capability = {"parameters": ["text", "voice", "lang_code"]}
    result = worker._speech(
        Model(),
        capability,
        worker.input_value(
            "text-to-speech", {"text": "Hello", "voice": "Vivian", "language": "English"}
        ),
        None,
        tmp_path / "result.wav",
    )
    assert result == {
        "relativeOutput": "result.wav",
        "mediaType": "audio/wav",
        "sampleRate": 44100,
        "channels": 1,
        "frames": 6,
        "durationSeconds": 6 / 44100,
        "backend": "mlx",
    }
    with wave.open(str(tmp_path / "result.wav"), "rb") as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getnframes()) == (44100, 1, 6)


def test_another_family_uses_builtin_default_and_reference_transcript(tmp_path):
    reference = tmp_path / "reference.wav"

    class Model:
        def generate(self, text, ref_audio=None, ref_text=None, **kwargs):
            assert (text, ref_audio, ref_text, kwargs) == (
                "[sighs] Fine.",
                str(reference),
                "A voice sample.",
                {},
            )
            yield sample(rate=24000)

    capability = {"parameters": ["text", "ref_audio", "ref_text"]}
    value = worker.input_value(
        "text-to-speech",
        {
            "text": "[sighs] Fine.",
            "voice": "reference",
            "audioArtifactId": ARTIFACT,
            "referenceText": "A voice sample.",
        },
    )
    assert (
        worker._speech(Model(), capability, value, reference, tmp_path / "result.wav")["sampleRate"]
        == 24000
    )


def test_kwargs_alone_do_not_authorize_silently_ignored_controls():
    class Model:
        def generate(self, text, **kwargs):
            return ()

    with pytest.raises(ValueError, match="MODEL_INPUT_INVALID"):
        worker._tts_kwargs(
            Model(),
            {"parameters": ["text", "voice"]},
            {
                "text": "Hello",
                "voice": "Vivian",
                "language": "auto",
                "speed": 1,
            },
            None,
        )


@pytest.mark.parametrize(
    "audio",
    [
        np.array([float("nan")], dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros(8000 * 31, dtype=np.float32),
    ],
)
def test_invalid_waveforms_are_removed(tmp_path, audio):
    class Model:
        def generate(self, text):
            yield sample(audio, 8000)

    with pytest.raises(ValueError):
        worker._speech(
            Model(),
            {"parameters": ["text"]},
            worker.input_value("text-to-speech", {"text": "Hello"}),
            None,
            tmp_path / "result.wav",
        )
    assert not (tmp_path / "result.wav").exists()


def test_inconsistent_native_chunk_rates_are_rejected(tmp_path):
    class Model:
        def generate(self, text):
            yield sample(rate=24000)
            yield sample(rate=44100)

    with pytest.raises(ValueError, match="MODEL_GENERATION_FAILED"):
        worker._speech(
            Model(),
            {"parameters": ["text"]},
            worker.input_value("text-to-speech", {"text": "Hello"}),
            None,
            tmp_path / "result.wav",
        )
    assert not (tmp_path / "result.wav").exists()


def test_transcription_uses_library_api_without_claiming_word_timestamps(tmp_path):
    class Model:
        def generate(self, audio, *, language=None, stream=True, word_timestamps=True):
            assert (audio, language, stream, word_timestamps) == (
                str(tmp_path / "input.wav"),
                "sk",
                False,
                False,
            )
            return SimpleNamespace(text="Dobrý deň", language="sk", segments=[])

    value = worker.input_value(
        "automatic-speech-recognition",
        {
            "audioArtifactId": ARTIFACT,
            "language": "sk",
        },
    )
    assert worker._transcription(
        Model(),
        {
            "parameters": ["audio", "language", "stream", "word_timestamps"],
        },
        value,
        tmp_path / "input.wav",
    ) == {"text": "Dobrý deň", "language": "sk"}
    with pytest.raises(ValueError, match="MODEL_INPUT_INVALID"):
        worker.input_value(
            "automatic-speech-recognition",
            {
                "audioArtifactId": ARTIFACT,
                "timestamps": True,
            },
        )


def test_loader_uses_approved_family_and_local_files_only(tmp_path, monkeypatch):
    from oryn_engine import mlx_audio_tasks

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "fish_drama3"}))
    capability = {"family": "fish_drama3", "category": "tts", "parameters": ["text"]}
    monkeypatch.setattr(mlx_audio_tasks, "validate_config", lambda config, task: config)
    monkeypatch.setattr(mlx_audio_tasks, "capability", lambda config, task: capability)
    monkeypatch.setattr(worker, "_preflight", lambda *_args: None)
    calls = []
    model = object()
    monkeypatch.setitem(
        sys.modules,
        "mlx_audio.tts.utils",
        SimpleNamespace(
            load_model=lambda *args, **kwargs: calls.append((args, kwargs)) or model,
        ),
    )
    assert worker._load(tmp_path, "text-to-speech") == (model, capability)
    assert calls == [
        (
            (tmp_path,),
            {
                "model_type": "fish_drama3",
                "model_name_parts": [],
                "strict": True,
                "trust_remote_code": False,
                "local_files_only": True,
            },
        )
    ]


def test_numeric_voice_arrays_are_bounded_and_never_unpickled(tmp_path):
    np.savez(tmp_path / "voices.npz", voice=np.ones((4, 16), dtype=np.float32))
    worker._numeric_arrays(tmp_path)
    np.save(tmp_path / "unsafe.npy", np.array([{"callable": "payload"}], dtype=object))
    with pytest.raises(ValueError, match="MODEL_RUNTIME_INCOMPATIBLE"):
        worker._numeric_arrays(tmp_path)


def test_tensor_headers_are_checked_before_backend_allocation(tmp_path, monkeypatch):
    from oryn_engine import mlx_audio_tasks

    monkeypatch.setattr(mlx_audio_tasks, "selected_files", lambda files, *_args: tuple(files))
    (tmp_path / "model.safetensors").write_bytes(b"invalid tensor bytes")
    with pytest.raises(ValueError):
        worker._preflight(tmp_path, {}, "text-to-speech")


@pytest.mark.parametrize(
    "name,metadata",
    [
        ("tokenizer_config.json", {"vocab_file": "/outside/checkpoint/vocab.json"}),
        ("tokenizer/tokenizer_config.json", {"auto_map": {"AutoTokenizer": "custom.Tokenizer"}}),
        ("preprocessor_config.json", {"trust_remote_code": True}),
        ("codec/config.json", {"auto_map": {"AutoModel": "custom.Codec"}}),
    ],
)
def test_component_constructor_metadata_cannot_select_files_or_code(
    tmp_path, monkeypatch, name, metadata
):
    from safetensors.numpy import save_file

    from oryn_engine import mlx_audio_tasks

    monkeypatch.setattr(mlx_audio_tasks, "selected_files", lambda files, *_args: tuple(files))
    save_file({"weight": np.zeros(1, dtype=np.float32)}, tmp_path / "model.safetensors")
    path = tmp_path / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        worker._preflight(tmp_path, {}, "text-to-speech")


@pytest.mark.skipif(importlib.util.find_spec("mlx_audio") is None, reason="isolated MLX-Audio pack")
def test_native_tiny_moonshine_loads_and_transcribes_offline(tmp_path):
    """Execute a real tiny native architecture through the unchanged task worker."""
    import dataclasses

    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_audio.stt.models.moonshine import Model, ModelConfig

    from oryn_engine import mlx_audio_tasks

    config = ModelConfig(
        hidden_size=16,
        intermediate_size=32,
        encoder_num_hidden_layers=1,
        decoder_num_hidden_layers=1,
        encoder_num_attention_heads=2,
        decoder_num_attention_heads=2,
        vocab_size=16,
        tie_word_embeddings=False,
        eos_token_id=0,
    )
    if (
        mlx_audio_tasks.capability(dataclasses.asdict(config), "automatic-speech-recognition")
        is None
    ):
        pytest.skip("MLX-Audio signed capability table not regenerated")
    request = payload(tmp_path, "automatic-speech-recognition", {"audioArtifactId": ARTIFACT})
    root = tmp_path / "checkpoint"
    model = Model(config)
    # Greedy decoding terminates at EOS after exercising native audio encoding,
    # attention and decoding. These random weights measure execution, not quality.
    model.proj_out.weight = mx.zeros_like(model.proj_out.weight)
    weights = dict(tree_flatten(model.parameters()))
    for name, array in weights.items():
        if "conv" in name and "weight" in name and array.ndim == 3:
            weights[name] = array.transpose(0, 2, 1)
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    (root / "config.json").write_text(json.dumps(dataclasses.asdict(config)))
    audio_path = tmp_path / "input.wav"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(np.zeros(16000, dtype="<i2").tobytes())
    request["inputPath"] = str(audio_path)
    assert worker.generate(request) == {"text": "", "language": None}


@pytest.mark.skipif(importlib.util.find_spec("mlx_audio") is None, reason="isolated MLX-Audio pack")
@pytest.mark.parametrize("decoder_dim", [512, 768])
def test_native_soprano_generates_a_waveform_through_the_same_worker(tmp_path, decoder_dim):
    """A second real family exercises tokenizer, strict load and acoustic decoding."""
    import dataclasses

    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_audio.tts.models.soprano import Model, ModelConfig
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from oryn_engine import mlx_audio_tasks

    request = payload(tmp_path, "text-to-speech", {"text": "Hello."})
    root = tmp_path / "checkpoint"
    config = ModelConfig(
        model_type="soprano",
        model_path=str(root),
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=3,
        max_position_embeddings=128,
        rope_theta=10000,
        head_dim=8,
        tie_word_embeddings=False,
    )
    values = dataclasses.asdict(config)
    if mlx_audio_tasks.capability(values, "text-to-speech") is None:
        pytest.skip("MLX-Audio signed capability table not regenerated")
    # A local three-token vocabulary and deterministic numeric weights emit one
    # acoustic token then STOP, keeping this complete inference bounded.
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[STOP]": 0, "[AUDIO]": 1, "[TEXT]": 2}, unk_token="[TEXT]")
        ),
        unk_token="[TEXT]",
        eos_token="[STOP]",
        pad_token="[STOP]",
    )
    tokenizer.save_pretrained(root)
    # Represent both original and 1.1 acoustic graphs under the same opaque
    # selected-model path. The installed directory name carries no family hint.
    config.decoder_config.decoder_dim = decoder_dim
    config.decoder_config.decoder_intermediate_dim = decoder_dim * 3
    config.decoder_config.input_kernel = 1 if decoder_dim == 768 else 3
    model = Model(config)
    embeddings = np.zeros((3, 16), dtype=np.float32)
    embeddings[2, 0] = 1
    embeddings[1, 1] = 1
    embeddings[0, 2] = 1
    model.language_model.embed_tokens.weight = mx.array(embeddings)
    projection = np.zeros((3, 16), dtype=np.float32)
    projection[1, 0] = 100
    projection[0, 1] = 100
    projection[2, :] = -100
    model.language_model.lm_head.weight = mx.array(projection)
    layer = model.language_model.layers[0]
    layer.self_attn.o_proj.weight = mx.zeros_like(layer.self_attn.o_proj.weight)
    layer.mlp.down_proj.weight = mx.zeros_like(layer.mlp.down_proj.weight)
    mx.save_safetensors(str(root / "model.safetensors"), dict(tree_flatten(model.parameters())))
    # The native config reconstructs its default acoustic decoder; retaining a
    # JSON dict here would be a different, unsupported native config layout.
    values.pop("decoder_config")
    (root / "config.json").write_text(json.dumps(values))
    result = worker.generate(request)
    assert result["sampleRate"] == 32000
    assert 0 < result["durationSeconds"] <= 30
    assert result["frames"] > 0
    with wave.open(str(tmp_path / "result.wav"), "rb") as audio:
        assert (audio.getframerate(), audio.getnframes()) == (32000, result["frames"])


def test_soprano_decoder_dimensions_come_from_tensor_graph():
    tensors = {
        "decoder.decoder.embed.weight": (768, 1, 16),
        "decoder.decoder.convnext.0.pwconv1.weight": (2304, 768),
        "decoder.decoder.convnext.0.pwconv2.weight": (768, 2304),
        "decoder.decoder.convnext.0.dwconv.weight": (768, 3, 1),
        "decoder.head.out.weight": (2050, 768),
    }
    assert worker._soprano_decoder_config({"hidden_size": 16}, tensors) == {
        "decoder_num_layers": 1,
        "decoder_dim": 768,
        "decoder_intermediate_dim": 2304,
        "input_kernel": 1,
        "dw_kernel": 3,
        "n_fft": 2048,
    }
    with pytest.raises(ValueError, match="MODEL_RUNTIME_INCOMPATIBLE"):
        worker._soprano_decoder_config(
            {"hidden_size": 16, "decoder_config": {"decoder_dim": 512}}, tensors
        )


def test_generate_validates_media_binding_before_model_load(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "_load", lambda *_args: pytest.fail("must not load"))
    monkeypatch.setattr(worker, "_local_loaders", lambda _root: contextlib.nullcontext())
    with pytest.raises(ValueError, match="MODEL_INPUT_INVALID"):
        worker.generate(
            payload(tmp_path, "automatic-speech-recognition", {"audioArtifactId": ARTIFACT})
        )


def test_pretrained_helpers_only_accept_checkpoint_scoped_paths(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    codec = root / "codec"
    codec.mkdir()
    assert worker._local_pretrained_path(codec, root) == str(codec)
    for value in ("org/remote-codec", str(tmp_path), "../model", root / "missing"):
        with pytest.raises(ValueError, match="MODEL_RUNTIME_INCOMPATIBLE"):
            worker._local_pretrained_path(value, root)


def test_native_pretrained_hooks_force_safe_local_flags(tmp_path, monkeypatch):
    import huggingface_hub
    import torch
    from transformers import AutoTokenizer

    calls = []

    def from_pretrained(cls, model_path, **kwargs):
        calls.append((cls, model_path, kwargs))
        return object()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", classmethod(from_pretrained))
    with worker._local_loaders(tmp_path):
        AutoTokenizer.from_pretrained(str(tmp_path), trust_remote_code=True, local_files_only=False)
        for call in (
            lambda: AutoTokenizer.from_pretrained("org/remote-tokenizer"),
            lambda: huggingface_hub.hf_hub_download("org/codec", "model.safetensors"),
            lambda: torch.load(tmp_path / "unsafe.pth"),
        ):
            with pytest.raises(ValueError, match="MODEL_RUNTIME_INCOMPATIBLE"):
                call()
    assert calls == [
        (AutoTokenizer, str(tmp_path), {"trust_remote_code": False, "local_files_only": True})
    ]
