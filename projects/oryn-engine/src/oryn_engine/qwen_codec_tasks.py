"""Standalone original Qwen 12Hz codec, sharing the complete Breeze codec graph."""

from __future__ import annotations

import copy
import wave

from oryn_engine.breeze_tasks import DATA, validate_codec
from oryn_engine.model_metadata import metadata_file, preflight_safetensors

FAMILY = "qwen3_tts_tokenizer_12hz"
ARCHITECTURE = "Qwen3TTSTokenizerV2Model"
TASK = "audio-to-audio"
FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
SAMPLE_RATE = 24000
HOP = 1920
MAX_FRAMES = 375


def supports(config, task):
    return task == TASK and config.get("model_type") == FAMILY


def validate_config(config):
    return validate_codec(config)


def validate_processor(value):
    import json

    expected = json.loads((DATA / "codec_preprocessor.json").read_text())
    if value != expected or any(type(value.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError("QWEN_CODEC_PROCESSOR")


def finite(value):
    import torch

    if not bool(torch.isfinite(value).all()):
        raise ValueError("QWEN_CODEC_NONFINITE")
    return value


def native_model(config, *, meta=False):
    import torch
    from transformers.models.mimi.modeling_mimi import MimiEuclideanCodebook, MimiVectorQuantization

    from oryn_engine.native_breeze.codec import Qwen3TTSTokenizerV2Model
    from oryn_engine.native_breeze.codec_config import Qwen3TTSTokenizerV2Config

    validate_config(config)
    c = Qwen3TTSTokenizerV2Config(**copy.deepcopy(config))
    for part in [c, c.encoder_config, c.decoder_config]:
        part._attn_implementation = "eager"

    class CheckedCodebook(MimiEuclideanCodebook):
        def quantize(self, hidden_states):
            # Preserve native nearest-centroid math; reject nonfinite distances
            # before argmin can silently turn them into an apparently valid code.
            distances = torch.cdist(hidden_states[None].float(), self.embed[None].float(), p=2)[0]
            return finite(distances).argmin(dim=-1)

    with torch.device("meta" if meta else "cpu"):
        model = Qwen3TTSTokenizerV2Model(c).eval()
        for module in model.encoder.modules():
            if isinstance(module, MimiVectorQuantization):
                module.codebook = CheckedCodebook(c.encoder_config)
    model.decoder.decoder[-1].register_forward_hook(lambda module, args, output: finite(output))
    return model


def validate_weight_shapes(config, shapes):
    model = native_model(config, meta=True)
    if {k: tuple(v.shape) for k, v in model.state_dict().items()} != {
        k: tuple(v) for k, v in shapes.items()
    }:
        raise ValueError("QWEN_CODEC_WEIGHTS_INCOMPLETE")


def load_components(root, config, backend):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    if backend not in {"cpu", "mps"} or root.is_symlink():
        raise ValueError("QWEN_CODEC_SCOPE")
    for name in FILES:
        if not (root / name).is_file() or (root / name).is_symlink():
            raise ValueError("QWEN_CODEC_EXPORT_INCOMPLETE")
    if {p.name for p in root.glob("*.safetensors")} != {"model.safetensors"} or any(
        root.glob("*.safetensors.index.json")
    ):
        raise ValueError("QWEN_CODEC_EXPORT_INCOMPLETE")
    validate_config(config)
    validate_processor(metadata_file(root / "preprocessor_config.json"))
    validate_weight_shapes(config, preflight_safetensors(root))
    state = load_file(root / "model.safetensors")
    for key, tensor in state.items():
        if tensor.dtype not in {torch.float32, torch.float16, torch.bfloat16} or not bool(
            torch.isfinite(tensor).all()
        ):
            raise ValueError("QWEN_CODEC_NUMERIC_WEIGHT")
        if key.endswith(".initialized") and not bool((tensor == 1).all()):
            raise ValueError("QWEN_CODEC_UNINITIALIZED_CODEBOOK")
    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True, assign=True)
    return model.to(backend).eval(), None, config


def validate_codes(codes):
    if (
        not isinstance(codes, list)
        or not 1 <= len(codes) <= MAX_FRAMES
        or any(
            not isinstance(row, list)
            or len(row) != 16
            or any(type(i) is not int or not 0 <= i < 2048 for i in row)
            for row in codes
        )
    ):
        raise ValueError("QWEN_CODEC_CODES")


def read_audio(path):
    import numpy as np
    import torch

    if (
        path is None
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 2 * SAMPLE_RATE * 30 + 4096
    ):
        raise ValueError("QWEN_CODEC_AUDIO_SCOPE")
    with wave.open(str(path), "rb") as f:
        count = f.getnframes()
        if (
            f.getcomptype() != "NONE"
            or f.getnchannels() != 1
            or f.getsampwidth() != 2
            or f.getframerate() != SAMPLE_RATE
            or not HOP <= count <= MAX_FRAMES * HOP
        ):
            raise ValueError("QWEN_CODEC_AUDIO_FORMAT")
        raw = f.readframes(count)
        if len(raw) != count * 2:
            raise ValueError("QWEN_CODEC_AUDIO_TRUNCATED")
    return torch.from_numpy(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768).unsqueeze(0)


def execute(model, value, backend, input_path, output_path):
    import numpy as np
    import torch

    if value.get("sampleRate", SAMPLE_RATE) != SAMPLE_RATE:
        raise ValueError("QWEN_CODEC_SAMPLE_RATE")
    operation = value.get("operation")
    if operation == "encode":
        if (
            output_path is not None
            or set(value) - {"operation", "sampleRate", "audioArtifactId"}
            or "audioArtifactId" not in value
        ):
            raise ValueError("QWEN_CODEC_ENCODE_INPUT")
        audio = read_audio(input_path).to(backend)
        codes = model.encode(
            audio, padding_mask=torch.ones_like(audio, dtype=torch.bool)
        ).audio_codes[0]
        expected = (audio.shape[1] + HOP - 1) // HOP
        if codes.shape != (expected, 16):
            raise ValueError("QWEN_CODEC_ENCODE_GEOMETRY")
        data = codes.cpu().tolist()
        validate_codes(data)
        return {
            "codes": data,
            "sampleRate": SAMPLE_RATE,
            "durationSeconds": len(data) * HOP / SAMPLE_RATE,
        }
    if (
        operation != "decode"
        or input_path is not None
        or output_path is None
        or set(value) - {"operation", "sampleRate", "codes"}
    ):
        raise ValueError("QWEN_CODEC_DECODE_INPUT")
    codes = value.get("codes")
    validate_codes(codes)
    audio = finite(
        model.decode(torch.tensor([codes], dtype=torch.long, device=backend)).audio_values[0]
    )
    frames = len(codes) * HOP
    if audio.ndim != 1 or audio.numel() != frames or float(audio.abs().max()) > 1:
        raise ValueError("QWEN_CODEC_DECODE_GEOMETRY")
    pcm = np.round(audio.float().cpu().numpy() * 32767).astype("<i2")
    with wave.open(str(output_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())
    return {
        "relativeOutput": "result.wav",
        "mediaType": "audio/wav",
        "sampleRate": SAMPLE_RATE,
        "channels": 1,
        "frames": frames,
        "durationSeconds": frames / SAMPLE_RATE,
        "backend": backend,
    }
