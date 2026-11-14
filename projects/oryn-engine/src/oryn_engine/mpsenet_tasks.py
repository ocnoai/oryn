"""Bounded original MP-SENet speech enhancement in the shared native environment."""

from __future__ import annotations

import copy
import json
import wave
from pathlib import Path

from oryn_engine.model_metadata import preflight_safetensors, validate_model_config

FAMILY = "oryn_mpsenet"
ARCHITECTURE = "MPSENet"
TASK = "audio-to-audio"
DATA = Path(__file__).parent / "native_mpsenet"
FILES = ("config.json", "model.safetensors")
SAMPLE_RATE = 16000
HOP = 100
SEGMENT = 32000


def supports(config, task):
    return task == TASK and (
        config.get("model_type") == FAMILY
        or set(config) == {"h", "num_tsblocks"}
        and isinstance(config.get("h"), dict)
        and set(config["h"])
        == {
            "beta",
            "compress_factor",
            "dense_channel",
            "hop_size",
            "n_fft",
            "num_tsconformers",
            "sampling_rate",
            "segment_size",
            "win_size",
        }
    )


def normalized_config(config, task):
    if not supports(config, task):
        return config
    c = copy.deepcopy(config)
    c.setdefault("model_type", FAMILY)
    c.setdefault("architectures", [ARCHITECTURE])
    validate_config(c)
    return c


def validate_config(config):
    validate_model_config(config)
    expected = json.loads((DATA / "config.json").read_text())
    expected.update(model_type=FAMILY, architectures=[ARCHITECTURE])
    if (
        set(config) != set(expected)
        or config.get("model_type") != FAMILY
        or config.get("architectures") != [ARCHITECTURE]
        or type(config["num_tsblocks"]) is not int
        or not 1 <= config["num_tsblocks"] <= 4
        or not isinstance(config["h"], dict)
        or set(config["h"]) != set(expected["h"])
    ):
        raise ValueError("MPSENET_CONFIGURATION")
    for key, val in expected["h"].items():
        if key == "dense_channel":
            n = config["h"][key]
            if type(n) is not int or not 8 <= n <= 64 or n % 4:
                raise ValueError("MPSENET_GEOMETRY")
        elif type(config["h"][key]) is not type(val) or config["h"][key] != val:
            raise ValueError("MPSENET_CONFIGURATION:" + key)
    return config


def finite(value):
    import torch

    if not bool(torch.isfinite(value).all()):
        raise ValueError("MPSENET_NONFINITE")
    return value


def native_model(config, *, meta=False):
    import torch

    from oryn_engine.native_mpsenet.model import LearnableSigmoid2d, MPSENet

    validate_config(config)

    class CheckedSigmoid(LearnableSigmoid2d):
        def forward(self, x):
            return self.beta * torch.sigmoid(finite(self.slope * x))

    with torch.device("meta" if meta else "cpu"):
        model = MPSENet(config["h"], config["num_tsblocks"]).eval()
        model.mask_decoder.lsigmoid = CheckedSigmoid(
            config["h"]["n_fft"] // 2 + 1, beta=config["h"]["beta"]
        )
    for layer in (model.phase_decoder.phase_conv_r, model.phase_decoder.phase_conv_i):
        layer.register_forward_hook(lambda module, args, output: finite(output))
    return model


def validate_weight_shapes(config, shapes):
    expected = {k: tuple(v.shape) for k, v in native_model(config, meta=True).state_dict().items()}
    if expected != {k: tuple(v) for k, v in shapes.items()}:
        raise ValueError("MPSENET_WEIGHTS_INCOMPLETE")


def load_components(root, config, backend):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    if backend not in {"cpu", "mps"} or root.is_symlink():
        raise ValueError("MPSENET_SCOPE")
    if (
        any(not (root / name).is_file() or (root / name).is_symlink() for name in FILES)
        or {p.name for p in root.glob("*.safetensors")} != {"model.safetensors"}
        or any(root.glob("*.safetensors.index.json"))
    ):
        raise ValueError("MPSENET_EXPORT_INCOMPLETE")
    validate_config(config)
    validate_weight_shapes(config, preflight_safetensors(root))
    state = load_file(root / "model.safetensors")
    if any(
        v.dtype not in {torch.float32, torch.float16, torch.bfloat16}
        or not bool(torch.isfinite(v).all())
        for v in state.values()
    ):
        raise ValueError("MPSENET_NUMERIC_WEIGHT")
    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True, assign=True)
    return model.to(backend).eval(), None, config


def read_audio(path):
    import numpy as np
    import torch

    if (
        path is None
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 2 * SAMPLE_RATE * 30 + 4096
    ):
        raise ValueError("MPSENET_AUDIO_SCOPE")
    with wave.open(str(path), "rb") as f:
        count = f.getnframes()
        if (
            f.getcomptype() != "NONE"
            or f.getnchannels() != 1
            or f.getsampwidth() != 2
            or f.getframerate() != SAMPLE_RATE
            or not 400 <= count <= SAMPLE_RATE * 30
        ):
            raise ValueError("MPSENET_AUDIO_FORMAT")
        raw = f.readframes(count)
        if len(raw) != count * 2:
            raise ValueError("MPSENET_AUDIO_TRUNCATED")
    return torch.from_numpy(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768)


def enhance(model, audio, backend):
    import torch

    from oryn_engine.native_mpsenet.model import mag_pha_istft, mag_pha_stft

    norm = finite((audio.numel() / audio.square().sum()).sqrt())
    audio = finite(audio * norm).unsqueeze(0)
    window = model.hann_window.cpu()
    results = []
    for start in range(0, audio.shape[1], SEGMENT):
        end = min(start + SEGMENT, audio.shape[1])
        last = audio.shape[1] - end < SEGMENT // 2
        if last:
            end = audio.shape[1]
        piece = audio[:, start:end]
        amp, phase, _ = mag_pha_stft(piece, window, 400, HOP, 400, 0.3)
        out_amp, out_phase, out_complex = model(amp.to(backend), phase.to(backend))
        finite(out_amp)
        finite(out_phase)
        finite(out_complex)
        piece = mag_pha_istft(out_amp.cpu(), out_phase.cpu(), window, 400, HOP, 400, 0.3)
        results.append(finite(piece))
        if last:
            break
    result = finite(torch.cat(results, dim=-1).squeeze(0) / norm)
    if result.shape != (audio.shape[1] // HOP * HOP,):
        raise ValueError("MPSENET_OUTPUT_GEOMETRY")
    return result


def execute(model, value, backend, input_path, output_path):
    import numpy as np

    if (
        value.get("operation") != "enhance"
        or value.get("sampleRate", SAMPLE_RATE) != SAMPLE_RATE
        or "audioArtifactId" not in value
        or set(value) - {"operation", "sampleRate", "audioArtifactId"}
        or output_path is None
    ):
        raise ValueError("MPSENET_INPUT")
    result = enhance(model, read_audio(input_path), backend)
    # libsndfile's PCM16 writer saturates and floors scaled floats; preserve that
    # boundary after rejecting nonfinite network output, without peak normalization.
    pcm = np.clip(np.floor(result.numpy() * 32768), -32768, 32767).astype("<i2")
    with wave.open(str(output_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())
    return dict(
        relativeOutput="result.wav",
        mediaType="audio/wav",
        sampleRate=SAMPLE_RATE,
        channels=1,
        frames=len(pcm),
        durationSeconds=len(pcm) / SAMPLE_RATE,
        backend=backend,
    )


def validate_output(path, result, input_path, backend):
    if (
        not isinstance(result, dict)
        or any(type(result.get(k)) is not int for k in ("sampleRate", "channels", "frames"))
        or type(result.get("durationSeconds")) not in {int, float}
    ):
        raise ValueError("MPSENET_OUTPUT")
    if input_path.is_symlink():
        raise ValueError("MPSENET_INPUT")
    with wave.open(str(input_path), "rb") as source:
        if (
            source.getcomptype() != "NONE"
            or source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getframerate() != SAMPLE_RATE
            or not 400 <= source.getnframes() <= SAMPLE_RATE * 30
        ):
            raise ValueError("MPSENET_INPUT")
        expected = source.getnframes() // HOP * HOP
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 2 * SAMPLE_RATE * 30 + 4096
        or result
        != {
            "relativeOutput": "result.wav",
            "mediaType": "audio/wav",
            "sampleRate": SAMPLE_RATE,
            "channels": 1,
            "frames": expected,
            "durationSeconds": expected / SAMPLE_RATE,
            "backend": backend,
        }
    ):
        raise ValueError("MPSENET_OUTPUT")
    with wave.open(str(path), "rb") as f:
        if (
            f.getcomptype() != "NONE"
            or f.getnchannels() != 1
            or f.getsampwidth() != 2
            or f.getframerate() != SAMPLE_RATE
            or f.getnframes() != expected
            or len(f.readframes(expected)) != expected * 2
        ):
            raise ValueError("MPSENET_OUTPUT")
