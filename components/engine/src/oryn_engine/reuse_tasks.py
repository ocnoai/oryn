"""Closed native RE-USE enhancement profiles backed by the shared Mamba graph."""

from __future__ import annotations

import copy
import json
import wave
from pathlib import Path

from oryn_engine.model_metadata import preflight_safetensors, validate_model_config

FAMILY = "oryn_reuse"
TASK = "audio-to-audio"
DATA = Path(__file__).parent / "reuse_assets"
FILES = ("config.json", "model.safetensors")
SAMPLE_RATE = 8000
MAX_SECONDS = 2


def is_causal(config):
    return "maximum_layer" in config["model_cfg"]


def architecture(config):
    return "SEMambaDecoderList" if is_causal(config) else "SEMamba"


def supports(config, task):
    return task == TASK and (
        config.get("model_type") == FAMILY
        or set(config) == {"env_setting", "model_cfg", "stft_cfg"}
        and isinstance(config.get("model_cfg"), dict)
        and config["model_cfg"].get("compress_factor") == "relu_log1p"
    )


def normalized_config(config, task):
    if not supports(config, task):
        return config
    if not isinstance(config.get("model_cfg"), dict):
        raise ValueError("REUSE_CONFIGURATION")
    c = copy.deepcopy(config)
    c.setdefault("model_type", FAMILY)
    c.setdefault("architectures", [architecture(c)])
    validate_config(c)
    return c


def validate_config(c):
    validate_model_config(c)
    if not isinstance(c.get("model_cfg"), dict):
        raise ValueError("REUSE_CONFIGURATION")
    causal = is_causal(c)
    expected = json.loads((DATA / ("causal.json" if causal else "offline.json")).read_text())
    expected.update(model_type=FAMILY, architectures=[architecture(c)])
    if set(c) != set(expected):
        raise ValueError("REUSE_CONFIGURATION")
    for name in ["env_setting", "stft_cfg", "model_type", "architectures"]:
        if json.dumps(c[name], sort_keys=True) != json.dumps(expected[name], sort_keys=True):
            raise ValueError("REUSE_CONFIGURATION:" + name)
    value = c["model_cfg"]
    original = expected["model_cfg"]
    if set(value) != set(original):
        raise ValueError("REUSE_CONFIGURATION")
    for key, val in original.items():
        if key not in {"hid_feature", "num_tfmamba", "maximum_layer"} and (
            type(value[key]) is not type(val) or value[key] != val
        ):
            raise ValueError("REUSE_CONFIGURATION:" + key)
    width, layers = value["hid_feature"], value["num_tfmamba"]
    if (
        type(width) is not int
        or not 8 <= width <= 64
        or width % 8
        or type(layers) is not int
        or not 1 <= layers <= (12 if causal else 30)
        or causal
        and (type(value["maximum_layer"]) is not int or value["maximum_layer"] != layers)
    ):
        raise ValueError("REUSE_GEOMETRY")
    return c


def finite(value):
    import torch

    if not bool(torch.isfinite(value).all()):
        raise ValueError("REUSE_NONFINITE")
    return value


def native_model(config, *, meta=False):
    import torch

    from oryn_engine.native_reuse import Decoder, ReuseGraph

    validate_config(config)
    with torch.device("meta" if meta else "cpu"):
        model = ReuseGraph(config).eval()
    for module in model.modules():
        if isinstance(module, Decoder) and module.phase:
            for layer in (module.phase_conv_r, module.phase_conv_i):
                layer.register_forward_hook(lambda module, args, out: finite(out))
    return model


def validate_weight_shapes(config, shapes):
    expected = {k: tuple(v.shape) for k, v in native_model(config, meta=True).state_dict().items()}
    if expected != {k: tuple(v) for k, v in shapes.items()}:
        raise ValueError("REUSE_WEIGHTS_INCOMPLETE")


def load_components(root, config, backend):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    if backend not in {"cpu", "mps"} or root.is_symlink():
        raise ValueError("REUSE_SCOPE")
    if (
        any(not (root / name).is_file() or (root / name).is_symlink() for name in FILES)
        or {p.name for p in root.glob("*.safetensors")} != {"model.safetensors"}
        or any(root.glob("*.safetensors.index.json"))
    ):
        raise ValueError("REUSE_EXPORT_INCOMPLETE")
    validate_config(config)
    validate_weight_shapes(config, preflight_safetensors(root))
    state = load_file(root / "model.safetensors")
    for key, value in state.items():
        if value.dtype not in {torch.float32, torch.float16, torch.bfloat16} or not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError("REUSE_NUMERIC_WEIGHT")
        if key.endswith(".A_log") and not bool(torch.isfinite(value.float().exp()).all()):
            raise ValueError("REUSE_STATE_DECAY")
    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True, assign=True)
    return model.to(backend).eval(), None, config


def audio_geometry(path):
    if (
        path is None
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 2 * SAMPLE_RATE * MAX_SECONDS + 4096
    ):
        raise ValueError("REUSE_AUDIO_SCOPE")
    with wave.open(str(path), "rb") as f:
        count = f.getnframes()
        if (
            f.getcomptype() != "NONE"
            or f.getnchannels() != 1
            or f.getsampwidth() != 2
            or f.getframerate() != SAMPLE_RATE
            or not 320 <= count <= SAMPLE_RATE * MAX_SECONDS
        ):
            raise ValueError("REUSE_AUDIO_FORMAT")
        data = f.readframes(count)
        if len(data) != count * 2:
            raise ValueError("REUSE_AUDIO_TRUNCATED")
    return count, data


def read_audio(path):
    import numpy as np
    import torch

    _, data = audio_geometry(path)
    return torch.from_numpy(np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768)


def spectral_input(audio, config):
    import torch

    hop = config["stft_cfg"]["hop_size"]
    window = torch.hann_window(320)
    value = torch.stft(
        audio.unsqueeze(0),
        320,
        hop_length=hop,
        win_length=320,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    return value.abs().log1p(), value.angle()


def spectral_output(amp, phase, count, config):
    import torch

    finite(amp)
    finite(phase)
    magnitude = finite(amp.relu().expm1())
    # Preserve the publisher's suppression of frames whose positive spectrum
    # occupies less than half the frequency bins, before inverse compression.
    remove = (magnitude == 0).sum(1) / magnitude.shape[1] > 0.5
    amp = amp.clone()
    amp[:, :, remove[0]] = 0
    magnitude = finite(amp.relu().expm1())
    value = torch.complex(magnitude * phase.cos(), magnitude * phase.sin())
    audio = finite(
        torch.istft(
            value,
            320,
            hop_length=config["stft_cfg"]["hop_size"],
            win_length=320,
            window=torch.hann_window(320),
            center=True,
        )
    )[0]
    if audio.numel() < count:
        audio = torch.nn.functional.pad(audio, (0, count - audio.numel()), value=1e-8)
    return audio[:count]


def enhance(model, audio, config, backend):
    amp, phase = spectral_input(audio, config)
    predicted, angle, complex_value = model(
        amp.to(backend), phase.to(backend), ahead=2 if is_causal(config) else 0
    )
    finite(complex_value)
    return spectral_output(predicted.cpu(), angle.cpu(), audio.numel(), config)


def execute(model, config, value, backend, input_path, output_path):
    import numpy as np

    if (
        value.get("operation") != "enhance"
        or value.get("sampleRate", SAMPLE_RATE) != SAMPLE_RATE
        or "audioArtifactId" not in value
        or set(value) - {"operation", "sampleRate", "audioArtifactId"}
        or output_path is None
    ):
        raise ValueError("REUSE_INPUT")
    audio = finite(enhance(model, read_audio(input_path), config, backend))
    pcm = np.clip(np.floor(audio.numpy() * 32768), -32768, 32767).astype("<i2")
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
    count, _ = audio_geometry(input_path)
    if (
        not isinstance(result, dict)
        or any(type(result.get(k)) is not int for k in ("sampleRate", "channels", "frames"))
        or type(result.get("durationSeconds")) not in {int, float}
    ):
        raise ValueError("REUSE_OUTPUT")
    if result != {
        "relativeOutput": "result.wav",
        "mediaType": "audio/wav",
        "sampleRate": SAMPLE_RATE,
        "channels": 1,
        "frames": count,
        "durationSeconds": count / SAMPLE_RATE,
        "backend": backend,
    }:
        raise ValueError("REUSE_OUTPUT")
    actual, _ = audio_geometry(path)
    if count != actual:
        raise ValueError("REUSE_OUTPUT")
