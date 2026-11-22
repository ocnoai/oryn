"""Closed complete three-branch TIGER DnR separation and bounded WAV stems."""

from __future__ import annotations

import copy
import json
import wave
from pathlib import Path

from oryn_engine.model_metadata import preflight_safetensors, validate_model_config

FAMILY = "oryn_tiger_dnr"
ARCHITECTURE = "TIGERDNR"
TASK = "audio-to-audio"
DATA = Path(__file__).parent / "native_tiger"
FILES = ("config.json", "model.safetensors")
STEMS = ("dialog", "effect", "music")
SAMPLE_RATE = 44100
MAX_SECONDS = 12


def supports(config, task):
    return task == TASK and (
        config.get("model_type") == FAMILY
        or {
            "att_hid_chan",
            "att_kernel_size",
            "att_n_head",
            "att_stride",
            "in_channels",
            "num_blocks",
            "num_sources",
            "out_channels",
            "sample_rate",
            "stride",
            "upsampling_depth",
            "win",
        }
        == set(config)
    )


def normalized_config(config, task):
    if not supports(config, task):
        return config
    c = copy.deepcopy(config)
    c.setdefault("model_type", FAMILY)
    c.setdefault("architectures", [ARCHITECTURE])
    validate_config(c)
    return c


def validate_config(c):
    validate_model_config(c)
    expected = json.loads((DATA / "config.json").read_text())
    expected.update(model_type=FAMILY, architectures=[ARCHITECTURE])
    if set(c) != set(expected):
        raise ValueError("TIGER_CONFIGURATION")
    variable = {"in_channels", "out_channels", "num_blocks", "upsampling_depth", "att_hid_chan"}
    for key, value in expected.items():
        if key not in variable and (type(c[key]) is not type(value) or c[key] != value):
            raise ValueError("TIGER_CONFIGURATION:" + key)
    limits = {
        "in_channels": (8, 256),
        "out_channels": (12, 132),
        "num_blocks": (1, 8),
        "upsampling_depth": (2, 5),
        "att_hid_chan": (1, 4),
    }
    if (
        any(type(c[k]) is not int or not low <= c[k] <= high for k, (low, high) in limits.items())
        or c["out_channels"] % 12
    ):
        raise ValueError("TIGER_GEOMETRY")
    return c


def finite(value):
    import torch

    if not bool(torch.isfinite(value).all()):
        raise ValueError("TIGER_NONFINITE")
    return value


def native_model(config, *, meta=False):
    import torch

    from oryn_engine.native_tiger.model import TIGERDNR

    validate_config(config)
    with torch.device("meta" if meta else "cpu"):
        model = TIGERDNR(
            **{k: v for k, v in config.items() if k not in {"model_type", "architectures"}}
        ).eval()
    # Reject overflow before sigmoid can conceal nonfinite mask or gate values.
    for branch in (model.dialog, model.effect, model.music):
        for head in branch.mask:
            head.register_forward_hook(lambda module, args, out: finite(out))
    for module in model.modules():
        if isinstance(module, torch.nn.Sigmoid):
            module.register_forward_pre_hook(lambda module, args: (finite(args[0]),))
    return model


def validate_weight_shapes(config, shapes):
    expected = {k: tuple(v.shape) for k, v in native_model(config, meta=True).state_dict().items()}
    if expected != {k: tuple(v) for k, v in shapes.items()}:
        raise ValueError("TIGER_WEIGHTS_INCOMPLETE")


def load_components(root, config, backend):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    if backend not in {"cpu", "mps"} or root.is_symlink():
        raise ValueError("TIGER_SCOPE")
    if (
        any(not (root / name).is_file() or (root / name).is_symlink() for name in FILES)
        or {p.name for p in root.glob("*.safetensors")} != {"model.safetensors"}
        or any(root.glob("*.safetensors.index.json"))
    ):
        raise ValueError("TIGER_EXPORT_INCOMPLETE")
    validate_config(config)
    validate_weight_shapes(config, preflight_safetensors(root))
    state = load_file(root / "model.safetensors")
    if any(
        value.dtype not in {torch.float32, torch.float16, torch.bfloat16}
        or not bool(torch.isfinite(value).all())
        for value in state.values()
    ):
        raise ValueError("TIGER_NUMERIC_WEIGHT")
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
        raise ValueError("TIGER_AUDIO_SCOPE")
    with wave.open(str(path), "rb") as f:
        count = f.getnframes()
        if (
            f.getcomptype() != "NONE"
            or f.getnchannels() != 1
            or f.getsampwidth() != 2
            or f.getframerate() != SAMPLE_RATE
            or not 2048 <= count <= SAMPLE_RATE * MAX_SECONDS
        ):
            raise ValueError("TIGER_AUDIO_FORMAT")
        data = f.readframes(count)
        if len(data) != count * 2:
            raise ValueError("TIGER_AUDIO_TRUNCATED")
    return count, data


def read_audio(path):
    import numpy as np
    import torch

    _, data = audio_geometry(path)
    return torch.from_numpy(np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768)


def separate(model, audio):
    return tuple(finite(x[0]) for x in model(audio[None, None]))


def execute(model, value, backend, input_path, output_path):
    import numpy as np

    if (
        value.get("operation") != "separate"
        or value.get("sampleRate", SAMPLE_RATE) != SAMPLE_RATE
        or "audioArtifactId" not in value
        or set(value) - {"operation", "sampleRate", "audioArtifactId"}
        or output_path is None
    ):
        raise ValueError("TIGER_INPUT")
    audio = read_audio(input_path)
    outputs = separate(model, audio)
    stems = []
    if len(outputs) != 3 or any(x.shape != audio.shape for x in outputs):
        raise ValueError("TIGER_OUTPUT")
    for name, samples in zip(STEMS, outputs, strict=True):
        path = output_path.parent / (name + ".wav")
        if path.exists() or path.is_symlink():
            raise ValueError("TIGER_OUTPUT_SCOPE")
        pcm = np.clip(np.floor(samples.numpy() * 32768), -32768, 32767).astype("<i2")
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SAMPLE_RATE)
            f.writeframes(pcm.tobytes())
        stems.append(
            dict(
                relativeOutput=path.name,
                mediaType="audio/wav",
                sampleRate=SAMPLE_RATE,
                channels=1,
                frames=len(pcm),
                durationSeconds=len(pcm) / SAMPLE_RATE,
            )
        )
    return dict(stems=stems, backend=backend)


def validate_outputs(directory, result, input_path, backend):
    count, _ = audio_geometry(input_path)
    if (
        not isinstance(result, dict)
        or set(result) != {"stems", "backend"}
        or result["backend"] != backend
        or not isinstance(result["stems"], list)
        or len(result["stems"]) != 3
        or directory.is_symlink()
    ):
        raise ValueError("TIGER_OUTPUT")
    paths = []
    for name, value in zip(STEMS, result["stems"], strict=True):
        if (
            not isinstance(value, dict)
            or any(type(value.get(k)) is not int for k in ("sampleRate", "channels", "frames"))
            or type(value.get("durationSeconds")) not in {int, float}
        ):
            raise ValueError("TIGER_OUTPUT")
        if value != {
            "relativeOutput": name + ".wav",
            "mediaType": "audio/wav",
            "sampleRate": SAMPLE_RATE,
            "channels": 1,
            "frames": count,
            "durationSeconds": count / SAMPLE_RATE,
        }:
            raise ValueError("TIGER_OUTPUT")
        path = directory / (name + ".wav")
        actual, _ = audio_geometry(path)
        if count != actual:
            raise ValueError("TIGER_OUTPUT")
        paths.append(path)
    return paths
