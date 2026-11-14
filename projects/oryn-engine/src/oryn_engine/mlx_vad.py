"""Sealed Silero v5 MLX graph, loaded exclusively from original safetensors.

The recurrent graph is checked against the official v5 ONNX implementation.
State and audio context belong to one request, never to a resident model.
"""

from __future__ import annotations

from pathlib import Path

from oryn_engine.model_metadata import metadata_file, preflight_safetensors

MODEL_TYPE = "silero_vad_v5"
CONFIG = {
    "model_type": MODEL_TYPE,
    "sample_rate": 16000,
    "chunk_size": 512,
    "context_size": 64,
    "filter_length": 256,
    "hop_length": 128,
    "encoder_channels": [129, 128, 64, 64, 128],
    "encoder_kernel_sizes": [3, 3, 3, 3],
    "encoder_strides": [1, 2, 2, 1],
    "lstm_hidden_size": 128,
    "lstm_num_layers": 1,
}
SHAPES = {
    "stft.weight": (258, 256, 1),
    "lstm.Wx": (512, 128),
    "lstm.Wh": (512, 128),
    "lstm.bias": (512,),
    "decoder.weight": (1, 1, 128),
    "decoder.bias": (1,),
}
for _index, (_in, _out) in enumerate(zip([129, 128, 64, 64], [128, 64, 64, 128], strict=True)):
    SHAPES[f"encoder.{_index}.weight"] = (_out, 3, _in)
    SHAPES[f"encoder.{_index}.bias"] = (_out,)


def validate_config(config):
    # No ignored dispatch, quantization, or preprocessing fields. This graph has
    # one reviewed geometry; a different model needs its own native binding.
    if config != CONFIG or any(type(config[k]) is not type(v) for k, v in CONFIG.items()):
        raise ValueError("MLX_VAD_CONFIGURATION_UNSUPPORTED")
    for key in ("encoder_channels", "encoder_kernel_sizes", "encoder_strides"):
        if any(type(v) is not int for v in config[key]):
            raise ValueError("MLX_VAD_CONFIGURATION_UNSUPPORTED")
    return config


def load_weights(root):
    import mlx.core as mx

    validate_config(metadata_file(root / "config.json"))
    path = root / "model.safetensors"
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size > 2 * 1024**2
        or {p.name for p in root.glob("*.safetensors")} != {path.name}
        or preflight_safetensors(root) != SHAPES
    ):
        raise ValueError("MLX_VAD_WEIGHTS_UNSUPPORTED")
    weights = mx.load(str(path))
    if set(weights) != set(SHAPES) or any(
        v.dtype != mx.float32 or tuple(v.shape) != SHAPES[k] or not mx.all(mx.isfinite(v)).item()
        for k, v in weights.items()
    ):
        raise ValueError("MLX_VAD_WEIGHTS_UNSUPPORTED")
    return weights


def forward(weights, audio, hidden, cell):
    import mlx.core as mx

    # Native STFT reflects 64 samples at the right boundary of the 576-sample
    # context/window. MLX convolution uses output/kernel/input weight order.
    audio = mx.concatenate([audio, audio[:, mx.arange(574, 510, -1)]], axis=1)[..., None]
    value = mx.conv1d(audio, weights["stft.weight"], stride=128)
    value = mx.sqrt(value[..., :129] ** 2 + value[..., 129:] ** 2)
    for index, stride in enumerate((1, 2, 2, 1)):
        value = mx.maximum(
            mx.conv1d(value, weights[f"encoder.{index}.weight"], stride=stride, padding=1)
            + weights[f"encoder.{index}.bias"],
            0,
        )
    gates = (
        value[:, 0] @ weights["lstm.Wx"].T + hidden @ weights["lstm.Wh"].T + weights["lstm.bias"]
    )
    incoming, forget, candidate, outgoing = mx.split(gates, 4, axis=-1)
    cell = mx.sigmoid(forget) * cell + mx.sigmoid(incoming) * mx.tanh(candidate)
    hidden = mx.sigmoid(outgoing) * mx.tanh(cell)
    probability = mx.sigmoid(
        mx.conv1d(mx.maximum(hidden, 0)[:, None, :], weights["decoder.weight"])
        + weights["decoder.bias"]
    )
    mx.eval(probability, hidden, cell)
    if not mx.all(mx.isfinite(hidden)).item() or not mx.all(mx.isfinite(cell)).item():
        raise ValueError("MLX_VAD_STATE_INVALID")
    return probability.item(), hidden, cell


def execute(payload, event):
    import mlx.core as mx
    import numpy as np

    from oryn_engine.onnx_tasks import input_value, output_value
    from oryn_engine.vad_tasks import intervals, read_audio

    if (
        set(payload) != {"modelRoot", "task", "backend", "input", "inputPath", "outputPath"}
        or payload["task"] != "voice-activity-detection"
        or payload["backend"] != "mlx"
        or payload["outputPath"] is not None
    ):
        raise ValueError("MLX_VAD_REQUEST_INVALID")
    root, path = Path(payload["modelRoot"]), Path(payload["inputPath"])
    if not root.is_absolute() or root.is_symlink() or not root.is_dir() or not path.is_absolute():
        raise ValueError("MLX_VAD_REQUEST_INVALID")
    value = input_value(payload["task"], payload["input"])
    audio, duration = read_audio(path)
    event("job.progress", {"stage": "loading-model", "progress": 0.05})
    weights = load_weights(root)
    try:
        hidden, cell = mx.zeros((1, 128)), mx.zeros((1, 128))
        context = np.zeros(64, dtype=np.float32)
        probabilities = []
        for offset in range(0, len(audio), 512):
            event(
                "job.progress",
                {"stage": "running-model", "progress": 0.2 + 0.7 * offset / len(audio)},
            )
            chunk = np.pad(audio[offset : offset + 512], (0, max(0, offset + 512 - len(audio))))
            window = mx.array(np.concatenate([context, chunk])[None, :])
            probability, hidden, cell = forward(weights, window, hidden, cell)
            probabilities.append(probability)
            context = chunk[-64:]
        result = {
            "segments": intervals(
                probabilities,
                duration,
                value.get("threshold", 0.5),
                value.get("minSpeechSeconds", 0.25),
                value.get("minSilenceSeconds", 0.1),
            ),
            "durationSeconds": duration,
        }
        output_value(payload["task"], result)
        return result
    finally:
        del weights
        mx.clear_cache()
