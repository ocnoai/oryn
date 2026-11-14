"""Bounded recurrent VAD with second-based, nonoverlapping speech intervals.

The state/context binding follows the native Silero v5/v6 ONNX interface, not
repository Python. Only the sealed 16 kHz, 512-sample state128 path is executed.
"""

from __future__ import annotations

import math
import wave


def validate_graph(graph):
    inputs = {v["name"]: v for v in graph["inputs"]}
    outputs = {v["name"]: v for v in graph["outputs"]}
    if inputs.keys() != {"input", "state", "sr"} or outputs.keys() != {"output", "stateN"}:
        raise ValueError("ONNX_VAD_BINDING_UNSUPPORTED")
    specs = (
        (inputs["input"], 1, [1, 576]),
        (inputs["state"], 1, [2, 1, 128]),
        (inputs["sr"], 7, []),
        (outputs["output"], 1, [1, 1]),
        (outputs["stateN"], 1, [2, 1, 128]),
    )
    for value, dtype, shape in specs:
        if (
            value["dtype"] != dtype
            or len(value["shape"]) != len(shape)
            or any(
                isinstance(d, int) and d != expected
                for d, expected in zip(value["shape"], shape, strict=True)
            )
        ):
            raise ValueError("ONNX_VAD_BINDING_UNSUPPORTED")


def read_audio(path):
    import numpy as np
    from scipy.signal import resample_poly

    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 12 * 1024**2:
        raise ValueError("ONNX_AUDIO_LIMIT")
    with wave.open(str(path), "rb") as source:
        rate, channels, count = source.getframerate(), source.getnchannels(), source.getnframes()
        if (
            source.getsampwidth() != 2
            or source.getcomptype() != "NONE"
            or channels not in {1, 2}
            or rate not in {8000, 16000, 32000, 44100, 48000}
            or not 0 < count / rate <= 30
        ):
            raise ValueError("ONNX_AUDIO_FORMAT_UNSUPPORTED")
        raw = source.readframes(count)
        if len(raw) != count * channels * 2:
            raise ValueError("ONNX_AUDIO_TRUNCATED")
    values = (
        np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, channels).mean(axis=1)
        / 32768
    )
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        values = resample_poly(values, 16000 // divisor, rate // divisor).astype(np.float32)
    return values, count / rate


def intervals(probabilities, duration, threshold, min_speech, min_silence):
    start = pending = None
    result = []
    for index, probability in enumerate(probabilities):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("ONNX_VAD_PROBABILITY_INVALID")
        time = index * 512 / 16000
        if probability >= threshold:
            pending = None
            if start is None:
                start = time
        elif probability < max(0.01, threshold - 0.15) and start is not None:
            if pending is None:
                pending = time
            if time - pending >= min_silence:
                if pending - start >= min_speech:
                    result.append({"start": start, "end": pending})
                start = pending = None
    if start is not None:
        end = min(duration, pending if pending is not None else duration)
        if end - start >= min_speech:
            result.append({"start": start, "end": end})
    return result


def execute(session, value, path, event):
    import numpy as np

    audio, duration = read_audio(path)
    state, context = np.zeros((2, 1, 128), dtype=np.float32), np.zeros((1, 64), dtype=np.float32)
    probabilities = []
    for offset in range(0, len(audio), 512):
        chunk = audio[offset : offset + 512]
        chunk = np.pad(chunk, (0, 512 - len(chunk)))[None, :]
        value_in = np.concatenate([context, chunk], axis=1)
        event(
            "job.progress", {"stage": "running-model", "progress": 0.2 + 0.7 * offset / len(audio)}
        )
        probability, state = session.run(
            ["output", "stateN"],
            {"input": value_in, "state": state, "sr": np.array(16000, dtype=np.int64)},
        )
        if (
            probability.shape != (1, 1)
            or state.shape != (2, 1, 128)
            or not np.isfinite(state).all()
        ):
            raise ValueError("ONNX_VAD_STATE_INVALID")
        probabilities.append(float(probability[0, 0]))
        context = value_in[:, -64:]
    segments = intervals(
        probabilities,
        duration,
        value.get("threshold", 0.5),
        value.get("minSpeechSeconds", 0.25),
        value.get("minSilenceSeconds", 0.1),
    )
    return {"segments": segments, "durationSeconds": duration}
