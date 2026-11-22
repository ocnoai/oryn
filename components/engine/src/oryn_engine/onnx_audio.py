"""Bounded DFN3 speech enhancement over a data-only ONNX graph.

The fixed 48 kHz front end uses the DeepFilterNet 3 Vorbis window, ERB bands
and normalization equations. No checkpoint selects preprocessing or imports.
Each channel keeps its own normalization state; output preserves channel count.
"""

from __future__ import annotations

import math
import wave

ARCHITECTURE = "dfn3-48000-200"
ERB_WIDTHS = (2,) * 13 + (5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20, 24, 28, 31, 37, 42, 50, 56, 67)


def validate_graph(graph):
    inputs = {item["name"]: item for item in graph["inputs"]}
    outputs = {item["name"]: item for item in graph["outputs"]}
    shapes = {
        "spec": [1, 1, 200, 481, 2],
        "feat_erb": [1, 1, 200, 32],
        "feat_spec": [1, 1, 200, 96, 2],
    }
    if inputs.keys() != shapes.keys() or outputs.keys() != {"spec_enhanced"}:
        raise ValueError("ONNX_AUDIO_BINDING_UNSUPPORTED")
    for name, shape in shapes.items():
        if inputs[name]["dtype"] != 1 or inputs[name]["shape"] != shape:
            raise ValueError("ONNX_AUDIO_BINDING_UNSUPPORTED")
    output = outputs["spec_enhanced"]
    if output["dtype"] != 1 or output["shape"] != shapes["spec"]:
        raise ValueError("ONNX_AUDIO_BINDING_UNSUPPORTED")


def input_value(value):
    if (
        not set(value) <= {"operation", "audioArtifactId", "sampleRate"}
        or value.get("operation") != "enhance"
        or not value.get("audioArtifactId")
        or value.get("sampleRate", 48000) != 48000
    ):
        raise ValueError("ONNX_AUDIO_OPERATION_UNSUPPORTED")
    return value


def read_audio(path):
    import numpy as np
    from scipy.signal import resample_poly

    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 12 * 1024**2:
        raise ValueError("ONNX_AUDIO_LIMIT")
    with wave.open(str(path), "rb") as source:
        count, rate, channels = source.getnframes(), source.getframerate(), source.getnchannels()
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
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, channels) / 32768
    if rate != 48000:
        divisor = math.gcd(rate, 48000)
        audio = resample_poly(audio, 48000 // divisor, rate // divisor, axis=0).astype(np.float32)
    return audio


def window():
    import numpy as np

    return np.sin(np.pi / 2 * np.sin(np.pi * (np.arange(960) + 0.5) / 960) ** 2).astype(np.float32)


def analysis(audio):
    import numpy as np
    from scipy.fft import rfft

    padded = np.pad(audio, (480, ((len(audio) + 479) // 480) * 480 + 960 - len(audio)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, 960)[::480]
    spec = (rfft(frames * window(), axis=1) / np.float32(960)).astype(np.complex64)
    power = spec.real**2 + spec.imag**2
    offsets = np.cumsum((0, *ERB_WIDTHS))
    erb = np.stack(
        [
            power[:, start:end].mean(axis=1)
            for start, end in zip(offsets[:-1], offsets[1:], strict=True)
        ],
        axis=1,
    )
    decibels = 10 * np.log10(erb + np.float32(1e-10))
    state = -60 + np.arange(32, dtype=np.float32) * np.float32(-30 / 31)
    unit = np.float32(0.001) + np.arange(96, dtype=np.float32) * np.float32((0.0001 - 0.001) / 95)
    erb_features, spec_features = np.empty_like(decibels), np.empty_like(spec[:, :96])
    alpha = np.float32(0.99)
    complement = np.float32(1) - alpha
    for index in range(len(spec)):
        state = decibels[index] * complement + state * alpha
        erb_features[index] = (decibels[index] - state) / 40
        unit = np.abs(spec[index, :96]) * complement + unit * alpha
        spec_features[index] = spec[index, :96] / np.sqrt(unit)
    return spec, erb_features, spec_features


def synthesis(spec):
    import numpy as np
    from scipy.fft import irfft

    frames = irfft(spec, n=960, axis=1) * np.float32(960) * window()
    result = np.zeros((len(spec) - 1) * 480 + 960, dtype=np.float32)
    for index, frame in enumerate(frames):
        result[index * 480 : index * 480 + 960] += frame
    return result[480:]


def enhance(session, audio, event):
    import numpy as np

    spec, erb, features = analysis(audio)
    enhanced = np.zeros_like(spec)
    for start in range(0, len(spec), 200):
        count = min(200, len(spec) - start)
        lookahead = min(200, max(0, len(spec) - start - 2))
        spectrum = np.zeros((1, 1, 200, 481, 2), dtype=np.float32)
        erb_input = np.zeros((1, 1, 200, 32), dtype=np.float32)
        feature_input = np.zeros((1, 1, 200, 96, 2), dtype=np.float32)
        spectrum[0, 0, :count, :, 0] = spec[start : start + count].real
        spectrum[0, 0, :count, :, 1] = spec[start : start + count].imag
        erb_input[0, 0, :lookahead] = erb[start + 2 : start + 2 + lookahead]
        feature_input[0, 0, :lookahead, :, 0] = features[start + 2 : start + 2 + lookahead].real
        feature_input[0, 0, :lookahead, :, 1] = features[start + 2 : start + 2 + lookahead].imag
        event("job.progress", {"stage": "running-model", "progress": 0.2 + 0.7 * start / len(spec)})
        output = session.run(
            ["spec_enhanced"], {"spec": spectrum, "feat_erb": erb_input, "feat_spec": feature_input}
        )[0]
        if output.shape != spectrum.shape or not np.isfinite(output).all():
            raise ValueError("ONNX_AUDIO_OUTPUT_INVALID")
        enhanced[start : start + count] = (
            output[0, 0, :count, :, 0] + 1j * output[0, 0, :count, :, 1]
        )
    return synthesis(enhanced)[: len(audio)]


def execute(session, value, path, output_path, event):
    import numpy as np

    input_value(value)
    audio = read_audio(path)
    if output_path.name != "result.wav" or output_path.exists() or output_path.is_symlink():
        raise ValueError("ONNX_AUDIO_OUTPUT_SCOPE")
    result = np.stack([enhance(session, channel, event) for channel in audio.T], axis=1)
    if result.shape != audio.shape or not np.isfinite(result).all() or np.max(np.abs(result)) > 8:
        raise ValueError("ONNX_AUDIO_OUTPUT_INVALID")
    with output_path.open("xb") as stream, wave.open(stream, "wb") as target:
        target.setnchannels(result.shape[1])
        target.setsampwidth(2)
        target.setframerate(48000)
        target.writeframes(np.rint(np.clip(result, -1, 1) * 32767).astype("<i2").tobytes())
    return {
        "relativeOutput": "result.wav",
        "mediaType": "audio/wav",
        "sampleRate": 48000,
        "channels": result.shape[1],
        "frames": len(result),
    }
