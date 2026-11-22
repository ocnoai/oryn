import wave

import numpy as np
import pytest

from oryn_engine.vad_tasks import intervals, read_audio, validate_graph


def test_speech_intervals_use_seconds_and_do_not_bridge_silence():
    scores = [0.0] * 10 + [0.9] * 20 + [0.0] * 10 + [0.9] * 20 + [0.0] * 10
    result = intervals(scores, 2.24, 0.5, 0.25, 0.1)
    assert result == [{"start": 0.32, "end": 0.96}, {"start": 1.28, "end": 1.92}]
    assert intervals([0.0] * 100, 3.2, 0.5, 0.25, 0.1) == []
    assert intervals([0.9] * 3, 0.09, 0.5, 0.25, 0.1) == []
    assert intervals([0.9] * 32, 1, 0.5, 0.25, 0.1) == [{"start": 0, "end": 1}]
    for bad in (float("nan"), 1.1, -0.1):
        with pytest.raises(ValueError):
            intervals([bad], 0.032, 0.5, 0.25, 0.1)


def test_pcm_decoder_keeps_duration_and_rejects_long_corrupt_or_symlinked_audio(tmp_path):
    path = tmp_path / "audio.wav"

    def write(seconds, rate=48000):
        with wave.open(str(path), "wb") as stream:
            stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            stream.writeframes(np.zeros(int(seconds * rate), dtype="<i2").tobytes())

    write(2)
    audio, duration = read_audio(path)
    assert duration == 2 and audio.shape == (32000,) and not audio.any()
    write(30.1)
    with pytest.raises(ValueError):
        read_audio(path)
    write(1)
    link = tmp_path / "link.wav"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        read_audio(link)
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError, match="TRUNCATED"):
        read_audio(path)


def test_vad_never_borrows_a_different_recurrent_state_binding():
    graph = {
        "inputs": [
            {"name": "input", "dtype": 1, "shape": [None, None]},
            {"name": "state", "dtype": 1, "shape": [2, None, 128]},
            {"name": "sr", "dtype": 7, "shape": []},
        ],
        "outputs": [
            {"name": "output", "dtype": 1, "shape": [None, 1]},
            {"name": "stateN", "dtype": 1, "shape": [None, None, None]},
        ],
    }
    validate_graph(graph)
    graph["inputs"][1]["shape"][-1] = 64
    with pytest.raises(ValueError, match="ONNX_VAD_BINDING_UNSUPPORTED"):
        validate_graph(graph)
