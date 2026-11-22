"""Task-specific audio bounds are checked before payload allocation."""

import struct
import threading
import wave
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from oryn_engine.jobs import JobExecutionError
from oryn_engine.runners import RunnerRegistry, TransformersASRHandle, _decode_wave


def silence(path, seconds):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(1)
        stream.setframerate(16_000)
        stream.writeframes(b"\x80" * (16_000 * seconds))


@pytest.mark.parametrize("seconds", [30, 31])
def test_transcription_enforces_profile_duration_before_processing(tmp_path, seconds):
    path = tmp_path / "synthetic.wav"
    silence(path, seconds)
    seen = []

    def processor(audio, **_kwargs):
        seen.append(int(audio.size))
        raise RuntimeError("Stop at bounded preprocessing; no model execution")

    job = SimpleNamespace(
        input={"audioArtifactId": "artifact_probe", "language": "auto", "timestamps": False},
        input_artifacts=(
            {
                "artifactId": "artifact_probe",
                "internalPath": str(path),
                "mimeType": "audio/wav",
                "sizeBytes": path.stat().st_size,
            },
        ),
    )
    with pytest.raises(JobExecutionError) as error:
        if seconds == 31:
            with patch.object(
                wave.Wave_read, "readframes", side_effect=AssertionError("Oversized payload read")
            ):
                RunnerRegistry._transcribe(
                    job,
                    TransformersASRHandle(None, processor, "cpu"),
                    threading.Event(),
                    lambda *_: None,
                )
        else:
            RunnerRegistry._transcribe(
                job,
                TransformersASRHandle(None, processor, "cpu"),
                threading.Event(),
                lambda *_: None,
            )
    assert error.value.code == ("AUDIO_TOO_LARGE" if seconds == 31 else "AUDIO_PROCESSING_FAILED")
    assert seen == ([] if seconds == 31 else [480_000])


def test_decoder_preserves_existing_non_transcription_duration_default(tmp_path):
    path = tmp_path / "synthetic.wav"
    silence(path, 31)
    assert _decode_wave(path, path.stat().st_size).size == 496_000


def header(path, data_bytes, *, channels=1, rate=16_000, width=1):
    path.write_bytes(
        struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            data_bytes + 36,
            b"WAVE",
            b"fmt ",
            16,
            1,
            channels,
            rate,
            rate * channels * width,
            channels * width,
            width * 8,
            b"data",
            data_bytes,
        )
    )


def test_projected_pcm_size_cannot_bypass_existing_file_budget(tmp_path):
    path = tmp_path / "header-only.wav"
    # Only a 44-byte fixture is created. The header claims >1GiB within four hours.
    header(path, 2 * 1024**3, channels=32, rate=192_000)
    with (
        patch.object(
            wave.Wave_read, "readframes", side_effect=AssertionError("Oversized payload read")
        ),
        pytest.raises(JobExecutionError) as error,
    ):
        _decode_wave(path, path.stat().st_size)
    assert error.value.code == "AUDIO_TOO_LARGE"


def test_truncated_pcm_is_rejected(tmp_path):
    path = tmp_path / "header-only.wav"
    header(path, 16_000)
    with pytest.raises(JobExecutionError) as error:
        _decode_wave(path, path.stat().st_size)
    assert error.value.code == "AUDIO_DECODE_FAILED"


def test_transcription_rejects_excessive_decoded_samples_before_reading(tmp_path):
    path = tmp_path / "header-only.wav"
    # A 44-byte fixture describes a legal-duration but excessive multichannel input.
    header(path, 30 * 192_000 * 32 * 3, channels=32, rate=192_000, width=3)
    job = SimpleNamespace(
        input={"audioArtifactId": "artifact_probe", "language": "auto", "timestamps": False},
        input_artifacts=(
            {
                "artifactId": "artifact_probe",
                "internalPath": str(path),
                "mimeType": "audio/wav",
                "sizeBytes": path.stat().st_size,
            },
        ),
    )
    with (
        patch.object(
            wave.Wave_read, "readframes", side_effect=AssertionError("Oversized payload read")
        ),
        pytest.raises(JobExecutionError) as error,
    ):
        RunnerRegistry._transcribe(
            job, TransformersASRHandle(None, None, "cpu"), threading.Event(), lambda *_: None
        )
    assert error.value.code == "AUDIO_TOO_LARGE"
