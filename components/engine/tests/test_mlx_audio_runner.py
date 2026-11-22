"""MLX-Audio integration with installed records, artifact bindings and the runner."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import wave
from pathlib import Path

import pytest

from oryn_engine.artifacts import ArtifactOutputStaging
from oryn_engine.jobs import EngineJob, JobExecutionError
from oryn_engine.runners import RunnerRegistry, SharedTaskPackHandle
from oryn_engine.runtime_packs.execution import PackRuntime

ARTIFACT = "artifact_" + "r" * 24


def record(tmp_path, task="text-to-speech", *, candidate=True):
    revision = "a" * 40
    cache = tmp_path / "cache"
    snapshot = cache / "models--fixture--mlx-audio" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    contents = {
        "config.json": json.dumps({"model_type": "fish_drama3"}).encode(),
        "model.safetensors": b"selected fixture bytes",
    }
    files = []
    for name, data in contents.items():
        path = snapshot / name
        path.write_bytes(data)
        files.append(
            {
                "name": name,
                "size": len(data),
                "content_sha256": hashlib.sha256(data).hexdigest(),
                "cache_path": str(path.relative_to(cache)),
            }
        )
    # The selected-model view must not expose unrelated repository contents.
    (snapshot / "unselected.py").write_text("raise RuntimeError('never load')")
    value = {
        "model": {
            "provider": "huggingface",
            "repository": "fixture/mlx-audio",
            "revision": revision,
        },
        "runner": "runtime-pack",
        "cacheRoot": str(cache),
        "files": files,
        "runtimePack": {
            "packId": "oryn.mlx-audio",
            "runnerId": "mlx-audio.shared-tasks",
            "taskContract": "speech-synthesis-v1"
            if task == "text-to-speech"
            else "transcription-v1",
        },
        "technicalCapabilities": {"task": task},
    }
    if candidate:
        value["_runtimeCandidate"] = {"backend": "mlx"}
    return value


def registry(tmp_path, monkeypatch):
    outputs = ArtifactOutputStaging(tmp_path / "outputs", "engine_mlx_audio")
    result = RunnerRegistry(outputs)
    runtime = PackRuntime(Path(sys.executable).resolve(), None, inherit_base_runtime=True)
    monkeypatch.setattr(result._pack_locator, "resolve", lambda _pack: runtime)
    return result


def reference(tmp_path):
    path = tmp_path / "reference.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(1600 * 2))
    return {
        "artifactId": ARTIFACT,
        "internalPath": str(path),
        "mimeType": "audio/wav",
        "sizeBytes": path.stat().st_size,
        "access": "read",
    }


def job(model_record, value, bindings=()):
    return EngineJob(
        job_id="job_mlx_audio",
        session_id="session_mlx_audio",
        model=model_record["model"],
        input=value,
        input_artifacts=bindings,
        invocation_digest="fixture-digest",
        state="running",
        enqueued_at=0,
    )


class AudioProcess:
    def __init__(self, *, damage=None, cancel=False):
        self.calls = []
        self.damage = damage
        self.cancel = cancel

    def invoke(self, pack, runtime, command, payload, **options):
        del runtime
        assert (pack.pack_id, command, payload["backend"]) == ("oryn.mlx-audio", "generate", "mlx")
        view = Path(payload["modelRoot"])
        assert {path.name for path in view.iterdir()} == {"config.json", "model.safetensors"}
        self.calls.append(dict(payload))
        options["event_sink"]("job.progress", {"stage": "running-model", "progress": 0.3})
        if payload["task"] == "automatic-speech-recognition":
            assert payload["outputPath"] is None
            result = {"text": "A local transcription.", "language": "en"}
        else:
            output = Path(payload["outputPath"])
            with wave.open(str(output), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(44100)
                audio.writeframes(bytes(4410 * 2))
            result = {
                "relativeOutput": "result.wav",
                "mediaType": "audio/wav",
                "sampleRate": 44100,
                "channels": 1,
                "frames": 4410,
                "durationSeconds": 0.1,
                "backend": "mlx",
            }
        if self.damage:
            result.update(self.damage)
        if self.cancel:
            options["cancellation"].set()
        return result


@pytest.mark.parametrize("candidate", [True, False])
def test_audio_pack_loads_as_shared_task_including_legacy_records(tmp_path, monkeypatch, candidate):
    runners = registry(tmp_path, monkeypatch)
    handle = runners.load(record(tmp_path, candidate=candidate))
    assert isinstance(handle, SharedTaskPackHandle)
    assert (handle.pack.pack_id, handle.task, handle.backend) == (
        "oryn.mlx-audio",
        "text-to-speech",
        "mlx",
    )


def test_authorized_reference_and_transcript_reach_worker_and_native_wav_is_staged(
    tmp_path, monkeypatch
):
    runners = registry(tmp_path, monkeypatch)
    model_record = record(tmp_path)
    handle = runners.load(model_record)
    process = AudioProcess()
    runners._pack_process = process
    binding = reference(tmp_path)
    value = {
        "text": "Hello.",
        "voice": "reference",
        "audioArtifactId": ARTIFACT,
        "referenceText": "The words in the reference.",
    }
    events = []
    result = runners.execute(
        job(model_record, value, (binding,)),
        handle,
        threading.Event(),
        lambda name, data: events.append((name, data)),
    )
    sent = process.calls[0]
    assert sent["inputPath"] == binding["internalPath"]
    assert sent["input"]["referenceText"] == value["referenceText"]
    assert sent["input"]["language"] == "auto"
    assert result["kind"] == "artifact"
    assert result["artifact"]["mimeType"] == "audio/wav"
    output = Path(result["artifact"]["internalPath"])
    with wave.open(str(output), "rb") as audio:
        assert (audio.getframerate(), audio.getnframes()) == (44100, 4410)
    assert events == [("job.progress", {"stage": "running-model", "progress": 0.3})]
    assert not Path(sent["modelRoot"]).exists()
    assert not Path(sent["outputPath"]).exists()


@pytest.mark.parametrize(
    "damage",
    [
        {"sampleRate": 24000},
        {"frames": 4409},
        {"backend": "cpu"},
        {"durationSeconds": float("nan")},
        {"relativeOutput": "another.wav"},
    ],
)
def test_forged_waveform_metadata_never_publishes_an_artifact(tmp_path, monkeypatch, damage):
    runners = registry(tmp_path, monkeypatch)
    model_record = record(tmp_path)
    runners._pack_process = AudioProcess(damage=damage)
    handle = runners.load(model_record)
    with pytest.raises(JobExecutionError) as error:
        runners.execute(
            job(model_record, {"text": "Hello."}), handle, threading.Event(), lambda *_: None
        )
    assert error.value.code == "PACK_OUTPUT_INVALID"
    assert not list(runners.outputs.instance_root.rglob("output_*"))


def test_cancellation_after_worker_result_prevents_publication_and_cleans_temporary_data(
    tmp_path, monkeypatch
):
    runners = registry(tmp_path, monkeypatch)
    model_record = record(tmp_path)
    process = AudioProcess(cancel=True)
    runners._pack_process = process
    handle = runners.load(model_record)
    with pytest.raises(JobExecutionError) as error:
        runners.execute(
            job(model_record, {"text": "Hello."}), handle, threading.Event(), lambda *_: None
        )
    assert error.value.code == "JOB_CANCELLED"
    assert not list(runners.outputs.instance_root.rglob("output_*"))
    assert not Path(process.calls[0]["outputPath"]).exists()
    assert not Path(process.calls[0]["modelRoot"]).exists()


def test_transcription_uses_authorized_audio_and_standard_structured_result(tmp_path, monkeypatch):
    runners = registry(tmp_path, monkeypatch)
    model_record = record(tmp_path, "automatic-speech-recognition")
    process = AudioProcess()
    runners._pack_process = process
    handle = runners.load(model_record)
    binding = reference(tmp_path)
    result = runners.execute(
        job(model_record, {"audioArtifactId": ARTIFACT}, (binding,)),
        handle,
        threading.Event(),
        lambda *_: None,
    )
    assert result == {
        "kind": "structured",
        "value": {"text": "A local transcription.", "language": "en"},
    }
    assert process.calls[0]["inputPath"] == binding["internalPath"]
    assert process.calls[0]["input"]["timestamps"] is False


def test_unbound_reference_is_rejected_before_worker_launch(tmp_path, monkeypatch):
    runners = registry(tmp_path, monkeypatch)
    model_record = record(tmp_path)
    process = AudioProcess()
    runners._pack_process = process
    handle = runners.load(model_record)
    with pytest.raises(JobExecutionError) as error:
        runners.execute(
            job(model_record, {"text": "Hello", "voice": "reference", "audioArtifactId": ARTIFACT}),
            handle,
            threading.Event(),
            lambda *_: None,
        )
    assert error.value.code == "ARTIFACT_BINDING_INVALID"
    assert process.calls == []
