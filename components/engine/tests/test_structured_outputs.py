import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from oryn_engine.structured_outputs import MAX_BYTES, read_result, write_result


def vectors():
    vector = [1 / 64] * 4096
    return dict(
        dimensions=4096,
        normalized=True,
        scoring="sum-max-cosine",
        documentVectors=[vector] * 32,
        queries=[dict(vectors=[vector], score=1)],
    )


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "path",
        "mime",
        "backend",
        "size",
        "digest",
        "extra",
        "truncated",
        "symlink",
        "directory",
        "missing",
    ],
)
def test_file_binding_preserves_full_vectors_and_rejects_substitution(tmp_path, damage):
    path = tmp_path / "result.json"
    value = vectors()
    descriptor = write_result(path, value, "cpu")
    assert path.stat().st_size > 1024 * 1024
    if damage in {"path", "mime", "backend", "size", "digest", "extra"}:
        key, changed = {
            "path": ("relativeOutput", "../other.json"),
            "mime": ("mediaType", "text/plain"),
            "backend": ("backend", "mps"),
            "size": ("sizeBytes", MAX_BYTES + 1),
            "digest": ("sha256", "a" * 64),
            "extra": ("modelRoot", "/outside"),
        }[damage]
        descriptor[key] = changed
    elif damage == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    elif damage == "symlink":
        other = tmp_path / "outside.json"
        path.rename(other)
        path.symlink_to(other)
    elif damage in {"directory", "missing"}:
        path.unlink()
        if damage == "directory":
            path.mkdir()
    if damage:
        with pytest.raises(ValueError):
            read_result(path, descriptor, "cpu")
    else:
        assert read_result(path, descriptor, "cpu") == value
        with pytest.raises(FileExistsError):
            write_result(path, dict(changed=True), "cpu")
        assert read_result(path, descriptor, "cpu") == value


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":1,"x":2}',
        b"[]",
        b'{"x":',
        b"\xff",
        b'{"x":' + b"[" * 33 + b"0" + b"]" * 33 + b"}",
    ],
)
def test_invalid_structured_json_is_not_parsed_as_a_result(tmp_path, raw):
    path = tmp_path / "result.json"
    path.write_bytes(raw)
    descriptor = dict(
        relativeOutput=path.name,
        mediaType="application/json",
        backend="cpu",
        sizeBytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    with pytest.raises(ValueError):
        read_result(path, descriptor, "cpu")


@pytest.mark.parametrize("value", [dict(x=float("nan")), dict(x="x" * MAX_BYTES), []])
def test_invalid_or_oversized_write_leaves_no_output(tmp_path, value):
    path = tmp_path / "result.json"
    with pytest.raises(ValueError):
        write_result(path, value, "cpu")
    assert not path.exists()


@pytest.mark.parametrize("damage", [None, "shape", "score", "digest", "backend", "cancel"])
def test_shared_runner_consumes_only_its_bound_file_then_cleans_it(tmp_path, damage):
    from oryn_engine.jobs import EngineJob, JobExecutionError
    from oryn_engine.runners import RunnerRegistry

    cache = tmp_path / "models"
    model = dict(provider="huggingface", repository="fixture/retrieval", revision="a" * 40)
    root = cache / "models--fixture--retrieval" / "snapshots" / model["revision"]
    root.mkdir(parents=True)
    image = tmp_path / "page.png"
    image.write_bytes(b"leased input; fixture worker does not decode")
    artifact = "artifact_" + "a" * 24
    job = EngineJob(
        job_id="job_vectors",
        session_id="session_vectors",
        model=model,
        input=dict(imageArtifactId=artifact, queries=["query"]),
        input_artifacts=(
            dict(
                artifactId=artifact,
                internalPath=str(image),
                mimeType="image/png",
                sizeBytes=image.stat().st_size,
            ),
        ),
        invocation_digest="digest",
        state="running",
        enqueued_at=0,
    )
    handle = SimpleNamespace(
        record=dict(model=model, cacheRoot=str(cache)),
        pack=SimpleNamespace(pack_id="oryn.transformers-pytorch"),
        runtime=None,
        task="visual-document-retrieval",
        backend="cpu",
    )
    cancellation = threading.Event()
    paths = []
    registry = RunnerRegistry()

    def invoke(record, pack, runtime, command, payload, **kwargs):
        assert payload["inputPath"] == str(image)
        path = Path(payload["outputPath"])
        paths.append(path)
        assert path.name == "result.json"
        result = vectors()
        if damage == "shape":
            result["documentVectors"][0] = result["documentVectors"][0][:-1]
        elif damage == "score":
            result["queries"][0]["score"] = 0.1
        descriptor = write_result(path, result, "mps" if damage == "backend" else "cpu")
        if damage == "digest":
            path.write_bytes(path.read_bytes().replace(b"0.015625", b"0.015626", 1))
        elif damage == "cancel":
            cancellation.set()
        return descriptor

    registry._invoke_pack = invoke
    if damage:
        with pytest.raises(JobExecutionError) as caught:
            registry._run_shared_task(job, handle, cancellation, lambda *_: None)
        assert caught.value.code == (
            "JOB_CANCELLED" if damage == "cancel" else "PACK_OUTPUT_INVALID"
        )
    else:
        result = registry._run_shared_task(job, handle, cancellation, lambda *_: None)
        assert result == dict(kind="structured", value=vectors())
    assert paths and all(not path.parent.exists() for path in paths)
