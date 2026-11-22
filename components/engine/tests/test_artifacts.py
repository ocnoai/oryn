from pathlib import Path

import pytest

from oryn_engine.artifacts import (
    ArtifactBindingError,
    ArtifactInputValidator,
    ArtifactOutputStaging,
)


def binding(path: Path, *, size: int = 3) -> dict[str, object]:
    return {
        "artifactId": "artifact_abcdefghijklmnopqrstuvwxyz",
        "internalPath": str(path),
        "mimeType": "audio/wav",
        "sizeBytes": size,
        "name": "sample.wav",
        "access": "read",
    }


def test_validates_read_only_regular_file_inside_controlled_root(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    artifact = input_root / "object"
    artifact.write_bytes(b"abc")

    result = ArtifactInputValidator(input_root).validate(binding(artifact))

    assert result.internal_path == artifact.resolve()
    assert result.access == "read"


@pytest.mark.parametrize("path_kind", ["traversal", "symlink", "directory"])
def test_rejects_path_escape_and_non_regular_input(tmp_path: Path, path_kind: str) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    outside = tmp_path / "private"
    outside.write_bytes(b"abc")
    if path_kind == "traversal":
        candidate = input_root / ".." / "private"
    elif path_kind == "symlink":
        candidate = input_root / "link"
        candidate.symlink_to(outside)
    else:
        candidate = input_root

    with pytest.raises(ArtifactBindingError) as raised:
        ArtifactInputValidator(input_root).validate(binding(candidate))

    assert raised.value.code in {"ARTIFACT_PATH_INVALID", "ARTIFACT_BINDING_INVALID"}


def test_rejects_size_or_access_mismatch_atomically(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    first = input_root / "first"
    second = input_root / "second"
    first.write_bytes(b"abc")
    second.write_bytes(b"xyz")
    validator = ArtifactInputValidator(input_root)

    with pytest.raises(ArtifactBindingError, match="ARTIFACT_BINDING_INVALID"):
        validator.validate_all([binding(first), binding(second, size=4)])
    invalid_access = binding(first)
    invalid_access["access"] = "write"
    with pytest.raises(ArtifactBindingError, match="ARTIFACT_BINDING_INVALID"):
        validator.validate(invalid_access)


def test_output_writer_flushes_closes_and_atomically_finalizes_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        "oryn_engine.artifacts.os.fsync", lambda descriptor: calls.append(descriptor)
    )
    staging = ArtifactOutputStaging(tmp_path / "staging", "engine_current")

    with staging.writer("job_output", "audio/wav", "generated.wav") as writer:
        writer.write(b"generated-bytes")
        descriptor = writer.finalize()

    assert calls
    assert descriptor.internal_path.read_bytes() == b"generated-bytes"
    assert descriptor.size_bytes == len(b"generated-bytes")
    assert descriptor.protocol_descriptor()["internalPath"] == str(descriptor.internal_path)
    assert not list(descriptor.internal_path.parent.glob("*.partial"))


def test_output_staging_rejects_escape_and_cleans_orphan_instances(tmp_path: Path) -> None:
    root = tmp_path / "staging"
    orphan = root / "engine_old" / "job_old"
    orphan.mkdir(parents=True)
    (orphan / "partial").write_bytes(b"partial")

    staging = ArtifactOutputStaging(root, "engine_current")
    assert not (root / "engine_old").exists()
    outside = tmp_path / "outside"
    outside.write_bytes(b"no")
    with pytest.raises(ArtifactBindingError, match="ARTIFACT_PATH_INVALID"):
        staging.validate_final_path("job_output", outside)
