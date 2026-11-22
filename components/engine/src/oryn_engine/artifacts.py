from __future__ import annotations

import contextlib
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

ARTIFACT_ID = re.compile(r"^artifact_[A-Za-z0-9_-]{20,80}$")
DOMAIN_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{1,127}$")


class ArtifactBindingError(ValueError):
    """A trusted input binding failed defense-in-depth validation."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class EngineInputArtifactBinding:
    artifact_id: str
    internal_path: Path
    mime_type: str
    size_bytes: int
    name: str | None
    access: str = "read"


class ArtifactInputValidator:
    def __init__(self, input_root: Path) -> None:
        if not input_root.is_absolute():
            raise ArtifactBindingError("ARTIFACT_PATH_INVALID")
        try:
            self.input_root = input_root.resolve(strict=True)
        except OSError as exc:
            raise ArtifactBindingError("ARTIFACT_PATH_INVALID") from exc
        if not self.input_root.is_dir():
            raise ArtifactBindingError("ARTIFACT_PATH_INVALID")

    def validate(self, value: object) -> EngineInputArtifactBinding:
        if not isinstance(value, dict):
            raise ArtifactBindingError("ARTIFACT_BINDING_INVALID")
        artifact_id = value.get("artifactId")
        internal_path = value.get("internalPath")
        mime_type = value.get("mimeType")
        size_bytes = value.get("sizeBytes")
        name = value.get("name")
        access = value.get("access")
        if (
            not isinstance(artifact_id, str)
            or ARTIFACT_ID.fullmatch(artifact_id) is None
            or not isinstance(internal_path, str)
            or not isinstance(mime_type, str)
            or not mime_type
            or len(mime_type) > 255
            or any(character in mime_type for character in "\r\n\0")
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
            or (name is not None and (not isinstance(name, str) or len(name) > 255))
            or access != "read"
        ):
            raise ArtifactBindingError("ARTIFACT_BINDING_INVALID")

        candidate = Path(internal_path)
        if not candidate.is_absolute() or candidate.is_symlink():
            raise ArtifactBindingError("ARTIFACT_PATH_INVALID")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.input_root)
            stat = resolved.stat()
        except (OSError, ValueError) as exc:
            raise ArtifactBindingError("ARTIFACT_PATH_INVALID") from exc
        if not resolved.is_file() or stat.st_size != size_bytes:
            raise ArtifactBindingError("ARTIFACT_BINDING_INVALID")
        return EngineInputArtifactBinding(
            artifact_id=artifact_id,
            internal_path=resolved,
            mime_type=mime_type,
            size_bytes=size_bytes,
            name=name,
        )

    def validate_all(self, values: object) -> tuple[EngineInputArtifactBinding, ...]:
        if not isinstance(values, list) or len(values) > 32:
            raise ArtifactBindingError("ARTIFACT_BINDING_INVALID")
        # No binding is returned until every required input has validated.
        return tuple(self.validate(value) for value in values)


def protocol_binding(binding: EngineInputArtifactBinding) -> dict[str, Any]:
    return {
        "artifactId": binding.artifact_id,
        "internalPath": str(binding.internal_path),
        "mimeType": binding.mime_type,
        "sizeBytes": binding.size_bytes,
        **({"name": binding.name} if binding.name is not None else {}),
        "access": "read",
    }


@dataclass(frozen=True)
class EngineOutputArtifact:
    output_id: str
    internal_path: Path
    mime_type: str
    size_bytes: int
    name: str | None = None

    def protocol_descriptor(self) -> dict[str, Any]:
        return {
            "outputId": self.output_id,
            "internalPath": str(self.internal_path),
            "mimeType": self.mime_type,
            "sizeBytes": self.size_bytes,
            **({"name": self.name} if self.name is not None else {}),
        }


class OutputArtifactWriter:
    """A bounded-scope writer whose descriptor exists only after fsync/close/finalize."""

    def __init__(
        self,
        staging: ArtifactOutputStaging,
        job_id: str,
        mime_type: str,
        name: str | None,
    ) -> None:
        self._staging = staging
        self._job_id = staging.validate_id(job_id)
        self._mime_type = validate_mime_type(mime_type)
        self._name = validate_output_name(name)
        self.output_id = f"output_{uuid.uuid4().hex}"
        job_root = staging.job_root(self._job_id)
        job_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._temporary = job_root / f".{self.output_id}.partial"
        self._final = job_root / self.output_id
        self._stream: BinaryIO = self._temporary.open("xb")
        self._closed = False
        self._finalized = False

    def write(self, value: bytes | bytearray | memoryview) -> int:
        if self._closed or self._finalized:
            raise ArtifactBindingError("ARTIFACT_IO_ERROR")
        return self._stream.write(value)

    def finalize(self) -> EngineOutputArtifact:
        if self._finalized:
            raise ArtifactBindingError("ARTIFACT_IO_ERROR")
        try:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            self._closed = True
            os.replace(self._temporary, self._final)
            self._final.chmod(0o600)
            resolved = self._staging.validate_final_path(self._job_id, self._final)
            size_bytes = resolved.stat().st_size
        except OSError as exc:
            self.abort()
            raise ArtifactBindingError("ARTIFACT_IO_ERROR") from exc
        self._finalized = True
        return EngineOutputArtifact(
            output_id=self.output_id,
            internal_path=resolved,
            mime_type=self._mime_type,
            size_bytes=size_bytes,
            name=self._name,
        )

    def abort(self) -> None:
        if not self._closed:
            with contextlib.suppress(OSError):
                self._stream.close()
            self._closed = True
        self._temporary.unlink(missing_ok=True)
        if not self._finalized:
            self._final.unlink(missing_ok=True)

    def __enter__(self) -> OutputArtifactWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if not self._finalized:
            self.abort()


class ArtifactOutputStaging:
    """Engine-owned staging constrained to one engine-instance/job subtree."""

    def __init__(self, root: Path, engine_instance_id: str) -> None:
        if not root.is_absolute():
            raise ArtifactBindingError("ARTIFACT_PATH_INVALID")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root = root.resolve(strict=True)
        if not self.root.is_dir() or self.root.is_symlink():
            raise ArtifactBindingError("ARTIFACT_PATH_INVALID")
        self.engine_instance_id = self.validate_id(engine_instance_id)
        self.cleanup_orphans()
        self.instance_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    @property
    def instance_root(self) -> Path:
        return self.root / self.engine_instance_id

    @staticmethod
    def validate_id(value: str) -> str:
        if not isinstance(value, str) or DOMAIN_ID.fullmatch(value) is None:
            raise ArtifactBindingError("ARTIFACT_PATH_INVALID")
        return value

    def job_root(self, job_id: str) -> Path:
        return self.instance_root / self.validate_id(job_id)

    def writer(self, job_id: str, mime_type: str, name: str | None = None) -> OutputArtifactWriter:
        return OutputArtifactWriter(self, job_id, mime_type, name)

    def validate_final_path(self, job_id: str, value: Path) -> Path:
        if not value.is_absolute() or value.is_symlink():
            raise ArtifactBindingError("ARTIFACT_PATH_INVALID")
        try:
            resolved = value.resolve(strict=True)
            resolved.relative_to(self.job_root(job_id).resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise ArtifactBindingError("ARTIFACT_PATH_INVALID") from exc
        if not resolved.is_file():
            raise ArtifactBindingError("ARTIFACT_PATH_INVALID")
        return resolved

    def cleanup_job(self, job_id: str) -> None:
        shutil.rmtree(self.job_root(job_id), ignore_errors=True)

    def cleanup_orphans(self) -> None:
        for child in self.root.iterdir():
            if child.name != self.engine_instance_id:
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        # A restarted instance ID should never inherit partially finalized files.
        if self.instance_root.exists():
            shutil.rmtree(self.instance_root, ignore_errors=True)


def validate_mime_type(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 255
        or any(character in value for character in "\r\n\0")
    ):
        raise ArtifactBindingError("ARTIFACT_BINDING_INVALID")
    return value.strip().lower()


def validate_output_name(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 255
        or any(character in value for character in "\r\n\0")
    ):
        raise ArtifactBindingError("ARTIFACT_BINDING_INVALID")
    return value
