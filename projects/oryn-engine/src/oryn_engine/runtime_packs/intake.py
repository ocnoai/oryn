from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from oryn_engine.runtime_packs.registry import RuntimePackRegistry


@dataclass(frozen=True)
class IntakeResult:
    state: str
    evidence_digest: str
    matches: tuple[str, ...]
    blocking_reasons: tuple[str, ...]


class ModelIntakeHarness:
    """Deterministic static intake before hardware/quality conformance is attempted."""

    def __init__(self, registry: RuntimePackRegistry) -> None:
        self._registry = registry

    def inspect(self, candidate: dict[str, Any]) -> IntakeResult:
        repository = candidate.get("repository")
        revision = candidate.get("revision")
        architectures = candidate.get("architectures", [])
        task = candidate.get("task")
        files = candidate.get("files", [])
        reasons: list[str] = []
        if candidate.get("requiresCustomCode"):
            reasons.append("CUSTOM_CODE_REQUIRED")
        if not isinstance(files, list) or any(
            not isinstance(item, str) or item.startswith("/") or ".." in item.split("/")
            for item in files
        ):
            reasons.append("MODEL_METADATA_INVALID")
        matches = ()
        if (
            isinstance(repository, str)
            and isinstance(revision, str)
            and isinstance(architectures, list)
            and all(isinstance(item, str) for item in architectures)
            and (task is None or isinstance(task, str))
        ):
            matches = tuple(
                item.manifest_id
                for item in self._registry.match(
                    repository=repository,
                    revision=revision,
                    architectures=architectures,
                    task=task,
                )
            )
        else:
            reasons.append("MODEL_METADATA_INVALID")
        if not matches:
            reasons.append("RUNNER_UNSUPPORTED")
        evidence = json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
        return IntakeResult(
            "static-rejected" if reasons else "static-approved",
            hashlib.sha256(evidence).hexdigest(),
            matches,
            tuple(dict.fromkeys(reasons)),
        )
