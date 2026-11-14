from __future__ import annotations

import hashlib
from typing import Any

from oryn_engine.compatibility_plane import (
    SOURCE_SCHEMA_VERSION,
    CapabilityRegistry,
    compile_registry,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _capability(
    identifier: str,
    source_kind: str,
    architecture: str,
    task: str,
    formats: list[str],
    family: str,
    task_adapter: str,
    conformance: str = "family-conformant",
) -> dict[str, Any]:
    return {
        "id": identifier,
        "sourceKind": source_kind,
        "architecture": architecture,
        "task": task,
        "formats": formats,
        "familyAdapter": {"id": family, "version": "1.0.0"},
        "taskAdapter": {"id": task_adapter, "version": "1.0.0"},
        "conformance": conformance,
    }


def framework_sources() -> tuple[dict[str, Any], ...]:
    from oryn_engine.generated_capabilities import generated_sources

    return generated_sources()


def synthetic_non_framework_source() -> dict[str, Any]:
    return {
        "schemaVersion": SOURCE_SCHEMA_VERSION,
        "provider": {
            "id": "oryn.source.synthetic",
            "version": "1.0.0",
            "generatorDigest": _digest("synthetic-provider-fixture-v1"),
        },
        "foundation": {
            "id": "oryn.runtime.synthetic",
            "version": "1.0.0",
            "digest": _digest("synthetic-foundation-v1"),
        },
        "capabilities": [
            _capability(
                "synthetic.table.reranking",
                "columnar-fixture",
                "bounded-pairwise-ranker",
                "text-ranking",
                ["columnar-v1"],
                "oryn.family.synthetic-ranker",
                "oryn.task.reranking-v1",
                "certified",
            )
        ],
    }


def framework_registry() -> CapabilityRegistry:
    return compile_registry(framework_sources())
