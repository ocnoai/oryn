from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

SCHEMA_VERSION = "oryn-model-descriptor/1"
SOURCE_SCHEMA_VERSION = "oryn-capability-source/1"
REGISTRY_SCHEMA_VERSION = "oryn-capability-registry/1"
IDENTIFIER = re.compile(r"^[a-z][a-z0-9.-]{0,127}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_SOURCE_KEYS = frozenset(
    {
        "callable",
        "command",
        "entrypoint",
        "environment",
        "executable",
        "loaderOptions",
        "path",
        "python",
        "repository",
        "shell",
    }
)

AxisName = Literal["access", "artifacts", "runtime", "task", "resources", "conformance"]
Disposition = Literal[
    "verified-load",
    "trust-required",
    "authorization-required",
    "not-admissible-on-this-device",
    "unsupported",
]


class CompatibilityPlaneError(ValueError):
    pass


@dataclass(frozen=True)
class ModelDescriptor:
    provider: str
    repository: str
    revision: str
    source_kind: str
    architectures: tuple[str, ...]
    task: str | None
    formats: tuple[str, ...]
    selected_files_digest: str
    requires_repository_code: bool
    access: str
    estimated_memory_bytes: int

    def __post_init__(self) -> None:
        if (
            self.provider != "huggingface"
            or not re.fullmatch(r"[A-Za-z0-9._-]{1,96}/[A-Za-z0-9._-]{1,96}", self.repository)
            or not re.fullmatch(r"[0-9a-f]{40,64}", self.revision)
            or not IDENTIFIER.fullmatch(self.source_kind)
            or not self.architectures
            or len(self.architectures) > 32
            or any(not 1 <= len(value) <= 128 for value in self.architectures)
            or len(self.formats) > 16
            or not DIGEST.fullmatch(self.selected_files_digest)
            or self.access not in {"available", "authorization-required", "denied"}
            or not 0 <= self.estimated_memory_bytes <= 2**63 - 1
        ):
            raise CompatibilityPlaneError("Model descriptor is invalid.")


@dataclass(frozen=True)
class NormalizedCapabilityTuple:
    capability_id: str
    source_kind: str
    architecture: str
    task: str
    formats: tuple[str, ...]
    foundation_id: str
    foundation_version: str
    family_adapter_id: str
    family_adapter_version: str
    task_adapter_id: str
    task_adapter_version: str
    conformance: str
    policy_state: str = "admitted"
    policy_reasons: tuple[str, ...] = ()
    constraints: dict[str, Any] | None = None

    def trusted_projection(self) -> dict[str, str]:
        return {
            "foundationId": self.foundation_id,
            "foundationVersion": self.foundation_version,
            "familyAdapterId": self.family_adapter_id,
            "familyAdapterVersion": self.family_adapter_version,
            "taskAdapterId": self.task_adapter_id,
            "taskAdapterVersion": self.task_adapter_version,
        }


@dataclass(frozen=True)
class CapabilityRegistry:
    schema_version: str
    tuples: tuple[NormalizedCapabilityTuple, ...]
    digest: str
    source_digests: tuple[str, ...]

    def match(self, descriptor: ModelDescriptor) -> NormalizedCapabilityTuple | None:
        candidates = [
            value
            for value in self.tuples
            if value.policy_state == "admitted"
            and value.source_kind == descriptor.source_kind
            and value.architecture in descriptor.architectures
            and (descriptor.task is None or value.task == descriptor.task)
            and set(value.formats).issubset(descriptor.formats)
        ]
        return min(candidates, key=lambda value: value.capability_id) if candidates else None


@dataclass(frozen=True)
class EligibilityAxis:
    state: str
    reasons: tuple[str, ...] = ()
    deferred_by: str | None = None

    def protocol_value(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reasons": list(self.reasons),
            **({"deferredBy": self.deferred_by} if self.deferred_by else {}),
        }


@dataclass(frozen=True)
class EligibilityResult:
    axes: dict[AxisName, EligibilityAxis]
    disposition: Disposition | None
    matched: NormalizedCapabilityTuple | None

    def protocol_value(self) -> dict[str, Any]:
        return {name: value.protocol_value() for name, value in self.axes.items()}


def compile_registry(sources: Iterable[object]) -> CapabilityRegistry:
    normalized: list[NormalizedCapabilityTuple] = []
    source_digests: list[str] = []
    seen_ids: set[str] = set()
    for raw in sources:
        if not isinstance(raw, dict) or set(raw) != {
            "schemaVersion",
            "provider",
            "foundation",
            "capabilities",
        }:
            raise CompatibilityPlaneError("Capability source envelope is invalid.")
        _reject_executable_fields(raw)
        if raw.get("schemaVersion") not in {SOURCE_SCHEMA_VERSION, "oryn-capability-source/2"}:
            raise CompatibilityPlaneError("Capability source schema is unsupported.")
        provider = _bounded_record(raw.get("provider"), {"id", "version", "generatorDigest"})
        foundation = _bounded_record(raw.get("foundation"), {"id", "version", "digest"})
        _identifier(provider.get("id"))
        _version(provider.get("version"))
        _digest(provider.get("generatorDigest"))
        foundation_id = _identifier(foundation.get("id"))
        foundation_version = _version(foundation.get("version"))
        _digest(foundation.get("digest"))
        capabilities = raw.get("capabilities")
        if not isinstance(capabilities, list) or not 1 <= len(capabilities) <= 4096:
            raise CompatibilityPlaneError("Capability source list is invalid.")
        for item in capabilities:
            capability = _bounded_record(
                item,
                {
                    "id",
                    "sourceKind",
                    "architecture",
                    "task",
                    "formats",
                    "familyAdapter",
                    "taskAdapter",
                    "conformance",
                }
                | (
                    {"policy", "constraints"}
                    if raw["schemaVersion"] == "oryn-capability-source/2"
                    else set()
                ),
            )
            capability_id = _identifier(capability.get("id"))
            if capability_id in seen_ids:
                raise CompatibilityPlaneError("Capability IDs must be unique.")
            seen_ids.add(capability_id)
            family = _bounded_record(capability.get("familyAdapter"), {"id", "version"})
            task_adapter = _bounded_record(capability.get("taskAdapter"), {"id", "version"})
            policy_state, policy_reasons, constraints = "admitted", (), None
            if raw["schemaVersion"] == "oryn-capability-source/2":
                policy = _bounded_record(capability["policy"], {"state", "reasons"})
                policy_state = _enum(policy["state"], {"admitted", "withheld"})
                reasons = policy["reasons"]
                if (
                    not isinstance(reasons, list)
                    or len(reasons) > 16
                    or any(
                        not isinstance(r, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{1,127}", r)
                        for r in reasons
                    )
                ):
                    raise CompatibilityPlaneError("Capability policy reasons are invalid.")
                policy_reasons = tuple(reasons)
                constraints = _bounded_record(
                    capability["constraints"],
                    {
                        "containerVersions",
                        "quantizations",
                        "tokenizers",
                        "templates",
                        "maxContextTokens",
                    },
                )
                for field in ("containerVersions", "quantizations", "tokenizers", "templates"):
                    _string_tuple(constraints[field], 32)
                if (
                    type(constraints["maxContextTokens"]) is not int
                    or not 1 <= constraints["maxContextTokens"] <= 8192
                ):
                    raise CompatibilityPlaneError("Capability context bound is invalid.")
            values = NormalizedCapabilityTuple(
                capability_id=capability_id,
                source_kind=_identifier(capability.get("sourceKind")),
                architecture=_bounded_string(capability.get("architecture"), 128),
                task=_identifier(capability.get("task")),
                formats=_string_tuple(capability.get("formats"), 16),
                foundation_id=foundation_id,
                foundation_version=foundation_version,
                family_adapter_id=_identifier(family.get("id")),
                family_adapter_version=_version(family.get("version")),
                task_adapter_id=_identifier(task_adapter.get("id")),
                task_adapter_version=_version(task_adapter.get("version")),
                policy_state=policy_state,
                policy_reasons=policy_reasons,
                constraints=constraints,
                conformance=_enum(
                    capability.get("conformance"),
                    {"certified", "family-conformant", "untested", "failed"},
                ),
            )
            normalized.append(values)
        source_digests.append(_canonical_digest(raw))
    ordered = tuple(sorted(normalized, key=lambda value: value.capability_id))
    material = {
        "schemaVersion": REGISTRY_SCHEMA_VERSION,
        "sourceDigests": sorted(source_digests),
        "capabilities": [value.__dict__ for value in ordered],
    }
    return CapabilityRegistry(
        REGISTRY_SCHEMA_VERSION,
        ordered,
        _canonical_digest(material),
        tuple(sorted(source_digests)),
    )


def evaluate_eligibility(
    descriptor: ModelDescriptor,
    registry: CapabilityRegistry,
    *,
    resource_admissible: bool | None,
    artifact_reasons: tuple[str, ...] = (),
    runtime_reasons: tuple[str, ...] = (),
    exact_certified: bool = False,
) -> EligibilityResult:
    matched = registry.match(descriptor)
    access = EligibilityAxis(
        descriptor.access, () if descriptor.access == "available" else ("MODEL_ACCESS_RESTRICTED",)
    )
    deferred = descriptor.access == "authorization-required"
    artifacts = _axis_or_deferred(
        "safe" if descriptor.formats else "incomplete",
        () if descriptor.formats else ("MODEL_SUBSET_INCOMPLETE",),
        deferred and not descriptor.formats,
    )
    runtime = _axis_or_deferred(
        "supported" if matched else "unsupported",
        () if matched else ("RUNTIME_CAPABILITY_UNSUPPORTED",),
        deferred and matched is None,
    )
    task = _axis_or_deferred(
        "supported" if matched else "unknown",
        () if matched else ("TASK_ADAPTER_UNAVAILABLE",),
        deferred and matched is None,
    )
    resources = _axis_or_deferred(
        "not-evaluated"
        if resource_admissible is None
        else "admissible"
        if resource_admissible
        else "not-admissible",
        () if resource_admissible is not False else ("MODEL_MEMORY_INSUFFICIENT",),
        resource_admissible is not None and deferred and descriptor.estimated_memory_bytes == 0,
    )
    if descriptor.requires_repository_code:
        runtime = EligibilityAxis("unsupported", ("CUSTOM_CODE_REQUIRED",))
    conformance_state = matched.conformance if matched else "untested"
    # Capability tables describe families. They cannot certify an arbitrary revision.
    if conformance_state == "certified" and not exact_certified:
        conformance_state = "family-conformant"
    if exact_certified and matched and conformance_state != "failed":
        conformance_state = "certified"
    conformance = _axis_or_deferred(
        conformance_state,
        ()
        if conformance_state in {"certified", "family-conformant"}
        else ("CONFORMANCE_REQUIRED",),
        deferred and matched is None,
    )
    axes: dict[AxisName, EligibilityAxis] = {
        "access": access,
        "artifacts": artifacts,
        "runtime": runtime,
        "task": task,
        "resources": resources,
        "conformance": conformance,
    }
    if artifact_reasons:
        axes["artifacts"] = EligibilityAxis("incomplete", artifact_reasons)
    if runtime_reasons:
        axes["runtime"] = EligibilityAxis("unsupported", runtime_reasons)
    return EligibilityResult(
        axes, _derive_disposition(axes) if resource_admissible is not None else None, matched
    )


def semantic_diff(
    previous: CapabilityRegistry, current: CapabilityRegistry
) -> dict[str, list[str]]:
    before = {value.capability_id: value for value in previous.tuples}
    after = {value.capability_id: value for value in current.tuples}
    return {
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
        "changed": sorted(key for key in before.keys() & after.keys() if before[key] != after[key]),
    }


def _derive_disposition(axes: dict[AxisName, EligibilityAxis]) -> Disposition:
    # Authorization is the disposition whenever access is the only unresolved
    # gate. Facts established without credentials stay established; requiring
    # every other axis to be deferred would incorrectly turn a known-compatible
    # gated model into ``unsupported``.
    hard_states = {
        "denied",
        "failed",
        "incomplete",
        "not-admissible",
        "unsafe",
        "unknown",
        "unsupported",
    }
    if axes["access"].state == "authorization-required" and all(
        name == "access" or value.state == "deferred" or value.state not in hard_states
        for name, value in axes.items()
    ):
        return "authorization-required"
    if axes["resources"].state == "not-admissible" and all(
        value.state not in {"unsafe", "unsupported", "failed", "denied"}
        for name, value in axes.items()
        if name != "resources"
    ):
        return "not-admissible-on-this-device"
    if (
        axes["access"].state != "available"
        or axes["artifacts"].state != "safe"
        or axes["runtime"].state != "supported"
        or axes["task"].state != "supported"
        or axes["conformance"].state == "failed"
    ):
        return "unsupported"
    return "verified-load" if axes["conformance"].state == "certified" else "trust-required"


def _axis_or_deferred(state: str, reasons: tuple[str, ...], deferred: bool) -> EligibilityAxis:
    return (
        EligibilityAxis("deferred", ("INSPECTION_DEFERRED_AUTHORIZATION",), "authorization")
        if deferred
        else EligibilityAxis(state, reasons)
    )


def _reject_executable_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_SOURCE_KEYS:
                raise CompatibilityPlaneError("Capability source contains executable authority.")
            _reject_executable_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_executable_fields(item)


def _bounded_record(value: object, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CompatibilityPlaneError("Capability source record is invalid.")
    return value


def _identifier(value: object) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise CompatibilityPlaneError("Capability identifier is invalid.")
    return value


def _version(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value):
        raise CompatibilityPlaneError("Capability version is invalid.")
    return value


def _digest(value: object) -> str:
    if not isinstance(value, str) or not DIGEST.fullmatch(value):
        raise CompatibilityPlaneError("Capability digest is invalid.")
    return value


def _bounded_string(value: object, maximum: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise CompatibilityPlaneError("Capability string is invalid.")
    return value


def _string_tuple(value: object, maximum: int) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= maximum
        or any(not isinstance(item, str) or not 1 <= len(item) <= 128 for item in value)
    ):
        raise CompatibilityPlaneError("Capability list is invalid.")
    return tuple(sorted(set(value)))


def _enum(value: object, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise CompatibilityPlaneError("Capability value is invalid.")
    return value


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
