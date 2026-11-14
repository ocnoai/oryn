from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oryn_engine.runtime_packs.manifests import (
    MAX_TASK_CONTRACTS,
    ManifestError,
    SignedManifestVerifier,
    _identifier,
    _record,
    _string,
    _string_tuple,
)


@dataclass(frozen=True)
class FoundationManifest:
    foundation_id: str
    version: str
    publisher: str
    abi: str
    minimum_safe_version: str
    dependency_closure: str
    platforms: tuple[str, ...]
    backends: tuple[str, ...]
    formats: tuple[str, ...]
    dtypes: tuple[str, ...]
    quantizations: tuple[str, ...]
    worker_abi: str
    network_policy: str
    filesystem_roots: tuple[str, ...]
    license_summary: str
    license_notice: str
    digest: str


@dataclass(frozen=True)
class AdapterManifest:
    adapter_id: str
    version: str
    publisher: str
    foundation_id: str
    foundation_abi: str
    minimum_foundation_version: str
    architectures: tuple[str, ...]
    formats: tuple[str, ...]
    task_contracts: tuple[str, ...]
    backends: tuple[str, ...]
    dtypes: tuple[str, ...]
    quantizations: tuple[str, ...]
    required_files: tuple[str, ...]
    forbidden_suffixes: tuple[str, ...]
    resource_estimator: str
    equivalence_class: str
    legacy_pack_ids: tuple[str, ...]
    digest: str


def parse_foundation_manifest(payload: dict[str, Any], digest: str) -> FoundationManifest:
    expected = {
        "kind",
        "schemaVersion",
        "foundationId",
        "version",
        "publisher",
        "abi",
        "minimumSafeVersion",
        "dependencyClosure",
        "platforms",
        "backends",
        "formats",
        "dtypes",
        "quantizations",
        "workerAbi",
        "policy",
        "license",
    }
    if set(payload) != expected or payload.get("kind") != "runtime-foundation":
        raise ManifestError("FOUNDATION_MANIFEST_INVALID", "Foundation manifest is invalid.")
    if payload.get("schemaVersion") != 1:
        raise ManifestError("FOUNDATION_MANIFEST_INVALID", "Foundation schema version is invalid.")
    policy = _record(payload.get("policy"), "policy")
    if set(policy) != {"network", "filesystemRoots"} or policy.get("network") != "deny":
        raise ManifestError("FOUNDATION_POLICY_INVALID", "Foundation policy is invalid.")
    roots = _string_tuple(policy.get("filesystemRoots"), "filesystem roots", 8)
    if not set(roots).issubset({"model", "artifact-input", "artifact-output", "scratch"}):
        raise ManifestError("FOUNDATION_POLICY_INVALID", "Foundation roots are invalid.")
    license_value = _record(payload.get("license"), "license")
    if set(license_value) != {"summary", "notice"}:
        raise ManifestError("FOUNDATION_MANIFEST_INVALID", "Foundation license is invalid.")
    return FoundationManifest(
        _identifier(payload.get("foundationId"), "foundation ID"),
        _version(payload.get("version")),
        _identifier(payload.get("publisher"), "publisher"),
        _string(payload.get("abi"), "ABI", 32),
        _version(payload.get("minimumSafeVersion")),
        _string(payload.get("dependencyClosure"), "dependency closure", 128),
        _string_tuple(payload.get("platforms"), "platforms", 16),
        _string_tuple(payload.get("backends"), "backends", 16),
        _string_tuple(payload.get("formats"), "formats", 32),
        _string_tuple(payload.get("dtypes"), "dtypes", 16),
        _string_tuple(payload.get("quantizations"), "quantizations", 32, allow_empty=True),
        _string(payload.get("workerAbi"), "worker ABI", 32),
        "deny",
        roots,
        _string(license_value.get("summary"), "license summary", 256),
        _string(license_value.get("notice"), "license notice", 1024),
        digest,
    )


def parse_adapter_manifest(payload: dict[str, Any], digest: str) -> AdapterManifest:
    expected = {
        "kind",
        "schemaVersion",
        "adapterId",
        "version",
        "publisher",
        "foundation",
        "architectures",
        "formats",
        "taskContracts",
        "backends",
        "dtypes",
        "quantizations",
        "requiredFiles",
        "forbiddenSuffixes",
        "resourceEstimator",
        "equivalenceClass",
        "legacyPackIds",
    }
    if set(payload) != expected or payload.get("kind") != "runtime-adapter":
        raise ManifestError("ADAPTER_MANIFEST_INVALID", "Adapter manifest is invalid.")
    if payload.get("schemaVersion") != 1:
        raise ManifestError("ADAPTER_MANIFEST_INVALID", "Adapter schema version is invalid.")
    foundation = _record(payload.get("foundation"), "foundation")
    if set(foundation) != {"id", "abi", "minimumVersion"}:
        raise ManifestError("ADAPTER_MANIFEST_INVALID", "Adapter foundation is invalid.")
    forbidden = _string_tuple(
        payload.get("forbiddenSuffixes"), "forbidden suffixes", 64, allow_empty=True
    )
    if any(not item.startswith(".") or "/" in item or "\\" in item for item in forbidden):
        raise ManifestError("ADAPTER_MANIFEST_INVALID", "Forbidden suffix is invalid.")
    return AdapterManifest(
        _identifier(payload.get("adapterId"), "adapter ID"),
        _version(payload.get("version")),
        _identifier(payload.get("publisher"), "publisher"),
        _identifier(foundation.get("id"), "foundation ID"),
        _string(foundation.get("abi"), "foundation ABI", 32),
        _version(foundation.get("minimumVersion")),
        _string_tuple(payload.get("architectures"), "architectures", 128, allow_empty=True),
        _string_tuple(payload.get("formats"), "formats", 32),
        _string_tuple(payload.get("taskContracts"), "task contracts", MAX_TASK_CONTRACTS),
        _string_tuple(payload.get("backends"), "backends", 16),
        _string_tuple(payload.get("dtypes"), "dtypes", 16),
        _string_tuple(payload.get("quantizations"), "quantizations", 32, allow_empty=True),
        _string_tuple(payload.get("requiredFiles"), "required files", 256),
        forbidden,
        _identifier(payload.get("resourceEstimator"), "resource estimator"),
        _identifier(payload.get("equivalenceClass"), "equivalence class"),
        _string_tuple(payload.get("legacyPackIds"), "legacy pack IDs", 16, allow_empty=True),
        digest,
    )


class FoundationRegistry:
    def __init__(self, verifier: SignedManifestVerifier, *, worker_abi: str = "1") -> None:
        self._verifier = verifier
        self._worker_abi = worker_abi
        self._foundations: dict[tuple[str, str], FoundationManifest] = {}
        self._adapters: dict[tuple[str, str], AdapterManifest] = {}
        self._active_foundations: dict[str, str] = {}
        self._active_adapters: dict[str, str] = {}
        self._revoked_foundations: set[tuple[str, str]] = set()
        self._revoked_adapters: set[tuple[str, str]] = set()

    @property
    def foundations(self) -> tuple[FoundationManifest, ...]:
        return tuple(self._foundations[key] for key in sorted(self._foundations))

    @property
    def adapters(self) -> tuple[AdapterManifest, ...]:
        return tuple(self._adapters[key] for key in sorted(self._adapters))

    def register_foundation(self, envelope: object) -> FoundationManifest:
        payload, digest = self._verifier.verify(envelope)
        foundation = parse_foundation_manifest(payload, digest)
        if foundation.worker_abi != self._worker_abi:
            raise ManifestError("FOUNDATION_ABI_INCOMPATIBLE", "Foundation ABI is incompatible.")
        if _semver_tuple(foundation.version) < _semver_tuple(foundation.minimum_safe_version):
            raise ManifestError("FOUNDATION_VERSION_UNSAFE", "Foundation version is unsafe.")
        key = (foundation.foundation_id, foundation.version)
        current = self._foundations.get(key)
        if current is not None and current.digest != foundation.digest:
            raise ManifestError("FOUNDATION_IDENTITY_CONFLICT", "Foundation identity conflicts.")
        self._foundations[key] = foundation
        self._active_foundations[foundation.foundation_id] = foundation.version
        return foundation

    def register_adapter(self, envelope: object) -> AdapterManifest:
        payload, digest = self._verifier.verify(envelope)
        adapter = parse_adapter_manifest(payload, digest)
        foundation = self.active_foundation(adapter.foundation_id)
        self._validate_adapter_foundation(adapter, foundation)
        key = (adapter.adapter_id, adapter.version)
        current = self._adapters.get(key)
        if current is not None and current.digest != adapter.digest:
            raise ManifestError("ADAPTER_IDENTITY_CONFLICT", "Adapter identity conflicts.")
        self._adapters[key] = adapter
        self._active_adapters[adapter.adapter_id] = adapter.version
        return adapter

    def active_foundation(self, foundation_id: str) -> FoundationManifest:
        version = self._active_foundations.get(foundation_id)
        if version is None:
            raise ManifestError("FOUNDATION_NOT_INSTALLED", "Foundation is not active.")
        return self._foundations[(foundation_id, version)]

    def revoke_foundation(self, foundation_id: str, version: str) -> None:
        key = (foundation_id, version)
        if key not in self._foundations:
            raise ManifestError("FOUNDATION_NOT_INSTALLED", "Foundation is not installed.")
        self._revoked_foundations.add(key)
        if self._active_foundations.get(foundation_id) == version:
            self._active_foundations.pop(foundation_id, None)
        for adapter in self.adapters:
            if adapter.foundation_id == foundation_id:
                self._active_adapters.pop(adapter.adapter_id, None)

    def activate_foundation(self, foundation_id: str, version: str) -> FoundationManifest:
        key = (foundation_id, version)
        foundation = self._foundations.get(key)
        if foundation is None:
            raise ManifestError("FOUNDATION_NOT_INSTALLED", "Foundation is not installed.")
        if key in self._revoked_foundations:
            raise ManifestError("FOUNDATION_REVOKED", "Foundation version is revoked.")
        if _semver_tuple(foundation.version) < _semver_tuple(foundation.minimum_safe_version):
            raise ManifestError("FOUNDATION_VERSION_UNSAFE", "Foundation version is unsafe.")
        self._active_foundations[foundation_id] = version
        return foundation

    def revoke_adapter(self, adapter_id: str, version: str) -> None:
        key = (adapter_id, version)
        if key not in self._adapters:
            raise ManifestError("ADAPTER_NOT_INSTALLED", "Adapter is not installed.")
        self._revoked_adapters.add(key)
        if self._active_adapters.get(adapter_id) == version:
            self._active_adapters.pop(adapter_id, None)

    def activate_adapter(self, adapter_id: str, version: str) -> AdapterManifest:
        key = (adapter_id, version)
        adapter = self._adapters.get(key)
        if adapter is None:
            raise ManifestError("ADAPTER_NOT_INSTALLED", "Adapter is not installed.")
        if key in self._revoked_adapters:
            raise ManifestError("ADAPTER_REVOKED", "Adapter version is revoked.")
        foundation = self.active_foundation(adapter.foundation_id)
        self._validate_adapter_foundation(adapter, foundation)
        self._active_adapters[adapter_id] = version
        return adapter

    def adapter_for_legacy_pack(self, pack_id: str) -> AdapterManifest | None:
        matches = [
            adapter
            for adapter in self.adapters
            if self._active_adapters.get(adapter.adapter_id) == adapter.version
            and pack_id in adapter.legacy_pack_ids
        ]
        if len(matches) > 1:
            raise ManifestError("ADAPTER_IDENTITY_CONFLICT", "Legacy pack mapping is ambiguous.")
        return matches[0] if matches else None

    @staticmethod
    def _validate_adapter_foundation(
        adapter: AdapterManifest, foundation: FoundationManifest
    ) -> None:
        if foundation.abi != adapter.foundation_abi:
            raise ManifestError("ADAPTER_ABI_INCOMPATIBLE", "Adapter ABI is incompatible.")
        if _semver_tuple(foundation.version) < _semver_tuple(adapter.minimum_foundation_version):
            raise ManifestError("ADAPTER_FOUNDATION_UNSAFE", "Foundation version is too old.")
        for values, supported, code, message in (
            (adapter.backends, foundation.backends, "ADAPTER_BACKEND_INCOMPATIBLE", "backend"),
            (adapter.formats, foundation.formats, "ADAPTER_FORMAT_INCOMPATIBLE", "format"),
            (adapter.dtypes, foundation.dtypes, "ADAPTER_DTYPE_INCOMPATIBLE", "dtype"),
            (
                adapter.quantizations,
                foundation.quantizations,
                "ADAPTER_QUANTIZATION_INCOMPATIBLE",
                "quantization",
            ),
        ):
            if not set(values).issubset(supported):
                raise ManifestError(code, f"Adapter {message} is invalid.")


def _version(value: object) -> str:
    version = _string(value, "version", 64)
    _semver_tuple(version)
    return version


def _semver_tuple(value: str) -> tuple[int, int, int]:
    parts = value.split("-", 1)[0].split(".")
    if len(parts) != 3 or any(not item.isdigit() for item in parts):
        raise ManifestError("FOUNDATION_MANIFEST_INVALID", "Version is invalid.")
    return tuple(int(item) for item in parts)  # type: ignore[return-value]
