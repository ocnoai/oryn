from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

IDENTIFIER = re.compile(r"^[a-z][a-z0-9.-]{0,127}$")
SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[a-z0-9.-]+)?$")
REVISION = re.compile(r"^[0-9a-f]{40,64}$")
RSA_SHA256_DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")
MAX_MANIFEST_BYTES = 256 * 1024
MAX_TASK_CONTRACTS = 64


class ManifestError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class TrustedPublisherKey:
    key_id: str
    modulus: int
    exponent: int = 65_537


@dataclass(frozen=True)
class RunnerDeclaration:
    runner_id: str
    task_contracts: tuple[str, ...]
    backends: tuple[str, ...]


@dataclass(frozen=True)
class PackManifest:
    pack_id: str
    version: str
    publisher: str
    abi: str
    entrypoint: str
    dependency_closure: str
    runners: tuple[RunnerDeclaration, ...]
    network_policy: str
    filesystem_roots: tuple[str, ...]
    digest: str
    provider: dict[str, Any] | None = None

    def runner(self, runner_id: str) -> RunnerDeclaration | None:
        return next((item for item in self.runners if item.runner_id == runner_id), None)


@dataclass(frozen=True)
class CompatibilityManifest:
    manifest_id: str
    pack_id: str
    pack_version: str
    runner_id: str
    task_contract: str
    level: str
    repository: str | None
    revision: str | None
    architectures: tuple[str, ...]
    tasks: tuple[str, ...]
    required_files: tuple[str, ...]
    capabilities: dict[str, Any]
    resources: dict[str, int]
    trusted_metadata: dict[str, Any]
    digest: str

    def matches(
        self, repository: str, revision: str, architectures: set[str], task: str | None
    ) -> bool:
        if self.repository is not None:
            return self.repository == repository and self.revision == revision
        architecture_match = bool(architectures.intersection(self.architectures))
        task_match = task is not None and task in self.tasks
        return architecture_match or task_match


class SignedManifestVerifier:
    """Verifies bounded RSA-SHA256 envelopes using pinned public publisher keys.

    Verification is deliberately small and dependency-free so it can run before a pack's
    dependency closure is imported. Private signing material is never present in Oryn.
    """

    def __init__(self, keys: dict[str, TrustedPublisherKey]) -> None:
        self._keys = dict(keys)

    def verify(self, envelope: object) -> tuple[dict[str, Any], str]:
        if not isinstance(envelope, dict) or set(envelope) != {"payload", "integrity"}:
            raise ManifestError("PACK_MANIFEST_INVALID", "Signed manifest envelope is invalid.")
        payload = envelope.get("payload")
        integrity = envelope.get("integrity")
        if not isinstance(payload, dict) or not isinstance(integrity, dict):
            raise ManifestError("PACK_MANIFEST_INVALID", "Signed manifest envelope is invalid.")
        if set(integrity) != {"algorithm", "keyId", "digest", "signature"}:
            raise ManifestError("PACK_MANIFEST_INVALID", "Manifest integrity metadata is invalid.")
        if integrity.get("algorithm") != "rsa-sha256":
            raise ManifestError(
                "PACK_SIGNATURE_INVALID", "Manifest signature algorithm is invalid."
            )
        key_id = integrity.get("keyId")
        key = self._keys.get(key_id) if isinstance(key_id, str) else None
        if key is None:
            raise ManifestError(
                "PACK_PUBLISHER_UNTRUSTED", "Manifest publisher key is not trusted."
            )
        encoded = canonical_json(payload)
        if len(encoded) > MAX_MANIFEST_BYTES:
            raise ManifestError("PACK_MANIFEST_INVALID", "Manifest exceeds the size limit.")
        digest = hashlib.sha256(encoded).hexdigest()
        claimed_digest = integrity.get("digest")
        if not isinstance(claimed_digest, str) or not hmac.compare_digest(digest, claimed_digest):
            raise ManifestError("PACK_DIGEST_INVALID", "Manifest content digest does not match.")
        signature_value = integrity.get("signature")
        if not isinstance(signature_value, str) or len(signature_value) > 2048:
            raise ManifestError("PACK_SIGNATURE_INVALID", "Manifest signature is invalid.")
        try:
            signature = base64.b64decode(signature_value, validate=True)
        except ValueError as exc:
            raise ManifestError("PACK_SIGNATURE_INVALID", "Manifest signature is invalid.") from exc
        if not _verify_rsa_sha256(key, bytes.fromhex(digest), signature):
            raise ManifestError("PACK_SIGNATURE_INVALID", "Manifest signature is invalid.")
        return payload, digest


def parse_pack_manifest(payload: dict[str, Any], digest: str) -> PackManifest:
    _exact_keys(
        {key: value for key, value in payload.items() if key != "provider"},
        {
            "kind",
            "schemaVersion",
            "packId",
            "version",
            "publisher",
            "abi",
            "runtime",
            "dependencyClosure",
            "runners",
            "policy",
        },
    )
    if payload.get("kind") != "runtime-pack" or payload.get("schemaVersion") != 1:
        raise ManifestError("PACK_MANIFEST_INVALID", "Runtime-pack schema version is invalid.")
    pack_id = _identifier(payload.get("packId"), "pack ID")
    version = _string(payload.get("version"), "version", 64)
    if not SEMVER.fullmatch(version):
        raise ManifestError("PACK_MANIFEST_INVALID", "Pack version is invalid.")
    publisher = _identifier(payload.get("publisher"), "publisher")
    abi = _string(payload.get("abi"), "ABI", 32)
    runtime = _record(payload.get("runtime"), "runtime")
    _exact_keys(runtime, {"kind", "entrypoint"})
    if runtime.get("kind") != "python":
        raise ManifestError("PACK_MANIFEST_INVALID", "Pack runtime kind is unsupported.")
    entrypoint = _string(runtime.get("entrypoint"), "entrypoint", 256)
    provider = payload.get("provider")
    if provider is not None:
        provider = _record(provider, "provider")
        _exact_keys(provider, {"abi", "catalog", "catalogSha256", "tasks", "libraries"})
        if provider["abi"] != "1":
            raise ManifestError("PACK_ABI_INCOMPATIBLE", "Provider ABI is incompatible.")
        catalog = _string(provider["catalog"], "provider catalog", 256)
        if (
            not re.fullmatch(r"[a-z][a-z0-9_]*/catalog\.json", catalog)
            or not isinstance(provider["catalogSha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", provider["catalogSha256"])
        ):
            raise ManifestError("PACK_MANIFEST_INVALID", "Provider catalog identity is invalid.")
        _string_tuple(provider["tasks"], "provider tasks", 32)
        _string_tuple(provider["libraries"], "provider libraries", 32)
    external_provider = provider is not None and re.fullmatch(
        r"oryn_[a-z][a-z0-9_]*_provider\.worker", entrypoint
    )
    if (
        not entrypoint.startswith("oryn_engine.runtime_packs.") and not external_provider
    ) or ":" in entrypoint:
        raise ManifestError("PACK_ENTRYPOINT_FORBIDDEN", "Pack entrypoint is not built into Oryn.")
    dependency_closure = _string(payload.get("dependencyClosure"), "dependency closure", 128)
    raw_runners = payload.get("runners")
    if not isinstance(raw_runners, list) or not 1 <= len(raw_runners) <= 64:
        raise ManifestError("PACK_MANIFEST_INVALID", "Pack runners are invalid.")
    runners: list[RunnerDeclaration] = []
    seen: set[str] = set()
    for raw in raw_runners:
        value = _record(raw, "runner")
        _exact_keys(value, {"runnerId", "taskContracts", "backends"})
        runner_id = _identifier(value.get("runnerId"), "runner ID")
        if runner_id in seen:
            raise ManifestError("PACK_MANIFEST_INVALID", "Pack runner IDs must be unique.")
        seen.add(runner_id)
        runners.append(
            RunnerDeclaration(
                runner_id,
                _string_tuple(value.get("taskContracts"), "task contracts", MAX_TASK_CONTRACTS),
                _string_tuple(value.get("backends"), "backends", 16),
            )
        )
    policy = _record(payload.get("policy"), "policy")
    _exact_keys(policy, {"network", "filesystemRoots"})
    if policy.get("network") != "deny":
        raise ManifestError("PACK_POLICY_INVALID", "Inference-time networking must be denied.")
    filesystem_roots = _string_tuple(policy.get("filesystemRoots"), "filesystem roots", 8)
    if not set(filesystem_roots).issubset(
        {"model", "artifact-input", "artifact-output", "scratch"}
    ):
        raise ManifestError("PACK_POLICY_INVALID", "Pack filesystem policy is invalid.")
    return PackManifest(
        pack_id,
        version,
        publisher,
        abi,
        entrypoint,
        dependency_closure,
        tuple(runners),
        "deny",
        filesystem_roots,
        digest,
        provider,
    )


def parse_compatibility_manifest(payload: dict[str, Any], digest: str) -> CompatibilityManifest:
    required_fields = {
        "kind",
        "schemaVersion",
        "manifestId",
        "runtime",
        "model",
        "match",
        "taskContract",
        "level",
        "requiredFiles",
        "capabilities",
        "resources",
    }
    if not required_fields.issubset(payload) or not set(payload).issubset(
        required_fields | {"metadata"}
    ):
        raise ManifestError("PACK_MANIFEST_INVALID", "Manifest fields are invalid.")
    if payload.get("kind") != "model-compatibility" or payload.get("schemaVersion") != 1:
        raise ManifestError("PACK_MANIFEST_INVALID", "Compatibility schema version is invalid.")
    manifest_id = _identifier(payload.get("manifestId"), "manifest ID")
    runtime = _record(payload.get("runtime"), "runtime")
    _exact_keys(runtime, {"packId", "packVersion", "runnerId"})
    pack_id = _identifier(runtime.get("packId"), "pack ID")
    pack_version = _string(runtime.get("packVersion"), "pack version", 64)
    runner_id = _identifier(runtime.get("runnerId"), "runner ID")
    model = payload.get("model")
    repository: str | None = None
    revision: str | None = None
    if model is not None:
        model_value = _record(model, "model")
        _exact_keys(model_value, {"provider", "repository", "revision"})
        if model_value.get("provider") != "huggingface":
            raise ManifestError("PACK_MANIFEST_INVALID", "Model provider is invalid.")
        repository = _repository(model_value.get("repository"))
        revision = _string(model_value.get("revision"), "revision", 64)
        if not REVISION.fullmatch(revision):
            raise ManifestError("PACK_MANIFEST_INVALID", "Model revision must be immutable.")
    match = _record(payload.get("match"), "match")
    _exact_keys(match, {"architectures", "tasks"})
    architectures = _string_tuple(match.get("architectures"), "architectures", 64, allow_empty=True)
    tasks = _string_tuple(match.get("tasks"), "tasks", 32, allow_empty=True)
    if repository is None and not architectures and not tasks:
        raise ManifestError("PACK_MANIFEST_INVALID", "Compatibility match is empty.")
    task_contract = _identifier(payload.get("taskContract"), "task contract")
    level = payload.get("level")
    if level not in {"native-declarative", "audited-adapter", "converted-artifact"}:
        raise ManifestError("PACK_MANIFEST_INVALID", "Compatibility level is invalid.")
    required_files = _string_tuple(payload.get("requiredFiles"), "required files", 256)
    for pattern in required_files:
        path = PurePosixPath(pattern)
        if path.is_absolute() or ".." in path.parts or "\\" in pattern:
            raise ManifestError("PACK_MANIFEST_INVALID", "Required model file path is unsafe.")
    capabilities = _record(payload.get("capabilities"), "capabilities")
    if len(canonical_json(capabilities)) > 64 * 1024:
        raise ManifestError("PACK_MANIFEST_INVALID", "Capabilities exceed the size limit.")
    resources_raw = _record(payload.get("resources"), "resources")
    _exact_keys(resources_raw, {"estimatedResidentBytes", "workingMemoryBytes"})
    resources = {
        name: _non_negative_integer(resources_raw.get(name), name)
        for name in ("estimatedResidentBytes", "workingMemoryBytes")
    }
    metadata_raw = payload.get("metadata", {})
    metadata = _record(metadata_raw, "metadata")
    if len(canonical_json(metadata)) > 64 * 1024:
        raise ManifestError("PACK_MANIFEST_INVALID", "Trusted metadata exceeds the size limit.")
    _validate_trusted_metadata(metadata)
    return CompatibilityManifest(
        manifest_id,
        pack_id,
        pack_version,
        runner_id,
        task_contract,
        str(level),
        repository,
        revision,
        architectures,
        tasks,
        required_files,
        dict(capabilities),
        resources,
        dict(metadata),
        digest,
    )


def _validate_trusted_metadata(value: dict[str, Any]) -> None:
    allowed = {
        "license",
        "source",
        "languages",
        "voices",
        "conformance",
        "quantization",
        "contextLength",
        "chatTemplate",
        "modelFile",
        "adapterArchitecture",
        "inferenceDefaults",
        "generationProfiles",
    }
    if not set(value).issubset(allowed) or (
        value and not {"license", "conformance"}.issubset(value)
    ):
        raise ManifestError("PACK_MANIFEST_INVALID", "Trusted metadata fields are invalid.")
    if "license" in value:
        _string(value["license"], "license", 128)
    if "quantization" in value:
        quantization = _string(value["quantization"], "quantization", 32)
        if not re.fullmatch(r"[A-Z0-9_]{2,32}", quantization):
            raise ManifestError("PACK_MANIFEST_INVALID", "Trusted quantization is invalid.")
    if "contextLength" in value:
        context_length = _non_negative_integer(value["contextLength"], "context length")
        if not 128 <= context_length <= 4_194_304:
            raise ManifestError("PACK_MANIFEST_INVALID", "Trusted context length is invalid.")
    if "chatTemplate" in value:
        from oryn_engine.chat_formats import names

        if value["chatTemplate"] not in names():
            raise ManifestError("PACK_MANIFEST_INVALID", "Trusted chat template is invalid.")
    if "modelFile" in value:
        from oryn_engine.runtime_packs.gguf import safe_member_name

        if not safe_member_name(_string(value["modelFile"], "model file", 256)):
            raise ManifestError("PACK_MANIFEST_INVALID", "Trusted model file is invalid.")
    if "adapterArchitecture" in value:
        adapter_architecture = _string(value["adapterArchitecture"], "adapter architecture", 128)
        if not re.fullmatch(r"[a-z][a-z0-9.-]{0,127}", adapter_architecture):
            raise ManifestError("PACK_MANIFEST_INVALID", "Trusted adapter architecture is invalid.")
    if "inferenceDefaults" in value:
        defaults = _record(value["inferenceDefaults"], "inference defaults")
        if not defaults or not set(defaults).issubset(
            {"width", "height", "steps", "guidanceScale"}
        ):
            raise ManifestError("PACK_MANIFEST_INVALID", "Inference defaults are invalid.")
        for name in ("width", "height", "steps"):
            if name in defaults and (
                not isinstance(defaults[name], int)
                or isinstance(defaults[name], bool)
                or not 1 <= defaults[name] <= 4096
            ):
                raise ManifestError("PACK_MANIFEST_INVALID", "Inference defaults are invalid.")
        guidance = defaults.get("guidanceScale")
        if guidance is not None and (
            not isinstance(guidance, (int, float))
            or isinstance(guidance, bool)
            or not 0 <= float(guidance) <= 30
        ):
            raise ManifestError("PACK_MANIFEST_INVALID", "Inference defaults are invalid.")
    if "generationProfiles" in value:
        _validate_generation_profiles(value["generationProfiles"])
    if "languages" in value:
        _string_tuple(value["languages"], "languages", 32, allow_empty=True)
    if "voices" in value:
        voices = value["voices"]
        if not isinstance(voices, list) or len(voices) > 128:
            raise ManifestError("PACK_MANIFEST_INVALID", "Trusted voices metadata is invalid.")
        seen: set[str] = set()
        for raw_voice in voices:
            voice = _record(raw_voice, "voice")
            _exact_keys(voice, {"id", "language", "name"})
            voice_id = _string(voice.get("id"), "voice ID", 64)
            if not re.fullmatch(r"[a-z][a-z0-9_-]*", voice_id):
                raise ManifestError("PACK_MANIFEST_INVALID", "Trusted voice ID is invalid.")
            if voice_id in seen:
                raise ManifestError("PACK_MANIFEST_INVALID", "Trusted voice IDs must be unique.")
            seen.add(voice_id)
            _string(voice.get("language"), "voice language", 32)
            _string(voice.get("name"), "voice name", 128)
    if "source" in value:
        source = _record(value["source"], "source")
        _exact_keys(source, {"provider", "repository", "revision"})
        if source.get("provider") != "huggingface":
            raise ManifestError("PACK_MANIFEST_INVALID", "Trusted source provider is invalid.")
        _repository(source.get("repository"))
        revision = _string(source.get("revision"), "source revision", 64)
        if not REVISION.fullmatch(revision):
            raise ManifestError("PACK_MANIFEST_INVALID", "Trusted source revision is invalid.")
    if "conformance" in value:
        conformance = _record(value["conformance"], "conformance")
        _exact_keys(conformance, {"status", "platforms"})
        if conformance.get("status") not in {"passed", "fixture-only"}:
            raise ManifestError("PACK_MANIFEST_INVALID", "Conformance status is invalid.")
        _string_tuple(conformance.get("platforms"), "conformance platforms", 16)


def _validate_generation_profiles(raw: object) -> None:
    if not isinstance(raw, list) or not 1 <= len(raw) <= 64:
        raise ManifestError("PACK_MANIFEST_INVALID", "Generation profiles are invalid.")
    seen: set[str] = set()
    for raw_descriptor in raw:
        descriptor = _record(raw_descriptor, "generation profile descriptor")
        _exact_keys(descriptor, {"match", "profile"})
        match = _record(descriptor.get("match"), "generation profile match")
        if not 1 <= len(match) <= 16:
            raise ManifestError("PACK_MANIFEST_INVALID", "Generation profile match is invalid.")
        for name, expected in match.items():
            if (
                not isinstance(name, str)
                or not re.fullmatch(r"_?[A-Za-z][A-Za-z0-9_]{0,127}", name)
                or not isinstance(expected, (str, bool, int, float))
                or isinstance(expected, str)
                and not 1 <= len(expected) <= 256
            ):
                raise ManifestError("PACK_MANIFEST_INVALID", "Generation profile match is invalid.")
        profile = _record(descriptor.get("profile"), "generation profile")
        _exact_keys(profile, {"profileId", "recommended", "parameters"})
        profile_id = _string(profile.get("profileId"), "generation profile ID", 128)
        if not re.fullmatch(r"[a-z][a-z0-9.-]{0,127}", profile_id) or profile_id in seen:
            raise ManifestError("PACK_MANIFEST_INVALID", "Generation profile ID is invalid.")
        seen.add(profile_id)
        if not isinstance(profile.get("recommended"), bool):
            raise ManifestError("PACK_MANIFEST_INVALID", "Generation profile is invalid.")
        parameters = _record(profile.get("parameters"), "generation profile parameters")
        _exact_keys(parameters, {"width", "height", "steps", "guidanceScale"})
        for name, raw_parameter in parameters.items():
            parameter = _record(raw_parameter, "generation parameter")
            _exact_keys(
                parameter,
                {"default", "minimum", "maximum", "step", "supported", "editable"},
            )
            numeric = [
                parameter.get("default"),
                parameter.get("minimum"),
                parameter.get("maximum"),
                parameter.get("step"),
            ]
            integer_parameter = name != "guidanceScale"
            if (
                not all(
                    isinstance(item, int if integer_parameter else (int, float))
                    and not isinstance(item, bool)
                    for item in numeric
                )
                or not 0 <= float(numeric[1]) <= float(numeric[0]) <= float(numeric[2]) <= 4096
                or not 0 < float(numeric[3]) <= 4096
                or not isinstance(parameter.get("supported"), bool)
                or not isinstance(parameter.get("editable"), bool)
                or parameter.get("editable") is True
                and parameter.get("supported") is not True
            ):
                raise ManifestError("PACK_MANIFEST_INVALID", "Generation parameter is invalid.")


def canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    except (TypeError, ValueError, RecursionError) as exc:
        raise ManifestError("PACK_MANIFEST_INVALID", "Manifest is not canonical JSON.") from exc


def _verify_rsa_sha256(key: TrustedPublisherKey, digest: bytes, signature: bytes) -> bool:
    size = (key.modulus.bit_length() + 7) // 8
    if len(signature) != size:
        return False
    decoded = pow(int.from_bytes(signature), key.exponent, key.modulus).to_bytes(size, "big")
    expected_tail = RSA_SHA256_DIGEST_INFO + digest
    padding_size = size - len(expected_tail) - 3
    if padding_size < 8:
        return False
    expected = b"\x00\x01" + (b"\xff" * padding_size) + b"\x00" + expected_tail
    return hmac.compare_digest(decoded, expected)


def _record(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError("PACK_MANIFEST_INVALID", f"Manifest {label} is invalid.")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise ManifestError("PACK_MANIFEST_INVALID", "Manifest fields are invalid.")


def _string(value: object, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ManifestError("PACK_MANIFEST_INVALID", f"Manifest {label} is invalid.")
    return value


def _identifier(value: object, label: str) -> str:
    result = _string(value, label, 128)
    if not IDENTIFIER.fullmatch(result):
        raise ManifestError("PACK_MANIFEST_INVALID", f"Manifest {label} is invalid.")
    return result


def _repository(value: object) -> str:
    repository = _string(value, "repository", 193)
    parts = repository.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise ManifestError("PACK_MANIFEST_INVALID", "Manifest repository is invalid.")
    return repository


def _string_tuple(
    value: object,
    label: str,
    maximum: int,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum or (not value and not allow_empty):
        raise ManifestError("PACK_MANIFEST_INVALID", f"Manifest {label} is invalid.")
    result = tuple(_string(item, label, 256) for item in value)
    if len(set(result)) != len(result):
        raise ManifestError("PACK_MANIFEST_INVALID", f"Manifest {label} contains duplicates.")
    return result


def _non_negative_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 2**63 - 1:
        raise ManifestError("PACK_MANIFEST_INVALID", f"Manifest {label} is invalid.")
    return value
