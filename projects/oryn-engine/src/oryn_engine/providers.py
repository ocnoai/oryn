"""Backend-neutral provider ABI, discovery and admission.

Provider implementations and capability catalogs live in signed runtime packs. This
module never imports a backend library or interprets a model/quantization name.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from oryn_engine.jobs import JobExecutionError
from oryn_engine.runtime_packs.execution import StreamingPackProcess
from oryn_engine.runtime_routing import canonical_digest

PROVIDER_COMMANDS = frozenset({"describe", "inspect", "prepare"})
MAX_CATALOG_BYTES = 4 * 1024**2
MAX_METADATA_FILES = 24


def provider_variant(value: object) -> bool:
    return isinstance(value, str) and value.startswith("oryn.provider.")


def safe_relative(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 1024
        and "\\" not in value
        and not PurePosixPath(value).is_absolute()
        and all(part not in {"", ".", ".."} for part in value.split("/"))
    )


def verify_catalog(pack, runtime) -> dict:
    declaration = pack.provider
    if declaration is None or runtime.site_packages is None:
        raise ValueError("Provider declaration is unavailable")
    root = runtime.site_packages.resolve()
    path = root / declaration["catalog"]
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Provider catalog escapes its pack")
    if not path.is_file() or path.stat().st_size > MAX_CATALOG_BYTES:
        raise ValueError("Provider catalog is missing or oversized")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != declaration["catalogSha256"]:
        raise ValueError("Provider catalog differs from its signed build")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schemaVersion") != 1:
        raise ValueError("Provider catalog schema is invalid")
    code_files = value.get("codeFiles")
    if not isinstance(code_files, dict) or not 1 <= len(code_files) <= 1000:
        raise ValueError("Provider code identity is unavailable")
    total = 0
    for name, digest in code_files.items():
        if not safe_relative(name) or not name.endswith(".py"):
            raise ValueError("Provider code path is invalid")
        source = root / name
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError("Provider code escaped its pack")
        total += source.stat().st_size
        if total > 16 * 1024**2 or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError("Provider code differs from its signed catalog")
    return value


@dataclass(frozen=True)
class ProviderResult:
    pack: Any
    runtime: Any
    candidate: dict[str, Any]


class ProviderRegistry:
    def __init__(self, packs, locator, process=None):
        self.packs = packs
        self.locator = locator
        self.process = process or StreamingPackProcess()
        self.diagnostics: list[str] = []

    def inspect(self, inventory: dict, hub, *, access: bool, pack_ids=None) -> list[ProviderResult]:
        results = []
        for declared in self.packs.packs:
            if declared.provider is None or (
                pack_ids is not None and declared.pack_id not in pack_ids
            ):
                continue
            pack = self.packs.active_pack(declared.pack_id)
            if declared.version != pack.version:
                continue
            if inventory.get("task") not in pack.provider["tasks"]:
                continue
            runtime = self.locator.resolve(pack)
            if runtime is None:
                continue
            verify_catalog(pack, runtime)
            metadata: dict[str, dict] = {}
            # Three bounded metadata fetch rounds, then evaluate the final
            # snapshot. Config -> component index -> selected shard header can
            # consume all three; returning the pre-fetch result loses its offer.
            for _attempt in range(4):
                result = self.process.invoke(
                    pack,
                    runtime,
                    "inspect",
                    {"inventory": inventory, "metadata": metadata},
                    cancellation=threading.Event(),
                    event_sink=lambda *_: None,
                    timeout_seconds=60,
                )
                requests = result.get("metadataNeeded", [])
                if not requests or not access:
                    break
                if not isinstance(requests, list) or len(requests) > MAX_METADATA_FILES:
                    raise ValueError("Provider requested too much metadata")
                if _attempt == 3:
                    raise ValueError("Provider metadata expansion did not converge")
                files = {f["name"] for f in inventory["files"]}
                added_metadata = False
                for request in requests:
                    if not isinstance(request, dict) or set(request) != {"file", "kind"}:
                        raise ValueError("Provider metadata request is invalid")
                    name = request["file"]
                    if name not in files or not safe_relative(name):
                        raise ValueError("Provider requested an undeclared artifact")
                    if name in metadata:
                        continue
                    added_metadata = True
                    try:
                        if request["kind"] == "json" and name.endswith(".json"):
                            metadata[name] = hub.read_json(
                                inventory["repository"], inventory["revision"], name
                            )
                        elif request["kind"] == "safetensors-header" and name.endswith(
                            ".safetensors"
                        ):
                            metadata[name] = hub.read_tensor_metadata(
                                inventory["repository"], inventory["revision"], name
                            )
                        elif request["kind"] == "safetensors-structure" and name.endswith(
                            ".safetensors"
                        ):
                            metadata[name] = hub.read_tensor_structure(
                                inventory["repository"], inventory["revision"], name
                            )
                        else:
                            raise ValueError("Provider metadata format is invalid")
                    except (AttributeError, OSError):
                        metadata[name] = {"unavailable": True}
                if not added_metadata:
                    raise ValueError("Provider metadata expansion made no progress")
            diagnostics = result.get("diagnostics", [])
            if (
                not isinstance(diagnostics, list)
                or len(diagnostics) > 32
                or any(
                    not isinstance(d, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", d)
                    for d in diagnostics
                )
            ):
                raise ValueError("Provider diagnostics are invalid")
            self.diagnostics.extend(diagnostics)
            candidates = result.get("candidates")
            if not isinstance(candidates, list) or len(candidates) > 32:
                raise ValueError("Provider inspection result is invalid")
            for candidate in candidates:
                validate_candidate(candidate, inventory, pack)
                results.append(ProviderResult(pack, runtime, candidate))
        return results


def validate_candidate(value: object, inventory: dict, pack) -> None:
    fields = {
        "id",
        "architecture",
        "task",
        "taskContract",
        "runnerId",
        "backend",
        "precision",
        "quantization",
        "selectedFiles",
        "components",
        "estimatedMemoryBytes",
        "inputModalities",
        "outputModalities",
        "resourceEstimate",
        "generationProfile",
        "capabilities",
        "diagnostics",
        "plan",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Provider candidate fields are invalid")
    for key in ("id", "architecture", "task", "taskContract", "runnerId", "backend", "precision"):
        if not isinstance(value[key], str) or not 1 <= len(value[key]) <= 128:
            raise ValueError("Provider candidate identity is invalid")
    if not re.fullmatch(r"[a-z][a-z0-9.-]{0,127}", value["id"]):
        raise ValueError("Provider variant is invalid")
    runner = pack.runner(value["runnerId"])
    if (
        runner is None
        or value["taskContract"] not in runner.task_contracts
        or value["backend"] not in runner.backends
    ):
        raise ValueError("Provider task exceeds its signed declaration")
    if value["task"] != inventory["task"] or value["task"] not in pack.provider["tasks"]:
        raise ValueError("Provider changed the requested task")
    for key in ("inputModalities", "outputModalities"):
        if (
            not isinstance(value[key], list)
            or not value[key]
            or any(m not in {"text", "image", "audio", "video"} for m in value[key])
        ):
            raise ValueError("Provider modalities are invalid")
    if value["quantization"] is not None and (
        not isinstance(value["quantization"], str) or len(value["quantization"]) > 128
    ):
        raise ValueError("Provider quantization is invalid")
    files = {f["name"] for f in inventory["files"]}
    selected = value["selectedFiles"]
    if (
        not isinstance(selected, list)
        or len(set(selected)) != len(selected)
        or not set(selected) <= files
    ):
        raise ValueError("Provider selected undeclared files")
    if any(not safe_relative(f) for f in selected):
        raise ValueError("Provider selected an invalid path")
    estimate = value["estimatedMemoryBytes"]
    if type(estimate) is not int or not 0 <= estimate <= 2**63 - 1:
        raise ValueError("Provider estimate is invalid")
    diagnostics = value["diagnostics"]
    if (
        not isinstance(diagnostics, list)
        or len(diagnostics) > 32
        or any(
            not isinstance(d, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", d)
            for d in diagnostics
        )
    ):
        raise ValueError("Provider diagnostics are invalid")
    plan = value["plan"]
    if not isinstance(plan, dict) or plan.get("catalogDigest") != pack.provider["catalogSha256"]:
        raise ValueError("Provider plan is from a different build")
    if len(json.dumps(value, allow_nan=False)) > 256 * 1024:
        raise ValueError("Provider candidate exceeds its bound")
    from oryn_protocol import ProtocolSchemaStore

    if not ProtocolSchemaStore().is_valid("modelGenerationProfile", value["generationProfile"]):
        raise ValueError("Provider generation profile is invalid")


def bind_plan(result: ProviderResult, inventory: dict) -> dict:
    selected = set(result.candidate["selectedFiles"])
    artifacts = [f for f in inventory["files"] if f["name"] in selected]
    plan = {
        "schemaVersion": 1,
        "packId": result.pack.pack_id,
        "packVersion": result.pack.version,
        "packDigest": result.pack.digest,
        "catalogDigest": result.pack.provider["catalogSha256"],
        "model": {
            "provider": "huggingface",
            "repository": inventory["repository"],
            "revision": inventory["revision"],
        },
        "files": artifacts,
        "taskContract": result.candidate["taskContract"],
        "runnerId": result.candidate["runnerId"],
        "variantId": result.candidate["id"],
        "generationProfile": result.candidate["generationProfile"],
        "providerPlan": result.candidate["plan"],
    }
    return {**plan, "digest": canonical_digest(plan)}


def verify_plan(plan: object, pack, runtime, record: dict) -> dict:
    if not isinstance(plan, dict) or plan.get("schemaVersion") != 1:
        raise JobExecutionError("RUNTIME_PLAN_INVALID", "Provider plan is invalid.")
    expected = canonical_digest({k: v for k, v in plan.items() if k != "digest"})
    if (
        plan.get("digest") != expected
        or plan.get("packId") != pack.pack_id
        or plan.get("packVersion") != pack.version
        or plan.get("packDigest") != pack.digest
        or pack.provider is None
        or plan.get("catalogDigest") != pack.provider["catalogSha256"]
        or plan.get("model") != record.get("model")
    ):
        raise JobExecutionError("RUNTIME_PLAN_INVALID", "Provider plan or runtime changed.")
    if (
        not isinstance(plan.get("files"), list)
        or not plan["files"]
        or len(plan["files"]) > 10000
        or not isinstance(plan.get("providerPlan"), dict)
    ):
        raise JobExecutionError("RUNTIME_PLAN_INVALID", "Provider artifact plan is invalid.")
    actual = {f["name"]: (f["size"], f["blob_id"]) for f in record.get("files", [])}
    if any(actual.get(f["name"]) != (f["sizeBytes"], f["blobId"]) for f in plan["files"]):
        raise JobExecutionError("RUNTIME_PLAN_INVALID", "Provider artifact selection changed.")
    verify_catalog(pack, runtime)
    return plan
