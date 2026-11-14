"""Record/replay the production inspector's bounded, data-only Hub reads."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import threading
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import httpx
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers

from oryn_engine.audit_profiles import TARGETS
from oryn_engine.models import HuggingFaceClient, ModelStore, ModelStoreError
from oryn_engine.runtime_packs.builtin_foundations import builtin_foundation_registry
from oryn_engine.runtime_routing import canonical_digest


class HubApiGate:
    """Share the anonymous Hub API quota across all workers in this process."""

    def __init__(self):
        self.lock = threading.Lock()
        self.next_request = 0.0

    def wait(self):
        while True:
            with self.lock:
                delay = self.next_request - time.monotonic()
                if delay <= 0:
                    # Anonymous Hub API quota is 500 requests per five-minute window.
                    self.next_request = time.monotonic() + 0.7
                    return
            time.sleep(min(delay, 30))

    def limited(self, headers):
        retry = headers.get("retry-after", "")
        reset = re.search(r"(?:^|;)\s*t=(\d+)", headers.get("ratelimit", ""))
        seconds = int(retry) if retry.isdigit() else int(reset[1]) if reset else 60
        with self.lock:
            self.next_request = max(self.next_request, time.monotonic() + seconds + 1)


API_GATE = HubApiGate()


@lru_cache(maxsize=1)
def json_parser_digest():
    # A rejected document must be reconsidered when its parser changes. Do not
    # turn transient HTTP/access/budget failures into persistent incompatibility.
    import inspect

    from oryn_engine.models import _safe_metadata_name

    root = Path(__file__).parent
    return canonical_digest(
        [
            hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("model_metadata.py", "timesfm3_tasks.py")
        ]
        + [inspect.getsource(HuggingFaceClient.read_json), inspect.getsource(_safe_metadata_name)]
    )


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def error_code(error: Exception) -> str:
    if isinstance(error, OSError) and str(error) in {
        "SNAPSHOT_METADATA_MISSING",
        "METADATA_READ_BUDGET_EXCEEDED",
        "METADATA_FILE_NOT_IN_INVENTORY",
        "ONNX_METADATA_BUDGET_EXCEEDED",
    }:
        return str(error)
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status in {401, 403}:
        return "MODEL_ACCESS_RESTRICTED"
    if status == 404:
        return "MODEL_OR_REVISION_NOT_FOUND"
    if status == 429:
        return "METADATA_RATE_LIMITED"
    return getattr(error, "code", None) or (
        "METADATA_NETWORK_ERROR" if status or isinstance(error, OSError) else type(error).__name__
    )


class BoundedAuditClient(HuggingFaceClient):
    """Use the production graph parser with a per-model HTTP range budget."""

    # A twelve-layer embedded-weight encoder has more than 128 separated
    # metadata blocks. The parser still caps total fetched bytes at 16 MiB;
    # this request/time budget permits that graph without downloading weights.
    MAX_GRAPH_REQUESTS = 512
    GRAPH_SECONDS = 90

    def inventory_info(self, *args, **kwargs):
        for attempt in range(3):
            API_GATE.wait()
            try:
                return self._api.model_info(*args, **kwargs)
            except Exception as exc:
                response = getattr(exc, "response", None)
                if getattr(response, "status_code", None) != 429 or attempt == 2:
                    raise
                API_GATE.limited(response.headers)

    def read_onnx_metadata(self, repository, revision, filename, sizes):
        from oryn_engine.onnx_graph import safe_path, scan_graph

        if not re.fullmatch(r"[0-9a-f]{40,64}", revision) or not safe_path(filename):
            raise ValueError("ONNX_METADATA_IDENTITY_INVALID")
        if not hasattr(self, "graph_deadline"):
            self.graph_deadline = time.monotonic() + self.GRAPH_SECONDS
            self.graph_requests = 0
        url = hf_hub_url(repository, filename, revision=revision, endpoint="https://huggingface.co")

        def read_range(start, length):
            remaining = self.graph_deadline - time.monotonic()
            if remaining <= 0 or self.graph_requests >= self.MAX_GRAPH_REQUESTS:
                raise OSError("ONNX_METADATA_BUDGET_EXCEEDED")
            self.graph_requests += 1
            headers = build_hf_headers(token=self._token())
            headers["Range"] = f"bytes={start}-{start + length - 1}"
            with client.stream("GET", url, headers=headers, timeout=min(15, remaining)) as response:
                response.raise_for_status()
                expected = f"bytes {start}-{start + length - 1}/{sizes[filename]}"
                if response.status_code != 206 or response.headers.get("content-range") != expected:
                    raise ValueError("ONNX_RANGE_UNAVAILABLE")
                raw = bytearray()
                for chunk in response.iter_bytes(chunk_size=32768):
                    if time.monotonic() > self.graph_deadline or len(raw) + len(chunk) > length:
                        raise OSError("ONNX_METADATA_BUDGET_EXCEEDED")
                    raw.extend(chunk)
                return bytes(raw)

        with httpx.Client(follow_redirects=True) as client:
            return scan_graph(read_range, sizes[filename], filename, sizes)


class MetadataSnapshot:
    """Forbid downloads; expose only inventory, bounded JSON and structural reads.

    Snapshots freeze mutable refs until explicitly refreshed. Replaying a snapshot
    establishes metadata compatibility, never current access or executable support.
    """

    def __init__(
        self, path, repository, revision, *, offline=False, token=None, client=None, envelope=None
    ):
        self.path = path
        self.repository, self.requested_revision = repository, revision
        self.offline = offline
        self.client = client or BoundedAuditClient(token_provider=lambda: token)
        self._token_value = token
        self._dependencies = {}
        self._budget = self
        self.failures: dict[str, str] = {}
        self.provider_failures: dict[str, dict[str, str]] = {}
        self.current_provider = "common"
        self.reads = 0
        self.deadline = time.monotonic() + 120
        self.data = {
            "repository": repository,
            "requestedRevision": revision,
            "accessMode": "authenticated" if token else "anonymous",
            "metadata": {},
        }
        if envelope is None and path.exists():
            if path.stat().st_size > 128 * 1024**2:
                raise ValueError("SNAPSHOT_TOO_LARGE")
            envelope = json.loads(path.read_text())
        if envelope is not None:
            saved = envelope["payload"]
            if (
                envelope.get("schemaVersion") != 1
                or envelope.get("sha256") != canonical_digest(saved)
                or saved.get("repository") != repository
                or saved.get("requestedRevision") != revision
                or saved.get("accessMode") != self.data["accessMode"]
            ):
                raise ValueError("SNAPSHOT_IDENTITY_OR_DIGEST_MISMATCH")
            self.data = saved

    def dependency_snapshot(self, reference):
        """Only the engine's reviewed, exact base pin may extend this snapshot."""
        from oryn_engine.model_composition import binding

        value = binding(self.repository, self.data.get("revision"))
        if value is None or reference != value["base"] or self._budget is not self:
            raise ValueError("METADATA_DEPENDENCY_NOT_ADMITTED")
        key = canonical_digest(reference)
        if key not in self._dependencies:
            child = MetadataSnapshot(
                self.path.parent / ".embedded-dependencies" / self.path.stem / (key + ".json"),
                reference["repository"],
                reference["revision"],
                offline=self.offline,
                token=self._token_value,
                client=self.client,
                envelope=self.data.get("dependencies", {}).get(key),
            )
            child._budget = self

            def save_child():
                self.data.setdefault("dependencies", {})[key] = {
                    "schemaVersion": 1,
                    "sha256": child.digest,
                    "payload": child.data,
                }
                self.save()

            child.save = save_child
            self._dependencies[key] = child
        child = self._dependencies[key]
        child.current_provider = self.current_provider
        return child

    def save(self):
        save_json(self.path, {"schemaVersion": 1, "sha256": self.digest, "payload": self.data})

    @property
    def digest(self):
        return canonical_digest(self.data)

    def model_info(self, repo_id, revision, files_metadata=True):
        del files_metadata
        if repo_id != self.repository or revision not in {
            self.requested_revision,
            self.data.get("revision"),
        }:
            raise ValueError("METADATA_IDENTITY_CHANGED")
        if "inventory" not in self.data:
            if self.offline:
                raise ModelStoreError("SNAPSHOT_MISSING", "No frozen inventory is available.")
            fetch = getattr(self.client, "inventory_info", self.client._api.model_info)
            info = fetch(
                repo_id,
                revision=revision,
                files_metadata=True,
                token=self.client._token(),
                timeout=15,
            )
            if (
                not re.fullmatch(r"[0-9a-f]{40,64}", info.sha or "")
                or (re.fullmatch(r"[0-9a-f]{40,64}", revision) and revision != info.sha)
                or len(info.siblings or []) > 10000
            ):
                raise ValueError("INVALID_RESOLVED_INVENTORY")
            card = getattr(info, "card_data", None) or {}
            card = card.to_dict() if hasattr(card, "to_dict") else card
            files = []
            for item in info.siblings or []:
                lfs = getattr(item, "lfs", None)
                digest = (
                    lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
                )
                files.append(
                    {
                        "name": item.rfilename,
                        "sizeBytes": item.size,
                        "blobId": digest or getattr(item, "blob_id", None),
                    }
                )
            restricted = bool(getattr(info, "gated", False) or getattr(info, "private", False))
            self.data.update(
                revision=info.sha,
                collectedAt=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                accessGranted=not restricted or self.client.has_access(repo_id),
                inventory={
                    "config": getattr(info, "config", None) or {},
                    "task": getattr(info, "pipeline_tag", None),
                    "library": getattr(info, "library_name", None),
                    "card": {k: card[k] for k in ("base_model", "license") if k in card},
                    "gated": getattr(info, "gated", False),
                    "private": getattr(info, "private", False),
                    "gguf": getattr(info, "gguf", None),
                    "files": files,
                },
            )
            self.save()
            # Quota waiting is separate from the bounded model metadata read budget.
            self.deadline = time.monotonic() + 120
        inv = self.data["inventory"]
        names = {f["name"] for f in inv["files"]}
        from oryn_engine.model_metadata import root_config_name

        root = root_config_name(names)
        config = inv["config"]
        if root in names and self.data["accessGranted"]:
            config = self.read_json(repo_id, self.data["revision"], root)
        return SimpleNamespace(
            sha=self.data["revision"],
            config=config,
            pipeline_tag=inv["task"],
            library_name=inv["library"],
            card_data=inv["card"],
            gated=inv["gated"],
            private=inv["private"],
            gguf=inv["gguf"],
            siblings=[
                SimpleNamespace(
                    rfilename=f["name"],
                    size=f["sizeBytes"],
                    blob_id=f["blobId"],
                    lfs=SimpleNamespace(sha256=f["blobId"]),
                )
                for f in inv["files"]
            ],
        )

    def has_access(self, repo_id):
        return repo_id == self.repository and self.data.get("accessGranted", False)

    def _read(self, method, repository, revision, filename, *extra):
        if (repository, revision) != (self.repository, self.data.get("revision")):
            raise ValueError("METADATA_IDENTITY_CHANGED")
        if filename not in {f["name"] for f in self.data["inventory"]["files"]}:
            self.failures[filename] = "METADATA_FILE_NOT_IN_INVENTORY"
            self.provider_failures.setdefault(self.current_provider, {})[filename] = self.failures[
                filename
            ]
            raise OSError("METADATA_FILE_NOT_IN_INVENTORY")
        key = canonical_digest([method, filename, extra])
        if key in self.data["metadata"]:
            return self.data["metadata"][key]["value"]
        try:
            rejected = self.data.get("metadataRejections", {}).get(key)
            if rejected and rejected.get("parserDigest") == json_parser_digest():
                raise ModelStoreError(
                    "MODEL_METADATA_INVALID",
                    "Frozen JSON was rejected by this parser.",
                    details={"metadataRejection": rejected["evidence"]},
                )
            if self.offline:
                raise OSError("SNAPSHOT_METADATA_MISSING")
            if self._budget.reads >= 64 or time.monotonic() > self._budget.deadline:
                raise OSError("METADATA_READ_BUDGET_EXCEEDED")
            self._budget.reads += 1
            value = getattr(self.client, method)(repository, revision, filename, *extra)
            self.data["metadata"][key] = {"method": method, "file": filename, "value": value}
            self.data.get("metadataRejections", {}).pop(key, None)
            self.failures.pop(filename, None)
            self.provider_failures.get(self.current_provider, {}).pop(filename, None)
            self.save()
            return value
        except Exception as exc:
            evidence = (getattr(exc, "details", None) or {}).get("metadataRejection")
            if method == "read_json" and error_code(exc) == "MODEL_METADATA_INVALID" and evidence:
                self.data.setdefault("metadataRejections", {})[key] = {
                    "file": filename,
                    "method": method,
                    "parserDigest": json_parser_digest(),
                    "evidence": evidence,
                }
                self.save()
            self.failures[filename] = error_code(exc)
            self.provider_failures.setdefault(self.current_provider, {})[filename] = error_code(exc)
            raise

    def read_json(self, *args):
        return self._read("read_json", *args)

    def read_template(self, *args):
        return self._read("read_template", *args)

    def read_tensor_metadata(self, *args):
        return self._read("read_tensor_metadata", *args)

    def read_tensor_structure(self, *args):
        return self._read("read_tensor_structure", *args)

    def read_vision_labels(self, *args):
        return self._read("read_vision_labels", *args)

    def read_ctranslate2_metadata(self, *args):
        return self._read("read_ctranslate2_metadata", *args)

    def read_numeric_archive(self, *args):
        return self._read("read_numeric_archive", *args)

    def read_onnx_metadata(self, *args):
        return self._read("read_onnx_metadata", *args)

    def read_gguf_metadata(self, *args):
        return self._read("read_gguf_metadata", *args)

    def download(self, *args, **kwargs):
        raise RuntimeError("WEIGHT_DOWNLOAD_FORBIDDEN_IN_METADATA_AUDIT")


class InspectionOnlyProvider:
    def invoke(self, pack, runtime, command, payload, **kwargs):
        if command != "inspect" or pack.pack_id != "oryn.mflux-mlx":
            raise RuntimeError("EXECUTION_FORBIDDEN_IN_METADATA_AUDIT")
        from oryn_mflux_provider.inspection import inspect_inventory

        return inspect_inventory(payload["inventory"], payload["metadata"])


class CatalogInspectionStore(ModelStore):
    """Evaluate signed loader contracts with an explicitly assumed runtime/device.

    These assumptions remove local pack installation and memory pressure from L1.
    The resulting `executable` flag MUST NOT be published as execution evidence.
    Production ModelStore behavior is unchanged.
    """

    def __init__(self, root, hub, *, target_platform="macos-arm64", gguf_memory_budget=2**60):
        self.target = TARGETS[target_platform]
        # L1 assumes unlimited memory unless a budget is given, so the most preferred
        # GGUF variant is reported; a budget reports the variant that machine would pick.
        self.gguf_memory_budget = gguf_memory_budget
        super().__init__(
            root,
            hub=hub,
            runtime_available=True,
            runtime_planning_state_provider=lambda: ("balanced", 2**60),
        )
        self.provider_profiles = []
        self._provider_process = InspectionOnlyProvider()
        self._pack_runtime_locator = SimpleNamespace(resolve=self._metadata_runtime)

    @staticmethod
    def _metadata_runtime(pack):
        if not pack.provider:
            return SimpleNamespace()
        spec = importlib.util.find_spec("oryn_mflux_provider")
        if pack.pack_id != "oryn.mflux-mlx" or spec is None:
            raise RuntimeError("METADATA_PROVIDER_UNAVAILABLE")
        return SimpleNamespace(site_packages=Path(spec.origin).parent.parent)

    def _hardware_info(self):
        return self.target.hardware()

    def _gguf_memory_budget(self):
        return self.gguf_memory_budget

    @staticmethod
    def foundation_digest():
        registry = builtin_foundation_registry()
        return canonical_digest(
            {
                "foundations": sorted(f.digest for f in registry.foundations),
                "adapters": sorted(a.digest for a in registry.adapters),
            }
        )

    def inspect(self, *args, **kwargs):
        profile = super().inspect(*args, **kwargs)
        # L1 can describe a candidate, but cannot give a device load action.
        for item in [profile, *(r["profile"] for r in self.provider_profiles)]:
            item["loadDisposition"] = None
            item.pop("runtimePlan", None)
            # The production inspector may reject an absent/unknown memory
            # estimate. L1 does not measure resources, so retain the artifact
            # and runtime failures without claiming this machine lacks memory.
            if "blockingReasons" in item:
                item["blockingReasons"] = [
                    reason
                    for reason in item["blockingReasons"]
                    if reason != "MODEL_MEMORY_INSUFFICIENT"
                ]
            item["eligibility"] = {
                **(item.get("eligibility") or {}),
                "resources": {"state": "not-evaluated", "reasons": []},
            }
        return profile

    def _available_pack_backends(self, declared):
        return [b for b in self.target.backends if b in declared]

    def _executable_backends(self):
        return list(self.target.backends)

    def _inspect_backend(self, *args, **kwargs):
        pack = self._runtime_packs.active_pack(kwargs["backend"])
        adapter = self._runtime_foundations.adapter_for_legacy_pack(pack.pack_id)
        foundation = (
            self._runtime_foundations.active_foundation(adapter.foundation_id) if adapter else None
        )
        eligible = self.target.eligible_backends(pack, adapter, foundation)
        if not eligible:
            profile = {
                "model": args[0],
                "exactRevision": kwargs["info_value"].sha,
                "executable": False,
                "executableBackends": [],
                "selectedFiles": [],
                "technicalCapabilities": {},
                "blockingReasons": ["RUNTIME_PLATFORM_UNDECLARED"],
            }
            self.provider_profiles.append({"provider": pack.pack_id, "profile": profile})
            return profile
        previous = self._hub.current_provider
        self._hub.current_provider = kwargs["backend"]
        offers = kwargs.get("offers")
        start = len(offers) if offers is not None else 0
        try:
            profile = super()._inspect_backend(*args, **kwargs)
        finally:
            self._hub.current_provider = previous
        if offers is not None:
            for number in range(start, len(offers)):
                candidate, projected, plan = offers[number]
                runner = pack.runner(projected.get("runtimePack", {}).get("runnerId"))
                if (
                    candidate.identity.backend not in eligible
                    or (projected.get("runtimePack") and runner is None)
                    or (runner is not None and candidate.identity.backend not in runner.backends)
                ):
                    projected = {
                        **projected,
                        "executable": False,
                        "executableBackends": [],
                        "blockingReasons": [
                            *projected.get("blockingReasons", []),
                            "RUNTIME_TUPLE_UNDECLARED",
                        ],
                    }
                    offers[number] = (
                        replace(candidate, runtime_available=False, task_compatible=False),
                        projected,
                        plan,
                    )
        if profile:
            self.provider_profiles.append({"provider": kwargs["backend"], "profile": profile})
        return profile
