"""Release-bound content index. Repository names and paths are not content keys.

Profiles are build artifacts authenticated by the signed capability build. The
advisory Registry is deliberately not consulted. Only a fixed installed adapter
may be selected; no module name/entrypoint from a profile is ever imported.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from urllib.parse import unquote, urlsplit

from oryn_engine.source_inspector_worker import MAX_SOURCE_BYTES, NORMALIZER, InspectionError

MAX_FILES = 64
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MAX_PROFILES = 20_000
MAX_CONTENTS = 100_000
MAX_SHARDS = 64
MAX_SHARD_BYTES = 1024 * 1024
MAX_CATALOG_BYTES = 64 * 1024 * 1024
MAX_ALIASES = 8
SOURCE_IO_SECONDS = 60
DIGEST = re.compile(r"[0-9a-f]{64}")
MODULE = re.compile(r"[A-Za-z_][A-Za-z_0-9]{0,127}")
ADAPTERS = frozenset({"oryn.lfm3.torch.v1"})
POLICY = "bundled-only-no-upstream-kernels-v1"


def content_collection(digests):
    """Sorted multiset; deliberately excludes filenames and retains duplicates."""
    values = list(digests)
    if (
        not values
        or len(values) > MAX_FILES
        or any(not isinstance(value, str) or DIGEST.fullmatch(value) is None for value in values)
    ):
        raise InspectionError("Invalid content collection")
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()


class ImplementationIndex:
    def __init__(self, profiles, *, identity):
        if not isinstance(profiles, list) or len(profiles) > MAX_PROFILES:
            raise InspectionError("Implementation profile limit")
        self.identity = identity
        self.profiles = {}
        self.raw = defaultdict(list)
        self.structural = defaultdict(list)
        self.contents = set()
        for profile in profiles:
            self._add(profile)

    def _add(self, p):
        if not isinstance(p, dict) or set(p) != {
            "id",
            "version",
            "normalizer",
            "adapter",
            "modules",
            "bindings",
            "state",
            "sourcePolicy",
            "representations",
            "references",
        }:
            raise InspectionError("Unknown implementation profile schema")
        if (
            not isinstance(p["id"], str)
            or re.fullmatch(r"[a-z0-9.-]{1,128}", p["id"]) is None
            or p["id"] in self.profiles
            or type(p["version"]) is not int
            or p["version"] != 1
            or p["normalizer"] != NORMALIZER
            or p["adapter"] not in ADAPTERS
            or p["state"] not in {"active", "revoked", "recognition-only"}
            or p["sourcePolicy"] != POLICY
            or not isinstance(p["representations"], list)
            or any(
                r not in {"bf16-cpu", "fp8-unsupported", "nvfp4-unsupported"}
                for r in p["representations"]
            )
            or not isinstance(p["references"], list)
            or len(p["references"]) > 16
            or not isinstance(p["modules"], dict)
            or not 1 <= len(p["modules"]) <= MAX_FILES
            or not isinstance(p["bindings"], dict)
            or not 1 <= len(p["bindings"]) <= 16
        ):
            raise InspectionError("Invalid implementation identity or authority")
        for role, record in p["modules"].items():
            if (
                MODULE.fullmatch(role) is None
                or not isinstance(record, dict)
                or set(record)
                != {
                    "raw",
                    "structural",
                    "exports",
                }
            ):
                raise InspectionError("Invalid implementation module role")
            if any(
                (record[k] is not None or k == "raw")
                and (not isinstance(record[k], str) or DIGEST.fullmatch(record[k]) is None)
                for k in ("raw", "structural")
            ):
                raise InspectionError("Invalid implementation content identity")
            if (
                not isinstance(record["exports"], list)
                or len(record["exports"]) > 64
                or any(
                    not isinstance(x, str) or MODULE.fullmatch(x) is None for x in record["exports"]
                )
            ):
                raise InspectionError("Invalid implementation exports")
            self.contents.add(record["raw"])
            if len(self.contents) > MAX_CONTENTS:
                raise InspectionError("Catalog content capacity exceeded")
        for key, binding in p["bindings"].items():
            if (
                key not in {"AutoConfig", "AutoModel", "AutoModelForCausalLM"}
                or not isinstance(binding, list)
                or len(binding) != 2
                or binding[0] not in p["modules"]
                or binding[1] not in p["modules"][binding[0]]["exports"]
            ):
                raise InspectionError("Invalid loader binding")
        self.profiles[p["id"]] = p
        if p["state"] == "revoked":
            return
        for index, kind in ((self.raw, "raw"), (self.structural, "structural")):
            if any(m[kind] is None for m in p["modules"].values()):
                continue
            key = content_collection(m[kind] for m in p["modules"].values())
            index[key].append(p["id"])
            if len(index[key]) > MAX_ALIASES:
                raise InspectionError("Ambiguous implementation collection")

    def match(self, modules, auto_map):
        if (
            not isinstance(modules, dict)
            or not 1 <= len(modules) <= MAX_FILES
            or not isinstance(auto_map, dict)
        ):
            return None
        for kind, index in (("raw", self.raw), ("structural", self.structural)):
            digests = [m.get(f"{kind}Sha256") for m in modules.values()]
            if any(d is None for d in digests):
                continue
            candidates = index.get(content_collection(digests), ())
            matches = []
            for profile_id in candidates:
                p = self.profiles[profile_id]
                if set(auto_map) != set(p["bindings"]):
                    continue
                role_paths = {}
                valid = True
                for role, expected in p["modules"].items():
                    paths = [
                        path
                        for path, facts in modules.items()
                        if facts[f"{kind}Sha256"] == expected[kind]
                        and set(expected["exports"]).issubset(facts["exports"])
                    ]
                    if len(paths) != 1:
                        valid = False
                        break
                    role_paths[role] = paths[0]
                if not valid or len(set(role_paths.values())) != len(modules):
                    continue
                for key, (role, class_name) in p["bindings"].items():
                    if auto_map[key] != f"{role_paths[role][:-3]}.{class_name}":
                        valid = False
                if valid:
                    matches.append((p, kind, role_paths))
            if len(matches) == 1:
                return matches[0]
            if matches:
                return None
        return None


def build_binding(root: Path) -> dict:
    """Exact code/runtime identity authenticated by the existing release signer."""
    relative = [
        "source_inspector_worker.py",
        "source_inspection.py",
        "reviewed_implementations.py",
        "data/source-inspection.sb",
        "data/reviewed-inference.sb",
        "native_lfm3.py",
        "runtime_packs/lfm3_worker.py",
        "runtime_packs/transformers_worker.py",
        "runtime_packs/execution.py",
        "runtime_packs/broker.py",
        "models.py",
        "runners.py",
    ]
    files = {
        name: hashlib.sha256(_bounded_installed(root / name, 1024 * 1024)).hexdigest()
        for name in relative
    }
    catalog = root / "data/reviewed-implementations.json"
    return {
        "schemaVersion": 1,
        "normalizer": NORMALIZER,
        "files": files,
        "catalogSha256": hashlib.sha256(_bounded_installed(catalog, 256 * 1024)).hexdigest(),
        "parserPython": (
            f"{sys.implementation.name}-{sys.version_info.major}."
            f"{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "dependencies": {name: version(name) for name in ("torch", "safetensors", "tokenizers")},
    }


def _bounded_installed(path, limit):
    if path.is_symlink():
        raise InspectionError("Installed implementation data is aliased")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if not 0 < len(raw) <= limit:
        raise InspectionError("Installed implementation byte limit")
    return raw


def installed_index():
    from oryn_engine.generated_capabilities import generated_artifact

    root = Path(__file__).parent
    binding = generated_artifact()["buildInputs"].get("reviewedImplementations")
    if not isinstance(binding, dict) or binding != build_binding(root):
        raise InspectionError("REVIEWED_IMPLEMENTATION_UNAVAILABLE")
    raw = _bounded_installed(root / "data/reviewed-implementations.json", 256 * 1024)
    manifest = json.loads(raw)
    shards = manifest.get("shards") if isinstance(manifest, dict) else None
    if not isinstance(shards, list) or not 1 <= len(shards) <= MAX_SHARDS:
        raise InspectionError("Invalid implementation shard list")
    identities = []
    for shard in shards:
        if (
            not isinstance(shard, dict)
            or not isinstance(shard.get("name"), str)
            or re.fullmatch(r"reviewed-implementations-[a-z0-9-]+\.json", shard["name"]) is None
        ):
            raise InspectionError("Invalid implementation shard name")
        path = root / "data" / shard["name"]
        if path.is_symlink():
            raise InspectionError("Aliased implementation shard")
        stat = path.stat()
        identities.append(
            (
                shard["name"],
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
        )
    # Recheck immutable build bytes/runtime on every admission/job; reuse the
    # activated index only while every shard's identity is unchanged. Revocation,
    # replacement and in-place mutation cannot revive a cached active profile.
    return _cached_index((json.dumps(binding, sort_keys=True), tuple(identities)))


@lru_cache(maxsize=1)
def _cached_index(snapshot):
    from oryn_engine.generated_capabilities import generated_artifact

    root = Path(__file__).parent
    binding = generated_artifact()["buildInputs"].get("reviewedImplementations")
    if not isinstance(binding, dict) or binding != build_binding(root):
        raise InspectionError("REVIEWED_IMPLEMENTATION_UNAVAILABLE")
    raw = (root / "data/reviewed-implementations.json").read_bytes()
    if len(raw) > 256 * 1024:
        raise InspectionError("Catalog root byte limit")
    manifest = json.loads(raw)
    if (
        set(manifest) != {"schemaVersion", "normalizer", "shards"}
        or manifest["schemaVersion"] != 1
        or manifest["normalizer"] != NORMALIZER
        or not isinstance(manifest["shards"], list)
        or not 1 <= len(manifest["shards"]) <= MAX_SHARDS
    ):
        raise InspectionError("Unknown catalog schema or normalizer")
    profiles, seen, total = [], set(), 0
    for shard in manifest["shards"]:
        if (
            not isinstance(shard, dict)
            or set(shard) != {"name", "sha256", "sizeBytes"}
            or not isinstance(shard["name"], str)
            or re.fullmatch(r"reviewed-implementations-[a-z0-9-]+\.json", shard["name"]) is None
            or shard["name"] in seen
            or type(shard["sizeBytes"]) is not int
            or not 0 < shard["sizeBytes"] <= MAX_SHARD_BYTES
        ):
            raise InspectionError("Invalid implementation shard")
        seen.add(shard["name"])
        path = root / "data" / shard["name"]
        if path.is_symlink() or path.stat().st_size != shard["sizeBytes"]:
            raise InspectionError("Implementation shard identity mismatch")
        data = path.read_bytes()
        total += len(data)
        if total > MAX_CATALOG_BYTES or hashlib.sha256(data).hexdigest() != shard["sha256"]:
            raise InspectionError("Implementation shard integrity mismatch")
        part = json.loads(data)
        if not isinstance(part, list) or len(profiles) + len(part) > MAX_PROFILES:
            raise InspectionError("Implementation profile capacity exceeded")
        profiles.extend(part)
    return ImplementationIndex(profiles, identity=binding["catalogSha256"])


def inspect_repository(config, files, read_source, *, parser=None, cancellation=None):
    """Bounded transitive local-module closure; no source is written to model roots."""
    from oryn_engine.source_inspection import inspect_source

    index = installed_index()
    parser = parser or inspect_source
    auto_map = config.get("auto_map")
    if not isinstance(auto_map, dict) or not 1 <= len(auto_map) <= 16:
        raise InspectionError("Unknown source declarations")
    pending = []
    for value in auto_map.values():
        if not isinstance(value, str) or value.count(".") != 1:
            raise InspectionError("Cross-repository or nested loader declaration")
        module, name = value.split(".")
        if not MODULE.fullmatch(module) or not MODULE.fullmatch(name):
            raise InspectionError("Invalid loader declaration")
        pending.append(f"{module}.py")
    available = {f.name: f for f in files}
    modules, total = {}, 0
    deadline = time.monotonic() + SOURCE_IO_SECONDS
    while pending:
        if time.monotonic() > deadline or cancellation is not None and cancellation.is_set():
            raise InspectionError("Source inspection cancelled or timed out")
        path = pending.pop()
        if path in modules:
            continue
        if (
            len(modules) >= MAX_FILES
            or path not in available
            or not 0 < available[path].size <= MAX_SOURCE_BYTES
        ):
            raise InspectionError("Missing source or inspection budget exceeded")
        # read_source must use immutable revision and bounded streaming I/O.
        raw = bytes(read_source(path, MAX_SOURCE_BYTES))
        total += len(raw)
        if len(raw) != available[path].size or total > MAX_TOTAL_BYTES:
            raise InspectionError("Source size changed or total budget exceeded")
        blob = getattr(available[path], "blob_id", None)
        if (
            isinstance(blob, str)
            and DIGEST.fullmatch(blob)
            and hashlib.sha256(raw).hexdigest() != blob
        ):
            raise InspectionError("Immutable source digest mismatch")
        if isinstance(blob, str) and re.fullmatch(r"[0-9a-f]{40}", blob):
            git = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            if git != blob:
                raise InspectionError("Immutable source blob mismatch")
        facts = parser(raw, cancellation=cancellation)
        if facts.get("normalizer") != NORMALIZER:
            raise InspectionError("Source normalizer mismatch")
        modules[path] = facts
        for binding in facts["imports"]:
            name = binding["module"]
            if binding["level"] != 1 or not isinstance(name, str) or not MODULE.fullmatch(name):
                raise InspectionError("Unresolved relative module binding")
            pending.append(name + ".py")
    for facts in modules.values():
        for binding in facts["imports"]:
            target = modules[binding["module"] + ".py"]
            if any(name not in target["exports"] for name in binding["names"]):
                raise InspectionError("Unresolved imported symbol")
    if time.monotonic() > deadline:
        raise InspectionError("Source inspection deadline exceeded")
    match = index.match(modules, auto_map)
    if match is None:
        return {"state": "unknown", "normalizer": NORMALIZER, "catalogDigest": index.identity}
    profile, kind, roles = match
    evidence = {
        "state": "recognized",
        "match": "exact" if kind == "raw" else "structural",
        "profileId": profile["id"],
        "profileVersion": profile["version"],
        "normalizer": NORMALIZER,
        "catalogDigest": index.identity,
        "adapterId": profile["adapter"],
        "sourcePolicy": POLICY,
        "sourceDigest": content_collection(f["rawSha256"] for f in modules.values()),
        "files": [
            {
                "name": name,
                "rawSha256": facts["rawSha256"],
                "structuralSha256": facts["structuralSha256"],
            }
            for name, facts in sorted(modules.items())
        ],
    }
    return evidence


def validate_hf_context(context, repository):
    if not isinstance(context, dict) or set(context) != {"sourceURL", "tabId", "documentId"}:
        raise InspectionError("Invalid trusted HF document context")
    if any(
        not isinstance(context[k], str) or not 0 < len(context[k]) <= 256
        for k in ("tabId", "documentId")
    ):
        raise InspectionError("Invalid trusted document identity")
    url = context["sourceURL"]
    if not isinstance(url, str) or len(url) > 2048:
        raise InspectionError("Invalid trusted HF source URL")
    parsed = urlsplit(url)
    path = unquote(parsed.path).strip("/")
    if (
        parsed.scheme != "https"
        or parsed.hostname != "huggingface.co"
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or parsed.query
        or parsed.fragment
        or path != repository
        and not path.startswith(repository + "/tree/")
    ):
        raise InspectionError("Source inspection is restricted to the trusted HF model page")


def validate_evidence(evidence, *, require_admission=True):
    index = installed_index()
    if (
        not isinstance(evidence, dict)
        or evidence.get("state") != "recognized"
        or evidence.get("normalizer") != NORMALIZER
        or evidence.get("catalogDigest") != index.identity
        or evidence.get("adapterId") not in ADAPTERS
    ):
        raise InspectionError("Reviewed implementation identity changed")
    profile = index.profiles.get(evidence.get("profileId"))
    if (
        not profile
        or profile["state"] != "active"
        or profile["version"] != evidence.get("profileVersion")
        or profile["adapter"] != evidence["adapterId"]
    ):
        raise InspectionError("Reviewed implementation revoked or unavailable")
    if require_admission and evidence.get("admitted") is not True:
        raise InspectionError("Reviewed implementation was not admitted")
    files = evidence.get("files")
    kind = {"exact": "raw", "structural": "structural"}.get(evidence.get("match"))
    if kind is None or not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
        raise InspectionError("Incomplete reviewed source evidence")
    if any(
        not isinstance(f, dict)
        or not isinstance(f.get("rawSha256"), str)
        or DIGEST.fullmatch(f["rawSha256"]) is None
        or not isinstance(f.get(f"{kind}Sha256"), str)
        or DIGEST.fullmatch(f[f"{kind}Sha256"]) is None
        for f in files
    ):
        raise InspectionError("Malformed reviewed source evidence")
    if (
        content_collection(f[f"{kind}Sha256"] for f in files)
        != content_collection(m[kind] for m in profile["modules"].values())
        or evidence.get("sourceDigest") != content_collection(f["rawSha256"] for f in files)
        or evidence.get("sourcePolicy") != POLICY
    ):
        raise InspectionError("Reviewed source collection changed")
    return profile
