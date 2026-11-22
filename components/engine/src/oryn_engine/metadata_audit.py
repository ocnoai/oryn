"""CSV-driven layer-one support tests; no weights, inference or LLM judgments."""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import threading
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urlsplit

from oryn_protocol.task_contracts import REGISTRY_DIGEST, TASK_CONTRACTS, validate_requirements

from oryn_engine.audit_metadata import (
    CatalogInspectionStore,
    MetadataSnapshot,
    error_code,
    save_json,
)
from oryn_engine.audit_profiles import TARGET_CHOICES, profile_provenance, targets_for
from oryn_engine.capability_sources import framework_sources
from oryn_engine.compatibility_plane import compile_registry
from oryn_engine.native_capability_sources import native_sources
from oryn_engine.providers import safe_relative
from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
from oryn_engine.runtime_routing import canonical_digest, runtime_build_digest

STATUSES = {
    "metadata-compatible",
    "metadata-incompatible",
    "metadata-incomplete",
    "access-required",
    "invalid-input",
    "error",
}
LINK_COLUMNS = ("url", "link", "model_url", "hf_url", "repository", "model")
ASSUMPTIONS = {
    "layer": "metadata-only",
    "runtimePacks": "signed catalog assumed available; not installed or executed by this test",
    "backends": ["mlx", "metal", "mps", "cpu", "coreml"],
    "deviceResources": "not-tested; memory estimates are not measured requirements",
    "weightIntegrity": "not-tested; file identities are host declarations",
    "loading": "not-tested",
    "inference": "not-tested",
    "packagedApp": "not-tested",
}


def parse_reference(value: str, revision: str = "") -> tuple[str, str]:
    value = value.strip()
    url_revision = ""
    if "://" in value:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
            raise ValueError("Only https://huggingface.co model links are accepted")
        parts = parsed.path.strip("/").split("/")
        if len(parts) == 4 and parts[2] == "tree":
            url_revision = unquote(parts[3])
        elif len(parts) != 2:
            raise ValueError("Use a repository link or /tree/revision, not a file or subdirectory")
        value = "/".join(parts[:2])
    if (
        not re.fullmatch(r"[A-Za-z0-9._-]{1,96}/[A-Za-z0-9._-]{1,96}", value)
        or value.split("/")[0] in {"datasets", "spaces", "collections"}
        or any(part in {".", ".."} for part in value.split("/"))
    ):
        raise ValueError("Invalid Hugging Face model repository")
    if revision and url_revision and revision != url_revision:
        raise ValueError("CSV revision conflicts with the link revision")
    revision = revision or url_revision or "main"
    if not re.fullmatch(r"[A-Za-z0-9._/-]{1,128}", revision) or ".." in revision:
        raise ValueError("Invalid revision")
    return value, revision


def read_rows(path: Path, column: str | None = None) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream, strict=True))
    rows = [r for r in rows if any(v.strip() for v in r)]
    if not rows:
        raise ValueError("CSV has no model rows")
    header = [v.strip().lower() for v in rows[0]]
    candidates = [name for name in LINK_COLUMNS if name in header]
    selected = column.lower() if column else next(iter(candidates), None)
    if selected is None:
        if any(len(r) != 1 for r in rows):
            raise ValueError("CSV needs a url column, or use --url-column")
        header, selected = ["url"], "url"
    else:
        rows = rows[1:]
    if selected not in header or len(set(header)) != len(header) or not rows:
        raise ValueError("CSV headers are ambiguous, missing, or contain no data")
    result = []
    for number, row in enumerate(rows, 1):
        values = dict(zip(header, [v.strip() for v in row], strict=False))
        values["input"] = values.get(selected, "")
        values["row"] = number
        if len(row) != len(header):
            values["inputError"] = "CSV row has a different number of columns than the header"
        result.append(values)
    return result


def requirements_for(task, config=None):
    from oryn_engine.diffusion_tasks import pipeline_contract, pipeline_contracts
    from oryn_engine.model_inspection import effective_task

    family = config.get("_class_name", "") if isinstance(config, dict) else ""
    family = family if isinstance(family, str) else ""
    if (
        task == "keypoint-detection"
        and isinstance(config, dict)
        and effective_task("transformers", task, config) == "pose-estimation"
    ):
        contract = TASK_CONTRACTS["pose-estimation-v1"]
        return {
            "task": task,
            "contract": {"id": contract["id"], "revision": 1},
            "inputModalities": contract["inputModalities"],
            "outputModalities": contract["outputModalities"],
        }
    native_contract = pipeline_contract(family, task)
    contract = (
        TASK_CONTRACTS.get(native_contract)
        if native_contract in pipeline_contracts(family)
        else None
    )
    contract = contract or next(
        (c for c in TASK_CONTRACTS.values() if task in [c["task"], *c.get("taskAliases", [])]), None
    )
    if contract is None:
        return None
    requirements = {
        "task": task,
        "contract": {"id": contract["id"], "revision": contract["revision"]},
        "inputModalities": contract["inputModalities"],
        "outputModalities": contract["outputModalities"],
    }
    if contract["id"] == "speech-synthesis-v1":
        # Reference audio is an explicitly negotiated speech branch. A fixed-voice
        # benchmark must not acquire a new mandatory input from that extension.
        requirements["inputModalities"] = ["text"]
        from oryn_engine import sopro_tasks

        try:
            speech = sopro_tasks.normalized_config(config or {}, task)
        except (ValueError, TypeError):
            speech = {}
        if sopro_tasks.supports(speech, task):
            requirements["inputModalities"] = ["text", "audio"]
            requirements["requiredFeatures"] = ["referenceAudio"]
    if contract["id"] == "audio-transformation-v1":
        # The canonical result is a union: enhanced/separated audio OR codec
        # tokens. An audio-to-audio benchmark requires playable audio; it does
        # not simultaneously require the optional codec-token output branch.
        # An encoder-only component therefore still needs an audio composition.
        requirements["outputModalities"] = ["audio"]
    if contract["id"] == "chat-v1":
        requirements["inputModalities"] = (
            ["text", "image"] if task == "image-text-to-text" else ["text"]
        )
        if task == "image-text-to-text":
            requirements["requiredFeatures"] = ["multimodalConversation"]
    if (
        contract["id"] == "multimodal-generation-v1"
        and isinstance(config, dict)
        and config.get("model_type") == "gemma4"
    ):
        from oryn_engine.gemma4_tasks import input_modes

        # The native conditional graph has image/audio encoders and a text head.
        # The contract's optional audio-output branch is not this graph's task.
        modes = input_modes(config)
        requirements["inputModalities"] = ["text", *modes]
        requirements["outputModalities"] = ["text"]
        requirements["requiredFeatures"] = [mode + "Grounding" for mode in modes]
        requirements["parameterValues"] = {"outputMode": "text"}
    validate_requirements(requirements)
    return requirements


def selected_files(value):
    files = json.loads(value)
    if (
        not isinstance(files, list)
        or not files
        or any(not safe_relative(f) for f in files)
        or len(set(files)) != len(files)
    ):
        raise ValueError("files must be a nonempty JSON array of unique relative filenames")
    return files


def classify(profile, failures):
    reasons = sorted(set(profile.get("blockingReasons", [])) - {"CONFORMANCE_REQUIRED"})
    selected = {f["name"] for f in profile.get("selectedFiles", [])}
    # Missing metadata in a declined alternative must not invalidate a complete route.
    invalid = {name: code for name, code in failures.items() if code == "MODEL_METADATA_INVALID"}
    missing = {
        name: code for name, code in failures.items() if name in selected and name not in invalid
    }
    if profile.get("executable") and selected and profile.get("taskProfile"):
        if selected & invalid.keys():
            return (
                "metadata-incompatible",
                sorted(set(reasons) | {"MODEL_METADATA_INVALID"}),
                missing,
            )
        return (
            "metadata-incomplete" if missing else "metadata-compatible",
            sorted(set(reasons) | ({"SELECTED_METADATA_UNAVAILABLE"} if missing else set())),
            missing,
        )
    if "MODEL_ACCESS_RESTRICTED" in reasons:
        return "access-required", reasons, failures
    if (
        set(failures) - invalid.keys()
        or profile.get("providerDiagnostics")
        or any("UNRESOLVED" in r or "INSPECTION_FAILED" in r for r in reasons)
    ):
        return "metadata-incomplete", reasons, failures
    return "metadata-incompatible", reasons, missing


def _inspect_row(row, snapshot, *, store_factory, target_platform):
    result = {
        "row": row["row"],
        "input": row["input"],
        "expected": row.get("expected", ""),
        "repository": snapshot.repository,
        "requestedRevision": snapshot.requested_revision,
        "task": row.get("task"),
        "status": "metadata-incomplete",
        "reasons": [],
        "selectedFiles": [],
        "accessMode": snapshot.data["accessMode"],
        "executionVerified": False,
    }
    try:
        from oryn_engine.model_composition import CompositionHub

        composed_hub = CompositionHub(snapshot)
        info = composed_hub.model_info(snapshot.repository, snapshot.requested_revision)
        result["revision"] = info.sha
        task = row.get("task") or info.pipeline_tag
        result["task"] = task
        result["taskSource"] = "csv" if row.get("task") else "hub-metadata"
        result["metadata"] = {
            "architectures": info.config.get("architectures", []),
            "pipelineClass": info.config.get("_class_name"),
            "modelType": info.config.get("model_type"),
            "library": info.library_name,
            "hubTask": info.pipeline_tag,
        }
        if not composed_hub.has_access(snapshot.repository):
            result.update(status="access-required", reasons=["MODEL_ACCESS_RESTRICTED"])
        elif not task:
            result["reasons"] = ["TASK_UNSPECIFIED"]
        elif (requirements := requirements_for(task, info.config)) is None:
            result.update(status="metadata-incompatible", reasons=["TASK_CONTRACT_MISSING"])
        else:
            model = {
                "provider": "huggingface",
                "repository": snapshot.repository,
                "revision": info.sha,
            }
            if row.get("files"):
                model["files"] = selected_files(row["files"])
            with tempfile.TemporaryDirectory(prefix="oryn-metadata-audit-") as temporary:
                store = store_factory(Path(temporary), snapshot, target_platform=target_platform)
                profile = store.inspect(model, requirements)
                selected_provider = profile.get("runtimePack", {}).get("packId")
                selected_failures = {
                    **snapshot.provider_failures.get("common", {}),
                    **snapshot.provider_failures.get(selected_provider, {}),
                }
                status, reasons, missing = classify(
                    profile, selected_failures if profile.get("executable") else snapshot.failures
                )
                result.update(
                    status=status,
                    reasons=reasons,
                    missingEvidence=missing,
                    requirements=requirements,
                    inspectorProfile=profile,
                    providerResults=store.provider_profiles,
                    selectedFiles=profile.get("selectedFiles", []),
                    estimatedMemoryBytes=profile.get("estimatedMemoryBytes"),
                    taskProfile=profile.get("taskProfile"),
                    runtime=profile.get("runtimePack"),
                    nextLayerReasons=sorted(
                        set(profile.get("blockingReasons", [])) & {"CONFORMANCE_REQUIRED"}
                    ),
                )
    except Exception as exc:
        code = error_code(exc)
        if code == "MODEL_ACCESS_RESTRICTED":
            status = "access-required"
        elif code == "MODEL_METADATA_INVALID":
            status = "metadata-incompatible"
        elif (
            snapshot.failures
            or code
            in {
                "SNAPSHOT_MISSING",
                "MODEL_NOT_FOUND",
                "MODEL_OR_REVISION_NOT_FOUND",
                "METADATA_RATE_LIMITED",
                "MODEL_METADATA_UNAVAILABLE",
                "METADATA_NETWORK_ERROR",
            }
            or type(exc).__module__.startswith(("httpx", "huggingface_hub"))
        ):
            status = "metadata-incomplete"
        else:
            status = "error"
        # Do not serialize exception text: HTTP exceptions can contain credentials/URLs.
        result.update(
            status=status,
            reasons=[code],
            missingEvidence={
                name: reason
                for name, reason in snapshot.failures.items()
                if reason != "MODEL_METADATA_INVALID"
            },
        )
    result["revision"] = snapshot.data.get("revision")
    result["snapshotDigest"] = snapshot.digest
    result["snapshot"] = str(snapshot.path)
    result["metadataReadFailures"] = dict(snapshot.failures)
    result["metadataReadFailuresByProvider"] = snapshot.provider_failures
    inventory = {f["name"]: f for f in snapshot.data.get("inventory", {}).get("files", [])}
    result["selectedFiles"] = [inventory.get(f["name"], f) for f in result["selectedFiles"]]
    from oryn_engine.model_composition import binding, source_ref

    composition = binding(snapshot.repository, result["revision"])
    if composition is not None:
        result["composition"] = composition
        dependencies = {
            (x["payload"]["repository"], x["payload"].get("revision")): x["payload"]
            for x in snapshot.data.get("dependencies", {}).values()
        }
        sources = {(snapshot.repository, result["revision"]): snapshot.data, **dependencies}
        selected = []
        for item in result["selectedFiles"]:
            ref, name = source_ref(composition, item["name"])
            original = next(
                (
                    f
                    for f in sources.get((ref["repository"], ref["revision"]), {})
                    .get("inventory", {})
                    .get("files", [])
                    if f["name"] == name
                ),
                {},
            )
            selected.append({**original, **item, "source": {**ref, "file": name}})
        result["selectedFiles"] = selected
    result["identity"] = canonical_digest(
        [
            result["repository"],
            result.get("revision") or result["requestedRevision"],
            result.get("task") or row.get("task"),
            result["selectedFiles"] or row.get("files", ""),
        ]
    )
    result["originalIdentity"] = canonical_digest(
        [
            result["repository"],
            result.get("revision") or result["requestedRevision"],
            result.get("task") or row.get("task"),
        ]
    )
    return result


def inspect_row(
    row, snapshot, *, store_factory=CatalogInspectionStore, target_platform="macos-arm64"
):
    profiles = {}
    for target in targets_for(target_platform):
        # Read failures are causal to a profile, not inherited from a declined route.
        snapshot.failures, snapshot.provider_failures = {}, {}
        profiles[target.id] = _inspect_row(
            row, snapshot, store_factory=store_factory, target_platform=target.id
        )
    passing = [key for key, result in profiles.items() if result["status"] == "metadata-compatible"]
    priority = (
        "error",
        "invalid-input",
        "access-required",
        "metadata-incomplete",
        "metadata-incompatible",
    )
    chosen = (
        profiles[passing[0]]
        if passing
        else next(
            result
            for status in priority
            for result in profiles.values()
            if result["status"] == status
        )
    )
    result = dict(chosen)
    result.update(
        targetPlatform=target_platform,
        passingProfiles=passing,
        profileResults=profiles,
        snapshotDigest=snapshot.digest,
    )
    return result


def failed(result, policy):
    if result["status"] in {"error", "invalid-input"}:
        return True
    expected = result.get("expected")
    return (
        (bool(expected) and result["status"] != expected)
        if policy == "unexpected"
        else result["status"] != "metadata-compatible"
        if policy == "non-candidate"
        else False
    )


def write_reports(output, results, provenance, policy):
    unique = {
        r.get("originalIdentity", r["identity"]): r
        for r in results
        if r.get("identity") and r.get("revision")
    }
    profiles = targets_for(provenance.get("targetPlatform", "macos-arm64"))
    compatible = {
        t.id: {key for key, value in unique.items() if t.id in value.get("passingProfiles", [])}
        for t in profiles
    }
    mac = compatible.get("macos-arm64", set())
    nvidia = set().union(
        *(compatible.get(t, set()) for t in ("linux-x64-cuda", "windows-x64-cuda"))
    )
    measured_both = "macos-arm64" in compatible and any(
        t in compatible for t in ("linux-x64-cuda", "windows-x64-cuda")
    )
    summary = {
        "inputRows": len(results),
        "uniqueIdentities": len(unique),
        "unresolvedRows": sum(not r.get("revision") for r in results),
        "rowStatuses": dict(sorted(Counter(r["status"] for r in results).items())),
        "uniqueStatuses": dict(sorted(Counter(r["status"] for r in unique.values()).items())),
        "failedChecks": sum(failed(r, policy) for r in results),
        "executedModels": 0,
        "failOn": policy,
        "profileStatuses": {
            t.id: dict(
                sorted(
                    Counter(
                        r.get("profileResults", {}).get(t.id, {}).get("status", "error")
                        for r in results
                    ).items()
                )
            )
            for t in profiles
        },
        "platformMetadataCoverage": {
            **{key: len(value) for key, value in compatible.items()},
            "nvidia": len(nvidia)
            if any(t in compatible for t in ("linux-x64-cuda", "windows-x64-cuda"))
            else None,
            "macNvidiaIntersection": len(mac & nvidia) if measured_both else None,
            "macNvidiaUnion": len(mac | nvidia) if measured_both else None,
        },
    }
    save_json(
        output / "results.json",
        {
            "schemaVersion": 1,
            "assumptions": {
                **ASSUMPTIONS,
                "backends": sorted(
                    {
                        b
                        for t in targets_for(provenance.get("targetPlatform", "macos-arm64"))
                        for b in t.backends
                    }
                ),
                "targetProfiles": provenance.get("targetProfiles"),
            },
            "provenance": provenance,
            "summary": summary,
            "results": results,
        },
    )
    fields = [
        "row",
        "input",
        "repository",
        "requestedRevision",
        "revision",
        "task",
        "status",
        "expected",
        "reasons",
        "nextLayerReasons",
        "missingEvidence",
        "metadata",
        "runtime",
        "estimatedMemoryBytes",
        "accessMode",
        "selectedFiles",
        "snapshotDigest",
        "snapshot",
        "executionVerified",
        "originalIdentity",
        "targetPlatform",
        "passingProfiles",
        "profileResults",
    ]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    k: json.dumps(result[k], sort_keys=True)
                    if isinstance(result.get(k), (dict, list))
                    else result.get(k, "")
                    for k in fields
                }
            )
    suite = ET.Element(
        "testsuite",
        name="oryn-model-metadata",
        tests=str(len(results)),
        failures=str(summary["failedChecks"]),
        skipped=str(
            sum(
                not failed(r, policy)
                and r["status"] != "metadata-compatible"
                and r.get("expected") != r["status"]
                for r in results
            )
        ),
    )
    for result in results:
        case = ET.SubElement(
            suite,
            "testcase",
            classname="model.metadata",
            name=(
                f"{result['row']}: {result.get('repository', result['input'])}"
                f"@{result.get('revision', '')}:{result.get('task', '')}"
            ),
        )
        message = result["status"] + ": " + ", ".join(result["reasons"])
        if failed(result, policy):
            ET.SubElement(case, "failure", message=message).text = json.dumps(result)
        elif (
            result["status"] != "metadata-compatible" and result.get("expected") != result["status"]
        ):
            ET.SubElement(case, "skipped", message=message)
        ET.SubElement(case, "system-out").text = message
    ET.ElementTree(suite).write(output / "junit.xml", encoding="utf-8", xml_declaration=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV of Hugging Face links or repository IDs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, help="Frozen metadata directory; default: OUTPUT/snapshots"
    )
    parser.add_argument("--url-column")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target-platform", choices=TARGET_CHOICES, default="macos-arm64")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--offline", action="store_true", help="Only read existing snapshots")
    modes.add_argument(
        "--refresh", action="store_true", help="Resolve mutable refs and collect metadata again"
    )
    parser.add_argument("--token-env", help="Explicit opt-in to a token environment variable")
    parser.add_argument(
        "--gguf-memory-budget-gib",
        type=float,
        help="Choose GGUF variants for a machine with this much usable memory",
    )
    parser.add_argument(
        "--fail-on", choices=["non-candidate", "unexpected", "none"], default="non-candidate"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    token = os.environ.get(args.token_env) if args.token_env else None
    if args.token_env and not token:
        parser.error("The requested token environment variable is empty")
    try:
        rows = read_rows(args.csv, args.url_column)
    except (ValueError, OSError, csv.Error) as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.cache or args.output / "snapshots"
    store_factory = CatalogInspectionStore
    if args.gguf_memory_budget_gib is not None:
        if not 0 < args.gguf_memory_budget_gib <= 16384:
            parser.error("--gguf-memory-budget-gib must be between 0 and 16384")
        store_factory = functools.partial(
            CatalogInspectionStore,
            gguf_memory_budget=int(args.gguf_memory_budget_gib * 1024**3),
        )
    locks = defaultdict(threading.Lock)
    refreshed = set()

    def check(row):
        try:
            if row.get("inputError"):
                raise ValueError(row["inputError"])
            if row.get("expected") and row["expected"] not in STATUSES:
                raise ValueError("Unknown expected status")
            repository, revision = parse_reference(row["input"], row.get("revision", ""))
            if row.get("files"):
                selected_files(row["files"])
        except ValueError as exc:
            return {
                "row": row["row"],
                "input": row["input"],
                "status": "invalid-input",
                "expected": row.get("expected", ""),
                "reasons": [str(exc)],
                "executionVerified": False,
            }
        key = canonical_digest([repository, revision, "authenticated" if token else "anonymous"])
        with locks[key]:
            path = cache / (key + ".json")
            if args.refresh and key not in refreshed:
                path.unlink(missing_ok=True)
                refreshed.add(key)
            try:
                snapshot = MetadataSnapshot(
                    path, repository, revision, offline=args.offline, token=token
                )
                result = inspect_row(
                    row,
                    snapshot,
                    target_platform=args.target_platform,
                    store_factory=store_factory,
                )
            except Exception as exc:
                result = {
                    "row": row["row"],
                    "input": row["input"],
                    "repository": repository,
                    "status": "error",
                    "reasons": [error_code(exc)],
                    "executionVerified": False,
                }
            save_json(args.output / "rows" / f"{row['row']:06d}.json", result)
            print(f"[{row['row']}/{len(rows)}] {repository}: {result['status']}", file=sys.stderr)
            return result

    provenance = audit_provenance(
        args.csv,
        args.target_platform,
        offline=args.offline,
        access_mode="authenticated" if token else "anonymous",
    )
    if args.gguf_memory_budget_gib is not None:
        provenance["ggufMemoryBudgetBytes"] = int(args.gguf_memory_budget_gib * 1024**3)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(check, rows))
    summary = write_reports(args.output, results, provenance, args.fail_on)
    print(json.dumps(summary, indent=2))
    if any(r["status"] in {"error", "invalid-input"} for r in results):
        return 2
    return 1 if summary["failedChecks"] else 0


def audit_provenance(csv_path, target_platform, *, offline, access_mode):
    return {
        "engineBuildDigest": runtime_build_digest(),
        "engineSourceDigest": canonical_digest(
            [
                (
                    str(p.relative_to(Path(__file__).parent)),
                    hashlib.sha256(p.read_bytes()).hexdigest(),
                )
                for p in sorted(Path(__file__).parent.rglob("*.py"))
            ]
        ),
        "capabilityRegistryDigest": compile_registry(
            (*framework_sources(), *native_sources())
        ).digest,
        "runtimePackDigests": {p.pack_id: p.digest for p in builtin_registry().packs},
        "foundationCatalogDigest": CatalogInspectionStore.foundation_digest(),
        "taskRegistryDigest": REGISTRY_DIGEST,
        "inputSha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "hostPlatform": platform.system(),
        "offline": offline,
        "accessMode": access_mode,
        "targetPlatform": target_platform,
        "targetProfiles": profile_provenance(target_platform),
    }


if __name__ == "__main__":
    raise SystemExit(main())
