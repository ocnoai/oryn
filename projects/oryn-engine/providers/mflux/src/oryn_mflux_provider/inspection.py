"""Data-only checkpoint resolution shared by inspection and execution."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath


def catalog():
    raw = Path(__file__).with_name("catalog.json").read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _under(name, directory):
    return PurePosixPath(name).parent.as_posix() == (directory or ".")


def numbered_shards(weights):
    """Older native MFLUX saves predate indexes and use 0..N component shards."""
    return 1 <= len(weights) <= 256 and {PurePosixPath(n).name for n in weights} == {
        f"{i}.safetensors" for i in range(len(weights))
    }


def legacy_metadata(headers):
    """Every unindexed shard must declare the same native serialization version."""
    if not headers:
        return False
    versions = [h.get("metadata", {}).get("mflux_version") for h in headers]
    quantizations = [h.get("metadata", {}).get("quantization_level") for h in headers]
    return (
        all(isinstance(v, str) and v == versions[0] for v in versions)
        and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", versions[0]) is not None
        and all(v is None or isinstance(v, str) for v in quantizations)
        and all(v == quantizations[0] for v in quantizations)
    )


def _base_models(inventory):
    config = inventory.get("config", {})
    card = inventory.get("card", {})
    values = []
    architectures = config.get("architectures")
    architectures = architectures if isinstance(architectures, list) else []
    for value in (
        config.get("base_model"),
        config.get("original_model"),
        card.get("base_model"),
        config.get("model_type"),
        config.get("_class_name"),
        *architectures,
    ):
        if isinstance(value, str):
            values.append(value.casefold())
        elif isinstance(value, list):
            values.extend(v.casefold() for v in value if isinstance(v, str))
    return values


def resolve_entry(inventory, data):
    bases = _base_models(inventory)
    contract = (
        "image-editing-v1" if inventory.get("task") == "image-to-image" else "text-to-image-v1"
    )
    entries = [e for e in data["entries"] if e["taskContract"] == contract]
    matches = [
        e
        for e in entries
        if any(
            v
            in {
                a.casefold()
                for a in [*e["aliases"], e["architecture"], e["architecture"] + "Pipeline"]
            }
            for v in bases
        )
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        return None
    # Exact upstream repository identity is a metadata hint, never an execution branch.
    exact = [
        e
        for e in entries
        if inventory["repository"].casefold() in {a.casefold() for a in e["aliases"]}
    ]
    if len(exact) == 1:
        return exact[0]
    # Naming only proposes a family; component and tensor validation still applies.
    name = inventory["repository"].split("/")[-1].casefold()
    hinted = [
        (len(a), e)
        for e in entries
        for a in e["aliases"]
        if "/" not in a and len(a) >= 4 and a.casefold() in name
    ]
    if hinted:
        longest = max(n for n, _ in hinted)
        choices = {e["id"]: e for n, e in hinted if n == longest}
        if len(choices) == 1:
            return next(iter(choices.values()))
    return None


def generation_profile(entry):
    defaults = entry["defaults"]
    specifications = {
        "width": (64, 1024, 32),
        "height": (64, 1024, 32),
        "steps": (1, 50, 1),
        "guidanceScale": (0, 20, 0.1),
    }
    return {
        "profileId": "provider.mflux." + entry["id"],
        "provenance": "runtime-declared",
        "recommended": True,
        "parameters": {
            k: {
                "default": min(max(defaults[k], lo), hi),
                "minimum": lo,
                "maximum": hi,
                "step": step,
                "supported": k != "guidanceScale"
                or entry["controls"][k] in {"honored", "conditional"},
                "editable": k != "guidanceScale"
                or entry["controls"][k] in {"honored", "conditional"},
            }
            for k, (lo, hi, step) in specifications.items()
        },
    }


def inspect_inventory(inventory, metadata):
    data, digest = catalog()
    names = {f["name"]: f for f in inventory["files"]}
    # MFLUX saved weights have indexes without the Diffusers component configuration.
    # A Hub tag is a hint; a normal Diffusers export remains available to its own provider.
    saved = any(n.endswith("model.safetensors.index.json") for n in names)
    if inventory.get("library") not in {"mflux", "mlx"} and not saved:
        return {"candidates": []}
    if "config.json" in names:
        if "config.json" not in metadata:
            return {"candidates": [], "metadataNeeded": [{"file": "config.json", "kind": "json"}]}
        # Hub's abbreviated config omits older saves' original_model declaration.
        # Resolve against the pinned JSON data and keep it in the selected graph.
        config = metadata["config.json"]
        if not isinstance(config, dict) or config.get("unavailable"):
            return {"candidates": [], "diagnostics": ["MODEL_METADATA_INVALID"]}
        inventory = {**inventory, "config": {**inventory.get("config", {}), **config}}
    entry = resolve_entry(inventory, data)
    structural = False
    indexed_krea = (
        entry is not None
        and entry["architecture"] == "Krea2"
        and "transformer/model.safetensors.index.json" in names
    )
    if indexed_krea:
        from oryn_mflux_provider.krea_layout import inspect_saved_krea

        matched, requests = inspect_saved_krea(inventory, metadata, data)
        if requests:
            return {"candidates": [], "metadataNeeded": requests}
        if matched is None:
            return {"candidates": [], "diagnostics": ["MODEL_INDEX_GRAPH_INVALID"]}
        entry = entry | {
            "components": [
                c | {"directory": c.get("indexedNativeDirectory", c["directory"])}
                for c in entry["components"]
            ]
        }
    # A family hint cannot bypass complete validation for a different saved
    # layout. Unindexed nonnumbered FLUX shards need structural resolution even
    # when their repository name happens to contain a recognized family alias.
    alternate_layout = any(
        "flux_swift_version" in h["metadata"]
        for name, h in metadata.items()
        if name.endswith(".safetensors")
        and isinstance(h, dict)
        and isinstance(h.get("metadata"), dict)
    )
    if entry is not None and entry["architecture"] == "Flux1":
        for component in entry["components"]:
            directory = component["directory"]
            weights = sorted(
                n for n in names if _under(n, directory) and n.endswith(".safetensors")
            )
            alternate_layout |= (
                f"{directory}/model.safetensors.index.json" not in names
                and len(weights) > 1
                and not numbered_shards(weights)
            )
    if entry is None or alternate_layout:
        from oryn_mflux_provider.structure import resolve_flux_graph

        entry, requests = resolve_flux_graph(inventory, metadata, data)
        if requests:
            return {"candidates": [], "metadataNeeded": requests}
        structural = entry is not None
    if entry is None:
        return {"candidates": [], "diagnostics": ["MODEL_ARCHITECTURE_UNRESOLVED"]}
    selected, components, diagnostics, requests = set(), [], [], []
    if "config.json" in names:
        selected.add("config.json")
    for component in entry["components"]:
        directory = component["directory"]
        prefix = directory + "/" if directory else ""
        index_name = prefix + "model.safetensors.index.json"
        weights = sorted(n for n in names if _under(n, directory) and n.endswith(".safetensors"))
        index = metadata.get(index_name)
        legacy = index_name not in names and len(weights) > 1 and numbered_shards(weights)
        if index_name in names:
            selected.add(index_name)
            if index is None:
                requests.append({"file": index_name, "kind": "json"})
            elif isinstance(index.get("weight_map"), dict):
                refs = set(index["weight_map"].values())
                if not refs or any(
                    not isinstance(n, str)
                    or "/" in n
                    or "\\" in n
                    or not n.endswith(".safetensors")
                    for n in refs
                ):
                    diagnostics.append("MODEL_SHARDS_INVALID")
                else:
                    weights = sorted(prefix + n for n in refs)
                    if any(n not in names for n in weights):
                        diagnostics.append("MODEL_SUBSET_INCOMPLETE")
                        weights = [n for n in weights if n in names]
            else:
                diagnostics.append("MODEL_SHARDS_INVALID")
        elif len(weights) != 1 and not legacy and not structural:
            diagnostics.append("MODEL_SUBSET_INCOMPLETE")
        if legacy:
            missing = [n for n in weights if n not in metadata]
            requests.extend({"file": n, "kind": "safetensors-header"} for n in missing)
            if not missing and not legacy_metadata([metadata[n] for n in weights]):
                diagnostics.append("MODEL_SHARDS_INVALID")
        if not weights:
            diagnostics.append("MODEL_SUBSET_INCOMPLETE")
        selected.update(weights)
        selected.update(
            n
            for n in names
            if _under(n, directory) and n.endswith(".json") and n != "model_index.json"
        )
        bits = None
        precision = component["precision"] or "checkpoint"
        if weights:
            header = metadata.get(weights[0])
            if header is None:
                requests.append({"file": weights[0], "kind": "safetensors-header"})
            elif header.get("unavailable"):
                diagnostics.append("TENSOR_METADATA_UNAVAILABLE")
            else:
                raw = header.get("metadata", {}).get("quantization_level")
                if raw is not None and raw not in ("None", "null"):
                    try:
                        bits = int(raw)
                    except (TypeError, ValueError):
                        diagnostics.append("QUANTIZATION_METADATA_INVALID")
                    # Loading support is supplied by the backend, not by repository names.
                    if bits is not None and bits not in data["quantizationBits"]:
                        diagnostics.append("QUANTIZATION_UNSUPPORTED")
                precision = "/".join(header.get("dtypes", [])) or precision
        components.append(
            {
                "name": component["name"],
                "directory": directory,
                "files": weights,
                "storedBytes": sum(names[n]["sizeBytes"] for n in weights),
                "quantizationBits": bits,
                "precision": precision,
                "metadataCoverage": "complete-index-graph"
                if indexed_krea
                else "complete-graph"
                if structural
                else "all-shards"
                if legacy
                else "first-shard",
            }
        )
    for tokenizer in entry["tokenizers"]:
        directories = [tokenizer["directory"], *tokenizer["fallbacks"]]
        directory = next(
            (
                d
                for d in directories
                if any(
                    _under(n, d) and ("tokenizer" in n or n.endswith((".model", ".spm")))
                    for n in names
                )
            ),
            directories[0],
        )
        assets = [
            n
            for n in names
            if _under(n, directory)
            and PurePosixPath(n).suffix in {".json", ".txt", ".model", ".spm", ".tiktoken"}
        ]
        if not assets:
            diagnostics.append("MODEL_TOKENIZER_MISSING")
        selected.update(assets)
    if requests:
        return {"candidates": [], "metadataNeeded": requests}
    weight_bytes = sum(c["storedBytes"] for c in components)
    largest_shard = max((names[n]["sizeBytes"] for c in components for n in c["files"]), default=0)
    # Include loading overlap even with encoder eviction: low-RAM doesn't eliminate
    # initialization peaks. No family-specific factor or invented measured peak.
    workspace = 3 * 1024**3
    peak = weight_bytes + largest_shard + workspace if weight_bytes else 0
    quantizations = sorted(
        {c["quantizationBits"] for c in components if c["quantizationBits"] is not None}
    )
    quantization = "+".join(f"INT{b}" for b in quantizations) or None
    precisions = {c["precision"] for c in components}
    precision = (
        {"BF16": "bf16", "F16": "fp16", "F32": "fp32"}.get(next(iter(precisions)), "mixed")
        if len(precisions) == 1 and not quantization
        else "mixed"
    )
    plan = {
        "schemaVersion": 1,
        "catalogDigest": digest,
        "entryId": entry["id"],
        "selectedFiles": sorted(selected),
        "components": components,
        "mode": "low-memory" if entry["controls"]["lowMemory"] else "standard",
        "maximumPixels": 1024**2,
        "estimatedMemoryBytes": peak,
    }
    return {
        "candidates": [
            {
                "id": "oryn.provider.mflux." + entry["id"],
                "architecture": entry["architecture"],
                "task": "image-to-image"
                if entry["taskContract"] == "image-editing-v1"
                else "text-to-image",
                "taskContract": entry["taskContract"],
                "runnerId": "mflux.image",
                "inputModalities": ["text", "image"]
                if entry["taskContract"] == "image-editing-v1"
                else ["text"],
                "outputModalities": ["image"],
                "backend": "mlx",
                "precision": precision,
                "quantization": quantization,
                "selectedFiles": sorted(selected),
                "components": components,
                "estimatedMemoryBytes": peak,
                "resourceEstimate": {
                    "kind": "reservation",
                    "residentBytes": 0,
                    "peakBytes": peak,
                    "confidence": "conservative",
                    "maximumPixels": 1024**2,
                },
                "generationProfile": generation_profile(entry),
                "capabilities": {
                    "deterministicSeed": True,
                    "streaming": False,
                    "imageEditing": entry["taskContract"] == "image-editing-v1",
                    "instructionEditing": entry["taskContract"] == "image-editing-v1",
                },
                "diagnostics": sorted(set(diagnostics)),
                "plan": plan,
            }
        ]
    }
