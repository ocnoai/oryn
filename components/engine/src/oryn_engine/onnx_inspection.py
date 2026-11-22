"""ONNX offers on the common discovery and admission boundary."""

from __future__ import annotations

import httpx

from oryn_engine import onnx_classifier, onnx_tdt
from oryn_engine.compatibility_plane import ModelDescriptor, evaluate_eligibility
from oryn_engine.onnx_graph import safe_path
from oryn_engine.onnx_tasks import (
    TASK_CONTRACTS,
    validate_config,
    validate_graph,
    validate_pooling,
    variant_id,
)
from oryn_engine.runtime_planning import RuntimeCandidate, RuntimeIdentity
from oryn_engine.runtime_routing import canonical_digest


def inspect_onnx(store, repository, revision, files, config, task, access, offers):
    if task not in TASK_CONTRACTS:
        return None
    graphs = [
        f.name for f in sorted(files, key=lambda f: (f.size, f.name)) if f.name.endswith(".onnx")
    ]
    if not graphs:
        return None
    pack = store._runtime_packs.active_pack("oryn.onnx-runtime")
    adapter = store._runtime_foundations.adapter_for_legacy_pack(pack.pack_id)
    foundation = store._runtime_foundations.active_foundation(adapter.foundation_id)
    available = store._pack_runtime_locator.resolve(pack) is not None
    sizes = {f.name: f.size for f in files}
    assets = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "1_Pooling/config.json",
        "preprocessor_config.json",
    } & sizes.keys()
    base_reasons = []
    vision = None
    try:
        if task == "image-classification":
            config = onnx_classifier.resolve(config, [(f.name, f.size, f.blob_id) for f in files])
        config = validate_config(config, task)
        if task == "image-classification":
            if onnx_classifier.supports(config):
                selected_graph = onnx_classifier.configuration(config)
                if selected_graph not in graphs:
                    raise ValueError("ONNX_CLASSIFIER_GRAPH_UNAVAILABLE")
                graphs = [selected_graph]
                assets = {"config.json"} & sizes.keys()
                onnx_classifier.labels()
            else:
                from oryn_engine.onnx_vision import configuration

                selected_graph, labels_name = configuration(config)
                if selected_graph not in graphs or labels_name not in sizes:
                    raise ValueError("ONNX_VISION_METADATA_UNAVAILABLE")
                graphs = [selected_graph]
                assets = {"config.json", labels_name}
                if access:
                    vision = store._hub.read_vision_labels(repository, revision, labels_name)
        if task == "feature-extraction" and "1_Pooling/config.json" in assets and access:
            validate_pooling(
                store._hub.read_json(repository, revision, "1_Pooling/config.json"), config
            )
    except ValueError as exc:
        base_reasons.append(str(exc) if str(exc).startswith("ONNX_") else "MODEL_METADATA_INVALID")
    except (OSError, AttributeError, KeyError, httpx.HTTPError):
        base_reasons.append("ONNX_METADATA_UNAVAILABLE")
    if task == "audio-to-audio":
        assets = set()
    if task not in {"voice-activity-detection", "audio-to-audio", "image-classification"} and (
        "config.json" not in assets or not ({"tokenizer.json", "vocab.txt", "vocab.json"} & assets)
    ):
        base_reasons.append("MODEL_TOKENIZER_MISSING")
    whisper = task == "automatic-speech-recognition" and config.get("model_type") == "whisper"
    tdt = onnx_tdt.supports(config, task)
    variants = [(name,) for name in graphs]
    if whisper:
        from oryn_engine.onnx_whisper import graph_pairs

        assets.update(
            {"generation_config.json", "normalizer.json", "added_tokens.json"} & sizes.keys()
        )
        variants = sorted(graph_pairs(graphs), key=lambda pair: (sum(sizes[n] for n in pair), pair))
        if not variants:
            base_reasons.append("ONNX_WHISPER_GRAPH_INCOMPLETE")
            variants = [(graphs[0],)]
    if tdt:
        assets = {"config.json", "vocab.txt"} & sizes.keys()
        variants = onnx_tdt.graph_groups(graphs)
        if not variants:
            base_reasons.append("ONNX_TDT_GRAPH_INCOMPLETE")
            variants = [(graphs[0],)]
    if len(variants) > 16:
        base_reasons.append("ONNX_VARIANT_LIMIT")
    if (
        task == "automatic-speech-recognition"
        and not tdt
        and "preprocessor_config.json" not in assets
    ):
        base_reasons.append("MODEL_PROCESSOR_MISSING")
    _policy, available_memory = store._runtime_planning_state_provider()
    last = None
    for names in variants[:16]:
        reasons = list(base_reasons)
        selected, graph = assets | set(names), None
        if access and not reasons:
            try:
                members = [
                    store._hub.read_onnx_metadata(repository, revision, name, sizes)
                    for name in names
                ]
                graph = (
                    onnx_tdt.compose(members)
                    if tdt
                    else {
                        "encoder": members[0],
                        "decoder": members[1],
                        "operators": sorted(
                            set(members[0]["operators"]) | set(members[1]["operators"])
                        ),
                    }
                    if whisper
                    else members[0]
                )
                validate_graph(graph, task, config, vision=vision)
                for member in members:
                    selected.update(member["externalFiles"])
            except (ValueError, OSError, AttributeError, KeyError, httpx.HTTPError) as exc:
                reason = str(exc)
                reasons.append(
                    reason
                    if reason.startswith("ONNX_")
                    and len(reason) < 128
                    and reason.replace("_", "").isalnum()
                    else "ONNX_METADATA_UNAVAILABLE"
                )
        if any(not safe_path(n) or n not in sizes for n in selected):
            reasons.append("MODEL_SUBSET_INCOMPLETE")
        picked = [f for f in files if f.name in selected]
        size = sum(f.size for f in picked)
        memory = size * 3 + 512 * 1024**2
        descriptor = ModelDescriptor(
            "huggingface",
            repository,
            revision,
            "onnx",
            (
                onnx_classifier.ARCHITECTURE
                if task == "image-classification" and onnx_classifier.supports(config)
                else "camie-tagger"
                if task == "image-classification"
                else "silero-vad-state128"
                if task == "voice-activity-detection"
                else "dfn3-48000-200"
                if task == "audio-to-audio"
                else str(config.get("model_type", "unknown")),
            ),
            task,
            ("onnx",)
            if task == "image-classification" and onnx_classifier.supports(config)
            else ("onnx", "json")
            if task == "image-classification"
            else ("onnx",)
            if task in {"voice-activity-detection", "audio-to-audio"}
            else ("onnx", "json", "tokenizer"),
            canonical_digest([(f.name, f.size, f.blob_id) for f in picked]),
            bool(config.get("auto_map")),
            "available" if access else "authorization-required",
            memory,
        )
        for backend in store._available_pack_backends(pack.runner("onnx.encoder").backends):
            runtime_reasons = tuple(
                reasons
                + ([] if available else ["RUNTIME_DEPENDENCY_UNAVAILABLE"])
                + (["ONNX_COREML_FULL_GRAPH_UNQUALIFIED"] if backend == "coreml" else [])
            )
            eligibility = evaluate_eligibility(
                descriptor,
                store._capability_registry,
                resource_admissible=None
                if available_memory is None
                else memory <= available_memory,
                runtime_reasons=runtime_reasons,
            )
            identity = RuntimeIdentity(
                foundation.foundation_id,
                foundation.version,
                adapter.adapter_id,
                adapter.version,
                variant_id(names, task),
                backend,
                "fp32",
                "int4"
                if graph and "com.microsoft::MatMulNBits" in graph["operators"]
                else "int8"
                if graph and any("Integer" in op or "Quantize" in op for op in graph["operators"])
                else None,
            )
            candidate = RuntimeCandidate(
                identity,
                f"huggingface:{repository}@{revision}",
                "exact-model",
                TASK_CONTRACTS[task],
                memory,
                size,
                "untested",
                False,
                False,
                available,
                True,
                foundation.abi == adapter.foundation_abi,
                not reasons,
                not reasons,
                access,
                True,
                not runtime_reasons,
                integrity_digest=canonical_digest(
                    {
                        "files": descriptor.selected_files_digest,
                        "pack": pack.digest,
                        "capabilities": store._capability_registry.digest,
                        "graph": graph,
                    }
                ),
                performance_score=49 if backend == "coreml" else 50,
            )
            profile = {
                "model": {
                    "provider": "huggingface",
                    "repository": repository,
                    "revision": revision,
                },
                "exactRevision": revision,
                "executable": eligibility.disposition == "trust-required",
                "executableBackends": [backend]
                if eligibility.disposition == "trust-required"
                else [],
                "requiresCustomCode": bool(config.get("auto_map")),
                "estimatedDiskBytes": size,
                "estimatedMemoryBytes": memory,
                "selectedFiles": [{"name": f.name, "sizeBytes": f.size} for f in picked],
                "technicalCapabilities": {
                    "task": task,
                    "architectures": [descriptor.architectures[0]],
                    "inputModalities": ["image"]
                    if task == "image-classification"
                    else ["audio"]
                    if task
                    in {
                        "voice-activity-detection",
                        "automatic-speech-recognition",
                        "audio-to-audio",
                    }
                    else ["text"],
                    "outputModalities": ["embedding"]
                    if task == "feature-extraction"
                    else ["text"]
                    if task == "automatic-speech-recognition"
                    else ["audio"]
                    if task == "audio-to-audio"
                    else ["structured"],
                    "capabilities": {},
                },
                "runner": "runtime-pack",
                "descriptorSchemaVersion": "oryn-model-descriptor/1",
                "capabilityRegistryDigest": store._capability_registry.digest,
                "eligibility": eligibility.protocol_value(),
                "loadDisposition": eligibility.disposition,
                "runtimePack": {
                    "packId": pack.pack_id,
                    "packVersion": pack.version,
                    "runnerId": "onnx.encoder",
                    "taskContract": TASK_CONTRACTS[task],
                    "compatibilityLevel": "native-declarative",
                    "manifestDigest": pack.digest,
                },
                "blockingReasons": sorted(
                    {reason for axis in eligibility.axes.values() for reason in axis.reasons}
                ),
            }
            offers.append((candidate, profile, None))
            last = profile
    return last
