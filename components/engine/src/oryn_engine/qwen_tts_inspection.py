"""Qwen TTS metadata offers using the common planner and trust boundary."""

from __future__ import annotations

import httpx

from oryn_engine import qwen_tts
from oryn_engine.compatibility_plane import ModelDescriptor, evaluate_eligibility
from oryn_engine.runtime_planning import RuntimeCandidate, RuntimeIdentity
from oryn_engine.runtime_routing import canonical_digest


def inspect_qwen_tts(store, repository, revision, files, config, task, access, offers):
    if not qwen_tts.supports(config, task):
        return None
    pack = store._runtime_packs.active_pack(qwen_tts.PACK_ID)
    adapter = store._runtime_foundations.adapter_for_legacy_pack(pack.pack_id)
    foundation = store._runtime_foundations.active_foundation(adapter.foundation_id)
    available = store._pack_runtime_locator.resolve(pack) is not None
    names = {f.name for f in files}
    reasons, codec, controls = [], None, {}
    if not names >= qwen_tts.REQUIRED_FILES:
        reasons.append("MODEL_SUBSET_INCOMPLETE")
    if access and not reasons:
        try:
            config = store._hub.read_json(repository, revision, "config.json")
            codec = store._hub.read_json(repository, revision, "speech_tokenizer/config.json")
            tokenizer = store._hub.read_json(repository, revision, "tokenizer_config.json")
            qwen_tts.validate_config(config, codec)
            qwen_tts.validate_tokenizer_metadata(tokenizer)
            controls = qwen_tts.controls(config)
        except ValueError:
            reasons.append("MODEL_METADATA_INVALID")
        except (OSError, AttributeError, KeyError, httpx.HTTPError):
            reasons.append("MODEL_METADATA_UNAVAILABLE")
    picked = [f for f in files if f.name in qwen_tts.REQUIRED_FILES]
    size = sum(f.size for f in picked)
    memory = size * 3 + 1024**3
    _, available_memory = store._runtime_planning_state_provider()
    descriptor = ModelDescriptor(
        "huggingface",
        repository,
        revision,
        "qwen-tts",
        (qwen_tts.ARCHITECTURE,),
        task,
        ("safetensors", "json", "tokenizer"),
        canonical_digest([(f.name, f.size, f.blob_id) for f in picked]),
        bool(config.get("auto_map")),
        "available" if access else "authorization-required",
        memory,
    )
    last = None
    for backend in store._available_pack_backends(
        pack.runner("qwen-tts.speech-synthesis").backends
    ):
        runtime_reasons = tuple(reasons + ([] if available else ["RUNTIME_DEPENDENCY_UNAVAILABLE"]))
        eligibility = evaluate_eligibility(
            descriptor,
            store._capability_registry,
            resource_admissible=None if available_memory is None else memory <= available_memory,
            runtime_reasons=runtime_reasons,
        )
        identity = RuntimeIdentity(
            foundation.foundation_id,
            foundation.version,
            adapter.adapter_id,
            adapter.version,
            "qwen-tts.custom-voice-12hz",
            backend,
            "fp32",
            None,
        )
        candidate = RuntimeCandidate(
            identity,
            f"huggingface:{repository}@{revision}",
            "exact-model",
            "speech-synthesis-v1",
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
                    "config": config,
                    "codec": codec,
                }
            ),
        )
        profile = {
            "model": {"provider": "huggingface", "repository": repository, "revision": revision},
            "exactRevision": revision,
            "executable": eligibility.disposition == "trust-required",
            "executableBackends": [backend] if eligibility.disposition == "trust-required" else [],
            "requiresCustomCode": bool(config.get("auto_map")),
            "estimatedDiskBytes": size,
            "estimatedMemoryBytes": memory,
            "selectedFiles": [{"name": f.name, "sizeBytes": f.size} for f in picked],
            "technicalCapabilities": {
                "task": task,
                "architectures": [qwen_tts.ARCHITECTURE],
                "inputModalities": ["text"],
                "outputModalities": ["audio"],
                "capabilities": {"speechSynthesis": True, "audioOutput": True},
            },
            "runner": "runtime-pack",
            "descriptorSchemaVersion": "oryn-model-descriptor/1",
            "capabilityRegistryDigest": store._capability_registry.digest,
            "eligibility": eligibility.protocol_value(),
            "loadDisposition": eligibility.disposition,
            "runtimePack": {
                "packId": pack.pack_id,
                "packVersion": pack.version,
                "runnerId": "qwen-tts.speech-synthesis",
                "taskContract": "speech-synthesis-v1",
                "compatibilityLevel": "native-declarative",
                "manifestDigest": pack.digest,
                "trustedMetadata": controls,
            },
            "blockingReasons": sorted(
                {r for axis in eligibility.axes.values() for r in axis.reasons}
            ),
        }
        offers.append((candidate, profile, None))
        last = profile
    return last
