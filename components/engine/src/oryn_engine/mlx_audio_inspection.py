"""MLX-Audio offers in the common model discovery and runtime planning loop."""

from __future__ import annotations

import httpx

from oryn_engine import mlx_audio_tasks
from oryn_engine.model_inspection import inspect_descriptor
from oryn_engine.runtime_planning import RuntimeCandidate, RuntimeIdentity
from oryn_engine.runtime_routing import canonical_digest


def inspect_mlx_audio(store, repository, revision, files, config, task, access, offers):
    task = mlx_audio_tasks.effective_task(config, task)
    binding = mlx_audio_tasks.capability(config, task)
    if binding is None:
        return None
    reasons = []
    codec = None
    if access:
        try:
            config = store._hub.read_json(repository, revision, "config.json")
            mlx_audio_tasks.validate_config(config, task)
            binding = mlx_audio_tasks.capability(config, task)
            codec_name = mlx_audio_tasks.QWEN3_TTS_CODEC_CONFIG
            if binding["family"] == "qwen3_tts" and any(f.name == codec_name for f in files):
                codec = store._hub.read_json(repository, revision, codec_name)
                mlx_audio_tasks.validate_qwen3_tts_codec(codec)
        except ValueError:
            reasons.append("MODEL_METADATA_INVALID")
        except (OSError, AttributeError, KeyError, httpx.HTTPError):
            reasons.append("MODEL_METADATA_UNAVAILABLE")
    pack = store._runtime_packs.active_pack(mlx_audio_tasks.PACK_ID)
    adapter = store._runtime_foundations.adapter_for_legacy_pack(pack.pack_id)
    foundation = store._runtime_foundations.active_foundation(adapter.foundation_id)
    available = store._pack_runtime_locator.resolve(pack) is not None
    from oryn_engine.model_layouts import assess_layout

    layout = assess_layout([f.name for f in files], config, task, source_kind="mlx-audio")
    picked = [f for f in files if f.name in layout.selected_files]
    size = sum(f.size for f in picked)
    memory = int(size * 1.35) + 1024**3
    hardware = store._hardware_info()
    if not hardware.unified_memory:
        reasons.append("BACKEND_UNAVAILABLE")
    if not available:
        reasons.append("RUNTIME_DEPENDENCY_UNAVAILABLE")
    _, available_memory = store._runtime_planning_state_provider()
    assessment = inspect_descriptor(
        repository=repository,
        revision=revision,
        files=[f.name for f in picked],
        config=config,
        task=task,
        registry=store._capability_registry,
        access="available" if access else "authorization-required",
        source_kind="mlx-audio",
        file_identities=[(f.name, f.size, f.blob_id) for f in picked],
        estimated_memory=memory,
        available_memory=available_memory,
        artifact_reasons=layout.artifact_reasons,
        runtime_reasons=tuple(reasons),
    )
    eligibility = assessment.eligibility
    quantization = config.get("quantization", {})
    bits = quantization.get("bits") if isinstance(quantization, dict) else None
    contract = mlx_audio_tasks.TASK_CONTRACTS[task]
    candidate = RuntimeCandidate(
        RuntimeIdentity(
            foundation.foundation_id,
            foundation.version,
            adapter.adapter_id,
            adapter.version,
            "oryn.generated.mlx-audio",
            "mlx",
            "quantized" if quantization else "native",
            f"int{bits}" if type(bits) is int and bits in {3, 4, 5, 6, 8} else None,
        ),
        f"huggingface:{repository}@{revision}",
        "exact-model",
        contract,
        memory,
        size,
        "untested",
        False,
        False,
        available,
        True,
        foundation.abi == adapter.foundation_abi,
        not reasons,
        not layout.artifact_reasons,
        access,
        True,
        not reasons,
        integrity_digest=canonical_digest(
            {
                "files": assessment.descriptor.selected_files_digest,
                "foundation": foundation.digest,
                "pack": pack.digest,
                "config": config,
                **({"codec": codec} if codec is not None else {}),
                "capabilities": store._capability_registry.digest,
            }
        ),
    )
    speech = task == "text-to-speech"
    reference = bool(binding and "ref_audio" in binding["parameters"])
    profile = {
        "model": {"provider": "huggingface", "repository": repository, "revision": revision},
        "exactRevision": revision,
        "executable": eligibility.disposition == "trust-required",
        "executableBackends": ["mlx"] if eligibility.disposition == "trust-required" else [],
        "requiresCustomCode": bool(config.get("auto_map")),
        "estimatedDiskBytes": size,
        "estimatedMemoryBytes": memory,
        "selectedFiles": [{"name": f.name, "sizeBytes": f.size} for f in picked],
        "technicalCapabilities": {
            "task": task,
            "architectures": [config.get("model_type") or config.get("architecture") or "unknown"],
            "inputModalities": ["text", "audio"]
            if speech and reference
            else ["text"]
            if speech
            else ["audio"],
            "outputModalities": ["audio"] if speech else ["text"],
            "capabilities": {
                "speechSynthesis" if speech else "speechRecognition": True,
                **({"audioOutput": True, "referenceAudio": reference} if speech else {}),
                "streaming": False,
            },
        },
        "runner": "runtime-pack",
        "descriptorSchemaVersion": "oryn-model-descriptor/1",
        "capabilityRegistryDigest": store._capability_registry.digest,
        "eligibility": eligibility.protocol_value(),
        "loadDisposition": eligibility.disposition,
        "runtimePack": {
            "packId": pack.pack_id,
            "packVersion": pack.version,
            "runnerId": mlx_audio_tasks.RUNNER_ID,
            "taskContract": contract,
            "compatibilityLevel": "native-declarative",
            "manifestDigest": pack.digest,
        },
        "blockingReasons": sorted({r for axis in eligibility.axes.values() for r in axis.reasons}),
    }
    offers.append((candidate, profile, None))
    return profile
