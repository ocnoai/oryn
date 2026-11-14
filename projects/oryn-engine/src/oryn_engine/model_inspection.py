"""The shared static admission boundary for live inspection and frozen audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oryn_engine.compatibility_plane import (
    CapabilityRegistry,
    EligibilityResult,
    ModelDescriptor,
    evaluate_eligibility,
)
from oryn_engine.model_layouts import LayoutAssessment, assess_layout
from oryn_engine.runtime_routing import canonical_digest


@dataclass(frozen=True)
class InspectionAssessment:
    descriptor: ModelDescriptor
    layout: LayoutAssessment
    eligibility: EligibilityResult


def effective_task(source_kind: str, task: str | None, config: dict) -> str | None:
    if (
        source_kind == "transformers"
        and task == "keypoint-detection"
        and config.get("model_type") in {"vitpose", "sapiens2"}
    ):
        return "pose-estimation"
    if task == "sentence-similarity":
        return "feature-extraction"
    if (
        source_kind == "mlx"
        and task == "image-text-to-text"
        and config.get("model_type") == "qwen3_5_moe"
        and not config.get("vision_config")
    ):
        # A text-only export cannot acquire image support from its family name.
        return "text-generation"
    return task


def inspect_descriptor(
    *,
    repository: str,
    revision: str,
    files: list[str],
    config: dict[str, Any],
    task: str | None,
    registry: CapabilityRegistry,
    access: str,
    source_kind: str = "transformers",
    file_identities: object = None,
    estimated_memory: int = 0,
    available_memory: int | None = None,
    artifact_reasons: tuple[str, ...] = (),
    runtime_reasons: tuple[str, ...] = (),
) -> InspectionAssessment:
    from oryn_engine.model_metadata import validate_model_config

    try:
        if source_kind == "diffusers":
            from oryn_engine.flat_ddpm import normalized_index

            config = normalized_index(config)
        if source_kind == "mlx":
            from oryn_engine.mlx_bolding import normalized_config

            config = normalized_config(config, task)
        validate_model_config(config)
    except (ValueError, OverflowError):
        runtime_reasons = (*runtime_reasons, "MODEL_METADATA_INVALID")
        config = {}  # Invalid data cannot reach native layout/class selectors.
    task = effective_task(source_kind, task, config)
    if source_kind == "transformers":
        from oryn_engine.generated_capabilities import normalized_transformers_config
        from oryn_engine.timm_tasks import normalized_config

        try:
            config = normalized_transformers_config(normalized_config(config, task), task)
        except (ValueError, TypeError):
            runtime_reasons = (*runtime_reasons, "TASK_CONFIGURATION_UNSUPPORTED")
    if source_kind == "mlx":
        from oryn_engine import mlx_bolding, mlx_kolibri, mlx_prism
        from oryn_engine.generated_capabilities import mlx_model_types
        from oryn_engine.runtime_packs.mlx_lm_worker import validate_config

        try:
            if mlx_kolibri.supports(config):
                mlx_kolibri.validate_config(config)
                if task not in {"text-generation", "conversational", None}:
                    raise ValueError("KOLIBRI_TASK")
            elif mlx_bolding.supports(config):
                mlx_bolding.normalized_config(config, task)
            elif mlx_prism.supports(config) and task in {"text-generation", "conversational", None}:
                mlx_prism.validate_config(config)
            elif task == "voice-activity-detection":
                from oryn_engine.mlx_vad import validate_config as validate_vad_config

                validate_vad_config(config)
            elif (
                task == "any-to-any"
                or task == "image-text-to-text"
                and config.get("model_type") == "gemma4"
            ):
                from oryn_engine.mlx_gemma4 import validate_config as validate_gemma4

                validate_gemma4(config)
            elif task == "image-text-to-text":
                from oryn_engine.mlx_vision import validate_vision_config

                validate_vision_config(config)
            else:
                validate_config(config, mlx_model_types())
                if task in {"text-generation", "conversational", None} and config.get(
                    "model_type"
                ) not in {"gpt_oss", "gemma4_unified"}:
                    from oryn_engine import chat_formats

                    # The generic MLX chat route frames prompts only from the reviewed table.
                    template = chat_formats.template_from_config(config)
                    if template and chat_formats.select(template) is None:
                        runtime_reasons = (*runtime_reasons, "CHAT_FORMAT_UNREVIEWED")
        except (ValueError, TypeError):
            runtime_reasons = (*runtime_reasons, "MODEL_RUNTIME_INCOMPATIBLE")
    task = effective_task(source_kind, task, config)
    layout = assess_layout(files, config, task, source_kind=source_kind)
    names = layout.selected_files
    formats = set()
    if source_kind == "transformers" and "pytorch_model.bin" in names:
        from oryn_engine.numeric_archive_tasks import selected

        if selected(config, task, names):
            formats.add("torch-numeric-zip")
    if source_kind == "diffusers":
        from oryn_engine.numeric_unet2d import selected as numeric_unet2d_selected

        if numeric_unet2d_selected(config, names):
            formats.add("torch-numeric-zip")
    for suffix, label in (
        (".safetensors", "safetensors"),
        (".json", "json"),
        (".gguf", "gguf"),
        (".onnx", "onnx"),
        (".msgpack", "flax-msgpack"),
        (".h5", "hdf5-weights"),
    ):
        if any(name.endswith(suffix) for name in names):
            formats.add(label)
    if any(
        "tokenizer" in name
        or "vocab" in name
        or name.endswith((".model", ".spm"))
        or name == "tekken.json"
        and config.get("model_type") == "voxtral"
        for name in names
    ):
        formats.add("tokenizer")
    if config.get("model_type") == "oryn_ct2_whisper" and "model.bin" in names:
        formats.add("ctranslate2")
    if {"model_index.json", "modular_model_index.json"} & set(names):
        formats.add("diffusers")
    architectures = config.get("architectures") or ["unknown"]
    if source_kind == "diffusers":
        architectures = [config.get("_class_name", "unknown")]
    elif source_kind == "mlx":
        architectures = [config.get("model_type", "unknown")]
    elif source_kind == "transformers" and task == "keypoint-detection":
        # The generated Auto mapping binds this task by native model_type; older
        # checkpoints retain the upstream class name from before its rename.
        architectures = [config.get("model_type", "unknown")]
    descriptor = ModelDescriptor(
        "huggingface",
        repository,
        revision,
        source_kind,
        tuple(architectures),
        task,
        tuple(sorted(formats)),
        canonical_digest(file_identities if file_identities is not None else names),
        bool(config.get("auto_map") or config.get("custom_pipelines")),
        access,
        estimated_memory,
    )
    eligibility = evaluate_eligibility(
        descriptor,
        registry,
        resource_admissible=None
        if available_memory is None
        else 0 < estimated_memory <= available_memory,
        artifact_reasons=tuple(sorted(set((*artifact_reasons, *layout.artifact_reasons)))),
        runtime_reasons=tuple(sorted(set((*runtime_reasons, *layout.runtime_reasons)))),
    )
    return InspectionAssessment(descriptor, layout, eligibility)
