"""Closed reusable task adapters over the existing packaged Transformers foundation.

No repository IDs, import strings, execution options or custom code are accepted.
The same contracts are used for inspection, Browser validation and isolated execution.
"""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from oryn_engine import (
    audio8_tasks,
    audio_understanding,
    bidirectional_embeddings,
    breeze_tasks,
    coverage_tasks,
    ct2_tasks,
    ctc_tasks,
    decision_tasks,
    flax_weights,
    forecast_tasks,
    gemma4_tasks,
    jina_ranking,
    layoutlm_tasks,
    llava_video_qwen,
    molparser_tasks,
    mpsenet_tasks,
    numeric_archive_tasks,
    omnivoice_tasks,
    parakeet_tasks,
    pii_tasks,
    pose_tasks,
    qwen_codec_tasks,
    qwen_retrieval,
    reuse_tasks,
    rnnt_tasks,
    sopro_tasks,
    speechx_tasks,
    tabular_tasks,
    tiger_tasks,
    trocr_tasks,
    videollama3_tasks,
    wav2vec_heads,
)
from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

TASK_CONTRACTS = {
    **numeric_archive_tasks.TASK_CONTRACTS,
    "audio-to-audio": "audio-transformation-v1",
    "text-to-speech": "speech-synthesis-v1",
    "audio-classification": "audio-classification-v1",
    **qwen_retrieval.TASK_CONTRACTS,
    **flax_weights.TASK_CONTRACTS,
    **gemma4_tasks.TASK_CONTRACTS,
    **tabular_tasks.TASK_CONTRACTS,
    **coverage_tasks.TASK_CONTRACTS,
    **forecast_tasks.TASK_CONTRACTS,
    **audio_understanding.TASK_CONTRACTS,
    **ctc_tasks.TASK_CONTRACTS,
    **pose_tasks.TASK_CONTRACTS,
    "zero-shot-classification": "zero-shot-classification-v1",
    "image-feature-extraction": "image-embeddings-v1",
    "image-segmentation": "image-segmentation-v1",
    "mask-generation": "mask-generation-v1",
    "zero-shot-object-detection": "zero-shot-object-detection-v1",
    "image-to-text": "image-to-text-v1",
    "visual-question-answering": "visual-question-answering-v1",
    "image-classification": "image-classification-v1",
    "question-answering": "question-answering-v1",
    "translation": "translation-v1",
    "token-classification": "token-classification-v1",
    "depth-estimation": "depth-estimation-v1",
}
# Reusing a shared executor does not turn a base protocol task into an extension.
# These contracts were available before additive task negotiation shipped.
NEGOTIATED_TASK_CONTRACTS = (frozenset(TASK_CONTRACTS.values()) | {"chat-v2"}) - {
    "fill-mask-v1",
    "audio-classification-v1",
    *qwen_retrieval.TASK_CONTRACTS.values(),
    "transcription-v1",
    "image-classification-v1",
    "question-answering-v1",
    "translation-v1",
    "token-classification-v1",
    "depth-estimation-v1",
    *flax_weights.TASK_CONTRACTS.values(),
}
VIDEO_TASKS = coverage_tasks.VIDEO_TASKS
AUDIO_TASKS = frozenset(
    {
        "audio-classification",
        "text-to-speech",
        "audio-to-audio",
        *audio_understanding.TASK_CONTRACTS,
        *ctc_tasks.TASK_CONTRACTS,
    }
)
MULTIMODAL_TASKS = frozenset(gemma4_tasks.TASK_CONTRACTS)
IMAGE_TASKS = frozenset(TASK_CONTRACTS) - {
    *qwen_retrieval.TASK_CONTRACTS,
    *flax_weights.TEXT_TASKS,
    "fill-mask",
    *MULTIMODAL_TASKS,
    *tabular_tasks.TASK_CONTRACTS,
    "zero-shot-classification",
    "table-question-answering",
    "question-answering",
    "translation",
    "token-classification",
    "time-series-forecasting",
    *VIDEO_TASKS,
    *AUDIO_TASKS,
}
ARTIFACT_TASKS = frozenset({"image-segmentation", "mask-generation", "text-to-speech"})
AUDIO_OUTPUT_TASKS = frozenset({"text-to-speech", "audio-to-audio"})
# These are sealed task behaviors, not per-model architecture lists. The build generator
# derives architecture identities from the corresponding pinned Auto mappings.
TASK_FAMILIES = {
    "audio-to-audio": {
        qwen_codec_tasks.FAMILY,
        mpsenet_tasks.FAMILY,
        reuse_tasks.FAMILY,
        tiger_tasks.FAMILY,
    },
    "text-to-speech": {sopro_tasks.FAMILY, omnivoice_tasks.FAMILY, breeze_tasks.FAMILY},
    "audio-classification": wav2vec_heads.FAMILIES | {speechx_tasks.FAMILY},
    **qwen_retrieval.TASK_FAMILIES,
    "feature-extraction": {"qwen3_vl", "ministral3", "qwen3_bidirec", "pplx_contextual_qwen3_5"},
    **flax_weights.TASK_FAMILIES,
    "text-classification": {"bert", "roberta", "oryn_llama_turn"},
    "token-classification": {"pii_masking"},
    "depth-estimation": {"depth_anything_3"},
    "question-answering": {"electra"},
    "translation": {"marian", "qwen3_5", "qwen3_5_text", "hunyuan_v1_dense"},
    **gemma4_tasks.TASK_FAMILIES,
    **tabular_tasks.TASK_FAMILIES,
    **coverage_tasks.TASK_FAMILIES,
    **forecast_tasks.TASK_FAMILIES,
    **audio_understanding.TASK_FAMILIES,
    **ctc_tasks.TASK_FAMILIES,
    "automatic-speech-recognition": {
        "wav2vec2",
        audio8_tasks.FAMILY,
        rnnt_tasks.FAMILY,
        parakeet_tasks.FAMILY,
        ct2_tasks.FAMILY,
    },
    **pose_tasks.TASK_FAMILIES,
    "image-feature-extraction": {
        "dinov2",
        "dinov2_with_registers",
        "dinov3_vit",
        "vit",
        "timm_wrapper",
    },
    "image-classification": {"timm_wrapper"},
    "image-segmentation": {"segformer", "mask2former"},
    "mask-generation": {"sam", "sam_hq", "sam2_video"},
    "zero-shot-object-detection": {
        "grounding-dino",
        "mm-grounding-dino",
        "owlvit",
        "owlv2",
        "omdet-turbo",
    },
    "image-to-text": {
        molparser_tasks.FAMILY,
        "llava_next",
        "vision-encoder-decoder",
        "glm_ocr",
        "blip",
        "blip-2",
        "pix2struct",
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_5",
        "llava",
    },
    "visual-question-answering": {
        "llava_next",
        "blip",
        "blip-2",
        "vilt",
        "pix2struct",
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_5",
        "llava",
    },
}
MAX_IMAGE_BYTES = 20 * 1024**2
MAX_RESULT_BYTES = 256 * 1024
QWEN_VISION_FAMILIES = frozenset({"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_5"})


def validate_processor_config(config: dict[str, Any], family: str) -> None:
    if family == speechx_tasks.FAMILY:
        speechx_tasks.validate_processor(config)
        return
    if family in wav2vec_heads.FAMILIES:
        wav2vec_heads.validate_processor(config, family)
        return
    if family == "oryn_pixai":
        from oryn_engine.pixai_tasks import validate_processor

        validate_processor(config)
        return
    if family == "oryn_isnet":
        from oryn_engine.isnet_tasks import validate_processor

        validate_processor(config)
        return
    if family == "voxtral" and "feature_extractor_type" in config:
        from oryn_engine.voxtral_tasks import processor_options

        processor_options(config)
        return
    if family in audio_understanding.FLAMINGO_FAMILIES and "feature_extractor" in config:
        audio_understanding.flamingo_processor_options(config, family)
        return
    if family == "colqwen2":
        family = "qwen2_vl"
    if family not in QWEN_VISION_FAMILIES | {"glm_ocr"}:
        validate_model_config(config)
        return
    # Qwen calls pixel AREAS shortest/longest_edge. These are not side lengths.
    # Repository budgets are never used: the sealed constructor below fixes them.
    bounded = deepcopy(config)
    stack = [bounded]
    identities = {
        "Qwen2VLImageProcessor",
        "Qwen2VLImageProcessorFast",
        "Qwen2VLVideoProcessor",
        "Qwen2_5VLVideoProcessor",
        "Qwen3VLVideoProcessor",
    }
    if family == "glm_ocr":
        identities = {"Glm46VImageProcessor", "Glm46VVideoProcessor"}
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            identity = value.get("image_processor_type", value.get("video_processor_type"))
            if identity is not None:
                if not isinstance(identity, str) or identity not in identities:
                    raise ValueError("Qwen processor identity is not admitted")
                for key in ("size", "min_pixels", "max_pixels"):
                    if key in value:
                        raw = value[key]
                        if key in {"min_pixels", "max_pixels"} and raw is None:
                            # Native fast processors serialize unset aliases as
                            # null; the declared size and sealed runtime budget
                            # remain independently validated.
                            del value[key]
                            continue
                        if (
                            key == "size"
                            and isinstance(raw, dict)
                            and set(raw) == {"min_pixels", "max_pixels"}
                        ):
                            if (
                                value.get("min_pixels", raw["min_pixels"]) != raw["min_pixels"]
                                or value.get("max_pixels", raw["max_pixels"]) != raw["max_pixels"]
                            ):
                                raise ValueError("Qwen pixel-area aliases disagree")
                            raw = {
                                "shortest_edge": raw["min_pixels"],
                                "longest_edge": raw["max_pixels"],
                            }
                        if (
                            key == "size"
                            and isinstance(raw, dict)
                            and set(raw)
                            == {"min_pixels", "max_pixels", "shortest_edge", "longest_edge"}
                        ):
                            if (
                                raw["min_pixels"] != raw["shortest_edge"]
                                or raw["max_pixels"] != raw["longest_edge"]
                                or value.get("min_pixels", raw["min_pixels"]) != raw["min_pixels"]
                                or value.get("max_pixels", raw["max_pixels"]) != raw["max_pixels"]
                            ):
                                raise ValueError("Qwen pixel-area aliases disagree")
                            raw = {k: raw[k] for k in ("shortest_edge", "longest_edge")}
                        axes = list(raw.values()) if isinstance(raw, dict) else [raw]
                        if (
                            isinstance(raw, dict) and set(raw) != {"shortest_edge", "longest_edge"}
                        ) or any(
                            type(axis) is not int or not 1 <= axis <= 64 * 1024**2 for axis in axes
                        ):
                            raise ValueError("Qwen pixel-area metadata is invalid")
                        del value[key]
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    validate_model_config(bounded)


def qwen_image_options(
    config: dict[str, Any], vision: dict[str, Any], *, video_only: bool = False
) -> dict[str, Any]:
    image_config = config.get("image_processor", config)
    if isinstance(image_config, dict):
        image_config = dict(image_config)
        # Inactive generic ImageProcessingMixin fields are not constructor
        # options for the sealed Qwen processor. Active overrides still fail.
        inactive = {
            "crop_size": None,
            "device": None,
            "disable_grouping": None,
            "do_center_crop": None,
            "do_pad": None,
            "input_data_format": None,
            "pad_size": None,
            "return_tensors": None,
            "default_to_square": True,
        }
        for key, default in inactive.items():
            if key in image_config and image_config[key] is default:
                del image_config[key]
    allowed = {
        "image_processor_type",
        "processor_class",
        "data_format",
        "do_convert_rgb",
        "do_normalize",
        "do_rescale",
        "do_resize",
        "image_mean",
        "image_std",
        "size",
        "max_pixels",
        "min_pixels",
        "merge_size",
        "patch_size",
        "temporal_patch_size",
        "resample",
        "rescale_factor",
    }
    if not isinstance(image_config, dict) or not set(image_config) <= allowed:
        raise ValueError("Qwen image preprocessing options are not admitted")
    for key, expected in (
        ("do_convert_rgb", True),
        ("do_normalize", True),
        ("do_rescale", True),
        ("resample", 3),
        ("rescale_factor", 1 / 255),
        ("data_format", "channels_first"),
    ):
        if key in image_config and image_config[key] != expected:
            raise ValueError("Qwen image preprocessing behavior is unsupported")
    resize = image_config.get("do_resize", True)
    if type(resize) is not bool or not resize and not video_only:
        raise ValueError("Qwen image preprocessing behavior is unsupported")
    patch = vision.get("patch_size", 14)
    merge = vision.get("spatial_merge_size", 2)
    temporal = vision.get("temporal_patch_size", 2)
    if type(patch) is not int or patch not in {14, 16} or merge != 2 or temporal != 2:
        raise ValueError("Qwen image patch geometry is unsupported")
    options = {
        "patch_size": patch,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "size": {"shortest_edge": (patch * 2) ** 2, "longest_edge": 1024**2},
    }
    if not resize:
        # This processor is unused by the video-only contract. Preserve its
        # declaration without relaxing the image contracts' resize requirement.
        options["do_resize"] = False
    for key, expected in (("patch_size", patch), ("merge_size", 2), ("temporal_patch_size", 2)):
        if key in image_config and image_config[key] != expected:
            raise ValueError("Image processor/model patch geometry disagrees")
    import math

    for key in ("image_mean", "image_std"):
        values = image_config.get(key)
        if values is not None:
            if (
                not isinstance(values, list)
                or len(values) != 3
                or any(
                    type(v) not in {int, float}
                    or not math.isfinite(v)
                    or not (0 < v <= 5 if key == "image_std" else -5 <= v <= 5)
                    for v in values
                )
            ):
                raise ValueError("Image normalization is invalid")
            options[key] = values
    return options


def qwen_video_options(
    config: dict[str, Any], vision: dict[str, Any], family: str
) -> dict[str, Any]:
    """Own sampling/resize budgets; preserve the declared native pixel transform."""
    import math

    validate_processor_config(config, family)
    expected = {
        "qwen2_vl": {"Qwen2VLVideoProcessor"},
        "qwen2_5_vl": {"Qwen2VLVideoProcessor", "Qwen2_5VLVideoProcessor"},
        "qwen3_vl": {"Qwen3VLVideoProcessor"},
        "qwen3_5": {"Qwen3VLVideoProcessor"},
    }
    if family not in expected or not isinstance(config, dict):
        raise ValueError("Qwen video processor family is unsupported")
    values = dict(config)
    identity = values.pop("video_processor_type", None)
    if config and identity not in expected[family]:
        raise ValueError("Qwen video processor identity disagrees")
    processor_class = {
        "qwen2_vl": "Qwen2VLProcessor",
        "qwen2_5_vl": "Qwen2_5_VLProcessor",
        "qwen3_vl": "Qwen3VLProcessor",
        "qwen3_5": "Qwen3VLProcessor",
    }[family]
    if values.get("processor_class", processor_class) != processor_class:
        raise ValueError("Qwen video processor class disagrees")
    for key in ("do_sample_frames", "return_metadata", "cap_pixels_per_frame"):
        value = values.pop(key, None)
        if value is not None and type(value) is not bool:
            raise ValueError("Qwen video sampling declaration is invalid")
    for key, maximum in (
        ("min_frames", 4096),
        ("max_frames", 4096),
        ("num_frames", 4096),
        ("max_video_tokens", 65536),
    ):
        value = values.pop(key, None)
        if value is not None and (type(value) is not int or not 1 <= value <= maximum):
            raise ValueError("Qwen video sampling declaration is invalid")
    if (config.get("min_frames") or 1) > (config.get("max_frames") or 4096):
        raise ValueError("Qwen video frame bounds disagree")
    fps = values.pop("fps", None)
    if fps is not None and (
        type(fps) not in {int, float} or not math.isfinite(fps) or not 0 < fps <= 120
    ):
        raise ValueError("Qwen video frame rate is invalid")
    if values.pop("video_metadata", None) is not None:
        raise ValueError("Repository video timestamps cannot replace input timestamps")
    options = qwen_image_options(values, vision, video_only=True)
    # A saved do_resize=False expects caller-resized frames. Our video path owns
    # that same native resize before normalization/patchification, with bounded
    # areas instead of the repository's potentially enormous inference budget.
    options.update(
        do_resize=True,
        size={"shortest_edge": 56**2, "longest_edge": 256 * 28**2},
        min_frames=8,
        max_frames=8,
        do_sample_frames=False,
        cap_pixels_per_frame=True,
        max_video_tokens=4096,
    )
    return options


@lru_cache(maxsize=1)
def _schemas():
    from oryn_protocol.schema import ProtocolSchemaStore

    return ProtocolSchemaStore()


def validate_task_input(task: str, value: object) -> dict[str, Any]:
    contract = TASK_CONTRACTS.get(task)
    if contract is None or not _schemas().is_valid(f"taskInput.{contract}", value):
        raise ValueError("Task input is invalid or unavailable")
    # JSON Schema numbers reject NaN inconsistently between bindings; serialization is strict.
    if len(json.dumps(value, allow_nan=False).encode()) > 128 * 1024:
        raise ValueError("Task input exceeds aggregate byte limit")
    result = dict(value)
    if task == "text-to-speech":
        result.setdefault("voice", "reference" if "audioArtifactId" in result else "auto")
        result.setdefault("speed", 1)
    if task in {"summarization", "translation"}:
        result.setdefault("maxTokens", 128)
    if task in {"text-classification", "fill-mask"}:
        result.setdefault("topK", 5)
    if task == "question-answering":
        result.setdefault("maxAnswerLength", 64)
    if "labels" in result:
        result["labels"] = [label.strip() for label in result["labels"]]
        if len(set(result["labels"])) != len(result["labels"]):
            raise ValueError("Candidate labels must be distinct")
    if task == "zero-shot-classification":
        result.setdefault("multiLabel", False)
    if task in {"image-feature-extraction", "feature-extraction"}:
        result.setdefault("normalize", True)
    if task == "feature-extraction" and len(result["texts"]) > qwen_retrieval.MAX_BATCH:
        raise ValueError("QWEN_RETRIEVAL_BATCH_LIMIT")
    if task == "text-ranking":
        from oryn_protocol.task_contracts import TASK_CONTRACTS as definitions

        result.setdefault(
            "instruction",
            definitions[contract]["inputSchema"]["properties"]["instruction"]["default"],
        )
    if task == "zero-shot-object-detection":
        result.setdefault("threshold", 0.3)
    if task in {"image-to-text", "visual-question-answering"}:
        result.setdefault("maxTokens", 128)
    if task == "mask-generation" and not any(p["label"] == 1 for p in result["points"]):
        raise ValueError("At least one positive point is required")
    coverage_tasks.validate_relations(task, result)
    from oryn_protocol.task_contracts import validate_input_relations

    validate_input_relations(contract, result)
    if task in MULTIMODAL_TASKS:
        gemma4_tasks.validate_input(result)
    return result


def result_byte_limit(task, result):
    if (
        task == "visual-document-retrieval"
        and isinstance(result, dict)
        and result.get("dimensions") == 4096
    ):
        from oryn_engine.structured_outputs import MAX_BYTES

        return MAX_BYTES
    return (
        900 * 1024
        if task in {"visual-document-retrieval", "feature-extraction", "pose-estimation"}
        else MAX_RESULT_BYTES
    )


def validate_task_output(task: str, result: object) -> dict[str, Any]:
    from oryn_protocol.task_contracts import TASK_CONTRACTS as definitions
    from oryn_protocol.task_contracts import validate_output_relations

    contract = TASK_CONTRACTS.get(task)
    validation_value = (
        {"kind": "structured", "value": result}
        if task in MULTIMODAL_TASKS or task == "audio-to-audio"
        else result
    )
    valid = (
        (contract is not None and _schemas().is_valid(f"taskOutput.{contract}", validation_value))
        if contract and definitions[contract].get("outputSchema")
        else _schemas().is_valid(f"taskResult.{contract}", {"kind": "structured", "value": result})
    )
    if not valid:
        raise ValueError("Task result violates its contract")
    if len(json.dumps(result, allow_nan=False).encode()) > result_byte_limit(task, result):
        raise ValueError("Task result exceeds aggregate byte limit")
    validate_output_relations(contract, validation_value)
    if task == "image-feature-extraction":
        if result["dimensions"] != len(result["vector"]):
            raise ValueError("Embedding dimension mismatch")
        if result["normalized"]:
            import math

            if not math.isclose(sum(x * x for x in result["vector"]), 1, abs_tol=1e-4):
                raise ValueError("Embedding normalization mismatch")
    if task == "zero-shot-classification":
        labels = [item["label"] for item in result["predictions"]]
        if len(set(labels)) != len(labels):
            raise ValueError("Duplicate prediction label")
    if task == "zero-shot-object-detection":
        for item in result["detections"]:
            x0, y0, x1, y1 = item["box"]
            if not 0 <= x0 <= x1 <= result["width"] or not 0 <= y0 <= y1 <= result["height"]:
                raise ValueError("Detection box exceeds image bounds")
    return result


def nli_label_ids(config: object) -> tuple[int, int]:
    labels = (
        config.get("id2label") if isinstance(config, dict) else getattr(config, "id2label", None)
    )
    if not isinstance(labels, dict) or not 2 <= len(labels) <= 3:
        raise ValueError("NLI label semantics are unavailable")
    normalized = {}
    for key, value in labels.items():
        if not isinstance(value, str) or str(key) not in {str(i) for i in range(len(labels))}:
            raise ValueError("NLI label mapping is invalid")
        label = value.lower().strip()
        if label in normalized:
            raise ValueError("NLI labels are ambiguous")
        normalized[label] = int(key)
    # Binary NLI checkpoints can explicitly name the negative class
    # not_entailment. It includes neutral examples, so this binding is only
    # valid for a two-class head, never as an alias in a three-class model.
    negative = (
        "not_entailment" if set(normalized) == {"entailment", "not_entailment"} else "contradiction"
    )
    if "entailment" not in normalized or negative not in normalized:
        raise ValueError("Explicit entailment and negative NLI labels are required")
    if len(normalized) == 3 and set(normalized) != {"entailment", "contradiction", "neutral"}:
        raise ValueError("Three-class NLI requires an explicit neutral label")
    return normalized["entailment"], normalized[negative]


def validate_task_config(task: str, config: dict[str, Any], *, numeric_archive=False) -> None:
    from oryn_engine import (
        birefnet_tasks,
        contextual_embeddings,
        isnet_tasks,
        openclip_tasks,
        turn_classification,
    )

    validate_model_config(config)
    if numeric_archive:
        numeric_archive_tasks.validate_config(config, task)
        return
    if turn_classification.supports(config, task):
        turn_classification.validate_config(config)
        return
    if layoutlm_tasks.supports(config, task):
        layoutlm_tasks.validate_config(config)
        return
    if llava_video_qwen.supports(config, task):
        llava_video_qwen.validate_config(config)
        return
    if videollama3_tasks.supports(config, task):
        videollama3_tasks.validate_config(config)
        return
    if tiger_tasks.supports(config, task):
        tiger_tasks.validate_config(config)
        return
    if reuse_tasks.supports(config, task):
        reuse_tasks.validate_config(config)
        return
    if mpsenet_tasks.supports(config, task):
        mpsenet_tasks.validate_config(config)
        return
    if qwen_codec_tasks.supports(config, task):
        qwen_codec_tasks.validate_config(config)
        return
    if molparser_tasks.supports(config, task):
        molparser_tasks.validate_config(config)
        return
    if breeze_tasks.supports(config, task):
        breeze_tasks.validate_config(config)
        return
    if omnivoice_tasks.supports(config, task):
        omnivoice_tasks.validate_config(config)
        return
    if sopro_tasks.supports(config, task):
        sopro_tasks.validate_config(config)
        return
    if speechx_tasks.supports(config, task):
        speechx_tasks.validate_config(config)
        return
    if wav2vec_heads.supports(config, task):
        wav2vec_heads.validate_config(config)
        return
    if config.get("quantization_config") or config.get("quantization"):
        raise ValueError("Quantized Transformers task loading is not admitted")
    if task == "keypoint-detection" and config.get("model_type") == "rtmw":
        from oryn_engine.rtmw_tasks import validate_config

        validate_config(config)
        return
    if task == "image-classification" and config.get("model_type") == "oryn_pixai":
        from oryn_engine.pixai_tasks import validate_config

        validate_config(config)
        return
    if task == "zero-shot-image-classification" and config.get("model_type") == "tipsv2":
        from oryn_engine.tips_tasks import validate_config

        validate_config(config)
        return
    if task == openclip_tasks.TASK and config.get("model_type") == openclip_tasks.FAMILY:
        openclip_tasks.validate_config(config)
        return
    if task == "image-segmentation" and config.get("model_type") == birefnet_tasks.FAMILY:
        birefnet_tasks.validate_config(config)
        return
    if task == "image-segmentation" and config.get("model_type") == isnet_tasks.FAMILY:
        isnet_tasks.validate_config(config)
        return
    if jina_ranking.supports(config, task):
        jina_ranking.validate_config(config)
        return
    if pii_tasks.supports(config, task):
        pii_tasks.validate_config(config)
        return
    if contextual_embeddings.supports(config, task):
        contextual_embeddings.validate_config(config)
        return
    if task == "zero-shot-classification":
        if decision_tasks.supports(config, task):
            decision_tasks.validate_config(config)
        else:
            nli_label_ids(config)
    elif config.get("model_type") not in TASK_FAMILIES.get(task, set()):
        raise ValueError("Native task behavior is unavailable for this family")
    coverage_tasks.validate_config(task, config)
    if task in {"question-answering", "translation"}:
        from oryn_engine import instruction_translation
        from oryn_engine.hdf5_tasks import validate_config

        if instruction_translation.supports(config, task):
            instruction_translation.adapter(config).validate_config(config)
        else:
            validate_config(config, task)
    if task in qwen_retrieval.TASK_CONTRACTS:
        if bidirectional_embeddings.supports(task, config):
            bidirectional_embeddings.validate_config(config)
        else:
            qwen_retrieval.validate_config(config)
    if task in flax_weights.TASK_CONTRACTS:
        if task == "text-classification" and config.get("model_type") == "roberta":
            from oryn_engine.hdf5_tasks import validate_config

            validate_config(config, task)
        else:
            flax_weights.validate_config(config, task)
    if task == "time-series-forecasting":
        forecast_tasks.validate_config(config)
    if task == "depth-estimation":
        from oryn_engine.depth3_tasks import validate_config

        validate_config(config)
    if task in tabular_tasks.TASK_CONTRACTS:
        tabular_tasks.validate_config(config)
    if task in MULTIMODAL_TASKS:
        gemma4_tasks.validate_config(config)
    if task == "automatic-speech-recognition":
        from oryn_engine.onnx_asr import validate_config

        if audio8_tasks.supports(config, task):
            audio8_tasks.validate_config(config)
        elif ct2_tasks.supports(config, task):
            ct2_tasks.validate_config(config)
        elif parakeet_tasks.supports(config, task):
            parakeet_tasks.validate_config(config)
        else:
            (rnnt_tasks.validate_config if rnnt_tasks.supports(config, task) else validate_config)(
                config
            )
    if task == "audio-text-to-text":
        audio_understanding.validate_config(config)
    if task == "pose-estimation":
        pose_tasks.validate_config(config)
    if config.get("model_type") == "vision-encoder-decoder":
        trocr_tasks.validate_config(config)
    if config.get("model_type") == "llava_next":
        from oryn_engine.llava_next_tasks import validate_config

        validate_config(config)
    if task == "mask-generation" and config.get("model_type") == "sam2_video":
        from oryn_engine.sam2_tasks import validate_config

        validate_config(config)


def open_task_image(path: Path, *, thumbnail=True, preserve_mode=False):
    from PIL import Image

    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= MAX_IMAGE_BYTES:
        raise ValueError("Image file exceeds input bounds")
    with Image.open(path) as source:
        # Check header dimensions before decoding/decompression allocates pixels.
        if source.format not in {"PNG", "JPEG", "WEBP"} or getattr(source, "n_frames", 1) != 1:
            raise ValueError("Image container is not admitted")
        if not 1 <= source.width <= 4096 or not 1 <= source.height <= 4096:
            raise ValueError("Image dimensions exceed input bounds")
        source.load()
        image = source.copy() if preserve_mode else source.convert("RGB")
    # New task contracts deliberately share the current 1024px preprocessing ceiling.
    original = image.size
    if thumbnail:
        image.thumbnail((1024, 1024))
    return image, original


def _move(inputs, backend):
    import torch

    # Native point processors can produce float64 coordinates; MPS accepts float32.
    return {
        key: item.to(
            device=backend, dtype=torch.float32 if item.dtype == torch.float64 else item.dtype
        )
        if isinstance(item, torch.Tensor)
        else item
        for key, item in inputs.items()
    }


def load_task(root: Path, task: str, backend: str):
    import torch
    import transformers as tf

    from oryn_engine.generated_capabilities import architectures_for, normalized_transformers_config

    if backend not in {"cpu", "mps", "cuda"}:
        raise ValueError("Backend is unavailable")
    from oryn_engine.model_metadata import root_config_name

    config = metadata_file(root / root_config_name({p.name for p in root.iterdir()}))
    from oryn_engine.timm_tasks import normalized_config

    config = normalized_transformers_config(
        tabular_tasks.normalized_config(normalized_config(config, task), task), task
    )
    validate_task_config(
        task,
        config,
        numeric_archive=numeric_archive_tasks.selected(
            config, task, {p.name for p in root.iterdir()}
        ),
    )
    architectures = config.get("architectures", [])
    if task == "keypoint-detection":
        architectures = [config.get("model_type", "unknown")]
    if (
        not isinstance(architectures, list)
        or len(architectures) != 1
        or not set(architectures) <= architectures_for(task)
    ):
        raise ValueError("Task architecture is not in the signed capability source")
    from oryn_engine.hdf5_tasks import supports as supports_hdf5

    if numeric_archive_tasks.selected(config, task, {p.name for p in root.iterdir()}):
        return numeric_archive_tasks.load_components(root, task, backend)
    if config.get("model_type") == "rtmw":
        from oryn_engine.rtmw_tasks import load

        return load(root, config).to(backend), None, config
    if config.get("model_type") == "tipsv2":
        from oryn_engine.tips_tasks import load_components

        return load_components(root, config, backend)
    if config.get("model_type") == "oryn_open_clip":
        from oryn_engine.openclip_tasks import load_components

        return load_components(root, config, backend)
    if config.get("model_type") == "birefnet":
        from oryn_engine.birefnet_tasks import load

        return load(root, config).to(backend), None, config
    if config.get("model_type") == "oryn_pixai":
        from oryn_engine.pixai_tasks import load

        return load(root, config).to(backend), None, config
    if config.get("model_type") == "oryn_isnet":
        from oryn_engine.isnet_tasks import load

        return load(root, config).to(backend), None, config

    if (
        supports_hdf5(config, task)
        and (root / "tf_model.h5").is_file()
        and not any(root.glob("*.safetensors*"))
        and not (flax_weights.supports(config, task) and (root / "flax_model.msgpack").is_file())
    ):
        from oryn_engine.hdf5_tasks import load_components

        return load_components(root, task, backend)
    from oryn_engine import turn_classification

    if turn_classification.supports(config, task):
        return turn_classification.load_components(root, config, backend)
    if task in flax_weights.TASK_CONTRACTS or (
        task == "automatic-speech-recognition"
        and (root / "flax_model.msgpack").is_file()
        and not any(root.glob("*.safetensors*"))
    ):
        return flax_weights.load_components(root, task, backend)
    if decision_tasks.supports(config, task):
        return decision_tasks.load_components(root, config, backend)
    if pii_tasks.supports(config, task):
        return pii_tasks.load_components(root, config, backend)
    if audio8_tasks.supports(config, task):
        return audio8_tasks.load_components(root, config, backend)
    if ct2_tasks.supports(config, task):
        return ct2_tasks.load_components(root, config, backend)
    if rnnt_tasks.supports(config, task):
        return rnnt_tasks.load_components(root, config, backend)
    if parakeet_tasks.supports(config, task):
        return parakeet_tasks.load_components(root, config, backend)
    if layoutlm_tasks.supports(config, task):
        return layoutlm_tasks.load_components(root, config, backend)
    if llava_video_qwen.supports(config, task):
        return llava_video_qwen.load_components(root, config, backend)
    if videollama3_tasks.supports(config, task):
        return videollama3_tasks.load_components(root, config, backend)
    if tiger_tasks.supports(config, task):
        return tiger_tasks.load_components(root, config, backend)
    if reuse_tasks.supports(config, task):
        return reuse_tasks.load_components(root, config, backend)
    if mpsenet_tasks.supports(config, task):
        return mpsenet_tasks.load_components(root, config, backend)
    if qwen_codec_tasks.supports(config, task):
        return qwen_codec_tasks.load_components(root, config, backend)
    if molparser_tasks.supports(config, task):
        return molparser_tasks.load_components(root, config, backend)
    if breeze_tasks.supports(config, task):
        return breeze_tasks.load_components(root, config, backend)
    if omnivoice_tasks.supports(config, task):
        return omnivoice_tasks.load_components(root, config, backend)
    if sopro_tasks.supports(config, task):
        return sopro_tasks.load_components(root, config, backend)
    if speechx_tasks.supports(config, task):
        return speechx_tasks.load_components(root, config, backend)
    if wav2vec_heads.supports(config, task):
        return wav2vec_heads.load_components(root, config, backend)
    if task == "depth-estimation" and config.get("model_type") == "depth_anything_3":
        from oryn_engine.depth3_tasks import load

        return load(root, config).to(backend), None, config
    if task == "time-series-forecasting" and config.get("model_type") == "exaone-demand":
        from oryn_engine.exaone_demand import load

        return load(root, config).to(backend), None, config
    if task == "time-series-forecasting" and config.get("model_type") == "yug":
        from oryn_engine.yug_tasks import load

        return load(root, config).to(backend), None, config
    if task == "time-series-forecasting" and config.get("model_type") == "aurora":
        from oryn_engine.aurora_tasks import load

        return load(root, config).to(backend), None, config
    if config.get("model_type") == "llava_next":
        from oryn_engine.llava_next_tasks import load_components

        return load_components(root, config, backend)
    if config.get("model_type") == "pplx_contextual_qwen3_5":
        from oryn_engine.contextual_embeddings import load_components

        return load_components(root, config, backend)
    tensor_shapes = preflight_safetensors(root)
    family = config["model_type"]
    if jina_ranking.supports(config, task):
        return jina_ranking.load_components(root, config, backend)
    if task == "translation":
        from oryn_engine.instruction_translation import adapter

        return adapter(config).load_components(root, config, backend)
    if task in qwen_retrieval.TASK_CONTRACTS:
        if bidirectional_embeddings.supports(task, config):
            return bidirectional_embeddings.load_components(root, config, backend)
        return qwen_retrieval.load_components(root, task, config, backend)
    if task in tabular_tasks.TASK_CONTRACTS:
        return tabular_tasks.load(root, config, backend), None, config
    if task == "visual-document-retrieval" and family == "colqwen3":
        from oryn_engine.colqwen3_tasks import load_components

        return load_components(root, config, backend)
    if task == "visual-document-retrieval" and family == "jina_embeddings_v4":
        from oryn_engine.jina4_tasks import load_components

        return load_components(root, config, backend)
    if task == "visual-document-retrieval" and family == "qwen3_5":
        from oryn_engine.colqwen35_tasks import load_components

        return load_components(root, config, backend)
    metadata_names = (
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
    ) + (
        ("video_preprocessor_config.json",)
        if family in {"vjepa2", "sam2_video"}
        or task == "video-text-to-text"
        and family in QWEN_VISION_FAMILIES
        else ()
    )
    for filename in metadata_names:
        if (root / filename).exists():
            validate_processor_config(metadata_file(root / filename), family)
    common = {"local_files_only": True, "trust_remote_code": False}
    if task in MULTIMODAL_TASKS:
        processor, loader = gemma4_tasks.load_components(root, config, common)
    elif family == "timm_wrapper":
        from oryn_engine.timm_tasks import load_components

        processor, loader, native_timm_config = load_components(root, config, task)
    elif task == "time-series-forecasting":
        processor, loader = None, tf.AutoModelForTimeSeriesPrediction
    elif task == "audio-text-to-text":
        processor, loader = audio_understanding.load_components(root, config, common)
    elif task == "automatic-speech-recognition":
        processor, loader = ctc_tasks.load_components(root, config, common)
    elif task == "pose-estimation":
        processor, loader = pose_tasks.load_components(root, config, common)
    elif family == "vision-encoder-decoder":
        processor, loader = trocr_tasks.load_components(root, config, common)
    elif family == "llava_next_video":
        from oryn_engine.llava_video import load_components

        processor, loader = load_components(root, config, common)
    elif task in coverage_tasks.TASK_CONTRACTS and task != "video-text-to-text":
        processor, loader = coverage_tasks.load_components(root, task, config, common)
    elif task == "zero-shot-classification":
        processor = tf.AutoTokenizer.from_pretrained(root, **common)
        loader = tf.AutoModelForSequenceClassification
    elif task == "image-feature-extraction":
        processor = tf.AutoImageProcessor.from_pretrained(root, **common)
        loader = tf.AutoModel
    elif task == "image-segmentation":
        if family == "segformer":
            preprocessing = metadata_file(root / "preprocessor_config.json")
            identity = preprocessing.get(
                "image_processor_type", preprocessing.get("feature_extractor_type")
            )
            if identity not in {"SegformerImageProcessor", "SegformerFeatureExtractor"}:
                raise ValueError("SegFormer processor identity is not admitted")
            # Transformers 5 removed the old FeatureExtractor alias; the native image
            # processor implements the same reusable family behavior and bounded settings.
            processor = tf.SegformerImageProcessor.from_pretrained(root, **common)
        else:
            processor = tf.AutoImageProcessor.from_pretrained(root, **common)
        loader = (
            tf.AutoModelForUniversalSegmentation
            if family == "mask2former"
            else tf.AutoModelForSemanticSegmentation
        )
    elif task == "mask-generation":
        processor = tf.AutoProcessor.from_pretrained(root, **common)
        loader = tf.Sam2VideoModel if family == "sam2_video" else tf.AutoModelForMaskGeneration
    elif task == "zero-shot-object-detection":
        processor = tf.AutoProcessor.from_pretrained(root, **common)
        loader = tf.AutoModelForZeroShotObjectDetection
    else:
        if family in QWEN_VISION_FAMILIES | {"glm_ocr"}:
            processor_file = root / "processor_config.json"
            if not processor_file.exists():
                processor_file = root / "preprocessor_config.json"
            options = qwen_image_options(
                metadata_file(processor_file),
                config.get("vision_config", {}),
                video_only=task == "video-text-to-text" and family in QWEN_VISION_FAMILIES,
            )
            image_processor = (
                tf.Glm46VImageProcessor(**options)
                if family == "glm_ocr"
                else tf.Qwen2VLImageProcessor(**options)
            )
            tokenizer = tf.AutoTokenizer.from_pretrained(root, **common)
            tokenizer.chat_template = None
            processor_class = {
                "qwen2_vl": tf.Qwen2VLProcessor,
                "qwen2_5_vl": tf.Qwen2_5_VLProcessor,
                "qwen3_vl": tf.Qwen3VLProcessor,
                "qwen3_5": tf.Qwen3VLProcessor,
                "glm_ocr": tf.Glm46VProcessor,
            }[family]
            # Image tasks require an unused one-frame video processor. Video
            # tasks retain native pixel transforms with owned sampling budgets.
            video_class = (
                tf.Glm46VVideoProcessor
                if family == "glm_ocr"
                else tf.Qwen2VLVideoProcessor
                if family in {"qwen2_vl", "qwen2_5_vl"}
                else tf.Qwen3VLVideoProcessor
            )
            video_file = root / "video_preprocessor_config.json"
            video_processor = (
                video_class(
                    **qwen_video_options(
                        metadata_file(video_file) if video_file.exists() else {},
                        config.get("vision_config", {}),
                        family,
                    )
                )
                if task == "video-text-to-text" and family in QWEN_VISION_FAMILIES
                else video_class(
                    size={"shortest_edge": 56**2, "longest_edge": 256 * 28**2},
                    min_frames=8,
                    max_frames=8,
                    do_sample_frames=False,
                    cap_pixels_per_frame=True,
                    max_video_tokens=4096,
                )
                if task == "video-text-to-text"
                else video_class(
                    size=options["size"], min_frames=1, max_frames=1, do_sample_frames=False
                )
            )
            processor = processor_class(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=video_processor,
            )
        else:
            processor = tf.AutoProcessor.from_pretrained(root, **common)
        loader = (
            tf.AutoModelForVisualQuestionAnswering
            if task == "visual-question-answering" and family in {"blip", "blip-2", "vilt"}
            else tf.AutoModelForImageTextToText
        )
    model_options = {}
    if family == "sam2_video":
        from oryn_engine.sam2_tasks import native_config

        model_options["config"] = native_config(config)
    if family == "timm_wrapper":
        model_options["config"] = native_timm_config
    zero_biases = set()
    if family == "colqwen2":
        from oryn_engine.visual_retrieval import legacy_weight_mapping

        mapping = legacy_weight_mapping(tensor_shapes)
        if mapping:
            model_options["key_mapping"] = mapping
    if family == "videomae" and any(name.endswith(".q_bias") for name in tensor_shapes):
        # Native builds moved the legacy separate q/v bias vectors into Linear layers.
        # Where a build adds a key bias, it must reproduce the old constant zero.
        # Match the complete old layout before applying this data-only rename.
        layers, hidden = config.get("num_hidden_layers"), config.get("hidden_size")
        for index in range(layers):
            prefix = f"videomae.encoder.layer.{index}.attention.attention"
            if any(
                tensor_shapes.get(prefix + suffix) != (hidden,) for suffix in (".q_bias", ".v_bias")
            ) or any(
                prefix + suffix in tensor_shapes
                for suffix in (".query.bias", ".key.bias", ".value.bias")
            ):
                raise ValueError("VideoMAE bias layout is incomplete or ambiguous")
            zero_biases.add(prefix + ".key.bias")
        model_options["key_mapping"] = {
            r"\.attention\.attention\.q_bias$": ".attention.attention.query.bias",
            r"\.attention\.attention\.v_bias$": ".attention.attention.value.bias",
        }
    if family == "omdet-turbo":
        # The upstream compatibility converter probes the Hub for legacy backbone
        # names. Construct the reviewed native backbone locally, with no resolver.
        if config.get("backbone") != "swin_tiny_patch4_window7_224":
            raise ValueError("OmDet backbone is not admitted")
        expected = {"always_partition": True, "img_size": 640, "out_indices": [1, 2, 3]}
        if config.get("backbone_kwargs") != expected or config.get("image_size") != 640:
            raise ValueError("OmDet backbone options are not admitted")
        if config.get("backbone_config") is not None:
            raise ValueError("OmDet backbone override is not admitted")
        native_config = dict(config)
        native_config["backbone_config"] = tf.TimmBackboneConfig(
            backbone="swin_tiny_patch4_window7_224", **expected
        )
        native_config["disable_custom_kernels"] = True
        model_options["config"] = tf.OmDetTurboConfig.from_dict(native_config)
    from oryn_engine import timesfm3_tasks, timesfm_original

    if task == "video-text-to-text" and family == "qwen3_vl":
        model = coverage_tasks.load_qwen3_video(root, config, tensor_shapes)
        loading = {}
    elif task == "time-series-forecasting" and family == "timesfm3":
        model = timesfm3_tasks.load(root, config)
        loading = {}
    elif task == "time-series-forecasting" and timesfm_original.applies(config):
        model = timesfm_original.load(root, config)
        loading = {}
    elif task == "time-series-forecasting" and family == "t5":
        model = forecast_tasks.load_chronos(root, config)
        loading = {}
    elif family == "timm_wrapper":
        from oryn_engine.timm_tasks import load_model

        model = load_model(root, loader, native_timm_config)
        loading = {}
    else:
        model, loading = loader.from_pretrained(
            root,
            **common,
            **model_options,
            use_safetensors=True,
            dtype={"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
                gemma4_tasks.precision(config, backend)
            ]
            if task in MULTIMODAL_TASKS
            else torch.float32,
            output_loading_info=True,
        )
    # Transformers >= 5.20 builds VideoMAE keys without a bias; zero only what exists.
    zero_biases &= {name for name, _ in model.named_parameters()}
    if (
        set(loading.get("missing_keys", [])) != zero_biases
        or loading.get("mismatched_keys")
        or loading.get("error_msgs")
        or (
            task in MULTIMODAL_TASKS
            or family in audio_understanding.FLAMINGO_FAMILIES
            or family == "vision-encoder-decoder"
            or family == "sapiens2"
            or family == "llava_next_video"
            or family == "neomme"
            or family == "voxtral"
            or task == "video-text-to-text"
        )
        and loading.get("unexpected_keys")
    ):
        raise ValueError("Selected task weights are incomplete; random task heads are not admitted")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in zero_biases:
                parameter.zero_()
    model = model.to(backend)
    model.eval()
    if task == "video-text-to-text":
        # Pinned Transformers fills unspecified generation fields from this
        # object even when generate() receives a separate GenerationConfig.
        # Repository defaults must not change the owned video task policy.
        text_config = config.get("text_config", config)
        model.generation_config = tf.GenerationConfig(
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=text_config.get("eos_token_id", config.get("eos_token_id")),
            pad_token_id=config.get("pad_token_id", processor.tokenizer.pad_token_id),
        )
    if family == "vision-encoder-decoder":
        trocr_tasks.configure_generation(model, config)
    return model, processor, config


def nli_scores(logits, entailment: int, contradiction: int, *, multi_label: bool):
    import torch

    if (
        logits.ndim != 2
        or not 1 <= logits.shape[0] <= 32
        or logits.shape[1] not in {2, 3}
        or type(entailment) is not int
        or type(contradiction) is not int
        or not 0 <= entailment < logits.shape[1]
        or not 0 <= contradiction < logits.shape[1]
        or entailment == contradiction
        or not bool(torch.isfinite(logits).all())
    ):
        raise ValueError("NLI output is invalid")
    logits = logits.float()
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("NLI output exceeds score precision")
    if multi_label:
        return torch.softmax(logits[:, [contradiction, entailment]], dim=-1)[:, 1]
    return torch.softmax(logits[:, entailment], dim=0)


def _zero_shot(model, processor, value, backend):
    import torch

    entailment, contradiction = nli_label_ids(model.config)
    batches = []
    labels = value["labels"]
    for offset in range(0, len(labels), 4):
        batch = labels[offset : offset + 4]
        inputs = processor(
            [value["text"]] * len(batch),
            [f"This example is {label}." for label in batch],
            padding=True,
            truncation="only_first",
            max_length=512,
            return_tensors="pt",
        )
        batches.append(model(**_move(inputs, backend)).logits.float().cpu())
    scores = nli_scores(
        torch.cat(batches), entailment, contradiction, multi_label=value["multiLabel"]
    )
    return {
        "predictions": sorted(
            [
                {"label": label, "score": float(score)}
                for label, score in zip(labels, scores, strict=True)
            ],
            key=lambda item: -item["score"],
        ),
        "multiLabel": value["multiLabel"],
    }


def image_vector(output, *, normalize: bool):
    import torch

    hidden = output.last_hidden_state
    if hidden.ndim != 3 or hidden.shape[0] != 1 or not 1 <= hidden.shape[2] <= 8192:
        raise ValueError("Image encoder output shape is invalid")
    # ViT and admitted DINO families expose the CLS embedding at token zero.
    vector = hidden[0, 0].float()
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("Image encoder returned non-finite values")
    if normalize:
        norm = torch.linalg.vector_norm(vector)
        if float(norm) <= 1e-12:
            raise ValueError("Image embedding cannot be normalized")
        vector = vector / norm
    return {
        "vector": vector.cpu().tolist(),
        "dimensions": int(vector.numel()),
        "normalized": normalize,
        "pooling": "cls",
    }


def _vision_prompt(family: str, question: str) -> str:
    # Oryn-owned renderers. Repository Jinja templates are never executed by these tasks.
    if family == "glm_ocr":
        return (
            "[gMASK]<sop><|user|>\n<|begin_of_image|><|image|><|end_of_image|>"
            "Text Recognition:<|assistant|>\n"
        )
    if family in {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_5"}:
        return (
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{question}<|im_end|>\n<|im_start|>assistant\n"
        )
    if family == "llava":
        return f"USER: <image>\n{question}\nASSISTANT:"
    return question


def _vision_text(model, processor, config, task, image, value, backend):
    import torch

    family = config["model_type"]
    if family == "vision-encoder-decoder":
        return trocr_tasks.execute(model, processor, config, image, value, backend)
    if family == "llava_next":
        from oryn_engine.llava_next_tasks import execute

        return execute(model, processor, config, image, value, backend)
    question = value.get("question", "Describe this image.")
    kwargs = {"images": image, "return_tensors": "pt"}
    if family != "blip" or task == "visual-question-answering":
        kwargs["text"] = _vision_prompt(family, question)
    inputs = _move(dict(processor(**kwargs)), backend)
    if "input_ids" in inputs and inputs["input_ids"].shape[-1] > 4096:
        raise ValueError("Vision prompt exceeds token bound")
    if family == "blip":
        from oryn_engine.hdf5_blip import token_limit

        limit = token_limit(config)
        if not 1 <= value["maxTokens"] <= limit or (
            "input_ids" in inputs and inputs["input_ids"].shape[-1] > limit
        ):
            raise ValueError("BLIP_INPUT_OR_OUTPUT_TOKEN_LIMIT")
    if family == "vilt":
        logits = model(**inputs).logits[0].float()
        if logits.numel() > 65_536 or not bool(torch.isfinite(logits).all()):
            raise ValueError("VQA logits are invalid")
        answer = str(model.config.id2label[int(logits.argmax())])
    else:
        generated = model.generate(**inputs, max_new_tokens=value["maxTokens"], do_sample=False)
        if not getattr(model.config, "is_encoder_decoder", False) and family not in {
            "blip",
            "blip-2",
            "pix2struct",
        }:
            generated = generated[:, inputs["input_ids"].shape[-1] :]
        answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    if not answer or len(answer.encode()) > 16_384:
        raise ValueError("Vision text result is empty or oversized")
    return {"text": answer}


def _detection(model, processor, config, image, original, value, backend):
    import torch

    family = config["model_type"]
    labels = value["labels"]
    inputs = _move(
        dict(
            processor(
                images=image,
                text=[labels]
                if family in {"owlvit", "owlv2", "omdet-turbo"}
                else ". ".join(labels) + ".",
                return_tensors="pt",
                padding=True,
            )
        ),
        backend,
    )
    output = model(**inputs)
    targets = [(original[1], original[0])]
    if family in {"owlvit", "owlv2"}:
        result = processor.post_process_grounded_object_detection(
            output, threshold=value["threshold"], target_sizes=targets, text_labels=[labels]
        )[0]
    elif family == "omdet-turbo":
        result = processor.post_process_grounded_object_detection(
            output, text_labels=[labels], threshold=value["threshold"], target_sizes=targets
        )[0]
    else:
        result = processor.post_process_grounded_object_detection(
            output,
            inputs["input_ids"],
            threshold=value["threshold"],
            text_threshold=value["threshold"],
            target_sizes=targets,
        )[0]
    names = result.get("text_labels", result.get("labels"))
    detections = []
    for score, name, box in zip(result["scores"], names, result["boxes"], strict=True):
        if len(detections) == 256:
            break
        coordinates = box.float().cpu()
        if (
            coordinates.numel() != 4
            or not bool(torch.isfinite(coordinates).all())
            or not 0 <= float(score) <= 1
        ):
            raise ValueError("Detection output is invalid")
        x0, y0, x1, y1 = coordinates.tolist()
        if x0 > x1 or y0 > y1:
            raise ValueError("Detection coordinates are invalid")
        label = name if isinstance(name, str) else labels[int(name)]
        if not isinstance(label, str) or len(label) > 1024:
            raise ValueError("Detection label is invalid")
        detections.append(
            {
                "label": label,
                "score": float(score),
                "box": [
                    max(0, min(original[0], x0)),
                    max(0, min(original[1], y0)),
                    max(0, min(original[0], x1)),
                    max(0, min(original[1], y1)),
                ],
            }
        )
    return {"detections": detections, "width": original[0], "height": original[1]}


def execute_task(
    root: Path,
    task: str,
    value: object,
    backend: str,
    input_path: Path | None,
    output_path: Path | None,
    emit,
):
    import numpy as np
    import torch
    from PIL import Image

    value = validate_task_input(task, value)
    if task == "feature-extraction" and {"inputType", "contextual"} & value.keys():
        from oryn_engine.contextual_embeddings import supports

        if not supports(metadata_file(root / "config.json"), task):
            raise ValueError("EMBEDDING_INPUT_MODE_UNSUPPORTED")
    if task == "audio-classification" and "classificationHead" in value:
        from oryn_engine.generated_capabilities import normalized_transformers_config

        if not speechx_tasks.supports(
            normalized_transformers_config(metadata_file(root / "config.json"), task), task
        ):
            raise ValueError("AUDIO_CLASSIFICATION_HEAD_UNSUPPORTED")
    image = original = None
    if task in IMAGE_TASKS:
        if input_path is None:
            raise ValueError("Authorized image input is missing")
        from oryn_engine.model_metadata import root_config_name

        image_config = metadata_file(root / root_config_name({p.name for p in root.iterdir()}))
        native_timm = "pretrained_cfg" in image_config
        image, original = open_task_image(
            input_path,
            preserve_mode=image_config.get("model_type") in {"cls_vitdet", "oryn_pixai"},
            thumbnail=image_config.get("model_type")
            not in {
                "cls_vitdet",
                "oryn_pixai",
                "tipsv2",
                "rtmw",
                "jina_embeddings_v4",
                "molparser_vision_encoder_decoder",
            }
            and not native_timm
            and "model_cfg" not in image_config
            and image_config.get("architectures") != ["BiRefNet"]
            and image_config.get("architectures") != ["JinaEmbeddingsV4Model"]
            and image_config.get("architectures") not in (["BriaRMBG"], ["ISNetDIS"])
            and image_config.get("model_type") != "llava_next"
            and task not in {"pose-estimation", "depth-estimation"},
        )
    emit("job.progress", {"stage": "loading-model", "progress": 0.1})
    model, processor, config = load_task(root, task, backend)
    emit("job.state", {"state": "running"})
    emit("job.progress", {"stage": "running-model", "progress": 0.3})
    with torch.inference_mode():
        from oryn_engine import turn_classification

        if tiger_tasks.supports(config, task):
            return tiger_tasks.execute(model, value, backend, input_path, output_path)
        if reuse_tasks.supports(config, task):
            return reuse_tasks.execute(model, config, value, backend, input_path, output_path)
        if mpsenet_tasks.supports(config, task):
            return mpsenet_tasks.execute(model, value, backend, input_path, output_path)
        if qwen_codec_tasks.supports(config, task):
            result = qwen_codec_tasks.execute(model, value, backend, input_path, output_path)
            return validate_task_output(task, result) if value["operation"] == "encode" else result
        if breeze_tasks.supports(config, task):
            return breeze_tasks.execute(
                model, processor, config, value, backend, input_path, output_path
            )
        if omnivoice_tasks.supports(config, task):
            return omnivoice_tasks.execute(
                model, processor, config, value, backend, input_path, output_path
            )
        if sopro_tasks.supports(config, task):
            return sopro_tasks.execute(
                model, processor, config, value, backend, input_path, output_path
            )
        if numeric_archive_tasks.selected(config, task, {p.name for p in root.iterdir()}):
            result = numeric_archive_tasks.execute(
                model, processor, config, task, value, backend, image, input_path, original
            )
        elif turn_classification.supports(config, task):
            result = turn_classification.execute(model, processor, config, value, backend)
        elif pii_tasks.supports(config, task):
            result = pii_tasks.execute(model, processor, config, value, backend)
        elif speechx_tasks.supports(config, task):
            result = speechx_tasks.execute(model, processor, config, value, backend, input_path)
        elif wav2vec_heads.supports(config, task):
            result = wav2vec_heads.execute(model, processor, config, value, backend, input_path)
        elif task in qwen_retrieval.TASK_CONTRACTS:
            from oryn_engine import contextual_embeddings

            if contextual_embeddings.supports(config, task):
                result = contextual_embeddings.execute(model, processor, config, value, backend)
            elif jina_ranking.supports(config, task):
                result = jina_ranking.execute(model, processor, config, value, backend)
            elif bidirectional_embeddings.supports(task, config):
                result = bidirectional_embeddings.execute(model, processor, config, value, backend)
            else:
                result = qwen_retrieval.execute(model, processor, config, task, value, backend)
        elif task == "translation":
            from oryn_engine import instruction_translation
            from oryn_engine.hdf5_marian import translate

            result = (
                instruction_translation.adapter(config).execute(
                    model, processor, config, value, backend
                )
                if instruction_translation.supports(config, task)
                else translate(model, processor, config, value, backend)
            )
        elif task == "question-answering":
            from oryn_engine.hdf5_tasks import answer_question

            result = answer_question(model, processor, config, value, backend)
        elif task == "text-classification" and config.get("model_type") == "roberta":
            from oryn_engine.hdf5_tasks import classify

            result = classify(model, processor, config, value, backend)
        elif config.get("model_type") == "oryn_pixai":
            from oryn_engine.pixai_tasks import execute

            result = execute(model, config, image, value, backend)
        elif config.get("model_type") == "rtmw":
            from oryn_engine.rtmw_tasks import execute

            result = execute(model, config, image, value, backend)
        elif config.get("model_type") == "tipsv2":
            from oryn_engine.tips_tasks import execute

            result = execute(model, processor, config, image, value, backend)
        elif config.get("model_type") == "oryn_open_clip":
            from oryn_engine.openclip_tasks import execute

            result = execute(model, processor, image, value, backend)
        elif task in flax_weights.TASK_CONTRACTS:
            result = flax_weights.execute(model, processor, value, backend, config, image)
        elif task == "image-classification":
            from oryn_engine.timm_tasks import classify

            result = classify(model, processor, image, config, value, backend)
        elif task == "depth-estimation":
            from oryn_engine.depth3_tasks import execute

            result = execute(model, image, backend)
        elif task == "time-series-forecasting":
            result = forecast_tasks.execute(model, config, value, backend)
        elif task in tabular_tasks.TASK_CONTRACTS:
            result = tabular_tasks.execute(model, task, value, backend)
        elif task in MULTIMODAL_TASKS:
            result = gemma4_tasks.execute(model, processor, config, value, backend, input_path)
        elif task == "audio-text-to-text":
            result = audio_understanding.execute(model, processor, value, backend, input_path)
        elif task == "automatic-speech-recognition":
            execute = (
                audio8_tasks.execute
                if audio8_tasks.supports(config, task)
                else ct2_tasks.execute
                if ct2_tasks.supports(config, task)
                else parakeet_tasks.execute
                if parakeet_tasks.supports(config, task)
                else rnnt_tasks.execute
                if rnnt_tasks.supports(config, task)
                else ctc_tasks.execute
            )
            result = execute(model, processor, config, value, backend, input_path, emit)
        elif task == "pose-estimation":
            result = pose_tasks.execute(model, processor, config, image, value, backend)
        elif task in coverage_tasks.TASK_CONTRACTS:
            result = coverage_tasks.execute(
                task, model, processor, config, value, backend, input_path, image, original
            )
        elif task == "zero-shot-classification":
            result = (
                decision_tasks.execute(model, processor, config, value, backend)
                if decision_tasks.supports(config, task)
                else _zero_shot(model, processor, value, backend)
            )
        elif task == "image-feature-extraction":
            inputs = _move(dict(processor(images=image, return_tensors="pt")), backend)
            result = image_vector(model(**inputs), normalize=value["normalize"])
        elif task == "zero-shot-object-detection":
            result = _detection(model, processor, config, image, original, value, backend)
        elif molparser_tasks.supports(config, task):
            result = molparser_tasks.execute(model, processor, config, image, value, backend)
        elif task in {"image-to-text", "visual-question-answering"}:
            result = _vision_text(model, processor, config, task, image, value, backend)
        else:
            if output_path is None:
                raise ValueError("Authorized output staging is missing")
            if task == "image-segmentation":
                if config.get("model_type") == "birefnet":
                    from oryn_engine.birefnet_tasks import segmentation

                    mask = segmentation(model, image, backend)
                elif config.get("model_type") == "oryn_isnet":
                    from oryn_engine.isnet_tasks import segmentation

                    mask = segmentation(model, image, backend)
                else:
                    inputs = _move(dict(processor(images=image, return_tensors="pt")), backend)
                    output = model(**inputs)
                    mask = processor.post_process_semantic_segmentation(
                        output, target_sizes=[(image.height, image.width)]
                    )[0]
                if (
                    mask.ndim != 2
                    or tuple(mask.shape) != (image.height, image.width)
                    or int(mask.min()) < 0
                    or int(mask.max()) > 65535
                ):
                    raise ValueError("Segmentation output is invalid")
                rendered = Image.fromarray(mask.cpu().numpy().astype(np.uint16))
            else:
                points = [
                    [
                        [p["x"] * (image.width - 1), p["y"] * (image.height - 1)]
                        for p in value["points"]
                    ]
                ]
                labels = [[p["label"] for p in value["points"]]]
                if config["model_type"] == "sam2_video":
                    from oryn_engine.sam2_tasks import predict_mask

                    mask = predict_mask(model, processor, image, points, labels, backend)
                else:
                    inputs = dict(
                        processor(
                            images=image,
                            input_points=points,
                            input_labels=labels,
                            return_tensors="pt",
                        )
                    )
                    sizes = inputs.pop("original_sizes")
                    reshaped = inputs.pop("reshaped_input_sizes")
                    output = model(**_move(inputs, backend), multimask_output=True)
                    if output.iou_scores.numel() > 3 or not bool(
                        torch.isfinite(output.iou_scores).all()
                    ):
                        raise ValueError("Mask scores exceed bounds")
                    masks = processor.post_process_masks(output.pred_masks.cpu(), sizes, reshaped)[
                        0
                    ][0]
                    if (
                        masks.ndim != 3
                        or masks.shape[0] > 3
                        or tuple(masks.shape[1:]) != (image.height, image.width)
                    ):
                        raise ValueError("Mask dimensions exceed bounds")
                    mask = masks[int(output.iou_scores.flatten().argmax())]
                rendered = Image.fromarray(mask.numpy().astype(np.uint8) * 255)
            # Restore bounded spatial coordinates with nearest-neighbor interpolation.
            rendered = rendered.resize(original, resample=Image.Resampling.NEAREST)
            ids, counts = np.unique(np.asarray(rendered), return_counts=True)
            if len(ids) > 512:
                raise ValueError("Segmentation label count exceeds bounds")
            labels = (
                config.get("id2label", {})
                if task == "image-segmentation"
                else {"0": "background", "255": "foreground"}
            )
            if any(str(int(i)) not in labels for i in ids):
                raise ValueError("Segmentation produced an unlabeled pixel ID")
            metadata = {
                "width": original[0],
                "height": original[1],
                "encoding": "class-id-16bit" if task == "image-segmentation" else "binary-8bit",
                "segments": [
                    {"id": int(i), "label": labels[str(int(i))], "pixelCount": int(n)}
                    for i, n in zip(ids, counts, strict=True)
                ],
            }
            rendered.save(output_path, format="PNG")
            result = {
                "relativeOutput": output_path.name,
                "mediaType": "image/png",
                "width": original[0],
                "height": original[1],
                "metadata": metadata,
            }
    del model
    if backend == "mps":
        torch.mps.empty_cache()
    if len(json.dumps(result, allow_nan=False).encode()) > result_byte_limit(task, result):
        raise ValueError("Task result exceeds bounded IPC")
    return result
