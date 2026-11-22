"""Shared bounded static file-layout checks for inspection and the offline audit."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from oryn_engine import decision_tasks

EXECUTABLE_SUFFIXES = frozenset({".py", ".pyc", ".so", ".dll", ".dylib", ".exe", ".sh"})
UNSAFE_SUFFIXES = frozenset({".bin", ".ckpt", ".pkl", ".pickle", ".pt", ".pth"})
SAFE_SUFFIXES = frozenset(
    {".json", ".safetensors", ".jinja", ".txt", ".model", ".spm", ".tiktoken"}
)
SHARD = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.safetensors$")
WEIGHT_INDEX = re.compile(r".+\.safetensors\.index(?:\.[A-Za-z0-9_-]+)?\.json$")


def weight_index_name(prefix: str, selected_files: set[str]) -> str | None:
    """Resolve declared native index conventions without inventing a double suffix."""
    stem = prefix.removesuffix(".safetensors")
    candidates = {f"{stem}.safetensors.index.json"}
    if "." in stem:
        base, variant = stem.rsplit(".", 1)
        candidates.add(f"{base}.safetensors.index.{variant}.json")
    matches = candidates & selected_files
    return next(iter(matches)) if len(matches) == 1 else None


@dataclass(frozen=True)
class LayoutAssessment:
    source_kind: str
    selected_files: tuple[str, ...]
    artifact_reasons: tuple[str, ...]
    runtime_reasons: tuple[str, ...]


def assess_layout(
    files: list[str] | tuple[str, ...],
    config: dict[str, Any],
    task: str | None,
    *,
    source_kind: str = "transformers",
) -> LayoutAssessment:
    reasons: list[str] = []
    runtime: list[str] = []
    if source_kind == "diffusers":
        from oryn_engine.flat_ddpm import normalized_index

        try:
            config = normalized_index(config)
        except (ValueError, TypeError):
            return LayoutAssessment(source_kind, (), (), ("PIPELINE_COMPONENT_UNSUPPORTED",))
        if not isinstance(config.get("_class_name"), str):
            return LayoutAssessment(source_kind, (), (), ("PIPELINE_COMPONENT_UNSUPPORTED",))
    if source_kind == "transformers":
        from oryn_engine.generated_capabilities import normalized_transformers_config
        from oryn_engine.tabular_tasks import normalized_config as normalized_tabular_config
        from oryn_engine.timm_tasks import normalized_config

        try:
            config = normalized_transformers_config(
                normalized_tabular_config(normalized_config(config, task), task), task
            )
        except (ValueError, TypeError):
            runtime.append("TASK_CONFIGURATION_UNSUPPORTED")
    if (
        len(files) > 10_000
        or len(set(files)) != len(files)
        or any(
            not name
            or len(name) > 1024
            or "\\" in name
            or PurePosixPath(name).is_absolute()
            or any(part in {"..", "."} for part in name.split("/"))
            for name in files
        )
    ):
        return LayoutAssessment(source_kind, (), ("MODEL_FILE_LAYOUT_INVALID",), ())
    if source_kind == "mlx-audio":
        from oryn_engine.mlx_audio_tasks import selected_files

        try:
            selected = selected_files(files, config, task)
        except (ValueError, TypeError):
            return LayoutAssessment(source_kind, (), ("MODEL_SUBSET_INCOMPLETE",), ())
        return LayoutAssessment(source_kind, selected, (), ())
    if source_kind == "diffusers":
        from oryn_engine import minimax_h3, minimax_music

        if minimax_h3.supports(config):
            try:
                selected = minimax_h3.selected_files(files, config)
                if task not in {"text-to-video", "image-text-to-video"}:
                    raise ValueError("H3_TASK")
            except (ValueError, TypeError):
                return LayoutAssessment(source_kind, (), ("MODEL_SUBSET_INCOMPLETE",), ())
            return LayoutAssessment(source_kind, selected, (), ())
        if minimax_music.supports(config):
            try:
                selected = minimax_music.selected_files(files, config)
                if task not in {"text-to-audio", "text-to-music"}:
                    raise ValueError("MUSIC3_TASK")
            except (ValueError, TypeError):
                return LayoutAssessment(source_kind, (), ("MODEL_SUBSET_INCOMPLETE",), ())
            return LayoutAssessment(source_kind, selected, (), ())
        from oryn_engine import numeric_unet2d

        if numeric_unet2d.selected(config, files):
            from oryn_engine.diffusion_tasks import validate_pipeline_index

            runtime = [] if task == "unconditional-image-generation" else ["TASK_INCOMPATIBLE"]
            try:
                # Exact bundled component identities, as for safetensors pipelines.
                validate_pipeline_index(config)
            except (ValueError, TypeError):
                runtime.append("PIPELINE_COMPONENT_UNSUPPORTED")
            if config.get("auto_map") or config.get("custom_pipelines"):
                runtime.append("CUSTOM_CODE_REQUIRED")
            return LayoutAssessment(
                source_kind,
                (
                    "config.json",
                    numeric_unet2d.WEIGHTS,
                    "model_index.json",
                    "scheduler_config.json",
                ),
                (),
                tuple(sorted(set(runtime))),
            )
    if source_kind == "transformers" and config.get("model_type") == "oryn_ct2_whisper":
        from oryn_engine.ct2_tasks import FILES, supports, validate_config

        try:
            validate_config(config)
            if not supports(config, task):
                raise ValueError("CT2_TASK")
        except (ValueError, TypeError):
            runtime.append("TASK_CONFIGURATION_UNSUPPORTED")
        if not set(files) >= FILES:
            reasons.append("MODEL_WEIGHTS_INCOMPLETE")
        if "vocabulary.json" in files:
            reasons.append("MODEL_CONFIG_AMBIGUOUS")
        return LayoutAssessment(
            source_kind, tuple(sorted(FILES & set(files))), tuple(reasons), tuple(runtime)
        )
    from oryn_engine import numeric_archive_tasks

    if source_kind == "transformers" and numeric_archive_tasks.selected(config, task, files):
        from oryn_engine.numeric_coverage import PROCESSOR_TASKS, TOKENIZER_FREE_TASKS

        tokenizer_free = {"depth-estimation", *TOKENIZER_FREE_TASKS}
        processor_tasks = {"depth-estimation", "zero-shot-object-detection", *PROCESSOR_TASKS}
        assets = set() if task in tokenizer_free else numeric_archive_tasks.TOKENIZER_FILES
        if task in processor_tasks:
            assets = assets | {"preprocessor_config.json"}
        selected = tuple(
            sorted(
                set(files)
                & {
                    "config.json",
                    "pytorch_model.bin",
                    *assets,
                }
            )
        )
        try:
            numeric_archive_tasks.validate_config(config, task)
        except (ValueError, TypeError):
            runtime.append("TASK_CONFIGURATION_UNSUPPORTED")
        if "config.json" not in selected:
            reasons.append("MODEL_SUBSET_INCOMPLETE")
        if config.get("model_type") == "m2m_100" and "tokenizer_config.json" not in selected:
            reasons.append("MODEL_TOKENIZER_MISSING")
        if task in processor_tasks and "preprocessor_config.json" not in selected:
            reasons.append("MODEL_PROCESSOR_MISSING")
        if config.get("model_type") == "tapas" and "vocab.txt" not in selected:
            reasons.append("MODEL_TOKENIZER_MISSING")
        if (
            task == "table-question-answering"
            and config.get("model_type") == "bart"
            and "tokenizer_config.json" not in selected
        ):
            # The TAPEX table serializer binding is declared only by this file.
            reasons.append("MODEL_TOKENIZER_MISSING")
        if task not in tokenizer_free and not {
            "tokenizer.json",
            "vocab.txt",
            "vocab.json",
            "sentencepiece.bpe.model",
            "tokenizer.model",
            "spiece.model",
        } & set(selected):
            reasons.append("MODEL_TOKENIZER_MISSING")
        return LayoutAssessment(source_kind, selected, tuple(reasons), tuple(runtime))
    # Alternate exports are separate candidates, not authority to change this loader.
    selected = tuple(
        sorted(
            name
            for name in files
            if PurePosixPath(name).suffix in SAFE_SUFFIXES
            and not name.startswith(("onnx/", "openvino/"))
        )
    )
    if source_kind in {"transformers", "mlx"}:
        from oryn_engine import mlx_bolding, mlx_kolibri, mlx_prism

        if source_kind == "mlx" and mlx_kolibri.supports(config):
            try:
                selected = mlx_kolibri.selected_files(files, config)
            except (ValueError, TypeError):
                return LayoutAssessment(source_kind, (), ("MODEL_SUBSET_INCOMPLETE",), ())
            return LayoutAssessment(source_kind, selected, (), ())
        if source_kind == "mlx" and mlx_bolding.supports(config):
            try:
                selected = mlx_bolding.selected_files(files, config)
            except (ValueError, TypeError):
                return LayoutAssessment(source_kind, (), ("MODEL_SUBSET_INCOMPLETE",), ())
            return LayoutAssessment(source_kind, selected, (), ())
        if source_kind == "mlx" and mlx_prism.supports(config):
            try:
                mlx_prism.validate_files(files, config)
            except (ValueError, TypeError):
                reasons.append("MODEL_SUBSET_INCOMPLETE")
        openclip = source_kind == "transformers" and config.get("model_type") == "oryn_open_clip"
        mlx_vad = source_kind == "mlx" and task == "voice-activity-detection"
        selected = tuple(
            name
            for name in selected
            if "/" not in name
            or name.startswith("chat_templates/")
            or task == "visual-document-retrieval"
            and config.get("model_type") == "jina_embeddings_v4"
            and name in {"adapters/adapter_config.json", "adapters/adapter_model.safetensors"}
            or task == "feature-extraction"
            and config.get("model_type") in {"qwen3_vl", "ministral3", "qwen3_bidirec"}
            and name == "1_Pooling/config.json"
        )
        if mlx_vad:
            from oryn_engine.mlx_vad import validate_config

            selected = tuple(
                name for name in selected if name in {"config.json", "model.safetensors"}
            )
            try:
                validate_config(config)
            except (ValueError, TypeError):
                runtime.append("TASK_CONFIGURATION_UNSUPPORTED")
        if source_kind == "transformers" and config.get("model_type") == "tabpfn_v3_5":
            from oryn_engine.tabular_tasks import weight_name

            selected = tuple(name for name in selected if name in {"config.json", weight_name()})
        if source_kind == "transformers" and config.get("model_type") in {
            "mitra_tab2d",
            "tabpfn_mix",
        }:
            selected = tuple(
                name for name in selected if name in {"config.json", "model.safetensors"}
            )
        # Native from_pretrained's default graph must not mix an alternate precision
        # export into the selected installation (e.g. Whisper's fp32 shard graph).
        if (
            source_kind == "transformers"
            and not openclip
            and "model.safetensors" in selected
            and "model.safetensors.index.json" not in selected
        ):
            # Native single-file loading selects this exact name. Other publisher
            # weight exports must not be installed into its preflight graph.
            selected = tuple(
                name
                for name in selected
                if not name.endswith(".safetensors")
                or name == "model.safetensors"
                or name == "adapters/adapter_model.safetensors"
                and task == "visual-document-retrieval"
                and config.get("model_type") == "jina_embeddings_v4"
                or name == "readout.safetensors"
                and task == "zero-shot-classification"
                and decision_tasks.supports(config, task)
            )
        if "model.safetensors" in selected or "model.safetensors.index.json" in selected:
            selected = tuple(
                name
                for name in selected
                if not re.fullmatch(
                    r"model\.(?!safetensors(?:-|\.))[A-Za-z0-9_-]+(?:-\d{5}-of-\d{5})?\.safetensors"
                    r"(?:\.index\.json)?|model\.safetensors\.index\.[A-Za-z0-9_-]+\.json",
                    name,
                )
            )
        if source_kind == "transformers" and config.get("model_type") == "voxtral":
            # The publisher also ships a consolidated Mistral-format alternative.
            # This route loads only the complete native Transformers export.
            selected = tuple(
                name
                for name in selected
                if not name.endswith(".safetensors")
                or name == "model.safetensors"
                or re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name)
            )
        if openclip:
            if "config.json" in files or "model_index.json" in files:
                reasons.append("MODEL_CONFIG_AMBIGUOUS")
            selected = tuple(
                name
                for name in selected
                if name in {"open_clip_config.json", "open_clip_model.safetensors"}
            )
        if ("open_clip_config.json" if openclip else "config.json") not in selected:
            reasons.append("MODEL_CONFIG_MISSING")
        from oryn_engine import layoutlm_tasks

        if layoutlm_tasks.supports(config, task):
            selected = tuple(sorted(name for name in files if name in layoutlm_tasks.FILES))
            if not set(layoutlm_tasks.FILES) <= set(selected):
                reasons.append("MODEL_SUBSET_INCOMPLETE")
        if config.get("model_type") == "molparser_vision_encoder_decoder":
            from oryn_engine.molparser_tasks import FILES

            selected = tuple(sorted(name for name in files if name in FILES))
            if not set(FILES) <= set(selected):
                reasons.append("MODEL_SUBSET_INCOMPLETE")
        if config.get("model_type") in {"videollama3_qwen2", "oryn_llava_video_qwen2"}:
            if config["model_type"] == "videollama3_qwen2":
                from oryn_engine.videollama3_tasks import REQUIRED_FILES
            else:
                from oryn_engine.llava_video_qwen import REQUIRED_FILES

            selected = tuple(
                sorted(
                    name
                    for name in selected
                    if name in REQUIRED_FILES
                    or name.endswith((".safetensors", ".safetensors.index.json"))
                )
            )
            if not set(REQUIRED_FILES) <= set(selected):
                reasons.append("MODEL_SUBSET_INCOMPLETE")
        if config.get("model_type") == "oryn_tiger_dnr":
            from oryn_engine.tiger_tasks import FILES

            selected = tuple(sorted(name for name in files if name in FILES))
            if not set(FILES) <= set(selected):
                reasons.append("MODEL_SUBSET_INCOMPLETE")
        if config.get("model_type") == "oryn_reuse":
            from oryn_engine.reuse_tasks import FILES

            selected = tuple(sorted(name for name in files if name in FILES))
            if not set(FILES) <= set(selected):
                reasons.append("MODEL_SUBSET_INCOMPLETE")
        if config.get("model_type") == "oryn_mpsenet":
            from oryn_engine.mpsenet_tasks import FILES

            selected = tuple(sorted(name for name in files if name in FILES))
            if not set(FILES) <= set(selected):
                reasons.append("MODEL_SUBSET_INCOMPLETE")
        if config.get("model_type") == "qwen3_tts_tokenizer_12hz":
            from oryn_engine.qwen_codec_tasks import FILES

            selected = tuple(sorted(name for name in files if name in FILES))
            if not set(FILES) <= set(selected):
                reasons.append("MODEL_SUBSET_INCOMPLETE")
        if config.get("model_type") == "breeze":
            from oryn_engine.breeze_tasks import REQUIRED_FILES

            selected = tuple(
                sorted(
                    name
                    for name in files
                    if name in REQUIRED_FILES
                    or name in {"model.safetensors", "model.safetensors.index.json"}
                    or re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name)
                )
            )
            if not set(REQUIRED_FILES) <= set(selected) or not {
                "model.safetensors",
                "model.safetensors.index.json",
            } & set(selected):
                reasons.append("MODEL_SUBSET_INCOMPLETE")
        if config.get("model_type") == "omnivoice":
            from oryn_engine.omnivoice_tasks import FILES

            selected = tuple(sorted(name for name in files if name in FILES))
            if not set(FILES) <= set(selected):
                reasons.append("MODEL_SUBSET_INCOMPLETE")
        if config.get("model_type") == "oryn_sopro":
            from oryn_engine.sopro_tasks import WEIGHTS

            selected = tuple(
                sorted(
                    name for name in files if name in {*WEIGHTS, "config.json", "tokenizer.model"}
                )
            )
            if not {*WEIGHTS, "config.json", "tokenizer.model"} <= set(selected):
                reasons.append("MODEL_SUBSET_INCOMPLETE")
        weights = [
            name
            for name in selected
            if name.endswith(".safetensors") and not PurePosixPath(name).name.startswith("adapter_")
        ]
        from oryn_engine.flax_weights import TASK_CONTRACTS as FLAX_TASKS
        from oryn_engine.flax_weights import supports as supports_flax
        from oryn_engine.flax_weights import validate_config as validate_flax
        from oryn_engine.hdf5_tasks import FORMAT_ONLY_TASKS
        from oryn_engine.instruction_translation import supports as supports_instruction_translation

        if (
            source_kind == "transformers"
            and not weights
            and "flax_model.msgpack" in files
            and supports_flax(config, task)
        ):
            try:
                validate_flax(config, task)
                selected = tuple(sorted((*selected, "flax_model.msgpack")))
                weights = ["flax_model.msgpack"]
            except ValueError:
                runtime.append("TASK_CONFIGURATION_UNSUPPORTED")
        if not weights:
            from oryn_engine.hdf5_tasks import supports as supports_hdf5
            from oryn_engine.hdf5_tasks import validate_config as validate_hdf5

            if (
                source_kind == "transformers"
                and "tf_model.h5" in files
                and supports_hdf5(config, task)
            ):
                try:
                    validate_hdf5(config, task)
                    selected = tuple(sorted((*selected, "tf_model.h5")))
                    weights = ["tf_model.h5"]
                except ValueError:
                    runtime.append("TASK_CONFIGURATION_UNSUPPORTED")
        if not weights:
            reasons.append("MODEL_SUBSET_INCOMPLETE")
        if (
            config.get("model_type") == "marian"
            and "tf_model.h5" in selected
            and not {"source.spm", "target.spm", "vocab.json", "tokenizer_config.json"}
            <= set(selected)
        ):
            reasons.append("MODEL_TOKENIZER_MISSING")
        groups: dict[str, tuple[int, set[int]]] = {}
        for name in weights:
            match = SHARD.fullmatch(name)
            if match:
                prefix, index, total = match.groups()
                count, indices = groups.setdefault(prefix, (int(total), set()))
                if count != int(total) or not 1 <= int(index) <= count <= 10_000:
                    reasons.append("MODEL_SHARDS_INVALID")
                indices.add(int(index))
        for prefix, (count, indices) in groups.items():
            if len(indices) != count or weight_index_name(prefix, set(selected)) is None:
                reasons.append("MODEL_SUBSET_INCOMPLETE")
        from oryn_engine.shared_tasks import (
            AUDIO_TASKS,
            IMAGE_TASKS,
            MULTIMODAL_TASKS,
            TASK_CONTRACTS,
            VIDEO_TASKS,
            validate_task_config,
        )

        if (
            source_kind == "transformers"
            and task in TASK_CONTRACTS
            and task != "fill-mask"
            and (
                task not in {*FLAX_TASKS, *FORMAT_ONLY_TASKS}
                or supports_instruction_translation(config, task)
                or set(selected) & {"flax_model.msgpack", "tf_model.h5"}
                or openclip
                or config.get("model_type") in {"oryn_pixai", "tipsv2"}
            )
            and (
                task != "automatic-speech-recognition"
                or config.get("model_type")
                in {"wav2vec2", "nemotron3_5_asr", "parakeet_tdt", "audio8_asr_infinite"}
            )
            and (
                task != "image-classification"
                or config.get("model_type") in {"timm_wrapper", "oryn_pixai"}
            )
            and (task != "token-classification" or config.get("model_type") == "pii_masking")
            and (
                task != "audio-classification"
                or config.get("model_type")
                in {"oryn_wav2vec2_age_gender", "oryn_wav2vec2_dual_hypersphere", "oryn_speechx"}
            )
            and (task != "depth-estimation" or config.get("model_type") == "depth_anything_3")
            and (
                task not in {"feature-extraction", "text-ranking"}
                or config.get("model_type") == "qwen3_vl"
                or task == "text-ranking"
                and config.get("architectures") == ["JinaForRanking"]
                or task == "feature-extraction"
                and config.get("model_type")
                in {"ministral3", "qwen3_bidirec", "pplx_contextual_qwen3_5"}
            )
        ):
            try:
                validate_task_config(task, config)
            except (ValueError, TypeError):
                runtime.append("TASK_CONFIGURATION_UNSUPPORTED")
        media = task in IMAGE_TASKS | VIDEO_TASKS | AUDIO_TASKS | MULTIMODAL_TASKS or task in {
            "image-classification",
            "object-detection",
            "depth-estimation",
            "zero-shot-image-classification",
            "audio-classification",
        }
        needs_text = (
            not media
            and task
            not in {"time-series-forecasting", "tabular-classification", "tabular-regression"}
        ) or task in {
            "zero-shot-image-classification",
            "text-to-speech",
            "zero-shot-object-detection",
            "image-to-text",
            "visual-question-answering",
            "document-question-answering",
            "visual-document-retrieval",
            "video-text-to-text",
            "audio-text-to-text",
            "automatic-speech-recognition",
            "any-to-any",
        }
        if (
            needs_text
            and not openclip
            and not mlx_vad
            and not any(
                "tokenizer" in name
                or name.endswith((".model", ".spm"))
                or name == "vocab.json"
                or name == "tekken.json"
                and config.get("model_type") == "voxtral"
                or name == "vocab.txt"
                and config.get("model_type") == "bert"
                and "tf_model.h5" in selected
                for name in selected
            )
        ):
            reasons.append("MODEL_TOKENIZER_MISSING")
        needs_processor = (
            media
            and not (
                task == "document-question-answering" and config.get("model_type") == "layoutlm"
            )
            and config.get("model_type")
            not in {
                "timm_wrapper",
                "depth_anything_3",
                "oryn_open_clip",
                "birefnet",
                "parakeet_tdt",
                "oryn_sopro",
                "omnivoice",
                "breeze",
                "oryn_mpsenet",
                "oryn_reuse",
                "oryn_tiger_dnr",
            }
        )
        processor_names = {"preprocessor_config.json", "processor_config.json"}
        if config.get("model_type") == "parakeet_tdt":
            processor_names = {"ternary.json"}
            if {"processor_config.json", "preprocessor_config.json", "tokenizer_config.json"} & set(
                selected
            ):
                reasons.append("TASK_CONFIGURATION_UNSUPPORTED")
        if config.get("model_type") == "parakeet_tdt" and not {
            "ternary.json",
            "tokenizer.json",
        } <= set(selected):
            reasons.append("MODEL_PROCESSOR_MISSING")
        if config.get("model_type") == "voxtral":
            processor_names = {"preprocessor_config.json"}
            if "tekken.json" not in selected:
                reasons.append("MODEL_TOKENIZER_MISSING")
        if config.get("model_type") == "tipsv2":
            if "tokenizer.model" not in selected:
                reasons.append("MODEL_TOKENIZER_MISSING")
            if not {"tokenizer_config.json", "processor_config.json"} <= set(selected):
                reasons.append("MODEL_PROCESSOR_MISSING")
        if config.get("model_type") in {"sapiens2", "rtmw"}:
            processor_names = {"preprocessor_config.json"}
        if config.get("model_type") == "jina_embeddings_v4":
            from oryn_engine.jina4_tasks import ADAPTER_WEIGHT, METADATA_NAMES

            if not {*METADATA_NAMES, ADAPTER_WEIGHT} <= set(selected):
                reasons.append("MODEL_SUBSET_INCOMPLETE")
            if not ("tokenizer.json" in selected or {"vocab.json", "merges.txt"} <= set(selected)):
                reasons.append("MODEL_TOKENIZER_MISSING")
        if config.get("model_type") in {"llava_next_video", "llava_next"}:
            from oryn_engine.llava_video import METADATA_NAMES

            if not {*METADATA_NAMES, "tokenizer.json"} <= set(selected):
                reasons.append("MODEL_PROCESSOR_MISSING")
        if config.get("model_type") == "neomme":
            from oryn_engine.neomme_retrieval import METADATA_NAMES

            if not {*METADATA_NAMES, "tokenizer.json"} <= set(selected):
                reasons.append("MODEL_PROCESSOR_MISSING")
        if task == "visual-document-retrieval" and config.get("architectures") == ["ColQwen3_5"]:
            from oryn_engine.colqwen35_tasks import METADATA_NAMES

            if not {*METADATA_NAMES, "tokenizer.json"} <= set(selected):
                reasons.append("MODEL_PROCESSOR_MISSING")
        from oryn_engine import bidirectional_embeddings, qwen_retrieval

        if (
            source_kind == "mlx"
            and (
                config.get("model_type") == "gemma4_unified"
                or config.get("model_type") == "gemma4"
                and task in {"image-text-to-text", "any-to-any"}
            )
            or task == "text-ranking"
            and config.get("architectures") == ["JinaForRanking"]
            or task == "token-classification"
            and config.get("model_type") == "pii_masking"
            or config.get("model_type") == "pplx_contextual_qwen3_5"
            or config.get("model_type") == "oryn_llama_turn"
        ) and not {"tokenizer_config.json", "tokenizer.json"} <= set(selected):
            reasons.append("MODEL_TOKENIZER_MISSING")

        if (
            task == "automatic-speech-recognition"
            and config.get("model_type") == "nemotron3_5_asr"
            and not {"tokenizer.json", "tokenizer_config.json", "processor_config.json"}
            <= set(selected)
        ):
            reasons.append("MODEL_PROCESSOR_MISSING")
        if config.get("model_type") == "audio8_asr_infinite" and not {
            "tokenizer.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "semantic_vad_heads.safetensors",
            "model.safetensors.index.json",
        } <= set(selected):
            reasons.append("MODEL_SUBSET_INCOMPLETE")
        if decision_tasks.supports(config, task) and not {
            *decision_tasks.METADATA_NAMES,
            "tokenizer.json",
            "readout.safetensors",
        } <= set(selected):
            reasons.append("MODEL_SUBSET_INCOMPLETE")

        if bidirectional_embeddings.supports(task, config) and not {
            *bidirectional_embeddings.METADATA_NAMES,
            "tokenizer.json",
        } <= set(selected):
            reasons.append("MODEL_TOKENIZER_MISSING")

        if qwen_retrieval.supports(task, config) and not set(
            qwen_retrieval.metadata_names(task)
        ) <= set(selected):
            reasons.append("MODEL_PROCESSOR_MISSING")
        if config.get("model_type") == "vision-encoder-decoder" and task == "image-to-text":
            processor_names = {"preprocessor_config.json"}
            if "tokenizer_config.json" not in selected:
                reasons.append("MODEL_TOKENIZER_MISSING")
        if config.get("model_type") in {"clip", "wav2vec2", "vit", "blip", "segformer"} and set(
            selected
        ) & {
            "flax_model.msgpack",
            "tf_model.h5",
        }:
            processor_names = {"preprocessor_config.json"}
        if task == "video-classification" and config.get("model_type") == "vjepa2":
            processor_names.add("video_preprocessor_config.json")
        if (
            not mlx_vad
            and (needs_processor or task in {"automatic-speech-recognition", "image-text-to-text"})
        ) and not processor_names.intersection(selected):
            reasons.append("MODEL_PROCESSOR_MISSING")
        # Generic Torch loading does not promise a quantized dependency/kernel implementation.
        if source_kind == "transformers" and any(
            config.get(name) is not None for name in ("quantization_config", "quantization")
        ):
            runtime.append("QUANTIZATION_UNSUPPORTED")
    if source_kind == "diffusers":
        from oryn_engine.diffusion_tasks import UNCONDITIONAL_PIPELINES

        flat_unconditional = (
            config.get("_class_name") in UNCONDITIONAL_PIPELINES
            and "config.json" in selected
            and "scheduler_config.json" in selected
            and not any(name.startswith("unet/") for name in selected)
        )
        data_fields = (
            {"sample_sigmas"} if config.get("_class_name") == "QwenImage21Pipeline" else set()
        )
        component_names = {
            name
            for name, value in config.items()
            if not name.startswith("_")
            and name not in data_fields
            and isinstance(value, list)
            and len(value) == 2
            and value != [None, None]
        }
        selected = tuple(
            name
            for name in selected
            if name == "model_index.json"
            or name.split("/", 1)[0] in component_names
            or (
                flat_unconditional
                and (
                    name in {"config.json", "scheduler_config.json"}
                    or (
                        "/" not in name
                        and name.startswith("diffusion_pytorch_model")
                        and name.endswith((".safetensors", ".json"))
                    )
                )
            )
        )
        selected = tuple(
            name
            for name in selected
            if not (
                any(
                    name.endswith(f".{variant}.safetensors")
                    and name.replace(f".{variant}.safetensors", ".safetensors") in selected
                    for variant in ("fp16", "non_ema")
                )
            )
        )
        if "model_index.json" not in selected or not isinstance(config.get("_class_name"), str):
            reasons.append("MODEL_CONFIG_MISSING")
        from oryn_engine.diffusion_tasks import validate_pipeline_index

        try:
            validate_pipeline_index(config)
        except ValueError:
            runtime.append("PIPELINE_COMPONENT_UNSUPPORTED")
        for name, component in config.items():
            if name in data_fields:
                # Validated by the pipeline contract above, including two-step grids.
                continue
            if name.startswith("_") or not isinstance(component, list) or component == [None, None]:
                continue
            if (
                len(component) != 2
                or (
                    component[0] not in {"diffusers", "transformers"}
                    and component != ["stable_diffusion", "StableDiffusionSafetyChecker"]
                )
                or not isinstance(component[1], str)
                or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", component[1])
                or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", name)
            ):
                runtime.append("CUSTOM_CODE_REQUIRED")
                continue
            assets = [file for file in selected if file.startswith(name + "/")]
            if flat_unconditional:
                assets = (
                    [f for f in selected if f == "scheduler_config.json"]
                    if name == "scheduler"
                    else [
                        f
                        for f in selected
                        if f == "config.json" or f.startswith("diffusion_pytorch_model")
                    ]
                )
            if not assets:
                reasons.append("MODEL_SUBSET_INCOMPLETE")
            klass = component[1]
            if "Tokenizer" in klass:
                if not any(
                    "tokenizer" in file or "vocab" in file or file.endswith((".model", ".spm"))
                    for file in assets
                ):
                    reasons.append("MODEL_TOKENIZER_MISSING")
            elif not any(
                word in klass for word in ("Scheduler", "Processor", "FeatureExtractor")
            ) and (
                not any(file.endswith(".safetensors") for file in assets)
                or ("config.json" if flat_unconditional else f"{name}/config.json") not in assets
            ):
                reasons.append("MODEL_SUBSET_INCOMPLETE")
    if config.get("auto_map"):
        runtime.append("CUSTOM_CODE_REQUIRED")
    return LayoutAssessment(
        source_kind, selected, tuple(sorted(set(reasons))), tuple(sorted(set(runtime)))
    )


def validate_weight_index(
    index: object, selected_files: set[str], *, expected_shards: set[str] | None = None
) -> None:
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError("Weight index is invalid.")
    mapping = index["weight_map"]
    if not mapping or len(mapping) > 1_000_000:
        raise ValueError("Weight index is invalid.")
    for tensor, name in mapping.items():
        if not isinstance(tensor, str) or not isinstance(name, str) or name not in selected_files:
            raise ValueError("Weight index references an unselected file.")
        if not name.endswith(".safetensors") or ".." in PurePosixPath(name).parts:
            raise ValueError("Weight index references an unsafe file.")
    groups = {}
    for name in set(mapping.values()):
        match = SHARD.fullmatch(name)
        if match:
            prefix, number, total = match.groups()
            count, numbers = groups.setdefault(prefix, (int(total), set()))
            if count != int(total) or not 1 <= int(number) <= count <= 10_000:
                raise ValueError("Weight index has inconsistent shard declarations.")
            numbers.add(int(number))
    if any(len(numbers) != count for count, numbers in groups.values()):
        raise ValueError("Weight index does not cover the complete declared shard graph.")
    if expected_shards is not None and not expected_shards.issubset(mapping.values()):
        raise ValueError("Weight index does not cover the complete selected shard graph.")
