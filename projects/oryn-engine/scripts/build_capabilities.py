"""Generate capability data from pinned installed libraries, then sign with an external build key.

The private key path is a build-only input. It is never copied into the artifact or runtime.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import re
import subprocess
from pathlib import Path

from oryn_engine.compatibility_plane import compile_registry
from oryn_engine.reviewed_implementations import build_binding
from oryn_engine.runtime_packs.llama_cpp_worker import VISION_LIBRARY
from oryn_engine.runtime_routing import canonical_digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-package", type=Path, required=True)
    parser.add_argument("--diffusers-package", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama-binary", type=Path, required=True)
    args = parser.parse_args()
    if args.llama_binary.name == "llama-mtmd-cli":
        parser.error(
            "--llama-binary requires the text entrypoint (llama-completion), not the vision CLI"
        )
    from transformers.models.auto import modeling_auto

    groups = {
        "image-text-to-text": (
            "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",
            "vision-language",
            "chat-v1",
        ),
        "text-generation": ("MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", "causal-lm", "chat-v1"),
        "text-classification": (
            "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES",
            "sequence-classification",
            "text-classification-v1",
        ),
        "token-classification": (
            "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES",
            "token-classification",
            "token-classification-v1",
        ),
        "question-answering": (
            "MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES",
            "question-answering",
            "question-answering-v1",
        ),
        "fill-mask": ("MODEL_FOR_MASKED_LM_MAPPING_NAMES", "masked-lm", "fill-mask-v1"),
        "translation": (
            "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES",
            "seq2seq",
            "translation-v1",
        ),
        "summarization": (
            "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES",
            "seq2seq",
            "summarization-v1",
        ),
        "image-classification": (
            "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES",
            "image-classification",
            "image-classification-v1",
        ),
        "object-detection": (
            "MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES",
            "object-detection",
            "object-detection-v1",
        ),
        "depth-estimation": (
            "MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES",
            "depth-estimation",
            "depth-estimation-v1",
        ),
        "audio-classification": (
            "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES",
            "audio-classification",
            "audio-classification-v1",
        ),
        "zero-shot-image-classification": (
            "MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES",
            "zero-shot-image-classification",
            "zero-shot-image-classification-v1",
        ),
        "text-ranking": (
            "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES",
            "sequence-classification",
            "reranking-v1",
        ),
    }
    generator_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    mapping_digest = hashlib.sha256(Path(modeling_auto.__file__).read_bytes()).hexdigest()
    transformers_vision_sources = {
        name: hashlib.sha256(
            (Path(modeling_auto.__file__).parents[2] / name).read_bytes()
        ).hexdigest()
        for name in (
            *[
                f"models/{family}/{kind}_{family}.py"
                for family in (
                    "bert",
                    "distilbert",
                    "modernbert",
                    "xlm_roberta",
                    "bart",
                    "m2m_100",
                    "dpt",
                    "bit",
                    "t5",
                    "mobilebert",
                    "owlvit",
                    "tapas",
                )
                for kind in ("modeling", "configuration")
            ],
            "models/dpt/image_processing_dpt.py",
            "image_processing_backends.py",
            "models/nllb/tokenization_nllb.py",
            "models/bart/tokenization_bart.py",
            "models/t5/tokenization_t5.py",
            "models/owlvit/image_processing_owlvit.py",
            "models/owlvit/processing_owlvit.py",
            "models/clip/tokenization_clip.py",
            "models/tapas/tokenization_tapas.py",
            "generation/utils.py",
            "audio_utils.py",
            "feature_extraction_utils.py",
            "feature_extraction_sequence_utils.py",
            "models/whisper/feature_extraction_whisper.py",
            "models/whisper/modeling_whisper.py",
            "models/whisper/configuration_whisper.py",
            "models/nemotron3_5_asr/configuration_nemotron3_5_asr.py",
            "models/nemotron3_5_asr/modeling_nemotron3_5_asr.py",
            "models/nemotron3_5_asr/generation_nemotron3_5_asr.py",
            "models/nemotron_asr_streaming/configuration_nemotron_asr_streaming.py",
            "models/nemotron_asr_streaming/modeling_nemotron_asr_streaming.py",
            "models/nemotron_asr_streaming/generation_nemotron_asr_streaming.py",
            "models/nemotron_asr_streaming/feature_extraction_nemotron_asr_streaming.py",
            "models/qwen2/modeling_qwen2.py",
            "models/qwen2/configuration_qwen2.py",
            "models/qwen2/tokenization_qwen2.py",
            "models/llama/modeling_llama.py",
            "models/llama/configuration_llama.py",
            "cache_utils.py",
            "masking_utils.py",
            "models/voxtral_realtime/modeling_voxtral_realtime.py",
            "models/voxtral_realtime/configuration_voxtral_realtime.py",
            "models/voxtral_realtime/feature_extraction_voxtral_realtime.py",
            "models/parakeet/modeling_parakeet.py",
            "models/parakeet/generation_parakeet.py",
            "models/parakeet/tokenization_parakeet.py",
            "models/parakeet/feature_extraction_parakeet.py",
            "models/parakeet/configuration_parakeet.py",
            "models/sam3/modeling_sam3.py",
            "models/sam3/configuration_sam3.py",
            "models/tipsv2/modeling_tipsv2.py",
            "models/tipsv2/configuration_tipsv2.py",
            "models/qwen2_5_vl/modeling_qwen2_5_vl.py",
            "models/qwen2_5_vl/configuration_qwen2_5_vl.py",
        )
    }

    def entry(identifier, architecture, task, family, contract, kind="transformers"):
        media = task in {
            "image-classification",
            "object-detection",
            "depth-estimation",
            "audio-classification",
            "zero-shot-image-classification",
        }
        return {
            "id": identifier,
            "sourceKind": kind,
            "architecture": architecture,
            "task": task,
            "formats": ["safetensors", "json"] + ([] if media else ["tokenizer"]),
            "familyAdapter": {"id": f"oryn.family.{family}", "version": "1.0.0"},
            "taskAdapter": {"id": f"oryn.task.{contract}", "version": "1.0.0"},
            "conformance": "untested",
        }

    entries = []
    for task, (mapping, family, contract) in groups.items():
        classes = sorted(
            {
                name
                for value in getattr(modeling_auto, mapping).values()
                for name in (value if isinstance(value, tuple) else (value,))
            }
        )
        entries.extend(
            entry(f"transformers.{task}.{index}", name, task, family, contract)
            for index, name in enumerate(classes)
        )
    entries.append(
        {
            **entry(
                "transformers.ct2.whisper",
                "CTranslate2Whisper",
                "automatic-speech-recognition",
                "automatic-speech-recognition",
                "transcription-v1",
            ),
            "formats": ["ctranslate2", "json", "tokenizer"],
        }
    )
    from oryn_engine.shared_tasks import TASK_CONTRACTS, TASK_FAMILIES

    task_mappings = {
        "translation": (
            "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES",
            "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",
            "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES",
        ),
        "question-answering": ("MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES",),
        "feature-extraction": ("MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",),
        "text-ranking": ("MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",),
        "zero-shot-image-classification": (
            "MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES",
        ),
        "text-classification": ("MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES",),
        "summarization": ("MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES",),
        "any-to-any": ("MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",),
        "pose-estimation": ("MODEL_FOR_POSE_ESTIMATION_MAPPING_NAMES",),
        "image-classification": ("MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES",),
        "automatic-speech-recognition": (
            "MODEL_FOR_CTC_MAPPING_NAMES",
            "MODEL_FOR_RNNT_MAPPING_NAMES",
            "MODEL_FOR_TDT_MAPPING_NAMES",
        ),
        "audio-text-to-text": (
            "MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES",
            "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES",
        ),
        "time-series-forecasting": ("MODEL_FOR_TIME_SERIES_PREDICTION_MAPPING_NAMES",),
        "visual-document-retrieval": ("MODEL_FOR_PRETRAINING_MAPPING_NAMES",),
        "table-question-answering": (
            "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES",
            "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES",
        ),
        "document-question-answering": ("MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES",),
        "keypoint-detection": ("MODEL_FOR_KEYPOINT_DETECTION_MAPPING_NAMES",),
        "video-classification": ("MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES",),
        "video-text-to-text": ("MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",),
        "zero-shot-classification": (
            "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES",
            "MODEL_MAPPING_NAMES",
        ),
        "image-feature-extraction": ("MODEL_MAPPING_NAMES",),
        "image-segmentation": (
            "MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES",
            "MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES",
        ),
        "mask-generation": ("MODEL_FOR_MASK_GENERATION_MAPPING_NAMES",),
        "zero-shot-object-detection": ("MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES",),
        "image-to-text": ("MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",),
        "visual-question-answering": (
            "MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES",
            "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",
        ),
    }
    for task, mappings in task_mappings.items():
        classes = sorted(
            {
                name
                for mapping in mappings
                for family, value in getattr(modeling_auto, mapping).items()
                if task not in TASK_FAMILIES or family in TASK_FAMILIES[task]
                for name in (value if isinstance(value, tuple) else (value,))
                if task != "zero-shot-classification"
                or mapping != "MODEL_MAPPING_NAMES"
                or name in {"Qwen3_5Model", "Gemma4TextModel"}
            }
        )
        if task == "translation":
            from oryn_engine.instruction_translation import (
                ARCHITECTURES as TRANSLATION_ARCHITECTURES,
            )

            classes = sorted(set(classes) & {"MarianMTModel", *TRANSLATION_ARCHITECTURES.values()})
        if task == "visual-document-retrieval":
            # The pinned native retrieval class is exported without an Auto mapping.
            from transformers import NeoMMEForRetrieval

            classes = sorted({*classes, NeoMMEForRetrieval.__name__})
        for index, name in enumerate(classes):
            prefix = (
                "transformers.shared"
                if task
                in {"image-classification", "feature-extraction", "text-ranking", "translation"}
                else "transformers"
            )
            item = entry(f"{prefix}.{task}.{index}", name, task, task, TASK_CONTRACTS[task])
            if task in {"summarization", "text-classification", "zero-shot-image-classification"}:
                item["id"] = f"transformers.flax.{task}.{index}"
                item["formats"] = ["flax-msgpack", "json", "tokenizer"]
            if task in {
                "image-feature-extraction",
                "image-segmentation",
                "mask-generation",
                "keypoint-detection",
                "video-classification",
                "time-series-forecasting",
                "pose-estimation",
            }:
                item["formats"] = ["safetensors", "json"]
            if not (
                task == "question-answering"
                or task == "translation"
                and name == "MarianMTModel"
                or task == "text-classification"
                and name == "RobertaForSequenceClassification"
            ):
                entries.append(item)
            from oryn_engine.hdf5_tasks import ARCHITECTURES as HDF5_ARCHITECTURES

            if any(t == task and n == name for (t, _), n in HDF5_ARCHITECTURES.items()):
                hdf5_family = next(
                    f for (t, f), n in HDF5_ARCHITECTURES.items() if t == task and n == name
                )
                entries.append(
                    {
                        **item,
                        "id": f"transformers.hdf5.{task}.{hdf5_family}",
                        "formats": ["hdf5-weights", "json"]
                        + (
                            []
                            if task in {"image-feature-extraction", "image-segmentation"}
                            else ["tokenizer"]
                        ),
                    }
                )
            if task == "automatic-speech-recognition" and name == "Wav2Vec2ForCTC":
                entries.append(
                    {
                        **item,
                        "id": f"transformers.flax.{task}.{index}",
                        "formats": ["flax-msgpack", "json", "tokenizer"],
                    }
                )
        if task == "summarization":
            # Older T5 Flax exports declare the base class. The closed loader
            # requires its complete graph and a tied shared-embedding LM head;
            # no missing head is initialized or inferred from repository code.
            base = modeling_auto.MODEL_MAPPING_NAMES.get("t5")
            if base == "T5Model":
                item = entry(
                    "transformers.flax.summarization.tied-t5-base",
                    base,
                    task,
                    task,
                    TASK_CONTRACTS[task],
                )
                item["formats"] = ["flax-msgpack", "json", "tokenizer"]
                entries.append(item)
        if task == "keypoint-detection":
            # Auto loading keys on model_type, including checkpoints predating the
            # upstream SuperPointModel -> ForKeypointDetection class rename.
            for family in sorted(TASK_FAMILIES[task]):
                if family in modeling_auto.MODEL_FOR_KEYPOINT_DETECTION_MAPPING_NAMES:
                    item = entry(
                        f"transformers.{task}.family.{family}",
                        family,
                        task,
                        task,
                        TASK_CONTRACTS[task],
                    )
                    item["formats"] = ["safetensors", "json"]
                    entries.append(item)

    from oryn_engine.numeric_archive_tasks import ARCHITECTURES as NUMERIC_ARCHITECTURES

    for (task, family), architecture in NUMERIC_ARCHITECTURES.items():
        mapping_names = (groups[task][0],) if task in groups else task_mappings[task]
        assert any(
            getattr(modeling_auto, name).get(family) == architecture for name in mapping_names
        )
        identifier_family = family.replace("_", "-")
        item = entry(
            f"transformers.numeric-archive.{task}.{identifier_family}",
            architecture,
            task,
            f"numeric-archive-{task}",
            TASK_CONTRACTS[task],
        )
        item["formats"] = ["torch-numeric-zip", "json"] + (
            [] if task in {"depth-estimation", "video-classification"} else ["tokenizer"]
        )
        entries.append(item)

    # SAM 2's image Auto mapping drops the temporal modules. Load its complete
    # native video model for the bounded first-frame image-mask invocation.
    sam2_native = modeling_auto.MODEL_MAPPING_NAMES["sam2_video"]
    entries.append(
        entry(
            "transformers.mask-generation.sam2-video",
            sam2_native,
            "mask-generation",
            "mask-generation",
            TASK_CONTRACTS["mask-generation"],
        )
    )
    entries[-1]["formats"] = ["safetensors", "json"]

    from transformers import Sam3ViTModel

    assert Sam3ViTModel.__name__ == "Sam3ViTModel"
    item = entry(
        "transformers.pixai.image-classification",
        "PixAITagger",
        "image-classification",
        "pixai",
        "image-classification-v1",
    )
    item["formats"] = ["safetensors", "json"]
    entries.append(item)

    from oryn_engine.native_rtmw.model import RTMWModel

    assert RTMWModel.__name__ == "RTMWModel"
    item = entry(
        "transformers.rtmw.keypoint-detection",
        "rtmw",
        "keypoint-detection",
        "rtmw",
        "keypoint-detection-v1",
    )
    item["formats"] = ["safetensors", "json"]
    entries.append(item)

    from open_clip.model import CLIP

    assert CLIP.__name__ == "CLIP"
    item = entry(
        "transformers.openclip.zero-shot-image-classification",
        "OpenCLIPModel",
        "zero-shot-image-classification",
        "zero-shot-image-classification",
        "zero-shot-image-classification-v1",
    )
    item["formats"] = ["safetensors", "json"]
    entries.append(item)

    # Chronos ships native classes outside the Transformers Auto mappings. Bind
    # the installed implementation and its source hashes into the same authority.
    from chronos.chronos2.model import Chronos2Model
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    from timesfm3.torch.model import TimesFM3Torch

    from oryn_engine.native_aurora.modeling_aurora import AuroraForPrediction
    from oryn_engine.native_exaone_demand.model import EXAONEDemand
    from oryn_engine.native_yug.model import Yug_Model

    for native in (
        ChronosBoltModelForForecasting,
        Chronos2Model,
        TimesFM3Torch,
        EXAONEDemand,
        Yug_Model,
        AuroraForPrediction,
    ):
        item = entry(
            f"transformers.forecasting.{native.__name__.lower().replace('_', '-')}",
            native.__name__,
            "time-series-forecasting",
            "time-series-forecasting",
            "time-series-forecasting-v1",
        )
        item["formats"] = ["safetensors", "json"]
        entries.append(item)

    from oryn_engine.native_birefnet.model import BiRefNet
    from oryn_engine.native_isnet.model import ISNetDIS

    item = entry(
        "transformers.segmentation.birefnet",
        BiRefNet.__name__,
        "image-segmentation",
        "image-segmentation",
        "image-segmentation-v1",
    )
    item["formats"] = ["safetensors", "json"]
    entries.append(item)

    from depth_anything_3.model.da3 import DepthAnything3Net, NestedDepthAnything3Net

    item = entry(
        "transformers.segmentation.isnet",
        ISNetDIS.__name__,
        "image-segmentation",
        "image-segmentation",
        "image-segmentation-v1",
    )
    item["formats"] = ["safetensors", "json"]
    entries.append(item)

    for native in (DepthAnything3Net, NestedDepthAnything3Net):
        item = entry(
            f"transformers.depth.{native.__name__.lower()}",
            native.__name__,
            "depth-estimation",
            "depth-estimation",
            "depth-estimation-v1",
        )
        item["formats"] = ["safetensors", "json"]
        entries.append(item)

    contextual = entry(
        "transformers.embeddings.pplx-contextual",
        "PplxContextualModel",
        "feature-extraction",
        "embeddings",
        "embeddings-v1",
    )
    contextual["formats"] = ["safetensors", "json", "tokenizer"]
    entries.append(contextual)

    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D

    colqwen3 = entry(
        "transformers.visual-retrieval.colqwen3",
        "ColQwen3",
        "visual-document-retrieval",
        "visual-document-retrieval",
        "visual-document-retrieval-v1",
    )
    colqwen3["formats"] = ["safetensors", "json", "tokenizer"]
    entries.append(colqwen3)

    colqwen35 = entry(
        "transformers.visual-retrieval.colqwen35",
        "ColQwen3_5",
        "visual-document-retrieval",
        "visual-document-retrieval",
        "visual-document-retrieval-v1",
    )
    colqwen35["formats"] = ["safetensors", "json", "tokenizer"]
    entries.append(colqwen35)

    audio8 = entry(
        "transformers.asr.audio8",
        "Audio8ASRInfiniteForConditionalGeneration",
        "automatic-speech-recognition",
        "automatic-speech-recognition",
        "transcription-v1",
    )
    audio8["formats"] = ["safetensors", "json", "tokenizer"]
    entries.append(audio8)
    turn = entry(
        "transformers.classification.llama-turn",
        "OrynLlamaTurnForClassification",
        "text-classification",
        "text-classification",
        "text-classification-v1",
    )
    turn["formats"] = ["safetensors", "json", "tokenizer"]
    entries.append(turn)

    jina4 = entry(
        "transformers.visual-retrieval.jina4",
        "JinaEmbeddingsV4Model",
        "visual-document-retrieval",
        "visual-document-retrieval",
        "visual-document-retrieval-v1",
    )
    jina4["formats"] = ["safetensors", "json", "tokenizer"]
    entries.append(jina4)

    from oryn_engine.tabpfnmix_tasks import native_class

    FoundationTransformer = native_class()
    from tabpfn.architectures.tabpfn_v3_5 import TabPFNV3p5

    from oryn_engine.tabular_tasks import TASK_CONTRACTS as tabular_contracts

    for task, contract in tabular_contracts.items():
        for native in (TabPFNV3p5, Tab2D, FoundationTransformer):
            item = entry(
                f"transformers.{task}.{native.__name__.lower()}",
                native.__name__,
                task,
                task,
                contract,
            )
            item["formats"] = ["safetensors", "json"]
            entries.append(item)

    # These tasks have framework-specific semantics beyond universal AutoModel behavior.
    for index, (architecture, task, family, contract) in enumerate(
        [
            (
                "WhisperForConditionalGeneration",
                "automatic-speech-recognition",
                "speech-seq2seq",
                "transcription-v1",
            ),
            (
                "Qwen3ASRForConditionalGeneration",
                "automatic-speech-recognition",
                "speech-seq2seq",
                "transcription-v1",
            ),
            (
                "Qwen3ASRForTokenClassification",
                "forced-alignment",
                "audio-token-classification",
                "forced-alignment-v1",
            ),
            ("MusicgenForConditionalGeneration", "text-to-audio", "musicgen", "text-to-music-v1"),
            ("Qwen3ForCausalLM", "feature-extraction", "causal-lm", "embeddings-v1"),
            ("Qwen3ForCausalLM", "text-ranking", "causal-lm", "reranking-v1"),
        ]
    ):
        entries.append(
            entry(f"transformers.specialized.{index}", architecture, task, family, contract)
        )
    entries.append(
        entry(
            "transformers.shared.feature-extraction.ministral3",
            modeling_auto.MODEL_MAPPING_NAMES["ministral3"],
            "feature-extraction",
            "feature-extraction",
            "embeddings-v1",
        )
    )
    giga = entry(
        "transformers.shared.feature-extraction.qwen3-bidirectional",
        "Qwen3BidirectionalModel",
        "feature-extraction",
        "feature-extraction",
        "embeddings-v1",
    )
    giga["formats"] = ["safetensors", "json", "tokenizer"]
    entries.append(giga)
    entries.append(
        entry(
            "transformers.shared.speech.sopro",
            "OrynSoproForSpeechSynthesis",
            "text-to-speech",
            "text-to-speech",
            "speech-synthesis-v1",
        )
    )
    entries.append(
        entry(
            "transformers.shared.speech.omnivoice",
            "OmniVoice",
            "text-to-speech",
            "text-to-speech",
            "speech-synthesis-v1",
        )
    )
    entries.append(
        entry(
            "transformers.shared.speech.breeze",
            "BreezeForConditionalGeneration",
            "text-to-speech",
            "text-to-speech",
            "speech-synthesis-v1",
        )
    )
    entries.append(
        entry(
            "transformers.shared.llava-video-qwen",
            "LlavaQwenForCausalLM",
            "video-text-to-text",
            "video-text-to-text",
            "video-question-answering-v1",
        )
    )
    entries.append(
        entry(
            "transformers.shared.videollama3",
            "Videollama3Qwen2ForCausalLM",
            "video-text-to-text",
            "video-text-to-text",
            "video-question-answering-v1",
        )
    )
    entries.append(
        entry(
            "transformers.shared.molparser",
            "MolParserVisionEncoderDecoderModel",
            "image-to-text",
            "image-to-text",
            "image-to-text-v1",
        )
    )
    entries.append(
        entry(
            "transformers.shared.qwen-codec",
            "Qwen3TTSTokenizerV2Model",
            "audio-to-audio",
            "audio-to-audio",
            "audio-transformation-v1",
        )
    )
    entries[-1]["formats"] = ["safetensors", "json"]
    entries.append(
        entry(
            "transformers.shared.tiger-dnr",
            "TIGERDNR",
            "audio-to-audio",
            "audio-to-audio",
            "audio-transformation-v1",
        )
    )
    entries[-1]["formats"] = ["safetensors", "json"]
    entries.append(
        entry(
            "transformers.shared.reuse",
            "SEMamba",
            "audio-to-audio",
            "audio-to-audio",
            "audio-transformation-v1",
        )
    )
    entries[-1]["formats"] = ["safetensors", "json"]
    entries.append(
        entry(
            "transformers.shared.reuse-causal",
            "SEMambaDecoderList",
            "audio-to-audio",
            "audio-to-audio",
            "audio-transformation-v1",
        )
    )
    entries[-1]["formats"] = ["safetensors", "json"]
    entries.append(
        entry(
            "transformers.shared.mpsenet",
            "MPSENet",
            "audio-to-audio",
            "audio-to-audio",
            "audio-transformation-v1",
        )
    )
    entries[-1]["formats"] = ["safetensors", "json"]
    from oryn_engine.speechx_tasks import ARCHITECTURE as SPEECHX_ARCHITECTURE
    from oryn_engine.speechx_tasks import FAMILY as SPEECHX_FAMILY
    from oryn_engine.wav2vec_heads import ARCHITECTURES as AUDIO_HEAD_ARCHITECTURES

    AUDIO_HEAD_ARCHITECTURES = AUDIO_HEAD_ARCHITECTURES | {SPEECHX_FAMILY: SPEECHX_ARCHITECTURE}
    for family, architecture in AUDIO_HEAD_ARCHITECTURES.items():
        audio_head = entry(
            "transformers.shared.audio-classification." + family.replace("_", "-"),
            architecture,
            "audio-classification",
            "audio-classification",
            "audio-classification-v1",
        )
        audio_head["formats"] = ["safetensors", "json"]
        entries.append(audio_head)
    entries.append(
        entry(
            "transformers.shared.text-ranking.jina-v3",
            "JinaForRanking",
            "text-ranking",
            "text-ranking",
            "reranking-v1",
        )
    )
    entries.append(
        entry(
            "transformers.shared.token-classification.pii",
            "PiiMaskingModel",
            "token-classification",
            "token-classification",
            "token-classification-v1",
        )
    )
    encoders = {"bert", "distilbert", "roberta", "xlm-roberta", "mpnet", "modernbert"}
    for index, (model_type, architecture) in enumerate(modeling_auto.MODEL_MAPPING_NAMES.items()):
        if model_type in encoders and isinstance(architecture, str):
            entries.append(
                entry(
                    f"transformers.encoder.{index}",
                    architecture,
                    "feature-extraction",
                    "encoder",
                    "embeddings-v1",
                )
            )
    mlx_inputs = {}
    mlx_types = []
    for path in sorted((args.mlx_package / "models").glob("*.py")):
        tree = ast.parse(path.read_text())
        names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        if {"Model", "ModelArgs"}.issubset(names):
            mlx_types.append(path.stem)
            mlx_inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    mlx_entries = [
        entry(f"mlx.chat.{index}", name, "text-generation", "mlx-lm", "chat-v1", "mlx")
        for index, name in enumerate(mlx_types)
    ]
    mlx_types.append("kolibri1")
    mlx_entries.append(
        entry("mlx.chat.kolibri1", "kolibri1", "text-generation", "mlx-lm", "chat-v1", "mlx")
    )
    mlx_types.append("prism_hadamard_qwen35")
    mlx_entries.append(
        entry(
            "mlx.chat.prism-hadamard-qwen35",
            "prism_hadamard_qwen35",
            "text-generation",
            "mlx-lm",
            "chat-v1",
            "mlx",
        )
    )
    import importlib.util
    from importlib.metadata import version

    from oryn_engine.mlx_vision import REVIEWED_FAMILIES, REVIEWED_MLX_VLM

    vision_package = Path(importlib.util.find_spec("mlx_vlm").origin).parent
    if version("mlx-vlm") != REVIEWED_MLX_VLM:
        raise ValueError("The reviewed MLX vision build is required")
    vision_sources = {
        path.relative_to(vision_package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(vision_package.rglob("*.py"))
    }
    native_vision_types = {
        path.name
        for path in (vision_package / "models").iterdir()
        if path.is_dir()
        and (path / (path.name + ".py")).is_file()
        and "class Model(" in (path / (path.name + ".py")).read_text()
    }
    if not native_vision_types >= REVIEWED_FAMILIES:
        raise ValueError("A reviewed MLX vision family is absent from the native build")
    if "gemma4_unified" not in native_vision_types:
        raise ValueError("The reviewed Gemma 4 Unified text graph is absent")
    mlx_types.append("gemma4_unified")
    mlx_entries.append(
        entry(
            "mlx.chat.gemma4-unified",
            "gemma4_unified",
            "text-generation",
            "mlx-lm",
            "chat-v1",
            "mlx",
        )
    )
    mlx_entries.extend(
        entry(
            f"mlx.vision.{name.replace('_', '-')}",
            name,
            "image-text-to-text",
            "mlx-vision",
            "chat-v1",
            "mlx",
        )
        for name in sorted(native_vision_types & REVIEWED_FAMILIES)
    )
    if "gemma4" not in native_vision_types:
        raise ValueError("The reviewed Gemma 4 multimodal graph is absent")
    for task, contract in (
        ("image-text-to-text", "chat-v1"),
        ("any-to-any", "multimodal-generation-v1"),
    ):
        mlx_entries.append(
            entry("mlx.gemma4." + task, "gemma4", task, "mlx-gemma4", contract, "mlx")
        )
    from oryn_engine import mlx_vad

    vad_entry = entry(
        "mlx.vad.silero-v5",
        mlx_vad.MODEL_TYPE,
        "voice-activity-detection",
        "mlx-vad",
        "voice-activity-detection-v1",
        "mlx",
    )
    vad_entry["formats"] = ["safetensors", "json"]
    mlx_entries.append(vad_entry)
    from oryn_engine import (
        audio_understanding,
        flax_weights,
        mlx_vision,
        qwen_retrieval,
        trocr_tasks,
    )

    flax_digest = hashlib.sha256(Path(flax_weights.__file__).read_bytes()).hexdigest()
    native_sources = {
        **{
            name: hashlib.sha256((Path(mlx_vision.__file__).parent / name).read_bytes()).hexdigest()
            for name in (
                "runtime_packs/onnx_worker.py",
                "onnx_vision.py",
                "onnx_classifier.py",
                "data/imagenet-efficientnet-lite4-labels.json",
                "jina_onnx.py",
                "onnx_tasks.py",
                "onnx_tdt.py",
                "onnx_graph.py",
                "onnx_nbits.py",
                "onnx_inspection.py",
                "tekken_tokenizer.py",
                "voxtral_tasks.py",
                "model_layouts.py",
                "model_inspection.py",
                "task_profiles.py",
                "decision_tasks.py",
                "gemma_decision.py",
                "mlx_gemma.py",
                "mlx_gemma4.py",
                "depth3_tasks.py",
                "llava_next_tasks.py",
                "coverage_tasks.py",
                "audio8_tasks.py",
                "turn_classification.py",
                "speechx_tasks.py",
                "sopro_tasks.py",
                "omnivoice_tasks.py",
                "breeze_tasks.py",
                "molparser_tasks.py",
                "qwen_codec_tasks.py",
                "mpsenet_tasks.py",
                "native_qwen_bpe.py",
                "llava_video_qwen.py",
                *[
                    p.relative_to(Path(mlx_vision.__file__).parent).as_posix()
                    for p in sorted(
                        (Path(mlx_vision.__file__).parent / "llava_video_qwen_assets").rglob("*")
                    )
                    if p.is_file()
                ],
                "videollama3_tasks.py",
                *[
                    p.relative_to(Path(mlx_vision.__file__).parent).as_posix()
                    for p in sorted(
                        (Path(mlx_vision.__file__).parent / "videollama3_assets").rglob("*")
                    )
                    if p.is_file()
                ],
                "tiger_tasks.py",
                *[
                    p.relative_to(Path(mlx_vision.__file__).parent).as_posix()
                    for p in sorted((Path(mlx_vision.__file__).parent / "native_tiger").rglob("*"))
                    if p.is_file() and p.suffix != ".pyc" and "__pycache__" not in p.parts
                ],
                "reuse_tasks.py",
                "native_reuse.py",
                "reuse_assets/offline.json",
                "reuse_assets/causal.json",
                "reuse_assets/provenance.json",
                *[
                    p.relative_to(Path(mlx_vision.__file__).parent).as_posix()
                    for p in sorted(
                        (Path(mlx_vision.__file__).parent / "native_mpsenet").rglob("*")
                    )
                    if p.is_file() and p.suffix != ".pyc" and "__pycache__" not in p.parts
                ],
                "layoutlm_tasks.py",
                *[
                    p.relative_to(Path(mlx_vision.__file__).parent).as_posix()
                    for p in sorted(
                        (Path(mlx_vision.__file__).parent / "layoutlm_assets").rglob("*")
                    )
                    if p.is_file()
                ],
                *(
                    p.relative_to(Path(mlx_vision.__file__).parent).as_posix()
                    for p in sorted(
                        (Path(mlx_vision.__file__).parent / "molparser_assets").rglob("*")
                    )
                    if p.is_file() and p.suffix in {".json", ".txt"}
                ),
                *(
                    p.relative_to(Path(mlx_vision.__file__).parent).as_posix()
                    for p in sorted((Path(mlx_vision.__file__).parent / "native_breeze").rglob("*"))
                    if p.is_file() and p.suffix in {".py", ".json", ".txt"}
                ),
                *(
                    p.relative_to(Path(mlx_vision.__file__).parent).as_posix()
                    for p in sorted(
                        (Path(mlx_vision.__file__).parent / "native_omnivoice").rglob("*")
                    )
                    if p.is_file() and p.suffix in {".py", ".json", ".txt"}
                ),
                *(
                    p.relative_to(Path(mlx_vision.__file__).parent).as_posix()
                    for p in sorted((Path(mlx_vision.__file__).parent / "native_sopro").rglob("*"))
                    if p.is_file() and p.suffix in {".py", ".json", ".txt"}
                ),
                "minimax_music.py",
                "minimax_h3.py",
                "modular_components.py",
                "video_audio_artifacts.py",
                "flat_ddpm.py",
                "numeric_unet2d.py",
                "audio8_model.py",
                "third_party/Audio8-LICENSE",
                "rnnt_tasks.py",
                "ct2_tasks.py",
                "ct2_weights.py",
                "parakeet_tasks.py",
                "contextual_embeddings.py",
                "colqwen35_tasks.py",
                "visual_retrieval.py",
                "structured_outputs.py",
                "runtime_packs/transformers_worker.py",
                "exaone_demand.py",
                "native_exaone_demand/__init__.py",
                "native_exaone_demand/adapter.py",
                "native_exaone_demand/config.py",
                "native_exaone_demand/layers.py",
                "native_exaone_demand/model.py",
                "native_exaone_demand/provenance.json",
                "native_exaone_demand/LICENSE",
                "native_exaone_demand/NOTICE.md",
                "native_exaone_demand/DEMAND-NOTICE.md",
                "yug_tasks.py",
                "native_yug/__init__.py",
                "native_yug/model.py",
                "native_yug/provenance.json",
                "native_yug/LICENSE",
                "native_yug/NOTICE",
                "aurora_tasks.py",
                "openclip_tasks.py",
                "birefnet_tasks.py",
                "isnet_tasks.py",
                "pixai_tasks.py",
                "tips_tasks.py",
                "rtmw_tasks.py",
                "jina4_tasks.py",
                "native_rtmw/__init__.py",
                "native_rtmw/model.py",
                "native_rtmw/LICENSE",
                "native_rtmw/NOTICE.md",
                "native_rtmw/provenance.json",
                "native_isnet/__init__.py",
                "native_isnet/model.py",
                "native_isnet/LICENSE",
                "native_isnet/NOTICE.md",
                "native_isnet/provenance.json",
                "native_birefnet/__init__.py",
                "native_birefnet/model.py",
                "native_birefnet/LICENSE",
                "native_birefnet/NOTICE.md",
                "native_birefnet/provenance.json",
                "native_aurora/__init__.py",
                "native_aurora/configuration_aurora.py",
                "native_aurora/flow_loss.py",
                "native_aurora/modality_connector.py",
                "native_aurora/modeling_aurora.py",
                "native_aurora/prototype_retriever.py",
                "native_aurora/util_functions.py",
                "native_aurora/bert_config/config.json",
                "native_aurora/vit_config/config.json",
                "native_aurora/vit_config/preprocessor_config.json",
                "native_aurora/provenance.json",
                "native_aurora/README.md",
                "native_aurora/NOTICE.md",
                "runtime_packs/mlx_lm_worker.py",
                "jina_ranking.py",
                "pii_tasks.py",
                "wav2vec_heads.py",
                "bioes.py",
                "models.py",
                "mlx_gpt_oss.py",
                "mlx_prism.py",
                "mlx_bolding.py",
                "mlx_kolibri.py",
                "native_kolibri.py",
                "third_party/Kolibri-MLX-LICENSE",
                "third_party/Kolibri-MLX-NOTICE",
                "third_party/Prism-MLX-LICENSE",
                "gguf_translation.py",
                "runtime_packs/llama_cpp_worker.py",
                "runners.py",
            )
        },
        "giga_embeddings.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "giga_embeddings.py").read_bytes()
        ).hexdigest(),
        "colqwen3_tasks.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "colqwen3_tasks.py").read_bytes()
        ).hexdigest(),
        "timesfm3_tasks.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "timesfm3_tasks.py").read_bytes()
        ).hexdigest(),
        "forecast_tasks.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "forecast_tasks.py").read_bytes()
        ).hexdigest(),
        "shared_tasks.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "shared_tasks.py").read_bytes()
        ).hexdigest(),
        "mlx_vad.py": hashlib.sha256(Path(mlx_vad.__file__).read_bytes()).hexdigest(),
        "flax_weights.py": flax_digest,
        "sapiens2_pose.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "sapiens2_pose.py").read_bytes()
        ).hexdigest(),
        "bidirectional_embeddings.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "bidirectional_embeddings.py").read_bytes()
        ).hexdigest(),
        "llava_video.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "llava_video.py").read_bytes()
        ).hexdigest(),
        "neomme_retrieval.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "neomme_retrieval.py").read_bytes()
        ).hexdigest(),
        "model_support.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "model_support.py").read_bytes()
        ).hexdigest(),
        "runners.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "runners.py").read_bytes()
        ).hexdigest(),
        "model_metadata.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "model_metadata.py").read_bytes()
        ).hexdigest(),
        "generated_capabilities.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "generated_capabilities.py").read_bytes()
        ).hexdigest(),
        **{
            name: hashlib.sha256((Path(mlx_vision.__file__).parent / name).read_bytes()).hexdigest()
            for name in (
                "numeric_state_dict.py",
                "numeric_torch_archive.py",
                "numeric_archive_tasks.py",
                "numeric_seq2seq.py",
                "numeric_dpt.py",
                "numeric_mobilebert.py",
                "numeric_owlvit.py",
                "numeric_tapas.py",
                "numeric_coverage.py",
                "numeric_layoutlmv2.py",
                "audit_metadata.py",
            )
        },
        "numeric_hdf5.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "numeric_hdf5.py").read_bytes()
        ).hexdigest(),
        "hdf5_tasks.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "hdf5_tasks.py").read_bytes()
        ).hexdigest(),
        "hdf5_blip.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "hdf5_blip.py").read_bytes()
        ).hexdigest(),
        "hdf5_segformer.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "hdf5_segformer.py").read_bytes()
        ).hexdigest(),
        "instruction_translation.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "instruction_translation.py").read_bytes()
        ).hexdigest(),
        "hunyuan_translation.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "hunyuan_translation.py").read_bytes()
        ).hexdigest(),
        "qwen_translation.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "qwen_translation.py").read_bytes()
        ).hexdigest(),
        "hdf5_marian.py": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "hdf5_marian.py").read_bytes()
        ).hexdigest(),
        "data/sapiens2-joints.json": hashlib.sha256(
            (Path(mlx_vision.__file__).parent / "data/sapiens2-joints.json").read_bytes()
        ).hexdigest(),
        "qwen_retrieval.py": hashlib.sha256(Path(qwen_retrieval.__file__).read_bytes()).hexdigest(),
        "trocr_tasks.py": hashlib.sha256(Path(trocr_tasks.__file__).read_bytes()).hexdigest(),
        "mlx_vision.py": hashlib.sha256(Path(mlx_vision.__file__).read_bytes()).hexdigest(),
        "audio_understanding.py": hashlib.sha256(
            Path(audio_understanding.__file__).read_bytes()
        ).hexdigest(),
    }

    def source(identifier, foundation, capabilities, digest):
        return {
            "schemaVersion": "oryn-capability-source/1",
            "provider": {"id": identifier, "version": "1.0.0", "generatorDigest": generator_digest},
            "foundation": {
                "id": foundation,
                "version": "1.2.0" if foundation == "oryn.runtime.pytorch" else "1.0.0",
                "digest": digest,
            },
            "capabilities": capabilities,
        }

    diffusers_mapping = args.diffusers_package / "pipelines" / "auto_pipeline.py"
    diffusion = {}
    for node in ast.parse(diffusers_mapping.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else ""
            if name in {"AUTO_TEXT2IMAGE_PIPELINES_MAPPING", "AUTO_IMAGE2IMAGE_PIPELINES_MAPPING"}:
                diffusion[name] = {
                    item.elts[0].value: item.elts[1].id for item in node.value.args[0].elts
                }
    # Other upstream pipelines require extra controls or unvalidated memory profiles.
    # Emit only families for which Oryn implements the full bounded input contract.
    diffusers_entries = []
    for task, mapping in [
        ("text-to-image", "AUTO_TEXT2IMAGE_PIPELINES_MAPPING"),
        ("image-to-image", "AUTO_IMAGE2IMAGE_PIPELINES_MAPPING"),
    ]:
        for family in ("stable-diffusion", "stable-diffusion-xl"):
            # AutoPipeline supports an image-editing route from the original text pipeline.
            names = {
                diffusion[mapping][family],
                diffusion["AUTO_TEXT2IMAGE_PIPELINES_MAPPING"][family],
            }
            for name in sorted(names):
                item = entry(
                    f"diffusers.{task}.{name.lower()}",
                    name,
                    task,
                    "diffusion-pipeline",
                    f"{task}-v1",
                    "diffusers",
                )
                item["formats"] = ["diffusers", "safetensors", "json", "tokenizer"]
                diffusers_entries.append(item)
    from oryn_engine.diffusion_tasks import (
        IMAGE_PIPELINES,
        UNCONDITIONAL_PIPELINES,
        VIDEO_BINDINGS,
        VIDEO_PIPELINES,
        pipeline_contracts,
    )
    from oryn_engine.minimax_music import FAMILY as MUSIC3_FAMILY

    for music_task in ("text-to-audio", "text-to-music"):
        item = entry(
            f"diffusers.{music_task}.minimax-music3",
            MUSIC3_FAMILY,
            music_task,
            "diffusion-pipeline",
            "text-to-music-v1",
            "diffusers",
        )
        item["formats"] = ["diffusers", "safetensors", "json", "tokenizer"]
        diffusers_entries.append(item)
    for h3_task in ("text-to-video", "image-text-to-video"):
        item = entry(
            f"diffusers.{h3_task}.minimax-h3",
            "MiniMaxH3ModularPipeline",
            h3_task,
            "diffusion-pipeline",
            f"{h3_task}-v1",
            "diffusers",
        )
        item["formats"] = ["diffusers", "safetensors", "json", "tokenizer"]
        diffusers_entries.append(item)
    pipeline_sources = {}
    for family, name, relative in (
        ("z-image", "ZImagePipeline", "z_image/pipeline_z_image.py"),
        ("flux2-klein", "Flux2KleinPipeline", "flux2/pipeline_flux2_klein.py"),
        (
            "qwen-image-edit",
            "QwenImageEditPlusPipeline",
            "qwenimage/pipeline_qwenimage_edit_plus.py",
        ),
        (
            "qwen-image-layers",
            "QwenImageLayeredPipeline",
            "qwenimage/pipeline_qwenimage_layered.py",
        ),
        ("qwen-image-2.1", "QwenImage21Pipeline", "qwenimage21/pipeline_qwenimage21.py"),
        ("cogvideox", "CogVideoXPipeline", "cogvideo/pipeline_cogvideox.py"),
    ):
        path = args.diffusers_package / "pipelines" / relative
        tree = ast.parse(path.read_text())
        native = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
        )
        call = next(
            node
            for node in native.body
            if isinstance(node, ast.FunctionDef) and node.name == "__call__"
        )
        parameters = {arg.arg for arg in call.args.args + call.args.kwonlyargs}
        if (
            not {
                "prompt",
                "generator",
                "num_inference_steps",
                "callback_on_step_end",
            }
            | (
                {"true_cfg_scale", "output_resolution"}
                if name == "QwenImage21Pipeline"
                else {"guidance_scale", "max_sequence_length"}
            )
            <= parameters
        ):
            raise ValueError("Native image pipeline contract changed")
        if name not in IMAGE_PIPELINES and name not in VIDEO_PIPELINES:
            raise ValueError("Pipeline has no sealed Oryn behavior")
        pipeline_sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        for contract in sorted(
            {"text-to-video-v1"} if name in VIDEO_PIPELINES else pipeline_contracts(name)
        ):
            task = {
                "text-to-image-v1": "text-to-image",
                "image-editing-v1": "image-to-image",
                "image-layers-v1": "image-text-to-image",
                "text-to-video-v1": "text-to-video",
            }[contract]
            required = (
                {"layers", "resolution", "image", "true_cfg_scale"}
                if contract == "image-layers-v1"
                else {"width", "height"}
            )
            if contract == "image-editing-v1":
                required.add("image")
            if contract == "text-to-video-v1":
                required.update({"num_frames", "negative_prompt", "max_sequence_length"})
            if not required <= parameters:
                raise ValueError("Native image task parameters changed")
            item = entry(
                f"diffusers.{task}.{family}",
                name,
                task,
                "diffusion-pipeline",
                contract,
                "diffusers",
            )
            item["formats"] = ["diffusers", "safetensors", "json", "tokenizer"]
            diffusers_entries.append(item)
    for name, relative in (
        ("DDPMPipeline", "ddpm/pipeline_ddpm.py"),
        ("ScoreSdeVePipeline", "deprecated/score_sde_ve/pipeline_score_sde_ve.py"),
    ):
        path = args.diffusers_package / "pipelines" / relative
        tree = ast.parse(path.read_text())
        native = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
        )
        call = next(
            node
            for node in native.body
            if isinstance(node, ast.FunctionDef) and node.name == "__call__"
        )
        parameters = {arg.arg for arg in call.args.args + call.args.kwonlyargs}
        if (
            not {"batch_size", "generator", "num_inference_steps", "output_type"} <= parameters
            or name not in UNCONDITIONAL_PIPELINES
        ):
            raise ValueError("Native unconditional task parameters changed")
        pipeline_sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        item = entry(
            f"diffusers.unconditional.{name.lower()}",
            name,
            "unconditional-image-generation",
            "diffusion-pipeline",
            "unconditional-image-generation-v1",
            "diffusers",
        )
        item["formats"] = ["diffusers", "safetensors", "json"]
        diffusers_entries.append(item)
        # Early native exports publish only plain numeric PyTorch archives.
        numeric = entry(
            f"diffusers.unconditional.numeric-archive.{name.lower()}",
            name,
            "unconditional-image-generation",
            "diffusion-pipeline",
            "unconditional-image-generation-v1",
            "diffusers",
        )
        numeric["formats"] = ["diffusers", "torch-numeric-zip", "json"]
        diffusers_entries.append(numeric)
    for family, native_name, relative in (
        ("WanPipeline", "WanPipeline", "wan/pipeline_wan.py"),
        ("LTXPipeline", "LTXPipeline", "ltx/pipeline_ltx.py"),
        ("LTXPipeline", "LTXImageToVideoPipeline", "ltx/pipeline_ltx_image2video.py"),
    ):
        path = args.diffusers_package / "pipelines" / relative
        tree = ast.parse(path.read_text())
        native = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == native_name
        )
        call = next(
            node
            for node in native.body
            if isinstance(node, ast.FunctionDef) and node.name == "__call__"
        )
        parameters = {arg.arg for arg in call.args.args + call.args.kwonlyargs}
        required = {
            "prompt",
            "generator",
            "num_inference_steps",
            "height",
            "width",
            "num_frames",
            "guidance_scale",
            "negative_prompt",
            "max_sequence_length",
        }
        if native_name == "LTXImageToVideoPipeline":
            required.add("image")
        if family == "LTXPipeline":
            required.add("frame_rate")
        if not required <= parameters:
            raise ValueError("Native video binding parameters changed")
        pipeline_sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        for contract, loader in VIDEO_BINDINGS[family].items():
            if loader != native_name:
                continue
            task = contract.removesuffix("-v1")
            item = entry(
                f"diffusers.{task}.{family.lower()}",
                family,
                task,
                "diffusion-pipeline",
                contract,
                "diffusers",
            )
            item["formats"] = ["diffusers", "safetensors", "json", "tokenizer"]
            diffusers_entries.append(item)
    diffusion_digest = canonical_digest(
        {
            "mapping": hashlib.sha256(diffusers_mapping.read_bytes()).hexdigest(),
            "bindings": {
                name: hashlib.sha256(
                    (Path(mlx_vision.__file__).parent / name).read_bytes()
                ).hexdigest()
                for name in (
                    "diffusion_tasks.py",
                    "diffusion_lora.py",
                    "model_composition.py",
                    "data/model-compositions.json",
                    "qwen_image_21.py",
                    "minimax_music.py",
                    "minimax_h3.py",
                    "modular_components.py",
                    "video_audio_artifacts.py",
                    "flat_ddpm.py",
                    "numeric_unet2d.py",
                    "numeric_torch_archive.py",
                    "numeric_state_dict.py",
                    "wan_video.py",
                    "runtime_packs/diffusers_worker.py",
                )
            },
        }
    )
    depth_inputs = {
        "depthDependencies": {
            name: importlib.metadata.version(name)
            for name in ("oryn-depth-native", "opencv-python-headless", "addict", "omegaconf")
        },
        "depthNativeSources": {
            path.relative_to(
                Path(__import__("depth_anything_3").__file__).parent
            ).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__import__("depth_anything_3").__file__).parent.rglob("*.py"))
        },
        "opencvBinaryName": next(Path(__import__("cv2").__file__).parent.glob("cv2*.so")).name,
        "opencvBinarySha256": hashlib.sha256(
            next(Path(__import__("cv2").__file__).parent.glob("cv2*.so")).read_bytes()
        ).hexdigest(),
    }
    import ctranslate2

    ct2_root = Path(ctranslate2.__file__).parent
    ct2_inputs = {
        "ctranslate2": importlib.metadata.version("ctranslate2"),
        "ctranslate2Sources": {
            path.relative_to(ct2_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(ct2_root.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".so", ".dylib"}
        },
    }
    import librosa

    rnnt_inputs = {
        "rnntDependencies": {
            name: importlib.metadata.version(name) for name in ("librosa", "numba", "llvmlite")
        },
        "librosaSources": {
            path.relative_to(Path(librosa.__file__).parent).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(Path(librosa.__file__).parent.rglob("*.py"))
        },
    }
    openclip_inputs = {
        "openclipDependencies": {
            name: importlib.metadata.version(name) for name in ("open-clip-torch", "ftfy")
        },
        "openclipSources": {
            module: {
                path.relative_to(
                    Path(importlib.import_module(module).__file__).parent
                ).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(importlib.import_module(module).__file__).parent.rglob("*"))
                if path.is_file()
                and (path.suffix == ".py" or path.name == "bpe_simple_vocab_16e6.txt.gz")
            }
            for module in ("open_clip", "ftfy")
        },
    }
    # A private reviewed adapter, selected only after Browser-owned source
    # recognition. Repository metadata alone cannot authorize this graph.
    entries.append(
        {
            "architecture": "OrynReviewedLfm3ForCausalLM",
            "conformance": "untested",
            "familyAdapter": {"id": "oryn.family.causal-lm", "version": "1.0.0"},
            "formats": ["safetensors", "json", "tokenizer"],
            "id": "transformers.reviewed-lfm3.chat.v1",
            "sourceKind": "transformers",
            "task": "text-generation",
            "taskAdapter": {"id": "oryn.task.chat-v1", "version": "1.0.0"},
        }
    )
    sources = [
        source(
            "oryn.source.transformers",
            "oryn.runtime.pytorch",
            entries,
            canonical_digest(
                {
                    "mapping": mapping_digest,
                    "flaxSource": flax_digest,
                    "audioSource": native_sources["audio_understanding.py"],
                    "voxtralSource": native_sources["voxtral_tasks.py"],
                    "tekkenSource": native_sources["tekken_tokenizer.py"],
                    "layoutSource": native_sources["model_layouts.py"],
                    "inspectionSource": native_sources["model_inspection.py"],
                    "taskProfileSource": native_sources["task_profiles.py"],
                    "decisionSource": native_sources["decision_tasks.py"],
                    "jinaRankingSource": native_sources["jina_ranking.py"],
                    "gemmaDecisionSource": native_sources["gemma_decision.py"],
                    "depth3Source": native_sources["depth3_tasks.py"],
                    "llavaNextSource": native_sources["llava_next_tasks.py"],
                    "contextualEmbeddingSource": native_sources["contextual_embeddings.py"],
                    "depthRuntime": depth_inputs,
                    "piiTasksSource": native_sources["pii_tasks.py"],
                    "audioHeadsSource": native_sources["wav2vec_heads.py"],
                    "bioesSource": native_sources["bioes.py"],
                    "tiktoken": importlib.metadata.version("tiktoken"),
                    "tiktokenBinarySha256": hashlib.sha256(
                        Path(importlib.import_module("tiktoken._tiktoken").__file__).read_bytes()
                    ).hexdigest(),
                    "trocrSource": native_sources["trocr_tasks.py"],
                    "qwenRetrievalSource": native_sources["qwen_retrieval.py"],
                    "bidirectionalEmbeddingsSource": native_sources["bidirectional_embeddings.py"],
                    "gigaEmbeddingsSource": native_sources["giga_embeddings.py"],
                    "llavaVideoSource": native_sources["llava_video.py"],
                    "neommeRetrievalSource": native_sources["neomme_retrieval.py"],
                    "nativeRoutingSource": native_sources["runners.py"],
                    "nativeTaskSource": native_sources["model_support.py"],
                    "metadataSource": native_sources["model_metadata.py"],
                    "nativeIdentitySource": native_sources["generated_capabilities.py"],
                    "timesfm3Source": native_sources["timesfm3_tasks.py"],
                    "colqwen3Source": native_sources["colqwen3_tasks.py"],
                    "colqwen35Source": native_sources["colqwen35_tasks.py"],
                    "forecastTaskSource": native_sources["forecast_tasks.py"],
                    "exaoneDemandSource": native_sources["exaone_demand.py"],
                    "exaoneDemandNative": {
                        k: v
                        for k, v in native_sources.items()
                        if k.startswith("native_exaone_demand/")
                    },
                    "yugSource": native_sources["yug_tasks.py"],
                    "yugNative": {
                        k: v for k, v in native_sources.items() if k.startswith("native_yug/")
                    },
                    "auroraSource": native_sources["aurora_tasks.py"],
                    "openclipSource": native_sources["openclip_tasks.py"],
                    "birefnetSource": native_sources["birefnet_tasks.py"],
                    "isnetSource": native_sources["isnet_tasks.py"],
                    "pixaiSource": native_sources["pixai_tasks.py"],
                    "tipsSource": native_sources["tips_tasks.py"],
                    "rtmwSource": native_sources["rtmw_tasks.py"],
                    "jina4Source": native_sources["jina4_tasks.py"],
                    "rtmwNative": {
                        k: v for k, v in native_sources.items() if k.startswith("native_rtmw/")
                    },
                    "transformersVisionSources": transformers_vision_sources,
                    "isnetNative": {
                        k: v for k, v in native_sources.items() if k.startswith("native_isnet/")
                    },
                    "birefnetNative": {
                        k: v for k, v in native_sources.items() if k.startswith("native_birefnet/")
                    },
                    "openclipRuntime": openclip_inputs,
                    "auroraNative": {
                        k: v for k, v in native_sources.items() if k.startswith("native_aurora/")
                    },
                    "sharedTaskSource": native_sources["shared_tasks.py"],
                    "coverageTaskSource": native_sources["coverage_tasks.py"],
                    "audio8TaskSource": native_sources["audio8_tasks.py"],
                    "turnClassificationSource": native_sources["turn_classification.py"],
                    "speechxSource": native_sources["speechx_tasks.py"],
                    "soproSource": native_sources["sopro_tasks.py"],
                    "omnivoiceSource": native_sources["omnivoice_tasks.py"],
                    "breezeSource": native_sources["breeze_tasks.py"],
                    "molparserSource": native_sources["molparser_tasks.py"],
                    "qwenCodecSource": native_sources["qwen_codec_tasks.py"],
                    "qwenBpeSource": native_sources["native_qwen_bpe.py"],
                    "llavaVideoQwenSource": native_sources["llava_video_qwen.py"],
                    "llavaVideoQwenAssets": {
                        k: v
                        for k, v in native_sources.items()
                        if k.startswith("llava_video_qwen_assets/")
                    },
                    "videollama3Source": native_sources["videollama3_tasks.py"],
                    "videollama3Assets": {
                        k: v
                        for k, v in native_sources.items()
                        if k.startswith("videollama3_assets/")
                    },
                    "tigerSource": native_sources["tiger_tasks.py"],
                    "tigerNative": {
                        k: v for k, v in native_sources.items() if k.startswith("native_tiger/")
                    },
                    "reuseSource": native_sources["reuse_tasks.py"],
                    "reuseNative": native_sources["native_reuse.py"],
                    "reuseAssets": {
                        k: v for k, v in native_sources.items() if k.startswith("reuse_assets/")
                    },
                    "mpsenetSource": native_sources["mpsenet_tasks.py"],
                    "mpsenetNative": {
                        k: v for k, v in native_sources.items() if k.startswith("native_mpsenet/")
                    },
                    "layoutlmSource": native_sources["layoutlm_tasks.py"],
                    "layoutlmAssets": {
                        k: v for k, v in native_sources.items() if k.startswith("layoutlm_assets/")
                    },
                    "molparserAssets": {
                        k: v for k, v in native_sources.items() if k.startswith("molparser_assets/")
                    },
                    "breezeNative": {
                        k: v for k, v in native_sources.items() if k.startswith("native_breeze/")
                    },
                    "omnivoiceNative": {
                        k: v for k, v in native_sources.items() if k.startswith("native_omnivoice/")
                    },
                    "soproNative": {
                        k: v for k, v in native_sources.items() if k.startswith("native_sopro/")
                    },
                    "audio8GraphSource": native_sources["audio8_model.py"],
                    "audio8License": native_sources["third_party/Audio8-LICENSE"],
                    "rnntTaskSource": native_sources["rnnt_tasks.py"],
                    "parakeetTaskSource": native_sources["parakeet_tasks.py"],
                    "rnntRuntime": rnnt_inputs,
                    "ctranslate2Runtime": ct2_inputs,
                    "ctranslate2TaskSource": native_sources["ct2_tasks.py"],
                    "ctranslate2ContainerSource": native_sources["ct2_weights.py"],
                    "numericArchiveSources": {
                        k: native_sources[k]
                        for k in (
                            "numeric_state_dict.py",
                            "numeric_torch_archive.py",
                            "numeric_archive_tasks.py",
                            "numeric_seq2seq.py",
                            "numeric_dpt.py",
                            "numeric_mobilebert.py",
                            "numeric_owlvit.py",
                            "numeric_tapas.py",
                            "numeric_coverage.py",
                            "numeric_layoutlmv2.py",
                        )
                    },
                    "numericHdf5Source": native_sources["numeric_hdf5.py"],
                    "hdf5TasksSource": native_sources["hdf5_tasks.py"],
                    "hdf5BlipSource": native_sources["hdf5_blip.py"],
                    "hdf5SegformerSource": native_sources["hdf5_segformer.py"],
                    "qwenTranslationSource": native_sources["qwen_translation.py"],
                    "instructionTranslationSource": native_sources["instruction_translation.py"],
                    "hunyuanTranslationSource": native_sources["hunyuan_translation.py"],
                    "hdf5MarianSource": native_sources["hdf5_marian.py"],
                    "h5py": importlib.metadata.version("h5py"),
                    "sacremoses": importlib.metadata.version("sacremoses"),
                    "sapiens2PoseSource": native_sources["sapiens2_pose.py"],
                    "sapiens2JointData": native_sources["data/sapiens2-joints.json"],
                }
            ),
        ),
        source(
            "oryn.source.mlx-lm",
            "oryn.runtime.mlx",
            mlx_entries,
            canonical_digest(
                [
                    mlx_inputs,
                    vision_sources,
                    {
                        name: native_sources[name]
                        for name in (
                            "mlx_vad.py",
                            "mlx_vision.py",
                            "mlx_gemma.py",
                            "mlx_gemma4.py",
                            "mlx_gpt_oss.py",
                            "mlx_prism.py",
                            "mlx_bolding.py",
                            "mlx_kolibri.py",
                            "native_kolibri.py",
                            "third_party/Kolibri-MLX-LICENSE",
                            "third_party/Kolibri-MLX-NOTICE",
                            "third_party/Prism-MLX-LICENSE",
                            "runtime_packs/mlx_lm_worker.py",
                            "gemma_decision.py",
                            "model_metadata.py",
                            "model_layouts.py",
                            "task_profiles.py",
                        )
                    },
                ]
            ),
        ),
    ]
    # The signed capability source names the active immutable foundation metadata revision.
    sources[1]["foundation"]["version"] = json.loads(
        (args.output / "mlx-foundation.json").read_text()
    )["payload"]["version"]
    sources.append(
        source("oryn.source.diffusers", "oryn.runtime.pytorch", diffusers_entries, diffusion_digest)
    )
    build_root = Path(__file__).resolve().parents[1] / "build-inputs"
    policy = json.loads((build_root / "llama-chat-policy.json").read_text())
    from oryn_engine.chat_formats import validate_table

    chat_formats = json.loads((build_root / "chat-formats.json").read_text())
    chat_format_names = set(validate_table(chat_formats))
    native_source = (build_root / f"llama-arch-{policy['sourceRevision'][:8]}.cpp").read_bytes()
    if hashlib.sha256(native_source).hexdigest() != policy["sourceSha256"]:
        raise ValueError("Pinned llama.cpp capability source changed")
    architectures = sorted(
        set(re.findall(r'\{ LLM_ARCH_[A-Z0-9_]+,\s+"([a-z0-9._-]+)"', native_source.decode()))
    )
    if not set(policy["admitted"]) <= set(architectures):
        raise ValueError("Policy claims an architecture absent from the native build")
    from oryn_engine.runtime_packs.gguf import QUANTIZATION_NAMES

    if len(set(policy["quantizations"])) != len(policy["quantizations"]) or not set(
        policy["quantizations"]
    ) <= set(QUANTIZATION_NAMES.values()):
        raise ValueError("Policy names a quantization the bounded GGUF reader does not know")
    for templates in policy["admitted"].values():
        if (
            not isinstance(templates, list)
            or not templates
            or len(templates) != len(set(templates))
            or not set(templates) <= chat_format_names - {"plain"}
        ):
            raise ValueError("Native template policy must name distinct sealed renderers")
    native_entries = []
    vision_library = args.llama_binary.with_name(VISION_LIBRARY).read_bytes()
    for family, projectors in policy.get("vision", {}).items():
        if (
            family not in policy["admitted"]
            or not projectors
            or any(
                not isinstance(name, str) or name.encode() not in vision_library
                for name in projectors
            )
        ):
            raise ValueError(
                "Vision policy names a projector absent from the pinned native library"
            )
    for architecture in architectures:
        admitted = architecture in policy["admitted"]
        # Capability identifiers have no underscores; the architecture keeps its exact name.
        slug = architecture.replace("_", "-")
        item = entry(
            f"llama-cpp.chat.{slug}",
            architecture,
            "text-generation",
            f"gguf.{slug}",
            "chat-v1",
            "gguf",
        )
        item["formats"] = ["gguf"]
        item["policy"] = {
            "state": "admitted" if admitted else "withheld",
            "reasons": [] if admitted else [policy["withheldReason"]],
        }
        item["constraints"] = {
            "containerVersions": ["2", "3"],
            "quantizations": policy["quantizations"],
            "tokenizers": policy["tokenizers"].get(architecture, policy["tokenizers"]["default"]),
            "templates": policy["admitted"].get(architecture, ["unavailable"]),
            "maxContextTokens": 8192,
        }
        native_entries.append(item)
        from oryn_engine.gguf_translation import ARCHITECTURES as TRANSLATION_ARCHITECTURES

        if architecture in TRANSLATION_ARCHITECTURES:
            translation_item = json.loads(json.dumps(item))
            translation_item.update(
                id=f"llama-cpp.translation.{slug}",
                task="translation",
                taskAdapter={"id": "oryn.task.translation-v1", "version": "1.0.0"},
            )
            translation_item["constraints"]["maxContextTokens"] = 4096
            native_entries.append(translation_item)
        if architecture in policy.get("vision", {}):
            image_item = json.loads(json.dumps(item))
            image_item.update(id=f"llama-cpp.vision.{slug}", task="image-text-to-text")
            native_entries.append(image_item)
    native_binary_digest = hashlib.sha256(args.llama_binary.read_bytes()).hexdigest()
    native_provider = source(
        "oryn.source.llama-cpp",
        "oryn.runtime.llama-cpp",
        native_entries,
        canonical_digest(
            {
                "binary": native_binary_digest,
                "sources": {
                    name: native_sources[name]
                    for name in (
                        "gguf_translation.py",
                        "qwen_translation.py",
                        "runtime_packs/llama_cpp_worker.py",
                        "runners.py",
                        "models.py",
                        "task_profiles.py",
                    )
                },
            }
        ),
    )
    native_provider["schemaVersion"] = "oryn-capability-source/2"
    sources.append(native_provider)
    import onnxruntime
    from onnxruntime.capi.onnxruntime_pybind11_state import get_all_operator_schema

    from oryn_engine.onnx_tasks import ENCODER_FAMILIES
    from oryn_engine.onnx_tasks import TASK_CONTRACTS as ONNX_TASKS

    onnx_operators = {}
    for schema in get_all_operator_schema():
        if schema.domain in {"", "ai.onnx"}:
            onnx_operators.setdefault(schema.name, []).append(schema.since_version)
        elif schema.domain == "com.microsoft" and schema.name == "MatMulNBits":
            onnx_operators.setdefault("com.microsoft::MatMulNBits", []).append(schema.since_version)
    onnx_operators = {k: sorted(set(v)) for k, v in sorted(onnx_operators.items())}
    onnx_entries = []
    from oryn_engine import onnx_audio

    audio_digest = hashlib.sha256(Path(onnx_audio.__file__).read_bytes()).hexdigest()
    native_sources["onnx_audio.py"] = audio_digest
    for task, contract in ONNX_TASKS.items():
        if task == "image-classification":
            from oryn_engine.onnx_vision import ARCHITECTURE

            item = entry("onnx.vision.camie", ARCHITECTURE, task, "onnx-vision", contract, "onnx")
            item["formats"] = ["onnx", "json"]
            onnx_entries.append(item)
            from oryn_engine import onnx_classifier

            item = entry(
                "onnx.vision.efficientnet-lite4",
                onnx_classifier.ARCHITECTURE,
                task,
                "onnx-classifier",
                contract,
                "onnx",
            )
            item["formats"] = ["onnx"]
            onnx_entries.append(item)
            continue
        if task == "audio-to-audio":
            item = entry(
                "onnx.audio.dfn3", onnx_audio.ARCHITECTURE, task, "onnx-dfn3", contract, "onnx"
            )
            item["formats"] = ["onnx"]
            onnx_entries.append(item)
            continue
        if task == "automatic-speech-recognition":
            if "wav2vec2" not in modeling_auto.MODEL_FOR_CTC_MAPPING_NAMES:
                raise ValueError("Native Wav2Vec2 CTC mapping is unavailable")
            item = entry("onnx.ctc.wav2vec2", "wav2vec2", task, "onnx-ctc", contract, "onnx")
            item["formats"] = ["onnx", "json", "tokenizer"]
            onnx_entries.append(item)
            if "whisper" not in modeling_auto.MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES:
                raise ValueError("Native Whisper mapping is unavailable")
            item = entry("onnx.asr.tdt", "nemo-conformer-tdt", task, "onnx-tdt", contract, "onnx")
            item["formats"] = ["onnx", "json", "tokenizer"]
            onnx_entries.append(item)
            item = entry("onnx.asr.whisper", "whisper", task, "onnx-whisper", contract, "onnx")
            item["formats"] = ["onnx", "json", "tokenizer"]
            onnx_entries.append(item)
            continue
        if task == "voice-activity-detection":
            item = entry(
                "onnx.voice-activity-detection.state128",
                "silero-vad-state128",
                task,
                "onnx-vad-state128",
                contract,
                "onnx",
            )
            item["formats"] = ["onnx"]
            onnx_entries.append(item)
            continue
        mapping = (
            modeling_auto.MODEL_MAPPING_NAMES
            if task == "feature-extraction"
            else modeling_auto.MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
            if task == "token-classification"
            else modeling_auto.MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
        )
        for family in sorted(ENCODER_FAMILIES & mapping.keys()):
            item = entry(
                f"onnx.{task}.{family.replace('_', '-')}",
                family,
                task,
                "onnx-encoder",
                contract,
                "onnx",
            )
            item["formats"] = ["onnx", "json", "tokenizer"]
            onnx_entries.append(item)
    sources.append(
        source(
            "oryn.source.onnx-encoders",
            "oryn.runtime.onnx",
            onnx_entries,
            canonical_digest(
                {
                    "operators": onnx_operators,
                    "audioSource": audio_digest,
                    "classifierPreprocessor": {
                        "version": depth_inputs["depthDependencies"]["opencv-python-headless"],
                        "binarySha256": depth_inputs["opencvBinarySha256"],
                    },
                    "nativeSources": {
                        name: native_sources[name]
                        for name in (
                            "onnx_vision.py",
                            "onnx_classifier.py",
                            "data/imagenet-efficientnet-lite4-labels.json",
                            "runtime_packs/onnx_worker.py",
                            "jina_onnx.py",
                            "onnx_tasks.py",
                            "onnx_tdt.py",
                            "onnx_graph.py",
                            "onnx_nbits.py",
                            "onnx_inspection.py",
                        )
                    },
                }
            ),
        )
    )
    qwen_runtime = json.loads((build_root / "qwen-tts-runtime.json").read_text())
    for name in ("qwen_tts.py", "qwen_tts_inspection.py", "runtime_packs/qwen_tts_worker.py"):
        native_sources[name] = hashlib.sha256(
            (build_root.parent / "src/oryn_engine" / name).read_bytes()
        ).hexdigest()
    sources.append(
        source(
            "oryn.source.qwen-tts",
            "oryn.runtime.pytorch-speech",
            [
                entry(
                    "qwen-tts.custom-voice-12hz",
                    "Qwen3TTSForConditionalGeneration",
                    "text-to-speech",
                    "qwen-tts",
                    "speech-synthesis-v1",
                    "qwen-tts",
                )
            ],
            canonical_digest(
                {
                    "runtime": qwen_runtime,
                    "adapter": native_sources["qwen_tts.py"],
                    "inspection": native_sources["qwen_tts_inspection.py"],
                    "worker": native_sources["runtime_packs/qwen_tts_worker.py"],
                }
            ),
        )
    )
    compile_registry(sources)
    # Record native task/model-type resolution for publisher configs that omit
    # architectures. Ambiguous Auto mappings remain ambiguous at inspection.
    native_task_architectures = {}
    mappings_by_task = {task: (group[0],) for task, group in groups.items()}
    for task, mappings in task_mappings.items():
        mappings_by_task[task] = tuple(set(mappings_by_task.get(task, ())) | set(mappings))
    mappings_by_task["feature-extraction"] = (
        "MODEL_MAPPING_NAMES",
        "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",
    )
    for task, mappings in mappings_by_task.items():
        admitted = {e["architecture"] for e in entries if e["task"] == task}
        families = {}
        for mapping in mappings:
            for family, value in getattr(modeling_auto, mapping).items():
                names = value if isinstance(value, tuple) else (value,)
                families.setdefault(family, set()).update(set(names) & admitted)
        native_task_architectures[task] = {
            family: sorted(names) for family, names in sorted(families.items()) if names
        }
    native_task_architectures.setdefault("automatic-speech-recognition", {})[
        "audio8_asr_infinite"
    ] = ["Audio8ASRInfiniteForConditionalGeneration"]
    native_task_architectures.setdefault("text-classification", {})["oryn_llama_turn"] = [
        "OrynLlamaTurnForClassification"
    ]
    native_task_architectures.setdefault("text-ranking", {}).setdefault("qwen3", []).append(
        "JinaForRanking"
    )
    native_task_architectures.setdefault("token-classification", {})["pii_masking"] = [
        "PiiMaskingModel"
    ]
    for family, architecture in AUDIO_HEAD_ARCHITECTURES.items():
        native_task_architectures.setdefault("audio-classification", {})[family] = [architecture]
    artifact = {
        "schemaVersion": 1,
        "sources": sources,
        "mlxModelTypes": mlx_types,
        "buildInputs": {
            "reviewedImplementations": build_binding(
                Path(__file__).resolve().parents[1] / "src/oryn_engine"
            ),
            **rnnt_inputs,
            **ct2_inputs,
            "tiktoken": importlib.metadata.version("tiktoken"),
            "tiktokenBinarySha256": hashlib.sha256(
                Path(importlib.import_module("tiktoken._tiktoken").__file__).read_bytes()
            ).hexdigest(),
            "qwenTtsRuntime": qwen_runtime,
            "transformersTaskArchitectures": native_task_architectures,
            "chatFormats": chat_formats,
            "transformersVisionSources": transformers_vision_sources,
            "h5py": importlib.metadata.version("h5py"),
            "sacremoses": importlib.metadata.version("sacremoses"),
            "autogluonTabular": importlib.metadata.version("autogluon.tabular"),
            "tabpfnMixSources": {
                path.relative_to(
                    Path(
                        importlib.import_module("autogluon.tabular.models.tabpfnmix").__file__
                    ).parent
                ).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(
                    Path(
                        importlib.import_module("autogluon.tabular.models.tabpfnmix").__file__
                    ).parent.rglob("*.py")
                )
            },
            "mitraSources": {
                path.relative_to(
                    Path(importlib.import_module("autogluon.tabular.models.mitra").__file__).parent
                ).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(
                    Path(
                        importlib.import_module("autogluon.tabular.models.mitra").__file__
                    ).parent.rglob("*.py")
                )
            },
            "tabpfn": importlib.metadata.version("tabpfn"),
            "tabpfnSources": {
                path.relative_to(
                    Path(__import__("tabpfn").__file__).parent
                ).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__import__("tabpfn").__file__).parent.rglob("*.py"))
            },
            "chronosForecasting": importlib.metadata.version("chronos-forecasting"),
            "timesfm": importlib.metadata.version("timesfm"),
            **depth_inputs,
            **openclip_inputs,
            "timesfm3Sources": {
                path.relative_to(
                    Path(__import__("timesfm3").__file__).parent
                ).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__import__("timesfm3").__file__).parent.rglob("*.py"))
            },
            "chronosSources": {
                path.relative_to(
                    Path(__import__("chronos").__file__).parent
                ).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__import__("chronos").__file__).parent.rglob("*.py"))
            },
            "timm": importlib.metadata.version("timm"),
            "timmArchitectures": sorted(__import__("timm").list_models()),
            "timmSources": {
                path.relative_to(
                    Path(__import__("timm").__file__).parent
                ).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__import__("timm").__file__).parent.rglob("*.py"))
            },
            "onnxruntime": onnxruntime.__version__,
            "onnxOperators": onnx_operators,
            "onnxNativeSha256": hashlib.sha256(
                Path(onnxruntime.capi.onnxruntime_pybind11_state.__file__).read_bytes()
            ).hexdigest(),
            "transformers": importlib.metadata.version("transformers"),
            "transformersMappingSha256": mapping_digest,
            "diffusersMappingSha256": hashlib.sha256(diffusers_mapping.read_bytes()).hexdigest(),
            "diffusionEngineSources": {
                name: hashlib.sha256(
                    (Path(mlx_vision.__file__).parent / name).read_bytes()
                ).hexdigest()
                for name in (
                    "diffusion_lora.py",
                    "model_composition.py",
                    "data/model-compositions.json",
                    "qwen_image_21.py",
                    "diffusion_tasks.py",
                    "minimax_music.py",
                    "minimax_h3.py",
                    "modular_components.py",
                    "video_audio_artifacts.py",
                    "runtime_packs/diffusers_worker.py",
                )
            },
            "diffusersPipelineSources": pipeline_sources,
            "diffusersSchedulerSources": {
                name: hashlib.sha256((args.diffusers_package / name).read_bytes()).hexdigest()
                for name in (
                    "schedulers/scheduling_ddpm.py",
                    "schedulers/scheduling_flow_match_euler_discrete.py",
                    "guiders/classifier_free_guidance.py",
                    "modular_pipelines/modular_pipeline.py",
                    "modular_pipelines/modular_pipeline_utils.py",
                    "modular_pipelines/minimax_music3/modular_pipeline.py",
                    "modular_pipelines/minimax_music3/modular_blocks_minimax_music3.py",
                    "modular_pipelines/minimax_music3/encoders.py",
                    "modular_pipelines/minimax_music3/before_denoise.py",
                    "modular_pipelines/minimax_music3/denoise.py",
                    "modular_pipelines/minimax_music3/decoders.py",
                    "modular_pipelines/minimax_h3/modular_pipeline.py",
                    "modular_pipelines/minimax_h3/modular_blocks_minimax_h3.py",
                    "modular_pipelines/minimax_h3/before_encoder.py",
                    "modular_pipelines/minimax_h3/before_denoise.py",
                    "modular_pipelines/minimax_h3/encoders.py",
                    "modular_pipelines/minimax_h3/denoise.py",
                    "modular_pipelines/minimax_h3/decoders.py",
                    "modular_pipelines/minimax_h3/references.py",
                    "schedulers/scheduling_minimax_h3.py",
                    "schedulers/scheduling_utils.py",
                    "utils/torch_utils.py",
                )
            },
            "diffusersModelSources": {
                name: hashlib.sha256(
                    (args.diffusers_package / "models" / name).read_bytes()
                ).hexdigest()
                for name in (
                    "transformers/transformer_qwenimage21.py",
                    "transformers/transformer_minimax_music3.py",
                    "transformers/minimax_music3_rvq_depth_decoder.py",
                    "condition_embedders/condition_embedder_minimax_music3.py",
                    "autoencoders/minimax_music3_vocoder.py",
                    "transformers/transformer_minimax_h3.py",
                    "autoencoders/autoencoder_kl_minimax_h3.py",
                    "autoencoders/autoencoder_kl_minimax_h3_audio.py",
                    "autoencoders/autoencoder_kl_qwenimage21.py",
                    "transformers/cogvideox_transformer_3d.py",
                    "autoencoders/autoencoder_kl_cogvideox.py",
                    "embeddings.py",
                    "unets/unet_2d.py",
                    "unets/unet_2d_blocks.py",
                    "resnet.py",
                    "attention_processor.py",
                )
            },
            "mlxModelSources": mlx_inputs,
            "orynNativeSources": native_sources,
            "mlxVlm": version("mlx-vlm"),
            "mlxVlmSources": vision_sources,
            "generatorSha256": generator_digest,
            "llamaSourceRevision": policy["sourceRevision"],
            "llamaSourceSha256": policy["sourceSha256"],
            "llamaBinarySha256": native_binary_digest,
            "llamaVisionBinarySha256": hashlib.sha256(
                args.llama_binary.with_name("llama-mtmd-cli").read_bytes()
            ).hexdigest(),
            "llamaVisionLibrarySha256": hashlib.sha256(
                args.llama_binary.with_name(VISION_LIBRARY).read_bytes()
            ).hexdigest(),
            "llamaVisionProjectors": policy.get("vision", {}),
            "llamaPolicySha256": hashlib.sha256(
                (build_root / "llama-chat-policy.json").read_bytes()
            ).hexdigest(),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (args.output / "capabilities.json").write_bytes(raw)
    payload = {
        "kind": "oryn-capability-build",
        "schemaVersion": 1,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    signature = subprocess.run(
        ["openssl", "dgst", "-sha256", "-sign", str(args.signing_key)],
        input=encoded,
        capture_output=True,
        check=True,
    ).stdout
    envelope = {
        "payload": payload,
        "integrity": {
            "algorithm": "rsa-sha256",
            "keyId": "oryn-sprint47-development-2026",
            "digest": hashlib.sha256(encoded).hexdigest(),
            "signature": base64.b64encode(signature).decode(),
        },
    }
    (args.output / "capabilities.signature.json").write_text(json.dumps(envelope, indent=2) + "\n")
    print(f"Generated {len(entries)} Transformers and {len(mlx_entries)} MLX entries.")


if __name__ == "__main__":
    main()
