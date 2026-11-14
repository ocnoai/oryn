from __future__ import annotations

from typing import Literal

from oryn_engine.generated_capabilities import architectures_for

TransformersWorkload = Literal[
    "asr",
    "audio-classification",
    "causal-chat",
    "depth-estimation",
    "embedding",
    "encoder-embedding",
    "cross-encoder-reranking",
    "fill-mask",
    "forced-alignment",
    "image-classification",
    "multimodal-chat",
    "musicgen",
    "object-detection",
    "question-answering",
    "reranking",
    "seq2seq",
    "text-classification",
    "token-classification",
    "zero-shot-image-classification",
]


ASR_ARCHITECTURES = architectures_for("automatic-speech-recognition")
FORCED_ALIGNMENT_ARCHITECTURES = architectures_for("forced-alignment")
CAUSAL_CHAT_ARCHITECTURES = architectures_for("text-generation")
MULTIMODAL_CHAT_ARCHITECTURES = architectures_for("image-text-to-text")
MUSICGEN_ARCHITECTURES = architectures_for("text-to-audio")
TEXT_CLASSIFICATION_ARCHITECTURES = architectures_for("text-classification")
TOKEN_CLASSIFICATION_ARCHITECTURES = architectures_for("token-classification")
QUESTION_ANSWERING_ARCHITECTURES = architectures_for("question-answering")
FILL_MASK_ARCHITECTURES = architectures_for("fill-mask")
SEQ2SEQ_ARCHITECTURES = architectures_for(
    "summarization", family_adapter="oryn.family.seq2seq"
) | architectures_for("translation", family_adapter="oryn.family.seq2seq")
IMAGE_CLASSIFICATION_ARCHITECTURES = architectures_for("image-classification")
AUDIO_CLASSIFICATION_ARCHITECTURES = architectures_for("audio-classification")
OBJECT_DETECTION_ARCHITECTURES = architectures_for("object-detection")
DEPTH_ESTIMATION_ARCHITECTURES = architectures_for("depth-estimation")
ZERO_SHOT_IMAGE_CLASSIFICATION_ARCHITECTURES = architectures_for("zero-shot-image-classification")
EMBEDDING_ARCHITECTURES = architectures_for("feature-extraction")
RERANKING_ARCHITECTURES = architectures_for("text-ranking")


def transformers_workload(
    architectures: list[str] | set[str], task: str | None
) -> TransformersWorkload | None:
    from oryn_engine.shared_tasks import TASK_CONTRACTS

    values = set(architectures)
    # This audio checkpoint keeps a token-classification pipeline tag. Preserve
    # its more specific task before considering reusable text-label adapters.
    if task in {"forced-alignment", "token-classification"} and (
        FORCED_ALIGNMENT_ARCHITECTURES.intersection(values)
    ):
        return "forced-alignment"
    if (
        (
            task not in {"summarization", "translation"}
            or task == "translation"
            and not SEQ2SEQ_ARCHITECTURES.intersection(values)
        )
        and task in TASK_CONTRACTS
        and (
            architectures_for(task, family_adapter=f"oryn.family.{task}")
            # Numeric-archive graphs carry the same signed task contract.
            | architectures_for(task, family_adapter=f"oryn.family.numeric-archive-{task}")
        ).intersection(values)
    ):
        return task
    if task in {
        "feature-extraction",
        "sentence-similarity",
    } and EMBEDDING_ARCHITECTURES.intersection(values):
        return (
            "embedding" if CAUSAL_CHAT_ARCHITECTURES.intersection(values) else "encoder-embedding"
        )
    if task == "text-ranking" and RERANKING_ARCHITECTURES.intersection(values):
        return (
            "reranking"
            if CAUSAL_CHAT_ARCHITECTURES.intersection(values)
            else "cross-encoder-reranking"
        )
    # Task-specialized Qwen checkpoints retain the causal-LM architecture. The
    # signed pipeline mapping must therefore win over the broad architecture
    # match or they would accidentally expose the chat contract.
    if task == "feature-extraction" and CAUSAL_CHAT_ARCHITECTURES.intersection(values):
        return "embedding"
    if task == "text-ranking" and CAUSAL_CHAT_ARCHITECTURES.intersection(values):
        return "reranking"
    if task == "text-classification" and TEXT_CLASSIFICATION_ARCHITECTURES.intersection(values):
        return "text-classification"
    if task == "token-classification" and TOKEN_CLASSIFICATION_ARCHITECTURES.intersection(values):
        return "token-classification"
    if task == "question-answering" and QUESTION_ANSWERING_ARCHITECTURES.intersection(values):
        return "question-answering"
    if task == "fill-mask" and FILL_MASK_ARCHITECTURES.intersection(values):
        return "fill-mask"
    if task in {"translation", "summarization"} and SEQ2SEQ_ARCHITECTURES.intersection(values):
        return "seq2seq"
    if task == "image-classification" and IMAGE_CLASSIFICATION_ARCHITECTURES.intersection(values):
        return "image-classification"
    if task == "audio-classification" and AUDIO_CLASSIFICATION_ARCHITECTURES.intersection(values):
        return "audio-classification"
    if task == "object-detection" and OBJECT_DETECTION_ARCHITECTURES.intersection(values):
        return "object-detection"
    if task == "depth-estimation" and DEPTH_ESTIMATION_ARCHITECTURES.intersection(values):
        return "depth-estimation"
    if (
        task == "zero-shot-image-classification"
        and ZERO_SHOT_IMAGE_CLASSIFICATION_ARCHITECTURES.intersection(values)
    ):
        return "zero-shot-image-classification"
    if task == "image-text-to-text" and MULTIMODAL_CHAT_ARCHITECTURES.intersection(values):
        return "multimodal-chat"
    if ASR_ARCHITECTURES.intersection(values) or task == "automatic-speech-recognition":
        return "asr"
    if CAUSAL_CHAT_ARCHITECTURES.intersection(values):
        return "causal-chat"
    if MULTIMODAL_CHAT_ARCHITECTURES.intersection(values):
        return "multimodal-chat"
    if MUSICGEN_ARCHITECTURES.intersection(values):
        return "musicgen"
    return None
