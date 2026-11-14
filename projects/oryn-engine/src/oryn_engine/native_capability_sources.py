from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Any

from oryn_engine.capability_sources import _capability
from oryn_engine.compatibility_plane import (
    SOURCE_SCHEMA_VERSION,
    CapabilityRegistry,
    compile_registry,
)

MAX_ONNX_GRAPH_BYTES = 16 * 1024 * 1024 * 1024
MAX_ONNX_EXTERNAL_FILES = 256
FORBIDDEN_NATIVE_SUFFIXES = {".dll", ".dylib", ".so", ".py", ".exe"}


def _native_capabilities():
    from oryn_engine.generated_capabilities import generated_sources

    return tuple(
        item
        for source in generated_sources()
        if source["provider"]["id"] == "oryn.source.llama-cpp"
        for item in source["capabilities"]
    )


def _admitted_native():
    return tuple(item for item in _native_capabilities() if item["policy"]["state"] == "admitted")


LLAMA_CPP_CHAT_ARCHITECTURES = tuple(item["architecture"] for item in _admitted_native())
# The signed policy lists quantizations in preference order: the first six keep the
# historical choice for every existing repository; later entries are fallbacks only.
LLAMA_CPP_CHAT_QUANTIZATIONS = tuple(
    dict.fromkeys(q for item in _admitted_native() for q in item["constraints"]["quantizations"])
)
_TRANSFORMERS_TO_GGUF_ARCHITECTURE = {
    "Gemma3ForCausalLM": "gemma3",
    "LlamaForCausalLM": "llama",
    "Qwen2ForCausalLM": "qwen2",
    "Qwen3ForCausalLM": "qwen3",
    "Qwen3MoeForCausalLM": "qwen3moe",
    "Qwen3_5MoeForConditionalGeneration": "qwen35moe",
    "Lfm2ForCausalLM": "lfm2",
    "Qwen3_5ForCausalLM": "qwen35",
    "Qwen3_5ForConditionalGeneration": "qwen35",
}
_MODEL_TYPE_TO_GGUF_ARCHITECTURE = {
    "gemma3": "gemma3",
    "llama": "llama",
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "qwen3_moe": "qwen3moe",
    "qwen3_5_moe": "qwen35moe",
    "lfm2": "lfm2",
    "qwen3_5": "qwen35",
}


class NativeDescriptorError(ValueError):
    pass


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def native_sources() -> tuple[dict[str, Any], ...]:
    onnx = {
        "schemaVersion": SOURCE_SCHEMA_VERSION,
        "provider": {
            "id": "oryn.source.onnxruntime",
            "version": "1.0.0",
            "generatorDigest": _digest("onnxruntime-1.22-operators-coreml-cpu-v1"),
        },
        "foundation": {
            "id": "oryn.runtime.onnx",
            "version": "1.0.0",
            "digest": _digest("onnxruntime-1.22-coreml-cpu-abi1"),
        },
        "capabilities": [
            _capability(
                "onnx.kokoro.speech-synthesis",
                "onnx",
                "kokoro-v1",
                "text-to-speech",
                ["onnx", "json", "tokenizer"],
                "oryn.family.onnx.kokoro-v1",
                "oryn.task.speech-synthesis-v1",
                "certified",
            ),
            _capability(
                "onnx.bert.sequence-classification",
                "onnx",
                "bert-sequence-classification-v1",
                "text-classification",
                ["onnx", "json", "tokenizer"],
                "oryn.family.onnx.bert-encoder-v1",
                "oryn.task.text-classification-v1",
                "untested",
            ),
        ],
    }
    return (onnx,)


def native_registry() -> CapabilityRegistry:
    from oryn_engine.generated_capabilities import generated_sources

    return compile_registry((*generated_sources(), *native_sources()))


def normalized_gguf_architecture(info: object, config: dict[str, Any]) -> str | None:
    """Resolve Hub discovery metadata into llama.cpp's GGUF identifier namespace.

    Hugging Face exposes the container-native identifier in ``ModelInfo.gguf``.
    Older Hub responses may omit that summary, so the Transformers mapping is a
    bounded fallback only.  The downloaded GGUF header remains authoritative.
    """

    summary = getattr(info, "gguf", None)
    if isinstance(summary, dict):
        architecture = summary.get("architecture")
        if architecture in LLAMA_CPP_CHAT_ARCHITECTURES:
            return str(architecture)
        if isinstance(architecture, str) and re.fullmatch(r"[a-z0-9_.-]{1,128}", architecture):
            return architecture

    architectures = config.get("architectures")
    if isinstance(architectures, list):
        mapped = {
            _TRANSFORMERS_TO_GGUF_ARCHITECTURE[item]
            for item in architectures
            if isinstance(item, str) and item in _TRANSFORMERS_TO_GGUF_ARCHITECTURE
        }
        if len(mapped) == 1:
            return mapped.pop()

    model_type = config.get("model_type")
    return _MODEL_TYPE_TO_GGUF_ARCHITECTURE.get(model_type) if isinstance(model_type, str) else None


def llama_cpp_chat_template(architecture: str, embedded: str | None = None) -> str | None:
    """Return the sealed task adapter implemented for an admitted family."""

    item = next((item for item in _admitted_native() if item["architecture"] == architecture), None)
    if item is None:
        return None
    templates = item["constraints"]["templates"]
    if embedded is None:
        return templates[0]  # provisional discovery only; installation checks the header
    from oryn_engine.chat_formats import select

    return select(embedded, allowed=templates)


def validate_gguf_policy(metadata, template=None):
    item = next(
        (item for item in _admitted_native() if item["architecture"] == metadata.architecture), None
    )
    if item is None:
        raise NativeDescriptorError("GGUF native task is withheld by policy.")
    constraints = item["constraints"]
    if (
        str(metadata.version) not in constraints["containerVersions"]
        or metadata.quantization not in constraints["quantizations"]
        or metadata.tokenizer_model not in constraints["tokenizers"]
        or (template is not None and template not in constraints["templates"])
    ):
        raise NativeDescriptorError("GGUF tuple is outside its signed policy.")
    # Embedded Jinja remains data. Only known framing markers select the sealed renderer.
    selected = llama_cpp_chat_template(metadata.architecture, metadata.chat_template)
    if selected is None or (template is not None and template != selected):
        raise NativeDescriptorError("GGUF chat framing does not match the sealed renderer.")
    return constraints


def validate_onnx_selected_files(files: Iterable[tuple[str, int, str]]) -> tuple[str, ...]:
    """Validate immutable ONNX graph/external-data membership before a worker sees it."""

    normalized: list[str] = []
    aggregate = 0
    graph_count = 0
    external_count = 0
    seen: set[str] = set()
    for raw_name, size, digest in files:
        path = PurePosixPath(raw_name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "" in path.parts
            or len(raw_name) > 1_024
            or raw_name in seen
            or not isinstance(size, int)
            or isinstance(size, bool)
            or not 1 <= size <= MAX_ONNX_GRAPH_BYTES
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or path.suffix.lower() in FORBIDDEN_NATIVE_SUFFIXES
            or path.name.startswith(".")
        ):
            raise NativeDescriptorError("ONNX selected-file set is invalid.")
        seen.add(raw_name)
        aggregate += size
        if aggregate > MAX_ONNX_GRAPH_BYTES:
            raise NativeDescriptorError("ONNX selected-file set exceeds its size bound.")
        if path.suffix.lower() == ".onnx":
            graph_count += 1
        elif path.suffix.lower() in {".data", ".bin", ".json", ".txt", ".model"}:
            external_count += 1
        else:
            raise NativeDescriptorError("ONNX selected-file type is unsupported.")
        if external_count > MAX_ONNX_EXTERNAL_FILES:
            raise NativeDescriptorError("ONNX external-data set is too large.")
        normalized.append(raw_name)
    if graph_count != 1:
        raise NativeDescriptorError("Exactly one ONNX graph is required.")
    return tuple(sorted(normalized))
