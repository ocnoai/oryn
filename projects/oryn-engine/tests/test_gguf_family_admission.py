"""Exercise catalog admission through the worker's real, bounded GGUF reader.

The framing fragments and container/tokenizer/quantization tuples come from the
audited LFM3, Spark-X2.6 and Flint GGUF headers. The tiny tensor is a structural
fixture, not an executable model or evidence of native inference conformance.
"""

from __future__ import annotations

import struct

import pytest

from oryn_engine.native_capability_sources import (
    NativeDescriptorError,
    llama_cpp_chat_template,
    validate_gguf_policy,
)
from oryn_engine.runtime_packs import llama_cpp_worker as worker
from oryn_engine.runtime_packs.gguf import GGUFError

CASES = (
    pytest.param(
        "lfm3",
        "lfm3",
        "{{- bos_token -}}<|im_start|>system<|im_end|>"
        "<|im_start|>user<|im_end|><|im_start|>assistant<think>"
        "CONTINUE_FINAL_MESSAGE_TAG ",
        15,
        12,
        "Q4_K_M",
        id="lfm3-owned-format-q4-k-m",
    ),
    pytest.param(
        "spark2_5",
        "spark",
        "<｜start▁of▁sentence｜><|System|><|User|><|Bot|><｜end▁of▁sentence｜>",
        15,
        12,
        "Q4_K_M",
        id="spark2-5-spark-q4-k-m",
    ),
    pytest.param(
        "flint",
        "flint",
        "<|begin_of_text|><|system|><|user|><|assistant|><|end_of_text|>",
        7,
        8,
        "Q8_0",
        id="flint-flint-q8-0",
    ),
)


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf(
    architecture: str,
    template: str,
    *,
    tokenizer: str = "gpt2",
    file_type: int = 15,
    tensor_type: int = 12,
) -> bytes:
    def entry(name, kind, value):
        return _string(name) + struct.pack("<I", kind) + value

    metadata = [
        entry("general.architecture", 8, _string(architecture)),
        entry("general.file_type", 4, struct.pack("<I", file_type)),
        entry(f"{architecture}.context_length", 4, struct.pack("<I", 4096)),
        entry("tokenizer.ggml.model", 8, _string(tokenizer)),
        entry("tokenizer.chat_template", 8, _string(template)),
        entry("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + _string("a") + _string("b")),
    ]
    elements, block_bytes = {12: (256, 144), 8: (32, 34)}[tensor_type]
    tensor = _string("token_embd.weight") + struct.pack("<IQIQ", 1, elements, tensor_type, 0)
    raw = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata)) + b"".join(metadata) + tensor
    raw += bytes(-len(raw) % 32)
    return raw + bytes(block_bytes)


@pytest.mark.parametrize("architecture,format_name,embedded,file_type,tensor_type,quant", CASES)
def test_worker_admits_new_families_from_validated_headers(
    tmp_path, architecture, format_name, embedded, file_type, tensor_type, quant
):
    # Even an executable-looking repository template remains metadata. Admission
    # selects the owned format; it never evaluates the repository's Jinja.
    template = "{{ raise_exception('repository template must stay inert') }}" + embedded
    model = tmp_path / "model.gguf"
    model.write_bytes(_gguf(architecture, template, file_type=file_type, tensor_type=tensor_type))

    path, metadata = worker._model({"modelRoot": str(tmp_path), "modelFile": model.name})

    assert path == model.resolve()
    assert (metadata.architecture, metadata.version, metadata.tokenizer_model) == (
        architecture,
        3,
        "gpt2",
    )
    assert metadata.quantization == quant
    assert llama_cpp_chat_template(architecture, metadata.chat_template) == format_name
    validate_gguf_policy(metadata, format_name)


@pytest.mark.parametrize("architecture,format_name,embedded,file_type,tensor_type,quant", CASES)
@pytest.mark.parametrize("invalid", ["tokenizer", "template", "quantization"])
def test_new_family_admission_keeps_unreviewed_header_tuples_closed(
    tmp_path, architecture, format_name, embedded, file_type, tensor_type, quant, invalid
):
    model = tmp_path / "model.gguf"
    model.write_bytes(
        _gguf(
            architecture,
            "{{ arbitrary_repository_template }}" if invalid == "template" else embedded,
            tokenizer="custom-repository-tokenizer" if invalid == "tokenizer" else "gpt2",
            file_type=36 if invalid == "quantization" else file_type,
            tensor_type=tensor_type,
        )
    )

    with pytest.raises(NativeDescriptorError):
        worker._model({"modelRoot": str(tmp_path), "modelFile": model.name})


@pytest.mark.parametrize("architecture,format_name,embedded,file_type,tensor_type,quant", CASES)
def test_new_family_cannot_override_its_trusted_prompt_format(
    tmp_path, architecture, format_name, embedded, file_type, tensor_type, quant
):
    model = tmp_path / "model.gguf"
    model.write_bytes(_gguf(architecture, embedded, file_type=file_type, tensor_type=tensor_type))
    _, metadata = worker._model({"modelRoot": str(tmp_path), "modelFile": model.name})

    with pytest.raises(NativeDescriptorError, match="outside its signed policy"):
        validate_gguf_policy(metadata, "llama3")


@pytest.mark.parametrize("architecture,format_name,embedded,file_type,tensor_type,quant", CASES)
def test_catalog_admission_does_not_bypass_tensor_extent_validation(
    tmp_path, architecture, format_name, embedded, file_type, tensor_type, quant
):
    model = tmp_path / "model.gguf"
    raw = _gguf(architecture, embedded, file_type=file_type, tensor_type=tensor_type)
    model.write_bytes(raw[:-1])

    with pytest.raises(GGUFError, match="offset"):
        worker._model({"modelRoot": str(tmp_path), "modelFile": model.name})
