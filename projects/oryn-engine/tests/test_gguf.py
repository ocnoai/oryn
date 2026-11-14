from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path

import pytest

from oryn_engine.runtime_packs.gguf import (
    GGUFError,
    inspect_gguf,
    inspect_gguf_set,
    inspect_projector,
    split_file_set,
)


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _metadata(key: str, value_type: int, value: bytes) -> bytes:
    return _string(key) + struct.pack("<I", value_type) + value


def _valid_gguf(
    *,
    tensor_offset: int = 0,
    split_count: int = 1,
    tensor_type: int = 0,
    elements: int = 1,
    file_type: int = 10,
    architecture: str = "llama",
    tensor_name: str = "token_embd.weight",
    payload_bytes: int | None = None,
) -> bytes:
    metadata = [
        _metadata("general.architecture", 8, _string(architecture)),
        _metadata("general.name", 8, _string("Oryn test model")),
        _metadata("general.file_type", 4, struct.pack("<I", file_type)),
        _metadata("general.alignment", 4, struct.pack("<I", 32)),
        _metadata("split.count", 4, struct.pack("<I", split_count)),
        _metadata(f"{architecture}.context_length", 4, struct.pack("<I", 4096)),
        _metadata("tokenizer.ggml.model", 8, _string("gpt2")),
        _metadata("tokenizer.chat_template", 8, _string("{{ messages }}")),
        _metadata(
            "tokenizer.ggml.tokens",
            9,
            struct.pack("<IQ", 8, 2) + _string("alpha") + _string("beta"),
        ),
    ]
    tensor = _string(tensor_name) + struct.pack("<IQIQ", 1, elements, tensor_type, tensor_offset)
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata))
    value = header + b"".join(metadata) + tensor
    value += b"\x00" * ((32 - len(value) % 32) % 32)
    return value + (
        bytes(payload_bytes)
        if payload_bytes is not None
        else bytes(136)
        if tensor_type == 23
        else struct.pack("<f", 1.0)
    )


@pytest.mark.parametrize(
    "file_type,tensor_type,elements,size,expected",
    [
        (1, 1, 2, 4, "F16"),
        (14, 12, 256, 144, "Q4_K_S"),
    ],
)
def test_native_float_and_q4_small_block_storage(
    tmp_path, file_type, tensor_type, elements, size, expected
):
    path = tmp_path / "model.gguf"
    path.write_bytes(
        _valid_gguf(
            file_type=file_type, tensor_type=tensor_type, elements=elements, payload_bytes=size
        )
    )
    assert inspect_gguf(path).quantization == expected
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(GGUFError, match="offset"):
        inspect_gguf(path)


@pytest.mark.parametrize("architecture", ["qwen35", "qwen35moe"])
@pytest.mark.parametrize("tensor_type", [1, 12, 30])
def test_state_space_convolution_requires_native_f32_before_worker_loading(
    tmp_path, architecture, tensor_type
):
    path = tmp_path / "model.gguf"
    path.write_bytes(
        _valid_gguf(
            architecture=architecture,
            tensor_name="blk.0.ssm_conv1d.weight",
            tensor_type=tensor_type,
        )
    )
    with pytest.raises(GGUFError, match="convolution requires F32"):
        inspect_gguf(path)
    path.write_bytes(_valid_gguf(architecture=architecture, tensor_name="blk.0.ssm_conv1d.weight"))
    assert inspect_gguf(path).architecture == architecture


def test_strict_gguf_inspection_reports_execution_metadata(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(_valid_gguf())

    result = inspect_gguf(path)

    assert result.version == 3
    assert result.architecture == "llama"
    assert result.quantization == "Q2_K"
    assert result.context_length == 4096
    assert result.token_count == 2
    assert result.tensor_count == 1


@pytest.mark.parametrize(
    ("architecture", "projector_kind"),
    [
        ("qwen2vl", "qwen2vl_merger"),
        ("qwen2vl", "qwen2.5vl_merger"),
        ("qwen35", "qwen3vl_merger"),
        ("qwen35moe", "qwen3vl_merger"),
    ],
)
def test_projector_requires_complete_matching_native_graph(tmp_path, architecture, projector_kind):
    language_path = tmp_path / "language.gguf"
    language_path.write_bytes(_valid_gguf())
    language = replace(
        inspect_gguf(language_path), architecture=architecture, embedding_length=1536
    )
    projector = tmp_path / "mmproj-test-f16.gguf"

    def content(dimension=1536, kind=projector_kind, tensor_type=1):
        metadata = [
            _metadata("general.architecture", 8, _string("clip")),
            _metadata("general.file_type", 4, struct.pack("<I", 1)),
            _metadata("clip.has_vision_encoder", 7, b"\x01"),
            _metadata("clip.projector_type", 8, _string(kind)),
            _metadata("clip.vision.projection_dim", 4, struct.pack("<I", dimension)),
        ]
        tensor = _string("mm.0.weight") + struct.pack("<IQIQ", 1, 1, tensor_type, 0)
        data = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata)) + b"".join(metadata) + tensor
        return data + bytes((-len(data)) % 32) + b"\x00\x3c"

    projector.write_bytes(content())
    assert inspect_projector(projector, language).projection_length == 1536
    with pytest.raises(GGUFError):
        inspect_projector(projector, replace(language, architecture="llama"))
    wrong_family = "qwen35" if architecture == "qwen2vl" else "qwen2vl"
    with pytest.raises(GGUFError):
        inspect_projector(projector, replace(language, architecture=wrong_family))
    # A projector must never be admitted as a language model.
    with pytest.raises(GGUFError):
        inspect_gguf(projector)
    for bad in (content(2048), content(kind="unreviewed"), content()[:-1], content(tensor_type=8)):
        projector.write_bytes(bad)
        with pytest.raises(GGUFError):
            inspect_projector(projector, language)
    projector.unlink()
    with pytest.raises(GGUFError):
        inspect_projector(projector, language)


def test_iq4_xs_requires_complete_native_quantization_blocks(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    complete = _valid_gguf(tensor_type=23, elements=256, file_type=30)
    path.write_bytes(complete)
    assert inspect_gguf(path).quantization == "IQ4_XS"
    for invalid in (
        complete[:-1],
        _valid_gguf(tensor_type=23, elements=255),
        _valid_gguf(tensor_type=23, elements=512),
    ):
        path.write_bytes(invalid)
        with pytest.raises(GGUFError):
            inspect_gguf(path)


@pytest.mark.parametrize(
    "content",
    [
        b"NOPE" + _valid_gguf()[4:],
        _valid_gguf(tensor_offset=1),
        _valid_gguf(split_count=2),
        _valid_gguf()[:-1],
    ],
)
def test_strict_gguf_inspection_rejects_invalid_or_split_files(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(content)

    with pytest.raises(GGUFError):
        inspect_gguf(path)


def split_fixture(index, *, total=2, aggregate=2, tensor_name=None):
    metadata = [
        _metadata("split.count", 2, struct.pack("<H", total)),
        _metadata("split.no", 2, struct.pack("<H", index)),
        _metadata("split.tensors.count", 5, struct.pack("<i", aggregate)),
    ]
    if index == 0:
        metadata += [
            _metadata("general.architecture", 8, _string("llama")),
            _metadata("general.name", 8, _string("Split fixture")),
            _metadata("general.file_type", 4, struct.pack("<I", 15)),
            _metadata("llama.context_length", 4, struct.pack("<I", 4096)),
            _metadata("tokenizer.ggml.model", 8, _string("gpt2")),
            _metadata("tokenizer.chat_template", 8, _string("<|im_start|><|im_end|>")),
            _metadata(
                "tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + _string("a") + _string("b")
            ),
        ]
    tensor = _string(tensor_name or f"layer.{index}.weight") + struct.pack("<IQIQ", 1, 1, 0, 0)
    value = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata)) + b"".join(metadata) + tensor
    value += bytes((-len(value)) % 32)
    return value + struct.pack("<f", 1.0)


def test_complete_split_set_checks_all_members_and_shared_tensor_count(tmp_path):
    names = [f"model-Q4_K_M-{i:05d}-of-00002.gguf" for i in (1, 2)]
    paths = [tmp_path / name for name in names]
    for i, path in enumerate(paths):
        path.write_bytes(split_fixture(i))
    result = inspect_gguf_set(paths, names)
    assert result.tensor_count == 2
    assert result.file_size == sum(p.stat().st_size for p in paths)
    assert result.token_count == 2
    assert result.quantization == "Q4_K_M"
    for content in [
        split_fixture(0),
        split_fixture(1, total=3),
        split_fixture(1, aggregate=3),
        split_fixture(1, tensor_name="layer.0.weight"),
        split_fixture(1)[:-1],
    ]:
        paths[1].write_bytes(content)
        with pytest.raises(GGUFError):
            inspect_gguf_set(paths, names)
    assert not split_file_set(names[:1], names[0])
    assert not split_file_set(names, names[1])
    assert not split_file_set(names + [names[0]], names[0])
    assert split_file_set(names, names[0]) == names


def _projector(*entries):
    metadata = [
        _metadata("general.architecture", 8, _string("clip")),
        _metadata("general.file_type", 4, struct.pack("<I", 1)),
        _metadata("clip.has_vision_encoder", 7, b"\x01"),
        _metadata("clip.vision.projection_dim", 4, struct.pack("<I", 2560)),
        *(_metadata(key, 8, _string(value)) for key, value in entries),
    ]
    tensor = _string("mm.0.weight") + struct.pack("<IQIQ", 1, 1, 1, 0)
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata)) + b"".join(metadata) + tensor
    return data + bytes((-len(data)) % 32) + b"\x00\x3c"


@pytest.mark.parametrize(
    "entries,admitted",
    [
        ([("clip.vision.projector_type", "gemma4v")], True),
        ([("clip.projector_type", "gemma4v")], True),
        ([("clip.projector_type", "gemma4v"), ("clip.vision.projector_type", "gemma4v")], True),
        ([("clip.projector_type", "gemma4v"), ("clip.vision.projector_type", "gemma3")], False),
        ([("clip.vision.projector_type", "gemma4a")], False),
        ([], False),
    ],
)
def test_multimodal_projector_names_its_vision_graph_per_modality(tmp_path, entries, admitted):
    language_path = tmp_path / "language.gguf"
    language_path.write_bytes(_valid_gguf())
    language = replace(inspect_gguf(language_path), architecture="gemma4", embedding_length=2560)
    projector = tmp_path / "mmproj-gemma4-f16.gguf"
    projector.write_bytes(_projector(*entries))
    if admitted:
        assert inspect_projector(projector, language).projector_type == "gemma4v"
    else:
        with pytest.raises(GGUFError):
            inspect_projector(projector, language)


@pytest.mark.parametrize(
    "name,valid",
    [
        ("model-Q4_K_M.gguf", True),
        ("Q4_K_M/model-Q4_K_M-00001-of-00003.gguf", True),
        ("UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf", True),
        ("a/b/model.gguf", False),
        ("../model.gguf", False),
        ("./model.gguf", False),
        ("/model.gguf", False),
        (".hidden/model.gguf", False),
        ("folder\\model.gguf", False),
        ("folder/model.bin", False),
        ("folder/", False),
        (None, False),
    ],
)
def test_member_names_allow_the_root_or_one_safe_folder(name, valid):
    from oryn_engine.runtime_packs.gguf import safe_member_name

    assert safe_member_name(name) is valid


QK_K, HALF = 256, 2


@pytest.mark.parametrize(
    "file_type,tensor_type,block_bytes,expected",
    [
        # Block sizes as written in llama.cpp ggml-common.h static_asserts.
        (19, 16, HALF + QK_K // 8 * 2, "IQ2_XXS"),
        (20, 17, HALF + QK_K // 8 * 2 + QK_K // 32, "IQ2_XS"),
        (23, 18, HALF + 3 * (QK_K // 8), "IQ3_XXS"),
        (24, 19, HALF + QK_K // 8 + QK_K // 16, "IQ1_S"),
        (26, 21, HALF + 13 * (QK_K // 32) + QK_K // 64, "IQ3_S"),
        (28, 22, HALF + QK_K // 4 + QK_K // 16, "IQ2_S"),
        (31, 29, QK_K // 8 + QK_K // 16 + QK_K // 32, "IQ1_M"),
    ],
)
def test_i_quant_blocks_match_the_pinned_ggml_layout(
    tmp_path, file_type, tensor_type, block_bytes, expected
):
    from oryn_engine.runtime_packs.gguf import GGML_TYPE_LAYOUTS

    assert GGML_TYPE_LAYOUTS[tensor_type] == (QK_K, block_bytes)
    path = tmp_path / "model.gguf"
    path.write_bytes(
        _valid_gguf(
            file_type=file_type,
            tensor_type=tensor_type,
            elements=QK_K,
            payload_bytes=block_bytes,
        )
    )
    assert inspect_gguf(path).quantization == expected
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(GGUFError):
        inspect_gguf(path)


@pytest.mark.parametrize(
    "file_type,tensor_type,admitted",
    [(1, 1, True), (0, 0, True), (32, 30, True), (7, 8, False), (15, 12, False)],
)
def test_projectors_must_stay_unquantized_but_may_be_bf16(
    tmp_path, file_type, tensor_type, admitted
):
    """F32, F16 and BF16 are plain floats; real Qwen3.8 BF16 projectors run on Metal."""
    language_path = tmp_path / "language.gguf"
    language_path.write_bytes(_valid_gguf())
    language = replace(inspect_gguf(language_path), architecture="qwen35", embedding_length=1536)
    metadata = [
        _metadata("general.architecture", 8, _string("clip")),
        _metadata("general.file_type", 4, struct.pack("<I", file_type)),
        _metadata("clip.has_vision_encoder", 7, b"\x01"),
        _metadata("clip.projector_type", 8, _string("qwen3vl_merger")),
        _metadata("clip.vision.projection_dim", 4, struct.pack("<I", 1536)),
    ]
    elements, payload = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34), 12: (256, 144)}[tensor_type]
    tensor = _string("mm.0.weight") + struct.pack("<IQIQ", 1, elements, tensor_type, 0)
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata)) + b"".join(metadata) + tensor
    projector = tmp_path / "mmproj-test.gguf"
    projector.write_bytes(data + bytes((-len(data)) % 32) + bytes(payload))
    if admitted:
        assert inspect_projector(projector, language).projection_length == 1536
    else:
        with pytest.raises(GGUFError):
            inspect_projector(projector, language)


def test_q2_0_blocks_follow_the_pinned_llama_cpp_layout(tmp_path):
    """block_q2_0 is 64 values in 18 bytes at llama.cpp 9d1e0361."""
    path = tmp_path / "model.gguf"
    complete = _valid_gguf(file_type=41, tensor_type=42, elements=64, payload_bytes=18)
    path.write_bytes(complete)
    result = inspect_gguf(path)
    assert result.quantization == "Q2_0" and result.tensor_types == (42,)
    for invalid in (
        complete[:-1],
        _valid_gguf(file_type=41, tensor_type=42, elements=65, payload_bytes=18),
    ):
        path.write_bytes(invalid)
        with pytest.raises(GGUFError):
            inspect_gguf(path)


def test_nvfp4_tensors_stay_unadmitted_until_they_run_on_metal(tmp_path):
    """The pinned Metal library has no NVFP4 kernels, so the reader rejects type 40."""
    path = tmp_path / "model.gguf"
    path.write_bytes(_valid_gguf(file_type=19, tensor_type=40, elements=64, payload_bytes=36))
    with pytest.raises(GGUFError, match="tensor type is unsupported"):
        inspect_gguf(path)


def _ranges(data: bytes, calls: list[tuple[int, int]]):
    def read_range(start, length):
        calls.append((start, length))
        return data[start : start + length]

    return read_range


def test_remote_header_read_matches_the_local_reader_and_never_reads_tensor_data(tmp_path):
    from oryn_engine.runtime_packs.gguf import describe, inspect_remote_header

    payload = 3 * 1024 * 1024
    data = _valid_gguf(file_type=1, tensor_type=1, elements=payload // 2, payload_bytes=payload)
    path = tmp_path / "model.gguf"
    path.write_bytes(data)
    calls = []
    remote = inspect_remote_header(_ranges(data, calls), len(data))
    assert describe(remote) == describe(inspect_gguf(path))
    assert describe(remote)["tensorTypes"] == [1]
    # One bounded range covers the header; the 3 MiB of tensor data is never requested.
    assert calls == [(0, 1024 * 1024)]


def test_remote_header_reads_are_bounded_and_strict(tmp_path):
    from oryn_engine.runtime_packs import gguf

    data = _valid_gguf()
    with pytest.raises(GGUFError, match="size"):
        gguf.inspect_remote_header(_ranges(data, []), 10)
    with pytest.raises(GGUFError, match="truncated"):
        gguf.inspect_remote_header(lambda start, length: b"GGUF", len(data))
    # A header larger than the remote budget is rejected before more is fetched.
    oversized = _valid_gguf(tensor_name="t" * 1000)
    calls = []
    source = gguf._RangeSource(_ranges(oversized, calls), len(oversized), limit=64)
    with pytest.raises(GGUFError, match="remote read limit"):
        gguf._inspect_source(source, len(oversized), "remote")
    assert sum(length for _start, length in calls) <= 64


def _labelled_projector(tmp_path, file_type, tensor_type, kind="qwen3vl_merger", dimension=4096):
    metadata = [
        _metadata("general.architecture", 8, _string("clip")),
        _metadata("general.file_type", 4, struct.pack("<I", file_type)),
        _metadata("clip.has_vision_encoder", 7, b"\x01"),
        _metadata("clip.projector_type", 8, _string(kind)),
        _metadata("clip.vision.projection_dim", 4, struct.pack("<I", dimension)),
    ]
    elements, payload = {0: (1, 4), 30: (1, 2), 8: (32, 34)}[tensor_type]
    tensor = _string("mm.0.weight") + struct.pack("<IQIQ", 1, elements, tensor_type, 0)
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata)) + b"".join(metadata) + tensor
    path = tmp_path / "mmproj-model.gguf"
    path.write_bytes(data + bytes((-len(data)) % 32) + bytes(payload))
    return path


def test_projectors_are_judged_by_their_tensors_not_their_file_type_label(tmp_path):
    """Converters copy the language model's label: NVIDIA's Nemotron projectors store BF16
    tensors under an NVFP4 file-type label (39)."""
    language_path = tmp_path / "language.gguf"
    language_path.write_bytes(_valid_gguf(architecture="qwen35"))
    language = replace(inspect_gguf(language_path), embedding_length=4096)
    assert inspect_projector(_labelled_projector(tmp_path, 39, 30), language).projector_type == (
        "qwen3vl_merger"
    )
    for path in (
        # Compressed tensors stay rejected whatever the label says.
        _labelled_projector(tmp_path, 1, 8),
        # A projector type not reviewed for this language graph.
        _labelled_projector(tmp_path, 39, 30, kind="nemotron_omni"),
        # A width that differs from the language graph.
        _labelled_projector(tmp_path, 39, 30, dimension=2048),
    ):
        with pytest.raises(GGUFError):
            inspect_projector(path, language)
