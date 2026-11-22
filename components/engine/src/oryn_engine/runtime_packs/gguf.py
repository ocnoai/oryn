from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO

MAX_GGUF_BYTES = 128 * 1024 * 1024 * 1024
MAX_METADATA_ENTRIES = 65_536
MAX_TENSORS = 1_000_000
MAX_ARRAY_ITEMS = 2_000_000
MAX_STRING_BYTES = 16 * 1024 * 1024
MAX_METADATA_BYTES = 512 * 1024 * 1024
MAX_TENSOR_DIMENSIONS = 4
SPLIT_NAME = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf$")
MAX_SPLITS = 4096
_MEMBER_FOLDER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_MEMBER_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\.gguf")


def safe_member_name(name: object) -> bool:
    """A repository-relative GGUF member: a file at the root or inside one folder.

    Large repositories keep each quantization in its own folder. Deeper paths,
    absolute paths, backslashes and dot segments are never members.
    """
    if not isinstance(name, str) or not 6 <= len(name) <= 256 or "\\" in name:
        return False
    parts = name.split("/")
    if len(parts) == 1:
        return _MEMBER_FILE.fullmatch(parts[0]) is not None
    return (
        len(parts) == 2
        and _MEMBER_FOLDER.fullmatch(parts[0]) is not None
        and _MEMBER_FILE.fullmatch(parts[1]) is not None
    )


VALUE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
INTEGER_TYPES = {0, 1, 2, 3, 4, 5, 10, 11}
FLOAT_TYPES = {6, 12}
STRING_TYPE = 8
ARRAY_TYPE = 9
QUANTIZATION_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    # llama.h LLAMA_FTYPE_MOSTLY_Q2_0 at 9d1e0361.
    41: "Q2_0",
}
GGML_TYPE_LAYOUTS = {
    0: (1, 4),
    1: (1, 2),
    2: (32, 18),
    3: (32, 20),
    6: (32, 22),
    7: (32, 24),
    8: (32, 34),
    9: (32, 40),
    10: (256, 84),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    15: (256, 292),
    20: (32, 18),
    # ggml type 23, block_iq4_xs: 2 + 2 + 256/64 + 256/2 bytes.
    # Verified against llama.cpp de699957 and 9d1e0361 ggml.h / ggml-common.h (build-inputs).
    23: (256, 136),
    24: (1, 1),
    25: (1, 2),
    26: (1, 4),
    27: (1, 8),
    28: (1, 8),
    30: (1, 2),
    # i-quant blocks, from the static_asserts in llama.cpp ggml-common.h at de699957 and
    # 9d1e0361 (identical): QK_K = 256 values per block, ggml_half = 2 bytes.
    16: (256, 66),  # block_iq2_xxs: 2 + QK_K/8 * 2
    17: (256, 74),  # block_iq2_xs: 2 + QK_K/8 * 2 + QK_K/32
    18: (256, 98),  # block_iq3_xxs: 2 + 3 * (QK_K/8)
    19: (256, 50),  # block_iq1_s: 2 + QK_K/8 + QK_K/16
    21: (256, 110),  # block_iq3_s: 2 + 13 * (QK_K/32) + QK_K/64
    22: (256, 82),  # block_iq2_s: 2 + QK_K/4 + QK_K/16
    29: (256, 56),  # block_iq1_m: QK_K/8 + QK_K/16 + QK_K/32
    # From the static_asserts in llama.cpp ggml-common.h at 9d1e0361 (ggml.h type numbers).
    # The pinned Metal library has Q2_0 kernels. NVFP4 (type 40) is not admitted: that
    # build has no Metal kernels for it, and it has not run here.
    42: (64, 18),  # block_q2_0: ggml_half + QK2_0/4
}
# A remote header read fetches only metadata and tensor descriptions, never tensor data.
MAX_REMOTE_HEADER_BYTES = 64 * 1024 * 1024
_REMOTE_CHUNK_BYTES = 1024 * 1024


class GGUFError(ValueError):
    pass


@dataclass(frozen=True)
class GGUFMetadata:
    version: int
    architecture: str
    name: str
    tokenizer_model: str
    chat_template: str
    context_length: int
    quantization: str
    file_type: int
    token_count: int
    tensor_count: int
    file_size: int
    split_count: int = 1
    split_index: int = 0
    split_tensor_count: int = 0
    tensor_names: tuple[str, ...] = ()
    embedding_length: int = 0
    projector_type: str = ""
    projection_length: int = 0
    tensor_types: tuple[int, ...] = ()


class _Reader:
    def __init__(self, source: BinaryIO, file_size: int) -> None:
        self.source = source
        self.file_size = file_size
        self.metadata_bytes = 0

    def read(self, size: int, *, metadata: bool = False) -> bytes:
        if size < 0 or size > self.file_size - self.source.tell():
            raise GGUFError("GGUF structure exceeds the file boundary")
        if metadata:
            self.metadata_bytes += size
            if self.metadata_bytes > MAX_METADATA_BYTES:
                raise GGUFError("GGUF metadata exceeds the size limit")
        value = self.source.read(size)
        if len(value) != size:
            raise GGUFError("GGUF is truncated")
        return value

    def u32(self, *, metadata: bool = False) -> int:
        return struct.unpack("<I", self.read(4, metadata=metadata))[0]

    def u64(self, *, metadata: bool = False) -> int:
        return struct.unpack("<Q", self.read(8, metadata=metadata))[0]

    def string(self, *, limit: int = MAX_STRING_BYTES, metadata: bool = True) -> str:
        size = self.u64(metadata=metadata)
        if size > limit:
            raise GGUFError("GGUF string exceeds the size limit")
        raw = self.read(size, metadata=metadata)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError("GGUF string is not valid UTF-8") from exc

    def skip_value(self, value_type: int) -> int | float | bool | str | None:
        if value_type in INTEGER_TYPES:
            raw = self.read(VALUE_SIZES[value_type], metadata=True)
            signed = value_type in {1, 3, 5, 11}
            return int.from_bytes(raw, "little", signed=signed)
        if value_type in FLOAT_TYPES:
            format_ = "<f" if value_type == 6 else "<d"
            value = struct.unpack(format_, self.read(VALUE_SIZES[value_type], metadata=True))[0]
            if not math.isfinite(value):
                raise GGUFError("GGUF floating metadata is not finite")
            return value
        if value_type == 7:
            value = self.read(1, metadata=True)[0]
            if value not in {0, 1}:
                raise GGUFError("GGUF boolean metadata is invalid")
            return bool(value)
        if value_type == STRING_TYPE:
            return self.string()
        if value_type == ARRAY_TYPE:
            element_type = self.u32(metadata=True)
            count = self.u64(metadata=True)
            if count > MAX_ARRAY_ITEMS or element_type == ARRAY_TYPE:
                raise GGUFError("GGUF metadata array is invalid")
            if element_type in VALUE_SIZES:
                self.read(VALUE_SIZES[element_type] * count, metadata=True)
            elif element_type == STRING_TYPE:
                for _ in range(count):
                    self.string(limit=1 * 1024 * 1024)
            else:
                raise GGUFError("GGUF metadata type is unsupported")
            return count
        raise GGUFError("GGUF metadata type is unsupported")


def inspect_gguf(path: Path) -> GGUFMetadata:
    return _inspect_member(path)


def inspect_projector(path: Path, language: GGUFMetadata) -> GGUFMetadata:
    """A projector is a separate graph member; it cannot pass the language loader."""
    result = _inspect_member(path, projector=True)
    check_projector(result, language.architecture, language.embedding_length)
    return result


def check_projector(result: GGUFMetadata, architecture: str, embedding_length: object) -> None:
    """The projector type is reviewed for the language graph and its width matches it.

    Plain-float storage is decided by the tensors themselves (F32, F16 or BF16, checked
    while reading). The file-type label is not: converters copy it from the language
    model, so a BF16 projector can carry an NVFP4 label.
    """
    from oryn_engine.generated_capabilities import generated_artifact

    admitted = generated_artifact()["buildInputs"].get("llamaVisionProjectors", {})
    if (
        result.projector_type not in admitted.get(architecture, [])
        or type(embedding_length) is not int
        or not 1 <= embedding_length <= 65536
        or result.projection_length != embedding_length
        or not result.tensor_types
        or not set(result.tensor_types) <= {0, 1, 30}
    ):
        raise GGUFError("GGUF projector does not match the reviewed language graph")


class _RangeSource:
    """Sequential reads over bounded byte ranges; nothing past the header budget is fetched."""

    def __init__(self, read_range, file_size: int, limit: int = MAX_REMOTE_HEADER_BYTES) -> None:
        self._read_range = read_range
        self._file_size = file_size
        self._limit = limit
        self._buffer = bytearray()
        self._position = 0
        self._chunk = _REMOTE_CHUNK_BYTES

    def tell(self) -> int:
        return self._position

    def read(self, size: int) -> bytes:
        end = self._position + size
        if end > self._limit:
            raise GGUFError("GGUF header exceeds the remote read limit")
        while len(self._buffer) < end:
            start = len(self._buffer)
            wanted = max(self._chunk, end - start)
            length = min(wanted, self._limit - start, self._file_size - start)
            if length <= 0:
                break
            data = self._read_range(start, length)
            if not isinstance(data, bytes) or len(data) != length:
                raise GGUFError("GGUF header range is truncated")
            self._buffer.extend(data)
            self._chunk = min(self._chunk * 2, 16 * 1024 * 1024)
        value = bytes(self._buffer[self._position : end])
        self._position += len(value)
        return value


def inspect_remote_header(
    read_range, file_size: int, *, count: int = 1, projector: bool = False
) -> GGUFMetadata:
    """Read one GGUF member's metadata and tensor descriptions through bounded ranges.

    Tensor data is never fetched. For a split set only the first member is read, because
    it carries the metadata. Installation still re-reads every downloaded file in full.
    """
    if type(file_size) is not int or not 64 <= file_size <= MAX_GGUF_BYTES:
        raise GGUFError("GGUF file size is invalid")
    if type(count) is not int or not 1 <= count <= MAX_SPLITS:
        raise GGUFError("GGUF split count is invalid")
    return _inspect_source(
        _RangeSource(read_range, file_size), file_size, "remote", count=count, projector=projector
    )


def describe(metadata: GGUFMetadata) -> dict[str, Any]:
    """JSON-safe header facts used to choose a variant before any download."""
    return {
        "version": metadata.version,
        "architecture": metadata.architecture,
        "quantization": metadata.quantization,
        "fileType": metadata.file_type,
        "tokenizerModel": metadata.tokenizer_model,
        "chatTemplate": metadata.chat_template,
        "contextLength": metadata.context_length,
        "embeddingLength": metadata.embedding_length,
        "projectorType": metadata.projector_type,
        "projectionLength": metadata.projection_length,
        "tensorTypes": list(metadata.tensor_types),
    }


def split_file_set(names: list[str], first: str) -> list[str]:
    """Return only a complete canonical shard set; never treat one shard as a model."""
    match = SPLIT_NAME.fullmatch(first)
    if match is None:
        return [first] if first in names else []
    prefix, index, total = match.groups()
    count = int(total)
    if int(index) != 1 or not 1 <= count <= MAX_SPLITS:
        return []
    expected = [f"{prefix}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]
    related = [n for n in names if (m := SPLIT_NAME.fullmatch(n)) and m[1] == prefix]
    return expected if sorted(related) == expected else []


def inspect_gguf_set(paths: list[Path], names: list[str]) -> GGUFMetadata:
    if not names or len(paths) != len(names) or split_file_set(names, names[0]) != names:
        raise GGUFError("GGUF split set is incomplete or inconsistent")
    if len(paths) == 1 and not SPLIT_NAME.fullmatch(names[0]):
        return inspect_gguf(paths[0])
    first = _inspect_member(paths[0], count=len(paths))
    members = [first] + [
        _inspect_member(path, count=len(paths), index=i, inherited=first)
        for i, path in enumerate(paths[1:], 1)
    ]
    tensors = [name for member in members for name in member.tensor_names]
    total_size = sum(m.file_size for m in members)
    if (
        len(tensors) != len(set(tensors))
        or len(tensors) != first.split_tensor_count
        or not 1 <= len(tensors) <= MAX_TENSORS
        or total_size > MAX_GGUF_BYTES
    ):
        raise GGUFError("GGUF split tensors or aggregate size are inconsistent")
    return replace(
        first, tensor_count=len(tensors), file_size=total_size, tensor_names=tuple(tensors)
    )


def _inspect_member(
    path: Path, *, count=1, index=0, inherited: GGUFMetadata | None = None, projector=False
) -> GGUFMetadata:
    if not path.is_file():
        raise GGUFError("GGUF file is unavailable")
    file_size = path.stat().st_size
    if not 64 <= file_size <= MAX_GGUF_BYTES:
        raise GGUFError("GGUF file size is invalid")
    with path.open("rb") as source:
        return _inspect_source(
            source,
            file_size,
            path.name,
            count=count,
            index=index,
            inherited=inherited,
            projector=projector,
        )


def _inspect_source(
    source: BinaryIO,
    file_size: int,
    display_name: str,
    *,
    count=1,
    index=0,
    inherited: GGUFMetadata | None = None,
    projector=False,
) -> GGUFMetadata:
    reader = _Reader(source, file_size)
    if reader.read(4) != b"GGUF":
        raise GGUFError("GGUF magic is invalid")
    version = reader.u32()
    if version not in {2, 3}:
        raise GGUFError("GGUF version is unsupported")
    tensor_count = reader.u64()
    metadata_count = reader.u64()
    if (
        not (0 if count > 1 and index == 0 else 1) <= tensor_count <= MAX_TENSORS
        or metadata_count > MAX_METADATA_ENTRIES
    ):
        raise GGUFError("GGUF header counts are invalid")

    metadata: dict[str, Any] = {}
    token_count = 0
    seen_keys: set[str] = set()
    for _ in range(metadata_count):
        key = reader.string(limit=1_024)
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,1024}", key) or key in seen_keys:
            raise GGUFError("GGUF metadata key is invalid")
        seen_keys.add(key)
        if inherited is not None and key not in {
            "split.no",
            "split.count",
            "split.tensors.count",
        }:
            raise GGUFError("Secondary GGUF shards may only declare split metadata")
        value_type = reader.u32(metadata=True)
        value = reader.skip_value(value_type)
        if key == "tokenizer.ggml.tokens":
            if value_type != ARRAY_TYPE or not isinstance(value, int):
                raise GGUFError("GGUF tokenizer tokens metadata is invalid")
            token_count = value
        elif key in {
            "general.architecture",
            "general.name",
            "general.file_type",
            "general.alignment",
            "split.count",
            "split.no",
            "split.tensors.count",
            "tokenizer.ggml.model",
            "tokenizer.chat_template",
            "clip.projector_type",
            "clip.vision.projector_type",
            "clip.vision.projection_dim",
            "clip.has_vision_encoder",
        } or key.endswith((".context_length", ".embedding_length")):
            metadata[key] = value

    if inherited is not None:
        metadata.update(
            {
                "general.architecture": inherited.architecture,
                "general.name": inherited.name,
                "general.file_type": inherited.file_type,
                "tokenizer.ggml.model": inherited.tokenizer_model,
                "tokenizer.chat_template": inherited.chat_template,
                f"{inherited.architecture}.context_length": inherited.context_length,
                f"{inherited.architecture}.embedding_length": inherited.embedding_length,
            }
        )
        token_count = inherited.token_count

    architecture = metadata.get("general.architecture")
    name = metadata.get("general.name", display_name)
    tokenizer_model = metadata.get("tokenizer.ggml.model")
    chat_template = metadata.get("tokenizer.chat_template")
    file_type = metadata.get("general.file_type")
    context_length = metadata.get(f"{architecture}.context_length")
    alignment = metadata.get("general.alignment", 32)
    split_count = metadata.get("split.count", 1)
    if projector:
        # Multi-modality projectors name the vision graph per modality.
        declared = {
            metadata[key]
            for key in ("clip.projector_type", "clip.vision.projector_type")
            if key in metadata
        }
        if (
            architecture != "clip"
            or metadata.get("clip.has_vision_encoder") is not True
            or type(metadata.get("clip.vision.projection_dim")) is not int
            or not 1 <= metadata["clip.vision.projection_dim"] <= 65536
            or len(declared) != 1
            or not all(isinstance(value, str) for value in declared)
        ):
            raise GGUFError("GGUF projector metadata is incomplete")
        metadata["clip.projector_type"] = declared.pop()
        # These language-only fields do not apply to a projector. The ordinary
        # language path retains its mandatory tokenizer/template/context checks.
        tokenizer_model, chat_template, context_length, token_count = (
            "projector",
            "none",
            128,
            1,
        )
        file_type = metadata.get("general.file_type", 1)
    if (
        not isinstance(architecture, str)
        or not re.fullmatch(r"[a-z0-9_.-]{1,128}", architecture)
        or not isinstance(name, str)
        or not 1 <= len(name) <= 1_024
        or not isinstance(tokenizer_model, str)
        or not 1 <= len(tokenizer_model) <= 128
        or not isinstance(chat_template, str)
        or not 1 <= len(chat_template) <= 1 * 1024 * 1024
        or not isinstance(file_type, int)
        or isinstance(file_type, bool)
        or not 0 <= file_type <= 255
        or not isinstance(context_length, int)
        or isinstance(context_length, bool)
        or not 128 <= context_length <= 4_194_304
        or not isinstance(alignment, int)
        or isinstance(alignment, bool)
        or alignment < 8
        or alignment > 4_096
        or alignment & (alignment - 1)
        or type(split_count) is not int
        or split_count != count
        or type(metadata.get("split.no", 0)) is not int
        or metadata.get("split.no", 0) != index
        or (
            count > 1
            and (
                type(metadata.get("split.tensors.count")) is not int
                or not 1 <= metadata["split.tensors.count"] <= MAX_TENSORS
                or (
                    inherited is not None
                    and metadata["split.tensors.count"] != inherited.split_tensor_count
                )
            )
        )
        or not 1 <= token_count <= MAX_ARRAY_ITEMS
    ):
        raise GGUFError("GGUF required metadata is invalid or unsupported")

    tensor_names: set[str] = set()
    tensors: list[tuple[int, int]] = []
    tensor_types: set[int] = set()
    for _ in range(tensor_count):
        tensor_name = reader.string(limit=1_024, metadata=False)
        if not tensor_name or tensor_name in tensor_names:
            raise GGUFError("GGUF tensor name is invalid")
        tensor_names.add(tensor_name)
        dimensions = reader.u32()
        if not 1 <= dimensions <= MAX_TENSOR_DIMENSIONS:
            raise GGUFError("GGUF tensor dimensions are invalid")
        element_count = 1
        for _dimension in range(dimensions):
            size = reader.u64()
            if not 1 <= size <= 2**31 or element_count > 2**63 // size:
                raise GGUFError("GGUF tensor shape is invalid")
            element_count *= size
        tensor_type = reader.u32()
        if (
            architecture in {"qwen35", "qwen35moe"}
            and re.fullmatch(r"blk\.\d+\.ssm_conv1d\.weight", tensor_name)
            and tensor_type != 0
        ):
            raise GGUFError("Native Qwen state-space convolution requires F32 weights")
        # Projectors stay unquantized: F32, F16 or BF16 (plain floats, verified on Metal).
        if projector and tensor_type not in {0, 1, 30}:
            raise GGUFError("Projector tensor quantization is not admitted")
        layout = GGML_TYPE_LAYOUTS.get(tensor_type)
        if layout is None:
            raise GGUFError("GGUF tensor type is unsupported")
        tensor_types.add(tensor_type)
        offset = reader.u64()
        if offset % alignment:
            raise GGUFError("GGUF tensor offset is misaligned")
        block_elements, block_bytes = layout
        if element_count % block_elements:
            raise GGUFError("GGUF tensor shape is incompatible with its data type")
        tensors.append((offset, element_count // block_elements * block_bytes))

    tensor_data_offset = (source.tell() + alignment - 1) // alignment * alignment
    data_size = file_size - tensor_data_offset
    ranges = sorted((offset, offset + size) for offset, size in tensors)
    if (
        (tensor_data_offset >= file_size and tensor_count > 0)
        or any(start >= end or end > data_size for start, end in ranges)
        or any(
            current[0] < previous[1] for previous, current in zip(ranges, ranges[1:], strict=False)
        )
    ):
        raise GGUFError("GGUF tensor data offset is invalid")

    quantization = QUANTIZATION_NAMES.get(file_type, f"TYPE_{file_type}")
    return GGUFMetadata(
        version=version,
        architecture=architecture,
        name=name,
        tokenizer_model=tokenizer_model,
        chat_template=chat_template,
        context_length=context_length,
        quantization=quantization,
        file_type=file_type,
        token_count=token_count,
        tensor_count=tensor_count,
        file_size=file_size,
        embedding_length=metadata.get(f"{architecture}.embedding_length", 0),
        projector_type=metadata.get("clip.projector_type", ""),
        projection_length=metadata.get("clip.vision.projection_dim", 0),
        split_count=count,
        split_index=index,
        split_tensor_count=metadata.get("split.tensors.count", tensor_count),
        tensor_names=tuple(sorted(tensor_names)),
        tensor_types=tuple(sorted(tensor_types)),
    )
