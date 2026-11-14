"""Bounded plain dense tensors from PyTorch ZIP archives, without torch.load.

Only the uncompressed state-dictionary container is recognized. Its directory is
decoded by the closed numeric_state_dict grammar; names in that document never
select Python objects, imports, constructors or callable reducers.
"""

from __future__ import annotations

import io
import os
import re
import stat
import sys
import zipfile
from pathlib import Path

from oryn_engine.numeric_state_dict import MAX_BYTES, MAX_TENSORS, parse

MAX_ARCHIVE_BYTES = 256 * 1024**3
MAX_METADATA_BYTES = 16 * 1024**2


class MetadataRanges(io.RawIOBase):
    def __init__(self, read_range, size):
        if type(size) is not int or not 22 <= size <= MAX_ARCHIVE_BYTES:
            raise ValueError("NUMERIC_ARCHIVE_SIZE")
        self._read_range, self.size = read_range, size
        self.position = self.bytes = self.calls = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in {0, 1, 2}:
            raise ValueError("NUMERIC_ARCHIVE_SEEK")
        target = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if not 0 <= target <= self.size:
            raise ValueError("NUMERIC_ARCHIVE_SEEK")
        self.position = target
        return target

    def read(self, length=-1):
        length = self.size - self.position if length < 0 else length
        if length == 0:
            return b""
        if (
            type(length) is not int
            or not 0 < length <= MAX_BYTES
            or self.position + length > self.size
            or self.bytes + length > MAX_METADATA_BYTES
            or self.calls >= 64
        ):
            raise ValueError("NUMERIC_ARCHIVE_METADATA_LIMIT")
        value = self._read_range(self.position, length)
        if not isinstance(value, bytes) or len(value) != length:
            raise ValueError("NUMERIC_ARCHIVE_TRUNCATED")
        self.position += length
        self.bytes += length
        self.calls += 1
        return value


def _directory(archive):
    entries = archive.infolist()
    if archive.comment or not 3 <= len(entries) <= MAX_TENSORS + 6:
        raise ValueError("NUMERIC_ARCHIVE_DIRECTORY")
    names = {x.filename for x in entries}
    if len(names) != len(entries):
        raise ValueError("NUMERIC_ARCHIVE_DUPLICATE_MEMBER")
    roots = {name.split("/")[0] for name in names}
    if len(roots) != 1:
        raise ValueError("NUMERIC_ARCHIVE_DIRECTORY")
    root = roots.pop()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", root):
        raise ValueError("NUMERIC_ARCHIVE_DIRECTORY")
    optional = {"byteorder", ".data/serialization_id", ".format_version", ".storage_alignment"}
    members = {}
    ordered = sorted(entries, key=lambda x: x.header_offset)
    if ordered[0].header_offset != 0:
        raise ValueError("NUMERIC_ARCHIVE_PREFIX")
    for index, entry in enumerate(ordered):
        name = entry.filename[len(root) + 1 :]
        if (
            entry.filename != root + "/" + name
            or entry.orig_filename != entry.filename
            or name not in {"data.pkl", "version"} | optional
            and not re.fullmatch(r"data/[0-9]{1,64}", name)
            or entry.compress_type != zipfile.ZIP_STORED
            or entry.compress_size != entry.file_size
            or entry.flag_bits & ~(0x08 | 0x800)
            or stat.S_IFMT(entry.external_attr >> 16) not in {0, stat.S_IFREG}
        ):
            raise ValueError("NUMERIC_ARCHIVE_MEMBER")
        boundary = (
            ordered[index + 1].header_offset if index + 1 < len(ordered) else archive.start_dir
        )
        if (
            not 0
            <= entry.header_offset
            < entry.header_offset + 30 + len(entry.filename) + entry.file_size
            <= boundary
        ):
            raise ValueError("NUMERIC_ARCHIVE_OVERLAP")
        if name not in {"data.pkl"} and not name.startswith("data/") and entry.file_size > 128:
            raise ValueError("NUMERIC_ARCHIVE_METADATA_LIMIT")
        members[name] = entry
    if (
        not {"data.pkl", "version"} <= members.keys()
        or not 3 <= members["data.pkl"].file_size <= MAX_BYTES
    ):
        raise ValueError("NUMERIC_ARCHIVE_DIRECTORY")
    if archive.read(members["version"]) not in {b"1\n", b"2\n", b"3\n"}:
        raise ValueError("NUMERIC_ARCHIVE_VERSION")
    # Older PyTorch archives omit byteorder and use native host byte order.
    # Qualified engine platforms are little-endian; never reinterpret a marked
    # big-endian archive or silently extend this to an unqualified host.
    if sys.byteorder != "little" or (
        "byteorder" in members and archive.read(members["byteorder"]) != b"little"
    ):
        raise ValueError("NUMERIC_ARCHIVE_BYTE_ORDER")
    for name in optional - {"byteorder"}:
        if name in members and not re.fullmatch(rb"[0-9]{1,64}\n?", archive.read(members[name])):
            raise ValueError("NUMERIC_ARCHIVE_BOOKKEEPING")
    tensors = parse(archive.read(members["data.pkl"]))
    storages = {value.storage.key: value.storage for value in tensors.values()}
    if {"data/" + key for key in storages} != {
        name for name in members if name.startswith("data/")
    }:
        raise ValueError("NUMERIC_ARCHIVE_STORAGE_MEMBERS")
    for key, storage in storages.items():
        if members["data/" + key].file_size != storage.elements * storage.itemsize:
            raise ValueError("NUMERIC_ARCHIVE_STORAGE_SIZE")
    return tensors, members


def inspect_archive(read_range, size):
    """Inspect ZIP and tensor directories only, never numeric storage members."""
    stream = MetadataRanges(read_range, size)
    try:
        if stream.read(4) != b"PK\x03\x04":
            raise ValueError("NUMERIC_ARCHIVE_FORMAT")
        with zipfile.ZipFile(stream) as archive:
            tensors, members = _directory(archive)
        return {
            "format": "torch-numeric-zip-v1",
            "byteOrder": "little",
            "tensors": {
                name: {
                    "dtype": value.storage.dtype,
                    "shape": list(value.shape),
                    "storage": value.storage.key,
                    "offset": value.offset,
                    "stride": list(value.stride),
                }
                for name, value in tensors.items()
            },
            "storages": {
                name.removeprefix("data/"): {"sizeBytes": member.file_size, "crc32": member.CRC}
                for name, member in members.items()
                if name.startswith("data/")
            },
        }
    except (zipfile.BadZipFile, UnicodeError, NotImplementedError, EOFError) as exc:
        raise ValueError("NUMERIC_ARCHIVE_INVALID") from exc


def read_weights(path: Path, *, expected_shapes, maximum_bytes):
    """Load validated numeric bytes only after the complete graph shape check."""
    import torch

    if (
        not isinstance(path, Path)
        or path.is_symlink()
        or not path.is_file()
        or not 22 <= path.stat().st_size <= MAX_ARCHIVE_BYTES
        or type(maximum_bytes) is not int
        or not 0 < maximum_bytes <= MAX_ARCHIVE_BYTES
        or not isinstance(expected_shapes, dict)
        or not 1 <= len(expected_shapes) <= MAX_TENSORS
    ):
        raise ValueError("NUMERIC_ARCHIVE_SCOPE")
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("NUMERIC_ARCHIVE_PLATFORM")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise ValueError("NUMERIC_ARCHIVE_SCOPE") from exc
    with os.fdopen(descriptor, "rb") as source:
        information = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(information.st_mode)
            or not 22 <= information.st_size <= MAX_ARCHIVE_BYTES
        ):
            raise ValueError("NUMERIC_ARCHIVE_SCOPE")
        size = information.st_size

        def read_range(start, length):
            source.seek(start)
            return source.read(length)

        metadata = inspect_archive(read_range, size)
        if {k: v["shape"] for k, v in metadata["tensors"].items()} != expected_shapes:
            raise ValueError("NUMERIC_ARCHIVE_GRAPH_INCOMPLETE")
        if sum(x["sizeBytes"] for x in metadata["storages"].values()) > maximum_bytes:
            raise ValueError("NUMERIC_ARCHIVE_WEIGHT_LIMIT")
        source.seek(0)
        try:
            with zipfile.ZipFile(source) as archive:
                tensors, members = _directory(archive)
                stores = {}
                types = {
                    "F32": torch.float32,
                    "F16": torch.float16,
                    "BF16": torch.bfloat16,
                    "F64": torch.float64,
                    "I64": torch.int64,
                    "I32": torch.int32,
                    "I16": torch.int16,
                    "I8": torch.int8,
                    "U8": torch.uint8,
                    "BOOL": torch.bool,
                }
                for value in tensors.values():
                    key = value.storage.key
                    if key in stores:
                        continue
                    entry = members["data/" + key]
                    raw = bytearray(entry.file_size)
                    with archive.open(entry) as storage:
                        view = memoryview(raw)
                        offset = 0
                        while offset < len(raw):
                            count = storage.readinto(view[offset : offset + 1024**2])
                            if not count:
                                raise ValueError("NUMERIC_ARCHIVE_TRUNCATED")
                            offset += count
                        if storage.read(1):
                            raise ValueError("NUMERIC_ARCHIVE_STORAGE_SIZE")
                    stores[key] = torch.frombuffer(raw, dtype=types[value.storage.dtype])
                    if value.storage.dtype == "BOOL" and bool(
                        (torch.frombuffer(raw, dtype=torch.uint8) > 1).any()
                    ):
                        raise ValueError("NUMERIC_ARCHIVE_BOOLEAN")
                result = {}
                for name, value in tensors.items():
                    count = 1
                    for dimension in value.shape:
                        count *= dimension
                    storage_slice = stores[value.storage.key][value.offset : value.offset + count]
                    tensor = storage_slice.as_strided(value.shape, value.stride)
                    if tensor.is_floating_point():
                        # Validate the dense physical span without materializing
                        # a potentially large transposed tensor.
                        flat = storage_slice
                        for start in range(0, flat.numel(), 1024**2):
                            if not bool(torch.isfinite(flat[start : start + 1024**2]).all()):
                                raise ValueError("NUMERIC_ARCHIVE_NONFINITE")
                    result[name] = tensor
                return result
        except (zipfile.BadZipFile, UnicodeError, NotImplementedError, EOFError) as exc:
            raise ValueError("NUMERIC_ARCHIVE_INVALID") from exc
