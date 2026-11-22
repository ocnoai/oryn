"""Bounded numeric CTranslate2 directories; no pickle or repository loaders."""

from __future__ import annotations

import math
import re
import struct
import time

DTYPE_BYTES = {0: 4, 1: 1, 2: 2, 3: 4, 4: 2, 5: 2}
NAME = re.compile(r"[a-z][a-z0-9_/]{0,255}")


def scan_directory(read_range, size):
    if type(size) is not int or not 64 <= size <= 16 * 1024**3:
        raise ValueError("CT2_FILE_BOUND")
    position, buffer_start, transferred, consumed = 0, 0, 0, 0
    buffer = b""
    deadline = time.monotonic() + 300

    def read(length):
        nonlocal position, buffer_start, buffer, transferred, consumed
        if not 0 < length <= 512 or position + length > size or time.monotonic() > deadline:
            raise ValueError("CT2_DIRECTORY_BOUND")
        if not buffer_start <= position <= buffer_start + len(buffer) - length:
            count = min(max(length, 512), size - position)
            transferred += count
            if transferred > 2 * 1024**2:
                raise ValueError("CT2_DIRECTORY_BOUND")
            buffer, buffer_start = read_range(position, count), position
            if not isinstance(buffer, bytes) or len(buffer) != count:
                raise ValueError("CT2_RANGE_INVALID")
        value = buffer[position - buffer_start : position - buffer_start + length]
        position += length
        consumed += length
        return value

    def number(fmt):
        return struct.unpack("<" + fmt, read(struct.calcsize(fmt)))[0]

    def string():
        count = number("H")
        if not 2 <= count <= 256:
            raise ValueError("CT2_STRING_BOUND")
        raw = read(count)
        if raw[-1] != 0 or b"\0" in raw[:-1]:
            raise ValueError("CT2_STRING_INVALID")
        return raw[:-1].decode("ascii")

    version = number("I")
    if version != 6:
        raise ValueError("CT2_BINARY_VERSION")
    spec, revision, count = string(), number("I"), number("I")
    if spec != "WhisperSpec" or revision != 3 or not 1 <= count <= 4096:
        raise ValueError("CT2_SPECIFICATION")
    tensors = {}
    for _ in range(count):
        name, rank = string(), number("B")
        if not NAME.fullmatch(name) or name in tensors or rank > 4:
            raise ValueError("CT2_VARIABLE_INVALID")
        shape = [number("I") for _ in range(rank)]
        dtype, byte_count = number("B"), number("I")
        if (
            dtype not in DTYPE_BYTES
            or any(not 0 < dim <= 262144 for dim in shape)
            or byte_count != math.prod(shape) * DTYPE_BYTES[dtype]
            or position + byte_count > size
        ):
            raise ValueError("CT2_VARIABLE_BOUND")
        tensor = {"shape": shape, "dtype": dtype, "dataOffset": position, "bytes": byte_count}
        if not shape and dtype in {1, 2, 3}:
            tensor["controlValue"] = number({1: "b", 2: "h", 3: "i"}[dtype])
        else:
            position += byte_count
        tensors[name] = tensor
    aliases = {}
    count = number("I")
    if count > 4096:
        raise ValueError("CT2_ALIAS_BOUND")
    for _ in range(count):
        name, target = string(), string()
        if not NAME.fullmatch(name) or name in tensors or name in aliases or target not in tensors:
            raise ValueError("CT2_ALIAS_INVALID")
        aliases[name] = target
    if position != size:
        raise ValueError("CT2_TRAILING_BYTES")
    return {
        "binaryVersion": version,
        "spec": spec,
        "specRevision": revision,
        "fileBytes": size,
        "tensors": tensors,
        "aliases": aliases,
        "directoryBytesConsumed": consumed,
        "transferredBytes": transferred,
    }


def validate_whisper(directory):
    if (
        directory.get("binaryVersion") != 6
        or directory.get("spec") != "WhisperSpec"
        or directory.get("specRevision") != 3
    ):
        raise ValueError("CT2_SPECIFICATION")
    tensors, aliases = directory["tensors"], directory["aliases"]
    all_tensors = tensors | {name: tensors[target] for name, target in aliases.items()}
    try:
        vocab, hidden = all_tensors["decoder/embeddings/weight"]["shape"]
        conv_hidden, mels, kernel = all_tensors["encoder/conv1/weight"]["shape"]
        if (
            vocab not in {51865, 51866}
            or not 16 <= hidden <= 1536
            or hidden % 2
            or conv_hidden != hidden
            or mels not in {80, 128}
            or kernel != 3
        ):
            raise ValueError("CT2_GEOMETRY")
        counts, heads = {}, {}
        for side in ("encoder", "decoder"):
            indices = {
                int(match[1])
                for name in all_tensors
                if (match := re.fullmatch(side + r"/layer_(\d+)/.+", name))
            }
            if not 1 <= len(indices) <= 32 or indices != set(range(len(indices))):
                raise ValueError("CT2_LAYERS")
            counts[side] = len(indices)
            heads[side] = tensors[side + "/num_heads"]["controlValue"]
            if type(heads[side]) is not int or not 1 <= heads[side] <= 32 or hidden % heads[side]:
                raise ValueError("CT2_HEADS")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("CT2_GRAPH_INCOMPLETE") from exc
    expected = {
        "decoder/embeddings/weight": [vocab, hidden],
        "decoder/projection/weight": [vocab, hidden],
        "encoder/conv1/weight": [hidden, mels, 3],
        "encoder/conv1/bias": [hidden],
        "encoder/conv2/weight": [hidden, hidden, 3],
        "encoder/conv2/bias": [hidden],
        "encoder/position_encodings/encodings": [1500, hidden],
        "decoder/position_encodings/encodings": [448, hidden],
    }
    for side in ("encoder", "decoder"):
        for value in ("beta", "gamma"):
            expected[f"{side}/layer_norm/{value}"] = [hidden]
        for index in range(counts[side]):
            base = f"{side}/layer_{index}/"
            parts = {
                "self_attention": [(3 * hidden, hidden), (hidden, hidden)],
                "ffn": [(4 * hidden, hidden), (hidden, 4 * hidden)],
            }
            if side == "decoder":
                parts["attention"] = [(hidden, hidden), (2 * hidden, hidden), (hidden, hidden)]
            for part, linears in parts.items():
                for value in ("beta", "gamma"):
                    expected[base + part + "/layer_norm/" + value] = [hidden]
                for layer, (out, ins) in enumerate(linears):
                    expected[f"{base}{part}/linear_{layer}/weight"] = [out, ins]
                    expected[f"{base}{part}/linear_{layer}/bias"] = [out]
    controls = {
        "decoder/activation": (1, 3),
        "decoder/alibi": (1, 0),
        "decoder/alignment_heads": (2, 1),
        "decoder/alignment_layer": (2, -1),
        "decoder/num_heads": (2, heads["decoder"]),
        "decoder/pre_norm": (1, 1),
        "decoder/scale_embeddings": (1, 0),
        "decoder/start_from_zero_embedding": (1, 0),
        "encoder/num_heads": (2, heads["encoder"]),
    }
    for name in ("decoder/alibi_use_positive_positions", "decoder/scale_alibi"):
        if name in all_tensors:
            controls[name] = (1, 0)
    if set(all_tensors) != expected.keys() | controls.keys():
        raise ValueError("CT2_GRAPH_INCOMPLETE")
    for name, shape in expected.items():
        tensor = all_tensors[name]
        if tensor["shape"] != shape or tensor["dtype"] not in {0, 4}:
            raise ValueError("CT2_WEIGHT_LAYOUT")
    for name, (dtype, value) in controls.items():
        tensor = all_tensors[name]
        if (
            name in aliases
            or tensor["shape"]
            or tensor["dtype"] != dtype
            or tensor.get("controlValue") != value
        ):
            raise ValueError("CT2_GRAPH_CONTROL")
    return {
        "vocab_size": vocab,
        "hidden_size": hidden,
        "num_mel_bins": mels,
        "encoder_layers": counts["encoder"],
        "decoder_layers": counts["decoder"],
        "encoder_heads": heads["encoder"],
        "decoder_heads": heads["decoder"],
    }


def validate_numeric(path, directory):
    import numpy as np

    for tensor in directory["tensors"].values():
        if tensor["dtype"] not in {0, 4}:
            continue
        dtype = "<f4" if tensor["dtype"] == 0 else "<f2"
        with path.open("rb") as stream:
            stream.seek(tensor["dataOffset"])
            remaining = tensor["bytes"]
            while remaining:
                raw = stream.read(min(remaining, 4 * 1024**2))
                if not raw or not bool(np.isfinite(np.frombuffer(raw, dtype=dtype)).all()):
                    raise ValueError("CT2_NONFINITE_WEIGHTS")
                remaining -= len(raw)


def inspect_file(path, *, numeric=False):
    if path.is_symlink() or not path.is_file():
        raise ValueError("CT2_FILE_INVALID")
    with path.open("rb") as stream:

        def read(start, length):
            stream.seek(start)
            return stream.read(length)

        directory = scan_directory(read, path.stat().st_size)
    geometry = validate_whisper(directory)
    if numeric:
        validate_numeric(path, directory)
    return directory, geometry
