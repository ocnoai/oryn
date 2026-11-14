"""Bounded, data-only ONNX discovery without reading embedded tensor payloads.

Field numbers are from ONNX v1.18.0 onnx.proto. The same scanner is used against
authenticated immutable HTTP ranges and the selected local files before loading.
It is not an ONNX evaluator; native operators are admitted by the signed build.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

MAX_GRAPH_BYTES = 4 * 1024**3
MAX_METADATA_BYTES = 16 * 1024**2


def safe_path(name):
    return (
        isinstance(name, str)
        and re.fullmatch(r"[A-Za-z0-9_. /-]{1,512}", name) is not None
        and all(p not in {"", ".", ".."} for p in name.split("/"))
    )


class Reader:
    def __init__(self, read_range, size):
        if type(size) is not int or not 1 <= size <= MAX_GRAPH_BYTES:
            raise ValueError("ONNX_GRAPH_LIMIT")
        self.read_range, self.size = read_range, size
        self.cache, self.bytes, self.fields = {}, 0, 0

    def read(self, start, length):
        if start < 0 or length < 0 or start + length > self.size:
            raise ValueError("ONNX_TRUNCATED")
        if length > MAX_METADATA_BYTES:
            raise ValueError("ONNX_METADATA_LIMIT")
        result = bytearray()
        while length:
            block = start // 32768 * 32768
            if block not in self.cache:
                count = min(32768, self.size - block)
                self.bytes += count
                if self.bytes > MAX_METADATA_BYTES:
                    raise ValueError("ONNX_METADATA_LIMIT")
                data = self.read_range(block, count)
                if len(data) != count:
                    raise ValueError("ONNX_TRUNCATED")
                self.cache[block] = data
            part = self.cache[block][start - block : start - block + length]
            result.extend(part)
            start, length = start + len(part), length - len(part)
        return bytes(result)

    def integer(self, offset, end):
        value = 0
        for index in range(10):
            if offset >= end:
                raise ValueError("ONNX_TRUNCATED")
            byte = self.read(offset, 1)[0]
            offset += 1
            if index == 9 and byte > 1:
                raise ValueError("ONNX_INTEGER_LIMIT")
            value |= (byte & 127) << (7 * index)
            if byte < 128:
                return value, offset
        raise ValueError("ONNX_INTEGER_LIMIT")

    def message(self, start, end):
        result = []
        while start < end:
            self.fields += 1
            if self.fields > 500000:
                raise ValueError("ONNX_FIELD_LIMIT")
            tag, start = self.integer(start, end)
            field, wire = tag >> 3, tag & 7
            if not 1 <= field < 2**29:
                raise ValueError("ONNX_FIELD_INVALID")
            if wire == 0:
                value, start = self.integer(start, end)
            elif wire in {1, 2, 5}:
                length, start = (
                    self.integer(start, end) if wire == 2 else (8 if wire == 1 else 4, start)
                )
                if length > end - start:
                    raise ValueError("ONNX_TRUNCATED")
                value, start = (start, start + length), start + length
            else:
                raise ValueError("ONNX_FIELD_INVALID")
            result.append((field, wire, value))
        return result

    def text(self, field):
        if field[1] != 2 or field[2][1] - field[2][0] > 4096:
            raise ValueError("ONNX_STRING_LIMIT")
        return self.read(field[2][0], field[2][1] - field[2][0]).decode("utf-8", "strict")

    def children(self, field):
        if field[1] != 2:
            raise ValueError("ONNX_FIELD_INVALID")
        return self.message(*field[2])


def _one(fields, number, required=False):
    found = [f for f in fields if f[0] == number]
    if len(found) > 1 or (required and not found):
        raise ValueError("ONNX_AMBIGUOUS_FIELD")
    return found[0] if found else None


def scan_graph(read_range, size, filename, sizes):
    if not safe_path(filename) or not filename.endswith(".onnx"):
        raise ValueError("ONNX_PATH_INVALID")
    reader = Reader(read_range, size)
    external, operators, inputs, outputs = set(), set(), [], []
    initializers = {}
    nodes = 0

    def tensor(entries):
        location = _one(entries, 14)
        refs = [f for f in entries if f[0] == 13]
        if location and (location[1] != 0 or location[2] not in {0, 1}):
            raise ValueError("ONNX_EXTERNAL_INVALID")
        if not refs and (location is None or location[2] == 0):
            return
        if (
            not location
            or location[2] != 1
            or any(f[0] in {4, 5, 6, 7, 9, 10, 11} for f in entries)
        ):
            raise ValueError("ONNX_EXTERNAL_INVALID")
        metadata = {}
        for ref in refs:
            fields = reader.children(ref)
            if len(fields) != 2:
                raise ValueError("ONNX_EXTERNAL_INVALID")
            key = reader.text(_one(fields, 1, True))
            if key in metadata or key not in {"location", "offset", "length", "checksum"}:
                raise ValueError("ONNX_EXTERNAL_INVALID")
            metadata[key] = reader.text(_one(fields, 2, True))
        name = metadata.get("location")
        if not safe_path(name):
            raise ValueError("ONNX_EXTERNAL_PATH")
        path = str(PurePosixPath(filename).parent / name)
        if not path.endswith((".data", ".bin", ".onnx_data")):
            raise ValueError("ONNX_EXTERNAL_PATH")
        length, offset = metadata.get("length"), metadata.get("offset", "0")
        if not re.fullmatch(r"[0-9]{1,20}", offset) or (
            length is not None and not re.fullmatch(r"[0-9]{1,20}", length)
        ):
            raise ValueError("ONNX_EXTERNAL_RANGE")
        total = sizes.get(path, 0)
        offset = int(offset)
        length = int(length) if length is not None else total - offset
        if not 0 < length <= total or offset + length > total:
            raise ValueError("ONNX_EXTERNAL_RANGE")
        external.add(path)
        if len(external) > 256:
            raise ValueError("ONNX_EXTERNAL_LIMIT")

    def value_info(field):
        fields = reader.children(field)
        name = reader.text(_one(fields, 1, True))
        type_fields = reader.children(_one(fields, 2, True))
        if len(type_fields) != 1 or type_fields[0][0] != 1:
            raise ValueError("ONNX_TENSOR_IO_REQUIRED")
        info = reader.children(type_fields[0])
        dtype = _one(info, 1, True)
        if dtype[1] != 0 or dtype[2] not in {1, 6, 7, 9, 10, 11}:
            raise ValueError("ONNX_TENSOR_TYPE_UNSUPPORTED")
        shape = []
        for dimension in reader.children(_one(info, 2, True)):
            if dimension[0] != 1:
                raise ValueError("ONNX_SHAPE_INVALID")
            dims = reader.children(dimension)
            if not dims:
                shape.append(None)
                continue
            if len(dims) != 1 or dims[0][0] not in {1, 2}:
                raise ValueError("ONNX_SHAPE_INVALID")
            value = dims[0][2] if dims[0][1] == 0 else reader.text(dims[0])
            if isinstance(value, int) and not 1 <= value <= 1048576:
                raise ValueError("ONNX_SHAPE_INVALID")
            shape.append(value)
        if len(shape) > 8:
            raise ValueError("ONNX_SHAPE_INVALID")
        return {"name": name, "dtype": dtype[2], "shape": shape}

    def visit(entries, kind, depth=0):
        nonlocal nodes
        if depth > 24:
            raise ValueError("ONNX_GRAPH_DEPTH")
        if kind == "tensor":
            tensor(entries)
            return
        if kind == "node":
            nodes += 1
            if nodes > 20000:
                raise ValueError("ONNX_NODE_LIMIT")
            op = reader.text(_one(entries, 4, True))
            domain = _one(entries, 7)
            domain = reader.text(domain) if domain else ""
            if "::" in op:
                raise ValueError("ONNX_OPERATOR_UNSUPPORTED")
            if domain == "com.microsoft":
                from oryn_engine.onnx_nbits import OPERATOR, validate_node

                if op != "MatMulNBits" or depth != 1 or opsets.get(domain) != 1:
                    raise ValueError("ONNX_OPERATOR_UNSUPPORTED")
                validate_node(reader, entries, initializers)
                op, domain = OPERATOR, ""
            if (
                domain not in {"", "ai.onnx"}
                or op
                in {
                    "Loop",
                    "Scan",
                    "RandomNormal",
                    "RandomUniform",
                    "RandomNormalLike",
                    "RandomUniformLike",
                    "Multinomial",
                }
                or _one(entries, 8)
            ):
                raise ValueError("ONNX_OPERATOR_UNSUPPORTED")
            operators.add(op)
        edges = {
            "graph": {1: "node", 5: "tensor", 15: "sparse"},
            "node": {5: "attribute"},
            "attribute": {
                5: "tensor",
                6: "graph",
                10: "tensor",
                11: "graph",
                22: "sparse",
                23: "sparse",
            },
            "sparse": {1: "tensor", 2: "tensor"},
        }
        for field in entries:
            child = edges.get(kind, {}).get(field[0])
            if child:
                visit(reader.children(field), child, depth + 1)
            elif kind == "graph" and depth == 0 and field[0] in {11, 12}:
                (inputs if field[0] == 11 else outputs).append(value_info(field))

    model = reader.message(0, size)
    version = _one(model, 1, True)
    if version[1] != 0 or not 3 <= version[2] <= 10 or any(f[0] in {20, 25, 26} for f in model):
        raise ValueError("ONNX_MODEL_UNSUPPORTED")
    opsets = {}
    for field in [f for f in model if f[0] == 8]:
        parts = reader.children(field)
        domain = _one(parts, 1)
        domain = reader.text(domain) if domain else ""
        version = _one(parts, 2, True)
        if (
            domain not in {"", "ai.onnx", "com.microsoft"}
            or domain in opsets
            or version[1] != 0
            or not (version[2] == 1 if domain == "com.microsoft" else 7 <= version[2] <= 22)
        ):
            raise ValueError("ONNX_OPSET_UNSUPPORTED")
        opsets[domain] = version[2]
    standard = set(opsets) & {"", "ai.onnx"}
    if len(standard) != 1:
        raise ValueError("ONNX_OPSET_UNSUPPORTED")
    graph = reader.children(_one(model, 7, True))
    if "com.microsoft" in opsets:
        for field in graph:
            if field[0] == 5:
                parts = reader.children(field)
                name = reader.text(_one(parts, 8, True))
                if name in initializers:
                    raise ValueError("ONNX_AMBIGUOUS_INITIALIZER")
                initializers[name] = parts
    visit(graph, "graph")
    # Original Whisper exports expose per-layer key/value and attention tensors
    # alongside logits (up to 193 outputs for 32 layers). Task bindings still
    # validate every output name/shape; this only bounds structural discovery.
    if not nodes or not 1 <= len(inputs) <= 8 or not 1 <= len(outputs) <= 256:
        raise ValueError("ONNX_GRAPH_INVALID")
    if any(len({v["name"] for v in values}) != len(values) for values in (inputs, outputs)):
        raise ValueError("ONNX_IO_AMBIGUOUS")
    return {
        "externalFiles": sorted(external),
        "operators": sorted(operators),
        "opset": opsets[next(iter(standard))],
        **({"operatorDomains": {"com.microsoft": 1}} if "com.microsoft" in opsets else {}),
        "inputs": inputs,
        "outputs": outputs,
    }
