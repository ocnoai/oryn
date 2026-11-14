"""Closed native ONNX Runtime symmetric block-Q4 matrix metadata."""

from __future__ import annotations

OPERATOR = "com.microsoft::MatMulNBits"


def validate_node(reader, entries, tensors):
    from oryn_engine.onnx_graph import _one

    names = [reader.text(f) for f in entries if f[0] == 1]
    outputs = [reader.text(f) for f in entries if f[0] == 2]
    if len(names) != 3 or any(not name for name in names) or len(outputs) != 1 or not outputs[0]:
        raise ValueError("ONNX_NBITS_INPUTS_UNSUPPORTED")
    attributes = {}
    for field in entries:
        if field[0] != 5:
            continue
        parts = reader.children(field)
        name = reader.text(_one(parts, 1, True))
        value, kind = _one(parts, 3, True), _one(parts, 20, True)
        if (
            name in attributes
            or name not in {"K", "N", "bits", "block_size", "accuracy_level"}
            or {p[0] for p in parts} != {1, 3, 20}
            or value[1] != 0
            or kind[1:] != (0, 2)
        ):
            raise ValueError("ONNX_NBITS_ATTRIBUTES_UNSUPPORTED")
        attributes[name] = value[2]
    k, n = attributes.get("K"), attributes.get("N")
    block = attributes.get("block_size", 128)
    if (
        type(k) is not int
        or type(n) is not int
        or not 1 <= k <= 65536
        or not 1 <= n <= 65536
        or k * n > 2**29
        or attributes.get("bits", 4) != 4
        or block not in {16, 32, 64, 128, 256}
        or attributes.get("accuracy_level", 0) not in {0, 1}
    ):
        raise ValueError("ONNX_NBITS_GEOMETRY_UNSUPPORTED")
    blocks = (k + block - 1) // block
    required = ((names[1], 2, [n, blocks, block // 2]), (names[2], 1, [n, blocks]))
    for name, dtype, shape in required:
        parts = tensors.get(name)
        if parts is None:
            raise ValueError("ONNX_NBITS_CONSTANT_REQUIRED")
        actual_dtype = _one(parts, 2, True)
        dimensions = []
        for field in parts:
            if field[0] != 1:
                continue
            if field[1] == 0:
                dimensions.append(field[2])
            elif field[1] == 2:
                start, end = field[2]
                while start < end:
                    value, start = reader.integer(start, end)
                    dimensions.append(value)
                    if len(dimensions) > 8:
                        raise ValueError("ONNX_NBITS_TENSOR_INVALID")
            else:
                raise ValueError("ONNX_NBITS_TENSOR_INVALID")
        admitted_shapes = [shape, [n * blocks]] if name == names[2] else [shape]
        if actual_dtype[1:] != (0, dtype) or dimensions not in admitted_shapes:
            raise ValueError("ONNX_NBITS_TENSOR_INVALID")
