"""Bounded file transfer for structured results larger than the worker wire limit.

The engine chooses the destination inside its per-request temporary directory.
The worker returns only a small descriptor; the engine verifies and consumes the
file before that directory is removed. No artifact path is exposed to clients.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat

MAX_BYTES = 12 * 1024 * 1024


def _validate_structure(value):
    stack = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > 32 or nodes > 1_100_000:
            raise ValueError("STRUCTURED_OUTPUT_COMPLEXITY")
        if isinstance(item, dict):
            if len(item) > 256 or any(not isinstance(k, str) or len(k) > 256 for k in item):
                raise ValueError("STRUCTURED_OUTPUT_KEYS")
            stack.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            if len(item) > 4096:
                raise ValueError("STRUCTURED_OUTPUT_ARRAY")
            stack.extend((v, depth + 1) for v in item)
        elif isinstance(item, str):
            if len(item) > 65536:
                raise ValueError("STRUCTURED_OUTPUT_STRING")
        elif type(item) in {float, int}:
            if not math.isfinite(item) or abs(item) > 2**63 - 1:
                raise ValueError("STRUCTURED_OUTPUT_NUMBER")
        elif item is not None and not isinstance(item, bool):
            raise ValueError("STRUCTURED_OUTPUT_VALUE")


def write_result(path, value, backend):
    if not isinstance(value, dict) or backend not in {"cpu", "mps", "cuda"}:
        raise ValueError("STRUCTURED_OUTPUT_VALUE")
    _validate_structure(value)
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if not 1 <= len(raw) <= MAX_BYTES:
        raise ValueError("STRUCTURED_OUTPUT_SIZE")
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(raw)
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise
    return dict(
        relativeOutput=path.name,
        mediaType="application/json",
        backend=backend,
        sizeBytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def read_result(path, descriptor, backend):
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) != {"relativeOutput", "mediaType", "backend", "sizeBytes", "sha256"}
        or descriptor["relativeOutput"] != path.name
        or descriptor["mediaType"] != "application/json"
        or descriptor["backend"] != backend
        or type(descriptor["sizeBytes"]) is not int
        or not 1 <= descriptor["sizeBytes"] <= MAX_BYTES
    ):
        raise ValueError("STRUCTURED_OUTPUT_DESCRIPTOR")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("STRUCTURED_OUTPUT_DUPLICATE")
            result[key] = value
        return result

    def invalid_constant(_):
        raise ValueError("STRUCTURED_OUTPUT_NONFINITE")

    try:
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != descriptor["sizeBytes"]:
                raise ValueError("STRUCTURED_OUTPUT_FILE")
            raw = stream.read(MAX_BYTES + 1)
        if (
            len(raw) != descriptor["sizeBytes"]
            or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]
        ):
            raise ValueError("STRUCTURED_OUTPUT_DIGEST")
        result = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
        if not isinstance(result, dict):
            raise ValueError("STRUCTURED_OUTPUT_VALUE")
        _validate_structure(result)
        return result
    except (OSError, UnicodeError, RecursionError) as exc:
        raise ValueError("STRUCTURED_OUTPUT_FILE") from exc
