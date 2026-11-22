from __future__ import annotations

import json
import threading
from typing import Any, BinaryIO


class ProtocolWriter:
    """The only writer allowed to emit engine stdout protocol bytes."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def write(self, message: dict[str, Any]) -> None:
        payload = (
            json.dumps(
                message,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        with self._lock:
            self._stream.write(payload)
            self._stream.flush()
