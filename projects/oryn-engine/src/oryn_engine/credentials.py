from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from typing import BinaryIO

TOKEN_PATTERN = re.compile(r"^hf_[A-Za-z0-9]{20,512}$")
MAX_CREDENTIAL_MESSAGE_BYTES = 1024


class HubCredentialVault:
    """Memory-only credential state populated by the supervising Browser."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._token: str | None = None

    def set(self, token: str | None) -> None:
        if token is not None and not TOKEN_PATTERN.fullmatch(token):
            raise ValueError("Invalid Hugging Face credential")
        with self._lock:
            self._token = token

    def get(self) -> str | None:
        with self._lock:
            return self._token


@dataclass
class SupervisorCredentialChannel:
    vault: HubCredentialVault
    input_stream: BinaryIO
    acknowledgment_stream: BinaryIO

    @classmethod
    def from_environment(cls, vault: HubCredentialVault) -> SupervisorCredentialChannel | None:
        input_value = os.environ.pop("ORYN_HF_CREDENTIAL_INPUT_FD", None)
        acknowledgment_value = os.environ.pop("ORYN_HF_CREDENTIAL_ACK_FD", None)
        if input_value is None and acknowledgment_value is None:
            return None
        if not input_value or not acknowledgment_value:
            raise RuntimeError("Incomplete supervised credential channel")
        input_fd = int(input_value)
        acknowledgment_fd = int(acknowledgment_value)
        if input_fd < 3 or acknowledgment_fd < 3 or input_fd == acknowledgment_fd:
            raise RuntimeError("Invalid supervised credential channel")
        return cls(
            vault,
            os.fdopen(input_fd, "rb", buffering=0),
            os.fdopen(acknowledgment_fd, "wb", buffering=0),
        )

    def start(self) -> threading.Thread:
        thread = threading.Thread(
            target=self._read,
            name="oryn-credential-channel",
            daemon=True,
        )
        thread.start()
        return thread

    def _read(self) -> None:
        while True:
            raw = self.input_stream.readline(MAX_CREDENTIAL_MESSAGE_BYTES + 2)
            if not raw:
                return
            if len(raw) > MAX_CREDENTIAL_MESSAGE_BYTES + 1 or not raw.endswith(b"\n"):
                return
            try:
                message = json.loads(raw.decode("utf-8"))
                revision = message["revision"]
                token = message["token"]
                if (
                    not isinstance(message, dict)
                    or set(message) != {"revision", "token"}
                    or not isinstance(revision, int)
                    or isinstance(revision, bool)
                    or revision < 1
                    or revision > 2**31 - 1
                    or (token is not None and not isinstance(token, str))
                ):
                    return
                self.vault.set(token)
                self.acknowledgment_stream.write(f"{revision}\n".encode())
                self.acknowledgment_stream.flush()
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError, OSError):
                return
