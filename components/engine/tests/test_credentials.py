from __future__ import annotations

import json
import os
from pathlib import Path

from oryn_engine.credentials import HubCredentialVault, SupervisorCredentialChannel


def test_supervisor_credential_channel_updates_memory_and_acknowledges() -> None:
    input_read, input_write = os.pipe()
    acknowledgment_read, acknowledgment_write = os.pipe()
    vault = HubCredentialVault()
    channel = SupervisorCredentialChannel(
        vault,
        os.fdopen(input_read, "rb", buffering=0),
        os.fdopen(acknowledgment_write, "wb", buffering=0),
    )
    thread = channel.start()
    token = f"hf_{'a' * 32}"

    with os.fdopen(input_write, "wb", buffering=0) as writer:
        writer.write(json.dumps({"revision": 1, "token": token}).encode() + b"\n")
        with os.fdopen(acknowledgment_read, "rb", buffering=0) as reader:
            assert reader.readline() == b"1\n"
            assert vault.get() == token
            writer.write(json.dumps({"revision": 2, "token": None}).encode() + b"\n")
            assert reader.readline() == b"2\n"
            assert vault.get() is None

    thread.join(timeout=1)
    assert not thread.is_alive()


def test_supervisor_credential_channel_requires_both_private_descriptors(
    monkeypatch, tmp_path: Path
) -> None:
    del tmp_path
    vault = HubCredentialVault()
    monkeypatch.setenv("ORYN_HF_CREDENTIAL_INPUT_FD", "3")
    monkeypatch.delenv("ORYN_HF_CREDENTIAL_ACK_FD", raising=False)

    try:
        SupervisorCredentialChannel.from_environment(vault)
    except RuntimeError as error:
        assert "Incomplete" in str(error)
    else:
        raise AssertionError("incomplete credential channels must fail closed")
