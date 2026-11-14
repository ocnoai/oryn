from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

from oryn_engine.app import run
from oryn_engine.protocol.dispatcher import ProtocolDispatcher

ENGINE_ROOT = Path(__file__).resolve().parents[1]


def _run_engine(messages: list[dict[str, object]]) -> subprocess.CompletedProcess[str]:
    payload = "".join(json.dumps(message, separators=(",", ":")) + "\n" for message in messages)
    environment = os.environ.copy()
    source_root = str(ENGINE_ROOT / "src")
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{existing_python_path}" if existing_python_path else source_root
    )
    return subprocess.run(
        [sys.executable, "-m", "oryn_engine"],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )


def test_handshake_ping_and_capabilities_round_trip() -> None:
    result = _run_engine(
        [
            {
                "type": "request",
                "id": "req_handshake_test",
                "method": "protocol.handshake",
                "params": {
                    "supportedVersions": [1],
                    "browserVersion": "0.1.0",
                    "maxMessageBytes": 16 * 1024 * 1024,
                },
            },
            {
                "v": 1,
                "type": "request",
                "id": "req_ping_test",
                "method": "protocol.ping",
                "params": {},
            },
            {
                "v": 1,
                "type": "request",
                "id": "req_caps_test",
                "method": "runtime.capabilities",
                "params": {},
            },
        ]
    )

    assert result.returncode == 0, result.stderr
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    responses = [message for message in messages if message["type"] == "response"]
    assert len(responses) == 3
    assert responses[0]["ok"] is True
    assert "v" not in responses[0]
    assert responses[1]["result"]["state"] == "ready"
    assert responses[2]["result"]["scheduler"]["lanes"] == [
        {"id": "generation", "capacity": 1, "global": True}
    ]


def test_request_before_handshake_is_rejected_without_exiting() -> None:
    result = _run_engine(
        [
            {
                "v": 1,
                "type": "request",
                "id": "req_early",
                "method": "protocol.ping",
                "params": {},
            }
        ]
    )

    assert result.returncode == 0
    response = json.loads(result.stdout)
    assert response["error"]["code"] == "PROTOCOL_HANDSHAKE_REQUIRED"


def test_invalid_json_is_fatal_and_never_contaminates_stdout() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ENGINE_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-m", "oryn_engine"],
        input="not-json\n",
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "Fatal protocol input error" in result.stderr


def test_oversized_line_is_fatal_without_stdout_contamination() -> None:
    output = BytesIO()
    result = run(BytesIO(b"x" * (16 * 1024 * 1024 + 1) + b"\n"), output)

    assert result == 2
    assert output.getvalue() == b""


def test_invalid_utf8_truncated_json_and_structural_bombs_are_fatal() -> None:
    payloads = [
        b"\xc3(\n",
        b'{"truncated":\n',
        ("[" * 66 + "0" + "]" * 66 + "\n").encode(),
        (json.dumps([0] * 50_001) + "\n").encode(),
    ]
    for payload in payloads:
        output = BytesIO()
        assert run(BytesIO(payload), output) == 2
        assert output.getvalue() == b""


def test_control_plane_ping_can_overtake_a_slow_valid_request() -> None:
    class SlowCapabilitiesDispatcher(ProtocolDispatcher):
        def _dispatch_method(self, method, message, request_id):  # type: ignore[no-untyped-def]
            if method == "runtime.capabilities":
                time.sleep(0.1)
            return super()._dispatch_method(method, message, request_id)

    input_stream = BytesIO(
        b"\n".join(
            json.dumps(message, separators=(",", ":")).encode()
            for message in [
                {
                    "type": "request",
                    "id": "req_handshake_concurrent",
                    "method": "protocol.handshake",
                    "params": {
                        "supportedVersions": [1],
                        "browserVersion": "0.1.0",
                        "maxMessageBytes": 16 * 1024 * 1024,
                    },
                },
                {
                    "v": 1,
                    "type": "request",
                    "id": "req_slow_caps",
                    "method": "runtime.capabilities",
                    "params": {},
                },
                {
                    "v": 1,
                    "type": "request",
                    "id": "req_fast_ping",
                    "method": "protocol.ping",
                    "params": {},
                },
            ]
        )
        + b"\n"
    )
    output = BytesIO()

    assert run(input_stream, output, SlowCapabilitiesDispatcher()) == 0
    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    response_ids = [message["id"] for message in messages if message["type"] == "response"]
    assert response_ids == ["req_handshake_concurrent", "req_fast_ping", "req_slow_caps"]
    assert all(isinstance(message, dict) for message in messages)


def test_graceful_shutdown_acknowledges_then_emits_runtime_state() -> None:
    result = _run_engine(
        [
            {
                "type": "request",
                "id": "req_handshake_shutdown",
                "method": "protocol.handshake",
                "params": {
                    "supportedVersions": [1],
                    "browserVersion": "0.1.0",
                    "maxMessageBytes": 16 * 1024 * 1024,
                },
            },
            {
                "v": 1,
                "type": "request",
                "id": "req_shutdown",
                "method": "runtime.shutdown",
                "params": {"reason": "browser-exit"},
            },
        ]
    )

    assert result.returncode == 0, result.stderr
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    shutdown = next(message for message in messages if message.get("id") == "req_shutdown")
    state = next(message for message in messages if message.get("event") == "runtime.state")
    assert shutdown["result"] == {"accepted": True}
    assert state["data"] == {"state": "shutting-down"}
    assert state["seq"] > 0
