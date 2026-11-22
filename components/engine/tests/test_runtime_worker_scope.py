"""Pure broker rejection/canonicalization tests; no worker or Seatbelt execution."""

import json
import sys
import threading
from pathlib import Path

import pytest

from oryn_engine.jobs import JobExecutionError
from oryn_engine.runtime_packs import execution
from oryn_engine.runtime_packs.broker import WorkerBroker


@pytest.mark.parametrize(
    "marked,contract,data,accepted",
    [
        (True, "chat-v2", {"reasoningPrefix": True}, True),
        (False, "chat-v2", {"reasoningPrefix": True}, False),
        (True, None, {"reasoningPrefix": True}, False),
        (True, "chat-v2", {"reasoningPrefix": "true"}, False),
        (True, "chat-v2", {"reasoningPrefix": True, "other": True}, False),
    ],
)
def test_private_chat_framing_crosses_real_worker_abi_only_when_admitted(
    tmp_path, marked, contract, data, accepted
):
    from types import SimpleNamespace

    (tmp_path / "framing_probe.py").write_text(
        "import json, sys\njson.loads(sys.stdin.readline())\n"
        + "print("
        + repr(json.dumps({"type": "event", "event": "chat.framing", "data": data}))
        + ", flush=True)\n"
        + "print("
        + repr(json.dumps({"type": "result", "ok": True, "result": {"text": "fixture"}}))
        + ", flush=True)\n"
    )
    pack = SimpleNamespace(
        abi="1",
        pack_id="oryn.llama-cpp",
        version="0.5.0",
        provider=None,
        entrypoint="framing_probe",
    )
    events = []

    def invoke():
        return execution.StreamingPackProcess().invoke(
            pack,
            execution.PackRuntime(
                Path(sys.executable), tmp_path, inherit_base_runtime=True, chat_v2=marked
            ),
            "generate",
            {"taskContract": contract} if contract else {},
            cancellation=threading.Event(),
            event_sink=lambda name, value: events.append((name, value)),
            timeout_seconds=10,
        )

    if accepted:
        assert invoke() == {"text": "fixture"}
        assert events == [("chat.framing", data)]
    else:
        with pytest.raises(JobExecutionError) as error:
            invoke()
        assert error.value.code == "PACK_PROTOCOL_INVALID"
        assert events == []


@pytest.mark.parametrize(
    "marked,data,accepted",
    [
        (True, {"reasoningPrefix": False}, True),
        (False, {"reasoningPrefix": False}, False),
        (True, {"reasoningPrefix": 1}, False),
    ],
)
def test_broker_client_retains_the_same_private_framing_gate(monkeypatch, marked, data, accepted):
    from types import SimpleNamespace

    from oryn_engine.runtime_packs import broker

    messages = (
        json.dumps({"type": "event", "event": "chat.framing", "data": data})
        + "\n"
        + json.dumps({"type": "result", "ok": True, "result": {"text": "fixture"}})
        + "\n"
    ).encode()

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def settimeout(self, *_):
            pass

        def connect(self, *_):
            pass

        def sendall(self, *_):
            pass

        def recv(self, *_):
            return messages

    connection = Connection()
    monkeypatch.setenv("ORYN_WORKER_BROKER_TOKEN", "synthetic-launch-token")
    monkeypatch.setenv("ORYN_WORKER_BROKER_SOCKET", "/private/tmp/synthetic-no-socket")
    monkeypatch.setattr(broker.socket, "socket", lambda *_: connection)
    monkeypatch.setattr(broker.select, "select", lambda *_: ([connection], [], []))
    seen = []

    def invoke():
        return broker.invoke_broker(
            SimpleNamespace(pack_id="oryn.llama-cpp", version="0.5.0"),
            "generate",
            {"taskContract": "chat-v2"},
            chat_v2=marked,
            cancellation=threading.Event(),
            event_sink=lambda name, value: seen.append((name, value)),
            timeout_seconds=2,
        )

    if accepted:
        assert invoke() == {"text": "fixture"}
        assert seen == [("chat.framing", data)]
    else:
        with pytest.raises(JobExecutionError) as error:
            invoke()
        assert error.value.code == "PACK_PROTOCOL_INVALID"
        assert seen == []


def test_worker_prefers_pack_dependencies_and_retains_base_runtime(tmp_path):
    from types import SimpleNamespace

    (tmp_path / "packaging.py").write_text("__version__ = 'pack-specific'\n")
    (tmp_path / "dependency_probe.py").write_text(
        "import json, packaging, oryn_engine\n"
        "print(json.dumps({'type': 'result', 'ok': True, "
        "'result': {'version': packaging.__version__, "
        "'baseAvailable': oryn_engine.__name__ == 'oryn_engine'}}))\n"
    )
    pack = SimpleNamespace(
        abi="1",
        pack_id="test.dependencies",
        version="1.0.0",
        provider=None,
        entrypoint="dependency_probe",
    )
    result = execution.StreamingPackProcess().invoke(
        pack,
        execution.PackRuntime(Path(sys.executable), tmp_path, inherit_base_runtime=True),
        "probe",
        {},
        cancellation=threading.Event(),
        event_sink=lambda *_: None,
        timeout_seconds=10,
    )
    assert result == {"version": "pack-specific", "baseAvailable": True}


def broker_request(root):
    broker = object.__new__(WorkerBroker)
    broker.token = "dummy-launch-token"
    broker.roots = {key: (root,) for key in ("models", "inputs", "outputs")}
    return broker, {
        "token": broker.token,
        "packId": "oryn.mlx-lm",
        "version": "0.1.0",
        "command": "generate",
        "payload": {"modelRoot": str(root)},
        "timeoutSeconds": 1,
    }


@pytest.mark.parametrize("key", ["inputPath", "outputPath", "previewPath"])
def test_broker_rejects_directory_artifact_scopes(tmp_path, key):
    root = tmp_path / "allowed"
    root.mkdir()
    child = root / "subdirectory"
    child.mkdir()
    broker, request = broker_request(root)
    for path in (root, child):
        with pytest.raises(ValueError):
            broker._validate(request | {"payload": request["payload"] | {key: str(path)}})
    # Model-directory scope remains intentionally supported.
    assert broker._validate(request)["payload"]["modelRoot"] == str(root)


@pytest.mark.parametrize("key", ["inputPath", "outputPath", "previewPath"])
def test_broker_returns_canonical_artifact_paths(tmp_path, key):
    root = tmp_path / "allowed"
    root.mkdir()
    target = root / "dummy.txt"
    target.write_text("dummy")
    alias = tmp_path / "alias.txt"
    alias.symlink_to(target)
    broker, request = broker_request(root)
    received = request | {"payload": request["payload"] | {key: str(alias)}}
    validated = broker._validate(received)
    assert validated["payload"][key] == str(target)
    assert received["payload"][key] == str(alias)
    outside = tmp_path / "unapproved.txt"
    with pytest.raises(ValueError):
        broker._validate(request | {"payload": request["payload"] | {key: str(outside)}})
    ordinary = root / "new.txt"
    assert broker._validate(request | {"payload": request["payload"] | {key: str(ordinary)}})[
        "payload"
    ][key] == str(ordinary)


def test_scope_builder_uses_broker_canonical_path_and_rejects_aliases(tmp_path, monkeypatch):
    import oryn_engine

    root = tmp_path / "allowed"
    root.mkdir()
    artifact = root / "dummy.txt"
    artifact.write_text("dummy")
    alias = tmp_path / "alias.txt"
    alias.symlink_to(artifact)
    repo = Path(oryn_engine.__file__).resolve().parents[4]
    monkeypatch.setenv(
        "ORYN_PACK_SANDBOX_PROFILE",
        str(repo / "components/browser/apps/browser/build/security/runtime-pack.sb"),
    )
    monkeypatch.setenv("ORYN_ENGINE_ROOT", str(tmp_path / "engine"))
    monkeypatch.setenv("ORYN_SANDBOX_REQUIRED", "1")
    monkeypatch.setattr(execution.sys, "platform", "darwin")
    monkeypatch.setattr(execution.os, "confstr", lambda _key: str(tmp_path / "cache"))
    runtime = execution.PackRuntime(Path(sys.executable), None)
    with pytest.raises(JobExecutionError) as error:
        execution.StreamingPackProcess._sandboxed_arguments(
            runtime,
            {"inputPath": str(alias)},
            str(root),
            ["UNEXECUTED"],
        )
    assert error.value.code == "PACK_SANDBOX_INVALID"
    broker, request = broker_request(root)
    validated = broker._validate(
        request
        | {
            "payload": request["payload"]
            | {
                "inputPath": str(alias),
                "outputPath": str(alias),
            }
        }
    )
    args = execution.StreamingPackProcess._sandboxed_arguments(
        runtime,
        validated["payload"],
        str(root),
        ["UNEXECUTED"],
    )
    parameters = dict(
        args[i + 1].split("=", 1) for i, value in enumerate(args[:-1]) if value == "-D"
    )
    assert parameters["ARTIFACT_INPUT_ROOT"] == str(root)
    assert parameters["ARTIFACT_OUTPUT_ROOT"] == str(root)
