"""Private launch broker: sibling Seatbelt workers, never nested sandbox reinitialization."""

from __future__ import annotations

import contextlib
import hmac
import json
import os
import secrets
import select
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from oryn_engine.jobs import JobExecutionError

MAX_REQUEST = 1024 * 1024
# Separate data-only protocol budget; ordinary pack envelopes stay at 1 MiB.
MAX_SOURCE_REQUEST = 1400 * 1024
SOURCE_INSPECTOR = "oryn.internal.source-inspector"


def scoped_path(value: object, roots: tuple[Path, ...], *, directory: bool) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("Invalid worker path")
    path = Path(value).resolve(strict=directory)
    if not any(path.is_relative_to(root) for root in roots):
        raise ValueError("Worker path exceeds its scope")
    if directory and not path.is_dir():
        raise ValueError("Worker directory is unavailable")
    if not directory and (
        path.is_dir() or not any(path.parent.is_relative_to(root) for root in roots)
    ):
        raise ValueError("Worker artifact must be a file inside its scope")
    return path


class WorkerBroker:
    """Only the supervised Engine can request a signed pack with scoped data paths."""

    def __init__(self, scratch: Path, roots: dict[str, tuple[Path, ...]]) -> None:
        # A private short path avoids Darwin's 104-byte AF_UNIX path limit. It
        # contains only this authenticated socket, never model or runtime files.
        self.directory = tempfile.TemporaryDirectory(prefix="oryn-worker-", dir="/private/tmp")
        self.path = Path(self.directory.name) / "rpc"
        self.token = secrets.token_hex(32)
        self.roots = roots
        self.stopped = threading.Event()
        self.slots = threading.BoundedSemaphore(4)
        self.connections: set[socket.socket] = set()
        self.workers: set[threading.Thread] = set()
        self.lock = threading.Lock()
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(self.path))
        self.path.chmod(0o600)
        self.server.listen(4)
        self.server.settimeout(0.2)
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stopped.set()
        self.server.close()
        with self.lock:
            connections = tuple(self.connections)
        for connection in connections:
            with contextlib.suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)
        self.thread.join(timeout=1)
        with self.lock:
            workers = tuple(self.workers)
        for worker in workers:
            worker.join(timeout=11)
        self.path.unlink(missing_ok=True)
        self.directory.cleanup()

    def _accept(self) -> None:
        while not self.stopped.is_set():
            try:
                connection, _ = self.server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            if not self.slots.acquire(blocking=False):
                connection.close()
                continue
            with self.lock:
                self.connections.add(connection)
            worker = threading.Thread(target=self._serve, args=(connection,), daemon=True)
            with self.lock:
                self.workers.add(worker)
            worker.start()

    def _validate(self, request: object) -> dict[str, Any]:
        if not isinstance(request, dict) or set(request) != {
            "token",
            "packId",
            "version",
            "command",
            "payload",
            "timeoutSeconds",
        }:
            raise ValueError("Invalid broker request")
        if not isinstance(request["token"], str) or not hmac.compare_digest(
            request["token"], self.token
        ):
            raise ValueError("Invalid broker authority")
        if request["packId"] == SOURCE_INSPECTOR:
            from oryn_engine.source_inspector_worker import NORMALIZER

            payload = request["payload"]
            if (
                request["version"] != NORMALIZER
                or request["command"] != "inspect-source"
                or request["timeoutSeconds"] != 5
                or not isinstance(payload, dict)
                or set(payload) != {"source"}
                or not isinstance(payload["source"], str)
                or len(payload["source"]) > 1398104
            ):
                raise ValueError("Invalid source-inspector request")
            return request
        from oryn_engine.providers import PROVIDER_COMMANDS

        if request["command"] not in {"probe", "generate", *PROVIDER_COMMANDS}:
            raise ValueError("Invalid worker command")
        timeout = request["timeoutSeconds"]
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 < timeout <= 7200
        ):
            raise ValueError("Invalid worker timeout")
        payload = request["payload"]
        if not isinstance(payload, dict):
            raise ValueError("Invalid worker payload")
        validated_payload = dict(payload)
        # All filesystem authority is assigned by Browser launch configuration.
        # Payloads cannot grant a model access to ambient files or select executable code.
        for key, roots, directory in (
            ("modelRoot", self.roots["models"], True),
            ("inputPath", self.roots["inputs"], False),
            ("outputPath", self.roots["outputs"], False),
            ("previewPath", self.roots["outputs"], False),
        ):
            value = payload.get(key)
            if value is not None:
                validated_payload[key] = str(scoped_path(value, roots, directory=directory))
        if request["command"] in {"generate", "prepare"} and "modelRoot" not in payload:
            raise ValueError("Missing model scope")
        return {**request, "payload": validated_payload}

    def _serve(self, connection: socket.socket) -> None:
        from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
        from oryn_engine.runtime_packs.execution import PackRuntimeLocator, StreamingPackProcess

        def send(value: dict) -> None:
            encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode() + b"\n"
            if len(encoded) > MAX_REQUEST:
                raise ValueError("Worker response exceeded bounds")
            connection.sendall(encoded)

        broker = self

        class Cancellation:
            def is_set(self) -> bool:
                if broker.stopped.is_set():
                    return True
                try:
                    readable, _, _ = select.select([connection], [], [], 0)
                    # No further client input is permitted; disconnect means cancellation.
                    return bool(readable)
                except (OSError, ValueError):
                    return True

        try:
            connection.settimeout(5)
            data = bytearray()
            while b"\n" not in data and len(data) <= MAX_SOURCE_REQUEST:
                chunk = connection.recv(65536)
                if not chunk:
                    return
                data.extend(chunk)
            if len(data) > MAX_SOURCE_REQUEST or data.count(b"\n") != 1 or not data.endswith(b"\n"):
                raise ValueError("Invalid broker frame")
            request = self._validate(json.loads(data))
            if request["packId"] == SOURCE_INSPECTOR:
                import base64

                from oryn_engine.source_inspection import launch_source_inspector

                result = launch_source_inspector(
                    base64.b64decode(request["payload"]["source"], validate=True),
                    cancellation=Cancellation(),
                )
                send({"type": "result", "ok": True, "result": result})
                return
            if len(data) > MAX_REQUEST:
                raise ValueError("Pack request exceeds its unchanged bound")
            pack = builtin_registry().active_pack(request["packId"])
            if pack.version != request["version"]:
                raise ValueError("Worker build changed")
            runtime = PackRuntimeLocator().resolve(pack)
            if runtime is None:
                raise JobExecutionError("RUNTIME_DEPENDENCY_UNAVAILABLE", "Pack unavailable")
            result = StreamingPackProcess().invoke(
                pack,
                runtime,
                request["command"],
                request["payload"],
                cancellation=Cancellation(),
                event_sink=lambda name, value: send(
                    {"type": "event", "event": name, "data": value}
                ),
                timeout_seconds=request["timeoutSeconds"],
            )
            send({"type": "result", "ok": True, "result": result})
        except JobExecutionError as exc:
            with contextlib.suppress(OSError):
                send({"type": "result", "ok": False, "error": {"code": exc.code}})
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            with contextlib.suppress(OSError):
                send({"type": "result", "ok": False, "error": {"code": "PACK_BROKER_REJECTED"}})
        finally:
            connection.close()
            with self.lock:
                self.connections.discard(connection)
                self.workers.discard(threading.current_thread())
            self.slots.release()


def invoke_source_inspector(raw: bytes, *, cancellation=None):
    import base64
    from types import SimpleNamespace

    from oryn_engine.source_inspector_worker import NORMALIZER, InspectionError

    try:
        return invoke_broker(
            SimpleNamespace(pack_id=SOURCE_INSPECTOR, version=NORMALIZER),
            "inspect-source",
            {"source": base64.b64encode(raw).decode("ascii")},
            cancellation=cancellation or threading.Event(),
            event_sink=lambda *_: None,
            timeout_seconds=5,
        )
    except JobExecutionError as exc:
        raise InspectionError(exc.code) from None


def invoke_broker(pack, command, payload, *, cancellation, event_sink, timeout_seconds):
    from oryn_engine.runtime_packs.execution import StreamingPackProcess

    if cancellation.is_set():
        raise JobExecutionError("JOB_CANCELLED", "Runtime-pack work was cancelled.")
    request = {
        "token": os.environ["ORYN_WORKER_BROKER_TOKEN"],
        "packId": pack.pack_id,
        "version": pack.version,
        "command": command,
        "payload": payload,
        "timeoutSeconds": timeout_seconds,
    }
    encoded = json.dumps(request, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    limit = MAX_SOURCE_REQUEST if pack.pack_id == SOURCE_INSPECTOR else MAX_REQUEST
    if len(encoded) > limit:
        raise JobExecutionError("PACK_MESSAGE_TOO_LARGE", "Worker request exceeded bounds.")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(5)
        connection.connect(os.environ["ORYN_WORKER_BROKER_SOCKET"])
        connection.sendall(encoded)
        started = time.monotonic()
        buffer = bytearray()
        while True:
            if cancellation.is_set():
                raise JobExecutionError("JOB_CANCELLED", "Runtime-pack work was cancelled.")
            if time.monotonic() - started > timeout_seconds:
                raise JobExecutionError("PACK_TIMEOUT", "Runtime-pack process timed out.")
            if not select.select([connection], [], [], 0.1)[0]:
                continue
            chunk = connection.recv(65536)
            if not chunk:
                raise JobExecutionError("PACK_CRASHED", "Runtime-pack connection closed.")
            buffer.extend(chunk)
            while b"\n" in buffer:
                raw, _, rest = buffer.partition(b"\n")
                buffer = bytearray(rest)
                if len(raw) > MAX_REQUEST:
                    raise JobExecutionError(
                        "PACK_PROTOCOL_INVALID", "Worker response exceeded bounds."
                    )
                message = StreamingPackProcess._message(raw)
                if (
                    message.get("type") == "event"
                    and message.get("event")
                    in {"job.progress", "job.state", "job.output.delta", "pack.preview"}
                    and isinstance(message.get("data"), dict)
                ):
                    event_sink(message["event"], message["data"])
                    if cancellation.is_set():
                        raise JobExecutionError("JOB_CANCELLED", "Runtime-pack work was cancelled.")
                elif message.get("type") == "result":
                    if message.get("ok") is True and isinstance(message.get("result"), dict):
                        return message["result"]
                    raise JobExecutionError(
                        StreamingPackProcess._worker_error_code(message.get("error")),
                        "Runtime-pack execution failed.",
                    )
                else:
                    raise JobExecutionError("PACK_PROTOCOL_INVALID", "Invalid worker response.")
            if len(buffer) > MAX_REQUEST:
                raise JobExecutionError("PACK_PROTOCOL_INVALID", "Worker response exceeded bounds.")
