"""Sealed, bounded parser launch. Unsupported containment always fails closed."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import queue
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil

from oryn_engine.source_inspector_worker import (
    MAX_REQUEST_BYTES,
    MAX_RESULT_BYTES,
    MAX_SOURCE_BYTES,
    NORMALIZER,
    InspectionError,
)

PARSER_WALL_SECONDS = 5
PARSER_MEMORY_BYTES = 256 * 1024 * 1024
PARSER_CPU_SECONDS = 3
_SLOTS = threading.BoundedSemaphore(2)
_TRANSACTIONS = threading.BoundedSemaphore(2)


def inspection_operation_id(context):
    identity = json.dumps(
        [context["tabId"], context["documentId"]], ensure_ascii=False, separators=(",", ":")
    ).encode()
    return "source_" + hashlib.sha256(identity).hexdigest()


def bounded_inspection(inspect, cancellation, *, seconds=60):
    """Bound the entire trusted data-reader transaction, including stalled HTTP.

    Python cannot forcibly terminate an HTTP-reading thread. At most two daemon
    readers may remain until their socket timeout; their slots stay occupied and
    their late results are discarded. They cannot install, execute or attach a
    model. Parser processes separately enforce the fixed OS policy.
    """
    if cancellation.is_set():
        raise InspectionError("Source inspection cancelled")
    if not _TRANSACTIONS.acquire(blocking=False):
        raise InspectionError("Source inspection capacity exceeded")
    result = queue.Queue(maxsize=1)

    def read():
        try:
            result.put((True, inspect()))
        except Exception as exc:
            result.put((False, exc))
        finally:
            _TRANSACTIONS.release()

    thread = threading.Thread(target=read, name="oryn-source-data-reader", daemon=True)
    try:
        thread.start()
    except Exception:
        _TRANSACTIONS.release()
        raise
    deadline = time.monotonic() + seconds
    while True:
        if cancellation.is_set():
            raise InspectionError("Source inspection cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            cancellation.set()
            raise InspectionError("Source inspection deadline exceeded")
        try:
            success, value = result.get(timeout=min(0.05, remaining))
        except queue.Empty:
            continue
        if cancellation.is_set():
            raise InspectionError("Source inspection cancelled")
        if success:
            return value
        raise value


def inspect_source(raw: bytes, *, cancellation=None) -> dict:
    """Only the supervised Engine's authenticated broker can launch a sibling."""
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_SOURCE_BYTES:
        raise InspectionError("Source byte limit")
    if os.environ.get("ORYN_WORKER_BROKER_SOCKET"):
        from oryn_engine.runtime_packs.broker import invoke_source_inspector

        return invoke_source_inspector(raw, cancellation=cancellation)
    return launch_source_inspector(raw, cancellation=cancellation)


def launch_source_inspector(raw: bytes, *, cancellation=None) -> dict:
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise InspectionError("SOURCE_SANDBOX_UNAVAILABLE")
    if not _SLOTS.acquire(timeout=PARSER_WALL_SECONDS):
        raise InspectionError("Source parser capacity exceeded")
    try:
        return _launch(raw, cancellation)
    finally:
        _SLOTS.release()


def _launch(raw, cancellation):
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_SOURCE_BYTES:
        raise InspectionError("Source byte limit")
    if cancellation is not None and cancellation.is_set():
        raise InspectionError("Source inspection cancelled")
    worker = Path(__file__).with_name("source_inspector_worker.py").resolve(strict=True)
    profile = Path(__file__).with_name("data") / "source-inspection.sb"
    python = Path(sys.executable).resolve(strict=True)
    root = Path(sys.base_prefix).resolve(strict=True)
    frame = json.dumps({"source": base64.b64encode(raw).decode("ascii")}).encode()
    if len(frame) > MAX_REQUEST_BYTES:
        raise InspectionError("Source request limit")
    # This is not a runtime-pack override. The executable, script and policy
    # are fixed installed resources; no caller-supplied path/profile is accepted.
    with tempfile.TemporaryDirectory(prefix="oryn-source-parser-") as scratch:
        command = [
            "/usr/bin/sandbox-exec",
            "-f",
            str(profile),
            "-D",
            f"PYTHON={python}",
            "-D",
            f"PYTHON_ROOT={root}",
            "-D",
            f"WORKER={worker}",
            str(python),
            "-I",
            "-S",
            "-B",
            str(worker),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=scratch,
            env={"HOME": scratch, "TMPDIR": scratch, "PATH": "/usr/bin:/bin"},
            close_fds=True,
            start_new_session=True,
        )
        assert process.stdin is not None and process.stdout is not None
        output = bytearray()
        pending = memoryview(frame)
        deadline = time.monotonic() + PARSER_WALL_SECONDS
        selector = selectors.DefaultSelector()
        selector.register(process.stdin, selectors.EVENT_WRITE)
        selector.register(process.stdout, selectors.EVENT_READ)
        monitored = psutil.Process(process.pid)
        try:
            while selector.get_map():
                if time.monotonic() > deadline:
                    raise InspectionError("Source parser deadline exceeded")
                if cancellation is not None and cancellation.is_set():
                    raise InspectionError("Source inspection cancelled")
                try:
                    if monitored.memory_info().rss > PARSER_MEMORY_BYTES:
                        raise InspectionError("Source parser resident-memory limit")
                except psutil.NoSuchProcess:
                    pass
                for key, _ in selector.select(timeout=0.05):
                    if key.fileobj is process.stdin:
                        written = os.write(process.stdin.fileno(), pending[:4096])
                        pending = pending[written:]
                        if not pending:
                            selector.unregister(process.stdin)
                            process.stdin.close()
                    else:
                        chunk = os.read(process.stdout.fileno(), MAX_RESULT_BYTES + 1 - len(output))
                        output.extend(chunk)
                        if len(output) > MAX_RESULT_BYTES:
                            raise InspectionError("Source parser result limit")
                        if not chunk:
                            selector.unregister(process.stdout)
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InspectionError("Source parser failed") from exc
        finally:
            selector.close()
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=1)
            process.stdout.close()
            if not process.stdin.closed:
                process.stdin.close()
    if process.returncode != 0 or len(output) > MAX_RESULT_BYTES:
        raise InspectionError("Source parser failed or exceeded its result limit")
    message = json.loads(output)
    if message.get("ok") is not True:
        raise InspectionError("SOURCE_INSPECTION_REJECTED")
    result = message.get("result")
    if (
        not isinstance(result, dict)
        or result.get("schemaVersion") != 1
        or result.get("normalizer") != NORMALIZER
        or result.get("sizeBytes") != len(raw)
        or result.get("rawSha256") != hashlib.sha256(raw).hexdigest()
    ):
        raise InspectionError("Source parser result identity mismatch")
    return result
