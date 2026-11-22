from __future__ import annotations

import json
import logging
import queue
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from json import JSONDecodeError
from typing import BinaryIO

from oryn_engine.credentials import HubCredentialVault, SupervisorCredentialChannel
from oryn_engine.protocol.dispatcher import ProtocolDispatcher
from oryn_engine.protocol.writer import ProtocolWriter

LOGGER = logging.getLogger("oryn_engine")
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


class ProtocolCorruptionError(RuntimeError):
    """Raised when message framing can no longer be trusted."""


def _read_line(stream: BinaryIO, max_message_bytes: int) -> bytes:
    line = stream.readline(max_message_bytes + 2)
    if not line:
        return b""
    if len(line) > max_message_bytes + 1 or (
        len(line) > max_message_bytes and not line.endswith(b"\n")
    ):
        raise ProtocolCorruptionError("Protocol input exceeded the maximum line size.")
    if not line.endswith(b"\n"):
        raise ProtocolCorruptionError("Protocol input ended before a complete JSON Line.")
    return line[:-1].removesuffix(b"\r")


def run(
    input_stream: BinaryIO = sys.stdin.buffer,
    output_stream: BinaryIO = sys.stdout.buffer,
    dispatcher: ProtocolDispatcher | None = None,
) -> int:
    writer = ProtocolWriter(output_stream)
    credential_channel = None
    if dispatcher is None:
        credential_vault = HubCredentialVault()
        credential_channel = SupervisorCredentialChannel.from_environment(credential_vault)
        if credential_channel is not None:
            credential_channel.start()
        active_dispatcher = ProtocolDispatcher(hugging_face_token_provider=credential_vault.get)
    else:
        active_dispatcher = dispatcher
    active_dispatcher.set_event_sink(writer.write)
    incoming: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=256)
    shutdown_requested = threading.Event()
    reader = threading.Thread(
        target=_read_messages,
        args=(input_stream, active_dispatcher, incoming),
        name="oryn-protocol-reader",
        daemon=True,
    )
    reader.start()
    executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="oryn-control")
    futures: set[Future[tuple[dict[str, object], bool]]] = set()
    futures_changed = threading.Condition()

    def completed(
        future: Future[tuple[dict[str, object], bool]],
        method: str,
    ) -> None:
        try:
            response, should_exit = future.result()
            writer.write(response)
            active_dispatcher.response_written(method, response)
            if should_exit:
                if method == "runtime.shutdown" and response.get("ok") is True:
                    active_dispatcher.publish_event("runtime.state", {"state": "shutting-down"})
                shutdown_requested.set()
        except Exception:
            LOGGER.exception("Unhandled control-plane worker failure")
            shutdown_requested.set()
        finally:
            # A completed result must not survive for the lifetime of the Engine.
            with futures_changed:
                futures.discard(future)
                futures_changed.notify_all()

    exit_code = 0
    try:
        while not shutdown_requested.is_set():
            try:
                kind, value = incoming.get(timeout=0.05)
            except queue.Empty:
                continue
            if kind == "error":
                LOGGER.error("Fatal protocol input error: %s", value)
                exit_code = 2
                break
            if kind == "eof":
                break
            message = value
            if not isinstance(message, dict):
                LOGGER.error("Fatal protocol envelope error: top-level value is not an object")
                exit_code = 2
                break
            if not active_dispatcher.handshake_complete:
                response, should_exit = active_dispatcher.dispatch(message)
                writer.write(response)
                if response.get("ok") is True and active_dispatcher.handshake_complete:
                    active_dispatcher.start_runtime_services()
                if should_exit:
                    shutdown_requested.set()
                continue
            method = str(message.get("method", ""))
            if method == "job.create":
                response, should_exit = active_dispatcher.dispatch(message)
                writer.write(response)
                active_dispatcher.response_written(method, response)
                if should_exit:
                    shutdown_requested.set()
                continue
            with futures_changed:
                while len(futures) >= 128 and not shutdown_requested.is_set():
                    futures_changed.wait(timeout=0.01)
            if shutdown_requested.is_set():
                break
            future = executor.submit(active_dispatcher.dispatch, message)
            with futures_changed:
                futures.add(future)
            future.add_done_callback(
                lambda done, request_method=method: completed(done, request_method)
            )
            del future
    finally:
        active_dispatcher.stop_runtime_services()
        if exit_code != 0:
            with futures_changed:
                pending_futures = tuple(futures)
            # cancel() may run a callback synchronously: never hold its condition.
            for future in pending_futures:
                future.cancel()
        executor.shutdown(wait=exit_code == 0, cancel_futures=exit_code != 0)
        # The Browser closes stdin after acknowledging runtime.shutdown. Give the
        # reader time to observe EOF so CPython never finalizes around a live
        # buffered-stdin daemon thread.
        reader.join(timeout=5)
    return exit_code


def _read_messages(
    stream: BinaryIO,
    dispatcher: ProtocolDispatcher,
    incoming: queue.Queue[tuple[str, object]],
) -> None:
    while True:
        try:
            raw_line = _read_line(stream, dispatcher.max_message_bytes)
            if raw_line == b"":
                incoming.put(("eof", None))
                return
            message = json.loads(raw_line.decode("utf-8"))
            _validate_json_complexity(message)
        except (
            ProtocolCorruptionError,
            UnicodeDecodeError,
            JSONDecodeError,
            RecursionError,
        ) as exc:
            incoming.put(("error", exc))
            return
        incoming.put(("message", message))


def _validate_json_complexity(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > 100_000 or depth > 64:
            raise ProtocolCorruptionError("Protocol input exceeded structural complexity limits.")
        if isinstance(item, list):
            if len(item) > 50_000:
                raise ProtocolCorruptionError(
                    "Protocol input exceeded structural complexity limits."
                )
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, dict):
            if len(item) > 10_000:
                raise ProtocolCorruptionError(
                    "Protocol input exceeded structural complexity limits."
                )
            stack.extend((child, depth + 1) for child in item.values())


def main() -> int:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    LOGGER.info("Local AI Engine starting")
    try:
        return run()
    except BrokenPipeError:
        LOGGER.info("Browser protocol pipe closed")
        return 0
    except KeyboardInterrupt:
        LOGGER.info("Local AI Engine interrupted")
        return 0
    except Exception:
        LOGGER.exception("Unhandled engine failure")
        return 1
