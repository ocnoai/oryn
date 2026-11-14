"""Post-fix retention regression on actual app.run; same bounded fixture as baseline."""

import copy
import gc
import json
import queue
import threading
import time
import tracemalloc
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def test_completed_control_futures_are_released():
    import oryn_engine

    args = SimpleNamespace(repo=Path(oryn_engine.__file__).resolve().parents[4])
    substitutions = []
    from oryn_engine import app

    template = json.loads(
        (args.repo / "projects/oryn-protocol/fixtures/v1/runtime-capabilities.json").read_text()
    )
    capability = template.get("result", template)
    references = []

    class ObservedExecutor(ThreadPoolExecutor):
        def submit(self, *a, **kw):
            future = super().submit(*a, **kw)
            references.append(weakref.ref(future))
            return future

    class Input:
        def __init__(self):
            self.q = queue.Queue(maxsize=4)

        def readline(self, _limit):
            return self.q.get(timeout=5)

    class Output:
        def __init__(self):
            self.count = 0
            self.condition = threading.Condition()

        def write(self, payload):
            message = json.loads(payload)
            assert message["ok"] is True
            with self.condition:
                self.count += 1
                self.condition.notify_all()
            return len(payload)

        def flush(self):
            pass

        def wait(self, count):
            with self.condition:
                assert self.condition.wait_for(lambda: self.count >= count, timeout=5)

    class Dispatcher:
        handshake_complete = True
        max_message_bytes = 16 * 1024 * 1024

        def set_event_sink(self, sink):
            pass

        def stop_runtime_services(self):
            pass

        def response_written(self, method, response):
            pass

        def dispatch(self, message):
            return {
                "v": 1,
                "type": "response",
                "id": message["id"],
                "ok": True,
                "result": copy.deepcopy(capability),
            }, False

    source, output = Input(), Output()
    done, failures = [], []

    def run():
        try:
            done.append(app.run(source, output, Dispatcher()))
        except BaseException as error:
            failures.append(type(error).__name__)

    batches = []
    tracemalloc.start()
    with patch.object(app, "ThreadPoolExecutor", ObservedExecutor):
        thread = threading.Thread(target=run)
        thread.start()
        count = 0
        try:
            for size in (32, 64, 128):
                for _ in range(size):
                    count += 1
                    source.q.put(
                        json.dumps(
                            {
                                "v": 1,
                                "type": "request",
                                "id": f"req_{count}",
                                "method": "runtime.capabilities",
                                "params": {},
                            }
                        ).encode()
                        + b"\n",
                        timeout=2,
                    )
                    output.wait(count)
                # Give the last response callback time to leave its finally block.
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    gc.collect()
                    if sum(ref() is not None for ref in references) <= 1:
                        break
                    time.sleep(0.005)
                live = [ref() for ref in references if ref() is not None]
                batches.append(
                    {
                        "completed_requests": count,
                        "retained_futures": len(live),
                        "unfinished_futures": sum(not future.done() for future in live),
                        "retained_result_json_bytes": sum(
                            len(json.dumps(future.result()[0])) for future in live if future.done()
                        ),
                        "tracemalloc_current_bytes": tracemalloc.get_traced_memory()[0],
                    }
                )
                del live
                # The final response callback can still be unwinding when write()
                # wakes the observer. At most this one Future may remain briefly.
                assert batches[-1]["retained_futures"] <= 1, "COMPLETED_FUTURES_RETAINED"
                assert tracemalloc.get_traced_memory()[0] < 16 * 1024 * 1024
        finally:
            source.q.put(b"", timeout=2)
            thread.join(timeout=6)
    assert not thread.is_alive(), "RUN_DID_NOT_EXIT"
    gc.collect()
    assert not failures and done == [0]
    assert sum(ref() is not None for ref in references) == 0
    print(
        json.dumps(
            {
                "stage": "actual_app_run_with_deterministic_fixture_dispatcher",
                "batches": batches,
                "after_run_retained_futures": sum(ref() is not None for ref in references),
                "exit_codes": done,
                "failures": failures,
                "unused_import_substitutions": substitutions,
                "real_dispatcher_and_browser_reachability_executed": False,
            },
            sort_keys=True,
        )
    )
