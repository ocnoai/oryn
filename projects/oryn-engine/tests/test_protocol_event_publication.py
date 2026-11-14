"""Post-fix event publication regression; main thread releases blocked writer."""

import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace


def test_concurrent_event_publication_preserves_sequence(tmp_path):
    import oryn_engine

    args = SimpleNamespace(
        repo=Path(oryn_engine.__file__).resolve().parents[4], scratch=tmp_path, portable=False
    )
    substitutions = []
    from oryn_engine.protocol.dispatcher import ProtocolDispatcher
    from oryn_engine.protocol.writer import ProtocolWriter

    root = args.repo / "projects/oryn-protocol/fixtures/v1"
    resource = json.loads((root / "runtime-resources-event.json").read_text())
    job = json.loads((root / "job-state-event.json").read_text())
    output = io.BytesIO()
    writer = ProtocolWriter(output)
    allocated, release = threading.Event(), threading.Event()
    failures = []

    def sink(message):
        if message["seq"] == 1:
            allocated.set()
            assert release.wait(3), "SCHEDULING_SEAM_TIMED_OUT"
        writer.write(message)

    dispatcher = object.__new__(ProtocolDispatcher)
    dispatcher._state_lock = threading.RLock()
    dispatcher._event_lock = threading.RLock()
    dispatcher._event_sequence = 0
    dispatcher._event_sink = sink

    def first():
        try:
            dispatcher._resource_event(resource["event"], resource["data"])
        except BaseException as error:
            failures.append(type(error).__name__)

    attempted, second_done = threading.Event(), threading.Event()

    def second():
        attempted.set()
        try:
            dispatcher._job_event(job["event"], job["data"])
        except BaseException as error:
            failures.append(type(error).__name__)
        finally:
            second_done.set()

    thread = threading.Thread(target=first)
    other = threading.Thread(target=second)
    thread.start()
    second_completed_before_release = None
    try:
        assert allocated.wait(3)
        other.start()
        assert attempted.wait(3)
        # The main thread must release the sink; it never calls the contending
        # callback itself, which would deadlock after the publication fix.
        second_completed_before_release = second_done.wait(0.25)
    finally:
        release.set()
        thread.join(timeout=3)
        if other.ident is not None:
            other.join(timeout=3)
    assert not thread.is_alive() and not other.is_alive()
    assert not failures
    data = output.getvalue()
    assert len(data) <= 16 * 1024
    messages = [json.loads(line) for line in data.splitlines()]
    assert [m["seq"] for m in messages] == [1, 2], "EVENT_PUBLICATION_REORDERED"
    assert second_completed_before_release is False
    control_output = io.BytesIO()
    control_writer = ProtocolWriter(control_output)
    control = object.__new__(ProtocolDispatcher)
    control._state_lock = threading.RLock()
    control._event_lock = threading.RLock()
    control._event_sequence = 0
    control._event_sink = control_writer.write
    control._resource_event(resource["event"], resource["data"])
    control._job_event(job["event"], job["data"])
    control_sequences = [json.loads(line)["seq"] for line in control_output.getvalue().splitlines()]
    assert control_sequences == [1, 2]
    schema_valid = None
    if not args.portable:
        from oryn_protocol import ProtocolSchemaStore

        schemas = ProtocolSchemaStore(
            args.repo / "projects/oryn-protocol/schemas/v1/protocol-v1.schema.json"
        )
        definitions = {
            "runtime.resources.updated": "runtimeResourcesUpdatedEvent",
            "job.state": "jobStateEvent",
        }
        schema_valid = [schemas.is_valid(definitions[m["event"]], m) for m in messages]
        assert all(schema_valid), "INVALID_EVENT_FIXTURE"
    args.scratch.mkdir(parents=True, exist_ok=True)
    # Parent must promote this exact declared file through its retained-descriptor procedure.
    (args.scratch / "two-events.jsonl").write_bytes(data)
    print(
        json.dumps(
            {
                "stage": "actual_callbacks_and_writer_postfix_contention",
                "sequences": [m["seq"] for m in messages],
                "bytes": len(data),
                "schema_valid": schema_valid,
                "failures": failures,
                "real_resource_monitor_started": False,
                "unused_import_substitutions": substitutions,
                "serial_negative_control_sequences": control_sequences,
                "second_completed_before_first_sink_release": second_completed_before_release,
                "browser_receiver_executed": False,
                "artifact": "two-events.jsonl",
            },
            sort_keys=True,
        )
    )
