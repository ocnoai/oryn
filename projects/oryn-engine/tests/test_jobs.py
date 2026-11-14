from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from oryn_engine.jobs import EngineJob, JobExecutionError, JobManager


@dataclass(frozen=True)
class FakeSession:
    model: dict[str, str]


class FakeSessions:
    def get(self, session_id: str) -> FakeSession:
        return FakeSession(
            {
                "provider": "huggingface",
                "repository": f"org/{session_id}",
                "revision": "a" * 40,
            }
        )


class ControlledRunner:
    def __init__(self) -> None:
        self.started: dict[str, threading.Event] = {}
        self.release: dict[str, threading.Event] = {}
        self.order: list[str] = []

    def __call__(
        self,
        job: EngineJob,
        cancellation: threading.Event,
        emit: Any,
    ) -> object:
        self.order.append(job.job_id)
        started = self.started.setdefault(job.job_id, threading.Event())
        release = self.release.setdefault(job.job_id, threading.Event())
        started.set()
        emit("job.progress", {"stage": "executing"})
        while not release.wait(0.005):
            if cancellation.is_set():
                return {"kind": "structured", "data": None}
        if isinstance(job.input, dict) and job.input.get("fail"):
            raise JobExecutionError("TEST_FAILURE", "injected")
        return {"kind": "structured", "data": {"jobId": job.job_id}}


def wait_for(predicate: Any, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("Timed out waiting for job state")
        time.sleep(0.005)


def test_one_global_fifo_lane_across_sessions_models_and_submitters() -> None:
    runner = ControlledRunner()
    events: list[tuple[str, dict[str, Any]]] = []
    activity: list[tuple[int, int]] = []
    manager = JobManager(
        FakeSessions(),
        lambda name, data: events.append((name, data)),
        runner,
        lambda running, queued: activity.append((running, queued)),
    )
    try:
        first = manager.create("job_first", "session_model_a", {"prompt": "a"})
        assert first["state"] == "running"
        assert events == []  # response-before-event activation
        manager.response_written("job_first")
        assert runner.started.setdefault("job_first", threading.Event()).wait(1)

        second = manager.create("job_second", "session_model_b", {"prompt": "b"})
        third = manager.create("job_third", "session_model_a", {"prompt": "c"})
        assert second["state"] == "queued" and second["queue"]["position"] == 1
        assert third["state"] == "queued" and third["queue"]["position"] == 2
        manager.response_written("job_second")
        manager.response_written("job_third")
        assert not runner.started.setdefault("job_second", threading.Event()).is_set()

        runner.release.setdefault("job_first", threading.Event()).set()
        assert runner.started["job_second"].wait(1)
        runner.release.setdefault("job_second", threading.Event()).set()
        assert runner.started.setdefault("job_third", threading.Event()).wait(1)
        runner.release.setdefault("job_third", threading.Event()).set()
        wait_for(lambda: manager.status("job_third")["state"] == "completed")

        assert runner.order == ["job_first", "job_second", "job_third"]
        assert max(running for running, _queued in activity) == 1
        for job_id in ["job_first", "job_second", "job_third"]:
            sequences = [data["sequence"] for _name, data in events if data["jobId"] == job_id]
            assert sequences == sorted(set(sequences))
            terminals = [
                name
                for name, data in events
                if data["jobId"] == job_id
                and name in {"job.completed", "job.failed", "job.cancelled"}
            ]
            assert len(terminals) == 1
    finally:
        manager.shutdown()


def test_queued_and_running_cancellation_are_isolated_and_idempotent() -> None:
    runner = ControlledRunner()
    events: list[tuple[str, dict[str, Any]]] = []
    manager = JobManager(FakeSessions(), lambda name, data: events.append((name, data)), runner)
    try:
        manager.create("job_running", "session_a", {})
        manager.response_written("job_running")
        assert runner.started.setdefault("job_running", threading.Event()).wait(1)
        manager.create("job_cancelled", "session_b", {})
        manager.response_written("job_cancelled")

        cancelled = manager.cancel("job_cancelled", "website")
        duplicate = manager.cancel("job_cancelled", "website")
        assert cancelled["state"] == duplicate["state"] == "cancelled"
        assert "job_cancelled" not in runner.order
        assert manager.status("job_running")["state"] == "running"

        manager.cancel("job_running", "user")
        wait_for(lambda: manager.status("job_running")["state"] == "cancelled")
        terminals = [name for name, _data in events if name == "job.cancelled"]
        assert len(terminals) == 2
    finally:
        manager.shutdown()


def test_failure_and_completion_both_advance_the_queue() -> None:
    runner = ControlledRunner()
    manager = JobManager(FakeSessions(), runner=runner)
    try:
        manager.create("job_failure", "session_a", {"fail": True})
        manager.response_written("job_failure")
        manager.create("job_after_failure", "session_b", {})
        manager.response_written("job_after_failure")
        assert runner.started.setdefault("job_failure", threading.Event()).wait(1)
        runner.release.setdefault("job_failure", threading.Event()).set()
        assert runner.started.setdefault("job_after_failure", threading.Event()).wait(1)
        assert manager.status("job_failure")["state"] == "failed"
        runner.release.setdefault("job_after_failure", threading.Event()).set()
        wait_for(lambda: manager.status("job_after_failure")["state"] == "completed")
    finally:
        manager.shutdown()


def test_job_id_retry_is_idempotent_and_conflict_safe() -> None:
    runner = ControlledRunner()
    manager = JobManager(FakeSessions(), runner=runner)
    try:
        first = manager.create("job_retry", "session_a", {"prompt": "same"})
        assert manager.create("job_retry", "session_a", {"prompt": "same"}) == first
        try:
            manager.create("job_retry", "session_a", {"prompt": "different"})
        except Exception as exc:
            assert getattr(exc, "code", None) == "JOB_ID_CONFLICT"
        else:
            raise AssertionError("Expected conflicting retry to fail")
    finally:
        manager.shutdown()


def test_terminal_job_retention_is_count_bounded() -> None:
    runner = ControlledRunner()
    manager = JobManager(FakeSessions(), runner=runner, max_terminal_jobs=1)
    try:
        for job_id in ("job_old", "job_new"):
            manager.create(job_id, "session_a", {})
            manager.response_written(job_id)
            assert runner.started.setdefault(job_id, threading.Event()).wait(1)
            runner.release.setdefault(job_id, threading.Event()).set()
            wait_for(lambda current=job_id: manager.status(current)["state"] == "completed")
        try:
            manager.status("job_old")
        except Exception as exc:
            assert getattr(exc, "code", None) == "JOB_NOT_FOUND"
        else:
            raise AssertionError("Expected the oldest terminal job to be pruned")
        assert manager.status("job_new")["state"] == "completed"
    finally:
        manager.shutdown()


def test_streaming_deltas_are_ordered_and_keep_the_global_lane() -> None:
    release = threading.Event()
    second_started = threading.Event()
    events: list[tuple[str, dict[str, Any]]] = []

    def stream_runner(job: EngineJob, cancellation: threading.Event, emit: Any) -> object:
        del cancellation
        if job.job_id == "job_stream":
            emit("job.state", {"state": "streaming"})
            emit("job.output.delta", {"channel": "text", "delta": "one "})
            emit("job.output.delta", {"channel": "text", "delta": "two"})
            assert release.wait(1)
        else:
            second_started.set()
        return {"kind": "structured", "value": {"text": "done"}}

    manager = JobManager(
        FakeSessions(), lambda name, data: events.append((name, data)), stream_runner
    )
    try:
        manager.create("job_stream", "session_a", {})
        manager.response_written("job_stream")
        wait_for(lambda: manager.status("job_stream")["state"] == "streaming")
        manager.create("job_after_stream", "session_b", {})
        manager.response_written("job_after_stream")
        assert manager.status("job_after_stream")["state"] == "queued"
        assert not second_started.is_set()

        release.set()
        assert second_started.wait(1)
        wait_for(lambda: manager.status("job_after_stream")["state"] == "completed")
        stream_events = [(name, data) for name, data in events if data["jobId"] == "job_stream"]
        assert [
            data.get("delta") for name, data in stream_events if name == "job.output.delta"
        ] == [
            "one ",
            "two",
        ]
        terminal_index = next(
            index for index, (name, _data) in enumerate(stream_events) if name == "job.completed"
        )
        assert all(
            name != "job.output.delta" for name, _data in stream_events[terminal_index + 1 :]
        )
        sequences = [data["sequence"] for _name, data in stream_events]
        assert sequences == sorted(set(sequences))
    finally:
        manager.shutdown()
