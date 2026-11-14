from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from oryn_engine.protocol.errors import protocol_error

TERMINAL_STATES = {"completed", "failed", "cancelled"}
JobEventSink = Callable[[str, dict[str, Any]], None]
RunnerEvent = Callable[[str, dict[str, Any]], None]
Runner = Callable[["EngineJob", threading.Event, RunnerEvent], object]


class SessionProvider(Protocol):
    def get(self, session_id: str) -> object: ...


class JobManagerError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class JobExecutionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class EngineJob:
    job_id: str
    session_id: str
    model: dict[str, str]
    input: object
    input_artifacts: tuple[object, ...]
    invocation_digest: str
    state: str
    enqueued_at: int
    task_profile: dict | None = None
    task_requirements: dict | None = None
    sequence: int = 0
    started_at: int | None = None
    completed_at: int | None = None
    cancellation_reason: str | None = None
    cancellation: threading.Event = field(default_factory=threading.Event)
    result: object | None = None
    error: dict[str, Any] | None = None
    queue: dict[str, Any] | None = None
    execution: dict[str, Any] = field(default_factory=dict)
    runtime_fallbacks: list[dict[str, str]] = field(default_factory=list)


class JobManager:
    """One process-global FIFO generation lane with terminal-state enforcement."""

    def __init__(
        self,
        sessions: SessionProvider,
        event_sink: JobEventSink | None = None,
        runner: Runner | None = None,
        activity_sink: Callable[[int, int], None] | None = None,
        model_activity_sink: Callable[[dict[str, str], int, int, int | None], None] | None = None,
        max_jobs: int = 256,
        max_terminal_jobs: int = 256,
        terminal_retention_ms: int = 15 * 60 * 1000,
    ) -> None:
        self.sessions = sessions
        self._event_sink = event_sink
        self._runner = runner or _unsupported_runner
        self._activity_sink = activity_sink
        self._model_activity_sink = model_activity_sink
        self._max_jobs = max_jobs
        self._max_terminal_jobs = max_terminal_jobs
        self._terminal_retention_ms = terminal_retention_ms
        self._jobs: dict[str, EngineJob] = {}
        self._queue: deque[str] = deque()
        self._current: str | None = None
        self._activated: set[str] = set()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._stopping = False
        self._thread = threading.Thread(
            target=self._dispatch_loop, name="oryn-generation-dispatcher", daemon=True
        )
        self._thread.start()

    def set_event_sink(self, sink: JobEventSink) -> None:
        self._event_sink = sink

    def create(
        self,
        job_id: str,
        session_id: str,
        input_value: object,
        input_artifacts: tuple[object, ...] = (),
    ) -> dict[str, Any]:
        session = self.sessions.get(session_id)
        model = getattr(session, "model", None)
        if not isinstance(model, dict):
            raise JobManagerError("SESSION_NOT_FOUND", "The Model Session does not exist.")
        profile = getattr(session, "result", {}).get("taskProfile")
        # Negotiated sessions enforce effective controls; legacy sessions retain their defaults.
        requirements = getattr(session, "requirements", None) or {}
        if profile is not None and requirements.get("contract"):
            from oryn_protocol.task_contracts import effective_input, validate_invocation

            try:
                validate_invocation(profile, input_value, requirements)
                input_value = effective_input(profile, input_value)
            except ValueError as exc:
                raise JobManagerError(
                    "JOB_INPUT_INVALID", "Input violates the negotiated task contract."
                ) from exc
        digest = _invocation_digest(session_id, input_value, input_artifacts)
        with self._condition:
            self._prune_terminal_locked()
            existing = self._jobs.get(job_id)
            if existing is not None:
                if existing.session_id != session_id or existing.invocation_digest != digest:
                    raise JobManagerError(
                        "JOB_ID_CONFLICT", "The job ID is bound to another invocation."
                    )
                return self._reference(existing)
            live_count = sum(job.state not in TERMINAL_STATES for job in self._jobs.values())
            if live_count >= self._max_jobs:
                raise JobManagerError("JOB_QUEUE_FULL", "The generation queue is full.")
            state = "running" if self._current is None else "queued"
            job = EngineJob(
                job_id=job_id,
                session_id=session_id,
                model=dict(model),
                input=input_value,
                input_artifacts=input_artifacts,
                invocation_digest=digest,
                state=state,
                enqueued_at=int(time.time() * 1000),
                task_profile=profile if requirements.get("contract") else None,
                task_requirements=requirements,
            )
            self._jobs[job_id] = job
            if state == "running":
                self._current = job_id
            else:
                self._queue.append(job_id)
                job.queue = self._queue_status(self._queue.index(job_id))
            self._sync_activity_locked()
            return self._reference(job)

    def response_written(self, job_id: str) -> None:
        with self._condition:
            job = self._require(job_id)
            self._activated.add(job_id)
            if job.state == "queued":
                self._emit_locked(job, "job.state", {"state": "queued", "queue": job.queue})
            elif job.state == "running":
                self._emit_locked(job, "job.state", {"state": "running"})
            self._condition.notify_all()

    def status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return self._status(self._require(job_id))

    def cancel(self, job_id: str, reason: str) -> dict[str, Any]:
        with self._condition:
            job = self._require(job_id)
            if job.state in TERMINAL_STATES:
                return self._status(job)
            job.cancellation_reason = reason
            job.cancellation.set()
            if job.state == "queued":
                self._queue.remove(job_id)
                self._terminal_locked(job, "cancelled", "job.cancelled", {"reason": reason})
                self._recompute_queue_locked()
                self._sync_activity_locked()
            self._condition.notify_all()
            return self._status(job)

    def shutdown(self) -> None:
        with self._condition:
            self._stopping = True
            for job in tuple(self._jobs.values()):
                if job.state not in TERMINAL_STATES:
                    job.cancellation_reason = "runtime-shutdown"
                    job.cancellation.set()
                    if job.state == "queued":
                        self._terminal_locked(
                            job,
                            "cancelled",
                            "job.cancelled",
                            {"reason": "runtime-shutdown"},
                        )
                    elif job.job_id not in self._activated:
                        self._terminal_locked(
                            job,
                            "cancelled",
                            "job.cancelled",
                            {"reason": "runtime-shutdown"},
                        )
                        if self._current == job.job_id:
                            self._current = None
            self._queue.clear()
            self._sync_activity_locked()
            self._condition.notify_all()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=5)

    def _dispatch_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(self._dispatchable_or_stopping)
                if self._stopping and self._current is None:
                    return
                if self._current is None:
                    continue
                job = self._jobs[self._current]
                if job.job_id not in self._activated:
                    continue
                job.started_at = int(time.time() * 1000)
            started = time.monotonic()
            try:
                result = self._runner(
                    job,
                    job.cancellation,
                    lambda name, data, current=job: self._runner_event(current, name, data),
                )
                if job.task_profile is not None:
                    from oryn_protocol.task_contracts import validate_result

                    try:
                        validate_result(job.task_profile, result, job.input)
                    except ValueError as exc:
                        raise JobExecutionError(
                            "TASK_OUTPUT_INVALID", "The result violates the task contract."
                        ) from exc
                with self._condition:
                    if job.cancellation.is_set():
                        self._terminal_locked(
                            job,
                            "cancelled",
                            "job.cancelled",
                            {"reason": job.cancellation_reason or "website"},
                        )
                    else:
                        job.result = result
                        self._terminal_locked(
                            job,
                            "completed",
                            "job.completed",
                            {
                                "result": result,
                                "execution": {
                                    **job.execution,
                                    **(
                                        {"fallbacks": list(job.runtime_fallbacks)}
                                        if job.runtime_fallbacks
                                        else {}
                                    ),
                                    "durationMs": int((time.monotonic() - started) * 1000),
                                },
                            },
                        )
            except Exception as exc:
                with self._condition:
                    if job.cancellation.is_set():
                        self._terminal_locked(
                            job,
                            "cancelled",
                            "job.cancelled",
                            {"reason": job.cancellation_reason or "website"},
                        )
                    else:
                        code = (
                            exc.code
                            if isinstance(exc, JobExecutionError)
                            else "JOB_EXECUTION_FAILED"
                        )
                        job.error = protocol_error(
                            code,
                            "The job failed during local execution.",
                            category="job",
                        )
                        self._terminal_locked(job, "failed", "job.failed", {"error": job.error})
            finally:
                with self._condition:
                    if self._current == job.job_id:
                        self._current = None
                    self._promote_next_locked()
                    self._sync_activity_locked()
                    self._condition.notify_all()

    def _runner_event(self, job: EngineJob, event: str, data: dict[str, Any]) -> None:
        if event not in {"job.progress", "job.output.delta", "job.preview", "job.state"}:
            raise JobExecutionError("JOB_EVENT_INVALID", "The runner emitted an invalid event.")
        with self._condition:
            if job.state in TERMINAL_STATES or job.cancellation.is_set():
                return
            if event == "job.state":
                state = data.get("state")
                if state not in {"loading", "running", "streaming", "saving"}:
                    raise JobExecutionError("JOB_STATE_INVALID", "The runner state is invalid.")
                job.state = str(state)
            self._emit_locked(job, event, data)

    def _promote_next_locked(self) -> None:
        while self._queue:
            next_id = self._queue.popleft()
            candidate = self._jobs[next_id]
            if candidate.state != "queued":
                continue
            candidate.state = "running"
            candidate.queue = None
            self._current = next_id
            if next_id in self._activated:
                self._emit_locked(candidate, "job.state", {"state": "running"})
            break
        self._recompute_queue_locked()

    def _recompute_queue_locked(self) -> None:
        for index, job_id in enumerate(self._queue):
            job = self._jobs[job_id]
            current = self._queue_status(index)
            if job.queue != current:
                job.queue = current
                if job_id in self._activated:
                    self._emit_locked(job, "job.queue.updated", current)

    def _emit_locked(self, job: EngineJob, event: str, data: dict[str, Any]) -> None:
        if job.state in TERMINAL_STATES and event not in {
            "job.completed",
            "job.failed",
            "job.cancelled",
        }:
            return
        job.sequence += 1
        if self._event_sink is not None:
            self._event_sink(
                event,
                {
                    "jobId": job.job_id,
                    "sessionId": job.session_id,
                    "sequence": job.sequence,
                    **data,
                },
            )

    def _terminal_locked(
        self, job: EngineJob, state: str, event: str, data: dict[str, Any]
    ) -> None:
        if job.state in TERMINAL_STATES:
            return
        job.state = state
        job.completed_at = int(time.time() * 1000)
        job.queue = None
        self._emit_locked(job, event, data)
        self._prune_terminal_locked()

    def _prune_terminal_locked(self) -> None:
        now = int(time.time() * 1000)
        terminal = sorted(
            (job for job in self._jobs.values() if job.state in TERMINAL_STATES),
            key=lambda job: job.completed_at or 0,
        )
        excess = max(0, len(terminal) - self._max_terminal_jobs)
        for index, job in enumerate(terminal):
            expired = (
                job.completed_at is not None
                and now - job.completed_at > self._terminal_retention_ms
            )
            if index < excess or expired:
                self._jobs.pop(job.job_id, None)
                self._activated.discard(job.job_id)

    def _reference(self, job: EngineJob) -> dict[str, Any]:
        return {
            "jobId": job.job_id,
            "sessionId": job.session_id,
            "state": job.state,
            **({"queue": job.queue} if job.queue is not None else {}),
        }

    def _status(self, job: EngineJob) -> dict[str, Any]:
        return {
            **self._reference(job),
            "sequence": job.sequence,
            **({"result": job.result} if job.result is not None else {}),
            **({"error": job.error} if job.error is not None else {}),
        }

    def _require(self, job_id: str) -> EngineJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobManagerError("JOB_NOT_FOUND", "The job does not exist.")
        return job

    @staticmethod
    def _queue_status(index: int) -> dict[str, Any]:
        return {
            "lane": "generation",
            "position": index + 1,
            "jobsAhead": index,
            "reason": "generation-slot",
        }

    def _sync_activity_locked(self) -> None:
        if self._activity_sink is not None:
            running = 1 if self._current is not None else 0
            queued = sum(self._jobs[job_id].state == "queued" for job_id in self._queue)
            self._activity_sink(running, queued)
        if self._model_activity_sink is not None:
            models: dict[str, dict[str, str]] = {}
            for job in self._jobs.values():
                models[json.dumps(job.model, sort_keys=True)] = job.model
            for model_key, model in models.items():
                relevant = [
                    job
                    for job in self._jobs.values()
                    if json.dumps(job.model, sort_keys=True) == model_key
                ]
                running_jobs = sum(
                    job.state in {"loading", "running", "streaming", "saving"} for job in relevant
                )
                queued_jobs = sum(job.state == "queued" for job in relevant)
                ranks = [
                    int(job.queue["position"])
                    for job in relevant
                    if job.state == "queued" and job.queue is not None
                ]
                self._model_activity_sink(
                    model,
                    running_jobs,
                    queued_jobs,
                    min(ranks) if ranks else None,
                )

    def _dispatchable_or_stopping(self) -> bool:
        return self._stopping or (self._current is not None and self._current in self._activated)


def _invocation_digest(
    session_id: str, input_value: object, input_artifacts: tuple[object, ...]
) -> str:
    try:
        encoded = json.dumps(
            [session_id, input_value, input_artifacts],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise JobManagerError("JOB_INPUT_INVALID", "The invocation input is invalid.") from exc
    return hashlib.sha256(encoded).hexdigest()


def _unsupported_runner(job: EngineJob, cancellation: threading.Event, emit: RunnerEvent) -> object:
    del job, cancellation, emit
    raise JobExecutionError("RUNNER_NOT_AVAILABLE", "No execution runner is available.")
