from __future__ import annotations

import gc
import importlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from oryn_engine.models import ModelStore, ModelStoreError
from oryn_engine.runtime.resources import MemoryPressure, ResourceManager

ResidencyReason = Literal[
    "explicit-load",
    "job-required",
    "explicit-unload",
    "memory-pressure",
    "memory-budget-change",
    "shutdown",
    "load-failed",
]
ResidencyEventSink = Callable[[str, dict[str, Any]], None]
ModelLoader = Callable[[dict[str, Any]], Any]
ModelUnloader = Callable[[Any], None]


@dataclass
class ResidentModel:
    installation_id: str
    model: dict[str, str]
    estimated_bytes: int
    handle: Any
    state: str = "loaded"
    active_sessions: int = 0
    running_jobs: int = 0
    queued_jobs: int = 0
    queue_rank: int | None = None
    last_used_at: int = field(default_factory=lambda: int(time.time() * 1000))
    unload_when_idle: bool = False


class ResidencyManager:
    """Owns ephemeral model residency independently from installation and jobs."""

    def __init__(
        self,
        models: ModelStore,
        resources: ResourceManager,
        event_sink: ResidencyEventSink | None = None,
        loader: ModelLoader | None = None,
        unloader: ModelUnloader | None = None,
        cache_cleanup: Callable[[], None] | None = None,
    ) -> None:
        self.models = models
        self.resources = resources
        self._event_sink = event_sink
        self._loader = loader or self._default_loader
        self._unloader = unloader or (lambda _handle: None)
        self._cache_cleanup = cache_cleanup or self._backend_cache_cleanup
        self._resident: dict[str, ResidentModel] = {}
        self._usage: dict[str, dict[str, int | None]] = {}
        self._transitioning: set[str] = set()
        self._lock = threading.RLock()
        self._baseline_process_bytes = int(
            self.resources.snapshot().get("processMemory", {}).get("residentBytes", 0)
        )
        self.models.set_runtime_state_provider(self.runtime_state)
        self.resources.set_pressure_listener(self.handle_pressure)

    def set_event_sink(self, sink: ResidencyEventSink) -> None:
        self._event_sink = sink

    def load(
        self,
        operation_id: str,
        model: dict[str, Any],
        *,
        reason: ResidencyReason = "explicit-load",
    ) -> dict[str, Any]:
        self.models.validate_operation_id(operation_id)
        record = self.models.installed_record(model)
        key = self.models.model_key(record["model"])
        with self._lock:
            existing = self._resident.get(key)
            if existing and existing.state == "loaded":
                existing.last_used_at = int(time.time() * 1000)
                return self._result(operation_id, existing)
            if key in self._transitioning:
                raise ModelStoreError(
                    "MODEL_RESIDENCY_BUSY", "The model residency is already changing."
                )
            self._transitioning.add(key)

        estimated = int(record.get("estimatedResidentBytes", record["logicalSizeBytes"]))
        self._emit(record["model"], "loading", reason, estimated)
        try:
            self._admit(key, estimated)
            handle = self._loader(record)
            resident = ResidentModel(
                installation_id=record["installationId"],
                model=record["model"],
                estimated_bytes=estimated,
                handle=handle,
            )
            with self._lock:
                usage = self._usage.get(key, {})
                resident.active_sessions = int(usage.get("activeSessions") or 0)
                resident.running_jobs = int(usage.get("runningJobs") or 0)
                resident.queued_jobs = int(usage.get("queuedJobs") or 0)
                resident.queue_rank = usage.get("queueRank")
                self._resident[key] = resident
            self._sync_resources()
            self._emit(record["model"], "loaded", reason, estimated)
            self._inventory_changed()
            return self._result(operation_id, resident)
        except Exception as exc:
            self._emit(record["model"], "error", "load-failed", estimated)
            if isinstance(exc, ModelStoreError):
                raise
            raise ModelStoreError(
                "MODEL_LOAD_FAILED", "The installed model could not be loaded."
            ) from exc
        finally:
            with self._lock:
                self._transitioning.discard(key)

    def unload(
        self,
        operation_id: str,
        model: dict[str, Any],
        *,
        mode: str = "fail-if-busy",
        reason: ResidencyReason = "explicit-unload",
    ) -> dict[str, Any]:
        self.models.validate_operation_id(operation_id)
        exact = self.models.installed_record(model)["model"]
        key = self.models.model_key(exact)
        with self._lock:
            resident = self._resident.get(key)
            if resident is None:
                return {"operationId": operation_id, "model": exact, "state": "unloaded"}
            if resident.running_jobs:
                if mode == "when-idle":
                    resident.unload_when_idle = True
                    return {"operationId": operation_id, "model": exact, "state": "deferred"}
                raise ModelStoreError("MODEL_BUSY", "The model has active running work.")
        self._evict(key, reason)
        return {"operationId": operation_id, "model": exact, "state": "unloaded"}

    def set_usage(
        self,
        model: dict[str, Any],
        *,
        active_sessions: int | None = None,
        running_jobs: int | None = None,
        queued_jobs: int | None = None,
        queue_rank: int | None = None,
    ) -> None:
        key = self.models.model_key(model)
        should_unload = False
        with self._lock:
            resident = self._resident.get(key)
            usage = self._usage.setdefault(
                key,
                {
                    "activeSessions": 0,
                    "runningJobs": 0,
                    "queuedJobs": 0,
                    "queueRank": None,
                },
            )
            if active_sessions is not None:
                usage["activeSessions"] = max(0, active_sessions)
                if resident:
                    resident.active_sessions = max(0, active_sessions)
            if running_jobs is not None:
                usage["runningJobs"] = max(0, running_jobs)
                if resident:
                    resident.running_jobs = max(0, running_jobs)
            if queued_jobs is not None:
                usage["queuedJobs"] = max(0, queued_jobs)
                if resident:
                    resident.queued_jobs = max(0, queued_jobs)
            usage["queueRank"] = queue_rank
            if resident:
                resident.queue_rank = queue_rank
                resident.last_used_at = int(time.time() * 1000)
                should_unload = resident.running_jobs == 0 and resident.unload_when_idle
        if should_unload:
            self._evict(key, "explicit-unload")
        elif (
            resident
            and running_jobs == 0
            and self.resources.snapshot()["memoryPressure"] != "normal"
        ):
            self.handle_pressure(
                self.resources.snapshot()["memoryPressure"], self.resources.snapshot()
            )
        self._complete_pending_removal(model)

    def remove_installation(
        self,
        operation_id: str,
        installation_id: str,
        *,
        mode: str = "fail-if-busy",
    ) -> dict[str, Any]:
        if mode not in {"fail-if-busy", "when-idle"}:
            raise ModelStoreError("MODEL_REFERENCE_INVALID", "Removal mode is invalid.")
        record = self.models.record_by_installation(installation_id)
        preview = self.models.remove_preview(installation_id)
        if not preview["canRemoveNow"]:
            if mode == "fail-if-busy":
                raise ModelStoreError(
                    "MODEL_BUSY",
                    "The model has active sessions or work.",
                    details={"blockers": preview["blockers"]},
                )
            return self.models.mark_removing(operation_id, installation_id)
        self.models.mark_removing(operation_id, installation_id)
        key = self.models.model_key(record["model"])
        with self._lock:
            resident = self._resident.get(key)
        if resident is not None:
            self._evict(key, "explicit-unload")
        return self.models.complete_remove(installation_id)

    def _complete_pending_removal(self, model: dict[str, Any]) -> None:
        pending = self.models.pending_removal(model)
        if pending is None:
            return
        _operation_id, installation_id = pending
        runtime = self.runtime_state(model)
        if any(
            int(runtime.get(field, 0)) > 0
            for field in ("activeSessions", "runningJobs", "queuedJobs")
        ):
            return
        key = self.models.model_key(model)
        with self._lock:
            resident = self._resident.get(key)
        if resident is not None:
            self._evict(key, "explicit-unload")
        self.models.complete_remove(installation_id)

    def runtime_state(self, model: dict[str, Any]) -> dict[str, Any]:
        key = self.models.model_key(model)
        with self._lock:
            resident = self._resident.get(key)
            if resident is None:
                usage = self._usage.get(key, {})
                return {
                    "residencyState": "loading" if key in self._transitioning else "unloaded",
                    "activeSessions": int(usage.get("activeSessions") or 0),
                    "runningJobs": int(usage.get("runningJobs") or 0),
                    "queuedJobs": int(usage.get("queuedJobs") or 0),
                }
            return {
                "residencyState": resident.state,
                "activeSessions": resident.active_sessions,
                "runningJobs": resident.running_jobs,
                "queuedJobs": resident.queued_jobs,
                "lastUsedAt": resident.last_used_at,
            }

    def handle(self, model: dict[str, Any]) -> Any:
        key = self.models.model_key(model)
        with self._lock:
            resident = self._resident.get(key)
            if resident is None or resident.state != "loaded":
                raise ModelStoreError("MODEL_NOT_LOADED", "The model is not resident.")
            resident.last_used_at = int(time.time() * 1000)
            return resident.handle

    def handle_pressure(self, pressure: MemoryPressure, snapshot: dict[str, Any]) -> None:
        if pressure == "normal":
            return
        while True:
            with self._lock:
                candidates = [item for item in self._resident.values() if not item.running_jobs]
                if not candidates:
                    return
                candidate = min(candidates, key=self._eviction_score)
                key = self.models.model_key(candidate.model)
                resident_total = sum(item.estimated_bytes for item in self._resident.values())
                budget = snapshot["localAI"].get("memoryBudgetBytes")
                available = snapshot.get("systemMemory", {}).get("availableBytes")
                total = snapshot.get("systemMemory", {}).get("totalBytes")
                reached_target = (
                    pressure == "high"
                    and (budget is None or resident_total <= int(budget * 0.75))
                    and (not total or not available or available / total >= 0.2)
                )
                if reached_target:
                    return
            self._evict(key, "memory-pressure")
            if pressure == "high":
                return

    def shutdown(self) -> None:
        with self._lock:
            keys = [
                self.models.model_key(item.model)
                for item in self._resident.values()
                if not item.running_jobs
            ]
        for key in keys:
            self._evict(key, "shutdown")

    def _admit(self, requested_key: str, estimated: int) -> None:
        snapshot = self.resources.snapshot()
        pressure = snapshot["memoryPressure"]
        if pressure == "critical":
            self.handle_pressure(pressure, snapshot)
        while True:
            with self._lock:
                current = sum(item.estimated_bytes for item in self._resident.values())
                budget = snapshot["localAI"].get("memoryBudgetBytes")
                available = snapshot.get("systemMemory", {}).get("availableBytes")
                process_resident = snapshot.get("processMemory", {}).get("residentBytes", 0)
                non_model_overhead = max(
                    self._baseline_process_bytes,
                    process_resident - current,
                    0,
                )
                fits_budget = budget is None or non_model_overhead + current + estimated <= budget
                # `estimated` is already a peak figure (resident weights plus declared/calculated
                # working memory), while psutil.available includes safely reclaimable macOS pages.
                # Applying another percentage haircut double-counts headroom and blocks large but
                # admissible unified-memory models. Native pressure remains an independent hard
                # gate.
                fits_available = available is None or estimated <= max(0, int(available))
                if fits_budget and fits_available and pressure != "critical":
                    return
                candidates = [
                    item
                    for key, item in self._resident.items()
                    if key != requested_key and not item.running_jobs
                ]
                if not candidates:
                    available_for_model = None if available is None else max(0, int(available))
                    raise ModelStoreError(
                        "MODEL_MEMORY_INSUFFICIENT",
                        "The model cannot be admitted under the current memory policy.",
                        retryable=True,
                        details={
                            "requiredBytes": estimated,
                            "availableBytes": available,
                            "availableForModelBytes": available_for_model,
                            "memoryBudgetBytes": budget,
                            "currentResidentModelBytes": current,
                            "processOverheadBytes": non_model_overhead,
                            "memoryPressure": pressure,
                        },
                    )
                victim = min(candidates, key=self._eviction_score)
                victim_key = self.models.model_key(victim.model)
            self._evict(victim_key, "memory-pressure")
            snapshot = self.resources.snapshot()
            pressure = snapshot["memoryPressure"]

    @staticmethod
    def _eviction_score(item: ResidentModel) -> tuple[int, int, int, int]:
        queue_preference = 1 if item.queue_rank is not None else 0
        session_preference = 1 if item.active_sessions else 0
        return (queue_preference, session_preference, item.last_used_at, -item.estimated_bytes)

    def _evict(self, key: str, reason: ResidencyReason) -> None:
        with self._lock:
            resident = self._resident.get(key)
            if resident is None:
                return
            if resident.running_jobs:
                raise ModelStoreError("MODEL_BUSY", "The running model cannot be evicted.")
            resident.state = "unloading"
        self._emit(resident.model, "unloading", reason, resident.estimated_bytes)
        self._unloader(resident.handle)
        resident.handle = None
        self._cache_cleanup()
        with self._lock:
            self._resident.pop(key, None)
        self._sync_resources()
        self._emit(resident.model, "unloaded", reason)
        self._inventory_changed()

    def _sync_resources(self) -> None:
        with self._lock:
            total = sum(item.estimated_bytes for item in self._resident.values())
        self.resources.set_resident_model_bytes(total)

    def _emit(
        self,
        model: dict[str, str],
        state: str,
        reason: ResidencyReason,
        estimated: int | None = None,
    ) -> None:
        data: dict[str, Any] = {"model": model, "state": state, "reason": reason}
        if estimated is not None:
            data["estimatedResidentBytes"] = estimated
        if self._event_sink:
            self._event_sink("model.residency.changed", data)

    def _inventory_changed(self) -> None:
        if self._event_sink:
            self._event_sink("model.inventory.changed", {"reason": "reconciled"})

    @staticmethod
    def _result(operation_id: str, resident: ResidentModel) -> dict[str, Any]:
        return {
            "operationId": operation_id,
            "model": resident.model,
            "state": "loaded",
            "estimatedResidentBytes": resident.estimated_bytes,
        }

    @staticmethod
    def _default_loader(record: dict[str, Any]) -> dict[str, Any]:
        # The default loader returns an opaque installed-model handle.
        return {"installationId": record["installationId"]}

    @staticmethod
    def _backend_cache_cleanup() -> None:
        gc.collect()
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            return
        cuda = getattr(torch, "cuda", None)
        if (
            cuda is not None
            and callable(getattr(cuda, "is_available", None))
            and cuda.is_available()
            and callable(getattr(cuda, "empty_cache", None))
        ):
            cuda.empty_cache()
        mps = getattr(torch, "mps", None)
        backend = getattr(getattr(torch, "backends", None), "mps", None)
        if (
            mps is not None
            and backend is not None
            and callable(getattr(backend, "is_available", None))
            and backend.is_available()
            and callable(getattr(mps, "empty_cache", None))
        ):
            mps.empty_cache()
