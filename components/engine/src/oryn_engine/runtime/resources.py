from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import psutil

from oryn_engine.runtime.hardware import HardwareManager

MemoryPressure = Literal["normal", "high", "critical"]
ResourceEventSink = Callable[[str, dict[str, Any]], None]
PressureListener = Callable[[MemoryPressure, dict[str, Any]], None]


class StalePolicyRevisionError(ValueError):
    pass


@dataclass(frozen=True)
class ResourceSample:
    total_bytes: int
    available_bytes: int
    process_resident_bytes: int
    accelerator: dict[str, Any] | None = None
    process_cpu_percent: float | None = None
    sample_window_ms: int | None = None


@dataclass(frozen=True)
class RuntimePolicy:
    revision: int = 0
    memory_budget_bytes: int | None = None
    battery_mode: str = "normal"
    allow_speculative_preload: bool = False
    runtime_selection_policy: str = "balanced"


class MemoryPressureController:
    def __init__(self, recovery_samples: int = 2) -> None:
        self.state: MemoryPressure = "normal"
        self.recovery_samples = recovery_samples
        self._recovery_count = 0

    def update(
        self,
        sample: ResourceSample,
        memory_budget_bytes: int | None,
        estimated_resident_model_bytes: int,
    ) -> tuple[MemoryPressure, MemoryPressure]:
        previous = self.state
        raw = self._raw_pressure(sample, memory_budget_bytes, estimated_resident_model_bytes)
        if _pressure_rank(raw) > _pressure_rank(self.state):
            self.state = raw
            self._recovery_count = 0
        elif _pressure_rank(raw) < _pressure_rank(self.state):
            if self._safe_to_recover(sample, memory_budget_bytes, estimated_resident_model_bytes):
                self._recovery_count += 1
                if self._recovery_count >= self.recovery_samples:
                    self.state = raw
                    self._recovery_count = 0
            else:
                self._recovery_count = 0
        else:
            self._recovery_count = 0
        return previous, self.state

    @staticmethod
    def _raw_pressure(
        sample: ResourceSample,
        memory_budget_bytes: int | None,
        estimated_resident_model_bytes: int,
    ) -> MemoryPressure:
        available_ratio = (
            sample.available_bytes / sample.total_bytes if sample.total_bytes > 0 else 1.0
        )
        local_usage = max(estimated_resident_model_bytes, sample.process_resident_bytes)
        budget_ratio = local_usage / memory_budget_bytes if memory_budget_bytes else 0.0
        if available_ratio <= 0.08 or budget_ratio >= 1.0:
            return "critical"
        if available_ratio <= 0.15 or budget_ratio >= 0.85:
            return "high"
        return "normal"

    def _safe_to_recover(
        self,
        sample: ResourceSample,
        memory_budget_bytes: int | None,
        estimated_resident_model_bytes: int,
    ) -> bool:
        available_ratio = (
            sample.available_bytes / sample.total_bytes if sample.total_bytes > 0 else 1.0
        )
        local_usage = max(estimated_resident_model_bytes, sample.process_resident_bytes)
        budget_ratio = local_usage / memory_budget_bytes if memory_budget_bytes else 0.0
        if self.state == "critical":
            return available_ratio >= 0.12 and budget_ratio <= 0.9
        return available_ratio >= 0.2 and budget_ratio <= 0.75


class ResourceManager:
    def __init__(
        self,
        event_sink: ResourceEventSink | None = None,
        sample_interval_seconds: float = 1.0,
        hardware: HardwareManager | None = None,
    ) -> None:
        self._event_sink = event_sink
        self._pressure_listener: PressureListener | None = None
        self._sample_interval_seconds = sample_interval_seconds
        self._hardware = hardware or HardwareManager()
        self._controller = MemoryPressureController()
        self._policy = RuntimePolicy()
        self._estimated_resident_model_bytes = 0
        self._generation_running = 0
        self._generation_queued = 0
        self._sequence = 0
        self._active_execution: dict[str, Any] | None = None
        self._throughput: dict[str, Any] | None = None
        self._process = psutil.Process()
        self._last_sample_at = time.monotonic()
        self._last_snapshot: dict[str, Any] | None = None
        self._last_emitted_at = 0.0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def hardware(self) -> HardwareManager:
        return self._hardware

    def set_event_sink(self, sink: ResourceEventSink) -> None:
        self._event_sink = sink

    def set_pressure_listener(self, listener: PressureListener) -> None:
        self._pressure_listener = listener

    @property
    def memory_budget_bytes(self) -> int | None:
        with self._lock:
            return self._policy.memory_budget_bytes

    def set_resident_model_bytes(self, value: int, *, emit: bool = True) -> None:
        with self._lock:
            self._estimated_resident_model_bytes = max(0, value)
        self.capture(emit=emit, notify_pressure=False)

    def set_generation_activity(self, running: int, queued: int) -> None:
        with self._lock:
            self._generation_running = 1 if running else 0
            self._generation_queued = max(0, queued)
        self.capture(emit=True, notify_pressure=False)

    def set_active_execution(
        self,
        *,
        job_id: str | None = None,
        model: str | None = None,
        task: str | None = None,
        estimated_bytes: int | None = None,
    ) -> None:
        with self._lock:
            values = {
                "jobId": job_id,
                "model": model,
                "task": task,
                "estimatedBytes": max(0, estimated_bytes) if estimated_bytes is not None else None,
            }
            self._active_execution = {
                key: value for key, value in values.items() if value is not None
            } or None
        self.capture(emit=True, notify_pressure=False)

    def record_throughput(self, value: float, unit: str, sample_window_ms: int) -> None:
        if unit not in {"tokens/s", "steps/s", "samples/s", "frames/s"}:
            raise ValueError("Unsupported throughput unit.")
        with self._lock:
            self._throughput = {
                "value": max(0.0, float(value)),
                "unit": unit,
                "sampleWindowMs": max(1, min(600_000, int(sample_window_ms))),
            }
        self.capture(emit=True, notify_pressure=False)

    def runtime_planning_state(self) -> tuple[str, int]:
        sample = self._sample()
        with self._lock:
            policy = self._policy.runtime_selection_policy
            budget = self._policy.memory_budget_bytes
            resident = self._estimated_resident_model_bytes
        available = sample.available_bytes
        if budget is not None:
            available = min(available, max(0, budget - resident))
        # psutil.available already includes memory macOS can reclaim without swapping. Candidate
        # estimates separately include resident plus working memory, so another percentage reserve
        # here would count the same headroom twice and reject valid unified-memory workloads.
        return policy, max(0, int(available))

    def generation_allowed(self) -> bool:
        """Return whether a new generation job may start under battery policy."""
        with self._lock:
            mode = self._policy.battery_mode
        if mode != "disable":
            return True
        try:
            battery = psutil.sensors_battery()
        except (AttributeError, OSError):
            # An unavailable battery signal must not disable desktop/headless systems.
            return True
        return battery is None or bool(battery.power_plugged)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.capture(emit=True)
        self._thread = threading.Thread(
            target=self._monitor, name="oryn-memory-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=2)
        self._thread = None

    def apply_policy(self, params: dict[str, Any]) -> dict[str, Any]:
        revision = int(params["policyRevision"])
        with self._lock:
            if revision <= self._policy.revision:
                raise StalePolicyRevisionError("Runtime policy revision is stale.")
            self._policy = RuntimePolicy(
                revision=revision,
                memory_budget_bytes=params.get("memoryBudgetBytes"),
                battery_mode=params.get("batteryMode", "normal"),
                allow_speculative_preload=params.get("allowSpeculativePreload", False),
                runtime_selection_policy=params.get("runtimeSelectionPolicy", "balanced"),
            )
        snapshot = self.capture(emit=True)
        return {"appliedRevision": revision, "resources": snapshot}

    def snapshot(self) -> dict[str, Any]:
        return self.capture(emit=False, notify_pressure=False)

    def capture(
        self,
        sample: ResourceSample | None = None,
        *,
        emit: bool,
        notify_pressure: bool = True,
    ) -> dict[str, Any]:
        current = sample or self._sample()
        with self._lock:
            previous_pressure, pressure = self._controller.update(
                current,
                self._policy.memory_budget_bytes,
                self._estimated_resident_model_bytes,
            )
            self._sequence += 1
            snapshot: dict[str, Any] = {
                "snapshotId": f"resource_{self._sequence}",
                "timestamp": int(time.time() * 1000),
                "memoryPressure": pressure,
                "systemMemory": {
                    "totalBytes": current.total_bytes,
                    "availableBytes": current.available_bytes,
                },
                "processMemory": {"residentBytes": current.process_resident_bytes},
                "supervisorMemory": {"residentBytes": current.process_resident_bytes},
                "localAI": {
                    "estimatedResidentModelBytes": self._estimated_resident_model_bytes,
                },
                "accelerator": current.accelerator or {"backend": "cpu"},
                "generationLane": {
                    "capacity": 1,
                    "running": self._generation_running,
                    "queued": self._generation_queued,
                },
            }
            if current.process_cpu_percent is not None:
                snapshot["processMemory"]["cpuPercent"] = max(
                    0.0, min(10_000.0, current.process_cpu_percent)
                )
            if current.sample_window_ms is not None:
                snapshot["processMemory"]["sampleWindowMs"] = max(
                    1, min(60_000, current.sample_window_ms)
                )
            if self._active_execution:
                snapshot["runningJob"] = dict(self._active_execution)
            if self._throughput:
                snapshot["throughput"] = dict(self._throughput)
            if self._policy.memory_budget_bytes is not None:
                snapshot["localAI"]["memoryBudgetBytes"] = self._policy.memory_budget_bytes
            should_emit = emit and self._material_change(snapshot)
            self._last_snapshot = snapshot
            if should_emit:
                self._last_emitted_at = time.monotonic()
        if should_emit and self._event_sink:
            self._event_sink("runtime.resources.updated", snapshot)
            if previous_pressure != pressure:
                self._event_sink(
                    "runtime.memoryPressure.changed",
                    {
                        "previous": previous_pressure,
                        "current": pressure,
                        "snapshotId": snapshot["snapshotId"],
                    },
                )
        if notify_pressure and self._pressure_listener:
            self._pressure_listener(pressure, snapshot)
        return snapshot

    def _material_change(self, snapshot: dict[str, Any]) -> bool:
        previous = self._last_snapshot
        if previous is None or previous["memoryPressure"] != snapshot["memoryPressure"]:
            return True
        if previous["localAI"].get("memoryBudgetBytes") != snapshot["localAI"].get(
            "memoryBudgetBytes"
        ):
            return True
        if previous["generationLane"] != snapshot["generationLane"]:
            return True
        old_available = previous["systemMemory"]["availableBytes"]
        new_available = snapshot["systemMemory"]["availableBytes"]
        total = max(1, snapshot["systemMemory"]["totalBytes"])
        return abs(new_available - old_available) / total >= 0.05 or (
            time.monotonic() - self._last_emitted_at >= 30
        )

    def _sample(self) -> ResourceSample:
        memory = psutil.virtual_memory()
        current_time = time.monotonic()
        sample_window_ms = max(1, int((current_time - self._last_sample_at) * 1000))
        self._last_sample_at = current_time
        resident = self._process.memory_info().rss
        return ResourceSample(
            total_bytes=int(memory.total),
            available_bytes=int(memory.available),
            process_resident_bytes=int(resident),
            accelerator=self._hardware.accelerator_metrics(),
            process_cpu_percent=float(self._process.cpu_percent(interval=None)),
            sample_window_ms=min(60_000, sample_window_ms),
        )

    def _monitor(self) -> None:
        while not self._stop.wait(self._sample_interval_seconds):
            try:
                self.capture(emit=True)
            except Exception:
                # Sampling failure is non-fatal; the control plane remains responsive.
                continue


def _pressure_rank(pressure: MemoryPressure) -> int:
    return {"normal": 0, "high": 1, "critical": 2}[pressure]
