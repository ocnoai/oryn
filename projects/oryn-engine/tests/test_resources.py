from __future__ import annotations

import pytest

from oryn_engine.runtime import hardware as hardware_module
from oryn_engine.runtime.hardware import HardwareManager
from oryn_engine.runtime.resources import (
    MemoryPressureController,
    ResourceManager,
    ResourceSample,
    StalePolicyRevisionError,
)


def sample(available: int, total: int = 1000, process: int = 10) -> ResourceSample:
    return ResourceSample(
        total_bytes=total,
        available_bytes=available,
        process_resident_bytes=process,
        accelerator={"backend": "cpu"},
    )


def test_pressure_escalates_immediately_and_recovers_with_hysteresis() -> None:
    controller = MemoryPressureController(recovery_samples=2)

    assert controller.update(sample(400), None, 0) == ("normal", "normal")
    assert controller.update(sample(140), None, 0) == ("normal", "high")
    assert controller.update(sample(70), None, 0) == ("high", "critical")
    assert controller.update(sample(400), None, 0) == ("critical", "critical")
    assert controller.update(sample(400), None, 0) == ("critical", "normal")


def test_budget_pressure_and_stale_policy_revision() -> None:
    manager = ResourceManager()
    applied = manager.apply_policy({"policyRevision": 2, "memoryBudgetBytes": 512_000_000})
    assert applied["appliedRevision"] == 2

    try:
        manager.apply_policy({"policyRevision": 1, "memoryBudgetBytes": 512_000_000})
    except StalePolicyRevisionError:
        pass
    else:
        raise AssertionError("Expected stale runtime policy to be rejected.")


def test_battery_disable_blocks_only_when_unplugged(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ResourceManager()
    manager.apply_policy({"policyRevision": 1, "batteryMode": "disable"})

    monkeypatch.setattr(
        "oryn_engine.runtime.resources.psutil.sensors_battery",
        lambda: type("Battery", (), {"power_plugged": False})(),
    )
    assert manager.generation_allowed() is False

    monkeypatch.setattr(
        "oryn_engine.runtime.resources.psutil.sensors_battery",
        lambda: type("Battery", (), {"power_plugged": True})(),
    )
    assert manager.generation_allowed() is True


def test_missing_battery_signal_does_not_disable_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ResourceManager()
    manager.apply_policy({"policyRevision": 1, "batteryMode": "disable"})
    monkeypatch.setattr("oryn_engine.runtime.resources.psutil.sensors_battery", lambda: None)
    assert manager.generation_allowed() is True


def test_simulated_external_ram_pressure_emits_resource_and_transition_events() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    manager = ResourceManager(event_sink=lambda name, data: events.append((name, data)))

    normal = manager.capture(sample(500), emit=True)
    high = manager.capture(sample(100), emit=True)
    critical = manager.capture(sample(50), emit=True)

    assert [normal["memoryPressure"], high["memoryPressure"], critical["memoryPressure"]] == [
        "normal",
        "high",
        "critical",
    ]
    assert [name for name, _ in events].count("runtime.resources.updated") == 3
    transitions = [data for name, data in events if name == "runtime.memoryPressure.changed"]
    assert [(item["previous"], item["current"]) for item in transitions] == [
        ("normal", "high"),
        ("high", "critical"),
    ]


def test_apple_silicon_is_unified_and_allocator_bytes_do_not_equal_machine_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hardware_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(HardwareManager, "_torch", staticmethod(lambda: None))
    info = HardwareManager().detect()
    controller = MemoryPressureController()
    mps_sample = ResourceSample(
        total_bytes=1000,
        available_bytes=500,
        process_resident_bytes=10,
        accelerator={"backend": "mps", "allocatedBytes": 900},
    )

    assert info.unified_memory is True
    assert info.preferred_backend == "cpu"
    assert controller.update(mps_sample, None, 0) == ("normal", "normal")


def test_runtime_planning_uses_os_reclaimable_availability_without_a_second_haircut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ResourceManager()
    monkeypatch.setattr(manager, "_sample", lambda: sample(477, total=640))

    policy, available = manager.runtime_planning_state()

    assert policy == "balanced"
    assert available == 477

    manager.set_resident_model_bytes(100, emit=False)
    manager.apply_policy({"policyRevision": 1, "memoryBudgetBytes": 500})
    _, budget_limited = manager.runtime_planning_state()
    assert budget_limited == 400


def test_optional_resource_provenance_is_bounded_and_task_native() -> None:
    manager = ResourceManager()
    manager.set_active_execution(
        job_id="job_resource_test",
        model="org/model",
        task="chat",
        estimated_bytes=1234,
    )
    manager.record_throughput(42.5, "tokens/s", 1_500)

    snapshot = manager.capture(
        ResourceSample(
            total_bytes=10_000,
            available_bytes=8_000,
            process_resident_bytes=512,
            accelerator={"backend": "cpu"},
            process_cpu_percent=12.5,
            sample_window_ms=1_000,
        ),
        emit=False,
    )

    assert snapshot["supervisorMemory"] == {"residentBytes": 512}
    assert snapshot["processMemory"]["cpuPercent"] == 12.5
    assert snapshot["processMemory"]["sampleWindowMs"] == 1_000
    assert snapshot["runningJob"] == {
        "jobId": "job_resource_test",
        "model": "org/model",
        "task": "chat",
        "estimatedBytes": 1234,
    }
    assert snapshot["throughput"] == {
        "value": 42.5,
        "unit": "tokens/s",
        "sampleWindowMs": 1_500,
    }


def test_throughput_rejects_unknown_units() -> None:
    manager = ResourceManager()
    with pytest.raises(ValueError):
        manager.record_throughput(1.0, "percent", 1_000)
