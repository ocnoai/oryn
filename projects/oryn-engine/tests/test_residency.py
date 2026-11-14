from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from oryn_engine.jobs import EngineJob, JobManager
from oryn_engine.models import DownloadProgress, ModelStore, ModelStoreError
from oryn_engine.residency import ResidencyManager
from oryn_engine.runtime.resources import ResourceManager, ResourceSample

SHA = "a" * 40


@pytest.mark.parametrize("available", [False, True])
def test_cleanup_calls_only_available_accelerators(monkeypatch, available):
    calls = []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: available, empty_cache=lambda: calls.append("cuda")
        ),
        mps=SimpleNamespace(empty_cache=lambda: calls.append("mps")),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: available)),
    )
    monkeypatch.setattr("oryn_engine.residency.importlib.import_module", lambda name: torch)
    ResidencyManager._backend_cache_cleanup()
    assert calls == (["cuda", "mps"] if available else [])


class FakeHub:
    def model_info(self, repo_id: str, revision: str, files_metadata: bool = True) -> Any:
        del revision, files_metadata
        suffix = repo_id.split("/")[-1].encode().hex()[:8]
        sha = (suffix + SHA)[:40]
        siblings = [
            SimpleNamespace(
                rfilename="model.safetensors",
                size=100_000_000,
                lfs=None,
            )
        ]
        siblings.extend(
            SimpleNamespace(rfilename=name, size=0, lfs=None)
            for name in ("config.json", "tokenizer.json")
        )
        return SimpleNamespace(
            sha=sha,
            config={"architectures": ["Qwen2ForCausalLM"]},
            pipeline_tag="text-generation",
            siblings=siblings,
        )

    def download(
        self,
        repo_id: str,
        filename: str,
        revision: str,
        cache_dir: str,
        progress: DownloadProgress | None = None,
    ) -> str:
        target = Path(cache_dir) / "snapshots" / revision / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        size = next(
            item.size
            for item in self.model_info(repo_id, revision).siblings
            if item.rfilename == filename
        )
        target.write_bytes(b"x" * size)
        if progress is not None:
            progress(4, 4)
        return str(target)


def requested(name: str) -> dict[str, str]:
    return {"provider": "huggingface", "repository": f"org/{name}", "revision": "main"}


def installed_models(store: ModelStore, count: int) -> list[dict[str, str]]:
    result = []
    for index in range(count):
        installed = store.install(f"op_install_{index}", requested(f"model-{index}"))
        result.append(installed["model"])
    return result


def resource_sample(available: int) -> ResourceSample:
    return ResourceSample(
        total_bytes=1000,
        available_bytes=available,
        process_resident_bytes=0,
        accelerator={"backend": "cpu"},
    )


def test_one_model_loads_and_unload_keeps_installation(tmp_path: Path) -> None:
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    model = installed_models(store, 1)[0]
    unloaded: list[object] = []
    cleanup: list[bool] = []
    manager = ResidencyManager(
        store,
        ResourceManager(),
        loader=lambda record: record["installationId"],
        unloader=unloaded.append,
        cache_cleanup=lambda: cleanup.append(True),
    )

    assert manager.load("op_load_one", model)["state"] == "loaded"
    assert store.inventory()["models"][0]["residencyState"] == "loaded"
    assert manager.unload("op_unload_one", model)["state"] == "unloaded"
    inventory = store.inventory()["models"][0]
    assert inventory["installationState"] == "installed"
    assert inventory["residencyState"] == "unloaded"
    assert unloaded
    assert cleanup == [True]


def test_two_models_coexist_and_third_evicts_idle_under_budget(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    models = installed_models(store, 3)
    resources = ResourceManager()
    baseline = resources.snapshot()["processMemory"]["residentBytes"]
    resources.apply_policy({"policyRevision": 1, "memoryBudgetBytes": baseline + 260_000_000})
    manager = ResidencyManager(store, resources, lambda name, data: events.append((name, data)))

    manager.load("op_load_a", models[0])
    manager.load("op_load_b", models[1])
    assert sum(item["residencyState"] == "loaded" for item in store.inventory()["models"]) == 2

    manager.load("op_load_c", models[2])
    inventory = store.inventory()["models"]
    assert sum(item["residencyState"] == "loaded" for item in inventory) == 2
    assert any(
        name == "model.residency.changed"
        and data["state"] == "unloaded"
        and data["reason"] == "memory-pressure"
        for name, data in events
    )


def test_memory_rejection_reports_safe_actionable_admission_measurements(
    tmp_path: Path,
) -> None:
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    model = installed_models(store, 1)[0]
    resources = ResourceManager()
    baseline = resources.snapshot()["processMemory"]["residentBytes"]
    budget = baseline + 10_000_000
    resources.apply_policy({"policyRevision": 1, "memoryBudgetBytes": budget})
    manager = ResidencyManager(store, resources)

    with pytest.raises(ModelStoreError) as rejected:
        manager.load("op_load_without_headroom", model)

    assert rejected.value.code == "MODEL_MEMORY_INSUFFICIENT"
    assert rejected.value.retryable is True
    details = rejected.value.details
    assert details is not None
    assert details["requiredBytes"] == 125_000_000
    assert details["memoryBudgetBytes"] == budget
    assert details["currentResidentModelBytes"] == 0
    assert details["processOverheadBytes"] >= baseline
    assert details["availableForModelBytes"] == details["availableBytes"]
    assert details["memoryPressure"] in {"normal", "high", "critical"}


def test_reclaimable_available_memory_admits_a_conservatively_estimated_large_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    model = installed_models(store, 1)[0]
    store._records[0]["estimatedResidentBytes"] = 44_600_000_000
    resources = ResourceManager()
    monkeypatch.setattr(
        resources,
        "_sample",
        lambda: ResourceSample(
            total_bytes=64_000_000_000,
            available_bytes=47_700_000_000,
            process_resident_bytes=250_000_000,
            accelerator={"backend": "mps"},
        ),
    )
    loaded: list[str] = []
    manager = ResidencyManager(
        store,
        resources,
        loader=lambda record: loaded.append(record["installationId"]) or "large-model",
    )

    assert manager.load("op_load_large_model", model)["state"] == "loaded"
    assert loaded


def test_available_memory_remains_a_hard_admission_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    model = installed_models(store, 1)[0]
    store._records[0]["estimatedResidentBytes"] = 48_000_000_000
    resources = ResourceManager()
    monkeypatch.setattr(
        resources,
        "_sample",
        lambda: ResourceSample(
            total_bytes=64_000_000_000,
            available_bytes=47_700_000_000,
            process_resident_bytes=250_000_000,
            accelerator={"backend": "mps"},
        ),
    )

    with pytest.raises(ModelStoreError) as rejected:
        ResidencyManager(store, resources).load("op_reject_oversized_model", model)

    assert rejected.value.code == "MODEL_MEMORY_INSUFFICIENT"
    assert rejected.value.details is not None
    assert rejected.value.details["availableForModelBytes"] == 47_700_000_000


def test_external_pressure_evicts_idle_without_a_new_ai_request(tmp_path: Path) -> None:
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    models = installed_models(store, 2)
    resources = ResourceManager()
    manager = ResidencyManager(store, resources)
    manager.load("op_load_a", models[0])
    manager.load("op_load_b", models[1])

    resources.capture(resource_sample(100), emit=True)

    assert sum(item["residencyState"] == "loaded" for item in store.inventory()["models"]) == 1


def test_running_model_is_never_evicted_and_deferred_unload_runs_when_idle(
    tmp_path: Path,
) -> None:
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    models = installed_models(store, 2)
    resources = ResourceManager()
    manager = ResidencyManager(store, resources)
    manager.load("op_load_a", models[0])
    manager.load("op_load_b", models[1])
    manager.set_usage(models[0], running_jobs=1)

    resources.capture(resource_sample(50), emit=True)
    states = {item["model"]["repository"]: item for item in store.inventory()["models"]}
    assert states[models[0]["repository"]]["residencyState"] == "loaded"
    assert states[models[1]["repository"]]["residencyState"] == "unloaded"
    with pytest.raises(ModelStoreError) as error:
        manager.unload("op_unload_busy", models[0])
    assert error.value.code == "MODEL_BUSY"
    assert manager.unload("op_unload_later", models[0], mode="when-idle")["state"] == "deferred"
    manager.set_usage(models[0], running_jobs=0)
    assert store.inventory()["models"][0]["residencyState"] == "unloaded"


def test_busy_removal_can_defer_and_blocks_new_binding_until_idle(tmp_path: Path) -> None:
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    model = installed_models(store, 1)[0]
    manager = ResidencyManager(
        store,
        ResourceManager(),
        loader=lambda record: SimpleNamespace(
            backend="cpu", repository=record["model"]["repository"]
        ),
        unloader=lambda _handle: None,
        cache_cleanup=lambda: None,
    )
    manager.load("load_busy_remove", model)
    manager.set_usage(model, active_sessions=1, running_jobs=1, queued_jobs=1, queue_rank=0)
    installation_id = store.inventory()["models"][0]["installationId"]
    preview = store.remove_preview(installation_id)
    assert preview == {
        "installationId": installation_id,
        "canRemoveNow": False,
        "activeSessions": 1,
        "runningJobs": 1,
        "queuedJobs": 1,
        "reclaimableBytes": 100_000_000,
        "sharedBytesRetained": 0,
        "blockers": ["active-sessions", "running-jobs", "queued-jobs"],
    }
    with pytest.raises(ModelStoreError) as busy:
        manager.remove_installation("remove_busy_now", installation_id)
    assert busy.value.code == "MODEL_BUSY"

    deferred = manager.remove_installation("remove_when_idle", installation_id, mode="when-idle")
    assert deferred["state"] == "deferred"
    assert store.inventory()["models"][0]["installationState"] == "removing"
    with pytest.raises(ModelStoreError) as pending:
        store.installed_record(model)
    assert pending.value.code == "MODEL_REMOVAL_PENDING"

    manager.set_usage(model, active_sessions=0, running_jobs=0, queued_jobs=0, queue_rank=None)
    assert store.inventory() == {"models": []}


def test_loaded_state_resets_after_engine_crash_but_installation_persists(tmp_path: Path) -> None:
    first_store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    model = installed_models(first_store, 1)[0]
    ResidencyManager(first_store, ResourceManager()).load("op_load_before_crash", model)
    assert first_store.inventory()["models"][0]["residencyState"] == "loaded"

    restarted_store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    ResidencyManager(restarted_store, ResourceManager())
    inventory = restarted_store.inventory()["models"][0]
    assert inventory["installationState"] == "installed"
    assert inventory["residencyState"] == "unloaded"


def test_next_queued_model_is_preferred_but_remains_evictable_when_critical(
    tmp_path: Path,
) -> None:
    unloaded: list[str] = []
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    models = installed_models(store, 3)
    resources = ResourceManager()
    manager = ResidencyManager(
        store,
        resources,
        event_sink=lambda name, data: (
            unloaded.append(data["model"]["repository"])
            if name == "model.residency.changed" and data["state"] == "unloaded"
            else None
        ),
    )
    for index, model in enumerate(models):
        manager.load(f"op_load_queue_{index}", model)
    manager.set_usage(models[0], queued_jobs=1, queue_rank=0)

    resources.capture(resource_sample(50), emit=True)

    assert unloaded[-1] == models[0]["repository"]
    assert all(item["residencyState"] == "unloaded" for item in store.inventory()["models"])


@dataclass(frozen=True)
class QueueSession:
    model: dict[str, str]


class QueueSessions:
    def __init__(self, models: dict[str, dict[str, str]]) -> None:
        self.models = models

    def get(self, session_id: str) -> QueueSession:
        return QueueSession(self.models[session_id])


def test_two_model_queue_survives_pressure_eviction_and_reloads_safely(tmp_path: Path) -> None:
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    model_a, model_b = installed_models(store, 2)
    resources = ResourceManager()
    loaded: list[str] = []
    unloaded: list[str] = []

    def load(record: dict[str, Any]) -> SimpleNamespace:
        loaded.append(record["model"]["repository"])
        return SimpleNamespace(backend="cpu", repository=record["model"]["repository"])

    residency = ResidencyManager(
        store,
        resources,
        loader=load,
        unloader=lambda handle: unloaded.append(handle.repository),
        cache_cleanup=lambda: None,
    )
    residency.load("load_model_a", model_a)
    residency.load("load_model_b", model_b)
    assert all(item["residencyState"] == "loaded" for item in store.inventory()["models"])

    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()

    def execute(job: EngineJob, cancellation: threading.Event, emit: Any) -> object:
        del cancellation, emit
        residency.load(f"load_{job.job_id}", job.model, reason="job-required")
        residency.handle(job.model)
        if job.job_id == "job_origin_a":
            first_started.set()
            assert release_first.wait(2)
        else:
            second_started.set()
        return {"kind": "structured", "value": {"model": job.model["repository"]}}

    sessions = QueueSessions({"session_a": model_a, "session_b": model_b})
    jobs = JobManager(
        sessions,
        runner=execute,
        activity_sink=resources.set_generation_activity,
        model_activity_sink=lambda model, running, queued, rank: residency.set_usage(
            model, running_jobs=running, queued_jobs=queued, queue_rank=rank
        ),
    )
    try:
        jobs.create("job_origin_a", "session_a", {"prompt": "first"})
        jobs.response_written("job_origin_a")
        assert first_started.wait(1)
        queued = jobs.create("job_origin_b", "session_b", {"prompt": "second"})
        jobs.response_written("job_origin_b")
        assert queued["state"] == "queued" and queued["queue"]["position"] == 1

        resources.capture(resource_sample(100), emit=True)
        inventory_by_repo = {
            item["model"]["repository"]: item for item in store.inventory()["models"]
        }
        assert inventory_by_repo[model_a["repository"]]["residencyState"] == "loaded"
        assert inventory_by_repo[model_b["repository"]]["residencyState"] == "unloaded"
        assert not second_started.is_set()

        resources.capture(resource_sample(300), emit=True)
        resources.capture(resource_sample(300), emit=True)
        release_first.set()
        assert second_started.wait(1)
        deadline = time.monotonic() + 1
        while jobs.status("job_origin_b")["state"] != "completed":
            assert time.monotonic() < deadline
            time.sleep(0.005)

        assert loaded.count(model_b["repository"]) == 2
        assert model_b["repository"] in unloaded
        assert store.inventory()["models"][1]["residencyState"] == "loaded"
    finally:
        release_first.set()
        jobs.shutdown()
