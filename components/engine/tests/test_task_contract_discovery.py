from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from oryn_protocol.task_contracts import (
    TASK_CONTRACTS,
    make_profile,
    requirement_failures,
    validate_profile,
)
from test_models import FakeHub, model
from test_providers import Checkpoint, setup_store

from oryn_engine.jobs import JobExecutionError, JobManager, JobManagerError
from oryn_engine.models import ModelStore, ModelStoreError
from oryn_engine.provider_discovery import (
    record_for_offer,
)
from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

PACKS = {p.pack_id for p in builtin_registry().packs}


def test_every_declared_runner_contract_has_shared_definition():
    for pack in builtin_registry().packs:
        for runner in pack.runners:
            assert set(runner.task_contracts) <= TASK_CONTRACTS.keys()


def test_all_seven_backends_are_asked_even_after_a_valid_offer(tmp_path, monkeypatch):
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    original = store._inspect_backend
    asked = []

    def inspect(*args, **kwargs):
        asked.append(kwargs["backend"])
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "_inspect_backend", inspect)
    result = store.inspect(model())
    assert result["executable"]
    assert set(asked) == PACKS
    validate_profile(result["taskProfile"])


@pytest.mark.parametrize("pack_id", ["oryn.ace-step-mlx", "oryn.kokoro-onnx"])
def test_manifest_backends_return_common_offers(tmp_path, pack_id):
    manifest = next(m for m in builtin_registry().compatibility if m.pack_id == pack_id)
    names = [p.replace("*", "fixture") for p in manifest.required_files]
    hub = FakeHub(
        config={"architectures": list(manifest.architectures)},
        files=[(name, 1, "a" * 64) for name in names],
        sha=manifest.revision,
        pipeline_tag=manifest.tasks[0],
    )
    store = ModelStore(
        tmp_path,
        hub=hub,
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 128 * 1024**3),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda pack: None if pack.provider else object()
    )
    result = store.inspect(model(manifest.repository, manifest.revision))
    assert result["executable"], result
    assert result["runtimePack"]["packId"] == pack_id
    assert result["taskProfile"]["contract"]["id"] == manifest.task_contract
    assert store._execution_offers[store.model_key(result["model"])]
    validate_profile(result["taskProfile"])


@pytest.mark.parametrize(
    "task,contract,architecture",
    [
        ("text-generation", "chat-v1", "Qwen2ForCausalLM"),
        ("feature-extraction", "embeddings-v1", "BertModel"),
        ("text-classification", "text-classification-v1", "BertForSequenceClassification"),
        ("forced-alignment", "forced-alignment-v1", "Qwen3ASRForTokenClassification"),
        ("automatic-speech-recognition", "transcription-v1", "WhisperForConditionalGeneration"),
        ("text-to-audio", "text-to-music-v1", "MusicgenForConditionalGeneration"),
    ],
)
def test_in_process_tasks_have_common_offers(tmp_path, task, contract, architecture):
    store = ModelStore(
        tmp_path,
        hub=FakeHub(config={"architectures": [architecture]}, pipeline_tag=task),
        runtime_available=True,
    )
    result = store.inspect(
        model("org/arbitrary-copy"), {"task": task, "contract": {"id": contract, "revision": 1}}
    )
    assert result["executable"], result
    assert result["taskProfile"]["contract"]["id"] == contract
    assert store._execution_offers[store.model_key(result["model"])]


def test_mlx_and_transformers_offers_are_aggregated(tmp_path):
    hub = FakeHub(
        config={
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "quantization": {"bits": 4, "group_size": 64},
        },
        pipeline_tag="text-generation",
    )
    store = ModelStore(
        tmp_path,
        hub=hub,
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 64 * 1024**3),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda pack: None if pack.provider else object()
    )
    result = store.inspect(model("someone/arbitrary-model"))
    assert result["executable"], result
    offers = store._execution_offers[store.model_key(result["model"])]
    assert any(o.get("runtimePack", {}).get("packId") == "oryn.mlx-lm" for o in offers.values())
    assert result["taskProfile"]["contract"]["id"] == "chat-v1"


def test_requested_task_overrides_hub_hint(tmp_path):
    hub = FakeHub(config={"architectures": ["BertModel"]}, pipeline_tag="text-classification")
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    result = store.inspect(
        model(), {"task": "feature-extraction", "contract": {"id": "embeddings-v1", "revision": 1}}
    )
    assert result["executable"], result
    assert result["taskProfile"]["task"] == "feature-extraction"


def test_provider_crash_cannot_hide_another_offer(tmp_path, monkeypatch):
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    original = store._inspect_backend

    def inspect(*args, **kwargs):
        if kwargs["backend"] == "oryn.mflux-mlx":
            raise RuntimeError("private path must not leak")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "_inspect_backend", inspect)
    result = store.inspect(model())
    assert result["executable"]
    assert result["providerDiagnostics"] == [
        {"providerId": "oryn.mflux-mlx", "code": "PROVIDER_INSPECTION_FAILED"}
    ]


def test_mflux_offer_uses_same_task_contract_as_other_backends(tmp_path):
    checkpoint = Checkpoint()
    store = setup_store(tmp_path, checkpoint)
    result = store.inspect(model(checkpoint.repository, checkpoint.inventory["revision"]))
    assert result["executable"]
    validate_profile(result["taskProfile"])
    assert result["taskProfile"]["contract"]["id"] == "text-to-image-v1"
    with pytest.raises(ModelStoreError, match="requirements are invalid"):
        store.inspect(
            model(checkpoint.repository, checkpoint.inventory["revision"]),
            {
                "contract": {"id": "text-to-image-v1", "revision": 1},
                "parameterValues": {"width": 13},
            },
        )


def test_word_timing_requirements_reject_unknown_and_qwen(tmp_path):
    requirement = {
        "contract": {"id": "transcription-v1", "revision": 1},
        "requiredFeatures": ["wordTimestamps"],
        "parameterValues": {"timestamps": True},
    }
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    result = store.inspect(model(), requirement)
    assert not result["executable"]
    assert requirement_failures(result["taskProfile"], requirement)


def test_session_parameters_are_checked_before_jobs_are_queued():
    p = make_profile("transcription-v1", features={"wordTimestamps": "supported"})
    r = {
        "contract": p["contract"],
        "requiredFeatures": ["wordTimestamps"],
        "parameterValues": {"language": "sk", "timestamps": True},
    }
    session = SimpleNamespace(
        model=model("org/chosen-model", "a" * 40), result={"taskProfile": p}, requirements=r
    )
    manager = JobManager(SimpleNamespace(get=lambda _: session))
    try:
        with pytest.raises(JobManagerError, match="negotiated task"):
            manager.create(
                "job_wrong",
                "session_one",
                {"audioArtifactId": "artifact_" + "a" * 24, "language": "en", "timestamps": True},
            )
        assert not manager._jobs
    finally:
        manager.shutdown()


def test_bound_offer_rejects_changed_profile_or_files(tmp_path):
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    exact = store.install("install_contract", model())["model"]
    record = store.installed_record(exact)
    candidate = record["runtimePlan"]["selected"]
    changed = copy.deepcopy(record)
    changed["executionOffers"][candidate["candidateId"]]["taskProfile"]["parameters"]["timestamps"][
        "default"
    ] = True
    with pytest.raises(JobExecutionError):
        record_for_offer(changed, candidate)

    changed = copy.deepcopy(record)
    changed["files"][0]["blob_id"] = "f" * 64
    with pytest.raises(JobExecutionError):
        record_for_offer(changed, candidate)


def test_install_keeps_its_inspection_context_when_another_inspection_interleaves(
    tmp_path, monkeypatch
):
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    inspect = store.inspect

    def interleaved(model_ref, requirements=None):
        original = inspect(model_ref, requirements)
        competing = inspect(
            model_ref,
            {
                "contract": {"id": "transcription-v1", "revision": 1},
                "requiredFeatures": ["wordTimestamps"],
            },
        )
        assert not competing["executable"]
        return original

    monkeypatch.setattr(store, "inspect", interleaved)
    installed = store.install("install_interleaved", model())
    record = store.installed_record(installed["model"])
    selected = record["runtimePlan"]["selected"]
    assert record_for_offer(record, selected)["taskProfile"] == record["taskProfile"]


def test_attached_contract_and_controls_are_pinned(tmp_path):
    from test_sessions import FakeHub as ChatHub

    from oryn_engine.residency import ResidencyManager
    from oryn_engine.runtime.resources import ResourceManager
    from oryn_engine.sessions import ModelSessionRegistry

    store = ModelStore(tmp_path, hub=ChatHub(), runtime_available=True)
    exact = store.install("install_chat", model("org/chat"))["model"]
    selected = store.installed_record(exact)["runtimePlan"]["selected"]
    store.set_runtime_override(exact, selected["candidateId"])
    store.set_runtime_override(exact, None)
    sessions = ModelSessionRegistry(store, ResidencyManager(store, ResourceManager()))
    requirements = {
        "contract": {"id": "chat-v1", "revision": 1},
        "parameterValues": {"maxTokens": 8},
    }
    result = sessions.attach(
        "session_contract", exact, ensure_loaded=False, requirements=requirements
    )
    assert result["taskProfile"]["contract"] == requirements["contract"]
    requirements["parameterValues"]["maxTokens"] = 9
    assert sessions.get("session_contract").requirements["parameterValues"]["maxTokens"] == 8
    with pytest.raises(ModelStoreError) as conflict:
        sessions.attach("session_contract", exact, ensure_loaded=False, requirements=requirements)
    assert conflict.value.code == "SESSION_ID_CONFLICT"


def test_evicted_inspection_cannot_install_without_its_bound_offers(tmp_path, monkeypatch):
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    inspect = store.inspect

    def expired(model_ref, requirements=None):
        result = inspect(model_ref, requirements)
        store._inspection_contexts.clear()
        return result

    monkeypatch.setattr(store, "inspect", expired)
    with pytest.raises(ModelStoreError) as error:
        store.install("install_expired", model())
    assert error.value.code == "RUNTIME_PLAN_INVALID"
    assert not store._records


@pytest.mark.parametrize("valid", [True, False])
def test_negotiated_jobs_apply_defaults_and_validate_result(valid):
    from test_jobs import wait_for

    profile = make_profile("chat-v1")
    session = SimpleNamespace(
        model=model("org/chat", "a" * 40),
        result={"taskProfile": profile},
        requirements={"contract": profile["contract"]},
    )
    seen = []

    def runner(job, cancellation, emit):
        seen.append(job.input)
        return {"kind": "structured", "value": {"text": "Hello" if valid else 123}}

    manager = JobManager(SimpleNamespace(get=lambda _: session), runner=runner)
    try:
        manager.create(
            "job_contract", "session_contract", {"messages": [{"role": "user", "content": "Hello"}]}
        )
        manager.response_written("job_contract")
        wait_for(lambda: manager.status("job_contract")["state"] in {"completed", "failed"})
        assert seen[0]["maxTokens"] == 128
        assert manager.status("job_contract")["state"] == ("completed" if valid else "failed")
    finally:
        manager.shutdown()


def test_diffusers_and_gguf_use_common_profiles(tmp_path):
    from oryn_engine.diffusion_tasks import IMAGE_PIPELINES

    family = "Flux2KleinPipeline"
    config = {
        "_class_name": family,
        **{k: list(v) for k, v in IMAGE_PIPELINES[family].items()},
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "tokenizer": ["transformers", "Qwen2TokenizerFast"],
    }
    files = ["model_index.json", "scheduler/scheduler_config.json", "tokenizer/tokenizer.json"]
    for component in IMAGE_PIPELINES[family]:
        files.extend([f"{component}/config.json", f"{component}/model.safetensors"])
    for name, hub, pack, contract, requirements in [
        (
            "image",
            FakeHub(
                config=config,
                files=[(f, 10, "a" * 64) for f in files],
                pipeline_tag="image-to-image",
            ),
            "oryn.diffusers-pytorch",
            "image-editing-v1",
            {"contract": {"id": "image-editing-v1", "revision": 1}, "task": "image-to-image"},
        ),
        (
            "gguf",
            FakeHub(
                config={
                    "model_type": "qwen3_5",
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                },
                files=[("model-Q4_K_M.gguf", 1000, "a" * 64)],
                pipeline_tag="conversational",
                gguf={"architecture": "qwen35", "context_length": 4096},
            ),
            "oryn.llama-cpp",
            "chat-v1",
            {"contract": {"id": "chat-v1", "revision": 1}},
        ),
    ]:
        store = ModelStore(
            tmp_path / name,
            hub=hub,
            runtime_available=True,
            runtime_planning_state_provider=lambda: ("balanced", 128 * 1024**3),
        )
        store._pack_runtime_locator = SimpleNamespace(
            resolve=lambda p: None if p.provider else object()
        )
        result = store.inspect(model("org/" + name), requirements)
        assert result["executable"], result
        assert result["runtimePack"]["packId"] == pack
        assert result["taskProfile"]["contract"]["id"] == contract
        assert store._execution_offers[store.model_key(result["model"])]


def test_runtime_fallback_cannot_change_negotiated_controls(monkeypatch):
    import threading

    from oryn_engine.runners import RunnerRegistry, RuntimePlanHandle

    primary = make_profile("chat-v1")
    alternate = make_profile(
        "chat-v1", constraints={"maxTokens": {"maximum": 8}}, defaults={"maxTokens": 8}
    )
    runner = RunnerRegistry()
    first_handle, second_handle = object(), object()
    attempts = []

    def execute(job, active, cancellation, emit):
        attempts.append(active)
        raise JobExecutionError("MODEL_LOAD_FAILED", "Injected initialization failure")

    monkeypatch.setattr(runner, "_execute_single", execute)
    monkeypatch.setattr(runner, "_load_single", lambda record: second_handle)
    monkeypatch.setattr(runner, "unload", lambda active: None)
    handle = RuntimePlanHandle(
        first_handle,
        {"candidateId": "primary"},
        [({"candidateId": "alternate"}, {"taskProfile": alternate})],
        active_task_profile=primary,
    )
    job = SimpleNamespace(
        task_profile=primary,
        task_requirements={"contract": primary["contract"]},
        input={"messages": [{"role": "user", "content": "Hello"}], "maxTokens": 16},
        runtime_fallbacks=[],
    )
    with pytest.raises(JobExecutionError) as error:
        runner._execute_runtime_plan(job, handle, threading.Event(), lambda *args: None)
    assert error.value.code == "TASK_CONTRACT_INCOMPATIBLE"
    assert attempts == [first_handle]
