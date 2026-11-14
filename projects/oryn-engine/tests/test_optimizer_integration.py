"""Routing integration fixtures; these do not qualify native model execution."""

from types import SimpleNamespace

from test_models import FakeHub, model

from oryn_engine.models import ModelStore

CPU = "oryn.transformers-pytorch"
MLX = "oryn.mlx-lm"


def store_with_offers(path):
    store = ModelStore(
        path,
        hub=FakeHub(
            config={
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
            },
            pipeline_tag="text-generation",
        ),
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 128 * 1024**3),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda pack: None if pack.provider else object()
    )
    return store


def hint(*engines):
    return {"snapshotDigest": "f" * 64, "engines": list(engines)}


def test_generated_offer_cannot_override_a_missing_signed_task_tuple(tmp_path):
    config = {
        "model_type": "bert",
        "architectures": ["UnreviewedClassifier"],
        "id2label": {"0": "contradiction", "1": "entailment"},
    }
    hub = FakeHub(config=config, pipeline_tag="zero-shot-classification")
    hub.read_json = lambda *args, **kwargs: config
    store = ModelStore(
        tmp_path,
        hub=hub,
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 128 * 1024**3),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda pack: None if pack.provider else object()
    )
    result = store.inspect(
        model("org/unreviewed"),
        {
            "task": "zero-shot-classification",
            "contract": {"id": "zero-shot-classification-v1", "revision": 1},
        },
        hint(CPU),
    )
    assert not result["executable"]
    assert "RUNTIME_CAPABILITY_UNSUPPORTED" in result["blockingReasons"]


def engine(profile):
    return (profile.get("runtimePack") or {}).get("packId", CPU)


def test_metadata_preference_reaches_installation_and_session(tmp_path):
    from oryn_engine.residency import ResidencyManager
    from oryn_engine.runtime.resources import ResourceManager
    from oryn_engine.sessions import ModelSessionRegistry

    store = store_with_offers(tmp_path)
    first = store.inspect(model("org/chat"), recommendation=hint(CPU, MLX))
    assert engine(first) == CPU
    assert {o["engine"] for o in first["routingFacts"]["offers"]} == {CPU, MLX}
    exact = {**first["model"], "files": sorted(f["name"] for f in first["selectedFiles"])}
    final = store.inspect(exact, recommendation=hint(MLX, CPU))
    assert engine(final) == MLX
    assert final["runtimePlan"]["selected"]["precision"] == "native"
    installed = store.install("install_optimizer", exact, recommendation=hint(MLX, CPU))
    assert installed["model"] == exact
    assert engine(store.installed_record(exact)) == MLX
    sessions = ModelSessionRegistry(store, ResidencyManager(store, ResourceManager()))
    attached = sessions.attach("session_optimizer", exact, ensure_loaded=False)
    assert attached["model"] == exact
    assert attached["runtime"]["backend"] == final["runtimePlan"]["selected"]["backend"]


def test_installed_preference_is_offline_and_prepared_before_attachment(tmp_path):
    store = store_with_offers(tmp_path)
    exact = store.install("install_original", model("org/chat"), recommendation=hint(CPU, MLX))[
        "model"
    ]
    assert engine(store.installed_record(exact)) == CPU
    store._hub.model_info = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("network"))
    planned = store.inspect(exact, recommendation=hint(MLX, CPU))
    assert engine(planned) == MLX
    assert planned["runtimePreparationRequired"]
    assert not planned["installationRequired"]
    # Inspection is read-only; only installation/preparation can bind the new route.
    assert engine(store.installed_record(exact)) == CPU
    store.install("install_rebind", exact, recommendation=hint(MLX, CPU))
    assert engine(store.installed_record(exact)) == MLX
    restored = store_with_offers(tmp_path)
    restored._hub.model_info = store._hub.model_info
    assert engine(restored.inspect(exact, recommendation=hint(CPU, MLX))) == CPU


def test_user_override_and_active_sessions_keep_the_bound_route(tmp_path):
    store = store_with_offers(tmp_path)
    exact = store.install("install_bound", model("org/chat"), recommendation=hint(CPU, MLX))[
        "model"
    ]
    selected = store.installed_record(exact)["runtimePlan"]["selected"]["candidateId"]
    store.set_runtime_override(exact, selected)
    assert engine(store.inspect(exact, recommendation=hint(MLX, CPU))) == CPU
    store.set_runtime_override(exact, None)
    store.set_runtime_state_provider(lambda _: {"activeSessions": 1})
    assert engine(store.inspect(exact, recommendation=hint(MLX, CPU))) == CPU


def test_runtime_update_requires_new_inspection_and_preparation(tmp_path, monkeypatch):
    import pytest

    from oryn_engine.models import ModelStoreError

    store = store_with_offers(tmp_path)
    exact = store.install("install_original", model("org/chat"), recommendation=hint(CPU))["model"]
    monkeypatch.setattr("oryn_engine.models.runtime_build_digest", lambda: "1" * 64)
    monkeypatch.setattr("oryn_engine.runtime_routing.runtime_build_digest", lambda: "1" * 64)
    profile = store.inspect(exact, recommendation=hint(CPU))
    assert profile["executable"] and profile["runtimePreparationRequired"]
    store.set_runtime_state_provider(lambda _: {"activeSessions": 1})
    with pytest.raises(ModelStoreError, match="Release active sessions"):
        store.install("install_active", exact, recommendation=hint(CPU))
    store.set_runtime_state_provider(lambda _: {})
    store.install("install_updated", exact, recommendation=hint(CPU))
    assert store.installed_record(exact)["runtimeBuildDigest"] == "1" * 64
    from oryn_engine.provider_discovery import record_for_offer

    record = store.installed_record(exact)
    assert record_for_offer(record, record["runtimePlan"]["selected"])["model"] == exact


def test_preferences_cannot_override_task_or_memory_gates(tmp_path):
    store = store_with_offers(tmp_path)
    result = store.inspect(
        model("org/chat"),
        {
            "contract": {"id": "chat-v1", "revision": 1},
            "requiredFeatures": ["multimodalConversation"],
        },
        hint(MLX),
    )
    assert not result["executable"]
    store._runtime_planning_state_provider = lambda: ("balanced", 1)
    assert not store.inspect(model("org/chat"), recommendation=hint(MLX))["executable"]


def test_native_mlx_weights_do_not_borrow_torch_support(tmp_path):
    store = store_with_offers(tmp_path)
    store._hub.config["quantization"] = {"bits": 4, "group_size": 64}
    profile = store.inspect(model("org/chat"), recommendation=hint(CPU, MLX))
    assert profile["executable"] and engine(profile) == MLX
    assert {o["engine"] for o in profile["routingFacts"]["offers"]} == {MLX}


def test_generic_gguf_discovery_selects_complete_shards_and_rejects_partial_pin(tmp_path):
    files = [(f"model-Q4_K_M-{i:05d}-of-00002.gguf", 1024, "a" * 64) for i in (1, 2)]
    store = store_with_offers(tmp_path)
    store._hub = FakeHub(
        config={},
        files=files,
        pipeline_tag="text-generation",
        gguf={"architecture": "qwen2", "context_length": 4096},
    )
    profile = store.inspect(model("org/split"))
    assert profile["executable"]
    assert sorted(f["name"] for f in profile["selectedFiles"]) == [f[0] for f in files]
    assert profile["estimatedDiskBytes"] == 2048
    assert not store.inspect({**profile["model"], "files": [files[0][0]]})["executable"]
