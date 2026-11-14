from dataclasses import replace

import pytest
from test_models import FakeHub, model
from test_providers import Checkpoint, setup_store
from test_runtime_planning import GIB, candidate

from oryn_engine.model_metadata import validate_model_config
from oryn_engine.models import ModelStore, ModelStoreError
from oryn_engine.runtime_routing import (
    PerformanceEvidence,
    RoutingContext,
    RoutingEvidenceStore,
    RuntimeRoutingService,
)


def test_diffusers_legacy_head_count_validates_the_effective_dimension():
    config = {"_class_name": "UNet2DConditionModel", "num_attention_heads": None}
    validate_model_config(config | {"attention_head_dim": [2, 4]})
    for bad in (None, 0, 513, [2, 513], True):
        with pytest.raises(ValueError):
            validate_model_config(config | {"attention_head_dim": bad})
    with pytest.raises(ValueError):
        validate_model_config({"num_attention_heads": None, "attention_head_dim": 2})


def test_pin_survives_inspection_installation_and_restart(tmp_path):
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    initial = store.inspect(model())
    pinned = {**initial["model"], "files": sorted(f["name"] for f in initial["selectedFiles"])}
    inspected = store.inspect(pinned)
    assert inspected["executable"]
    assert inspected["model"] == pinned
    store.install("install_pinned", pinned)
    restored = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    assert restored.installed_record(pinned)["model"] == pinned
    assert restored.inspect(pinned)["model"] == pinned
    assert store.model_key(pinned) != store.model_key(initial["model"])
    for offer in store._execution_offers[store.model_key(pinned)].values():
        assert offer["model"] == pinned


def test_native_provider_binding_retains_the_exact_file_pin(tmp_path):
    checkpoint = Checkpoint("qwen-image", 4)
    store = setup_store(tmp_path, checkpoint)
    initial = store.inspect(
        {"provider": "huggingface", "repository": checkpoint.repository, "revision": "main"}
    )
    pinned = {**initial["model"], "files": sorted(f["name"] for f in initial["selectedFiles"])}
    profile = store.inspect(pinned)
    assert profile["executable"]
    store.install("install_provider_pin", pinned)
    record = store.installed_record(pinned)
    assert record["providerPlans"]
    for bound in record["providerPlans"].values():
        assert bound["model"] == pinned


@pytest.mark.parametrize("files", [["../weights"], ["/weights"], ["a", "a"], [], ["a//b"]])
def test_bad_file_pin_rejected(tmp_path, files):
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    with pytest.raises(ModelStoreError):
        store.inspect({**model(revision="a" * 40), "files": files})


def test_missing_pin_cannot_silently_select_another_subset(tmp_path):
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    profile = store.inspect({**model(revision="a" * 40), "files": ["missing.gguf"]})
    assert not profile["executable"]
    with pytest.raises(ModelStoreError):
        store.install("install_missing", profile["model"])


def test_config_is_soft_and_measurements_override_only_with_comparable_evidence(tmp_path):
    a, b = candidate("a"), candidate("b")
    store = RoutingEvidenceStore(tmp_path / "measurements.json")
    service = RuntimeRoutingService(store)
    context = RoutingContext("catalog", "device", "os", "workload")
    options = dict(
        context=context,
        available_memory_bytes=16 * GIB,
        available_backends=("cpu",),
        recommendation_scores={a.identity.candidate_id: 10},
        recommendation_digest="f" * 64,
        now=150,
    )
    assert service.plan([a, b], **options).selected.identity == a.identity
    store.record(PerformanceEvidence(store.key(b, context), 100, 200, 99, 1, GIB, 3), now=150)
    assert service.plan([a, b], **options).selected.identity == a.identity
    store.record(PerformanceEvidence(store.key(a, context), 100, 200, 1, 1, GIB, 3), now=150)
    assert service.plan([a, b], **options).selected.identity == b.identity
    blocked = replace(a, task_compatible=False)
    assert service.plan([blocked, b], **options).selected.identity == b.identity
    assert (
        service.plan(
            [a, b], **{**options, "context": replace(context, task_settings_digest="changed")}
        ).selected.identity
        == a.identity
    )


@pytest.mark.parametrize("metric", ["latency-ms", "real-time-factor"])
def test_task_metric_lower_is_better_and_cold_warm_are_incomparable(tmp_path, metric):
    contract = "transcription-v1" if metric == "real-time-factor" else "text-to-image-v1"
    a, b = (replace(candidate(name), task_contract=contract) for name in ("a", "b"))
    store = RoutingEvidenceStore(tmp_path / "measurements.json")
    service = RuntimeRoutingService(store)
    context = RoutingContext("catalog", "device", "os", "task-workload")
    for route, value in [(a, 20), (b, 2)]:
        store.record(
            PerformanceEvidence(store.key(route, context), 100, 200, 1, 1, GIB, 3, metric, value),
            now=150,
        )
    options = dict(
        context=context,
        available_memory_bytes=16 * GIB,
        available_backends=("cpu",),
        now=150,
        recommendation_scores={a.identity.candidate_id: 10},
    )
    assert service.plan([a, b], **options).selected.identity == b.identity
    store.record(
        PerformanceEvidence(store.key(b, context), 100, 200, 1, 1, GIB, 3, metric, 2, "cold-load"),
        now=150,
    )
    assert service.plan([a, b], **options).selected.identity == a.identity


def test_foreign_task_metric_does_not_override_configuration(tmp_path):
    a, b = candidate("a"), candidate("b")
    store = RoutingEvidenceStore(tmp_path / "measurements.json")
    context = RoutingContext("catalog", "device", "os", "chat-workload")
    for route, value in [(a, 20), (b, 2)]:
        store.record(
            PerformanceEvidence(
                store.key(route, context), 100, 200, 1, 1, GIB, 3, "real-time-factor", value
            ),
            now=150,
        )
    plan = RuntimeRoutingService(store).plan(
        [a, b],
        context=context,
        available_memory_bytes=16 * GIB,
        available_backends=("cpu",),
        now=150,
        recommendation_scores={a.identity.candidate_id: 10},
    )
    assert plan.selected.identity == a.identity
