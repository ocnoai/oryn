from dataclasses import replace
from pathlib import Path

import pytest
from test_runtime_planning import GIB, candidate

from oryn_engine.runtime_planning import RuntimePlanError, RuntimePolicy
from oryn_engine.runtime_routing import (
    PerformanceEvidence,
    RoutingContext,
    RoutingEvidenceStore,
    RuntimeRoutingService,
)


def test_non_token_measurements_need_no_invented_throughput():
    observation = PerformanceEvidence(
        "a" * 64, 100, 200, 0, 0, 1024, 2, metric="latency-ms", metric_value=12
    )
    assert observation.valid(150)
    assert not replace(observation, metric="decode-tokens-per-second").valid(150)


def test_failed_conformance_cannot_win_or_be_explicitly_selected():
    service = RuntimeRoutingService()
    failed = replace(candidate("failed"), conformance="failed")
    safe = candidate("safe")
    context = RoutingContext("catalog", "mac", "os", "settings")
    result = service.plan(
        [safe, failed],
        context=context,
        available_memory_bytes=16 * GIB,
        available_backends=("cpu",),
    )
    assert result.selected.identity == safe.identity
    assert (
        "CONFORMANCE_FAILED"
        in next(x for x in result.evaluations if not x.admitted).rejection_reasons
    )
    with pytest.raises(RuntimePlanError, match="RUNTIME_OVERRIDE_UNAVAILABLE"):
        service.plan(
            [safe, failed],
            context=context,
            available_memory_bytes=16 * GIB,
            available_backends=("cpu",),
            selected_candidate=failed.identity.candidate_id,
        )


def test_routing_never_substitutes_logical_models():
    service = RuntimeRoutingService()
    values = [candidate("one"), replace(candidate("two"), logical_model_id="another-model")]
    with pytest.raises(RuntimePlanError, match="RUNTIME_IDENTITY_MISMATCH"):
        service.plan(
            values,
            context=RoutingContext("c", "d", "s", "t"),
            available_memory_bytes=16 * GIB,
            available_backends=("cpu",),
        )


def test_measured_routing_and_stale_evidence(tmp_path: Path):
    store = RoutingEvidenceStore(tmp_path / "measurements.json")
    service = RuntimeRoutingService(store)
    context = RoutingContext("catalog", "mac16gb", "os-build", "chat-8k")
    slow, fast = candidate("slow"), candidate("fast")
    for route, throughput in ((slow, 10), (fast, 50)):
        store.record(
            PerformanceEvidence(store.key(route, context), 100, 200, throughput, 200, GIB, 3),
            now=150,
        )
    result = service.plan(
        [slow, fast],
        context=context,
        available_memory_bytes=16 * GIB,
        available_backends=("cpu",),
        policy=RuntimePolicy.PERFORMANCE,
        now=150,
    )
    assert result.selected.identity == fast.identity
    assert "PERFORMANCE_MEASURED" in result.trusted_summary()["reasonCodes"]
    stale = service.plan(
        [slow, fast],
        context=context,
        available_memory_bytes=16 * GIB,
        available_backends=("cpu",),
        now=201,
    )
    assert "PERFORMANCE_DEFAULT" in stale.trusted_summary()["reasonCodes"]
    assert result.plan_id != stale.plan_id
    changed = replace(context, system="new-os")
    assert store.key(fast, changed) != store.key(fast, context)
    assert store.key(replace(fast, integrity_digest="changed-files"), context) != store.key(
        fast, context
    )


def test_partial_measurements_do_not_create_a_fictitious_comparison(tmp_path: Path):
    store = RoutingEvidenceStore(tmp_path / "measurements.json")
    context = RoutingContext("c", "d", "s", "t")
    first, second = candidate("a"), candidate("b")
    store.record(
        PerformanceEvidence(store.key(first, context), 100, 200, 1000, 100, GIB, 3), now=150
    )
    result = RuntimeRoutingService(store).plan(
        [first, second],
        context=context,
        available_memory_bytes=16 * GIB,
        available_backends=("cpu",),
        now=150,
    )
    assert all(x.candidate.evidence_source == "default" for x in result.evaluations)


def test_corrupt_or_executable_evidence_is_ignored(tmp_path: Path):
    path = tmp_path / "evidence.json"
    path.write_text('{"schemaVersion":1,"records":[{"command":"run-me"}]}')
    assert RoutingEvidenceStore(path).read() == {}


def test_exact_route_has_no_silent_fallback():
    a, b = candidate("a"), candidate("b")
    result = RuntimeRoutingService().plan(
        [a, b],
        context=RoutingContext("c", "d", "s", "t"),
        available_memory_bytes=16 * GIB,
        available_backends=("cpu",),
        selected_candidate=a.identity.candidate_id,
    )
    assert result.selected.identity == a.identity
    assert result.fallbacks == ()


def test_repeated_failures_quarantine_only_the_exact_build_and_variant():
    service = RuntimeRoutingService()
    primary, alternate = candidate("primary"), candidate("alternate")
    context = RoutingContext("catalog", "mac", "os", "settings")
    options = dict(context=context, available_memory_bytes=16 * GIB, available_backends=("cpu",))
    plan = service.plan([primary, alternate], **options)
    failed = plan.selected
    for _ in range(3):
        service.record_failure(plan.plan_id, failed.identity.candidate_id)
    assert service.is_quarantined(plan.plan_id, failed.identity.candidate_id)
    next_plan = service.plan([primary, alternate], **options)
    assert next_plan.selected.identity != failed.identity
    renewed = replace(failed, integrity_digest="new-reviewed-build")
    assert service.plan([renewed], **options).selected == renewed


def test_restored_execution_plan_is_tracked_without_in_memory_planning():
    service = RuntimeRoutingService()
    record = {
        "model": {"repository": "same/model", "revision": "a" * 40},
        "files": [{"name": "model.safetensors", "content_sha256": "1" * 64}],
        "runtimePack": {"manifestDigest": "pack-v1"},
        "runtimePlan": {
            "planId": "restored-plan",
            "selected": {"candidateId": "same-candidate", "backend": "mlx"},
            "fallbacks": [],
        },
    }
    service.track_execution(record)
    for _ in range(3):
        service.record_failure("restored-plan", "same-candidate")
    service.track_execution(record)
    assert service.is_quarantined("restored-plan", "same-candidate")
    record["runtimePack"]["manifestDigest"] = "new-build"
    service.track_execution(record)
    assert not service.is_quarantined("restored-plan", "same-candidate")
