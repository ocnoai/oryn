from __future__ import annotations

import itertools

import pytest

from oryn_engine.runtime_planning import (
    RuntimeCandidate,
    RuntimeIdentity,
    RuntimePlanError,
    RuntimePlanner,
    RuntimePolicy,
)

GIB = 1024**3


def candidate(
    name: str,
    *,
    backend: str = "cpu",
    memory: int = GIB,
    performance: int = 50,
    quality: int = 50,
    trusted: bool = True,
    equivalence: str = "exact",
) -> RuntimeCandidate:
    return RuntimeCandidate(
        identity=RuntimeIdentity(
            "oryn.runtime.pytorch",
            "1.0.0",
            f"oryn.adapter.{name}",
            "1.0.0",
            f"variant-{name}",
            backend,
            "fp16",
            None,
        ),
        logical_model_id="huggingface:org/model@" + "a" * 40,
        equivalence_class=equivalence,
        task_contract="chat-v1",
        estimated_memory_bytes=memory,
        download_bytes=0,
        conformance="passed",
        installed=False,
        resident=False,
        runtime_available=True,
        trusted=trusted,
        abi_compatible=True,
        task_compatible=True,
        files_complete=True,
        access_granted=True,
        license_allowed=True,
        security_allowed=True,
        quality_score=quality,
        performance_score=performance,
        reliability_score=80,
    )


def plan(values: list[RuntimeCandidate], policy: RuntimePolicy = RuntimePolicy.BALANCED):
    return RuntimePlanner().plan(
        values,
        policy=policy,
        available_memory_bytes=16 * GIB,
        preferred_backends=("mps", "cpu"),
    )


def test_ranking_is_deterministic_for_every_input_order() -> None:
    values = [
        candidate("fast", backend="mps", performance=95),
        candidate("small", memory=GIB // 2),
        candidate("quality", quality=100),
    ]
    winners = {
        plan(list(permutation)).selected.identity.candidate_id
        for permutation in itertools.permutations(values)
    }
    assert len(winners) == 1


def test_no_blocked_candidate_can_win_regardless_of_soft_score() -> None:
    blocked = candidate("blocked", performance=100, quality=100, trusted=False)
    selected = plan([blocked, candidate("safe")]).selected
    assert selected.identity.adapter_id == "oryn.adapter.safe"


def test_memory_and_security_are_hard_gates_for_every_policy() -> None:
    for policy in RuntimePolicy:
        selected = plan(
            [candidate("huge", memory=32 * GIB, quality=100), candidate("safe")], policy
        ).selected
        assert selected.identity.adapter_id == "oryn.adapter.safe"


def test_removing_winner_selects_next_equivalent_candidate() -> None:
    values = [candidate("metal", backend="mps"), candidate("cpu")]
    first = plan(values)
    second = plan([item for item in values if item != first.selected])
    assert second.selected == first.fallbacks[0]


def test_fallback_never_crosses_equivalence_class() -> None:
    result = plan([candidate("exact"), candidate("substitute", equivalence="other-model")])
    assert result.fallbacks == ()


def test_all_rejected_returns_stable_resource_error() -> None:
    with pytest.raises(RuntimePlanError) as error:
        RuntimePlanner().plan(
            [candidate("huge", memory=32 * GIB)],
            available_memory_bytes=16 * GIB,
            preferred_backends=("cpu",),
        )
    assert error.value.code == "RUNTIME_RESOURCES_INSUFFICIENT"


def test_candidate_and_plan_ids_are_stable() -> None:
    first = plan([candidate("stable")])
    second = plan([candidate("stable")])
    assert first.plan_id == second.plan_id
    assert first.selected.identity.candidate_id == second.selected.identity.candidate_id


def test_adding_a_strictly_worse_candidate_does_not_change_the_winner() -> None:
    winner = candidate("winner", backend="mps", memory=GIB, performance=90, quality=90)
    worse = candidate("worse", backend="cpu", memory=2 * GIB, performance=10, quality=10)
    assert plan([winner]).selected == plan([winner, worse]).selected


def test_each_policy_uses_its_declared_priority_without_bypassing_gates() -> None:
    fast = candidate("fast", performance=100, quality=20, memory=3 * GIB)
    quality = candidate("quality", performance=20, quality=100, memory=3 * GIB)
    small = candidate("small", performance=20, quality=20, memory=GIB // 2)
    values = [fast, quality, small]
    assert plan(values, RuntimePolicy.PERFORMANCE).selected.identity.adapter_id.endswith("fast")
    assert plan(values, RuntimePolicy.QUALITY).selected.identity.adapter_id.endswith("quality")
    assert plan(values, RuntimePolicy.MEMORY_SAVER).selected.identity.adapter_id.endswith("small")
    assert plan(values, RuntimePolicy.BALANCED).selected in values


def test_exact_model_authority_is_preserved_across_runtime_fallbacks() -> None:
    result = plan([candidate("metal", backend="mps"), candidate("cpu")])
    assert result.fallbacks
    assert all(
        item.logical_model_id == result.selected.logical_model_id for item in result.fallbacks
    )


def test_user_override_selects_only_an_admitted_candidate() -> None:
    automatic = candidate("automatic", backend="mps", performance=100)
    requested = candidate("requested", backend="cpu", performance=20)
    result = RuntimePlanner().plan(
        [automatic, requested],
        available_memory_bytes=16 * GIB,
        preferred_backends=("mps", "cpu"),
        preferred_candidate_id=requested.identity.candidate_id,
    )
    assert result.selected == requested
    assert result.trusted_summary()["reasonCodes"][0] == "USER_OVERRIDE"


def test_user_override_cannot_bypass_security_or_memory_gates() -> None:
    safe = candidate("safe")
    blocked = candidate("blocked", trusted=False, performance=100)
    with pytest.raises(RuntimePlanError) as failure:
        RuntimePlanner().plan(
            [safe, blocked],
            available_memory_bytes=16 * GIB,
            preferred_backends=("cpu",),
            preferred_candidate_id=blocked.identity.candidate_id,
        )
    assert failure.value.code == "RUNTIME_OVERRIDE_UNAVAILABLE"


@pytest.mark.parametrize(
    ("preferred_backends", "winner_backend"),
    [
        (("mps", "mlx", "cpu"), "mps"),
        (("cuda", "cpu"), "cuda"),
        (("cpu",), "cpu"),
    ],
)
def test_hardware_backend_matrices_are_ranked_without_crossing_hard_gates(
    preferred_backends: tuple[str, ...], winner_backend: str
) -> None:
    values = [
        candidate("mps", backend="mps"),
        candidate("mlx", backend="mlx"),
        candidate("cuda", backend="cuda"),
        candidate("cpu", backend="cpu"),
    ]
    result = RuntimePlanner().plan(
        values,
        available_memory_bytes=16 * GIB,
        preferred_backends=preferred_backends,
    )
    assert result.selected.identity.backend == winner_backend
    assert all(
        evaluation.admitted == (evaluation.candidate.identity.backend in preferred_backends)
        for evaluation in result.evaluations
    )
