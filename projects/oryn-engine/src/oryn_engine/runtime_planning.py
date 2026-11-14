from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

ROUTING_POLICY_VERSION = "2"


class RuntimePolicy(StrEnum):
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    MEMORY_SAVER = "memory-saver"
    QUALITY = "quality"


@dataclass(frozen=True)
class RuntimeIdentity:
    foundation_id: str
    foundation_version: str
    adapter_id: str
    adapter_version: str
    variant_id: str
    backend: str
    precision: str
    quantization: str | None

    @property
    def candidate_id(self) -> str:
        encoded = json.dumps(
            {
                "adapter": [self.adapter_id, self.adapter_version],
                "backend": self.backend,
                "foundation": [self.foundation_id, self.foundation_version],
                "precision": self.precision,
                "quantization": self.quantization,
                "variant": self.variant_id,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return f"candidate-{hashlib.sha256(encoded).hexdigest()[:24]}"


@dataclass(frozen=True)
class RuntimeCandidate:
    identity: RuntimeIdentity
    logical_model_id: str
    equivalence_class: str
    task_contract: str
    estimated_memory_bytes: int
    download_bytes: int
    conformance: str
    installed: bool
    resident: bool
    runtime_available: bool
    trusted: bool
    abi_compatible: bool
    task_compatible: bool
    files_complete: bool
    access_granted: bool
    license_allowed: bool
    security_allowed: bool
    quality_score: int = 50
    performance_score: int = 50
    reliability_score: int = 50
    evidence_source: str = "default"
    evidence_digest: str = ""
    integrity_digest: str = ""
    preference_score: int = 0
    quarantined: bool = False


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: RuntimeCandidate
    admitted: bool
    rejection_reasons: tuple[str, ...]
    score: tuple[int, ...]
    ranking_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RuntimePlan:
    plan_id: str
    policy: RuntimePolicy
    selected: RuntimeCandidate
    fallbacks: tuple[RuntimeCandidate, ...]
    evaluations: tuple[CandidateEvaluation, ...]
    override_candidate_id: str | None = None

    def trusted_summary(self) -> dict[str, Any]:
        selected = self.selected
        selected_evaluation = next(
            item for item in self.evaluations if item.candidate.identity == selected.identity
        )
        return {
            "planId": self.plan_id,
            "policy": self.policy.value,
            "reasonCodes": (
                (["USER_OVERRIDE"] if self.override_candidate_id else [])
                + list(selected_evaluation.ranking_reasons)
            ),
            "selected": _candidate_summary(selected),
            "fallbacks": [_candidate_summary(item) for item in self.fallbacks],
            "rejected": [
                {
                    "candidateId": item.candidate.identity.candidate_id,
                    "reasonCodes": list(item.rejection_reasons),
                }
                for item in self.evaluations
                if not item.admitted
            ][:16],
        }


class RuntimePlanError(ValueError):
    def __init__(self, code: str, evaluations: tuple[CandidateEvaluation, ...]) -> None:
        super().__init__(code)
        self.code = code
        self.evaluations = evaluations


class RuntimePlanner:
    """Deterministic hard-gate-first selector over Engine-authenticated candidates."""

    def plan(
        self,
        candidates: list[RuntimeCandidate] | tuple[RuntimeCandidate, ...],
        *,
        policy: RuntimePolicy = RuntimePolicy.BALANCED,
        available_memory_bytes: int,
        preferred_backends: tuple[str, ...],
        preferred_candidate_id: str | None = None,
        context_digest: str = "",
    ) -> RuntimePlan:
        # A caller may not turn runtime selection into model or task substitution.
        if (
            len({item.logical_model_id for item in candidates}) > 1
            or len({item.task_contract for item in candidates}) > 1
        ):
            raise RuntimePlanError("RUNTIME_IDENTITY_MISMATCH", ())
        if len({item.identity.candidate_id for item in candidates}) != len(candidates):
            raise RuntimePlanError("RUNTIME_IDENTITY_CONFLICT", ())
        backend_rank = {
            name: len(preferred_backends) - index for index, name in enumerate(preferred_backends)
        }
        evaluations = tuple(
            self._evaluate(
                candidate,
                policy=policy,
                available_memory_bytes=available_memory_bytes,
                backend_rank=backend_rank,
            )
            for candidate in sorted(candidates, key=lambda item: item.identity.candidate_id)
        )
        admitted = sorted(
            (item for item in evaluations if item.admitted),
            key=lambda item: (item.score, item.candidate.identity.candidate_id),
            reverse=True,
        )
        if not admitted:
            code = (
                "RUNTIME_RESOURCES_INSUFFICIENT"
                if evaluations
                and all("MEMORY_INSUFFICIENT" in item.rejection_reasons for item in evaluations)
                else "RUNTIME_PLAN_UNAVAILABLE"
            )
            raise RuntimePlanError(code, evaluations)

        override = next(
            (
                item
                for item in admitted
                if item.candidate.identity.candidate_id == preferred_candidate_id
            ),
            None,
        )
        if preferred_candidate_id is not None and override is None:
            raise RuntimePlanError("RUNTIME_OVERRIDE_UNAVAILABLE", evaluations)
        if override is not None:
            admitted.remove(override)
            admitted.insert(0, override)

        selected = admitted[0].candidate
        fallbacks = (
            tuple(
                item.candidate
                for item in admitted[1:]
                if item.candidate.logical_model_id == selected.logical_model_id
                and item.candidate.equivalence_class == selected.equivalence_class
            )[:8]
            if override is None
            else ()
        )
        plan_key = {
            "candidates": [selected.identity.candidate_id]
            + [item.identity.candidate_id for item in fallbacks],
            "logicalModel": selected.logical_model_id,
            "policy": policy.value,
            "override": override.candidate.identity.candidate_id if override is not None else None,
            "policyVersion": ROUTING_POLICY_VERSION,
            "context": context_digest,
            "inputs": [
                {
                    "candidate": item.candidate.identity.candidate_id,
                    "integrity": item.candidate.integrity_digest,
                    "evidence": item.candidate.evidence_digest,
                    "score": item.score,
                    "rejected": item.rejection_reasons,
                }
                for item in evaluations
            ],
        }
        digest = hashlib.sha256(
            json.dumps(plan_key, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        return RuntimePlan(
            plan_id=f"plan-{digest[:24]}",
            policy=policy,
            selected=selected,
            fallbacks=fallbacks,
            evaluations=evaluations,
            override_candidate_id=(
                override.candidate.identity.candidate_id if override is not None else None
            ),
        )

    @staticmethod
    def _evaluate(
        candidate: RuntimeCandidate,
        *,
        policy: RuntimePolicy,
        available_memory_bytes: int,
        backend_rank: dict[str, int],
    ) -> CandidateEvaluation:
        reasons: list[str] = []
        gates = (
            (not candidate.quarantined, "RUNTIME_QUARANTINED"),
            (candidate.conformance != "failed", "CONFORMANCE_FAILED"),
            (
                candidate.conformance
                in {"passed", "developer-tested", "fixture-only", "family-conformant", "untested"},
                "CONFORMANCE_INVALID",
            ),
            (
                isinstance(candidate.estimated_memory_bytes, int)
                and not isinstance(candidate.estimated_memory_bytes, bool)
                and 0 < candidate.estimated_memory_bytes <= 2**63 - 1
                and isinstance(candidate.download_bytes, int)
                and not isinstance(candidate.download_bytes, bool)
                and 0 <= candidate.download_bytes <= 2**63 - 1,
                "RESOURCE_ESTIMATE_INVALID",
            ),
            (candidate.trusted, "AUTHORITY_UNTRUSTED"),
            (candidate.abi_compatible, "ABI_INCOMPATIBLE"),
            (candidate.task_compatible, "TASK_INCOMPATIBLE"),
            (candidate.files_complete, "MODEL_SUBSET_INCOMPLETE"),
            (candidate.access_granted, "MODEL_ACCESS_RESTRICTED"),
            (candidate.license_allowed, "MODEL_LICENSE_BLOCKED"),
            (candidate.security_allowed, "MODEL_SECURITY_BLOCKED"),
            (candidate.runtime_available, "RUNTIME_DEPENDENCY_UNAVAILABLE"),
            (candidate.identity.backend in backend_rank, "BACKEND_UNAVAILABLE"),
            (
                isinstance(candidate.estimated_memory_bytes, int)
                and candidate.estimated_memory_bytes <= max(0, available_memory_bytes),
                "MEMORY_INSUFFICIENT",
            ),
        )
        reasons.extend(code for allowed, code in gates if not allowed)
        if reasons:
            return CandidateEvaluation(candidate, False, tuple(reasons), (), ())

        conformance = {"passed": 300, "developer-tested": 250, "fixture-only": 100}.get(
            candidate.conformance, 150
        )
        memory_margin = max(0, available_memory_bytes - candidate.estimated_memory_bytes)
        memory_score = min(300, memory_margin // (256 * 1024 * 1024))
        backend_score = backend_rank[candidate.identity.backend] * 25
        locality_score = (80 if candidate.resident else 0) + (40 if candidate.installed else 0)
        download_score = -min(200, candidate.download_bytes // (256 * 1024 * 1024))
        if policy == RuntimePolicy.PERFORMANCE:
            soft = (
                candidate.performance_score * 5,
                conformance,
                candidate.reliability_score * 2,
                backend_score,
                candidate.quality_score,
                memory_score,
                locality_score,
                download_score,
            )
            policy_reason = "POLICY_PERFORMANCE"
        elif policy == RuntimePolicy.MEMORY_SAVER:
            soft = (
                memory_score * 4,
                conformance,
                candidate.reliability_score * 2,
                locality_score,
                backend_score,
                candidate.quality_score,
                download_score,
            )
            policy_reason = "POLICY_MEMORY_SAVER"
        elif policy == RuntimePolicy.QUALITY:
            soft = (
                candidate.quality_score * 5,
                conformance,
                candidate.reliability_score * 2,
                candidate.performance_score,
                backend_score,
                memory_score,
                locality_score,
                download_score,
            )
            policy_reason = "POLICY_QUALITY"
        else:
            soft = (
                conformance,
                candidate.reliability_score * 3,
                memory_score,
                candidate.performance_score * 2,
                backend_score,
                locality_score,
                candidate.quality_score,
                download_score,
            )
            policy_reason = "POLICY_BALANCED"
        soft = (candidate.preference_score, *soft)
        ranking_reasons = [
            policy_reason,
            "PERFORMANCE_MEASURED"
            if candidate.evidence_source == "measured"
            else "PERFORMANCE_DEFAULT",
        ]
        if candidate.preference_score:
            ranking_reasons.append("RUNTIME_SOFT_PREFERENCE")
        if candidate.resident:
            ranking_reasons.append("ALREADY_RESIDENT")
        elif candidate.installed:
            ranking_reasons.append("ALREADY_INSTALLED")
        ranking_reasons.extend(
            ("CONFORMANCE_ACCEPTED", f"BACKEND_{candidate.identity.backend.upper()}")
        )
        return CandidateEvaluation(
            candidate, True, (), tuple(int(item) for item in soft), tuple(ranking_reasons)
        )


def _candidate_summary(candidate: RuntimeCandidate) -> dict[str, Any]:
    identity = candidate.identity
    return {
        "candidateId": identity.candidate_id,
        "foundation": {"id": identity.foundation_id, "version": identity.foundation_version},
        "adapter": {"id": identity.adapter_id, "version": identity.adapter_version},
        "variantId": identity.variant_id,
        "backend": identity.backend,
        "precision": identity.precision,
        **({"quantization": identity.quantization} if identity.quantization else {}),
        "estimatedMemoryBytes": candidate.estimated_memory_bytes,
    }
