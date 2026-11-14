"""Engine-owned routing. Repository metadata never supplies policy or performance evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path

from oryn_engine.runtime_planning import (
    RuntimeCandidate,
    RuntimePlan,
    RuntimePlanError,
    RuntimePlanner,
    RuntimePolicy,
)

GENERATED_PACKS = {
    "qwen-tts.custom-voice-12hz": "oryn.qwen-tts-pytorch",
    "oryn.generated.transformers-tasks": "oryn.transformers-pytorch",
    "oryn.generated.mlx-chat": "oryn.mlx-lm",
    "oryn.generated.mlx-vad": "oryn.mlx-lm",
    "oryn.generated.diffusers": "oryn.diffusers-pytorch",
    "oryn.generated.gguf-chat": "oryn.llama-cpp",
}

MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
MAX_EVIDENCE_RECORDS = 512
MAX_EVIDENCE_AGE_SECONDS = 7 * 24 * 3600


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def generated_pack_id(variant_id: object) -> str | None:
    import re

    if not isinstance(variant_id, str):
        return None
    if re.fullmatch(r"oryn\.onnx\.[a-f0-9]{24}", variant_id):
        return "oryn.onnx-runtime"
    return GENERATED_PACKS.get(variant_id)


@lru_cache(maxsize=1)
def runtime_build_digest() -> str:
    root = os.environ.get("ORYN_ENGINE_ROOT")
    manifest = Path(root) / "release-manifest.json" if root else None
    if (
        manifest is not None
        and manifest.is_file()
        and manifest.stat().st_size <= MAX_EVIDENCE_BYTES
    ):
        return hashlib.sha256(manifest.read_bytes()).hexdigest()
    package = Path(__file__).parent
    return canonical_digest(
        [
            (str(path.relative_to(package)), hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted(package.rglob("*.py"))
        ]
    )


@lru_cache(maxsize=1)
def device_class() -> str:
    # Model/chip/RAM only: no serial number, hostname, user identity, or prompts.
    values = [platform.machine(), platform.processor()]
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.model", "machdep.cpu.brand_string", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            values.append(result.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return canonical_digest(values)


@dataclass(frozen=True)
class RoutingContext:
    catalog_digest: str
    device: str
    system: str
    task_settings_digest: str

    @classmethod
    def current(cls, catalog_digest: str, settings: object) -> RoutingContext:
        return cls(
            canonical_digest([catalog_digest, runtime_build_digest()]),
            device_class(),
            f"{platform.system()}:{platform.release()}:{platform.mac_ver()[0]}",
            canonical_digest(settings),
        )

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True)
class PerformanceEvidence:
    key: str
    captured_at: float
    expires_at: float
    tokens_per_second: float
    first_token_ms: float
    peak_memory_bytes: int
    sample_count: int
    metric: str = "decode-tokens-per-second"
    metric_value: float | None = None
    phase: str = "warm-inference"

    def valid(self, now: float) -> bool:
        numbers = (self.captured_at, self.expires_at, self.tokens_per_second, self.first_token_ms)
        return (
            self.metric in {"decode-tokens-per-second", "ttft-ms", "latency-ms", "real-time-factor"}
            and self.phase in {"warm-inference", "cold-load"}
            and (self.metric == "decode-tokens-per-second" or self.metric_value is not None)
            and (
                self.metric_value is None
                or isinstance(self.metric_value, (int, float))
                and not isinstance(self.metric_value, bool)
                and math.isfinite(self.metric_value)
                and self.metric_value > 0
            )
            and len(self.key) == 64
            and all(c in "0123456789abcdef" for c in self.key)
            and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                for x in numbers
            )
            and 0 <= self.captured_at <= now < self.expires_at
            and 0 < self.expires_at - self.captured_at <= MAX_EVIDENCE_AGE_SECONDS
            and (
                0 < self.tokens_per_second <= 100_000
                if self.metric == "decode-tokens-per-second"
                else 0 <= self.tokens_per_second <= 100_000
            )
            and 0 <= self.first_token_ms <= 3_600_000
            and isinstance(self.peak_memory_bytes, int)
            and not isinstance(self.peak_memory_bytes, bool)
            and 0 < self.peak_memory_bytes <= 2**63 - 1
            and isinstance(self.sample_count, int)
            and not isinstance(self.sample_count, bool)
            and 2 <= self.sample_count <= 10_000
        )


class RoutingEvidenceStore:
    """Bounded local measurements; never an authority for certification or admission."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @staticmethod
    def key(candidate: RuntimeCandidate, context: RoutingContext) -> str:
        return canonical_digest(
            {
                "model": candidate.logical_model_id,
                "candidate": candidate.identity.candidate_id,
                "filesAndBuild": candidate.integrity_digest,
                "task": candidate.task_contract,
                "context": context.digest,
            }
        )

    def read(self, *, now: float | None = None) -> dict[str, PerformanceEvidence]:
        captured = time.time() if now is None else now
        try:
            if self.path.is_symlink() or self.path.stat().st_size > MAX_EVIDENCE_BYTES:
                return {}
            raw = json.loads(self.path.read_text())
            if not isinstance(raw, dict) or set(raw) != {"schemaVersion", "records"}:
                return {}
            records = raw["records"]
            if (
                raw["schemaVersion"] != 1
                or not isinstance(records, list)
                or len(records) > MAX_EVIDENCE_RECORDS
            ):
                return {}
            result: dict[str, PerformanceEvidence] = {}
            for value in records:
                item = PerformanceEvidence(**value)
                if item.valid(captured):
                    if item.key in result:
                        return {}
                    result[item.key] = item
            return result
        except (OSError, ValueError, TypeError, AttributeError):
            return {}

    def record(self, evidence: PerformanceEvidence, *, now: float | None = None) -> None:
        captured = time.time() if now is None else now
        if not evidence.valid(captured) or self.path.is_symlink():
            raise ValueError("Routing measurement is invalid.")
        records = self.read(now=captured)
        records[evidence.key] = evidence
        ordered = sorted(records.values(), key=lambda x: (x.captured_at, x.key))[
            -MAX_EVIDENCE_RECORDS:
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        if temporary.is_symlink():
            raise ValueError("Routing evidence staging is invalid.")
        temporary.write_text(
            json.dumps(
                {"schemaVersion": 1, "records": [asdict(item) for item in ordered]},
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        temporary.replace(self.path)


class RuntimeRoutingService:
    """One internal route selector; no inference, downloads, or permission authority."""

    def __init__(self, evidence: RoutingEvidenceStore | None = None) -> None:
        self.evidence = evidence
        self.planner = RuntimePlanner()
        self._plans: dict[str, RuntimePlan] = {}
        self._failures: dict[str, tuple[int, float]] = {}
        self._execution_keys: dict[str, dict[str, str]] = {}

    def track_execution(self, record: dict) -> None:
        """Bind restored installed plans for failure tracking; this grants no admission."""
        plan = record.get("runtimePlan", {})
        plan_id = plan.get("planId")
        if not isinstance(plan_id, str):
            return
        scope = canonical_digest(
            {
                "model": record.get("model"),
                "files": record.get("files"),
                "pack": record.get("runtimePack"),
                "catalog": record.get("capabilityRegistryDigest"),
                "build": runtime_build_digest(),
            }
        )
        self._execution_keys[plan_id] = {
            candidate["candidateId"]: canonical_digest([scope, candidate])
            for candidate in [plan.get("selected"), *plan.get("fallbacks", [])]
            if isinstance(candidate, dict) and isinstance(candidate.get("candidateId"), str)
        }
        while len(self._execution_keys) > 128:
            self._execution_keys.pop(next(iter(self._execution_keys)))

    def _tracked_key(self, plan_id: str | None, candidate_id: str) -> str | None:
        key = self._execution_keys.get(plan_id or "", {}).get(candidate_id)
        if key is not None:
            return key
        plan = self._plans.get(plan_id or "")
        if plan is not None:
            return next(
                (
                    self._failure_key(item.candidate)
                    for item in plan.evaluations
                    if item.candidate.identity.candidate_id == candidate_id
                ),
                None,
            )
        return None

    @staticmethod
    def _failure_key(candidate: RuntimeCandidate) -> str:
        return canonical_digest(
            [
                candidate.logical_model_id,
                candidate.identity.candidate_id,
                candidate.integrity_digest,
            ]
        )

    def record_failure(self, plan_id: str | None, candidate_id: str) -> None:
        key = self._tracked_key(plan_id, candidate_id)
        if key is None:
            return
        count, captured = self._failures.get(key, (0, 0))
        now = time.monotonic()
        self._failures[key] = (count + 1 if now - captured < 3600 else 1, now)
        while len(self._failures) > MAX_EVIDENCE_RECORDS:
            self._failures.pop(next(iter(self._failures)))

    def is_quarantined(self, plan_id: str | None, candidate_id: str) -> bool:
        key = self._tracked_key(plan_id, candidate_id)
        count, captured = self._failures.get(key or "", (0, 0))
        return count >= 3 and time.monotonic() - captured < 3600

    def _quarantined(self, candidate: RuntimeCandidate) -> bool:
        count, captured = self._failures.get(self._failure_key(candidate), (0, 0))
        return count >= 3 and time.monotonic() - captured < 3600

    def measurement(self, candidate: RuntimeCandidate, context: RoutingContext) -> dict | None:
        record = (
            self.evidence.read().get(RoutingEvidenceStore.key(candidate, context))
            if self.evidence
            else None
        )
        if record is None or not self._metric_applies(record, candidate):
            return None
        return {
            "metric": record.metric,
            "value": record.metric_value
            if record.metric_value is not None
            else record.tokens_per_second,
            "phase": record.phase,
            "comparisonKey": canonical_digest(
                [
                    context.device,
                    context.system,
                    context.task_settings_digest,
                    candidate.task_contract,
                    record.metric,
                    record.phase,
                ]
            ),
            "measuredAt": record.captured_at * 1000,
            "expiresAt": record.expires_at * 1000,
            "sampleCount": record.sample_count,
        }

    @staticmethod
    def _metric_applies(record: PerformanceEvidence, candidate: RuntimeCandidate) -> bool:
        if record.metric in {"decode-tokens-per-second", "ttft-ms"}:
            return candidate.task_contract == "chat-v1"
        if record.metric == "real-time-factor":
            return candidate.task_contract in {"transcription-v1", "captions-v1", "subtitles-v1"}
        return True

    def plan(
        self,
        candidates: list[RuntimeCandidate],
        *,
        context: RoutingContext,
        available_memory_bytes: int,
        available_backends: tuple[str, ...],
        policy: RuntimePolicy = RuntimePolicy.BALANCED,
        preferred_foundation: str | None = None,
        selected_candidate: str | None = None,
        now: float | None = None,
        recommendation_scores: dict[str, int] | None = None,
        recommendation_digest: str = "",
    ) -> RuntimePlan:
        records = self.evidence.read(now=now) if self.evidence else {}
        measurements = {
            item.identity.candidate_id: records.get(RoutingEvidenceStore.key(item, context))
            for item in candidates
        }
        for item in candidates:
            measured = measurements[item.identity.candidate_id]
            if measured and not self._metric_applies(measured, item):
                measurements[item.identity.candidate_id] = None
        # Comparing a measured candidate against invented defaults gives a false winner.
        # Compare measurements only when every available route has comparable evidence.
        eligible = [
            item
            for item in candidates
            if self.planner._evaluate(
                item,
                policy=policy,
                available_memory_bytes=available_memory_bytes,
                backend_rank={b: 1 for b in available_backends},
            ).admitted
        ]
        complete = bool(eligible) and all(measurements[x.identity.candidate_id] for x in eligible)
        complete = (
            complete
            and len(
                {
                    (
                        measurements[x.identity.candidate_id].metric,
                        measurements[x.identity.candidate_id].phase,
                    )
                    for x in eligible
                    if measurements[x.identity.candidate_id]
                }
            )
            == 1
        )

        def performance(e):
            value = e.metric_value if e.metric_value is not None else e.tokens_per_second
            return value if e.metric == "decode-tokens-per-second" else 1 / value

        maximum = max((performance(x) for x in measurements.values() if x), default=1)
        prepared = []
        for item in candidates:
            measured = measurements[item.identity.candidate_id] if complete else None
            prepared.append(
                replace(
                    item,
                    performance_score=max(1, round(100 * performance(measured) / maximum))
                    if measured
                    else 50,
                    evidence_source="measured" if measured else "default",
                    evidence_digest=canonical_digest(asdict(measured))
                    if measured
                    else recommendation_digest,
                    # Measurements can increase estimates; they cannot lower admission gates.
                    estimated_memory_bytes=max(
                        item.estimated_memory_bytes, measured.peak_memory_bytes
                    )
                    if measured
                    else item.estimated_memory_bytes,
                    preference_score=(
                        10000 if item.identity.foundation_id == preferred_foundation else 0
                    )
                    + (
                        max(1, round(100 * performance(measured) / maximum))
                        if measured
                        and policy in {RuntimePolicy.BALANCED, RuntimePolicy.PERFORMANCE}
                        else 0
                        if complete
                        else (recommendation_scores or {}).get(item.identity.candidate_id, 0)
                    ),
                    quarantined=item.quarantined or self._quarantined(item),
                )
            )
        plan = self.planner.plan(
            prepared,
            policy=policy,
            available_memory_bytes=available_memory_bytes,
            preferred_backends=available_backends,
            preferred_candidate_id=selected_candidate,
            context_digest=context.digest,
        )
        self._plans[plan.plan_id] = plan
        while len(self._plans) > 128:
            self._plans.pop(next(iter(self._plans)))
        return plan

    @staticmethod
    def validate_scope(candidates: list[RuntimeCandidate], model: str, task: str) -> None:
        if any(item.logical_model_id != model or item.task_contract != task for item in candidates):
            raise RuntimePlanError("RUNTIME_IDENTITY_MISMATCH", ())
