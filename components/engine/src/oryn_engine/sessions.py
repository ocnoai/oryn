from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from typing import Any

from oryn_engine.models import ModelStore, ModelStoreError
from oryn_engine.residency import ResidencyManager


@dataclass(frozen=True)
class EngineModelSession:
    session_id: str
    model: dict[str, str]
    result: dict[str, Any]
    requirements: dict[str, Any] | None = None


class ModelSessionRegistry:
    """Engine-side model binding; browser retains origin/document authority."""

    def __init__(self, models: ModelStore, residency: ResidencyManager) -> None:
        self.models = models
        self.residency = residency
        self._sessions: dict[str, EngineModelSession] = {}
        self._lock = threading.RLock()

    def attach(
        self,
        session_id: str,
        model: dict[str, Any],
        *,
        ensure_loaded: bool = True,
        requirements: dict | None = None,
    ) -> dict[str, Any]:
        self.models.validate_operation_id(session_id)
        from oryn_protocol.task_contracts import requirement_failures, validate_requirements

        from oryn_engine.task_profiles import profile_for_execution

        requirements = copy.deepcopy(requirements or {})
        try:
            validate_requirements(requirements)
        except ValueError as exc:
            raise ModelStoreError(
                "TASK_REQUIREMENTS_INVALID", "Task requirements are invalid."
            ) from exc
        record = self.models.installed_record(model)
        exact = record["model"]
        with self._lock:
            existing = self._sessions.get(session_id)
            if existing:
                if (
                    self.models.model_key(existing.model) != self.models.model_key(exact)
                    or existing.requirements != requirements
                ):
                    raise ModelStoreError(
                        "SESSION_ID_CONFLICT",
                        "The session ID is already bound to a different model.",
                    )
                return existing.result

        profile = self._execution_profile(record)
        task_profile = profile_for_execution(
            record, contract_id=requirements.get("contract", {}).get("id")
        )
        if requirement_failures(task_profile, requirements):
            raise ModelStoreError(
                "TASK_CONTRACT_INCOMPATIBLE",
                "The installed model does not meet the requested task contract.",
            )
        if not profile["executable"]:
            raise ModelStoreError("MODEL_NOT_EXECUTABLE", "The model cannot be attached.")
        residency_state = "unloaded"
        if ensure_loaded:
            self.residency.load(f"load_{session_id}", exact, reason="job-required")
            residency_state = "loaded"
        backend = profile["executableBackends"][0]
        if ensure_loaded:
            backend = str(getattr(self.residency.handle(exact), "backend", backend))
        device_class = backend if backend in {"cpu", "mps", "cuda"} else "other"
        result = {
            "sessionId": session_id,
            "model": exact,
            "runtime": {
                "runner": profile["runner"],
                "backend": backend,
                "deviceClass": device_class,
            },
            "technicalCapabilities": profile["technicalCapabilities"],
            "residencyState": residency_state,
            **({"taskProfile": task_profile} if task_profile else {}),
        }
        session = EngineModelSession(session_id, exact, result, requirements)
        with self._lock:
            conflicting = self._sessions.get(session_id)
            if conflicting:
                if (
                    self.models.model_key(conflicting.model) != self.models.model_key(exact)
                    or conflicting.requirements != requirements
                ):
                    raise ModelStoreError(
                        "SESSION_ID_CONFLICT",
                        "The session ID is already bound to a different model.",
                    )
                return conflicting.result
            self._sessions[session_id] = session
            count = self._session_count(exact)
        self.residency.set_usage(exact, active_sessions=count)
        return result

    def release(self, session_id: str) -> dict[str, bool]:
        self.models.validate_operation_id(session_id)
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return {"released": False}
            count = self._session_count(session.model)
        self.residency.set_usage(session.model, active_sessions=count)
        return {"released": True}

    def get(self, session_id: str) -> EngineModelSession:
        self.models.validate_operation_id(session_id)
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise ModelStoreError("SESSION_NOT_FOUND", "The Model Session does not exist.")
        return session

    def release_all(self) -> None:
        with self._lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self.release(session_id)

    def _session_count(self, model: dict[str, str]) -> int:
        key = self.models.model_key(model)
        return sum(
            self.models.model_key(session.model) == key for session in self._sessions.values()
        )

    def _execution_profile(self, record: dict[str, Any]) -> dict[str, Any]:
        if (
            isinstance(record.get("runner"), str)
            and isinstance(record.get("executableBackends"), list)
            and record["executableBackends"]
            and isinstance(record.get("technicalCapabilities"), dict)
        ):
            return {
                "executable": True,
                "runner": record["runner"],
                "executableBackends": record["executableBackends"],
                "technicalCapabilities": record["technicalCapabilities"],
            }
        return self.models.inspect(record["model"])
