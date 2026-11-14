from __future__ import annotations

import logging
import os
import platform
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from oryn_protocol import ProtocolSchemaStore

from oryn_engine import __version__
from oryn_engine.artifacts import (
    ArtifactBindingError,
    ArtifactInputValidator,
    ArtifactOutputStaging,
    protocol_binding,
)
from oryn_engine.jobs import JobManager, JobManagerError
from oryn_engine.models import HuggingFaceClient, ModelStore, ModelStoreError
from oryn_engine.protocol.errors import protocol_error
from oryn_engine.residency import ResidencyManager
from oryn_engine.runners import RunnerRegistry
from oryn_engine.runtime.capabilities import runtime_capabilities
from oryn_engine.runtime.resources import ResourceManager, StalePolicyRevisionError
from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
from oryn_engine.runtime_packs.builtin_pack_catalog import builtin_pack_catalog
from oryn_engine.runtime_packs.lifecycle import PackLifecycleManager
from oryn_engine.runtime_packs.manifests import ManifestError
from oryn_engine.sessions import ModelSessionRegistry

SUPPORTED_VERSIONS = (1,)
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
LOGGER = logging.getLogger("oryn_engine.protocol")


class ProtocolDispatcher:
    def __init__(
        self,
        schemas: ProtocolSchemaStore | None = None,
        resources: ResourceManager | None = None,
        models: ModelStore | None = None,
        hugging_face_token_provider: Callable[[], str | None] | None = None,
    ) -> None:
        self.schemas = schemas or ProtocolSchemaStore()
        self.engine_instance_id = f"engine_{uuid.uuid4().hex}"
        self.max_message_bytes = MAX_MESSAGE_BYTES
        self.handshake_complete = False
        self.shutting_down = False
        self.seen_request_ids: set[str] = set()
        self._state_lock = threading.RLock()
        self._event_lock = threading.RLock()
        self._event_sequence = 0
        self._event_sink: Callable[[dict[str, Any]], None] | None = None
        self.resources = resources or ResourceManager()
        self.resources.set_event_sink(self._resource_event)
        self.models = models
        self._hugging_face_token_provider = hugging_face_token_provider or (lambda: None)
        self.runners = RunnerRegistry()
        self.residency: ResidencyManager | None = None
        self.negotiated_task_contracts: frozenset[str] = frozenset()
        self.sessions: ModelSessionRegistry | None = None
        self.jobs: JobManager | None = None
        self.artifact_inputs: ArtifactInputValidator | None = None
        self.artifact_outputs: ArtifactOutputStaging | None = None
        self.pack_lifecycle: PackLifecycleManager | None = None
        if models is not None:
            self.residency = ResidencyManager(
                models,
                self.resources,
                self._model_event,
                loader=self.runners.load,
                unloader=self.runners.unload,
            )
            self.sessions = ModelSessionRegistry(models, self.residency)

    def set_event_sink(self, sink: Callable[[dict[str, Any]], None]) -> None:
        self._event_sink = sink
        if self.models is not None:
            self.models.set_event_sink(self._model_event)
        if self.residency is not None:
            self.residency.set_event_sink(self._model_event)
        if self.jobs is not None:
            self.jobs.set_event_sink(self._job_event)

    def start_runtime_services(self) -> None:
        self.resources.start()

    def stop_runtime_services(self) -> None:
        if self.jobs is not None:
            self.jobs.shutdown()
        if self.sessions is not None:
            self.sessions.release_all()
        if self.residency is not None:
            self.residency.shutdown()
        self.resources.stop()

    def publish_event(self, name: str, data: dict[str, Any]) -> None:
        # Sequence assignment and publication share one order across emitters.
        with self._event_lock:
            if self._event_sink is not None:
                self._event_sink(self.event(name, data))

    def _resource_event(self, name: str, data: dict[str, Any]) -> None:
        self.publish_event(name, data)

    def _model_event(self, name: str, data: dict[str, Any]) -> None:
        self.publish_event(name, data)

    def _job_event(self, name: str, data: dict[str, Any]) -> None:
        self.publish_event(name, data)

    def response_written(self, method: str, response: dict[str, Any]) -> None:
        if (
            method == "job.create"
            and response.get("ok") is True
            and self.jobs is not None
            and isinstance(response.get("result"), dict)
        ):
            job_id = response["result"].get("jobId")
            if isinstance(job_id, str):
                self.jobs.response_written(job_id)

    def event(self, event: str, data: dict[str, Any]) -> dict[str, Any]:
        with self._event_lock:
            self._event_sequence += 1
            sequence = self._event_sequence
        return {
            "v": 1,
            "type": "event",
            "event": event,
            "seq": sequence,
            "timestamp": int(time.time() * 1000),
            "data": data,
        }

    def _response(
        self,
        request_id: str,
        *,
        result: object | None = None,
        error: dict[str, Any] | None = None,
        negotiated: bool | None = None,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {
            "type": "response",
            "id": request_id,
            "ok": error is None,
        }
        use_version = self.handshake_complete if negotiated is None else negotiated
        if use_version:
            response["v"] = 1
        if error is None:
            response["result"] = result
        else:
            response["error"] = error
        return response

    def _schema_error(self, request_id: str, *, negotiated: bool | None = None) -> dict[str, Any]:
        return self._response(
            request_id,
            error=protocol_error(
                "PROTOCOL_SCHEMA_INVALID",
                "The request payload is invalid.",
            ),
            negotiated=negotiated,
        )

    def dispatch(self, message: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        request_id = message.get("id")
        if not isinstance(request_id, str) or not request_id:
            request_id = "req_invalid_envelope"

        with self._state_lock:
            if not self.handshake_complete:
                return self._dispatch_before_handshake(message, request_id)

            if message.get("method") == "protocol.handshake":
                return (
                    self._response(
                        request_id,
                        error=protocol_error(
                            "PROTOCOL_HANDSHAKE_ALREADY_COMPLETE",
                            "The protocol handshake is already complete.",
                        ),
                    ),
                    False,
                )

            if not self.schemas.is_valid("protocolRequest", message):
                return self._schema_error(request_id), False

            if request_id in self.seen_request_ids:
                return (
                    self._response(
                        request_id,
                        error=protocol_error(
                            "PROTOCOL_DUPLICATE_REQUEST_ID",
                            "The request ID has already been used for this engine instance.",
                        ),
                    ),
                    False,
                )
            self.seen_request_ids.add(request_id)

            if self.shutting_down and message.get("method") != "protocol.ping":
                return (
                    self._response(
                        request_id,
                        error=protocol_error(
                            "PROTOCOL_SHUTTING_DOWN",
                            "The runtime is shutting down.",
                        ),
                    ),
                    False,
                )

        method = message["method"]
        try:
            return self._dispatch_method(method, message, request_id)
        except ModelStoreError as exc:
            category = (
                "resource"
                if exc.code in {"DISK_SPACE_INSUFFICIENT", "MODEL_MEMORY_INSUFFICIENT"}
                else "session"
                if exc.code.startswith("SESSION_")
                else "model"
            )
            return (
                self._response(
                    request_id,
                    error=protocol_error(
                        exc.code,
                        str(exc),
                        retryable=exc.retryable,
                        details=exc.details,
                        category=category,
                    ),
                ),
                False,
            )
        except (JobManagerError, ArtifactBindingError) as exc:
            category = "artifact" if isinstance(exc, ArtifactBindingError) else "job"
            return (
                self._response(
                    request_id,
                    error=protocol_error(
                        exc.code,
                        "The job request could not be accepted.",
                        category=category,
                    ),
                ),
                False,
            )
        except ManifestError as exc:
            return (
                self._response(
                    request_id,
                    error=protocol_error(exc.code, str(exc), category="runtime"),
                ),
                False,
            )
        except Exception:
            diagnostic_id = f"diag_{uuid.uuid4().hex}"
            LOGGER.exception("Protocol request failed (diagnosticId=%s)", diagnostic_id)
            return (
                self._response(
                    request_id,
                    error=protocol_error(
                        "PROTOCOL_INTERNAL_ERROR",
                        "The engine could not complete the request.",
                        diagnostic_id=diagnostic_id,
                    ),
                ),
                False,
            )

    def _require_task_contract(self, model):
        from oryn_engine.shared_tasks import NEGOTIATED_TASK_CONTRACTS, TASK_CONTRACTS

        record = self._model_store().installed_record(model)
        from oryn_engine.diffusion_tasks import MEDIA_CONTRACTS

        contract = record.get("runtimePack", {}).get("taskContract")
        if contract not in MEDIA_CONTRACTS:
            contract = TASK_CONTRACTS.get(record.get("technicalCapabilities", {}).get("task"))
        if (
            contract in NEGOTIATED_TASK_CONTRACTS | MEDIA_CONTRACTS
            and contract not in self.negotiated_task_contracts
        ):
            raise ModelStoreError(
                "TASK_CONTRACT_NOT_NEGOTIATED", "The Browser does not support this task contract."
            )

    def _dispatch_method(
        self,
        method: str,
        message: dict[str, Any],
        request_id: str,
    ) -> tuple[dict[str, Any], bool]:
        if method == "task.contracts.negotiate":
            from oryn_engine.shared_tasks import NEGOTIATED_TASK_CONTRACTS

            if not self.schemas.is_valid("taskContractsRequest", message):
                return self._schema_error(request_id), False
            from oryn_engine.diffusion_tasks import MEDIA_CONTRACTS

            requested = frozenset(message["params"]["contracts"])
            # Negotiation can only add contracts understood by both peers; never revoke a live
            # session's contract or infer permission from a website/model metadata field.
            with self._state_lock:
                self.negotiated_task_contracts |= requested.intersection(
                    NEGOTIATED_TASK_CONTRACTS | MEDIA_CONTRACTS
                )
                negotiated = sorted(self.negotiated_task_contracts)
            return self._response(request_id, result={"contracts": negotiated}), False
        if method == "protocol.ping":
            if not self.schemas.is_valid("pingRequest", message):
                return self._schema_error(request_id), False
            state = "shutting-down" if self.shutting_down else "ready"
            return self._response(
                request_id,
                result={"engineInstanceId": self.engine_instance_id, "state": state},
            ), False

        if method == "runtime.capabilities":
            if not self.schemas.is_valid("runtimeCapabilitiesRequest", message):
                return self._schema_error(request_id), False
            return self._response(
                request_id,
                result=runtime_capabilities(self.engine_instance_id, self.max_message_bytes),
            ), False

        if method == "runtime.resources":
            if not self.schemas.is_valid("runtimeResourcesRequest", message):
                return self._schema_error(request_id), False
            return self._response(request_id, result=self.resources.snapshot()), False

        if method == "runtime.policy.update":
            if not self.schemas.is_valid("runtimePolicyUpdateRequest", message):
                return self._schema_error(request_id), False
            try:
                result = self.resources.apply_policy(message["params"])
            except StalePolicyRevisionError:
                return self._response(
                    request_id,
                    error=protocol_error(
                        "RUNTIME_POLICY_STALE",
                        "The runtime policy revision is stale.",
                        category="resource",
                    ),
                ), False
            return self._response(request_id, result=result), False

        if method == "runtime.pack.inventory":
            if not self.schemas.is_valid("runtimePackInventoryRequest", message):
                return self._schema_error(request_id), False
            return self._response(request_id, result=self._pack_manager().inventory()), False

        if method == "runtime.pack.cleanup":
            if not self.schemas.is_valid("runtimePackCleanupRequest", message):
                return self._schema_error(request_id), False
            return self._response(request_id, result=self._pack_manager().cleanup()), False

        if method == "runtime.shutdown":
            if not self.schemas.is_valid("runtimeShutdownRequest", message):
                return self._schema_error(request_id), False
            with self._state_lock:
                self.shutting_down = True
            return self._response(request_id, result={"accepted": True}), True

        if method in {"model.inspect", "model.source.inspect"}:
            source_method = method == "model.source.inspect"
            schema = "modelSourceInspectRequest" if source_method else "modelInspectRequest"
            if not self.schemas.is_valid(schema, message):
                return self._schema_error(request_id), False
            return self._response(
                request_id,
                result=self._model_store().inspect(
                    message["params"]["model"],
                    message["params"].get("requirements"),
                    **(
                        {"source_context": message["params"]["sourceContext"]}
                        if source_method
                        else {}
                    ),
                    **(
                        {"recommendation": message["params"]["recommendation"]}
                        if "recommendation" in message["params"]
                        else {}
                    ),
                ),
            ), False

        if method == "model.runtime.override":
            if not self.schemas.is_valid("modelRuntimeOverrideRequest", message):
                return self._schema_error(request_id), False
            params = message["params"]
            return self._response(
                request_id,
                result=self._model_store().set_runtime_override(
                    params["model"], params["candidateId"]
                ),
            ), False

        if method == "model.catalog.match":
            if not self.schemas.is_valid("modelCatalogMatchRequest", message):
                return self._schema_error(request_id), False
            params = message["params"]
            return self._response(
                request_id,
                result=self._model_store().catalog(params["task"], params["outputModalities"]),
            ), False

        if method in {"model.install", "model.source.install"}:
            source_method = method == "model.source.install"
            schema = "modelSourceInstallRequest" if source_method else "modelInstallRequest"
            if not self.schemas.is_valid(schema, message):
                return self._schema_error(request_id), False
            params = message["params"]
            return self._response(
                request_id,
                result=self._model_store().install(
                    params["operationId"],
                    params["model"],
                    params.get("requirements"),
                    **({"source_context": params["sourceContext"]} if source_method else {}),
                    **(
                        {"recommendation": params["recommendation"]}
                        if "recommendation" in params
                        else {}
                    ),
                ),
            ), False

        if method == "model.install.cancel":
            if not self.schemas.is_valid("modelInstallCancelRequest", message):
                return self._schema_error(request_id), False
            accepted = self._model_store().cancel_install(message["params"]["operationId"])
            return self._response(request_id, result={"accepted": accepted}), False

        if method == "model.load":
            if not self.schemas.is_valid("modelLoadRequest", message):
                return self._schema_error(request_id), False
            params = message["params"]
            self._require_task_contract(params["model"])
            return self._response(
                request_id,
                result=self._residency_manager().load(
                    params["operationId"], params["model"], reason="explicit-load"
                ),
            ), False

        if method == "model.unload":
            if not self.schemas.is_valid("modelUnloadRequest", message):
                return self._schema_error(request_id), False
            params = message["params"]
            return self._response(
                request_id,
                result=self._residency_manager().unload(
                    params["operationId"],
                    params["model"],
                    mode=params.get("mode", "fail-if-busy"),
                    reason="explicit-unload",
                ),
            ), False

        if method == "model.inventory.list":
            if not self.schemas.is_valid("modelInventoryRequest", message):
                return self._schema_error(request_id), False
            return self._response(request_id, result=self._model_store().inventory()), False

        if method == "model.remove.preview":
            if not self.schemas.is_valid("modelRemovePreviewRequest", message):
                return self._schema_error(request_id), False
            return self._response(
                request_id,
                result=self._model_store().remove_preview(message["params"]["installationId"]),
            ), False

        if method == "model.remove":
            if not self.schemas.is_valid("modelRemoveRequest", message):
                return self._schema_error(request_id), False
            params = message["params"]
            return self._response(
                request_id,
                result=self._residency_manager().remove_installation(
                    params["operationId"],
                    params["installationId"],
                    mode=params["mode"],
                ),
            ), False

        if method == "session.attach":
            if not self.schemas.is_valid("sessionAttachRequest", message):
                return self._schema_error(request_id), False
            params = message["params"]
            self._require_task_contract(params["model"])
            return self._response(
                request_id,
                result=self._session_registry().attach(
                    params["sessionId"],
                    params["model"],
                    ensure_loaded=params.get("ensureLoaded", True),
                    requirements=params.get("requirements"),
                ),
            ), False

        if method == "session.release":
            if not self.schemas.is_valid("sessionReleaseRequest", message):
                return self._schema_error(request_id), False
            return self._response(
                request_id,
                result=self._session_registry().release(message["params"]["sessionId"]),
            ), False

        if method == "job.create":
            if not self.schemas.is_valid("jobCreateRequest", message):
                return self._schema_error(request_id), False
            if not self.resources.generation_allowed():
                raise JobManagerError(
                    "LOCAL_AI_DISABLED_ON_BATTERY",
                    "Local AI generation is disabled while running on battery.",
                )
            params = message["params"]
            from oryn_engine.shared_tasks import TASK_CONTRACTS, validate_task_input

            session = self._session_registry().get(params["sessionId"])
            record = self._model_store().installed_record(session.model)
            task = record.get("technicalCapabilities", {}).get("task")
            if task in TASK_CONTRACTS:
                self._require_task_contract(session.model)
                try:
                    validate_task_input(task, params["input"])
                except ValueError as exc:
                    raise JobManagerError(
                        "JOB_INPUT_INVALID", "The task input is invalid."
                    ) from exc
            from oryn_engine.diffusion_tasks import MEDIA_CONTRACTS, validate_media_input

            contract = record.get("runtimePack", {}).get("taskContract")
            if contract in MEDIA_CONTRACTS:
                self._require_task_contract(session.model)
                try:
                    validate_media_input(contract, params["input"])
                except ValueError as exc:
                    raise JobManagerError(
                        "JOB_INPUT_INVALID", "Image task input is invalid."
                    ) from exc
            bindings = self._validate_artifact_bindings(params.get("inputArtifacts", []))
            return self._response(
                request_id,
                result=self._job_manager().create(
                    params["jobId"], params["sessionId"], params["input"], bindings
                ),
            ), False

        if method == "job.status":
            if not self.schemas.is_valid("jobStatusRequest", message):
                return self._schema_error(request_id), False
            return self._response(
                request_id,
                result=self._job_manager().status(message["params"]["jobId"]),
            ), False

        if method == "job.cancel":
            if not self.schemas.is_valid("jobCancelRequest", message):
                return self._schema_error(request_id), False
            params = message["params"]
            return self._response(
                request_id,
                result=self._job_manager().cancel(params["jobId"], params["reason"]),
            ), False

        return self._response(
            request_id,
            error=protocol_error(
                "PROTOCOL_UNKNOWN_METHOD",
                "The requested protocol method is not supported.",
            ),
        ), False

    def _model_store(self) -> ModelStore:
        if self.models is not None:
            return self.models
        root_value = os.environ.get("ORYN_MODEL_ROOT")
        if not root_value or not Path(root_value).is_absolute():
            raise ModelStoreError(
                "MODEL_STORAGE_UNAVAILABLE",
                "The controlled model storage root is unavailable.",
            )
        self.models = ModelStore(
            Path(root_value),
            event_sink=self._model_event,
            hub=HuggingFaceClient(self._hugging_face_token_provider),
            runtime_planning_state_provider=self.resources.runtime_planning_state,
        )
        self.runners.routing = self.models.runtime_routing
        return self.models

    def _pack_manager(self) -> PackLifecycleManager:
        if self.pack_lifecycle is not None:
            return self.pack_lifecycle
        configured = os.environ.get("ORYN_RUNTIME_PACK_ROOT")
        if configured and Path(configured).is_absolute():
            root = Path(configured).resolve()
        else:
            root = (self._model_store().root.parent / "runtime-packs").resolve()
        machine = platform.machine().lower()
        platform_name = (
            "macos-arm64"
            if sys.platform == "darwin" and machine in {"arm64", "aarch64"}
            else "windows-amd64"
            if sys.platform == "win32" and machine in {"amd64", "x86_64"}
            else "linux-amd64"
            if sys.platform.startswith("linux") and machine in {"amd64", "x86_64"}
            else "unsupported"
        )
        self.pack_lifecycle = PackLifecycleManager(
            root, builtin_pack_catalog(), builtin_registry(), platform=platform_name
        )
        return self.pack_lifecycle

    def _residency_manager(self) -> ResidencyManager:
        if self.residency is None:
            self.residency = ResidencyManager(
                self._model_store(),
                self.resources,
                self._model_event,
                loader=self.runners.load,
                unloader=self.runners.unload,
            )
        return self.residency

    def _session_registry(self) -> ModelSessionRegistry:
        if self.sessions is None:
            residency = self._residency_manager()
            self.sessions = ModelSessionRegistry(self._model_store(), residency)
        return self.sessions

    def _job_manager(self) -> JobManager:
        if self.jobs is None:
            self.jobs = JobManager(
                self._session_registry(),
                self._job_event,
                runner=self._run_job,
                activity_sink=self.resources.set_generation_activity,
                model_activity_sink=lambda model, running, queued, rank: (
                    self._residency_manager().set_usage(
                        model,
                        running_jobs=running,
                        queued_jobs=queued,
                        queue_rank=rank,
                    )
                ),
            )
        return self.jobs

    def _run_job(
        self,
        job: Any,
        cancellation: threading.Event,
        emit: Callable[[str, dict[str, Any]], None],
    ) -> object:
        residency = self._residency_manager()
        residency.load(f"load_{job.job_id}", job.model, reason="job-required")
        self.resources.set_active_execution(
            job_id=job.job_id,
            model=str(job.model.get("repository", "")) or None,
        )
        outputs = self._artifact_output_staging()
        self.runners.outputs = outputs
        try:
            return self.runners.execute(job, residency.handle(job.model), cancellation, emit)
        except Exception:
            outputs.cleanup_job(job.job_id)
            raise
        finally:
            self.resources.set_active_execution()

    def _artifact_output_staging(self) -> ArtifactOutputStaging:
        if self.artifact_outputs is None:
            root_value = os.environ.get("ORYN_ARTIFACT_OUTPUT_ROOT")
            if not root_value or not Path(root_value).is_absolute():
                raise ArtifactBindingError("ARTIFACT_PATH_INVALID")
            self.artifact_outputs = ArtifactOutputStaging(Path(root_value), self.engine_instance_id)
        return self.artifact_outputs

    def _validate_artifact_bindings(self, values: object) -> tuple[dict[str, Any], ...]:
        if values == []:
            return ()
        if self.artifact_inputs is None:
            root_value = os.environ.get("ORYN_ARTIFACT_INPUT_ROOT")
            if not root_value or not Path(root_value).is_absolute():
                raise ArtifactBindingError("ARTIFACT_PATH_INVALID")
            self.artifact_inputs = ArtifactInputValidator(Path(root_value))
        return tuple(
            protocol_binding(binding) for binding in self.artifact_inputs.validate_all(values)
        )

    def _dispatch_before_handshake(
        self,
        message: dict[str, Any],
        request_id: str,
    ) -> tuple[dict[str, Any], bool]:
        if message.get("method") != "protocol.handshake":
            return (
                self._response(
                    request_id,
                    error=protocol_error(
                        "PROTOCOL_HANDSHAKE_REQUIRED",
                        "A protocol handshake is required before other requests.",
                    ),
                    negotiated=False,
                ),
                False,
            )

        if not self.schemas.is_valid("handshakeRequest", message):
            return self._schema_error(request_id, negotiated=False), False

        supported = message["params"]["supportedVersions"]
        selected = next((version for version in SUPPORTED_VERSIONS if version in supported), None)
        if selected is None:
            return (
                self._response(
                    request_id,
                    error=protocol_error(
                        "PROTOCOL_VERSION_UNSUPPORTED",
                        "No compatible protocol version is available.",
                        details={"engineSupportedVersions": list(SUPPORTED_VERSIONS)},
                    ),
                    negotiated=False,
                ),
                True,
            )

        self.max_message_bytes = min(message["params"]["maxMessageBytes"], MAX_MESSAGE_BYTES)
        response = self._response(
            request_id,
            result={
                "selectedVersion": selected,
                "engineVersion": __version__,
                "engineInstanceId": self.engine_instance_id,
                "maxMessageBytes": self.max_message_bytes,
            },
            negotiated=False,
        )
        self.handshake_complete = True
        self.seen_request_ids.add(request_id)
        return response, False
