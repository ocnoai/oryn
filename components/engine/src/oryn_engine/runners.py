from __future__ import annotations

import contextlib
import hashlib
import math
import os
import queue
import re
import struct
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from oryn_engine.artifacts import ArtifactOutputStaging
from oryn_engine.jobs import EngineJob, JobExecutionError, RunnerEvent
from oryn_engine.model_support import transformers_workload
from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
from oryn_engine.runtime_packs.execution import (
    PackRuntime,
    PackRuntimeLocator,
    StreamingPackProcess,
)
from oryn_engine.runtime_packs.manifests import PackManifest
from oryn_engine.runtime_routing import GENERATED_PACKS, RuntimeRoutingService

INITIALIZATION_FAILURES = frozenset(
    {
        "MODEL_LOAD_FAILED",
        "MODEL_GENERATION_FAILED",
        "PACK_CRASHED",
        "PACK_TIMEOUT",
        "RUNTIME_BACKEND_UNAVAILABLE",
        "RUNTIME_DEPENDENCY_UNAVAILABLE",
    }
)

MAX_AUDIO_BYTES = 1024 * 1024 * 1024
MAX_AUDIO_SECONDS = 4 * 60 * 60
# Bound pre-resampling float32 samples for the 30-second transcription profile.
MAX_TRANSCRIPTION_DECODED_BYTES = 64 * 1024 * 1024
WHISPER_SAMPLE_RATE = 16_000


def _preserve_checkpoint_dtype(workload: str) -> bool:
    """Keep safe text-model checkpoint precision instead of expanding every weight to FP32."""

    return workload in {"causal-chat", "multimodal-chat", "embedding", "reranking"}


@dataclass
class TransformersASRHandle:
    model: Any
    processor: Any
    backend: str
    family: str = "speech-seq2seq"


@dataclass
class TransformersForcedAlignmentHandle:
    model: Any
    processor: Any
    backend: str


@dataclass
class TransformersCausalLMHandle:
    model: Any
    tokenizer: Any
    backend: str
    processor: Any = None


@dataclass
class TransformersEmbeddingHandle:
    model: Any
    tokenizer: Any
    backend: str
    pooling: str = "last-token"


@dataclass
class TransformersRerankerHandle:
    model: Any
    tokenizer: Any
    backend: str
    scoring: str = "causal"


@dataclass
class TransformersTaskHandle:
    model: Any
    processor: Any
    backend: str
    task: str


@dataclass
class TransformersMusicgenHandle:
    model: Any
    processor: Any
    backend: str


@dataclass(frozen=True)
class AceStepPackHandle:
    record: dict[str, Any]
    pack: PackManifest
    runtime: PackRuntime


@dataclass(frozen=True)
class SpeechPackHandle:
    record: dict[str, Any]
    pack: PackManifest
    runtime: PackRuntime


@dataclass(frozen=True)
class DiffusersPackHandle:
    record: dict[str, Any]
    pack: PackManifest
    runtime: PackRuntime


@dataclass(frozen=True)
class ProviderPackHandle:
    record: dict[str, Any]
    pack: PackManifest
    runtime: PackRuntime


@dataclass(frozen=True)
class LlamaCppPackHandle:
    record: dict[str, Any]
    pack: PackManifest
    runtime: PackRuntime


@dataclass
class ReviewedImplementationHandle:
    record: dict[str, Any]
    pack: PackManifest
    runtime: PackRuntime
    backend: str = "cpu"


@dataclass
class MlxLmPackHandle:
    record: dict[str, Any]
    pack: PackManifest
    runtime: PackRuntime


@dataclass
class SharedTaskPackHandle:
    record: dict[str, Any]
    pack: PackManifest
    runtime: PackRuntime
    task: str
    backend: str


@dataclass
class RuntimePlanHandle:
    active: object
    active_candidate: dict[str, Any]
    remaining_records: list[tuple[dict[str, Any], dict[str, Any]]]
    plan_id: str | None = None
    active_task_profile: dict[str, Any] | None = None

    @property
    def backend(self) -> str:
        return str(self.active_candidate.get("backend", "other"))


class RunnerRegistry:
    """Explicit allow-list of bundled model loaders and workload runners."""

    def __init__(self, outputs: ArtifactOutputStaging | None = None) -> None:
        self.outputs = outputs
        self._runtime_packs = builtin_registry()
        self._pack_locator = PackRuntimeLocator()
        self._pack_process = StreamingPackProcess()
        self.routing = RuntimeRoutingService()

    def load(self, record: dict[str, Any]) -> object:
        _verify_installed_files(record)
        plan = record.get("runtimePlan")
        if not isinstance(plan, dict) or not isinstance(plan.get("selected"), dict):
            return self._load_single(record)
        candidates = [plan["selected"], *plan.get("fallbacks", [])]
        prepared = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            try:
                prepared.append((candidate, self._record_for_candidate(record, candidate)))
            except JobExecutionError:
                if candidate == plan["selected"]:
                    raise
                # An inspected alternative may need files the user has not installed.
                continue
        self.routing.track_execution(record)
        last_error: JobExecutionError | None = None
        for index, (candidate, candidate_record) in enumerate(prepared):
            try:
                if self.routing.is_quarantined(plan.get("planId"), candidate["candidateId"]):
                    raise JobExecutionError(
                        "RUNTIME_QUARANTINED", "This execution route repeatedly failed."
                    )
                active = self._load_single(candidate_record)
            except JobExecutionError as exc:
                if exc.code in INITIALIZATION_FAILURES:
                    self.routing.record_failure(plan.get("planId"), candidate["candidateId"])
                last_error = exc
                continue
            return RuntimePlanHandle(
                active,
                candidate,
                prepared[index + 1 :],
                plan.get("planId"),
                candidate_record.get("taskProfile"),
            )
        if last_error is not None:
            raise last_error
        raise JobExecutionError("RUNNER_NOT_AVAILABLE", "The runtime plan has no usable candidate.")

    def _record_for_candidate(
        self, record: dict[str, Any], candidate: dict[str, Any]
    ) -> dict[str, Any]:
        from oryn_engine.provider_discovery import record_for_offer

        record = record_for_offer(record, candidate)
        variant_id = candidate.get("variantId")
        if variant_id == "oryn.builtin.transformers":
            return dict(record) | {"_runtimeCandidate": dict(candidate)}
        from oryn_engine.providers import provider_variant

        if provider_variant(variant_id):
            bound = record.get("providerPlans", {}).get(candidate.get("candidateId"))
            if not isinstance(bound, dict):
                raise JobExecutionError(
                    "RUNTIME_PLAN_INVALID", "Provider candidate has no bound plan."
                )
            pack = self._runtime_packs.active_pack(bound["packId"])
            runtime = self._pack_locator.resolve(pack)
            if runtime is None:
                raise JobExecutionError(
                    "RUNTIME_DEPENDENCY_UNAVAILABLE", "Provider pack is unavailable."
                )
            from oryn_engine.providers import verify_plan
            from oryn_engine.runtime_packs.builtin_foundations import builtin_foundation_registry

            verify_plan(bound, pack, runtime, record)
            foundations = builtin_foundation_registry()
            adapter = foundations.adapter_for_legacy_pack(pack.pack_id)
            if adapter is None:
                raise JobExecutionError(
                    "RUNTIME_PLAN_INVALID", "Provider foundation is unavailable."
                )
            foundation = foundations.active_foundation(adapter.foundation_id)
            if (
                candidate.get("adapter") != {"id": adapter.adapter_id, "version": adapter.version}
                or candidate.get("foundation")
                != {"id": foundation.foundation_id, "version": foundation.version}
                or candidate.get("backend") not in adapter.backends
            ):
                raise JobExecutionError(
                    "RUNTIME_PLAN_INVALID", "Provider runtime identity changed."
                )
            return dict(record) | {
                "_runtimeCandidate": dict(candidate),
                "_providerPlan": bound,
                "runtimePack": {
                    "packId": pack.pack_id,
                    "packVersion": pack.version,
                    "runnerId": bound["runnerId"],
                    "taskContract": bound["taskContract"],
                    "compatibilityLevel": "native-declarative",
                    "manifestDigest": pack.digest,
                },
            }
        onnx_variant = isinstance(variant_id, str) and variant_id.startswith("oryn.onnx.")
        if onnx_variant:
            from oryn_engine.onnx_tasks import variant_id as onnx_variant_id

            graphs = [f["name"] for f in record.get("files", []) if f["name"].endswith(".onnx")]
            try:
                expected = onnx_variant_id(
                    graphs, record.get("technicalCapabilities", {}).get("task")
                )
            except ValueError as exc:
                raise JobExecutionError(
                    "RUNTIME_PLAN_INVALID", "ONNX graph identity changed."
                ) from exc
            if variant_id != expected:
                raise JobExecutionError("RUNTIME_PLAN_INVALID", "ONNX graph identity changed.")
        if variant_id in GENERATED_PACKS or onnx_variant:
            from oryn_engine.runtime_packs.builtin_foundations import builtin_foundation_registry

            pack_id = "oryn.onnx-runtime" if onnx_variant else GENERATED_PACKS[variant_id]
            pack = self._runtime_packs.active_pack(pack_id)
            foundations = builtin_foundation_registry()
            adapter = foundations.adapter_for_legacy_pack(pack_id)
            declared = record.get("runtimePack", {})
            if (
                adapter is None
                or declared.get("packId") != pack.pack_id
                or declared.get("manifestDigest") != pack.digest
                or candidate.get("backend") not in adapter.backends
                or candidate.get("foundation", {}).get("id") != adapter.foundation_id
                or candidate.get("adapter", {}).get("id") != adapter.adapter_id
            ):
                raise JobExecutionError("RUNTIME_PLAN_INVALID", "Runtime identity changed.")
            foundation = foundations.active_foundation(adapter.foundation_id)
            if (
                candidate.get("foundation", {}).get("version") != foundation.version
                or candidate.get("adapter", {}).get("version") != adapter.version
            ):
                raise JobExecutionError("RUNTIME_PLAN_INVALID", "Runtime version changed.")
            return dict(record) | {"_runtimeCandidate": dict(candidate)}
        compatibility = next(
            (item for item in self._runtime_packs.compatibility if item.manifest_id == variant_id),
            None,
        )
        if compatibility is None:
            raise JobExecutionError(
                "RUNTIME_PLAN_INVALID", "The runtime plan references an unavailable variant."
            )
        candidate_record = dict(record)
        candidate_record["runtimePack"] = {
            "packId": compatibility.pack_id,
            "packVersion": compatibility.pack_version,
            "runnerId": compatibility.runner_id,
            "taskContract": compatibility.task_contract,
            "compatibilityLevel": compatibility.level,
            "manifestDigest": compatibility.digest,
            **(
                {"trustedMetadata": dict(compatibility.trusted_metadata)}
                if compatibility.trusted_metadata
                else {}
            ),
        }
        candidate_record["_runtimeCandidate"] = dict(candidate)
        return candidate_record

    def _load_single(
        self, record: dict[str, Any]
    ) -> (
        TransformersASRHandle
        | TransformersForcedAlignmentHandle
        | TransformersCausalLMHandle
        | TransformersEmbeddingHandle
        | TransformersMusicgenHandle
        | TransformersRerankerHandle
        | TransformersTaskHandle
        | AceStepPackHandle
        | SpeechPackHandle
        | DiffusersPackHandle
        | ProviderPackHandle
        | LlamaCppPackHandle
    ):
        runtime_pack = record.get("runtimePack")
        if record.get("sourceRecognition"):
            from oryn_engine.reviewed_implementations import validate_evidence

            try:
                validate_evidence(record["sourceRecognition"])
                if record.get("_runtimeCandidate", {}).get("backend", "cpu") != "cpu":
                    raise ValueError("Unqualified reviewed implementation backend")
                pack = self._runtime_packs.active_pack("oryn.transformers-pytorch")
                runtime = self._pack_locator.resolve(pack)
                if runtime is None:
                    raise ValueError("Reviewed implementation runtime unavailable")
            except (ValueError, OSError) as exc:
                raise JobExecutionError(
                    "MODEL_LOAD_FAILED", "Reviewed implementation unavailable"
                ) from exc
            return ReviewedImplementationHandle(dict(record), pack, runtime)
        if isinstance(record.get("_providerPlan"), dict):
            pack = self._runtime_packs.active_pack(runtime_pack["packId"])
            runtime = self._pack_locator.resolve(pack)
            if runtime is None:
                raise JobExecutionError(
                    "RUNTIME_DEPENDENCY_UNAVAILABLE", "Provider pack is unavailable."
                )
            return ProviderPackHandle(dict(record), pack, runtime)
        if isinstance(runtime_pack, dict) and runtime_pack.get("packId") == "oryn.ace-step-mlx":
            pack = self._runtime_packs.active_pack("oryn.ace-step-mlx")
            runtime = self._pack_locator.resolve(pack)
            if runtime is None:
                raise JobExecutionError(
                    "RUNTIME_DEPENDENCY_UNAVAILABLE",
                    "The ACE-Step runtime-pack dependency closure is unavailable.",
                )
            return AceStepPackHandle(dict(record), pack, runtime)
        if isinstance(runtime_pack, dict) and runtime_pack.get("packId") in {
            "oryn.kokoro-onnx",
            "oryn.qwen-tts-pytorch",
        }:
            pack = self._runtime_packs.active_pack(runtime_pack["packId"])
            runtime = self._pack_locator.resolve(pack)
            if runtime is None:
                raise JobExecutionError(
                    "RUNTIME_DEPENDENCY_UNAVAILABLE",
                    "The speech runtime dependency closure is unavailable.",
                )
            return SpeechPackHandle(dict(record), pack, runtime)
        if (
            isinstance(runtime_pack, dict)
            and runtime_pack.get("packId") == "oryn.diffusers-pytorch"
        ):
            pack = self._runtime_packs.active_pack("oryn.diffusers-pytorch")
            runtime = self._pack_locator.resolve(pack)
            if runtime is None:
                raise JobExecutionError(
                    "RUNTIME_DEPENDENCY_UNAVAILABLE",
                    "The Diffusers runtime-pack dependency closure is unavailable.",
                )
            return DiffusersPackHandle(dict(record), pack, runtime)
        if isinstance(runtime_pack, dict) and runtime_pack.get("packId") == "oryn.mlx-lm":
            pack = self._runtime_packs.active_pack("oryn.mlx-lm")
            runtime = self._pack_locator.resolve(pack)
            if runtime is None:
                raise JobExecutionError(
                    "RUNTIME_DEPENDENCY_UNAVAILABLE", "The MLX-LM adapter is unavailable."
                )
            if runtime_pack.get("taskContract") == "multimodal-generation-v1":
                return SharedTaskPackHandle(dict(record), pack, runtime, "any-to-any", "mlx")
            if runtime_pack.get("taskContract") == "voice-activity-detection-v1":
                return SharedTaskPackHandle(
                    dict(record), pack, runtime, "voice-activity-detection", "mlx"
                )
            return MlxLmPackHandle(dict(record), pack, runtime)
        if isinstance(runtime_pack, dict) and runtime_pack.get("packId") == "oryn.llama-cpp":
            pack = self._runtime_packs.active_pack("oryn.llama-cpp")
            runtime = self._pack_locator.resolve(pack)
            if runtime is None:
                raise JobExecutionError(
                    "RUNTIME_DEPENDENCY_UNAVAILABLE",
                    "The llama.cpp runtime-pack dependency closure is unavailable.",
                )
            return LlamaCppPackHandle(dict(record), pack, runtime)
        if isinstance(runtime_pack, dict) and runtime_pack.get("packId") in {
            "oryn.onnx-runtime",
            "oryn.mlx-audio",
        }:
            pack = self._runtime_packs.active_pack(runtime_pack["packId"])
            runtime = self._pack_locator.resolve(pack)
            if runtime is None:
                raise JobExecutionError(
                    "RUNTIME_DEPENDENCY_UNAVAILABLE", "The audio/task runtime is unavailable."
                )
            return SharedTaskPackHandle(
                dict(record),
                pack,
                runtime,
                record["technicalCapabilities"]["task"],
                record.get("_runtimeCandidate", {}).get(
                    "backend", "mlx" if pack.pack_id == "oryn.mlx-audio" else "cpu"
                ),
            )
        if record.get("runner") != "transformers" and not (
            record.get("runner") == "runtime-pack"
            and isinstance(runtime_pack, dict)
            and runtime_pack.get("packId") == "oryn.transformers-pytorch"
        ):
            raise JobExecutionError("RUNNER_NOT_AVAILABLE", "No bundled runner supports the model.")
        capabilities = record.get("technicalCapabilities", {})
        from oryn_engine.flax_weights import TASK_CONTRACTS as FLAX_TASKS
        from oryn_engine.generated_capabilities import architectures_for
        from oryn_engine.hdf5_tasks import FORMAT_ONLY_TASKS
        from oryn_engine.model_support import SEQ2SEQ_ARCHITECTURES
        from oryn_engine.shared_tasks import TASK_CONTRACTS

        task = capabilities.get("task")
        if (
            (
                task
                not in {
                    *FLAX_TASKS,
                    *FORMAT_ONLY_TASKS,
                    "image-classification",
                    "token-classification",
                    "depth-estimation",
                }
                or task == "translation"
                and not SEQ2SEQ_ARCHITECTURES.intersection(capabilities.get("architectures", []))
                or isinstance(runtime_pack, dict)
                and runtime_pack.get("runnerId") == "transformers.shared-tasks"
            )
            and task in TASK_CONTRACTS
            and (
                architectures_for(task, family_adapter=f"oryn.family.{task}").intersection(
                    capabilities.get("architectures", [])
                )
                or isinstance(runtime_pack, dict)
                and runtime_pack.get("runnerId") == "transformers.shared-tasks"
                and architectures_for(
                    task, family_adapter=f"oryn.family.numeric-archive-{task}"
                ).intersection(capabilities.get("architectures", []))
            )
        ):
            pack = self._runtime_packs.active_pack("oryn.transformers-pytorch")
            runtime = self._pack_locator.resolve(pack)
            if runtime is None:
                raise JobExecutionError(
                    "RUNTIME_DEPENDENCY_UNAVAILABLE", "Shared task runtime is unavailable."
                )
            backend = record.get("_runtimeCandidate", {}).get("backend")
            if backend is None:
                from oryn_engine.runtime.hardware import HardwareManager

                backend = HardwareManager().detect().preferred_backend
            return SharedTaskPackHandle(dict(record), pack, runtime, capabilities["task"], backend)
        architectures = set(capabilities.get("architectures", []))
        workload = transformers_workload(architectures, capabilities.get("task"))
        if workload is None:
            raise JobExecutionError(
                "RUNNER_NOT_AVAILABLE", "The Transformers workload is not enabled yet."
            )

        try:
            import torch
            from transformers import (
                AutoModel,
                AutoModelForAudioClassification,
                AutoModelForCausalLM,
                AutoModelForDepthEstimation,
                AutoModelForImageClassification,
                AutoModelForImageTextToText,
                AutoModelForMaskedLM,
                AutoModelForMultimodalLM,
                AutoModelForObjectDetection,
                AutoModelForQuestionAnswering,
                AutoModelForSeq2SeqLM,
                AutoModelForSequenceClassification,
                AutoModelForSpeechSeq2Seq,
                AutoModelForTokenClassification,
                AutoModelForZeroShotImageClassification,
                AutoProcessor,
                AutoTokenizer,
                MusicgenForConditionalGeneration,
            )
        except ImportError as exc:
            raise JobExecutionError(
                "RUNTIME_DEPENDENCY_UNAVAILABLE", "The Transformers runtime is unavailable."
            ) from exc

        model_ref = record["model"]
        common = {
            "revision": model_ref["revision"],
            "cache_dir": str(record["cacheRoot"]),
            "local_files_only": True,
            "trust_remote_code": False,
        }
        try:
            if workload == "musicgen":
                try:
                    processor = AutoProcessor.from_pretrained(model_ref["repository"], **common)
                except (AttributeError, ImportError, TypeError):
                    # Some generated Transformers integration checkpoints retain the
                    # pre-5.x list form for this tokenizer field.
                    processor = AutoProcessor.from_pretrained(
                        model_ref["repository"], **common, extra_special_tokens={}
                    )
                model = MusicgenForConditionalGeneration.from_pretrained(
                    model_ref["repository"],
                    **common,
                    use_safetensors=True,
                    dtype=torch.float32,
                )
            elif workload == "asr":
                processor = AutoProcessor.from_pretrained(model_ref["repository"], **common)
                if "Qwen3ASRForConditionalGeneration" in architectures:
                    model = AutoModelForMultimodalLM.from_pretrained(
                        model_ref["repository"],
                        **common,
                        use_safetensors=True,
                        dtype="auto",
                    )
                else:
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        model_ref["repository"],
                        **common,
                        use_safetensors=True,
                        dtype=torch.float32,
                    )
            elif workload == "forced-alignment":
                processor = AutoProcessor.from_pretrained(model_ref["repository"], **common)
                model = AutoModelForTokenClassification.from_pretrained(
                    model_ref["repository"],
                    **common,
                    use_safetensors=True,
                    dtype="auto",
                )
            elif workload in {
                "audio-classification",
                "depth-estimation",
                "fill-mask",
                "image-classification",
                "object-detection",
                "question-answering",
                "seq2seq",
                "text-classification",
                "token-classification",
                "zero-shot-image-classification",
            }:
                media_workloads = {
                    "audio-classification",
                    "depth-estimation",
                    "image-classification",
                    "object-detection",
                    "zero-shot-image-classification",
                }
                processor = (
                    AutoProcessor.from_pretrained(model_ref["repository"], **common)
                    if workload in media_workloads
                    else AutoTokenizer.from_pretrained(model_ref["repository"], **common)
                )
                model_class = {
                    "audio-classification": AutoModelForAudioClassification,
                    "depth-estimation": AutoModelForDepthEstimation,
                    "fill-mask": AutoModelForMaskedLM,
                    "image-classification": AutoModelForImageClassification,
                    "object-detection": AutoModelForObjectDetection,
                    "question-answering": AutoModelForQuestionAnswering,
                    "seq2seq": AutoModelForSeq2SeqLM,
                    "text-classification": AutoModelForSequenceClassification,
                    "token-classification": AutoModelForTokenClassification,
                    "zero-shot-image-classification": AutoModelForZeroShotImageClassification,
                }[workload]
                model = model_class.from_pretrained(
                    model_ref["repository"],
                    **common,
                    use_safetensors=True,
                    dtype="auto",
                    **(
                        {"output_loading_info": True}
                        if workload == "zero-shot-image-classification"
                        else {}
                    ),
                )
                if workload == "zero-shot-image-classification":
                    model, loading = model
                    if any(
                        loading.get(key)
                        for key in (
                            "missing_keys",
                            "unexpected_keys",
                            "mismatched_keys",
                            "error_msgs",
                        )
                    ):
                        raise ValueError("The complete image/text graph was not loaded")
            elif workload == "multimodal-chat":
                processor = AutoProcessor.from_pretrained(model_ref["repository"], **common)
                model = AutoModelForImageTextToText.from_pretrained(
                    model_ref["repository"],
                    **common,
                    use_safetensors=True,
                    dtype="auto",
                )
            elif workload in {"embedding", "encoder-embedding", "cross-encoder-reranking"}:
                processor = AutoTokenizer.from_pretrained(model_ref["repository"], **common)
                model_class = (
                    AutoModelForSequenceClassification
                    if workload == "cross-encoder-reranking"
                    else AutoModel
                )
                model = model_class.from_pretrained(
                    model_ref["repository"],
                    **common,
                    use_safetensors=True,
                    dtype="auto",
                )
            else:
                processor = AutoTokenizer.from_pretrained(model_ref["repository"], **common)
                model = AutoModelForCausalLM.from_pretrained(
                    model_ref["repository"],
                    **common,
                    use_safetensors=True,
                    dtype="auto" if _preserve_checkpoint_dtype(workload) else torch.float32,
                )
        except Exception as exc:
            raise JobExecutionError(
                "MODEL_LOAD_FAILED", "The locally installed model could not be loaded."
            ) from exc

        runtime_candidate = record.get("_runtimeCandidate", {})
        requested_backend = runtime_candidate.get("backend")
        backend = "cpu"
        if requested_backend == "cuda":
            if not bool(getattr(torch.cuda, "is_available", lambda: False)()):
                raise JobExecutionError(
                    "RUNTIME_BACKEND_UNAVAILABLE", "The planned CUDA backend is unavailable."
                )
            try:
                model = model.to("cuda")
                backend = "cuda"
            except (RuntimeError, NotImplementedError) as exc:
                raise JobExecutionError(
                    "RUNTIME_BACKEND_UNAVAILABLE", "The planned CUDA backend could not load."
                ) from exc
        elif requested_backend == "mps" or (
            requested_backend is None
            and bool(getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)())
        ):
            if not bool(
                getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()
            ):
                raise JobExecutionError(
                    "RUNTIME_BACKEND_UNAVAILABLE", "The planned Metal backend is unavailable."
                )
            try:
                model = model.to("mps")
                backend = "mps"
            except (RuntimeError, NotImplementedError) as exc:
                if requested_backend == "mps":
                    raise JobExecutionError(
                        "RUNTIME_BACKEND_UNAVAILABLE", "The planned Metal backend could not load."
                    ) from exc
                # CPU is the compatibility fallback. Convert there when a checkpoint dtype is
                # unsupported by the selected CPU kernels; the successful MPS path retains the
                # checkpoint's safe BF16/FP16 precision and materially lower memory footprint.
                model = model.to(device="cpu", dtype=torch.float32)
        model.eval()
        if workload == "musicgen":
            return TransformersMusicgenHandle(
                model=model,
                processor=processor,
                backend=backend,
            )
        if workload == "asr":
            return TransformersASRHandle(
                model=model,
                processor=processor,
                backend=backend,
                family=(
                    "qwen3-asr"
                    if "Qwen3ASRForConditionalGeneration" in architectures
                    else "speech-seq2seq"
                ),
            )
        if workload == "forced-alignment":
            return TransformersForcedAlignmentHandle(
                model=model,
                processor=processor,
                backend=backend,
            )
        if workload in {"embedding", "encoder-embedding"}:
            return TransformersEmbeddingHandle(
                model,
                processor,
                backend,
                "mean" if workload == "encoder-embedding" else "last-token",
            )
        if workload in {"reranking", "cross-encoder-reranking"}:
            return TransformersRerankerHandle(
                model,
                processor,
                backend,
                "cross-encoder" if workload == "cross-encoder-reranking" else "causal",
            )
        if workload in {
            "audio-classification",
            "depth-estimation",
            "fill-mask",
            "image-classification",
            "object-detection",
            "question-answering",
            "seq2seq",
            "text-classification",
            "token-classification",
            "zero-shot-image-classification",
        }:
            return TransformersTaskHandle(model, processor, backend, workload)
        return TransformersCausalLMHandle(
            model=model,
            tokenizer=processor.tokenizer if workload == "multimodal-chat" else processor,
            backend=backend,
            processor=processor if workload == "multimodal-chat" else None,
        )

    @staticmethod
    def unload(handle: object) -> None:
        if isinstance(handle, RuntimePlanHandle):
            RunnerRegistry.unload(handle.active)
            handle.remaining_records.clear()
            return
        if isinstance(
            handle,
            (
                AceStepPackHandle,
                SpeechPackHandle,
                DiffusersPackHandle,
                ProviderPackHandle,
                LlamaCppPackHandle,
                MlxLmPackHandle,
                ReviewedImplementationHandle,
            ),
        ):
            return
        if not isinstance(
            handle,
            (
                TransformersASRHandle,
                TransformersForcedAlignmentHandle,
                TransformersCausalLMHandle,
                TransformersEmbeddingHandle,
                TransformersMusicgenHandle,
                TransformersRerankerHandle,
                TransformersTaskHandle,
            ),
        ):
            return
        with contextlib.suppress(RuntimeError, AttributeError):
            handle.model.to("cpu")
        handle.model = None
        if isinstance(
            handle,
            (
                TransformersASRHandle,
                TransformersForcedAlignmentHandle,
                TransformersMusicgenHandle,
                TransformersTaskHandle,
            ),
        ):
            handle.processor = None
        else:
            handle.tokenizer = None

    def execute(
        self,
        job: EngineJob,
        handle: object,
        cancellation: threading.Event,
        emit: RunnerEvent,
    ) -> object:
        if isinstance(handle, RuntimePlanHandle):
            return self._execute_runtime_plan(job, handle, cancellation, emit)
        return self._execute_single(job, handle, cancellation, emit)

    def _execute_runtime_plan(
        self,
        job: EngineJob,
        handle: RuntimePlanHandle,
        cancellation: threading.Event,
        emit: RunnerEvent,
    ) -> object:
        while True:
            candidate = handle.active_candidate
            job.execution = {
                "runner": str(candidate.get("adapter", {}).get("id", "runtime-adapter")),
                "backend": str(candidate.get("backend", "other")),
                "foundationId": str(candidate.get("foundation", {}).get("id", "unknown")),
                "adapterId": str(candidate.get("adapter", {}).get("id", "unknown")),
                "candidateId": str(candidate.get("candidateId", "candidate-unknown")),
                "variantId": str(candidate.get("variantId", "unknown")),
                "precision": str(candidate.get("precision", "unknown")),
                **(
                    {"quantization": str(candidate["quantization"])}
                    if candidate.get("quantization")
                    else {}
                ),
                **({"planId": handle.plan_id} if handle.plan_id else {}),
            }
            visible_output = False

            def guarded_emit(name: str, data: dict[str, Any]) -> None:
                nonlocal visible_output
                if name in {"job.output.delta", "job.preview"}:
                    visible_output = True
                emit(name, data)

            try:
                if job.task_profile is not None:
                    from oryn_protocol.task_contracts import validate_invocation

                    effective_profile = handle.active_task_profile or getattr(
                        handle.active, "record", {}
                    ).get("taskProfile", job.task_profile)
                    if (
                        _chat_v2(job)
                        and effective_profile.get("contract", {}).get("id") == "chat-v1"
                    ):
                        from oryn_engine.task_profiles import profile_for_execution

                        effective_profile = profile_for_execution(
                            {
                                **getattr(handle.active, "record", {}),
                                "taskProfile": effective_profile,
                            },
                            contract_id="chat-v2",
                        )
                    try:
                        validate_invocation(effective_profile, job.input, job.task_requirements)
                    except ValueError as exc:
                        raise JobExecutionError(
                            "TASK_CONTRACT_INCOMPATIBLE",
                            "The runtime cannot honor the pinned task settings.",
                        ) from exc
                return self._execute_single(job, handle.active, cancellation, guarded_emit)
            except JobExecutionError as exc:
                if cancellation.is_set() or exc.code == "JOB_CANCELLED":
                    raise
                if exc.code in INITIALIZATION_FAILURES:
                    self.routing.record_failure(
                        handle.plan_id, str(candidate.get("candidateId", ""))
                    )
                if visible_output or exc.code not in INITIALIZATION_FAILURES:
                    raise
                self.unload(handle.active)
                next_handle: object | None = None
                next_candidate: dict[str, Any] | None = None
                while handle.remaining_records and next_handle is None:
                    candidate, candidate_record = handle.remaining_records.pop(0)
                    try:
                        if self.routing.is_quarantined(handle.plan_id, candidate["candidateId"]):
                            continue
                        next_handle = self._load_single(candidate_record)
                        next_candidate = candidate
                        handle.active_task_profile = candidate_record.get("taskProfile")
                    except JobExecutionError as next_error:
                        if next_error.code in INITIALIZATION_FAILURES:
                            self.routing.record_failure(handle.plan_id, candidate["candidateId"])
                        continue
                if next_handle is None or next_candidate is None:
                    raise
                prior_candidate_id = str(handle.active_candidate.get("candidateId", "unknown"))
                handle.active = next_handle
                handle.active_candidate = next_candidate
                job.runtime_fallbacks.append(
                    {
                        "fromCandidateId": prior_candidate_id,
                        "toCandidateId": str(next_candidate.get("candidateId", "unknown")),
                        "reasonCode": exc.code,
                    }
                )
                emit(
                    "job.progress",
                    {
                        "stage": "runtime-fallback",
                        "progress": 0.0,
                    },
                )

    def _execute_single(self, job, handle, cancellation, emit):
        if (job.task_profile or {}).get("contract", {}).get("id") != "chat-v2":
            return self._execute_raw(job, handle, cancellation, emit)
        from oryn_engine.chat_v2 import ReasoningStream, result_value

        parser = ReasoningStream()
        streamed = []

        def separated(name, data):
            if name == "job.output.delta" and data.get("channel") == "text":
                if not streamed and job.chat_reasoning_prefix:
                    parser.mode = "reasoning"
                streamed.append(data["delta"])
                for channel, text in parser.feed(data["delta"]):
                    emit(name, {"channel": channel, "delta": text})
            else:
                emit(name, data)

        raw = self._execute_raw(job, handle, cancellation, separated)
        value = raw.get("value", {}) if isinstance(raw, dict) else {}
        if not isinstance(value.get("text"), str):
            raise JobExecutionError("PACK_OUTPUT_INVALID", "Chat result is invalid")
        aggregate = "".join(streamed)
        if value.get("_legacyWhitespaceTrim") and value["text"].strip() == aggregate.strip():
            value = {**value, "text": aggregate}
        if not value["text"].startswith(aggregate):
            raise JobExecutionError("PACK_OUTPUT_INVALID", "Chat stream and final output disagree")
        if not streamed and job.chat_reasoning_prefix:
            parser.mode = "reasoning"
        if len(value["text"]) > len(aggregate):
            for channel, text in parser.feed(value["text"][len(aggregate) :]):
                emit("job.output.delta", {"channel": channel, "delta": text})
        for channel, text in parser.finish():
            emit("job.output.delta", {"channel": channel, "delta": text})
        return {
            "kind": "structured",
            "value": result_value(
                value,
                parser,
                job.input.get("thinking"),
                job.task_profile.get("thinking", {"support": "unknown", "default": None}),
            ),
        }

    def _execute_raw(
        self,
        job: EngineJob,
        handle: object,
        cancellation: threading.Event,
        emit: RunnerEvent,
    ) -> object:
        if isinstance(handle, SharedTaskPackHandle):
            return self._run_shared_task(job, handle, cancellation, emit)
        if isinstance(handle, TransformersASRHandle):
            return self._transcribe(job, handle, cancellation, emit)
        if isinstance(handle, TransformersForcedAlignmentHandle):
            return self._force_align(job, handle, cancellation, emit)
        if isinstance(handle, TransformersCausalLMHandle):
            return self._generate_chat(job, handle, cancellation, emit)
        if isinstance(handle, TransformersEmbeddingHandle):
            return self._embed(job, handle, cancellation, emit)
        if isinstance(handle, TransformersRerankerHandle):
            return self._rerank(job, handle, cancellation, emit)
        if isinstance(handle, TransformersMusicgenHandle):
            return self._generate_song(job, handle, cancellation, emit)
        if isinstance(handle, TransformersTaskHandle):
            return self._run_transformers_task(job, handle, cancellation, emit)
        if isinstance(handle, AceStepPackHandle):
            return self._generate_ace_step(job, handle, cancellation, emit)
        if isinstance(handle, SpeechPackHandle):
            return self._synthesize_speech(job, handle, cancellation, emit)
        if isinstance(handle, DiffusersPackHandle):
            return self._generate_diffusion(job, handle, cancellation, emit)
        if isinstance(handle, ProviderPackHandle):
            return self._generate_provider(job, handle, cancellation, emit)
        if isinstance(handle, MlxLmPackHandle):
            return self._generate_mlx(job, handle, cancellation, emit)
        if isinstance(handle, ReviewedImplementationHandle):
            return self._generate_reviewed(job, handle, cancellation, emit)
        if isinstance(handle, LlamaCppPackHandle):
            return self._generate_llama_cpp(job, handle, cancellation, emit)
        raise JobExecutionError("RUNNER_NOT_AVAILABLE", "The resident runner is invalid.")

    def _generate_reviewed(self, job, handle, cancellation, emit):
        from oryn_engine.reviewed_implementations import validate_evidence

        proof = handle.record["sourceRecognition"]
        try:
            validate_evidence(proof)
        except (ValueError, OSError) as exc:
            raise JobExecutionError("MODEL_LOAD_FAILED", "Reviewed implementation changed") from exc
        value = _chat_input(job.input, chat_v2=_chat_v2(job))
        if _chat_v2(job):
            value.pop("thinking", None)
        model = handle.record["model"]
        cache = Path(handle.record["cacheRoot"]).resolve()
        root = (
            cache
            / f"models--{model['repository'].replace('/', '--')}"
            / "snapshots"
            / model["revision"]
        ).resolve()
        if not root.is_relative_to(cache) or not root.is_dir():
            raise JobExecutionError("MODEL_LOAD_FAILED", "Reviewed model data is unavailable")
        emitted = 0

        def event(name, data):
            nonlocal emitted
            if (
                name != "job.output.delta"
                or set(data) != {"channel", "delta"}
                or (data["channel"] != "text" or not isinstance(data["delta"], str))
            ):
                raise JobExecutionError("PACK_OUTPUT_INVALID", "Reviewed stream is invalid")
            emitted += len(data["delta"].encode())
            if emitted > 4 * 1024 * 1024:
                raise JobExecutionError("PACK_OUTPUT_INVALID", "Reviewed stream exceeds bounds")
            emit(name, data)

        result = self._invoke_pack(
            handle.record,
            handle.pack,
            handle.runtime,
            "generate",
            {
                "implementationId": proof["adapterId"],
                "profileId": proof["profileId"],
                "configDigest": proof["configDigest"],
                "catalogDigest": proof["catalogDigest"],
                "modelRoot": str(root),
                "backend": "cpu",
                "input": value,
            },
            cancellation=cancellation,
            event_sink=event,
            timeout_seconds=30 * 60,
        )
        if (
            not isinstance(result.get("text"), str)
            or result.get("backend") != "cpu"
            or len(result["text"].encode()) > 4 * 1024 * 1024
        ):
            raise JobExecutionError("PACK_OUTPUT_INVALID", "Reviewed result is invalid")
        return {"kind": "structured", "value": {"text": result["text"]}}

    def _invoke_pack(self, record, pack, runtime, command, payload, **options):
        if isinstance(record.get("_providerPlan"), dict):
            names = {f["name"] for f in record["_providerPlan"]["files"]}
            record = record | {"files": [f for f in record["files"] if f["name"] in names]}
        with _selected_model_view(record, Path(payload["modelRoot"])) as root:
            return self._pack_process.invoke(
                pack, runtime, command, {**payload, "modelRoot": str(root)}, **options
            )

    def _run_shared_task(self, job, handle, cancellation, emit):
        from oryn_engine.shared_tasks import (
            ARTIFACT_TASKS,
            validate_task_input,
            validate_task_output,
        )

        if handle.pack.pack_id == "oryn.onnx-runtime" or (
            handle.pack.pack_id == "oryn.mlx-lm" and handle.task == "voice-activity-detection"
        ):
            from oryn_engine.onnx_tasks import input_value as validate_task_input
            from oryn_engine.onnx_tasks import output_value as validate_task_output

        if handle.pack.pack_id == "oryn.mlx-audio":
            from oryn_engine.runtime_packs.mlx_audio_worker import (
                input_value as validate_task_input,
            )
            from oryn_engine.runtime_packs.mlx_audio_worker import (
                output_value as validate_task_output,
            )

        try:
            value = validate_task_input(handle.task, job.input)
        except ValueError as exc:
            raise JobExecutionError("JOB_INPUT_INVALID", "The task input is invalid.") from exc
        model = handle.record["model"]
        cache = Path(handle.record["cacheRoot"]).resolve()
        root = (
            cache
            / f"models--{model['repository'].replace('/', '--')}"
            / "snapshots"
            / model["revision"]
        ).resolve()
        if not root.is_relative_to(cache) or not root.is_dir():
            raise JobExecutionError(
                "MODEL_LOAD_FAILED", "The selected model subset is unavailable."
            )
        binding = (
            _image_binding(job.input_artifacts, value["imageArtifactId"])
            if "imageArtifactId" in value
            else None
        )
        if binding and (
            binding["sizeBytes"] > 20 * 1024**2
            or binding["mimeType"] not in {"image/png", "image/jpeg", "image/webp"}
        ):
            raise JobExecutionError("JOB_INPUT_INVALID", "Image input exceeds task bounds.")
        if "videoArtifactId" in value:
            binding = next(
                (
                    item
                    for item in job.input_artifacts
                    if item["artifactId"] == value["videoArtifactId"]
                ),
                None,
            )
            if (
                binding is None
                or binding["mimeType"] not in {"video/mp4", "video/webm"}
                or not (1 <= binding["sizeBytes"] <= 64 * 1024**2)
            ):
                raise JobExecutionError(
                    "JOB_INPUT_INVALID", "Authorized video exceeds task bounds."
                )
        if "audioArtifactId" in value:
            binding = _audio_binding(job.input_artifacts, value["audioArtifactId"])
            if (
                type(binding.get("sizeBytes")) is not int
                or not 1 <= binding["sizeBytes"] <= 12 * 1024**2
            ):
                raise JobExecutionError(
                    "JOB_INPUT_INVALID", "Authorized audio exceeds task bounds."
                )
        with tempfile.TemporaryDirectory(prefix="oryn-task-output-") as temporary:
            audio_output = (
                handle.pack.pack_id == "oryn.onnx-runtime"
                and handle.task == "audio-to-audio"
                or handle.task == "text-to-speech"
                or handle.task == "audio-to-audio"
                and value.get("operation") in {"decode", "enhance", "separate"}
            )
            structured_file = handle.task == "visual-document-retrieval"
            output_path = (
                Path(temporary) / "result.json"
                if structured_file
                else Path(temporary) / "result.wav"
                if audio_output
                else Path(temporary) / "result.png"
                if handle.task in ARTIFACT_TASKS and handle.task != "audio-to-audio"
                else None
            )
            result = self._invoke_pack(
                handle.record,
                handle.pack,
                handle.runtime,
                "generate",
                {
                    "modelRoot": str(root),
                    "task": handle.task,
                    "backend": handle.backend,
                    "input": value,
                    "inputPath": binding["internalPath"] if binding else None,
                    "outputPath": str(output_path) if output_path else None,
                },
                cancellation=cancellation,
                event_sink=emit,
                timeout_seconds=3600,
            )
            _checkpoint(cancellation)
            if audio_output:
                if self.outputs is None:
                    raise JobExecutionError("PACK_OUTPUT_INVALID", "Audio staging is unavailable.")
                if handle.task == "audio-to-audio" and value.get("operation") == "separate":
                    paths = _validate_tiger_audio(
                        output_path.parent, result, Path(binding["internalPath"]), handle.backend
                    )
                    artifacts = [
                        self._stage_file(job, path, "audio/wav", cancellation) for path in paths
                    ]
                    return {
                        "kind": "multi",
                        "outputs": [
                            {"kind": "artifact", "artifact": artifact.protocol_descriptor()}
                            for artifact in artifacts
                        ],
                    }
                if handle.task == "audio-to-audio" and value.get("operation") == "decode":
                    _validate_codec_audio(output_path, result, value, handle.backend)
                elif (
                    handle.task == "audio-to-audio"
                    and handle.pack.pack_id == "oryn.transformers-pytorch"
                ):
                    validator = (
                        _validate_reuse_audio
                        if value.get("sampleRate") == 8000
                        else _validate_mpsenet_audio
                    )
                    validator(output_path, result, Path(binding["internalPath"]), handle.backend)
                elif handle.task == "text-to-speech":
                    if handle.pack.pack_id == "oryn.mlx-audio":
                        _validate_mlx_audio_speech(output_path, result, handle.backend)
                    else:
                        _validate_synthesized_audio(output_path, result, handle.backend)
                    if (
                        set(result)
                        != {
                            "relativeOutput",
                            "mediaType",
                            "sampleRate",
                            "channels",
                            "frames",
                            "durationSeconds",
                            "backend",
                        }
                        or result["relativeOutput"] != "result.wav"
                        or result["mediaType"] != "audio/wav"
                        or result["frames"] > 30 * result["sampleRate"]
                    ):
                        raise JobExecutionError("PACK_OUTPUT_INVALID", "Speech output is invalid.")
                else:
                    _validate_enhanced_audio(output_path, result, Path(binding["internalPath"]))
                artifact = self._stage_file(job, output_path, "audio/wav", cancellation)
                return {"kind": "artifact", "artifact": artifact.protocol_descriptor()}
            if output_path and not structured_file:
                if (
                    self.outputs is None
                    or set(result) != {"relativeOutput", "mediaType", "width", "height", "metadata"}
                    or result["relativeOutput"] != "result.png"
                    or result["mediaType"] != "image/png"
                ):
                    raise JobExecutionError("PACK_OUTPUT_INVALID", "Task output is invalid.")
                if (
                    output_path.is_symlink()
                    or not output_path.is_file()
                    or not 1 <= output_path.stat().st_size <= 64 * 1024**2
                ):
                    raise JobExecutionError("PACK_OUTPUT_INVALID", "Task artifact is invalid.")
                if any(
                    type(result[k]) is not int or not 1 <= result[k] <= 4096
                    for k in ("width", "height")
                ):
                    raise JobExecutionError(
                        "PACK_OUTPUT_INVALID", "Task artifact dimensions are invalid."
                    )
                _validate_generated_image(output_path, result["width"], result["height"])
                _validate_mask_metadata(output_path, result["metadata"])
                artifact = self._stage_file(job, output_path, "image/png", cancellation)
                combined = {
                    "kind": "multi",
                    "outputs": [
                        {"kind": "artifact", "artifact": artifact.protocol_descriptor()},
                        {"kind": "structured", "value": result["metadata"]},
                    ],
                }
                from oryn_protocol.task_contracts import make_profile, validate_result

                from oryn_engine.shared_tasks import TASK_CONTRACTS

                try:
                    validate_result(make_profile(TASK_CONTRACTS[handle.task]), combined)
                except ValueError as exc:
                    raise JobExecutionError(
                        "PACK_OUTPUT_INVALID", "Mask metadata is invalid."
                    ) from exc
                return combined
            try:
                if structured_file:
                    from oryn_engine.structured_outputs import read_result

                    result = read_result(output_path, result, handle.backend)
                    _checkpoint(cancellation)
                validate_task_output(handle.task, result)
            except ValueError as exc:
                raise JobExecutionError("PACK_OUTPUT_INVALID", "Task result is invalid.") from exc
            return {"kind": "structured", "value": result}

    def _generate_mlx(
        self,
        job: EngineJob,
        handle: MlxLmPackHandle,
        cancellation: threading.Event,
        emit: RunnerEvent,
    ) -> object:
        vision = "image" in handle.record.get("technicalCapabilities", {}).get(
            "inputModalities", []
        )
        value = _chat_input(job.input, allow_image=vision, chat_v2=_chat_v2(job))
        modern_worker = _chat_v2(job) and getattr(handle.runtime, "chat_v2", False)
        if _chat_v2(job) and not modern_worker:
            value.pop("thinking", None)
            if value["maxTokens"] > 512:
                raise JobExecutionError(
                    "JOB_INPUT_INVALID", "The installed worker admits at most 512 output tokens"
                )
        binding = (
            _image_binding(job.input_artifacts, value["imageArtifactId"])
            if "imageArtifactId" in value
            else None
        )
        if vision and binding is None:
            raise JobExecutionError(
                "JOB_INPUT_INVALID", "This vision route requires an owned image."
            )
        model = handle.record["model"]
        cache = Path(handle.record["cacheRoot"]).resolve()
        root = (
            cache
            / f"models--{model['repository'].replace('/', '--')}"
            / "snapshots"
            / model["revision"]
        ).resolve()
        if not root.is_relative_to(cache) or not root.is_dir():
            raise JobExecutionError("MODEL_LOAD_FAILED", "The MLX model subset is unavailable.")
        emitted = 0

        def event(name: str, data: dict[str, Any]) -> None:
            nonlocal emitted
            if name == "chat.framing" and _chat_v2(job):
                if set(data) != {"reasoningPrefix"} or type(data["reasoningPrefix"]) is not bool:
                    raise JobExecutionError("PACK_OUTPUT_INVALID", "Chat framing is invalid")
                job.chat_reasoning_prefix = data["reasoningPrefix"]
                return
            if name == "job.output.delta":
                if (
                    set(data) != {"channel", "delta"}
                    or data.get("channel") != "text"
                    or not isinstance(data.get("delta"), str)
                ):
                    raise JobExecutionError("PACK_OUTPUT_INVALID", "MLX stream is invalid.")
                emitted += len(data["delta"].encode())
                if emitted > 4 * 1024 * 1024:
                    raise JobExecutionError("PACK_OUTPUT_INVALID", "MLX stream is oversized.")
            emit(name, data)

        result = self._invoke_pack(
            handle.record,
            handle.pack,
            handle.runtime,
            "generate",
            {
                "modelRoot": str(root),
                **({"taskContract": "chat-v2"} if modern_worker else {}),
                "input": value,
                **({"inputPath": binding["internalPath"]} if binding else {}),
            },
            cancellation=cancellation,
            event_sink=event,
            timeout_seconds=3600,
        )
        if (
            not isinstance(result.get("text"), str)
            or len(result["text"].encode()) > 4 * 1024 * 1024
        ):
            raise JobExecutionError("PACK_OUTPUT_INVALID", "MLX output is invalid.")
        return {
            "kind": "structured",
            "value": {
                **(
                    {k: result[k] for k in ("finishReason", "usage", "metadata") if k in result}
                    if _chat_v2(job)
                    else {}
                ),
                "text": result["text"],
            },
        }

    def _generate_llama_cpp(
        self,
        job: EngineJob,
        handle: LlamaCppPackHandle,
        cancellation: threading.Event,
        emit: RunnerEvent,
    ) -> object:
        runtime_pack = handle.record.get("runtimePack", {})
        trusted = runtime_pack.get("trustedMetadata", {})
        from oryn_engine import gguf_translation

        translation = runtime_pack.get("taskContract") == gguf_translation.CONTRACT
        projector = trusted.get("projectorFile")
        if translation:
            try:
                gguf_translation.validate_binding(
                    trusted.get("adapterArchitecture"), trusted.get("chatTemplate"), projector
                )
                input_value = gguf_translation.validate_input(job.input)
            except (ValueError, TypeError) as exc:
                raise JobExecutionError("JOB_INPUT_INVALID", str(exc)) from exc
        else:
            input_value = _chat_input(
                job.input, allow_image=projector is not None, chat_v2=_chat_v2(job)
            )
        modern_worker = _chat_v2(job) and getattr(handle.runtime, "chat_v2", False)
        if _chat_v2(job) and not modern_worker:
            input_value.pop("thinking", None)
            if input_value["maxTokens"] > 512:
                raise JobExecutionError(
                    "JOB_INPUT_INVALID", "The installed worker admits at most 512 output tokens"
                )
        image_id = input_value.pop("imageArtifactId", None)
        if projector and image_id is None:
            raise JobExecutionError(
                "JOB_INPUT_INVALID", "The projector route requires an owned image."
            )
        binding = _image_binding(job.input_artifacts, image_id) if image_id else None
        quantization = trusted.get("quantization")
        context_limit = trusted.get("contextLength")
        from oryn_engine import chat_formats

        chat_template = trusted.get("chatTemplate")
        model_file = trusted.get("modelFile")
        architecture = trusted.get("adapterArchitecture")
        if not isinstance(architecture, str):
            declared = handle.record.get("technicalCapabilities", {}).get("architectures", [])
            if isinstance(declared, list) and len(declared) == 1 and isinstance(declared[0], str):
                architecture = declared[0]
        if (
            not isinstance(quantization, str)
            or not isinstance(context_limit, int)
            or isinstance(context_limit, bool)
            or chat_template not in chat_formats.names()
            or not isinstance(model_file, str)
            or not isinstance(architecture, str)
        ):
            raise JobExecutionError(
                "MODEL_METADATA_INVALID", "Trusted GGUF execution metadata is incomplete."
            )
        model = handle.record["model"]
        model_root = (
            Path(handle.record["cacheRoot"])
            / f"models--{model['repository'].replace('/', '--')}"
            / "snapshots"
            / model["revision"]
        ).resolve()
        cache_root = Path(handle.record["cacheRoot"]).resolve()
        if (
            not model_root.is_relative_to(cache_root)
            or not model_root.is_dir()
            or not (model_root / model_file).is_file()
        ):
            raise JobExecutionError("MODEL_LOAD_FAILED", "The GGUF model subset is unavailable.")
        emitted_bytes = 0

        def pack_event(name: str, data: dict[str, Any]) -> None:
            nonlocal emitted_bytes
            if name == "chat.framing" and _chat_v2(job):
                if set(data) != {"reasoningPrefix"} or type(data["reasoningPrefix"]) is not bool:
                    raise JobExecutionError("PACK_OUTPUT_INVALID", "Chat framing is invalid")
                job.chat_reasoning_prefix = data["reasoningPrefix"]
                return
            if name != "job.output.delta":
                emit(name, data)
                return
            if translation:
                raise JobExecutionError(
                    "PACK_OUTPUT_INVALID", "Translation emitted an undeclared stream."
                )
            if (
                set(data) != {"channel", "delta"}
                or data.get("channel") != "text"
                or not isinstance(data.get("delta"), str)
                or not data["delta"]
            ):
                raise JobExecutionError("PACK_OUTPUT_INVALID", "llama.cpp stream is invalid.")
            emitted_bytes += len(data["delta"].encode("utf-8"))
            if emitted_bytes > 4 * 1024 * 1024:
                raise JobExecutionError("PACK_OUTPUT_INVALID", "llama.cpp stream is too large.")
            emit(name, data)

        result = self._invoke_pack(
            handle.record,
            handle.pack,
            handle.runtime,
            "generate",
            {
                **({"taskContract": gguf_translation.CONTRACT} if translation else {}),
                **({"taskContract": "chat-v2"} if modern_worker else {}),
                "modelRoot": str(model_root),
                "modelFile": model_file,
                **(
                    {"projectorFile": projector, "inputPath": binding["internalPath"]}
                    if binding
                    else {}
                ),
                "input": {
                    **input_value,
                    **({} if translation else {"chatTemplate": chat_template}),
                    "expectedArchitecture": architecture,
                    "expectedQuantization": quantization,
                    "contextLimit": context_limit,
                },
            },
            cancellation=cancellation,
            event_sink=pack_event,
            timeout_seconds=30 * 60,
        )
        text = result.get("text")
        if (
            not isinstance(text, str)
            or not text
            or len(text.encode("utf-8")) > 4 * 1024 * 1024
            or result.get("backend") not in {"metal", "cuda", "cpu"}
            or result.get("quantization") != quantization
            or result.get("contextLength") != context_limit
            or not isinstance(result.get("runtimeContextLimit"), int)
            or not 128 <= result["runtimeContextLimit"] <= context_limit
        ):
            raise JobExecutionError("PACK_OUTPUT_INVALID", "llama.cpp output is invalid.")
        if translation:
            try:
                value = gguf_translation.validate_output(text)
            except ValueError as exc:
                raise JobExecutionError("PACK_OUTPUT_INVALID", str(exc)) from exc
            if result["runtimeContextLimit"] > gguf_translation.MAX_CONTEXT:
                raise JobExecutionError("PACK_OUTPUT_INVALID", "Translation context is invalid.")
            return {"kind": "structured", "value": value}
        return {
            "kind": "structured",
            "value": {
                **(
                    {k: result[k] for k in ("finishReason", "usage", "metadata") if k in result}
                    if _chat_v2(job)
                    else {}
                ),
                "text": text,
                **({"_legacyWhitespaceTrim": True} if _chat_v2(job) and not modern_worker else {}),
            },
        }

    def _generate_diffusion(
        self,
        job: EngineJob,
        handle: DiffusersPackHandle,
        cancellation: threading.Event,
        emit: RunnerEvent,
    ) -> object:
        runtime_pack = handle.record.get("runtimePack", {})
        contract = runtime_pack.get("taskContract")
        bound_backend = handle.record.get("_runtimeCandidate", {}).get(
            "backend"
        ) or handle.record.get("runtimePlan", {}).get("selected", {}).get("backend")
        if bound_backend not in {"cpu", "mps", "cuda"}:
            raise JobExecutionError(
                "RUNTIME_PLAN_INVALID", "The media runtime backend is not bound."
            )
        if (
            contract == "text-to-image-v1"
            and isinstance(job.input, dict)
            and "imageArtifactId" in job.input
            and handle.record.get("technicalCapabilities", {})
            .get("capabilities", {})
            .get("imageEditing")
            is True
        ):
            from oryn_engine.diffusion_tasks import pipeline_contract

            architectures = handle.record.get("technicalCapabilities", {}).get("architectures", [])
            family = architectures[0] if len(architectures) == 1 else ""
            contract = pipeline_contract(family, "image-to-image")
        raw_input = job.input
        profile = handle.record.get("generationProfile")
        if isinstance(raw_input, dict) and profile is not None:
            from oryn_protocol.task_contracts import TASK_CONTRACTS

            parameters = TASK_CONTRACTS.get(contract, {}).get("parameterNames", ())
            defaults = {
                key: value["default"]
                for key, value in profile["parameters"].items()
                if key in parameters
            }
            raw_input = {**defaults, **raw_input}
        is_h3 = "MiniMaxH3ModularPipeline" in handle.record.get("technicalCapabilities", {}).get(
            "architectures", []
        )
        if is_h3:
            from oryn_engine.minimax_h3 import validate_input

            try:
                input_value = validate_input(raw_input, contract)
            except (ValueError, TypeError) as exc:
                raise JobExecutionError("JOB_INPUT_INVALID", "Video input is invalid.") from exc
        else:
            input_value = _diffusion_input(raw_input, contract)
        if self.outputs is None:
            raise JobExecutionError("ARTIFACT_IO_ERROR", "Artifact output staging is unavailable.")
        model = handle.record["model"]
        model_root = (
            Path(handle.record["cacheRoot"])
            / f"models--{model['repository'].replace('/', '--')}"
            / "snapshots"
            / model["revision"]
        ).resolve()
        cache_root = Path(handle.record["cacheRoot"]).resolve()
        if not model_root.is_relative_to(cache_root) or not model_root.is_dir():
            raise JobExecutionError(
                "MODEL_LOAD_FAILED", "The Diffusers model subset is unavailable."
            )

        input_path: Path | None = None
        if contract in {
            "image-to-image-v1",
            "image-to-video-v1",
            "image-editing-v1",
            "image-layers-v1",
            "image-text-to-video-v1",
        }:
            binding = _image_binding(job.input_artifacts, input_value["imageArtifactId"])
            input_path = Path(binding["internalPath"])
        is_image = contract in {
            "unconditional-image-generation-v1",
            "text-to-image-v1",
            "image-to-image-v1",
            "image-editing-v1",
            "image-layers-v1",
        }
        reference_paths = []
        if contract == "image-editing-v1":
            from oryn_engine.diffusion_tasks import validate_reference_images

            architectures = handle.record.get("technicalCapabilities", {}).get("architectures", [])
            validate_reference_images(
                architectures[0] if len(architectures) == 1 else "",
                1 + len(input_value.get("referenceImageArtifactIds", [])),
                runtime_pack.get("trustedMetadata", {}).get("composition"),
            )
            reference_paths = [
                _image_binding(job.input_artifacts, artifact_id)["internalPath"]
                for artifact_id in input_value.get("referenceImageArtifactIds", [])
            ]
        is_music = contract == "text-to-music-v1"
        media_type = "audio/wav" if is_music else "image/png" if is_image else "video/mp4"
        output_name = "audio.wav" if is_music else "image.png" if is_image else "video.mp4"
        size_limit = (
            16 * 1024 * 1024
            if is_music
            else 256 * 1024 * 1024
            if is_image
            else 2 * 1024 * 1024 * 1024
        )

        with tempfile.TemporaryDirectory(prefix="oryn-diffusers-output-") as output_directory:
            directory = Path(output_directory)
            output_path = directory / output_name
            preview_path = directory / "preview.png"
            preview_sequence = 0

            def pack_event(name: str, data: dict[str, Any]) -> None:
                nonlocal preview_sequence
                if name != "pack.preview":
                    emit(name, data)
                    return
                if (
                    set(data) != {"relativeOutput", "mediaType", "sequence"}
                    or data.get("relativeOutput") != preview_path.name
                    or data.get("mediaType") != "image/png"
                    or not isinstance(data.get("sequence"), int)
                    or isinstance(data.get("sequence"), bool)
                    or data["sequence"] <= preview_sequence
                    or not preview_path.is_file()
                    or preview_path.stat().st_size > 16 * 1024 * 1024
                ):
                    raise JobExecutionError("PACK_OUTPUT_INVALID", "Diffusers preview is invalid.")
                preview_sequence = data["sequence"]
                preview = self._stage_file(job, preview_path, "image/png", cancellation)
                emit("job.preview", {"output": preview.protocol_descriptor()})

            result = self._invoke_pack(
                handle.record,
                handle.pack,
                handle.runtime,
                "generate",
                {
                    "modelRoot": str(model_root),
                    "outputPath": str(output_path),
                    "previewPath": str(preview_path) if is_image else None,
                    "inputPath": str(input_path) if input_path is not None else None,
                    **({"referenceImagePaths": reference_paths} if reference_paths else {}),
                    "taskContract": contract,
                    "backend": bound_backend,
                    "input": input_value,
                    **(
                        {"composition": runtime_pack["trustedMetadata"]["composition"]}
                        if runtime_pack.get("trustedMetadata", {}).get("composition") is not None
                        else {}
                    ),
                },
                cancellation=cancellation,
                event_sink=pack_event,
                timeout_seconds=2 * 60 * 60,
            )
            if contract == "image-layers-v1":
                from oryn_engine.diffusion_tasks import validate_layer_outputs

                try:
                    paths = validate_layer_outputs(result, directory, input_value["layers"])
                except (ValueError, OSError) as exc:
                    raise JobExecutionError(
                        "PACK_OUTPUT_INVALID", "Image layers are invalid."
                    ) from exc
                try:
                    outputs = [
                        {
                            "kind": "artifact",
                            "artifact": self._stage_file(
                                job, path, "image/png", cancellation
                            ).protocol_descriptor(),
                        }
                        for path in paths
                    ]
                    _checkpoint(cancellation)
                except BaseException:
                    self.outputs.cleanup_job(job.job_id)
                    raise
                return {"kind": "multi", "outputs": outputs}
            if (
                result.get("relativeOutput") != output_path.name
                or result.get("mediaType") != media_type
                or not is_music
                and result.get("width") != input_value["width"]
                or not is_music
                and result.get("height") != input_value["height"]
                or result.get("backend") != bound_backend
                or not output_path.is_file()
                or not 1 <= output_path.stat().st_size <= size_limit
            ):
                raise JobExecutionError(
                    "PACK_OUTPUT_INVALID", "Diffusers returned an invalid output."
                )
            if is_music:
                from oryn_engine.minimax_music import validate_audio_artifact

                try:
                    validate_audio_artifact(output_path, result, input_value["durationSeconds"])
                except (OSError, ValueError, EOFError) as exc:
                    raise JobExecutionError(
                        "PACK_OUTPUT_INVALID", "Music output is invalid."
                    ) from exc
            elif is_h3:
                from oryn_engine.video_audio_artifacts import validate_video_audio

                try:
                    validate_video_audio(output_path, result, input_value)
                except (OSError, ValueError, EOFError) as exc:
                    raise JobExecutionError(
                        "PACK_OUTPUT_INVALID", "Video soundtrack output is invalid."
                    ) from exc
            elif is_image:
                _validate_generated_image(output_path, input_value["width"], input_value["height"])
            else:
                _validate_generated_video(output_path, result, input_value)
            artifact = self._stage_file(job, output_path, media_type, cancellation)
        emit("job.progress", {"stage": "publishing-artifact", "progress": 1.0})
        return {"kind": "artifact", "artifact": artifact.protocol_descriptor()}

    def _generate_provider(self, job, handle, cancellation, emit):
        from oryn_engine.providers import verify_plan

        bound = verify_plan(
            handle.record["_providerPlan"], handle.pack, handle.runtime, handle.record
        )
        defaults = {k: v["default"] for k, v in bound["generationProfile"]["parameters"].items()}
        raw_input = defaults | job.input if isinstance(job.input, dict) else job.input
        value = _diffusion_input(raw_input, bound["taskContract"])
        binding = (
            _image_binding(job.input_artifacts, value["imageArtifactId"])
            if "imageArtifactId" in value
            else None
        )
        if self.outputs is None:
            raise JobExecutionError("ARTIFACT_IO_ERROR", "Artifact output staging is unavailable.")
        model = handle.record["model"]
        root = (
            Path(handle.record["cacheRoot"])
            / f"models--{model['repository'].replace('/', '--')}"
            / "snapshots"
            / model["revision"]
        )
        with tempfile.TemporaryDirectory(prefix="oryn-provider-output-") as temporary:
            output = Path(temporary) / "image.png"
            result = self._invoke_pack(
                handle.record,
                handle.pack,
                handle.runtime,
                "generate",
                {
                    "modelRoot": str(root.resolve()),
                    "outputPath": str(output),
                    "plan": bound["providerPlan"],
                    "input": value,
                    "inputPath": binding["internalPath"] if binding else None,
                },
                cancellation=cancellation,
                event_sink=emit,
                timeout_seconds=7200,
            )
            if (
                result.get("relativeOutput") != output.name
                or result.get("mediaType") != "image/png"
                or result.get("catalogDigest") != bound["catalogDigest"]
                or result.get("backend") != handle.record["_runtimeCandidate"]["backend"]
                or result.get("width") != value["width"]
                or result.get("height") != value["height"]
                or result.get("seed") != value["seed"]
            ):
                raise JobExecutionError(
                    "PACK_OUTPUT_INVALID", "Provider result differs from its plan."
                )
            _validate_generated_image(output, value["width"], value["height"])
            artifact = self._stage_file(job, output, "image/png", cancellation)
        emit("job.progress", {"stage": "publishing-artifact", "progress": 1.0})
        return {"kind": "artifact", "artifact": artifact.protocol_descriptor()}

    def _stage_file(
        self,
        job: EngineJob,
        path: Path,
        media_type: str,
        cancellation: threading.Event,
    ) -> Any:
        assert self.outputs is not None
        _checkpoint(cancellation)
        with (
            path.open("rb") as source,
            self.outputs.writer(job.job_id, media_type, path.name) as writer,
        ):
            while chunk := source.read(1024 * 1024):
                _checkpoint(cancellation)
                writer.write(chunk)
            _checkpoint(cancellation)
            return writer.finalize()

    def _synthesize_speech(
        self,
        job: EngineJob,
        handle: SpeechPackHandle,
        cancellation: threading.Event,
        emit: RunnerEvent,
    ) -> object:
        qwen = handle.pack.pack_id == "oryn.qwen-tts-pytorch"
        if qwen:
            from oryn_engine.qwen_tts import input_value as validate_speech

            try:
                input_value = validate_speech(job.input)
            except ValueError as exc:
                raise JobExecutionError(
                    "JOB_INPUT_INVALID", "Speech-synthesis input is invalid."
                ) from exc
        else:
            input_value = _speech_input(job.input)
        if self.outputs is None:
            raise JobExecutionError("ARTIFACT_IO_ERROR", "Artifact output staging is unavailable.")
        model = handle.record["model"]
        model_root = (
            Path(handle.record["cacheRoot"])
            / f"models--{model['repository'].replace('/', '--')}"
            / "snapshots"
            / model["revision"]
        ).resolve()
        cache_root = Path(handle.record["cacheRoot"]).resolve()
        if not model_root.is_relative_to(cache_root) or not model_root.is_dir():
            raise JobExecutionError("MODEL_LOAD_FAILED", "The speech model subset is unavailable.")
        with tempfile.TemporaryDirectory(prefix="oryn-speech-output-") as output_directory:
            output_path = Path(output_directory) / "speech.wav"
            result = self._invoke_pack(
                handle.record,
                handle.pack,
                handle.runtime,
                "generate",
                {
                    "modelRoot": str(model_root),
                    "outputPath": str(output_path),
                    "input": input_value,
                    **({"backend": handle.record["_runtimeCandidate"]["backend"]} if qwen else {}),
                },
                cancellation=cancellation,
                event_sink=emit,
                timeout_seconds=10 * 60,
            )
            if (
                result.get("relativeOutput") != output_path.name
                or result.get("mediaType") != "audio/wav"
                or result.get("voice") != input_value["voice"]
                or result.get("language") != input_value["language"]
                or not output_path.is_file()
                or output_path.stat().st_size > MAX_AUDIO_BYTES
            ):
                raise JobExecutionError(
                    "PACK_OUTPUT_INVALID", "Speech worker returned invalid audio."
                )
            if qwen:
                _validate_synthesized_audio(
                    output_path, result, handle.record["_runtimeCandidate"]["backend"]
                )
            with output_path.open("rb") as source:
                header = source.read(12)
                if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
                    raise JobExecutionError(
                        "PACK_OUTPUT_INVALID", "Speech worker returned invalid audio."
                    )
                source.seek(0)
                with self.outputs.writer(job.job_id, "audio/wav", output_path.name) as writer:
                    while chunk := source.read(1024 * 1024):
                        _checkpoint(cancellation)
                        writer.write(chunk)
                    artifact = writer.finalize()
        emit("job.progress", {"stage": "publishing-artifact", "progress": 1.0})
        return {"kind": "artifact", "artifact": artifact.protocol_descriptor()}

    @staticmethod
    def _embed(
        job: EngineJob,
        handle: TransformersEmbeddingHandle,
        cancellation: threading.Event,
        emit: RunnerEvent,
    ) -> object:
        input_value = _embedding_input(job.input)
        _checkpoint(cancellation)
        emit("job.progress", {"stage": "encoding", "progress": 0.05})
        try:
            import torch
            import torch.nn.functional as functional

            encoded = handle.tokenizer(
                input_value["texts"],
                padding=True,
                truncation=True,
                max_length=min(
                    8192,
                    int(
                        getattr(
                            getattr(handle.model, "config", None), "max_position_embeddings", 512
                        )
                    ),
                ),
                return_tensors="pt",
            )
            model_inputs = {
                key: value.to(handle.backend)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask"}
            }
            attention_mask = model_inputs["attention_mask"]
            _checkpoint(cancellation)
            emit("job.state", {"state": "running"})
            emit("job.progress", {"stage": "embedding", "progress": 0.2})
            with torch.inference_mode():
                output = handle.model(**model_inputs, output_hidden_states=True, return_dict=True)
            hidden = output.hidden_states[-1]
            if handle.pooling == "mean":
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                rows = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                positions = torch.arange(hidden.shape[1], device=hidden.device).expand_as(
                    attention_mask
                )
                last_indices = (
                    positions.masked_fill(attention_mask == 0, -1).max(dim=1).values.clamp(min=0)
                )
                rows = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_indices]
            if input_value["normalize"]:
                # Public vectors are float32. Normalize there as well: half/bfloat
                # normalization can otherwise return visibly non-unit vectors.
                rows = functional.normalize(rows.float(), p=2, dim=1)
            values = rows.detach().to("cpu", dtype=torch.float32).tolist()
        except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
            raise JobExecutionError("EMBEDDING_FAILED", "Text embedding failed.") from exc
        _checkpoint(cancellation)
        if (
            len(values) != len(input_value["texts"])
            or not values
            or not isinstance(values[0], list)
            or not values[0]
            or any(len(row) != len(values[0]) for row in values)
            or any(not math.isfinite(float(number)) for row in values for number in row)
        ):
            raise JobExecutionError("EMBEDDING_OUTPUT_INVALID", "Embedding output is invalid.")
        emit("job.progress", {"stage": "finalizing-output", "progress": 1.0})
        return {
            "kind": "structured",
            "value": {
                "embeddings": values,
                "dimensions": len(values[0]),
                "normalized": input_value["normalize"],
            },
        }

    @staticmethod
    def _rerank(
        job: EngineJob,
        handle: TransformersRerankerHandle,
        cancellation: threading.Event,
        emit: RunnerEvent,
    ) -> object:
        input_value = _reranking_input(job.input)
        if handle.scoring == "cross-encoder":
            return _cross_encoder_rerank(handle, input_value, cancellation, emit)
        _checkpoint(cancellation)
        try:
            import torch

            prefix = (
                "<|im_start|>system\nJudge whether the Document meets the requirements based on "
                'the Query and the Instruct provided. Note that the answer can only be "yes" or '
                '"no".<|im_end|>\n<|im_start|>user\n'
            )
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            prefix_tokens = handle.tokenizer.encode(prefix, add_special_tokens=False)
            suffix_tokens = handle.tokenizer.encode(suffix, add_special_tokens=False)
            available = 8_192 - len(prefix_tokens) - len(suffix_tokens)
            pairs = [
                f"<Instruct>: {input_value['instruction']}\n<Query>: {input_value['query']}"
                f"\n<Document>: {document}"
                for document in input_value["documents"]
            ]
            encoded = handle.tokenizer(
                pairs,
                padding=False,
                truncation=True,
                max_length=available,
                add_special_tokens=False,
            )
            encoded["input_ids"] = [
                prefix_tokens + row + suffix_tokens for row in encoded["input_ids"]
            ]
            padded = handle.tokenizer.pad(encoded, padding=True, return_tensors="pt")
            model_inputs = {
                key: value.to(handle.backend)
                for key, value in padded.items()
                if key in {"input_ids", "attention_mask"}
            }
            true_id = handle.tokenizer.convert_tokens_to_ids("yes")
            false_id = handle.tokenizer.convert_tokens_to_ids("no")
            if not isinstance(true_id, int) or not isinstance(false_id, int):
                raise ValueError("ranking answer tokens are unavailable")
            emit("job.state", {"state": "running"})
            emit("job.progress", {"stage": "reranking", "progress": 0.2})
            _checkpoint(cancellation)
            with torch.inference_mode():
                logits = handle.model(**model_inputs).logits[:, -1, [false_id, true_id]]
                scores = torch.softmax(logits.to(torch.float32), dim=1)[:, 1].cpu().tolist()
        except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
            raise JobExecutionError("RERANKING_FAILED", "Text reranking failed.") from exc
        _checkpoint(cancellation)
        if len(scores) != len(input_value["documents"]) or any(
            not math.isfinite(float(score)) or not 0 <= float(score) <= 1 for score in scores
        ):
            raise JobExecutionError("RERANKING_OUTPUT_INVALID", "Reranking output is invalid.")
        ranked = sorted(
            ({"index": index, "score": float(score)} for index, score in enumerate(scores)),
            key=lambda item: (-item["score"], item["index"]),
        )[: input_value["topK"]]
        emit("job.progress", {"stage": "finalizing-output", "progress": 1.0})
        return {"kind": "structured", "value": {"results": ranked}}

    def _generate_ace_step(
        self,
        job: EngineJob,
        handle: AceStepPackHandle,
        cancellation: threading.Event,
        emit: RunnerEvent,
    ) -> object:
        input_value = _song_input(job.input, ace_step=True)
        if self.outputs is None:
            raise JobExecutionError("ARTIFACT_IO_ERROR", "Artifact output staging is unavailable.")
        model = handle.record["model"]
        repository_directory = model["repository"].replace("/", "--")
        model_root = (
            Path(handle.record["cacheRoot"])
            / f"models--{repository_directory}"
            / "snapshots"
            / model["revision"]
        ).resolve()
        cache_root = Path(handle.record["cacheRoot"]).resolve()
        if not model_root.is_relative_to(cache_root) or not model_root.is_dir():
            raise JobExecutionError(
                "MODEL_LOAD_FAILED", "The ACE-Step model subset is unavailable."
            )

        with tempfile.TemporaryDirectory(prefix="oryn-ace-step-output-") as output_directory:
            output_path = Path(output_directory) / "generated-music.wav"
            result = self._invoke_pack(
                handle.record,
                handle.pack,
                handle.runtime,
                "generate",
                {
                    "modelRoot": str(model_root),
                    "outputPath": str(output_path),
                    "input": input_value,
                },
                cancellation=cancellation,
                event_sink=emit,
                timeout_seconds=2 * 60 * 60,
            )
            if (
                result.get("relativeOutput") != output_path.name
                or result.get("mediaType") != "audio/wav"
                or not output_path.is_file()
                or output_path.stat().st_size > MAX_AUDIO_BYTES
            ):
                raise JobExecutionError("PACK_OUTPUT_INVALID", "ACE-Step returned invalid audio.")
            with output_path.open("rb") as source:
                header = source.read(12)
                if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
                    raise JobExecutionError(
                        "PACK_OUTPUT_INVALID", "ACE-Step returned invalid audio."
                    )
                source.seek(0)
                with self.outputs.writer(job.job_id, "audio/wav", output_path.name) as writer:
                    while chunk := source.read(1024 * 1024):
                        _checkpoint(cancellation)
                        writer.write(chunk)
                    artifact = writer.finalize()
        emit("job.progress", {"stage": "publishing-artifact", "progress": 1.0})
        return {
            "kind": "artifact",
            "artifact": artifact.protocol_descriptor(),
        }

    def _generate_song(
        self,
        job: EngineJob,
        handle: TransformersMusicgenHandle,
        cancellation: threading.Event,
        emit: RunnerEvent,
    ) -> object:
        input_value = _song_input(job.input)
        _checkpoint(cancellation)
        emit("job.progress", {"stage": "encoding-prompt", "progress": 0.02})
        try:
            import torch
            from transformers import StoppingCriteria, StoppingCriteriaList

            model_inputs = _musicgen_inputs(handle, input_value["prompt"])
            audio_config = handle.model.config.audio_encoder
            sample_rate = int(audio_config.sampling_rate)
            ratios = tuple(int(value) for value in audio_config.upsampling_ratios)
            hop_length = math.prod(ratios)
            if not 8_000 <= sample_rate <= 192_000 or not 1 <= hop_length <= sample_rate:
                raise ValueError("invalid MusicGen audio configuration")
            frame_rate = sample_rate / hop_length
            max_tokens = max(1, min(1_503, math.ceil(input_value["durationSeconds"] * frame_rate)))
            initial_length: int | None = None
            last_progress = 0.02

            class ProgressCancellationCriteria(StoppingCriteria):
                def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                    nonlocal initial_length, last_progress
                    del scores, kwargs
                    length = int(input_ids.shape[-1])
                    initial_length = length if initial_length is None else initial_length
                    generated = max(0, length - initial_length)
                    progress = min(0.92, 0.05 + (generated / max_tokens) * 0.87)
                    if progress - last_progress >= 0.02:
                        last_progress = progress
                        emit("job.progress", {"stage": "generating-audio", "progress": progress})
                    return cancellation.is_set()

            emit("job.state", {"state": "running"})
            emit("job.progress", {"stage": "generating-audio", "progress": 0.05})
            with torch.inference_mode():
                audio = handle.model.generate(
                    **model_inputs,
                    do_sample=True,
                    guidance_scale=3.0,
                    max_new_tokens=max_tokens,
                    stopping_criteria=StoppingCriteriaList([ProgressCancellationCriteria()]),
                )
        except JobExecutionError:
            raise
        except Exception as exc:
            raise JobExecutionError("SONG_GENERATION_FAILED", "Music generation failed.") from exc

        _checkpoint(cancellation)
        if self.outputs is None:
            raise JobExecutionError("ARTIFACT_IO_ERROR", "Artifact output staging is unavailable.")
        emit("job.state", {"state": "saving"})
        emit("job.progress", {"stage": "writing-audio", "progress": 0.94})
        try:
            with self.outputs.writer(job.job_id, "audio/wav", "generated-song.wav") as writer:
                _write_pcm_wave(writer, audio, sample_rate, cancellation)
                artifact = writer.finalize()
        except JobExecutionError:
            raise
        except Exception as exc:
            raise JobExecutionError(
                "ARTIFACT_IO_ERROR", "Generated audio could not be staged."
            ) from exc
        _checkpoint(cancellation)
        emit("job.progress", {"stage": "publishing-artifact", "progress": 1.0})
        return {"kind": "artifact", "artifact": artifact.protocol_descriptor()}

    @staticmethod
    def _generate_chat(
        job: EngineJob,
        handle: TransformersCausalLMHandle,
        cancellation: threading.Event,
        emit: RunnerEvent,
    ) -> object:
        input_value = _chat_input(
            job.input, allow_image=handle.processor is not None, chat_v2=_chat_v2(job)
        )
        _checkpoint(cancellation)
        emit("job.progress", {"stage": "encoding"})
        from oryn_engine import chat_formats

        # The repository template is read as text to choose reviewed markers, never rendered.
        template = chat_formats.template_text(handle.processor, handle.tokenizer)
        chat_format = chat_formats.select(template) if template else "plain"
        if chat_format is None:
            raise JobExecutionError(
                "MODEL_NOT_EXECUTABLE", "This model's chat format is not in the reviewed list."
            )
        control, thinking_off = None, False
        if _chat_v2(job):
            from oryn_engine.chat_v2 import reasoning_prefix, reviewed_control

            control, thinking_off = reviewed_control(
                getattr(handle.model.config, "model_type", None),
                template or "",
                chat_format,
                input_value.get("thinking"),
            )
            job.chat_reasoning_prefix = reasoning_prefix(
                chat_format, thinking_off, thinking_on=control["effective"] is True
            )
        if "imageArtifactId" in input_value and not chat_formats.supports_image(
            chat_format, handle.tokenizer.get_vocab()
        ):
            raise JobExecutionError(
                "MODEL_NOT_EXECUTABLE", "This model's image placement is not in the reviewed list."
            )
        try:
            if handle.processor is not None:
                messages = input_value["messages"]
                image = None
                image_index = None
                if "imageArtifactId" in input_value:
                    image = _bounded_chat_image(job.input_artifacts, input_value["imageArtifactId"])
                    image_index = max(i for i, m in enumerate(messages) if m["role"] == "user")
                prompt = chat_formats.render(
                    chat_format, messages, image_index=image_index, thinking_off=thinking_off
                )
                if control and control["effective"] is True:
                    prompt += "<think>\n"
                encoded = handle.processor(
                    text=prompt, **({"images": [image]} if image else {}), return_tensors="pt"
                )
                model_inputs = {
                    key: value.to(handle.backend) if hasattr(value, "to") else value
                    for key, value in encoded.items()
                }
            else:
                prompt = chat_formats.render(
                    chat_format, input_value["messages"], thinking_off=thinking_off
                )
                if control and control["effective"] is True:
                    prompt += "<think>\n"
                encoded = handle.tokenizer(prompt, return_tensors="pt", truncation=False)
                model_inputs = {
                    key: value.to(handle.backend)
                    for key, value in encoded.items()
                    if key in {"input_ids", "attention_mask"}
                }
            text_config = getattr(handle.model.config, "text_config", handle.model.config)
            context_limit = min(int(getattr(text_config, "max_position_embeddings", 8192)), 8192)
            if int(encoded["input_ids"].shape[-1]) + input_value["maxTokens"] > context_limit:
                raise ValueError("Chat exceeds its admitted context limit")
            if "input_ids" not in model_inputs:
                raise ValueError("tokenizer omitted input_ids")
        except Exception as exc:
            raise JobExecutionError("CHAT_INPUT_ENCODING_FAILED", "Chat encoding failed.") from exc

        _checkpoint(cancellation)
        emit("job.state", {"state": "streaming"})
        emit("job.progress", {"stage": "generating"})
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

            class CancellationCriteria(StoppingCriteria):
                def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                    del input_ids, scores, kwargs
                    return cancellation.is_set()

            streamer = TextIteratorStreamer(
                handle.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=0.1,
            )
            generation_error: list[Exception] = []
            generated_tokens: list[Any] = []

            def generate() -> None:
                try:
                    options: dict[str, Any] = {
                        **model_inputs,
                        "streamer": streamer,
                        "max_new_tokens": input_value["maxTokens"],
                        "do_sample": input_value["temperature"] > 0,
                        "stopping_criteria": StoppingCriteriaList([CancellationCriteria()]),
                        "pad_token_id": handle.tokenizer.pad_token_id
                        or handle.tokenizer.eos_token_id,
                    }
                    if options["do_sample"]:
                        options.update(
                            {
                                "temperature": input_value["temperature"],
                                "top_p": input_value["topP"],
                            }
                        )
                    generated_tokens.append(handle.model.generate(**options))
                except Exception as exc:  # captured and re-raised on the dispatcher thread
                    generation_error.append(exc)
                    streamer.end()

            worker = threading.Thread(
                target=generate,
                name=f"oryn-chat-{job.job_id}",
                daemon=True,
            )
            worker.start()
            chunks: list[str] = []
            pending = ""
            last_emit = time.monotonic()
            cancellation_started: float | None = None
            iterator = iter(streamer)
            while True:
                try:
                    value = next(iterator)
                except queue.Empty:
                    if cancellation.is_set():
                        cancellation_started = cancellation_started or time.monotonic()
                        if not worker.is_alive() or time.monotonic() - cancellation_started >= 5:
                            break
                    continue
                except StopIteration:
                    break
                if value:
                    pending += str(value)
                now = time.monotonic()
                if pending and (
                    len(pending) >= 24
                    or pending.endswith((" ", "\n", ".", "!", "?"))
                    or now - last_emit >= 0.05
                ):
                    chunks.append(pending)
                    emit("job.output.delta", {"channel": "text", "delta": pending})
                    pending = ""
                    last_emit = now
            worker.join(timeout=5)
            if worker.is_alive():
                cancellation.set()
                raise JobExecutionError(
                    "GENERATION_INTERRUPTION_FAILED", "Text generation did not stop safely."
                )
            if generation_error:
                raise generation_error[0]
            if pending:
                chunks.append(pending)
                emit("job.output.delta", {"channel": "text", "delta": pending})
        except JobExecutionError:
            raise
        except Exception as exc:
            raise JobExecutionError("CHAT_GENERATION_FAILED", "Text generation failed.") from exc

        _checkpoint(cancellation)
        emit("job.progress", {"stage": "finalizing-output"})
        value = {"text": "".join(chunks)}
        if _chat_v2(job):
            count = int(generated_tokens[0].shape[-1]) - int(encoded["input_ids"].shape[-1])
            eos = handle.tokenizer.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            stopped = int(generated_tokens[0][0, -1]) in eos_ids
            value.update(
                finishReason="stop"
                if stopped
                else "length"
                if count >= input_value["maxTokens"]
                else "unknown",
                usage={
                    "inputTokens": int(encoded["input_ids"].shape[-1]),
                    "outputTokens": count,
                    "reasoningTokens": None,
                    "answerTokens": None,
                },
                metadata={"thinking": control},
            )
        return {"kind": "structured", "value": value}

    @staticmethod
    def _transcribe(
        job: EngineJob,
        handle: TransformersASRHandle,
        cancellation: threading.Event,
        emit: RunnerEvent,
    ) -> object:
        input_value = _asr_input(job.input)
        binding = _audio_binding(job.input_artifacts, input_value["audioArtifactId"])
        _checkpoint(cancellation)
        emit("job.progress", {"stage": "decoding-audio"})
        audio = _decode_wave(
            Path(binding["internalPath"]),
            int(binding["sizeBytes"]),
            max_seconds=30,
            max_decoded_bytes=MAX_TRANSCRIPTION_DECODED_BYTES,
        )

        _checkpoint(cancellation)
        emit("job.progress", {"stage": "preparing-inputs"})
        try:
            if handle.family == "qwen3-asr":
                requested_language = input_value.get("language")
                processed = handle.processor.apply_transcription_request(
                    audio=audio,
                    language=(None if requested_language in {None, "auto"} else requested_language),
                    sampling_rate=WHISPER_SAMPLE_RATE,
                    return_tensors="pt",
                )
                qwen_inputs = _move_inputs_to_model(dict(processed), handle.model, handle.backend)
                features = None
                attention_mask = None
            else:
                processed = handle.processor(
                    audio,
                    sampling_rate=WHISPER_SAMPLE_RATE,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                features = processed.input_features.to(handle.backend)
                attention_mask = getattr(processed, "attention_mask", None)
                qwen_inputs = {}
        except Exception as exc:
            raise JobExecutionError(
                "AUDIO_PROCESSING_FAILED", "Audio preprocessing failed."
            ) from exc

        _checkpoint(cancellation)
        emit("job.state", {"state": "running"})
        emit("job.progress", {"stage": "transcribing"})
        try:
            import torch
            from transformers import StoppingCriteria, StoppingCriteriaList

            class CancellationCriteria(StoppingCriteria):
                def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                    del input_ids, scores, kwargs
                    return cancellation.is_set()

            generate_options: dict[str, Any] = {
                "stopping_criteria": StoppingCriteriaList([CancellationCriteria()]),
                "max_new_tokens": (
                    512
                    if handle.family == "qwen3-asr"
                    else max(
                        1,
                        min(
                            440,
                            int(getattr(handle.model.config, "max_target_positions", 448)) - 8,
                        ),
                    )
                ),
            }
            if handle.family != "qwen3-asr" and attention_mask is not None:
                generate_options["attention_mask"] = attention_mask.to(handle.backend)
            language = input_value.get("language")
            if handle.family != "qwen3-asr" and isinstance(language, str) and language != "auto":
                generate_options.update({"language": language, "task": "transcribe"})
            if handle.family != "qwen3-asr" and input_value.get("timestamps") is True:
                generate_options["return_timestamps"] = True
                # Attention-derived token timing remains available for conformant
                # Whisper fine-tunes that do not emit timestamp tokens themselves.
                if _supports_whisper_attention_timestamps(handle.model):
                    generate_options["return_token_timestamps"] = True
                    if handle.backend == "mps":
                        # Transformers 5.17's deferred Metal stop check assumes
                        # cache.layers, which EncoderDecoderCache does not expose.
                        # A static cache keeps KV reuse and avoids that rollback
                        # path while retaining attention-derived word timestamps.
                        generate_options["cache_implementation"] = "static"
                        generate_options["disable_compile"] = True
            with torch.inference_mode():
                if handle.family == "qwen3-asr":
                    token_ids = handle.model.generate(**qwen_inputs, **generate_options)
                else:
                    token_ids = handle.model.generate(features, **generate_options)
        except Exception as exc:
            raise JobExecutionError("ASR_INFERENCE_FAILED", "Speech recognition failed.") from exc

        _checkpoint(cancellation)
        emit("job.progress", {"stage": "decoding-text"})
        try:
            if handle.family == "qwen3-asr":
                prompt_length = int(qwen_inputs["input_ids"].shape[1])
                generated = token_ids[:, prompt_length:]
                decoded = handle.processor.decode(generated[0], return_format="parsed")
                if not isinstance(decoded, dict):
                    raise ValueError("Qwen ASR returned an invalid result")
                text = str(decoded.get("transcription", "")).strip()
                detected_language = decoded.get("language")
                segments: list[dict[str, Any]] = []
            else:
                text, detected_language, segments = _decode_transcript(
                    handle.processor,
                    token_ids,
                    timestamps=input_value.get("timestamps") is True,
                )
        except Exception as exc:
            raise JobExecutionError(
                "ASR_OUTPUT_INVALID", "The transcript could not be decoded."
            ) from exc
        _checkpoint(cancellation)
        return {
            "kind": "structured",
            "value": {
                "text": text,
                "language": (
                    detected_language
                    if input_value.get("language") == "auto"
                    else input_value.get("language")
                ),
                **({"segments": segments} if input_value.get("timestamps") is True else {}),
            },
        }

    @staticmethod
    def _force_align(
        job: EngineJob,
        handle: TransformersForcedAlignmentHandle,
        cancellation: threading.Event,
        emit: RunnerEvent,
    ) -> object:
        value = _forced_alignment_input(job.input)
        binding = _audio_binding(job.input_artifacts, value["audioArtifactId"])
        _checkpoint(cancellation)
        emit("job.progress", {"stage": "decoding-audio"})
        audio = _decode_wave(Path(binding["internalPath"]), int(binding["sizeBytes"]))
        emit("job.progress", {"stage": "preparing-alignment"})
        try:
            model_inputs, word_lists = handle.processor.prepare_forced_aligner_inputs(
                audio=audio,
                transcript=value["transcript"],
                language=value["language"],
                sampling_rate=WHISPER_SAMPLE_RATE,
                return_tensors="pt",
            )
            runtime_inputs = _move_inputs_to_model(dict(model_inputs), handle.model, handle.backend)
        except Exception as exc:
            raise JobExecutionError(
                "ALIGNMENT_LANGUAGE_UNSUPPORTED",
                "The forced aligner cannot prepare this transcript and language.",
            ) from exc
        _checkpoint(cancellation)
        emit("job.state", {"state": "running"})
        emit("job.progress", {"stage": "aligning-transcript"})
        try:
            import torch

            with torch.inference_mode():
                logits = handle.model(**runtime_inputs).logits
            decoded = handle.processor.decode_forced_alignment(
                logits=logits,
                input_ids=model_inputs["input_ids"],
                word_lists=word_lists,
                timestamp_token_id=handle.model.config.timestamp_token_id,
            )
            segments = _forced_alignment_segments(decoded, value["timestamps"])
        except JobExecutionError:
            raise
        except Exception as exc:
            raise JobExecutionError(
                "ALIGNMENT_OUTPUT_INVALID", "The forced-alignment result is invalid."
            ) from exc
        _checkpoint(cancellation)
        emit("job.progress", {"stage": "normalizing-alignment"})
        return {"kind": "structured", "value": {"segments": segments}}

    @staticmethod
    def _run_transformers_task(
        job: EngineJob,
        handle: TransformersTaskHandle,
        cancellation: threading.Event,
        emit: RunnerEvent,
    ) -> object:
        """Execute one closed, Oryn-owned task adapter over a sealed AutoModel family."""

        value = _closed_task_input(handle.task, job.input)
        token_metadata: dict[str, Any] = {}
        _checkpoint(cancellation)
        emit("job.progress", {"stage": "preparing-inputs"})
        try:
            import torch

            if handle.task == "audio-classification":
                binding = _audio_binding(job.input_artifacts, value["audioArtifactId"])
                audio = _decode_wave(Path(binding["internalPath"]), int(binding["sizeBytes"]))
                processed = handle.processor(
                    audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt"
                )
            elif handle.task in {
                "depth-estimation",
                "image-classification",
                "object-detection",
                "zero-shot-image-classification",
            }:
                binding = _image_binding(job.input_artifacts, value["imageArtifactId"])
                from PIL import Image

                with Image.open(binding["internalPath"]) as source:
                    source.load()
                    if source.width * source.height > 16_777_216:
                        raise ValueError("image exceeds pixel bound")
                    image = source.convert("RGB")
                if handle.task == "zero-shot-image-classification":
                    siglip = handle.model.config.model_type in {"siglip", "siglip2"}
                    processed = handle.processor(
                        text=[f"This is a photo of {label}." for label in value["labels"]]
                        if siglip
                        else value["labels"],
                        images=image,
                        return_tensors="pt",
                        **(
                            {"padding": "max_length", "max_length": 64, "truncation": False}
                            if siglip
                            else {"padding": True}
                        ),
                    )
                    if siglip and processed["input_ids"].shape[-1] != 64:
                        raise ValueError("SigLIP text exceeds its native token bound")
                else:
                    processed = handle.processor(images=image, return_tensors="pt")
            elif handle.task == "question-answering":
                processed = handle.processor(
                    value["question"],
                    value["context"],
                    return_tensors="pt",
                    truncation="only_second",
                    max_length=512,
                    return_offsets_mapping=True,
                )
                token_metadata = {
                    "offsets": processed.pop("offset_mapping")[0].tolist(),
                    "sequence_ids": processed.sequence_ids(),
                }
            else:
                processed = handle.processor(
                    value["text"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    **(
                        {"return_offsets_mapping": True}
                        if handle.task == "token-classification"
                        else {}
                    ),
                )
                if handle.task == "token-classification":
                    token_metadata = {"offsets": processed.pop("offset_mapping")[0].tolist()}
            inputs = {
                key: item.to(handle.backend) if hasattr(item, "to") else item
                for key, item in dict(processed).items()
            }
        except JobExecutionError:
            raise
        except Exception as exc:
            raise JobExecutionError(
                "TASK_INPUT_PROCESSING_FAILED", "The bounded model input could not be prepared."
            ) from exc

        _checkpoint(cancellation)
        emit("job.state", {"state": "running"})
        emit("job.progress", {"stage": "running-model"})
        try:
            with torch.inference_mode():
                if handle.task == "seq2seq":
                    from transformers import StoppingCriteria, StoppingCriteriaList

                    class CancelGeneration(StoppingCriteria):
                        def __call__(self, _input_ids, _scores, **_kwargs):
                            return cancellation.is_set()

                    generated = handle.model.generate(
                        **inputs,
                        max_new_tokens=value["maxTokens"],
                        do_sample=False,
                        stopping_criteria=StoppingCriteriaList([CancelGeneration()]),
                    )
                    result: object = {
                        "text": handle.processor.batch_decode(generated, skip_special_tokens=True)[
                            0
                        ].strip()
                    }
                else:
                    output = handle.model(**inputs)
                    result = _normalize_transformers_task_output(
                        handle.task,
                        value,
                        output,
                        handle.processor,
                        handle.model,
                        inputs,
                        image.size if "image" in locals() else None,
                        token_metadata,
                    )
        except Exception as exc:
            raise JobExecutionError("TASK_INFERENCE_FAILED", "The model task failed.") from exc
        _checkpoint(cancellation)
        emit("job.progress", {"stage": "normalizing-output"})
        return {"kind": "structured", "value": result}


def _closed_task_input(task: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise JobExecutionError("JOB_INPUT_INVALID", "The task input is invalid.")
    allowed: dict[str, set[str]] = {
        "text-classification": {"text", "topK"},
        "token-classification": {"text"},
        "question-answering": {"question", "context", "maxAnswerLength"},
        "fill-mask": {"text", "topK"},
        "seq2seq": {"text", "maxTokens"},
        "image-classification": {"imageArtifactId", "topK"},
        "audio-classification": {"audioArtifactId", "topK"},
        "object-detection": {"imageArtifactId", "threshold"},
        "depth-estimation": {"imageArtifactId"},
        "zero-shot-image-classification": {"imageArtifactId", "labels"},
    }
    if task not in allowed or set(value) - allowed[task]:
        raise JobExecutionError("JOB_INPUT_INVALID", "The task input is invalid.")
    result = dict(value)
    if task in {"text-classification", "token-classification", "fill-mask", "seq2seq"}:
        if not isinstance(value.get("text"), str) or not 1 <= len(value["text"].strip()) <= 32_768:
            raise JobExecutionError("JOB_INPUT_INVALID", "The text input is invalid.")
        result["text"] = value["text"].strip()
    if task == "question-answering":
        if (
            not isinstance(value.get("question"), str)
            or not 1 <= len(value["question"].strip()) <= 4_096
            or not isinstance(value.get("context"), str)
            or not 1 <= len(value["context"].strip()) <= 65_536
        ):
            raise JobExecutionError("JOB_INPUT_INVALID", "The question input is invalid.")
        result["question"] = value["question"].strip()
        result["context"] = value["context"].strip()
        result["maxAnswerLength"] = _bounded_task_integer(value.get("maxAnswerLength", 64), 1, 128)
    if task in {"text-classification", "fill-mask", "image-classification", "audio-classification"}:
        result["topK"] = _bounded_task_integer(value.get("topK", 5), 1, 20)
    if task == "seq2seq":
        result["maxTokens"] = _bounded_task_integer(value.get("maxTokens", 256), 1, 512)
    if task in {
        "image-classification",
        "object-detection",
        "depth-estimation",
        "zero-shot-image-classification",
    } and not isinstance(value.get("imageArtifactId"), str):
        raise JobExecutionError("JOB_INPUT_INVALID", "The image binding is invalid.")
    if task == "audio-classification" and not isinstance(value.get("audioArtifactId"), str):
        raise JobExecutionError("JOB_INPUT_INVALID", "The audio binding is invalid.")
    if task == "object-detection":
        threshold = value.get("threshold", 0.5)
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not 0 <= threshold <= 1
        ):
            raise JobExecutionError("JOB_INPUT_INVALID", "The detection threshold is invalid.")
        result["threshold"] = float(threshold)
    if task == "zero-shot-image-classification":
        labels = value.get("labels")
        if (
            not isinstance(labels, list)
            or not 2 <= len(labels) <= 64
            or any(
                not isinstance(item, str) or not 1 <= len(item.strip()) <= 128 for item in labels
            )
        ):
            raise JobExecutionError("JOB_INPUT_INVALID", "The candidate labels are invalid.")
        result["labels"] = [item.strip() for item in labels]
    return result


def _bounded_task_integer(value: object, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise JobExecutionError("JOB_INPUT_INVALID", "The task option is invalid.")
    return value


def _normalize_transformers_task_output(
    task: str,
    value: dict[str, Any],
    output: Any,
    processor: Any,
    model: Any,
    inputs: dict[str, Any],
    image_size: tuple[int, int] | None,
    token_metadata: dict[str, Any] | None = None,
) -> object:
    import torch

    id2label = getattr(getattr(model, "config", None), "id2label", {})
    if task in {"text-classification", "image-classification", "audio-classification"}:
        logits = output.logits[0].float()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Classifier returned non-finite scores")
        problem = getattr(getattr(model, "config", None), "problem_type", None)
        probabilities = (
            torch.sigmoid(logits)
            if logits.numel() == 1 or problem == "multi_label_classification"
            else torch.softmax(logits, dim=-1)
        )
        scores, indices = torch.topk(probabilities, min(value["topK"], probabilities.numel()))
        return {
            "predictions": [
                {
                    "label": str(id2label.get(int(index), int(index))),
                    "labelId": int(index),
                    "score": float(score),
                }
                for score, index in zip(scores.cpu(), indices.cpu(), strict=True)
            ]
        }
    if task == "token-classification":
        probabilities = torch.softmax(output.logits[0].float(), dim=-1)
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError("Token classifier returned non-finite scores")
        scores, labels = probabilities.max(dim=-1)
        ids = labels.cpu().tolist()
        token_ids = inputs["input_ids"][0].cpu().tolist()
        tokens = processor.convert_ids_to_tokens(token_ids)
        offsets = (token_metadata or {}).get("offsets")
        specials = set(getattr(processor, "all_special_ids", []))
        return {
            "tokens": [
                {
                    "token": str(token),
                    "labelId": int(label),
                    "label": str(id2label.get(int(label), int(label))),
                    "score": float(scores[index]),
                    **({"start": offsets[index][0], "end": offsets[index][1]} if offsets else {}),
                }
                for index, (token, label) in enumerate(zip(tokens, ids, strict=True))
                if token_ids[index] not in specials
                and (not offsets or offsets[index][1] > offsets[index][0])
            ]
        }
    if task == "question-answering":
        metadata = token_metadata or {}
        offsets, sequences = metadata.get("offsets"), metadata.get("sequence_ids")
        if not offsets or not sequences:
            raise ValueError("Question answering requires context token offsets")
        starts = output.start_logits[0].float().cpu()
        ends = output.end_logits[0].float().cpu()
        if not bool(torch.isfinite(starts).all() and torch.isfinite(ends).all()):
            raise ValueError("Question answering returned non-finite scores")
        eligible = [
            i for i, seq in enumerate(sequences) if seq == 1 and offsets[i][1] > offsets[i][0]
        ]
        spans = [
            (float(starts[start] + ends[end]), start, end)
            for start in eligible
            for end in eligible
            if start <= end < start + value["maxAnswerLength"]
        ]
        if not spans:
            raise ValueError("No bounded context answer is available")
        _, start, end = max(spans, key=lambda span: (span[0], -span[1], -span[2]))
        character_start, character_end = offsets[start][0], offsets[end][1]
        return {
            "answer": value["context"][character_start:character_end],
            "startToken": start,
            "endToken": end,
            "start": character_start,
            "end": character_end,
        }
    if task == "fill-mask":
        mask_id = getattr(processor, "mask_token_id", None)
        if mask_id is None:
            raise ValueError("the tokenizer does not define a mask token")
        positions = (inputs["input_ids"][0] == mask_id).nonzero(as_tuple=False).flatten()
        if positions.numel() != 1:
            raise ValueError("exactly one mask token is required")
        probabilities = torch.softmax(output.logits[0, int(positions[0])].float(), dim=-1)
        scores, indices = torch.topk(probabilities, min(value["topK"], probabilities.numel()))
        return {
            "predictions": [
                {"token": processor.decode([int(index)]).strip(), "score": float(score)}
                for score, index in zip(scores.cpu(), indices.cpu(), strict=True)
            ]
        }
    if task == "zero-shot-image-classification":
        logits = getattr(output, "logits_per_image", None)
        if logits is None:
            logits = output.logits
        if (
            logits.ndim != 2
            or tuple(logits.shape) != (1, len(value["labels"]))
            or not bool(torch.isfinite(logits).all())
        ):
            raise ValueError("Image/text classifier returned invalid scores")
        logits = logits[0].float()
        scores = (
            (
                torch.sigmoid(logits)
                if model.config.model_type in {"siglip", "siglip2"}
                else torch.softmax(logits, dim=-1)
            )
            .cpu()
            .tolist()
        )
        return {
            "predictions": sorted(
                [
                    {"label": label, "score": float(score)}
                    for label, score in zip(value["labels"], scores, strict=True)
                ],
                key=lambda p: -p["score"],
            )
        }
    if task == "object-detection":
        if image_size is None:
            raise ValueError("image size is missing")
        target = torch.tensor([[image_size[1], image_size[0]]], device=output.logits.device)
        normalized = processor.post_process_object_detection(
            output, threshold=value["threshold"], target_sizes=target
        )[0]
        return {
            "detections": [
                {
                    "labelId": int(label),
                    "score": float(score),
                    "box": [round(float(item), 3) for item in box],
                }
                for score, label, box in list(
                    zip(
                        normalized["scores"],
                        normalized["labels"],
                        normalized["boxes"],
                        strict=True,
                    )
                )[:256]
            ]
        }
    if task == "depth-estimation":
        depth = output.predicted_depth[0].float().cpu()
        return {
            "width": int(depth.shape[-1]),
            "height": int(depth.shape[-2]),
            "minimum": float(depth.min()),
            "maximum": float(depth.max()),
            "mean": float(depth.mean()),
        }
    raise ValueError("task output is unsupported")


def _song_input(value: object, *, ace_step: bool = False) -> dict[str, Any]:
    allowed = {"prompt", "durationSeconds"}
    if ace_step:
        allowed.update({"lyrics", "seed"})
    if not isinstance(value, dict) or set(value) - allowed:
        raise JobExecutionError("JOB_INPUT_INVALID", "Song input is invalid.")
    prompt = value.get("prompt")
    duration = value.get("durationSeconds", 10)
    if (
        not isinstance(prompt, str)
        or not 1 <= len(prompt.strip()) <= 2_000
        or not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(float(duration))
        or not (10 if ace_step else 1) <= float(duration) <= (180 if ace_step else 30)
    ):
        raise JobExecutionError("JOB_INPUT_INVALID", "Song input is invalid.")
    result: dict[str, Any] = {"prompt": prompt.strip(), "durationSeconds": float(duration)}
    if ace_step:
        lyrics = value.get("lyrics", "[Instrumental]")
        seed = value.get("seed", -1)
        if (
            not isinstance(lyrics, str)
            or len(lyrics) > 12_000
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or not -1 <= seed <= 2**32 - 1
        ):
            raise JobExecutionError("JOB_INPUT_INVALID", "Song input is invalid.")
        result.update({"lyrics": lyrics, "seed": seed})
    return result


def _musicgen_inputs(handle: TransformersMusicgenHandle, prompt: str) -> dict[str, Any]:
    import torch

    vocab_size = int(handle.model.config.text_encoder.vocab_size)
    processed = handle.processor(text=[prompt], padding=True, return_tensors="pt")
    input_ids = processed.get("input_ids")
    attention_mask = processed.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or input_ids.numel() == 0:
        raise ValueError("MusicGen processor omitted input IDs")
    if int(input_ids.min()) < 0 or int(input_ids.max()) >= vocab_size:
        parameter_count = sum(int(parameter.numel()) for parameter in handle.model.parameters())
        if not (4 <= vocab_size <= 128 and parameter_count <= 1_000_000):
            raise ValueError("MusicGen tokenizer does not match the text encoder")
        # Bounded synthetic conformance checkpoints can intentionally pair a generated tokenizer
        # with a deliberately tiny text encoder. Keep that format-level fixture path deterministic
        # without granting repository identity any execution authority. Both the encoder vocabulary
        # and the complete parameter count are capped so ordinary user models cannot enter it.
        values = [2 + (value % (vocab_size - 2)) for value in prompt.encode("utf-8")[:64]]
        input_ids = torch.tensor([values or [2]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
    if not isinstance(attention_mask, torch.Tensor):
        attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.to(handle.backend),
        "attention_mask": attention_mask.to(handle.backend),
    }


def _write_pcm_wave(
    writer: Any,
    audio: Any,
    sample_rate: int,
    cancellation: threading.Event,
) -> None:
    import torch

    if not isinstance(audio, torch.Tensor) or audio.ndim != 3 or audio.shape[0] != 1:
        raise ValueError("MusicGen output shape is invalid")
    source_channels = int(audio.shape[1])
    frames = int(audio.shape[2])
    if source_channels not in {1, 2} or frames <= 0:
        raise ValueError("MusicGen audio dimensions are invalid")
    # Chromium consistently decodes 16-bit PCM WAV. Mono generations are
    # represented as dual-mono so long-form output remains a realistic large,
    # seekable binary artifact without renderer-side expansion.
    channels = 2
    bytes_per_sample = 2
    data_size = channels * frames * bytes_per_sample
    if data_size > 0xFFFFFFFF - 36:
        raise ValueError("MusicGen audio is too large for a WAV artifact")
    writer.write(
        b"RIFF"
        + struct.pack("<I", 36 + data_size)
        + b"WAVEfmt "
        + struct.pack(
            "<IHHIIHH",
            16,
            1,
            channels,
            sample_rate,
            sample_rate * channels * bytes_per_sample,
            channels * bytes_per_sample,
            bytes_per_sample * 8,
        )
        + b"data"
        + struct.pack("<I", data_size)
    )
    source = torch.nan_to_num(audio[0].detach().to("cpu", dtype=torch.float32)).clamp(-1, 1)
    if source_channels == 1:
        source = source.repeat(2, 1)
    block_frames = max(1, sample_rate)
    for offset in range(0, frames, block_frames):
        _checkpoint(cancellation)
        block = source[:, offset : offset + block_frames].transpose(0, 1).contiguous()
        pcm = (block * 32_767).round().to(torch.int16).numpy().astype("<i2", copy=False)
        writer.write(pcm.tobytes())


def _asr_input(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise JobExecutionError("JOB_INPUT_INVALID", "ASR input must be an object.")
    if set(value) - {"audioArtifactId", "language", "timestamps"}:
        raise JobExecutionError("JOB_INPUT_INVALID", "ASR input contains unsupported fields.")
    artifact_id = value.get("audioArtifactId")
    language = value.get("language", "auto")
    timestamps = value.get("timestamps", False)
    if (
        not isinstance(artifact_id, str)
        or not isinstance(language, str)
        or not 1 <= len(language) <= 32
        or not isinstance(timestamps, bool)
    ):
        raise JobExecutionError("JOB_INPUT_INVALID", "ASR input is invalid.")
    return {"audioArtifactId": artifact_id, "language": language, "timestamps": timestamps}


def _embedding_input(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - {"texts", "normalize"}:
        raise JobExecutionError("JOB_INPUT_INVALID", "Embedding input is invalid.")
    texts = value.get("texts")
    normalize = value.get("normalize", True)
    if (
        not isinstance(texts, list)
        or not 1 <= len(texts) <= 64
        or not isinstance(normalize, bool)
        or any(not isinstance(item, str) or not 1 <= len(item.strip()) <= 8_192 for item in texts)
        or sum(len(item) for item in texts) > 65_536
    ):
        raise JobExecutionError("JOB_INPUT_INVALID", "Embedding input is invalid.")
    return {"texts": [item.strip() for item in texts], "normalize": normalize}


def _speech_input(value: object) -> dict[str, Any]:
    allowed = {"text", "voice", "language", "speed"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise JobExecutionError("JOB_INPUT_INVALID", "Speech-synthesis input is invalid.")
    text = value.get("text")
    voice = value.get("voice", "af_heart")
    language = value.get("language", "en-us")
    speed = value.get("speed", 1.0)
    voice_languages = {
        "af_heart": "en-us",
        "af_bella": "en-us",
        "af_nicole": "en-us",
        "af_sarah": "en-us",
        "am_adam": "en-us",
        "am_michael": "en-us",
        "bf_emma": "en-gb",
        "bf_isabella": "en-gb",
        "bm_george": "en-gb",
        "bm_lewis": "en-gb",
    }
    if (
        not isinstance(text, str)
        or not 1 <= len(text.strip()) <= 350
        or voice not in voice_languages
        or language != voice_languages.get(voice)
        or not isinstance(speed, (int, float))
        or isinstance(speed, bool)
        or not math.isfinite(float(speed))
        or not 0.5 <= float(speed) <= 2.0
    ):
        raise JobExecutionError("JOB_INPUT_INVALID", "Speech-synthesis input is invalid.")
    return {
        "text": text.strip(),
        "voice": voice,
        "language": language,
        "speed": float(speed),
    }


def _diffusion_input(value: object, contract: object) -> dict[str, Any]:
    if contract == "text-to-music-v1":
        from oryn_engine.minimax_music import validate_input

        try:
            return validate_input(value)
        except (ValueError, TypeError) as exc:
            raise JobExecutionError("JOB_INPUT_INVALID", "Music input is invalid.") from exc
    if contract == "image-text-to-video-v1":
        from oryn_protocol.task_contracts import effective_input, make_profile, validate_invocation

        try:
            profile = make_profile(contract)
            validate_invocation(profile, value)
            return effective_input(profile, value)
        except ValueError as exc:
            raise JobExecutionError("JOB_INPUT_INVALID", "Video input is invalid.") from exc
    if contract == "unconditional-image-generation-v1":
        from oryn_engine.runtime_packs.diffusers_worker import _validated_input

        try:
            return _validated_input(value, contract)
        except ValueError as exc:
            raise JobExecutionError("JOB_INPUT_INVALID", "Unconditional input is invalid.") from exc
    from oryn_engine.diffusion_tasks import MEDIA_CONTRACTS, validate_media_input

    if isinstance(contract, str) and contract in MEDIA_CONTRACTS:
        try:
            return validate_media_input(contract, value)
        except ValueError as exc:
            raise JobExecutionError("JOB_INPUT_INVALID", "Image task input is invalid.") from exc
    contracts = {
        "text-to-image-v1",
        "image-to-image-v1",
        "text-to-video-v1",
        "image-to-video-v1",
    }
    if contract not in contracts or not isinstance(value, dict):
        raise JobExecutionError("JOB_INPUT_INVALID", "Diffusion input is invalid.")
    is_video = contract in {"text-to-video-v1", "image-to-video-v1"}
    needs_image = contract in {"image-to-image-v1", "image-to-video-v1"}
    allowed = {
        "prompt",
        "negativePrompt",
        "width",
        "height",
        "steps",
        "guidanceScale",
        "seed",
    }
    if is_video:
        allowed.update({"frames", "fps"})
    if needs_image:
        allowed.add("imageArtifactId")
    if contract == "image-to-image-v1":
        allowed.add("strength")
    if set(value) - allowed:
        raise JobExecutionError("JOB_INPUT_INVALID", "Diffusion input contains unsupported fields.")

    prompt = value.get("prompt")
    negative_prompt = value.get("negativePrompt", "")
    width = value.get("width", 512)
    height = value.get("height", 512)
    steps = value.get("steps", 20)
    guidance = value.get("guidanceScale", 7.5)
    seed = value.get("seed", 0)
    maximum_side = 1_280 if is_video else 2_048
    maximum_pixels = 1_280 * 720 if is_video else 2_048 * 2_048
    if (
        not isinstance(prompt, str)
        or not 1 <= len(prompt.strip()) <= 4_096
        or not isinstance(negative_prompt, str)
        or len(negative_prompt) > 4_096
        or not isinstance(width, int)
        or isinstance(width, bool)
        or not 64 <= width <= maximum_side
        or width % 8 != 0
        or not isinstance(height, int)
        or isinstance(height, bool)
        or not 64 <= height <= maximum_side
        or height % 8 != 0
        or width * height > maximum_pixels
        or not isinstance(steps, int)
        or isinstance(steps, bool)
        or not 1 <= steps <= 100
        or not isinstance(guidance, (int, float))
        or isinstance(guidance, bool)
        or not math.isfinite(float(guidance))
        or not 0 <= float(guidance) <= 30
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed <= 2**32 - 1
    ):
        raise JobExecutionError("JOB_INPUT_INVALID", "Diffusion input is invalid.")
    result: dict[str, Any] = {
        "prompt": prompt.strip(),
        "negativePrompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "guidanceScale": float(guidance),
        "seed": seed,
    }
    if is_video:
        frames = value.get("frames", 16)
        fps = value.get("fps", 8)
        if (
            not isinstance(frames, int)
            or isinstance(frames, bool)
            or not 1 <= frames <= 120
            or not isinstance(fps, int)
            or isinstance(fps, bool)
            or not 1 <= fps <= 60
            or frames / fps > 20
            or width * height * frames > 96 * 1024 * 1024
        ):
            raise JobExecutionError("JOB_INPUT_INVALID", "Diffusion video bounds are invalid.")
        result.update({"frames": frames, "fps": fps})
    if needs_image:
        artifact_id = value.get("imageArtifactId")
        if not isinstance(artifact_id, str) or not re.fullmatch(
            r"artifact_[A-Za-z0-9_-]{20,80}", artifact_id
        ):
            raise JobExecutionError("JOB_INPUT_INVALID", "Diffusion image input is invalid.")
        result["imageArtifactId"] = artifact_id
    if contract == "image-to-image-v1":
        strength = value.get("strength", 0.8)
        if (
            not isinstance(strength, (int, float))
            or isinstance(strength, bool)
            or not math.isfinite(float(strength))
            or not 0.05 <= float(strength) <= 1
        ):
            raise JobExecutionError(
                "JOB_INPUT_INVALID", "Image transformation strength is invalid."
            )
        result["strength"] = float(strength)
    return result


def _reranking_input(value: object) -> dict[str, Any]:
    allowed = {"query", "documents", "instruction", "topK"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise JobExecutionError("JOB_INPUT_INVALID", "Reranking input is invalid.")
    query = value.get("query")
    documents = value.get("documents")
    instruction = value.get(
        "instruction", "Given a web search query, retrieve relevant passages that answer the query"
    )
    top_k = value.get("topK", len(documents) if isinstance(documents, list) else 0)
    if (
        not isinstance(query, str)
        or not 1 <= len(query.strip()) <= 8_192
        or not isinstance(documents, list)
        or not 1 <= len(documents) <= 64
        or any(
            not isinstance(document, str) or not 1 <= len(document.strip()) <= 16_384
            for document in documents
        )
        or sum(len(document) for document in documents) + len(query) > 262_144
        or not isinstance(instruction, str)
        or not 1 <= len(instruction.strip()) <= 2_048
        or not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or not 1 <= top_k <= len(documents)
    ):
        raise JobExecutionError("JOB_INPUT_INVALID", "Reranking input is invalid.")
    return {
        "query": query.strip(),
        "documents": [document.strip() for document in documents],
        "instruction": instruction.strip(),
        "topK": top_k,
    }


def _chat_v2(job):
    return (job.task_profile or {}).get("contract", {}).get("id") == "chat-v2"


def _chat_input(
    value: object, *, allow_image: bool = False, chat_v2: bool = False
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - (
        {"imageArtifactId"} if allow_image else set()
    ) - ({"thinking"} if chat_v2 else set()) - {
        "messages",
        "maxTokens",
        "temperature",
        "topP",
    }:
        raise JobExecutionError("JOB_INPUT_INVALID", "Chat input is invalid.")
    messages = value.get("messages")
    max_tokens = value.get("maxTokens", 4096 if chat_v2 else 128)
    temperature = value.get("temperature", 0.7)
    top_p = value.get("topP", 0.9)
    if (
        not isinstance(messages, list)
        or not 1 <= len(messages) <= 128
        or not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or not 1 <= max_tokens <= (8192 if chat_v2 else 512)
        or ("thinking" in value and type(value["thinking"]) is not bool)
        or not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
        or not 0 <= temperature <= 2
        or not isinstance(top_p, (int, float))
        or isinstance(top_p, bool)
        or not math.isfinite(float(top_p))
        or not 0 < top_p <= 1
    ):
        raise JobExecutionError("JOB_INPUT_INVALID", "Chat input is invalid.")
    normalized: list[dict[str, str]] = []
    total_characters = 0
    for message in messages:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            or not message["content"]
            or len(message["content"]) > 32_768
        ):
            raise JobExecutionError("JOB_INPUT_INVALID", "Chat messages are invalid.")
        total_characters += len(message["content"])
        normalized.append({"role": message["role"], "content": message["content"]})
    if total_characters > 65_536 or not any(item["role"] == "user" for item in normalized):
        raise JobExecutionError("JOB_INPUT_INVALID", "Chat messages are invalid.")
    image = value.get("imageArtifactId")
    if image is not None and (
        not isinstance(image, str) or not re.fullmatch(r"artifact_[A-Za-z0-9_-]{20,80}", image)
    ):
        raise JobExecutionError("JOB_INPUT_INVALID", "Image reference is invalid.")
    return {
        "messages": normalized,
        **({"thinking": value["thinking"]} if "thinking" in value else {}),
        **({"imageArtifactId": image} if image is not None else {}),
        "maxTokens": max_tokens,
        "temperature": float(temperature),
        "topP": float(top_p),
    }


def _audio_binding(values: tuple[object, ...], artifact_id: str) -> dict[str, Any]:
    matches = [
        value
        for value in values
        if isinstance(value, dict) and value.get("artifactId") == artifact_id
    ]
    if len(matches) != 1:
        raise JobExecutionError("ARTIFACT_BINDING_INVALID", "The audio artifact is not bound.")
    binding = matches[0]
    if binding.get("mimeType") not in {"audio/wav", "audio/x-wav", "audio/wave"}:
        raise JobExecutionError("AUDIO_FORMAT_UNSUPPORTED", "Only PCM WAV audio is supported.")
    return binding


def _image_binding(values: tuple[object, ...], artifact_id: str) -> dict[str, Any]:
    matches = [
        value
        for value in values
        if isinstance(value, dict) and value.get("artifactId") == artifact_id
    ]
    if len(matches) != 1:
        raise JobExecutionError("ARTIFACT_BINDING_INVALID", "The image artifact is not bound.")
    binding = matches[0]
    if binding.get("mimeType") not in {"image/png", "image/jpeg", "image/webp"}:
        raise JobExecutionError(
            "IMAGE_FORMAT_UNSUPPORTED", "The image artifact format is unsupported."
        )
    size = binding.get("sizeBytes")
    path = binding.get("internalPath")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or not 1 <= size <= 256 * 1024 * 1024
        or not isinstance(path, str)
    ):
        raise JobExecutionError("ARTIFACT_BINDING_INVALID", "The image artifact is invalid.")
    return binding


def _validate_mask_metadata(path: Path, metadata: object) -> None:
    """Bind worker mask semantics to the PNG before publishing either output."""
    import numpy as np
    from PIL import Image

    try:
        if not isinstance(metadata, dict):
            raise ValueError("Missing mask metadata")
        with Image.open(path) as image:
            expected_mode = {"binary-8bit": "L", "class-id-16bit": "I;16"}.get(
                metadata.get("encoding")
            )
            if expected_mode is None or image.mode != expected_mode:
                raise ValueError("Mask encoding differs")
            if image.size != (metadata.get("width"), metadata.get("height")):
                raise ValueError("Mask geometry differs")
            ids, counts = np.unique(np.asarray(image), return_counts=True)
        declared = metadata.get("segments")
        if not isinstance(declared, list) or len(declared) != len(ids):
            raise ValueError("Mask labels differ")
        actual = {int(i): int(n) for i, n in zip(ids, counts, strict=True)}
        if (
            any(not isinstance(s, dict) for s in declared)
            or {s.get("id"): s.get("pixelCount") for s in declared} != actual
        ):
            raise ValueError("Mask pixel counts differ")
    except (ValueError, TypeError, OSError) as exc:
        raise JobExecutionError(
            "PACK_OUTPUT_INVALID", "Mask metadata differs from its pixels."
        ) from exc


def _validate_mlx_audio_speech(path: Path, result: dict[str, Any], backend: str) -> None:
    try:
        rate, frames = result.get("sampleRate"), result.get("frames")
        if (
            path.is_symlink()
            or not path.is_file()
            or result.get("backend") != backend
            or type(rate) is not int
            or not 8000 <= rate <= 192000
            or result.get("channels") != 1
            or type(frames) is not int
            or not 1 <= frames <= rate * 30
            or type(result.get("durationSeconds")) not in {int, float}
            or not math.isfinite(result["durationSeconds"])
            or abs(result["durationSeconds"] - frames / rate) > 1e-9
            or path.stat().st_size > 12 * 1024**2
        ):
            raise ValueError("Invalid MLX-Audio speech metadata")
        with wave.open(str(path), "rb") as audio:
            if (
                audio.getcomptype() != "NONE"
                or audio.getsampwidth() != 2
                or audio.getframerate() != rate
                or audio.getnchannels() != 1
                or audio.getnframes() != frames
                or len(audio.readframes(frames)) != 2 * frames
            ):
                raise ValueError("Speech waveform differs from metadata")
    except (ValueError, TypeError, KeyError, OSError, EOFError, wave.Error) as exc:
        raise JobExecutionError("PACK_OUTPUT_INVALID", "Speech output is invalid.") from exc


def _validate_synthesized_audio(path: Path, result: dict[str, Any], backend: str) -> None:
    try:
        from oryn_engine.qwen_tts import MAX_CODES

        if (
            path.is_symlink()
            or result.get("backend") != backend
            or result.get("sampleRate") != 24000
            or result.get("channels") != 1
            or type(result.get("frames")) is not int
            or not 1 <= result["frames"] <= MAX_CODES * 1920
            or type(result.get("durationSeconds")) not in {float, int}
            or abs(result["durationSeconds"] - result["frames"] / 24000) > 1e-9
        ):
            raise ValueError("Invalid speech metadata")
        with wave.open(str(path), "rb") as audio:
            if (
                audio.getcomptype() != "NONE"
                or audio.getsampwidth() != 2
                or audio.getframerate() != 24000
                or audio.getnchannels() != 1
                or audio.getnframes() != result["frames"]
                or len(audio.readframes(result["frames"])) != 2 * result["frames"]
            ):
                raise ValueError("Speech waveform differs from metadata")
    except (ValueError, TypeError, KeyError, OSError, EOFError, wave.Error) as exc:
        raise JobExecutionError("PACK_OUTPUT_INVALID", "Speech output is invalid.") from exc


def _validate_codec_audio(path: Path, result: dict[str, Any], value: dict, backend: str) -> None:
    from oryn_engine.qwen_codec_tasks import HOP, validate_codes

    try:
        validate_codes(value.get("codes"))
        if (
            set(result)
            != {
                "relativeOutput",
                "mediaType",
                "sampleRate",
                "channels",
                "frames",
                "durationSeconds",
                "backend",
            }
            or result["relativeOutput"] != "result.wav"
            or result["mediaType"] != "audio/wav"
            or result["frames"] != len(value["codes"]) * HOP
        ):
            raise ValueError("Codec output differs from requested frames")
        _validate_synthesized_audio(path, result, backend)
    except (ValueError, TypeError, KeyError) as exc:
        raise JobExecutionError("PACK_OUTPUT_INVALID", "Codec output is invalid.") from exc


def _validate_tiger_audio(
    directory: Path, result: dict[str, Any], input_path: Path, backend: str
) -> list[Path]:
    from oryn_engine.tiger_tasks import validate_outputs

    try:
        return validate_outputs(directory, result, input_path, backend)
    except (ValueError, TypeError, KeyError, OSError, EOFError, wave.Error) as exc:
        raise JobExecutionError(
            "PACK_OUTPUT_INVALID", "Separated audio output is invalid."
        ) from exc


def _validate_reuse_audio(
    path: Path, result: dict[str, Any], input_path: Path, backend: str
) -> None:
    from oryn_engine.reuse_tasks import validate_output

    try:
        validate_output(path, result, input_path, backend)
    except (ValueError, TypeError, KeyError, OSError, EOFError, wave.Error) as exc:
        raise JobExecutionError("PACK_OUTPUT_INVALID", "Enhanced audio output is invalid.") from exc


def _validate_mpsenet_audio(
    path: Path, result: dict[str, Any], input_path: Path, backend: str
) -> None:
    from oryn_engine.mpsenet_tasks import validate_output

    try:
        validate_output(path, result, input_path, backend)
    except (ValueError, TypeError, KeyError, OSError, EOFError, wave.Error) as exc:
        raise JobExecutionError("PACK_OUTPUT_INVALID", "Enhanced audio output is invalid.") from exc


def _validate_enhanced_audio(path: Path, result: dict[str, Any], input_path: Path) -> None:
    try:
        if (
            set(result) != {"relativeOutput", "mediaType", "sampleRate", "channels", "frames"}
            or result["relativeOutput"] != "result.wav"
            or result["mediaType"] != "audio/wav"
            or any(type(result[k]) is not int for k in ("sampleRate", "channels", "frames"))
            or result["sampleRate"] != 48000
            or result["channels"] not in {1, 2}
            or not 1 <= result["frames"] <= 48000 * 30
            or path.is_symlink()
            or not path.is_file()
            or not 1 <= path.stat().st_size <= 6 * 1024**2
        ):
            raise ValueError("Invalid audio metadata")
        with wave.open(str(input_path), "rb") as source, wave.open(str(path), "rb") as target:
            if (
                target.getcomptype() != "NONE"
                or target.getsampwidth() != 2
                or target.getframerate() != result["sampleRate"]
                or target.getnchannels() != result["channels"]
                or target.getnframes() != result["frames"]
                or target.getnchannels() != source.getnchannels()
                or target.getnframes()
                != (source.getnframes() * 48000 + source.getframerate() - 1)
                // source.getframerate()
                or len(target.readframes(result["frames"]))
                != result["frames"] * result["channels"] * 2
            ):
                raise ValueError("Audio geometry differs")
    except (ValueError, TypeError, KeyError, OSError, EOFError, wave.Error) as exc:
        raise JobExecutionError("PACK_OUTPUT_INVALID", "Enhanced audio output is invalid.") from exc


def _validate_generated_image(path: Path, width: int, height: int) -> None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (width, height):
                raise ValueError("image dimensions or format do not match")
    except Exception as exc:
        raise JobExecutionError(
            "PACK_OUTPUT_INVALID", "Diffusers image output is invalid."
        ) from exc


def _validate_generated_video(
    path: Path, result: dict[str, Any], input_value: dict[str, Any]
) -> None:
    try:
        with path.open("rb") as source:
            header = source.read(32)
    except OSError as exc:
        raise JobExecutionError(
            "PACK_OUTPUT_INVALID", "Diffusers video output is invalid."
        ) from exc
    if (
        len(header) < 12
        or header[4:8] != b"ftyp"
        or result.get("frames") != input_value["frames"]
        or result.get("fps") != input_value["fps"]
    ):
        raise JobExecutionError("PACK_OUTPUT_INVALID", "Diffusers video output is invalid.")


def _decode_wave(
    path: Path,
    declared_size: int,
    *,
    max_seconds: int = MAX_AUDIO_SECONDS,
    max_decoded_bytes: int | None = None,
) -> Any:
    if not 0 < declared_size <= MAX_AUDIO_BYTES:
        raise JobExecutionError("AUDIO_TOO_LARGE", "The audio artifact exceeds the ASR limit.")
    try:
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frame_count = source.getnframes()
            compression = source.getcomptype()
            if (
                channels < 1
                or channels > 32
                or sample_width not in {1, 2, 3, 4}
                or sample_rate < 8_000
                or sample_rate > 192_000
                or compression != "NONE"
                or frame_count <= 0
            ):
                raise JobExecutionError(
                    "AUDIO_FORMAT_UNSUPPORTED", "The WAV format is unsupported."
                )
            pcm_bytes = frame_count * channels * sample_width
            if (
                frame_count > sample_rate * max_seconds
                or pcm_bytes > MAX_AUDIO_BYTES
                or (
                    max_decoded_bytes is not None and frame_count * channels * 4 > max_decoded_bytes
                )
            ):
                raise JobExecutionError(
                    "AUDIO_TOO_LARGE", "The audio artifact exceeds the task limit."
                )
            raw = source.readframes(frame_count)
            if len(raw) != pcm_bytes:
                raise JobExecutionError("AUDIO_DECODE_FAILED", "The WAV artifact is truncated.")
    except JobExecutionError:
        raise
    except (OSError, EOFError, wave.Error) as exc:
        raise JobExecutionError("AUDIO_DECODE_FAILED", "The WAV artifact is invalid.") from exc

    import numpy as np

    if sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        values = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        integers = values[:, 0] | (values[:, 1] << 8) | (values[:, 2] << 16)
        integers = np.where(integers & 0x800000, integers - 0x1000000, integers)
        samples = integers.astype(np.float32) / 8388608.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if sample_rate != WHISPER_SAMPLE_RATE and samples.size:
        output_size = max(1, round(samples.size * WHISPER_SAMPLE_RATE / sample_rate))
        source_positions = np.arange(samples.size, dtype=np.float64)
        target_positions = np.linspace(0, samples.size - 1, output_size, dtype=np.float64)
        samples = np.interp(target_positions, source_positions, samples).astype(np.float32)
    return samples


def _checkpoint(cancellation: threading.Event) -> None:
    if cancellation.is_set():
        raise JobExecutionError("JOB_CANCELLED", "The job was cancelled.")


def _decode_transcript(
    processor: Any,
    token_ids: Any,
    *,
    timestamps: bool,
) -> tuple[str, str | None, list[dict[str, Any]]]:
    sequences = token_ids.get("sequences") if isinstance(token_ids, dict) else token_ids
    if sequences is None:
        raise ValueError("generated transcript has no token sequence")
    tokens = sequences[0]
    raw_special = processor.tokenizer.decode(tokens, skip_special_tokens=False)
    match = re.search(r"<\|([a-z]{2,3})\|>", raw_special)
    language = match.group(1) if match else None
    if timestamps:
        token_timestamps = (
            token_ids.get("token_timestamps") if isinstance(token_ids, dict) else None
        )
        if token_timestamps is not None and hasattr(processor.tokenizer, "_decode_asr"):
            with contextlib.suppress(TypeError, ValueError, KeyError, AttributeError):
                time_precision = (
                    float(
                        getattr(getattr(processor, "feature_extractor", None), "chunk_length", 30.0)
                    )
                    / 1_500.0
                )
                text, metadata = processor.tokenizer._decode_asr(
                    [
                        {
                            "tokens": sequences.detach().cpu().numpy(),
                            "token_timestamps": token_timestamps.detach().cpu().numpy(),
                        }
                    ],
                    return_timestamps="word",
                    return_language=False,
                    time_precision=time_precision,
                )
                chunks = metadata.get("chunks") if isinstance(metadata, dict) else None
                segments = _timestamp_segments(chunks)
                if isinstance(text, str) and segments:
                    return text.strip(), language, segments
        with contextlib.suppress(TypeError, ValueError, KeyError, AttributeError):
            decoded = processor.tokenizer.decode(
                tokens,
                skip_special_tokens=True,
                output_offsets=True,
            )
            if isinstance(decoded, dict) and isinstance(decoded.get("text"), str):
                return (
                    decoded["text"].strip(),
                    language,
                    _timestamp_segments(decoded.get("offsets")),
                )
    text = processor.batch_decode(sequences, skip_special_tokens=True)[0]
    if not isinstance(text, str):
        raise ValueError("decoded transcript is not text")
    return text.strip(), language, []


def _timestamp_segments(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 100_000:
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            continue
        timestamp = item.get("timestamp")
        if (
            not isinstance(timestamp, (tuple, list))
            or len(timestamp) != 2
            or not all(isinstance(number, (int, float)) for number in timestamp)
        ):
            continue
        start, end = float(timestamp[0]), float(timestamp[1])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            continue
        result.append({"text": item["text"].strip(), "start": start, "end": end})
    return result


def _move_inputs_to_model(values: dict[str, Any], model: Any, backend: str) -> dict[str, Any]:
    """Move tensor inputs while matching floating features to the checkpoint dtype."""

    model_dtype = getattr(model, "dtype", None)
    result: dict[str, Any] = {}
    for key, value in values.items():
        if not hasattr(value, "to"):
            result[key] = value
            continue
        kwargs: dict[str, Any] = {"device": backend}
        if getattr(getattr(value, "dtype", None), "is_floating_point", False) and model_dtype:
            kwargs["dtype"] = model_dtype
        result[key] = value.to(**kwargs)
    return result


def _supports_whisper_attention_timestamps(model: Any) -> bool:
    heads = getattr(getattr(model, "generation_config", None), "alignment_heads", None)
    layers = getattr(getattr(model, "config", None), "decoder_layers", None)
    attention_heads = getattr(getattr(model, "config", None), "decoder_attention_heads", None)
    return bool(
        isinstance(heads, (list, tuple))
        and heads
        and isinstance(layers, int)
        and layers > 0
        and isinstance(attention_heads, int)
        and attention_heads > 0
        and all(
            isinstance(item, (list, tuple))
            and len(item) == 2
            and all(isinstance(index, int) and not isinstance(index, bool) for index in item)
            and 0 <= item[0] < layers
            and 0 <= item[1] < attention_heads
            for item in heads
        )
    )


def _forced_alignment_input(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "audioArtifactId",
        "transcript",
        "language",
        "timestamps",
    }:
        raise JobExecutionError("JOB_INPUT_INVALID", "The forced-alignment input is invalid.")
    artifact_id = value.get("audioArtifactId")
    transcript = value.get("transcript")
    language = value.get("language")
    timestamps = value.get("timestamps")
    if (
        not isinstance(artifact_id, str)
        or not isinstance(transcript, str)
        or not 1 <= len(transcript.strip()) <= 262_144
        or not isinstance(language, str)
        or not 1 <= len(language) <= 32
        or timestamps not in {"word", "segment"}
    ):
        raise JobExecutionError("JOB_INPUT_INVALID", "The forced-alignment input is invalid.")
    return {
        "audioArtifactId": artifact_id,
        "transcript": transcript.strip(),
        "language": language,
        "timestamps": timestamps,
    }


def _forced_alignment_segments(value: object, granularity: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list):
        raise JobExecutionError(
            "ALIGNMENT_OUTPUT_INVALID", "The aligner returned an invalid batch."
        )
    words: list[dict[str, Any]] = []
    for item in value[0]:
        if not isinstance(item, dict) or set(item) != {"text", "start_time", "end_time"}:
            raise JobExecutionError("ALIGNMENT_OUTPUT_INVALID", "An aligned word is invalid.")
        text = item["text"]
        start = item["start_time"]
        end = item["end_time"]
        if (
            not isinstance(text, str)
            or not text.strip()
            or not isinstance(start, (int, float))
            or isinstance(start, bool)
            or not isinstance(end, (int, float))
            or isinstance(end, bool)
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or not 0 <= float(start) < float(end) <= MAX_AUDIO_SECONDS
        ):
            raise JobExecutionError("ALIGNMENT_OUTPUT_INVALID", "An aligned word is invalid.")
        words.append({"text": text.strip(), "start": float(start), "end": float(end)})
    if not 1 <= len(words) <= 100_000:
        raise JobExecutionError("ALIGNMENT_OUTPUT_INVALID", "The aligned word count is invalid.")
    if granularity == "word":
        return words
    return _group_alignment_words(words)


def _group_alignment_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for word in words:
        previous = pending[-1] if pending else None
        candidate = " ".join([*(str(item["text"]) for item in pending), str(word["text"])])
        boundary = bool(
            pending
            and (
                len(candidate) > 84
                or (previous is not None and float(word["start"]) - float(previous["end"]) > 0.8)
            )
        )
        if boundary:
            result.append(
                {
                    "text": " ".join(str(item["text"]) for item in pending),
                    "start": float(pending[0]["start"]),
                    "end": float(pending[-1]["end"]),
                }
            )
            pending = []
        pending.append(word)
        if str(word["text"]).endswith((".", "!", "?")):
            result.append(
                {
                    "text": " ".join(str(item["text"]) for item in pending),
                    "start": float(pending[0]["start"]),
                    "end": float(pending[-1]["end"]),
                }
            )
            pending = []
    if pending:
        result.append(
            {
                "text": " ".join(str(item["text"]) for item in pending),
                "start": float(pending[0]["start"]),
                "end": float(pending[-1]["end"]),
            }
        )
    return result


def _cross_encoder_rerank(
    handle: TransformersRerankerHandle,
    value: dict[str, Any],
    cancellation: threading.Event,
    emit: RunnerEvent,
) -> object:
    import torch

    _checkpoint(cancellation)
    encoded = handle.tokenizer(
        [value["query"]] * len(value["documents"]),
        value["documents"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    inputs = {
        key: tensor.to(handle.backend)
        for key, tensor in encoded.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }
    emit("job.state", {"state": "running"})
    with torch.inference_mode():
        logits = handle.model(**inputs).logits.float()
    if logits.ndim != 2 or logits.shape[1] not in {1, 2}:
        raise JobExecutionError(
            "RERANKING_OUTPUT_INVALID", "Reranker requires one or two output labels."
        )
    scores = (
        (
            torch.sigmoid(logits[:, 0])
            if logits.shape[1] == 1
            else torch.softmax(logits, dim=-1)[:, 1]
        )
        .cpu()
        .tolist()
    )
    _checkpoint(cancellation)
    if any(not math.isfinite(score) for score in scores):
        raise JobExecutionError("RERANKING_OUTPUT_INVALID", "Reranker returned non-finite scores.")
    ranked = sorted(
        ({"index": index, "score": score} for index, score in enumerate(scores)),
        key=lambda item: (-item["score"], item["index"]),
    )[: value["topK"]]
    return {"kind": "structured", "value": {"results": ranked}}


def _bounded_chat_image(bindings: tuple[object, ...], artifact_id: str) -> Any:
    from PIL import Image

    binding = _image_binding(bindings, artifact_id)
    path = Path(binding["internalPath"])
    if binding["sizeBytes"] > 20 * 1024**2 or path.stat().st_size != binding["sizeBytes"]:
        raise JobExecutionError("JOB_INPUT_INVALID", "Chat image exceeds its byte limit.")
    try:
        with Image.open(path) as image:
            if image.width * image.height > 4096**2 or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Image size or frame count is invalid")
            image.load()
            result = image.convert("RGB")
            result.thumbnail((1024, 1024))
            return result
    except Exception as exc:
        raise JobExecutionError("JOB_INPUT_INVALID", "Chat image is invalid.") from exc


def _verify_installed_files(record: dict[str, Any]) -> None:
    files = record.get("files", [])
    if not any("content_sha256" in item for item in files):
        return  # Older installations retain their existing admission path until reinstallation.
    root = Path(record["cacheRoot"]).resolve()
    for item in files:
        try:
            path = (root / item["cache_path"]).resolve(strict=True)
            if not path.is_relative_to(root) or path.stat().st_size != item["size"]:
                raise ValueError("Model file changed")
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != item["content_sha256"]:
                raise ValueError("Model file changed")
        except (KeyError, OSError, ValueError) as exc:
            raise JobExecutionError(
                "MODEL_INTEGRITY_FAILED", "Installed model files changed; reinstall the model."
            ) from exc


@contextlib.contextmanager
def _selected_model_view(record: dict[str, Any], model_root: Path):
    """Give the sandbox direct selected files without granting access to the whole blob cache."""
    files = record.get("files", [])
    if not files:
        yield model_root
        return
    cache = Path(record["cacheRoot"]).resolve()
    from oryn_engine.model_composition import record_binding

    try:
        composition = record_binding(record)
    except ValueError as exc:
        raise JobExecutionError("MODEL_INTEGRITY_FAILED", "Model composition changed.") from exc
    if len(files) > 10_000 or not model_root.resolve().is_relative_to(cache):
        raise JobExecutionError("MODEL_INTEGRITY_FAILED", "Model file scope is invalid.")
    with tempfile.TemporaryDirectory(prefix=".oryn-model-view-", dir=cache) as temporary:
        view = Path(temporary)
        for item in files:
            name = item.get("name")
            if (
                not isinstance(name, str)
                or PurePosixPath(name).is_absolute()
                or "\\" in name
                or any(part in {"", ".", ".."} for part in name.split("/"))
            ):
                raise JobExecutionError("MODEL_INTEGRITY_FAILED", "Model file name is invalid.")
            try:
                if composition is not None:
                    relative = item.get("cache_path")
                    if not isinstance(relative, str) or not item.get("content_sha256"):
                        raise ValueError("Composition file lacks its verified source")
                    source = (cache / relative).resolve(strict=True)
                else:
                    source = (model_root / name).resolve(strict=True)
            except (OSError, ValueError) as exc:
                raise JobExecutionError(
                    "MODEL_INTEGRITY_FAILED", "A selected model file is unavailable."
                ) from exc
            if not source.is_relative_to(cache) or not source.is_file():
                raise JobExecutionError("MODEL_INTEGRITY_FAILED", "Model file escaped its scope.")
            target = view / name
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError as exc:
                raise JobExecutionError(
                    "MODEL_LOAD_FAILED", "Could not prepare the selected model files."
                ) from exc
        yield view
