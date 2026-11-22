from __future__ import annotations

from pathlib import Path
from typing import Any

from oryn_engine.models import ModelStore, ModelStoreError
from oryn_engine.protocol.dispatcher import ProtocolDispatcher

EXACT_MODEL = {
    "provider": "huggingface",
    "repository": "test/model",
    "revision": "a" * 40,
}


def _handshake(dispatcher: ProtocolDispatcher) -> dict[str, object]:
    response, should_exit = dispatcher.dispatch(
        {
            "type": "request",
            "id": "req_handshake",
            "method": "protocol.handshake",
            "params": {
                "supportedVersions": [1],
                "browserVersion": "0.1.0",
                "maxMessageBytes": 16 * 1024 * 1024,
            },
        }
    )
    assert should_exit is False
    return response


def test_duplicate_request_id_is_rejected() -> None:
    dispatcher = ProtocolDispatcher()
    _handshake(dispatcher)
    request = {
        "v": 1,
        "type": "request",
        "id": "req_ping",
        "method": "protocol.ping",
        "params": {},
    }

    first, _ = dispatcher.dispatch(request)
    second, _ = dispatcher.dispatch(request)

    assert first["ok"] is True
    assert second["error"]["code"] == "PROTOCOL_DUPLICATE_REQUEST_ID"


def test_unknown_fields_are_rejected() -> None:
    dispatcher = ProtocolDispatcher()
    _handshake(dispatcher)
    response, _ = dispatcher.dispatch(
        {
            "v": 1,
            "type": "request",
            "id": "req_ping_extra",
            "method": "protocol.ping",
            "params": {},
            "unexpected": True,
        }
    )

    assert response["error"]["code"] == "PROTOCOL_SCHEMA_INVALID"


def test_invalid_post_handshake_version_is_rejected() -> None:
    dispatcher = ProtocolDispatcher()
    _handshake(dispatcher)
    response, _ = dispatcher.dispatch(
        {
            "v": 2,
            "type": "request",
            "id": "req_wrong_version",
            "method": "protocol.ping",
            "params": {},
        }
    )

    assert response["error"]["code"] == "PROTOCOL_SCHEMA_INVALID"


def test_unsupported_protocol_version_requests_clean_exit() -> None:
    dispatcher = ProtocolDispatcher()
    response, should_exit = dispatcher.dispatch(
        {
            "type": "request",
            "id": "req_handshake_unsupported",
            "method": "protocol.handshake",
            "params": {
                "supportedVersions": [2],
                "browserVersion": "0.1.0",
                "maxMessageBytes": 16 * 1024 * 1024,
            },
        }
    )

    assert should_exit is True
    assert response["error"]["code"] == "PROTOCOL_VERSION_UNSUPPORTED"


def test_method_specific_params_are_strict_and_unknown_method_is_stable() -> None:
    dispatcher = ProtocolDispatcher()
    _handshake(dispatcher)
    invalid, _ = dispatcher.dispatch(
        {
            "v": 1,
            "type": "request",
            "id": "req_ping_invalid",
            "method": "protocol.ping",
            "params": {"unexpected": True},
        }
    )
    unknown, _ = dispatcher.dispatch(
        {
            "v": 1,
            "type": "request",
            "id": "req_unknown",
            "method": "engine.exec",
            "params": {},
        }
    )

    assert invalid["error"]["code"] == "PROTOCOL_SCHEMA_INVALID"
    assert unknown["error"]["code"] == "PROTOCOL_UNKNOWN_METHOD"


def test_signed_model_catalog_is_available_through_a_bounded_protocol_method(
    tmp_path: Path,
) -> None:
    dispatcher = ProtocolDispatcher(models=ModelStore(tmp_path, runtime_available=True))
    _handshake(dispatcher)
    response, should_exit = dispatcher.dispatch(
        {
            "v": 1,
            "type": "request",
            "id": "req_catalog",
            "method": "model.catalog.match",
            "params": {"task": "text-generation", "outputModalities": ["text"]},
        }
    )

    assert should_exit is False
    assert response["ok"] is True
    assert response["result"]["models"] == [
        {
            "provider": "huggingface",
            "repository": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "revision": "f56c2fbf7fabc03ffff3494d3c2f8d06c774493d",
        },
        {
            "provider": "huggingface",
            "repository": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "revision": "9217f5db79a29953eb74d5343926648285ec7e67",
        },
    ]


def test_internal_error_is_sanitized_and_has_diagnostic_id() -> None:
    class FailingDispatcher(ProtocolDispatcher):
        def _dispatch_method(self, method, message, request_id):  # type: ignore[no-untyped-def]
            raise RuntimeError("/Users/private/secret should stay in stderr")

    dispatcher = FailingDispatcher()
    _handshake(dispatcher)
    response, should_exit = dispatcher.dispatch(
        {
            "v": 1,
            "type": "request",
            "id": "req_failure",
            "method": "protocol.ping",
            "params": {},
        }
    )

    assert should_exit is False
    assert response["error"]["code"] == "PROTOCOL_INTERNAL_ERROR"
    assert response["error"]["diagnosticId"].startswith("diag_")
    assert "private" not in response["error"]["message"]


def test_memory_admission_error_is_a_retryable_resource_error() -> None:
    class MemoryRejectingDispatcher(ProtocolDispatcher):
        def _dispatch_method(self, method, message, request_id):  # type: ignore[no-untyped-def]
            raise ModelStoreError(
                "MODEL_MEMORY_INSUFFICIENT",
                "The model cannot be admitted under the current memory policy.",
                retryable=True,
                details={
                    "requiredBytes": 20_000_000_000,
                    "availableForModelBytes": 15_000_000_000,
                    "memoryPressure": "normal",
                },
            )

    dispatcher = MemoryRejectingDispatcher()
    _handshake(dispatcher)
    response, should_exit = dispatcher.dispatch(
        {
            "v": 1,
            "type": "request",
            "id": "req_memory_rejected",
            "method": "protocol.ping",
            "params": {},
        }
    )

    assert should_exit is False
    assert response["error"] == {
        "code": "MODEL_MEMORY_INSUFFICIENT",
        "category": "resource",
        "message": "The model cannot be admitted under the current memory policy.",
        "retryable": True,
        "details": {
            "requiredBytes": 20_000_000_000,
            "availableForModelBytes": 15_000_000_000,
            "memoryPressure": "normal",
        },
    }


def test_runtime_resources_and_policy_revisions() -> None:
    dispatcher = ProtocolDispatcher()
    _handshake(dispatcher)
    update = {
        "v": 1,
        "type": "request",
        "id": "req_policy_new",
        "method": "runtime.policy.update",
        "params": {"policyRevision": 3, "memoryBudgetBytes": 2_147_483_648},
    }
    applied, _ = dispatcher.dispatch(update)
    stale, _ = dispatcher.dispatch({**update, "id": "req_policy_stale"})
    resources, _ = dispatcher.dispatch(
        {
            "v": 1,
            "type": "request",
            "id": "req_resources",
            "method": "runtime.resources",
            "params": {},
        }
    )

    assert applied["result"]["appliedRevision"] == 3
    assert stale["error"]["code"] == "RUNTIME_POLICY_STALE"
    assert resources["result"]["generationLane"] == {"capacity": 1, "running": 0, "queued": 0}


def test_signed_runtime_pack_inventory_and_cleanup_protocol(tmp_path: Path) -> None:
    dispatcher = ProtocolDispatcher(models=ModelStore(tmp_path, runtime_available=True))
    _handshake(dispatcher)
    inventory, should_exit = dispatcher.dispatch(
        {
            "v": 1,
            "type": "request",
            "id": "req_pack_inventory",
            "method": "runtime.pack.inventory",
            "params": {},
        }
    )
    cleanup, _ = dispatcher.dispatch(
        {
            "v": 1,
            "type": "request",
            "id": "req_pack_cleanup",
            "method": "runtime.pack.cleanup",
            "params": {},
        }
    )

    assert should_exit is False
    assert inventory["ok"] is True
    assert inventory["result"]["catalogVersion"] == 8
    assert len(inventory["result"]["catalogDigest"]) == 64
    assert inventory["result"]["thirdPartyPublishersEnabled"] is False
    assert inventory["result"]["automaticExecutableUpdatesEnabled"] is False
    assert {item["packId"] for item in inventory["result"]["packs"]} == {
        "oryn.ace-step-mlx",
        "oryn.diffusers-pytorch",
        "oryn.kokoro-onnx",
        "oryn.llama-cpp",
        "oryn.mflux-mlx",
        "oryn.mlx-audio",
        "oryn.mlx-lm",
        "oryn.onnx-runtime",
        "oryn.qwen-tts-pytorch",
        "oryn.transformers-pytorch",
    }
    assert cleanup["result"] == {"removedPacks": 0, "reclaimedBytes": 0}


def test_runtime_override_protocol_is_strict_and_dispatches_to_model_store(
    tmp_path: Path, monkeypatch: Any
) -> None:
    store = ModelStore(tmp_path, runtime_available=True)
    calls: list[tuple[dict[str, object], str | None]] = []
    result = {
        "planId": "plan-" + "1" * 24,
        "policy": "balanced",
        "reasonCodes": ["USER_OVERRIDE"],
        "selected": {
            "candidateId": "candidate-" + "2" * 24,
            "foundation": {"id": "oryn.runtime.pytorch", "version": "2.8.0"},
            "adapter": {"id": "oryn.adapter.transformers", "version": "1.0.0"},
            "variantId": "compatibility-" + "3" * 24,
            "backend": "cpu",
            "precision": "fp32",
            "estimatedMemoryBytes": 1024,
        },
        "fallbacks": [],
        "rejected": [],
    }

    def set_override(model: dict[str, object], candidate_id: str | None) -> dict[str, object]:
        calls.append((model, candidate_id))
        return result

    monkeypatch.setattr(store, "set_runtime_override", set_override)
    dispatcher = ProtocolDispatcher(models=store)
    _handshake(dispatcher)
    accepted, _ = dispatcher.dispatch(
        {
            "v": 1,
            "type": "request",
            "id": "req_runtime_override",
            "method": "model.runtime.override",
            "params": {"model": EXACT_MODEL, "candidateId": "candidate-" + "2" * 24},
        }
    )
    rejected, _ = dispatcher.dispatch(
        {
            "v": 1,
            "type": "request",
            "id": "req_runtime_override_invalid",
            "method": "model.runtime.override",
            "params": {"model": EXACT_MODEL, "candidateId": "unsafe"},
        }
    )

    assert accepted["result"] == result
    assert calls == [(EXACT_MODEL, "candidate-" + "2" * 24)]
    assert rejected["error"]["code"] == "PROTOCOL_SCHEMA_INVALID"
