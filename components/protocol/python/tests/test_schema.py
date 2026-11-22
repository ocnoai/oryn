from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from oryn_protocol import ProtocolSchemaStore

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "v1"


def test_paired_gguf_projector_metadata_is_a_bounded_filename() -> None:
    validator = ProtocolSchemaStore().validator("trustedModelMetadata")
    value = {"license": "Repository-defined", "modelFile": "model.gguf",
             "projectorFile": "mmproj-f16.gguf",
             "conformance": {"status": "fixture-only", "platforms": ["cpu"]}}
    validator.validate(value)
    for path in ("../mmproj.gguf", "/tmp/mmproj.gguf", "projector.safetensors"):
        with pytest.raises(ValidationError):
            validator.validate({**value, "projectorFile": path})


@pytest.mark.parametrize(
    ("fixture_name", "definition"),
    [
        ("handshake-request.json", "handshakeRequest"),
        ("handshake-success.json", "handshakeResponse"),
        ("handshake-version-error.json", "handshakeResponse"),
        ("ping-request.json", "protocolRequest"),
        ("ping-response.json", "protocolResponse"),
        ("runtime-capabilities-request.json", "protocolRequest"),
        ("runtime-capabilities.json", "protocolResponse"),
        ("error-response.json", "protocolResponse"),
        ("runtime-shutdown-request.json", "runtimeShutdownRequest"),
        ("runtime-shutdown-response.json", "protocolResponse"),
        ("runtime-state-event.json", "runtimeStateEvent"),
        ("runtime-resources.json", "protocolResponse"),
        ("runtime-memory-pressure-event.json", "runtimeMemoryPressureChangedEvent"),
        ("runtime-policy-update-request.json", "runtimePolicyUpdateRequest"),
        ("runtime-resources-event.json", "runtimeResourcesUpdatedEvent"),
        ("model-inspect-request.json", "modelInspectRequest"),
        ("model-runtime-override-request.json", "modelRuntimeOverrideRequest"),
        ("model-install-progress.json", "modelInstallProgressEvent"),
        ("model-load-request.json", "modelLoadRequest"),
        ("model-residency-event.json", "modelResidencyChangedEvent"),
        ("model-remove-request.json", "modelRemoveRequest"),
        ("session-attach-request.json", "sessionAttachRequest"),
        ("job-create-request.json", "jobCreateRequest"),
        ("job-state-event.json", "jobStateEvent"),
        ("job-completed-event.json", "jobCompletedEvent"),
        ("job-cancelled-event.json", "jobCancelledEvent"),
    ],
)
def test_golden_fixture_matches_schema(fixture_name: str, definition: str) -> None:
    value = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))
    ProtocolSchemaStore().validator(definition).validate(value)


def test_resource_response_result_matches_snapshot_schema() -> None:
    response = json.loads((FIXTURES / "runtime-resources.json").read_text(encoding="utf-8"))
    ProtocolSchemaStore().validator("runtimeResourceSnapshot").validate(response["result"])


def test_runtime_pack_inventory_and_cleanup_are_strict() -> None:
    schemas = ProtocolSchemaStore()
    inventory = {
        "catalogVersion": 1,
        "catalogDigest": "a" * 64,
        "thirdPartyPublishersEnabled": False,
        "automaticExecutableUpdatesEnabled": False,
        "packs": [
            {
                "packId": "oryn.llama-cpp",
                "version": "0.1.0",
                "publisher": "oryn.first-party",
                "abi": "1",
                "dependencyClosure": "llama.cpp-b10218",
                "platforms": ["macos-arm64"],
                "distribution": "bundled",
                "state": "active",
                "bundled": True,
                "removable": False,
                "sizeBytes": 0,
                "minimumSafeVersion": "0.1.0",
                "revoked": False,
                "license": {
                    "summary": "MIT native runtime",
                    "notice": "Upstream notice is packaged with the runtime.",
                    "access": "public",
                    "acceptanceRequired": False,
                    "accepted": True,
                },
            }
        ],
    }
    schemas.validator("runtimePackInventoryResult").validate(inventory)
    schemas.validator("runtimePackCleanupResult").validate(
        {"removedPacks": 0, "reclaimedBytes": 0}
    )
    with pytest.raises(ValidationError):
        schemas.validator("runtimePackInventoryResult").validate(
            {**inventory, "thirdPartyPublishersEnabled": True}
        )


@pytest.mark.parametrize(
    ("fixture_name", "definition"),
    [
        ("model-inspect-response.json", "candidateExecutionProfile"),
        ("model-inventory-response.json", "modelInventoryResult"),
        ("model-remove-preview-response.json", "modelRemovePreviewResult"),
        ("model-remove-response.json", "modelRemoveResult"),
        ("session-attached-response.json", "sessionAttachedResult"),
        ("job-queued-response.json", "jobReference"),
    ],
)
def test_model_response_result_schema(fixture_name: str, definition: str) -> None:
    response = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))
    ProtocolSchemaStore().validator(definition).validate(response["result"])
