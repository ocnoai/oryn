from __future__ import annotations

import copy
import hashlib

import pytest

from oryn_engine.capability_sources import (
    framework_registry,
    framework_sources,
    synthetic_non_framework_source,
)
from oryn_engine.compatibility_plane import (
    SCHEMA_VERSION,
    CompatibilityPlaneError,
    ModelDescriptor,
    compile_registry,
    evaluate_eligibility,
    semantic_diff,
)


def descriptor(**overrides: object) -> ModelDescriptor:
    values: dict[str, object] = {
        "provider": "huggingface",
        "repository": "org/model",
        "revision": "a" * 40,
        "source_kind": "transformers",
        "architectures": ("Qwen3ForCausalLM",),
        "task": "text-generation",
        "formats": ("json", "safetensors", "tokenizer"),
        "selected_files_digest": hashlib.sha256(b"files").hexdigest(),
        "requires_repository_code": False,
        "access": "available",
        "estimated_memory_bytes": 1_000,
    }
    values.update(overrides)
    return ModelDescriptor(**values)  # type: ignore[arg-type]


def test_framework_and_non_framework_sources_compile_through_one_schema() -> None:
    registry = compile_registry((*framework_sources(), synthetic_non_framework_source()))
    assert registry.schema_version.endswith("/1")
    assert len(registry.digest) == 64
    assert registry.match(descriptor()) is not None
    assert (
        registry.match(
            descriptor(
                source_kind="columnar-fixture",
                architectures=("bounded-pairwise-ranker",),
                task="text-ranking",
                formats=("columnar-v1",),
            )
        ).foundation_id
        == "oryn.runtime.synthetic"
    )  # type: ignore[union-attr]


def test_compiler_is_deterministic_and_reports_semantic_diff() -> None:
    forward = compile_registry(framework_sources())
    reverse = compile_registry(reversed(framework_sources()))
    assert forward.digest == reverse.digest
    reduced_source = copy.deepcopy(framework_sources()[0])
    reduced_source["capabilities"] = [
        item
        for item in reduced_source["capabilities"]
        if item["id"] != "transformers.specialized.3"
    ]
    reduced = compile_registry((reduced_source, *framework_sources()[1:]))
    diff = semantic_diff(forward, reduced)
    assert diff["removed"] == ["transformers.specialized.3"]


@pytest.mark.parametrize("field", ["executable", "entrypoint", "command", "loaderOptions"])
def test_provider_cannot_add_executable_authority(field: str) -> None:
    source = copy.deepcopy(synthetic_non_framework_source())
    source["capabilities"][0][field] = "hostile"
    with pytest.raises(CompatibilityPlaneError):
        compile_registry((source,))


def test_eligibility_axes_remain_independent_and_drive_disposition() -> None:
    result = evaluate_eligibility(descriptor(), framework_registry(), resource_admissible=True)
    assert result.disposition == "trust-required"
    assert result.axes["runtime"].state == "supported"
    assert result.matched is not None
    blocked = evaluate_eligibility(
        descriptor(requires_repository_code=True), framework_registry(), resource_admissible=True
    )
    assert blocked.disposition == "unsupported"
    assert blocked.axes["runtime"].reasons == ("CUSTOM_CODE_REQUIRED",)
    constrained = evaluate_eligibility(
        descriptor(), framework_registry(), resource_admissible=False
    )
    assert constrained.disposition == "not-admissible-on-this-device"


def test_safe_untested_capability_requires_user_trust_instead_of_being_unsupported() -> None:
    result = evaluate_eligibility(
        descriptor(
            architectures=("Qwen3ASRForConditionalGeneration",),
            task="automatic-speech-recognition",
        ),
        framework_registry(),
        resource_admissible=True,
    )

    assert result.axes["conformance"].state == "untested"
    assert result.axes["conformance"].reasons == ("CONFORMANCE_REQUIRED",)
    assert result.disposition == "trust-required"


def test_authorization_deferral_is_causal_and_cannot_hide_known_security_block() -> None:
    unknown = evaluate_eligibility(
        descriptor(
            access="authorization-required",
            architectures=("UnknownArchitecture",),
            formats=(),
            estimated_memory_bytes=0,
        ),
        framework_registry(),
        resource_admissible=True,
    )
    assert unknown.disposition == "authorization-required"
    assert all(
        axis.state == "deferred" and axis.deferred_by == "authorization"
        for name, axis in unknown.axes.items()
        if name != "access"
    )
    blocked = evaluate_eligibility(
        descriptor(
            access="authorization-required",
            requires_repository_code=True,
            architectures=("UnknownArchitecture",),
            formats=(),
            estimated_memory_bytes=0,
        ),
        framework_registry(),
        resource_admissible=True,
    )
    assert blocked.disposition == "unsupported"
    assert blocked.axes["runtime"].state == "unsupported"


def test_known_compatible_gated_model_is_authorization_required() -> None:
    result = evaluate_eligibility(
        descriptor(access="authorization-required"),
        framework_registry(),
        resource_admissible=True,
    )

    assert result.disposition == "authorization-required"
    assert result.axes["runtime"].state == "supported"
    assert result.axes["task"].state == "supported"
    assert result.axes["conformance"].state == "untested"


def test_descriptor_identity_is_strict_and_versioned() -> None:
    assert SCHEMA_VERSION == "oryn-model-descriptor/1"
    with pytest.raises(CompatibilityPlaneError):
        descriptor(revision="main")
