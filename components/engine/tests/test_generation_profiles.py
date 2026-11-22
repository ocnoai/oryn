import pytest

from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
from oryn_engine.runtime_packs.generation_profiles import generation_profile


def diffusers_metadata() -> dict[str, object]:
    manifest = next(
        item
        for item in builtin_registry().compatibility
        if item.manifest_id == "oryn.compat.generic-safe-diffusers-image"
    )
    return dict(manifest.trusted_metadata)


def test_distilled_krea_profile_is_derived_from_declarative_pipeline_metadata() -> None:
    profile = generation_profile(
        task_contract="text-to-image-v1",
        trusted_metadata=diffusers_metadata(),
        declarative_metadata={"_class_name": "Krea2Pipeline", "is_distilled": True},
    )

    assert profile is not None
    assert profile["profileId"] == "diffusers.krea2.distilled"
    assert profile["provenance"] == "runtime-declared"
    assert profile["recommended"] is True
    assert profile["parameters"]["steps"]["default"] == 8
    assert profile["parameters"]["guidanceScale"] == {
        "default": 0.0,
        "minimum": 0.0,
        "maximum": 30.0,
        "step": 0.1,
        "supported": True,
        "editable": True,
    }


def test_base_krea_and_unknown_diffusers_do_not_share_a_fake_standard_profile() -> None:
    base = generation_profile(
        task_contract="text-to-image-v1",
        trusted_metadata=diffusers_metadata(),
        declarative_metadata={"_class_name": "Krea2Pipeline", "is_distilled": False},
    )
    unknown = generation_profile(
        task_contract="text-to-image-v1",
        trusted_metadata=diffusers_metadata(),
        declarative_metadata={"_class_name": "SomeFuturePipeline"},
    )

    assert base is not None and base["profileId"] == "diffusers.krea2.base"
    assert base["parameters"]["steps"]["default"] == 28
    assert base["parameters"]["guidanceScale"]["default"] == 4.5
    assert unknown is not None and unknown["profileId"] == "runtime.generic.fallback"
    assert unknown["recommended"] is False


@pytest.mark.parametrize(
    "pipeline_class",
    ["FluxPipeline", "StableDiffusionXLPipeline", "AutoPipelineForText2Image", "FuturePipeline"],
)
def test_other_pipeline_families_load_without_model_specific_profile_rules(
    pipeline_class: str,
) -> None:
    profile = generation_profile(
        task_contract="text-to-image-v1",
        trusted_metadata=diffusers_metadata(),
        declarative_metadata={"_class_name": pipeline_class},
    )

    assert profile is not None
    assert profile["profileId"] == "runtime.generic.fallback"
    assert profile["recommended"] is False
    assert all(parameter["editable"] for parameter in profile["parameters"].values())


def test_signed_runtime_defaults_take_precedence_over_adapter_derivation() -> None:
    profile = generation_profile(
        task_contract="text-to-image-v1",
        trusted_metadata={
            "inferenceDefaults": {"width": 768, "height": 512, "steps": 6, "guidanceScale": 1},
            "conformance": {"status": "passed", "platforms": ["macos-arm64-mps"]},
        },
        declarative_metadata={"_class_name": "Krea2Pipeline", "is_distilled": True},
    )

    assert profile is not None
    assert profile["provenance"] == "runtime-tested"
    assert profile["parameters"]["width"]["default"] == 768
    assert profile["parameters"]["steps"]["default"] == 6
    assert profile["parameters"]["guidanceScale"]["default"] == 1.0


def test_unverified_signed_defaults_are_declared_but_not_described_as_tested() -> None:
    profile = generation_profile(
        task_contract="text-to-image-v1",
        trusted_metadata={"inferenceDefaults": {"steps": 12, "guidanceScale": 3}},
    )

    assert profile is not None
    assert profile["provenance"] == "runtime-declared"


def test_non_media_contract_has_no_generation_profile() -> None:
    assert generation_profile(task_contract="chat-v1") is None
