import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from oryn_engine.diffusion_tasks import image_options, validate_pipeline_index
from oryn_engine.model_layouts import assess_layout
from oryn_engine.models import ModelStore
from oryn_engine.qwen_image_21 import (
    generation_defaults,
    processor_options,
    reference_image,
    sample_sigmas,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/qwen-image-2.1.json").read_text())
TURBO = json.loads((Path(__file__).parent / "fixtures/qwen-image-2.1-turbo.json").read_text())


@pytest.mark.parametrize("sigmas", [TURBO["modelIndex"]["sample_sigmas"], [1.0, 0.4], None])
def test_sampling_grid_is_data_and_does_not_select_component_files(sigmas):
    index = {**TURBO["modelIndex"], "sample_sigmas": sigmas}
    validate_pipeline_index(index)
    layout = assess_layout(
        [f["name"] for f in TURBO["files"]] + ["sample_sigmas/unused.json"],
        index,
        "text-to-image",
        source_kind="diffusers",
    )
    assert not layout.artifact_reasons and not layout.runtime_reasons
    assert "sample_sigmas/unused.json" not in layout.selected_files
    assert "processor/processor_config.json" in layout.selected_files


@pytest.mark.parametrize(
    "sigmas",
    [
        [],
        [1.0] * 51,
        [True],
        [float("nan")],
        [float("inf")],
        [1.1],
        [0],
        [-0.1],
        [0.5, 1.0],
        [0.5, 0.5],
        ["model_repo", "Runner"],
        {"auto_map": "repo.Runner"},
    ],
)
def test_sampling_grid_rejects_invalid_or_executable_metadata(sigmas):
    index = {**TURBO["modelIndex"], "sample_sigmas": sigmas}
    with pytest.raises(ValueError):
        validate_pipeline_index(index)
    layout = assess_layout(
        [f["name"] for f in TURBO["files"]], index, "text-to-image", source_kind="diffusers"
    )
    assert "PIPELINE_COMPONENT_UNSUPPORTED" in layout.runtime_reasons


def test_turbo_grid_does_not_waive_repository_component_protection():
    index = {**TURBO["modelIndex"], "processor": ["model_repo", "Processor"]}
    layout = assess_layout(
        [f["name"] for f in TURBO["files"]], index, "text-to-image", source_kind="diffusers"
    )
    assert "CUSTOM_CODE_REQUIRED" in layout.runtime_reasons
    defaults = generation_defaults(TURBO["modelIndex"])
    assert defaults["parameters"]["steps"] == {
        "default": 8,
        "minimum": 8,
        "maximum": 8,
        "step": 1,
        "supported": True,
        "editable": False,
    }
    assert defaults["provenance"] == "adapter-derived"
    assert sample_sigmas({}) is None


def test_actual_qwen21_layout_has_processor_owned_tokenizer_and_no_repository_code():
    index = FIXTURE["modelIndex"]
    validate_pipeline_index(index)
    layout = assess_layout(
        [f["name"] for f in FIXTURE["files"]], index, "text-to-image", source_kind="diffusers"
    )
    assert not layout.artifact_reasons and not layout.runtime_reasons
    assert "processor/tokenizer.json" in layout.selected_files
    assert sum(name.endswith(".safetensors") for name in layout.selected_files) == 7
    for change in (
        {"tokenizer": ["transformers", "Qwen2Tokenizer"]},
        {"processor": ["model_repo", "Processor"]},
        {"vae": ["diffusers", "AutoencoderKLQwenImage"]},
    ):
        with pytest.raises(ValueError):
            validate_pipeline_index({**index, **change})


@pytest.mark.parametrize("fixture", [FIXTURE, TURBO])
def test_real_processor_defaults_are_bounded_and_active_overrides_rejected(fixture):
    metadata, vision = fixture["processor"], fixture["textEncoder"]["vision_config"]
    assert processor_options(metadata, vision)["size"]["longest_edge"] == 1024**2
    for change in (
        {"device": "cuda"},
        {"do_pad": True},
        {"do_resize": False},
        {"patch_size": 14},
        {"auto_map": {"Processor": "model_repo.Processor"}},
    ):
        with pytest.raises(ValueError):
            processor_options(
                {**metadata, "image_processor": {**metadata["image_processor"], **change}}
                if "image_processor" in metadata
                else {**metadata, **change},
                vision,
            )


def test_native_options_do_not_silently_round_dimensions_or_discard_guidance():
    value = {"width": 512, "height": 512, "steps": 40, "guidanceScale": 1.0, "negativePrompt": ""}
    options = image_options(
        "QwenImage21Pipeline",
        "text-to-image-v1",
        value,
        {"guidance_scale": 1.0, "negative_prompt": ""},
    )
    assert options == {"true_cfg_scale": 1.0, "negative_prompt": "", "output_resolution": 512}
    for change in (
        {"width": 80},
        {"width": 2048, "height": 2048},
        {"steps": 51},
        {"guidanceScale": 11},
    ):
        with pytest.raises(ValueError):
            image_options("QwenImage21Pipeline", "text-to-image-v1", {**value, **change}, {})
    defaults = generation_defaults()
    assert defaults["parameters"]["guidanceScale"]["default"] == 1.0
    assert defaults["parameters"]["width"]["step"] == 32


def test_rgba_reference_keeps_alpha_and_rejects_unbounded_aspect(tmp_path):
    from PIL import Image

    path = tmp_path / "reference.png"
    Image.new("RGBA", (64, 64), (200, 10, 10, 80)).save(path)
    image = reference_image(path)
    assert image.mode == "RGBA" and image.getpixel((0, 0))[3] == 80
    Image.new("RGB", (640, 64)).save(path)
    with pytest.raises(ValueError):
        reference_image(path)


@pytest.mark.parametrize("fixture,steps", [(FIXTURE, 40), (TURBO, 8)])
def test_exact_checkpoint_recognized_without_claiming_conformance_or_waiving_memory(
    tmp_path, fixture, steps
):
    class Hub:
        def read_json(self, _repository, _revision, filename):
            parent = str(Path(filename).parent)
            return {
                "weight_map": {
                    f"tensor-{i}": Path(f["name"]).name
                    for i, f in enumerate(fixture["files"])
                    if str(Path(f["name"]).parent) == parent and f["name"].endswith(".safetensors")
                }
            }

        def model_info(self, *_args, **_kwargs):
            return SimpleNamespace(
                sha=fixture["revision"],
                config=deepcopy(fixture["modelIndex"]),
                pipeline_tag="text-to-image",
                library_name="diffusers",
                gated=False,
                private=False,
                siblings=[
                    SimpleNamespace(rfilename=f["name"], size=f["size"], blob_id="a" * 40, lfs=None)
                    for f in fixture["files"]
                ],
            )

    available = 256 * 1024**3
    store = ModelStore(
        tmp_path,
        hub=Hub(),
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", available),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _pack: None if _pack.provider else object()
    )
    model = {
        "provider": "huggingface",
        "repository": fixture["repository"],
        "revision": fixture["revision"],
    }
    profile = store.inspect(model)
    assert profile["executable"] and profile["loadDisposition"] == "trust-required"
    assert profile["runtimePack"]["packId"] == "oryn.diffusers-pytorch"
    assert profile["technicalCapabilities"]["architectures"] == ["QwenImage21Pipeline"]
    assert profile["technicalCapabilities"]["capabilities"]["instructionEditing"] is True
    assert profile["estimatedDiskBytes"] > 32_000_000_000
    assert profile["estimatedMemoryBytes"] > profile["estimatedDiskBytes"]
    assert profile["generationProfile"]["parameters"]["steps"]["default"] == steps
    assert not profile["requiresCustomCode"]
    assert profile["blockingReasons"] == ["CONFORMANCE_REQUIRED"]
    from oryn_protocol.schema import ProtocolSchemaStore

    assert ProtocolSchemaStore().is_valid("candidateExecutionProfile", profile)

    available = 1024**3
    blocked = store.inspect(model)
    assert not blocked["executable"]
    assert any("RESOURCE" in reason or "MEMORY" in reason for reason in blocked["blockingReasons"])
    assert "ARCHITECTURE_UNSUPPORTED" not in blocked["blockingReasons"]


@pytest.mark.parametrize("fixture", [FIXTURE, TURBO])
def test_installed_qwen21_defaults_survive_refresh_and_stale_evidence_is_removed(tmp_path, fixture):
    store = ModelStore(tmp_path, runtime_available=True)
    index = store.cache_root / "qwen21/model_index.json"
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps(fixture["modelIndex"]))
    record = {
        "runtimePack": {"packId": "oryn.diffusers-pytorch"},
        "runtimePlan": {"selected": {"variantId": "oryn.generated.diffusers"}},
        "capabilityRegistryDigest": store._capability_registry.digest,
        "files": [{"name": "model_index.json", "cache_path": "qwen21/model_index.json"}],
    }
    assert store._refresh_installed_generation_profile(record)
    assert record["generationProfile"] == generation_defaults(fixture["modelIndex"])
    assert not store._refresh_installed_generation_profile(record)
    record["capabilityRegistryDigest"] = "stale"
    assert store._refresh_installed_generation_profile(record)
    assert "generationProfile" not in record
