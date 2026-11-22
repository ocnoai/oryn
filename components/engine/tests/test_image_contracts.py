import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image

from oryn_engine.diffusion_tasks import (
    IMAGE_PIPELINES,
    image_options,
    klein_generation_defaults,
    validate_layer_outputs,
    validate_media_input,
    validate_pipeline_index,
)
from oryn_engine.runtime_packs.diffusers_worker import WorkerFailure, _generate_layers

BASE = {"imageArtifactId": "artifact_" + "a" * 24, "prompt": "Separate the foreground"}


def test_klein_base_and_distilled_have_distinct_effective_controls():
    from oryn_engine.task_profiles import profile_for_execution

    index = {
        "_class_name": "Flux2KleinPipeline",
        **{name: list(value) for name, value in IMAGE_PIPELINES["Flux2KleinPipeline"].items()},
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "tokenizer": ["transformers", "Qwen2Tokenizer"],
    }
    base = klein_generation_defaults({**index, "is_distilled": False})
    distilled = klein_generation_defaults({**index, "is_distilled": True})
    assert base["parameters"]["steps"]["default"] == 50
    assert base["parameters"]["guidanceScale"]["default"] == 4.0
    assert distilled["parameters"]["steps"]["default"] == 4
    profile = profile_for_execution(
        {
            "runtimePack": {"packId": "oryn.diffusers-pytorch", "taskContract": "image-editing-v1"},
            "generationProfile": distilled,
            "technicalCapabilities": {"architectures": ["Flux2KleinPipeline"]},
        }
    )
    assert profile["parameters"]["guidanceScale"]["valueSchema"]["const"] == 1.0
    assert profile["parameters"]["guidanceScale"]["handling"] == "fixed"
    assert profile["parameters"]["negativePrompt"]["valueSchema"]["const"] == ""
    value = validate_media_input("image-editing-v1", BASE)
    with pytest.raises(ValueError, match="classifier-free"):
        image_options(
            "Flux2KleinPipeline",
            "image-editing-v1",
            {**value, "guidanceScale": 4},
            {},
            distilled=True,
        )
    with pytest.raises(ValueError, match="Distillation"):
        klein_generation_defaults({**index, "is_distilled": "true"})


def test_native_layer_profile_honors_explicit_contract_features_and_modalities(tmp_path):
    from oryn_protocol.task_contracts import requirement_failures
    from test_models import FakeHub

    from oryn_engine.models import ModelFile, ModelStore
    from oryn_engine.task_profiles import profile_for_execution

    raw = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)._inspect_generated_pack(
        "org/layers",
        "a" * 40,
        [ModelFile("model_index.json", 10, "b" * 64)],
        {"_class_name": "QwenImageLayeredPipeline"},
        "image-text-to-image",
        True,
        source_kind="diffusers",
    )
    profile = profile_for_execution(raw)
    requirements = {
        "task": "image-text-to-image",
        "contract": {"id": "image-layers-v1", "revision": 1},
        "requiredFeatures": ["imageEditing"],
        "outputModalities": ["image", "structured"],
    }
    assert not requirement_failures(profile, requirements)
    assert requirement_failures(
        profile,
        {
            **requirements,
            "task": "image-to-image",
            "outputModalities": ["image"],
            "contract": {"id": "image-to-image-v1", "revision": 1},
        },
    )
    assert requirement_failures(profile, {**requirements, "inputModalities": ["audio"]})


def test_legacy_diffusion_pipeline_rejects_unselected_shard_reference(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    from oryn_engine.runtime_packs.diffusers_worker import _validate_model

    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "StableDiffusionPipeline",
                "unet": ["diffusers", "UNet2DConditionModel"],
            }
        )
    )
    component = tmp_path / "unet"
    component.mkdir()
    save_file({"weight": np.zeros((2, 2), dtype=np.float32)}, component / "model.safetensors")
    _validate_model(tmp_path)
    (component / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {"weight": "../unselected.safetensors"},
            }
        )
    )
    with pytest.raises(ValueError):
        _validate_model(tmp_path)


@pytest.mark.parametrize("contract", ["image-editing-v1", "image-layers-v1"])
def test_media_contracts_reject_authority_and_unimplemented_controls(contract):
    assert validate_media_input(contract, BASE)["steps"] == 20
    for field in ("backend", "modelRoot", "strength", "runtimeOptions", "trust_remote_code"):
        with pytest.raises(ValueError):
            validate_media_input(contract, {**BASE, field: 1})
    for change in ({"steps": 51}, {"seed": -1}, {"guidanceScale": 11}, {"prompt": " "}):
        with pytest.raises(ValueError):
            validate_media_input(contract, {**BASE, **change})
    invalid = {"width": 65} if contract == "image-editing-v1" else {"layers": 5}
    with pytest.raises(ValueError):
        validate_media_input(contract, {**BASE, **invalid})


def test_native_pipeline_components_are_closed():
    for family, components in IMAGE_PIPELINES.items():
        index = {
            "_class_name": family,
            **{key: list(value) for key, value in components.items()},
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "tokenizer": ["transformers", "Qwen2Tokenizer"],
        }
        if family == "QwenImage21Pipeline":
            del index["tokenizer"]
        validate_pipeline_index(index)
        for change in ({"custom_pipeline": "repo.code"}, {"transformer": ["repo", "Code"]}):
            with pytest.raises(ValueError):
                validate_pipeline_index({**index, **change})


def test_editing_and_layers_map_native_semantics_without_silent_option_loss():
    value = validate_media_input("image-editing-v1", BASE)
    options = {"guidance_scale": 3, "negative_prompt": "", "image": object()}
    mapped = image_options("Flux2KleinPipeline", "image-editing-v1", value, options)
    assert mapped["image"] is options["image"] and mapped["guidance_scale"] == 3
    assert "negative_prompt" not in mapped and "strength" not in mapped
    with pytest.raises(ValueError):
        image_options(
            "Flux2KleinPipeline", "image-editing-v1", {**value, "negativePrompt": "x"}, options
        )
    mapped = image_options("QwenImageEditPlusPipeline", "image-editing-v1", value, options)
    assert mapped["true_cfg_scale"] == 3 and "guidance_scale" not in mapped
    layers = validate_media_input("image-layers-v1", BASE)
    mapped = image_options("QwenImageLayeredPipeline", "image-layers-v1", layers, options)
    assert (mapped["layers"], mapped["resolution"]) == (4, 640)
    with pytest.raises(ValueError):
        image_options("ZImagePipeline", "image-editing-v1", value, options)


def test_layer_set_validation_rejects_partial_escape_or_inconsistent_geometry(tmp_path):
    descriptors = []
    for i in range(2):
        Image.new("RGBA", (64, 64), (255, 0, 0, 128)).save(tmp_path / f"layer-{i}.png")
        descriptors.append(
            {
                "relativeOutput": f"layer-{i}.png",
                "mediaType": "image/png",
                "width": 64,
                "height": 64,
            }
        )
    result = {"backend": "mps", "layers": descriptors}
    assert len(validate_layer_outputs(result, tmp_path, 2)) == 2
    for change in (
        {"relativeOutput": "../layer-0.png"},
        {"width": 63},
        {"width": True},
        {"backend": "mps"},
    ):
        bad = deepcopy(result)
        bad["layers"][0].update(change)
        with pytest.raises(ValueError):
            validate_layer_outputs(bad, tmp_path, 2)
    with pytest.raises(ValueError):
        validate_layer_outputs(result, tmp_path, 3)
    Image.new("RGB", (64, 64)).save(tmp_path / "layer-1.png")
    with pytest.raises(ValueError):
        validate_layer_outputs(result, tmp_path, 2)
    (tmp_path / "layer-1.png").unlink()
    (tmp_path / "layer-1.png").symlink_to(tmp_path / "layer-0.png")
    with pytest.raises(ValueError):
        validate_layer_outputs(result, tmp_path, 2)


def test_layered_native_filter_cannot_publish_files(tmp_path):
    import torch

    class QwenImageLayeredPipeline:
        def __call__(self, **options):
            return SimpleNamespace(
                images=[[Image.new("RGBA", (64, 64))]], nsfw_content_detected=[True]
            )

    source = tmp_path / "input.png"
    Image.new("RGB", (64, 64)).save(source)
    with pytest.raises(WorkerFailure, match="MODEL_OUTPUT_FILTERED"):
        _generate_layers(
            QwenImageLayeredPipeline(),
            validate_media_input("image-layers-v1", {**BASE, "layers": 1}),
            source,
            tmp_path,
            torch,
            "cpu",
        )
    assert not list(tmp_path.glob("layer-*.png"))


def test_filtered_pipeline_cannot_publish_unchecked_latent_preview(tmp_path):
    from oryn_engine.runtime_packs.diffusers_worker import _preview_callback

    calls = []
    pipe = SimpleNamespace(
        safety_checker=object(),
        vae=SimpleNamespace(decode=lambda *_args, **_kwargs: calls.append("decode")),
    )
    values = {"latents": SimpleNamespace(ndim=4)}
    target = tmp_path / "preview.png"
    assert _preview_callback(target, 2)(pipe, 1, None, values) is values
    assert not calls and not target.exists()
