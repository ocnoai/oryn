"""Sealed contracts for additional native pipelines in the existing Diffusers pack."""

from __future__ import annotations

from typing import Any

# One task behavior per native pipeline; model repositories cannot extend these classes.
IMAGE_PIPELINES = {
    "ZImagePipeline": {
        "transformer": ("diffusers", "ZImageTransformer2DModel"),
        "vae": ("diffusers", "AutoencoderKL"),
        "text_encoder": ("transformers", "Qwen3Model"),
    },
    "Flux2KleinPipeline": {
        "transformer": ("diffusers", "Flux2Transformer2DModel"),
        "vae": ("diffusers", "AutoencoderKLFlux2"),
        "text_encoder": ("transformers", "Qwen3ForCausalLM"),
    },
}


QWEN_IMAGE_COMPONENTS = {
    "transformer": ("diffusers", "QwenImageTransformer2DModel"),
    "vae": ("diffusers", "AutoencoderKLQwenImage"),
    "text_encoder": ("transformers", "Qwen2_5_VLForConditionalGeneration"),
    "processor": ("transformers", "Qwen2VLProcessor"),
}
IMAGE_PIPELINES.update(
    {
        "QwenImageEditPlusPipeline": QWEN_IMAGE_COMPONENTS,
        "QwenImageLayeredPipeline": QWEN_IMAGE_COMPONENTS,
        "QwenImage21Pipeline": {
            "transformer": ("diffusers", "QwenImage21Transformer2DModel"),
            "vae": ("diffusers", "AutoencoderKLQwenImage21"),
            "text_encoder": ("transformers", "Qwen3VLForConditionalGeneration"),
            "processor": ("transformers", "Qwen3VLProcessor"),
        },
    }
)
MEDIA_CONTRACTS = frozenset({"image-editing-v1", "image-layers-v1"})
MULTI_REFERENCE_PIPELINES = frozenset(
    {"QwenImage21Pipeline", "QwenImageEditPlusPipeline", "Flux2KleinPipeline"}
)


def reference_image_limits(family, composition=None):
    minimum, maximum = 1, 3 if family in MULTI_REFERENCE_PIPELINES else 1
    if composition is not None:
        from oryn_engine.model_composition import validate_binding

        binding = validate_binding(composition)
        if binding["family"] != family:
            raise ValueError("Reference image pipeline differs from its composition")
        limits = binding.get("referenceImages", {})
        minimum, maximum = limits.get("minimum", minimum), limits.get("maximum", maximum)
    return minimum, maximum


def validate_reference_images(family, count, composition=None):
    minimum, maximum = reference_image_limits(family, composition)
    if type(count) is not int or not minimum <= count <= maximum:
        raise ValueError(f"This pipeline requires {minimum} to {maximum} reference images")


UNCONDITIONAL_PIPELINES = {
    "DDPMPipeline": {
        "unet": ("diffusers", "UNet2DModel"),
        "scheduler": ("diffusers", "DDPMScheduler"),
    },
    "ScoreSdeVePipeline": {
        "unet": ("diffusers", "UNet2DModel"),
        "scheduler": ("diffusers", "ScoreSdeVeScheduler"),
    },
}

# Bounded native video binding. Other generated pipeline identities remain
# discoverable but do not inherit a video invocation contract from their name.
VIDEO_PIPELINES = {
    "CogVideoXPipeline": {
        "transformer": ("diffusers", "CogVideoXTransformer3DModel"),
        "vae": ("diffusers", "AutoencoderKLCogVideoX"),
        "text_encoder": ("transformers", "T5EncoderModel"),
    },
    "WanPipeline": {
        "transformer": ("diffusers", "WanTransformer3DModel"),
        "vae": ("diffusers", "AutoencoderKLWan"),
        "text_encoder": ("transformers", "UMT5EncoderModel"),
    },
    "LTXPipeline": {
        "transformer": ("diffusers", "LTXVideoTransformer3DModel"),
        "vae": ("diffusers", "AutoencoderKLLTXVideo"),
        "text_encoder": ("transformers", "T5EncoderModel"),
    },
}
VIDEO_SCHEDULERS = {
    "CogVideoXPipeline": "CogVideoXDDIMScheduler",
    "WanPipeline": "UniPCMultistepScheduler",
    "LTXPipeline": "FlowMatchEulerDiscreteScheduler",
}
VIDEO_BINDINGS = {
    "CogVideoXPipeline": {"text-to-video-v1": "CogVideoXPipeline"},
    "WanPipeline": {"text-to-video-v1": "WanPipeline"},
    "LTXPipeline": {
        "text-to-video-v1": "LTXPipeline",
        "image-to-video-v1": "LTXImageToVideoPipeline",
        "image-text-to-video-v1": "LTXImageToVideoPipeline",
    },
}


def pipeline_contract(family: str, task: str) -> str:
    if family == "MiniMaxMusic3ModularPipeline" and task in {"text-to-audio", "text-to-music"}:
        return "text-to-music-v1"
    if task == "image-to-image" and family in {
        "Flux2KleinPipeline",
        "QwenImageEditPlusPipeline",
        "QwenImage21Pipeline",
    }:
        return "image-editing-v1"
    if task == "image-text-to-image" and family == "QwenImageLayeredPipeline":
        return "image-layers-v1"
    return f"{task}-v1"


def pipeline_contracts(family: str) -> set[str]:
    if not isinstance(family, str):
        return set()
    if family == "MiniMaxMusic3ModularPipeline":
        return {"text-to-music-v1"}
    if family == "MiniMaxH3ModularPipeline":
        return {"text-to-video-v1", "image-text-to-video-v1"}
    if family in VIDEO_BINDINGS:
        return set(VIDEO_BINDINGS[family])
    if family in UNCONDITIONAL_PIPELINES:
        return {"unconditional-image-generation-v1"}
    return {
        "ZImagePipeline": {"text-to-image-v1"},
        "Flux2KleinPipeline": {"text-to-image-v1", "image-editing-v1"},
        "QwenImageEditPlusPipeline": {"image-editing-v1"},
        "QwenImageLayeredPipeline": {"image-layers-v1"},
        "QwenImage21Pipeline": {"text-to-image-v1", "image-editing-v1"},
    }.get(family, set())


def unconditional_geometry(config):
    from oryn_engine.model_metadata import validate_model_config

    validate_model_config(config)
    size = config.get("sample_size")
    if type(size) is int:
        size = [size, size]
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or any(type(v) is not int or not 8 <= v <= 1024 or v % 8 for v in size)
        or config.get("in_channels") != 3
        or config.get("out_channels") != 3
    ):
        raise ValueError("Unconditional image geometry/channels are not admitted")
    return {"height": size[0], "width": size[1]}


def unconditional_generation_defaults(geometry):
    """Expose native raster limits using the existing public profile schema."""
    from oryn_engine.runtime_packs.generation_profiles import generation_profile

    result = generation_profile(task_contract="text-to-image-v1")
    result.update(profileId="runtime.unconditional.native-raster", provenance="adapter-derived")
    for name in ("width", "height"):
        result["parameters"][name].update(
            default=geometry[name],
            minimum=geometry[name],
            maximum=geometry[name],
            step=8,
            editable=False,
        )
    result["parameters"]["steps"].update(default=50, maximum=1000)
    result["parameters"]["guidanceScale"].update(
        default=0.0, maximum=0.0, supported=False, editable=False
    )
    return result


def klein_generation_defaults(index: dict[str, Any]) -> dict[str, Any]:
    """Closed native defaults; checkpoint flags cannot add controls or certify a run."""
    from oryn_engine.runtime_packs.generation_profiles import generation_profile

    validate_pipeline_index(index)
    if index.get("_class_name") != "Flux2KleinPipeline":
        raise ValueError("Klein defaults require a Klein pipeline")
    distilled = index.get("is_distilled", False)
    result = generation_profile(
        task_contract="image-editing-v1",
        trusted_metadata={
            "inferenceDefaults": {
                "width": 512,
                "height": 512,
                "steps": 4 if distilled else 50,
                "guidanceScale": 1.0 if distilled else 4.0,
            }
        },
    )
    result["profileId"] = (
        "runtime.flux2-klein.distilled" if distilled else "runtime.flux2-klein.base"
    )
    result["provenance"] = "adapter-derived"
    for name in ("width", "height"):
        result["parameters"][name].update(maximum=1024, step=16)
    result["parameters"]["steps"]["maximum"] = 50
    result["parameters"]["guidanceScale"].update(
        maximum=1.0 if distilled else 10.0, supported=not distilled, editable=not distilled
    )
    return result


def validate_media_input(contract: str, value: object) -> dict[str, Any]:
    from oryn_engine.shared_tasks import _schemas

    if contract not in MEDIA_CONTRACTS or not _schemas().is_valid(f"taskInput.{contract}", value):
        raise ValueError("Image task input is invalid")
    from oryn_protocol.task_contracts import validate_input_relations

    validate_input_relations(contract, value)
    result = {"negativePrompt": "", "steps": 20, "guidanceScale": 1.0, "seed": 0}
    result.update(
        {"width": 512, "height": 512}
        if contract == "image-editing-v1"
        else {"resolution": 640, "layers": 4}
    )
    result.update(value)
    if not result["prompt"].strip():
        raise ValueError("An explicit image prompt is required")
    return result


def validate_pipeline_index(index: dict[str, Any]) -> None:
    if index.get("_class_name") == "MiniMaxH3ModularPipeline":
        from oryn_engine.minimax_h3 import validate_index

        validate_index(index)
        return
    from oryn_engine import minimax_music

    if minimax_music.supports(index):
        minimax_music.validate_index(index)
        return

    from oryn_engine.flat_ddpm import MARKER, normalized_index

    index = normalized_index(index)
    family = index.get("_class_name")
    if family is not None and not isinstance(family, str):
        raise ValueError("Pipeline class identity is invalid")
    if family in UNCONDITIONAL_PIPELINES:
        components = UNCONDITIONAL_PIPELINES[family]
        # Early native DDPM exports recorded this obsolete loader hint. The
        # sealed loader constructs the bundled classes directly; it never imports it.
        legacy_module = (
            {"_module"}
            if family == "DDPMPipeline" and index.get("_module") == "modeling_ddpm.py"
            else set()
        )
        if MARKER in index and (family != "DDPMPipeline" or index[MARKER] is not True):
            raise ValueError("Flat DDPM marker is invalid")
        if not set(index) <= set(components) | {
            "_class_name",
            "_diffusers_version",
        } | legacy_module | ({MARKER} if index.get(MARKER) is True else set()) or any(
            index.get(name) != list(identity) for name, identity in components.items()
        ):
            raise ValueError("Unconditional pipeline components are not admitted")
        return
    if family in VIDEO_PIPELINES:
        components = VIDEO_PIPELINES[family]
        native_options = set()
        if family == "WanPipeline":
            native_options = {"boundary_ratio", "expand_timesteps", "transformer_2"}
            if (
                index.get("boundary_ratio") is not None
                or type(index.get("expand_timesteps", False)) is not bool
                or index.get("transformer_2", [None, None]) != [None, None]
            ):
                raise ValueError("Wan video requires one native transformer")
        if (
            not set(index)
            <= set(components)
            | {
                "scheduler",
                "tokenizer",
                "_class_name",
                "_diffusers_version",
            }
            | native_options
        ):
            raise ValueError("Video pipeline has undeclared components/options")
        if any(index.get(name) != list(identity) for name, identity in components.items()):
            raise ValueError("Video pipeline component identity is not admitted")
        if index.get("scheduler") != ["diffusers", VIDEO_SCHEDULERS[family]] or index.get(
            "tokenizer"
        ) not in (["transformers", "T5Tokenizer"], ["transformers", "T5TokenizerFast"]):
            raise ValueError("Video scheduler or tokenizer is not admitted")
        return
    if family not in IMAGE_PIPELINES:
        return
    components = IMAGE_PIPELINES[family]
    allowed = set(components) | {"scheduler", "tokenizer", "_class_name", "_diffusers_version"}
    if family == "QwenImage21Pipeline":
        from oryn_engine.qwen_image_21 import sample_sigmas

        # Its tokenizer is owned by the processor; a separate tokenizer is not a component.
        allowed.remove("tokenizer")
        allowed.add("sample_sigmas")
        sample_sigmas(index)
    if family == "Flux2KleinPipeline":
        allowed.add("is_distilled")
        if "is_distilled" in index and type(index["is_distilled"]) is not bool:
            raise ValueError("Distillation metadata is invalid")
    if not set(index) <= allowed:
        raise ValueError("Pipeline has undeclared components/options")
    for name, identity in components.items():
        if index.get(name) != list(identity):
            raise ValueError("Pipeline component identity is not admitted")
    if index.get("scheduler") != ["diffusers", "FlowMatchEulerDiscreteScheduler"]:
        raise ValueError("Pipeline scheduler is not admitted")
    if family != "QwenImage21Pipeline" and index.get("tokenizer") not in (
        ["transformers", "Qwen2Tokenizer"],
        ["transformers", "Qwen2TokenizerFast"],
    ):
        raise ValueError("Pipeline tokenizer is not admitted")


def image_options(
    family: str, contract: str, value: dict[str, Any], options: dict[str, Any], *, distilled=False
) -> dict[str, Any]:
    if family not in IMAGE_PIPELINES:
        return options
    if contract not in pipeline_contracts(family):
        raise ValueError("This pipeline has no admitted contract for this task")
    if family == "Flux2KleinPipeline" and distilled and value["guidanceScale"] != 1.0:
        raise ValueError("Distilled Klein does not implement classifier-free guidance")
    if contract == "image-layers-v1":
        options = dict(options)
        options["true_cfg_scale"] = options.pop("guidance_scale")
        options.update(
            resolution=value["resolution"], layers=value["layers"], max_sequence_length=512
        )
        return options
    multiple = 32 if family == "QwenImage21Pipeline" else 16
    if value["width"] % multiple or value["height"] % multiple:
        raise ValueError(f"This pipeline requires image dimensions divisible by {multiple}")
    if value["width"] * value["height"] > 1024**2:
        raise ValueError("This pipeline exceeds the admitted image area")
    options = dict(options)
    if family == "QwenImage21Pipeline":
        if value["steps"] > 50 or value["guidanceScale"] > 10:
            raise ValueError("Qwen Image 2.1 exceeds the admitted generation limits")
        options["true_cfg_scale"] = options.pop("guidance_scale")
        # Bound the reference-image preprocessing separately from output dimensions.
        # This native pipeline has no max_sequence_length or guidance_scale argument.
        options["output_resolution"] = 512
        return options
    options["max_sequence_length"] = 512
    if family == "Flux2KleinPipeline":
        if value["negativePrompt"]:
            raise ValueError("This pipeline does not support a negative prompt")
        del options["negative_prompt"]
    if family == "QwenImageEditPlusPipeline":
        options["true_cfg_scale"] = options.pop("guidance_scale")
    return options


def video_options(
    family: str,
    contract: str,
    value: dict[str, Any],
    options: dict[str, Any],
    *,
    expanded_wan=False,
) -> dict[str, Any]:
    graph_family = "LTXPipeline" if family == "LTXImageToVideoPipeline" else family
    spatial_multiple = 32 if graph_family == "LTXPipeline" or expanded_wan else 16
    if expanded_wan and graph_family != "WanPipeline":
        raise ValueError("Expanded timesteps require a Wan native pipeline")
    if (
        graph_family not in VIDEO_BINDINGS
        or contract not in VIDEO_BINDINGS[graph_family]
        or VIDEO_BINDINGS[graph_family][contract] != family
    ):
        raise ValueError("Video pipeline task binding is unavailable")
    if (
        not 64 <= value["width"] <= 512
        or not 64 <= value["height"] <= 512
        or value["width"] % spatial_multiple
        or value["height"] % spatial_multiple
    ):
        raise ValueError("Video dimensions exceed the native binding")
    frames = {1, 9, 17} if graph_family == "LTXPipeline" else {1, 5, 9, 13, 17}
    if value["frames"] not in frames or value["steps"] > 30:
        raise ValueError("Video duration or steps exceed the native binding")
    return {
        **options,
        "max_sequence_length": {"CogVideoXPipeline": 226, "WanPipeline": 512, "LTXPipeline": 128}[
            graph_family
        ],
        **({"frame_rate": value["fps"]} if graph_family == "LTXPipeline" else {}),
    }


def seal_tokenizer(tokenizer) -> None:
    # Native pipelines call this fixed method. Repository Jinja is never compiled or rendered.
    vocabulary = tokenizer.get_vocab()
    if not all(token in vocabulary for token in ("<|im_start|>", "<|im_end|>")):
        raise ValueError("Pipeline tokenizer framing is unsupported")

    def render(messages, *, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        if tokenize or not add_generation_prompt or len(messages) != 1:
            raise ValueError("Unexpected native prompt contract")
        message = messages[0]
        if set(message) != {"role", "content"} or message["role"] != "user":
            raise ValueError("Unexpected native prompt role")
        from oryn_engine.chat_formats import render

        return render("chatml", [message], thinking_off=not enable_thinking)

    tokenizer.apply_chat_template = render


def validate_layer_outputs(result, directory, expected_count):
    """Validate the entire output set before any layer becomes a Browser artifact."""
    from PIL import Image

    if not isinstance(result, dict) or set(result) != {"layers", "backend"}:
        raise ValueError("Layer result is invalid")
    if result["backend"] not in {"cpu", "mps", "cuda"}:
        raise ValueError("Layer backend is invalid")
    layers = result["layers"]
    if not isinstance(layers, list) or len(layers) != expected_count or not 1 <= len(layers) <= 4:
        raise ValueError("Layer count is invalid")
    paths, total, geometry = [], 0, None
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict) or set(layer) != {
            "relativeOutput",
            "mediaType",
            "width",
            "height",
        }:
            raise ValueError("Layer descriptor is invalid")
        if layer["relativeOutput"] != f"layer-{index}.png" or layer["mediaType"] != "image/png":
            raise ValueError("Layer path or type is invalid")
        if type(layer["width"]) is not int or type(layer["height"]) is not int:
            raise ValueError("Layer dimensions are invalid")
        path = directory / layer["relativeOutput"]
        if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 16 * 1024**2:
            raise ValueError("Layer file is invalid")
        total += path.stat().st_size
        if total > 64 * 1024**2:
            raise ValueError("Layer output budget exceeded")
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGBA" or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Layer must be a single RGBA PNG")
            if (
                not 1 <= image.width <= 2048
                or not 1 <= image.height <= 2048
                or image.width * image.height > 1024**2
            ):
                raise ValueError("Layer dimensions exceed the contract")
            if image.size != (layer["width"], layer["height"]) or geometry not in {
                None,
                image.size,
            }:
                raise ValueError("Layer coordinate systems disagree")
            geometry = image.size
            image.verify()
        paths.append(path)
    return paths
