from __future__ import annotations

import inspect
import json
import math
import os
import re
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

MAX_REQUEST_BYTES = 1024 * 1024
MAX_IMAGE_BYTES = 256 * 1024 * 1024
MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024
SAFE_COMPONENT_LIBRARIES = {"diffusers", "transformers"}
FORBIDDEN_MODEL_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".dll",
    ".dylib",
    ".exe",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".py",
    ".so",
}
IMAGE_CONTRACTS = {
    "text-to-image-v1",
    "image-to-image-v1",
    "image-editing-v1",
    "image-layers-v1",
    "unconditional-image-generation-v1",
}
VIDEO_CONTRACTS = {"text-to-video-v1", "image-to-video-v1", "image-text-to-video-v1"}


class WorkerFailure(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _message(value: dict[str, Any]) -> None:
    print(json.dumps(value, separators=(",", ":")), flush=True)


def _event(stage: str, progress: float) -> None:
    _message(
        {
            "type": "event",
            "event": "job.progress",
            "data": {"stage": stage, "progress": progress},
        }
    )


def _request() -> dict[str, Any]:
    line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
    if not line or len(line) > MAX_REQUEST_BYTES:
        raise ValueError("request size is invalid")
    value = json.loads(line)
    if not isinstance(value, dict) or set(value) != {
        "abi",
        "packId",
        "version",
        "command",
        "payload",
    }:
        raise ValueError("request envelope is invalid")
    if value["abi"] != "1" or value["packId"] != "oryn.diffusers-pytorch":
        raise ValueError("request identity is invalid")
    if value["command"] not in {"probe", "generate"} or not isinstance(value["payload"], dict):
        raise ValueError("request command is invalid")
    return value


def _validated_paths(payload: dict[str, Any]) -> tuple[Path, Path, Path | None, Path | None]:
    model_root = Path(payload.get("modelRoot", ""))
    output_path = Path(payload.get("outputPath", ""))
    preview_value = payload.get("previewPath")
    input_value = payload.get("inputPath")
    preview_path = Path(preview_value) if isinstance(preview_value, str) else None
    input_path = Path(input_value) if isinstance(input_value, str) else None
    if not model_root.is_absolute() or not model_root.is_dir() or not output_path.is_absolute():
        raise ValueError("runtime-pack paths are invalid")
    contract = payload.get("taskContract")
    expected_name = (
        "audio.wav"
        if contract == "text-to-music-v1"
        else "image.png"
        if contract in IMAGE_CONTRACTS
        else "video.mp4"
    )
    if output_path.name != expected_name or not output_path.parent.is_dir():
        raise ValueError("runtime-pack output path is invalid")
    if preview_path is not None and (
        not preview_path.is_absolute()
        or preview_path.parent != output_path.parent
        or preview_path.name != "preview.png"
    ):
        raise ValueError("runtime-pack preview path is invalid")
    if input_path is not None and (not input_path.is_absolute() or not input_path.is_file()):
        raise ValueError("runtime-pack input path is invalid")
    return model_root.resolve(), output_path.resolve(), preview_path, input_path


def _reference_paths(payload, value, contract):
    paths = payload.get("referenceImagePaths", [])
    if (
        not isinstance(paths, list)
        or len(paths) > 2
        or len(paths) != len(value.get("referenceImageArtifactIds", []))
        or (paths and contract != "image-editing-v1")
        or any(not isinstance(path, str) for path in paths)
    ):
        raise ValueError("Reference image bindings are invalid")
    result = tuple(Path(path) for path in paths)
    if any(not path.is_absolute() or not path.is_file() for path in result):
        raise ValueError("Reference image paths are invalid")
    return result


def _validate_model(model_root: Path) -> None:
    index_path = model_root / "model_index.json"
    repository_root = model_root.resolve()
    resolved_index = index_path.resolve()
    if (
        not index_path.is_file()
        or not resolved_index.is_relative_to(repository_root)
        or not resolved_index.is_file()
    ):
        raise ValueError("Diffusers model index is unavailable")
    if resolved_index.stat().st_size > MAX_REQUEST_BYTES:
        raise ValueError("Diffusers model index is invalid")
    from oryn_engine import flat_ddpm
    from oryn_engine.diffusion_tasks import IMAGE_PIPELINES, validate_pipeline_index
    from oryn_engine.model_metadata import (
        metadata_file,
        preflight_safetensors,
        validate_model_config,
    )

    index = flat_ddpm.normalized_index(metadata_file(resolved_index))
    if flat_ddpm.supports(index):
        flat_ddpm.read_configs(model_root)
    validate_pipeline_index(index)
    from oryn_engine import wan_video

    if wan_video.expanded(index):
        wan_video.read_configs(model_root)
    if index.get("_class_name") in IMAGE_PIPELINES:
        for component in set(IMAGE_PIPELINES[index["_class_name"]]) - {"processor"}:
            component_root = model_root / component
            validate_model_config(metadata_file(component_root / "config.json"))
    if not isinstance(index, dict):
        raise ValueError("Diffusers model index is invalid")
    class_name = index.get("_class_name")
    if not isinstance(class_name, str) or not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_]{0,127}", class_name
    ):
        raise ValueError("Diffusers pipeline class is invalid")
    for name, component in index.items():
        if name.startswith("_") or component is None:
            continue
        if isinstance(component, (bool, int, float, str)):
            continue
        if component == [None, None]:
            continue
        # Diffusers model indexes may contain bounded declarative arrays in addition to
        # [library, class] component references. They remain data-only and cannot select code.
        if (
            isinstance(component, list)
            and len(component) <= 256
            and all(item is None or isinstance(item, (bool, int, float, str)) for item in component)
            and not (
                len(component) == 2
                and isinstance(component[0], str)
                and isinstance(component[1], str)
            )
        ):
            continue
        if (
            not isinstance(component, list)
            or len(component) != 2
            or (
                component[0] not in SAFE_COMPONENT_LIBRARIES
                and component != ["stable_diffusion", "StableDiffusionSafetyChecker"]
            )
            or not isinstance(component[1], str)
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", component[1])
        ):
            raise ValueError("Diffusers component declaration is invalid")
    from oryn_engine import numeric_unet2d

    # The only admitted archive is read by the closed numeric reader, never torch.load.
    numeric_archive = (
        model_root / numeric_unet2d.WEIGHTS
        if numeric_unet2d.selected(index, [path.name for path in model_root.iterdir()])
        else None
    )
    if numeric_archive is not None:
        numeric_unet2d.read_configs(model_root)
    weight_roots = set()
    for path in model_root.rglob("*"):
        if path.is_file():
            resolved = path.resolve()
            if not resolved.is_relative_to(repository_root) or not resolved.is_file():
                raise ValueError("Diffusers model path escapes its repository cache")
            if path.suffix.lower() in FORBIDDEN_MODEL_SUFFIXES and path != numeric_archive:
                raise ValueError("Diffusers model contains an unsafe executable or weight format")
            if path.name == "config.json":
                validate_model_config(metadata_file(path))
            if path.name.endswith((".safetensors", ".safetensors.index.json")):
                weight_roots.add(path.parent)
    # All native families use the same shard/index boundary, including older SD/SDXL.
    # Never let a native loader interpret unchecked weight_map paths.
    for component_root in sorted(weight_roots):
        preflight_safetensors(component_root)


def _validated_input(value: object, contract: str) -> dict[str, Any]:
    from oryn_engine.diffusion_tasks import MEDIA_CONTRACTS, validate_media_input

    if contract in MEDIA_CONTRACTS:
        return validate_media_input(contract, value)
    if contract == "unconditional-image-generation-v1":
        from oryn_protocol.task_contracts import effective_input, make_profile, validate_invocation

        profile = make_profile(contract)
        validate_invocation(profile, value)
        return effective_input(profile, value)
    if contract == "image-text-to-video-v1":
        from oryn_protocol.task_contracts import effective_input, make_profile, validate_invocation

        profile = make_profile(contract)
        validate_invocation(profile, value)
        return {**effective_input(profile, value), "negativePrompt": ""}
    is_video = contract in VIDEO_CONTRACTS
    needs_image = contract in {"image-to-image-v1", "image-to-video-v1"}
    expected = {
        "prompt",
        "negativePrompt",
        "width",
        "height",
        "steps",
        "guidanceScale",
        "seed",
    }
    if is_video:
        expected.update({"frames", "fps"})
    if needs_image:
        expected.add("imageArtifactId")
    if contract == "image-to-image-v1":
        expected.add("strength")
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Diffusers input is invalid")
    prompt = value.get("prompt")
    negative = value.get("negativePrompt")
    width = value.get("width")
    height = value.get("height")
    steps = value.get("steps")
    guidance = value.get("guidanceScale")
    seed = value.get("seed")
    if (
        not isinstance(prompt, str)
        or not 1 <= len(prompt) <= 4_096
        or not isinstance(negative, str)
        or len(negative) > 4_096
        or not isinstance(width, int)
        or isinstance(width, bool)
        or not 64 <= width <= 2_048
        or width % 8
        or not isinstance(height, int)
        or isinstance(height, bool)
        or not 64 <= height <= 2_048
        or height % 8
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
        raise ValueError("Diffusers input is invalid")
    if is_video:
        frames = value.get("frames")
        fps = value.get("fps")
        if (
            not isinstance(frames, int)
            or isinstance(frames, bool)
            or not 1 <= frames <= 120
            or not isinstance(fps, int)
            or isinstance(fps, bool)
            or not 1 <= fps <= 60
            or frames / fps > 20
            or width * height > 1_280 * 720
            or width * height * frames > 96 * 1024 * 1024
        ):
            raise ValueError("Diffusers video input is invalid")
    elif width * height > 2_048 * 2_048:
        raise ValueError("Diffusers image input is invalid")
    if needs_image and not isinstance(value.get("imageArtifactId"), str):
        raise ValueError("Diffusers image input is invalid")
    if contract == "image-to-image-v1":
        strength = value.get("strength")
        if (
            not isinstance(strength, (int, float))
            or isinstance(strength, bool)
            or not math.isfinite(float(strength))
            or not 0.05 <= float(strength) <= 1
        ):
            raise ValueError("Diffusers image input is invalid")
    return value


def _load_image(path: Path | None) -> Any:
    if path is None or path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("input image is invalid")
    from PIL import Image

    with Image.open(path) as source:
        source.verify()
    with Image.open(path) as source:
        if source.width < 1 or source.height < 1 or source.width * source.height > 16_777_216:
            raise ValueError("input image dimensions are invalid")
        return source.convert("RGB")


def _backend(torch: Any) -> tuple[str, Any]:
    if bool(getattr(torch.cuda, "is_available", lambda: False)()):
        return "cuda", torch.float16
    if bool(getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()):
        return "mps", torch.float16
    return "cpu", torch.float32


def requested_backend(torch, backend):
    if backend == "cpu":
        return "cpu", torch.float32
    if backend == "cuda" and torch.cuda.is_available():
        return "cuda", torch.float16
    if backend == "mps" and torch.backends.mps.is_available():
        return "mps", torch.float16
    raise ValueError("The bound runtime backend is unavailable")


def _pipeline(
    model_root: Path, contract: str, torch: Any, *, backend_request=None, composition=None
) -> tuple[Any, str]:
    import diffusers
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image, DiffusionPipeline

    from oryn_engine import flat_ddpm

    index = flat_ddpm.normalized_index(
        json.loads((model_root / "model_index.json").read_text(encoding="utf-8"))
    )
    if not hasattr(diffusers, index["_class_name"]):
        raise ValueError("Diffusers pipeline is not bundled")
    backend, dtype = (
        _backend(torch) if backend_request is None else requested_backend(torch, backend_request)
    )
    loader: Any
    from oryn_engine.diffusion_tasks import (
        IMAGE_PIPELINES,
        UNCONDITIONAL_PIPELINES,
        seal_tokenizer,
        validate_pipeline_index,
    )

    validate_pipeline_index(index)
    component_overrides = {}
    from oryn_engine import wan_video

    if wan_video.expanded(index):
        wan_video.preflight(model_root)
        component_overrides.update(expand_timesteps=True, boundary_ratio=None, transformer_2=None)
    if index["_class_name"] == "QwenImage21Pipeline":
        from oryn_engine.qwen_image_21 import load_processor

        # The native constructor calls the processor; seal it before from_pretrained.
        component_overrides["processor"] = load_processor(model_root)
    if contract == "unconditional-image-generation-v1":
        if index["_class_name"] not in UNCONDITIONAL_PIPELINES:
            raise ValueError("Unconditional task binding is unavailable")
        loader = getattr(diffusers, index["_class_name"])
        dtype = torch.float32
    elif index["_class_name"] in IMAGE_PIPELINES:
        from oryn_engine.diffusion_tasks import pipeline_contracts

        if backend in {"mps", "cuda"}:
            dtype = torch.bfloat16
        if contract not in pipeline_contracts(index["_class_name"]):
            raise ValueError("Pipeline task contract is unavailable")
        loader = getattr(diffusers, index["_class_name"])
    elif contract in VIDEO_CONTRACTS:
        from oryn_engine.diffusion_tasks import VIDEO_BINDINGS

        if (
            index["_class_name"] not in VIDEO_BINDINGS
            or contract not in VIDEO_BINDINGS[index["_class_name"]]
        ):
            raise ValueError("Video pipeline task binding is unavailable")
        loader = getattr(diffusers, VIDEO_BINDINGS[index["_class_name"]][contract])
    elif contract == "text-to-image-v1":
        loader = AutoPipelineForText2Image
    elif contract == "image-to-image-v1":
        loader = AutoPipelineForImage2Image
    else:
        loader = DiffusionPipeline
    nested_safe_weights = tuple(
        path for path in model_root.rglob("*.safetensors") if path.parent != model_root
    )
    root_safe_weights = tuple(model_root.glob("*.safetensors"))
    from oryn_engine import numeric_unet2d

    if flat_ddpm.supports(index):
        if contract != "unconditional-image-generation-v1":
            raise ValueError("Flat DDPM task binding is unavailable")
        pipe = flat_ddpm.load_pipeline(model_root)
    elif numeric_unet2d.selected(index, [path.name for path in model_root.iterdir()]):
        if contract != "unconditional-image-generation-v1" or root_safe_weights:
            raise ValueError("Numeric UNet2D task binding is unavailable")
        pipe = numeric_unet2d.load_pipeline(model_root)
    elif nested_safe_weights:
        variant = (
            "fp16"
            if any(path.name.endswith(".fp16.safetensors") for path in nested_safe_weights)
            else None
        )
        pipe = loader.from_pretrained(
            str(model_root),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=dtype,
            variant=variant,
            **component_overrides,
        )
    elif len(root_safe_weights) == 1 and contract == "unconditional-image-generation-v1":
        from oryn_engine.diffusion_tasks import unconditional_geometry
        from oryn_engine.model_metadata import metadata_file

        unconditional_geometry(metadata_file(model_root / "config.json"))
        pipe = loader(
            unet=diffusers.UNet2DModel.from_pretrained(
                model_root, local_files_only=True, use_safetensors=True, torch_dtype=dtype
            ),
            scheduler=getattr(
                diffusers, UNCONDITIONAL_PIPELINES[index["_class_name"]]["scheduler"][1]
            ).from_pretrained(model_root, local_files_only=True),
        )
    elif len(root_safe_weights) == 1:
        pipeline_class = getattr(diffusers, index["_class_name"])
        from_single_file = getattr(pipeline_class, "from_single_file", None)
        if not callable(from_single_file):
            raise ValueError("Diffusers pipeline does not support a safe single-file checkpoint")
        pipe = from_single_file(
            str(root_safe_weights[0]),
            config=str(model_root),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=dtype,
        )
    else:
        raise ValueError("Diffusers model has no unambiguous safe weight layout")
    if index["_class_name"] == "QwenImage21Pipeline":
        from oryn_engine.qwen_image_21 import sample_sigmas, stable_mps_downsampling

        # The pinned native pipeline already accepts call-time sigmas. Restore the
        # validated checkpoint grid explicitly; older Diffusers ignores it on load.
        # Keep it out of native config: older Diffusers treats unknown public
        # config entries as modules when enumerating pipeline components.
        pipe._oryn_sample_sigmas = sample_sigmas(index)
        stable_mps_downsampling(pipe.vae)
    if index["_class_name"] in IMAGE_PIPELINES and index["_class_name"] != "QwenImage21Pipeline":
        seal_tokenizer(pipe.tokenizer)
        if index["_class_name"].startswith("QwenImage"):
            # Construct a closed image-only processor; do not render repository templates
            # or inherit a repository-controlled pixel budget/video preprocessing policy.
            import transformers as tf

            from oryn_engine.model_metadata import metadata_file
            from oryn_engine.shared_tasks import qwen_image_options, validate_processor_config

            processor_root = model_root / "processor"
            file = processor_root / "processor_config.json"
            if not file.exists():
                file = processor_root / "preprocessor_config.json"
            metadata = metadata_file(file)
            validate_processor_config(metadata, "qwen2_vl")
            config = metadata_file(model_root / "text_encoder/config.json")
            options = qwen_image_options(metadata, config.get("vision_config", {}))
            pipe.processor = tf.Qwen2VLProcessor(
                tokenizer=pipe.tokenizer,
                image_processor=tf.Qwen2VLImageProcessor(**options),
                video_processor=tf.Qwen3VLVideoProcessor(
                    size=options["size"], min_frames=1, max_frames=1, do_sample_frames=False
                ),
            )
            if index["_class_name"] == "QwenImageLayeredPipeline":
                pipe.vl_processor = pipe.processor
    if composition is not None:
        from oryn_engine.diffusion_lora import configure_pipeline

        configure_pipeline(pipe, model_root, composition)
    elif (model_root / "oryn_adapter").exists():
        raise ValueError("Adapter files require a pinned composition binding")
    if backend == "cpu":
        pipe = pipe.to("cpu")
    elif hasattr(pipe, "enable_model_cpu_offload") and backend == "cuda":
        pipe.enable_model_cpu_offload()
    elif contract in VIDEO_CONTRACTS:
        # CogVideoX constructs positional buffers in float64 on CPU. Apply the
        # admitted precision to those buffers as well as weights before MPS,
        # which cannot store float64. Native forward uses the same model dtype.
        pipe = pipe.to(backend, dtype=dtype)
    else:
        pipe = pipe.to(backend)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if contract in VIDEO_CONTRACTS:
        _enable_video_vae_tiling(pipe)
    return pipe, backend


def _enable_video_vae_tiling(pipe: Any) -> None:
    if type(pipe.vae).__name__ in {"AutoencoderKLWan", "AutoencoderKLLTXVideo"}:
        # Keep the pinned native overlap and temporal tile geometry intact.
        pipe.vae.enable_tiling()
        return
    # CogVideoX exposes tiling on its VAE, not on the pipeline. Bound the
    # native spatial decoder tiles so full clips do not exhaust MPS memory.
    pipe.vae.enable_slicing()
    # 240 aligns both native overlap fractions (1/6 vertically, 1/5
    # horizontally) with the eight-pixel latent grid. Arbitrary smaller
    # tiles can round the overlap differently and alter output geometry.
    pipe.vae.enable_tiling(tile_sample_min_height=240, tile_sample_min_width=240)


def _preview_callback(preview_path: Path | None, total_steps: int) -> Any:
    emitted = False

    def callback(pipe: Any, step: int, _timestep: Any, values: dict[str, Any]) -> dict[str, Any]:
        nonlocal emitted
        progress = min(0.9, 0.15 + (step + 1) / max(1, total_steps) * 0.7)
        _event("denoising", progress)
        if (
            emitted
            or preview_path is None
            or getattr(pipe, "safety_checker", None) is not None
            or step + 1 < max(1, total_steps // 2)
        ):
            # A latent preview bypasses the native final-image checker. Keep progress,
            # but publish no unchecked intermediate image from a filtered pipeline.
            return values
        latents = values.get("latents")
        try:
            if getattr(latents, "ndim", 0) != 4 or not hasattr(pipe, "vae"):
                return values
            scaling = float(getattr(pipe.vae.config, "scaling_factor", 0.18215))
            decoded = pipe.vae.decode(latents / scaling, return_dict=False)[0]
            image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
            image.thumbnail((256, 256))
            image.save(preview_path, format="PNG", optimize=True)
            if preview_path.stat().st_size <= MAX_IMAGE_BYTES:
                _message(
                    {
                        "type": "event",
                        "event": "pack.preview",
                        "data": {
                            "relativeOutput": preview_path.name,
                            "mediaType": "image/png",
                            "sequence": 1,
                        },
                    }
                )
                emitted = True
        except Exception:
            # Preview decode is optional; it must never make final generation fail.
            return values
        return values

    return callback


def _generate_image(
    pipe: Any,
    contract: str,
    value: dict[str, Any],
    input_path: Path | None,
    output_path: Path,
    preview_path: Path | None,
    torch: Any,
    *,
    reference_paths: tuple[Path, ...] = (),
) -> None:
    from oryn_engine.diffusion_lora import validate_sampling

    validate_sampling(pipe, value)
    family = type(pipe).__name__
    if reference_paths and contract != "image-editing-v1":
        raise ValueError("This task does not accept reference images")
    if contract == "image-editing-v1":
        from oryn_engine.diffusion_tasks import validate_reference_images

        validate_reference_images(
            family, 1 + len(reference_paths), getattr(pipe, "_oryn_composition", None)
        )
    generator = torch.Generator(device="cpu").manual_seed(value["seed"])
    if contract == "unconditional-image-generation-v1":
        from oryn_engine.diffusion_tasks import UNCONDITIONAL_PIPELINES

        if family not in UNCONDITIONAL_PIPELINES or input_path is not None:
            raise ValueError("Unconditional sampling has unexpected conditioning")
        sample_size = pipe.unet.config.sample_size
        height, width = (sample_size, sample_size) if type(sample_size) is int else sample_size
        if (width, height) != (value["width"], value["height"]):
            raise ValueError("Requested dimensions differ from native sample geometry")
        _event("sampling", 0.2)
        result = pipe(
            batch_size=1, generator=generator, num_inference_steps=value["steps"], output_type="pil"
        )
        images = getattr(result, "images", None)
        if not isinstance(images, list) or len(images) != 1 or images[0].size != (width, height):
            raise ValueError("Native unconditional sample geometry is invalid")
        images[0].convert("RGB").save(output_path, format="PNG")
        return
    options: dict[str, Any] = {
        "prompt": value["prompt"],
        "negative_prompt": value["negativePrompt"],
        "width": value["width"],
        "height": value["height"],
        "num_inference_steps": value["steps"],
        "guidance_scale": value["guidanceScale"],
        "generator": generator,
        "callback_on_step_end": _preview_callback(preview_path, value["steps"]),
        "callback_on_step_end_tensor_inputs": ["latents"],
    }
    if contract == "image-to-image-v1":
        options.update({"image": _load_image(input_path), "strength": value["strength"]})
    if contract == "image-editing-v1" and family != "QwenImage21Pipeline":
        from oryn_engine.shared_tasks import open_task_image

        images = [open_task_image(path)[0] for path in (input_path, *reference_paths)]
        options["image"] = images if reference_paths else images[0]
    from oryn_engine.diffusion_tasks import IMAGE_PIPELINES, image_options

    if family == "QwenImage21Pipeline":
        from oryn_engine.qwen_image_21 import reference_image, sample_sigmas, validate_prompts

        sigmas = sample_sigmas({"sample_sigmas": getattr(pipe, "_oryn_sample_sigmas", None)})
        if sigmas is not None:
            if value["steps"] != len(sigmas):
                raise ValueError(f"This checkpoint requires {len(sigmas)} denoising steps")
            options["sigmas"] = sigmas
        validate_prompts(pipe, value)
        if contract == "image-editing-v1":
            images = [reference_image(path) for path in (input_path, *reference_paths)]
            options["image"] = images if reference_paths else images[0]
    options = image_options(
        family,
        contract,
        value,
        options,
        distilled=getattr(getattr(pipe, "config", None), "is_distilled", False),
    )
    if family in IMAGE_PIPELINES:
        # Every option has reviewed semantics; never silently discard user requests.
        result = pipe(**options)
    else:
        required = {"image"} if contract == "image-to-image-v1" else {"prompt"}
        result = pipe(**_supported_options(pipe, options, required))
    filtered = getattr(result, "nsfw_content_detected", None)
    if filtered is not None and any(filtered):
        raise WorkerFailure("MODEL_OUTPUT_FILTERED")
    images = getattr(result, "images", None)
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("Diffusers image output is invalid")
    image = images[0]
    if image.width != value["width"] or image.height != value["height"]:
        image = image.resize((value["width"], value["height"]))
    image.convert("RGBA" if family == "QwenImage21Pipeline" else "RGB").save(
        output_path, format="PNG", optimize=True
    )


def _generate_layers(pipe, value, input_path, directory, torch, backend):
    from oryn_engine.diffusion_tasks import image_options, validate_layer_outputs
    from oryn_engine.shared_tasks import open_task_image

    image, _ = open_task_image(input_path)
    if max(image.size) / min(image.size) > 2:
        raise ValueError("Layer decomposition requires aspect ratio between 1:2 and 2:1")
    options = image_options(
        type(pipe).__name__,
        "image-layers-v1",
        value,
        {
            "image": image,
            "prompt": value["prompt"],
            "negative_prompt": value["negativePrompt"],
            "num_inference_steps": value["steps"],
            "guidance_scale": value["guidanceScale"],
            "generator": torch.Generator(device="cpu").manual_seed(value["seed"]),
            "callback_on_step_end": _preview_callback(None, value["steps"]),
        },
    )
    result = pipe(**options)
    if any(getattr(result, "nsfw_content_detected", None) or []):
        raise WorkerFailure("MODEL_OUTPUT_FILTERED")
    batches = getattr(result, "images", None)
    if (
        not isinstance(batches, list)
        or len(batches) != 1
        or not isinstance(batches[0], list)
        or len(batches[0]) != value["layers"]
    ):
        raise ValueError("Native layered image count is invalid")
    outputs = []
    for index, layer in enumerate(batches[0]):
        if layer.mode != "RGBA" or layer.width * layer.height > 1024**2 or max(layer.size) > 2048:
            raise ValueError("Native layer is not bounded RGBA")
        filename = f"layer-{index}.png"
        layer.save(directory / filename, format="PNG")
        outputs.append(
            {
                "relativeOutput": filename,
                "mediaType": "image/png",
                "width": layer.width,
                "height": layer.height,
            }
        )
    result = {"layers": outputs, "backend": backend}
    validate_layer_outputs(result, directory, value["layers"])
    return result


def _generate_video(
    pipe: Any,
    contract: str,
    value: dict[str, Any],
    input_path: Path | None,
    output_path: Path,
    torch: Any,
) -> None:
    import imageio.v3 as imageio
    import numpy as np

    generator = torch.Generator(device="cpu").manual_seed(value["seed"])
    options: dict[str, Any] = {
        "prompt": value["prompt"],
        "negative_prompt": value["negativePrompt"],
        "width": value["width"],
        "height": value["height"],
        "num_frames": value["frames"],
        "num_inference_steps": value["steps"],
        "guidance_scale": value["guidanceScale"],
        "generator": generator,
    }
    if contract in {"image-to-video-v1", "image-text-to-video-v1"}:
        options["image"] = _load_image(input_path)
    from oryn_engine.diffusion_tasks import video_options

    options = video_options(
        type(pipe).__name__,
        contract,
        value,
        options,
        expanded_wan=type(pipe).__name__ == "WanPipeline"
        and getattr(pipe.config, "expand_timesteps", False),
    )
    _event("denoising", 0.2)
    required = set(options)
    result = pipe(**_supported_options(pipe, options, required))
    raw_frames = getattr(result, "frames", None)
    has_outer_batch = (
        isinstance(raw_frames, np.ndarray) and raw_frames.ndim == 5 and raw_frames.shape[0] == 1
    ) or (
        isinstance(raw_frames, (list, tuple))
        and len(raw_frames) == 1
        and isinstance(raw_frames[0], (list, tuple, np.ndarray))
    )
    if has_outer_batch:
        raw_frames = raw_frames[0]
    if not isinstance(raw_frames, (list, tuple, np.ndarray)):
        raise ValueError("Diffusers video output is invalid")
    frames = list(raw_frames)
    if len(frames) != value["frames"]:
        raise ValueError("Diffusers video output is invalid")
    arrays = []
    for frame in frames:
        array = np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame)
        if array.ndim != 3 or array.shape[2] != 3 or array.shape[0] * array.shape[1] > 1_280 * 720:
            raise ValueError("Diffusers video frame is invalid")
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
                raise ValueError("Diffusers video frame is invalid")
            array = np.rint(array * 255).astype(np.uint8)
        elif array.dtype != np.uint8:
            raise ValueError("Diffusers video frame is invalid")
        if array.shape[:2] != (value["height"], value["width"]):
            raise ValueError("Diffusers video did not honor the requested geometry")
        arrays.append(array)
    _event("encoding-video", 0.9)
    imageio.imwrite(output_path, np.stack(arrays), fps=value["fps"], codec="libx264")


def _supported_options(pipe: Any, options: dict[str, Any], required: set[str]) -> dict[str, Any]:
    """Adapt Oryn's stable task contract to the trusted pipeline's native call signature."""
    signature = inspect.signature(pipe.__call__)
    accepts_arbitrary_options = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_arbitrary_options:
        return options
    supported = set(signature.parameters)
    if not required.issubset(supported):
        raise ValueError("Diffusers pipeline does not implement the requested task contract")
    return {name: option for name, option in options.items() if name in supported}


def _generate(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    if payload.get("taskContract") == "text-to-music-v1":
        from oryn_engine import minimax_music

        try:
            model_root, output_path, preview_path, input_path = _validated_paths(payload)
            value = minimax_music.validate_input(payload.get("input"))
            if (
                preview_path is not None
                or input_path is not None
                or payload.get("referenceImagePaths")
                or payload.get("composition")
                or payload.get("backend") not in {"cpu", "mps", "cuda"}
            ):
                raise ValueError("Music input bindings are invalid")
        except (OSError, TypeError, ValueError) as exc:
            raise WorkerFailure("MODEL_INPUT_INVALID") from exc
        try:
            result = minimax_music.generate(
                model_root, output_path, payload["backend"], value, _event
            )
            minimax_music.validate_audio_artifact(output_path, result, value["durationSeconds"])
            return result
        except Exception as exc:
            raise WorkerFailure(_runtime_failure_code(exc, "MODEL_GENERATION_FAILED")) from exc

    if payload.get("taskContract") in {"text-to-video-v1", "image-text-to-video-v1"}:
        from oryn_engine import minimax_h3
        from oryn_engine.model_metadata import metadata_file
        from oryn_engine.video_audio_artifacts import validate_video_audio

        try:
            model_root, output_path, preview_path, input_path = _validated_paths(payload)
            index = metadata_file(model_root / minimax_h3.INDEX)
        except (OSError, TypeError, ValueError) as exc:
            raise WorkerFailure("MODEL_FORMAT_UNSUPPORTED") from exc
        if minimax_h3.supports(index):
            try:
                contract = payload["taskContract"]
                value = minimax_h3.validate_input(payload.get("input"), contract)
                if (
                    preview_path is not None
                    or (input_path is not None) != (contract == minimax_h3.IMAGE_CONTRACT)
                    or payload.get("referenceImagePaths")
                    or payload.get("composition")
                    or payload.get("backend") not in {"cpu", "mps", "cuda"}
                ):
                    raise ValueError("H3 input bindings are invalid")
            except (OSError, TypeError, ValueError) as exc:
                raise WorkerFailure("MODEL_INPUT_INVALID") from exc
            try:
                result = minimax_h3.generate(
                    model_root,
                    output_path,
                    payload["backend"],
                    value,
                    _event,
                    contract=contract,
                    input_path=input_path,
                )
                validate_video_audio(output_path, result, value)
                return result
            except Exception as exc:
                raise WorkerFailure(_runtime_failure_code(exc, "MODEL_GENERATION_FAILED")) from exc

    try:
        model_root, output_path, preview_path, input_path = _validated_paths(payload)
        contract = payload.get("taskContract")
        value = payload.get("input")
        if contract not in IMAGE_CONTRACTS | VIDEO_CONTRACTS:
            raise ValueError("Diffusers task contract is invalid")
        value = _validated_input(value, contract)
        reference_paths = _reference_paths(payload, value, contract)
    except (OSError, TypeError, ValueError) as exc:
        raise WorkerFailure("MODEL_INPUT_INVALID") from exc
    try:
        _validate_model(model_root)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise WorkerFailure("MODEL_FORMAT_UNSUPPORTED") from exc
    _event("loading-pipeline", 0.02)
    try:
        if payload.get("backend") not in {"cpu", "mps", "cuda"}:
            raise ValueError("The runtime backend must be explicitly bound")
        pipe, backend = _pipeline(
            model_root,
            contract,
            torch,
            backend_request=payload["backend"],
            **({"composition": payload["composition"]} if "composition" in payload else {}),
        )
    except Exception as exc:
        raise WorkerFailure(_runtime_failure_code(exc, "MODEL_RUNTIME_INCOMPATIBLE")) from exc
    _event("preparing-inputs", 0.1)
    try:
        if contract == "image-layers-v1":
            return _generate_layers(pipe, value, input_path, output_path.parent, torch, backend)
        if contract in IMAGE_CONTRACTS:
            _generate_image(
                pipe,
                contract,
                value,
                input_path,
                output_path,
                preview_path,
                torch,
                reference_paths=reference_paths,
            )
            media_type = "image/png"
            limit = MAX_IMAGE_BYTES
        else:
            _generate_video(pipe, contract, value, input_path, output_path, torch)
            media_type = "video/mp4"
            limit = MAX_VIDEO_BYTES
    except WorkerFailure:
        raise
    except Exception as exc:
        raise WorkerFailure(_runtime_failure_code(exc, "MODEL_GENERATION_FAILED")) from exc
    if not output_path.is_file() or not 1 <= output_path.stat().st_size <= limit:
        raise WorkerFailure("ARTIFACT_IO_ERROR")
    _event("writing-artifact", 0.98)
    return {
        "relativeOutput": output_path.name,
        "mediaType": media_type,
        "backend": backend,
        "width": value["width"],
        "height": value["height"],
        **({"frames": value["frames"], "fps": value["fps"]} if contract in VIDEO_CONTRACTS else {}),
    }


def _runtime_failure_code(exc: Exception, fallback: str) -> str:
    if isinstance(exc, MemoryError):
        return "MODEL_MEMORY_INSUFFICIENT"
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "RUNTIME_DEPENDENCY_UNAVAILABLE"
    message = str(exc).lower()
    if any(
        marker in message
        for marker in (
            "out of memory",
            "cannot allocate memory",
            "memory allocation failed",
            "mps backend out of memory",
        )
    ):
        return "MODEL_MEMORY_INSUFFICIENT"
    return fallback


def main() -> None:
    try:
        request = _request()
        if request["command"] == "probe":
            import bitsandbytes
            import diffusers
            import imageio
            import torch

            backend, _dtype = _backend(torch)
            result = {
                "diffusers": diffusers.__version__,
                "bitsandbytes": getattr(bitsandbytes, "__version__", version("bitsandbytes")),
                "imageio": imageio.__version__,
                "backend": backend,
                "networkPolicy": os.environ.get("ORYN_PACK_NETWORK_POLICY"),
            }
        else:
            result = _generate(request["payload"])
        _message({"type": "result", "ok": True, "result": result})
    except WorkerFailure as exc:
        _message({"type": "result", "ok": False, "error": {"code": exc.code}})
    except (ImportError, ModuleNotFoundError):
        _message(
            {
                "type": "result",
                "ok": False,
                "error": {"code": "RUNTIME_DEPENDENCY_UNAVAILABLE"},
            }
        )
    except Exception as exc:
        _message(
            {
                "type": "result",
                "ok": False,
                "error": {"code": _runtime_failure_code(exc, "PACK_EXECUTION_FAILED")},
            }
        )


if __name__ == "__main__":
    main()
