"""Oryn-owned preprocessing for the bundled Qwen Image 2.1 pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
MAX_PROMPT_TOKENS = 512


def stable_mps_downsampling(vae):
    """Evaluate native parameter-free average shortcuts on CPU for MPS inputs.

    MPS corrupts the temporally padded six-dimensional grouping, including
    BF16 NaNs. Preserve the original pad/reshape/mean and dtype on CPU instead.
    """
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import QwenImage21AvgDown3D

    for module in vae.modules():
        if type(module) is QwenImage21AvgDown3D:
            original = module.forward

            def forward(x, original=original):
                if x.device.type == "mps":
                    return original(x.to("cpu")).to(x.device)
                return original(x)

            module.forward = forward


def estimated_memory(files: list[Any], backend: str) -> int:
    # Live model weights plus one component's transfer/loading copy and a bounded
    # activation allowance. CPU fp32 doubles bf16 checkpoints. This is an estimate,
    # not a device qualification; ordinary available-memory admission still applies.
    components: dict[str, int] = {}
    for item in files:
        if item.name.endswith(".safetensors"):
            name = item.name.split("/", 1)[0]
            components[name] = components.get(name, 0) + item.size
    multiplier = 2 if backend == "cpu" else 1
    return (
        multiplier * (sum(components.values()) + max(components.values(), default=0)) + 4 * 1024**3
    )


def sample_sigmas(index: dict[str, Any]) -> list[float] | None:
    """Admit a bounded sampling grid as data, never as a pipeline component."""
    value = index.get("sample_sigmas")
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 50
        or any(
            type(item) not in {int, float} or not math.isfinite(item) or not 0 < item <= 1
            for item in value
        )
        or any(left <= right for left, right in zip(value, value[1:], strict=False))
    ):
        raise ValueError("Qwen Image 2.1 sampling sigmas are invalid")
    return [float(item) for item in value]


def generation_defaults(index: dict[str, Any] | None = None) -> dict[str, Any]:
    from oryn_engine.runtime_packs.generation_profiles import generation_profile

    result = generation_profile(
        task_contract="text-to-image-v1",
        trusted_metadata={
            "inferenceDefaults": {"width": 512, "height": 512, "steps": 40, "guidanceScale": 1.0}
        },
    )
    assert result is not None
    result["profileId"] = "runtime.qwen-image-2.1"
    for name in ("width", "height"):
        result["parameters"][name].update(maximum=1024, step=32)
    result["parameters"]["steps"]["maximum"] = 50
    result["parameters"]["guidanceScale"]["maximum"] = 10.0
    sigmas = sample_sigmas(index or {})
    if sigmas is not None:
        result["profileId"] = "runtime.qwen-image-2.1.sampling-grid"
        result["provenance"] = "adapter-derived"
        result["parameters"]["steps"].update(
            default=len(sigmas), minimum=len(sigmas), maximum=len(sigmas), editable=False
        )
    return result


def processor_options(metadata: dict[str, Any], vision: dict[str, Any]) -> dict[str, Any]:
    from oryn_engine.shared_tasks import qwen_image_options, validate_processor_config

    # Serialized fast-processor defaults are inert. Do not accept device selection,
    # cropping, padding, or other repository-controlled preprocessing behavior.
    metadata = dict(metadata)
    for key in (
        "crop_size",
        "device",
        "disable_grouping",
        "do_center_crop",
        "do_pad",
        "input_data_format",
        "max_pixels",
        "min_pixels",
        "pad_size",
        "return_tensors",
    ):
        if key in metadata:
            if metadata[key] is not None:
                raise ValueError("Qwen Image 2.1 preprocessing override is not admitted")
            del metadata[key]
    if "default_to_square" in metadata and metadata.pop("default_to_square") is not True:
        raise ValueError("Qwen Image 2.1 preprocessing default is invalid")
    if metadata.get("processor_class") not in {None, "Qwen3VLProcessor"}:
        raise ValueError("Qwen Image 2.1 processor identity is invalid")
    if vision.get("patch_size") != 16:
        raise ValueError("Qwen Image 2.1 requires the native 16-pixel vision patches")
    validate_processor_config(metadata, "qwen3_vl")
    return qwen_image_options(metadata, vision)


def sealed_processor(tokenizer: Any, options: dict[str, Any]) -> Any:
    import transformers as tf

    for token in ("<|im_start|>", "<|im_end|>", "<|image_pad|>"):
        if token not in tokenizer.get_vocab() or len(tokenizer.encode(token)) != 1:
            raise ValueError("Qwen Image 2.1 tokenizer framing is unsupported")
    tokenizer.chat_template = None
    processor = tf.Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=tf.Qwen2VLImageProcessor(**options),
        video_processor=tf.Qwen3VLVideoProcessor(
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
            size=options["size"],
            min_frames=1,
            max_frames=1,
            do_sample_frames=False,
        ),
    )

    def system_tokens(messages, *, tokenize=True, return_dict=False):
        # Native __init__ derives the leading system-token count before its first call.
        # Supply the exact native framing without compiling repository Jinja templates.
        expected = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        if messages != expected or tokenize is not True or return_dict is not False:
            raise ValueError("Unexpected Qwen Image 2.1 prompt contract")
        return [
            tokenizer.encode(
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n", add_special_tokens=False
            )
        ]

    processor.apply_chat_template = system_tokens
    return processor


def load_processor(model_root: Path) -> Any:
    import transformers as tf

    from oryn_engine.model_metadata import metadata_file, validate_model_config

    root = model_root / "processor"
    metadata_path = root / "preprocessor_config.json"
    if not metadata_path.exists():
        metadata_path = root / "processor_config.json"
    metadata = metadata_file(metadata_path)
    config = metadata_file(model_root / "text_encoder/config.json")
    validate_model_config(config)
    tokenizer_config = metadata_file(root / "tokenizer_config.json")
    validate_model_config(tokenizer_config)
    tokenizer = tf.Qwen2TokenizerFast.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False, model_max_length=2048
    )
    if tokenizer.encode("<|image_pad|>") != [config.get("image_token_id")]:
        raise ValueError("Qwen Image 2.1 tokenizer and text encoder disagree")
    return sealed_processor(tokenizer, processor_options(metadata, config.get("vision_config", {})))


def validate_prompts(pipe: Any, value: dict[str, Any]) -> None:
    for name in ("prompt", "negativePrompt"):
        if (
            len(pipe.processor.tokenizer.encode(value[name], add_special_tokens=False))
            > MAX_PROMPT_TOKENS
        ):
            raise ValueError("Qwen Image 2.1 prompt exceeds 512 tokens")


def reference_image(path: Path | None) -> Any:
    from PIL import Image, ImageOps

    if path is None or not path.is_file() or not 0 < path.stat().st_size <= 20 * 1024**2:
        raise ValueError("Reference image is invalid")
    with Image.open(path) as image:
        if image.width * image.height > 16_777_216 or not 0.5 <= image.width / image.height <= 2:
            raise ValueError("Reference image must have aspect ratio between 1:2 and 2:1")
        image.verify()
    with Image.open(path) as image:
        result = ImageOps.exif_transpose(image).convert("RGBA")
        result.thumbnail((1024, 1024))
        return result
