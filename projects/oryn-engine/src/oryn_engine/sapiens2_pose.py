"""Complete native Sapiens2 pose graph and reviewed Sociopticon joint semantics."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from oryn_engine.model_metadata import validate_model_config


@lru_cache(maxsize=1)
def joint_metadata():
    # Build-owned factual labels, extracted without executing upstream config code.
    return json.loads((Path(__file__).parent / "data/sapiens2-joints.json").read_text())


def labels(config):
    expected = {str(i): name for i, name in enumerate(joint_metadata()["names"])}
    declared = config.get("id2label")
    placeholders = {str(i): f"LABEL_{i}" for i in range(308)}
    if declared != expected and declared != placeholders:
        raise ValueError("SAPIENS2_JOINT_SEMANTICS_UNSUPPORTED")
    pairs = config.get("flip_pairs")
    if (
        not isinstance(pairs, list)
        or len(pairs) != 146
        or any(
            not isinstance(p, list) or len(p) != 2 or any(type(i) is not int for i in p)
            for p in pairs
        )
        or sorted(sorted(p) for p in pairs) != joint_metadata()["flipPairs"]
    ):
        raise ValueError("SAPIENS2_JOINT_SEMANTICS_UNSUPPORTED")
    return expected


def image_size(config):
    size = config.get("image_size")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(v) is not int or not 64 <= v <= 1024 or v % 16 for v in size)
        or size[0] * size[1] > 1024 * 768
    ):
        raise ValueError("SAPIENS2_IMAGE_GEOMETRY_UNSUPPORTED")
    return {"height": size[0], "width": size[1]}


def validate_config(config):
    validate_model_config(config)
    labels(config)
    image_size(config)
    if (
        config.get("model_type") != "sapiens2"
        or config.get("architectures") != ["Sapiens2ForPoseEstimation"]
        or config.get("patch_size") != 16
        or config.get("num_channels", 3) != 3
        or config.get("quantization_config")
        or config.get("quantization")
    ):
        raise ValueError("SAPIENS2_CONFIGURATION_UNSUPPORTED")
    for name, low, high in (
        ("hidden_size", 16, 4096),
        ("intermediate_size", 16, 32768),
        ("num_hidden_layers", 1, 80),
        ("num_attention_heads", 1, 64),
        ("num_register_tokens", 0, 16),
    ):
        if type(config.get(name)) is not int or not low <= config[name] <= high:
            raise ValueError("SAPIENS2_CONFIGURATION_LIMIT")
    if config["hidden_size"] % (4 * config["num_attention_heads"]):
        raise ValueError("SAPIENS2_HEAD_GEOMETRY_UNSUPPORTED")
    heads = config.get("num_key_value_heads_per_layer")
    if (
        not isinstance(heads, list)
        or len(heads) != config["num_hidden_layers"]
        or any(type(h) is not int or h < 1 or config["num_attention_heads"] % h for h in heads)
    ):
        raise ValueError("SAPIENS2_ATTENTION_GEOMETRY_UNSUPPORTED")
    head = config.get("head_config")
    if (
        not isinstance(head, dict)
        or head.get("model_type") != "sapiens2_head"
        or head.get("use_pixel_shuffle") is not None
        and head["use_pixel_shuffle"] is not False
        or any(
            head.get(k) is not None
            for k in (
                "scale_conv_out_channels",
                "scale_conv_kernel_sizes",
                "scale_final_hidden_sizes",
                "scale_final_input_size",
            )
        )
    ):
        raise ValueError("SAPIENS2_POSE_HEAD_UNSUPPORTED")
    for prefix, length, kernel in (("upsample", 2, 4), ("conv", 3, 1)):
        channels = head.get(prefix + "_out_channels")
        if (
            not isinstance(channels, list)
            or len(channels) != length
            or any(type(c) is not int or not 4 <= c <= 4096 for c in channels)
            or head.get(prefix + "_kernel_sizes") != [kernel] * length
        ):
            raise ValueError("SAPIENS2_POSE_HEAD_UNSUPPORTED")


def processor_options(metadata, config):
    validate_model_config(metadata)
    expected = {
        "size": image_size(config),
        "do_resize": True,
        "do_center_crop": False,
        "do_rescale": True,
        "do_normalize": True,
        "do_reduce_labels": False,
        "do_pad": False,
        "resample": 2,
        "rescale_factor": 1 / 255,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
    }
    if metadata.get("image_processor_type") != "Sapiens2ImageProcessor":
        raise ValueError("SAPIENS2_PROCESSOR_UNSUPPORTED")
    for key, value in expected.items():
        if key not in metadata or type(metadata[key]) is not type(value) or metadata[key] != value:
            raise ValueError("SAPIENS2_PROCESSOR_UNSUPPORTED")
    # Inactive serialized generic fields never influence the owned constructor.
    return expected


def load_components(root, config, common):
    from transformers import Sapiens2ForPoseEstimation, Sapiens2ImageProcessor

    from oryn_engine.model_metadata import metadata_file

    processor = Sapiens2ImageProcessor(
        **processor_options(metadata_file(root / "preprocessor_config.json"), config)
    )
    return processor, Sapiens2ForPoseEstimation


def predictions(model, processor, config, image, boxes, backend):
    import torch

    size = image_size(config)
    results = []
    # One person crop at a time bounds activations independently of box count.
    for box in boxes:
        inputs = processor(images=image, boxes=[[box]], return_tensors="pt")
        pixels = inputs["pixel_values"]
        if tuple(pixels.shape) != (1, 3, size["height"], size["width"]):
            raise ValueError("SAPIENS2_PERSON_CROP_INVALID")
        output = model(pixel_values=pixels.to(backend))
        if (
            tuple(output.heatmaps.shape) != (1, 308, size["height"] // 4, size["width"] // 4)
            or not torch.isfinite(output.heatmaps).all()
        ):
            raise ValueError("SAPIENS2_HEATMAP_INVALID")
        # Geometry decoding is deterministic float32; CPU avoids requiring
        # device-specific support for the processor's matrix inverse operations.
        from transformers.models.sapiens2.modeling_sapiens2 import Sapiens2PoseEstimatorOutput

        output = Sapiens2PoseEstimatorOutput(heatmaps=output.heatmaps.float().cpu())
        decoded = processor.post_process_pose_estimation(output, boxes=[[box]], threshold=None)[0]
        if len(decoded) != 1 or decoded[0]["keypoints"].shape != (308, 2):
            raise ValueError("SAPIENS2_KEYPOINT_COUNT_INVALID")
        results.extend(decoded)
    return results
