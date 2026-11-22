"""Bounded SegFormer numeric graph and native image preprocessing."""

import math
import re

from oryn_engine.model_metadata import validate_model_config


def validate_config(config):
    if config.get("num_encoder_blocks", 4) != 4 or config.get("num_channels", 3) != 3:
        raise ValueError("HDF5_SEGFORMER_CONFIGURATION")
    if (
        config.get("hidden_act", "gelu") != "gelu"
        or config.get("reshape_last_stage", True) is not True
    ):
        raise ValueError("HDF5_SEGFORMER_CONFIGURATION")
    for name, low, high in (
        ("depths", 1, 48),
        ("hidden_sizes", 8, 2048),
        ("num_attention_heads", 1, 64),
        ("mlp_ratios", 1, 8),
        ("patch_sizes", 1, 15),
        ("strides", 1, 8),
        ("sr_ratios", 1, 16),
    ):
        v = config.get(name)
        if (
            not isinstance(v, list)
            or len(v) != 4
            or any(type(x) is not int or not low <= x <= high for x in v)
        ):
            raise ValueError("HDF5_SEGFORMER_CONFIGURATION")
    if any(
        w % h for w, h in zip(config["hidden_sizes"], config["num_attention_heads"], strict=True)
    ) or any(x % 2 != 1 for x in config["patch_sizes"]):
        raise ValueError("HDF5_SEGFORMER_CONFIGURATION")
    width = config.get("decoder_hidden_size")
    if type(width) is not int or not 8 <= width <= 4096:
        raise ValueError("HDF5_SEGFORMER_CONFIGURATION")
    labels = config.get("id2label")
    if (
        not isinstance(labels, dict)
        or not 2 <= len(labels) <= 512
        or set(labels) != {str(i) for i in range(len(labels))}
        or any(not isinstance(v, str) or not 1 <= len(v) <= 1024 for v in labels.values())
    ):
        raise ValueError("HDF5_SEGFORMER_LABELS")


def native_name(source, shape):
    match = re.fullmatch(
        r"(segformer|decode_head)/tf_segformer_for_semantic_segmentation(?:_\d+)?/\1/(.+):0", source
    )
    if not match:
        raise ValueError("HDF5_WEIGHT_NAME_UNSUPPORTED")
    prefix, leaf = match.groups()
    parts = leaf.split("/")
    kernel = parts[-1] == "kernel"
    parts[-1] = {
        "kernel": "weight",
        "gamma": "weight",
        "beta": "bias",
        "moving_mean": "running_mean",
        "moving_variance": "running_var",
    }.get(parts[-1], parts[-1])
    name = prefix + "." + ".".join(parts)
    for pattern, replacement in (
        (r"encoder\.patch_embeddings\.(\d+)\.", r"stages.\1.patch_embeddings."),
        (r"encoder\.block\.(\d+)\.", r"stages.\1.blocks."),
        (r"encoder\.layer_norm\.(\d+)", r"stages.\1.layer_norm"),
    ):
        name = re.sub(pattern, replacement, name)
    for old, new in (
        ("attention.self.query", "attention.q_proj"),
        ("attention.self.key", "attention.k_proj"),
        ("attention.self.value", "attention.v_proj"),
        ("attention.self.sr", "attention.sequence_reduction.sequence_reduction"),
        ("attention.self.layer_norm", "attention.sequence_reduction.layer_norm"),
        ("attention.output.dense", "attention.o_proj"),
        ("mlp.dense1", "mlp.fc1"),
        ("mlp.dense2", "mlp.fc2"),
        ("layer_norm_1", "layernorm_before"),
        ("layer_norm_2", "layernorm_after"),
        ("decode_head.linear_c", "decode_head.linear_projections"),
    ):
        name = name.replace(old, new)
    permutation = None
    if kernel:
        if len(shape) == 4:
            permutation = (3, 2, 0, 1)
        elif len(shape) == 2:
            permutation = (1, 0)
        else:
            raise ValueError("HDF5_WEIGHT_SHAPE_UNSUPPORTED")
    return name, permutation


def processor_options(metadata, config):
    validate_model_config(metadata)
    if metadata.get("image_processor_type", metadata.get("feature_extractor_type")) not in {
        "SegformerImageProcessor",
        "SegformerFeatureExtractor",
    }:
        raise ValueError("HDF5_SEGFORMER_PROCESSOR")
    size = metadata.get("size")
    if type(size) is int:
        size = dict.fromkeys(("height", "width"), size)
    if (
        not isinstance(size, dict)
        or set(size) != {"height", "width"}
        or any(type(v) is not int or not 32 <= v <= 1024 for v in size.values())
    ):
        raise ValueError("HDF5_SEGFORMER_PROCESSOR")
    for side in size.values():
        for stride, ratio in zip(config["strides"], config["sr_ratios"], strict=True):
            side = math.ceil(side / stride)
            if side < ratio:
                raise ValueError("HDF5_SEGFORMER_PROCESSOR")
    if metadata.get("resample", 2) != 2 or metadata.get("do_center_crop", False) is not False:
        raise ValueError("HDF5_SEGFORMER_PROCESSOR")
    options = {"size": size, "resample": 2}
    for name in ("do_resize", "do_rescale", "do_normalize"):
        if metadata.get(name, True) is not True:
            raise ValueError("HDF5_SEGFORMER_PROCESSOR")
        options[name] = True
    for name in ("image_mean", "image_std"):
        v = metadata.get(name)
        if (
            not isinstance(v, list)
            or len(v) != 3
            or any(type(x) not in {int, float} or not math.isfinite(x) or abs(x) > 100 for x in v)
            or name == "image_std"
            and any(x <= 0 for x in v)
        ):
            raise ValueError("HDF5_SEGFORMER_PROCESSOR")
        options[name] = v
    scale = metadata.get("rescale_factor", 1 / 255)
    reduce = metadata.get("do_reduce_labels", metadata.get("reduce_labels", False))
    if type(reduce) is not bool or metadata.get("reduce_labels", reduce) != reduce:
        raise ValueError("HDF5_SEGFORMER_PROCESSOR")
    if type(scale) not in {int, float} or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("HDF5_SEGFORMER_PROCESSOR")
    options.update(rescale_factor=scale, do_reduce_labels=reduce)
    return options
