"""Closed whole-body RTMW keypoints for a single person image/crop."""

from __future__ import annotations

import math
from types import SimpleNamespace

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

FAMILY = "rtmw"
TASK = "keypoint-detection"
ARCHITECTURE = "RTMWModel"
ALIASES = {"AutoConfig": "configuration_rtmw.RTMWConfig", "AutoModel": "modeling_rtmw.RTMWModel"}
FIXED = {
    "backbone_arch": "P5",
    "backbone_channel_attention": True,
    "backbone_deepen_factor": 1.0,
    "backbone_expand_ratio": 0.5,
    "decoder_normalize": False,
    "decoder_sigma": [4.9, 5.66],
    "decoder_use_dark": False,
    "gau_act_fn": "SiLU",
    "gau_drop_path": 0.0,
    "gau_dropout_rate": 0.0,
    "gau_expansion_factor": 2,
    "gau_pos_enc": False,
    "gau_use_rel_bias": False,
    "head_final_layer_kernel_size": 7,
    "neck_expand_ratio": 0.5,
    "neck_num_csp_blocks": 2,
    "neck_out_channels": None,
    "num_keypoints": 133,
    "score_max": 1.0,
    "score_min": 0.0,
    "simcc_split_ratio": 2.0,
}
PROCESSOR_KEYS = [
    "images",
    "do_resize",
    "size",
    "keep_aspect_ratio",
    "ensure_multiple_of",
    "resample",
    "do_rescale",
    "rescale_factor",
    "do_normalize",
    "image_mean",
    "image_std",
    "do_pad",
    "size_divisor",
    "return_tensors",
    "data_format",
    "input_data_format",
]
MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]


def normalized_config(config, task):
    if task != TASK or config.get("model_type") != FAMILY:
        return config
    if config.get("auto_map") == ALIASES:
        config = {k: v for k, v in config.items() if k != "auto_map"}
    if "architectures" not in config:
        config = config | {"architectures": [ARCHITECTURE]}
    validate_config(config)
    return config


def validate_config(config):
    validate_model_config(config)
    extra = {
        "model_type",
        "architectures",
        "backbone_widen_factor",
        "gau_hidden_dims",
        "gau_s",
        "head_in_channels",
        "head_in_featuremap_size",
        "input_size",
        "neck_in_channels",
    }
    required = set(FIXED) | extra
    if (
        not required <= config.keys()
        or not config.keys() <= required | {"transformers_version", "_name_or_path"}
        or config["model_type"] != FAMILY
        or config["architectures"] != [ARCHITECTURE]
        or any(type(config[k]) is not type(v) or config[k] != v for k, v in FIXED.items())
    ):
        raise ValueError("RTMW_CONFIGURATION")
    width = config["backbone_widen_factor"]
    if (
        type(width) not in {int, float}
        or not math.isfinite(width)
        or not 1 / 16 <= width <= 1.25
        or width * 16 != int(width * 16)
    ):
        raise ValueError("RTMW_DIMENSIONS")
    for key, low, high in (("gau_hidden_dims", 32, 256), ("gau_s", 8, 128)):
        if type(config[key]) is not int or not low <= config[key] <= high or config[key] % 2:
            raise ValueError("RTMW_DIMENSIONS")
    for key in ("head_in_featuremap_size", "neck_in_channels"):
        if not isinstance(config[key], list) or any(type(n) is not int for n in config[key]):
            raise ValueError("RTMW_DIMENSIONS")
    size = config["input_size"]
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(n) is not int or not 64 <= n <= 384 or n % 32 for n in size)
        or config["head_in_featuremap_size"] != [n // 32 for n in size]
        or type(config["head_in_channels"]) is not int
        or config["head_in_channels"] != int(1024 * width)
        or config["neck_in_channels"] != [int(n * width) for n in (256, 512, 1024)]
    ):
        raise ValueError("RTMW_DIMENSIONS")


def validate_processor(values, config):
    validate_model_config(values)
    width, height = config["input_size"]
    expected = {
        "_valid_processor_keys": PROCESSOR_KEYS,
        "do_normalize": True,
        "do_rescale": False,
        "do_resize": True,
        "image_mean": MEAN,
        "image_std": STD,
        "image_processor_type": "DPTImageProcessor",
        "size": {"height": height, "width": width},
    }
    if set(values) != set(expected) or any(
        type(values[k]) is not type(v) or values[k] != v for k, v in expected.items()
    ):
        raise ValueError("RTMW_PREPROCESSOR")


def native_model(config):
    from oryn_engine.native_rtmw.model import RTMWModel

    validate_config(config)
    return RTMWModel(SimpleNamespace(**config))


def validate_weight_shapes(config, shapes):
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    if shapes != {k: tuple(v.shape) for k, v in model.state_dict().items()}:
        raise ValueError("RTMW_WEIGHTS_INCOMPLETE")
    return model


def load(root, config):
    import torch
    from safetensors.torch import load_file

    validate_processor(metadata_file(root / "preprocessor_config.json"), config)
    model = validate_weight_shapes(config, preflight_safetensors(root))
    state = {}
    for path in sorted(root.glob("*.safetensors")):
        values = load_file(path)
        if state.keys() & values.keys():
            raise ValueError("RTMW_DUPLICATE_WEIGHTS")
        state.update(values)
    for name, value in state.items():
        if name.endswith(".num_batches_tracked"):
            if value.dtype != torch.int64 or bool((value < 0).any()):
                raise ValueError("RTMW_BATCH_COUNTER")
        else:
            if not value.is_floating_point() or not bool(torch.isfinite(value.float()).all()):
                raise ValueError("RTMW_WEIGHT_VALUES")
            if name.endswith(".running_var") and bool((value < 0).any()):
                raise ValueError("RTMW_BATCH_VARIANCE")
            state[name] = value.float()
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


def preprocess(image, size):
    import numpy as np
    import torch
    from PIL import Image

    # Published DPT PIL preprocessing: bicubic square/rectangle resize of RGB
    # byte pixels, then ImageNet byte-scale centering, with no rescale or padding.
    pixels = np.asarray(
        image.convert("RGB").resize(tuple(size), Image.Resampling.BICUBIC), dtype=np.float32
    )
    pixels = (pixels - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy())[None]


def execute(model, config, image, value, backend):
    import torch

    pred_x, pred_y = model(preprocess(image, config["input_size"]).to(backend))
    for prediction, size in zip((pred_x, pred_y), config["input_size"], strict=True):
        if tuple(prediction.shape) != (1, config["num_keypoints"], size * 2) or not bool(
            torch.isfinite(prediction).all()
        ):
            raise ValueError("RTMW_OUTPUT_INVALID")
    keypoints, scores = model.head.decoder.decode(pred_x, pred_y)
    keypoints, scores = keypoints[0].float().cpu(), scores[0].float().cpu()
    scores = scores.clamp(max=1)
    width, height = image.size
    keypoints *= torch.tensor([width / config["input_size"][0], height / config["input_size"][1]])
    points = []
    for index in sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i)):
        score = float(scores[index])
        # The native decoder marks nonpositive-confidence coordinates invalid.
        if score <= 0 or score < value["threshold"]:
            continue
        x, y = keypoints[index].tolist()
        if not all(math.isfinite(v) for v in (score, x, y)) or not (
            0 <= x <= width and 0 <= y <= height
        ):
            raise ValueError("RTMW_OUTPUT_INVALID")
        points.append({"x": x, "y": y, "score": score})
        if len(points) >= value["maxKeypoints"]:
            break
    return {"keypoints": points, "width": width, "height": height}
