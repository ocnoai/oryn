"""Closed OpenCLIP ViT/text graphs in the shared PyTorch task environment.

The native default BPE belongs to the pinned runtime. Repository Python,
pretrained identifiers, tokenizer overrides and checkpoint conversion are unused.
"""

from __future__ import annotations

import math

from oryn_engine.model_metadata import preflight_safetensors, validate_model_config

FAMILY = "oryn_open_clip"
ARCHITECTURE = "OpenCLIPModel"
TASK = "zero-shot-image-classification"
CONFIG = "open_clip_config.json"
WEIGHTS = "open_clip_model.safetensors"


def normalized_config(config, task):
    if task == TASK and ("model_cfg" in config or config.get("model_type") == FAMILY):
        if "model_type" not in config and "architectures" not in config:
            config = config | {"model_type": FAMILY, "architectures": [ARCHITECTURE]}
        validate_config(config)
    return config


def validate_config(config):
    validate_model_config(config)
    if (
        set(config) != {"model_cfg", "preprocess_cfg", "model_type", "architectures"}
        or config["model_type"] != FAMILY
        or config["architectures"] != [ARCHITECTURE]
    ):
        raise ValueError("OPENCLIP_CONFIGURATION")
    model = config["model_cfg"]
    if not isinstance(model, dict) or set(model) != {"embed_dim", "vision_cfg", "text_cfg"}:
        raise ValueError("OPENCLIP_GRAPH")
    vision, text = model["vision_cfg"], model["text_cfg"]
    specs = (
        (model, {"embed_dim": (8, 2048)}, False),
        (
            vision,
            {
                "image_size": (16, 512),
                "layers": (1, 48),
                "width": (8, 2048),
                "head_width": (4, 128),
                "patch_size": (2, 32),
            },
            True,
        ),
        (
            text,
            {
                "context_length": (77, 77),
                "vocab_size": (49408, 49408),
                "width": (8, 2048),
                "heads": (1, 32),
                "layers": (1, 32),
            },
            True,
        ),
    )
    for values, fields, exact in specs:
        if not isinstance(values, dict) or exact and set(values) != set(fields):
            raise ValueError("OPENCLIP_GRAPH")
        for key, (low, high) in fields.items():
            if type(values.get(key)) is not int or not low <= values[key] <= high:
                raise ValueError("OPENCLIP_DIMENSIONS")
    if (
        vision["width"] % vision["head_width"]
        or vision["image_size"] % vision["patch_size"]
        or (vision["image_size"] // vision["patch_size"]) ** 2 > 1024
        or text["width"] % text["heads"]
    ):
        raise ValueError("OPENCLIP_DIMENSIONS")
    preprocess = config["preprocess_cfg"]
    if (
        not isinstance(preprocess, dict)
        or set(preprocess) != {"mean", "std", "interpolation", "resize_mode"}
        or preprocess["interpolation"] != "bicubic"
        or preprocess["resize_mode"] != "shortest"
    ):
        raise ValueError("OPENCLIP_PREPROCESS")
    for key in ("mean", "std"):
        values = preprocess[key]
        if (
            not isinstance(values, list)
            or len(values) != 3
            or any(
                type(v) not in {int, float}
                or not math.isfinite(v)
                or not (0 <= v <= 1 if key == "mean" else 0 < v <= 1)
                for v in values
            )
        ):
            raise ValueError("OPENCLIP_PREPROCESS")


def native_model(config):
    from open_clip.model import CLIP

    validate_config(config)
    return CLIP(**config["model_cfg"], quick_gelu=False)


def validate_weight_shapes(config, shapes):
    from accelerate import init_empty_weights

    # The causal attention mask is deterministic and non-persistent.
    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    if shapes != {k: tuple(v.shape) for k, v in model.state_dict().items()}:
        raise ValueError("OPENCLIP_WEIGHTS_INCOMPLETE")
    return model


def load_components(root, config, backend):
    import torch
    from open_clip.tokenizer import SimpleTokenizer
    from open_clip.transform import PreprocessCfg, image_transform_v2
    from safetensors.torch import load_file

    model = validate_weight_shapes(config, preflight_safetensors(root))
    if {p.name for p in root.glob("*.safetensors")} != {WEIGHTS}:
        raise ValueError("OPENCLIP_WEIGHT_LAYOUT")
    state = load_file(root / WEIGHTS)
    if any(
        not v.is_floating_point() or not bool(torch.isfinite(v.float()).all())
        for v in state.values()
    ):
        raise ValueError("OPENCLIP_WEIGHT_VALUES")
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True, assign=True)
    if not bool(torch.isfinite(model.logit_scale.exp()).all()):
        raise ValueError("OPENCLIP_LOGIT_SCALE")
    transform = image_transform_v2(
        PreprocessCfg(
            size=config["model_cfg"]["vision_cfg"]["image_size"], **config["preprocess_cfg"]
        ),
        is_train=False,
    )
    return model.eval().to(backend), (transform, SimpleTokenizer(context_length=77)), config


def execute(model, processor, image, value, backend):
    import torch

    transform, tokenizer = processor
    labels = value["labels"]
    # Follow native OpenCLIP truncation, EOS pooling and normalized similarities.
    images = transform(image).unsqueeze(0).to(backend)
    tokens = tokenizer(labels).to(backend)
    scores, _ = model.get_logits(images, tokens)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("OPENCLIP_OUTPUT_INVALID")
    scores = scores.softmax(dim=-1)[0].float().cpu().tolist()
    order = sorted(range(len(labels)), key=lambda i: (-scores[i], i))
    return {"predictions": [{"label": labels[i], "score": scores[i]} for i in order]}
