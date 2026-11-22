"""Closed PixAI tagging on the pinned Transformers SAM3 vision backbone.

Original fused projections are bound to native projections without changing any
learned values. The unused original CLS-position row remains a required parameter.
Repository implementation code and dispatch never enter the runtime.
"""

from __future__ import annotations

import math
import re

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

FAMILY = "oryn_pixai"
ARCHITECTURE = "PixAITagger"
TASK = "image-classification"
ALIASES = {
    "AutoConfig": "tagger_pipeline.ViTDetClsConfig",
    "AutoModel": "tagger_pipeline.ViTDetCls",
}
PIPELINE = {
    "tagger": {
        "impl": "tagger_pipeline.TaggerPipeline",
        "pt": ["AutoModel"],
        "tf": [],
        "type": "image",
    }
}
FIXED = {
    "act_layer": "GELU",
    "bias_patch_embed": False,
    "drop_path_rate": 0.0,
    "dropout": 0.0,
    "dtype": "float32",
    "in_chans": 3,
    "init_values": None,
    "ln_post": False,
    "ln_pre": True,
    "norm_layer": "LayerNorm",
    "pretrain_use_cls_token": True,
    "qkv_bias": True,
    "rel_pos_blocks": [],
    "rel_pos_zero_init": True,
    "retain_cls_token": False,
    "return_interm_layers": False,
    "rope_pt_size": None,
    "tile_abs_pos": True,
    "use_abs_pos": True,
    "use_act_checkpoint": False,
    "use_interp_rope": True,
    "use_rope": True,
    "tags_best_threshold": None,
}


def normalized_config(config, task):
    if task != TASK:
        return config
    if config.get("model_type") == "cls_vitdet" and config.get("architectures") == ["ViTDetCls"]:
        config = config | {"model_type": FAMILY, "architectures": [ARCHITECTURE]}
        for key, value in (("auto_map", ALIASES), ("custom_pipelines", PIPELINE)):
            if config.get(key) == value:
                config = {k: v for k, v in config.items() if k != key}
    if config.get("model_type") == FAMILY:
        validate_config(config)
    return config


def validate_config(config):
    validate_model_config(config)
    dimensions = {
        "embed_dim": (8, 1536),
        "depth": (1, 40),
        "img_size": (16, 1024),
        "patch_size": (2, 32),
        "pretrain_img_size": (4, 512),
        "num_heads": (1, 32),
        "num_classes": (1, 50000),
        "window_size": (1, 32),
    }
    required = (
        set(FIXED)
        | set(dimensions)
        | {
            "model_type",
            "architectures",
            "mlp_ratio",
            "global_att_blocks",
            "tags",
            "tags_split",
            "category_best_threshold",
        }
    )
    if (
        not required <= config.keys()
        or not config.keys() <= required | {"transformers_version", "_name_or_path"}
        or config["model_type"] != FAMILY
        or config["architectures"] != [ARCHITECTURE]
        or any(type(config[k]) is not type(v) or config[k] != v for k, v in FIXED.items())
    ):
        raise ValueError("PIXAI_CONFIGURATION")
    for key, (low, high) in dimensions.items():
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError("PIXAI_DIMENSIONS")
    width, heads, patch = config["embed_dim"], config["num_heads"], config["patch_size"]
    ratio = config["mlp_ratio"]
    if (
        width % heads
        or (width // heads) % 4
        or config["img_size"] % patch
        or config["pretrain_img_size"] % patch
        or (config["img_size"] // patch) ** 2 > 5184
        or type(ratio) not in {int, float}
        or not math.isfinite(ratio)
        or not 1 <= ratio <= 8
        or width * ratio != int(width * ratio)
        or width * ratio > 8192
    ):
        raise ValueError("PIXAI_DIMENSIONS")
    blocks = config["global_att_blocks"]
    if (
        not isinstance(blocks, list)
        or not blocks
        or any(type(x) is not int or not 0 <= x < config["depth"] for x in blocks)
        or sorted(set(blocks)) != blocks
        or blocks[-1] != config["depth"] - 1
    ):
        raise ValueError("PIXAI_GLOBAL_ATTENTION")
    tags = config["tags"]
    if (
        not isinstance(tags, list)
        or len(tags) != config["num_classes"]
        or any(not isinstance(x, str) or not 1 <= len(x) <= 256 for x in tags)
        or len(set(tags)) != len(tags)
    ):
        raise ValueError("PIXAI_LABELS")
    splits, thresholds = config["tags_split"], config["category_best_threshold"]
    if (
        not isinstance(splits, list)
        or not 1 <= len(splits) <= 32
        or any(
            not isinstance(x, list)
            or len(x) != 2
            or not isinstance(x[0], str)
            or not 1 <= len(x[0]) <= 64
            or type(x[1]) is not int
            or not 1 <= x[1] <= len(tags)
            for x in splits
        )
        or len({x[0] for x in splits}) != len(splits)
        or sum(x[1] for x in splits) != len(tags)
        or not isinstance(thresholds, dict)
        or set(thresholds) != {x[0] for x in splits}
        or any(
            type(x) not in {int, float} or not math.isfinite(x) or not 0 <= x <= 1
            for x in thresholds.values()
        )
    ):
        raise ValueError("PIXAI_CATEGORIES")


def validate_processor(metadata, config=None):
    values = dict(metadata)
    if values.get("auto_map") == {"AutoImageProcessor": "tagger_pipeline.RescalePadProcessor"}:
        del values["auto_map"]
    validate_model_config(values)
    fixed = {
        "data_format": "channels_first",
        "default_to_square": True,
        "image_processor_type": "RescalePadProcessor",
        "rescale_factor": 1 / 255,
    }
    nulls = {
        "crop_size",
        "device",
        "disable_grouping",
        "do_center_crop",
        "do_convert_rgb",
        "do_normalize",
        "do_pad",
        "do_rescale",
        "do_resize",
        "image_mean",
        "image_std",
        "input_data_format",
        "pad_size",
        "resample",
        "return_tensors",
    }
    if (
        set(values) != set(fixed) | nulls | {"size"}
        or any(type(values[k]) is not type(v) or values[k] != v for k, v in fixed.items())
        or any(values[k] is not None for k in nulls)
        or type(values["size"]) is not int
        or not 16 <= values["size"] <= 1024
        or config is not None
        and values["size"] != config["img_size"]
    ):
        raise ValueError("PIXAI_PREPROCESSOR")


def native_model(config):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from transformers import Sam3ViTConfig, Sam3ViTModel

    validate_config(config)
    width, heads = config["embed_dim"], config["num_heads"]

    class AttentionPool(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(width, eps=1e-5)
            self.q = nn.Parameter(torch.empty(width))
            self.pos_embed = nn.Parameter(
                torch.empty((config["img_size"] // config["patch_size"]) ** 2, width)
            )
            self.kv = nn.Linear(width, width * 2)
            self.proj = nn.Linear(width, width)

        def forward(self, tokens):
            batch, count, _ = tokens.shape
            tokens = self.norm(tokens) + self.pos_embed
            query = (
                self.q.reshape(1, 1, heads, width // heads)
                .expand(batch, -1, -1, -1)
                .transpose(1, 2)
            )
            key, value = self.kv(tokens).chunk(2, dim=-1)
            key = key.reshape(batch, count, heads, width // heads).transpose(1, 2)
            value = value.reshape(batch, count, heads, width // heads).transpose(1, 2)
            result = F.scaled_dot_product_attention(query, key, value)
            return self.proj(result.transpose(1, 2).reshape(batch, width))

    class Tagger(nn.Module):
        def __init__(self):
            super().__init__()
            options = Sam3ViTConfig(
                hidden_size=width,
                intermediate_size=int(width * config["mlp_ratio"]),
                num_hidden_layers=config["depth"],
                num_attention_heads=heads,
                num_channels=3,
                image_size=config["img_size"],
                patch_size=config["patch_size"],
                hidden_act="gelu",
                layer_norm_eps=1e-5,
                window_size=config["window_size"],
                global_attn_indexes=config["global_att_blocks"],
                pretrain_image_size=config["pretrain_img_size"],
                rope_parameters={"rope_type": "axial", "rope_theta": 10000.0},
            )
            options._attn_implementation = "sdpa"
            self.backbone = Sam3ViTModel(options)
            self.unused_class_position = nn.Parameter(torch.empty(1, 1, width))
            self.head_pool = AttentionPool()
            self.head = nn.Linear(width, config["num_classes"])

        def forward(self, pixels):
            return self.head(self.head_pool(self.backbone(pixels).last_hidden_state))

    return Tagger()


def original_shapes(model):
    shapes = {}
    for name, value in model.state_dict().items():
        shape = tuple(value.shape)
        if name == "unused_class_position":
            continue
        if name == "backbone.embeddings.position_embeddings":
            shapes["pos_embed"] = (1, shape[1] + 1, shape[2])
        elif name == "backbone.embeddings.patch_embeddings.projection.weight":
            shapes["patch_embed.proj.weight"] = shape
        elif name.startswith("backbone.layer_norm."):
            shapes[name.replace("backbone.layer_norm.", "ln_pre.")] = shape
        elif name.startswith("backbone.layers."):
            key = (
                name.replace("backbone.layers.", "blocks.")
                .replace(".layer_norm1.", ".norm1.")
                .replace(".layer_norm2.", ".norm2.")
                .replace(".attention.o_proj.", ".attn.proj.")
            )
            if re.search(r"\.attention\.[qkv]_proj\.", key):
                key = re.sub(r"\.attention\.[qkv]_proj\.", ".attn.qkv.", key)
                shape = (shape[0] * 3, *shape[1:])
            shapes[key] = shape
        else:
            shapes[name] = shape
    return shapes


def mapped_weights(state):
    result = {}
    for name, value in state.items():
        if name == "pos_embed":
            result["backbone.embeddings.position_embeddings"] = value[:, 1:]
            result["unused_class_position"] = value[:, :1]
        elif name == "patch_embed.proj.weight":
            result["backbone.embeddings.patch_embeddings.projection.weight"] = value
        elif name.startswith("ln_pre."):
            result[name.replace("ln_pre.", "backbone.layer_norm.")] = value
        elif name.startswith("blocks."):
            key = (
                name.replace("blocks.", "backbone.layers.")
                .replace(".norm1.", ".layer_norm1.")
                .replace(".norm2.", ".layer_norm2.")
                .replace(".attn.proj.", ".attention.o_proj.")
            )
            if ".attn.qkv." in key:
                for part, tensor in zip(("q", "k", "v"), value.chunk(3, dim=0), strict=True):
                    result[key.replace(".attn.qkv.", f".attention.{part}_proj.")] = tensor
            else:
                result[key] = value
        else:
            result[name] = value
    return result


def validate_weight_shapes(config, shapes):
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    if shapes != original_shapes(model):
        raise ValueError("PIXAI_WEIGHTS_INCOMPLETE")
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
            raise ValueError("PIXAI_DUPLICATE_WEIGHTS")
        if any(
            not v.is_floating_point() or not bool(torch.isfinite(v.float()).all())
            for v in values.values()
        ):
            raise ValueError("PIXAI_WEIGHT_VALUES")
        state.update({k: v.float() for k, v in values.items()})
    model.load_state_dict(mapped_weights(state), strict=True, assign=True)
    return model.eval()


def preprocess(image, size):
    from PIL import Image
    from torchvision.transforms import functional as F

    if image.mode != "RGB":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image.convert("RGBA"))
        image = canvas.convert("RGB")
    pixels = F.to_tensor(image)
    height, width = pixels.shape[-2:]
    if (height, width) != (size, size):
        scale = min(size / height, size / width)
        resized = (int(height * scale), int(width * scale))
        if min(resized) < 1:
            raise ValueError("PIXAI_IMAGE_ASPECT")
        dh, dw = size - resized[0], size - resized[1]
        pixels = F.resize(pixels, list(resized))
        pixels = F.pad(pixels, [dw // 2, dh // 2, dw - dw // 2, dh - dh // 2], 0)
    return F.normalize(pixels, [0.5] * 3, [0.5] * 3)[None]


def execute(model, config, image, value, backend):
    import torch

    logits = model(preprocess(image, config["img_size"]).to(backend))
    if tuple(logits.shape) != (1, config["num_classes"]) or not bool(torch.isfinite(logits).all()):
        raise ValueError("PIXAI_OUTPUT_INVALID")
    scores = logits.sigmoid()[0].float().cpu().tolist()
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[: value.get("topK", 5)]
    return {
        "predictions": [
            {"labelId": i, "label": config["tags"][i], "score": scores[i]} for i in order
        ]
    }
