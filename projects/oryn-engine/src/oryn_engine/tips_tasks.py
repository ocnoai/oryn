"""Original TIPS contrastive encoders bound to the pinned native Tipsv2 graph.

Only tensor names and fused QKV storage change. The published temperature comes
from explicit metadata; all learned image/text weights remain mandatory.
"""

from __future__ import annotations

import math
import re

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

FAMILY = "tipsv2"
ARCHITECTURE = "Tipsv2Model"
TASK = "zero-shot-image-classification"
METADATA_NAMES = ("tokenizer_config.json", "processor_config.json")
ALIASES = {
    "AutoConfig": "configuration_tips.TIPSv2Config",
    "AutoModel": "modeling_tips.TIPSv2Model",
}
TEXT_FIXED = {
    "attention_dropout": 0.0,
    "bos_token_id": None,
    "eos_token_id": None,
    "hidden_act": "relu",
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-5,
    "max_position_embeddings": 64,
    "model_type": "tipsv2_text_model",
    "pad_token_id": 0,
    "pooling_epsilon": 1e-8,
    "scale_sqrt_depth": True,
}
VISION_FIXED = {
    "apply_layernorm": True,
    "attention_probs_dropout_prob": 0.0,
    "drop_path_rate": 0.0,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-6,
    "layerscale_value": 1.0,
    "mlp_ratio": 4,
    "model_type": "tipsv2_vision_model",
    "num_channels": 3,
    "num_register_tokens": 1,
    "qkv_bias": True,
    "reshape_hidden_states": True,
    "use_swiglu_ffn": True,
}
TOKENIZER = {
    "backend": "tokenizers",
    "bos_token": None,
    "do_lower_case": True,
    "eos_token": None,
    "model_max_length": 64,
    "pad_token": "<pad>",
    "processor_class": "Tipsv2Processor",
    "token_type_ids_pattern": "all_zeros",
    "tokenizer_class": "Tipsv2Tokenizer",
    "unk_token": "<unk>",
}


def normalized_config(config, task):
    if task != TASK or config.get("model_type") != FAMILY:
        return config
    if config.get("architectures") == ["TIPSv2Model"]:
        config = config | {"architectures": [ARCHITECTURE]}
        if config.get("auto_map") == ALIASES:
            config = {k: v for k, v in config.items() if k != "auto_map"}
    validate_config(config)
    return config


def _same(actual, expected):
    return type(actual) is type(expected) and actual == expected


def validate_config(config):
    validate_model_config(config)
    required = {
        "model_type",
        "architectures",
        "vision_config",
        "text_config",
        "temperature_init_value",
    }
    if (
        not required <= config.keys()
        or not config.keys() <= required | {"transformers_version", "_name_or_path"}
        or config["model_type"] != FAMILY
        or config["architectures"] != [ARCHITECTURE]
    ):
        raise ValueError("TIPS_CONFIGURATION")
    temperature = config["temperature_init_value"]
    if (
        type(temperature) not in {int, float}
        or not math.isfinite(temperature)
        or not 1e-5 <= temperature <= 1
    ):
        raise ValueError("TIPS_TEMPERATURE")
    text, vision = config["text_config"], config["vision_config"]
    common = {
        "hidden_size": (8, 1536),
        "num_attention_heads": (1, 24),
        "num_hidden_layers": (1, 40),
    }
    for values, fixed, dimensions, extra in (
        (
            text,
            TEXT_FIXED,
            common | {"intermediate_size": (8, 6144), "vocab_size": (8, 32000)},
            set(),
        ),
        (
            vision,
            VISION_FIXED,
            common | {"image_size": (16, 448), "patch_size": (2, 32)},
            {"stage_names", "out_features", "out_indices"},
        ),
    ):
        if (
            not isinstance(values, dict)
            or set(values) != set(fixed) | set(dimensions) | extra
            or any(not _same(values[k], v) for k, v in fixed.items())
        ):
            raise ValueError("TIPS_GRAPH")
        for key, (low, high) in dimensions.items():
            if type(values[key]) is not int or not low <= values[key] <= high:
                raise ValueError("TIPS_DIMENSIONS")
        if values["hidden_size"] % values["num_attention_heads"]:
            raise ValueError("TIPS_DIMENSIONS")
    layers = vision["num_hidden_layers"]
    if (
        text["hidden_size"] != vision["hidden_size"]
        or text["hidden_size"] % 2
        or text["num_hidden_layers"] > 12
        or vision["image_size"] % vision["patch_size"]
        or (vision["image_size"] // vision["patch_size"]) ** 2 > 1024
        or vision["stage_names"] != ["stem"] + [f"stage{i}" for i in range(1, layers + 1)]
        or vision["out_features"] != [f"stage{layers}"]
        or vision["out_indices"] != [layers]
    ):
        raise ValueError("TIPS_DIMENSIONS")


def validate_metadata(name, values, config):
    validate_model_config(values)
    if name == "tokenizer_config.json":
        if set(values) != set(TOKENIZER) or any(
            not _same(values[k], v) for k, v in TOKENIZER.items()
        ):
            raise ValueError("TIPS_TOKENIZER_METADATA")
    elif name == "processor_config.json":
        expected = {
            "do_convert_rgb": True,
            "do_normalize": False,
            "do_rescale": True,
            "do_resize": True,
            "image_processor_type": "Tipsv2ImageProcessor",
            "resample": 2,
            "rescale_factor": 1 / 255,
            "size": {
                "height": config["vision_config"]["image_size"],
                "width": config["vision_config"]["image_size"],
            },
        }
        image = values.get("image_processor")
        if (
            set(values) != {"image_processor", "processor_class"}
            or values["processor_class"] != "Tipsv2Processor"
            or not isinstance(image, dict)
            or set(image) != set(expected)
            or any(not _same(image[k], v) for k, v in expected.items())
        ):
            raise ValueError("TIPS_PREPROCESSOR")
    else:
        raise ValueError("TIPS_METADATA_LAYOUT")


def native_model(config):
    from transformers import Tipsv2Config, Tipsv2Model

    validate_config(config)
    options = Tipsv2Config(
        vision_config=config["vision_config"],
        text_config=config["text_config"],
        temperature_init_value=config["temperature_init_value"],
    )
    options._attn_implementation = "sdpa"
    return Tipsv2Model(options)


def original_shapes(model):
    shapes = {}
    for name, value in model.state_dict().items():
        if name == "temperature":
            continue
        shape = tuple(value.shape)
        if name.startswith("text_model."):
            key = (
                name.replace(
                    "text_model.embeddings.token_embedding.", "text_encoder.token_embedding."
                )
                .replace("text_model.final_layer_norm.", "text_encoder.ln_final.")
                .replace("text_model.encoder.layers.", "text_encoder.transformer.resblocks.")
                .replace(".layer_norm1.", ".ln_1.")
                .replace(".layer_norm2.", ".ln_2.")
                .replace(".mlp.fc1.", ".mlp.c_fc.")
                .replace(".mlp.fc2.", ".mlp.c_proj.")
                .replace(".self_attn.out_proj.", ".attn.out_proj.")
            )
            if ".self_attn." in key:
                key = re.sub(r"\.self_attn\.[qkv]_proj\.", ".attn.in_proj_", key)
                shape = (shape[0] * 3, *shape[1:])
        else:
            key = (
                name.replace(
                    "vision_model.embeddings.position_embeddings", "vision_encoder.pos_embed"
                )
                .replace(
                    "vision_model.embeddings.patch_embeddings.projection.",
                    "vision_encoder.patch_embed.proj.",
                )
                .replace("vision_model.embeddings.", "vision_encoder.")
                .replace("vision_model.layernorm.", "vision_encoder.norm.")
                .replace("vision_model.encoder.layer.", "vision_encoder.blocks.")
                .replace(".layer_scale1.lambda1", ".ls1.gamma")
                .replace(".layer_scale2.lambda1", ".ls2.gamma")
                .replace(".mlp.down_proj.", ".mlp.w3.")
                .replace(".attention.o_proj.", ".attn.proj.")
            )
            # Transformers >= 5.20 splits the original fused projections.
            if re.search(r"\.attention\.[qkv]_proj\.", key):
                key = re.sub(r"\.attention\.[qkv]_proj\.", ".attn.qkv.", key)
                shape = (shape[0] * 3, *shape[1:])
            elif re.search(r"\.mlp\.(gate|up)_proj\.", key):
                key = re.sub(r"\.mlp\.(gate|up)_proj\.", ".mlp.w12.", key)
                shape = (shape[0] * 2, *shape[1:])
        if key in shapes and shapes[key] != shape:
            raise ValueError("TIPS_NATIVE_GRAPH_CHANGED")
        shapes[key] = shape
    return shapes


def mapped_weights(state):
    result = {}
    for name, value in state.items():
        if name.startswith("text_encoder."):
            key = (
                name.replace("text_encoder.", "text_model.")
                .replace("text_model.token_embedding.", "text_model.embeddings.token_embedding.")
                .replace("text_model.ln_final.", "text_model.final_layer_norm.")
                .replace(".transformer.resblocks.", ".encoder.layers.")
                .replace(".ln_1.", ".layer_norm1.")
                .replace(".ln_2.", ".layer_norm2.")
                .replace(".mlp.c_fc.", ".mlp.fc1.")
                .replace(".mlp.c_proj.", ".mlp.fc2.")
                .replace(".attn.out_proj.", ".self_attn.out_proj.")
            )
            if ".attn.in_proj_" in key:
                for part, tensor in zip(("q", "k", "v"), value.chunk(3, dim=0), strict=True):
                    result[key.replace(".attn.in_proj_", f".self_attn.{part}_proj.")] = tensor
            else:
                result[key] = value
        elif name.startswith("vision_encoder."):
            key = name.replace("vision_encoder.", "vision_model.")
            direct = {
                "vision_model." + k: "vision_model.embeddings." + v
                for k, v in (
                    ("cls_token", "cls_token"),
                    ("mask_token", "mask_token"),
                    ("register_tokens", "register_tokens"),
                    ("pos_embed", "position_embeddings"),
                )
            }
            key = (
                direct.get(key, key)
                .replace(
                    "vision_model.patch_embed.proj.",
                    "vision_model.embeddings.patch_embeddings.projection.",
                )
                .replace("vision_model.norm.", "vision_model.layernorm.")
                .replace(".blocks.", ".encoder.layer.")
                .replace(".ls1.gamma", ".layer_scale1.lambda1")
                .replace(".ls2.gamma", ".layer_scale2.lambda1")
                .replace(".mlp.w3.", ".mlp.down_proj.")
                .replace(".attn.proj.", ".attention.o_proj.")
            )
            if ".attn.qkv." in key:
                for part, tensor in zip(("q", "k", "v"), value.chunk(3, dim=0), strict=True):
                    result[key.replace(".attn.qkv.", f".attention.{part}_proj.")] = tensor
            elif ".mlp.w12." in key:
                # SwiGLU: the activated half comes first, as in the library's own converter.
                for part, tensor in zip(("gate", "up"), value.chunk(2, dim=0), strict=True):
                    result[key.replace(".mlp.w12.", f".mlp.{part}_proj.")] = tensor
            else:
                result[key] = value
        else:
            raise ValueError("TIPS_WEIGHT_NAME")
    return result


def validate_weight_shapes(config, shapes):
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    if shapes != original_shapes(model):
        raise ValueError("TIPS_WEIGHTS_INCOMPLETE")
    return model


def load_tokenizer(root, config):
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2

    path = root / "tokenizer.model"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 2 * 1024**2:
        raise ValueError("TIPS_TOKENIZER_FILE")
    raw = path.read_bytes()
    proto = sentencepiece_model_pb2.ModelProto()
    try:
        proto.ParseFromString(raw)
        tokenizer = spm.SentencePieceProcessor(model_proto=raw)
    except Exception as exc:
        raise ValueError("TIPS_TOKENIZER_INVALID") from exc
    if (
        proto.trainer_spec.model_type != 2
        or len(proto.pieces) != config["text_config"]["vocab_size"]
        or (tokenizer.pad_id(), tokenizer.eos_id(), tokenizer.unk_id(), tokenizer.bos_id())
        != (0, 1, 2, 3)
        or [tokenizer.id_to_piece(i) for i in range(4)] != ["<pad>", "</s>", "<unk>", "<s>"]
        or any(not math.isfinite(p.score) or len(p.piece) > 1024 for p in proto.pieces)
        or len({p.piece for p in proto.pieces}) != len(proto.pieces)
    ):
        raise ValueError("TIPS_TOKENIZER_VOCABULARY")
    return tokenizer


def load_components(root, config, backend):
    import torch
    from safetensors.torch import load_file

    for name in METADATA_NAMES:
        validate_metadata(name, metadata_file(root / name), config)
    tokenizer = load_tokenizer(root, config)
    model = validate_weight_shapes(config, preflight_safetensors(root))
    state = {}
    for path in sorted(root.glob("*.safetensors")):
        values = load_file(path)
        if state.keys() & values.keys():
            raise ValueError("TIPS_DUPLICATE_WEIGHTS")
        if any(
            not v.is_floating_point() or not bool(torch.isfinite(v.float()).all())
            for v in values.values()
        ):
            raise ValueError("TIPS_WEIGHT_VALUES")
        state.update({k: v.float() for k, v in values.items()})
    native = mapped_weights(state)
    native["temperature"] = torch.tensor(config["temperature_init_value"], dtype=torch.float32)
    model.load_state_dict(native, strict=True, assign=True)
    return model.eval().to(backend), tokenizer, config


def preprocess(image, size):
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    return to_tensor(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR))[None]


def tokenize(tokenizer, labels):
    import torch

    tokens = torch.zeros((len(labels), 64), dtype=torch.long)
    for i, label in enumerate(labels):
        ids = tokenizer.encode(label.lower(), add_bos=False, add_eos=False)[:64]
        if not ids or not any(ids):
            raise ValueError("TIPS_EMPTY_LABEL")
        tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return tokens, (tokens != 0).long()


def execute(model, tokenizer, config, image, value, backend):
    import torch

    labels = value["labels"]
    ids, mask = tokenize(tokenizer, labels)
    result = model(
        pixel_values=preprocess(image, config["vision_config"]["image_size"]).to(backend),
        input_ids=ids.to(backend),
        attention_mask=mask.to(backend),
    )
    logits = result.logits_per_image
    if tuple(logits.shape) != (1, len(labels)) or not bool(torch.isfinite(logits).all()):
        raise ValueError("TIPS_OUTPUT_INVALID")
    scores = logits.softmax(dim=-1)[0].float().cpu().tolist()
    order = sorted(range(len(labels)), key=lambda i: (-scores[i], i))
    return {"predictions": [{"label": labels[i], "score": scores[i]} for i in order]}
