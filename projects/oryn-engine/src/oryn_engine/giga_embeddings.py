"""Reviewed bidirectional Qwen3 behavior; no repository modules or remote loading."""

from __future__ import annotations

from oryn_engine.model_metadata import validate_model_config

FAMILY = "qwen3_bidirec"
ARCHITECTURE = "Qwen3BidirectionalModel"
ALIASES = {
    "AutoConfig": "configuration_gigarembed.Qwen3BidirectionalConfig",
    "AutoModel": "modeling_gigarembed.Qwen3BidirectionalModel",
}
PROMPTS = {"query": "Instruct: Given a query, retrieve relevant passages\nQuery: ", "document": ""}


def native_config(config):
    if config.get("auto_map") not in (None, ALIASES):
        raise ValueError("GIGA_EMBEDDING_REMOTE_MAPPING")
    config = {k: v for k, v in config.items() if k != "auto_map"}
    validate_model_config(config)
    if (
        config.get("model_type") != FAMILY
        or config.get("architectures") != [ARCHITECTURE]
        or config.get("is_causal") is not False
        or config.get("use_cache") is not False
        or config.get("sliding_window") is not None
        or config.get("use_sliding_window", False)
        or config.get("rope_scaling") is not None
        or config.get("rope_parameters") is not None
        or config.get("attention_bias", False)
        or config.get("attention_dropout", 0.0) != 0.0
        or config.get("hidden_act", "silu") != "silu"
        or config.get("quantization_config")
        or config.get("quantization")
    ):
        raise ValueError("GIGA_EMBEDDING_CONFIGURATION_UNSUPPORTED")
    fields = {
        "hidden_size": 8192,
        "intermediate_size": 65536,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 256,
        "vocab_size": 262144,
        "max_position_embeddings": 32768,
    }
    for key, upper in fields.items():
        if type(config.get(key)) is not int or not 1 <= config[key] <= upper:
            raise ValueError("GIGA_EMBEDDING_DIMENSION_LIMIT")
    if (
        config["num_attention_heads"] % config["num_key_value_heads"]
        or config["head_dim"] % 2
        or config.get("layer_types") != ["full_attention"] * config["num_hidden_layers"]
        or type(config.get("rope_theta")) not in {int, float}
        or not 1 <= config["rope_theta"] <= 1e9
        or type(config.get("rms_norm_eps")) not in {int, float}
        or not 1e-8 <= config["rms_norm_eps"] <= 1e-3
    ):
        raise ValueError("GIGA_EMBEDDING_NATIVE_GEOMETRY")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if type(config.get(key)) is not int or not 0 <= config[key] < config["vocab_size"]:
            raise ValueError("GIGA_EMBEDDING_SPECIAL_TOKENS")
    allowed = fields.keys() | {
        "model_type",
        "architectures",
        "is_causal",
        "use_cache",
        "sliding_window",
        "use_sliding_window",
        "rope_scaling",
        "rope_parameters",
        "rope_theta",
        "rms_norm_eps",
        "attention_bias",
        "attention_dropout",
        "hidden_act",
        "layer_types",
        "max_window_layers",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "dtype",
        "torch_dtype",
        "tie_word_embeddings",
        "transformers_version",
        "initializer_range",
    }
    if not set(config) <= allowed:
        raise ValueError("GIGA_EMBEDDING_OPTIONS_UNSUPPORTED")
    return config


def normalized_config(config, task):
    if task == "feature-extraction" and config.get("model_type") == FAMILY:
        return native_config(config)
    return config


def native_model(config):
    from transformers import Qwen3Config, Qwen3Model

    values = native_config(config)
    options = Qwen3Config.from_dict(
        values | {"model_type": "qwen3", "architectures": ["Qwen3Model"]}
    )
    options._attn_implementation = "eager"
    model = Qwen3Model(options)
    for layer in model.layers:
        layer.self_attn.is_causal = False
    return model


def validate_weight_shapes(config, shapes):
    import torch

    with torch.device("meta"):
        model = native_model(config)
    prefix = "model." if shapes and all(k.startswith("model.") for k in shapes) else ""
    if {k.removeprefix(prefix): v for k, v in shapes.items()} != {
        k: tuple(v.shape) for k, v in model.state_dict().items()
    }:
        raise ValueError("GIGA_EMBEDDING_WEIGHTS_INCOMPLETE")
    return prefix


def load(root, config):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    from oryn_engine.model_metadata import preflight_safetensors

    prefix = validate_weight_shapes(config, preflight_safetensors(root))
    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    weights = {}
    for path in sorted(root.glob("*.safetensors")):
        state = {k.removeprefix(prefix): v for k, v in load_file(path).items()}
        if weights.keys() & state.keys() or any(
            not v.is_floating_point() or not bool(torch.isfinite(v).all()) for v in state.values()
        ):
            raise ValueError("GIGA_EMBEDDING_WEIGHTS_INVALID")
        weights.update({k: v.float() for k, v in state.items()})
    model.load_state_dict(weights, strict=True, assign=True)
    return model.eval()


def attention_mask(mask):
    import torch

    batch, length = mask.shape
    values = (1 - mask[:, None, None, :].float()) * torch.finfo(torch.float32).min
    # A native mask mapping bypasses causal-mask creation while preserving every
    # Qwen3 layer, rotary position, QK normalization and learned tensor.
    return {"full_attention": values.expand(batch, 1, length, length)}
