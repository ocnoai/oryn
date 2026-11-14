"""Bind the original safe TimesFM 2.5 tensor layout to the pinned native model.

The original layout uses fused QKV and different tensor names. This adapter only
renames/slices tensors in memory; the checkpoint identity and files stay unchanged.
Layout reference: Transformers' convert_timesfm2_5_original_to_hf.py and the pinned
TimesFm2_5ModelForPrediction implementation. Missing tensors are never initialized.
"""

from __future__ import annotations

import math


def applies(config):
    return config.get("model_type") == "timesfm" and "quantile_horizon_length" in config


def native_config(config):
    fields = {
        "context_length": 16384,
        "head_dim": 256,
        "hidden_size": 4096,
        "horizon_length": 4096,
        "intermediate_size": 16384,
        "num_attention_heads": 64,
        "num_hidden_layers": 64,
        "patch_length": 512,
        "quantile_horizon_length": 4096,
    }
    if (
        not applies(config)
        or config.get("architectures") != ["TimesFmModelForPrediction"]
        or not set(config)
        <= fields.keys()
        | {"architectures", "model_type", "quantiles", "rms_norm_eps", "torch_compile"}
        or config.get("torch_compile", False) is not False
        or config.get("quantiles") != [i / 10 for i in range(1, 10)]
    ):
        raise ValueError("Original TimesFM 2.5 declaration is unsupported")
    for key, bound in fields.items():
        value = config.get(key)
        if type(value) is not int or not 1 <= value <= bound:
            raise ValueError("Original TimesFM 2.5 dimensions exceed bounds")
    hidden = config["hidden_size"]
    if (
        hidden != config["head_dim"] * config["num_attention_heads"]
        or config["context_length"] % config["patch_length"]
        or config["quantile_horizon_length"] < config["horizon_length"]
        or 24 * hidden * config["intermediate_size"] * config["num_hidden_layers"] > 1_000_000_000
    ):
        raise ValueError("Original TimesFM 2.5 geometry is unsupported")
    epsilon = config.get("rms_norm_eps", 1e-6)
    if (
        type(epsilon) not in (int, float)
        or not math.isfinite(epsilon)
        or not 1e-8 <= epsilon <= 1e-3
    ):
        raise ValueError("Original TimesFM 2.5 normalization is invalid")
    return {
        **{key: config[key] for key in fields if key != "quantile_horizon_length"},
        "output_quantile_len": config["quantile_horizon_length"],
        "num_key_value_heads": config["num_attention_heads"],
        "max_position_embeddings": config["context_length"],
        "rms_norm_eps": epsilon,
        "quantiles": config["quantiles"],
        "decode_index": 5,
        "use_bias": False,
        "activation": "swish",
    }


def bindings(config):
    """Closed layout mapping: destination -> (original name, optional QKV slice)."""
    result = {}
    for destination, source, suffixes in (
        ("model.input_ff_layer", "tokenizer", ("weight", "bias")),
        ("output_projection_point", "output_projection_point", ("weight",)),
        ("output_projection_quantiles", "output_projection_quantiles", ("weight",)),
    ):
        for layer in ("input_layer", "output_layer", "residual_layer"):
            original = "hidden_layer" if layer == "input_layer" else layer
            for suffix in suffixes:
                result[f"{destination}.{layer}.{suffix}"] = (f"{source}.{original}.{suffix}", None)
    for index in range(config["num_hidden_layers"]):
        source, dest = f"stacked_xf.{index}", f"model.layers.{index}"
        for part, name in enumerate(("q_proj", "k_proj", "v_proj")):
            result[f"{dest}.self_attn.{name}.weight"] = (f"{source}.attn.qkv_proj.weight", part)
        for original, target in {
            "attn.out.weight": "self_attn.o_proj.weight",
            "attn.query_ln.scale": "self_attn.q_norm.weight",
            "attn.key_ln.scale": "self_attn.k_norm.weight",
            "attn.per_dim_scale.per_dim_scale": "self_attn.scaling",
            "ff0.weight": "mlp.fc1.weight",
            "ff1.weight": "mlp.fc2.weight",
            "pre_attn_ln.scale": "input_layernorm.weight",
            "post_attn_ln.scale": "post_attention_layernorm.weight",
            "pre_ff_ln.scale": "pre_feedforward_layernorm.weight",
            "post_ff_ln.scale": "post_feedforward_layernorm.weight",
        }.items():
            result[f"{dest}.{target}"] = (f"{source}.{original}", None)
    return result


def load(root, config):
    from safetensors.torch import load_file
    from transformers import TimesFm2_5Config, TimesFm2_5ModelForPrediction

    from oryn_engine.model_metadata import preflight_safetensors

    options = native_config(config)
    shapes = preflight_safetensors(root)
    mapping = bindings(config)
    if set(shapes) != {name for name, _ in mapping.values()}:
        raise ValueError("Original TimesFM 2.5 tensor graph is incomplete or ambiguous")
    model = TimesFm2_5ModelForPrediction(TimesFm2_5Config(**options))
    expected = model.state_dict()
    if expected.keys() != mapping.keys():
        raise ValueError("Pinned TimesFM 2.5 layout changed")
    for name, (source, part) in mapping.items():
        shape = tuple(expected[name].shape)
        if part is not None:
            shape = (3 * shape[0], *shape[1:])
        if shapes[source] != shape:
            raise ValueError("Original TimesFM 2.5 tensor shape disagrees with its configuration")
    tensors = {}
    for path in sorted(root.glob("*.safetensors")):
        state = load_file(path, device="cpu")
        if tensors.keys() & state.keys():
            raise ValueError("Original TimesFM 2.5 graph contains duplicate tensors")
        tensors.update(state)
    state = {
        dest: tensors[source] if part is None else tensors[source].chunk(3, dim=0)[part]
        for dest, (source, part) in mapping.items()
    }
    model.load_state_dict(state, strict=True)
    return model.eval()
