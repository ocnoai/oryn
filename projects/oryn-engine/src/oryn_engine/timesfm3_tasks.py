"""Closed TimesFM 3 config and complete native graph over local numeric tensors."""

from __future__ import annotations

ARCHITECTURE = "TimesFM3Torch"
FAMILY = "timesfm3"


def native_config(config):
    """Validate without importing Torch; unknown options never reach native code."""
    fixed = {
        "input_transform": "identity",
        "linear_detrending_threshold": 0.5,
        "quantiles": [i / 10 for i in range(1, 10)],
        "use_frozen_running_stats": False,
        "use_iterative_cpm_revin": True,
        "use_linear_detrending": True,
        "use_stitching": True,
        "use_variate_attention": True,
        "value_clip": 1e20,
    }
    options = {k: v for k, v in config.items() if k not in {"model_type", "architectures"}}
    if (
        config.get("model_type", FAMILY) != FAMILY
        or config.get("architectures", [ARCHITECTURE]) != [ARCHITECTURE]
        or set(options)
        != fixed.keys()
        | {"input_patch_len", "output_patch_len", "residual_block_config", "transformer_config"}
        or any(type(options[k]) is not type(v) or options[k] != v for k, v in fixed.items())
    ):
        raise ValueError("TimesFM 3 configuration is unsupported")
    residual = options["residual_block_config"]
    stack = options["transformer_config"]
    residual_fixed = {
        "activation": "relu",
        "dropout": 0.0,
        "identity_skip": False,
        "prenorm": "none",
        "use_bias": False,
    }
    transformer_fixed = {
        "attention_norm": "rms",
        "causal_attention": True,
        "debug_no_masking": False,
        "deterministic": True,
        "feedforward_norm": "rms",
        "ff_activation": "relu",
        "max_variates": 32,
        "paired_token_skip_second": False,
        "qk_norm": "rms",
        "training": True,
        "use_bias": False,
        "use_memory_efficient_attention": True,
        "use_rope_seq": True,
        "use_rope_var": False,
        "use_sdpa": True,
        "v_norm": "none",
    }
    if (
        not isinstance(stack, dict)
        or set(stack) != {"num_layers", "transformer", "use_remat"}
        or stack["use_remat"] is not True
    ):
        raise ValueError("TimesFM 3 transformer stack is unsupported")
    transformer = stack["transformer"]
    for part, constants, dimensions in (
        (residual, residual_fixed, {"hidden_dims": 4096, "output_dims": 4096}),
        (
            transformer,
            transformer_fixed,
            {"model_dims": 4096, "hidden_dims": 16384, "num_heads": 64},
        ),
    ):
        if (
            not isinstance(part, dict)
            or set(part) != constants.keys() | dimensions.keys()
            or any(type(part[k]) is not type(v) or part[k] != v for k, v in constants.items())
            or any(type(part[k]) is not int or not 1 <= part[k] <= n for k, n in dimensions.items())
        ):
            raise ValueError("TimesFM 3 native layer options are unsupported")
    patch, output, layers = (
        options["input_patch_len"],
        options["output_patch_len"],
        stack["num_layers"],
    )
    if any(
        type(v) is not int or not 1 <= v <= bound
        for v, bound in ((patch, 128), (output, 512), (layers, 64))
    ):
        raise ValueError("TimesFM 3 patch/layer dimensions exceed bounds")
    width, heads = transformer["model_dims"], transformer["num_heads"]
    if (
        output <= patch
        or output % patch
        or width != residual["output_dims"]
        or width % heads
        or (width // heads) % 2
        or 16 * width * max(width, transformer["hidden_dims"]) * layers > 1_000_000_000
    ):
        raise ValueError("TimesFM 3 native geometry exceeds bounds")
    return options


def normalized_config(config, task):
    if task != "time-series-forecasting":
        return config
    if config.get("model_type") == FAMILY or (
        "model_type" not in config
        and "architectures" not in config
        and {
            "input_patch_len",
            "output_patch_len",
            "residual_block_config",
            "transformer_config",
            "value_clip",
        }
        <= config.keys()
    ):
        native_config(config)
        return config | {"model_type": FAMILY, "architectures": [ARCHITECTURE]}
    return config


def settings(config):
    native_config(config)
    return {
        "context_length": 2048,
        "horizon_length": 512,
        "patch_length": config["input_patch_len"],
        "quantiles": config["quantiles"],
    }


def native_model(config):
    from timesfm3.torch.model import TimesFM3Torch

    return TimesFM3Torch(**native_config(config))


def validate_weight_shapes(config, shapes):
    import torch

    with torch.device("meta"):
        model = native_model(config)
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    if shapes != expected:
        raise ValueError("TimesFM 3 weights are incomplete or mismatched")


def load(root, config):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    from oryn_engine.model_metadata import preflight_safetensors

    validate_weight_shapes(config, preflight_safetensors(root))
    # Preserve deterministic rotary buffers while avoiding original-size random weights.
    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    tensors = {}
    for path in sorted(root.glob("*.safetensors")):
        state = load_file(path, device="cpu")
        if tensors.keys() & state.keys() or any(
            not value.is_floating_point() or not bool(torch.isfinite(value).all())
            for value in state.values()
        ):
            raise ValueError("TimesFM 3 tensors are duplicate or invalid")
        tensors.update({name: value.float() for name, value in state.items()})
    model.load_state_dict(tensors, strict=True, assign=True)
    return model.eval()


def predictions(model, context, horizon, backend):
    import math

    import torch

    # Same univariate, no-covariate path as the pinned upstream forecaster's
    # defaults: native RevIN/detrending/stitching, sorted quantiles, median point.
    # No caller-controlled normalization, truncation or extra learned graph.
    padded_horizon = math.ceil(horizon / model.output_patch_len) * model.output_patch_len
    full = model.decode(
        target=torch.tensor([[context]], dtype=torch.float32, device=backend),
        horizon=padded_horizon,
    )
    if full.shape != (1, 1, padded_horizon, 9):
        raise ValueError("TimesFM 3 native forecast shape is invalid")
    quantiles = full[:, 0, :horizon].sort(dim=-1).values
    points = quantiles[:, :, 4]
    return points, torch.cat((points.unsqueeze(-1), quantiles), dim=-1)
