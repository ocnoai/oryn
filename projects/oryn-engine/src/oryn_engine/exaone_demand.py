"""Bounded EXAONE Demand native forecasting with complete numeric-only weights."""

from __future__ import annotations

import math

from oryn_engine.model_metadata import preflight_safetensors, validate_model_config

FAMILY = "exaone-demand"
ARCHITECTURE = "EXAONEDemand"


def options(config):
    validate_model_config(config)
    fixed = {
        "model_type": FAMILY,
        "architectures": [ARCHITECTURE],
        "ssmax": True,
        "num_reg_tokens": 1,
        "adapter_enabled": True,
        "adapter_targets": ["to_q", "to_k", "to_v", "to_out", "w12", "w3"],
    }
    integer_bounds = {
        "dim_model": (16, 2048),
        "dim_embed": (16, 4096),
        "num_heads": (1, 32),
        "num_layers": (1, 48),
        "context_length": (16, 8192),
        "input_patch_size": (1, 64),
        "input_patch_stride": (1, 64),
        "output_patch_size": (1, 64),
        "max_output_patches": (1, 16),
        "adapter_alpha": (1, 128),
    }
    scalar_bounds = {
        "multiplier": (1.0, 4.0),
        "dropout": (0.0, 0.5),
        "adapter_dropout": (0.0, 0.5),
        "adapter_shared_weight": (0.0, 1.0),
        "adapter_temperature": (0.01, 10.0),
    }
    if (
        set(config)
        - (
            fixed.keys()
            | integer_bounds.keys()
            | scalar_bounds.keys()
            | {"adapter_ranks", "quantiles", "torch_dtype"}
        )
        or any(type(config.get(k)) is not type(v) or config[k] != v for k, v in fixed.items())
        or config.get("torch_dtype", "float32") != "float32"
    ):
        raise ValueError("EXAONE_DEMAND_CONFIGURATION")
    for key, (low, high) in integer_bounds.items():
        if type(config.get(key)) is not int or not low <= config[key] <= high:
            raise ValueError("EXAONE_DEMAND_DIMENSIONS")
    for key, (low, high) in scalar_bounds.items():
        if type(config.get(key)) not in {int, float} or not low <= config[key] <= high:
            raise ValueError("EXAONE_DEMAND_OPTIONS")
    ranks = config.get("adapter_ranks")
    levels = config.get("quantiles")
    if (
        not isinstance(ranks, list)
        or len(ranks) != 5
        or any(type(n) is not int or not 1 <= n <= 32 for n in ranks)
        or not isinstance(levels, list)
        or not 3 <= len(levels) <= 21
        or any(type(q) not in {int, float} or not 0 < q < 1 for q in levels)
        or levels != sorted(set(levels))
        or 0.5 not in levels
        or config["input_patch_size"] != config["input_patch_stride"]
        or config["input_patch_size"] != config["output_patch_size"]
        or config["dim_model"] % (2 * config["num_heads"])
        or config["dim_model"] // config["num_heads"] < 4
        or config["context_length"] % config["input_patch_size"]
        or config["output_patch_size"] * config["max_output_patches"] > 512
        or config["num_layers"] * config["dim_model"] ** 2 > 48 * 1024**2
    ):
        raise ValueError("EXAONE_DEMAND_GRAPH")
    return {
        k: v for k, v in config.items() if k not in {"model_type", "architectures", "torch_dtype"}
    }


def settings(config):
    native = options(config)
    return {
        "context_length": native["context_length"],
        "horizon_length": native["output_patch_size"] * native["max_output_patches"],
        "patch_length": native["input_patch_size"],
        "quantiles": native["quantiles"],
    }


def native_model(config):
    from oryn_engine.native_exaone_demand.config import EXAONEDemandConfig
    from oryn_engine.native_exaone_demand.model import EXAONEDemand

    return EXAONEDemand(EXAONEDemandConfig(**options(config)))


def validate_weight_shapes(config, shapes):
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    if shapes != expected:
        raise ValueError("EXAONE_DEMAND_WEIGHTS_INCOMPLETE")
    return model


def tied_parameters(model):
    groups = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter), []).append(name)
    return [names for names in groups.values() if len(names) > 1]


def load(root, config):
    import torch
    from safetensors.torch import load_file

    model = validate_weight_shapes(config, preflight_safetensors(root))
    state = {}
    for path in sorted(root.glob("*.safetensors")):
        values = load_file(path)
        if state.keys() & values.keys() or any(
            not v.is_floating_point() or not bool(torch.isfinite(v).all()) for v in values.values()
        ):
            raise ValueError("EXAONE_DEMAND_WEIGHT_VALUES")
        state.update({k: v.float() for k, v in values.items()})
    if not torch.equal(state["quantiles"], torch.tensor(config["quantiles"], dtype=torch.float32)):
        raise ValueError("EXAONE_DEMAND_QUANTILE_BINDING")
    if any(
        not torch.equal(state[names[0]], state[name])
        for names in tied_parameters(model)
        for name in names[1:]
    ):
        raise ValueError("EXAONE_DEMAND_TIED_WEIGHTS")
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


def predictions(model, context, horizon, backend):
    import torch

    patch = model.config.output_patch_size
    quantiles = model(
        context=torch.tensor([context], device=backend, dtype=torch.float32),
        num_output_patches=math.ceil(horizon / patch),
    ).transpose(1, 2)
    points = quantiles[:, :, model.config.quantiles.index(0.5)]
    return points, torch.cat((points.unsqueeze(-1), quantiles), dim=-1)
