"""Complete pinned Yug graph with bounded, direct univariate forecasts."""

from __future__ import annotations

from oryn_engine.model_metadata import preflight_safetensors, validate_model_config

FAMILY = "yug"
ARCHITECTURE = "Yug_Model"


def options(config):
    validate_model_config(config)
    fixed = {
        "model_type": FAMILY,
        "architectures": [ARCHITECTURE],
        "library_name": "yug",
        "freq_num": 9,
        "kernel_size": 3,
    }
    integers = {
        "patch_len": (1, 64),
        "output_dim": (1, 512),
        "hidden_dim": (16, 2048),
        "d_model": (16, 2048),
        "num_attn_heads": (1, 32),
        "num_decoder_layers": (1, 24),
    }
    scalars = {"dropout": (0, 0.5), "eps": (1e-8, 1e-3), "expansion": (1, 4)}
    if set(config) != fixed.keys() | integers.keys() | scalars.keys() | {"quantiles"} or any(
        type(config.get(k)) is not type(v) or config[k] != v for k, v in fixed.items()
    ):
        raise ValueError("YUG_CONFIGURATION")
    for key, (low, high) in integers.items():
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError("YUG_DIMENSIONS")
    for key, (low, high) in scalars.items():
        if type(config[key]) not in {int, float} or not low <= config[key] <= high:
            raise ValueError("YUG_OPTIONS")
    levels = config["quantiles"]
    if (
        not isinstance(levels, list)
        or not 3 <= len(levels) <= 21
        or any(type(q) not in {int, float} or not 0 < q < 1 for q in levels)
        or levels != sorted(set(levels))
        or 0.5 not in levels
        or config["d_model"] % (2 * config["num_attn_heads"])
        or config["d_model"] // config["num_attn_heads"] < 4
        or config["output_dim"] % config["patch_len"]
        or config["num_decoder_layers"] * config["d_model"] ** 2 > 24 * 960**2
    ):
        raise ValueError("YUG_GRAPH")
    return {
        k: v for k, v in config.items() if k not in {"model_type", "architectures", "library_name"}
    }


def settings(config):
    native = options(config)
    return {
        "context_length": 2048,
        "horizon_length": native["output_dim"],
        "patch_length": native["patch_len"],
        "quantiles": native["quantiles"],
    }


def native_model(config):
    from oryn_engine.native_yug.model import Yug_Model

    return Yug_Model(options(config))


def validate_weight_shapes(config, shapes):
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    if shapes != {k: tuple(v.shape) for k, v in model.state_dict().items()}:
        raise ValueError("YUG_WEIGHTS_INCOMPLETE")
    return model


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
            raise ValueError("YUG_WEIGHT_VALUES")
        state.update({k: v.float() for k, v in values.items()})
    # RoPE bases are persistent in the original graph. A changed basis would
    # implement a different architecture despite identical parameter shapes.
    for name, buffer in model.named_buffers():
        if not torch.equal(state[name], buffer):
            raise ValueError("YUG_ROTARY_BINDING")
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


def frequency(milliseconds):
    """Canonical whole fixed periods; absent/subsecond periods use native U/1.

    Numeric durations cannot identify calendar months, quarters or years.
    Choose the largest exactly dividing fixed unit, preserving its multiplier.
    """
    if milliseconds is not None:
        if type(milliseconds) is not int or not 1 <= milliseconds <= 31536000000:
            raise ValueError("YUG_FREQUENCY")
        for unit, bucket in ((604800000, 4), (86400000, 3), (3600000, 2), (60000, 1), (1000, 0)):
            if milliseconds % unit == 0:
                return bucket, milliseconds // unit
    return 8, 1


def predictions(model, context, horizon, backend, milliseconds=None):
    import torch

    config = model.config
    patch = config["patch_len"]
    # Align the last observed value with the last patch. Native masks exclude
    # left padding from both cumulative normalization and attention.
    padding = (-len(context)) % patch
    values = torch.tensor([[0.0] * padding + context], device=backend, dtype=torch.float32)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[:, :padding] = False
    values, mask = values.reshape(1, -1, patch), mask.reshape(1, -1, patch)
    bucket, scale = frequency(milliseconds)
    output, _, _ = model(
        values,
        attn_mask_target=mask,
        freq_id=torch.tensor([bucket], device=backend),
        scale=torch.tensor([scale], device=backend),
        num_input_patches=values.shape[1],
    )
    # The head layout is output patch, quantile, patch position. Select the
    # final observed anchor before converting to the shared horizon/quantile axes.
    count = len(config["quantiles"])
    quantiles = (
        output[:, -1]
        .reshape(1, config["output_dim"] // patch, count, patch)
        .permute(0, 1, 3, 2)
        .reshape(1, config["output_dim"], count)
    )
    points = quantiles[:, :, config["quantiles"].index(0.5)]
    return points, torch.cat((points.unsqueeze(-1), quantiles), dim=-1)
