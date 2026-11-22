"""Sealed native forecasting behavior over safe, local tensors."""

from __future__ import annotations

import math

TASK_CONTRACTS = {"time-series-forecasting": "time-series-forecasting-v1"}
TASK_FAMILIES = {
    "time-series-forecasting": {
        "timesfm",
        "timesfm2_5",
        "timesfm3",
        "t5",
        "exaone-demand",
        "yug",
        "aurora",
    }
}
CHRONOS_ARCHITECTURES = {"ChronosBoltModelForForecasting", "Chronos2Model"}


def settings(config):
    """Read native bounds without changing the original model configuration."""
    if config.get("model_type") == "aurora":
        from oryn_engine.aurora_tasks import settings as native_settings

        return native_settings(config)
    if config.get("model_type") == "yug":
        from oryn_engine.yug_tasks import settings as native_settings

        return native_settings(config)
    if config.get("model_type") == "exaone-demand":
        from oryn_engine.exaone_demand import settings as native_settings

        return native_settings(config)
    if config.get("model_type") == "timesfm3":
        from oryn_engine.timesfm3_tasks import settings as native_settings

        return native_settings(config)
    if config.get("model_type") != "t5":
        return config
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or len(architectures) != 1
        or architectures[0] not in CHRONOS_ARCHITECTURES
    ):
        raise ValueError("Native Chronos architecture is unavailable")
    native = config.get("chronos_config")
    if not isinstance(native, dict):
        raise ValueError("Native Chronos configuration is missing")
    bolt = architectures == ["ChronosBoltModelForForecasting"]
    allowed = {
        "context_length",
        "input_patch_size",
        "input_patch_stride",
        "quantiles",
        "use_reg_token",
    }
    allowed |= (
        {"prediction_length"}
        if bolt
        else {
            "output_patch_size",
            "max_output_patches",
            "use_arcsinh",
            "time_encoding_scale",
        }
    )
    if not set(native) <= allowed:
        raise ValueError("Native Chronos options are unsupported")
    patch = native.get("input_patch_size")
    if type(patch) is not int or not 1 <= patch <= 512 or native.get("input_patch_stride") != patch:
        raise ValueError("Native Chronos patch geometry is unsupported")
    for key in ("use_reg_token", "use_arcsinh"):
        if key in native and type(native[key]) is not bool:
            raise ValueError("Native Chronos flags are invalid")
    for key, maximum in {
        "d_model": 4096,
        "d_ff": 16384,
        "d_kv": 256,
        "num_heads": 64,
        "num_layers": 64,
    }.items():
        value = config.get(key)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("Native Chronos dimensions exceed bounds")
    if config["d_model"] != config["d_kv"] * config["num_heads"]:
        raise ValueError("Native Chronos head dimensions disagree")
    if 24 * config["d_model"] * config["d_ff"] * config["num_layers"] > 1_000_000_000:
        raise ValueError("Native Chronos construction exceeds the parameter budget")
    if config.get("feed_forward_proj", "relu") not in {
        "relu",
        "gelu",
        "gated-gelu",
        "gated-relu",
    } or (not bolt and config.get("is_gated_act", False)):
        raise ValueError("Native Chronos activation is unsupported")
    if bolt:
        decoder_layers = config.get("num_decoder_layers", config["num_layers"])
        if type(decoder_layers) is not int or not 1 <= decoder_layers <= 64:
            raise ValueError("Native Chronos decoder exceeds bounds")
        horizon = native.get("prediction_length")
    else:
        if native.get("use_reg_token") is not True:
            raise ValueError("Native Chronos-2 requires its regression token")
        patches = native.get("max_output_patches", 1)
        scale = native.get("time_encoding_scale", native.get("context_length"))
        if (
            native.get("output_patch_size") != patch
            or type(patches) is not int
            or not 1 <= patches <= 128
            or type(scale) is not int
            or not 1 <= scale <= 16384
        ):
            raise ValueError("Native Chronos output geometry is unsupported")
        horizon = patch * patches
    return {
        "context_length": native.get("context_length"),
        "horizon_length": horizon,
        "patch_length": patch,
        "quantiles": native.get("quantiles"),
    }


def validate_config(config):
    from oryn_engine import timesfm_original

    if timesfm_original.applies(config):
        timesfm_original.native_config(config)
    original = config
    config = settings(config)
    for field, upper in (
        ("context_length", 16384),
        ("horizon_length", 4096),
        ("patch_length", 512),
    ):
        value = config.get(field)
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError("Forecast configuration exceeds native bounds")
    levels = config.get("quantiles")
    if (
        not isinstance(levels, list)
        or not 1 <= len(levels) <= 99
        or any(
            isinstance(q, bool)
            or not isinstance(q, (int, float))
            or not math.isfinite(q)
            or not 0 < q < 1
            for q in levels
        )
        or levels != sorted(set(levels))
    ):
        raise ValueError("Native forecast quantiles are invalid")
    if original.get("model_type") == "t5" and 0.5 not in levels:
        raise ValueError("Chronos point predictions require the native median head")


def load_chronos(root, config):
    """Construct only pinned native classes; never use repository Auto mappings.

    Chronos' generic pretrained loader intentionally ignores missing patch heads
    for fine tuning. Inference must instead require every trained parameter.
    """
    import torch
    from chronos.chronos2.config import Chronos2CoreConfig
    from chronos.chronos2.model import Chronos2Model
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    from safetensors.torch import load_file
    from transformers import T5Config

    from oryn_engine.model_metadata import preflight_safetensors

    validate_config(config)
    shapes = preflight_safetensors(root)
    if config["architectures"] == ["Chronos2Model"]:
        model = Chronos2Model(Chronos2CoreConfig.from_dict(config))
    else:
        model = ChronosBoltModelForForecasting(T5Config.from_dict(config))
    expected = model.state_dict()
    # Native T5 shares the token embedding. Safe serialization stores one copy.
    aliases = {"encoder.embed_tokens.weight": "shared.weight"}
    if config["architectures"] == ["ChronosBoltModelForForecasting"]:
        aliases["decoder.embed_tokens.weight"] = "shared.weight"
    for alias, source in aliases.items():
        if alias in expected and alias not in shapes and source in shapes:
            shapes[alias] = shapes[source]
    if set(shapes) != set(expected) or any(
        tuple(expected[k].shape) != shape for k, shape in shapes.items()
    ):
        raise ValueError("Selected forecast weights are incomplete or mismatched")
    weights = {}
    for path in sorted(root.glob("*.safetensors")):
        state = load_file(path, device="cpu")
        if weights.keys() & state.keys():
            raise ValueError("Selected forecast weights contain duplicate tensors")
        weights.update(state)
    for alias, source in aliases.items():
        if alias in expected and alias not in weights:
            weights[alias] = weights[source]
        elif alias in weights and not torch.equal(weights[alias], weights[source]):
            raise ValueError("Native tied forecast weights disagree")
    model.load_state_dict(weights, strict=True)
    return model.eval()


def execute(model, config, value, backend):
    import torch

    validate_config(config)
    native_config = config
    config = settings(config)
    context, horizon = value["context"], value.get("horizon", 16)
    levels = value.get("quantiles", [])
    if len(context) > min(config["context_length"], 2048) or horizon > min(
        config["horizon_length"], 512
    ):
        raise ValueError("Forecast input exceeds the native context/horizon")
    if any(field in value for field in ("pastCovariates", "futureCovariates")):
        raise ValueError("This native binding does not implement covariates")
    if any(q not in config["quantiles"] for q in levels):
        raise ValueError("Requested quantiles are not native output heads")
    # No target lookahead, caller-controlled normalization or fabricated intervals.
    if native_config.get("model_type") == "aurora":
        from oryn_engine.aurora_tasks import predictions

        points, full = predictions(model, context, horizon, backend)
    elif native_config.get("model_type") == "yug":
        from oryn_engine.yug_tasks import predictions

        points, full = predictions(
            model, context, horizon, backend, value.get("frequencyMilliseconds")
        )
    elif native_config.get("model_type") == "exaone-demand":
        from oryn_engine.exaone_demand import predictions

        points, full = predictions(model, context, horizon, backend)
    elif native_config.get("model_type") == "timesfm3":
        from oryn_engine.timesfm3_tasks import predictions

        points, full = predictions(model, context, horizon, backend)
    elif native_config.get("model_type") == "t5":
        options = {}
        if native_config["architectures"] == ["Chronos2Model"]:
            options["num_output_patches"] = math.ceil(horizon / config["patch_length"])
        output = model(
            context=torch.tensor([context], device=backend, dtype=torch.float32), **options
        )
        quantiles = output.quantile_preds.transpose(1, 2)
        points = quantiles[:, :, config["quantiles"].index(0.5)]
        full = torch.cat((points.unsqueeze(-1), quantiles), dim=-1)
    else:
        output = model(
            past_values=[torch.tensor(context, device=backend, dtype=torch.float32)],
            return_dict=True,
        )
        points, full = output.mean_predictions, output.full_predictions
    if (
        points.ndim != 2
        or points.shape[0] != 1
        or points.shape[1] < horizon
        or full.ndim != 3
        or full.shape[0] != 1
        or full.shape[1] < horizon
        or full.shape[2] != len(config["quantiles"]) + 1
        or not bool(torch.isfinite(points).all())
        or not bool(torch.isfinite(full).all())
    ):
        raise ValueError("Native forecast output is invalid")
    result = {
        "predictions": points[0, :horizon].float().cpu().tolist(),
        "quantiles": [
            {
                "level": q,
                "values": full[0, :horizon, config["quantiles"].index(q) + 1]
                .float()
                .cpu()
                .tolist(),
            }
            for q in levels
        ],
    }
    if "timestamps" in value:
        result["timestamps"] = [
            value["timestamps"][-1] + (i + 1) * value["frequencyMilliseconds"]
            for i in range(horizon)
        ]
    from oryn_protocol.task_contracts import make_profile, validate_result

    validate_result(
        make_profile("time-series-forecasting-v1"), {"kind": "structured", "value": result}, value
    )
    return result
