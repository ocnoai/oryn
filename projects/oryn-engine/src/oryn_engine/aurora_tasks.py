"""Closed Aurora forecasting over the complete multimodal numerical graph."""

from __future__ import annotations

import re

from oryn_engine.model_metadata import preflight_safetensors, validate_model_config

FAMILY = "aurora"
ARCHITECTURE = "AuroraForPrediction"
ALIASES = {
    "AutoConfig": "configuration_aurora.AuroraConfig",
    "AutoModelForCausalLM": "modeling_aurora.AuroraForPrediction",
}
QUANTILES = [i / 10 for i in range(1, 10)]
SAMPLES = 100


def normalized_config(config, task):
    if task == "time-series-forecasting" and config.get("model_type") == FAMILY:
        if config.get("auto_map") == ALIASES:
            config = {k: v for k, v in config.items() if k != "auto_map"}
        options(config)
    return config


def options(config):
    validate_model_config(config)
    fixed = {
        "model_type": FAMILY,
        "architectures": [ARCHITECTURE],
        "hidden_act": "silu",
        "rope_theta": 10000,
        "torch_dtype": "float32",
        "norm_mode": "batch",
    }
    integers = {
        "hidden_size": (16, 512),
        "intermediate_size": (16, 2048),
        "token_len": (4, 96),
        "max_position_embeddings": (64, 10000),
        "num_attention_heads": (1, 16),
        "num_enc_layers": (1, 12),
        "num_dec_layers": (1, 12),
        "num_sampling_steps": (1, 50),
        "flow_loss_depth": (1, 6),
        "diffusion_batch_mul": (1, 8),
        "num_prototypes": (6, 1000),
        "num_retriever_enc_layers": (1, 4),
        "num_retriever_dec_layers": (1, 4),
        "num_text_cross_layers": (1, 4),
        "num_vision_cross_layers": (1, 4),
        "num_text_connect_layers": (1, 4),
        "num_vision_connect_layers": (1, 4),
        "num_distill": (1, 16),
    }
    scalars = {"dropout_rate": (0, 0.5), "mask_ratio": (0, 1)}
    optional = {"_name_or_path", "transformers_version"}
    if (
        set(config)
        - (fixed.keys() | integers.keys() | scalars.keys() | optional | {"threshold_ratio"})
        or any(type(config.get(k)) is not type(v) or config[k] != v for k, v in fixed.items())
        or any(k in config and not isinstance(config[k], str) for k in optional)
    ):
        raise ValueError("AURORA_CONFIGURATION")
    for key, (low, high) in integers.items():
        if type(config.get(key)) is not int or not low <= config[key] <= high:
            raise ValueError("AURORA_DIMENSIONS")
    for key, (low, high) in scalars.items():
        if type(config.get(key)) not in {int, float} or not low <= config[key] <= high:
            raise ValueError("AURORA_OPTIONS")
    ratios = config.get("threshold_ratio")
    if (
        not isinstance(ratios, list)
        or not 1 <= len(ratios) <= 4
        or any(type(r) not in {int, float} or not 0 < r < 1 for r in ratios)
        or ratios != sorted(set(ratios))
        or config["hidden_size"] % (2 * config["num_attention_heads"])
        or config["hidden_size"] // config["num_attention_heads"] < 4
    ):
        raise ValueError("AURORA_GRAPH")
    return {
        k: v
        for k, v in config.items()
        if k not in optional | {"model_type", "architectures", "torch_dtype"}
    }


def settings(config):
    native = options(config)
    return {
        "context_length": 2048,
        "horizon_length": 256,
        "patch_length": native["token_len"],
        "quantiles": QUANTILES,
    }


def native_model(config):
    from oryn_engine.native_aurora.configuration_aurora import AuroraConfig
    from oryn_engine.native_aurora.modeling_aurora import AuroraForPrediction

    return AuroraForPrediction(AuroraConfig(**options(config)))


def validate_weight_shapes(config, shapes):
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    if native_names(shapes) != {k: tuple(v.shape) for k, v in model.state_dict().items()}:
        raise ValueError("AURORA_WEIGHTS_INCOMPLETE")
    return model


def native_names(state):
    """Bind legacy ViT names to the pinned native graph without altering tensors."""
    result = {}
    for name, value in state.items():
        prefix = "model.VisionEncoder.model."
        if name.startswith(prefix + "encoder.layer."):
            leaf = re.sub(r"^encoder\.layer\.(\d+)\.", r"layers.\1.", name[len(prefix) :])
            for old, new in (
                ("attention.attention.query.", "attention.q_proj."),
                ("attention.attention.key.", "attention.k_proj."),
                ("attention.attention.value.", "attention.v_proj."),
                ("attention.output.dense.", "attention.o_proj."),
                ("intermediate.dense.", "mlp.fc1."),
                ("output.dense.", "mlp.fc2."),
            ):
                leaf = leaf.replace(old, new)
            name = prefix + leaf
        if name in result:
            raise ValueError("AURORA_DUPLICATE_NATIVE_WEIGHT")
        result[name] = value
    return result


def load(root, config):
    import torch
    from safetensors.torch import load_file

    model = validate_weight_shapes(config, preflight_safetensors(root))
    state = {}
    for path in sorted(root.glob("*.safetensors")):
        values = load_file(path)
        if state.keys() & values.keys():
            raise ValueError("AURORA_WEIGHT_VALUES")
        state.update(values)
    state = native_names(state)
    expected = model.state_dict()
    for name, value in state.items():
        if name.endswith(".num_batches_tracked") and expected[name].dtype == torch.int64:
            if value.dtype != torch.int64 or bool((value < 0).any()):
                raise ValueError("AURORA_BATCH_COUNTER")
        else:
            if not value.is_floating_point() or not bool(torch.isfinite(value.float()).all()):
                raise ValueError("AURORA_WEIGHT_VALUES")
            if name.endswith(".running_var") and bool((value < 0).any()):
                raise ValueError("AURORA_BATCH_VARIANCE")
            state[name] = value.float()
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


def predictions(model, context, horizon, backend):
    import torch

    # Native flow sampling draws its noise on CPU, then moves it to the model.
    # A fixed private stream makes the bounded empirical result reproducible
    # without changing the worker's ambient CPU random state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        samples = (
            model(
                input_ids=torch.tensor([context], device=backend, dtype=torch.float32),
                max_output_length=horizon,
                inference_token_len=model.config.token_len,
                revin=True,
                num_samples=SAMPLES,
                return_dict=True,
            )
            .logits.float()
            .cpu()
        )
    if samples.shape != (1, SAMPLES, horizon) or not bool(torch.isfinite(samples).all()):
        raise ValueError("AURORA_SAMPLES_INVALID")
    points = samples.mean(dim=1)
    quantiles = torch.quantile(samples, torch.tensor(QUANTILES), dim=1).permute(1, 2, 0)
    return points, torch.cat((points.unsqueeze(-1), quantiles), dim=-1)
