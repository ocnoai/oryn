import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import TimesFm2_5Config, TimesFm2_5ModelForPrediction

from oryn_engine import forecast_tasks, timesfm_original
from oryn_engine.shared_tasks import load_task


@pytest.fixture
def original(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(28)
    native = TimesFm2_5ModelForPrediction(
        TimesFm2_5Config(
            context_length=32,
            patch_length=4,
            horizon_length=16,
            hidden_size=16,
            intermediate_size=32,
            head_dim=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_hidden_layers=1,
            output_quantile_len=32,
            max_position_embeddings=32,
        )
    ).eval()
    config = {
        "architectures": ["TimesFmModelForPrediction"],
        "model_type": "timesfm",
        "context_length": 32,
        "patch_length": 4,
        "horizon_length": 16,
        "hidden_size": 16,
        "intermediate_size": 32,
        "head_dim": 4,
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "quantile_horizon_length": 32,
        "quantiles": [i / 10 for i in range(1, 10)],
        "rms_norm_eps": 1e-6,
        "torch_compile": False,
    }
    # Independently write the publisher layout from the native module parameters.
    state = {}
    for prefix, module in [
        ("tokenizer", native.model.input_ff_layer),
        ("output_projection_point", native.output_projection_point),
        ("output_projection_quantiles", native.output_projection_quantiles),
    ]:
        for name, value in module.state_dict().items():
            state[f"{prefix}.{name.replace('input_layer.', 'hidden_layer.')}"] = value
    layer = native.model.layers[0]
    state["stacked_xf.0.attn.qkv_proj.weight"] = torch.cat(
        [
            layer.self_attn.q_proj.weight,
            layer.self_attn.k_proj.weight,
            layer.self_attn.v_proj.weight,
        ]
    )
    for name, value in {
        "attn.out.weight": layer.self_attn.o_proj.weight,
        "attn.query_ln.scale": layer.self_attn.q_norm.weight,
        "attn.key_ln.scale": layer.self_attn.k_norm.weight,
        "attn.per_dim_scale.per_dim_scale": layer.self_attn.scaling,
        "ff0.weight": layer.mlp.fc1.weight,
        "ff1.weight": layer.mlp.fc2.weight,
        "pre_attn_ln.scale": layer.input_layernorm.weight,
        "post_attn_ln.scale": layer.post_attention_layernorm.weight,
        "pre_ff_ln.scale": layer.pre_feedforward_layernorm.weight,
        "post_ff_ln.scale": layer.post_feedforward_layernorm.weight,
    }.items():
        state[f"stacked_xf.0.{name}"] = value
    save_file(
        {k: v.detach().contiguous() for k, v in state.items()}, tmp_path / "model.safetensors"
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    yield tmp_path, config, native
    torch.set_num_threads(previous)


def test_original_layout_reproduces_native_weights_forecasts_and_time_axis(original):
    root, config, native = original
    loaded, processor, selected_config = load_task(root, "time-series-forecasting", "cpu")
    assert processor is None and selected_config == config
    assert loaded.config.model_type == "timesfm2_5"
    for key, value in native.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    value = {
        "context": list(range(1, 33)),
        "horizon": 8,
        # Untrained fixture heads need not be ordered. Exercise the native median;
        # the original-checkpoint acceptance cases exercise all three quantiles.
        "quantiles": [0.5],
        "timestamps": list(range(32)),
        "frequencyMilliseconds": 1,
    }
    with torch.inference_mode():
        expected = native(past_values=[torch.tensor(value["context"], dtype=torch.float32)])
        result = forecast_tasks.execute(loaded, config, value, "cpu")
    torch.testing.assert_close(
        torch.tensor(result["predictions"]), expected.mean_predictions[0, :8]
    )
    for item, column in zip(result["quantiles"], (5,), strict=True):
        torch.testing.assert_close(
            torch.tensor(item["values"]), expected.full_predictions[0, :8, column]
        )
    assert result["timestamps"] == list(range(32, 40))


@pytest.mark.parametrize("mutation", ["missing-head", "extra", "qkv-shape"])
def test_original_graph_must_be_complete_and_unambiguous(original, mutation):
    root, config, _ = original
    path = root / "model.safetensors"
    weights = load_file(path)
    if mutation == "missing-head":
        del weights["output_projection_quantiles.output_layer.weight"]
    elif mutation == "extra":
        weights["unexpected.weight"] = torch.zeros(1)
    else:
        weights["stacked_xf.0.attn.qkv_proj.weight"] = torch.zeros(47, 16)
    save_file(weights, path)
    with pytest.raises(ValueError, match="graph|shape"):
        timesfm_original.load(root, config)


@pytest.mark.parametrize(
    "change",
    [
        {"num_hidden_layers": 100000},
        {"hidden_size": 17},
        {"torch_compile": True},
        {"quantile_horizon_length": 4},
        {"rms_norm_eps": float("nan")},
        {"quantiles": [0.1, 0.9]},
        {"custom_loader": "remote.module"},
    ],
)
def test_original_config_rejects_unsupported_geometry_and_behavior(original, change):
    _, config, _ = original
    with pytest.raises(ValueError):
        forecast_tasks.validate_config(config | change)
