from __future__ import annotations

import json

import pytest
import torch

from oryn_engine.forecast_tasks import execute, load_chronos, settings, validate_config
from oryn_engine.shared_tasks import load_task
from oryn_engine.task_profiles import profile_for_execution


@pytest.fixture
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=["bolt", "chronos2"])
def checkpoint(request, tmp_path, single_thread):
    from chronos.chronos2.config import Chronos2CoreConfig
    from chronos.chronos2.model import Chronos2Model
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    from transformers import T5Config

    bolt = request.param == "bolt"
    config_class, model_class = (
        (T5Config, ChronosBoltModelForForecasting) if bolt else (Chronos2CoreConfig, Chronos2Model)
    )
    native = {
        "context_length": 32,
        "input_patch_size": 4,
        "input_patch_stride": 4,
        "quantiles": [0.1, 0.5, 0.9],
        "use_reg_token": True,
    }
    native.update(
        {"prediction_length": 8} if bolt else {"output_patch_size": 4, "max_output_patches": 4}
    )
    config = config_class(
        d_model=16,
        d_kv=8,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        decoder_start_token_id=0,
        chronos_config=native,
    )
    torch.manual_seed(61)
    model = model_class(config).eval()
    model.save_pretrained(tmp_path, safe_serialization=True)
    return tmp_path, model, json.loads((tmp_path / "config.json").read_text())


def test_native_chronos_safe_loading_forecasts_match_native_heads(checkpoint, monkeypatch):
    from huggingface_hub import HfApi

    def forbidden(*args, **kwargs):
        raise AssertionError("Forecast execution must stay offline")

    monkeypatch.setattr(HfApi, "model_info", forbidden)
    root, reference, config = checkpoint
    model, processor, loaded = load_task(root, "time-series-forecasting", "cpu")
    assert processor is None and loaded == config
    value = {
        "context": [1.0, 1.5, 2.0, 2.5, 3.0],
        "horizon": 6,
        "quantiles": [0.5],
        "timestamps": [1000, 2000, 3000, 4000, 5000],
        "frequencyMilliseconds": 1000,
    }
    options = {"num_output_patches": 2} if config["architectures"] == ["Chronos2Model"] else {}
    with torch.inference_mode():
        expected = reference(context=torch.tensor([value["context"]]), **options).quantile_preds
        actual = execute(model, loaded, value, "cpu")
    assert actual["predictions"] == pytest.approx(expected[0, 1, :6].tolist(), abs=1e-6)
    assert actual["quantiles"][0]["values"] == actual["predictions"]
    assert actual["timestamps"] == [6000, 7000, 8000, 9000, 10000, 11000]
    profile = profile_for_execution(
        {
            "runtimePack": {
                "runnerId": "transformers.shared-tasks",
                "taskContract": "time-series-forecasting-v1",
            }
        },
        config,
    )
    assert profile["features"]["covariates"] == "unsupported"


def test_native_chronos_missing_trained_patch_head_is_rejected(checkpoint):
    from safetensors.torch import load_file, save_file

    root, _, config = checkpoint
    path = root / "model.safetensors"
    weights = load_file(path)
    del weights["output_patch_embedding.output_layer.weight"]
    save_file(weights, path)
    with pytest.raises(ValueError, match="weights are incomplete"):
        load_chronos(root, config)


@pytest.mark.parametrize("change", ["architecture", "stride", "median", "dimension", "option"])
def test_native_chronos_rejects_unsupported_config(checkpoint, change):
    _, _, config = checkpoint
    if change == "architecture":
        config["architectures"] = ["T5ForConditionalGeneration"]
    elif change == "stride":
        config["chronos_config"]["input_patch_stride"] = 0
    elif change == "median":
        config["chronos_config"]["quantiles"] = [0.1, 0.9]
    elif change == "dimension":
        config["d_model"] = 2**30
    else:
        config["chronos_config"]["custom_model"] = "repository.py"
    with pytest.raises(ValueError):
        validate_config(config)


def test_native_chronos_enforces_context_horizon_and_native_quantiles(checkpoint):
    _, model, config = checkpoint
    bounds = settings(config)
    for value in [
        {"context": [1.0] * 33, "horizon": 4},
        {"context": [1.0, 2.0], "horizon": bounds["horizon_length"] + 1},
        {"context": [1.0, 2.0], "horizon": 4, "quantiles": [0.2]},
        {"context": [1.0, 2.0], "horizon": 4, "futureCovariates": [[3.0]]},
    ]:
        with pytest.raises(ValueError):
            execute(model, config, value, "cpu")
