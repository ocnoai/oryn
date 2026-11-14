import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from timesfm3.torch.model import TimesFM3Torch
from timesfm3.torch.timesfm3_forecaster import ModelConfig, TimesFM3Forecaster

from oryn_engine import forecast_tasks, timesfm3_tasks
from oryn_engine.generated_capabilities import normalized_transformers_config
from oryn_engine.model_metadata import parse_metadata, validate_metadata
from oryn_engine.shared_tasks import load_task


def original_config():
    return json.loads((Path(__file__).parent / "fixtures/timesfm3/config.json").read_text())


@pytest.fixture
def tiny(tmp_path):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(61627)
    config = original_config()
    config.update(input_patch_len=4, output_patch_len=8)
    config["residual_block_config"].update(hidden_dims=16, output_dims=16)
    config["transformer_config"]["num_layers"] = 1
    config["transformer_config"]["transformer"].update(hidden_dims=16, model_dims=16, num_heads=4)
    reference = TimesFM3Torch(**copy.deepcopy(config)).eval()
    save_file(reference.state_dict(), tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(config))
    yield tmp_path, config, reference
    torch.set_num_threads(old)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("length,horizon", [(9, 1), (16, 5), (33, 17)])
def test_complete_graph_matches_independent_native_forecaster(tiny, backend, length, horizon):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    root, config, reference = tiny
    loaded, processor, selected = load_task(root, "time-series-forecasting", backend)
    assert processor is None and selected["model_type"] == "timesfm3"
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name].cpu(), tensor, rtol=0, atol=0)

    # Independent upstream preprocessing/output wrapper, with its loader replaced
    # by the already-created reference fixture. No adapter helper makes inputs.
    class FixtureForecaster(TimesFM3Forecaster):
        def _init_model(self):
            self.model, self.device = reference, torch.device("cpu")

    forecaster = FixtureForecaster(
        ModelConfig(input_patch_length=4, output_patch_length=8, device="cpu")
    )
    for context in (
        np.sin(np.arange(length) * 0.3) + np.arange(length) * 0.02,
        np.full(length, 3.0),
        np.arange(length) * 0.7 - 4,
    ):
        value = {
            "context": context.tolist(),
            "horizon": horizon,
            "quantiles": [0.1, 0.5, 0.9],
            "timestamps": list(range(length)),
            "frequencyMilliseconds": 1,
        }
        expected = forecaster.predict(context, horizon, return_quantiles=True)
        with torch.inference_mode():
            observed = forecast_tasks.execute(loaded, selected, value, backend)
        np.testing.assert_allclose(observed["predictions"], expected.forecast, rtol=4e-4, atol=3e-5)
        for item, column in zip(observed["quantiles"], (0, 4, 8), strict=True):
            np.testing.assert_allclose(
                item["values"], expected.quantiles[:, column], rtol=4e-4, atol=3e-5
            )
        assert observed["timestamps"] == list(range(length, length + horizon))


@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "nonfinite", "integer"])
def test_no_missing_extra_or_invalid_native_weights(tiny, damage):
    root, config, _ = tiny
    path = root / "model.safetensors"
    weights = load_file(path)
    key = "output_head.weight"
    if damage == "missing":
        del weights[key]
    elif damage == "extra":
        weights["ignored_graph.weight"] = torch.zeros(1)
    elif damage == "shape":
        weights[key] = weights[key][:-1]
    elif damage == "integer":
        weights[key] = weights[key].long()
    else:
        weights[key][0, 0] = float("nan")
    save_file(weights, path)
    with pytest.raises(ValueError, match="incomplete|mismatched|invalid"):
        timesfm3_tasks.load(root, config)


@pytest.mark.parametrize(
    "path,value",
    [
        ("auto_map", {"AutoModel": "repository.Remote"}),
        ("custom_option", True),
        ("value_clip", 1e30),
        ("input_transform", "log"),
        ("use_stitching", False),
        ("input_patch_len", 0),
        ("output_patch_len", 33),
        ("quantiles", [0.1, 0.9]),
        ("transformer_config.num_layers", 1000),
        ("transformer_config.transformer.num_heads", 3),
        ("transformer_config.transformer.debug_no_masking", True),
        ("residual_block_config.dropout", 0.1),
        ("residual_block_config.output_dims", 64),
    ],
)
def test_closed_metadata_rejects_changed_graphs_and_behavior(path, value):
    config = original_config()
    target = config
    names = path.split(".")
    for key in names[:-1]:
        target = target[key]
    target[names[-1]] = value
    with pytest.raises(ValueError):
        timesfm3_tasks.native_config(config)


@pytest.mark.parametrize(
    "value",
    [
        {"value_clip": 1e20},
        {"other": 1e20},
        {"nested": {"value_clip": 1e20}},
        {"nested": None},
    ],
)
def test_large_arithmetic_clip_does_not_relax_other_metadata(value):
    if value == {"nested": None}:
        value = {"nested": original_config()}
    with pytest.raises(ValueError, match="number"):
        validate_metadata(value)


def test_original_metadata_parses_without_loading_native_libraries():
    config = original_config()
    assert parse_metadata(json.dumps(config).encode()) == config
    normalized = normalized_transformers_config(config, "time-series-forecasting")
    assert timesfm3_tasks.native_config(normalized) == config
    code = (
        "import json,sys; from oryn_engine.model_metadata import parse_metadata; "
        "from oryn_engine.generated_capabilities import normalized_transformers_config; "
        "c=normalized_transformers_config(parse_metadata(sys.stdin.buffer.read()),"
        "'time-series-forecasting'); assert c['model_type']=='timesfm3'; "
        "assert not {'torch','transformers','timesfm3'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", code], input=json.dumps(config).encode(), check=True)


@pytest.mark.parametrize("damage", [None, "missing-weights", "remote-code"])
def test_metadata_admission_uses_shared_planner_and_bounded_profile(tmp_path, damage):
    from oryn_protocol.task_contracts import effective_input, validate_invocation
    from test_models import FakeHub

    from oryn_engine.models import ModelStore, ModelStoreError
    from oryn_engine.runtime_packs.execution import PackRuntime

    config = original_config()
    if damage == "remote-code":
        config["auto_map"] = {"AutoModel": "repository.Model"}
    files = [("config.json", 100, "b" * 64)]
    if damage != "missing-weights":
        files.append(("model.safetensors", 100, "c" * 64))

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            if filename == "config.json":
                return parse_metadata(json.dumps(config).encode())
            raise OSError(filename)

    hub = Hub(config=config, files=files, pipeline_tag="time-series-forecasting")
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    request = {"provider": "huggingface", "repository": "test/timesfm3", "revision": "main"}
    if damage == "remote-code":
        with pytest.raises(ModelStoreError, match="metadata"):
            store.inspect(request)
        return
    result = store.inspect(request)
    assert not hub.downloaded
    if damage:
        assert not result["executable"]
        return
    assert result["executable"], result
    assert result["runtimePack"]["packId"] == "oryn.transformers-pytorch"
    profile = result["taskProfile"]
    value = effective_input(profile, {"context": [1.0] * 16})
    assert value["horizon"] == 16
    validate_invocation(profile, value | {"horizon": 512, "quantiles": [0.1, 0.5, 0.9]})
    for change in (
        {"context": [1.0] * 2049},
        {"horizon": 513},
        {"quantiles": [0.15]},
        {"pastCovariates": [[1.0]]},
    ):
        with pytest.raises(ValueError):
            validate_invocation(profile, value | change)
