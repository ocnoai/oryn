import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from oryn_engine import exaone_demand as binding
from oryn_engine import forecast_tasks, shared_tasks

FIXTURES = Path(__file__).parent / "fixtures/exaone_demand"


def original():
    return json.loads((FIXTURES / "config.json").read_text())


@pytest.fixture
def tiny(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    vectors = json.loads((FIXTURES / "reference.json").read_text())
    config = vectors["config"]
    torch.manual_seed(vectors["seed"])
    reference = binding.native_model(config).eval()
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if "lora_" in name:
                parameter.copy_(torch.randn_like(parameter) * 0.03)
    state = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    assert {
        k: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k, v in state.items()
    } == vectors["stateSha256"]
    save_file(state, tmp_path / "exaone-demand-1.0.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.py").write_text("raise RuntimeError('repository code must stay inert')")
    yield tmp_path, config, reference, vectors
    torch.set_num_threads(previous)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("case", range(5))
def test_complete_native_forecasts_match_independent_pinned_reference(tiny, backend, case):
    from oryn_engine.native_exaone_demand import adapter

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    root, config, reference, fixture = tiny
    loaded, processor, selected = shared_tasks.load_task(root, "time-series-forecasting", backend)
    assert processor is None and selected == config
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name].cpu(), tensor, rtol=0, atol=0)
    data = fixture["cases"][case]
    # PyTorch 2.15's float32 reductions differ by up to three ULPs from the
    # pinned reference. Keep a tight rounding bound and the forecast checks below.
    np.testing.assert_array_max_ulp(
        adapter.demand_stats(torch.tensor([data["context"]])).numpy(),
        np.asarray(data["routerStatistics"], dtype=np.float32),
        maxulp=4,
    )
    length = len(data["context"])
    value = {
        "context": data["context"],
        "horizon": data["horizon"],
        "quantiles": config["quantiles"],
        "timestamps": list(range(length)),
        "frequencyMilliseconds": 1,
    }
    with torch.inference_mode():
        _, full = binding.predictions(loaded, data["context"], data["horizon"], backend)
        actual = forecast_tasks.execute(loaded, selected, value | {"quantiles": []}, backend)
    np.testing.assert_allclose(actual["predictions"], data["quantiles"][1], rtol=4e-4, atol=3e-5)
    for column in range(len(config["quantiles"])):
        np.testing.assert_allclose(
            full[0, : data["horizon"], column + 1].cpu(),
            data["quantiles"][column],
            rtol=4e-4,
            atol=3e-5,
        )
    assert actual["timestamps"] == list(range(length, length + data["horizon"]))


@pytest.mark.parametrize(
    "damage",
    [
        "router",
        "branch",
        "base",
        "cross-series",
        "normalization",
        "head",
        "shape",
        "extra",
        "nan",
        "integer",
        "quantiles",
        "tied-conflict",
    ],
)
def test_whole_graph_router_branches_buffers_and_aliases_are_mandatory(tiny, damage):
    root, config, reference, _ = tiny
    path = root / "exaone-demand-1.0.safetensors"
    state = load_file(path)
    fragments = {
        "router": "gate.0.weight",
        "branch": "lora_b",
        "base": "to_q.base.weight",
        "cross-series": "encoder.block.1.block.1.to_q",
        "normalization": "encoder.norm.weight",
        "head": "output_embed.linear2.weight",
    }
    if damage in fragments:
        del state[next(k for k in state if fragments[damage] in k)]
    elif damage == "shape":
        state["output_embed.linear2.weight"] = torch.zeros(3, 3)
    elif damage == "extra":
        state["unreviewed.weight"] = torch.zeros(2)
    elif damage == "nan":
        state["output_embed.linear2.weight"][0, 0] = float("nan")
    elif damage == "integer":
        state["output_embed.linear2.weight"] = state["output_embed.linear2.weight"].to(torch.int32)
    elif damage == "quantiles":
        state["quantiles"][0] = 0.2
    else:
        group = binding.tied_parameters(reference)[0]
        state[group[-1]] = state[group[-1]] + 1
    save_file(state, path)
    with pytest.raises(ValueError):
        binding.load(root, config)


@pytest.mark.parametrize(
    "change",
    [
        {"auto_map": {"AutoModel": "remote.Model"}},
        {"dim_model": 17},
        {"num_heads": 0},
        {"num_layers": 49},
        {"dim_model": 2048, "num_layers": 48},
        {"adapter_ranks": [16, 4]},
        {"adapter_ranks": [16, 4, 4, 4, True]},
        {"adapter_targets": ["unreviewed"]},
        {"adapter_temperature": 0},
        {"quantiles": [0.1, 0.9]},
        {"quantiles": [0.5, 0.1, 0.9]},
        {"input_patch_stride": 8},
        {"output_patch_size": 32},
        {"ssmax": False},
        {"adapter_enabled": False},
        {"unreviewed": True},
        {"torch_dtype": "float64"},
    ],
)
def test_closed_native_configuration(change):
    with pytest.raises(ValueError):
        binding.options(original() | change)


def test_metadata_does_not_import_torch_or_native_source():
    code = f"""
import json,sys
from pathlib import Path
from oryn_engine.forecast_tasks import validate_config,settings
config=json.loads(Path({str(FIXTURES / "config.json")!r}).read_text())
validate_config(config)
assert settings(config)['horizon_length']==64
assert 'torch' not in sys.modules
assert 'oryn_engine.native_exaone_demand.model' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_vendored_numeric_modules_preserve_pinned_bytes_and_notices():
    from oryn_engine import native_exaone_demand

    root = Path(native_exaone_demand.__file__).parent
    manifest = json.loads((root / "provenance.json").read_text())
    assert manifest["revision"] == "5e6e5224f2a0205f146bc772082d7fa4f59bd7c5"
    for entry in manifest["files"]:
        assert hashlib.sha256((root / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
    assert not (root / "_hub.py").exists() and not (root / "_loading.py").exists()
    for name in ["LICENSE", "NOTICE.md", "DEMAND-NOTICE.md"]:
        assert (root / name).is_file()


def test_public_execution_limits_and_quantile_identity(tiny):
    from oryn_protocol.task_contracts import effective_input, validate_invocation, validate_result

    from oryn_engine.task_profiles import profile_for_execution

    root, config, reference, _ = tiny
    profile = profile_for_execution(
        {
            "runtimePack": {
                "runnerId": "transformers.shared-tasks",
                "taskContract": "time-series-forecasting-v1",
            }
        },
        config,
    )
    value = effective_input(profile, {"context": [1.0, 2.0, 0.0, 4.0] * 4})
    assert value["horizon"] == 16
    value["quantiles"] = [0.1, 0.5, 0.9]
    validate_invocation(profile, value)
    for extra in (
        {"horizon": 17},
        {"quantiles": [0.15]},
        {"context": [1.0] * 65},
        {"pastCovariates": [[1.0]] * 16, "futureCovariates": [[1.0]] * 16},
    ):
        with pytest.raises(ValueError):
            validate_invocation(profile, value | extra)
    # A controlled monotone readout exercises the public result envelope. Native
    # random heads above remain unchanged; their raw quantiles must not be sorted.
    with torch.no_grad():
        reference.output_embed.linear2.weight.zero_()
        reference.output_embed.linear2.bias.copy_(torch.tensor([-3.0] * 4 + [0.0] * 4 + [3.0] * 4))
        reference.output_embed.linear3.weight.zero_()
        reference.output_embed.linear3.bias.zero_()
    save_file(
        {k: v.detach().clone() for k, v in reference.state_dict().items()},
        root / "exaone-demand-1.0.safetensors",
    )
    result = shared_tasks.execute_task(
        root, "time-series-forecasting", value, "cpu", None, None, lambda *_: None
    )
    validate_result(profile, {"kind": "structured", "value": result}, value)
    assert len(result["predictions"]) == 16
    assert result["predictions"] == result["quantiles"][1]["values"]


def test_crossing_quantiles_are_rejected_without_reordering(tiny):
    root, config, _, vectors = tiny
    data = vectors["cases"][2]
    value = {k: data[k] for k in ("context", "horizon")} | {"quantiles": config["quantiles"]}
    loaded = binding.load(root, config)
    with torch.inference_mode():
        _, full = binding.predictions(loaded, data["context"], data["horizon"], "cpu")
        assert (full[..., 1:-1] > full[..., 2:]).any()
        with pytest.raises(ValueError, match="TASK_OUTPUT_INVALID"):
            forecast_tasks.execute(loaded, config, value, "cpu")


@pytest.mark.parametrize("damage", [None, "missing-weights", "remote", "router"])
def test_original_metadata_discovery(tmp_path, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    config = original()
    if damage == "remote":
        config["auto_map"] = {"AutoModel": "remote.Model"}
    if damage == "router":
        config["adapter_targets"] = ["to_q"]
    files = [("config.json", 100, "a" * 64)]
    if damage != "missing-weights":
        files.append(("exaone-demand-1.0.safetensors", 100, "b" * 64))
    hub = FakeHub(config=config, files=files, pipeline_tag="time-series-forecasting")
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "test/exaone-demand", "revision": "main"}
    )
    assert not hub.downloaded
    assert result["executable"] == (damage is None), result
    if damage is None:
        assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
        assert [f["name"] for f in result["selectedFiles"]] == [
            "config.json",
            "exaone-demand-1.0.safetensors",
        ]
