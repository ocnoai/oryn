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

from oryn_engine import forecast_tasks, shared_tasks
from oryn_engine import yug_tasks as binding

FIXTURES = Path(__file__).parent / "fixtures/yug"


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
    state = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    assert {
        k: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k, v in state.items()
    } == vectors["stateSha256"]
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.py").write_text("raise RuntimeError('repository code must stay inert')")
    yield tmp_path, config, reference, vectors
    torch.set_num_threads(previous)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("case", range(8))
def test_direct_forecasts_match_independent_upstream(tiny, backend, case):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    root, config, reference, fixture = tiny
    loaded, processor, selected = shared_tasks.load_task(root, "time-series-forecasting", backend)
    assert processor is None and selected == config
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name].cpu(), tensor, rtol=0, atol=0)
    data = fixture["cases"][case]
    milliseconds = data["frequencyMilliseconds"]
    assert list(binding.frequency(milliseconds)) == data["nativeFrequency"]
    value = {k: data[k] for k in ("context", "horizon")} | {"quantiles": []}
    if milliseconds is not None:
        value.update(
            frequencyMilliseconds=milliseconds,
            timestamps=[i * milliseconds for i in range(len(data["context"]))],
        )
    with torch.inference_mode():
        _, full = binding.predictions(
            loaded, data["context"], data["horizon"], backend, milliseconds
        )
        actual = forecast_tasks.execute(loaded, selected, value, backend)
    np.testing.assert_allclose(actual["predictions"], data["quantiles"][1], rtol=5e-4, atol=3e-5)
    for column in range(len(config["quantiles"])):
        np.testing.assert_allclose(
            full[0, : data["horizon"], column + 1].cpu(),
            data["quantiles"][column],
            rtol=5e-4,
            atol=3e-5,
        )
    if milliseconds is not None:
        length = len(data["context"])
        assert actual["timestamps"] == [
            i * milliseconds for i in range(length, length + data["horizon"])
        ]
    else:
        assert "timestamps" not in actual


@pytest.mark.parametrize(
    "damage",
    [
        "target-attention",
        "variate-attention",
        "cross-attention",
        "compression",
        "frequency",
        "unused-normalization",
        "head",
        "shape",
        "extra",
        "nan",
        "integer",
        "rotary",
    ],
)
def test_complete_graph_and_rotary_buffers_are_required(tiny, damage):
    root, config, _, _ = tiny
    path = root / "model.safetensors"
    state = load_file(path)
    fragments = {
        "target-attention": "self_attn_tar.query.weight",
        "variate-attention": "self_attn_var.query.weight",
        "cross-attention": "cross_attn_1.query.weight",
        "compression": "variate_mlp.",
        "frequency": "freq_embedding.freq_embedding.weight",
        "unused-normalization": "output_patch_norm.weight",
        "head": "output_layer.output_layer.weight",
    }
    if damage in fragments:
        del state[next(k for k in state if fragments[damage] in k)]
    elif damage == "shape":
        state["output_layer.output_layer.weight"] = torch.zeros(3, 3)
    elif damage == "extra":
        state["unreviewed.weight"] = torch.zeros(2)
    elif damage == "nan":
        state["output_layer.output_layer.weight"][0, 0] = float("nan")
    elif damage == "integer":
        state["output_layer.output_layer.weight"] = state["output_layer.output_layer.weight"].to(
            torch.int32
        )
    else:
        name = next(k for k in state if k.endswith("inv_freq"))
        state[name] = state[name] + 1
    save_file(state, path)
    with pytest.raises(ValueError):
        binding.load(root, config)


@pytest.mark.parametrize(
    "change",
    [
        {"auto_map": {"AutoModel": "remote.Model"}},
        {"d_model": 17},
        {"num_attn_heads": 0},
        {"num_decoder_layers": 25},
        {"d_model": 2048, "num_decoder_layers": 24},
        {"output_dim": 63},
        {"patch_len": True},
        {"freq_num": 8},
        {"library_name": "custom"},
        {"quantiles": [0.1, 0.9]},
        {"quantiles": [0.5, 0.1, 0.9]},
        {"quantiles": [0.1, 0.5, float("nan")]},
        {"eps": 0},
        {"expansion": float("inf")},
        {"unreviewed": True},
        {"kernel_size": 5},
    ],
)
def test_closed_native_configuration(change):
    with pytest.raises(ValueError):
        binding.options(original() | change)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, 31536000001])
def test_frequency_rejects_invalid_interval(value):
    with pytest.raises(ValueError, match="YUG_FREQUENCY"):
        binding.frequency(value)


def test_metadata_does_not_import_torch_or_native_source():
    code = f"""
import json,sys
from pathlib import Path
from oryn_engine.forecast_tasks import validate_config,settings
config=json.loads(Path({str(FIXTURES / "config.json")!r}).read_text())
validate_config(config)
assert settings(config)['horizon_length']==64
assert 'torch' not in sys.modules
assert 'oryn_engine.native_yug.model' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_vendored_source_preserves_pinned_bytes_and_notice():
    from oryn_engine import native_yug

    root = Path(native_yug.__file__).parent
    manifest = json.loads((root / "provenance.json").read_text())
    assert manifest["revision"] == "2ad7da7133d5cb5b27be3ac1cc5e7963e2ba4e79"
    for entry in manifest["files"]:
        assert hashlib.sha256((root / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
    assert sorted(p.name for p in root.glob("*.py")) == ["__init__.py", "model.py"]
    assert (root / "LICENSE").is_file() and (root / "NOTICE").is_file()


def test_public_profile_limits_and_quantile_identity(tiny):
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
        {"context": [1.0] * 2049},
        {"pastCovariates": [[1.0]] * 16, "futureCovariates": [[1.0]] * 16},
    ):
        with pytest.raises(ValueError):
            validate_invocation(profile, value | extra)
    with torch.no_grad():
        reference.output_layer.output_layer.weight.zero_()
        reference.output_layer.output_layer.bias.copy_(
            torch.tensor(([-3.0] * 4 + [0.0] * 4 + [3.0] * 4) * 4)
        )
        reference.output_layer.residual_layer.weight.zero_()
        reference.output_layer.residual_layer.bias.zero_()
    save_file(
        {k: v.detach().clone() for k, v in reference.state_dict().items()},
        root / "model.safetensors",
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


@pytest.mark.parametrize("damage", [None, "missing-weights", "remote", "frequency"])
def test_original_metadata_discovery(tmp_path, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    config = original()
    if damage == "remote":
        config["auto_map"] = {"AutoModel": "remote.Model"}
    if damage == "frequency":
        config["freq_num"] = 8
    files = [("config.json", 100, "a" * 64)]
    if damage != "missing-weights":
        files.append(("model.safetensors", 100, "b" * 64))
    hub = FakeHub(config=config, files=files, pipeline_tag="time-series-forecasting")
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "test/yug", "revision": "main"}
    )
    assert not hub.downloaded
    assert result["executable"] == (damage is None), result
    if damage is None:
        assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
        assert [f["name"] for f in result["selectedFiles"]] == ["config.json", "model.safetensors"]
