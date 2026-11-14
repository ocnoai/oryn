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

from oryn_engine import aurora_tasks as binding
from oryn_engine import forecast_tasks, shared_tasks

FIXTURES = Path(__file__).parent / "fixtures/aurora"


def original():
    return json.loads((FIXTURES / "config.json").read_text())


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    from oryn_engine.native_aurora import modality_connector as connector

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    vectors = json.loads((FIXTURES / "reference.json").read_text())
    config = vectors["config"]
    # Numerical tests shrink the fixed, bundled BERT/ViT components. Production
    # has no component-path/config override; original full shapes are audited separately.
    auxiliary = tmp_path / "auxiliary"
    (auxiliary / "vision").mkdir(parents=True)
    (auxiliary / "text").mkdir()
    for name, key in [
        ("vision/config.json", "vision"),
        ("text/config.json", "text"),
        ("vision/preprocessor_config.json", "processor"),
    ]:
        (auxiliary / name).write_text(json.dumps(vectors["auxiliary"][key]))
    monkeypatch.setattr(connector.VisionEncoder, "config_path", str(auxiliary / "vision"))
    monkeypatch.setattr(connector.UnifiedImageProcessor, "config_path", str(auxiliary / "vision"))
    monkeypatch.setattr(connector.TextEncoder, "config_path", str(auxiliary / "text"))
    torch.manual_seed(vectors["seed"])
    reference = binding.native_model(config).eval()
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name.startswith("flow_match.net.") and (
                "adaLN_modulation" in name or "final_layer.linear" in name
            ):
                parameter.copy_(torch.randn_like(parameter) * 0.03)
    state = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    assert {
        k: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k, v in state.items()
    } == vectors["stateSha256"]
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(config | {"auto_map": binding.ALIASES}))
    (tmp_path / "modeling_aurora.py").write_text(
        "raise RuntimeError('repository code must stay inert')"
    )
    yield tmp_path, config, reference, vectors
    torch.set_num_threads(previous)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("case", range(6))
def test_empirical_forecasts_match_independent_upstream(tiny, backend, case):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    root, config, reference, fixture = tiny
    loaded, processor, selected = shared_tasks.load_task(root, "time-series-forecasting", backend)
    assert processor is None and selected == config
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name].cpu(), tensor, rtol=0, atol=0)
    data = fixture["cases"][case]
    value = {k: data[k] for k in ("context", "horizon")} | {
        "quantiles": binding.QUANTILES,
        "timestamps": list(range(len(data["context"]))),
        "frequencyMilliseconds": 1,
    }
    with torch.inference_mode():
        actual = forecast_tasks.execute(loaded, selected, value, backend)
    np.testing.assert_allclose(actual["predictions"], data["predictions"], rtol=8e-4, atol=5e-5)
    np.testing.assert_allclose(
        np.asarray([q["values"] for q in actual["quantiles"]]).T,
        data["quantiles"],
        rtol=8e-4,
        atol=5e-5,
    )
    assert actual["timestamps"] == list(
        range(len(data["context"]), len(data["context"]) + data["horizon"])
    )


@pytest.mark.parametrize(
    "damage",
    [
        "vision",
        "text",
        "prototype",
        "flow",
        "connector",
        "head",
        "shape",
        "extra",
        "nan",
        "integer",
        "alias",
    ],
)
def test_complete_multimodal_weights_are_required(tiny, damage):
    root, config, _, _ = tiny
    path = root / "model.safetensors"
    state = load_file(path)
    fragments = {
        "vision": "VisionEncoder.model.layers.0.attention.q_proj.weight",
        "text": "TextEncoder.model.embeddings.word_embeddings.weight",
        "prototype": "retriever.prototypes",
        "flow": "flow_match.net.final_layer.linear.weight",
        "connector": "ModalityConnector.",
        "head": "linear_head.output_proj.weight",
    }
    if damage in fragments:
        del state[next(k for k in state if fragments[damage] in k)]
    elif damage == "shape":
        state["retriever.prototypes"] = torch.zeros(3, 3)
    elif damage == "extra":
        state["unreviewed.weight"] = torch.zeros(2)
    elif damage == "nan":
        state["retriever.prototypes"][0, 0] = float("nan")
    elif damage == "integer":
        state["retriever.prototypes"] = state["retriever.prototypes"].to(torch.int32)
    else:
        state["model.VisionEncoder.model.encoder.layer.0.attention.attention.query.weight"] = state[
            "model.VisionEncoder.model.layers.0.attention.q_proj.weight"
        ].clone()
    save_file(state, path)
    with pytest.raises(ValueError):
        binding.load(root, config)


def test_original_legacy_vision_names_preserve_every_weight(tiny):
    root, config, reference, _ = tiny
    state = load_file(root / "model.safetensors")
    legacy = {}
    for name, value in state.items():
        if name.startswith("model.VisionEncoder.model.layers."):
            name = name.replace(
                "model.VisionEncoder.model.layers.", "model.VisionEncoder.model.encoder.layer."
            )
            for new, old in [
                ("attention.q_proj.", "attention.attention.query."),
                ("attention.k_proj.", "attention.attention.key."),
                ("attention.v_proj.", "attention.attention.value."),
                ("attention.o_proj.", "attention.output.dense."),
                ("mlp.fc1.", "intermediate.dense."),
                ("mlp.fc2.", "output.dense."),
            ]:
                name = name.replace(new, old)
        legacy[name] = value
    save_file(legacy, root / "model.safetensors")
    loaded = binding.load(root, config)
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "change",
    [
        {"auto_map": {"AutoModel": "remote.Model"}},
        {"hidden_size": 17},
        {"num_attention_heads": 0},
        {"num_dec_layers": 13},
        {"num_sampling_steps": 51},
        {"num_prototypes": 1001},
        {"num_prototypes": True},
        {"token_len": 1},
        {"hidden_act": "gelu"},
        {"norm_mode": "custom"},
        {"threshold_ratio": [0.1, float("nan")]},
        {"threshold_ratio": [0.5, 0.1]},
        {"dropout_rate": float("inf")},
        {"torch_dtype": "float64"},
        {"vision_config": {}},
        {"text_config": {}},
        {"config_path": "/tmp/remote"},
        {"unreviewed": True},
    ],
)
def test_closed_config_and_only_reviewed_remote_aliases(change):
    with pytest.raises(ValueError):
        binding.normalized_config(original() | change, "time-series-forecasting")


def test_unknown_task_never_receives_alias_normalization():
    raw = original()
    assert binding.normalized_config(raw, "text-generation") is raw


def test_metadata_does_not_import_native_model():
    code = f"""
import json,sys
from pathlib import Path
from oryn_engine.generated_capabilities import normalized_transformers_config
from oryn_engine.forecast_tasks import validate_config,settings
raw=json.loads(Path({str(FIXTURES / "config.json")!r}).read_text())
config=normalized_transformers_config(raw,'time-series-forecasting')
validate_config(config)
assert settings(config)['horizon_length']==256
assert 'torch' not in sys.modules
assert 'oryn_engine.native_aurora.modeling_aurora' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_vendored_source_hashes_license_declaration_and_removed_loader():
    from oryn_engine import native_aurora

    root = Path(native_aurora.__file__).parent
    manifest = json.loads((root / "provenance.json").read_text())
    assert manifest["revision"] == "c495b02c1b151be52a3c174237ed240aa66e6384"
    for entry in manifest["files"]:
        assert hashlib.sha256((root / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
    assert not (root / "ts_generation_mixin.py").exists()
    assert "license: mit" in (root / "README.md").read_text()
    assert (root / "NOTICE.md").is_file()


def test_sampling_is_repeatable_and_preserves_cpu_random_state(tiny):
    root, config, _, vectors = tiny
    loaded = binding.load(root, config)
    context = vectors["cases"][2]["context"]
    torch.manual_seed(8912)
    initial = torch.get_rng_state()
    with torch.inference_mode():
        a = binding.predictions(loaded, context, 17, "cpu")
        assert torch.equal(torch.get_rng_state(), initial)
        torch.manual_seed(3112)
        b = binding.predictions(loaded, context, 17, "cpu")
    torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
    assert not torch.equal(a[1][..., 1], a[1][..., -1])


def test_public_limits_and_native_output(tiny):
    from oryn_protocol.task_contracts import effective_input, validate_invocation, validate_result

    from oryn_engine.task_profiles import profile_for_execution

    root, config, _, _ = tiny
    profile = profile_for_execution(
        {
            "runtimePack": {
                "runnerId": "transformers.shared-tasks",
                "taskContract": "time-series-forecasting-v1",
            }
        },
        config,
    )
    value = effective_input(
        profile, {"context": [1.0, 2.0, 0.0, 4.0] * 4, "quantiles": [0.1, 0.5, 0.9]}
    )
    assert value["horizon"] == 16
    validate_invocation(profile, value)
    for extra in (
        {"horizon": 257},
        {"quantiles": [0.15]},
        {"context": [1.0] * 2049},
        {"pastCovariates": [[1.0]] * 16, "futureCovariates": [[1.0]] * 16},
    ):
        with pytest.raises(ValueError):
            validate_invocation(profile, value | extra)
    result = shared_tasks.execute_task(
        root, "time-series-forecasting", value, "cpu", None, None, lambda *_: None
    )
    validate_result(profile, {"kind": "structured", "value": result}, value)
    assert len(result["predictions"]) == 16


@pytest.mark.parametrize("damage", [None, "missing-weights", "remote", "dimensions"])
def test_original_metadata_discovery(tmp_path, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore, ModelStoreError
    from oryn_engine.runtime_packs.execution import PackRuntime

    config = original()
    if damage == "remote":
        config["auto_map"] = {"AutoModel": "remote.Model"}
    if damage == "dimensions":
        config["num_prototypes"] = 10000
    files = [("config.json", 100, "a" * 64)]
    if damage != "missing-weights":
        files.append(("model.safetensors", 100, "b" * 64))
    hub = FakeHub(config=config, files=files, pipeline_tag="time-series-forecasting")
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    request = {"provider": "huggingface", "repository": "test/aurora", "revision": "main"}
    if damage in {"remote", "dimensions"}:
        with pytest.raises(ModelStoreError) as error:
            store.inspect(request)
        assert error.value.code == "MODEL_METADATA_INVALID"
        assert not hub.downloaded
        return
    result = store.inspect(request)
    assert not hub.downloaded
    assert result["executable"] == (damage is None), result
    if damage is None:
        assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
        assert [f["name"] for f in result["selectedFiles"]] == ["config.json", "model.safetensors"]


@pytest.mark.parametrize(
    "damage", ["negative-counter", "floating-counter", "negative-variance", "overflow"]
)
def test_native_batch_statistics_and_float_range_are_validated(tiny, damage):
    root, config, _, _ = tiny
    state = load_file(root / "model.safetensors")
    counter = next(k for k in state if k.endswith(".num_batches_tracked"))
    variance = next(k for k in state if k.endswith(".running_var"))
    if damage == "negative-counter":
        state[counter] = torch.tensor(-1, dtype=torch.int64)
    elif damage == "floating-counter":
        state[counter] = torch.tensor(0.0)
    elif damage == "negative-variance":
        state[variance][0] = -1
    else:
        state["retriever.prototypes"] = torch.full_like(
            state["retriever.prototypes"], 1e200, dtype=torch.float64
        )
    save_file(state, root / "model.safetensors")
    with pytest.raises(ValueError):
        binding.load(root, config)
