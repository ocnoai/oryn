import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from oryn_engine import tabpfnmix_tasks
from oryn_engine.shared_tasks import validate_task_input, validate_task_output


@pytest.fixture
def original(tmp_path):
    from oryn_engine.tabpfnmix_tasks import native_class

    FoundationTransformer = native_class()

    config = dict(
        n_features=3,
        n_classes=3,
        dim=16,
        n_layers=2,
        n_heads=2,
        attn_dropout=0.0,
        y_as_float_embedding=True,
        task="regression",
    )
    torch.manual_seed(13)
    model = FoundationTransformer(**config).eval()
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path, config, model


def values():
    return {
        "columns": [
            {"name": "x", "type": "numeric"},
            {"name": "category", "type": "categorical"},
        ],
        "trainingRows": [[float(i), "even" if i % 2 == 0 else "odd"] for i in range(16)],
        "targets": [float(i * 3 - 2) for i in range(16)],
        "queryRows": [[2.5, "even"], [12.5, "odd"]],
    }


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_loaded_native_graph_parity_and_training_only_preprocessing(original, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    root, raw, native = original
    config = tabpfnmix_tasks.normalized_config(raw, "tabular-regression")
    loaded = tabpfnmix_tasks.load(root, config, backend)
    native.to(backend)
    value = validate_task_input("tabular-regression", values())
    expected = tabpfnmix_tasks.execute(native, "tabular-regression", value, backend)
    result = tabpfnmix_tasks.execute(loaded, "tabular-regression", value, backend)
    validate_task_output("tabular-regression", result)
    assert next(loaded.parameters()).device.type == backend
    assert result["predictions"] == pytest.approx(expected["predictions"], abs=1e-5)
    # Changing a different query must not refit support preprocessing or affect
    # the first prediction. Native attention only lets queries attend support.
    value["queryRows"][1] = [1000.0, "unseen"]
    changed = tabpfnmix_tasks.execute(loaded, "tabular-regression", value, backend)
    assert changed["predictions"][0] == pytest.approx(result["predictions"][0], abs=1e-5)


@pytest.mark.parametrize("change", ["missing", "extra", "shape"])
def test_incomplete_safe_graph_is_rejected(original, change):
    root, raw, _ = original
    path = root / "model.safetensors"
    state = load_file(path)
    key = "final_layer2.weight"
    if change == "missing":
        del state[key]
    elif change == "extra":
        state["invented.weight"] = torch.ones(1, 1)
    else:
        state[key] = torch.ones(1, 1)
    save_file(state, path)
    with pytest.raises(ValueError, match="incomplete or incompatible"):
        tabpfnmix_tasks.load(
            root, tabpfnmix_tasks.normalized_config(raw, "tabular-regression"), "cpu"
        )


@pytest.mark.parametrize(
    "change",
    [
        {"n_features": 10000},
        {"n_layers": True},
        {"attn_dropout": 0.5},
        {"task": "classification"},
        {"y_as_float_embedding": False},
        {"model_type": "remote"},
        {"auto_map": {"A": "custom.A"}},
    ],
)
def test_closed_native_configuration(original, change):
    with pytest.raises(ValueError):
        tabpfnmix_tasks.normalized_config({**original[1], **change}, "tabular-regression")


def test_missing_values_after_a_constant_column_preserve_alignment(original):
    root, raw, _ = original
    model = tabpfnmix_tasks.load(
        root, tabpfnmix_tasks.normalized_config(raw, "tabular-regression"), "cpu"
    )
    value = values()
    value["columns"].insert(0, {"name": "constant", "type": "numeric"})
    value["trainingRows"] = [[1.0, *row] for row in value["trainingRows"]]
    value["queryRows"] = [[1.0, None, "unseen"], [1.0, 12.5, None]]
    value["trainingRows"][4][1] = None
    result = tabpfnmix_tasks.execute(
        model,
        "tabular-regression",
        validate_task_input("tabular-regression", value),
        "cpu",
    )
    validate_task_output("tabular-regression", result)
    assert len(result["predictions"]) == 2 and np.isfinite(result["predictions"]).all()


def test_native_classification_head_preserves_label_order(tmp_path):
    from oryn_engine.tabpfnmix_tasks import native_class

    FoundationTransformer = native_class()

    raw = dict(
        n_features=3,
        n_classes=3,
        dim=16,
        n_layers=2,
        n_heads=2,
        attn_dropout=0.0,
        y_as_float_embedding=False,
        task="classification",
    )
    model = FoundationTransformer(**raw).eval()
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    loaded = tabpfnmix_tasks.load(
        tmp_path,
        tabpfnmix_tasks.normalized_config(raw, "tabular-classification"),
        "cpu",
    )
    value = values()
    value["targets"] = ["low" if i < 8 else "high" for i in range(16)]
    result = tabpfnmix_tasks.execute(
        loaded,
        "tabular-classification",
        validate_task_input("tabular-classification", value),
        "cpu",
    )
    validate_task_output("tabular-classification", result)
    assert result["classes"] == ["high", "low"]
    assert len(result["predictions"]) == 2


def test_shared_loading_and_narrowed_profile_use_the_native_binding(original):
    from oryn_engine.model_layouts import assess_layout
    from oryn_engine.shared_tasks import load_task
    from oryn_engine.task_profiles import profile_for_execution

    root, raw, _ = original
    layout = assess_layout(
        ["config.json", "model.safetensors", "unused.ckpt"], raw, "tabular-regression"
    )
    assert not layout.artifact_reasons and not layout.runtime_reasons
    assert set(layout.selected_files) == {"config.json", "model.safetensors"}
    model, processor, config = load_task(root, "tabular-regression", "cpu")
    assert processor is None and model.n_features == 3
    profile = profile_for_execution(
        {
            "runtimePack": {
                "taskContract": "tabular-regression-v1",
                "runnerId": "transformers.shared-tasks",
            },
            "technicalCapabilities": {"task": "tabular-regression"},
        },
        config,
    )
    assert profile["limits"]["featureColumns"] == 3
