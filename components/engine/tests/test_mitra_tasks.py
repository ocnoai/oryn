import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from oryn_engine import tabular_tasks
from oryn_engine.shared_tasks import validate_task_input, validate_task_output


@pytest.fixture
def graph(tmp_path):
    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D

    def create(task, regression_outputs=1):
        torch.manual_seed(7)
        raw = dict(
            dim=16,
            n_heads=2,
            n_layers=1,
            dim_output=3 if task.endswith("classification") else regression_outputs,
            task="CLASSIFICATION" if task.endswith("classification") else "REGRESSION",
        )
        model = Tab2D(**raw, use_pretrained_weights=False, path_to_weights="", device="cpu").eval()
        save_file(model.state_dict(), tmp_path / "model.safetensors")
        (tmp_path / "config.json").write_text(json.dumps(raw))
        return tmp_path, tabular_tasks.normalized_config(raw, task), model

    return create


def values(task):
    return dict(
        columns=[dict(name="x", type="numeric"), dict(name="y", type="numeric")],
        trainingRows=[[float(i), float(i % 3)] for i in range(12)],
        queryRows=[[1.5, 2.0], [8.5, 1.0]],
        targets=["low" if i < 6 else "high" for i in range(12)]
        if task.endswith("classification")
        else [3.0 * i - 2 for i in range(12)],
    )


@pytest.mark.parametrize("task", ["tabular-classification", "tabular-regression"])
@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_native_graph_safe_loading_and_device_parity(graph, task, backend, monkeypatch):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    root, config, native = graph(task)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("unsafe loader called"))
    loaded = tabular_tasks.load(root, config, backend)
    for key, tensor in native.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key].cpu(), tensor)
    value = validate_task_input(task, values(task))
    actual = tabular_tasks.execute(loaded, task, value, backend)
    expected = tabular_tasks.execute(native, task, value, "cpu")
    validate_task_output(task, actual)
    if task.endswith("classification"):
        assert actual["classes"] == ["high", "low"]
        np.testing.assert_allclose(
            [r["probabilities"] for r in actual["predictions"]],
            [r["probabilities"] for r in expected["predictions"]],
            rtol=1e-4,
            atol=1e-5,
        )
    else:
        np.testing.assert_allclose(
            actual["predictions"], expected["predictions"], rtol=1e-4, atol=1e-5
        )


@pytest.mark.parametrize(
    "change",
    [
        {"dim": 4096},
        {"dim": True},
        {"n_heads": 3},
        {"task": "REGRESSION"},
        {"custom_code": "unsafe.py"},
    ],
)
def test_declaration_rejects_wrong_task_unbounded_or_custom_geometry(graph, change):
    _, config, _ = graph("tabular-classification")
    with pytest.raises(ValueError):
        tabular_tasks.normalized_config(config | change, "tabular-classification")


@pytest.mark.parametrize("change", ["missing", "extra", "shape"])
def test_graph_is_complete_and_exact_before_loading(graph, change):
    root, config, _ = graph("tabular-classification")
    path = root / "model.safetensors"
    weights = load_file(path)
    key = next(iter(weights))
    if change == "missing":
        del weights[key]
    elif change == "extra":
        weights["foreign.weight"] = torch.zeros(1)
    else:
        weights[key] = torch.zeros(1)
    save_file(weights, path)
    with pytest.raises(ValueError, match="incomplete or incompatible"):
        tabular_tasks.load(root, config, "cpu")


def test_class_bound_and_query_only_categories(graph):
    root, config, _ = graph("tabular-classification")
    model = tabular_tasks.load(root, config, "cpu")
    value = values("tabular-classification")
    value["columns"][1]["type"] = "categorical"
    for i, row in enumerate(value["trainingRows"]):
        row[1] = "a" if i % 2 else None
    value["queryRows"][0][1] = "query-only"
    value["queryRows"][1][1] = None
    result = tabular_tasks.execute(model, "tabular-classification", value, "cpu")
    validate_task_output("tabular-classification", result)
    value["targets"] = [str(i % 4) for i in range(12)]
    with pytest.raises(ValueError, match="class count"):
        tabular_tasks.execute(model, "tabular-classification", value, "cpu")


def test_binned_regression_preserves_native_bucket_and_inverse_scale(graph, monkeypatch):
    root, config, _ = graph("tabular-regression", regression_outputs=1000)
    model = tabular_tasks.load(root, config, "cpu")
    value = dict(
        columns=[dict(name="x", type="numeric")],
        trainingRows=[[0.0], [5.0], [10.0]],
        queryRows=[[1.0], [9.0]],
        targets=[0.0, 5.001, 10.0],
    )

    def native_forward(support, labels, queries, *masks):
        torch.testing.assert_close(labels, torch.tensor([[249, 500, 749]]))
        output = torch.full((1, 2, 1000), -20.0)
        output[0, 0, 250] = 20.0
        output[0, 1, 750] = 20.0
        return output

    monkeypatch.setattr(model, "forward", native_forward)
    result = tabular_tasks.execute(model, "tabular-regression", value, "cpu")
    np.testing.assert_allclose(result["predictions"], [0.01, 10.01], atol=1e-5)
    validate_task_output("tabular-regression", result)
