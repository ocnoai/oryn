import json
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import load_file, save_file
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.architectures.tabpfn_v3_5 import TabPFNV3p5Config, get_architecture
from tabpfn.base import ModelSpecs
from tabpfn.inference_config import InferenceConfig

from oryn_engine import tabular_tasks
from oryn_engine.shared_tasks import validate_task_input, validate_task_output


@pytest.fixture
def original(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(61)
    config = TabPFNV3p5Config(
        max_num_classes=4,
        num_buckets=8,
        embed_dim=8,
        dist_embed_num_blocks=1,
        dist_embed_num_heads=1,
        dist_embed_num_inducing_points=2,
        feature_group_size=1,
        feat_agg_num_blocks=1,
        feat_agg_num_heads=1,
        feat_agg_num_cls_tokens=2,
        nlayers=1,
        icl_num_heads=1,
        icl_num_kv_heads_test=1,
        decoder_head_dim=4,
        decoder_num_heads=1,
        softmax_scaling_mlp_hidden_dim=4,
        fourier_encoding_num_frequencies=2,
        cell_ecdf_num_frequencies=1,
        cell_ecdf_num_buckets=32,
    )
    inference = InferenceConfig(
        PREPROCESS_TRANSFORMS=[
            dict(
                name="none",
                categorical_name="ordinal_shuffled",
                append_original=False,
                differentiable=False,
                global_transformer_name=None,
                max_features_per_estimator=768,
                max_onehot_cardinality=None,
            )
        ],
        N_ESTIMATORS=1,
        POLYNOMIAL_FEATURES="no",
        SUBSAMPLE_SAMPLES=None,
        REGRESSION_Y_PREPROCESS_TRANSFORMS=[None, "safepower"],
        SOFTMAX_TEMPERATURE=1.0,
    )
    native = get_architecture(config).eval()
    metadata = dict(
        architecture_name="tabpfn_v3_5", config=asdict(config), inference_config=asdict(inference)
    )
    path = tmp_path / tabular_tasks.weight_name()
    save_file(
        {k: v.contiguous() for k, v in native.state_dict().items()},
        path,
        metadata={k: json.dumps(v) for k, v in metadata.items()},
    )
    declaration = tabular_tasks.normalized_config(
        {"model_name": "TabPFN-v3.5"}, "tabular-classification"
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_name": "TabPFN-v3.5"}))
    yield tmp_path, declaration, ModelSpecs(native, config, inference), metadata
    torch.set_num_threads(previous)


def values(task):
    return dict(
        columns=[dict(name="x", type="numeric"), dict(name="y", type="numeric")],
        trainingRows=[[float(i), float(i % 3)] for i in range(12)],
        targets=["low" if i < 6 else "high" for i in range(12)]
        if task.endswith("classification")
        else [float(i * 2) for i in range(12)],
        queryRows=[[2.5, 1.0], [9.5, 0.0]],
    )


@pytest.mark.parametrize("task", ["tabular-classification", "tabular-regression"])
@pytest.mark.parametrize(
    "backend",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS device unavailable"
            ),
        ),
    ],
)
def test_safe_original_native_prediction_parity(original, task, backend, monkeypatch):
    root, config, reference, _ = original
    # Both public inference and loading must stay off every pickle-capable path.
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("pickle-capable loader called"))
    loaded = tabular_tasks.load(root, config, backend)
    for key, value in reference.model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key].cpu(), value, rtol=0, atol=0)
    value = validate_task_input(task, values(task))
    cls = TabPFNClassifier if task.endswith("classification") else TabPFNRegressor
    estimator = cls(
        model_path=reference,
        device=backend,
        n_estimators=1,
        categorical_features_indices=[],
        random_state=0,
        n_preprocessing_jobs=1,
        inference_precision=torch.float32,
        enable_torch_compile=False,
        show_progress_bar=False,
    )
    with torch.inference_mode():
        estimator.fit(
            pd.DataFrame(value["trainingRows"], columns=["x", "y"]), np.asarray(value["targets"])
        )
        query = pd.DataFrame(value["queryRows"], columns=["x", "y"])
        expected = (
            estimator.predict_proba(query)
            if task.endswith("classification")
            else estimator.predict(query)
        )
        result = tabular_tasks.execute(loaded, task, value, backend)
    validate_task_output(task, result)
    actual = (
        [p["probabilities"] for p in result["predictions"]]
        if task.endswith("classification")
        else result["predictions"]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    if task.endswith("classification"):
        assert result["classes"] == estimator.classes_.tolist()
        assert [x["label"] for x in result["predictions"]] == estimator.classes_[
            expected.argmax(axis=1)
        ].tolist()


@pytest.mark.parametrize("change", ["missing", "extra", "shape"])
def test_incomplete_original_graph_is_rejected(original, change):
    root, config, _, metadata = original
    path = root / tabular_tasks.weight_name()
    state = load_file(path)
    key = next(k for k, v in state.items() if v.ndim == 2)
    if change == "missing":
        del state[key]
    elif change == "extra":
        state["invented.weight"] = torch.ones(2, 2)
    else:
        state[key] = torch.ones(1, 1)
    save_file(state, path, metadata={k: json.dumps(v) for k, v in metadata.items()})
    with pytest.raises(ValueError, match="incomplete or incompatible"):
        tabular_tasks.load(root, config, "cpu")


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("config", "nlayers", 1000000),
        ("config", "embed_dim", True),
        ("config", "custom_import", "os.system"),
        ("config", "use_rope", False),
        ("inference_config", "N_ESTIMATORS", 1000000),
        ("inference_config", "PREPROCESS_TRANSFORMS", [{"name": "custom"}]),
    ],
)
def test_native_configuration_cannot_expand_execution(original, section, key, value):
    metadata = deepcopy(original[3])
    metadata[section][key] = value
    with pytest.raises(ValueError):
        tabular_tasks.native_metadata(metadata)


def test_never_accepts_a_legacy_or_competing_weight_file(original):
    root, config, _, _ = original
    (root / "other.safetensors").write_bytes(b"bad")
    with pytest.raises(ValueError, match="single original safe"):
        tabular_tasks.load(root, config, "cpu")


def test_rectangles_targets_and_cell_types_are_checked_before_fit():
    value = values("tabular-classification")
    for key, bad in [
        ("targets", ["only-one"]),
        ("queryRows", [[1.0]]),
        ("queryRows", [[True, 1.0]]),
    ]:
        malformed = {**value, key: bad}
        with pytest.raises(ValueError):
            validate_task_input("tabular-classification", malformed)


def test_shared_loading_profile_and_layout_use_the_same_native_graph(original):
    from oryn_engine.model_layouts import assess_layout
    from oryn_engine.shared_tasks import load_task
    from oryn_engine.task_profiles import profile_for_execution

    root, _, _, _ = original
    layout = assess_layout(
        ["config.json", tabular_tasks.weight_name(), "unused.safetensors", "unsafe.ckpt"],
        {"model_name": "TabPFN-v3.5"},
        "tabular-classification",
    )
    assert not layout.artifact_reasons and not layout.runtime_reasons
    assert set(layout.selected_files) == {"config.json", tabular_tasks.weight_name()}
    loaded, processor, config = load_task(root, "tabular-classification", "cpu")
    assert isinstance(loaded, ModelSpecs) and processor is None
    profile = profile_for_execution(
        {
            "runtimePack": {
                "taskContract": "tabular-classification-v1",
                "runnerId": "transformers.shared-tasks",
            },
            "technicalCapabilities": {"task": "tabular-classification"},
        },
        config,
    )
    assert profile["limits"] == {"trainingRows": 256, "queryRows": 64, "featureColumns": 32}


def test_categories_missing_values_and_unseen_query_categories(original):
    root, config, _, _ = original
    value = values("tabular-classification")
    value["columns"][1]["type"] = "categorical"
    for index, row in enumerate(value["trainingRows"]):
        row[1] = None if index % 3 == 0 else ("a" if index % 2 else "b")
    value["queryRows"] = [[2.5, "new-category"], [None, "a"]]
    value = validate_task_input("tabular-classification", value)
    loaded = tabular_tasks.load(root, config, "cpu")
    with torch.inference_mode():
        result = tabular_tasks.execute(loaded, "tabular-classification", value, "cpu")
    validate_task_output("tabular-classification", result)
    assert len(result["predictions"]) == 2
    assert result["classes"] == ["high", "low"]
    assert all(abs(sum(row["probabilities"]) - 1) < 1e-5 for row in result["predictions"])


def test_original_metadata_keeps_native_float_rotary_base():
    from pathlib import Path

    original = json.loads(
        (Path(__file__).parent / "fixtures/tabpfn_original_metadata.json").read_text()
    )
    config, inference = tabular_tasks.native_metadata(original)
    assert config.feat_agg_rope_base == 100000.0
    assert inference.N_ESTIMATORS == 8
