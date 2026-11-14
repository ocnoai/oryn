"""Bounded native TabPFN inference over safe original checkpoint bytes.

Model selection uses the installed native family's default file, never repository IDs.
Only the sealed 3.5 architecture and data-only preprocessing settings are admitted.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import get_type_hints

TASK_CONTRACTS = {
    "tabular-classification": "tabular-classification-v1",
    "tabular-regression": "tabular-regression-v1",
}
TASK_FAMILIES = {task: {"tabpfn_v3_5", "mitra_tab2d", "tabpfn_mix"} for task in TASK_CONTRACTS}


def normalized_config(config, task):
    from oryn_engine.mitra_tasks import normalized_config as normalize_mitra
    from oryn_engine.tabpfnmix_tasks import normalized_config as normalize_mix

    config = normalize_mix(normalize_mitra(config, task), task)
    if task not in TASK_CONTRACTS or config.get("model_name") != "TabPFN-v3.5":
        return config
    if set(config) - {"model_name", "model_type", "architectures"}:
        raise ValueError("Unsupported native tabular declaration")
    if config.get("model_type", "tabpfn_v3_5") != "tabpfn_v3_5" or config.get(
        "architectures", ["TabPFNV3p5"]
    ) != ["TabPFNV3p5"]:
        raise ValueError("Conflicting native tabular declaration")
    return {**config, "model_type": "tabpfn_v3_5", "architectures": ["TabPFNV3p5"]}


def validate_config(config):
    if config.get("model_type") == "tabpfn_mix":
        from oryn_engine.tabpfnmix_tasks import FIELDS
        from oryn_engine.tabpfnmix_tasks import normalized_config as normalize_mix

        task = (
            "tabular-classification"
            if config.get("task") == "classification"
            else "tabular-regression"
        )
        if not config.keys() >= FIELDS or normalize_mix(config, task) != config:
            raise ValueError("Native TabPFN-Mix configuration is unavailable")
        return
    if config.get("model_type") == "mitra_tab2d":
        from oryn_engine.mitra_tasks import FIELDS
        from oryn_engine.mitra_tasks import normalized_config as normalize_mitra

        task = (
            "tabular-classification"
            if config.get("task") == "CLASSIFICATION"
            else "tabular-regression"
        )
        if not config.keys() >= FIELDS or normalize_mitra(config, task) != config:
            raise ValueError("Native Mitra configuration is unavailable")
        return
    if (
        normalized_config(config, "tabular-classification") != config
        or config.get("model_type") != "tabpfn_v3_5"
    ):
        raise ValueError("Native tabular configuration is unavailable")


def weight_name():
    from tabpfn.model_loading import ModelSource

    # The same trained graph implements both tasks. Explicit variants remain separate
    # artifacts and are never combined into a single installation.
    return ModelSource.get_v3_5().default_filename


def native_metadata(metadata):
    from tabpfn.architectures.tabpfn_v3_5 import TabPFNV3p5Config, parse_config
    from tabpfn.inference_config import InferenceConfig

    if (
        set(metadata) != {"architecture_name", "config", "inference_config"}
        or metadata["architecture_name"] != "tabpfn_v3_5"
    ):
        raise ValueError("Native tabular checkpoint declaration is unsupported")
    declaration = metadata["config"]
    if not isinstance(declaration, dict):
        raise ValueError("Native tabular geometry is invalid")
    defaults = asdict(TabPFNV3p5Config(max_num_classes=160, num_buckets=5000))
    field_types = get_type_hints(TabPFNV3p5Config)
    # These legacy declarations describe fixed behavior in the pinned 3.5 class.
    fixed = {
        "feat_agg_use_softmax_scaling": False,
        "icl_bf16": False,
        "inference_row_chunk_size": 2048,
        "layernorm_elementwise_affine": True,
        "use_fourier_cell_embedding": True,
        "use_mlp_heads": True,
        "use_nan_indicators": True,
        "use_qk_norm": True,
        "use_rope": True,
        "y_encoder_layernorm": True,
    }
    if set(declaration) - (defaults.keys() | fixed.keys()):
        raise ValueError("Unknown native tabular geometry")
    for key, value in declaration.items():
        if key in fixed:
            if type(value) is not type(fixed[key]) or value != fixed[key]:
                raise ValueError("Unsupported legacy tabular behavior")
            continue
        default = defaults[key]
        if key in {"icl_num_kv_heads", "icl_num_kv_heads_test"}:
            valid = value is None or type(value) is int and 1 <= value <= 16
        elif default is None:
            valid = value is None
        elif type(default) is bool:
            valid = type(value) is bool
        elif isinstance(default, (int, float)):
            valid = type(value) in (int, float) and math.isfinite(value) and 1 <= value <= default
            if field_types[key] is int:
                valid = valid and type(value) is int
        else:
            valid = value == default
        if not valid:
            raise ValueError("Native tabular dimensions exceed bounds")
    config, unused = parse_config(declaration)
    if set(unused) - fixed.keys():
        raise ValueError("Unbound native tabular configuration")
    inference = metadata["inference_config"]
    if not isinstance(inference, dict) or len(json.dumps(inference)) > 32768:
        raise ValueError("Native tabular preprocessing is invalid")
    transforms = inference.get("PREPROCESS_TRANSFORMS")
    if not isinstance(transforms, list) or len(transforms) != 1:
        raise ValueError("Native tabular preprocessing graph is unsupported")
    transform = transforms[0]
    if transform != {
        "append_original": False,
        "categorical_name": "ordinal_shuffled",
        "differentiable": False,
        "global_transformer_name": None,
        "max_features_per_estimator": 768,
        "max_onehot_cardinality": None,
        "name": "none",
    }:
        raise ValueError("Native tabular preprocessing graph is unsupported")
    if (
        type(inference.get("N_ESTIMATORS")) is not int
        or not 1 <= inference["N_ESTIMATORS"] <= 8
        or inference.get("POLYNOMIAL_FEATURES") != "no"
        or inference.get("SUBSAMPLE_SAMPLES") is not None
        or inference.get("REGRESSION_Y_PREPROCESS_TRANSFORMS") != [None, "safepower"]
    ):
        raise ValueError("Native tabular inference exceeds the reviewed behavior")
    settings = InferenceConfig(**inference)
    return config, settings


def load(root, config, backend):
    if config.get("model_type") == "tabpfn_mix":
        from oryn_engine.tabpfnmix_tasks import load as load_mix

        validate_config(config)
        return load_mix(root, config, backend)
    if config.get("model_type") == "mitra_tab2d":
        from oryn_engine.mitra_tasks import load as load_mitra

        validate_config(config)
        return load_mitra(root, config, backend)
    from safetensors import safe_open
    from safetensors.torch import load_file
    from tabpfn.architectures.tabpfn_v3_5 import get_architecture
    from tabpfn.base import ModelSpecs

    from oryn_engine.model_metadata import preflight_safetensors

    validate_config(config)
    path = root / weight_name()
    if (
        path.is_symlink()
        or not path.is_file()
        or not 1 <= path.stat().st_size <= 1024**3
        or list(root.glob("*.safetensors")) != [path]
    ):
        raise ValueError("A single original safe tabular graph is required")
    shapes = preflight_safetensors(root)
    if len(shapes) > 1000 or sum(math.prod(shape) for shape in shapes.values()) > 250_000_000:
        raise ValueError("Native tabular tensor graph exceeds bounds")
    with safe_open(path, framework="pt", device="cpu") as source:
        raw = source.metadata() or {}
    if sum(len(value) for value in raw.values()) > 65536:
        raise ValueError("Native tabular checkpoint metadata exceeds bounds")
    architecture, inference = native_metadata(
        {key: json.loads(value) for key, value in raw.items()}
    )
    model = get_architecture(architecture)
    expected = model.state_dict()
    if shapes != {key: tuple(value.shape) for key, value in expected.items()}:
        raise ValueError("Original tabular tensor graph is incomplete or incompatible")
    model.load_state_dict(load_file(path, device="cpu"), strict=True)
    model = model.to(backend).eval()
    return ModelSpecs(model=model, architecture_config=architecture, inference_config=inference)


def execute(spec, task, value, backend):
    from tabpfn.base import ModelSpecs

    if not isinstance(spec, ModelSpecs):
        from oryn_engine.tabpfnmix_tasks import native_class

        FoundationTransformer = native_class()

        from oryn_engine.mitra_tasks import execute as execute_mitra
        from oryn_engine.tabpfnmix_tasks import execute as execute_mix

        if isinstance(spec, FoundationTransformer):
            return execute_mix(spec, task, value, backend)
        return execute_mitra(spec, task, value, backend)
    import numpy as np
    import pandas as pd
    import torch
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    columns = value["columns"]
    names = [column["name"] for column in columns]
    categorical = [i for i, column in enumerate(columns) if column["type"] == "categorical"]
    training = pd.DataFrame(value["trainingRows"], columns=names)
    query = pd.DataFrame(value["queryRows"], columns=names)
    for column in columns:
        if column["type"] == "numeric":
            training[column["name"]] = training[column["name"]].astype(float)
            query[column["name"]] = query[column["name"]].astype(float)
        else:
            # Train-only category vocabulary: unseen query categories remain missing,
            # with no query rows or targets leaking into the fitting population.
            dtype = pd.CategoricalDtype(
                categories=sorted(training[column["name"]].dropna().unique())
            )
            training[column["name"]] = training[column["name"]].astype(dtype)
            query[column["name"]] = query[column["name"]].astype(dtype)
    classifier = task == "tabular-classification"
    estimator = (TabPFNClassifier if classifier else TabPFNRegressor)(
        model_path=spec,
        device=backend,
        categorical_features_indices=categorical,
        n_estimators=spec.inference_config.N_ESTIMATORS,
        random_state=0,
        n_preprocessing_jobs=1,
        inference_precision=torch.float32,
        enable_torch_compile=False,
        show_progress_bar=False,
    )
    estimator.fit(training, np.asarray(value["targets"]))
    if classifier:
        probabilities = estimator.predict_proba(query)
        classes = [str(item) for item in estimator.classes_]
        if (
            probabilities.shape != (len(query), len(classes))
            or not np.isfinite(probabilities).all()
        ):
            raise ValueError("Native tabular class output is invalid")
        return {
            "classes": classes,
            "predictions": [
                {"label": classes[int(row.argmax())], "probabilities": row.tolist()}
                for row in probabilities
            ],
        }
    predictions = estimator.predict(query)
    if predictions.shape != (len(query),) or not np.isfinite(predictions).all():
        raise ValueError("Native tabular regression output is invalid")
    return {"predictions": predictions.tolist()}
