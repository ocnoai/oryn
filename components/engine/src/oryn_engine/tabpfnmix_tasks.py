"""Bounded in-context inference with the packaged native TabPFN-Mix graph."""

from __future__ import annotations

import math

FIELDS = {
    "n_features",
    "n_classes",
    "dim",
    "n_layers",
    "n_heads",
    "attn_dropout",
    "y_as_float_embedding",
    "task",
}


def normalized_config(config, task):
    if not config.keys() >= FIELDS:
        return config
    expected = {
        "tabular-classification": "classification",
        "tabular-regression": "regression",
    }.get(task)
    if expected is None:
        return config
    if (
        set(config) - (FIELDS | {"model_type", "architectures"})
        or config["task"] != expected
        or config.get("model_type", "tabpfn_mix") != "tabpfn_mix"
        or config.get("architectures", ["FoundationTransformer"]) != ["FoundationTransformer"]
        or type(config["attn_dropout"]) not in (int, float)
        or config["attn_dropout"] != 0
        or type(config["y_as_float_embedding"]) is not bool
        or expected == "regression"
        and not config["y_as_float_embedding"]
    ):
        raise ValueError("Native TabPFN-Mix declaration is unsupported")
    for key, maximum in {
        "n_features": 100,
        "n_classes": 10,
        "dim": 512,
        "n_layers": 12,
        "n_heads": 8,
    }.items():
        if type(config[key]) is not int or not 1 <= config[key] <= maximum:
            raise ValueError("Native TabPFN-Mix dimensions exceed bounds")
    if config["dim"] % config["n_heads"] or config["n_classes"] < 2:
        raise ValueError("Native TabPFN-Mix attention or class geometry is invalid")
    return {
        **config,
        "model_type": "tabpfn_mix",
        "architectures": ["FoundationTransformer"],
    }


def native_class():
    from autogluon.tabular.models.tabpfnmix._internal.models.foundation import (
        foundation_transformer,
    )

    return foundation_transformer.FoundationTransformer


def load(root, config, backend):
    FoundationTransformer = native_class()
    from safetensors.torch import load_file

    from oryn_engine.model_metadata import preflight_safetensors

    path = root / "model.safetensors"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 512 * 1024**2:
        raise ValueError("A safe native TabPFN-Mix graph is required")
    if list(root.glob("*.safetensors")) != [path]:
        raise ValueError("Competing TabPFN-Mix tensor graphs are unsupported")
    shapes = preflight_safetensors(root)
    if len(shapes) > 1000 or sum(math.prod(s) for s in shapes.values()) > 80_000_000:
        raise ValueError("Native TabPFN-Mix tensor graph exceeds bounds")
    model = FoundationTransformer(**{key: config[key] for key in FIELDS})
    if shapes != {key: tuple(value.shape) for key, value in model.state_dict().items()}:
        raise ValueError("Original TabPFN-Mix tensor graph is incomplete or incompatible")
    model.load_state_dict(load_file(path, device="cpu"), strict=True)
    return model.to(backend).eval()


def execute(model, task, value, backend):
    import numpy as np
    import torch
    from autogluon.tabular.models.tabpfnmix._internal.core.y_transformer import (
        create_y_transformer,
    )
    from autogluon.tabular.models.tabpfnmix._internal.data.preprocessor import (
        Preprocessor,
    )

    classification = task == "tabular-classification"
    expected_task = "classification" if classification else "regression"
    if model.task != expected_task:
        raise ValueError("Native TabPFN-Mix task does not match the selected graph")
    training = np.empty((len(value["trainingRows"]), len(value["columns"])), dtype=np.float32)
    query = np.empty((len(value["queryRows"]), len(value["columns"])), dtype=np.float32)
    for index, column in enumerate(value["columns"]):
        train_values = [row[index] for row in value["trainingRows"]]
        query_values = [row[index] for row in value["queryRows"]]
        if column["type"] == "categorical":
            vocabulary = {
                v: i for i, v in enumerate(sorted({v for v in train_values if v is not None}))
            }
            train_values = [vocabulary.get(v, float("nan")) for v in train_values]
            query_values = [vocabulary.get(v, float("nan")) for v in query_values]
        training[:, index] = train_values
        query[:, index] = query_values
    if np.isinf(training).any() or np.isinf(query).any():
        raise ValueError("Native TabPFN-Mix features exceed numeric bounds")
    # Impute before the native transform removes constant columns, preserving
    # original column alignment. Every statistic comes only from training rows.
    for index in range(training.shape[1]):
        observed = training[:, index][~np.isnan(training[:, index])]
        mean = float(observed.mean(dtype=np.float64)) if observed.size else 0.0
        training[np.isnan(training[:, index]), index] = mean
        query[np.isnan(query[:, index]), index] = mean
    if not any(np.unique(column).size > 1 for column in training.T):
        raise ValueError("Native TabPFN-Mix requires a varying training feature")
    classes = sorted(set(value["targets"])) if classification else []
    if classification and len(classes) > model.n_classes:
        raise ValueError("Native TabPFN-Mix class count exceeds the selected graph")
    targets = np.asarray(
        [classes.index(v) for v in value["targets"]] if classification else value["targets"]
    )
    if not np.isfinite(targets).all():
        raise ValueError("Native TabPFN-Mix targets exceed numeric bounds")
    preprocessing = Preprocessor(
        max_features=model.n_features,
        use_quantile_transformer=True,
        use_feature_count_scaling=True,
        task=expected_task,
    )
    preprocessing.fit(training.copy(), targets.copy())
    target_transform = create_y_transformer(targets.copy(), expected_task)
    arrays = [
        preprocessing.transform(training.copy()),
        target_transform.transform(targets.copy()),
        preprocessing.transform(query.copy()),
    ]
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Native TabPFN-Mix preprocessing is nonfinite")
    support = torch.from_numpy(arrays[0]).float().unsqueeze(0).to(backend)
    labels = torch.from_numpy(arrays[1])
    labels = labels.float() if not classification or model.y_as_float_embedding else labels.long()
    if not bool(torch.isfinite(labels).all()):
        raise ValueError("Native TabPFN-Mix target encoding is nonfinite")
    labels = labels.unsqueeze(0).to(backend)
    queries = torch.from_numpy(arrays[2]).float().unsqueeze(0).to(backend)
    with torch.inference_mode():
        predictions = model(support, labels, queries)[0].float().cpu()
    if not bool(torch.isfinite(predictions).all()):
        raise ValueError("Native TabPFN-Mix output is not finite")
    if classification:
        probabilities = predictions[:, : len(classes)].softmax(dim=-1).numpy()
        return {
            "classes": classes,
            "predictions": [
                {"label": classes[int(row.argmax())], "probabilities": row.tolist()}
                for row in probabilities
            ],
        }
    values = target_transform.inverse_transform(predictions[:, 0].numpy())
    if not np.isfinite(values).all():
        raise ValueError("Native TabPFN-Mix regression is nonfinite")
    return {"predictions": values.tolist()}
