"""Exact native Tab2D graphs with bounded, train-only preprocessing."""

from __future__ import annotations

import math

FIELDS = {"dim", "dim_output", "n_heads", "n_layers", "task"}


def normalized_config(config, task):
    if not config.keys() >= FIELDS:
        return config
    expected = {
        "tabular-classification": "CLASSIFICATION",
        "tabular-regression": "REGRESSION",
    }.get(task)
    if expected is None:
        return config
    if (
        set(config) - (FIELDS | {"model_type", "architectures"})
        or config["task"] != expected
        or config.get("model_type", "mitra_tab2d") != "mitra_tab2d"
        or config.get("architectures", ["Tab2D"]) != ["Tab2D"]
    ):
        raise ValueError("Native Mitra declaration is unsupported")
    for key, maximum in {"dim": 512, "dim_output": 1000, "n_heads": 4, "n_layers": 12}.items():
        if type(config[key]) is not int or not 1 <= config[key] <= maximum:
            raise ValueError("Native Mitra dimensions exceed bounds")
    if (
        config["dim"] % config["n_heads"]
        or (expected == "REGRESSION" and config["dim_output"] not in {1, 1000})
        or (expected == "CLASSIFICATION" and not 2 <= config["dim_output"] <= 10)
    ):
        raise ValueError("Native Mitra output or attention geometry is invalid")
    return {**config, "model_type": "mitra_tab2d", "architectures": ["Tab2D"]}


def load(root, config, backend):
    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D
    from safetensors.torch import load_file

    from oryn_engine.model_metadata import preflight_safetensors

    path = root / "model.safetensors"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 512 * 1024**2:
        raise ValueError("A safe native Mitra graph is required")
    if list(root.glob("*.safetensors")) != [path]:
        raise ValueError("Competing Mitra tensor graphs are unsupported")
    shapes = preflight_safetensors(root)
    if len(shapes) > 1000 or sum(math.prod(s) for s in shapes.values()) > 80_000_000:
        raise ValueError("Native Mitra tensor graph exceeds bounds")
    model = Tab2D(
        **{k: config[k] for k in FIELDS},
        use_pretrained_weights=False,
        path_to_weights="",
        device=backend,
    )
    if shapes != {k: tuple(v.shape) for k, v in model.state_dict().items()}:
        raise ValueError("Original Mitra tensor graph is incomplete or incompatible")
    model.load_state_dict(load_file(path, device="cpu"), strict=True)
    return model.to(backend).eval()


def execute(model, task, value, backend):
    import numpy as np
    import torch
    from autogluon.tabular.models.mitra._internal.config.enums import Task
    from autogluon.tabular.models.mitra._internal.data.preprocessor import Preprocessor

    classification = task == "tabular-classification"
    expected_task = Task.CLASSIFICATION if classification else Task.REGRESSION
    if model.task != expected_task:
        raise ValueError("Native Mitra task does not match the selected graph")
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
    classes = sorted(set(value["targets"])) if classification else []
    if classification and len(classes) > model.dim_output:
        raise ValueError("Native Mitra class count exceeds the selected graph")
    targets = np.asarray(
        [classes.index(v) for v in value["targets"]] if classification else value["targets"]
    )
    preprocessing = Preprocessor(
        dim_embedding=None,
        n_classes=len(classes),
        dim_output=model.dim_output,
        use_quantile_transformer=False,
        use_feature_count_scaling=False,
        use_random_transforms=False,
        shuffle_classes=False,
        shuffle_features=False,
        random_mirror_regression=False,
        random_mirror_x=False,
        task=expected_task,
    )
    preprocessing.fit(training.copy(), targets.copy())
    support = torch.from_numpy(preprocessing.transform_X(training.copy())).unsqueeze(0).to(backend)
    queries = torch.from_numpy(preprocessing.transform_X(query.copy())).unsqueeze(0).to(backend)
    labels = torch.from_numpy(preprocessing.transform_y(targets.copy())).unsqueeze(0).to(backend)
    bins = None
    if not classification and model.dim_output == 1000:
        # Native Mitra's cross-entropy regression head uses fixed normalized bins,
        # followed by the most likely bin's midpoint and inverse target scaling.
        bins = torch.linspace(-0.5, 1.5, model.dim_output + 1, device=backend)
        labels = (torch.bucketize(labels, bins) - 1).clamp(0, model.dim_output - 1).long()

    def mask(count):
        return torch.zeros((1, count), dtype=torch.bool, device=backend)

    with torch.inference_mode():
        predictions = (
            model(
                support,
                labels,
                queries,
                mask(support.shape[2]),
                mask(support.shape[1]),
                mask(queries.shape[1]),
            )[0]
            .float()
            .cpu()
            .numpy()
        )
    if bins is not None:
        indices = torch.from_numpy(predictions.argmax(axis=-1)).to(backend)
        predictions = (bins[indices] + (bins[1] - bins[0]) / 2).cpu().numpy()
    predictions = preprocessing.inverse_transform_y(predictions)
    if not np.isfinite(predictions).all():
        raise ValueError("Native Mitra output is not finite")
    if classification:
        probabilities = torch.from_numpy(predictions[:, : len(classes)]).softmax(dim=-1).numpy()
        return {
            "classes": classes,
            "predictions": [
                {"label": classes[int(row.argmax())], "probabilities": row.tolist()}
                for row in probabilities
            ],
        }
    return {"predictions": predictions.tolist()}
