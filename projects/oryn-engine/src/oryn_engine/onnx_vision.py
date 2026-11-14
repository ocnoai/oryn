"""Closed vision-tagging semantics over the existing standard-operator ONNX runtime."""

from __future__ import annotations

import json
import re
from pathlib import Path

from oryn_engine.model_metadata import validate_metadata

ARCHITECTURE = "camie-tagger"
MAX_LABEL_BYTES = 12 * 1024**2
MAX_LABELS = 100_000


def configuration(config):
    required = {"model_name", "model_type", "models", "default_model"}
    if (
        not isinstance(config, dict)
        or not required <= config.keys()
        or not config.keys() <= required | {"description", "version", "tags", "license"}
        or config["model_name"] != ARCHITECTURE
        or config["model_type"] != "pytorch"
        or not isinstance(config["models"], list)
        or not 1 <= len(config["models"]) <= 8
    ):
        raise ValueError("ONNX_VISION_CONFIGURATION_UNSUPPORTED")
    validate_metadata(config)
    entries = config["models"]
    for item in entries:
        if (
            not isinstance(item, dict)
            or not {"name", "path"} <= item.keys()
            or not item.keys() <= {"name", "path", "description"}
            or not isinstance(item["name"], str)
            or not isinstance(item["path"], str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}\.(onnx|safetensors)", item["path"])
        ):
            raise ValueError("ONNX_VISION_CONFIGURATION_UNSUPPORTED")
    matches = [x for x in entries if x["name"] == config["default_model"]]
    if len(matches) != 1 or not matches[0]["path"].endswith(".onnx"):
        raise ValueError("ONNX_VISION_GRAPH_AMBIGUOUS")
    graph = matches[0]["path"]
    return graph, graph[:-5] + "-metadata.json"


def parse_labels(raw):
    """Large flat vocabularies have their own limits, never generic config exemptions."""
    if len(raw) > MAX_LABEL_BYTES:
        raise ValueError("ONNX_VISION_METADATA_LIMIT")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("ONNX_VISION_METADATA_DUPLICATE")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique)
        dataset = value["dataset_info"]
        mapping = dataset["tag_mapping"]
        # The remainder keeps the ordinary recursive JSON limits, including
        # depth, numbers and strings. Only these three flat maps may be large.
        validate_metadata(value | {"dataset_info": dataset | {"tag_mapping": {}}})
        count = dataset["total_tags"]
        labels, inverse, categories = (
            mapping[k] for k in ("idx_to_tag", "tag_to_idx", "tag_to_category")
        )
        if (
            type(count) is not int
            or not 1 <= count <= MAX_LABELS
            or set(mapping) != {"idx_to_tag", "tag_to_idx", "tag_to_category"}
            or not all(
                isinstance(x, dict) and len(x) == count for x in (labels, inverse, categories)
            )
            or set(labels) != {str(i) for i in range(count)}
        ):
            raise ValueError("ONNX_VISION_LABELS_INVALID")
        ordered = [labels[str(i)] for i in range(count)]
        declared_categories = dataset["categories"]
        if (
            not isinstance(declared_categories, list)
            or not 1 <= len(declared_categories) <= 32
            or any(not isinstance(x, str) or not 1 <= len(x) <= 64 for x in declared_categories)
            or any(not isinstance(x, str) or not 1 <= len(x) <= 256 for x in ordered)
            or len(set(ordered)) != count
            or set(inverse) != set(ordered)
            or set(categories) != set(ordered)
            or any(type(inverse[x]) is not int or inverse[x] != i for i, x in enumerate(ordered))
            or any(
                not isinstance(x, str) or x not in declared_categories for x in categories.values()
            )
        ):
            raise ValueError("ONNX_VISION_LABELS_INVALID")
        info = value["model_info"]
        size, context = info["img_size"], info["tag_context_size"]
        if (
            info["architecture"] != "ImageTagger with Vision Transformer"
            or type(size) is not int
            or not 16 <= size <= 1024
            or type(context) is not int
            or not 1 <= context <= min(count, 1024)
            or value["input_spec"]["shape"] != [3, size, size]
            or value["input_spec"]["dtype"] != "float32"
            or value["input_spec"].get("format") != "RGB image tensor"
            or value["input_spec"].get("normalization")
            != "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
            or value["usage"].get("postprocessing") != "Apply sigmoid to get probabilities"
        ):
            raise ValueError("ONNX_VISION_METADATA_INVALID")
        for name, width, dtype in (
            ("initial_predictions", count, "float32"),
            ("refined_predictions", count, "float32"),
            ("selected_candidates", context, "int64"),
        ):
            spec = value["output_spec"][name]
            if spec["shape"] != ["batch_size", width] or spec["dtype"] != dtype:
                raise ValueError("ONNX_VISION_METADATA_INVALID")
        return {"labels": ordered, "size": size, "context": context}
    except (KeyError, TypeError, RecursionError, OverflowError, json.JSONDecodeError) as exc:
        raise ValueError("ONNX_VISION_METADATA_INVALID") from exc


def local_labels(root, config):
    _, name = configuration(config)
    path = root / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_LABEL_BYTES:
        raise ValueError("ONNX_VISION_METADATA_UNAVAILABLE")
    with path.open("rb") as stream:
        return parse_labels(stream.read(MAX_LABEL_BYTES + 1))


def validate_graph(graph, metadata):
    if not isinstance(metadata, dict):
        raise ValueError("ONNX_VISION_METADATA_UNAVAILABLE")
    size, count, context = metadata["size"], len(metadata["labels"]), metadata["context"]
    inputs, outputs = graph["inputs"], graph["outputs"]

    def matches(item, name, dtype, dimensions):
        shape = item["shape"]
        return (
            item["name"] == name
            and item["dtype"] == dtype
            and len(shape) == len(dimensions) + 1
            and (shape[0] == 1 or isinstance(shape[0], str))
            and shape[1:] == dimensions
        )

    if len(inputs) != 1 or not matches(inputs[0], "input", 1, [3, size, size]):
        raise ValueError("ONNX_INPUT_BINDING_UNSUPPORTED")
    expected = {
        "initial_predictions": (1, count),
        "refined_predictions": (1, count),
        "selected_candidates": (7, context),
    }
    if (
        len(outputs) != 3
        or {x["name"] for x in outputs} != set(expected)
        or any(
            not matches(x, x["name"], expected[x["name"]][0], [expected[x["name"]][1]])
            for x in outputs
        )
    ):
        raise ValueError("ONNX_OUTPUT_BINDING_UNSUPPORTED")
    return "refined_predictions"


def pixels(path, size):
    import numpy as np
    from PIL import Image

    if (
        not isinstance(path, Path)
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 20 * 1024**2
    ):
        raise ValueError("ONNX_IMAGE_INPUT_INVALID")
    with Image.open(path) as image:
        width, height = image.size
        if image.format not in {"PNG", "JPEG", "WEBP"} or not 1 <= width * height <= 16_777_216:
            raise ValueError("ONNX_IMAGE_INPUT_INVALID")
        ratio = width / height
        resized = (size, int(size / ratio)) if ratio > 1 else (int(size * ratio), size)
        if min(resized) < 1:
            raise ValueError("ONNX_IMAGE_ASPECT_UNSUPPORTED")
        # RGB conversion is explicit also for grayscale; the publisher canvas
        # copies grayscale pixels into all three RGB channels equivalently.
        image = image.convert("RGB").resize(resized, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (124, 116, 104))
        canvas.paste(image, ((size - resized[0]) // 2, (size - resized[1]) // 2))
        value = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    return (
        (value - np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None])
        / np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    )[None]


def execute(session, metadata, value, input_path, event):
    import numpy as np

    image = pixels(input_path, metadata["size"])
    event("job.progress", {"stage": "running-model", "progress": 0.3})
    logits = session.run(["refined_predictions"], {"input": image})[0]
    if logits.shape != (1, len(metadata["labels"])) or not np.isfinite(logits).all():
        raise ValueError("ONNX_VISION_OUTPUT_INVALID")
    # Independent tag probabilities, never a softmax across unrelated labels.
    x = logits[0].astype(np.float64)
    probabilities = np.exp(-np.logaddexp(0, -x))
    indices = np.argsort(-probabilities, kind="stable")[: value.get("topK", 5)]
    return {
        "predictions": [
            {"label": metadata["labels"][i], "labelId": int(i), "score": float(probabilities[i])}
            for i in indices
        ]
    }
