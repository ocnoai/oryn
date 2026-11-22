"""Declarative classifier semantics for standard ONNX graphs, without repository code.

An explicit profile binds pixels, tensor names, labels and probability semantics.
Bare exports may acquire a profile only from their complete content identity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from oryn_engine.onnx_graph import safe_path

MODEL_TYPE = "oryn-onnx-image-classifier"
ARCHITECTURE = "efficientnet-lite4"
PROFILE = "imagenet-efficientnet-lite4-v1"
LABELS_SHA256 = "3e232637a09beeb733f409908cfc0ba256e83f06815684678abafd4c0990c757"
# Original ONNX Model Zoo graph, also published unchanged by inference4j.
CONTENT_PROFILES = {
    ("d111689907c06eea7c82e4833ddef758da6453b9d4cf60b7e99ca05c7cbd9c12", 51_946_641): PROFILE,
}


def supports(config):
    return config.get("model_type") == MODEL_TYPE


def configuration(config):
    if (
        not isinstance(config, dict)
        or set(config) != {"model_type", "profile", "graph"}
        or config["model_type"] != MODEL_TYPE
        or config["profile"] != PROFILE
        or not isinstance(config["graph"], str)
        or not safe_path(config["graph"])
        or not config["graph"].endswith(".onnx")
    ):
        raise ValueError("ONNX_CLASSIFIER_CONFIGURATION_UNSUPPORTED")
    return config["graph"]


def resolve(config, files):
    """Bind missing metadata using (name, size, SHA256) inventory records."""
    if config:
        return config
    matches = [
        (name, CONTENT_PROFILES[(digest, size)])
        for name, size, digest in files
        if name.endswith(".onnx") and (digest, size) in CONTENT_PROFILES
    ]
    if len(matches) > 1:
        raise ValueError("ONNX_CLASSIFIER_GRAPH_AMBIGUOUS")
    if not matches:
        return config
    name, profile = matches[0]
    result = {"model_type": MODEL_TYPE, "profile": profile, "graph": name}
    configuration(result)
    return result


def resolve_local(config, root, files):
    if config:
        return config
    sizes = {size for _, size in CONTENT_PROFILES}
    records = []
    for path in files:
        size = path.stat().st_size
        if path.suffix != ".onnx" or size not in sizes:
            continue
        # A remote blob identifier is insufficient at execution time. Verify
        # the entire installed graph in bounded chunks before adding semantics.
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append((path.relative_to(root).as_posix(), size, digest))
    return resolve(config, records)


def labels():
    raw = (Path(__file__).parent / "data/imagenet-efficientnet-lite4-labels.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != LABELS_SHA256:
        raise ValueError("ONNX_CLASSIFIER_LABELS_CHANGED")
    mapping = json.loads(raw)
    return [mapping[str(i)] for i in range(1000)]


def verify_preprocessor():
    from importlib.metadata import version

    import cv2

    from oryn_engine.generated_capabilities import generated_artifact

    inputs = generated_artifact()["buildInputs"]
    binary = Path(cv2.__file__).parent / inputs["opencvBinaryName"]
    with binary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if (
        version("opencv-python-headless") != inputs["depthDependencies"]["opencv-python-headless"]
        or digest != inputs["opencvBinarySha256"]
    ):
        raise ValueError("ONNX_CLASSIFIER_PREPROCESSOR_CHANGED")


def validate_graph(graph):
    if graph["inputs"] != [{"name": "images:0", "dtype": 1, "shape": [1, 224, 224, 3]}]:
        raise ValueError("ONNX_INPUT_BINDING_UNSUPPORTED")
    if graph["outputs"] != [{"name": "Softmax:0", "dtype": 1, "shape": [1, 1000]}]:
        raise ValueError("ONNX_OUTPUT_BINDING_UNSUPPORTED")
    return "Softmax:0"


def pixels(path):
    import cv2
    import numpy as np
    from PIL import Image

    if (
        not isinstance(path, Path)
        or path.is_symlink()
        or not path.is_file()
        or not 0 < path.stat().st_size <= 20 * 1024**2
    ):
        raise ValueError("ONNX_IMAGE_INPUT_INVALID")
    # Bound decoding before OpenCV allocates pixels. OpenCV also preserves the
    # publisher's RGB conversion, JPEG decoder and EXIF orientation behavior.
    with Image.open(path) as image:
        width, height = image.size
        if (
            image.format not in {"PNG", "JPEG", "WEBP"}
            or getattr(image, "n_frames", 1) != 1
            or not 1 <= width * height <= 16_777_216
            or max(width, height) / min(width, height) > 64
        ):
            raise ValueError("ONNX_IMAGE_INPUT_INVALID")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("ONNX_IMAGE_INPUT_INVALID")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    if width * height > 16_777_216:
        raise ValueError("ONNX_IMAGE_INPUT_INVALID")
    resized = (
        (256, int(256 * height / width)) if height > width else (int(256 * width / height), 256)
    )
    image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
    height, width, _ = image.shape
    image = image[
        int((height - 224) / 2) : int((height + 224) / 2),
        int((width - 224) / 2) : int((width + 224) / 2),
    ].astype(np.float32)
    image -= np.float32(127)
    image /= np.float32(128)
    return image[None]


def execute(session, value, input_path, event):
    import numpy as np

    image = pixels(input_path)
    event("job.progress", {"stage": "running-model", "progress": 0.3})
    output = session.run(["Softmax:0"], {"images:0": image})[0]
    if (
        output.shape != (1, 1000)
        or not np.isfinite(output).all()
        or (output < 0).any()
        or (output > 1).any()
        or not np.isclose(output.sum(dtype=np.float64), 1, rtol=1e-5, atol=1e-6)
    ):
        raise ValueError("ONNX_CLASSIFIER_OUTPUT_INVALID")
    probabilities = output[0]
    indices = np.argsort(-probabilities, kind="stable")[: value.get("topK", 5)]
    names = labels()
    return {
        "predictions": [
            {"label": names[i], "labelId": int(i), "score": float(probabilities[i])}
            for i in indices
        ]
    }
