import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from oryn_engine import onnx_classifier as binding
from oryn_engine import onnx_tasks

FIXTURES = Path(__file__).parent / "fixtures/onnx_classifier"
ARTIFACT = "artifact_" + "c" * 24


def document(name):
    return json.loads((FIXTURES / name).read_text())


def configuration(name="model.onnx"):
    return {"model_type": binding.MODEL_TYPE, "profile": binding.PROFILE, "graph": name}


def image(path, mode="RGB", width=47, height=31, orientation=None):
    channels = {"RGB": 3, "RGBA": 4, "L": 1}[mode]
    pixels = ((np.arange(width * height * channels) * 41 + 19) % 256).astype(np.uint8)
    pixels = pixels.reshape((height, width, channels) if channels > 1 else (height, width))
    value = Image.fromarray(pixels)
    options = {}
    if orientation:
        exif = Image.Exif()
        exif[274] = orientation
        options["exif"] = exif
    value.save(path, **options)
    return path


def publisher_pixels(path):
    """Independent calculation from the pinned ONNX Model Zoo example."""
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    shorter = int(100.0 * 224 / 87.5)
    dimensions = (shorter, int(shorter * h / w)) if h > w else (int(shorter * w / h), shorter)
    img = cv2.resize(img, dimensions, interpolation=cv2.INTER_LINEAR)
    h, w, _ = img.shape
    img = img[int((h - 224) / 2) : int((h + 224) / 2), int((w - 224) / 2) : int((w + 224) / 2)]
    img = np.asarray(img, dtype="float32")
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return np.expand_dims(img, axis=0)


@pytest.mark.parametrize(
    "mode,width,height,suffix,orientation",
    [
        ("RGB", 47, 31, ".png", None),
        ("RGB", 31, 47, ".png", None),
        ("RGBA", 17, 17, ".png", None),
        ("L", 19, 33, ".png", None),
        ("RGB", 79, 39, ".jpg", 6),
        ("RGB", 45, 33, ".webp", None),
    ],
)
def test_preprocessing_matches_publisher_rgb_resize_crop_normalization(
    tmp_path, mode, width, height, suffix, orientation
):
    path = image(tmp_path / ("input" + suffix), mode, width, height, orientation)
    actual = binding.pixels(path)
    assert actual.shape == (1, 224, 224, 3) and actual.dtype == np.float32
    np.testing.assert_array_equal(actual, publisher_pixels(path))
    assert actual.min() >= -127 / 128 and actual.max() <= 1


@pytest.mark.parametrize("damage", ["profile", "type", "extra", "path", "absolute", "suffix"])
def test_declarative_configuration_rejects_unknown_semantics(damage):
    config = configuration()
    if damage == "profile":
        config["profile"] = "raw-rgb"
    elif damage == "type":
        config["model_type"] = "efficientnet"
    elif damage == "extra":
        config["auto_map"] = {"AutoModel": "model.py"}
    else:
        config["graph"] = {
            "path": "../model.onnx",
            "absolute": "/model.onnx",
            "suffix": "model.pkl",
        }[damage]
    with pytest.raises(ValueError):
        onnx_tasks.validate_config(config, "image-classification")


def test_original_graph_identity_and_owned_labels_match_upstream():
    original = document("original-inventory.json")
    graph = next(x for x in original["files"] if x["name"].endswith(".onnx"))
    assert (graph["lfsSha"], graph["size"]) in binding.CONTENT_PROFILES
    upstream = document("upstream-inventory.json")
    upstream_graph = next(x for x in upstream["files"] if x["name"].endswith(".onnx"))
    assert (upstream_graph["lfsSha"], upstream_graph["size"]) == (graph["lfsSha"], graph["size"])
    assert binding.resolve(
        {}, [("renamed/export.onnx", graph["size"], graph["lfsSha"])]
    ) == configuration("renamed/export.onnx")
    assert (
        onnx_tasks.validate_graph(
            document("original-graph.json"), "image-classification", configuration()
        )
        == "Softmax:0"
    )
    labels = binding.labels()
    assert len(labels) == 1000
    assert labels[0] == "tench, Tinca tinca"
    assert labels[999] == "toilet tissue, toilet paper, bathroom tissue"


@pytest.mark.parametrize("damage", ["digest", "size", "duplicate", "config"])
def test_bare_graph_profile_requires_complete_unique_content_identity(damage):
    (digest, size), _ = next(iter(binding.CONTENT_PROFILES.items()))
    records = [("model.onnx", size, digest)]
    if damage == "digest":
        records = [("model.onnx", size, "a" * 64)]
    elif damage == "size":
        records = [("model.onnx", size - 1, digest)]
    elif damage == "duplicate":
        records += [("second.onnx", size, digest)]
        with pytest.raises(ValueError, match="AMBIGUOUS"):
            binding.resolve({}, records)
        return
    config = {"unknown": True} if damage == "config" else {}
    assert binding.resolve(config, records) == config


def test_installed_bare_graph_is_hashed_in_full_before_binding(tmp_path, monkeypatch):
    content = b"data-only graph identity" * 100_000
    path = tmp_path / "renamed.onnx"
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(binding, "CONTENT_PROFILES", {(digest, len(content)): binding.PROFILE})
    assert binding.resolve_local({}, tmp_path, [path]) == configuration(path.name)
    with path.open("r+b") as stream:
        stream.seek(len(content) - 1)
        stream.write(b"X")
    assert binding.resolve_local({}, tmp_path, [path]) == {}


@pytest.mark.parametrize(
    "damage",
    [
        "input-name",
        "input-type",
        "layout",
        "batch",
        "output-name",
        "classes",
        "extra-output",
        "operator",
    ],
)
def test_graph_binding_rejects_changed_tensor_semantics(damage):
    graph = document("original-graph.json")
    if damage == "input-name":
        graph["inputs"][0]["name"] = "input"
    elif damage == "input-type":
        graph["inputs"][0]["dtype"] = 2
    elif damage == "layout":
        graph["inputs"][0]["shape"] = [1, 3, 224, 224]
    elif damage == "batch":
        graph["inputs"][0]["shape"][0] = "batch"
    elif damage == "output-name":
        graph["outputs"][0]["name"] = "logits"
    elif damage == "classes":
        graph["outputs"][0]["shape"][-1] = 1001
    elif damage == "extra-output":
        graph["outputs"].append(copy.deepcopy(graph["outputs"][0]))
    else:
        graph["operators"].append("UnreviewedCustomOp")
    with pytest.raises(ValueError):
        onnx_tasks.validate_graph(graph, "image-classification", configuration())


@pytest.mark.parametrize(
    "damage", [None, "digest", "size", "missing", "geometry", "explicit", "bad-profile"]
)
def test_discovery_selects_metadata_bound_graph_without_original_weights(tmp_path, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    graph = document("original-graph.json")
    (digest, size), _ = next(iter(binding.CONTENT_PROFILES.items()))
    if damage == "geometry":
        graph["outputs"][0]["shape"][-1] = 999

    class Hub(FakeHub):
        def read_onnx_metadata(self, *args):
            assert args[2] == "model.onnx"
            return graph

    files = [
        ("model.onnx", size - (damage == "size"), "b" * 64 if damage == "digest" else digest),
        ("repository.py", 200, "a" * 64),
    ]
    config = {}
    if damage in {"explicit", "bad-profile"}:
        files.append(("config.json", 120, "c" * 64))
        config = configuration()
        if damage == "bad-profile":
            config["profile"] = "unknown"
    if damage == "missing":
        files = files[1:]
    hub = Hub(pipeline_tag="image-classification", files=files)
    hub.config = config
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "test/classifier", "revision": "main"}
    )
    if damage in {None, "explicit"}:
        assert result["executable"], result["blockingReasons"]
        assert result["technicalCapabilities"]["architectures"] == [binding.ARCHITECTURE]
        assert {x["name"] for x in result["selectedFiles"]} == (
            {"model.onnx", "config.json"} if config else {"model.onnx"}
        )
        assert result["taskProfile"]["contract"]["id"] == "image-classification-v1"
        assert result["executableBackends"] == ["cpu"]
    else:
        assert not result["executable"]


@pytest.fixture(scope="module")
def full_export(tmp_path_factory):
    import timm
    import torch

    root = tmp_path_factory.mktemp("complete-efficientnet-lite4")
    torch.manual_seed(617)
    network = timm.create_model("tf_efficientnet_lite4", pretrained=False).eval()
    assert sum(p.numel() for p in network.parameters()) == 13_006_568

    class Export(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = network

        def forward(self, pixels):
            return self.network(pixels.permute(0, 3, 1, 2)).reshape(1, 1000).softmax(-1)

    model = Export().eval()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            torch.zeros(1, 224, 224, 3),
            str(root / "model.onnx"),
            input_names=["images:0"],
            output_names=["Softmax:0"],
            opset_version=17,
            dynamo=False,
        )
    (root / "config.json").write_text(json.dumps(configuration()))
    yield root, model, copy.deepcopy(model).double()
    # Only generated fixtures: no original checkpoints are fetched by this test.
    for path in root.glob("*.onnx*"):
        path.unlink()


@pytest.mark.parametrize("width,height", [(47, 31), (31, 47)])
def test_complete_generated_efficientnet_matches_native_pytorch(
    full_export, tmp_path, width, height
):
    import torch

    root, model, high_precision = full_export
    path = image(tmp_path / "input.png", width=width, height=height)
    with torch.inference_mode():
        expected = model(torch.from_numpy(publisher_pixels(path))).numpy()[0]
        reference = high_precision(torch.from_numpy(publisher_pixels(path)).double()).numpy()[0]
    session, _, _, output = onnx_tasks.prepare(root, "image-classification", "cpu")
    native = session.run([output], {"images:0": binding.pixels(path)})[0][0]
    # This full 13M-parameter CNN accumulates float32 convolution/fusion rounding.
    # Check both implementations against independent float64 evaluation, with a
    # 0.01% relative probability budget and identical top classes. The float32
    # PyTorch reference itself differs from float64 after the full CNN depth.
    np.testing.assert_allclose(native, reference, rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(expected, reference, rtol=1e-4, atol=1e-8)
    actual = onnx_tasks.execute(
        root,
        "image-classification",
        {"imageArtifactId": ARTIFACT, "topK": 5},
        "cpu",
        lambda *_: None,
        path,
    )
    predictions = actual["predictions"]
    expected_ids = np.argsort(-expected, kind="stable")[:5]
    assert expected_ids.tolist() == np.argsort(-reference, kind="stable")[:5].tolist()
    assert len(predictions) == 5
    np.testing.assert_allclose(
        [x["score"] for x in predictions], reference[expected_ids], rtol=1e-4, atol=1e-8
    )
    assert [x["labelId"] for x in predictions[:5]] == np.argsort(-expected, kind="stable")[
        :5
    ].tolist()
    assert all(x["label"] == binding.labels()[x["labelId"]] for x in predictions)


def test_repeated_offline_worker_complete_graph_results(full_export, tmp_path):
    import os

    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    root, _, _ = full_export
    path = image(tmp_path / "input.png")
    pack = builtin_registry().active_pack("oryn.onnx-runtime")
    request = dict(
        abi=pack.abi,
        packId=pack.pack_id,
        version=pack.version,
        command="generate",
        payload=dict(
            modelRoot=str(root),
            task="image-classification",
            backend="cpu",
            input={"imageArtifactId": ARTIFACT, "topK": 5},
            inputPath=str(path),
            outputPath=None,
        ),
    )
    results = []
    for _ in range(2):
        child = subprocess.run(
            [sys.executable, "-m", "oryn_engine.runtime_packs.onnx_worker"],
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            timeout=60,
            env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        assert child.returncode == 0, child.stderr
        records = [json.loads(line) for line in child.stdout.splitlines()]
        result = next(x for x in records if x["type"] == "result")
        assert result["ok"], (result, child.stderr)
        results.append(result["result"])
    assert results[0] == results[1]
    assert len(results[0]["predictions"]) == 5


@pytest.mark.parametrize(
    "damage", ["nan", "negative", "over-one", "unnormalized", "classes", "batch"]
)
def test_probability_output_is_validated_without_second_softmax(tmp_path, damage):
    output = np.full((1, 1000), 0.001, dtype=np.float32)
    if damage == "classes":
        output = output[:, :-1]
    elif damage == "batch":
        output = output[0]
    else:
        output[0, 0] = {"nan": np.nan, "negative": -0.01, "over-one": 2, "unnormalized": 0.2}[
            damage
        ]
    session = SimpleNamespace(run=lambda *_: [output])
    with pytest.raises(ValueError, match="ONNX_CLASSIFIER_OUTPUT_INVALID"):
        binding.execute(session, {}, image(tmp_path / "input.png"), lambda *_: None)


def test_probabilities_and_stable_ties_keep_original_label_order(tmp_path):
    output = np.zeros((1, 1000), dtype=np.float32)
    output[0, 1:3] = 0.5
    session = SimpleNamespace(run=lambda *_: [output])
    result = binding.execute(session, {"topK": 2}, image(tmp_path / "input.png"), lambda *_: None)
    assert [x["labelId"] for x in result["predictions"]] == [1, 2]
    assert [x["score"] for x in result["predictions"]] == [0.5, 0.5]


@pytest.mark.parametrize("damage", ["symlink", "missing", "format", "animated", "aspect"])
def test_image_input_scope_and_resource_limits(tmp_path, damage):
    path = image(tmp_path / "input.png")
    if damage == "symlink":
        link = tmp_path / "link.png"
        link.symlink_to(path)
        path = link
    elif damage == "missing":
        path.unlink()
    elif damage == "format":
        path = image(tmp_path / "input.bmp")
    elif damage == "animated":
        Image.new("RGB", (3, 3)).save(
            path, save_all=True, append_images=[Image.new("RGB", (3, 3), "red")]
        )
    else:
        path = image(path, width=129, height=1)
    with pytest.raises(ValueError, match="ONNX_IMAGE_INPUT_INVALID"):
        binding.pixels(path)


@pytest.mark.parametrize("damage", ["labels", "preprocessor"])
def test_owned_preprocessing_and_labels_are_bound_to_signed_assets(monkeypatch, damage):
    if damage == "labels":
        monkeypatch.setattr(binding, "LABELS_SHA256", "a" * 64)
        with pytest.raises(ValueError, match="ONNX_CLASSIFIER_LABELS_CHANGED"):
            binding.labels()
    else:
        from oryn_engine import generated_capabilities

        artifact = copy.deepcopy(generated_capabilities.generated_artifact())
        artifact["buildInputs"]["opencvBinarySha256"] = "a" * 64
        monkeypatch.setattr(generated_capabilities, "generated_artifact", lambda: artifact)
        with pytest.raises(ValueError, match="ONNX_CLASSIFIER_PREPROCESSOR_CHANGED"):
            binding.verify_preprocessor()
