import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from oryn_engine import onnx_tasks, onnx_vision

FIXTURES = Path(__file__).parent / "fixtures/onnx_vision"
ARTIFACT = "artifact_" + "a" * 24


def document(name):
    return json.loads((FIXTURES / name).read_text())


def image(path, mode="RGB", width=37, height=19):
    channels = {"RGB": 3, "L": 1, "RGBA": 4}[mode]
    data = ((np.arange(width * height * channels, dtype=np.uint32) * 37 + 13) % 256).astype(
        np.uint8
    )
    data = data.reshape((height, width, channels) if channels > 1 else (height, width))
    Image.fromarray(data, mode).save(path)
    return path


def tiny(root, *, external=False):
    config, metadata = document("config.json"), document("tiny-metadata.json")
    graph_name, metadata_name = onnx_vision.configuration(config)
    (root / "config.json").write_text(json.dumps(config))
    (root / metadata_name).write_text(json.dumps(metadata))
    matrix = np.arange(24, dtype=np.float32).reshape(3, 8) / 17 - 0.5
    bias = np.array([-3, -2, -1, -0.5, 0, 1, 2, 3], dtype=np.float32)
    nodes = [
        helper.make_node("ReduceMean", ["input"], ["mean"], axes=[2, 3], keepdims=0),
        helper.make_node("MatMul", ["mean", "weight"], ["initial_predictions"]),
        helper.make_node("Add", ["initial_predictions", "bias"], ["refined_predictions"]),
        helper.make_node(
            "TopK", ["refined_predictions", "k"], ["top_values", "selected_candidates"]
        ),
    ]
    outputs = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, ["batch_size", 8])
        for n in ("initial_predictions", "refined_predictions")
    ]
    outputs.append(
        helper.make_tensor_value_info("selected_candidates", TensorProto.INT64, ["batch_size", 4])
    )
    graph = helper.make_graph(
        nodes,
        "vision-tagger",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch_size", 3, 64, 64])],
        outputs,
        [
            numpy_helper.from_array(matrix, "weight"),
            numpy_helper.from_array(bias, "bias"),
            helper.make_tensor("k", TensorProto.INT64, [1], [4]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.save_model(
        model,
        root / graph_name,
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location="vision.data",
        size_threshold=0,
    )
    return matrix, bias, metadata


@pytest.mark.parametrize("case", document("reference.json")["cases"])
def test_preprocessing_matches_independent_original_reference(tmp_path, case):
    path = image(tmp_path / "image.png", case["mode"], case["width"], case["height"])
    pixels = onnx_vision.pixels(path, 64)
    assert pixels.shape == (1, 3, 64, 64) and pixels.dtype == np.float32
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == case["sha256"]


@pytest.mark.parametrize("external", [False, True])
def test_complete_native_graph_refined_sigmoid_topk_and_label_identity(tmp_path, external):
    weight, bias, metadata = tiny(tmp_path, external=external)
    path = image(tmp_path / "image.png")
    pixels = onnx_vision.pixels(path, 64)
    # Independent numeric evaluation of this complete test graph, with a
    # deliberately distinct refinement head and independent tag probabilities.
    logits = pixels.mean(axis=(2, 3)) @ weight + bias
    probabilities = 1 / (1 + np.exp(-logits.astype(np.float64)))
    expected = np.argsort(-probabilities[0])[:3]
    result = onnx_tasks.execute(
        tmp_path,
        "image-classification",
        {"imageArtifactId": ARTIFACT, "topK": 3},
        "cpu",
        lambda *_: None,
        path,
    )
    assert [x["labelId"] for x in result["predictions"]] == expected.tolist()
    assert [x["label"] for x in result["predictions"]] == [
        metadata["dataset_info"]["tag_mapping"]["idx_to_tag"][str(i)] for i in expected
    ]
    np.testing.assert_allclose(
        [x["score"] for x in result["predictions"]], probabilities[0, expected], rtol=1e-6
    )
    assert sum(x["score"] for x in result["predictions"]) > 1
    if external:
        (tmp_path / "vision.data").unlink()
        with pytest.raises(ValueError, match="ONNX_EXTERNAL_RANGE"):
            onnx_tasks.prepare(tmp_path, "image-classification", "cpu")


def test_original_graph_and_bounded_large_vocabulary():
    metadata = document("tiny-metadata.json")
    count = 70527
    metadata["dataset_info"].update(
        total_tags=count,
        tag_mapping={
            "idx_to_tag": {str(i): f"tag_{i}" for i in range(count)},
            "tag_to_idx": {f"tag_{i}": i for i in range(count)},
            "tag_to_category": {f"tag_{i}": "general" for i in range(count)},
        },
    )
    metadata["model_info"].update(img_size=512, tag_context_size=256)
    metadata["input_spec"]["shape"] = [3, 512, 512]
    for name in ("initial_predictions", "refined_predictions"):
        metadata["output_spec"][name]["shape"] = ["batch_size", count]
    metadata["output_spec"]["selected_candidates"]["shape"] = ["batch_size", 256]
    raw = json.dumps(metadata).encode()
    parsed = onnx_vision.parse_labels(raw)
    assert len(parsed["labels"]) == count
    assert (
        onnx_tasks.validate_graph(
            document("original-onnx-graph.json"),
            "image-classification",
            document("config.json"),
            vision=parsed,
        )
        == "refined_predictions"
    )
    from oryn_engine.model_metadata import parse_metadata

    with pytest.raises(ValueError):
        parse_metadata(raw)


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate",
        "size",
        "count",
        "missing",
        "inverse",
        "boolean",
        "duplicate-label",
        "category",
        "long-label",
        "nested-label",
        "normalization",
        "output",
        "context",
        "depth",
        "nan",
    ],
)
def test_label_metadata_fails_closed(damage):
    value = document("tiny-metadata.json")
    mapping = value["dataset_info"]["tag_mapping"]
    label = mapping["idx_to_tag"]["0"]
    if damage == "duplicate":
        raw = b'{"dataset_info":{},"dataset_info":{}}'
    elif damage == "size":
        raw = b" " * (onnx_vision.MAX_LABEL_BYTES + 1)
    else:
        if damage == "count":
            value["dataset_info"]["total_tags"] = 100001
        elif damage == "missing":
            del mapping["idx_to_tag"]["0"]
        elif damage == "inverse":
            mapping["tag_to_idx"][label] = 2
        elif damage == "boolean":
            mapping["tag_to_idx"][label] = False
        elif damage == "duplicate-label":
            mapping["idx_to_tag"]["1"] = label
        elif damage == "category":
            mapping["tag_to_category"][label] = "unrecognized"
        elif damage == "long-label":
            mapping["idx_to_tag"]["0"] = "x" * 257
        elif damage == "nested-label":
            mapping["idx_to_tag"]["0"] = {}
        elif damage == "normalization":
            value["input_spec"]["normalization"] = "different"
        elif damage == "output":
            value["output_spec"]["refined_predictions"]["shape"][-1] = 7
        elif damage == "context":
            value["model_info"]["tag_context_size"] = 9
        elif damage == "depth":
            x = value
            for _ in range(40):
                x["nested"] = {}
                x = x["nested"]
        elif damage == "nan":
            value["model_info"]["embedding_dim"] = float("nan")
        raw = json.dumps(value).encode()
    with pytest.raises(ValueError):
        onnx_vision.parse_labels(raw)


@pytest.mark.parametrize("damage", ["path", "ambiguous", "default", "family", "custom", "options"])
def test_closed_root_binding(damage):
    config = document("config.json")
    if damage == "path":
        config["models"][1]["path"] = "../outside.onnx"
    elif damage == "ambiguous":
        config["models"].append(config["models"][1])
    elif damage == "default":
        config["default_model"] = config["models"][0]["name"]
    elif damage == "family":
        config["model_name"] = "unknown"
    elif damage == "custom":
        config["auto_map"] = {"AutoModel": "repo.Code"}
    else:
        config["normalization"] = "softmax"
    with pytest.raises(ValueError):
        onnx_tasks.validate_config(config, "image-classification")


@pytest.mark.parametrize(
    "damage",
    [
        "input-name",
        "input-dtype",
        "input-size",
        "batch",
        "output-name",
        "output-width",
        "output-dtype",
        "missing-head",
    ],
)
def test_graph_signature_must_match_all_original_outputs(damage):
    graph = document("original-onnx-graph.json")
    metadata = {"size": 512, "labels": ["tag"] * 70527, "context": 256}
    if damage == "input-name":
        graph["inputs"][0]["name"] = "other"
    elif damage == "input-dtype":
        graph["inputs"][0]["dtype"] = 10
    elif damage == "input-size":
        graph["inputs"][0]["shape"][2] = 224
    elif damage == "batch":
        graph["inputs"][0]["shape"][0] = 2
    elif damage == "output-name":
        graph["outputs"][0]["name"] = "other"
    elif damage == "output-width":
        graph["outputs"][1]["shape"][1] = 4
    elif damage == "output-dtype":
        graph["outputs"][2]["dtype"] = 1
    else:
        graph["outputs"].pop()
    with pytest.raises(ValueError):
        onnx_tasks.validate_graph(
            graph, "image-classification", document("config.json"), vision=metadata
        )


@pytest.mark.parametrize("damage", [None, "missing-labels", "bad-graph", "wrong-graph"])
def test_metadata_discovery_uses_only_declared_graph_and_label_sidecar(tmp_path, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    config = document("config.json")
    graph_name, label_name = onnx_vision.configuration(config)
    graph = document("original-onnx-graph.json")
    if damage == "bad-graph":
        graph["inputs"][0]["shape"][1] = 4

    class Hub(FakeHub):
        def read_vision_labels(self, *args):
            return {"size": 512, "labels": [f"tag_{i}" for i in range(70527)], "context": 256}

        def read_onnx_metadata(self, *args):
            return graph

    names = {
        "config.json",
        graph_name,
        label_name,
        "alternate.onnx",
        "camie-tagger-v2.safetensors",
        "onnx_inference.py",
    }
    if damage == "missing-labels":
        names.remove(label_name)
    if damage == "wrong-graph":
        names.remove(graph_name)
    store = ModelStore(
        tmp_path,
        hub=Hub(
            config=config,
            pipeline_tag="image-classification",
            files=[(n, 100, "b" * 64) for n in names],
        ),
        runtime_available=True,
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "test/vision", "revision": "main"}
    )
    if damage:
        assert not result["executable"]
    else:
        assert result["executable"]
        assert result["runtimePack"]["packId"] == "oryn.onnx-runtime"
        assert result["technicalCapabilities"]["inputModalities"] == ["image"]
        assert {x["name"] for x in result["selectedFiles"]} == {
            "config.json",
            graph_name,
            label_name,
        }
        assert result["taskProfile"]["contract"]["id"] == "image-classification-v1"


@pytest.mark.parametrize("damage", ["nan", "width", "shape"])
def test_invalid_predictions_fail_before_result_creation(tmp_path, damage):
    logits = np.zeros((1, 8), dtype=np.float32)
    if damage == "nan":
        logits[0, 0] = np.nan
    elif damage == "width":
        logits = logits[:, :7]
    else:
        logits = logits[0]
    session = SimpleNamespace(run=lambda *_: [logits])
    metadata = onnx_vision.parse_labels(json.dumps(document("tiny-metadata.json")).encode())
    with pytest.raises(ValueError, match="ONNX_VISION_OUTPUT_INVALID"):
        onnx_vision.execute(session, metadata, {}, image(tmp_path / "image.png"), lambda *_: None)


def test_image_geometry_symlinks_and_metadata_paths(tmp_path):
    tiny(tmp_path)
    path = image(tmp_path / "image.png", width=1024, height=1)
    with pytest.raises(ValueError, match="ONNX_IMAGE_ASPECT_UNSUPPORTED"):
        onnx_vision.pixels(path, 64)
    link = tmp_path / "linked.png"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="ONNX_IMAGE_INPUT_INVALID"):
        onnx_vision.pixels(link, 64)
    _, name = onnx_vision.configuration(document("config.json"))
    sidecar = tmp_path / name
    other = tmp_path / "other.json"
    sidecar.rename(other)
    sidecar.symlink_to(other)
    with pytest.raises(ValueError, match="ONNX_FILE_SCOPE_INVALID"):
        onnx_tasks.prepare(tmp_path, "image-classification", "cpu")


@pytest.mark.parametrize("damage", [None, "oversized", "redirect"])
def test_bounded_hub_label_reader_keeps_host_and_ordinary_config_limits(monkeypatch, damage):
    import httpx

    from oryn_engine.models import HuggingFaceClient, ModelStoreError

    raw = json.dumps(document("tiny-metadata.json")).encode()
    # Whitespace makes a valid document larger than the ordinary 1 MiB cap.
    raw += b" " * (1024 * 1024 + 1)
    if damage == "oversized":
        raw += b" " * onnx_vision.MAX_LABEL_BYTES

    def serve(request):
        if damage == "redirect":
            return httpx.Response(302, headers={"location": "https://outside.invalid/labels.json"})
        return httpx.Response(200, content=raw)

    client = httpx.Client(transport=httpx.MockTransport(serve))
    monkeypatch.setattr(httpx, "stream", client.stream)
    hub = HuggingFaceClient(token_provider=lambda: False)
    if damage:
        with pytest.raises(ModelStoreError, match="Metadata"):
            hub.read_vision_labels("test/vision", "a" * 40, "tags-metadata.json")
    else:
        assert (
            len(hub.read_vision_labels("test/vision", "a" * 40, "tags-metadata.json")["labels"])
            == 8
        )
        with pytest.raises(ModelStoreError):
            hub.read_json("test/vision", "a" * 40, "tags-metadata.json")
    client.close()


def test_isolated_worker_probe_prepare_and_public_image_result(tmp_path):
    import subprocess
    import sys

    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    pack = builtin_registry().active_pack("oryn.onnx-runtime")
    tiny(tmp_path)
    path = image(tmp_path / "image.png")
    payload = {
        "modelRoot": str(tmp_path),
        "task": "image-classification",
        "backend": "cpu",
        "input": {"imageArtifactId": ARTIFACT, "topK": 2},
        "inputPath": str(path),
        "outputPath": None,
    }
    for command in ("probe", "prepare", "generate"):
        request = {
            "abi": pack.abi,
            "packId": pack.pack_id,
            "version": pack.version,
            "command": command,
            "payload": payload,
        }
        child = subprocess.run(
            [sys.executable, "-m", "oryn_engine.runtime_packs.onnx_worker"],
            input=json.dumps(request) + "\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert child.returncode == 0, child.stderr
        records = [json.loads(line) for line in child.stdout.splitlines()]
        result = next(x for x in records if x["type"] == "result")
        assert result["ok"], (result, child.stderr)
        if command == "probe":
            assert "image-classification-v1" in result["result"]["taskContracts"]
        elif command == "prepare":
            assert result["result"]["prepared"]
        else:
            assert len(result["result"]["predictions"]) == 2


def test_signed_shared_adapter_declares_vision_without_requiring_text_assets():
    from oryn_engine.runtime_packs.builtin_foundations import builtin_foundation_registry

    adapter = builtin_foundation_registry().adapter_for_legacy_pack("oryn.onnx-runtime")
    assert "camie-tagger" in adapter.architectures
    assert "image-classification-v1" in adapter.task_contracts
    assert adapter.required_files == ("*.onnx",)
