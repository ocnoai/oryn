import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from oryn_engine import jina_onnx, onnx_tasks


def originals(task):
    name = (
        "jina-embeddings-v3"
        if task == "feature-extraction"
        else "jina-reranker-v2-base-multilingual"
    )
    root = Path(__file__).parent / "fixtures/jina_onnx" / name
    return {p.name: json.loads(p.read_text()) for p in root.glob("*.json")}


def tiny(root, task):
    from transformers import XLMRobertaTokenizer

    data = originals(task)
    config = data["config.json"] | {
        "vocab_size": 12,
        "hidden_size": 3,
        "max_position_embeddings": 64,
    }
    vocab = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
        "▁",
        "cat",
        "dog",
        "blue",
        "river",
        "sky",
        "the",
    ]
    processor = XLMRobertaTokenizer(vocab=[(token, -float(i)) for i, token in enumerate(vocab)])
    processor.save_pretrained(root)
    special = json.loads((root / "tokenizer_config.json").read_text())
    special.update(tokenizer_class="XLMRobertaTokenizer", model_max_length=64)
    (root / "tokenizer_config.json").write_text(json.dumps(special))
    (root / "config.json").write_text(json.dumps(config))
    inputs = [
        helper.make_tensor_value_info(n, TensorProto.INT64, ["batch", "sequence"])
        for n in ("input_ids", "attention_mask")
    ]
    initializers = []
    if task == "feature-extraction":
        # Every task has distinct learned values, so a missing/wrong selector
        # cannot accidentally pass the output comparison.
        values = np.arange(5 * 12 * 3, dtype=np.float32).reshape(5, 12, 3) / 100
        initializers = [numpy_helper.from_array(values, "adapters")]
        inputs.append(helper.make_tensor_value_info("task_id", TensorProto.INT64, []))
        nodes = [
            helper.make_node("Gather", ["adapters", "task_id"], ["selected"], axis=0),
            helper.make_node("Gather", ["selected", "input_ids"], ["text_embeds"], axis=0),
        ]
        output = helper.make_tensor_value_info(
            "text_embeds", TensorProto.FLOAT, ["batch", "sequence", 3]
        )
    else:
        nodes = [
            helper.make_node("Cast", ["input_ids"], ["floats"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMean", ["floats"], ["logits"], axes=[1], keepdims=1),
        ]
        output = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 1])
    graph = helper.make_graph(nodes, "sealed-jina", inputs, [output], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)], ir_version=9)
    onnx.save_model(
        model,
        root / "model.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    (root / "modeling_lora.py").write_text("raise RuntimeError('must remain inert')")
    return config, processor


@pytest.mark.parametrize("normalize", [False, True])
def test_native_selector_and_masked_mean_match_independent_task_values(tmp_path, normalize):
    _, tokenizer = tiny(tmp_path, "feature-extraction")
    texts = ["cat", "the river is blue"]
    result = onnx_tasks.execute(
        tmp_path,
        "feature-extraction",
        {"texts": texts, "normalize": normalize},
        "cpu",
        lambda *_: None,
    )
    values = np.arange(5 * 12 * 3, dtype=np.float32).reshape(5, 12, 3) / 100
    expected = []
    for text in texts:
        ids = tokenizer(text)["input_ids"]
        vector = values[4, ids].mean(0)
        expected.append(vector / np.linalg.norm(vector) if normalize else vector)
    np.testing.assert_allclose(result["embeddings"], expected, rtol=1e-6, atol=1e-6)
    assert result["normalized"] is normalize


def test_native_pair_ranking_uses_sigmoid_and_original_document_indexes(tmp_path):
    _, tokenizer = tiny(tmp_path, "text-ranking")
    docs = ["cat", "blue river", "dog"]
    result = onnx_tasks.execute(
        tmp_path,
        "text-ranking",
        {"query": "sky", "documents": docs, "topK": 2},
        "cpu",
        lambda *_: None,
    )
    expected = []
    for index, text in enumerate(docs):
        ids = tokenizer("sky", text_pair=text)["input_ids"]
        expected.append({"index": index, "score": float(1 / (1 + np.exp(-np.mean(ids))))})
    expected.sort(key=lambda row: -row["score"])
    assert [v["index"] for v in result["results"]] == [v["index"] for v in expected[:2]]
    np.testing.assert_allclose(
        [v["score"] for v in result["results"]], [v["score"] for v in expected[:2]], rtol=1e-6
    )


@pytest.mark.parametrize("task", ["feature-extraction", "text-ranking"])
def test_original_graph_metadata_and_safe_shared_inspection(tmp_path, task):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    data = originals(task)
    config = onnx_tasks.validate_config(data["config.json"], task)
    assert config["model_type"] == "xlm-roberta" and "auto_map" not in config
    assert onnx_tasks.validate_graph(data["graph.json"], task, config) == (
        "text_embeds" if task == "feature-extraction" else "logits"
    )

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return data[filename]

        def read_onnx_metadata(self, repository, revision, filename, sizes):
            return data["graph.json"]

    names = {"config.json", "tokenizer_config.json", "tokenizer.json", "onnx/model.onnx"} | set(
        data["graph.json"]["externalFiles"]
    )
    hub = Hub(
        config=data["config.json"], pipeline_tag=task, files=[(n, 100, "b" * 64) for n in names]
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "test/jina", "revision": "main"}
    )
    assert result["executable"], result
    assert result["runtimePack"]["packId"] == "oryn.onnx-runtime"
    assert not hub.downloaded


@pytest.mark.parametrize(
    "damage", ["missing", "vector", "float", "extra-input", "wrong-output", "wrong-rank", "width"]
)
def test_task_selector_and_output_geometry_are_closed(damage):
    data = originals("feature-extraction")
    graph = copy.deepcopy(data["graph.json"])
    if damage == "missing":
        graph["inputs"].pop()
    elif damage == "vector":
        graph["inputs"][-1]["shape"] = [1]
    elif damage == "float":
        graph["inputs"][-1]["dtype"] = 1
    elif damage == "extra-input":
        graph["inputs"].append({"name": "unknown", "dtype": 7, "shape": []})
    elif damage == "wrong-output":
        graph["outputs"][0]["name"] = "last_hidden_state"
    elif damage == "width":
        graph["outputs"][0]["shape"][-1] = 768
    else:
        graph["outputs"][0]["shape"] = ["batch", 1024]
    with pytest.raises(ValueError):
        onnx_tasks.validate_graph(graph, "feature-extraction", data["config.json"])


@pytest.mark.parametrize(
    "change",
    [
        {"auto_map": {"AutoModel": "unreviewed.Code"}},
        {"model_type": "bert"},
        {"lora_adaptations": list(reversed(jina_onnx.TASKS))},
        {"truncate_dim": 32},
        {"load_trained_adapters": False},
        {"lora_rank": 8},
        {"task_instructions": {"text-matching": "unreviewed prefix"}},
    ],
)
def test_metadata_cannot_choose_another_adapter_or_instruction(change):
    with pytest.raises(ValueError):
        onnx_tasks.validate_config(
            originals("feature-extraction")["config.json"] | change, "feature-extraction"
        )


def test_task_specific_aliases_do_not_admit_other_tasks_or_custom_code():
    for task, wrong in (
        ("feature-extraction", "text-ranking"),
        ("text-ranking", "text-classification"),
    ):
        with pytest.raises(ValueError):
            onnx_tasks.validate_config(originals(task)["config.json"], wrong)
    data = originals("feature-extraction")
    config = onnx_tasks.validate_config(data["config.json"], "feature-extraction")
    with pytest.raises(ValueError, match="POOLING"):
        onnx_tasks.validate_pooling({"pooling_mode_cls_token": True}, config)


def test_signed_onnx_source_inputs_match_the_owned_bindings():
    from oryn_engine.generated_capabilities import generated_artifact
    from oryn_engine.runtime_routing import canonical_digest

    artifact = generated_artifact()
    inputs = artifact["buildInputs"]
    names = (
        "jina_onnx.py",
        "onnx_tasks.py",
        "onnx_tdt.py",
        "onnx_graph.py",
        "onnx_nbits.py",
        "onnx_inspection.py",
        "onnx_vision.py",
        "onnx_classifier.py",
        "data/imagenet-efficientnet-lite4-labels.json",
        "runtime_packs/onnx_worker.py",
    )
    hashes = {
        name: hashlib.sha256((Path(onnx_tasks.__file__).parent / name).read_bytes()).hexdigest()
        for name in names
    }
    assert all(inputs["orynNativeSources"][name] == digest for name, digest in hashes.items())
    source = next(
        s for s in artifact["sources"] if s["provider"]["id"] == "oryn.source.onnx-encoders"
    )
    assert source["foundation"]["digest"] == canonical_digest(
        {
            "operators": inputs["onnxOperators"],
            "audioSource": inputs["orynNativeSources"]["onnx_audio.py"],
            "classifierPreprocessor": {
                "version": inputs["depthDependencies"]["opencv-python-headless"],
                "binarySha256": inputs["opencvBinarySha256"],
            },
            "nativeSources": hashes,
        }
    )
