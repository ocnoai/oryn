"""Complete native BERT block-Q4 graphs and public per-token offset semantics."""

import copy
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from oryn_engine import onnx_tasks
from oryn_engine.onnx_graph import scan_graph
from oryn_engine.onnx_nbits import OPERATOR

FIXTURES = Path(__file__).parent / "fixtures/onnx_tokens"


def scan(path):
    raw = path.read_bytes()
    return scan_graph(lambda s, n: raw[s : s + n], len(raw), path.name, {path.name: len(raw)})


@pytest.fixture
def tiny(tmp_path):
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    from transformers import BertConfig, BertForTokenClassification, BertTokenizer

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    words = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "alice",
        "hello",
        "world",
        "boris",
        "cafe",
        "##s",
        "!",
        ".",
        "žofia",
    ]
    tokenizer = BertTokenizer(vocab={w: i for i, w in enumerate(words)}, do_lower_case=True)
    tokenizer.save_pretrained(tmp_path)
    config = BertConfig(
        vocab_size=len(words),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        id2label={0: "O", 1: "B-NAME", 2: "I-NAME"},
        label2id={"O": 0, "B-NAME": 1, "I-NAME": 2},
        attention_probs_dropout_prob=0,
        hidden_dropout_prob=0,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(6151)
    reference = BertForTokenClassification(config).eval()
    config.architectures = ["BertForTokenClassification"]
    config.save_pretrained(tmp_path)
    sample = tokenizer("Alice hello world", return_tensors="pt")
    path = tmp_path / "model.onnx"
    torch.onnx.export(
        reference,
        (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
        str(path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamo=False,
        opset_version=18,
        dynamic_axes={
            n: {0: "batch", 1: "sequence"}
            for n in ["input_ids", "attention_mask", "token_type_ids", "logits"]
        },
    )
    raw = onnx.load(path)
    original_bytes = raw.SerializeToString()
    quantizer = MatMulNBitsQuantizer(raw, block_size=32, is_symmetric=True, accuracy_level=0)
    quantizer.process()
    quantizer.model.save_model_to_file(str(path), use_external_data_format=False)
    quantized = onnx.load(path)
    tensors = {t.name: numpy_helper.to_array(t) for t in quantized.graph.initializer}
    replaced = 0
    for node in quantized.graph.node:
        if node.domain != "com.microsoft":
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        assert node.op_type == "MatMulNBits"
        packed = tensors[node.input[1]]
        scales = tensors[node.input[2]].reshape(attrs["N"], -1)
        # Independent low/high-nibble unpacking, native symmetric zero point 8.
        integers = np.empty((*packed.shape[:2], packed.shape[2] * 2), dtype=np.int16)
        integers[..., 0::2] = packed & 15
        integers[..., 1::2] = packed >> 4
        matrix = ((integers - 8) * scales[..., None]).reshape(attrs["N"], -1)[:, : attrs["K"]]
        module_path = node.name.strip("/").rsplit("/", 1)[0].replace("/", ".")
        reference.get_submodule(module_path).weight.data.copy_(torch.from_numpy(matrix.copy()))
        replaced += 1
    assert replaced == 13
    yield tmp_path, reference, tokenizer, original_bytes
    torch.set_num_threads(old)


@pytest.mark.parametrize("scale_layout", ["flat", "matrix"])
@pytest.mark.parametrize("text", ["Alice hello world!", "Boris cafés. 未知"])
def test_native_whole_quantized_encoder_matches_dequantized_torch(tiny, scale_layout, text):
    root, reference, tokenizer, _ = tiny
    if scale_layout == "matrix":
        model = onnx.load(root / "model.onnx")
        ts = {t.name: t for t in model.graph.initializer}
        for node in model.graph.node:
            if node.domain == "com.microsoft":
                a = {v.name: helper.get_attribute_value(v) for v in node.attribute}
                scales = ts[node.input[2]]
                scales.dims[:] = [a["N"], (a["K"] + 31) // 32]
        onnx.save(model, root / "model.onnx")
    session, config, graph, output = onnx_tasks.prepare(root, "token-classification", "cpu")
    assert OPERATOR in graph["operators"] and graph["operatorDomains"] == {"com.microsoft": 1}
    encoded = tokenizer(
        text, return_tensors="pt", return_offsets_mapping=True, return_special_tokens_mask=True
    )
    inputs = {
        k: v for k, v in encoded.items() if k in {"input_ids", "attention_mask", "token_type_ids"}
    }
    with torch.inference_mode():
        expected = reference(**inputs).logits
    actual = session.run([output], {k: v.numpy() for k, v in inputs.items()})[0]
    np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-4, atol=2e-5)
    result = onnx_tasks.execute(
        root, "token-classification", {"text": text}, "cpu", lambda *_: None
    )
    probabilities = expected[0].softmax(-1)
    labels = probabilities.argmax(-1)
    wanted = []
    for i, (start, end) in enumerate(encoded["offset_mapping"][0].tolist()):
        if encoded["special_tokens_mask"][0, i] or start == end:
            continue
        label = int(labels[i])
        wanted.append(
            (
                tokenizer.convert_ids_to_tokens(int(inputs["input_ids"][0, i])),
                label,
                start,
                end,
                float(probabilities[i, label]),
            )
        )
    assert len(result["tokens"]) == len(wanted)
    for got, (token, label, start, end, score) in zip(result["tokens"], wanted, strict=True):
        assert got | {"score": score} == {
            "token": token,
            "labelId": label,
            "label": config["id2label"][str(label)],
            "score": score,
            "start": start,
            "end": end,
        }
        assert got["score"] == pytest.approx(score, abs=2e-5)
    assert all(t["end"] <= len(text) for t in result["tokens"])
    with pytest.raises(ValueError, match="TOKEN_LIMIT"):
        onnx_tasks.execute(
            root, "token-classification", {"text": "hello " * 100}, "cpu", lambda *_: None
        )


def q4_graph(root):
    packed = numpy_helper.from_array(np.zeros((3, 1, 16), dtype=np.uint8), "packed")
    scales = numpy_helper.from_array(np.ones((3, 1), dtype=np.float32), "scales")
    graph = helper.make_graph(
        [
            helper.make_node(
                "MatMulNBits",
                ["x", "packed", "scales"],
                ["y"],
                domain="com.microsoft",
                K=32,
                N=3,
                bits=4,
                block_size=32,
            )
        ],
        "q4",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        [packed, scales],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    path = root / "model.onnx"
    onnx.save(model, path)
    return model, path


@pytest.mark.parametrize(
    "damage",
    [
        "domain",
        "version",
        "operator",
        "overload",
        "bits",
        "block",
        "accuracy",
        "huge-k",
        "unknown-attribute",
        "duplicate-attribute",
        "float-attribute",
        "zero-points",
        "missing-constant",
        "packed-shape",
        "packed-dtype",
        "scale-shape",
        "scale-dtype",
        "duplicate-initializer",
        "spoofed-standard",
    ],
)
def test_closed_native_quantization_boundary(tmp_path, damage):
    model, path = q4_graph(tmp_path)
    node = model.graph.node[0]
    if damage == "domain":
        model.opset_import[-1].domain = node.domain = "unreviewed.native"
    elif damage == "version":
        model.opset_import[-1].version = 2
    elif damage == "operator":
        node.op_type = "UnreviewedKernel"
    elif damage == "overload":
        node.overload = "dynamic"
    elif damage in {"bits", "block", "accuracy", "huge-k"}:
        key, value = {
            "bits": ("bits", 8),
            "block": ("block_size", 24),
            "accuracy": ("accuracy_level", 4),
            "huge-k": ("K", 2**30),
        }[damage]
        attr = next((a for a in node.attribute if a.name == key), None)
        if attr is None:
            node.attribute.append(helper.make_attribute(key, value))
        else:
            attr.i = value
    elif damage == "unknown-attribute":
        node.attribute.append(helper.make_attribute("kernel", "custom"))
    elif damage == "duplicate-attribute":
        node.attribute.append(helper.make_attribute("K", 32))
    elif damage == "float-attribute":
        node.attribute[0].CopyFrom(helper.make_attribute("K", 32.0))
    elif damage == "zero-points":
        node.input.append("zero_points")
    elif damage == "missing-constant":
        del model.graph.initializer[1]
    elif damage == "packed-shape":
        model.graph.initializer[0].dims[0] = 4
    elif damage == "packed-dtype":
        model.graph.initializer[0].data_type = TensorProto.INT8
    elif damage == "scale-shape":
        model.graph.initializer[1].dims[:] = [1, 3]
    elif damage == "scale-dtype":
        model.graph.initializer[1].data_type = TensorProto.FLOAT16
    elif damage == "duplicate-initializer":
        model.graph.initializer.append(model.graph.initializer[0])
    else:
        node.domain = ""
        node.op_type = OPERATOR
    onnx.save(model, path)
    with pytest.raises(ValueError):
        scan(path)


def test_complete_original_metadata_fits_the_native_task():
    config = json.loads((FIXTURES / "config.json").read_text())
    review = json.loads((FIXTURES / "original-graph-review.json").read_text())
    graph = {k: review[k] for k in ["inputs", "outputs"]}
    graph.update(
        externalFiles=[],
        opset=18,
        operatorDomains={"com.microsoft": 1},
        operators=sorted(
            {(n["domain"] + "::" if n["domain"] else "") + n["op"] for n in review["nodes"]}
        ),
    )
    assert onnx_tasks.validate_graph(graph, "token-classification", config) == "logits"
    assert sum(n["domain"] == "com.microsoft" for n in review["nodes"]) == 37


@pytest.mark.parametrize(
    "damage",
    ["labels", "sequence", "rank", "dtype", "extra-output", "input", "architecture", "remote-code"],
)
def test_task_binding_does_not_borrow_other_outputs(damage):
    config = json.loads((FIXTURES / "config.json").read_text())
    review = json.loads((FIXTURES / "original-graph-review.json").read_text())
    graph = {
        "inputs": copy.deepcopy(review["inputs"]),
        "outputs": copy.deepcopy(review["outputs"]),
        "operators": ["Add"],
        "opset": 18,
        "externalFiles": [],
    }
    if damage == "labels":
        graph["outputs"][0]["shape"][-1] = 34
    elif damage == "sequence":
        graph["outputs"][0]["shape"][1] = 512
    elif damage == "rank":
        graph["outputs"][0]["shape"] = ["batch", 35]
    elif damage == "dtype":
        graph["outputs"][0]["dtype"] = 7
    elif damage == "extra-output":
        graph["outputs"].append({"name": "hidden", "dtype": 1, "shape": ["batch", "sequence", 384]})
    elif damage == "input":
        graph["inputs"][0]["dtype"] = 1
    elif damage == "architecture":
        config["architectures"] = ["BertModel"]
    else:
        config["auto_map"] = {"AutoModel": "repo.Custom"}
    with pytest.raises(ValueError):
        onnx_tasks.validate_graph(graph, "token-classification", config)


@pytest.mark.parametrize("block", [16, 32, 64, 128, 256])
@pytest.mark.parametrize("accuracy", [0, 1])
def test_native_block_geometry_and_partial_final_block(tmp_path, block, accuracy):
    import onnxruntime as ort

    k, n = block + 3, 7
    rng = np.random.default_rng(61)
    unsigned = rng.integers(0, 16, (n, 2, block), dtype=np.uint8)
    packed = unsigned[..., 0::2] | (unsigned[..., 1::2] << 4)
    scales = rng.uniform(0.01, 0.1, (n, 2)).astype(np.float32)
    x = rng.normal(size=(3, k)).astype(np.float32)
    weight = ((unsigned.astype(np.int16) - 8) * scales[..., None]).reshape(n, -1)[:, :k].T
    graph = helper.make_graph(
        [
            helper.make_node(
                "MatMulNBits",
                ["x", "packed", "scales"],
                ["y"],
                domain="com.microsoft",
                K=k,
                N=n,
                bits=4,
                block_size=block,
                accuracy_level=accuracy,
            )
        ],
        "q4",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, k])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, n])],
        [numpy_helper.from_array(packed, "packed"), numpy_helper.from_array(scales, "scales")],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    assert scan(path)["operators"] == [OPERATOR]
    actual = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
        ["y"], {"x": x}
    )[0]
    np.testing.assert_allclose(actual, x @ weight, rtol=2e-5, atol=2e-5)


def test_isolated_worker_probe_prepare_and_token_result(tiny):
    import subprocess
    import sys

    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    root, _, _, _ = tiny
    pack = builtin_registry().active_pack("oryn.onnx-runtime")
    value = {"text": "Alice cafés!"}
    payload = {
        "modelRoot": str(root),
        "task": "token-classification",
        "backend": "cpu",
        "input": value,
        "inputPath": None,
        "outputPath": None,
    }
    expected = onnx_tasks.execute(root, "token-classification", value, "cpu", lambda *_: None)
    for command in ["probe", "prepare", "generate", "generate"]:
        request = {
            "abi": pack.abi,
            "packId": pack.pack_id,
            "version": pack.version,
            "command": command,
            "payload": {} if command == "probe" else payload,
        }
        child = subprocess.run(
            [sys.executable, "-m", "oryn_engine.runtime_packs.onnx_worker"],
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert child.returncode == 0, child.stderr
        messages = [json.loads(s) for s in child.stdout.splitlines() if s.startswith("{")]
        result = messages[-1]
        assert result["ok"], (result, child.stderr)
        if command == "probe":
            assert "token-classification-v1" in result["result"]["taskContracts"]
        elif command == "prepare":
            assert result["result"] == {"prepared": True}
        else:
            assert result["result"] == expected


def test_metadata_discovery_retains_original_onnx_and_q4_identity(tmp_path):
    from types import SimpleNamespace

    from test_models import FakeHub

    from oryn_engine.models import ModelStore

    config = json.loads((FIXTURES / "config.json").read_text())
    review = json.loads((FIXTURES / "original-graph-review.json").read_text())
    graph = {k: review[k] for k in ["inputs", "outputs"]}
    graph.update(
        externalFiles=[],
        opset=18,
        operatorDomains={"com.microsoft": 1},
        operators=sorted(
            {(n["domain"] + "::" if n["domain"] else "") + n["op"] for n in review["nodes"]}
        ),
    )

    class Hub(FakeHub):
        def read_onnx_metadata(self, repository, revision, name, sizes):
            assert name == "onnx/model_q4.onnx"
            return copy.deepcopy(graph)

    hub = Hub(
        config=config,
        pipeline_tag="token-classification",
        files=[
            (n, 1000, "a" * 64)
            for n in [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "onnx/model_q4.onnx",
            ]
        ],
    )
    store = ModelStore(
        tmp_path,
        hub=hub,
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 64_000_000_000),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda pack: None if pack.provider else object()
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "renamed/native-q4-tokens", "revision": "main"}
    )
    assert result["executable"], result
    assert result["runtimePack"]["taskContract"] == "token-classification-v1"
    assert {f["name"] for f in result["selectedFiles"]} == {f[0] for f in hub.files}
    assert hub.downloaded == []
