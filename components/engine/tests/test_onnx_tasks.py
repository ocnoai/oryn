from __future__ import annotations

import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from oryn_engine.onnx_graph import Reader, scan_graph
from oryn_engine.onnx_tasks import execute, prepare, validate_graph, validate_pooling


def encoder(root, *, external=False, domain="", output="last_hidden_state"):
    root.mkdir(exist_ok=True)
    weight = numpy_helper.from_array(np.arange(36, dtype=np.float32).reshape(12, 3), "weight")
    graph = helper.make_graph(
        [helper.make_node("Gather", ["weight", "input_ids"], [output], axis=0, domain=domain)],
        "encoder",
        [
            helper.make_tensor_value_info(n, TensorProto.INT64, ["batch", "sequence"])
            for n in ("input_ids", "attention_mask")
        ],
        [helper.make_tensor_value_info(output, TensorProto.FLOAT, ["batch", "sequence", 3])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    if domain:
        model.opset_import.append(helper.make_opsetid(domain, 1))
    path = root / "model.onnx"
    onnx.save_model(
        model,
        path,
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "architectures": ["BertModel"],
                "hidden_size": 3,
                "num_attention_heads": 1,
                "max_position_embeddings": 64,
            }
        )
    )
    (root / "vocab.txt").write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ncat\ndog\nis\nan\nanimal\nhello\nworld\n"
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {"tokenizer_class": "BertTokenizer", "do_lower_case": True, "model_max_length": 64}
        )
    )
    return path


def scan(path):
    raw = path.read_bytes()
    return scan_graph(
        lambda s, n: raw[s : s + n],
        len(raw),
        path.name,
        {p.name: p.stat().st_size for p in path.parent.iterdir()},
    )


def test_graph_membership_and_native_masked_pooling(tmp_path):
    path = encoder(tmp_path, external=True)
    graph = scan(path)
    assert graph["externalFiles"] == ["weights.data"]
    assert graph["inputs"][0]["shape"] == ["batch", "sequence"]
    result = execute(
        tmp_path,
        "feature-extraction",
        {"texts": ["cat", "dog"], "normalize": False},
        "cpu",
        lambda *_: None,
    )
    assert result["dimensions"] == 3
    assert result["embeddings"][0] == pytest.approx([10, 11, 12])
    assert result["embeddings"][1] == pytest.approx([11, 12, 13])
    normalized = execute(tmp_path, "feature-extraction", {"texts": ["cat"]}, "cpu", lambda *_: None)
    assert np.linalg.norm(normalized["embeddings"][0]) == pytest.approx(1)
    (tmp_path / "weights.data").unlink()
    with pytest.raises(ValueError, match="ONNX_EXTERNAL_RANGE"):
        prepare(tmp_path, "feature-extraction", "cpu")


def test_external_ranges_paths_and_duplicate_keys_fail_before_native_loading(tmp_path):
    path = encoder(tmp_path, external=True)
    for key, value in (("location", "../outside.bin"), ("offset", "9999999"), ("length", "0")):
        model = onnx.load(path, load_external_data=False)
        entry = next(v for v in model.graph.initializer[0].external_data if v.key == key)
        previous = entry.value
        entry.value = value
        path.write_bytes(model.SerializeToString())
        with pytest.raises(ValueError, match="ONNX_EXTERNAL_"):
            scan(path)
        entry.value = previous
        path.write_bytes(model.SerializeToString())
    model = onnx.load(path, load_external_data=False)
    model.graph.initializer[0].external_data.add(key="location", value="weights.data")
    path.write_bytes(model.SerializeToString())
    with pytest.raises(ValueError, match="ONNX_EXTERNAL_INVALID"):
        scan(path)


def test_custom_operators_truncated_graphs_and_unknown_task_io_are_rejected(tmp_path):
    path = encoder(tmp_path, domain="checkpoint.custom")
    with pytest.raises(ValueError, match="ONNX_OPSET_UNSUPPORTED"):
        scan(path)
    path.unlink()
    path = encoder(tmp_path)
    graph = scan(path)
    graph["inputs"][0]["dtype"] = TensorProto.FLOAT
    with pytest.raises(ValueError, match="ONNX_INPUT_BINDING_UNSUPPORTED"):
        validate_graph(
            graph, "feature-extraction", json.loads((tmp_path / "config.json").read_text())
        )
    path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(ValueError, match="ONNX_TRUNCATED"):
        scan(path)


def test_reader_skips_large_raw_payload_without_allocating_or_reading_it():
    # Length-delimited bytes can be skipped by offset. Only the next field's
    # block is fetched; the 1 GiB raw tensor is never materialized.
    size = 1024**3
    raw = b"\x4a\x80\x80\x80\x80\x04"
    calls = []

    def read(start, length):
        calls.append((start, length))
        return (raw[start:] + bytes(length))[:length]

    reader = Reader(read, size + len(raw))
    assert reader.message(0, size + len(raw)) == [(9, 2, (len(raw), size + len(raw)))]
    assert sum(n for _, n in calls) == 32768


def test_preparation_rejects_symlink_external_data_and_unsupported_pooling(tmp_path):
    encoder(tmp_path, external=True)
    data = tmp_path / "weights.data"
    moved = tmp_path / "elsewhere.data"
    data.rename(moved)
    data.symlink_to(moved)
    with pytest.raises(ValueError, match="ONNX_FILE_SCOPE_INVALID"):
        prepare(tmp_path, "feature-extraction", "cpu")
    data.unlink()
    moved.rename(data)
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "1_Pooling/config.json").write_text('{"pooling_mode_max_tokens": true}')
    with pytest.raises(ValueError, match="ONNX_POOLING_UNSUPPORTED"):
        prepare(tmp_path, "feature-extraction", "cpu")


@pytest.mark.parametrize("output", ["last_hidden_state", "token_embeddings"])
@pytest.mark.parametrize("mode,expected", [("cls_token", [6, 7, 8]), ("mean_tokens", [10, 11, 12])])
def test_declared_pooling_runs_the_native_graph(tmp_path, output, mode, expected):
    encoder(tmp_path, output=output)
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "1_Pooling/config.json").write_text(
        json.dumps(
            {
                "word_embedding_dimension": 3,
                "pooling_mode_" + mode: True,
            }
        )
    )
    result = execute(
        tmp_path,
        "feature-extraction",
        {"texts": ["cat"], "normalize": False},
        "cpu",
        lambda *_: None,
    )
    assert result["embeddings"][0] == pytest.approx(expected)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"pooling_mode_cls_token": 1},
        {"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": True},
        {"pooling_mode_cls_token": True, "word_embedding_dimension": 4},
        {"pooling_mode_cls_token": True, "include_prompt": False},
        {"pooling_mode_cls_token": True, "custom_pooler": "repo.py"},
    ],
)
def test_ambiguous_or_unimplemented_pooling_is_rejected(options):
    with pytest.raises(ValueError, match="ONNX_POOLING_UNSUPPORTED"):
        validate_pooling(options, {"hidden_size": 3})


def test_ambiguous_token_outputs_are_rejected(tmp_path):
    graph = scan(encoder(tmp_path))
    graph["outputs"].append(graph["outputs"][0] | {"name": "token_embeddings"})
    with pytest.raises(ValueError, match="ONNX_OUTPUT_BINDING_UNSUPPORTED"):
        validate_graph(
            graph, "feature-extraction", json.loads((tmp_path / "config.json").read_text())
        )


def test_legacy_nomic_alias_is_inert_for_a_complete_native_onnx_encoder(tmp_path):
    from oryn_engine.onnx_tasks import native_config

    encoder(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text()) | {
        "model_type": "nomic_bert",
        "architectures": ["NomicBertModel"],
        "auto_map": {
            "AutoConfig": "nomic-ai/nomic-bert-2048--configuration_hf_nomic_bert.NomicBertConfig",
            "AutoModel": "nomic-ai/nomic-bert-2048--modeling_hf_nomic_bert.NomicBertModel",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    normalized = native_config(config, "feature-extraction")
    assert "auto_map" not in normalized and "auto_map" in config
    result = execute(
        tmp_path,
        "feature-extraction",
        {"texts": ["cat"], "normalize": False},
        "cpu",
        lambda *_: None,
    )
    assert result["embeddings"][0] == pytest.approx([10, 11, 12])
    for changed in (
        {"auto_map": {"AutoModel": "repository.ArbitraryModel"}},
        {"auto_map": {"AutoTokenizer": "repository.ArbitraryTokenizer"}},
        {"model_type": "bert"},
        {"architectures": ["NomicBertForSequenceClassification"]},
        {"nested": {"auto_map": {"AutoModel": "repository.ArbitraryModel"}}},
    ):
        with pytest.raises(ValueError, match="Repository code is prohibited"):
            native_config(config | changed, "feature-extraction")
    with pytest.raises(ValueError, match="Repository code is prohibited"):
        native_config(config, "text-classification")


@pytest.mark.parametrize("count,accepted", [(193, True), (257, False)])
def test_graph_scanner_bounds_whisper_auxiliary_outputs(tmp_path, count, accepted):
    outputs = [f"output_{i}" for i in range(count)]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], [name]) for name in outputs],
        "auxiliary",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1]) for name in outputs],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    path = tmp_path / "model.onnx"
    onnx.save_model(model, path)
    if accepted:
        assert len(scan(path)["outputs"]) == count
    else:
        with pytest.raises(ValueError, match="ONNX_GRAPH_INVALID"):
            scan(path)
