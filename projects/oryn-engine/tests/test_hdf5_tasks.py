import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from oryn_engine import hdf5_tasks, shared_tasks

FIXTURES = Path(__file__).parent / "fixtures/hdf5"


def test_original_numeric_descriptors_cover_the_complete_native_graph():
    from transformers import TapasConfig, TapasForQuestionAnswering

    config = json.loads((FIXTURES / "original-tiny-config.json").read_text())
    hdf5_tasks.validate_config(config, "table-question-answering")
    with torch.device("meta"):
        native = TapasForQuestionAnswering(TapasConfig.from_dict(config))
    expected = {k: tuple(v.shape) for k, v in native.state_dict().items()}
    actual = {}
    for item in json.loads((FIXTURES / "original-tiny-datasets.json").read_text()):
        name, transpose = hdf5_tasks.tapas_name(item["name"].lstrip("/"))
        assert name not in actual
        actual[name] = tuple(reversed(item["shape"])) if transpose else tuple(item["shape"])
    assert actual == expected


def fixture(root, aggregations=4):
    from transformers import TapasTokenizer

    shutil.copytree(FIXTURES / f"tapas-{aggregations}", root, dirs_exist_ok=True)
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "[EMPTY]",
        "name",
        "age",
        "a",
        "b",
        "10",
        "20",
    ]
    vocab += [f"extra_{i}" for i in range(12, 32)]
    vocab[-1] = "what"
    (root / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = TapasTokenizer(vocab_file=str(root / "vocab.txt"), model_max_length=128)
    tokenizer.save_pretrained(root)
    return json.loads((root / "config.json").read_text())


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("aggregations", [0, 4])
def test_complete_native_graph_matches_independent_tensorflow_outputs(
    tmp_path, backend, aggregations
):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    fixture(tmp_path, aggregations)
    model, _, _ = shared_tasks.load_task(tmp_path, "table-question-answering", backend)
    with (
        np.load(tmp_path / "reference.npz", allow_pickle=False) as reference,
        torch.inference_mode(),
    ):
        inputs = {
            k: torch.tensor(reference[k], dtype=torch.long, device=backend)
            for k in ["input_ids", "attention_mask", "token_type_ids"]
        }
        result = model(**inputs)
        torch.testing.assert_close(
            result.logits.cpu(), torch.tensor(reference["logits"]), atol=2e-5, rtol=2e-5
        )
        if aggregations:
            torch.testing.assert_close(
                result.logits_aggregation.cpu(),
                torch.tensor(reference["logits_aggregation"]),
                atol=2e-5,
                rtol=2e-5,
            )
        else:
            assert result.logits_aggregation is None


@pytest.mark.parametrize(
    "part", ["embedding", "encoder", "cell", "column", "aggregation", "extra", "shape", "duplicate"]
)
def test_no_missing_or_unreviewed_weight_is_initialized(tmp_path, part):
    fixture(tmp_path)
    with h5py.File(tmp_path / "tf_model.h5", "r+") as f:
        names = []
        f.visititems(
            lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None
        )
        matches = {
            "embedding": "word_embeddings",
            "encoder": "attention/self/query/kernel",
            "cell": "output/output_weights",
            "column": "column_output/column_output_weights",
            "aggregation": "aggregation_classifier/kernel",
        }
        if part in matches:
            del f[next(n for n in names if matches[part] in n)]
        elif part == "extra":
            f.create_dataset("rogue/head:0", data=np.zeros(2, np.float32))
        elif part == "shape":
            name = next(n for n in names if "attention/self/query/kernel" in n)
            del f[name]
            f.create_dataset(name, data=np.zeros((2, 2), np.float32))
        else:
            name = next(n for n in names if "attention/self/query/kernel" in n)
            duplicate = name.replace(
                "tf_tapas_for_question_answering_1", "tf_tapas_for_question_answering_99"
            )
            f.create_dataset(duplicate, data=f[name][()])
    with pytest.raises(ValueError, match="HDF5_WEIGHT_"):
        shared_tasks.load_task(tmp_path, "table-question-answering", "cpu")


def test_layout_preserves_safetensors_preference_and_safe_hdf5_fallback(tmp_path):
    from oryn_engine.model_layouts import assess_layout

    config = fixture(tmp_path)
    files = [p.name for p in tmp_path.iterdir()]
    layout = assess_layout(files, config, "table-question-answering")
    assert "tf_model.h5" in layout.selected_files
    assert not layout.artifact_reasons and not layout.runtime_reasons
    existing = assess_layout(files + ["model.safetensors"], config, "table-question-answering")
    assert (
        "tf_model.h5" not in existing.selected_files
        and "model.safetensors" in existing.selected_files
    )
    invalid = assess_layout(files, config | {"hidden_size": 0}, "table-question-answering")
    assert invalid.runtime_reasons and "tf_model.h5" not in invalid.selected_files
    # Existing native Safetensors loading still reaches the same table executor.
    model, _, _ = shared_tasks.load_task(tmp_path, "table-question-answering", "cpu")
    model.save_pretrained(tmp_path)
    loaded, _, _ = shared_tasks.load_task(tmp_path, "table-question-answering", "cpu")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_shared_execution_preserves_table_cell_coordinates_and_aggregation(tmp_path):
    from oryn_protocol.task_contracts import validate_invocation, validate_result

    from oryn_engine.coverage_tasks import table_answer
    from oryn_engine.task_profiles import profile_for_execution

    config = fixture(tmp_path, 0)
    value = {"question": "what age", "columns": ["name", "age"], "rows": [["a", "10"], ["b", "20"]]}
    result = shared_tasks.execute_task(
        tmp_path, "table-question-answering", value, "cpu", None, None, lambda *_: None
    )
    profile = profile_for_execution(
        {
            "technicalCapabilities": {"task": "table-question-answering"},
            "runtimePack": {"runnerId": "transformers.shared-tasks"},
        },
        config,
    )
    validate_result(profile, {"kind": "structured", "value": result}, value)
    assert profile["limits"] == {"tableRows": 15, "tableColumns": 15}
    oversized = value | {"rows": [["a", "10"]] * 16}
    with pytest.raises(ValueError):
        validate_invocation(profile, oversized)
    with pytest.raises(ValueError, match="row/column"):
        table_answer(None, None, config, oversized, "cpu")
    for cell in result["cells"]:
        assert cell["text"] == value["rows"][cell["row"]][cell["column"]]


@pytest.mark.parametrize(
    "change",
    [
        {"type_vocab_sizes": [3, 16]},
        {"num_attention_heads": 3},
        {"temperature": 0},
        {"num_aggregation_labels": 99},
        {"architectures": ["TapasModel"]},
    ],
)
def test_incompatible_task_configuration_is_rejected(tmp_path, change):
    config = fixture(tmp_path)
    with pytest.raises(ValueError):
        hdf5_tasks.validate_config(config | change, "table-question-answering")
