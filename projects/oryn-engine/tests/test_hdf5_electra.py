"""Numeric ELECTRA graph, QA offsets and legacy Safetensors route coverage."""

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from oryn_engine import hdf5_tasks, shared_tasks
from oryn_engine.model_layouts import assess_layout

FIXTURE = Path(__file__).parent / "fixtures/hdf5/electra"
TASK = "question-answering"


def fixture(root):
    from transformers import ElectraTokenizer

    shutil.copytree(FIXTURE, root, dirs_exist_ok=True)
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "who", "is", "elise", "cat", "paris"]
    words += [f"token_{i}" for i in range(10, 32)]
    ElectraTokenizer(vocab={w: i for i, w in enumerate(words)}).save_pretrained(root)
    return json.loads((root / "config.json").read_text())


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_full_extractive_graph_matches_independent_tf_logits(tmp_path, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    fixture(tmp_path)
    model, _, _ = shared_tasks.load_task(tmp_path, TASK, backend)
    with np.load(tmp_path / "reference.npz", allow_pickle=False) as ref, torch.inference_mode():
        output = model(
            **{
                k: torch.tensor(ref[k], device=backend, dtype=torch.long)
                for k in ["input_ids", "attention_mask", "token_type_ids"]
            }
        )
        for name in ["start_logits", "end_logits"]:
            torch.testing.assert_close(
                getattr(output, name).cpu(), torch.tensor(ref[name]), atol=3e-5, rtol=3e-4
            )


@pytest.mark.parametrize("part", ["embedding", "projection", "encoder", "head", "extra", "shape"])
def test_incomplete_graphs_are_rejected(tmp_path, part):
    fixture(tmp_path)
    with h5py.File(tmp_path / "tf_model.h5", "r+") as f:
        names = []
        f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
        if part == "extra":
            f.create_dataset("rogue/head:0", data=np.zeros(2, np.float32))
        else:
            fragment = {
                "embedding": "word_embeddings/weight",
                "projection": "embeddings_project/kernel",
                "encoder": "attention/self/query/kernel",
                "head": "qa_outputs/kernel",
                "shape": "qa_outputs/bias",
            }[part]
            name = next(n for n in names if fragment in n)
            del f[name]
            if part == "shape":
                f.create_dataset(name, data=np.zeros(4, np.float32))
    with pytest.raises(ValueError, match="HDF5_WEIGHT_"):
        hdf5_tasks.load_native(tmp_path, TASK)


def test_shared_qa_preserves_original_unicode_context_offsets(tmp_path):
    from oryn_protocol.task_contracts import validate_result

    from oryn_engine.task_profiles import profile_for_execution

    config = fixture(tmp_path)
    value = {"question": "who", "context": "  Élise  ", "maxAnswerLength": 1}
    result = shared_tasks.execute_task(tmp_path, TASK, value, "cpu", None, None, lambda *_: None)
    assert result["answer"] == "Élise" and (result["start"], result["end"]) == (2, 7)
    assert result["answer"] == value["context"][result["start"] : result["end"]]
    profile = profile_for_execution(
        {
            "technicalCapabilities": {"task": TASK},
            "runtimePack": {"runnerId": "transformers.shared-tasks"},
        },
        config,
    )
    validate_result(profile, {"kind": "structured", "value": result}, value)
    assert TASK not in shared_tasks.IMAGE_TASKS
    assert "question-answering-v1" not in shared_tasks.NEGOTIATED_TASK_CONTRACTS
    assert (
        shared_tasks.validate_task_input(TASK, {"question": "who", "context": "elise"})[
            "maxAnswerLength"
        ]
        == 64
    )


def test_token_budget_is_checked_without_truncating_the_context(tmp_path):
    config = fixture(tmp_path)
    model, tokenizer, _ = hdf5_tasks.load_components(tmp_path, TASK, "cpu")
    for question, context, reason in [
        ("who", "cat " * 70, "TOKEN_LIMIT"),
        (" ", "cat", "INPUT_EMPTY"),
    ]:
        with pytest.raises(ValueError, match=reason):
            hdf5_tasks.answer_question(
                model,
                tokenizer,
                config,
                {"question": question, "context": context, "maxAnswerLength": 1},
                "cpu",
            )


def test_safe_native_layouts_keep_their_existing_inspection_route(tmp_path):
    config = fixture(tmp_path)
    files = [p.name for p in tmp_path.iterdir()]
    layout = assess_layout(files, config, TASK)
    assert "tf_model.h5" in layout.selected_files
    assert not layout.artifact_reasons and not layout.runtime_reasons
    safe = assess_layout(files + ["model.safetensors"], config, TASK)
    assert "tf_model.h5" not in safe.selected_files
    bert = config | {"model_type": "bert", "architectures": ["BertForQuestionAnswering"]}
    legacy = assess_layout(files + ["model.safetensors"], bert, TASK)
    assert not legacy.runtime_reasons and not legacy.artifact_reasons


@pytest.mark.parametrize(
    "change",
    [
        {"embedding_size": 0},
        {"num_labels": 3},
        {"is_decoder": True},
        {"max_position_embeddings": 0},
    ],
)
def test_incompatible_configurations_are_rejected(tmp_path, change):
    config = fixture(tmp_path)
    with pytest.raises(ValueError):
        hdf5_tasks.validate_config(config | change, TASK)
