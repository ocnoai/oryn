"""Complete legacy BERT classifiers, safe family inference and vision metadata."""

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from oryn_engine import hdf5_tasks, shared_tasks
from oryn_engine.generated_capabilities import normalized_transformers_config
from oryn_engine.model_layouts import assess_layout
from oryn_engine.model_metadata import validate_model_config

FIXTURE = Path(__file__).parent / "fixtures/hdf5/bert"
TASK = "text-classification"


def fixture(root):
    shutil.copytree(FIXTURE, root, dirs_exist_ok=True)
    return json.loads((root / "config.json").read_text())


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_full_original_graph_matches_independent_tensorflow(tmp_path, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    fixture(tmp_path)
    model, tokenizer, config = shared_tasks.load_task(tmp_path, TASK, backend)
    assert config["model_type"] == "bert"
    assert len(tokenizer) == 32 and tokenizer("HELLO")["input_ids"] == [2, 5, 3]
    with np.load(tmp_path / "reference.npz", allow_pickle=False) as ref, torch.inference_mode():
        output = model(
            **{
                k: torch.tensor(ref[k], device=backend, dtype=torch.long)
                for k in ["input_ids", "attention_mask", "token_type_ids"]
            }
        )
        torch.testing.assert_close(
            output.logits.cpu(), torch.tensor(ref["logits"]), atol=3e-5, rtol=3e-4
        )


@pytest.mark.parametrize("part", ["embedding", "encoder", "pooler", "head", "extra", "shape"])
def test_no_incomplete_or_incompatible_original_graphs(tmp_path, part):
    fixture(tmp_path)
    with h5py.File(tmp_path / "tf_model.h5", "r+") as f:
        names = []
        f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
        if part == "extra":
            f.create_dataset("rogue/head:0", data=np.zeros(2, np.float32))
        else:
            fragment = {
                "embedding": "word_embeddings/weight",
                "encoder": "attention/self/query/kernel",
                "pooler": "pooler/dense/kernel",
                "head": "classifier/kernel",
                "shape": "classifier/bias",
            }[part]
            name = next(n for n in names if fragment in n)
            del f[name]
            if part == "shape":
                f.create_dataset(name, data=np.zeros(4, np.float32))
    with pytest.raises(ValueError, match="HDF5_WEIGHT_"):
        hdf5_tasks.load_native(tmp_path, TASK)


def test_shared_result_preserves_original_labels_and_probabilities(tmp_path):
    fixture(tmp_path)
    value = {"text": "hello finance", "topK": 3}
    result = shared_tasks.execute_task(tmp_path, TASK, value, "cpu", None, None, lambda *_: None)
    shared_tasks.validate_task_output(TASK, result)
    model, tokenizer, config = shared_tasks.load_task(tmp_path, TASK, "cpu")
    with torch.inference_mode():
        scores = model(**tokenizer(value["text"], return_tensors="pt")).logits[0].softmax(-1)
    assert {r["label"] for r in result["predictions"]} == set(config["id2label"].values())
    for item in result["predictions"]:
        index = config["label2id"][item["label"]]
        assert item["score"] == pytest.approx(scores[index].item(), abs=1e-6)
    with pytest.raises(ValueError, match="TOKEN_LIMIT"):
        shared_tasks.execute_task(
            tmp_path, TASK, value | {"text": "hello " * 130}, "cpu", None, None, lambda *_: None
        )


def test_legacy_layout_prefers_safe_weights_and_requires_vocabulary(tmp_path):
    config = fixture(tmp_path)
    files = [p.name for p in tmp_path.iterdir()]
    layout = assess_layout(files, config, TASK)
    assert not layout.artifact_reasons and not layout.runtime_reasons
    assert "tf_model.h5" in layout.selected_files
    missing = assess_layout([f for f in files if f != "vocab.txt"], config, TASK)
    assert "MODEL_TOKENIZER_MISSING" in missing.artifact_reasons
    safe = assess_layout(files + ["model.safetensors"], config, TASK)
    assert "tf_model.h5" not in safe.selected_files


def test_missing_family_uses_only_unambiguous_signed_native_mapping(monkeypatch):
    from oryn_engine import generated_capabilities as capabilities

    config = {"architectures": ["BertForSequenceClassification"]}
    assert normalized_transformers_config(config, TASK)["model_type"] == "bert"
    assert "model_type" not in config
    for change in [
        {"model_type": None},
        {"model_type": "roberta"},
        {"auto_map": {"AutoModel": "repository.Model"}},
        {"architectures": ["CustomBertForSequenceClassification"]},
        {"architectures": ["BertForSequenceClassification", "Other"]},
    ]:
        candidate = config | change
        assert normalized_transformers_config(candidate, TASK) == candidate
    assert normalized_transformers_config(config, "translation") == config
    monkeypatch.setattr(
        capabilities,
        "generated_artifact",
        lambda: {
            "buildInputs": {
                "transformersTaskArchitectures": {
                    TASK: {"bert": config["architectures"], "ambiguous": config["architectures"]}
                }
            }
        },
    )
    assert capabilities.normalized_transformers_config(config, TASK) == config


@pytest.mark.parametrize(
    "change",
    [
        {"is_decoder": True},
        {"vocab_size": 0},
        {"auto_map": {"AutoModel": "repo.Model"}},
        {"quantization_config": {"bits": 4}},
    ],
)
def test_legacy_family_inference_does_not_bypass_closed_validation(tmp_path, change):
    config = fixture(tmp_path)
    with pytest.raises(ValueError):
        hdf5_tasks.validate_config(config | change, TASK)


def test_empty_vocabulary_placeholder_applies_only_to_native_vit():
    config = {"model_type": "vit", "architectures": ["ViTForImageClassification"], "vocab_size": {}}
    validate_model_config(config)
    for change in [
        {"model_type": "bert"},
        {"vocab_size": {"hidden_size": 0}},
        {"vocab_size": 0},
        {"auto_map": {"AutoModel": "repo.Model"}},
    ]:
        with pytest.raises(ValueError):
            validate_model_config(config | change)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_native_vit_placeholder_preserves_complete_classifier_execution(tmp_path, backend):
    from PIL import Image
    from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    torch.manual_seed(6184)
    config = ViTConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=16,
        patch_size=4,
        id2label={0: "cat", 1: "dog"},
        label2id={"cat": 0, "dog": 1},
    )
    native = ViTForImageClassification(config).eval()
    native.save_pretrained(tmp_path)
    processor = ViTImageProcessor(size={"height": 16, "width": 16})
    processor.save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    original = json.loads(path.read_text())
    path.write_text(json.dumps(original | {"vocab_size": {}}))
    image = Image.fromarray(np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3))
    image.save(tmp_path / "image.png")
    from oryn_engine.runners import (
        RunnerRegistry,
        TransformersTaskHandle,
        _normalize_transformers_task_output,
    )

    validate_model_config(original | {"vocab_size": {}})
    handle = RunnerRegistry()._load_single(
        {
            "model": {"provider": "huggingface", "repository": str(tmp_path), "revision": "a" * 40},
            "runner": "transformers",
            "cacheRoot": str(tmp_path),
            "technicalCapabilities": {
                "task": "image-classification",
                "architectures": ["ViTForImageClassification"],
            },
            "_runtimeCandidate": {"backend": backend},
        }
    )
    assert isinstance(handle, TransformersTaskHandle) and handle.backend == backend
    inputs = {
        k: v.to(backend) for k, v in handle.processor(images=image, return_tensors="pt").items()
    }
    with torch.inference_mode():
        result = _normalize_transformers_task_output(
            "image-classification",
            {"topK": 2},
            handle.model(**inputs),
            handle.processor,
            handle.model,
            inputs,
            image,
        )
    with torch.inference_mode():
        expected = native(**processor(images=image, return_tensors="pt")).logits[0].softmax(-1)
    assert {r["label"] for r in result["predictions"]} == {"cat", "dog"}
    for item in result["predictions"]:
        assert item["score"] == pytest.approx(
            expected[config.label2id[item["label"]]].item(), abs=3e-5
        )


def test_padding_metadata_must_match_the_fixed_bert_tokenizer(tmp_path):
    config = fixture(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(config | {"pad_token_id": 1}))
    with pytest.raises(ValueError, match="HDF5_TOKENIZER_SPECIAL_TOKENS"):
        hdf5_tasks.load_components(tmp_path, TASK, "cpu")


@pytest.mark.parametrize(
    "task,family,architecture,workload",
    [
        ("question-answering", "electra", "ElectraForQuestionAnswering", "question-answering"),
        ("translation", "marian", "MarianMTModel", "seq2seq"),
    ],
)
def test_basic_native_routes_and_explicit_shared_routes_remain_distinct(
    tmp_path, task, family, architecture, workload
):
    from types import SimpleNamespace

    from transformers import ElectraTokenizer

    from oryn_engine.runners import RunnerRegistry, SharedTaskPackHandle, TransformersTaskHandle

    source = FIXTURE.parent / family
    model = hdf5_tasks.load_native(source, task)
    # The independent TF fixture deliberately perturbs position arrays too.
    # Retain them in this synthetic Safetensors export for numerical comparison.
    if family == "marian":
        model._keys_to_ignore_on_save = []
    model.save_pretrained(tmp_path)
    if family == "electra":
        ElectraTokenizer(vocab={"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}).save_pretrained(
            tmp_path
        )
    else:
        from oryn_engine.hdf5_marian import load_tokenizer

        load_tokenizer(source, json.loads((source / "config.json").read_text())).save_pretrained(
            tmp_path
        )
    record = {
        "model": {"provider": "huggingface", "repository": str(tmp_path), "revision": "a" * 40},
        "runner": "transformers",
        "cacheRoot": str(tmp_path),
        "technicalCapabilities": {"task": task, "architectures": [architecture]},
        "_runtimeCandidate": {"backend": "cpu"},
    }
    registry = RunnerRegistry()
    registry._pack_locator = SimpleNamespace(resolve=lambda _: pytest.fail("Native route changed"))
    handle = registry._load_single(record)
    assert isinstance(handle, TransformersTaskHandle) and handle.task == workload
    with np.load(source / "reference.npz", allow_pickle=False) as ref, torch.inference_mode():
        keys = [
            "input_ids",
            "attention_mask",
            "decoder_input_ids" if family == "marian" else "token_type_ids",
        ]
        result = handle.model(**{k: torch.tensor(ref[k], dtype=torch.long) for k in keys})
        for field in ["logits"] if family == "marian" else ["start_logits", "end_logits"]:
            torch.testing.assert_close(
                getattr(result, field), torch.tensor(ref[field]), rtol=4e-4, atol=4e-5
            )
    runtime = object()
    registry._pack_locator = SimpleNamespace(resolve=lambda _: runtime)
    shared = registry._load_single(
        record
        | {
            "runtimePack": {
                "packId": "oryn.transformers-pytorch",
                "runnerId": "transformers.shared-tasks",
            }
        }
    )
    assert (
        isinstance(shared, SharedTaskPackHandle)
        and shared.runtime is runtime
        and shared.task == task
    )
