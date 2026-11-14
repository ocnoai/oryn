"""Caption/VQA graph parity against independent TensorFlow BLIP fixtures."""

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from oryn_engine import hdf5_blip, hdf5_tasks, shared_tasks
from oryn_engine.model_layouts import assess_layout

FIXTURES = Path(__file__).parent / "fixtures/hdf5"
TASKS = {"caption": "image-to-text", "vqa": "visual-question-answering"}


def fixture(root, kind):
    from transformers import BertTokenizer, BlipImageProcessor

    shutil.copytree(FIXTURES / f"blip-{kind}", root, dirs_exist_ok=True)
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "b", "cat"]
    words += [f"token_{i}" for i in range(8, 30)]
    tokenizer = BertTokenizer(vocab={word: i for i, word in enumerate(words)})
    tokenizer.save_pretrained(root)
    BlipImageProcessor(size={"height": 16, "width": 16}).save_pretrained(root)
    return json.loads((root / "config.json").read_text())


@pytest.mark.parametrize("kind", TASKS)
@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_graph_matches_independent_tf_vision_encoder_and_decoder(tmp_path, kind, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    fixture(tmp_path, kind)
    model, processor, _ = shared_tasks.load_task(tmp_path, TASKS[kind], backend)
    assert len(processor.tokenizer) == 32
    assert processor.tokenizer.bos_token_id == 30
    assert processor.tokenizer.convert_tokens_to_ids("[ENC]") == 31
    assert processor.tokenizer.chat_template is None
    with np.load(tmp_path / "reference.npz", allow_pickle=False) as ref, torch.inference_mode():
        ids = torch.tensor(ref["input_ids"], device=backend, dtype=torch.long)
        mask = torch.tensor(ref["attention_mask"], device=backend, dtype=torch.long)
        images = model.vision_model(
            pixel_values=torch.tensor(ref["pixel_values"], device=backend)
        ).last_hidden_state
        torch.testing.assert_close(images.cpu(), torch.tensor(ref["vision"]), atol=4e-5, rtol=4e-4)
        hidden = images
        if kind == "vqa":
            hidden = model.text_encoder(
                input_ids=ids,
                attention_mask=mask,
                encoder_hidden_states=images,
                encoder_attention_mask=torch.ones(images.shape[:2], device=backend),
            ).last_hidden_state
        torch.testing.assert_close(hidden.cpu(), torch.tensor(ref["encoder"]), atol=4e-5, rtol=4e-4)
        logits = model.text_decoder(
            input_ids=ids,
            attention_mask=mask,
            encoder_hidden_states=hidden,
            encoder_attention_mask=torch.ones(hidden.shape[:2], device=backend),
        ).logits
        torch.testing.assert_close(logits.cpu(), torch.tensor(ref["logits"]), atol=4e-5, rtol=4e-4)
    # TF serialized this head independently; tying it would alter its logits.
    assert not torch.equal(
        model.text_decoder.cls.predictions.decoder.weight,
        model.text_decoder.bert.embeddings.word_embeddings.weight,
    )


@pytest.mark.parametrize("kind", TASKS)
@pytest.mark.parametrize("part", ["vision", "cross_attention", "head", "extra", "shape"])
def test_incomplete_or_invalid_graph_is_rejected(tmp_path, kind, part):
    fixture(tmp_path, kind)
    with h5py.File(tmp_path / "tf_model.h5", "r+") as f:
        names = []
        f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
        if part == "extra":
            f.create_dataset("rogue/head:0", data=np.zeros(2, np.float32))
        else:
            key = {
                "vision": "patch_embedding/kernel",
                "cross_attention": "crossattention/self/query/kernel",
                "head": "predictions/decoder/kernel",
                "shape": "position_embeddings/embeddings",
            }[part]
            name = next(n for n in names if key in n)
            del f[name]
            if part == "shape":
                f.create_dataset(name, data=np.zeros((2, 3), np.float32))
    with pytest.raises(ValueError, match="HDF5_WEIGHT_"):
        hdf5_tasks.load_native(tmp_path, TASKS[kind])


@pytest.mark.parametrize("kind", TASKS)
def test_shared_generation_contract_and_token_budget(tmp_path, kind):
    from oryn_protocol.task_contracts import validate_invocation

    from oryn_engine.task_profiles import profile_for_execution

    config = fixture(tmp_path, kind)
    task = TASKS[kind]
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "num_beams": 128})
    )
    image_path = tmp_path / "input.png"
    Image.new("RGB", (24, 20), (20, 80, 200)).save(image_path)
    value = {"imageArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa", "maxTokens": 3}
    if kind == "vqa":
        value["question"] = "a cat"
    result = shared_tasks.execute_task(
        tmp_path, task, value, "cpu", image_path, None, lambda *_: None
    )
    assert result["text"] == "cat cat cat"
    profile = profile_for_execution({"technicalCapabilities": {"task": task}}, config)
    assert profile["parameters"]["maxTokens"]["default"] == 63
    with pytest.raises(ValueError):
        validate_invocation(profile, value | {"maxTokens": 64})
    model, processor, _ = shared_tasks.load_task(tmp_path, task, "cpu")
    assert model.text_decoder.generation_config.num_beams == 1
    assert model.text_decoder.generation_config.do_sample is False
    with pytest.raises(ValueError, match="TOKEN_LIMIT"):
        shared_tasks._vision_text(
            model, processor, config, task, Image.open(image_path), value | {"maxTokens": 64}, "cpu"
        )
    if kind == "vqa":
        with pytest.raises(ValueError, match="TOKEN_LIMIT"):
            shared_tasks._vision_text(
                model,
                processor,
                config,
                task,
                Image.open(image_path),
                value | {"question": "cat " * 65},
                "cpu",
            )
    layout = assess_layout([p.name for p in tmp_path.iterdir()], config, task)
    assert not layout.artifact_reasons and not layout.runtime_reasons
    assert "tf_model.h5" in layout.selected_files


@pytest.mark.parametrize(
    "change", [{"size": 17}, {"resample": 2}, {"do_normalize": False}, {"image_std": [1, 0, 1]}]
)
def test_closed_processor_rejects_incompatible_settings(tmp_path, change):
    config = fixture(tmp_path, "caption")
    raw = json.loads((tmp_path / "preprocessor_config.json").read_text())
    with pytest.raises(ValueError, match="PROCESSOR"):
        hdf5_blip.processor_options(raw | change, config)


def test_mismatched_special_tokens_and_missing_question_encoder_are_rejected(tmp_path):
    config = fixture(tmp_path, "vqa")
    wrong = config | {"text_config": config["text_config"] | {"bos_token_id": 29}}
    with pytest.raises(ValueError, match="TOKENIZER"):
        hdf5_blip.load_processor(tmp_path, wrong)
    with h5py.File(tmp_path / "tf_model.h5", "r+") as f:
        del f["text_encoder"]
    with pytest.raises(ValueError, match="GRAPH_INCOMPLETE"):
        hdf5_tasks.load_native(tmp_path, TASKS["vqa"])
