"""Independent TensorFlow fixtures for shared numeric HDF5 task loading."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from oryn_engine import hdf5_tasks, shared_tasks
from oryn_engine.model_layouts import assess_layout

FIXTURES = Path(__file__).parent / "fixtures/hdf5"
TASKS = {
    "clip": "zero-shot-image-classification",
    "vit": "image-feature-extraction",
    "roberta": "text-classification",
}


def fixture(root, family):
    from transformers import (
        CLIPImageProcessor,
        CLIPProcessor,
        CLIPTokenizer,
        RobertaTokenizer,
        ViTImageProcessor,
    )

    shutil.copytree(FIXTURES / family, root, dirs_exist_ok=True)
    if family == "clip":
        vocab = {"<|startoftext|>": 0, "<pad>": 1, "<|endoftext|>": 2, "a</w>": 3, "b</w>": 4}
        vocab.update({f"extra_{i}": i for i in range(5, 32)})
        tokenizer = CLIPTokenizer(vocab=vocab, merges=[], pad_token="<pad>")
        processor = CLIPImageProcessor(
            size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
        )
        CLIPProcessor(tokenizer=tokenizer, image_processor=processor).save_pretrained(root)
        processor.save_pretrained(root)
    elif family == "vit":
        ViTImageProcessor(size={"height": 16, "width": 16}).save_pretrained(root)
    else:
        vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4}
        vocab.update({chr(97 + i): i + 5 for i in range(27)})
        RobertaTokenizer(vocab=vocab, merges=[]).save_pretrained(root)
    return json.loads((root / "config.json").read_text())


@pytest.mark.parametrize("family", TASKS)
@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_graph_matches_independent_tensorflow(tmp_path, family, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    fixture(tmp_path, family)
    model, _, _ = shared_tasks.load_task(tmp_path, TASKS[family], backend)
    with np.load(tmp_path / "reference.npz", allow_pickle=False) as ref, torch.inference_mode():
        inputs = {
            k: torch.tensor(ref[k], device=backend)
            for k in ("input_ids", "attention_mask", "pixel_values")
            if k in ref
        }
        output = model(**inputs)
        fields = {
            "clip": ["logits_per_image", "text_embeds", "image_embeds"],
            "vit": ["last_hidden_state", "pooler_output"],
            "roberta": ["logits"],
        }[family]
        for key in fields:
            torch.testing.assert_close(
                getattr(output, key).cpu(), torch.tensor(ref[key]), rtol=3e-4, atol=3e-5
            )


@pytest.mark.parametrize("family", TASKS)
@pytest.mark.parametrize("part", ["backbone", "head", "extra", "shape"])
def test_incomplete_or_wrong_native_graph_is_rejected(tmp_path, family, part):
    fixture(tmp_path, family)
    with h5py.File(tmp_path / "tf_model.h5", "r+") as f:
        names = []
        f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
        if part == "extra":
            f.create_dataset("rogue/output:0", data=np.zeros(2, np.float32))
        else:
            fragment = (
                {
                    "clip": "text_projection/kernel",
                    "vit": "pooler/dense/kernel",
                    "roberta": "classifier/out_proj/kernel",
                }[family]
                if part == "head"
                else "embeddings/"
            )
            name = next(n for n in names if fragment in n)
            del f[name]
            if part == "shape":
                f.create_dataset(name, data=np.zeros((2, 3), np.float32))
    with pytest.raises(ValueError, match="HDF5_WEIGHT_"):
        hdf5_tasks.load_native(tmp_path, TASKS[family])


def test_encoder_only_vit_uses_cls_without_creating_a_pooler(tmp_path):
    fixture(tmp_path, "vit")
    with h5py.File(tmp_path / "tf_model.h5", "r+") as f:
        del f["vit/tf_vi_t_model/vit/pooler"]
    model = hdf5_tasks.load_native(tmp_path, TASKS["vit"])
    assert model.pooler is None
    with np.load(tmp_path / "reference.npz", allow_pickle=False) as ref, torch.inference_mode():
        output = model(pixel_values=torch.tensor(ref["pixel_values"]))
        torch.testing.assert_close(
            output.last_hidden_state, torch.tensor(ref["last_hidden_state"]), rtol=3e-4, atol=3e-5
        )


@pytest.mark.parametrize("family", TASKS)
def test_shared_execution_and_layout_preserve_task_semantics(tmp_path, family):
    config = fixture(tmp_path, family)
    task = TASKS[family]
    files = [p.name for p in tmp_path.iterdir()]
    layout = assess_layout(files, config, task)
    assert not layout.artifact_reasons and not layout.runtime_reasons
    assert "tf_model.h5" in layout.selected_files
    safe = assess_layout(files + ["model.safetensors"], config, task)
    assert "tf_model.h5" not in safe.selected_files
    image_path = tmp_path / "input.png"
    Image.new("RGB", (24, 19), (30, 100, 200)).save(image_path)
    value = (
        {"text": "abc"}
        if family == "roberta"
        else {"imageArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa"}
    )
    if family == "clip":
        value["labels"] = ["a", "b"]
    result = shared_tasks.execute_task(
        tmp_path,
        task,
        value,
        "cpu",
        image_path if family != "roberta" else None,
        None,
        lambda *_: None,
    )
    shared_tasks.validate_task_output(task, result)
    if family == "vit":
        assert result["pooling"] == "cls" and result["dimensions"] == 32
        assert sum(x * x for x in result["vector"]) == pytest.approx(1, abs=1e-5)
    else:
        assert {p["label"] for p in result["predictions"]} == (
            {"a", "b"} if family == "clip" else {"negative", "neutral", "positive"}
        )
        assert sum(p["score"] for p in result["predictions"]) == pytest.approx(1)


@pytest.mark.parametrize(
    "family,change",
    [
        ("vit", {"image_size": 10000}),
        ("vit", {"patch_size": 3}),
        ("roberta", {"pad_token_id": 33}),
        ("roberta", {"num_attention_heads": 3}),
        ("clip", {"projection_dim": 0}),
    ],
)
def test_metadata_rejects_incompatible_graphs(family, change):
    config = json.loads((FIXTURES / family / "config.json").read_text())
    with pytest.raises(ValueError):
        hdf5_tasks.validate_config(config | change, TASKS[family])


def test_metadata_path_imports_no_inference_backend():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys; from pathlib import Path; "
                "from oryn_engine.hdf5_tasks import validate_config; "
                f"tasks={TASKS!r}; "
                "[validate_config(json.loads((Path(sys.argv[1])/f/'config.json').read_text()),t) "
                "for f,t in tasks.items()]; "
                "assert not {'torch','transformers','h5py','tensorflow'} & set(sys.modules)"
            ),
            str(FIXTURES),
        ],
        check=True,
    )


def test_roberta_position_budget_and_vit_processor_geometry(tmp_path):
    config = fixture(tmp_path, "roberta")
    model, tokenizer, _ = hdf5_tasks.load_components(tmp_path, TASKS["roberta"], "cpu")
    with pytest.raises(ValueError, match="INPUT_TOKEN_LIMIT"):
        hdf5_tasks.classify(model, tokenizer, config, {"text": "a" * 40, "topK": 3}, "cpu")
    vit = json.loads((FIXTURES / "vit/config.json").read_text())
    with pytest.raises(ValueError, match="PROCESSOR"):
        hdf5_tasks.vit_processor_options(
            {"image_processor_type": "ViTImageProcessor", "size": 17}, vit
        )


def test_legacy_clip_eos_preserves_native_pooling_and_rejects_added_vocabulary():
    from transformers import CLIPConfig

    from oryn_engine.flax_weights import validate_clip_config, validate_clip_tokenizer

    config = json.loads((FIXTURES / "original-clip-config.json").read_text())
    validate_clip_config(config)
    native = CLIPConfig.from_dict(json.loads(json.dumps(config)))
    assert native.text_config.eos_token_id == 49407
    assert native.text_config.vocab_size == 49408
    for field, value in [("eos_token_id", 3), ("vocab_size", 49409), ("hidden_size", 64)]:
        changed = config | {"text_config": config["text_config"] | {field: value}}
        with pytest.raises(ValueError):
            validate_clip_config(changed)

    class Tokenizer:
        eos_token_id = 49407

        def __len__(self):
            return 49408

    validate_clip_tokenizer(Tokenizer(), config)
    wrong = Tokenizer()
    wrong.eos_token_id = 2
    with pytest.raises(ValueError, match="TOKENIZER_INCOMPATIBLE"):
        validate_clip_tokenizer(wrong, config)

    class AddedTokens(Tokenizer):
        def __len__(self):
            return 49409

    with pytest.raises(ValueError, match="TOKENIZER_INCOMPATIBLE"):
        validate_clip_tokenizer(AddedTokens(), config)
    # Native legacy argmax and explicit EOS pooling select identical features,
    # including right padding with the EOS token as real CLIP tokenizers do.
    model = hdf5_tasks.load_native(FIXTURES / "clip", TASKS["clip"])
    ids = torch.tensor([[0, 7, 31, 1], [0, 9, 10, 31], [0, 31, 31, 31]])
    with torch.inference_mode():
        model.text_model.eos_token_id = 2
        before = model.text_model(input_ids=ids).pooler_output
        model.text_model.eos_token_id = 31
        after = model.text_model(input_ids=ids).pooler_output
    torch.testing.assert_close(before, after)
