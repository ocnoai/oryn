"""Independent TF translation graphs and fixed bilingual processor regression."""

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from oryn_engine import hdf5_marian, hdf5_tasks, shared_tasks
from oryn_engine.model_layouts import assess_layout

FIXTURE = Path(__file__).parent / "fixtures/hdf5/marian"
TASK = "translation"


def fixture(root):
    shutil.copytree(FIXTURE, root, dirs_exist_ok=True)
    return json.loads((root / "config.json").read_text())


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_encoder_decoder_and_logits_match_independent_tensorflow(tmp_path, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    fixture(tmp_path)
    model, _, _ = shared_tasks.load_task(tmp_path, TASK, backend)
    with np.load(tmp_path / "reference.npz", allow_pickle=False) as ref, torch.inference_mode():
        output = model(
            **{
                k: torch.tensor(ref[k], device=backend, dtype=torch.long)
                for k in ["input_ids", "attention_mask", "decoder_input_ids"]
            }
        )
        torch.testing.assert_close(
            output.logits.cpu(), torch.tensor(ref["logits"]), rtol=4e-4, atol=4e-5
        )
        torch.testing.assert_close(
            output.encoder_last_hidden_state.cpu(),
            torch.tensor(ref["encoder"]),
            rtol=4e-4,
            atol=4e-5,
        )
    assert model.lm_head.weight.data_ptr() == model.model.shared.weight.data_ptr()


@pytest.mark.parametrize(
    "part",
    ["shared", "encoder", "decoder", "cross_attention", "positions", "bias", "extra", "shape"],
)
def test_missing_or_incompatible_graphs_cannot_create_weights(tmp_path, part):
    fixture(tmp_path)
    with h5py.File(tmp_path / "tf_model.h5", "r+") as f:
        names = []
        f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
        if part == "extra":
            f.create_dataset("rogue/kernel:0", data=np.zeros((2, 3), np.float32))
        else:
            fragment = {
                "shared": "model.shared/embeddings",
                "encoder": "encoder/layers.0/self_attn/q_proj/kernel",
                "decoder": "decoder/layers.0/self_attn/q_proj/kernel",
                "cross_attention": "decoder/layers.0/encoder_attn/q_proj/kernel",
                "positions": "encoder/embed_positions/embeddings",
                "bias": "final_logits_bias:0",
                "shape": "final_logits_bias:0",
            }[part]
            name = next(n for n in names if fragment in n)
            del f[name]
            if part == "shape":
                f.create_dataset(name, data=np.zeros((1, 17), np.float32))
    with pytest.raises(ValueError, match="HDF5_WEIGHT_"):
        hdf5_tasks.load_native(tmp_path, TASK)


def test_shared_translation_uses_fixed_pair_and_bounded_owned_generation(tmp_path):
    from oryn_protocol.task_contracts import validate_invocation

    from oryn_engine.task_profiles import profile_for_execution

    config = fixture(tmp_path)
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"num_beams": 100, "do_sample": True})
    )
    value = {"text": "hello world", "maxTokens": 3}
    result = shared_tasks.execute_task(tmp_path, TASK, value, "cpu", None, None, lambda *_: None)
    shared_tasks.validate_task_output(TASK, result)
    model, tokenizer, _ = shared_tasks.load_task(tmp_path, TASK, "cpu")
    assert result["text"] == tokenizer.decode([8, 8], skip_special_tokens=True).strip()
    assert model.generation_config.num_beams == config.get("num_beams", 1)
    assert model.generation_config.do_sample is False and tokenizer.chat_template is None
    profile = profile_for_execution({"technicalCapabilities": {"task": TASK}}, config)
    assert profile["parameters"]["maxTokens"]["default"] == 63
    with pytest.raises(ValueError):
        validate_invocation(profile, value | {"maxTokens": 64})
    with pytest.raises(ValueError, match="TOKEN_LIMIT"):
        hdf5_marian.translate(model, tokenizer, config, value | {"maxTokens": 64}, "cpu")
    with pytest.raises(ValueError, match="TOKEN_LIMIT"):
        hdf5_marian.translate(model, tokenizer, config, value | {"text": "hello " * 70}, "cpu")
    with pytest.raises(ValueError, match="LANGUAGE_PAIR_FIXED"):
        hdf5_marian.translate(model, tokenizer, config, value | {"targetLanguage": "en"}, "cpu")
    assert TASK not in shared_tasks.IMAGE_TASKS
    assert "translation-v1" not in shared_tasks.NEGOTIATED_TASK_CONTRACTS


def test_tokenizer_layout_requires_both_native_sentencepiece_graphs(tmp_path):
    config = fixture(tmp_path)
    files = [p.name for p in tmp_path.iterdir()]
    valid = assess_layout(files, config, TASK)
    assert not valid.runtime_reasons and not valid.artifact_reasons
    assert "tf_model.h5" in valid.selected_files
    for missing in ["source.spm", "target.spm", "vocab.json", "tokenizer_config.json"]:
        invalid = assess_layout([name for name in files if name != missing], config, TASK)
        assert "MODEL_TOKENIZER_MISSING" in invalid.artifact_reasons
    safe = assess_layout(files + ["model.safetensors"], config, TASK)
    assert not safe.runtime_reasons and "tf_model.h5" not in safe.selected_files


def test_native_punctuation_normalization_is_required(tmp_path, monkeypatch):
    from transformers import MarianTokenizer

    config = fixture(tmp_path)
    tokenizer = hdf5_marian.load_tokenizer(tmp_path, config)
    assert tokenizer.normalize("“hello”—world…") == '"hello" - world...'
    monkeypatch.setattr(
        MarianTokenizer,
        "_setup_normalizer",
        lambda self: setattr(self, "punc_normalizer", lambda x: x),
    )
    with pytest.raises(ValueError, match="HDF5_MARIAN_TOKENIZER"):
        hdf5_marian.load_tokenizer(tmp_path, config)


@pytest.mark.parametrize(
    "change", [{"target_lang": "de"}, {"source_lang": None}, {"separate_vocabs": True}]
)
def test_ambiguous_language_metadata_is_rejected(change):
    with pytest.raises(ValueError):
        hdf5_marian.validate_tokenizer_metadata({"source_lang": "de", "target_lang": "en"} | change)


@pytest.mark.parametrize(
    "change",
    [
        {"tie_word_embeddings": False},
        {"share_encoder_decoder_embeddings": False},
        {"num_beams": 9},
        {"decoder_vocab_size": 34},
        {"d_model": 33},
        {"bad_words_ids": [[8]]},
    ],
)
def test_unsupported_graph_or_generation_metadata_is_rejected(tmp_path, change):
    config = fixture(tmp_path)
    with pytest.raises(ValueError):
        hdf5_tasks.validate_config(config | change, TASK)
