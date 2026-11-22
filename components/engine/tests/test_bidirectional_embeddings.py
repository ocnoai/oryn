import json

import pytest
import torch

from oryn_engine import bidirectional_embeddings as binding
from oryn_engine import shared_tasks


def tiny_model(root):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import Ministral3Config, Ministral3Model, TokenizersBackend

    vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<pad>": 3,
        "query": 4,
        "passage": 5,
        ":": 6,
        "blue": 7,
        "river": 8,
        "sky": 9,
        "the": 10,
        "is": 11,
    }
    vocab.update({f"extra_{i}": i for i in range(12, 64)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 1)]
    )
    processor = TokenizersBackend(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    processor.save_pretrained(root)
    config = Ministral3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=256,
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
        is_causal=False,
        pooling="avg",
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 4.0,
            "original_max_position_embeddings": 64,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": 0.1,
        },
    )
    torch.manual_seed(61)
    native = Ministral3Model(config).eval()
    native.save_pretrained(root)
    (root / "1_Pooling").mkdir()
    pooling = {
        "word_embedding_dimension": 16,
        "pooling_mode_cls_token": False,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": False,
        "pooling_mode_lasttoken": False,
        "include_prompt": True,
    }
    (root / "1_Pooling/config.json").write_text(json.dumps(pooling))
    (root / "config_sentence_transformers.json").write_text(
        json.dumps(
            {
                "prompts": {"query": "query: ", "document": "passage: "},
                "default_prompt_name": None,
                "similarity_fn_name": "cosine",
            }
        )
    )
    (root / "chat_template.jinja").write_text("{{ raise_exception('not used') }}")
    return native, processor


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("normalize", [False, True])
def test_mean_pooling_matches_unpadded_native_rows_and_bidirectional_attention(
    tmp_path, backend, normalize
):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    native, original = tiny_model(tmp_path)
    model, tokenizer, config = shared_tasks.load_task(tmp_path, "feature-extraction", backend)
    texts = ["query: blue", "passage: the river is blue"]
    expected = []
    with torch.inference_mode():
        for text in texts:
            inputs = original(text, return_tensors="pt")
            inputs = {k: inputs[k] for k in ("input_ids", "attention_mask")}
            hidden = native(**inputs, use_cache=False).last_hidden_state[0]
            vector = hidden.mean(0)
            expected.append(vector / vector.norm() if normalize else vector)
        result = binding.execute(
            model, tokenizer, config, {"texts": texts, "normalize": normalize}, backend
        )
        torch.testing.assert_close(
            torch.tensor(result["embeddings"]), torch.stack(expected), atol=2e-5, rtol=2e-4
        )
        # Changing a future token must affect the first token in this encoder.
        ids = torch.tensor([[1, 7, 8], [1, 7, 9]], device=backend)
        hidden = model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
        ).last_hidden_state
        assert not torch.allclose(hidden[0, 0], hidden[1, 0], atol=1e-6, rtol=1e-6)
    assert tokenizer.chat_template is None
    shared_tasks.validate_task_output("feature-extraction", result)


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape"])
def test_complete_native_graph_required(tmp_path, mutation):
    from safetensors.torch import load_file, save_file

    tiny_model(tmp_path)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    if mutation == "missing":
        del state["layers.0.self_attn.q_proj.weight"]
    elif mutation == "extra":
        state["unreviewed_head.weight"] = torch.zeros(2, 2)
    else:
        state["layers.0.self_attn.q_proj.weight"] = torch.zeros(2, 2)
    save_file(state, path)
    with pytest.raises((ValueError, RuntimeError)):
        shared_tasks.load_task(tmp_path, "feature-extraction", "cpu")


@pytest.mark.parametrize(
    "change",
    [
        {"is_causal": True},
        {"pooling": "last"},
        {"num_key_value_heads": 2},
        {"architectures": ["Ministral3ForCausalLM"]},
        {"auto_map": {"AutoModel": "repository.Model"}},
    ],
)
def test_rejects_causal_or_unreviewed_graph(tmp_path, change):
    tiny_model(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    with pytest.raises(ValueError):
        binding.validate_config(config | change)


def test_metadata_runtime_and_profile_share_the_same_limits(tmp_path):
    from oryn_protocol.task_contracts import validate_invocation

    from oryn_engine.model_layouts import assess_layout
    from oryn_engine.task_profiles import profile_for_execution

    tiny_model(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    files = [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()]
    layout = assess_layout(files, config, "feature-extraction")
    assert not layout.runtime_reasons and not layout.artifact_reasons
    missing = assess_layout(
        [p for p in files if p != "1_Pooling/config.json"], config, "feature-extraction"
    )
    assert missing.artifact_reasons
    profile = profile_for_execution(
        {
            "technicalCapabilities": {"task": "feature-extraction"},
            "runtimePack": {"runnerId": "transformers.shared-tasks"},
        },
        config,
    )
    assert profile["limits"] == {"batchItems": 4, "textCharacters": 8192, "embeddingDimensions": 16}
    with pytest.raises(ValueError):
        validate_invocation(profile, {"texts": ["blue"] * 5})
    value = {"texts": ["query: blue", "passage: blue"]}
    result = shared_tasks.execute_task(
        tmp_path, "feature-extraction", value, "cpu", None, None, lambda *_: None
    )
    assert result["dimensions"] == 16 and result["normalized"]
    model, tokenizer, config = shared_tasks.load_task(tmp_path, "feature-extraction", "cpu")
    with pytest.raises(ValueError, match="TOKEN_LIMIT"):
        binding.execute(
            model, tokenizer, config, {"texts": ["blue " * 257], "normalize": True}, "cpu"
        )
    with pytest.raises(ValueError, match="BATCH_LIMIT"):
        binding.execute(model, tokenizer, config, {"texts": ["blue"] * 5, "normalize": True}, "cpu")


@pytest.mark.parametrize(
    "name,change",
    [
        ("1_Pooling/config.json", {"pooling_mode_cls_token": True}),
        ("1_Pooling/config.json", {"include_prompt": 1}),
        ("config_sentence_transformers.json", {"default_prompt_name": "query"}),
        ("tokenizer_config.json", {"tokenizer_class": "RepositoryTokenizer"}),
    ],
)
def test_metadata_does_not_silently_change_embeddings(tmp_path, name, change):
    tiny_model(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    metadata = json.loads((tmp_path / name).read_text())
    with pytest.raises(ValueError):
        binding.validate_metadata(name, metadata | change, config)
