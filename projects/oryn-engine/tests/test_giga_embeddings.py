import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from oryn_engine import bidirectional_embeddings, giga_embeddings, shared_tasks


def originals():
    root = Path(__file__).parent / "fixtures/giga_embeddings"
    return {p.relative_to(root).as_posix(): json.loads(p.read_text()) for p in root.rglob("*.json")}


def test_sentence_similarity_alias_uses_the_same_owned_metadata_normalization():
    from oryn_engine.generated_capabilities import normalized_transformers_config

    config = originals()["config.json"]
    for task in ("feature-extraction", "sentence-similarity"):
        result = normalized_transformers_config(config, task)
        assert "auto_map" not in result
        assert result["model_type"] == "qwen3_bidirec"


@pytest.fixture
def tiny(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import Qwen3Config, Qwen3Model, TokenizersBackend

    data = originals()
    config = data["config.json"]
    config.update(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=128,
        max_window_layers=2,
        layer_types=["full_attention"] * 2,
    )
    vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "blue": 3,
        "river": 4,
        "sky": 5,
        "the": 6,
        "is": 7,
        "Query": 8,
        ":": 9,
    }
    vocab.update({f"extra_{i}": i for i in range(10, 64)})
    graph = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    graph.pre_tokenizer = pre_tokenizers.Whitespace()
    graph.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 1)]
    )
    tokenizer = TokenizersBackend(
        tokenizer_object=graph,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="</s>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(tmp_path)
    native_config = Qwen3Config(
        **{k: v for k, v in config.items() if k not in {"model_type", "auto_map", "architectures"}}
    )
    native_config._attn_implementation = "eager"
    torch.manual_seed(61610)
    native = Qwen3Model(native_config).eval()
    for layer in native.layers:
        layer.self_attn.is_causal = False
    save_file(
        {"model." + k: v.detach().clone() for k, v in native.state_dict().items()},
        tmp_path / "model.safetensors",
    )
    data["1_Pooling/config.json"]["word_embedding_dimension"] = 16
    for name, value in data.items():
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "modeling_gigarembed.py").write_text("raise RuntimeError('must remain inert')")
    (tmp_path / "chat_template.jinja").write_text("{{ raise_exception('must remain inert') }}")
    return tmp_path, config, native, tokenizer


def reference_hidden(model, ids, mask):
    # Independently reproduce the reviewed upstream layer loop, including absolute
    # positions and a full padding mask. Do not use the adapter's mask helper.
    hidden = model.embed_tokens(ids)
    positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    rotary = model.rotary_emb(hidden, positions)
    padding = torch.zeros((ids.shape[0], 1, ids.shape[1], ids.shape[1]), device=ids.device)
    padding.masked_fill_(~mask[:, None, None, :].bool(), torch.finfo(hidden.dtype).min)
    for layer in model.layers:
        hidden = layer(
            hidden,
            attention_mask=padding,
            position_ids=positions,
            position_embeddings=rotary,
            use_cache=False,
        )
    return model.norm(hidden)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("normalize", [False, True])
def test_native_bidirectional_graph_pooling_padding_and_normalization(tiny, backend, normalize):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    root, config, reference, original = tiny
    model, tokenizer, selected = shared_tasks.load_task(root, "feature-extraction", backend)
    assert all(layer.self_attn.is_causal is False for layer in model.layers)
    assert tokenizer.chat_template is None
    for key, tensor in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key].cpu(), tensor, rtol=0, atol=0)
    texts = ["Query: blue", "the river is blue"]
    with torch.inference_mode():
        expected = []
        for text in texts:
            tokens = original(text, return_tensors="pt")
            hidden = reference_hidden(reference, tokens["input_ids"], tokens["attention_mask"])
            vector = hidden[0].mean(0)
            expected.append(vector / vector.norm() if normalize else vector)
        result = bidirectional_embeddings.execute(
            model, tokenizer, selected, {"texts": texts, "normalize": normalize}, backend
        )
        torch.testing.assert_close(
            torch.tensor(result["embeddings"]), torch.stack(expected), atol=2e-5, rtol=2e-4
        )
        ids = torch.tensor([[1, 3, 4], [1, 3, 5]], device=backend)
        output = model(
            input_ids=ids,
            attention_mask=giga_embeddings.attention_mask(torch.ones_like(ids)),
            use_cache=False,
        ).last_hidden_state
        assert not torch.allclose(output[0, 0], output[1, 0], atol=1e-6, rtol=1e-6)
    shared_tasks.validate_task_output("feature-extraction", result)


@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "nonfinite", "mixed-prefix"])
def test_complete_numeric_graph_required(tiny, damage):
    root, _, _, _ = tiny
    path = root / "model.safetensors"
    state = load_file(path)
    key = "model.layers.0.self_attn.q_norm.weight"
    if damage == "missing":
        del state[key]
    elif damage == "extra":
        state["unknown.weight"] = torch.ones(1)
    elif damage == "shape":
        state[key] = torch.ones(1)
    elif damage == "mixed-prefix":
        state[key.removeprefix("model.")] = state.pop(key)
    else:
        state[key][0] = float("nan")
    save_file(state, path)
    with pytest.raises(ValueError, match="GIGA_EMBEDDING_WEIGHTS"):
        shared_tasks.load_task(root, "feature-extraction", "cpu")


@pytest.mark.parametrize(
    "change",
    [
        {"auto_map": {"AutoModel": "unknown.Code"}},
        {"is_causal": True},
        {"use_cache": True},
        {"layer_types": ["sliding_attention"] * 28},
        {"rope_scaling": {"type": "dynamic", "factor": 2}},
        {"head_dim": 63},
        {"num_key_value_heads": 3},
        {"unreviewed_head": True},
        {"quantization_config": {"bits": 4}},
        {"attention_bias": True},
    ],
)
def test_closed_metadata_graph(change):
    with pytest.raises(ValueError):
        giga_embeddings.native_config(originals()["config.json"] | change)


@pytest.mark.parametrize("damage", [None, "pooling", "prompt", "tokenizer", "remote-code"])
def test_metadata_admission_requires_owned_semantics_without_weights(tmp_path, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore, ModelStoreError
    from oryn_engine.runtime_packs.execution import PackRuntime

    data = originals()
    files = set(data) | {"model.safetensors", "tokenizer.json"}
    if damage == "pooling":
        files.remove("1_Pooling/config.json")
    elif damage == "prompt":
        data["config_sentence_transformers.json"]["default_prompt_name"] = "query"
    elif damage == "tokenizer":
        data["tokenizer_config.json"]["tokenizer_class"] = "RepositoryCode"
    elif damage == "remote-code":
        data["config.json"]["auto_map"]["AutoModel"] = "unknown.Code"

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return data[filename]

    hub = Hub(
        config=data["config.json"],
        pipeline_tag="feature-extraction",
        files=[(n, 100, "b" * 64) for n in files],
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    request = {"provider": "huggingface", "repository": "test/giga", "revision": "main"}
    if damage == "remote-code":
        with pytest.raises(ModelStoreError):
            store.inspect(request)
    else:
        result = store.inspect(request)
        assert result["executable"] is (damage is None), result
        if not damage:
            assert result["taskProfile"]["limits"] == {
                "batchItems": 4,
                "textCharacters": 8192,
                "embeddingDimensions": 1024,
            }
    assert not hub.downloaded


def test_owned_limits_and_public_task_route(tiny):
    root, _, _, _ = tiny
    result = shared_tasks.execute_task(
        root, "feature-extraction", {"texts": ["blue"]}, "cpu", None, None, lambda *_: None
    )
    assert result["dimensions"] == 16 and result["normalized"]
    model, tokenizer, config = shared_tasks.load_task(root, "feature-extraction", "cpu")
    for texts, error in ((["blue"] * 5, "BATCH_LIMIT"), (["blue " * 129], "TOKEN_LIMIT")):
        with pytest.raises(ValueError, match=error):
            bidirectional_embeddings.execute(
                model, tokenizer, config, {"texts": texts, "normalize": True}, "cpu"
            )
