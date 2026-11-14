"""Native listwise scoring, exact checkpoint geometry and metadata-only routing."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from oryn_engine import jina_ranking as ranking
from oryn_engine import shared_tasks
from oryn_engine.generated_capabilities import normalized_transformers_config
from oryn_engine.model_metadata import validate_model_config


def originals():
    root = Path(__file__).parent / "fixtures/jina_ranking"
    return {p.name: json.loads(p.read_text()) for p in root.glob("*.json")}


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3ForCausalLM

    root = tmp_path_factory.mktemp("jina-ranking-template")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    vocab = {token: i for i, token in enumerate(sorted(ByteLevel.alphabet()))}
    markers = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    for token in markers:
        vocab[token] = len(vocab)
    vocab.update({f"unused_{i}": i for i in range(len(vocab), 151670)})
    vocab.update({"<|embed_token|>": 151670, "<|rerank_token|>": 151671})
    tokenizer = Qwen2Tokenizer(
        vocab=vocab, merges=[], eos_token="<|endoftext|>", pad_token="<|endoftext|>"
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": markers + ["<|embed_token|>", "<|rerank_token|>"]}
    )
    tokenizer.save_pretrained(root)
    config = originals()["config.json"] | dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=len(vocab),
        max_position_embeddings=4096,
        max_window_layers=2,
    )
    native = Qwen3Config.from_dict({k: v for k, v in config.items() if k != "auto_map"})
    native._attn_implementation = "eager"
    torch.manual_seed(61613)
    model = Qwen3ForCausalLM(native).eval()
    projector = torch.nn.Sequential(
        torch.nn.Linear(16, 8, bias=False), torch.nn.ReLU(), torch.nn.Linear(8, 512, bias=False)
    ).eval()
    state = {k: v.detach().clone() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    state.update({"projector." + k: v.detach().clone() for k, v in projector.state_dict().items()})
    save_file(state, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps(config))
    (root / "modeling.py").write_text("raise RuntimeError('repository code must remain inert')")
    (root / "chat_template.jinja").write_text("{{ raise_exception('must remain inert') }}")
    try:
        yield root, model, projector, tokenizer, config
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def tiny(template, tmp_path):
    root, *rest = template
    shutil.copytree(root, tmp_path, dirs_exist_ok=True)
    return tmp_path, *rest


def reference_prompt(query, documents, instruction):
    # Independent transcription of the reviewed format_docs_prompts_func source.
    passages = [
        f'<passage id="{i}">\n{text}<|embed_token|>\n</passage>' for i, text in enumerate(documents)
    ]
    return (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the passages based "
        "on how relevant they are to the query. If the query is a question, how relevant a "
        "passage is depends on how well it answers the question. If not, try to analyze the "
        "intent of the query and assess how well each passage satisfies the intent. If an "
        "instruction is provided, you should follow the instruction when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
        f"I will provide you with {len(documents)} passages, each indicated by a numerical "
        f"identifier. Rank the passages based on their relevance to query: {query}\n"
        f"<instruct>\n{instruction}\n</instruct>\n"
        + "\n".join(passages)
        + f"\n<query>\n{query}<|rerank_token|>\n</query>"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("count", [1, 3])
def test_complete_native_causal_graph_projector_and_joint_scores(tiny, backend, count):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    root, reference, projector, original, config = tiny
    model, tokenizer, selected = shared_tasks.load_task(root, "text-ranking", backend)
    assert tokenizer.chat_template is None and "auto_map" not in selected
    assert all(layer.self_attn.is_causal for layer in model.model.layers)
    for key, tensor in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key].cpu(), tensor, rtol=0, atol=0)
    for key, tensor in projector.state_dict().items():
        torch.testing.assert_close(model.projector.state_dict()[key].cpu(), tensor, rtol=0, atol=0)
    value = dict(
        query="Čau, what is blue?", documents=["A blue river", "A red sky", "Water"][:count]
    )
    value = shared_tasks.validate_task_input("text-ranking", value)
    text = reference_prompt(value["query"], value["documents"], value["instruction"])
    assert ranking.prompt(value["query"], value["documents"], value["instruction"]) == text
    inputs = original(text, add_special_tokens=False, return_tensors="pt")
    with torch.inference_mode():
        hidden = reference(
            **inputs, use_cache=False, output_hidden_states=True, logits_to_keep=1
        ).hidden_states[-1]
        docs = projector(hidden[inputs["input_ids"] == 151670])
        query = projector(hidden[inputs["input_ids"] == 151671])
        expected = torch.nn.functional.cosine_similarity(docs, query.expand_as(docs), dim=-1)
        result = ranking.execute(model, tokenizer, config, value, backend)
    scores = {r["index"]: r["score"] for r in result["results"]}
    torch.testing.assert_close(
        torch.tensor([scores[i] for i in range(count)]), expected, atol=2e-4, rtol=2e-3
    )
    assert result["results"] == sorted(result["results"], key=lambda r: (-r["score"], r["index"]))
    shared_tasks.validate_task_output("text-ranking", result)


def test_shared_worker_preserves_indices_and_top_k(tiny):
    root, *_ = tiny
    value = {"query": "river", "documents": ["water", "sky", "blue"], "topK": 2}
    result = shared_tasks.execute_task(
        root, "text-ranking", value, "cpu", None, None, lambda *_: None
    )
    assert len(result["results"]) == 2
    assert len({r["index"] for r in result["results"]}) == 2
    assert all(0 <= r["index"] < 3 for r in result["results"])


@pytest.mark.parametrize(
    "damage", ["base-missing", "head-missing", "shape", "extra", "nonfinite", "tie"]
)
def test_full_native_graph_and_learned_head_required(tiny, damage):
    root, *_ = tiny
    path = root / "model.safetensors"
    state = load_file(path)
    if damage == "base-missing":
        del state["model.layers.1.self_attn.q_norm.weight"]
    elif damage == "head-missing":
        del state["projector.2.weight"]
    elif damage == "shape":
        state["projector.0.weight"] = torch.ones(1)
    elif damage == "extra":
        state["unreviewed.weight"] = torch.ones(1)
    elif damage == "tie":
        state["lm_head.weight"] = state["model.embed_tokens.weight"] + 1
    else:
        state["projector.2.weight"][0, 0] = float("nan")
    save_file(state, path)
    with pytest.raises(ValueError, match="JINA_RANKING_"):
        shared_tasks.load_task(root, "text-ranking", "cpu")


@pytest.mark.parametrize("damage", ["query", "document", "context", "framing"])
def test_input_is_bounded_without_silent_truncation(tiny, damage):
    root, _, _, _, config = tiny
    model, tokenizer, _ = shared_tasks.load_task(root, "text-ranking", "cpu")
    value = {"query": "blue", "documents": ["river"], "instruction": "Rank the passages."}
    if damage == "query":
        value["query"] = "a" * 513
    elif damage == "document":
        value["documents"] = ["a" * 2049]
    elif damage == "context":
        value["documents"] = ["a" * 1800] * 3
    else:
        value["documents"] = ["river <|rerank_token|>"]
    with pytest.raises(ValueError, match="JINA_RANKING_"):
        ranking.execute(model, tokenizer, config, value, "cpu")


def test_tokenizer_marker_identity_is_checked(tiny):
    root, _, _, tokenizer, config = tiny
    graph = json.loads((root / "tokenizer.json").read_text())
    vocab = graph["model"]["vocab"]
    vocab["<|embed_token|>"], vocab["<|rerank_token|>"] = 151671, 151670
    for item in graph["added_tokens"]:
        if item["content"] in {"<|embed_token|>", "<|rerank_token|>"}:
            item["id"] = vocab[item["content"]]
    (root / "tokenizer.json").write_text(json.dumps(graph))
    with pytest.raises(ValueError, match="JINA_RANKING_VOCABULARY"):
        ranking.load_tokenizer(root, config)


@pytest.mark.parametrize(
    "change",
    [
        {"auto_map": {"AutoModel": "other.Code"}},
        {"auto_map": {"AutoModel": "modeling.JinaForRanking", "AutoTokenizer": "other.Code"}},
        {"attention_bias": True},
        {"rope_scaling": {"type": "dynamic", "factor": 2}},
        {"sliding_window": 1024},
        {"head_dim": 63},
        {"num_key_value_heads": 3},
        {"quantization_config": {"bits": 4}},
    ],
)
def test_unreviewed_native_graph_rejected(change):
    with pytest.raises(ValueError):
        ranking.validate_config(
            normalized_transformers_config(originals()["config.json"] | change, "text-ranking")
        )


def test_reviewed_alias_does_not_admit_other_tasks():
    for task in ("feature-extraction", "text-generation", "zero-shot-classification"):
        with pytest.raises(ValueError):
            validate_model_config(normalized_transformers_config(originals()["config.json"], task))


@pytest.mark.parametrize("missing", [None, "tokenizer_config.json", "tokenizer.json"])
def test_original_metadata_selects_owned_adapter_without_weights(tmp_path, missing):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    data = originals()

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return data[filename]

    names = (set(data) | {"tokenizer.json", "model.safetensors"}) - {missing}
    hub = Hub(
        config=data["config.json"],
        pipeline_tag="text-ranking",
        files=[(n, 100, "b" * 64) for n in names],
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "test/jina-v3", "revision": "main"}
    )
    assert result["executable"] is (missing is None), result
    if missing is None:
        assert result["taskProfile"]["contract"]["id"] == "reranking-v1"
        assert result["taskProfile"]["inputModalities"] == ["text"]
    assert not hub.downloaded
