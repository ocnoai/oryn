"""Complete native Jina-v3 causal listwise ranking with owned prompt construction."""

from __future__ import annotations

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

DOC_TOKEN, QUERY_TOKEN = "<|embed_token|>", "<|rerank_token|>"
DOC_ID, QUERY_ID = 151670, 151671
MAX_CONTEXT = 4096
SYSTEM = (
    "You are a search relevance expert who can determine a ranking of the passages "
    "based on how relevant they are to the query. "
    "If the query is a question, how relevant a passage is depends on how well "
    "it answers the question. "
    "If not, try to analyze the intent of the query and assess how well each passage "
    "satisfies the intent. "
    "If an instruction is provided, you should follow the instruction when determining the ranking."
)


def supports(config, task):
    return (
        task == "text-ranking"
        and config.get("model_type") == "qwen3"
        and config.get("architectures") == ["JinaForRanking"]
    )


def normalized_config(config, task):
    if supports(config, task) and config.get("auto_map") == {
        "AutoModel": "modeling.JinaForRanking"
    }:
        config = {k: v for k, v in config.items() if k != "auto_map"}
        validate_config(config)
    return config


def validate_config(config):
    validate_model_config(config)
    if not supports(config, "text-ranking") or any(
        config.get(k) is not None for k in ("quantization", "quantization_config")
    ):
        raise ValueError("JINA_RANKING_CONFIGURATION")
    for key, low, high in (
        ("hidden_size", 16, 8192),
        ("intermediate_size", 16, 32768),
        ("num_hidden_layers", 1, 80),
        ("num_attention_heads", 1, 128),
        ("num_key_value_heads", 1, 128),
        ("head_dim", 4, 256),
        ("vocab_size", QUERY_ID + 1, 262144),
        ("max_position_embeddings", 4096, 1048576),
    ):
        if type(config.get(key)) is not int or not low <= config[key] <= high:
            raise ValueError("JINA_RANKING_DIMENSIONS")
    if (
        config["hidden_size"] % 2
        or config["head_dim"] % 2
        or config["num_attention_heads"] % config["num_key_value_heads"]
        or config.get("attention_bias", False) is not False
        or config.get("attention_dropout", 0.0) != 0
        or config.get("hidden_act", "silu") != "silu"
        or config.get("sliding_window") is not None
        or config.get("use_sliding_window", False) is not False
        or config.get("rope_scaling") is not None
        or config.get("layer_types", ["full_attention"] * config["num_hidden_layers"])
        != ["full_attention"] * config["num_hidden_layers"]
    ):
        raise ValueError("JINA_RANKING_GRAPH")
    rope = config.get("rope_parameters")
    if rope is not None and (
        not isinstance(rope, dict)
        or set(rope) - {"rope_type", "rope_theta"}
        or rope.get("rope_type", "default") != "default"
    ):
        raise ValueError("JINA_RANKING_POSITIONS")


def native_model(config):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    native = Qwen3Config.from_dict(dict(config))
    native._attn_implementation = "eager"
    model = Qwen3ForCausalLM(native)
    model.add_module(
        "projector",
        torch.nn.Sequential(
            torch.nn.Linear(config["hidden_size"], config["hidden_size"] // 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(config["hidden_size"] // 2, 512, bias=False),
        ),
    )
    return model


def validate_weight_shapes(config, shapes):
    import torch

    validate_config(config)
    with torch.device("meta"):
        expected = {k: tuple(v.shape) for k, v in native_model(config).state_dict().items()}
    if config.get("tie_word_embeddings") is True and "lm_head.weight" not in shapes:
        expected.pop("lm_head.weight")
    if shapes != expected:
        raise ValueError("JINA_RANKING_WEIGHTS_INCOMPLETE")


def validate_metadata(name, metadata, config):
    from oryn_engine.qwen_retrieval import validate_metadata as validate_tokenizer

    if name != "tokenizer_config.json":
        raise ValueError("JINA_RANKING_METADATA_UNSUPPORTED")
    validate_tokenizer(name, metadata, config)


def load_tokenizer(root, config):
    from transformers import Qwen2Tokenizer

    validate_metadata(
        "tokenizer_config.json", metadata_file(root / "tokenizer_config.json"), config
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    tokenizer.chat_template = None
    vocab = tokenizer.get_vocab()
    if not QUERY_ID < len(vocab) <= config["vocab_size"] or set(vocab.values()) != set(
        range(len(vocab))
    ):
        raise ValueError("JINA_RANKING_VOCABULARY")
    for token, expected in ((DOC_TOKEN, DOC_ID), (QUERY_TOKEN, QUERY_ID)):
        if vocab.get(token) != expected or tokenizer.encode(token, add_special_tokens=False) != [
            expected
        ]:
            raise ValueError("JINA_RANKING_VOCABULARY")
    for token in ("<|im_start|>", "<|im_end|>", "<think>", "</think>"):
        if token not in vocab or tokenizer.encode(token, add_special_tokens=False) != [
            vocab[token]
        ]:
            raise ValueError("JINA_RANKING_FRAMING")
    return tokenizer


def load_components(root, config, backend):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    validate_weight_shapes(config, preflight_safetensors(root))
    tokenizer = load_tokenizer(root, config)
    state = {}
    for path in sorted(root.glob("*.safetensors")):
        part = load_file(path)
        if state.keys() & part.keys() or any(not v.is_floating_point() for v in part.values()):
            raise ValueError("JINA_RANKING_WEIGHTS_INVALID")
        part = {k: v.float() for k, v in part.items()}
        if any(not bool(torch.isfinite(v).all()) for v in part.values()):
            raise ValueError("JINA_RANKING_WEIGHTS_INVALID")
        state.update(part)
    if config.get("tie_word_embeddings") is True:
        embedding = state["model.embed_tokens.weight"]
        if "lm_head.weight" in state and not torch.equal(state["lm_head.weight"], embedding):
            raise ValueError("JINA_RANKING_TIED_WEIGHTS")
        state["lm_head.weight"] = embedding
    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    model.to(backend).eval()
    return model, tokenizer, config


def prompt(query, documents, instruction):
    reserved = (
        DOC_TOKEN,
        QUERY_TOKEN,
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<|endoftext|>",
    )
    if any(token in text for text in [query, *documents, instruction] for token in reserved):
        raise ValueError("JINA_RANKING_RESERVED_FRAMING")
    user = (
        f"I will provide you with {len(documents)} passages, "
        "each indicated by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
        f"<instruct>\n{instruction}\n</instruct>\n"
        + "\n".join(
            f'<passage id="{i}">\n{doc}{DOC_TOKEN}\n</passage>' for i, doc in enumerate(documents)
        )
        + f"\n<query>\n{query}{QUERY_TOKEN}\n</query>"
    )
    return (
        "<|im_start|>system\n"
        + SYSTEM
        + "<|im_end|>\n<|im_start|>user\n"
        + user
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def execute(model, tokenizer, config, value, backend):
    import torch

    docs, query = value["documents"], value["query"]
    if (
        not 1 <= len(docs) <= 64
        or len(tokenizer.encode(query, add_special_tokens=False)) > 512
        or any(len(tokenizer.encode(doc, add_special_tokens=False)) > 2048 for doc in docs)
    ):
        raise ValueError("JINA_RANKING_INPUT_LIMIT")
    inputs = tokenizer(
        prompt(query, docs, value["instruction"]),
        add_special_tokens=False,
        truncation=False,
        return_tensors="pt",
    )
    if inputs["input_ids"].shape[-1] > min(MAX_CONTEXT, config["max_position_embeddings"]):
        raise ValueError("JINA_RANKING_CONTEXT_LIMIT")
    inputs = {k: v.to(backend) for k, v in inputs.items()}
    ids = inputs["input_ids"]
    if int((ids == DOC_ID).sum()) != len(docs) or int((ids == QUERY_ID).sum()) != 1:
        raise ValueError("JINA_RANKING_MARKER_GEOMETRY")
    # The native reference replaces lm_head with Identity; read the same final
    # hidden states directly while preserving every checkpoint module and weight.
    hidden = model.model(**inputs, use_cache=False).last_hidden_state
    doc_vectors = model.projector(hidden[ids == DOC_ID]).float()
    query_vector = model.projector(hidden[ids == QUERY_ID]).float()
    scores = torch.nn.functional.cosine_similarity(
        doc_vectors, query_vector.expand_as(doc_vectors), dim=-1
    )
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("JINA_RANKING_OUTPUT_INVALID")
    results = [{"index": i, "score": score} for i, score in enumerate(scores.cpu().tolist())]
    results.sort(key=lambda item: (-item["score"], item["index"]))
    return {"results": results[: value.get("topK", len(results))]}
