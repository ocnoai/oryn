"""Bounded mean-pooled bidirectional embeddings over complete native graphs.

The caller's text is preserved, including any query/passage prefixes. Repository
sentence-transformers modules and tokenizer class names never select Python code.
"""

from __future__ import annotations

from oryn_engine.model_metadata import metadata_file, validate_model_config

METADATA_NAMES = (
    "tokenizer_config.json",
    "1_Pooling/config.json",
    "config_sentence_transformers.json",
)
MAX_BATCH = 4
MAX_TOKENS = 4096


def supports(task, config):
    return task == "feature-extraction" and config.get("model_type") in {
        "ministral3",
        "qwen3_bidirec",
    }


def validate_config(config):
    if config.get("model_type") == "qwen3_bidirec":
        from oryn_engine.giga_embeddings import native_config

        native_config(config)
        return
    validate_model_config(config)
    if (
        config.get("model_type") != "ministral3"
        or config.get("architectures") != ["Ministral3Model"]
        or config.get("is_causal") is not False
        or config.get("pooling") != "avg"
        or config.get("sliding_window") is not None
        or config.get("quantization_config")
        or config.get("quantization")
    ):
        raise ValueError("BIDIRECTIONAL_EMBEDDING_CONFIGURATION_UNSUPPORTED")
    for name, low, high in (
        ("hidden_size", 8, 8192),
        ("intermediate_size", 8, 65536),
        ("num_hidden_layers", 1, 80),
        ("num_attention_heads", 1, 64),
        ("num_key_value_heads", 1, 64),
        ("head_dim", 8, 256),
        ("vocab_size", 32, 262144),
        ("max_position_embeddings", 64, 1048576),
    ):
        if type(config.get(name)) is not int or not low <= config[name] <= high:
            raise ValueError("BIDIRECTIONAL_EMBEDDING_CONFIGURATION_LIMIT")
    # Pruned models may have an attention projection wider than the residual stream.
    if config["num_attention_heads"] % config["num_key_value_heads"] or config["head_dim"] % 2:
        raise ValueError("BIDIRECTIONAL_EMBEDDING_ATTENTION_GEOMETRY")
    for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
        if type(config.get(name)) is not int or not 0 <= config[name] < config["vocab_size"]:
            raise ValueError("BIDIRECTIONAL_EMBEDDING_SPECIAL_TOKENS")
    rope = config.get("rope_parameters")
    if not isinstance(rope, dict) or rope.get("rope_type", rope.get("type")) not in {
        "default",
        "yarn",
    }:
        raise ValueError("BIDIRECTIONAL_EMBEDDING_ROPE_UNSUPPORTED")
    if "type" in rope and "rope_type" in rope and rope["type"] != rope["rope_type"]:
        raise ValueError("BIDIRECTIONAL_EMBEDDING_ROPE_UNSUPPORTED")


def validate_metadata(name, value, config):
    validate_model_config(value)
    if name == "1_Pooling/config.json":
        expected = {
            "word_embedding_dimension": config["hidden_size"],
            "pooling_mode_cls_token": False,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_tokens": True,
            "pooling_mode_mean_sqrt_len_tokens": False,
            "pooling_mode_weightedmean_tokens": False,
            "pooling_mode_lasttoken": False,
            "include_prompt": True,
        }
        if set(value) != set(expected) or any(
            type(value[k]) is not type(v) or value[k] != v for k, v in expected.items()
        ):
            raise ValueError("BIDIRECTIONAL_EMBEDDING_POOLING_UNSUPPORTED")
    elif name == "config_sentence_transformers.json":
        from oryn_engine.giga_embeddings import PROMPTS

        if (
            value.get("default_prompt_name") is not None
            or value.get("similarity_fn_name") != "cosine"
            or value.get("prompts")
            != (
                PROMPTS
                if config.get("model_type") == "qwen3_bidirec"
                else {"query": "query: ", "document": "passage: "}
            )
        ):
            raise ValueError("BIDIRECTIONAL_EMBEDDING_PROMPT_UNSUPPORTED")
    elif name == "tokenizer_config.json":
        expected = (
            "PreTrainedTokenizerFast"
            if config.get("model_type") == "qwen3_bidirec"
            else "TokenizersBackend"
        )
        if value.get("tokenizer_class") != expected:
            raise ValueError("BIDIRECTIONAL_EMBEDDING_TOKENIZER_UNSUPPORTED")
        for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
            if not isinstance(value.get(key), str) or not 1 <= len(value[key]) <= 128:
                raise ValueError("BIDIRECTIONAL_EMBEDDING_SPECIAL_TOKENS")


def load_components(root, config, backend):
    import torch
    from transformers import Ministral3Model, TokenizersBackend

    for name in METADATA_NAMES:
        validate_metadata(name, metadata_file(root / name), config)
    path = root / "tokenizer.json"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 32 * 1024**2:
        raise ValueError("BIDIRECTIONAL_EMBEDDING_TOKENIZER_UNAVAILABLE")
    metadata = metadata_file(root / "tokenizer_config.json")
    tokenizer = TokenizersBackend(
        tokenizer_file=str(path),
        padding_side="left",
        **{k: metadata[k] for k in ("bos_token", "eos_token", "pad_token", "unk_token")},
    )
    tokenizer.chat_template = None
    if len(tokenizer) != config["vocab_size"] or any(
        getattr(tokenizer, k) != config[k] for k in ("bos_token_id", "eos_token_id", "pad_token_id")
    ):
        raise ValueError("BIDIRECTIONAL_EMBEDDING_TOKENIZER_VOCABULARY")
    if config.get("model_type") == "qwen3_bidirec":
        from oryn_engine.giga_embeddings import load

        return load(root, config).to(backend).eval(), tokenizer, config
    model, loading = Ministral3Model.from_pretrained(
        root,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if any(
        loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError("BIDIRECTIONAL_EMBEDDING_WEIGHTS_INCOMPLETE")
    return model.to(backend).eval(), tokenizer, config


def execute(model, tokenizer, config, value, backend):
    import torch
    import torch.nn.functional as functional

    if not 1 <= len(value["texts"]) <= MAX_BATCH:
        raise ValueError("BIDIRECTIONAL_EMBEDDING_BATCH_LIMIT")
    tokens = tokenizer(value["texts"], padding=True, truncation=False, return_tensors="pt")
    if tokens["input_ids"].shape[1] > min(MAX_TOKENS, config["max_position_embeddings"]):
        raise ValueError("BIDIRECTIONAL_EMBEDDING_TOKEN_LIMIT")
    mask = tokens["attention_mask"].to(backend)
    # Explicit per-text positions preserve left-padding invariance, including YaRN.
    positions = (mask.long().cumsum(-1) - 1).clamp(min=0)
    native_mask = mask
    if config.get("model_type") == "qwen3_bidirec":
        from oryn_engine.giga_embeddings import attention_mask

        native_mask = attention_mask(mask)
    hidden = model(
        input_ids=tokens["input_ids"].to(backend),
        attention_mask=native_mask,
        position_ids=positions,
        use_cache=False,
    ).last_hidden_state.float()
    vectors = hidden.masked_fill(~mask.bool().unsqueeze(-1), 0).sum(1) / mask.sum(1, keepdim=True)
    if not torch.isfinite(vectors).all() or torch.any(vectors.norm(dim=-1) <= 1e-12):
        raise ValueError("BIDIRECTIONAL_EMBEDDING_OUTPUT_INVALID")
    if value["normalize"]:
        vectors = functional.normalize(vectors, dim=-1)
    return {
        "embeddings": vectors.cpu().tolist(),
        "dimensions": vectors.shape[1],
        "normalized": value["normalize"],
    }
