"""Closed native Gemma 4 text backbone and tokenizer for decision readouts."""

from __future__ import annotations

from oryn_engine.model_metadata import metadata_file, validate_model_config

TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "sot_token": "<|turn>",
    "eot_token": "<turn|>",
}
MAX_LABELS = 26


def supports(config):
    return config.get("model_type") == "gemma4_text" and config.get("architectures") == [
        "Gemma4TextModel"
    ]


def validate_config(config):
    validate_model_config(config)
    if not supports(config) or any(
        config.get(k) is not None for k in ("quantization", "quantization_config")
    ):
        raise ValueError("GEMMA_DECISION_CONFIGURATION")
    for key, low, high in (
        ("hidden_size", 16, 8192),
        ("intermediate_size", 16, 65536),
        ("num_hidden_layers", 2, 80),
        ("num_attention_heads", 1, 128),
        ("num_key_value_heads", 1, 128),
        ("head_dim", 8, 512),
        ("vocab_size", 32, 262144),
        ("vocab_size_per_layer_input", 32, 262144),
        ("max_position_embeddings", 4096, 1048576),
        ("sliding_window", 8, 4096),
        ("hidden_size_per_layer_input", 8, 1024),
        ("num_kv_shared_layers", 0, 79),
    ):
        if type(config.get(key)) is not int or not low <= config[key] <= high:
            raise ValueError("GEMMA_DECISION_GEOMETRY")
    layers = config.get("layer_types")
    if (
        not isinstance(layers, list)
        or len(layers) != config["num_hidden_layers"]
        or set(layers) != {"sliding_attention", "full_attention"}
        or layers[-1] != "full_attention"
        or config["num_attention_heads"] % config["num_key_value_heads"]
        or config["head_dim"] % 4
        or config["num_kv_shared_layers"] >= config["num_hidden_layers"]
        or config["vocab_size_per_layer_input"] != config["vocab_size"]
        or config.get("enable_moe_block") is not False
        or config.get("attention_bias") is not False
        or config.get("attention_dropout") != 0
        or config.get("attention_k_eq_v") is not False
        or config.get("hidden_activation") != "gelu_pytorch_tanh"
        or config.get("use_bidirectional_attention") is not None
        or config.get("use_double_wide_mlp") is not True
        or config.get("tie_word_embeddings") is not True
        or config.get("rope_parameters")
        != {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        }
    ):
        raise ValueError("GEMMA_DECISION_GRAPH")
    overrides = config.get("per_layer_config")
    expected = {f"{i:02}": i for i, layer in enumerate(layers) if layer == "full_attention"}
    if not isinstance(overrides, dict) or set(overrides) != set(expected):
        raise ValueError("GEMMA_DECISION_LAYER_OVERRIDES")
    for value in overrides.values():
        if (
            not isinstance(value, dict)
            or set(value) != {"head_dim"}
            or type(value["head_dim"]) is not int
            or not 16 <= value["head_dim"] <= 512
            or value["head_dim"] % 4
        ):
            raise ValueError("GEMMA_DECISION_LAYER_OVERRIDES")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if type(config.get(key)) is not int or not 0 <= config[key] < config["vocab_size"]:
            raise ValueError("GEMMA_DECISION_TOKEN_IDS")


def native_model(config):
    from transformers import Gemma4TextConfig, Gemma4TextModel

    native = Gemma4TextConfig.from_dict(dict(config))
    native._attn_implementation = "eager"
    return Gemma4TextModel(native)


def validate_tokenizer_metadata(metadata):
    validate_model_config(metadata)
    if metadata.get("tokenizer_class") != "GemmaTokenizer" or any(
        metadata.get(k) != v for k, v in TOKENS.items()
    ):
        raise ValueError("GEMMA_DECISION_TOKENIZER_METADATA")


def load_tokenizer(root, config):
    from transformers import TokenizersBackend

    metadata = metadata_file(root / "tokenizer_config.json")
    validate_tokenizer_metadata(metadata)
    path = root / "tokenizer.json"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 32 * 1024**2:
        raise ValueError("GEMMA_DECISION_TOKENIZER_GRAPH")
    tokenizer = TokenizersBackend(
        tokenizer_file=str(path), **{k: TOKENS[k] for k in ("bos_token", "eos_token", "pad_token")}
    )
    tokenizer.chat_template = None
    vocab = tokenizer.get_vocab()
    if len(vocab) != config["vocab_size"] or set(vocab.values()) != set(range(len(vocab))):
        raise ValueError("GEMMA_DECISION_VOCABULARY")
    for key, token in TOKENS.items():
        token_id = vocab.get(token)
        if (
            token_id is None
            or tokenizer.encode(token, add_special_tokens=False) != [token_id]
            or key + "_id" in config
            and token_id != config[key + "_id"]
        ):
            raise ValueError("GEMMA_DECISION_FRAMING")
    return tokenizer
