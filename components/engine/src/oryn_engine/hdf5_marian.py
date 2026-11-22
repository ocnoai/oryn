"""Complete numeric Marian translation graphs and fixed bilingual tokenizers."""

import re

from oryn_engine.model_metadata import metadata_file, validate_model_config


def validate_config(config):
    for key, low, high in (
        ("vocab_size", 8, 262144),
        ("d_model", 8, 4096),
        ("encoder_layers", 1, 48),
        ("decoder_layers", 1, 48),
        ("encoder_attention_heads", 1, 64),
        ("decoder_attention_heads", 1, 64),
        ("encoder_ffn_dim", 8, 16384),
        ("decoder_ffn_dim", 8, 16384),
        ("max_position_embeddings", 16, 4096),
    ):
        v = config.get(key)
        if type(v) is not int or not low <= v <= high:
            raise ValueError("HDF5_MARIAN_CONFIGURATION")
    if (
        config["d_model"] % 2
        or config["d_model"] % config["encoder_attention_heads"]
        or config["d_model"] % config["decoder_attention_heads"]
        or config.get("decoder_vocab_size", config["vocab_size"]) != config["vocab_size"]
        or config.get("activation_function", "swish") not in {"swish", "silu", "gelu", "relu"}
    ):
        raise ValueError("HDF5_MARIAN_CONFIGURATION")
    for key, default in (
        ("share_encoder_decoder_embeddings", True),
        ("tie_word_embeddings", True),
        ("static_position_embeddings", True),
        ("is_encoder_decoder", True),
        ("normalize_before", False),
        ("normalize_embedding", False),
        ("add_final_layer_norm", False),
    ):
        if config.get(key, default) is not default:
            raise ValueError("HDF5_MARIAN_CONFIGURATION")
    for key in ("eos_token_id", "pad_token_id", "decoder_start_token_id"):
        v = config.get(key)
        if type(v) is not int or not 0 <= v < config["vocab_size"]:
            raise ValueError("HDF5_MARIAN_TOKENS")
    if (
        config["eos_token_id"] == config["pad_token_id"]
        or config["decoder_start_token_id"] != config["pad_token_id"]
        or config.get("forced_eos_token_id", config["eos_token_id"]) != config["eos_token_id"]
        or config.get("forced_bos_token_id") is not None
        or config.get("bad_words_ids") not in (None, [[config["pad_token_id"]]])
    ):
        raise ValueError("HDF5_MARIAN_TOKENS")
    beams = config.get("num_beams", 1)
    if type(beams) is not int or not 1 <= beams <= 8:
        raise ValueError("HDF5_MARIAN_GENERATION")


def validate_tokenizer_metadata(metadata):
    validate_model_config(metadata)
    for key in ("source_lang", "target_lang"):
        value = metadata.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]{1,15}", value):
            raise ValueError("HDF5_MARIAN_LANGUAGE_PAIR")
    if (
        metadata["source_lang"] == metadata["target_lang"]
        or metadata.get("separate_vocabs", False) is not False
    ):
        raise ValueError("HDF5_MARIAN_LANGUAGE_PAIR")


def native_name(source, shape):
    if source == "final_logits_bias/final_logits_bias:0":
        return "final_logits_bias", None
    if source == "model/model.shared/model.shared/embeddings:0":
        return "model.shared.weight", None
    match = re.fullmatch(
        r"model/tf_marian_mt_model(?:_\d+)?/model/(encoder|decoder)/(.+):0", source
    )
    if not match:
        raise ValueError("HDF5_WEIGHT_NAME_UNSUPPORTED")
    prefix, leaf = match.groups()
    parts = leaf.split("/")
    kernel = parts[-1] == "kernel"
    parts[-1] = {"kernel": "weight", "embeddings": "weight", "gamma": "weight", "beta": "bias"}.get(
        parts[-1], parts[-1]
    )
    if kernel and len(shape) != 2:
        raise ValueError("HDF5_WEIGHT_SHAPE_UNSUPPORTED")
    return "model." + prefix + "." + ".".join(parts), (1, 0) if kernel else None


def prepare_native(model, config):
    from transformers import GenerationConfig

    model.tie_weights()
    model.generation_config = GenerationConfig(
        eos_token_id=config["eos_token_id"],
        pad_token_id=config["pad_token_id"],
        decoder_start_token_id=config["decoder_start_token_id"],
        forced_eos_token_id=config["eos_token_id"],
        bad_words_ids=[[config["pad_token_id"]]],
        num_beams=config.get("num_beams", 1),
        do_sample=False,
        use_cache=True,
    )


def load_tokenizer(root, config):
    from sacremoses import MosesPunctNormalizer
    from transformers import MarianTokenizer

    validate_tokenizer_metadata(metadata_file(root / "tokenizer_config.json"))
    tokenizer = MarianTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    if (
        len(tokenizer) != config["vocab_size"]
        or tokenizer.pad_token_id != config["pad_token_id"]
        or tokenizer.eos_token_id != config["eos_token_id"]
        or tokenizer.supported_language_codes
        or not isinstance(
            getattr(tokenizer.punc_normalizer, "__self__", None), MosesPunctNormalizer
        )
    ):
        raise ValueError("HDF5_MARIAN_TOKENIZER")
    tokenizer.chat_template = None
    return tokenizer


def token_limit(config):
    maximum = config.get("max_position_embeddings")
    if type(maximum) is not int or not 16 <= maximum <= 4096:
        raise ValueError("HDF5_MARIAN_CONFIGURATION")
    return min(512, maximum - 1)


def translate(model, tokenizer, config, value, backend):
    if "targetLanguage" in value:
        raise ValueError("HDF5_MARIAN_LANGUAGE_PAIR_FIXED")
    if not value["text"].strip():
        raise ValueError("HDF5_MARIAN_INPUT_EMPTY")
    inputs = tokenizer(value["text"], return_tensors="pt", truncation=False)
    if not 1 <= inputs["input_ids"].shape[1] <= min(
        512, config["max_position_embeddings"]
    ) or not 1 <= value["maxTokens"] <= token_limit(config):
        raise ValueError("HDF5_MARIAN_TOKEN_LIMIT")
    generated = model.generate(
        **{k: v.to(backend) for k, v in inputs.items()},
        max_new_tokens=value["maxTokens"],
        do_sample=False,
    )
    result = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    if not result or len(result.encode()) > 16384:
        raise ValueError("HDF5_MARIAN_OUTPUT_LIMIT")
    return {"text": result}
