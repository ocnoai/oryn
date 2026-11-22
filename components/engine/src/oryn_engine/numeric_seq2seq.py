"""Owned complete BART summarization and NLLB translation over numeric archives."""

from oryn_engine.flax_weights import decoding_options
from oryn_engine.model_metadata import validate_model_config

ARCHITECTURES = {
    ("summarization", "t5"): "T5ForConditionalGeneration",
    ("summarization", "bart"): "BartForConditionalGeneration",
    ("translation", "m2m_100"): "M2M100ForConditionalGeneration",
}
# ISO task parameters map to explicit native language tokens. Source follows the
# tokenizer's declared source language (the native NLLB default is English).
LANGUAGES = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "cs": "ces_Latn",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "hr": "hrv_Latn",
    "sr": "srp_Cyrl",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "bg": "bul_Cyrl",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "el": "ell_Grek",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "tr": "tur_Latn",
    "ar": "arb_Arab",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "zh-CN": "zho_Hans",
    "zh-TW": "zho_Hant",
    "vi": "vie_Latn",
    "id": "ind_Latn",
}


def supports(config, task):
    return (task, config.get("model_type")) in ARCHITECTURES


def validate_config(c, task):
    validate_model_config(c)
    if (
        not supports(c, task)
        or c.get("architectures") != [ARCHITECTURES[(task, c.get("model_type"))]]
        or c.get("auto_map")
        or c.get("quantization_config")
        or c.get("quantization")
        or c.get("is_encoder_decoder", True) is not True
    ):
        raise ValueError("NUMERIC_SEQ2SEQ_CONFIGURATION")
    if c["model_type"] == "t5":
        validate_t5(c)
        options(c)
        return
    for key, lo, hi in [
        ("vocab_size", 8, 300000),
        ("d_model", 8, 4096),
        ("encoder_layers", 1, 48),
        ("decoder_layers", 1, 48),
        ("encoder_attention_heads", 1, 64),
        ("decoder_attention_heads", 1, 64),
        ("encoder_ffn_dim", 8, 32768),
        ("decoder_ffn_dim", 8, 32768),
        ("max_position_embeddings", 16, 8192),
    ]:
        if type(c.get(key)) is not int or not lo <= c[key] <= hi:
            raise ValueError("NUMERIC_SEQ2SEQ_CONFIGURATION_LIMIT")
    if (
        c["d_model"] % 2
        or any(c["d_model"] % c[side + "_attention_heads"] for side in ["encoder", "decoder"])
        or c.get("activation_function", "relu" if c["model_type"] == "m2m_100" else "gelu")
        not in {"relu", "gelu", "gelu_new", "silu"}
    ):
        raise ValueError("NUMERIC_SEQ2SEQ_CONFIGURATION")
    for key in ["bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id"]:
        if type(c.get(key)) is not int or not 0 <= c[key] < c["vocab_size"]:
            raise ValueError("NUMERIC_SEQ2SEQ_TOKENS")
    for key in ["forced_bos_token_id", "forced_eos_token_id"]:
        if c.get(key) is not None and (
            type(c[key]) is not int or not 0 <= c[key] < c["vocab_size"]
        ):
            raise ValueError("NUMERIC_SEQ2SEQ_TOKENS")
    if c["pad_token_id"] == c["eos_token_id"]:
        raise ValueError("NUMERIC_SEQ2SEQ_TOKENS")
    for key in ["scale_embedding", "tie_word_embeddings", "tie_encoder_decoder"]:
        if key in c and type(c[key]) is not bool:
            raise ValueError("NUMERIC_SEQ2SEQ_CONFIGURATION")
    if c.get("tie_encoder_decoder", False):
        raise ValueError("NUMERIC_SEQ2SEQ_CONFIGURATION")
    if c["model_type"] == "m2m_100" and c.get("tokenizer_class") not in {
        None,
        "NllbTokenizer",
        "NllbTokenizerFast",
    }:
        raise ValueError("NUMERIC_SEQ2SEQ_TOKENIZER")
    options(c)


def options(c):
    result = {"prefix": c.get("prefix", "")}
    result.update(decoding_options(c))
    if (
        not isinstance(result["prefix"], str)
        or len(result["prefix"]) > 512
        or any(ord(x) < 32 for x in result["prefix"])
    ):
        raise ValueError("NUMERIC_SEQ2SEQ_PREFIX")
    return result


def prepare(model, tokenizer, c):
    from transformers import GenerationConfig, NllbTokenizer

    opts = options(c)
    model.generation_config = GenerationConfig(
        **{
            k: c.get(k)
            for k in [
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "decoder_start_token_id",
                "forced_bos_token_id",
                "forced_eos_token_id",
            ]
        },
        **{k: v for k, v in opts.items() if k != "prefix"},
        do_sample=False,
        # This native T5 version sizes its generation cache from encoder depth.
        # Full decoding also supports different encoder and decoder depths.
        use_cache=c["model_type"] != "t5"
        or c.get("num_decoder_layers", c.get("num_layers")) == c.get("num_layers"),
    )
    if (
        len(tokenizer) > c["vocab_size"]
        or tokenizer.pad_token_id != c["pad_token_id"]
        or tokenizer.eos_token_id != c["eos_token_id"]
    ):
        raise ValueError("NUMERIC_SEQ2SEQ_TOKENIZER")
    if c["model_type"] == "m2m_100":
        if not isinstance(tokenizer, NllbTokenizer):
            raise ValueError("NUMERIC_SEQ2SEQ_TOKENIZER")
        source = tokenizer.src_lang
        if source not in tokenizer.all_special_tokens:
            raise ValueError("NUMERIC_SEQ2SEQ_SOURCE_LANGUAGE")
        token = tokenizer.convert_tokens_to_ids(source)
        if (
            type(token) is not int
            or not 0 <= token < c["vocab_size"]
            or token == tokenizer.unk_token_id
        ):
            raise ValueError("NUMERIC_SEQ2SEQ_SOURCE_LANGUAGE")


def execute(model, tokenizer, c, task, value, backend):
    if not value["text"].strip():
        raise ValueError("NUMERIC_SEQ2SEQ_INPUT_EMPTY")
    inputs = tokenizer(options(c)["prefix"] + value["text"], return_tensors="pt", truncation=False)
    maximum = min(512, c.get("max_position_embeddings", 513) - 1)
    if (
        not 1 <= inputs["input_ids"].shape[1] <= maximum
        or not 1 <= value.get("maxTokens", 128) <= maximum
    ):
        raise ValueError("NUMERIC_SEQ2SEQ_TOKEN_LIMIT")
    extra = {}
    if task == "translation":
        token = target_language(value.get("targetLanguage"))
        index = tokenizer.convert_tokens_to_ids(token)
        if (
            token not in tokenizer.all_special_tokens
            or type(index) is not int
            or not 0 <= index < c["vocab_size"]
            or index == tokenizer.unk_token_id
        ):
            raise ValueError("NUMERIC_SEQ2SEQ_TARGET_LANGUAGE")
        extra["forced_bos_token_id"] = index
    output = model.generate(
        **{k: v.to(backend) for k, v in inputs.items() if k in {"input_ids", "attention_mask"}},
        max_new_tokens=value.get("maxTokens", 128),
        do_sample=False,
        **extra,
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    if not text or len(text.encode()) > 16384:
        raise ValueError("NUMERIC_SEQ2SEQ_OUTPUT_LIMIT")
    return {"text": text}


def expected_shapes(c):
    family = c["model_type"]
    if family == "t5":
        return t5_shapes(c)
    assert family in {"bart", "m2m_100"}
    d, v, p = c["d_model"], c["vocab_size"], c["max_position_embeddings"]
    out = {}

    def linear(name, rows, columns):
        out[name + ".weight"] = [rows, columns]
        out[name + ".bias"] = [rows]

    def norm(name):
        out[name + ".weight"] = [d]
        out[name + ".bias"] = [d]

    out["model.shared.weight"] = [v, d]
    out["lm_head.weight"] = [v, d]
    if family == "bart":
        out["final_logits_bias"] = [1, v]
    for side in ["encoder", "decoder"]:
        prefix = "model." + side
        out[prefix + ".embed_tokens.weight"] = [v, d]
        if family == "bart":
            out[prefix + ".embed_positions.weight"] = [p + 2, d]
            norm(prefix + ".layernorm_embedding")
        else:
            norm(prefix + ".layer_norm")
        for i in range(c[side + "_layers"]):
            layer = prefix + ".layers." + str(i)
            for attn in ["self_attn", "encoder_attn"] if side == "decoder" else ["self_attn"]:
                for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                    linear(layer + "." + attn + "." + proj, d, d)
                norm(layer + "." + attn + "_layer_norm")
            linear(layer + ".fc1", c[side + "_ffn_dim"], d)
            linear(layer + ".fc2", d, c[side + "_ffn_dim"])
            norm(layer + ".final_layer_norm")
    return out


def validate_tokenizer_metadata(config, metadata):
    validate_model_config(metadata)
    if (
        metadata.get("tokenizer_class") not in {"NllbTokenizer", "NllbTokenizerFast"}
        or metadata.get("auto_map")
        or config.get("tokenizer_class") not in {None, "NllbTokenizer", "NllbTokenizerFast"}
    ):
        raise ValueError("NUMERIC_SEQ2SEQ_TOKENIZER")
    source = metadata.get("src_lang")
    if source is not None and (not isinstance(source, str) or len(source) > 16):
        raise ValueError("NUMERIC_SEQ2SEQ_SOURCE_LANGUAGE")


def target_language(value):
    if isinstance(value, str) and value in LANGUAGES:
        return LANGUAGES[value]
    if not isinstance(value, str) or len(value) != 3 or not value.isascii() or not value.islower():
        raise ValueError("NUMERIC_SEQ2SEQ_TARGET_LANGUAGE_REQUIRED")
    from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES

    # Script variants share an ISO code. Never guess between them; the explicit
    # aliases above select supported variants such as zh-CN and zh-TW.
    choices = [token for token in FAIRSEQ_LANGUAGE_CODES if token.split("_")[0] == value]
    if len(choices) != 1:
        raise ValueError("NUMERIC_SEQ2SEQ_TARGET_LANGUAGE_REQUIRED")
    return choices[0]


def validate_t5(c):
    for key, default, lo, hi in [
        ("vocab_size", None, 8, 300000),
        ("d_model", None, 8, 4096),
        ("d_ff", None, 8, 32768),
        ("d_kv", 64, 1, 256),
        ("num_layers", None, 1, 48),
        ("num_decoder_layers", c.get("num_layers"), 1, 48),
        ("num_heads", None, 1, 64),
        ("relative_attention_num_buckets", 32, 4, 256),
        ("relative_attention_max_distance", 128, 8, 8192),
    ]:
        value = c.get(key, default)
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError("NUMERIC_SEQ2SEQ_CONFIGURATION_LIMIT")
    buckets = c.get("relative_attention_num_buckets", 32)
    if (
        buckets % 2
        or c.get("relative_attention_max_distance", 128) <= buckets // 2
        or c.get("feed_forward_proj", "relu") not in {"relu", "gated-gelu"}
        or c.get("tie_encoder_decoder", False) is not False
        or type(c.get("tie_word_embeddings", True)) is not bool
    ):
        raise ValueError("NUMERIC_SEQ2SEQ_CONFIGURATION")
    for key in ["pad_token_id", "eos_token_id", "decoder_start_token_id"]:
        if type(c.get(key)) is not int or not 0 <= c[key] < c["vocab_size"]:
            raise ValueError("NUMERIC_SEQ2SEQ_TOKENS")
    for key in ["bos_token_id", "forced_bos_token_id", "forced_eos_token_id"]:
        if c.get(key) is not None and (
            type(c[key]) is not int or not 0 <= c[key] < c["vocab_size"]
        ):
            raise ValueError("NUMERIC_SEQ2SEQ_TOKENS")
    if c["pad_token_id"] == c["eos_token_id"]:
        raise ValueError("NUMERIC_SEQ2SEQ_TOKENS")


def t5_shapes(c):
    d, f, v = c["d_model"], c["d_ff"], c["vocab_size"]
    heads, kv = c["num_heads"], c.get("d_kv", 64)
    out = {name + ".weight": [v, d] for name in ["shared", "lm_head"]}
    for side in ["encoder", "decoder"]:
        out[side + ".embed_tokens.weight"] = [v, d]
        out[side + ".final_layer_norm.weight"] = [d]
        layers = (
            c["num_layers"] if side == "encoder" else c.get("num_decoder_layers", c["num_layers"])
        )
        for i in range(layers):
            prefix = f"{side}.block.{i}.layer."
            attentions = (
                ["SelfAttention", "EncDecAttention"] if side == "decoder" else ["SelfAttention"]
            )
            for j, attention in enumerate(attentions):
                layer = prefix + str(j)
                out[layer + ".layer_norm.weight"] = [d]
                for name in ["q", "k", "v"]:
                    out[layer + "." + attention + "." + name + ".weight"] = [heads * kv, d]
                out[layer + "." + attention + ".o.weight"] = [d, heads * kv]
                if i == 0 and j == 0:
                    out[layer + "." + attention + ".relative_attention_bias.weight"] = [
                        c.get("relative_attention_num_buckets", 32),
                        heads,
                    ]
            ff = prefix + str(len(attentions))
            out[ff + ".layer_norm.weight"] = [d]
            for name in ["wi_0", "wi_1"] if c.get("feed_forward_proj") == "gated-gelu" else ["wi"]:
                out[ff + ".DenseReluDense." + name + ".weight"] = [f, d]
            out[ff + ".DenseReluDense.wo.weight"] = [d, f]
    return out
