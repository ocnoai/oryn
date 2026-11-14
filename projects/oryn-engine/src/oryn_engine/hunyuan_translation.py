"""Closed native HunYuan dense translation; repository templates are inert."""

from __future__ import annotations

import copy
import math

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

ARCHITECTURES = {"hunyuan_v1_dense": "HunYuanDenseV1ForCausalLM"}
LANGUAGES = {"en": "英语", "zh": "中文", "ja": "日语"}
MAX_CONTEXT = 4096
MARKERS = (
    "<｜hy_begin▁of▁sentence｜>",
    "<｜hy_User｜>",
    "<｜hy_Assistant｜>",
    "<｜hy_place▁holder▁no▁2｜>",
    "<｜hy_▁pad▁｜>",
)


def supports(config, task):
    return task == "translation" and config.get("model_type") in ARCHITECTURES


def text_config(config):
    return config


def validate_config(config):
    validate_model_config(config)
    if (
        config.get("model_type") not in ARCHITECTURES
        or config.get("architectures") != ["HunYuanDenseV1ForCausalLM"]
        or config.get("quantization_config")
        or config.get("quantization")
    ):
        raise ValueError("HUNYUAN_TRANSLATION_CONFIGURATION")
    for key, low, high in (
        ("vocab_size", 32, 300000),
        ("hidden_size", 16, 8192),
        ("intermediate_size", 16, 32768),
        ("num_hidden_layers", 1, 80),
        ("num_attention_heads", 1, 64),
        ("num_key_value_heads", 1, 64),
        ("head_dim", 4, 256),
        ("max_position_embeddings", 128, 1048576),
    ):
        value = config.get(key)
        if type(value) is not int or not low <= value <= high:
            raise ValueError("HUNYUAN_TRANSLATION_DIMENSIONS")
    if (
        config["num_attention_heads"] % config["num_key_value_heads"]
        or config["hidden_size"] != config["num_attention_heads"] * config["head_dim"]
        or config["head_dim"] % 2
        or config.get("attention_head_dim", config["head_dim"]) != config["head_dim"]
        or config.get("use_cla", False) is not False
        or config.get("use_qk_norm", True) is not True
        or config.get("use_rotary_pos_emb", True) is not True
        or config.get("add_classification_head", False) is not False
        or config.get("class_num", 0) != 0
        or config.get("mlp_bias", False) is not False
        or config.get("attention_bias", False) is not False
        or config.get("norm_type", "rms") != "rms"
        or config.get("hidden_act", "silu") != "silu"
        or config.get("pretraining_tp", 1) != 1
    ):
        raise ValueError("HUNYUAN_TRANSLATION_GRAPH")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if type(config.get(key)) is not int or not 0 <= config[key] < config["vocab_size"]:
            raise ValueError("HUNYUAN_TRANSLATION_TOKENS")
    if len({config[k] for k in ("bos_token_id", "eos_token_id", "pad_token_id")}) != 3:
        raise ValueError("HUNYUAN_TRANSLATION_TOKENS")
    if config.get("rope_parameters") is not None and config.get("rope_scaling") is not None:
        raise ValueError("HUNYUAN_TRANSLATION_POSITIONS")
    rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
    kind = rope.get("rope_type", rope.get("type", "default"))
    theta = rope.get("rope_theta", config.get("rope_theta", 10000.0))
    if (
        kind not in {"default", "dynamic"}
        or rope.get("type", kind) != kind
        or type(theta) not in {int, float}
        or not math.isfinite(theta)
        or not 100 <= theta <= 10000000
        or set(rope)
        - {
            "type",
            "rope_type",
            "rope_theta",
            "alpha",
            "factor",
            "beta_fast",
            "beta_slow",
            "mscale",
            "mscale_all_dim",
        }
        or any(rope.get(k, 1.0) != 1.0 for k in ("factor", "mscale", "mscale_all_dim"))
        or rope.get("beta_fast", 32) != 32
        or rope.get("beta_slow", 1) != 1
        or kind == "default"
        and "alpha" in rope
        or kind == "dynamic"
        and (type(rope.get("alpha")) not in {int, float} or not 1 <= rope["alpha"] <= 10000)
    ):
        raise ValueError("HUNYUAN_TRANSLATION_POSITIONS")


def native_config(config):
    from transformers import HunYuanDenseV1Config

    native = HunYuanDenseV1Config.from_dict(copy.deepcopy(config))
    native._attn_implementation = "eager"
    return native


def validate_weight_shapes(config, shapes):
    import torch
    from transformers import HunYuanDenseV1ForCausalLM

    validate_config(config)
    with torch.device("meta"):
        model = HunYuanDenseV1ForCausalLM(native_config(config))
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    missing = expected.keys() - shapes.keys()
    if config.get("tie_word_embeddings", False) and "model.embed_tokens.weight" in shapes:
        missing -= {"lm_head.weight"}
    if (
        missing
        or shapes.keys() - expected.keys()
        or any(tuple(v) != expected[k] for k, v in shapes.items())
    ):
        raise ValueError("HUNYUAN_TRANSLATION_WEIGHTS_INCOMPLETE")


def validate_metadata(name, value, config):
    validate_model_config(value)
    if name != "tokenizer_config.json" or (
        value.get("tokenizer_class") not in {"PreTrainedTokenizerFast", "TokenizersBackend"}
        or value.get("add_bos_token", False) is not False
        or value.get("add_eos_token", False) is not False
    ):
        raise ValueError("HUNYUAN_TRANSLATION_TOKENIZER")


def load_tokenizer(root, config):
    from transformers import PreTrainedTokenizerFast

    validate_config(config)
    validate_metadata(
        "tokenizer_config.json", metadata_file(root / "tokenizer_config.json"), config
    )
    path = root / "tokenizer.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024**2:
        raise ValueError("HUNYUAN_TRANSLATION_TOKENIZER")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    tokenizer.chat_template = None
    vocabulary = tokenizer.get_vocab()
    ids = [vocabulary.get(token) for token in MARKERS]
    if (
        len(tokenizer) != config["vocab_size"]
        or set(vocabulary.values()) != set(range(len(vocabulary)))
        or any(type(v) is not int for v in ids)
        or len(set(ids)) != len(ids)
        or any(
            tokenizer.encode(t, add_special_tokens=False) != [v]
            for t, v in zip(MARKERS, ids, strict=True)
        )
        or ids[0] != config["bos_token_id"]
        or ids[3] != config["eos_token_id"]
        or ids[4] != config["pad_token_id"]
        or tokenizer.bos_token_id != ids[0]
        or tokenizer.eos_token_id != ids[3]
        or tokenizer.pad_token_id != ids[4]
    ):
        raise ValueError("HUNYUAN_TRANSLATION_TOKENIZER")
    return tokenizer


def load_components(root, config, backend):
    from transformers import GenerationConfig, HunYuanDenseV1ForCausalLM

    tokenizer = load_tokenizer(root, config)
    validate_weight_shapes(config, preflight_safetensors(root))
    model, loading = HunYuanDenseV1ForCausalLM.from_pretrained(
        root,
        config=native_config(config),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype="float32",
        attn_implementation="eager",
        output_loading_info=True,
    )
    if any(
        loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError("HUNYUAN_TRANSLATION_WEIGHTS_INCOMPLETE")
    model.generation_config = GenerationConfig(
        bos_token_id=config["bos_token_id"],
        eos_token_id=config["eos_token_id"],
        pad_token_id=config["pad_token_id"],
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )
    return model.to(backend).eval(), tokenizer, config


def prompt(text, language):
    if language not in LANGUAGES or not text.strip():
        raise ValueError("HUNYUAN_TRANSLATION_INPUT")
    if "<｜" in text or "｜>" in text:
        raise ValueError("HUNYUAN_TRANSLATION_RESERVED_TOKEN")
    content = (
        f"将以下文本翻译成{LANGUAGES[language]},注意只需要输出翻译后的结果,不要额外解释:\n\n{text}"
    )
    return f"{MARKERS[0]}{MARKERS[1]}{content}{MARKERS[2]}"


def execute(model, tokenizer, config, value, backend):
    inputs = tokenizer(
        prompt(value["text"], value.get("targetLanguage", "en")),
        add_special_tokens=False,
        truncation=False,
        return_tensors="pt",
    )
    maximum = value["maxTokens"]
    count = inputs["input_ids"].shape[1]
    if (
        type(maximum) is not int
        or not 1 <= maximum <= 512
        or count + maximum > min(MAX_CONTEXT, config["max_position_embeddings"])
    ):
        raise ValueError("HUNYUAN_TRANSLATION_TOKEN_LIMIT")
    output = model.generate(
        **{k: v.to(backend) for k, v in inputs.items()}, max_new_tokens=maximum, do_sample=False
    )
    result = tokenizer.decode(output[0, count:], skip_special_tokens=True).strip()
    if not result or len(result.encode()) > 16384:
        raise ValueError("HUNYUAN_TRANSLATION_OUTPUT_LIMIT")
    return {"text": result}
