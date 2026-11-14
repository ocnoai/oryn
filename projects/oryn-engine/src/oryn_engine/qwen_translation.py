"""Bounded instruction translation over complete native Qwen3.5 checkpoints."""

from __future__ import annotations

import copy

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

FAMILIES = {"qwen3_5", "qwen3_5_text"}
ARCHITECTURES = {
    "qwen3_5": "Qwen3_5ForConditionalGeneration",
    "qwen3_5_text": "Qwen3_5ForCausalLM",
}
# The common reviewed directions of the text and document translation releases.
LANGUAGES = {"en": "英语", "zh": "中文", "ja": "日语"}
MAX_CONTEXT = 4096


def supports(config, task):
    return task == "translation" and config.get("model_type") in FAMILIES


def text_config(config):
    return config.get("text_config", {}) if config.get("model_type") == "qwen3_5" else config


def validate_config(config):
    validate_model_config(config)
    family = config.get("model_type")
    if (
        family not in FAMILIES
        or config.get("architectures") != [ARCHITECTURES[family]]
        or config.get("quantization_config")
        or config.get("quantization")
    ):
        raise ValueError("QWEN_TRANSLATION_CONFIGURATION")
    text = text_config(config)
    if (
        text.get("model_type") != "qwen3_5_text"
        or text.get("quantization_config")
        or text.get("quantization")
    ):
        raise ValueError("QWEN_TRANSLATION_CONFIGURATION")
    for key, low, high in (
        ("vocab_size", 32, 262144),
        ("hidden_size", 16, 8192),
        ("intermediate_size", 16, 32768),
        ("num_hidden_layers", 1, 80),
        ("num_attention_heads", 1, 64),
        ("num_key_value_heads", 1, 64),
        ("head_dim", 4, 512),
        ("max_position_embeddings", 128, 1048576),
        ("linear_conv_kernel_dim", 1, 8),
        ("linear_key_head_dim", 4, 256),
        ("linear_value_head_dim", 4, 256),
        ("linear_num_key_heads", 1, 64),
        ("linear_num_value_heads", 1, 64),
    ):
        v = text.get(key)
        if type(v) is not int or not low <= v <= high:
            raise ValueError("QWEN_TRANSLATION_DIMENSIONS")
    layers = text.get("layer_types")
    if (
        not isinstance(layers, list)
        or len(layers) != text["num_hidden_layers"]
        or not set(layers) <= {"linear_attention", "full_attention"}
        or text["num_attention_heads"] % text["num_key_value_heads"]
        or text["linear_num_value_heads"] % text["linear_num_key_heads"]
        or text.get("hidden_act", "silu") != "silu"
        or text.get("attention_bias", False) is not False
        or text.get("attn_output_gate", True) is not True
        or text.get("mtp_use_dedicated_embeddings", False) is not False
        or type(text.get("mtp_num_hidden_layers", 0)) is not int
        or not 0 <= text.get("mtp_num_hidden_layers", 0) <= 1
    ):
        raise ValueError("QWEN_TRANSLATION_GRAPH")
    rope = text.get("rope_parameters", {})
    fraction = rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25))
    sections = rope.get("mrope_section")
    if (
        rope.get("rope_type", "default") != "default"
        or rope.get("mrope_interleaved", True) is not True
        or type(fraction) not in {int, float}
        or not 0 < fraction <= 1
        or not isinstance(sections, list)
        or len(sections) != 3
        or any(type(v) is not int or not 0 <= v <= 128 for v in sections)
        or sum(sections) * 2 != int(text["head_dim"] * fraction)
    ):
        raise ValueError("QWEN_TRANSLATION_POSITIONS")
    if (
        type(text.get("eos_token_id")) is not int
        or not 0 <= text["eos_token_id"] < text["vocab_size"]
    ):
        raise ValueError("QWEN_TRANSLATION_TOKENS")
    if family == "qwen3_5":
        vision = config.get("vision_config", {})
        for key, low, high in (
            ("depth", 1, 48),
            ("hidden_size", 16, 4096),
            ("intermediate_size", 16, 16384),
            ("num_heads", 1, 64),
            ("patch_size", 1, 32),
            ("temporal_patch_size", 1, 4),
            ("spatial_merge_size", 1, 4),
            ("num_position_embeddings", 4, 4096),
        ):
            v = vision.get(key)
            if type(v) is not int or not low <= v <= high:
                raise ValueError("QWEN_TRANSLATION_VISION_GRAPH")
        if (
            vision["hidden_size"] % vision["num_heads"]
            or vision.get("out_hidden_size") != text["hidden_size"]
            or vision.get("deepstack_visual_indexes", []) != []
        ):
            raise ValueError("QWEN_TRANSLATION_VISION_GRAPH")
        tokens = [
            config.get(k)
            for k in (
                "image_token_id",
                "video_token_id",
                "vision_start_token_id",
                "vision_end_token_id",
            )
        ]
        if (
            any(type(v) is not int or not 0 <= v < text["vocab_size"] for v in tokens)
            or len(set(tokens)) != 4
        ):
            raise ValueError("QWEN_TRANSLATION_TOKENS")


def native_class(config):
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm

    base = (
        Qwen3_5ForConditionalGeneration if config["model_type"] == "qwen3_5" else Qwen3_5ForCausalLM
    )

    class NativeTranslation(base):
        # No repository tensor may hide behind upstream ignored-key patterns.
        _keys_to_ignore_on_load_unexpected = []

        def __init__(self, native):
            super().__init__(native)
            text = native.text_config if hasattr(native, "text_config") else native
            if getattr(text, "mtp_num_hidden_layers", 0):
                # Preserve the complete serialized auxiliary MTP graph as well.
                # Ordinary autoregressive translation never invokes speculative
                # prediction; no speculative-decoding capability is advertised.
                auxiliary = copy.deepcopy(text)
                auxiliary.layer_types = ["full_attention"]
                auxiliary.num_hidden_layers = 1
                self.mtp = torch.nn.Module()
                self.mtp.fc = torch.nn.Linear(2 * text.hidden_size, text.hidden_size, bias=False)
                self.mtp.layers = torch.nn.ModuleList([Qwen3_5DecoderLayer(auxiliary, 0)])
                for name in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden"):
                    setattr(self.mtp, name, Qwen3_5RMSNorm(text.hidden_size, eps=text.rms_norm_eps))

    return NativeTranslation


def native_config(config):
    from transformers import Qwen3_5Config, Qwen3_5TextConfig

    cls = Qwen3_5Config if config["model_type"] == "qwen3_5" else Qwen3_5TextConfig
    native = cls.from_dict(copy.deepcopy(config))
    native._attn_implementation = "eager"
    if hasattr(native, "text_config"):
        native.text_config._attn_implementation = "eager"
        native.vision_config._attn_implementation = "eager"
    return native


def validate_weight_shapes(config, shapes):
    import torch

    validate_config(config)
    with torch.device("meta"):
        model = native_class(config)(native_config(config))
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    missing = set(expected) - shapes.keys()
    text = text_config(config)
    embedding = (
        "model.language_model.embed_tokens.weight"
        if config["model_type"] == "qwen3_5"
        else "model.embed_tokens.weight"
    )
    if (
        config.get("tie_word_embeddings", text.get("tie_word_embeddings", False))
        and embedding in shapes
    ):
        missing -= {"lm_head.weight"}
    if (
        missing
        or shapes.keys() - expected.keys()
        or any(tuple(v) != expected[k] for k, v in shapes.items())
    ):
        raise ValueError("QWEN_TRANSLATION_WEIGHTS_INCOMPLETE")


def validate_metadata(name, value, config):
    from oryn_engine.qwen_retrieval import validate_metadata

    validate_metadata(name, value, config)


def load_tokenizer(root, config, *, require_eos_match=True):
    from transformers import Qwen2Tokenizer

    validate_config(config)
    validate_metadata(
        "tokenizer_config.json", metadata_file(root / "tokenizer_config.json"), config
    )
    text = text_config(config)
    tokenizer = Qwen2Tokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    tokenizer.chat_template = None
    special = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>"]
    if config["model_type"] == "qwen3_5":
        special += ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
    vocabulary = tokenizer.get_vocab()
    ids = [vocabulary.get(token) for token in special]
    if (
        not 32 <= len(tokenizer) <= text["vocab_size"]
        or text["vocab_size"] - len(tokenizer) >= 256
        or set(vocabulary.values()) != set(range(len(vocabulary)))
        or any(type(v) is not int or not 0 <= v < text["vocab_size"] for v in ids)
        or len(set(ids)) != len(ids)
        or any(
            tokenizer.encode(token, add_special_tokens=False) != [v]
            for token, v in zip(special, ids, strict=True)
        )
        or text["eos_token_id"] not in {ids[1], ids[4]}
        or tokenizer.eos_token_id not in {ids[1], ids[4]}
        or require_eos_match
        and tokenizer.eos_token_id != text["eos_token_id"]
        or type(tokenizer.pad_token_id) is not int
        or not 0 <= tokenizer.pad_token_id < text["vocab_size"]
    ):
        raise ValueError("QWEN_TRANSLATION_TOKENIZER")
    if config["model_type"] == "qwen3_5" and ids[5:] != [
        config[k]
        for k in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        )
    ]:
        raise ValueError("QWEN_TRANSLATION_TOKENIZER")
    return tokenizer


def load_components(root, config, backend):
    from transformers import GenerationConfig

    tokenizer = load_tokenizer(root, config)
    text = text_config(config)
    validate_weight_shapes(config, preflight_safetensors(root))
    model, loading = native_class(config).from_pretrained(
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
        raise ValueError("QWEN_TRANSLATION_WEIGHTS_INCOMPLETE")
    model.generation_config = GenerationConfig(
        eos_token_id=sorted({text["eos_token_id"], tokenizer.get_vocab()["<|im_end|>"]}),
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )
    return model.to(backend).eval(), tokenizer, config


def prompt(text, language):
    if language not in LANGUAGES or not text.strip():
        raise ValueError("QWEN_TRANSLATION_INPUT")
    if any(marker in text for marker in ("<|", "|>", "<think>", "</think>")):
        raise ValueError("QWEN_TRANSLATION_RESERVED_TOKEN")
    instruction = (
        f"请将以下文本翻译为{LANGUAGES[language]}，直接输出翻译结果，不要进行任何解释。\n\n{text}"
    )
    return (
        f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


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
        or count + maximum > min(MAX_CONTEXT, text_config(config)["max_position_embeddings"])
    ):
        raise ValueError("QWEN_TRANSLATION_TOKEN_LIMIT")
    output = model.generate(
        **{k: v.to(backend) for k, v in inputs.items()}, max_new_tokens=maximum, do_sample=False
    )
    result = tokenizer.decode(output[0, count:], skip_special_tokens=True).strip()
    if not result or len(result.encode()) > 16384:
        raise ValueError("QWEN_TRANSLATION_OUTPUT_LIMIT")
    return {"text": result}
