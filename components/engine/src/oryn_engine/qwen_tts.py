"""Closed CustomVoice binding for the separately pinned Qwen3 TTS runtime.

Inspection stays dependency-free. Native imports and complete graph validation run
only in the isolated pack; no repository model, processor or template is executed.
"""

from __future__ import annotations

import copy
import math

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

PACK_ID = "oryn.qwen-tts-pytorch"
ARCHITECTURE = "Qwen3TTSForConditionalGeneration"
MAX_CODES = 512
MAX_TEXT_TOKENS = 1024
VOICES = frozenset(
    {"aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"}
)
LANGUAGES = {
    "en-us": "english",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "fr": "french",
    "ru": "russian",
    "pt": "portuguese",
    "es": "spanish",
    "it": "italian",
    "auto": "auto",
}
MARKERS = {
    "im_start_token_id": "<|im_start|>",
    "im_end_token_id": "<|im_end|>",
    "tts_bos_token_id": "<tts_text_bos>",
    "tts_eos_token_id": "<tts_text_eod>",
    "tts_pad_token_id": "<tts_pad>",
}
REQUIRED_FILES = frozenset(
    {
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
        "speech_tokenizer/config.json",
        "speech_tokenizer/model.safetensors",
    }
)


def supports(config, task):
    return task == "text-to-speech" and config.get("model_type") == "qwen3_tts"


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("QWEN_TTS_DIMENSIONS")
    return value


def _dimensions(config, limits):
    for name, (low, high) in limits.items():
        _integer(config.get(name), low, high)


def validate_config(config, codec):
    for value in (config, codec):
        validate_model_config(value)
    if (
        config.get("architectures") != [ARCHITECTURE]
        or config.get("model_type") != "qwen3_tts"
        or config.get("tts_model_type") != "custom_voice"
        or config.get("tts_model_size") not in {"0b6", "1b7"}
        or config.get("tokenizer_type") != "qwen3_tts_tokenizer_12hz"
        or codec.get("model_type") != "qwen3_tts_tokenizer_12hz"
        or codec.get("architectures") != ["Qwen3TTSTokenizerV2Model"]
    ):
        raise ValueError("QWEN_TTS_GRAPH_IDENTITY")
    talker = config.get("talker_config", {})
    predictor = talker.get("code_predictor_config", {})
    encoder, decoder = codec.get("encoder_config", {}), codec.get("decoder_config", {})
    for value in (config, talker, predictor, codec, encoder, decoder):
        if value.get("quantization_config") or value.get("quantization"):
            raise ValueError("QWEN_TTS_QUANTIZATION_UNSUPPORTED")
        if (
            value.get("tie_word_embeddings", False)
            or value.get("pruned_heads")
            or value.get("add_cross_attention", False)
        ):
            raise ValueError("QWEN_TTS_GRAPH_OPTIONS")
    for value in (talker, predictor, encoder, decoder):
        _dimensions(
            value,
            {
                "hidden_size": (16, 4096),
                "intermediate_size": (16, 16384),
                "num_hidden_layers": (1, 64),
                "num_attention_heads": (1, 32),
                "num_key_value_heads": (1, 32),
                "head_dim": (4, 256),
                "max_position_embeddings": (1024, 131072),
            },
        )
        if (
            value["num_attention_heads"] % value["num_key_value_heads"]
            or value["head_dim"] % 2
            or value.get("attention_bias", False)
            or value.get("attention_dropout", 0) != 0
        ):
            raise ValueError("QWEN_TTS_ATTENTION")
        for key, low, high in (("rope_theta", 100, 10000000), ("rms_norm_eps", 1e-8, 0.1)):
            number = value.get(key, value.get("norm_eps", 1e-5))
            if (
                type(number) not in {float, int}
                or not math.isfinite(number)
                or not low <= number <= high
            ):
                raise ValueError("QWEN_TTS_NUMERIC_OPTIONS")
        if value in (encoder, decoder) and value.get("rope_scaling") is not None:
            raise ValueError("QWEN_TTS_POSITIONS")
    for value in (talker, predictor):
        if (
            value.get("hidden_act") != "silu"
            or value.get("num_code_groups") != 16
            or value.get("use_sliding_window", False)
            or value.get("sliding_window") is not None
            or value.get("layer_types", ["full_attention"] * value["num_hidden_layers"])
            != ["full_attention"] * value["num_hidden_layers"]
        ):
            raise ValueError("QWEN_TTS_TALKER")
    _dimensions(
        talker,
        {
            "text_hidden_size": (16, 4096),
            "text_vocab_size": (256, 300000),
            "vocab_size": (1056, 9216),
        },
    )
    _integer(predictor.get("vocab_size"), 32, 8192)
    if talker["vocab_size"] != predictor["vocab_size"] + 1024:
        raise ValueError("QWEN_TTS_CODEC_VOCABULARY")
    rope = talker.get("rope_scaling", {})
    if (
        rope.get("rope_type") != "default"
        or rope.get("type", "default") != "default"
        or rope.get("interleaved") is not True
        or predictor.get("rope_scaling") is not None
        or set(rope) - {"type", "rope_type", "interleaved", "mrope_section"}
        or not isinstance(rope.get("mrope_section"), list)
        or len(rope["mrope_section"]) != 3
        or any(type(v) is not int or v < 1 for v in rope["mrope_section"])
        or sum(rope["mrope_section"]) * 2 != talker["head_dim"]
    ):
        raise ValueError("QWEN_TTS_POSITIONS")
    for key in (*MARKERS, "assistant_token_id"):
        _integer(config.get(key), 0, talker["text_vocab_size"] - 1)
    if len({config[k] for k in (*MARKERS, "assistant_token_id")}) != 6:
        raise ValueError("QWEN_TTS_FRAMING_TOKENS")
    special = [
        talker.get(k)
        for k in (
            "codec_bos_id",
            "codec_eos_token_id",
            "codec_pad_id",
            "codec_think_id",
            "codec_nothink_id",
            "codec_think_bos_id",
            "codec_think_eos_id",
        )
    ]
    speakers, dialects, languages = (
        talker.get(k, {}) for k in ("spk_id", "spk_is_dialect", "codec_language_id")
    )
    if not speakers or set(speakers) - VOICES or set(dialects) != set(speakers):
        raise ValueError("QWEN_TTS_VOICES")
    if set(languages) != (set(LANGUAGES.values()) - {"auto"}) | {
        "beijing_dialect",
        "sichuan_dialect",
    }:
        raise ValueError("QWEN_TTS_LANGUAGES")
    if any(
        v is not False and v not in {"beijing_dialect", "sichuan_dialect"}
        for v in dialects.values()
    ):
        raise ValueError("QWEN_TTS_DIALECT")
    for value in [*special, *speakers.values(), *languages.values()]:
        _integer(value, predictor["vocab_size"], talker["vocab_size"] - 1)
    if len(set(special)) != len(special):
        raise ValueError("QWEN_TTS_CODEC_TOKENS")
    if (
        codec.get("input_sample_rate") != 24000
        or codec.get("output_sample_rate") != 24000
        or codec.get("decode_upsample_rate") != 1920
        or codec.get("encode_downsample_rate") != 1920
        or codec.get("encoder_valid_num_quantizers") != 16
        or decoder.get("num_quantizers") != 16
        or encoder.get("num_quantizers") != 32
        or decoder.get("codebook_size") != predictor["vocab_size"]
        or encoder.get("codebook_size") != predictor["vocab_size"]
        or decoder.get("upsample_rates") != [8, 5, 4, 3]
        or decoder.get("upsampling_ratios") != [2, 2]
        or encoder.get("upsampling_ratios") != [8, 6, 5, 4]
        or encoder.get("sampling_rate") != 24000
        or encoder.get("audio_channels") != 1
        or encoder.get("use_causal_conv") is not True
        or encoder.get("use_conv_shortcut") is not False
        or encoder.get("pad_mode") != "constant"
        or encoder.get("use_streaming", False)
        or encoder.get("hidden_act") != "gelu"
        or decoder.get("hidden_act") != "silu"
    ):
        raise ValueError("QWEN_TTS_AUDIO_CODEC")
    for value, keys in (
        (encoder, ("num_filters", "codebook_dim", "vector_quantization_hidden_dimension")),
        (
            decoder,
            ("latent_dim", "codebook_dim", "decoder_dim", "vector_quantization_hidden_dimension"),
        ),
    ):
        _dimensions(value, {k: (4, 2048) for k in keys})
    if decoder["decoder_dim"] % 16 or decoder["codebook_dim"] % 2:
        raise ValueError("QWEN_TTS_AUDIO_DIMENSIONS")
    _dimensions(
        encoder,
        {
            "compress": (1, 8),
            "dilation_growth_rate": (1, 4),
            "num_residual_layers": (1, 4),
            "kernel_size": (1, 15),
            "last_kernel_size": (1, 15),
            "residual_kernel_size": (1, 15),
            "upsample_groups": (1, 4096),
            "sliding_window": (1, 4096),
        },
    )
    _dimensions(decoder, {"sliding_window": (1, 4096)})
    if encoder.get("num_semantic_quantizers") != 1 or decoder.get("num_semantic_quantizers") != 1:
        raise ValueError("QWEN_TTS_SEMANTIC_CODEC")
    return config


def controls(config):
    voices = sorted(config["talker_config"]["spk_id"])
    return {
        "voices": [{"id": v, "name": v.replace("_", " ").title()} for v in voices],
        "languages": sorted(LANGUAGES),
        "defaultVoice": "ryan" if "ryan" in voices else voices[0],
    }


def input_value(value, config=None):
    if not isinstance(value, dict) or set(value) - {"text", "voice", "language", "speed"}:
        raise ValueError("QWEN_TTS_INPUT")
    default = controls(config)["defaultVoice"] if config else "ryan"
    text, voice, language, speed = (
        value.get(k, d)
        for k, d in (("text", None), ("voice", default), ("language", "en-us"), ("speed", 1))
    )
    if (
        not isinstance(text, str)
        or not 1 <= len(text) <= 350
        or not text.strip()
        or not isinstance(voice, str)
        or voice not in (config["talker_config"]["spk_id"] if config else VOICES)
        or not isinstance(language, str)
        or language not in LANGUAGES
        or type(speed) not in {int, float}
        or speed != 1
        or any(marker in text for marker in (*MARKERS.values(), "<|endoftext|>"))
    ):
        raise ValueError("QWEN_TTS_INPUT")
    return {"text": text, "voice": voice, "language": language, "speed": 1.0}


def native_models(config, codec):
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Config,
    )
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Model,
    )
    from transformers import GenerationConfig, PretrainedConfig

    validate_config(config, codec)
    configs = [
        Qwen3TTSConfig(**copy.deepcopy(config)),
        Qwen3TTSTokenizerV2Config(**copy.deepcopy(codec)),
    ]
    for native in configs:
        stack = [native]
        while stack:
            item = stack.pop()
            item._attn_implementation = "eager"
            stack.extend(v for v in vars(item).values() if isinstance(v, PretrainedConfig))
    models = Qwen3TTSForConditionalGeneration(configs[0]), Qwen3TTSTokenizerV2Model(configs[1])
    for model in models:
        for module in model.modules():
            if getattr(module, "generation_config", None) is not None:
                module.generation_config = GenerationConfig(
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=config["talker_config"]["codec_eos_token_id"],
                )
        model.eval()
    return models


def validate_weight_shapes(config, codec, graphs):
    import torch

    with torch.device("meta"):
        models = native_models(config, codec)
    for model, shapes in zip(models, graphs, strict=True):
        expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        if expected != {k: tuple(v) for k, v in shapes.items()}:
            raise ValueError("QWEN_TTS_WEIGHTS_INCOMPLETE")


def validate_tokenizer_metadata(value):
    validate_model_config(value)
    if value.get("tokenizer_class") not in {"Qwen2Tokenizer", "Qwen2TokenizerFast"}:
        raise ValueError("QWEN_TTS_TOKENIZER")


def load_tokenizer(root, config):
    from transformers import Qwen2Tokenizer

    validate_tokenizer_metadata(metadata_file(root / "tokenizer_config.json"))
    tokenizer = Qwen2Tokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    tokenizer.chat_template = None
    vocab = tokenizer.get_vocab()
    if (
        set(vocab.values()) != set(range(len(vocab)))
        or not 0 <= config["talker_config"]["text_vocab_size"] - len(vocab) <= 1024
        or any(
            tokenizer.encode(marker, add_special_tokens=False) != [config[key]]
            for key, marker in MARKERS.items()
        )
        or tokenizer.encode("assistant", add_special_tokens=False) != [config["assistant_token_id"]]
    ):
        raise ValueError("QWEN_TTS_TOKENIZER_VOCABULARY")
    return tokenizer


def preflight(root):
    config, codec = (
        metadata_file(root / "config.json"),
        metadata_file(root / "speech_tokenizer/config.json"),
    )
    validate_weight_shapes(
        config,
        codec,
        [preflight_safetensors(root), preflight_safetensors(root / "speech_tokenizer")],
    )
    tokenizer = load_tokenizer(root, config)
    return config, codec, tokenizer


def verify_runtime():
    import hashlib
    import importlib.util
    from importlib.metadata import version
    from pathlib import Path

    from oryn_engine.generated_capabilities import generated_artifact

    expected = generated_artifact()["buildInputs"]["qwenTtsRuntime"]
    versions = {name: version(name) for name in expected["dependencies"]}
    spec = importlib.util.find_spec("qwen_tts")
    if versions != expected["dependencies"] or spec is None or not spec.origin:
        raise ValueError("QWEN_TTS_DEPENDENCY_CLOSURE")
    package = Path(spec.origin).parent
    sources = {
        path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in package.rglob("*.py")
    }
    if sources != expected["sources"]:
        raise ValueError("QWEN_TTS_NATIVE_SOURCE_IDENTITY")
    return versions


def load(root, backend):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    if backend not in {"cpu", "mps", "cuda"}:
        raise ValueError("QWEN_TTS_BACKEND")
    config, codec, tokenizer = preflight(root)
    roots = root, root / "speech_tokenizer"
    # Keep native deterministic convolution/rotary buffers on CPU. Only parameters
    # are empty; every serialized parameter and persistent buffer is assigned below.
    with init_empty_weights(include_buffers=False):
        models = native_models(config, codec)
    for model, directory in zip(models, roots, strict=True):
        weights = load_file(directory / "model.safetensors", device="cpu")
        if any(not v.is_floating_point() or not torch.isfinite(v).all() for v in weights.values()):
            raise ValueError("QWEN_TTS_WEIGHT_DTYPE")
        model.load_state_dict(weights, strict=True, assign=True)
        model.to(device=backend, dtype=torch.float32).eval()
    return (*models, tokenizer, config)


def synthesize(model, codec, tokenizer, config, value, *, max_codes=MAX_CODES):
    import torch

    value = input_value(value, config)
    _integer(max_codes, 3, MAX_CODES)
    text = value["text"]
    prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    prefix = [config["im_start_token_id"], config["assistant_token_id"]]
    if len(ids) > MAX_TEXT_TOKENS or ids[:2] != prefix or ids[-3:-1] != prefix:
        raise ValueError("QWEN_TTS_TOKEN_BUDGET")
    with torch.inference_mode():
        codes, _ = model.generate(
            input_ids=[torch.tensor([ids], device=model.device)],
            instruct_ids=[None],
            languages=[LANGUAGES[value["language"]]],
            speakers=[value["voice"]],
            non_streaming_mode=True,
            max_new_tokens=max_codes,
            do_sample=False,
            subtalker_dosample=False,
            repetition_penalty=1.05,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            subtalker_top_k=0,
            subtalker_top_p=1.0,
            subtalker_temperature=1.0,
        )
        if (
            len(codes) != 1
            or not 1 <= codes[0].shape[0] <= max_codes
            or codes[0].shape[1] != 16
            or codes[0].min() < 0
            or codes[0].max() >= config["talker_config"]["code_predictor_config"]["vocab_size"]
        ):
            raise ValueError("QWEN_TTS_AUDIO_CODES")
        audio = (
            codec.decode(codes[0].unsqueeze(0), return_dict=True)
            .audio_values[0]
            .float()
            .cpu()
            .numpy()
        )
    if (
        audio.ndim != 1
        or not 1 <= audio.size <= MAX_CODES * 1920
        or not all(math.isfinite(v) for v in audio)
    ):
        raise ValueError("QWEN_TTS_AUDIO_OUTPUT")
    return audio, 24000
