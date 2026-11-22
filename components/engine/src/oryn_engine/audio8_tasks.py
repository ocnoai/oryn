"""Owned bounded Audio8 ASR over pinned native Voxtral and Qwen2 graphs.

Publisher graph/clock reviewed at Edge0-AI/Audio8-ASR-Infinite
c8ba8eea829be0339e8d7757f8ca52dac06e1e32 (Apache-2.0).
Repository Python files and templates are never imported or executed.
"""

from __future__ import annotations

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

FAMILY = "audio8_asr_infinite"
ARCHITECTURE = "Audio8ASRInfiniteForConditionalGeneration"
TASK = "automatic-speech-recognition"
METADATA_NAMES = ("preprocessor_config.json", "tokenizer_config.json")
ALIASES = {
    "AutoConfig": "configuration_audio8_asr_infinite.Audio8ASRInfiniteConfig",
    "AutoModelForCausalLM": (
        "modeling_audio8_asr_infinite.Audio8ASRInfiniteForConditionalGeneration"
    ),
}
FEATURES = {
    "feature_extractor_type": "VoxtralRealtimeFeatureExtractor",
    "feature_size": 128,
    "global_log_mel_max": 1.5,
    "hop_length": 160,
    "n_fft": 400,
    "padding_side": "right",
    "padding_value": 0.0,
    "return_attention_mask": True,
    "sampling_rate": 16000,
    "win_length": 400,
}
ROOT_FIELDS = {
    "hidden_size",
    "transformers_version",
    "frame_lens",
    "tie_word_embeddings",
    "max_frame_len",
    "weight_format_version",
    "num_delay_tokens_by_frame_len",
    "semantic_vad_horizons_seconds",
    "audio_config",
    "dtype",
    "use_frame_len_embedding",
    "streaming_frame_ms",
    "token_duration_ms",
    "streaming_n_left_pad_tokens",
    "processor_class",
    "tokenizer_class",
    "projection_size",
    "semantic_vad_num_classes",
    "audio_tower_frame_ms",
    "vocab_size",
    "model_type",
    "audio_length_per_tok",
    "bos_token_id",
    "architectures",
    "eos_token_id",
    "target_delay_ms",
    "supported_frame_lens",
    "default_num_delay_tokens",
    "pad_token_id",
    "use_cache",
    "projector_hidden_act",
    "streaming_n_left_pad_tokens_by_frame_len",
    "text_config",
}
AUDIO_FIELDS = {
    "attention_dropout",
    "hidden_size",
    "max_position_embeddings",
    "rope_parameters",
    "num_mel_bins",
    "vocab_size",
    "num_attention_heads",
    "model_type",
    "head_dim",
    "dtype",
    "rms_norm_eps",
    "num_hidden_layers",
    "intermediate_size",
    "initializer_range",
    "activation_function",
    "hidden_act",
    "sliding_window",
    "streaming_n_left_pad_tokens",
}
TEXT_FIELDS = {
    "attention_dropout",
    "hidden_size",
    "projection_size",
    "max_position_embeddings",
    "tie_word_embeddings",
    "rope_parameters",
    "vocab_size",
    "num_attention_heads",
    "num_key_value_heads",
    "model_type",
    "max_window_layers",
    "use_sliding_window",
    "bos_token_id",
    "dtype",
    "architectures",
    "eos_token_id",
    "rms_norm_eps",
    "num_hidden_layers",
    "intermediate_size",
    "initializer_range",
    "layer_types",
    "hidden_act",
    "pad_token_id",
    "use_cache",
    "sliding_window",
    "_name_or_path",
}
SPECIAL = ("[STREAMING_PAD]", "[STREAMING_WORD]", "[LANGUAGE_ZH]", "[LANGUAGE_EN]")


def supports(config, task):
    return task == TASK and config.get("model_type") == FAMILY


def normalized_config(config, task):
    if not supports(config, task):
        return config
    if config.get("auto_map") not in (None, ALIASES):
        raise ValueError("AUDIO8_REMOTE_MAPPING")
    clean = {k: v for k, v in config.items() if k != "auto_map"}
    validate_config(clean)
    return clean


def validate_config(config):
    validate_model_config(config)
    audio, text = config.get("audio_config"), config.get("text_config")
    if (
        not isinstance(audio, dict)
        or not isinstance(text, dict)
        or set(config) - ROOT_FIELDS
        or set(audio) - AUDIO_FIELDS
        or set(text) - TEXT_FIELDS
    ):
        raise ValueError("AUDIO8_CONFIGURATION")
    fixed = {
        "model_type": FAMILY,
        "architectures": [ARCHITECTURE],
        "weight_format_version": 2,
        "supported_frame_lens": [4, 6, 8],
        "frame_lens": [4, 6, 8],
        "audio_tower_frame_ms": 20,
        "audio_length_per_tok": 8,
        "max_frame_len": 8,
        "use_frame_len_embedding": True,
        "projector_hidden_act": "gelu",
        "semantic_vad_horizons_seconds": [0.5, 1.0, 2.0, 3.0],
        "semantic_vad_num_classes": 8,
        "streaming_n_left_pad_tokens": 18,
        "streaming_n_left_pad_tokens_by_frame_len": {"4": 18, "6": 12, "8": 9},
        "tie_word_embeddings": True,
        "processor_class": "VoxtralRealtimeFeatureExtractor",
        "tokenizer_class": "Qwen2Tokenizer",
        "default_num_delay_tokens": None,
        "streaming_frame_ms": None,
        "token_duration_ms": None,
        "target_delay_ms": [240, 320, 480, 560],
        "num_delay_tokens_by_frame_len": {
            "4": {"240": 3, "320": 4, "480": 6, "560": 7},
            "6": {"240": 2, "480": 4},
            "8": {"320": 2, "480": 3},
        },
    }
    audio_fixed = {
        "model_type": "voxtral_realtime_encoder",
        "num_mel_bins": 128,
        "activation_function": "gelu",
        "hidden_act": "silu",
        "attention_dropout": 0.0,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 1500,
        "sliding_window": 750,
        "streaming_n_left_pad_tokens": 18,
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
    }
    text_fixed = {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_act": "silu",
        "attention_dropout": 0.0,
        "rms_norm_eps": 1e-6,
        "sliding_window": None,
        "use_sliding_window": False,
        "tie_word_embeddings": True,
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
    }
    if any(
        part.get(k) != v
        for part, expected in ((config, fixed), (audio, audio_fixed), (text, text_fixed))
        for k, v in expected.items()
    ):
        raise ValueError("AUDIO8_GRAPH")
    for part, bounds in (
        (
            audio,
            {
                "hidden_size": (16, 2048),
                "intermediate_size": (32, 8192),
                "num_hidden_layers": (1, 48),
                "num_attention_heads": (1, 32),
                "head_dim": (4, 128),
            },
        ),
        (
            text,
            {
                "hidden_size": (16, 4096),
                "intermediate_size": (32, 16384),
                "num_hidden_layers": (1, 64),
                "num_attention_heads": (1, 32),
                "num_key_value_heads": (1, 32),
                "vocab_size": (16, 262144),
                "max_position_embeddings": (512, 131072),
            },
        ),
    ):
        for key, (low, high) in bounds.items():
            if type(part.get(key)) is not int or not low <= part[key] <= high:
                raise ValueError("AUDIO8_GEOMETRY")
    if (
        audio["head_dim"] % 2
        or text["hidden_size"] % text["num_attention_heads"]
        or (text["hidden_size"] // text["num_attention_heads"]) % 2
        or text["hidden_size"] % 2
        or text["num_attention_heads"] % text["num_key_value_heads"]
        or audio["intermediate_size"] != 4 * audio["hidden_size"]
        or text.get("layer_types") != ["full_attention"] * text["num_hidden_layers"]
    ):
        raise ValueError("AUDIO8_GEOMETRY")
    for key in ("hidden_size", "vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
        if config.get(key) != text.get(key):
            raise ValueError("AUDIO8_TEXT_IDENTITY")
    if (
        config.get("projection_size") != audio["hidden_size"] * 8
        or text.get("projection_size") != config["projection_size"]
    ):
        raise ValueError("AUDIO8_PROJECTION")
    ids = [text.get(k) for k in ("bos_token_id", "eos_token_id", "pad_token_id")]
    if (
        any(type(i) is not int or not 0 <= i < text["vocab_size"] for i in ids)
        or len(set(ids)) != 3
    ):
        raise ValueError("AUDIO8_TOKEN_IDS")
    return config


def validate_metadata(name, metadata, config):
    validate_model_config(metadata)
    if name == "preprocessor_config.json":
        if metadata != FEATURES:
            raise ValueError("AUDIO8_FRONTEND")
    elif name == "tokenizer_config.json":
        expected = {
            "add_prefix_space": False,
            "backend": "tokenizers",
            "bos_token": "<|im_start|>",
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "clean_up_tokenization_spaces": False,
            "errors": "replace",
            "extra_special_tokens": list(SPECIAL),
            "split_special_tokens": False,
            "tokenizer_class": "Qwen2Tokenizer",
            "unk_token": None,
        }
        if set(metadata) - set(expected) - {
            "is_local",
            "local_files_only",
            "model_max_length",
        } or any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError("AUDIO8_TOKENIZER")
    else:
        raise ValueError("AUDIO8_METADATA")


def components(root, config):
    from transformers import Qwen2Tokenizer, VoxtralRealtimeFeatureExtractor

    for name in METADATA_NAMES:
        validate_metadata(name, metadata_file(root / name), config)
    path = root / "tokenizer.json"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 16 * 1024**2:
        raise ValueError("AUDIO8_TOKENIZER_FILE")
    from tokenizers import Tokenizer

    original_vocab = Tokenizer.from_file(str(path)).get_vocab()
    tokenizer = Qwen2Tokenizer(
        tokenizer_file=str(path),
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        extra_special_tokens=list(SPECIAL),
        clean_up_tokenization_spaces=False,
    )
    tokenizer.chat_template = None
    vocab = tokenizer.get_vocab()
    if (
        vocab != original_vocab
        or not 16 <= len(vocab) <= config["vocab_size"]
        or config["vocab_size"] - len(vocab) > 512
        or set(vocab.values()) != set(range(len(vocab)))
        or any(
            getattr(tokenizer, k) != config[k]
            for k in ("bos_token_id", "eos_token_id", "pad_token_id")
        )
    ):
        raise ValueError("AUDIO8_VOCABULARY")
    for token in SPECIAL:
        if (
            token not in vocab
            or tokenizer.encode(token, add_special_tokens=False) != [vocab[token]]
            or vocab[token] not in tokenizer.all_special_ids
        ):
            raise ValueError("AUDIO8_CONTROL_TOKENS")
    return VoxtralRealtimeFeatureExtractor(
        **{k: v for k, v in FEATURES.items() if k != "feature_extractor_type"}
    ), tokenizer


def native_model(config):
    from oryn_engine.audio8_model import Audio8Model

    validate_config(config)
    return Audio8Model(config)


def load_components(root, config, backend):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    validate_config(config)
    if not (root / "model.safetensors.index.json").is_file():
        raise ValueError("AUDIO8_INDEX_REQUIRED")
    processor = components(root, config)
    shapes = preflight_safetensors(root)
    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    expected = model.state_dict()
    tied = "language_model.lm_head.weight"
    source = "language_model.model.embed_tokens.weight"
    if shapes != {k: tuple(v.shape) for k, v in expected.items() if k != tied}:
        raise ValueError("AUDIO8_WEIGHTS_INCOMPLETE")
    weights = {}
    for path in sorted(root.glob("*.safetensors")):
        values = load_file(path, device="cpu")
        if weights.keys() & values.keys():
            raise ValueError("AUDIO8_WEIGHTS_DUPLICATE")
        for key, value in values.items():
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError("AUDIO8_WEIGHTS_NUMERIC")
            weights[key] = value.float()
    weights[tied] = weights[source]
    model.load_state_dict(weights, assign=True, strict=True)
    model.language_model.lm_head.weight = model.language_model.model.embed_tokens.weight
    model.eval().to(backend)
    return model, processor, config


def audio_windows(audio):
    """Original 80 ms clock, 52.5 ms look-back and 2.5 ms look-ahead."""
    import numpy as np

    audio = np.asarray(audio)
    if audio.ndim != 1 or not 320 <= audio.size <= 480000 or not np.isfinite(audio).all():
        raise ValueError("AUDIO8_AUDIO_BOUND")
    stream = np.concatenate(
        (np.zeros(18 * 1280, np.float32), audio.astype(np.float32), np.zeros(17 * 1280, np.float32))
    )
    start, end = 0, 25 * 1280
    while end + 40 <= stream.size:
        yield stream[max(0, start - 840) : end + 40], (end - start) // 1280
        start, end = end, end + 1280


def transcribe(model, processor, config, audio, backend, language):
    import torch

    feature, tokenizer = processor
    if language not in {"en", "zh"}:
        raise ValueError("AUDIO8_LANGUAGE")
    vocab = tokenizer.get_vocab()
    pending = [config["bos_token_id"], vocab["[LANGUAGE_" + language.upper() + "]"]] + [
        vocab["[STREAMING_PAD]"]
    ] * 23
    audio_cache = text_cache = None
    seen = 0
    generated = []
    with torch.inference_mode():
        cond = model.condition()
        for window, count in audio_windows(audio):
            # CPU float32 STFT matches the pinned frontend on every device.
            features = feature._torch_extract_fbank_features(
                torch.from_numpy(window.copy()).unsqueeze(0), device="cpu", center=True
            ).to(backend)
            conv = model.audio_tower.embedder(features)
            if (
                conv.ndim != 3
                or conv.shape[0] != 1
                or conv.shape[1] < count * 4
                or not bool(torch.isfinite(conv).all())
            ):
                raise ValueError("AUDIO8_FRONTEND_OUTPUT")
            conv = conv[:, -count * 4 :, :]
            encoded = model.audio_tower(
                inputs_embeds=conv, past_key_values=audio_cache, use_cache=True, return_dict=True
            )
            audio_cache = encoded.past_key_values
            hidden = encoded.last_hidden_state
            if tuple(hidden.shape) != (
                1,
                count * 4,
                config["audio_config"]["hidden_size"],
            ) or not bool(torch.isfinite(hidden).all()):
                raise ValueError("AUDIO8_ENCODER_OUTPUT")
            projected = model.project(hidden, count)
            ids = torch.tensor([pending], dtype=torch.long, device=backend)
            if ids.shape != (1, count) or projected.shape != (1, count, config["hidden_size"]):
                raise ValueError("AUDIO8_CLOCK_ALIGNMENT")
            embeds = model.language_model.model.embed_tokens(ids) + projected
            seen += count
            if seen > 409:
                raise ValueError("AUDIO8_CLOCK_BOUND")
            out = model.language_model(
                inputs_embeds=embeds,
                attention_mask=torch.ones((1, seen), dtype=torch.long, device=backend),
                past_key_values=text_cache,
                use_cache=True,
                logits_to_keep=1,
                t_cond=cond,
            )
            text_cache = out.past_key_values
            logits = out.logits
            if logits.shape != (1, 1, config["vocab_size"]) or not bool(
                torch.isfinite(logits).all()
            ):
                raise ValueError("AUDIO8_LOGITS")
            # Preserve native greedy choice; reject a padded row without a token identity.
            logits = logits[0, -1, :].clone()
            logits[config["eos_token_id"]] = -torch.inf
            token = int(logits.argmax().item())
            if token >= len(vocab):
                raise ValueError("AUDIO8_OUTPUT_TOKEN")
            generated.append(token)
            pending = [token]
        if not 1 <= len(generated) <= 385:
            raise ValueError("AUDIO8_GENERATION_BOUND")
    text = tokenizer.decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    if len(text.encode()) > 16384:
        raise ValueError("AUDIO8_OUTPUT_BOUND")
    return {"text": text, "language": language}


def execute(model, processor, config, value, backend, input_path, emit):
    from oryn_engine.vad_tasks import read_audio

    language = value.get("language", "en")
    if value.get("timestamps", False) or language not in {"en", "zh"}:
        raise ValueError("AUDIO8_PARAMETERS")
    audio, _ = read_audio(input_path)
    emit("job.progress", {"stage": "transcribing", "progress": 0.3})
    return transcribe(model, processor, config, audio, backend, language)
