"""Instruction-grounded native audio families; templates and paths are pack-owned."""

from __future__ import annotations

TASK_CONTRACTS = {"audio-text-to-text": "audio-question-answering-v1"}
FLAMINGO_FAMILIES = frozenset({"audioflamingo3", "musicflamingo"})
TASK_FAMILIES = {"audio-text-to-text": {"qwen2_audio", "voxtral", *FLAMINGO_FAMILIES}}


def validate_config(config):
    if config.get("model_type") == "voxtral":
        from oryn_engine.voxtral_tasks import validate_config as validate_voxtral

        return validate_voxtral(config)
    if config.get("model_type") not in FLAMINGO_FAMILIES:
        return
    audio, text = config.get("audio_config", {}), config.get("text_config", {})
    if (
        not isinstance(audio, dict)
        or not isinstance(text, dict)
        or audio.get("model_type") != "audioflamingo3_encoder"
        or audio.get("num_mel_bins", 128) != 128
        or audio.get("max_source_positions", 1500) != 1500
        or text.get("model_type") != "qwen2"
    ):
        raise ValueError("Native Flamingo audio/text graph is unsupported")
    vocab = text.get("vocab_size")
    tokens = ["audio_token_id"] + (
        ["audio_bos_token_id", "audio_eos_token_id"]
        if config["model_type"] == "musicflamingo"
        else []
    )
    if (
        type(vocab) is not int
        or not 32 <= vocab <= 262144
        or any(type(config.get(key)) is not int or not 0 <= config[key] < vocab for key in tokens)
        or len({config[key] for key in tokens}) != len(tokens)
    ):
        raise ValueError("Native Flamingo special-token identities are invalid")


def flamingo_processor_options(metadata, family):
    from oryn_engine.model_metadata import validate_model_config

    validate_model_config(metadata)
    expected = "MusicFlamingoProcessor" if family == "musicflamingo" else "AudioFlamingo3Processor"
    if metadata.get("processor_class") != expected or set(metadata) - {
        "processor_class",
        "audio_token",
        "audio_bos_token",
        "audio_eos_token",
        "feature_extractor",
        "max_audio_len",
        "default_transcription_prompt",
    }:
        raise ValueError("Native Flamingo processor identity is unsupported")
    for key, value in {
        "audio_token": "<sound>",
        "audio_bos_token": "<|sound_bos|>",
        "audio_eos_token": "<|sound_eos|>",
    }.items():
        if key in metadata and metadata[key] != value:
            raise ValueError("Native Flamingo audio framing is unsupported")
    length = metadata.get("max_audio_len", 30)
    if type(length) is not int or not 30 <= length <= 1200:
        raise ValueError("Native Flamingo audio window is invalid")
    features = metadata.get("feature_extractor")
    expected_features = {
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": 128,
        "sampling_rate": 16000,
        "chunk_length": 30,
        "hop_length": 160,
        "n_fft": 400,
        "n_samples": 480000,
        "nb_max_frames": 3000,
        "padding_side": "right",
        "padding_value": 0.0,
        "return_attention_mask": True,
        "dither": 0.0,
    }
    if not isinstance(features, dict) or set(features) - {*expected_features, "processor_class"}:
        raise ValueError("Native Flamingo feature extraction is unsupported")
    for key, value in expected_features.items():
        if key in features and (type(features[key]) is not type(value) or features[key] != value):
            raise ValueError("Native Flamingo feature extraction is unsupported")
    if features.get("feature_extractor_type") != "WhisperFeatureExtractor":
        raise ValueError("Native Flamingo feature extraction is unsupported")
    return {
        key: value for key, value in expected_features.items() if key != "feature_extractor_type"
    }


def load_flamingo(root, config, common):
    import transformers as tf

    from oryn_engine.model_metadata import metadata_file

    family = config["model_type"]
    options = flamingo_processor_options(metadata_file(root / "processor_config.json"), family)
    tokenizer = tf.AutoTokenizer.from_pretrained(root, **common)
    tokenizer.chat_template = None
    tokens = {"<sound>": "audio_token_id"}
    if family == "musicflamingo":
        tokens.update(
            {"<|sound_bos|>": "audio_bos_token_id", "<|sound_eos|>": "audio_eos_token_id"}
        )
    for token, key in tokens.items():
        identifier = tokenizer.convert_tokens_to_ids(token)
        if identifier is None or identifier == tokenizer.unk_token_id or identifier != config[key]:
            raise ValueError("Native Flamingo tokenizer/model identities disagree")
    for token in ("<|im_start|>", "<|im_end|>"):
        identifier = tokenizer.convert_tokens_to_ids(token)
        if identifier is None or identifier == tokenizer.unk_token_id:
            raise ValueError("Native Flamingo conversation token is missing")
    processor_class, model_class = (
        (tf.MusicFlamingoProcessor, tf.MusicFlamingoForConditionalGeneration)
        if family == "musicflamingo"
        else (tf.AudioFlamingo3Processor, tf.AudioFlamingo3ForConditionalGeneration)
    )
    return processor_class(
        feature_extractor=tf.WhisperFeatureExtractor(**options),
        tokenizer=tokenizer,
        chat_template="",
        max_audio_len=30,
    ), model_class


def load_components(root, config, common):
    import transformers as tf

    from oryn_engine.model_metadata import metadata_file, validate_model_config

    if config.get("model_type") == "voxtral":
        from oryn_engine.voxtral_tasks import Processor

        return Processor(root, config), tf.VoxtralForConditionalGeneration
    if config.get("model_type") in FLAMINGO_FAMILIES:
        return load_flamingo(root, config, common)
    metadata = metadata_file(root / "preprocessor_config.json")
    validate_model_config(metadata)
    if metadata.get("feature_extractor_type") != "WhisperFeatureExtractor":
        raise ValueError("Qwen audio preprocessing identity is not admitted")
    if metadata.get("sampling_rate", 16000) != 16000 or metadata.get("chunk_length", 30) != 30:
        raise ValueError("Qwen audio preprocessing has unsupported timing")
    extractor = tf.WhisperFeatureExtractor.from_pretrained(root, **common)
    tokenizer = tf.AutoTokenizer.from_pretrained(root, **common)
    for attr, token in (
        ("audio_token", "<|AUDIO|>"),
        ("audio_bos_token", "<|audio_bos|>"),
        ("audio_eos_token", "<|audio_eos|>"),
    ):
        setattr(tokenizer, attr, token)
        identifier = tokenizer.convert_tokens_to_ids(token)
        if identifier is None or identifier == tokenizer.unk_token_id:
            raise ValueError("Qwen audio special token is missing")
    if tokenizer.convert_tokens_to_ids("<|AUDIO|>") != config.get("audio_token_index"):
        raise ValueError("Qwen audio encoder and tokenizer disagree")
    processor = tf.Qwen2AudioProcessor(
        feature_extractor=extractor, tokenizer=tokenizer, chat_template=""
    )
    return processor, tf.Qwen2AudioForConditionalGeneration


def execute(model, processor, value, backend, input_path):
    import torch

    from oryn_engine.runners import _decode_wave
    from oryn_engine.shared_tasks import _move

    if (
        input_path is None
        or input_path.is_symlink()
        or not input_path.is_file()
        or not 1 <= input_path.stat().st_size <= 12 * 1024**2
    ):
        raise ValueError("Owned audio input is unavailable or oversized")
    audio = _decode_wave(input_path, input_path.stat().st_size)
    if not 80 <= audio.size <= 30 * 16000:
        raise ValueError("Audio understanding requires 5 ms to 30 seconds of input")
    if model.config.model_type == "voxtral":
        from oryn_engine.voxtral_tasks import execute as execute_voxtral

        return execute_voxtral(model, processor, audio, value, backend)
    # Repository Jinja is never evaluated. Literal user text remains one user turn.
    instruction = value["instruction"]
    if any(
        marker in instruction
        for marker in (
            "<|im_start|>",
            "<|im_end|>",
            "<|AUDIO|>",
            "<|audio_bos|>",
            "<|audio_eos|>",
            "<sound>",
            "<|sound_bos|>",
            "<|sound_eos|>",
        )
    ):
        raise ValueError("Instruction contains reserved native framing")
    prompt = (
        "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>"
        + instruction
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    flamingo = model.config.model_type in FLAMINGO_FAMILIES
    if flamingo:
        # Literal pack-owned ChatML framing; the native processor expands the
        # sound token and adds Music Flamingo's audio boundaries exactly once.
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<sound>" + instruction + "<|im_end|>\n<|im_start|>assistant\n"
        )
    inputs = _move(
        dict(processor(text=[prompt], audio=[audio], sampling_rate=16000, return_tensors="pt")),
        backend,
    )
    if (
        inputs["input_ids"].shape[-1] > 4096
        or "input_features" not in inputs
        or not bool(torch.isfinite(inputs["input_features"]).all())
    ):
        raise ValueError("Audio context exceeds bounded native input")
    if flamingo and (
        tuple(inputs["input_features"].shape) != (1, 128, 3000)
        or "input_features_mask" not in inputs
        or tuple(inputs["input_features_mask"].shape) != (1, 3000)
    ):
        raise ValueError("Native Flamingo audio chunk geometry is invalid")
    generated = model.generate(
        **inputs, do_sample=False, max_new_tokens=value.get("maxTokens", 128)
    )
    text = processor.batch_decode(
        generated[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )[0].strip()
    if not text or len(text.encode()) > 16384:
        raise ValueError("Grounded audio answer is empty or oversized")
    return {"text": text}
