"""Native CPU Whisper over a validated CTranslate2 numeric container."""

from __future__ import annotations

import math
import re

from oryn_engine.model_metadata import metadata_file, validate_model_config

FAMILY = "oryn_ct2_whisper"
ARCHITECTURE = "CTranslate2Whisper"
CONFIG_KEYS = {"alignment_heads", "lang_ids", "suppress_ids", "suppress_ids_begin"}
FILES = {"config.json", "model.bin", "tokenizer.json", "vocabulary.txt"}


def normalized_config(config, task):
    if task == "automatic-speech-recognition" and set(config) == CONFIG_KEYS:
        validate_config(config)
        return config | {"model_type": FAMILY, "architectures": [ARCHITECTURE]}
    return config


def supports(config, task):
    return task == "automatic-speech-recognition" and config.get("model_type") == FAMILY


def validate_config(config, geometry=None):
    validate_model_config(config)
    if set(config) not in (CONFIG_KEYS, CONFIG_KEYS | {"model_type", "architectures"}):
        raise ValueError("CT2_CONFIGURATION")
    if "model_type" in config and (
        config["model_type"] != FAMILY or config["architectures"] != [ARCHITECTURE]
    ):
        raise ValueError("CT2_CONFIGURATION")
    vocab = geometry["vocab_size"] if geometry else 51866
    for key, limit in (("lang_ids", 100), ("suppress_ids", 4096), ("suppress_ids_begin", 4096)):
        ids = config[key]
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= limit
            or any(type(i) is not int or not 0 <= i < vocab for i in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError("CT2_TOKEN_CONFIG")
    heads = config["alignment_heads"]
    if (
        not isinstance(heads, list)
        or not 1 <= len(heads) <= 1024
        or any(
            not isinstance(h, list)
            or len(h) != 2
            or any(type(v) is not int or not 0 <= v < 32 for v in h)
            for h in heads
        )
    ):
        raise ValueError("CT2_ALIGNMENT_CONFIG")
    if geometry and any(
        layer >= geometry["decoder_layers"] or head >= geometry["decoder_heads"]
        for layer, head in heads
    ):
        raise ValueError("CT2_ALIGNMENT_CONFIG")


def components(root, config, geometry):
    from tokenizers import Tokenizer
    from transformers import WhisperFeatureExtractor

    validate_config(config, geometry)
    for name in ("tokenizer.json", "vocabulary.txt"):
        path = root / name
        if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 4 * 1024**2:
            raise ValueError("CT2_TOKENIZER_FILE")
    # Native CTranslate2 prioritizes vocabulary.json. Never let an unvalidated
    # companion replace the exact vocabulary.txt which this binding checks.
    if (root / "vocabulary.json").exists():
        raise ValueError("CT2_VOCABULARY_OVERRIDE")
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    vocabulary = (root / "vocabulary.txt").read_text().splitlines()
    vocab = tokenizer.get_vocab()
    if (
        len(vocabulary) != geometry["vocab_size"]
        or len(vocab) not in {len(vocabulary), len(vocabulary) - 1501}
        or set(vocab.values()) != set(range(len(vocab)))
        or any(vocab.get(token) != i for i, token in enumerate(vocabulary[: len(vocab)]))
        or len(vocab) != len(vocabulary)
        and (
            vocab.get("<|notimestamps|>") != len(vocab) - 1
            or vocabulary[len(vocab) :] != [f"<|{i * 0.02:.2f}|>" for i in range(1501)]
        )
    ):
        raise ValueError("CT2_VOCABULARY")
    markers = ("<|endoftext|>", "<|startoftranscript|>", "<|transcribe|>", "<|notimestamps|>")
    if any(
        marker not in vocab
        or tokenizer.encode(marker, add_special_tokens=False).ids != [vocab[marker]]
        for marker in markers
    ):
        raise ValueError("CT2_MARKERS")
    if vocab["<|notimestamps|>"] != len(vocabulary) - 1502 or vocabulary[-1501:] != [
        f"<|{i * 0.02:.2f}|>" for i in range(1501)
    ]:
        raise ValueError("CT2_TIMESTAMPS")
    languages = {i: vocabulary[i] for i in config["lang_ids"]}
    if any(not re.fullmatch(r"<\|[a-z]{2,3}\|>", value) for value in languages.values()):
        raise ValueError("CT2_LANGUAGES")
    return WhisperFeatureExtractor(
        feature_size=geometry["num_mel_bins"],
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=0.0,
        return_attention_mask=False,
    ), tokenizer


def load_components(root, config, backend):
    from oryn_engine.ct2_weights import inspect_file

    if backend != "cpu":
        raise ValueError("CT2_BACKEND_UNAVAILABLE")
    # Validate the actual runtime files independently of discovery's metadata.
    raw_config = metadata_file(root / "config.json")
    validate_config(raw_config)
    if {k: config[k] for k in CONFIG_KEYS} != raw_config:
        raise ValueError("CT2_CONFIG_IDENTITY")
    _, geometry = inspect_file(root / "model.bin", numeric=True)
    processor = components(root, config, geometry)
    import ctranslate2

    model = ctranslate2.models.Whisper(
        str(root),
        device="cpu",
        compute_type="float32",
        inter_threads=1,
        intra_threads=2,
        max_queued_batches=1,
    )
    if model.n_mels != geometry["num_mel_bins"]:
        raise ValueError("CT2_NATIVE_GEOMETRY")
    return model, processor, config


def transcribe(model, processor, config, audio):
    import ctranslate2
    import numpy as np

    feature, tokenizer = processor
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not 1 <= audio.size <= 480000 or not np.isfinite(audio).all():
        raise ValueError("CT2_AUDIO_BOUND")
    values = feature(
        audio,
        sampling_rate=16000,
        return_tensors="np",
        padding="max_length",
        max_length=480000,
        truncation=False,
    ).input_features
    if values.shape != (1, model.n_mels, 3000) or not np.isfinite(values).all():
        raise ValueError("CT2_FEATURES")
    encoded = model.encode(ctranslate2.StorageView.from_array(np.ascontiguousarray(values)))
    if not np.isfinite(np.asarray(encoded)).all():
        raise ValueError("CT2_ENCODER_OUTPUT")
    predictions = model.detect_language(encoded)
    if not isinstance(predictions, list) or len(predictions) != 1 or not predictions[0]:
        raise ValueError("CT2_LANGUAGE_OUTPUT")
    vocab = tokenizer.get_vocab()
    language, score = predictions[0][0]
    if (
        not re.fullmatch(r"<\|[a-z]{2,3}\|>", language)
        or vocab.get(language) not in config["lang_ids"]
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise ValueError("CT2_LANGUAGE_OUTPUT")
    prompt = [
        vocab[token]
        for token in ("<|startoftranscript|>", language, "<|transcribe|>", "<|notimestamps|>")
    ]
    result = model.generate(
        encoded,
        [prompt],
        beam_size=1,
        num_hypotheses=1,
        # Native Whisper halves this total bound: at most 224 new tokens.
        max_length=448,
        sampling_topk=1,
        sampling_temperature=1,
        suppress_blank=True,
        suppress_tokens=sorted(
            set(config["suppress_ids"])
            | set(range(vocab["<|notimestamps|>"] + 1, vocab["<|notimestamps|>"] + 1502))
        ),
        return_scores=True,
    )
    if (
        len(result) != 1
        or len(result[0].sequences_ids) != 1
        or len(result[0].scores) != 1
        or not math.isfinite(result[0].scores[0])
    ):
        raise ValueError("CT2_DECODING_RESULT")
    ids = result[0].sequences_ids[0]
    if len(ids) >= 224 or any(type(i) is not int or not 0 <= i < len(vocab) for i in ids):
        raise ValueError("CT2_OUTPUT_BOUND")
    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
    if len(text.encode()) > 16384:
        raise ValueError("CT2_OUTPUT_BOUND")
    return {"text": text, "language": language[2:-2]}


def execute(model, processor, config, value, backend, input_path, emit):
    from oryn_engine.vad_tasks import read_audio

    try:
        if (
            backend != "cpu"
            or value.get("timestamps", False)
            or value.get("language", "auto") != "auto"
        ):
            raise ValueError("CT2_PARAMETERS")
        audio, _ = read_audio(input_path)
        emit("job.progress", {"stage": "transcribing", "progress": 0.3})
        return transcribe(model, processor, config, audio)
    finally:
        model.unload_model()
