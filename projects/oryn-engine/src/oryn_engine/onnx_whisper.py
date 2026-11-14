"""Sealed Whisper encoder/decoder binding over the native ONNX foundation.

Greedy decoding recomputes the bounded decoder prefix. No repository generation
code, cache-graph branching, custom operators or timestamp alignment is used.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from oryn_engine.model_metadata import metadata_file


def graph_pairs(names):
    names = set(names)
    pairs = []
    for name in sorted(names):
        path = PurePosixPath(name)
        match = re.fullmatch(r"encoder_model(_(?:quantized|int8|uint8))?\.onnx", path.name)
        if match:
            decoder = str(path.parent / ("decoder_model" + (match[1] or "") + ".onnx"))
            if decoder in names:
                pairs.append((name, decoder))
    return pairs


def validate_config(config):
    if config.get("model_type") != "whisper":
        raise ValueError("ONNX_WHISPER_CONFIGURATION_UNSUPPORTED")
    for key, minimum, maximum in (
        ("d_model", 64, 2048),
        ("decoder_layers", 1, 32),
        ("encoder_layers", 1, 32),
        ("encoder_attention_heads", 1, 32),
        ("decoder_attention_heads", 1, 32),
        ("num_mel_bins", 80, 128),
        ("vocab_size", 51864, 51866),
        ("max_source_positions", 1500, 1500),
        ("max_target_positions", 448, 448),
    ):
        if type(config.get(key)) is not int or not minimum <= config[key] <= maximum:
            raise ValueError("ONNX_WHISPER_CONFIGURATION_UNSUPPORTED")
    if (
        config["d_model"] % config["decoder_attention_heads"]
        or config["d_model"] % config["encoder_attention_heads"]
        or config["num_mel_bins"] not in {80, 128}
    ):
        raise ValueError("ONNX_WHISPER_CONFIGURATION_UNSUPPORTED")
    for key in ("decoder_start_token_id", "eos_token_id", "pad_token_id"):
        if type(config.get(key)) is not int or not 0 <= config[key] < config["vocab_size"]:
            raise ValueError("ONNX_WHISPER_CONFIGURATION_UNSUPPORTED")
    for key in ("suppress_tokens", "begin_suppress_tokens"):
        ids = config.get(key, [])
        if (
            not isinstance(ids, list)
            or len(ids) > 4096
            or any(type(i) is not int or not 0 <= i < config["vocab_size"] for i in ids)
        ):
            raise ValueError("ONNX_WHISPER_CONFIGURATION_UNSUPPORTED")


def _shape(value, dtype, dimensions):
    if (
        value["dtype"] != dtype
        or len(value["shape"]) != len(dimensions)
        or any(
            type(actual) is int and expected is not None and actual != expected
            for actual, expected in zip(value["shape"], dimensions, strict=True)
        )
    ):
        raise ValueError("ONNX_WHISPER_BINDING_UNSUPPORTED")


def validate_graphs(graph, config):
    validate_config(config)
    encoder, decoder = graph["encoder"], graph["decoder"]
    ei, eo, di, do = (
        {v["name"]: v for v in g[k]}
        for g, k in (
            (encoder, "inputs"),
            (encoder, "outputs"),
            (decoder, "inputs"),
            (decoder, "outputs"),
        )
    )
    if (
        ei.keys() != {"input_features"}
        or "last_hidden_state" not in eo
        or di.keys() != {"input_ids", "encoder_hidden_states"}
        or "logits" not in do
    ):
        raise ValueError("ONNX_WHISPER_BINDING_UNSUPPORTED")
    _shape(ei["input_features"], 1, [1, config["num_mel_bins"], 3000])
    _shape(eo["last_hidden_state"], 1, [1, 1500, config["d_model"]])
    for name, value in eo.items():
        if name == "last_hidden_state":
            continue
        match = re.fullmatch(r"encoder_attentions\.(\d+)", name)
        if not match or int(match[1]) >= config["encoder_layers"]:
            raise ValueError("ONNX_WHISPER_BINDING_UNSUPPORTED")
        _shape(value, 1, [1, config["encoder_attention_heads"], 1500, 1500])
    _shape(di["encoder_hidden_states"], 1, [1, 1500, config["d_model"]])
    _shape(di["input_ids"], 7, [1, None])
    _shape(do["logits"], 1, [1, None, config["vocab_size"]])
    if type(di["input_ids"]["shape"][1]) is int or type(do["logits"]["shape"][1]) is int:
        raise ValueError("ONNX_WHISPER_BINDING_UNSUPPORTED")
    for name, value in do.items():
        if name == "logits":
            continue
        attention = re.fullmatch(r"(decoder|cross)_attentions\.(\d+)", name)
        if attention:
            if int(attention[2]) >= config["decoder_layers"]:
                raise ValueError("ONNX_WHISPER_BINDING_UNSUPPORTED")
            _shape(
                value,
                1,
                [
                    1,
                    config["decoder_attention_heads"],
                    None,
                    1500 if attention[1] == "cross" else None,
                ],
            )
            continue
        match = re.fullmatch(r"present\.(\d+)\.(decoder|encoder)\.(key|value)", name)
        if not match or not 0 <= int(match[1]) < config["decoder_layers"]:
            raise ValueError("ONNX_WHISPER_BINDING_UNSUPPORTED")
        _shape(
            value,
            1,
            [
                1,
                config["decoder_attention_heads"],
                1500 if match[2] == "encoder" else None,
                config["d_model"] // config["decoder_attention_heads"],
            ],
        )


def load_components(root, config):
    from transformers import WhisperFeatureExtractor, WhisperTokenizerFast

    validate_config(config)
    pre = metadata_file(root / "preprocessor_config.json")
    if (
        pre.get("feature_extractor_type") != "WhisperFeatureExtractor"
        or pre.get("sampling_rate") != 16000
        or pre.get("n_fft", 400) != 400
        or pre.get("hop_length") != 160
        or pre.get("chunk_length") != 30
        or pre.get("feature_size") != config["num_mel_bins"]
        or pre.get("padding_value", 0) != 0
    ):
        raise ValueError("ONNX_WHISPER_PREPROCESSOR_UNSUPPORTED")
    # Construct only the reviewed native feature extractor. Repository processor
    # classes, dispatch options and chat templates are never interpreted.
    feature = WhisperFeatureExtractor(
        feature_size=config["num_mel_bins"],
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=0,
    )
    tokenizer = WhisperTokenizerFast.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    no_timestamps = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    # Some original ONNX tokenizers omit the 1501 timestamp entries. Their
    # decoder retains those rows; this binding always suppresses them.
    if (
        no_timestamps != config["vocab_size"] - 1502
        or len(tokenizer) not in {config["vocab_size"], config["vocab_size"] - 1501}
        or tokenizer.eos_token_id != config["eos_token_id"]
        or tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        != config["decoder_start_token_id"]
    ):
        raise ValueError("ONNX_WHISPER_VOCABULARY_MISMATCH")
    return feature, tokenizer


def decode_beams(logits, prefix, config, no_timestamps, event):
    """Deterministic width-five beam decoding, bounded by the native context.

    Full-prefix graphs expose no cache inputs. Decode each beam independently
    so original exports with a static batch of one remain valid.
    """
    import numpy as np

    width = 5
    beams, finished = [(0.0, [])], []
    for _ in range(config["max_target_positions"] - len(prefix)):
        candidates = []
        for score, tokens in beams:
            scores = np.array(logits(prefix + tokens), dtype=np.float64, copy=True)
            if scores.shape != (config["vocab_size"],) or not np.isfinite(scores).all():
                raise ValueError("ONNX_WHISPER_DECODER_OUTPUT_INVALID")
            scores[config.get("suppress_tokens", [])] = -np.inf
            scores[no_timestamps + 1 :] = -np.inf
            if not tokens:
                scores[config.get("begin_suppress_tokens", [])] = -np.inf
            if not np.isfinite(scores).any():
                raise ValueError("ONNX_WHISPER_DECODER_OUTPUT_INVALID")
            scores -= scores.max()
            scores -= np.log(np.exp(scores).sum())
            for token in np.argsort(-scores, kind="stable")[: 2 * width]:
                if np.isfinite(scores[token]):
                    candidates.append((score + float(scores[token]), tokens + [int(token)]))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        next_beams = []
        for rank, (score, tokens) in enumerate(candidates):
            if tokens[-1] == config["eos_token_id"]:
                if rank < width:
                    finished.append((score / len(tokens), tokens[:-1]))
            else:
                next_beams.append((score, tokens))
                if len(next_beams) == width:
                    break
        finished.sort(key=lambda item: (-item[0], item[1]))
        finished = finished[:width]
        beams = next_beams
        if len(finished) == width or not beams:
            break
        event(
            "job.progress",
            {
                "stage": "running-model",
                "progress": 0.3
                + 0.6 * (len(prefix) + len(beams[0][1])) / config["max_target_positions"],
            },
        )
    if not finished:
        raise ValueError("ONNX_WHISPER_TOKEN_LIMIT")
    return finished[0][1]


def execute(sessions, root, config, value, path, event):
    import numpy as np
    from transformers.models.whisper.tokenization_whisper import LANGUAGES

    from oryn_engine.vad_tasks import read_audio

    if value.get("timestamps", False) or value.get("language", "auto") != "auto":
        raise ValueError("ONNX_WHISPER_PARAMETER_UNSUPPORTED")
    feature, tokenizer = load_components(root, config)
    audio, _duration = read_audio(path)
    if not np.any(audio):
        return {"text": "", "language": None}
    encoder, decoder = sessions
    features = feature(audio, sampling_rate=16000, return_tensors="np")["input_features"]
    if features.shape != (1, config["num_mel_bins"], 3000) or not np.isfinite(features).all():
        raise ValueError("ONNX_WHISPER_FEATURES_INVALID")
    event("job.progress", {"stage": "running-model", "progress": 0.2})
    hidden = encoder.run(["last_hidden_state"], {"input_features": features.astype(np.float32)})[0]
    if hidden.shape != (1, 1500, config["d_model"]) or not np.isfinite(hidden).all():
        raise ValueError("ONNX_WHISPER_ENCODER_OUTPUT_INVALID")

    def logits(ids):
        result = decoder.run(
            ["logits"],
            {"input_ids": np.asarray([ids], dtype=np.int64), "encoder_hidden_states": hidden},
        )[0]
        if result.shape != (1, len(ids), config["vocab_size"]) or not np.isfinite(result).all():
            raise ValueError("ONNX_WHISPER_DECODER_OUTPUT_INVALID")
        return result[0, -1].astype(np.float64)

    start = config["decoder_start_token_id"]
    first = logits([start])
    probability = np.exp(first - first.max())
    probability /= probability.sum()
    silence_id = tokenizer.convert_tokens_to_ids("<|notimestamps|>") - 1
    if tokenizer.convert_ids_to_tokens(silence_id) not in {"<|nospeech|>", "<|nocaptions|>"}:
        raise ValueError("ONNX_WHISPER_VOCABULARY_MISMATCH")
    if probability[silence_id] > 0.6:
        return {"text": "", "language": None}
    multilingual = config["vocab_size"] >= 51865
    language = "en"
    if multilingual:
        languages = [(code, tokenizer.convert_tokens_to_ids(f"<|{code}|>")) for code in LANGUAGES]
        languages = [(code, i) for code, i in languages if i != tokenizer.unk_token_id]
        if len(languages) < 90 or len({i for _, i in languages}) != len(languages):
            raise ValueError("ONNX_WHISPER_VOCABULARY_MISMATCH")
        language = max(languages, key=lambda item: first[item[1]])[0]
    # Reset optional native tokenizer defaults so English-only checkpoints never
    # acquire a language/task token from incidental repository tokenizer settings.
    tokenizer.language = None
    tokenizer.task = None
    forced = tokenizer.get_decoder_prompt_ids(
        task="transcribe" if multilingual else None,
        language=language if multilingual else None,
        no_timestamps=True,
    )
    ids = [start] + [token for _, token in forced]
    no_timestamps = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    generated = decode_beams(logits, ids, config, no_timestamps, event)
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if len(text.encode()) > 16384:
        raise ValueError("ONNX_WHISPER_OUTPUT_LIMIT")
    return {"text": text, "language": language}
