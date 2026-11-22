"""Bounded Wav2Vec2 CTC inference; decoder/cache graphs use separate bindings."""

from __future__ import annotations

from oryn_engine.model_metadata import metadata_file

MAX_CTC_VOCABULARY = 4096


def validate_config(config):
    if (
        config.get("model_type") != "wav2vec2"
        or type(config.get("vocab_size")) is not int
        or not 1 <= config["vocab_size"] <= MAX_CTC_VOCABULARY
        or type(config.get("pad_token_id")) is not int
        or not 0 <= config["pad_token_id"] < config["vocab_size"]
    ):
        raise ValueError("ONNX_CTC_CONFIGURATION_UNSUPPORTED")


def validate_graph(graph, config):
    inputs = {v["name"]: v for v in graph["inputs"]}
    outputs = {v["name"]: v for v in graph["outputs"]}
    if not {"input_values"} <= inputs.keys() <= {"input_values", "attention_mask"}:
        raise ValueError("ONNX_CTC_BINDING_UNSUPPORTED")
    for name, value in inputs.items():
        if (
            value["dtype"] != (1 if name == "input_values" else 7)
            or len(value["shape"]) != 2
            or isinstance(value["shape"][0], int)
            and value["shape"][0] != 1
            or isinstance(value["shape"][1], int)
        ):
            raise ValueError("ONNX_CTC_BINDING_UNSUPPORTED")
    if (
        outputs.keys() != {"logits"}
        or outputs["logits"]["dtype"] != 1
        or len(outputs["logits"]["shape"]) != 3
        or isinstance(outputs["logits"]["shape"][-1], int)
        and outputs["logits"]["shape"][-1] != config["vocab_size"]
    ):
        raise ValueError("ONNX_CTC_BINDING_UNSUPPORTED")


def execute(session, root, config, value, input_path, event):
    import numpy as np

    from oryn_engine.vad_tasks import read_audio

    if value.get("timestamps", False) or value.get("language", "auto") != "auto":
        raise ValueError("ONNX_CTC_PARAMETER_UNSUPPORTED")
    audio, _duration = read_audio(input_path)
    if len(audio) < 400:
        raise ValueError("ONNX_AUDIO_TOO_SHORT")
    feature, tokenizer = load_components(root, config)
    inputs = feature(audio, sampling_rate=16000, return_tensors="np")
    names = {i.name for i in session.get_inputs()}
    event("job.progress", {"stage": "running-model", "progress": 0.2})
    if not np.any(audio):
        return {"text": "", "language": None}
    logits = session.run(
        ["logits"],
        {
            name: np.asarray(inputs[name], dtype=np.float32 if name == "input_values" else np.int64)
            for name in names
        },
    )[0]
    return normalize_logits(logits, tokenizer, config)


def validate_processor_config(metadata):
    if (
        metadata.get("feature_extractor_type", "Wav2Vec2FeatureExtractor")
        != "Wav2Vec2FeatureExtractor"
        or metadata.get("sampling_rate") != 16000
        or metadata.get("feature_size") != 1
        or type(metadata.get("do_normalize")) is not bool
        or metadata.get("padding_value", 0) != 0
    ):
        raise ValueError("ONNX_CTC_PREPROCESSOR_UNSUPPORTED")


def load_components(root, config):
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

    validate_config(config)
    metadata = metadata_file(root / "preprocessor_config.json")
    validate_processor_config(metadata)
    feature = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0,
        do_normalize=metadata["do_normalize"],
        return_attention_mask=True,
    )
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id != config["pad_token_id"] or len(tokenizer) != config["vocab_size"]:
        raise ValueError("ONNX_CTC_VOCABULARY_MISMATCH")
    return feature, tokenizer


def normalize_logits(logits, tokenizer, config):
    import numpy as np

    if (
        logits.ndim != 3
        or logits.shape[0] != 1
        or not 1 <= logits.shape[1] <= 3000
        or logits.shape[2] != config["vocab_size"]
        or not np.isfinite(logits).all()
    ):
        raise ValueError("ONNX_CTC_OUTPUT_INVALID")
    # CTC must collapse repeats before removing blank tokens. Removing specials
    # first merges letters separated by a blank (for example the two l's in hello).
    text = tokenizer.decode(
        logits[0].argmax(-1), group_tokens=True, skip_special_tokens=False
    ).strip()
    if len(text.encode()) > 16384:
        raise ValueError("ONNX_CTC_OUTPUT_LIMIT")
    return {"text": text, "language": None}
