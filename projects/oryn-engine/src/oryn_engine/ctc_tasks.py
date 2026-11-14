"""Native safe Wav2Vec2 CTC; shares normalization with the ONNX binding."""

from __future__ import annotations

TASK_CONTRACTS = {"automatic-speech-recognition": "transcription-v1"}
TASK_FAMILIES = {"automatic-speech-recognition": {"wav2vec2"}}


def load_components(root, config, common):
    import transformers as tf

    from oryn_engine.onnx_asr import load_components as load_ctc_components

    return load_ctc_components(root, config), tf.Wav2Vec2ForCTC


def execute(model, processor, config, value, backend, input_path, emit):
    import torch

    from oryn_engine.onnx_asr import normalize_logits
    from oryn_engine.vad_tasks import read_audio

    if value.get("timestamps", False) or value.get("language", "auto") != "auto":
        raise ValueError("Native CTC has no language detection or word timing")
    audio, _ = read_audio(input_path)
    if len(audio) < 400:
        raise ValueError("CTC input is too short")
    if not bool(audio.any()):
        return {"text": "", "language": None}
    feature, tokenizer = processor
    inputs = feature(audio, sampling_rate=16000, return_tensors="pt")
    emit("job.progress", {"stage": "transcribing", "progress": 0.3})
    with torch.inference_mode():
        output = model(**{k: v.to(backend) for k, v in inputs.items()})
    return normalize_logits(output.logits.float().cpu().numpy(), tokenizer, config)
