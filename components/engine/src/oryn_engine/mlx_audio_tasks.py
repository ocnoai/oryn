"""Dependency-free admission for the signed, bundled MLX-Audio model families."""

from __future__ import annotations

from pathlib import PurePosixPath

from oryn_engine.model_metadata import validate_model_config

PACK_ID = "oryn.mlx-audio"
RUNNER_ID = "mlx-audio.shared-tasks"
QWEN3_TTS_CODEC_CONFIG = "speech_tokenizer/config.json"
TASK_CONTRACTS = {
    "text-to-speech": "speech-synthesis-v1",
    "automatic-speech-recognition": "transcription-v1",
}
SAFE_SUFFIXES = {
    ".json",
    ".safetensors",
    ".txt",
    ".model",
    ".spm",
    ".tiktoken",
    ".jinja",
    ".npy",
    ".npz",
}


def capability(config: dict, task: str | None) -> dict | None:
    from oryn_engine.generated_capabilities import generated_artifact

    if not isinstance(config, dict):
        return None
    model_type = config.get("model_type") or config.get("architecture")
    if not isinstance(model_type, str):
        return None
    table = generated_artifact()["buildInputs"].get("mlxAudio", {}).get("models", {})
    if task is None:
        matches = [entries[model_type] for entries in table.values() if model_type in entries]
        return matches[0] if len(matches) == 1 else None
    return table.get(task, {}).get(model_type)


def effective_task(config: dict, task: str | None) -> str | None:
    if task is not None:
        return task
    entry = capability(config, None)
    return {"tts": "text-to-speech", "stt": "automatic-speech-recognition"}.get(
        entry["category"] if entry else None
    )


def model_family(config: dict, task: str | None) -> str | None:
    entry = capability(config, task)
    return entry["family"] if entry else None


def validate_config(config: dict, task: str | None) -> dict:
    validate_model_config(config)
    if capability(config, task) is None:
        raise ValueError("MLX-Audio model architecture or task is unavailable.")
    if config.get("auto_map") or config.get("custom_pipelines") or config.get("trust_remote_code"):
        raise ValueError("MLX-Audio repository code is unavailable.")
    return config


def validate_qwen3_tts_codec(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("MLX-Audio speech tokenizer configuration is invalid.")
    validate_model_config(config)
    decoder = config.get("decoder_config")
    if decoder is None:
        decoder = {}
    declared_rate = config.get("decode_upsample_rate", 1920)
    if not isinstance(decoder, dict) or type(declared_rate) is not int or declared_rate < 1:
        raise ValueError("MLX-Audio speech tokenizer decoder geometry is invalid.")
    # The native tokenizer computes decoded lengths from the top-level rate,
    # while the decoder builds the waveform from these two lists of factors.
    decoder_rate = 1
    for factors in (
        decoder.get("upsample_rates", [8, 5, 4, 3]),
        decoder.get("upsampling_ratios", [2, 2]),
    ):
        if not isinstance(factors, list) or not factors:
            raise ValueError("MLX-Audio speech tokenizer decoder geometry is invalid.")
        for factor in factors:
            if type(factor) is not int or not 1 <= factor <= declared_rate // decoder_rate:
                raise ValueError("MLX-Audio speech tokenizer decoder geometry is invalid.")
            decoder_rate *= factor
    if decoder_rate != declared_rate:
        raise ValueError("MLX-Audio speech tokenizer decoder geometry is invalid.")


def selected_files(files, config: dict, task: str | None) -> tuple[str, ...]:
    validate_config(config, task)
    names = tuple(sorted(files))
    if (
        len(names) > 10_000
        or len(set(names)) != len(names)
        or any(
            not isinstance(name, str)
            or len(name) > 1024
            or "\\" in name
            or PurePosixPath(name).is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))
            for name in names
        )
    ):
        raise ValueError("MLX-Audio file layout is invalid.")
    selected = tuple(name for name in names if PurePosixPath(name).suffix in SAFE_SUFFIXES)
    if "config.json" not in selected or not any(name.endswith(".safetensors") for name in selected):
        raise ValueError("MLX-Audio safe model data is incomplete.")
    binding = capability(config, task)
    if not set(binding["requiredFiles"]).issubset(selected):
        raise ValueError("MLX-Audio companion model data is missing.")
    if binding["family"] == "qwen3_tts":
        codec_directory = PurePosixPath(QWEN3_TTS_CODEC_CONFIG).parent
        codec_weights = any(
            PurePosixPath(name).parent == codec_directory and name.endswith(".safetensors")
            for name in selected
        )
        model_weights = any(
            "/" not in name and name.endswith((".safetensors", ".npz")) for name in selected
        )
        if QWEN3_TTS_CODEC_CONFIG not in selected or not codec_weights or not model_weights:
            raise ValueError("MLX-Audio Qwen3-TTS model or speech tokenizer data is missing.")
    return selected
