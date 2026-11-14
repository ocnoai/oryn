from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

MAX_REQUEST_BYTES = 1024 * 1024
SAMPLE_RATE = 24_000
VOICE_LANGUAGES = {
    "af_heart": "en-us",
    "af_bella": "en-us",
    "af_nicole": "en-us",
    "af_sarah": "en-us",
    "am_adam": "en-us",
    "am_michael": "en-us",
    "bf_emma": "en-gb",
    "bf_isabella": "en-gb",
    "bm_george": "en-gb",
    "bm_lewis": "en-gb",
}


def _message(value: dict[str, Any]) -> None:
    print(json.dumps(value, separators=(",", ":")), flush=True)


def _event(stage: str, progress: float) -> None:
    _message(
        {
            "type": "event",
            "event": "job.progress",
            "data": {"stage": stage, "progress": progress},
        }
    )


def _request() -> dict[str, Any]:
    line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
    if not line or len(line) > MAX_REQUEST_BYTES:
        raise ValueError("request size is invalid")
    value = json.loads(line)
    if not isinstance(value, dict) or set(value) != {
        "abi",
        "packId",
        "version",
        "command",
        "payload",
    }:
        raise ValueError("request envelope is invalid")
    if value["abi"] != "1" or value["packId"] != "oryn.kokoro-onnx":
        raise ValueError("request identity is invalid")
    if value["command"] not in {"probe", "generate"} or not isinstance(value["payload"], dict):
        raise ValueError("request command is invalid")
    return value


def _validated_paths(payload: dict[str, Any]) -> tuple[Path, Path]:
    model_root = Path(payload.get("modelRoot", ""))
    output_path = Path(payload.get("outputPath", ""))
    if not model_root.is_absolute() or not model_root.is_dir() or not output_path.is_absolute():
        raise ValueError("runtime-pack paths are invalid")
    if output_path.name != "speech.wav" or not output_path.parent.is_dir():
        raise ValueError("runtime-pack output path is invalid")
    return model_root.resolve(), output_path.resolve()


def _input(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("input")
    if not isinstance(value, dict) or set(value) != {"text", "voice", "language", "speed"}:
        raise ValueError("speech input is invalid")
    text = value.get("text")
    voice = value.get("voice")
    language = value.get("language")
    speed = value.get("speed")
    if (
        not isinstance(text, str)
        or not 1 <= len(text) <= 350
        or voice not in VOICE_LANGUAGES
        or language != VOICE_LANGUAGES.get(voice)
        or not isinstance(speed, (int, float))
        or isinstance(speed, bool)
        or not math.isfinite(float(speed))
        or not 0.5 <= float(speed) <= 2.0
    ):
        raise ValueError("speech input is invalid")
    return {"text": text, "voice": voice, "language": language, "speed": float(speed)}


def _generate(payload: dict[str, Any]) -> dict[str, Any]:
    import espeakng_loader
    import numpy as np
    import onnxruntime as ort
    from phonemizer import phonemize
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    model_root, output_path = _validated_paths(payload)
    input_value = _input(payload)
    model_path = model_root / "onnx" / "model_quantized.onnx"
    tokenizer_path = model_root / "tokenizer.json"
    voice_path = model_root / "voices" / f"{input_value['voice']}.bin"
    if not model_path.is_file() or not tokenizer_path.is_file() or not voice_path.is_file():
        raise ValueError("Kokoro model subset is incomplete")

    _event("phonemizing", 0.05)
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    # eSpeak 1.52 stores its data path in a 160-byte native buffer. Packaged
    # application paths can legitimately exceed that limit, so expose the
    # immutable bundled data through one short, process-private temporary
    # symlink. No dependency or model file is copied or modified.
    with tempfile.TemporaryDirectory(prefix="oryn-espeak-") as data_parent:
        os.symlink(
            espeakng_loader.get_data_path(),
            Path(data_parent) / "espeak-ng-data",
            target_is_directory=True,
        )
        # The sandbox's own scratch root may already be long. A relative path
        # inside this process-private directory also bounds eSpeak's native
        # path buffer without widening any filesystem permission.
        previous_directory = Path.cwd()
        original_data_path = EspeakWrapper.data_path
        try:
            os.chdir(data_parent)
            # phonemizer-fork otherwise resolves '.' back to the oversized
            # absolute path before calling the native initializer. Keep the
            # reviewed relative binding for this isolated invocation only.
            EspeakWrapper.data_path = property(lambda _self: Path("."))
            phonemes = phonemize(
                input_value["text"],
                language=input_value["language"],
                backend="espeak",
                strip=True,
                preserve_punctuation=True,
                with_stress=True,
            )
        finally:
            EspeakWrapper.data_path = original_data_path
            os.chdir(previous_directory)
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocab = tokenizer.get("model", {}).get("vocab", {})
    if not isinstance(phonemes, str) or not isinstance(vocab, dict):
        raise ValueError("Kokoro tokenizer input is invalid")
    token_ids = [vocab[character] for character in phonemes if character in vocab]
    if not 1 <= len(token_ids) <= 510 or any(not isinstance(value, int) for value in token_ids):
        raise ValueError("speech text exceeds the Kokoro context limit")

    voices = np.fromfile(voice_path, dtype=np.float32).reshape(-1, 1, 256)
    if len(token_ids) >= voices.shape[0]:
        raise ValueError("voice style does not cover the text length")
    model_inputs = {
        "input_ids": np.asarray([[0, *token_ids, 0]], dtype=np.int64),
        "style": voices[len(token_ids)],
        "speed": np.asarray([input_value["speed"]], dtype=np.float32),
    }

    # CoreML's compiler writes outside the permitted scratch root even when
    # ModelCacheDirectory is provided. This signed adapter admits CPU only.
    # Do not silently dispatch a CPU-admitted job to another execution provider.
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("approved ONNX CPU provider is unavailable")
    providers = ["CPUExecutionProvider"]
    _event("synthesizing", 0.2)
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    audio = np.asarray(session.run(None, model_inputs)[0], dtype=np.float32).reshape(-1)
    if not 1 <= audio.size <= SAMPLE_RATE * 300 or not np.isfinite(audio).all():
        raise ValueError("Kokoro audio output is invalid")

    _event("writing-audio", 0.95)
    pcm = np.rint(np.clip(audio, -1, 1) * 32_767).astype("<i2", copy=False)
    with wave.open(str(output_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())
    return {
        "relativeOutput": output_path.name,
        "mediaType": "audio/wav",
        "sampleRate": SAMPLE_RATE,
        "durationSeconds": audio.size / SAMPLE_RATE,
        "backend": "cpu",
        "voice": input_value["voice"],
        "language": input_value["language"],
    }


def main() -> None:
    try:
        request = _request()
        if request["command"] == "probe":
            import onnxruntime as ort

            result = {
                "providers": [
                    p for p in ort.get_available_providers() if p == "CPUExecutionProvider"
                ]
            }
        else:
            result = _generate(request["payload"])
        _message({"type": "result", "ok": True, "result": result})
    except Exception:
        _message({"type": "result", "ok": False, "error": "PACK_EXECUTION_FAILED"})


if __name__ == "__main__":
    main()
