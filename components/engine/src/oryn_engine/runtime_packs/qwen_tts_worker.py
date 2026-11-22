"""Network-denied Qwen TTS worker. Protocol stdout contains JSON only."""

from __future__ import annotations

import contextlib
import json
import sys
import wave
from pathlib import Path

from oryn_engine import qwen_tts


def _message(value):
    print(json.dumps(value, separators=(",", ":")), flush=True)


def _event(stage, progress):
    _message(
        {"type": "event", "event": "job.progress", "data": {"stage": stage, "progress": progress}}
    )


def _generate(payload):
    import numpy as np

    if set(payload) != {"modelRoot", "outputPath", "input", "backend"}:
        raise ValueError("Invalid speech request")
    root, output = Path(payload["modelRoot"]), Path(payload["outputPath"])
    if (
        not root.is_absolute()
        or not root.is_dir()
        or not output.is_absolute()
        or output.name != "speech.wav"
        or not output.parent.is_dir()
        or output.exists()
    ):
        raise ValueError("Invalid speech paths")
    value = qwen_tts.input_value(payload["input"])
    _event("loading-speech", 0.05)
    with contextlib.redirect_stdout(sys.stderr):
        model, codec, tokenizer, config = qwen_tts.load(root, payload["backend"])
    _event("synthesizing", 0.2)
    with contextlib.redirect_stdout(sys.stderr):
        audio, rate = qwen_tts.synthesize(model, codec, tokenizer, config, value)
    _event("writing-audio", 0.95)
    pcm = np.rint(np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(output), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm.tobytes())
    return {
        "relativeOutput": output.name,
        "mediaType": "audio/wav",
        "sampleRate": rate,
        "frames": len(audio),
        "durationSeconds": len(audio) / rate,
        "channels": 1,
        "voice": value["voice"],
        "language": value["language"],
        "backend": payload["backend"],
    }


def main():
    try:
        raw = sys.stdin.buffer.readline(1024 * 1024 + 1)
        if not raw or len(raw) > 1024 * 1024:
            raise ValueError("Request exceeds bound")
        request = json.loads(raw)
        if (
            not isinstance(request, dict)
            or set(request) != {"abi", "packId", "version", "command", "payload"}
            or request["abi"] != "1"
            or request["packId"] != qwen_tts.PACK_ID
            or not isinstance(request["payload"], dict)
        ):
            raise ValueError("Invalid request")
        if request["command"] == "probe":
            result = {"versions": qwen_tts.verify_runtime()}
        elif request["command"] == "prepare":
            qwen_tts.verify_runtime()
            root = Path(request["payload"]["modelRoot"])
            if not root.is_absolute() or not root.is_dir():
                raise ValueError("Invalid model path")
            with contextlib.redirect_stdout(sys.stderr):
                qwen_tts.preflight(root)
            result = {"prepared": True}
        elif request["command"] == "generate":
            qwen_tts.verify_runtime()
            result = _generate(request["payload"])
        else:
            raise ValueError("Invalid command")
        _message({"type": "result", "ok": True, "result": result})
    except Exception:
        _message({"type": "result", "ok": False, "error": "PACK_EXECUTION_FAILED"})


if __name__ == "__main__":
    main()
