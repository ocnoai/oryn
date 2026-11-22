"""Offline ONNX worker. Checkpoint bytes never select imports or custom operators."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

from oryn_engine.model_metadata import parse_metadata
from oryn_engine.onnx_tasks import execute, prepare
from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
from oryn_engine.runtime_packs.transformers_worker import event, message


def main():
    try:
        request = parse_metadata(sys.stdin.buffer.readline(1024 * 1024 + 1))
        pack = builtin_registry().active_pack("oryn.onnx-runtime")
        if set(request) != {"abi", "packId", "version", "command", "payload"} or (
            request["abi"],
            request["packId"],
            request["version"],
        ) != (pack.abi, pack.pack_id, pack.version):
            raise ValueError("ONNX_WORKER_IDENTITY")
        if request["command"] == "probe":
            from importlib.metadata import version

            from oryn_engine.onnx_tasks import TASK_CONTRACTS, verify_runtime

            verify_runtime()
            if set(TASK_CONTRACTS.values()) != set(pack.runner("onnx.encoder").task_contracts):
                raise ValueError("ONNX_WORKER_CONTRACTS")
            message(
                {
                    "type": "result",
                    "ok": True,
                    "result": {
                        "onnxruntime": version("onnxruntime"),
                        "taskContracts": sorted(TASK_CONTRACTS.values()),
                        "backends": ["cpu"],
                    },
                }
            )
            return
        value = request["payload"]
        if not isinstance(value, dict) or set(value) != {
            "modelRoot",
            "task",
            "backend",
            "input",
            "inputPath",
            "outputPath",
        }:
            raise ValueError("ONNX_WORKER_INPUT")
        root = Path(value["modelRoot"])
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise ValueError("ONNX_MODEL_SCOPE")
        input_path = Path(value["inputPath"]) if isinstance(value["inputPath"], str) else None
        output_path = Path(value["outputPath"]) if isinstance(value["outputPath"], str) else None
        audio_output = request["command"] == "generate" and value["task"] == "audio-to-audio"
        if (
            audio_output != (output_path is not None)
            or (output_path is None and value["outputPath"] is not None)
            or (output_path is not None and not output_path.is_absolute())
        ):
            raise ValueError("ONNX_ARTIFACT_SCOPE")
        if request["command"] == "generate" and (
            (
                value["task"]
                in {
                    "voice-activity-detection",
                    "automatic-speech-recognition",
                    "audio-to-audio",
                    "image-classification",
                }
            )
            != (input_path is not None)
            or (input_path is not None and not input_path.is_absolute())
        ):
            raise ValueError("ONNX_ARTIFACT_SCOPE")
        with contextlib.redirect_stdout(sys.stderr):
            if request["command"] == "prepare":
                prepare(root, value["task"], value["backend"])
                result = {"prepared": True}
            elif request["command"] == "generate":
                result = execute(
                    root,
                    value["task"],
                    value["input"],
                    value["backend"],
                    event,
                    input_path,
                    output_path,
                )
            else:
                raise ValueError("ONNX_WORKER_COMMAND")
        message({"type": "result", "ok": True, "result": result})
    except Exception as exc:
        code = str(exc)
        if not code.startswith("ONNX_") or len(code) > 128 or not code.replace("_", "").isalnum():
            code = "MODEL_GENERATION_FAILED"
        print(f"ONNX worker failed: {type(exc).__name__}: {code}", file=sys.stderr)
        message({"type": "result", "ok": False, "error": {"code": code}})


if __name__ == "__main__":
    main()
