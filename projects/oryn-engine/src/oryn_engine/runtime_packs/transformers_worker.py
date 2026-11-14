"""One offline worker for the shared Transformers task contracts; no new environment."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

from oryn_engine.model_metadata import parse_metadata
from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
from oryn_engine.shared_tasks import (
    ARTIFACT_TASKS,
    AUDIO_OUTPUT_TASKS,
    AUDIO_TASKS,
    IMAGE_TASKS,
    MULTIMODAL_TASKS,
    TASK_CONTRACTS,
    VIDEO_TASKS,
    execute_task,
)


def message(value):
    print(json.dumps(value, separators=(",", ":"), allow_nan=False), flush=True)


def event(name, data):
    # Native framework progress/logs cannot corrupt the bounded protocol stream.
    sys.__stdout__.write(
        json.dumps({"type": "event", "event": name, "data": data}, separators=(",", ":")) + "\n"
    )
    sys.__stdout__.flush()


def generate(payload):
    if isinstance(payload, dict) and "implementationId" in payload:
        from oryn_engine.runtime_packs.lfm3_worker import generate as reviewed_generate

        return reviewed_generate(payload, event)
    if not isinstance(payload, dict) or set(payload) != {
        "modelRoot",
        "task",
        "backend",
        "input",
        "inputPath",
        "outputPath",
    }:
        raise ValueError("Invalid task payload")
    task = payload["task"]
    if not isinstance(task, str) or task not in TASK_CONTRACTS:
        raise ValueError("Unknown task")
    root = Path(payload["modelRoot"])
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("Invalid model scope")
    input_path = Path(payload["inputPath"]) if isinstance(payload["inputPath"], str) else None
    output_path = Path(payload["outputPath"]) if isinstance(payload["outputPath"], str) else None
    needs_media = task in IMAGE_TASKS | VIDEO_TASKS | AUDIO_TASKS | MULTIMODAL_TASKS
    if task in {"text-to-speech", "audio-to-audio"}:
        needs_media = isinstance(payload["input"], dict) and "audioArtifactId" in payload["input"]
    if needs_media != (input_path is not None) or (
        input_path is not None and not input_path.is_absolute()
    ):
        raise ValueError("Invalid image scope")
    structured_file = task == "visual-document-retrieval" and output_path is not None
    writes_artifact = task in ARTIFACT_TASKS or structured_file
    if task == "audio-to-audio":
        writes_artifact = (
            isinstance(payload["input"], dict) and payload["input"].get("operation") != "encode"
        )
    if writes_artifact != (output_path is not None):
        raise ValueError("Invalid output scope")
    if output_path is not None and (
        not output_path.is_absolute()
        or output_path.name
        != (
            "result.json"
            if structured_file
            else "result.wav"
            if task in AUDIO_OUTPUT_TASKS
            else "result.png"
        )
        or output_path.is_symlink()
        or not output_path.parent.is_dir()
    ):
        raise ValueError("Invalid output path")
    with contextlib.redirect_stdout(sys.stderr):
        result = execute_task(
            root,
            task,
            payload["input"],
            payload["backend"],
            input_path,
            None if structured_file else output_path,
            event,
        )
    if structured_file:
        from oryn_engine.shared_tasks import validate_task_output
        from oryn_engine.structured_outputs import write_result

        validate_task_output(task, result)
        return write_result(output_path, result, payload["backend"])
    if len(json.dumps(result, allow_nan=False).encode()) > 900 * 1024:
        raise ValueError("Structured result requires a bounded output file")
    return result


def main():
    try:
        request = parse_metadata(sys.stdin.buffer.readline(1024 * 1024 + 1))
        pack = builtin_registry().active_pack("oryn.transformers-pytorch")
        if (
            set(request) != {"abi", "packId", "version", "command", "payload"}
            or request["abi"] != pack.abi
            or request["packId"] != pack.pack_id
            or request["version"] != pack.version
        ):
            raise ValueError("Invalid worker identity")
        if request["command"] == "probe":
            import transformers

            if set(TASK_CONTRACTS.values()) != set(
                pack.runner("transformers.shared-tasks").task_contracts
            ):
                raise ValueError("Signed task declarations differ from the worker")
            result = {
                "transformers": transformers.__version__,
                "taskContracts": list(TASK_CONTRACTS.values()),
                "networkPolicy": "deny",
            }
        elif request["command"] == "generate":
            result = generate(request["payload"])
        else:
            raise ValueError("Invalid worker command")
        message({"type": "result", "ok": True, "result": result})
    except Exception as exc:
        print(f"Shared task worker failed: {type(exc).__name__}", file=sys.stderr)
        message({"type": "result", "ok": False, "error": {"code": "MODEL_GENERATION_FAILED"}})


if __name__ == "__main__":
    main()
