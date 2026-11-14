from __future__ import annotations

import codecs
import json
import math
import os
import re
import selectors
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from oryn_engine.runtime_packs.gguf import (
    GGUFMetadata,
    inspect_gguf_set,
    safe_member_name,
    split_file_set,
)

MAX_REQUEST_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
EXPECTED_BUILD = "11960"
EXPECTED_REVISION = "8642541b255bfb5dc39cd351a178be09013a55d6"
# Upstream names libraries by release version since b11xxx; the build number is above.
LIBRARY_VERSION = "0.8.0"
VISION_LIBRARY = f"libmtmd.{LIBRARY_VERSION}.dylib"


def _message(value: dict[str, Any]) -> None:
    print(json.dumps(value, separators=(",", ":")), flush=True)


def _event(event: str, data: dict[str, Any]) -> None:
    _message({"type": "event", "event": event, "data": data})


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
    if value["abi"] != "1" or value["packId"] != "oryn.llama-cpp" or value["version"] != "0.4.0":
        raise ValueError("request identity is invalid")
    if value["command"] not in {"probe", "generate"} or not isinstance(value["payload"], dict):
        raise ValueError("request command is invalid")
    return value


def _pack() -> tuple[Path, Path, str]:
    root_value = os.environ.get("ORYN_PACK_ROOT")
    if not isinstance(root_value, str):
        raise ValueError("runtime-pack root is unavailable")
    root = Path(root_value)
    executable = root / "native" / "llama-completion"
    marker = root / "site-packages" / "oryn-runtime-pack.json"
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or executable.is_symlink()
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
        or marker.is_symlink()
        or not marker.is_file()
    ):
        raise ValueError("native runtime-pack files are invalid")
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    if (
        not isinstance(marker_value, dict)
        or marker_value.get("packId") != "oryn.llama-cpp"
        or marker_value.get("version") != "0.4.0"
        or marker_value.get("sourceRevision") != EXPECTED_REVISION
        or marker_value.get("nativeBackend") not in {"metal", "cuda", "cpu"}
    ):
        raise ValueError("native runtime-pack marker is invalid")
    return root, executable, marker_value["nativeBackend"]


def _probe(executable: Path, root: Path, backend: str) -> dict[str, Any]:
    result = subprocess.run(
        [str(executable), "--version"],
        cwd=root,
        env=_native_environment(root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=10,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError("llama.cpp probe failed")
    # Upstream prints "version: 0.8.0-dev (build 11960, commit 8642541b2)" since b11xxx.
    match = re.search(r"version:\s+\S+\s+\(build\s+(\d+),\s+commit\s+([0-9a-f]+)\)", result.stdout)
    if (
        match is None
        or match.group(1) != EXPECTED_BUILD
        or not EXPECTED_REVISION.startswith(match.group(2))
    ):
        raise ValueError("llama.cpp identity is invalid")
    return {
        "build": EXPECTED_BUILD,
        "revision": EXPECTED_REVISION,
        "backend": backend,
        "networkPolicy": os.environ.get("ORYN_PACK_NETWORK_POLICY"),
    }


def _model(payload: dict[str, Any]) -> tuple[Path, GGUFMetadata]:
    model_value = payload.get("modelRoot")
    model_file = payload.get("modelFile")
    if not isinstance(model_value, str) or not safe_member_name(model_file):
        raise ValueError("GGUF model root is invalid")
    model_root = Path(model_value)
    if not model_root.is_absolute() or not model_root.is_dir():
        raise ValueError("GGUF model root is invalid")
    repository_root = model_root.resolve()
    model_path = (model_root / model_file).resolve()
    if not model_path.is_relative_to(repository_root) or not model_path.is_file():
        raise ValueError("GGUF model path escapes its repository cache")
    from oryn_engine.native_capability_sources import validate_gguf_policy

    # Split members share the selected file's folder, which may be the root.
    folder = Path(model_file).parent
    directory = (repository_root / folder).resolve()
    if not directory.is_relative_to(repository_root):
        raise ValueError("GGUF model path escapes its repository cache")
    names = split_file_set(
        [(folder / p.name).as_posix() for p in directory.iterdir() if p.is_file()], model_file
    )
    metadata = inspect_gguf_set([repository_root / n for n in names], names)
    validate_gguf_policy(metadata)
    return model_path, metadata


def _input(value: object) -> dict[str, Any]:
    expected = {
        "messages",
        "maxTokens",
        "temperature",
        "topP",
        "chatTemplate",
        "expectedArchitecture",
        "expectedQuantization",
        "contextLimit",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("llama.cpp input is invalid")
    messages = value.get("messages")
    max_tokens = value.get("maxTokens")
    temperature = value.get("temperature")
    top_p = value.get("topP")
    from oryn_engine import chat_formats

    template = value.get("chatTemplate")
    architecture = value.get("expectedArchitecture")
    quantization = value.get("expectedQuantization")
    context_limit = value.get("contextLimit")
    if (
        not isinstance(messages, list)
        or not 1 <= len(messages) <= 128
        or not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or not 1 <= max_tokens <= 512
        or not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
        or not 0 <= float(temperature) <= 2
        or not isinstance(top_p, (int, float))
        or isinstance(top_p, bool)
        or not math.isfinite(float(top_p))
        or not 0 < float(top_p) <= 1
        or template not in chat_formats.names()
        or not isinstance(architecture, str)
        or not re.fullmatch(r"[a-z0-9_.-]{1,128}", architecture)
        or not isinstance(quantization, str)
        or not re.fullmatch(r"[A-Z0-9_]{2,32}", quantization)
        or not isinstance(context_limit, int)
        or isinstance(context_limit, bool)
        or not 128 <= context_limit <= 1_048_576
    ):
        raise ValueError("llama.cpp input is invalid")
    total = 0
    normalized = []
    for message in messages:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            or not 1 <= len(message["content"]) <= 32_768
        ):
            raise ValueError("llama.cpp messages are invalid")
        total += len(message["content"])
        normalized.append({"role": message["role"], "content": message["content"]})
    if total > 65_536 or not any(message["role"] == "user" for message in normalized):
        raise ValueError("llama.cpp messages are invalid")
    return {
        "messages": normalized,
        "maxTokens": max_tokens,
        "temperature": float(temperature),
        "topP": float(top_p),
        "chatTemplate": template,
        "expectedArchitecture": architecture,
        "expectedQuantization": quantization,
        "contextLimit": context_limit,
    }


class CompletionText:
    """Hold the native CLI's fixed terminal footer out of streamed model text.

    Pinned tools/completion/completion.cpp emits this after EOG and then
    two final newlines. Retain only a bounded tail; literal nonterminal text stays.
    """

    footer = " [end of text]\n\n\n"

    def __init__(self) -> None:
        self.tail = ""

    def feed(self, text: str) -> str:
        self.tail += text
        count = max(0, len(self.tail) - len(self.footer))
        ready, self.tail = self.tail[:count], self.tail[count:]
        return ready

    def finish(self) -> str:
        ready = self.tail.removesuffix(self.footer)
        self.tail = ""
        return ready


def _render_prompt(messages: list[dict[str, str]], template: str) -> str:
    # Reviewed markers from the signed chat-format table; repository Jinja stays inert.
    from oryn_engine.chat_formats import render

    return render(template, messages)


def _native_environment(root: Path) -> dict[str, str]:
    native = root / "native"
    return {
        "HOME": tempfile.gettempdir(),
        "PATH": "/usr/bin:/bin",
        "DYLD_LIBRARY_PATH": str(native),
        "GGML_METAL_PATH_RESOURCES": str(native),
    }


def _generate(
    executable: Path,
    root: Path,
    backend: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    from oryn_engine import gguf_translation

    translation = payload.get("taskContract") == gguf_translation.CONTRACT
    if "taskContract" in payload and not translation:
        raise ValueError("GGUF task contract is invalid")
    translation_value = None
    if translation:
        if set(payload) != {"taskContract", "modelRoot", "modelFile", "input"}:
            raise ValueError("GGUF translation payload is invalid")
        raw = payload.get("input")
        trusted_fields = {"expectedArchitecture", "expectedQuantization", "contextLimit"}
        if not isinstance(raw, dict) or not trusted_fields <= raw.keys():
            raise ValueError("GGUF translation input is invalid")
        translation_value = gguf_translation.validate_input(
            {k: v for k, v in raw.items() if k not in trusted_fields}
        )
        value = _input(
            {
                **{k: raw[k] for k in trusted_fields},
                "messages": [{"role": "user", "content": translation_value["text"]}],
                "maxTokens": translation_value["maxTokens"],
                "temperature": 0,
                "topP": 1,
                "chatTemplate": "chatml",
            }
        )
    else:
        value = _input(payload.get("input"))
    model_path, metadata = _model(payload)
    from oryn_engine.native_capability_sources import validate_gguf_policy

    validate_gguf_policy(metadata, value["chatTemplate"])
    if (
        metadata.architecture != value["expectedArchitecture"]
        or metadata.quantization != value["expectedQuantization"]
        or metadata.context_length < value["contextLimit"]
    ):
        raise ValueError("GGUF metadata does not match its trusted compatibility manifest")
    runtime_context = min(
        value["contextLimit"], gguf_translation.MAX_CONTEXT if translation else 8192
    )
    if translation:
        gguf_translation.validate_binding(metadata.architecture, value["chatTemplate"])
        if metadata.tokenizer_model != "gpt2":
            raise ValueError("GGUF translation requires its native byte tokenizer")
        prompt = gguf_translation.render(translation_value, runtime_context)
    else:
        prompt = _render_prompt(value["messages"], value["chatTemplate"])
    output_limit = (
        gguf_translation.MAX_OUTPUT_BYTES + len(CompletionText.footer)
        if translation
        else MAX_OUTPUT_BYTES
    )
    vision = payload.get("projectorFile") is not None
    image = None
    projector = None
    if vision:
        import hashlib

        from oryn_engine.generated_capabilities import generated_artifact
        from oryn_engine.runtime_packs.gguf import inspect_projector
        from oryn_engine.shared_tasks import open_task_image

        name = payload["projectorFile"]
        if not safe_member_name(name):
            raise ValueError("Projector file is invalid")
        # Repository-relative like modelFile: next to the weights or at the root.
        repository_root = Path(payload["modelRoot"]).resolve()
        projector = repository_root / name
        if (
            projector.is_symlink()
            or not projector.resolve().is_relative_to(repository_root)
            or not projector.is_file()
        ):
            raise ValueError("Projector graph member is missing")
        inspect_projector(projector, metadata)
        if len(value["messages"]) != 1 or value["messages"][0]["role"] != "user":
            raise ValueError("This image adapter accepts a single user message")
        from oryn_engine.chat_formats import llama_cpp_vision_arguments

        # Raises for formats without a reviewed vision renderer.
        vision_template = llama_cpp_vision_arguments(value["chatTemplate"])
        prompt = value["messages"][0]["content"]
        image, _ = open_task_image(Path(payload["inputPath"]))
        executable = root / "native/llama-mtmd-cli"
        inputs = generated_artifact()["buildInputs"]
        for path, key in (
            (executable, "llamaVisionBinarySha256"),
            (root / "native" / VISION_LIBRARY, "llamaVisionLibrarySha256"),
        ):
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != inputs[key]:
                raise ValueError("Native vision build differs from its signed identity")
    if len(prompt.encode("utf-8")) > max(1, runtime_context - value["maxTokens"]) * 16:
        raise ValueError("GGUF prompt exceeds the bounded runtime context")
    _event("job.progress", {"stage": "loading-native-model", "progress": 0.02})
    with tempfile.TemporaryDirectory(prefix="oryn-llama-cpp-") as directory:
        prompt_path = Path(directory) / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        arguments = [
            str(executable),
            "--model",
            str(model_path),
            "--file",
            str(prompt_path),
            "--predict",
            str(value["maxTokens"]),
            "--ctx-size",
            str(runtime_context),
            "--batch-size",
            "512",
            "--ubatch-size",
            "128",
            "--threads",
            str(max(1, min(8, os.cpu_count() or 1))),
            "--temperature",
            str(value["temperature"]),
            "--top-p",
            str(value["topP"]),
            "--seed",
            "0",
            "--gpu-layers",
            "all" if backend in {"metal", "cuda"} else "0",
            "--no-perf",
            "--no-escape",
        ]
        if translation:
            arguments.append("--no-context-shift")
        if backend == "cpu":
            # Zero GPU layers alone still permits native KV/host-op offloading.
            # Keep the admitted CPU route on CPU, including hybrid MoE graphs.
            arguments += ["--device", "none", "--no-kv-offload", "--no-op-offload"]
        if vision:
            if backend == "cpu":
                arguments.append("--no-mmproj-offload")
            image_path = Path(directory) / "input.png"
            image.save(image_path, format="PNG")
            arguments += [
                "--mmproj",
                str(projector),
                "--image",
                str(image_path),
                "--image-min-tokens",
                "4",
                "--image-max-tokens",
                "1024",
                *vision_template,
                "--log-colors",
                "off",
            ]
        else:
            arguments += [
                "--no-conversation",
                "--no-display-prompt",
                "--no-warmup",
                "--simple-io",
                "--color",
                "off",
            ]
        process = subprocess.Popen(
            arguments,
            cwd=directory,
            env=_native_environment(root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        completion = None if vision else CompletionText()
        chunks: list[str] = []
        output_bytes = 0
        _event("job.state", {"state": "streaming"})
        _event("job.progress", {"stage": "generating", "progress": 0.1})
        try:
            while process.poll() is None:
                for key, _mask in selector.select(timeout=0.1):
                    chunk = os.read(key.fileobj.fileno(), 4096)
                    if not chunk:
                        continue
                    output_bytes += len(chunk)
                    if output_bytes > output_limit:
                        process.terminate()
                        raise ValueError("llama.cpp output exceeds the size limit")
                    text = decoder.decode(chunk)
                    if completion is not None:
                        text = completion.feed(text)
                    if text:
                        chunks.append(text)
                        if not translation:
                            _event("job.output.delta", {"channel": "text", "delta": text})
            trailing = process.stdout.read()
            if trailing:
                output_bytes += len(trailing)
                if output_bytes > output_limit:
                    raise ValueError("llama.cpp output exceeds the size limit")
                text = decoder.decode(trailing)
                if completion is not None:
                    text = completion.feed(text)
                if text:
                    chunks.append(text)
                    if not translation:
                        _event("job.output.delta", {"channel": "text", "delta": text})
            final_text = decoder.decode(b"", final=True)
            if completion is not None:
                final_text = completion.feed(final_text) + completion.finish()
            if final_text:
                chunks.append(final_text)
                if not translation:
                    _event("job.output.delta", {"channel": "text", "delta": final_text})
            if process.returncode != 0:
                raise ValueError("llama.cpp generation failed")
        finally:
            selector.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    text = "".join(chunks).strip()
    if not text:
        raise ValueError("llama.cpp generation produced no text")
    if translation:
        gguf_translation.validate_output(text)
    _event("job.progress", {"stage": "finalizing-output", "progress": 1.0})
    return {
        "text": text,
        "backend": backend,
        "quantization": metadata.quantization,
        # This is the admitted context ceiling, which can be lower than the
        # model's physical capacity. The header was checked against it above.
        "contextLength": value["contextLimit"],
        "runtimeContextLimit": runtime_context,
        "architecture": metadata.architecture,
    }


def main() -> None:
    try:
        request = _request()
        root, executable, backend = _pack()
        if request["command"] == "probe":
            result = _probe(executable, root, backend)
            result["vision"] = _probe(root / "native/llama-mtmd-cli", root, backend)
            from importlib.metadata import version
            from io import BytesIO

            from PIL import Image

            encoded_image = BytesIO()
            Image.new("RGB", (2, 2), "red").save(encoded_image, format="PNG")
            encoded_image.seek(0)
            with Image.open(encoded_image) as decoded_image:
                decoded_image.load()
                if decoded_image.getpixel((0, 0)) != (255, 0, 0):
                    raise ValueError("Native image codec probe failed")
            result["vision"]["pillow"] = version("pillow")
        else:
            result = _generate(executable, root, backend, request["payload"])
        _message({"type": "result", "ok": True, "result": result})
    except Exception:
        _message({"type": "result", "ok": False, "error": "PACK_EXECUTION_FAILED"})


if __name__ == "__main__":
    main()
