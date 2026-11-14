"""Offline MLX-LM chat through an Oryn-owned worker; no repository loader authority."""

from __future__ import annotations

import contextlib
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

MAX_JSON_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 4 * 1024 * 1024


def message(value: dict[str, Any]) -> None:
    print(json.dumps(value, separators=(",", ":")), flush=True)


def _event(event: str, data: dict[str, Any]) -> None:
    message({"type": "event", "event": event, "data": data})


def validate_config(config: object, supported: set[str]) -> dict[str, Any]:
    from oryn_engine.model_metadata import validate_model_config

    if not isinstance(config, dict) or config.get("model_type") not in supported:
        raise ValueError("MLX model architecture is unavailable.")
    validate_model_config(config)
    # Legacy quant_method paths select different native quantizers and are not admitted.
    if config.get("quantization_config") is not None and config[
        "quantization_config"
    ] != config.get("quantization"):
        raise ValueError("Legacy MLX quantizer identity is not admitted.")
    quantization = config.get("quantization")
    if quantization is None:
        return config
    if not isinstance(quantization, dict) or len(quantization) > 10_000:
        raise ValueError("MLX quantization layout is not admitted.")

    def settings(value):
        if (
            not isinstance(value, dict)
            or not set(value) <= {"bits", "group_size", "mode"}
            or type(value.get("bits")) is not int
            or value["bits"] not in {4, 5, 6, 8}
            or type(value.get("group_size")) is not int
            or value["group_size"] not in {32, 64, 128}
            or not (
                value.get("mode", "affine") == "affine"
                or config["model_type"] == "gpt_oss"
                and value.get("mode") == "mxfp4"
                and value["bits"] == 4
                and value["group_size"] == 32
            )
        ):
            raise ValueError("MLX quantization layout is not admitted.")

    settings(
        {key: quantization[key] for key in ("bits", "group_size", "mode") if key in quantization}
    )
    for name, value in quantization.items():
        if name in {"bits", "group_size", "mode"}:
            continue
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+){0,15}", name) or any(
            part.startswith("_") for part in name.split(".")
        ):
            raise ValueError("MLX layer name is invalid.")
        if value is not False:
            settings(value)
    return config


def validate_quantized_layers(model, config, tensors, *, language_only=True):
    """Names only look up an enumerated native module; never resolve attributes/imports."""
    quantization = config.get("quantization")
    if not quantization:
        return
    modules = dict(model.named_modules())
    # Match the reviewed Qwen 3.5 language-only wrapper's name sanitization without
    # loading tensors or executing repository transformations. Vision weights are inert.
    if language_only and config.get("model_type") in {"qwen3_5", "qwen3_5_moe"}:
        normalized = {}
        for name, shape in tensors.items():
            if name.startswith(("vision_tower.", "model.visual.")):
                continue
            if name.startswith("model.language_model."):
                name = "language_model.model." + name.removeprefix("model.language_model.")
            elif not name.startswith("language_model."):
                name = "language_model." + name
            if ".mtp." in name:
                continue
            if name in normalized:
                raise ValueError("MLX tensor names are ambiguous after native mapping.")
            normalized[name] = shape
        tensors = normalized
    if any(
        name.removesuffix(".scales") not in modules for name in tensors if name.endswith(".scales")
    ):
        raise ValueError("MLX scales reference an unknown native layer.")
    for name in set(quantization) - {"bits", "group_size", "mode"}:
        if name not in modules or not callable(getattr(modules[name], "to_quantized", None)):
            raise ValueError("MLX quantization references an unknown native layer.")
    for name, module in modules.items():
        if name in quantization and quantization[name] is False:
            if name + ".scales" in tensors:
                raise ValueError("Unquantized layer contains quantized tensors.")
            continue
        scale = tensors.get(name + ".scales")
        if scale is None:
            if name in quantization:
                raise ValueError("Quantized layer has no scales.")
            continue
        settings = quantization.get(name, quantization)
        weight = tensors.get(name + ".weight")
        biases = tensors.get(name + ".biases")
        native_weight = getattr(module, "weight", None)
        if (
            not callable(getattr(module, "to_quantized", None))
            or native_weight is None
            or len(native_weight.shape) not in {2, 3}
        ):
            raise ValueError("Quantization target is not an admitted native matrix.")
        rows, columns = tuple(native_weight.shape[:-1]), native_weight.shape[-1]
        if (
            columns % settings["group_size"]
            or columns * settings["bits"] % 32
            or scale != (*rows, columns // settings["group_size"])
            or (
                biases != scale
                if settings.get("mode", "affine") == "affine"
                else biases is not None
            )
            or weight != (*rows, columns * settings["bits"] // 32)
        ):
            raise ValueError("MLX quantized tensor layout does not match its native layer.")


def sealed_prompt(tokenizer, messages):
    from oryn_engine import chat_formats

    # The repository template is read as text to choose reviewed markers, never rendered.
    text = chat_formats.template_text(tokenizer)
    name = (
        chat_formats.select(text)
        if text
        else chat_formats.select_by_vocabulary(tokenizer.get_vocab())
    )
    if name is None:
        raise ValueError("MLX tokenizer has no admitted sealed chat renderer.")
    return chat_formats.render(name, messages)


def generate(payload: dict[str, Any]) -> dict[str, Any]:
    from oryn_engine.generated_capabilities import mlx_model_types
    from oryn_engine.runners import _chat_input

    if payload.get("task") == "any-to-any":
        from oryn_engine.mlx_gemma4 import execute

        return execute(payload, _event)
    if payload.get("task") == "voice-activity-detection":
        from oryn_engine.mlx_vad import execute

        return execute(payload, _event)
    if set(payload) not in ({"modelRoot", "input"}, {"modelRoot", "input", "inputPath"}):
        raise ValueError("MLX request fields are invalid.")
    root = Path(payload["modelRoot"])
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError("MLX model root is invalid.")
    if "inputPath" in payload:
        from oryn_engine.model_metadata import metadata_file

        if metadata_file(root / "config.json").get("model_type") == "gemma4":
            from oryn_engine.mlx_gemma4 import generate as generate_gemma4

            return generate_gemma4(root, payload["input"], Path(payload["inputPath"]), _event)
        from oryn_engine.mlx_vision import generate as generate_vision

        return generate_vision(root, payload["input"], Path(payload["inputPath"]), _event)
    config_path = root / "config.json"
    if not config_path.is_file() or config_path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("MLX config is missing or oversized.")
    supported = mlx_model_types()
    from oryn_engine import mlx_bolding, mlx_kolibri, mlx_prism
    from oryn_engine.model_metadata import metadata_file, preflight_safetensors

    config = metadata_file(config_path)
    value = _chat_input(payload["input"])
    if mlx_kolibri.supports(config):
        return mlx_kolibri.generate(root, config, value, _event)
    if mlx_bolding.supports(config):
        return mlx_bolding.generate(root, config, value, _event)
    if mlx_prism.supports(config):
        return mlx_prism.generate(root, config, value, _event)
    config = validate_config(config, supported)
    tensors = preflight_safetensors(root)
    value = _chat_input(payload["input"])
    if config["model_type"] == "gpt_oss":
        from oryn_engine.mlx_gpt_oss import generate as generate_gpt_oss

        return generate_gpt_oss(root, config, value, _event)
    if config["model_type"] == "gemma4_unified":
        from oryn_engine.mlx_gemma import generate as generate_gemma

        return generate_gemma(root, config, value, _event)
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load_model, load_tokenizer

    def classes(config: dict[str, Any]) -> tuple[Any, Any]:
        checked = validate_config(config, supported)
        # Only a build-generated library module can be imported. The repository cannot add one.
        module = importlib.import_module(f"mlx_lm.models.{checked['model_type']}")

        def checked_model(args):
            model = module.Model(args)
            validate_quantized_layers(model, checked, tensors)
            return model

        return checked_model, module.ModelArgs

    _event("job.progress", {"stage": "loading-model", "progress": 0.05})
    with contextlib.redirect_stdout(sys.stderr):
        model, _ = load_model(root, get_model_classes=classes)
        tokenizer = load_tokenizer(root, {"trust_remote_code": False, "local_files_only": True})
    prompt = sealed_prompt(tokenizer, value["messages"])
    context_limit = min(int(config.get("max_position_embeddings", 8192)), 8192)
    if len(tokenizer.encode(prompt)) + value["maxTokens"] > context_limit:
        raise ValueError("MLX request exceeds the admitted context limit.")
    chunks: list[str] = []
    output_bytes = 0
    _event("job.state", {"state": "running"})
    try:
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=value["maxTokens"],
            sampler=make_sampler(temp=value["temperature"], top_p=value["topP"]),
        ):
            text = response.text
            output_bytes += len(text.encode("utf-8"))
            if output_bytes > MAX_OUTPUT_BYTES:
                raise ValueError("MLX output exceeds its bound.")
            if text:
                chunks.append(text)
                _event("job.output.delta", {"channel": "text", "delta": text})
        return {"text": "".join(chunks), "backend": "mlx"}
    finally:
        del model
        mx.clear_cache()


def main() -> None:
    try:
        raw = sys.stdin.buffer.readline(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise ValueError("Request is oversized.")
        request = json.loads(raw)
        if (
            not isinstance(request, dict)
            or set(request) != {"abi", "packId", "version", "command", "payload"}
            or request["abi"] != "1"
            or request["packId"] != "oryn.mlx-lm"
            or request["version"] != "0.10.0"
        ):
            raise ValueError("Request identity is invalid.")
        if request["command"] == "probe":
            import mlx.core as mx
            import mlx_lm

            from oryn_engine.mlx_vision import verify_vision_build

            verify_vision_build()

            del mlx_lm
            if not mx.metal.is_available():
                raise ValueError("Metal is unavailable.")
            result = {"backend": "mlx", "available": True}
        elif request["command"] == "generate":
            result = generate(request["payload"])
        else:
            raise ValueError("Unsupported command.")
        message({"type": "result", "ok": True, "result": result})
    except Exception:
        message({"type": "result", "ok": False, "error": {"code": "MODEL_GENERATION_FAILED"}})


if __name__ == "__main__":
    main()
