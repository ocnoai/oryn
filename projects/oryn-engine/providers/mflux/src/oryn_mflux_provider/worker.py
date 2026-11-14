"""Provider worker: standard task input, native backend command, standard artifact."""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
from pathlib import Path

from oryn_mflux_provider.inspection import (
    catalog,
    inspect_inventory,
    legacy_metadata,
    numbered_shards,
)

MAX_MESSAGE_BYTES = 1024**2
_protocol_stdout = sys.stdout


def message(value):
    print(json.dumps(value, separators=(",", ":")), file=_protocol_stdout, flush=True)


def event(stage, progress):
    message(
        {"type": "event", "event": "job.progress", "data": {"stage": stage, "progress": progress}}
    )


def _entry(plan):
    data, digest = catalog()
    if (
        not isinstance(plan, dict)
        or plan.get("schemaVersion") != 1
        or plan.get("catalogDigest") != digest
    ):
        raise ValueError("Provider plan/catalog changed")
    entry = next((e for e in data["entries"] if e["id"] == plan.get("entryId")), None)
    if entry is None:
        raise ValueError("Provider entry is unavailable")
    return entry


def prepare(payload, *, generating=False):
    from oryn_engine.model_metadata import (
        metadata_file,
        preflight_safetensors,
        validate_model_config,
    )
    from oryn_engine.providers import safe_relative
    from oryn_engine.runners import _diffusion_input

    if set(payload) != {"modelRoot", "plan", "input"}:
        raise ValueError("Provider preparation fields are invalid")
    plan = payload["plan"]
    entry = _entry(plan)
    if set(plan) != {
        "schemaVersion",
        "catalogDigest",
        "entryId",
        "selectedFiles",
        "components",
        "mode",
        "maximumPixels",
        "estimatedMemoryBytes",
    }:
        raise ValueError("Provider plan fields are invalid")
    expected_mode = "low-memory" if entry["controls"]["lowMemory"] else "standard"
    if plan["mode"] != expected_mode or plan["maximumPixels"] != 1024**2:
        raise ValueError("Provider execution mode changed")
    raw_input = payload["input"]
    defaults = entry["defaults"]
    value = _diffusion_input(
        defaults | raw_input if isinstance(raw_input, dict) else raw_input,
        entry["taskContract"] if generating else "text-to-image-v1",
    )
    root = Path(payload["modelRoot"])
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("Provider model root is invalid")
    names = plan.get("selectedFiles")
    if (
        not isinstance(names, list)
        or len(names) > 10000
        or any(not safe_relative(n) for n in names)
    ):
        raise ValueError("Provider file selection is invalid")
    for name in names:
        path = root / name
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Selected provider artifact is unavailable")
        if name.endswith("config.json"):
            config = metadata_file(path)
            if config.get("auto_map") or config.get("custom_pipelines"):
                raise ValueError("Repository code cannot select provider execution")
            validate_model_config(config)
    directories = {str(p.parent) for p in map(Path, names) if p.suffix == ".safetensors"}
    for directory in directories:
        preflight_safetensors(root / directory)
    from safetensors import safe_open

    definitions = {c["name"]: c for c in entry["components"]}
    indexed_krea = any(
        c.get("metadataCoverage") == "complete-index-graph" for c in plan["components"]
    )
    structural = indexed_krea or any(
        c.get("metadataCoverage") == "complete-graph" for c in plan["components"]
    )
    marker = "complete-index-graph" if indexed_krea else "complete-graph"
    structural_metadata = {}
    if {c["name"] for c in plan["components"]} != set(definitions):
        raise ValueError("Provider component graph changed")
    for component in plan["components"]:
        definition = definitions[component["name"]]
        directory = (
            definition.get("indexedNativeDirectory", definition["directory"])
            if indexed_krea
            else definition["directory"]
        )
        if component["directory"] != directory or not component["files"]:
            raise ValueError("Provider component layout changed")
        if structural and (
            component.get("metadataCoverage") != marker
            or sorted(component["files"])
            != sorted(
                p.relative_to(root).as_posix()
                for p in (root / component["directory"]).glob("*.safetensors")
            )
        ):
            raise ValueError("Provider complete graph selection changed")
        headers = []
        for name in component["files"]:
            if name not in names or Path(name).parent.as_posix() != (component["directory"] or "."):
                raise ValueError("Provider component file changed")
            with safe_open(root / name, framework="numpy") as tensors:
                header = tensors.metadata() or {}
                headers.append({"metadata": header})
                if structural:
                    import struct

                    with (root / name).open("rb") as stream:
                        header_bytes = 8 + struct.unpack("<Q", stream.read(8))[0]
                    structural_metadata[name] = {
                        "metadata": header,
                        "headerBytes": header_bytes,
                        "dataBytes": (root / name).stat().st_size - header_bytes,
                        "tensors": {
                            key: {
                                "shape": tensors.get_slice(key).get_shape(),
                                "dtype": tensors.get_slice(key).get_dtype(),
                            }
                            for key in list(tensors.keys())
                        },
                    }
                raw = header.get("quantization_level")
                bits = None if raw in (None, "None", "null") else int(raw)
            if bits != component["quantizationBits"]:
                raise ValueError("Stored quantization differs across shards or from the plan")
        index = root / component["directory"] / "model.safetensors.index.json"
        if (
            not index.exists()
            and len(component["files"]) > 1
            and not structural
            and (not numbered_shards(component["files"]) or not legacy_metadata(headers))
        ):
            raise ValueError("Unindexed native component shards are inconsistent")
    if structural:
        inventory = {
            "task": entry["taskContract"].removesuffix("-v1"),
            "files": [{"name": n, "sizeBytes": (root / n).stat().st_size} for n in names],
        }
        if indexed_krea:
            from oryn_mflux_provider.krea_layout import inspect_saved_krea

            if entry["architecture"] != "Krea2":
                raise ValueError("Indexed native graph family changed")
            structural_metadata.update(
                {n: metadata_file(root / n) for n in names if n.endswith(".safetensors.index.json")}
            )
            matched, requests = inspect_saved_krea(inventory, structural_metadata, catalog()[0])
        else:
            from oryn_mflux_provider.structure import resolve_flux_graph

            matched, requests = resolve_flux_graph(inventory, structural_metadata, catalog()[0])
            if matched and matched["id"] != entry["id"]:
                raise ValueError("Provider complete native graph family changed")
        if requests or matched is None:
            raise ValueError("Provider complete native graph changed")
    if value["width"] % 32 or value["height"] % 32 or value["width"] * value["height"] > 1024**2:
        raise ValueError("Image dimensions exceed the provider plan")
    if value["steps"] > 50 or value["guidanceScale"] > 20:
        raise ValueError("Generation settings exceed the provider plan")
    if (
        entry["controls"]["guidanceScale"] not in {"honored", "conditional"}
        and value["guidanceScale"] != defaults["guidanceScale"]
    ):
        raise ValueError("The backend does not support the requested guidance")
    if value["negativePrompt"] and entry["controls"]["negativePrompt"] not in {
        "honored",
        "conditional",
    }:
        raise ValueError("The backend does not support a negative prompt")
    return {"entry": entry, "input": value, "plan": plan, "modelRoot": str(root.resolve())}


class Progress:
    def call_before_loop(self, **_):
        event("denoising", 0.2)

    def call_in_loop(self, t, config, **_):
        event("denoising", 0.2 + min(1, (t + 1) / max(1, config.num_inference_steps)) * 0.65)

    def call_after_loop(self, **_):
        event("decoding-image", 0.9)


@contextlib.contextmanager
def validated_weights():
    """Require each native component's complete graph after quantization setup."""
    import mlx.nn as nn
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mlx.utils import tree_flatten

    original_update = nn.Module.update
    original_single = WeightApplier.apply_and_quantize_single
    original_set = WeightApplier._set_weights
    original_load = WeightLoader._try_load_mflux_format
    original_component = WeightLoader._load_component
    guarded = set()

    def load(path):
        from oryn_mflux_provider.swift_layout import load_swift_component

        return load_swift_component(path, original_load(path), catalog()[0])

    def load_component(root, component, raw_weights_cache=None, save_subdir=None):
        from oryn_mflux_provider.krea_layout import saved_transformer_component

        component, save_subdir = saved_transformer_component(root, component, save_subdir)
        return original_component(root, component, raw_weights_cache, save_subdir)

    def update(model, parameters, strict=True):
        if id(model) in guarded:
            from oryn_mflux_provider.krea_layout import native_parameters
            from oryn_mflux_provider.structure import restore_flux_output_bias

            restore_flux_output_bias(model, parameters)
            parameters = native_parameters(model, parameters)
            expected = dict(tree_flatten(model.parameters()))
            actual = dict(tree_flatten(parameters))
            if set(actual) != set(expected) or any(
                getattr(actual[k], "shape", None) != expected[k].shape for k in expected
            ):
                raise ValueError("Native component weight graph is incomplete or inconsistent")
            strict = True
        return original_update(model, parameters, strict=strict)

    def single(weights, model, component, quantize_arg, quantization_predicate=None):
        guarded.add(id(model))
        try:
            return original_single(weights, model, component, quantize_arg, quantization_predicate)
        finally:
            guarded.discard(id(model))

    def set_weights(weights, models, components=None):
        if any(name not in weights.components for name in models):
            raise ValueError("Native component weights are missing")
        identities = {id(model) for model in models.values()}
        guarded.update(identities)
        try:
            return original_set(weights, models, components)
        finally:
            guarded.difference_update(identities)

    try:
        nn.Module.update = update
        WeightApplier.apply_and_quantize_single = staticmethod(single)
        WeightApplier._set_weights = staticmethod(set_weights)
        WeightLoader._try_load_mflux_format = staticmethod(load)
        WeightLoader._load_component = staticmethod(load_component)
        yield
    finally:
        nn.Module.update = original_update
        WeightApplier.apply_and_quantize_single = staticmethod(original_single)
        WeightApplier._set_weights = staticmethod(original_set)
        WeightLoader._try_load_mflux_format = staticmethod(original_load)
        WeightLoader._load_component = staticmethod(original_component)


@contextlib.contextmanager
def validated_decoding():
    import mlx.core as mx
    from mflux.utils.image_util import ImageUtil

    original = ImageUtil._denormalize

    def denormalize(images):
        # Check before clipping/casting: NaNs otherwise become a successful black PNG.
        if not bool(mx.all(mx.isfinite(images)).item()):
            raise FloatingPointError("The backend produced non-finite image values")
        return original(images)

    try:
        ImageUtil._denormalize = staticmethod(denormalize)
        yield
    finally:
        ImageUtil._denormalize = staticmethod(original)


def generate(payload):
    if set(payload) != {"modelRoot", "plan", "input", "outputPath", "inputPath"}:
        raise ValueError("Provider generation fields are invalid")
    prepared = prepare(
        {k: v for k, v in payload.items() if k not in {"outputPath", "inputPath"}}, generating=True
    )
    entry, value, plan = prepared["entry"], prepared["input"], prepared["plan"]
    output = Path(payload["outputPath"])
    if not output.is_absolute() or output.name != "image.png" or not output.parent.is_dir():
        raise ValueError("Provider output path is invalid")
    input_path = payload["inputPath"]
    if entry["taskContract"] == "image-editing-v1":
        from oryn_engine.shared_tasks import open_task_image

        if not isinstance(input_path, str) or not Path(input_path).is_absolute():
            raise ValueError("Owned image input is missing")
        image, _ = open_task_image(Path(input_path))
        input_path = output.parent / "condition.png"
        image.save(input_path, format="PNG")
    elif input_path is not None:
        raise ValueError("This provider task does not accept image input")
    # The module is from the build catalog. No module, command, or arbitrary flag is
    # accepted from the checkpoint or the task input.
    module = importlib.import_module(entry["command"])
    from mflux.callbacks.callback_registry import CallbackRegistry
    from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader
    from PIL import Image

    original_start = CallbackRegistry.start
    original_tokenizer = TokenizerLoader._load_raw_tokenizer

    def start(registry, *args, **kwargs):
        registry.register(Progress())
        return original_start(registry, *args, **kwargs)

    def tokenizer(*args, **kwargs):
        native_template = kwargs.get("chat_template")
        loaded = original_tokenizer(*args, **kwargs)
        # Pack-owned templates may be used. Repository templates are inert.
        loaded.chat_template = native_template
        if entry.get("promptBinding"):
            from oryn_mflux_provider.prompt_binding import render

            maximum = min(t["maxLength"] for t in entry["tokenizers"])

            def apply(messages, **options):
                prompt = render(messages, binding=entry["promptBinding"], **options)
                if len(loaded.encode(prompt, add_special_tokens=True)) > maximum:
                    raise ValueError("Encoder prompt exceeds the native token budget")
                return prompt

            loaded.apply_chat_template = apply
        return loaded

    argv = [
        entry["command"],
        "--model",
        prepared["modelRoot"],
        "--base-model",
        entry.get("baseModelId", entry["id"]),
        "--prompt=" + value["prompt"],
        "--output",
        str(output),
        "--seed",
        str(value["seed"]),
        "--steps",
        str(value["steps"]),
        "--width",
        str(value["width"]),
        "--height",
        str(value["height"]),
    ]
    if input_path is not None:
        argv.extend(["--image-paths", str(input_path)])
    if entry["controls"]["guidanceScale"] in {"honored", "conditional"}:
        argv.extend(["--guidance", str(value["guidanceScale"])])
    if value["negativePrompt"]:
        argv.append("--negative-prompt=" + value["negativePrompt"])
    if plan["mode"] == "low-memory" and entry["controls"]["lowMemory"]:
        argv.append("--low-ram")
    old_argv = sys.argv
    try:
        sys.argv = argv
        CallbackRegistry.start = start
        TokenizerLoader._load_raw_tokenizer = staticmethod(tokenizer)
        event("loading-model", 0.02)
        with contextlib.redirect_stdout(sys.stderr), validated_weights(), validated_decoding():
            module.main()
    finally:
        sys.argv = old_argv
        CallbackRegistry.start = original_start
        TokenizerLoader._load_raw_tokenizer = staticmethod(original_tokenizer)
    if output.is_symlink() or not output.is_file() or not 0 < output.stat().st_size <= 64 * 1024**2:
        raise ValueError("Provider output is missing or invalid")
    with Image.open(output) as image:
        if image.format != "PNG" or image.size != (value["width"], value["height"]):
            raise ValueError("Provider output does not match its task")
        image.verify()
    event("writing-artifact", 0.98)
    import mlx.core as mx

    return {
        "relativeOutput": "image.png",
        "mediaType": "image/png",
        "backend": "mlx",
        "width": value["width"],
        "height": value["height"],
        "seed": value["seed"],
        "catalogDigest": plan["catalogDigest"],
        "peakMetalAllocationBytes": int(mx.get_peak_memory()),
    }


def main():
    try:
        raw = sys.stdin.buffer.readline(MAX_MESSAGE_BYTES + 1)
        if not raw or len(raw) > MAX_MESSAGE_BYTES:
            raise ValueError("Provider request size is invalid")
        request = json.loads(raw)
        if not isinstance(request, dict) or set(request) != {
            "abi",
            "packId",
            "version",
            "command",
            "payload",
        }:
            raise ValueError("Provider envelope is invalid")
        if request["abi"] != "1" or request["packId"] != "oryn.mflux-mlx":
            raise ValueError("Provider identity is invalid")
        command, payload = request["command"], request["payload"]
        if command in {"probe", "describe"}:
            from importlib.metadata import version

            data, digest = catalog()
            result = {
                "providerAbi": "1",
                "catalogDigest": digest,
                "foundation": "oryn.runtime.mlx",
                "adapter": "oryn.adapter.mflux-image",
                "mflux": version("mflux"),
                "mlx": version("mlx"),
                "networkPolicy": os.environ.get("ORYN_PACK_NETWORK_POLICY"),
                "models": [e["id"] for e in data["entries"]],
                "uncovered": data["uncovered"],
            }
            if command == "describe":
                result["entries"] = data["entries"]
                result["quantizationBits"] = data["quantizationBits"]
        elif command == "inspect":
            if not isinstance(payload, dict) or set(payload) != {"inventory", "metadata"}:
                raise ValueError("Provider inspection fields are invalid")
            result = inspect_inventory(payload["inventory"], payload["metadata"])
        elif command == "prepare":
            prepared = prepare(payload)
            result = {
                "ready": True,
                "catalogDigest": prepared["plan"]["catalogDigest"],
                "estimatedMemoryBytes": prepared["plan"]["estimatedMemoryBytes"],
            }
        elif command == "generate":
            result = generate(payload)
        else:
            raise ValueError("Provider command is unavailable")
        message({"type": "result", "ok": True, "result": result})
    except Exception as exc:
        print(f"Provider failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        code = (
            "MODEL_GENERATION_FAILED"
            if isinstance(exc, FloatingPointError)
            else "MODEL_RUNTIME_INCOMPATIBLE"
        )
        message({"type": "result", "ok": False, "error": {"code": code}})


if __name__ == "__main__":
    main()
