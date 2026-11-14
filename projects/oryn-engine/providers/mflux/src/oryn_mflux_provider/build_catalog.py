"""Discover the pinned backend at build time; never inspect downloaded Python."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import site
from copy import deepcopy
from pathlib import Path


def _called_classes(module):
    tree = ast.parse(inspect.getsource(module))
    return {
        getattr(module, n.func.id)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and isinstance(getattr(module, n.func.id, None), type)
        and hasattr(getattr(module, n.func.id), "generate_image")
    }


def build_catalog():
    import mflux
    from mflux.cli.capabilities import build_capabilities
    from mflux.cli.defaults.defaults import QUANTIZE_CHOICES, model_inference_steps
    from mflux.models.common.cli.save import MODEL_CLASSES
    from mflux.models.common.config.model_config import AVAILABLE_MODELS
    from mflux.models.common.weights.loading.weight_definition import ComponentDefinition

    from oryn_mflux_provider.krea_layout import build_krea_graphs
    from oryn_mflux_provider.prompt_binding import binding_for
    from oryn_mflux_provider.structure import build_flux_graphs
    from oryn_mflux_provider.swift_layout import build_swift_aliases

    root = Path(next(iter(mflux.__path__)))
    upstream = build_capabilities()
    command_records, native_classes = [], set()
    for command in upstream["commands"]:
        if command["coverage"] != "full":
            continue
        module = importlib.import_module(command["module"])
        native_classes.update(_called_classes(module))
        parser = module.build_parser()
        # This binding implements a single text-to-image result. Extra required inputs
        # need a different task binding; a permissive Python signature is not enough.
        allowed_required = {"prompt", "model", "output"}
        if any(a.required and a.dest not in allowed_required for a in parser._actions):
            continue
        if any(
            g.required and not any(a.dest == "prompt" for a in g._group_actions)
            for g in parser._mutually_exclusive_groups
        ):
            continue
        if getattr(parser, "require_init_image", False):
            continue
        options = {o["flag"]: o for o in command["options"]}
        if "--depth-image-path" in options:
            continue
        required = {
            "--model",
            "--base-model",
            "--prompt",
            "--output",
            "--seed",
            "--steps",
            "--width",
            "--height",
        }
        if not required <= set(options):
            continue
        if any(options[k]["status"] not in {"honored", "conditional"} for k in required):
            continue
        command_records.append((command, module, _called_classes(module)))

    entries, uncovered = [], []
    for key, config in AVAILABLE_MODELS.items():
        specialized = {
            c
            for c in native_classes
            if inspect.signature(c.__init__).parameters.get("model_config")
            and inspect.signature(c.__init__).parameters["model_config"].default is config
        }
        cls = next(iter(specialized)) if len(specialized) == 1 else MODEL_CLASSES.get(key)
        if cls is not None:
            required_inputs = {
                name
                for name, p in inspect.signature(cls.generate_image).parameters.items()
                if p.default is inspect.Parameter.empty and name != "self"
            }
            if (
                not required_inputs <= {"seed", "prompt"}
                or config.controlnet_model
                or config.custom_transformer_model
            ):
                uncovered.append({"model": key, "reason": "ADDITIONAL_INPUT_BINDING_REQUIRED"})
                continue
        choices = [(c, m) for c, m, classes in command_records if cls in classes]
        if cls is None or not choices:
            uncovered.append({"model": key, "reason": "TASK_BINDING_UNAVAILABLE"})
            continue
        # Prefer the backend's dedicated command; use a shared class command otherwise.
        choices.sort(
            key=lambda item: (
                item[0]["traits"].get("default_model") not in {key, *config.aliases},
                len(item[0]["command"]),
                item[0]["command"],
            )
        )
        command, module = choices[0]
        owner = importlib.import_module(cls.__module__)
        definitions = {
            value
            for value in vars(owner).values()
            if isinstance(value, type)
            and hasattr(value, "get_components")
            and hasattr(value, "get_tokenizers")
        }
        if len(definitions) != 1:
            uncovered.append({"model": key, "reason": "COMPONENT_DESCRIPTION_UNAVAILABLE"})
            continue
        definition = next(iter(definitions))
        components = definition.get_components()
        tokenizers = definition.get_tokenizers()
        if any(c.download_url is not None for c in components):
            uncovered.append({"model": key, "reason": "EXTERNAL_COMPONENT_REQUIRED"})
            continue
        prompt_binding = binding_for(tokenizers)
        if (
            any(t.use_chat_template and not t.chat_template for t in tokenizers)
            and not prompt_binding
        ):
            uncovered.append({"model": key, "reason": "NATIVE_PROMPT_BINDING_REQUIRED"})
            continue
        save_dirs = ComponentDefinition.save_subdirs(components)
        native_params = inspect.signature(cls.generate_image).parameters
        options = {o["flag"]: o for o in command["options"]}
        guidance = getattr(module, "DEFAULT_GUIDANCE", None)
        if guidance is None and "guidance" in native_params:
            guidance = native_params["guidance"].default
        if not isinstance(guidance, (int, float)):
            guidance = 1.0
        entries.append(
            {
                "id": key,
                "architecture": cls.__name__,
                "aliases": sorted({key, config.model_name, *config.aliases}),
                "command": command["module"],
                "taskContract": "text-to-image-v1",
                "promptBinding": prompt_binding,
                "components": [
                    {
                        "name": c.name,
                        "directory": save_dirs[c.name],
                        "nativeDirectory": c.hf_subdir,
                        "nativeFiles": c.weight_files,
                        "precision": str(c.precision) if c.precision is not None else None,
                        "skipQuantization": c.skip_quantization,
                        **(
                            {"indexedNativeDirectory": "transformer"}
                            if cls.__name__ == "Krea2" and c.name == "transformer"
                            else {}
                        ),
                    }
                    for c in components
                ],
                "tokenizers": [
                    {
                        "directory": t.hf_subdir,
                        "fallbacks": t.fallback_subdirs or [],
                        "maxLength": min(t.max_length, 2048),
                        "nativeTemplate": t.chat_template,
                    }
                    for t in tokenizers
                ],
                "defaults": {
                    "width": 512,
                    "height": 512,
                    "steps": model_inference_steps(key),
                    "guidanceScale": float(guidance),
                },
                "controls": {
                    "guidanceScale": options.get("--guidance", {}).get("status", "unavailable")
                    if config.supports_guidance is not False
                    or options.get("--guidance", {}).get("status") != "conditional"
                    else "ignored",
                    "negativePrompt": options.get("--negative-prompt", {}).get(
                        "status", "unavailable"
                    ),
                    "lowMemory": options.get("--low-ram", {}).get("status") == "honored",
                },
            }
        )
    # The reviewed editing binding reuses the native Flux2 component graph and
    # one owned image. Derive configurations from the pinned CLI's family list.
    for command in upstream["commands"]:
        if command["command"] != "mflux-generate-flux2-edit" or command["coverage"] != "full":
            continue
        module = importlib.import_module(command["module"])
        classes = _called_classes(module)
        if len(classes) != 1:
            raise ValueError("Native image editor identity changed")
        cls = next(iter(classes))
        options = {o["flag"]: o for o in command["options"]}
        required = {
            "--model",
            "--base-model",
            "--prompt",
            "--output",
            "--seed",
            "--steps",
            "--width",
            "--height",
            "--image-paths",
        }
        if (
            not required <= set(options)
            or any(options[k]["status"] not in {"honored", "conditional"} for k in required)
            or "image_paths" not in inspect.signature(cls.generate_image).parameters
        ):
            raise ValueError("Native image conditioning binding changed")
        parser = module.build_parser()
        if any(
            a.required and a.dest not in {"prompt", "model", "output", "image_paths"}
            for a in parser._actions
        ):
            raise ValueError("Native image editor requires additional inputs")
        family = {module.DEFAULT_MODEL, *module.FAMILY_MODELS}
        for base in list(entries):
            if base["id"] not in family:
                continue
            entry = deepcopy(base)
            entry.update(
                id=base["id"] + ".edit",
                baseModelId=base["id"],
                architecture=cls.__name__,
                command=command["module"],
                taskContract="image-editing-v1",
            )
            entry["controls"]["lowMemory"] = options.get("--low-ram", {}).get("status") == "honored"
            entries.append(entry)
    sources = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }
    graphs = build_flux_graphs()
    return {
        "schemaVersion": 1,
        "providerId": "oryn.mflux",
        "backendVersion": upstream["mflux_version"],
        "binding": "mflux-cli-image-v2",
        "quantizationBits": list(QUANTIZE_CHOICES),
        "nativeFluxGraphs": graphs,
        "swiftFluxAliases": build_swift_aliases(graphs),
        "nativeKreaGraphs": build_krea_graphs(),
        "entries": entries,
        "uncovered": uncovered,
        "sources": sources,
        "codeFiles": {
            "oryn_mflux_provider/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "commandCoverage": [
            {"command": c["command"], "coverage": c["coverage"]} for c in upstream["commands"]
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.runtime_root:
        root = args.runtime_root.resolve(strict=True)
        site.addsitedir(str(root / "runtime-foundations/oryn.runtime.mlx/site-packages"))
        site.addsitedir(str(root / "runtime-packs/oryn.mflux-mlx/site-packages"))
    catalog = build_catalog()
    destination = Path(__file__).with_name("catalog.json")
    serialized = json.dumps(catalog, sort_keys=True, indent=2) + "\n"
    if args.check:
        if destination.read_text() != serialized:
            raise ValueError("Provider catalog drifted; regenerate and review capability changes")
    else:
        destination.write_text(serialized)
    print(
        json.dumps(
            {
                "catalog": str(destination),
                "models": len(catalog["entries"]),
                "uncovered": catalog["uncovered"],
            }
        )
    )


if __name__ == "__main__":
    main()
