"""Extract MLX-Audio's native model registry without importing model code."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

TASKS = {"tts": "text-to-speech", "stt": "automatic-speech-recognition"}
AMBIGUOUS = {"llama", "qwen3", "dense"}
# Library category membership is not itself a public task binding. These native
# defaults require an audio-understanding contract rather than transcription.
WITHHELD_FAMILIES = {
    "automatic-speech-recognition": {
        "moss_music": "The native default generates a musical description, not a transcript."
    }
}
# Companion data for reviewed native layouts. Graph weights may share the main
# SafeTensors file (Soprano); separate decoder filenames must not be invented.
FAMILY_REQUIRED_FILES = {
    "fish_drama3": ["codec.safetensors", "tokenizer.json", "tokenizer_config.json"],
    "kitten_tts": ["voices.npz"],
    "soprano": ["tokenizer.json", "tokenizer_config.json"],
}


def _task_generate(node: ast.FunctionDef, primary: str) -> bool:
    args = node.args
    positional = args.posonlyargs + args.args
    parameters = [a.arg for a in positional + args.kwonlyargs]
    required = [a.arg for a in positional[: len(positional) - len(args.defaults)]]
    required += [
        a.arg
        for a, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
        if default is None
    ]
    return primary in parameters and not set(required) - {"self", primary}


def _generate(root: Path, module: str, symbol: str = "Model", seen=None, *, primary=None):
    seen = set() if seen is None else seen
    key = (module, symbol)
    if key in seen:
        return None
    seen.add(key)
    path = root.joinpath(*module.split("."))
    package = path.is_dir()
    path = path / "__init__.py" if package else path.with_suffix(".py")
    if not path.is_file() or not path.is_relative_to(root):
        return None
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == symbol:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "generate":
                    return member
            for base in node.bases:
                if isinstance(base, ast.Name):
                    result = _generate(root, module, base.id, seen, primary=primary)
                    if result:
                        return result
            # Some upstream packages expose a wrapper whose attribute lookup
            # delegates to one of several bundled models. Only follow an
            # explicit self._model assignment and an explicit __getattr__ hook;
            # choose a unique generate signature matching this public task.
            delegates = any(
                isinstance(member, ast.FunctionDef)
                and member.name == "__getattr__"
                and any(
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "getattr"
                    and call.args
                    and isinstance(call.args[0], ast.Attribute)
                    and isinstance(call.args[0].value, ast.Name)
                    and call.args[0].value.id == "self"
                    and call.args[0].attr == "_model"
                    for call in ast.walk(member)
                )
                for member in node.body
            )
            if delegates and primary:
                candidates = []
                for assignment in ast.walk(node):
                    if not isinstance(assignment, ast.Assign) or not any(
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr == "_model"
                        for target in assignment.targets
                    ):
                        continue
                    for call in ast.walk(assignment.value):
                        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                            result = _generate(
                                root, module, call.func.id, set(seen), primary=primary
                            )
                            if result and _task_generate(result, primary):
                                candidates.append(result)
                unique = {(result.lineno, ast.dump(result)): result for result in candidates}
                if len(unique) == 1:
                    return next(iter(unique.values()))
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == symbol for t in node.targets)
            and isinstance(node.value, ast.Name)
        ):
            result = _generate(root, module, node.value.id, seen, primary=primary)
            if result:
                return result
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) != symbol:
                    continue
                if node.level:
                    parts = module.split(".") if package else module.split(".")[:-1]
                    parts = parts[: len(parts) - node.level + 1]
                    target = ".".join([*parts, *([node.module] if node.module else [])])
                elif node.module and node.module.startswith("mlx_audio."):
                    target = node.module.removeprefix("mlx_audio.")
                else:
                    continue
                result = _generate(root, target, alias.name, seen, primary=primary)
                if result:
                    return result
    return None


def build_catalog(root: Path, version: str) -> dict:
    models = {}
    for category, task in TASKS.items():
        families = {
            path.name
            for path in (root / category / "models").iterdir()
            if path.is_dir() and not path.name.startswith("_")
        }
        remapping = {}
        tree = ast.parse((root / category / "utils.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "MODEL_REMAPPING" for t in node.targets
            ):
                remapping = ast.literal_eval(node.value)
        admitted = families - AMBIGUOUS - WITHHELD_FAMILIES.get(task, {}).keys()
        bindings = {family: family for family in admitted}
        bindings.update({name: family for name, family in remapping.items() if family in admitted})
        entries = {}
        for name, family in sorted(bindings.items()):
            primary = "text" if category == "tts" else "audio"
            generate = _generate(root, f"{category}.models.{family}", primary=primary)
            if generate is None:
                continue
            args = generate.args
            positional = args.posonlyargs + args.args
            parameters = [a.arg for a in positional + args.kwonlyargs if a.arg != "self"]
            # Forced alignment and lower-level decoders have different contracts.
            if primary not in parameters:
                continue
            required = [a.arg for a in positional[: len(positional) - len(args.defaults)]]
            required += [
                a.arg
                for a, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
                if default is None
            ]
            if set(required) - {"self", primary}:
                continue
            defaults = {}
            for arg, value in [
                *zip(
                    positional[len(positional) - len(args.defaults) :], args.defaults, strict=True
                ),
                *zip(args.kwonlyargs, args.kw_defaults, strict=True),
            ]:
                try:
                    literal = ast.literal_eval(value)
                except (ValueError, TypeError):
                    continue
                if literal is None or type(literal) in {str, int, float, bool}:
                    defaults[arg.arg] = literal
            entries[name] = {
                "category": category,
                "family": family,
                "parameters": parameters,
                "defaults": defaults,
                "requiredFiles": FAMILY_REQUIRED_FILES.get(family, []),
            }
        models[task] = entries
    return {
        "version": version,
        "models": models,
        "sources": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))
        },
    }
