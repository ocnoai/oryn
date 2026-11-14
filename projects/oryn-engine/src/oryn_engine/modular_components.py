"""Closed native component bundles shared by modular pipeline adapters.

Class tables and workflow components come from engine code. Repository loading
hints are inert provenance; no native from_pretrained loader sees an index.
"""

from __future__ import annotations

import re

from oryn_engine.model_metadata import preflight_safetensors, validate_model_config


def validate_index(index, *, family, blocks, classes):
    validate_model_config(index)
    if (
        set(index) != {*classes, "_class_name", "_blocks_class_name", "_diffusers_version"}
        or index["_class_name"] != family
        or index["_blocks_class_name"] != blocks
    ):
        raise ValueError("MODULAR_PIPELINE")
    for name, identity in classes.items():
        value = index[name]
        if not isinstance(value, list) or len(value) != 3 or value[:2] != list(identity):
            raise ValueError("MODULAR_COMPONENT_IDENTITY")
        hint = value[2]
        if (
            not isinstance(hint, dict)
            or set(hint)
            != {"pretrained_model_name_or_path", "revision", "subfolder", "type_hint", "variant"}
            or not isinstance(hint["pretrained_model_name_or_path"], str)
            or not re.fullmatch(
                r"[A-Za-z0-9._-]{1,96}/[A-Za-z0-9._-]{1,96}", hint["pretrained_model_name_or_path"]
            )
            or hint["revision"] is not None
            and not re.fullmatch(r"[0-9a-f]{40}", str(hint["revision"]))
            or hint["subfolder"] != name
            or hint["type_hint"] != list(identity)
            or hint["variant"] is not None
        ):
            raise ValueError("MODULAR_COMPONENT_BINDING")


def select_files(files, *, required, weight_stems):
    names = set(files)
    if not set(required) <= names:
        raise ValueError("MODULAR_METADATA_MISSING")
    selected = set(required)
    for component, stem in weight_stems.items():
        prefix = component + "/"
        shards = {n for n in names if n.startswith(prefix) and n.endswith(".safetensors")}
        index = prefix + stem + ".safetensors.index.json"
        if shards == {prefix + stem + ".safetensors"}:
            if index in names:
                raise ValueError("MODULAR_AMBIGUOUS_INDEX")
        elif (
            not 1 <= len(shards) <= 32
            or shards
            != {
                f"{prefix}{stem}-{i:05d}-of-{len(shards):05d}.safetensors"
                for i in range(1, len(shards) + 1)
            }
            or index not in names
        ):
            raise ValueError("MODULAR_SHARDS")
        else:
            selected.add(index)
        selected.update(shards)
    return tuple(sorted(selected))


def validate_headers(selected, specs, read_header, read_json):
    for component, expected in specs.items():
        actual, owners = {}, {}
        for filename in selected:
            if not filename.startswith(component + "/") or not filename.endswith(".safetensors"):
                continue
            header = read_header(filename)
            if not isinstance(header, dict) or not isinstance(header.get("tensors"), dict):
                raise ValueError("MODULAR_HEADER")
            for name, record in header["tensors"].items():
                if (
                    name in actual
                    or not isinstance(record, dict)
                    or record.get("dtype") not in {"F32", "F16", "BF16"}
                ):
                    raise ValueError("MODULAR_TENSOR")
                actual[name] = tuple(record.get("shape", []))
                owners[name] = filename.split("/", 1)[1]
        if actual != expected:
            raise ValueError("MODULAR_GRAPH_INCOMPLETE")
        indexes = [
            n for n in selected if n.startswith(component + "/") and n.endswith(".index.json")
        ]
        if indexes and (len(indexes) != 1 or read_json(indexes[0]).get("weight_map") != owners):
            raise ValueError("MODULAR_INDEX_BINDING")


def load_components(root, selected, specs, factory, backend):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    if backend not in {"cpu", "mps", "cuda"}:
        raise ValueError("MODULAR_BACKEND")
    for name, expected in specs.items():
        if preflight_safetensors(root / name) != expected:
            raise ValueError("MODULAR_GRAPH_INCOMPLETE")
    # Parameters are deferred; native nonpersistent buffers retain their values.
    with init_empty_weights():
        components = factory()
    if set(components) != set(specs):
        raise ValueError("MODULAR_NATIVE_COMPONENTS")
    for name, model in components.items():
        if {n: tuple(t.shape) for n, t in model.state_dict().items()} != specs[name]:
            raise ValueError("MODULAR_NATIVE_GRAPH_CHANGED")
        weights = {}
        for filename in selected:
            if filename.startswith(name + "/") and filename.endswith(".safetensors"):
                for key, tensor in load_file(root / filename, device="cpu").items():
                    if (
                        key in weights
                        or tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32}
                        or not torch.isfinite(tensor).all().item()
                    ):
                        raise ValueError("MODULAR_WEIGHT_INVALID")
                    weights[key] = tensor
        model.load_state_dict(weights, strict=True, assign=True)
        model.to(device=backend, dtype=torch.float32).eval()
    return components
