"""Complete indexed native Krea graphs and reviewed legacy numeric layouts."""

from __future__ import annotations

from pathlib import PurePosixPath

DERIVED_FREQUENCY = "rotary_emb.inv_freq"


def build_krea_graphs():
    from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder
    from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
    from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
    from mlx.utils import tree_flatten

    result = {}
    for name, model in (
        ("transformer", Krea2Transformer()),
        ("text_encoder", Krea2TextEncoder()),
        ("vae", QwenVAE()),
    ):
        quantizable = {
            n + ".weight"
            for n, module in model.named_modules()
            if hasattr(module, "to_quantized") and module.weight.shape[-1] % 64 == 0
        }
        result[name] = {
            n: {"shape": list(value.shape), "quantizable": n in quantizable}
            for n, value in tree_flatten(model.parameters())
        }
    return result


def legacy_norm(name, actual, expected):
    return (
        name.endswith(".weight")
        and name.split(".")[-2] in {"norm", "norm1", "norm2", "norm_out"}
        and len(expected) >= 3
        and expected[1:] == [1] * (len(expected) - 1)
        and actual == expected[:1]
    )


def normalized_shapes(tensors, graph, component):
    result = {}
    for name, value in tensors.items():
        expected = graph.get(name, {}).get("shape", [])
        if component == "vae" and legacy_norm(name, value.get("shape"), expected):
            value = value | {"shape": expected}
        result[name] = value
    if component == "text_encoder" and DERIVED_FREQUENCY not in result:
        result[DERIVED_FREQUENCY] = {"shape": graph[DERIVED_FREQUENCY]["shape"], "dtype": "F32"}
    return result


def complete_names(names, graph, component):
    """Indexes establish complete parameter identity; the worker checks all shapes."""
    remaining = set(names)
    for name, value in graph.items():
        if name == DERIVED_FREQUENCY and component == "text_encoder" and name not in remaining:
            continue
        if name not in remaining:
            return False
        remaining.remove(name)
        if value["quantizable"] and name.endswith(".weight"):
            companions = {
                name.removesuffix(".weight") + suffix for suffix in (".scales", ".biases")
            }
            if remaining & companions:
                if not companions <= remaining:
                    return False
                remaining.difference_update(companions)
    return not remaining


def _valid_tensor(name, value, graph, bits, component):
    shape, dtype = value.get("shape"), value.get("dtype")
    expected = graph.get(name)
    if expected:
        shape_ok = shape == expected["shape"] or (
            component == "vae" and legacy_norm(name, shape, expected["shape"])
        )
        if dtype in {"F16", "BF16", "F32"} and shape_ok:
            return True
        return (
            expected["quantizable"]
            and bits in {4, 8}
            and len(expected["shape"]) == 2
            and dtype == "U32"
            and shape == [expected["shape"][0], expected["shape"][1] * bits // 32]
        )
    prefix, _, suffix = name.rpartition(".")
    expected = graph.get(prefix + ".weight", {})
    return (
        suffix in {"scales", "biases"}
        and expected.get("quantizable")
        and dtype in {"F16", "BF16", "F32"}
        and shape
        in [
            [expected["shape"][0], expected["shape"][1] // g]
            for g in (32, 64, 128)
            if expected["shape"][1] % g == 0
        ]
    )


def inspect_saved_krea(inventory, metadata, catalog):
    """Bound index identities plus sampled tensor geometry, or every installed header."""
    from oryn_mflux_provider.structure import matches_graph

    graphs = catalog.get("nativeKreaGraphs")
    if not graphs or inventory.get("task") != "text-to-image":
        return None, []
    files = {f["name"]: f for f in inventory["files"]}
    selected = {
        directory: sorted(
            n
            for n in files
            if PurePosixPath(n).parent.as_posix() == directory and n.endswith(".safetensors")
        )
        for directory in graphs
    }
    if (
        not all(selected.values())
        or sum(map(len, selected.values())) > 128
        or any(
            PurePosixPath(n).parent.as_posix() == "." and n.endswith(".safetensors") for n in files
        )
        or len(selected["vae"]) != 1
    ):
        return None, []
    index_names = {d: f"{d}/model.safetensors.index.json" for d in ("transformer", "text_encoder")}
    if "vae/model.safetensors.index.json" in files:
        index_names["vae"] = "vae/model.safetensors.index.json"
    if not all(n in files for n in index_names.values()):
        return None, []
    requests = [{"file": n, "kind": "json"} for n in index_names.values() if n not in metadata]
    requests += [
        {"file": names[0], "kind": "safetensors-structure"}
        for names in selected.values()
        if names[0] not in metadata
    ]
    if requests:
        return None, requests
    for directory, names in selected.items():
        first = metadata[names[0]]
        version = first.get("metadata", {}).get("mflux_version")
        raw_bits = first.get("metadata", {}).get("quantization_level")
        if version not in {"0.18.0", "0.20.0", "16gb-bench"} or raw_bits not in {
            "4",
            "8",
            "None",
            "null",
        }:
            return None, []
        bits = int(raw_bits) if raw_bits in {"4", "8"} else None
        graph = graphs[directory]
        if directory in index_names:
            index = metadata[index_names[directory]].get("weight_map")
            if (
                not isinstance(index, dict)
                or not index
                or not all(
                    isinstance(k, str)
                    and isinstance(v, str)
                    and PurePosixPath(v).name == v
                    and v.endswith(".safetensors")
                    and "\\" not in v
                    for k, v in index.items()
                )
            ):
                return None, []
            if {f"{directory}/{v}" for v in index.values()} != set(names):
                return None, []
        else:
            index = {k: PurePosixPath(names[0]).name for k in first.get("tensors", {})}
        if not complete_names(index, graph, directory):
            return None, []
        collected = {}
        for name in names:
            if name not in metadata:
                continue
            header = metadata[name]
            tensors = header.get("tensors")
            if (
                not isinstance(tensors, dict)
                or not tensors
                or collected.keys() & tensors.keys()
                or header.get("metadata", {}).get("mflux_version") != version
                or header.get("metadata", {}).get("quantization_level") != raw_bits
                or header.get("headerBytes", 0) + header.get("dataBytes", 0)
                != files[name]["sizeBytes"]
                or set(tensors) != {k for k, v in index.items() if v == PurePosixPath(name).name}
                or not all(_valid_tensor(k, v, graph, bits, directory) for k, v in tensors.items())
            ):
                return None, []
            collected.update(tensors)
        if len(collected) == len(index) and not matches_graph(
            normalized_shapes(collected, graph, directory), graph, bits
        ):
            return None, []
    return selected, []


def native_parameters(model, parameters):
    """Restore derived frequency and reshape only the old native VAE scale layout."""
    import mlx.core as mx
    from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder
    from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
    from mlx.utils import tree_flatten, tree_unflatten

    if type(model) not in {Krea2TextEncoder, QwenVAE}:
        return parameters
    actual = dict(tree_flatten(parameters))
    expected = dict(tree_flatten(model.parameters()))
    if type(model) is Krea2TextEncoder:
        native = expected[DERIVED_FREQUENCY]
        if DERIVED_FREQUENCY in actual and not bool(
            mx.array_equal(actual[DERIVED_FREQUENCY], native)
        ):
            raise ValueError("Krea derived rotary frequencies changed")
        actual.setdefault(DERIVED_FREQUENCY, native)
    else:
        for name, value in actual.items():
            if name in expected and legacy_norm(
                name, list(value.shape), list(expected[name].shape)
            ):
                actual[name] = value.reshape(expected[name].shape)
    return tree_unflatten(list(actual.items()))


def saved_transformer_component(root, component, save_subdir):
    from dataclasses import replace

    from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition

    if (
        root is not None
        and component.variant_selector is Krea2WeightDefinition._select_transformer_variant
        and (root / "transformer/model.safetensors.index.json").is_file()
        and not any(root.glob("*.safetensors"))
    ):
        return replace(component, hf_subdir="transformer", variant_selector=None), "transformer"
    return component, save_subdir
