"""Match native saved graphs using inert tensor geometry, without model aliases."""

from __future__ import annotations

import re
from pathlib import PurePosixPath


def build_flux_graphs():
    """Build-time only: MLX constructors are lazy; no weights are evaluated or saved."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux.model.flux_text_encoder.clip_encoder.clip_encoder import CLIPEncoder
    from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder
    from mflux.models.flux.model.flux_transformer.transformer import Transformer
    from mflux.models.flux.model.flux_vae.vae import VAE
    from mflux.models.flux.weights.flux_weight_definition import FluxWeightDefinition
    from mlx.utils import tree_flatten

    def describe(model):
        quantizable = {
            name + ".weight"
            for name, module in model.named_modules()
            if FluxWeightDefinition.quantization_predicate(name, module)
        }
        return {
            name: {"shape": list(value.shape), "quantizable": name in quantizable}
            for name, value in tree_flatten(model.parameters())
        }

    return {
        "components": {
            "vae": describe(VAE()),
            "text_encoder": describe(CLIPEncoder()),
            "text_encoder_2": describe(T5Encoder()),
        },
        # The backend instantiates pruned transformer depths from the saved graph.
        # Keep one block template per stack and its native maximum depth.
        "transformers": {
            family: {name: value for name, value in describe(Transformer(config, 1, 1)).items()}
            for family, config in (("schnell", ModelConfig.schnell()), ("dev", ModelConfig.dev()))
        },
        "maximumBlocks": {"transformer_blocks": 19, "single_transformer_blocks": 38},
    }


def _expanded_transformer(template, tensors, maxima):
    result = {k: v for k, v in template.items() if not any(k.startswith(p + ".") for p in maxima)}
    # Older native saves retain this learned affine bias. The worker restores
    # this exact parameter before strict loading; it is never discarded.
    bias = "norm_out.linear.bias"
    if bias in tensors and bias not in result:
        result[bias] = {
            "shape": [template["norm_out.linear.weight"]["shape"][0]],
            "quantizable": False,
        }
    for prefix, maximum in maxima.items():
        pattern = re.compile(re.escape(prefix) + r"\.([0-9]+)\.")
        indices = {int(match[1]) for name in tensors if (match := pattern.match(name))}
        if not indices or len(indices) > maximum or indices != set(range(len(indices))):
            return None
        block = {
            k.removeprefix(prefix + ".0."): v
            for k, v in template.items()
            if k.startswith(prefix + ".0.")
        }
        for index in range(len(indices)):
            result.update({f"{prefix}.{index}.{k}": v for k, v in block.items()})
    return result


def restore_flux_output_bias(model, parameters):
    """Preserve the older saved affine graph, including its nonzero bias."""
    import mlx.core as mx
    from mflux.models.flux.model.flux_transformer.transformer import Transformer

    if type(model) is not Transformer or "bias" in model.norm_out.linear:
        return
    norm = parameters.get("norm_out", {})
    linear = norm.get("linear", {}) if isinstance(norm, dict) else {}
    bias = linear.get("bias") if isinstance(linear, dict) else None
    if bias is None:
        return
    if getattr(bias, "shape", None) != (model.norm_out.embedding_dim * 2,) or bias.dtype not in (
        mx.float16,
        mx.bfloat16,
        mx.float32,
    ):
        raise ValueError("Saved FLUX output bias does not match its native weight graph")
    model.norm_out.linear.bias = bias


def matches_graph(tensors, expected, bits, group_size=None):
    """Check every tensor, including packed affine quantization companions."""
    if expected is None:
        return False
    consumed = set()
    for name, description in expected.items():
        value = tensors.get(name, {})
        shape = description["shape"]
        if value.get("dtype") in {"F16", "BF16", "F32"} and value.get("shape") == shape:
            consumed.add(name)
            continue
        if (
            not description["quantizable"]
            or bits not in {2, 3, 4, 5, 6, 8}
            or len(shape) != 2
            or shape[1] * bits % 32
            or value.get("dtype") != "U32"
            or value.get("shape") != [shape[0], shape[1] * bits // 32]
        ):
            return False
        prefix = name.removesuffix(".weight")
        scales, biases = tensors.get(prefix + ".scales", {}), tensors.get(prefix + ".biases", {})
        if (
            scales.get("dtype") not in {"F16", "BF16", "F32"}
            or biases.get("dtype") != scales.get("dtype")
            or biases.get("shape") != scales.get("shape")
            or scales.get("shape")
            not in [
                [shape[0], shape[1] // group]
                for group in ((group_size,) if group_size else (32, 64, 128))
                if shape[1] % group == 0
            ]
        ):
            return False
        consumed.update((name, prefix + ".scales", prefix + ".biases"))
    return consumed == set(tensors)


def resolve_flux_graph(inventory, metadata, catalog):
    """Resolve a complete saved FLUX graph; names never establish a match."""
    from oryn_mflux_provider.inspection import legacy_metadata
    from oryn_mflux_provider.swift_layout import remap_swift_tensors, swift_metadata

    if inventory.get("task") != "text-to-image":
        return None, []
    graphs = catalog.get("nativeFluxGraphs")
    if not graphs:
        return None, []
    files = {f["name"]: f for f in inventory["files"]}
    by_directory = {
        directory: sorted(
            n
            for n in files
            if PurePosixPath(n).parent.as_posix() == directory and n.endswith(".safetensors")
        )
        for directory in (*graphs["components"], "transformer")
    }
    # Complete graph matching also admits renamed unindexed native shards.
    # Indexed saves retain their explicit architecture/index resolution path.
    if not all(by_directory.values()) or any(
        f"{directory}/model.safetensors.index.json" in files for directory in by_directory
    ):
        return None, []
    names = [n for values in by_directory.values() for n in values]
    if len(names) > 24:
        return None, []
    requests = [{"file": n, "kind": "safetensors-structure"} for n in names if n not in metadata]
    if requests:
        return None, requests
    components, quantizations, groups, layouts = {}, {}, {}, set()
    for directory, names in by_directory.items():
        headers = [metadata[n] for n in names]
        swift = any("flux_swift_version" in h.get("metadata", {}) for h in headers)
        declaration = swift_metadata(headers, directory) if swift else None
        if (swift and declaration is None) or (not swift and not legacy_metadata(headers)):
            return None, []
        if swift and any(
            h["metadata"]["chunk_name"] != PurePosixPath(n).stem
            for n, h in zip(names, headers, strict=True)
        ):
            return None, []
        layouts.add("swift" if swift else "mflux")
        groups[directory] = declaration[1] if declaration else None
        raw = headers[0]["metadata"].get("quantization_level")
        if raw not in {None, "None", "null", "2", "3", "4", "5", "6", "8"}:
            return None, []
        quantizations[directory] = None if raw in {None, "None", "null"} else int(raw)
        merged = {}
        for name, header in zip(names, headers, strict=True):
            tensors = header.get("tensors")
            if (
                not isinstance(tensors, dict)
                or not tensors
                or merged.keys() & tensors.keys()
                or header.get("headerBytes", 0) + header.get("dataBytes", 0)
                != files[name]["sizeBytes"]
            ):
                return None, []
            merged.update(tensors)
        if swift:
            try:
                merged = remap_swift_tensors(merged, directory, catalog.get("swiftFluxAliases", {}))
            except ValueError:
                return None, []
        components[directory] = merged
    if len(layouts) != 1:
        return None, []
    if not all(
        matches_graph(components[d], g, quantizations[d], groups[d])
        for d, g in graphs["components"].items()
    ):
        return None, []
    matches = [
        family
        for family, graph in graphs["transformers"].items()
        if matches_graph(
            components["transformer"],
            _expanded_transformer(graph, components["transformer"], graphs["maximumBlocks"]),
            quantizations["transformer"],
            groups["transformer"],
        )
    ]
    if len(matches) != 1:
        return None, []
    return next((e for e in catalog["entries"] if e["id"] == matches[0]), None), []
