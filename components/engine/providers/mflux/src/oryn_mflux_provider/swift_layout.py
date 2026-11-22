"""Closed tensor-name binding for FLUX Swift's native MLX serialization."""

from __future__ import annotations


def build_swift_aliases(graphs):
    """Derive data-only aliases from the pinned backend's native HF mappings."""
    from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
    from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping

    from oryn_mflux_provider.structure import _expanded_transformer

    markers = {
        f"{prefix}.{i}.marker": {}
        for prefix, count in graphs["maximumBlocks"].items()
        for i in range(count)
    } | {"norm_out.linear.bias": {}}
    expected = graphs["components"] | {
        "transformer": _expanded_transformer(
            graphs["transformers"]["dev"], markers, graphs["maximumBlocks"]
        )
    }
    methods = {
        "vae": FluxWeightMapping.get_vae_mapping,
        "text_encoder": FluxWeightMapping.get_clip_encoder_mapping,
        "text_encoder_2": FluxWeightMapping.get_t5_encoder_mapping,
        "transformer": FluxWeightMapping.get_transformer_mapping,
    }
    result = {}
    for directory, getter in methods.items():
        flat = WeightMapper._build_flat_mapping(
            getter(), 24 if directory == "text_encoder_2" else 19, 4
        )
        names = {
            source: [name for name, _ in targets if name in expected[directory]]
            for source, targets in flat.items()
        }
        names = {source: targets for source, targets in names.items() if targets}
        # The Swift encoder stores one shared relative bias at its root. The
        # native backend exposes that same table through each of its 24 layers.
        if directory == "text_encoder_2":
            names["relative_attention_bias.weight"] = names.pop(
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
            )
        names.update({name: [name] for name in expected[directory]})
        result[directory] = names
    return result


def swift_metadata(headers, directory):
    """Validate complete shard identity; return the declared bits/group pair."""
    if not headers:
        return None
    declarations = [h.get("metadata", {}) for h in headers]
    first = declarations[0]
    fields = {
        "chunk_index",
        "chunk_name",
        "flux_swift_version",
        "group_size",
        "model_type",
        "quantization_date",
        "quantization_level",
        "quantization_method",
        "total_chunks",
    }
    if (
        first.get("quantization_level") not in {"4", "8"}
        or first.get("group_size") not in {"64", "128"}
        or len(headers) > 24
    ):
        return None
    for index, metadata in enumerate(declarations):
        if (
            set(metadata) != fields
            or metadata.get("flux_swift_version") != "1.0.0"
            or metadata.get("quantization_method") != "mlx-nn-quantize"
            or metadata.get("model_type") != directory
            or metadata.get("quantization_level") != first["quantization_level"]
            or metadata.get("group_size") != first["group_size"]
            or metadata.get("total_chunks") != str(len(headers))
            or metadata.get("chunk_index") != str(index)
            or not isinstance(metadata.get("chunk_name"), str)
            or not metadata["chunk_name"]
            or not isinstance(metadata.get("quantization_date"), str)
        ):
            return None
    return int(first["quantization_level"]), int(first["group_size"])


def remap_swift_tensors(tensors, directory, aliases):
    """Rename without arithmetic/transposition and retain every affine companion."""
    mapping = aliases.get(directory, {})
    result = {}
    for source, value in tensors.items():
        prefix, _, suffix = source.rpartition(".")
        key = prefix + ".weight" if suffix in {"scales", "biases"} else source
        targets = mapping.get(key)
        if not targets:
            raise ValueError("Unknown Swift tensor in complete native graph")
        for target in targets:
            if suffix in {"scales", "biases"}:
                target = target.removesuffix(".weight") + "." + suffix
            if target in result:
                raise ValueError("Colliding Swift tensor aliases")
            result[target] = value
    return result


def load_swift_component(path, loaded, data):
    """Adapt an already loaded safe component before native graph construction."""
    from mlx.utils import tree_flatten, tree_unflatten
    from safetensors import safe_open

    paths = sorted(path.glob("*.safetensors"))
    headers = []
    for file in paths:
        with safe_open(file, framework="numpy") as source:
            headers.append({"metadata": source.metadata() or {}})
    if not any("flux_swift_version" in h["metadata"] for h in headers):
        return loaded
    declaration = swift_metadata(headers, path.name)
    if (
        declaration is None
        or loaded[0] is None
        or loaded[1] != declaration[0]
        or any(h["metadata"]["chunk_name"] != p.stem for p, h in zip(paths, headers, strict=True))
    ):
        raise ValueError("Invalid Swift component serialization")
    values = remap_swift_tensors(dict(tree_flatten(loaded[0])), path.name, data["swiftFluxAliases"])
    return tree_unflatten(list(values.items())), loaded[1], loaded[2]
