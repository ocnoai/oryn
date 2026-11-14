"""Bounded data-only metadata shared by inspection, offline audits and workers."""

from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path
from typing import Any

MAX_JSON_BYTES = 1024 * 1024
# The pinned tokenizer library serializes these exact integers as "no declared
# maximum". They remain inert; every task supplies its own bounded token budget.
TOKENIZER_LENGTH_SENTINELS = {int(1e20), int(1e30)}
FLOAT32_MAX = 3.4028234663852886e38


def root_config_name(names) -> str:
    """Use the same authoritative root metadata in discovery, replay and execution."""
    return next(
        (
            name
            for name in (
                "model_index.json",
                "modular_model_index.json",
                "config.json",
                "open_clip_config.json",
            )
            if name in names
        ),
        "config.json",
    )


def validate_metadata(value: object) -> None:
    # This native arithmetic clip exceeds int64 but is finite FP32 data, never
    # an allocation bound. Admit it only at the root of a fully closed TimesFM
    # declaration; other numbers and nested uses keep the original bound.
    timesfm_clip = False
    if isinstance(value, dict) and value.get("value_clip") == 1e20:
        from oryn_engine.timesfm3_tasks import native_config

        try:
            native_config(value)
            timesfm_clip = True
        except (ValueError, TypeError, KeyError):
            pass
    # Native Granite exports retain the Mamba arithmetic clip even in an
    # attention-only graph. This exact finite sentinel is inert there; it must
    # not widen allocation dimensions, active state-space options or nested data.
    granite_clip = None
    if (
        isinstance(value, dict)
        and value.get("model_type") == "granitemoehybrid"
        and value.get("architectures") == ["GraniteMoeHybridForCausalLM"]
        and type(value.get("num_hidden_layers")) is int
        and 1 <= value["num_hidden_layers"] <= 80
        and value.get("layer_types") == ["attention"] * value["num_hidden_layers"]
    ):
        clip = value.get("time_step_limit")
        if (
            isinstance(clip, list)
            and len(clip) == 2
            and type(clip[0]) in {int, float}
            and clip[0] == 0
            and type(clip[1]) is float
            and clip[1] == FLOAT32_MAX
        ):
            granite_clip = clip
    stack = [(value, 0, None)]
    nodes = 0
    while stack:
        item, depth, field = stack.pop()
        nodes += 1
        if nodes > 100_000 or depth > 32:
            raise ValueError("Metadata exceeds structural limits")
        if item is None or isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            if timesfm_clip and depth == 1 and field == "value_clip":
                continue
            if (
                field == "model_max_length"
                and type(item) is int
                and item in TOKENIZER_LENGTH_SENTINELS
            ):
                continue
            if not math.isfinite(item) or abs(item) > 2**63 - 1:
                raise ValueError("Metadata number is invalid")
        elif isinstance(item, str):
            if len(item) > 65_536:
                raise ValueError("Metadata string exceeds limits")
        elif isinstance(item, list):
            if item is granite_clip and depth == 1 and field == "time_step_limit":
                continue
            if len(item) > 50_000:
                raise ValueError("Metadata array exceeds limits")
            stack.extend((child, depth + 1, None) for child in item)
        elif isinstance(item, dict):
            if len(item) > 10_000 or any(not isinstance(k, str) or len(k) > 256 for k in item):
                raise ValueError("Metadata object is invalid")
            stack.extend((child, depth + 1, key) for key, child in item.items())
        else:
            raise ValueError("Metadata is not JSON data")


def parse_metadata(
    raw: bytes, *, limit: int = MAX_JSON_BYTES, redundant_keys: bool = False
) -> dict[str, Any]:
    if len(raw) > limit:
        raise ValueError("Metadata exceeds byte limit")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            # Some native tokenizer configs repeat the same inert setting.
            # Permit only identical JSON values, including their scalar types;
            # first-wins and last-wins parsers must produce the same result.
            if key in result and (
                not redundant_keys
                or json.dumps(result[key], sort_keys=True) != json.dumps(value, sort_keys=True)
            ):
                raise ValueError("Duplicate metadata key")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("Metadata must be an object")
    validate_metadata(value)
    return value


def metadata_file(path: Path, *, limit: int = MAX_JSON_BYTES) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Metadata file is unavailable or oversized")
    with path.open("rb") as stream:
        value = parse_metadata(
            stream.read(limit + 1), limit=limit, redundant_keys=path.name == "tokenizer_config.json"
        )
    if path.name == "tokenizer_config.json":
        validate_tokenizer_config(value)
    return value


def validate_tokenizer_config(config: dict[str, Any]) -> None:
    """Tokenizer constructor metadata is literal data, never file authority."""
    validate_metadata(config)
    stack = [config]
    while stack:
        value = stack.pop()
        if isinstance(value, list):
            stack.extend(value)
            continue
        if not isinstance(value, dict):
            continue
        if (
            value.get("auto_map")
            or value.get("custom_pipelines")
            or value.get("trust_remote_code") is True
        ):
            raise ValueError("Repository code is prohibited")
        for key, item in value.items():
            if key == "init_inputs" and item not in (None, []):
                raise ValueError("Tokenizer positional constructor inputs are prohibited")
            if (
                key.endswith(("_file", "_files"))
                or key in {"tokenizer_object", "backend_tokenizer"}
            ) and item is not None:
                raise ValueError("Tokenizer file and backend overrides are prohibited")
            if key == "vocab" and item is not None:
                mapping = isinstance(item, dict) and all(
                    isinstance(token, str) and type(index) is int and 0 <= index < 1_000_000
                    for token, index in item.items()
                )
                sequence = isinstance(item, list) and all(
                    isinstance(token, str)
                    or (
                        isinstance(token, list)
                        and len(token) == 2
                        and isinstance(token[0], str)
                        and type(token[1]) in {int, float}
                    )
                    for token in item
                )
                if not (mapping or sequence):
                    raise ValueError("Tokenizer vocabulary must contain literal data")
                continue  # Token strings may themselves resemble metadata key names.
            if key == "merges" and item is not None:
                if not isinstance(item, list) or not all(
                    isinstance(merge, str)
                    or (
                        isinstance(merge, list)
                        and len(merge) == 2
                        and all(isinstance(part, str) for part in merge)
                    )
                    for merge in item
                ):
                    raise ValueError("Tokenizer merges must contain literal data")
                continue
            if key in {
                "added_tokens_decoder",
                "extra_special_tokens",
                "additional_special_tokens",
                "chat_template",
            }:
                continue  # These already-bounded token/template values are not loader options.
            stack.append(item)


def validate_inspection_tokenizer_metadata(config: dict[str, Any]) -> None:
    """Admit closed native declarations as data before family-specific inspection.

    Execution readers still validate their own bindings; generic tokenizer
    constructors continue to use validate_tokenizer_config directly.
    """
    if config.get("auto_map"):
        from oryn_engine import mlx_bolding, molparser_tasks, videollama3_tasks

        if config.get("tokenizer_class") == "BoldingMtpTokenizerFast":
            mlx_bolding.validate_tokenizer_metadata(config)
            return
        for family in (molparser_tasks, videollama3_tasks):
            expected = family.template("tokenizer_config.json")
            if config.get("auto_map") == expected["auto_map"]:
                family.validate_metadata(
                    "tokenizer_config.json", config, family.template("config.json")
                )
                return
    validate_tokenizer_config(config)


def validate_model_config(config: dict[str, Any]) -> None:
    validate_metadata(config)
    for key in ("model_type", "_class_name"):
        if key in config and (not isinstance(config[key], str) or not 1 <= len(config[key]) <= 128):
            raise ValueError("Model class identity is invalid")
    if "architectures" in config and (
        not isinstance(config["architectures"], list)
        or not 1 <= len(config["architectures"]) <= 32
        or any(not isinstance(v, str) or not 1 <= len(v) <= 128 for v in config["architectures"])
    ):
        raise ValueError("Model architecture identities are invalid")
    stack = [config]
    dimensions = {
        "hidden_size": 65_536,
        "intermediate_size": 262_144,
        "num_hidden_layers": 512,
        "num_attention_heads": 512,
        "num_key_value_heads": 512,
        "vocab_size": 1_000_000,
        "num_local_experts": 1024,
        "num_experts": 1024,
    }
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if (
                value.get("auto_map")
                or value.get("custom_pipelines")
                or value.get("model_file") is not None
                or value.get("trust_remote_code") is True
                or any(
                    key.startswith(("_attn_implementation", "_experts_implementation"))
                    and item is not None
                    for key, item in value.items()
                )
            ):
                raise ValueError("Repository code is prohibited")
            for key, maximum in dimensions.items():
                if key not in value:
                    continue
                raw = value[key]
                # Native ViT has no vocabulary. Older test exports serialized an
                # empty placeholder; only that inert empty object is permitted.
                if key == "vocab_size" and value.get("model_type") == "vit" and raw == {}:
                    continue
                # Granite uses zero experts to select its native dense MLP.
                # Require both expert counts to be explicit integer zeros.
                if (
                    key == "num_local_experts"
                    and type(raw) is int
                    and raw == 0
                    and value.get("model_type") == "granitemoehybrid"
                    and type(value.get("num_experts_per_tok")) is int
                    and value["num_experts_per_tok"] == 0
                ):
                    continue
                # Native dense Gemma 4 explicitly serializes its unused expert
                # count as null. MoE configurations still require a bounded count.
                if (
                    key == "num_experts"
                    and raw is None
                    and value.get("model_type") in {"gemma4_text", "gemma4_unified_text"}
                    and value.get("enable_moe_block") is False
                ):
                    continue
                # Diffusers UNet2DConditionModel derives its effective head count from
                # attention_head_dim when num_attention_heads is null (legacy naming).
                # Validate that effective count with the same bound, never skip the gate.
                if (
                    key == "num_attention_heads"
                    and raw is None
                    and value.get("_class_name") == "UNet2DConditionModel"
                ):
                    raw = value.get("attention_head_dim")
                axes = raw if isinstance(raw, list) and key == "num_attention_heads" else [raw]
                if not 1 <= len(axes) <= 32 or any(
                    type(axis) is not int or not 1 <= axis <= maximum for axis in axes
                ):
                    raise ValueError("Model dimensions exceed the admitted bound")
            for key in ("image_size", "size", "crop_size"):
                candidate = value.get(key)
                if candidate is not None and isinstance(candidate, (int, list, dict)):
                    axes = (
                        list(candidate.values())
                        if isinstance(candidate, dict)
                        else candidate
                        if isinstance(candidate, list)
                        else [candidate]
                    )
                    if any(type(axis) is not int or not 1 <= axis <= 2048 for axis in axes):
                        raise ValueError("Image preprocessing dimensions exceed the admitted bound")
            for key in ("max_patches", "num_patches", "image_seq_length"):
                if key in value and (type(value[key]) is not int or not 1 <= value[key] <= 4096):
                    raise ValueError("Image patch count exceeds the admitted bound")
            if "max_pixels" in value and (
                type(value["max_pixels"]) is not int or not 1 <= value["max_pixels"] <= 1024**2
            ):
                raise ValueError("Image pixel budget exceeds the admitted bound")
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)


def preflight_safetensors(
    root: Path,
    *,
    companion_files: frozenset[str] = frozenset(),
    weight_names: tuple[str, ...] | None = None,
) -> dict[str, tuple[int, ...]]:
    """Read only bounded headers before a native loader allocates weight/model tensors."""
    tensors = {}
    tensor_files: dict[str, str] = {}
    total = 0
    header_bytes = 0
    if weight_names is not None and (
        not weight_names
        or len(set(weight_names)) != len(weight_names)
        or any(not re.fullmatch(r"[A-Za-z0-9_-]+\.safetensors", name) for name in weight_names)
    ):
        raise ValueError("Selected weight names are invalid")
    paths = (
        sorted(root / name for name in weight_names)
        if weight_names is not None
        else sorted(root.glob("*.safetensors"))
    )
    if not 1 <= len(paths) <= 256:
        raise ValueError("Selected weight set is invalid")
    widths = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "I16": 2,
        "U16": 2,
        "I32": 4,
        "U32": 4,
        "I64": 8,
        "U64": 8,
        "F16": 2,
        "BF16": 2,
        "F32": 4,
        "F64": 8,
    }
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("Unsafe selected weight path")
        size = path.stat().st_size
        total += size
        if size < 8 or total > 128 * 1024**3:
            raise ValueError("Selected weights exceed size bound")
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            header_bytes += length
            if not 2 <= length <= 8 * 1024**2 or length + 8 > size or header_bytes > 32 * 1024**2:
                raise ValueError("Tensor header exceeds bounds")
            header = parse_metadata(stream.read(length), limit=8 * 1024**2)
        spans = []
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,256}", name) or name in tensors:
                raise ValueError("Tensor name is duplicate or invalid")
            if not isinstance(tensor, dict) or set(tensor) != {"dtype", "shape", "data_offsets"}:
                raise ValueError("Tensor declaration is invalid")
            shape, offsets, dtype = tensor["shape"], tensor["data_offsets"], tensor["dtype"]
            if (
                not isinstance(shape, list)
                or len(shape) > 6
                or any(type(n) is not int or not 1 <= n <= 1_000_000 for n in shape)
            ):
                raise ValueError("Tensor dimensions are invalid")
            if not isinstance(dtype, str) or dtype not in widths:
                raise ValueError("Tensor dtype is not admitted")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(n) is not int for n in offsets)
            ):
                raise ValueError("Tensor offsets are invalid")
            start, end = offsets
            if (
                not 0 <= start < end <= size - 8 - length
                or end - start != math.prod(shape) * widths[dtype]
            ):
                raise ValueError("Tensor size does not match its data range")
            spans.append((start, end))
            tensors[name] = tuple(shape)
            tensor_files[name] = path.name
            if len(tensors) > 100_000:
                raise ValueError("Too many tensors")
        previous = 0
        for start, end in sorted(spans):
            if start != previous:
                raise ValueError("Tensor data ranges overlap or contain gaps")
            previous = end
        if previous != size - 8 - length:
            raise ValueError("Unselected trailing tensor data")
    # Native checkpoint loaders consume these indexes again. Validate the exact mapping
    # before they see it, including shard identity, so an index cannot redirect a loader
    # to an unrelated file or trigger repeated reads through an inconsistent weight map.
    from oryn_engine.model_layouts import validate_weight_index

    indexes = sorted(root.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise ValueError("Ambiguous selected weight indexes")
    for path in indexes:
        index = metadata_file(path, limit=8 * 1024**2)
        # A build-owned adapter can separately bind a complete companion head.
        # Its tensors still pass every header/range/duplicate check above, while
        # the backbone index must cover exactly the remaining tensor graph.
        validate_weight_index(index, {weight.name for weight in paths} - companion_files)
        indexed_tensors = {
            name: filename
            for name, filename in tensor_files.items()
            if filename not in companion_files
        }
        if index["weight_map"] != indexed_tensors:
            raise ValueError("Weight index disagrees with selected tensor headers")
    return tensors
