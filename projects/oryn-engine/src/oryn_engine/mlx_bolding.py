"""Owned language-only Qwen3.5 PQ2/Hadamard backbone and Q8 MTP graph.

Repository Python files are identities, never executable inputs. The serialized
norm offsets, convolution layout, full vocabulary and shared MTP embedding are
explicit parts of this layout. Chat uses ordinary autoregression; retaining the
draft graph does not advertise speculative decoding.
"""

from __future__ import annotations

import copy
import hashlib
import re

from oryn_engine import mlx_prism
from oryn_engine.model_metadata import (
    MAX_JSON_BYTES,
    metadata_file,
    parse_metadata,
    preflight_safetensors,
    validate_model_config,
    validate_tokenizer_config,
)

LAYOUT = "bolding-mtp-v1"
MAX_CONTEXT, MAX_OUTPUT = 4096, 512
MTP = "language_model.mtp."
MTP_MODULES = {MTP + "fc"} | {
    MTP + "layers.0." + path
    for path in (
        "mlp.down_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "self_attn.q_proj",
        "self_attn.v_proj",
    )
}
METADATA = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "model-manifest.json",
    "tensor-map.json",
    "model.safetensors.index.json",
}
SIGN_FILE = "hadamard-signs.safetensors"


def supports(config):
    return config.get("model_type") == "qwen3_5" and (
        config.get("oryn_packed_layout") == LAYOUT
        or config.get("language_model_only") is True
        and isinstance(config.get("quantization"), dict)
        and config["quantization"].get("bits") == 2
    )


def normalized_config(config, task=None):
    if not supports(config):
        return config
    if task not in {None, "text-generation", "conversational"}:
        raise ValueError("BOLDING_TASK")
    validate_config(config)
    result = copy.deepcopy(config)
    result.pop("model_file", None)
    result["oryn_packed_layout"] = LAYOUT
    return result


def validate_config(config):
    clean = copy.deepcopy(config)
    if clean.get("oryn_packed_layout") == LAYOUT:
        if "model_file" in clean:
            raise ValueError("BOLDING_RUNTIME_BINDING")
        clean.pop("oryn_packed_layout")
    elif clean.pop("model_file", None) != "model.py":
        raise ValueError("BOLDING_RUNTIME_BINDING")
    validate_model_config(clean)
    if set(clean) != {
        "architectures",
        "language_model_only",
        "model_type",
        "quantization",
        "source_sha256",
        "text_config",
        "tie_word_embeddings",
        "transformers_version",
    } or (
        clean["model_type"] != "qwen3_5"
        or clean["architectures"] != ["Qwen3_5ForConditionalGeneration"]
        or clean["language_model_only"] is not True
        or clean["tie_word_embeddings"] is not False
        or not isinstance(clean["source_sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", clean["source_sha256"])
    ):
        raise ValueError("BOLDING_GRAPH")
    expected_q = {"bits": 2, "group_size": 128, "mode": "affine"} | {
        name: {"bits": 8, "group_size": 32} for name in MTP_MODULES
    }
    if clean["quantization"] != expected_q:
        raise ValueError("BOLDING_QUANTIZATION")
    t = clean["text_config"]
    if not isinstance(t, dict) or set(t) - mlx_prism.TEXT_FIELDS or "quantization" in t:
        raise ValueError("BOLDING_TEXT_OPTIONS")
    bounds = {
        "hidden_size": (128, 16384),
        "intermediate_size": (128, 65536),
        "num_hidden_layers": (2, 128),
        "num_attention_heads": (1, 128),
        "num_key_value_heads": (1, 128),
        "head_dim": (32, 256),
        "vocab_size": (256, 262144),
        "max_position_embeddings": (128, 1048576),
        "linear_num_key_heads": (1, 256),
        "linear_num_value_heads": (1, 256),
        "linear_key_head_dim": (32, 256),
        "linear_value_head_dim": (32, 256),
        "full_attention_interval": (2, 16),
    }
    if any(type(t.get(k)) is not int or not lo <= t[k] <= hi for k, (lo, hi) in bounds.items()):
        raise ValueError("BOLDING_GEOMETRY")
    fixed = {
        "model_type": "qwen3_5_text",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "output_gate_type": "swish",
        "hidden_act": "silu",
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
        "dtype": "bfloat16",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "use_cache": True,
        "pad_token_id": None,
    }
    if any(
        t.get(k) != v or isinstance(t.get(k), bool) != isinstance(v, bool) for k, v in fixed.items()
    ):
        raise ValueError("BOLDING_GRAPH")
    if (
        t["hidden_size"] % 128
        or t["intermediate_size"] % 128
        or t["head_dim"] % 32
        or t["linear_key_head_dim"] % 32
        or t["linear_value_head_dim"] % 32
        or t["num_attention_heads"] % t["num_key_value_heads"]
        or t["linear_num_value_heads"] % t["linear_num_key_heads"]
        or t["linear_num_value_heads"] * t["linear_value_head_dim"] % 128
        or t["num_attention_heads"] * t["head_dim"] % 128
        or t["full_attention_interval"] > t["num_hidden_layers"]
        or t.get("layer_types")
        != [
            "full_attention" if (i + 1) % t["full_attention_interval"] == 0 else "linear_attention"
            for i in range(t["num_hidden_layers"])
        ]
    ):
        raise ValueError("BOLDING_LAYER_GEOMETRY")
    r = t.get("rope_parameters")
    if (
        not isinstance(r, dict)
        or set(r)
        != {
            "mrope_interleaved",
            "mrope_section",
            "partial_rotary_factor",
            "rope_theta",
            "rope_type",
        }
        or r.get("rope_type") != "default"
        or r.get("mrope_interleaved") is not True
        or r.get("partial_rotary_factor") != 0.25
        or type(r.get("rope_theta")) not in {int, float}
        or not 10000 <= r["rope_theta"] <= 10000000
        or not isinstance(r.get("mrope_section"), list)
        or len(r["mrope_section"]) != 3
        or any(type(x) is not int or x < 1 for x in r["mrope_section"])
        or sum(r["mrope_section"]) * 8 != t["head_dim"]
        or type(t.get("bos_token_id")) is not int
        or not 0 <= t["bos_token_id"] < t["vocab_size"]
        or t.get("eos_token_id") != t["bos_token_id"]
    ):
        raise ValueError("BOLDING_POSITION_OR_STOP")
    return config


def parameter_specs(config):
    """Unpacked shapes and norm offsets, independently usable in metadata audits."""
    validate_config(config)
    t = config["text_config"]
    h, d, m, v = (t[k] for k in ("hidden_size", "head_dim", "intermediate_size", "vocab_size"))
    q, kv = t["num_attention_heads"] * d, t["num_key_value_heads"] * d
    key = t["linear_num_key_heads"] * t["linear_key_head_dim"]
    val = t["linear_num_value_heads"] * t["linear_value_head_dim"]
    specs = {
        "language_model.model.embed_tokens.weight": (v, h),
        "language_model.lm_head.weight": (v, h),
    }
    norms = set()

    def norm(name, width):
        specs[name + ".weight"] = (width,)
        norms.add(name + ".weight")

    def layer(prefix, kind):
        norm(prefix + "input_layernorm", h)
        norm(prefix + "post_attention_layernorm", h)
        for name, shape in {"gate": (m, h), "up": (m, h), "down": (h, m)}.items():
            specs[prefix + "mlp." + name + "_proj.weight"] = shape
        if kind == "full_attention":
            for name, shape in {"q": (2 * q, h), "k": (kv, h), "v": (kv, h), "o": (h, q)}.items():
                specs[prefix + "self_attn." + name + "_proj.weight"] = shape
            norm(prefix + "self_attn.q_norm", d)
            norm(prefix + "self_attn.k_norm", d)
        else:
            p = prefix + "linear_attn."
            specs.update(
                {
                    p + k: shape
                    for k, shape in {
                        "A_log": (t["linear_num_value_heads"],),
                        "dt_bias": (t["linear_num_value_heads"],),
                        "in_proj_a.weight": (t["linear_num_value_heads"], h),
                        "in_proj_b.weight": (t["linear_num_value_heads"], h),
                        "in_proj_qkv.weight": (2 * key + val, h),
                        "in_proj_z.weight": (val, h),
                        "out_proj.weight": (h, val),
                        "conv1d.weight": (2 * key + val, 1, 4),
                        "norm.weight": (t["linear_value_head_dim"],),
                    }.items()
                }
            )

    for i, kind in enumerate(t["layer_types"]):
        layer(f"language_model.model.layers.{i}.", kind)
    norm("language_model.model.norm", h)
    layer(MTP + "layers.0.", "full_attention")
    specs[MTP + "fc.weight"] = (h, 2 * h)
    for name in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden"):
        norm(MTP + name, h)
    return specs, norms


def packed_specs(config):
    specs, _ = parameter_specs(config)
    return {
        n.removesuffix(".weight"): (
            shape,
            8 if n.startswith(MTP) else 2,
            32 if n.startswith(MTP) else 128,
        )
        for n, shape in specs.items()
        if len(shape) == 2 and not n.endswith((".in_proj_a.weight", ".in_proj_b.weight"))
    }


def serialized_shapes(config):
    shapes, _ = parameter_specs(config)
    for name, ((rows, cols), bits, group) in packed_specs(config).items():
        shapes[name + ".weight"] = (rows, cols * bits // 32)
        shapes[name + ".scales"] = shapes[name + ".biases"] = (rows, cols // group)
    return shapes


def selected_files(files, config):
    validate_config(config)
    weights = {n for n in files if n.endswith(".safetensors")}
    shards = weights - {SIGN_FILE}
    if not set(files) >= METADATA or SIGN_FILE not in weights or not shards or len(shards) > 64:
        raise ValueError("BOLDING_FILES")
    if any(not re.fullmatch(r"model-\d{5}\.safetensors", n) for n in shards):
        raise ValueError("BOLDING_SHARDS")
    if shards != {f"model-{i:05d}.safetensors" for i in range(1, len(shards) + 1)}:
        raise ValueError("BOLDING_SHARD_ORDER")
    return tuple(sorted(METADATA | weights))


def expected_transformations(name, norms, packed):
    if name in norms:
        return ["subtract_gguf_norm_offset"]
    result = []
    if name.endswith(".conv1d.weight"):
        result.append("restore_mlx_conv1d_unsanitized_layout")
    if ".linear_attn." in name and not name.endswith((".norm.weight", ".out_proj.weight")):
        result.append("restore_gguf_tiled_gdn_v_to_grouped")
    if name.endswith(".A_log"):
        result.append("log_of_negated_source_A")
    module = name.removesuffix(".weight")
    if module in packed and packed[module][1] == 2:
        result.append(
            "apply_inverse_normalized_fwht_to_embedding_rows"
            if module.endswith(".embed_tokens")
            else "apply_normalized_fwht_to_input_activation"
        )
    return result


def validate_manifest(config, manifest, tensor_map, index):
    for value in (manifest, tensor_map, index):
        validate_model_config(value)
    shapes, norms = parameter_specs(config)
    packed = packed_specs(config)
    transformed = {n: shape for n, (shape, bits, _) in packed.items() if bits == 2}
    h = manifest.get("hadamard", {})
    if any(
        not isinstance(value, dict)
        for value in (
            h,
            manifest.get("source"),
            tensor_map.get("hadamard"),
            tensor_map.get("vocabulary"),
        )
    ):
        raise ValueError("BOLDING_MANIFEST")
    block = h.get("block_size")
    widths = sorted({shape[1] for shape in transformed.values()})
    if (
        manifest.get("schema") != "omlx-bolding-mtp-model-manifest/v1"
        or manifest.get("source", {}).get("sha256") != config["source_sha256"]
        or h.get("transform") != "normalized-sylvester-walsh-hadamard"
        or h.get("axis") != "input-last-dimension"
        or h.get("sign_mode") != "explicit"
        or h.get("gdn_v_grouped") is not True
        or h.get("version") != 1
        or type(block) is not int
        or block not in {128, 256, 512, 1024, 2048, 4096}
        or any(width % block for width in widths)
        or h.get("sign_widths") != widths
        or h.get("sign_file") != SIGN_FILE
        or h.get("folded_module_count") != len(transformed) - 1
        or h.get("inverse_embedding_count") != 1
    ):
        raise ValueError("BOLDING_HADAMARD")
    hashes = h.get("sign_vector_sha256")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != {str(w) for w in widths}
        or any(
            not isinstance(s, str) or not re.fullmatch(r"[0-9a-f]{64}", s) for s in hashes.values()
        )
    ):
        raise ValueError("BOLDING_SIGN_HASHES")
    records = manifest.get("quantized_modules")
    if not isinstance(records, list) or len(records) != len(transformed):
        raise ValueError("BOLDING_TRANSFORMS")
    seen = set()
    for record in records:
        name = record.get("runtime_module_path") if isinstance(record, dict) else None
        if name not in transformed or name in seen:
            raise ValueError("BOLDING_TRANSFORMS")
        seen.add(name)
        expected_role = (
            "inverse_embedding_rows"
            if name.endswith(".embed_tokens")
            else "folded_weight_input_transform"
        )
        if any(
            record.get(k) != v
            for k, v in {
                "hadamard_role": expected_role,
                "quantization": {"bits": 2, "group_size": 128},
                "runtime_parameter_path": name + ".weight",
                "sign_width": transformed[name][1],
                "target_shape": list(transformed[name]),
            }.items()
        ):
            raise ValueError("BOLDING_TRANSFORMS")
    if (
        tensor_map.get("schema") != "omlx-bolding-mtp-tensor-map/v1"
        or tensor_map.get("source_sha256") != config["source_sha256"]
        or tensor_map.get("quantization_config") != config["quantization"]
        or tensor_map.get("base_layer_count") != config["text_config"]["num_hidden_layers"]
        or tensor_map.get("mtp_layer_count") != 1
        or any(
            tensor_map.get("hadamard", {}).get(key) != h[key]
            for key in (
                "axis",
                "block_size",
                "gdn_v_grouped",
                "sign_mode",
                "sign_vector_sha256",
                "sign_widths",
                "transform",
                "version",
            )
        )
    ):
        raise ValueError("BOLDING_TENSOR_MAP")
    mappings = tensor_map.get("mappings")
    if not isinstance(mappings, list) or len(mappings) != len(shapes):
        raise ValueError("BOLDING_TENSOR_MAP")
    seen = set()
    for record in mappings:
        n = record.get("runtime_parameter_path") if isinstance(record, dict) else None
        if n not in shapes or n in seen or record.get("target_shape") != list(shapes[n]):
            raise ValueError("BOLDING_TENSOR_MAP")
        seen.add(n)
        transformations = record.get("transformations")
        if not isinstance(transformations, list) or any(
            not isinstance(s, str) for s in transformations
        ):
            raise ValueError("BOLDING_TENSOR_MAP")
        if ("subtract_gguf_norm_offset" in transformations) != (n in norms):
            raise ValueError("BOLDING_NORM_OFFSET")
        if ("restore_mlx_conv1d_unsanitized_layout" in transformations) != n.endswith(
            ".conv1d.weight"
        ):
            raise ValueError("BOLDING_CONV_LAYOUT")
        if transformations != expected_transformations(n, norms, packed):
            raise ValueError("BOLDING_UNREVIEWED_TRANSFORM")
    weight_map = index.get("weight_map")
    if (
        not isinstance(weight_map, dict)
        or set(weight_map) != set(serialized_shapes(config))
        or any(
            not isinstance(n, str) or not re.fullmatch(r"model-\d{5}\.safetensors", n)
            for n in weight_map.values()
        )
    ):
        raise ValueError("BOLDING_INDEX")
    vocab = tensor_map.get("vocabulary", {})
    interval = vocab.get("unused_token_id_range")
    size = config["text_config"]["vocab_size"]
    if (
        vocab.get("model_vocab_size") != size
        or vocab.get("source_vocab_size") != size
        or not isinstance(interval, list)
        or len(interval) != 2
        or any(type(i) is not int for i in interval)
        or not 256 <= interval[0] < interval[1] == size
        or config["text_config"]["bos_token_id"] >= interval[0]
        or not 1 <= vocab.get("unused_token_count", 0) == size - interval[0] <= 512
    ):
        raise ValueError("BOLDING_VOCABULARY")
    return h, tuple(interval)


def validate_tokenizer_metadata(value):
    clean = copy.deepcopy(value)
    if clean.pop("auto_map", None) != {
        "AutoTokenizer": [None, "tokenization_bolding_mtp.BoldingMtpTokenizerFast"]
    }:
        raise ValueError("BOLDING_TOKENIZER_BINDING")
    if clean.pop("tokenizer_class", None) != "BoldingMtpTokenizerFast":
        raise ValueError("BOLDING_TOKENIZER_BINDING")
    validate_tokenizer_config(clean)
    mlx_prism.validate_tokenizer_metadata(clean | {"tokenizer_class": "TokenizersBackend"})


def validate_headers(config, index, headers):
    expected = serialized_shapes(config)
    packed = packed_specs(config)
    actual = {}
    for filename, header in headers.items():
        if not isinstance(header, dict) or not isinstance(header.get("tensors"), dict):
            raise ValueError("BOLDING_HEADER")
        for name, record in header["tensors"].items():
            if name in actual or index["weight_map"].get(name) != filename:
                raise ValueError("BOLDING_INDEX_BINDING")
            shape = tuple(record.get("shape", []))
            dtype = record.get("dtype")
            packed_weight = name.endswith(".weight") and name.removesuffix(".weight") in packed
            if dtype not in ({"U32"} if packed_weight else {"F16", "BF16", "F32"}):
                raise ValueError("BOLDING_DTYPE")
            actual[name] = shape
    if actual != expected:
        raise ValueError("BOLDING_WEIGHTS_INCOMPLETE")


def inspect_metadata(config, read_json, read_header, files):
    selected = selected_files(files, config)
    manifest, tensor_map, index = [
        read_json(n)
        for n in ("model-manifest.json", "tensor-map.json", "model.safetensors.index.json")
    ]
    h, interval = validate_manifest(config, manifest, tensor_map, index)
    validate_tokenizer_metadata(read_json("tokenizer_config.json"))
    generation = read_json("generation_config.json")
    mlx_prism.validate_generation_metadata(config, generation)
    if any(n >= interval[0] for n in generation["eos_token_id"]):
        raise ValueError("BOLDING_STOP_TOKENS")
    shards = set(index["weight_map"].values())
    if shards != set(selected) - METADATA - {SIGN_FILE}:
        raise ValueError("BOLDING_SHARD_BINDING")
    validate_headers(config, index, {name: read_header(name) for name in sorted(shards)})
    signs = read_header(SIGN_FILE).get("tensors", {})
    if {n: (r.get("shape"), r.get("dtype")) for n, r in signs.items()} != {
        f"sign_{w}": ([w], "F32") for w in h["sign_widths"]
    }:
        raise ValueError("BOLDING_SIGN_LAYOUT")


def native_model(config):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5.config import TextConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import Qwen3_5MTPConfig
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp import Qwen3_5MTPDraftModel

    validate_config(config)

    class TextLanguageModel(LanguageModel):
        def get_rope_index(
            self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None
        ):
            # The native VLM helper reads vision_config even for text. This
            # language-only layout has no vision branch or padded media input.
            if (
                image_grid_thw is not None
                or video_grid_thw is not None
                or attention_mask is not None
            ):
                raise ValueError("BOLDING_TEXT_INPUT")
            positions = mx.broadcast_to(
                mx.arange(input_ids.shape[1], dtype=mx.int32)[None, :], input_ids.shape
            )
            return positions, mx.zeros((input_ids.shape[0], 1), dtype=mx.int32)

    model = nn.Module()
    model.language_model = TextLanguageModel(
        TextConfig.from_dict(copy.deepcopy(config["text_config"]))
    )
    model.language_model.mtp = Qwen3_5MTPDraftModel(
        Qwen3_5MTPConfig(text_config=copy.deepcopy(config["text_config"]))
    )
    return model


def packed_class():
    import mlx.core as mx

    class Packed(mlx_prism.packed_class()):
        def __init__(self, *args, bits, group, **kwargs):
            super().__init__(*args, **kwargs)
            self.bits, self.group = bits, group

        def __call__(self, x):
            if self.embedding:
                ids = x.reshape(-1)
                out = mx.dequantize(
                    self.weight[ids],
                    self.scales[ids],
                    self.biases[ids],
                    group_size=self.group,
                    bits=self.bits,
                ).reshape(*x.shape, -1)
                return self.transform(out, inverse=True)
            return mx.quantized_matmul(
                self.transform(x),
                self.weight,
                self.scales,
                self.biases,
                transpose=True,
                group_size=self.group,
                bits=self.bits,
            )

    return Packed


def load_model(root, config):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    from oryn_engine.mlx_vision import verify_vision_build

    verify_vision_build()
    files = selected_files(
        [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()], config
    )
    manifest, tensor_map, index = [
        metadata_file(root / n)
        for n in ("model-manifest.json", "tensor-map.json", "model.safetensors.index.json")
    ]
    h, _ = validate_manifest(config, manifest, tensor_map, index)
    actual = preflight_safetensors(root, companion_files=frozenset({SIGN_FILE}))
    expected = serialized_shapes(config) | {f"sign_{w}": (w,) for w in h["sign_widths"]}
    if actual != expected or set(index["weight_map"].values()) != set(files) - METADATA - {
        SIGN_FILE
    }:
        raise ValueError("BOLDING_WEIGHTS_INCOMPLETE")
    packed = packed_specs(config)
    weights, signs = {}, {}
    for filename in files:
        if not filename.endswith(".safetensors"):
            continue
        for name, tensor in mx.load(str(root / filename)).items():
            if filename == SIGN_FILE:
                if (
                    name not in {f"sign_{w}" for w in h["sign_widths"]}
                    or tensor.dtype != mx.float32
                ):
                    raise ValueError("BOLDING_SIGN_LAYOUT")
                if not bool(mx.all((tensor == 1) | (tensor == -1)).item()):
                    raise ValueError("BOLDING_SIGNS")
                raw = memoryview(tensor).tobytes()
                if hashlib.sha256(raw).hexdigest() != h["sign_vector_sha256"][str(tensor.size)]:
                    raise ValueError("BOLDING_SIGN_BINDING")
                signs[tensor.size] = tensor
                continue
            if name in weights or index["weight_map"].get(name) != filename:
                raise ValueError("BOLDING_INDEX_BINDING")
            is_packed = name.endswith(".weight") and name.removesuffix(".weight") in packed
            if is_packed:
                if tensor.dtype != mx.uint32:
                    raise ValueError("BOLDING_PACKED_DTYPE")
            elif tensor.dtype not in {mx.float16, mx.float32, mx.bfloat16} or not bool(
                mx.all(mx.isfinite(tensor)).item()
            ):
                raise ValueError("BOLDING_NONFINITE_WEIGHT")
            weights[name] = tensor
    if set(weights) != set(serialized_shapes(config)) or set(signs) != set(h["sign_widths"]):
        raise ValueError("BOLDING_WEIGHTS_INCOMPLETE")
    specs, norms = parameter_specs(config)
    for name in weights:
        if name in norms:
            weights[name] = weights[name].astype(mx.float32) + 1
        elif name.endswith(".conv1d.weight"):
            weights[name] = mx.moveaxis(weights[name], 2, 1)
    model = native_model(config)
    native_shapes = {n: tuple(v.shape) for n, v in tree_flatten(model.parameters())}
    native_expected = {
        n: (s[0], s[2], s[1]) if n.endswith(".conv1d.weight") else s for n, s in specs.items()
    }
    if native_shapes != native_expected:
        raise ValueError("BOLDING_NATIVE_GRAPH_CHANGED")
    Packed = packed_class()
    replacements = []
    for name, ((_, width), bits, group) in packed.items():
        module = Packed(
            *(weights[name + "." + suffix] for suffix in ("weight", "scales", "biases")),
            bits=bits,
            group=group,
            embedding=name.endswith(".embed_tokens"),
            block=h["block_size"] if bits == 2 else 0,
            signs=signs[width] if bits == 2 else None,
        )
        if bits == 2:
            weights[name + ".signs"] = signs[width]
        replacements.append((name, module))
    model.update_modules(tree_unflatten(replacements))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def load_tokenizer(root, config):
    from numbers import Integral

    from tokenizers import Tokenizer
    from transformers import TokenizersBackend

    validate_config(config)
    # This format declares a known repository tokenizer as an identity only.
    # Validate that exact declaration and every remaining option before using
    # Oryn's concrete tokenizer below; never give the declaration to a loader.
    metadata_path = root / "tokenizer_config.json"
    if (
        metadata_path.is_symlink()
        or not metadata_path.is_file()
        or metadata_path.stat().st_size > MAX_JSON_BYTES
    ):
        raise ValueError("BOLDING_TOKENIZER_METADATA")
    with metadata_path.open("rb") as stream:
        validate_tokenizer_metadata(
            parse_metadata(stream.read(MAX_JSON_BYTES + 1), redundant_keys=True)
        )
    _, interval = validate_manifest(
        config,
        *(
            metadata_file(root / n)
            for n in ("model-manifest.json", "tensor-map.json", "model.safetensors.index.json")
        ),
    )
    path = root / "tokenizer.json"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 32 * 1024**2:
        raise ValueError("BOLDING_TOKENIZER_GRAPH")
    vocab = Tokenizer.from_file(str(path)).get_vocab()
    if (
        len(vocab) != config["text_config"]["vocab_size"]
        or set(vocab.values()) != set(range(len(vocab)))
        or any(len(t.encode()) > 4096 for t in vocab)
    ):
        raise ValueError("BOLDING_VOCABULARY")

    class OwnedTokenizer(TokenizersBackend):
        def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
            if isinstance(ids, Integral):
                return (
                    ""
                    if interval[0] <= int(ids) < interval[1]
                    else super().convert_ids_to_tokens(
                        int(ids), skip_special_tokens=skip_special_tokens
                    )
                )
            return [
                self.convert_ids_to_tokens(i, skip_special_tokens=skip_special_tokens) for i in ids
            ]

        def decode(self, token_ids, **kwargs):
            if isinstance(token_ids, Integral):
                token_ids = [token_ids]
            elif hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            return super().decode(
                [int(i) for i in token_ids if not interval[0] <= int(i) < interval[1]], **kwargs
            )

    tok = OwnedTokenizer(
        tokenizer_file=str(path), eos_token=mlx_prism.MARKERS[1], pad_token=mlx_prism.MARKERS[2]
    )
    tok.chat_template = None
    if tok.get_vocab() != vocab or any(
        tok.encode(t, add_special_tokens=False) != [vocab.get(t)] for t in mlx_prism.MARKERS
    ):
        raise ValueError("BOLDING_TOKENIZER_CONTROLS")
    generation = metadata_file(root / "generation_config.json")
    mlx_prism.validate_generation_metadata(config, generation)
    if (
        generation["eos_token_id"] != [vocab[mlx_prism.MARKERS[1]], vocab[mlx_prism.MARKERS[2]]]
        or generation["bos_token_id"] != vocab[mlx_prism.MARKERS[2]]
        or any(n >= interval[0] for n in tok.all_special_ids)
    ):
        raise ValueError("BOLDING_STOP_TOKENS")
    return tok, set(generation["eos_token_id"])


def generate(root, config, value, event):
    validate_config(config)
    return mlx_prism.generate_with_loader(
        root, config, value, event, load_model=load_model, load_tokenizer=load_tokenizer
    )
