"""Owned packed Qwen3.5 graphs with optional explicit Hadamard transforms.

The original packed matrices and every floating vision tensor are mandatory.
Repository runtime modules are metadata identities only and are never imported.
"""

from __future__ import annotations

import copy
import re

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

FAMILY = "prism_hadamard_qwen35"
MAX_CONTEXT = 4096
MAX_OUTPUT = 512
MARKERS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
TEXT_FIELDS = set(
    [
        "attention_bias",
        "attention_dropout",
        "attn_output_gate",
        "bos_token_id",
        "dtype",
        "eos_token_id",
        "full_attention_interval",
        "head_dim",
        "hidden_act",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "layer_types",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "mamba_ssm_dtype",
        "max_position_embeddings",
        "model_type",
        "mtp_num_hidden_layers",
        "mtp_use_dedicated_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "output_gate_type",
        "pad_token_id",
        "partial_rotary_factor",
        "quantization",
        "rms_norm_eps",
        "rope_parameters",
        "tie_word_embeddings",
        "use_cache",
        "vocab_size",
    ]
)
VISION_FIELDS = set(
    [
        "deepstack_visual_indexes",
        "depth",
        "hidden_act",
        "hidden_size",
        "in_channels",
        "initializer_range",
        "intermediate_size",
        "model_type",
        "num_heads",
        "num_position_embeddings",
        "out_hidden_size",
        "patch_size",
        "spatial_merge_size",
        "temporal_patch_size",
    ]
)
ROOT_FIELDS = set(
    [
        "architectures",
        "eos_token_id",
        "image_token_id",
        "language_model_only",
        "model_type",
        "quantization",
        "text_config",
        "tie_word_embeddings",
        "transformers_version",
        "video_token_id",
        "vision_config",
        "vision_end_token_id",
        "vision_start_token_id",
        "base_model_type",
        "components",
        "gdn_activation_layout",
        "hadamard_config",
        "modules",
        "requires_runtime",
        "schema_version",
        "tensor_namespace",
    ]
)


def supports(config):
    return config.get("model_type") == FAMILY or (
        config.get("model_type") == "qwen3_5"
        and isinstance(config.get("quantization"), dict)
        and config["quantization"].get("bits") in {1, 2}
    )


def native_config(config):
    clean = copy.deepcopy(config)
    clean["model_type"] = "qwen3_5"
    clean.pop("quantization", None)
    clean["text_config"].pop("quantization", None)
    return clean


def validate_config(config):
    from oryn_engine.mlx_vision import validate_vision_config

    validate_model_config(config)
    if not supports(config):
        raise ValueError("PRISM_CONFIGURATION")
    q = config.get("quantization")
    if (
        not isinstance(q, dict)
        or set(q) - {"bits", "group_size", "mode"}
        or type(q.get("bits")) is not int
        or q["bits"] not in {1, 2}
        or q.get("group_size") != 128
        or q.get("mode", "affine") != "affine"
    ):
        raise ValueError("PRISM_QUANTIZATION")
    text, vision = config.get("text_config"), config.get("vision_config")
    if not isinstance(text, dict) or not isinstance(vision, dict):
        raise ValueError("PRISM_GRAPH")
    if set(config) - ROOT_FIELDS or set(text) - TEXT_FIELDS or set(vision) - VISION_FIELDS:
        raise ValueError("PRISM_UNREVIEWED_OPTION")
    custom_fields = {
        "base_model_type",
        "components",
        "gdn_activation_layout",
        "hadamard_config",
        "modules",
        "requires_runtime",
        "schema_version",
        "tensor_namespace",
    }
    if config["model_type"] != FAMILY and custom_fields & set(config):
        raise ValueError("PRISM_UNREVIEWED_OPTION")
    if config["model_type"] == FAMILY and {
        "architectures",
        "eos_token_id",
        "language_model_only",
    } & set(config):
        raise ValueError("PRISM_UNREVIEWED_OPTION")
    if text.get("quantization") not in (None, q) or config.get("quantization_config") is not None:
        raise ValueError("PRISM_QUANTIZATION")
    validate_vision_config(native_config(config))
    for part, bounds in (
        (
            text,
            {
                "hidden_size": (128, 16384),
                "intermediate_size": (128, 65536),
                "num_hidden_layers": (2, 128),
                "num_attention_heads": (1, 128),
                "num_key_value_heads": (1, 128),
                "head_dim": (32, 256),
                "vocab_size": (256, 262144),
                "max_position_embeddings": (128, 1048576),
            },
        ),
        (
            vision,
            {
                "depth": (1, 64),
                "hidden_size": (16, 4096),
                "intermediate_size": (16, 16384),
                "num_heads": (1, 64),
                "num_position_embeddings": (4, 16384),
            },
        ),
    ):
        if any(
            type(part.get(k)) is not int or not low <= part[k] <= high
            for k, (low, high) in bounds.items()
        ):
            raise ValueError("PRISM_GEOMETRY")
    rope = text.get("rope_parameters")
    if (
        not isinstance(rope, dict)
        or set(rope)
        != {
            "mrope_interleaved",
            "mrope_section",
            "partial_rotary_factor",
            "rope_theta",
            "rope_type",
        }
        or rope.get("rope_type") != "default"
        or rope.get("mrope_interleaved") is not True
        or rope.get("partial_rotary_factor") != 0.25
        or text.get("partial_rotary_factor") != 0.25
        or type(rope.get("rope_theta")) not in {int, float}
        or not 10000 <= rope["rope_theta"] <= 10000000
        or not isinstance(rope.get("mrope_section"), list)
        or len(rope["mrope_section"]) != 3
        or any(type(x) is not int or x < 1 for x in rope["mrope_section"])
        or sum(rope["mrope_section"]) * 8 != text["head_dim"]
    ):
        raise ValueError("PRISM_ROPE")
    if (
        text["num_attention_heads"] % text["num_key_value_heads"]
        or vision["hidden_size"] % vision["num_heads"]
        or text.get("attention_dropout") != 0
        or text.get("use_cache") is not True
        or text.get("mtp_use_dedicated_embeddings") is not False
        or type(text.get("mtp_num_hidden_layers")) is not int
        or text["mtp_num_hidden_layers"] not in ({0} if config["model_type"] == FAMILY else {0, 1})
    ):
        raise ValueError("PRISM_GRAPH")
    for part, expected in (
        (
            text,
            {
                "model_type": "qwen3_5_text",
                "attention_bias": False,
                "hidden_act": "silu",
                "attn_output_gate": True,
                "output_gate_type": "swish",
                "mamba_ssm_dtype": "float32",
                "linear_conv_kernel_dim": 4,
                "tie_word_embeddings": False,
                "rms_norm_eps": 1e-6,
            },
        ),
        (
            vision,
            {
                "model_type": "qwen3_5",
                "hidden_act": "gelu_pytorch_tanh",
                "deepstack_visual_indexes": [],
                "in_channels": 3,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
        ),
    ):
        if any(part.get(k) != v for k, v in expected.items()):
            raise ValueError("PRISM_GRAPH")
    if (
        config.get("tie_word_embeddings") is not False
        or vision.get("out_hidden_size") != text["hidden_size"]
    ):
        raise ValueError("PRISM_GRAPH")
    for key in ("hidden_size", "intermediate_size"):
        if text[key] % 128:
            raise ValueError("PRISM_PACKED_WIDTH")
    interval = text.get("full_attention_interval")
    if (
        type(interval) is not int
        or not 2 <= interval <= min(16, text["num_hidden_layers"])
        or text.get("layer_types")
        != [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(text["num_hidden_layers"])
        ]
    ):
        raise ValueError("PRISM_LAYER_ORDER")
    if text["linear_num_value_heads"] * text["linear_value_head_dim"] % 128:
        raise ValueError("PRISM_PACKED_WIDTH")
    if config["model_type"] == FAMILY:
        required = {
            "schema_version": 2,
            "base_model_type": "qwen3_5",
            "components": {"text": True, "vision": True, "mtp": False},
            "gdn_activation_layout": "grouped",
            "hadamard_config": "hadamard.json",
            "tensor_namespace": "mlx-vlm-qwen3_5",
            "requires_runtime": "runtime/artifact.py",
        }
        if q != {"bits": 2, "group_size": 128, "mode": "affine"} or any(
            config.get(k) != v for k, v in required.items()
        ):
            raise ValueError("PRISM_HADAMARD_SCHEMA")
        records = config.get("modules")
        if not isinstance(records, list) or not 1 <= len(records) <= 10000:
            raise ValueError("PRISM_MODULES")
        seen = set()
        for record in records:
            if (
                not isinstance(record, dict)
                or set(record) != {"path", "block", "embedding", "dtype"}
                or record["dtype"] != "float16"
                or type(record["embedding"]) is not bool
                or type(record["block"]) is not int
                or record["block"] not in {128, 256, 512, 1024, 2048, 4096}
                or not isinstance(record["path"], str)
                or len(record["path"]) > 256
                or record["path"] in seen
            ):
                raise ValueError("PRISM_MODULES")
            seen.add(record["path"])
    elif config.get("language_model_only") is not False or config.get("architectures") != [
        "Qwen3_5ForConditionalGeneration"
    ]:
        raise ValueError("PRISM_GRAPH")
    return config


def transform_specs(config):
    """Closed module identities; metadata paths never resolve Python attributes."""
    text = config["text_config"]
    hidden = text["hidden_size"]
    specs = {"lm_head": hidden, "model.embed_tokens": hidden}
    for i, kind in enumerate(text["layer_types"]):
        prefix = f"model.layers.{i}."
        specs.update(
            {
                prefix + "mlp." + k + "_proj": width
                for k, width in (
                    ("gate", hidden),
                    ("up", hidden),
                    ("down", text["intermediate_size"]),
                )
            }
        )
        if kind == "linear_attention":
            specs.update(
                {
                    prefix + "linear_attn." + k: width
                    for k, width in (
                        ("in_proj_qkv", hidden),
                        ("in_proj_z", hidden),
                        (
                            "out_proj",
                            text["linear_num_value_heads"] * text["linear_value_head_dim"],
                        ),
                    )
                }
            )
        else:
            specs.update(
                {
                    prefix + "self_attn." + k + "_proj": width
                    for k, width in (
                        ("q", hidden),
                        ("k", hidden),
                        ("v", hidden),
                        ("o", text["num_attention_heads"] * text["head_dim"]),
                    )
                }
            )
    return specs


def validate_hadamard(config, metadata):
    validate_config(config)
    expected = {
        "prism.hadamard." + k: v
        for k, v in {
            "version": 1,
            "transform": "normalized-sylvester-walsh-hadamard",
            "axis": "input-last-dimension",
            "sign_mode": "explicit",
            "gdn_v_grouped": True,
        }.items()
    }
    if set(metadata) != set(expected) | {
        "prism.hadamard." + k
        for k in (
            "block_size",
            "weight_names",
            "inverse_weight_names",
            "sign_widths",
            "sign_values",
        )
    } or any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("PRISM_HADAMARD_METADATA")
    specs = transform_specs(config)
    records = {r["path"]: r for r in config["modules"]}
    block = metadata["prism.hadamard.block_size"]
    if set(records) != set(specs) or any(
        r["block"] != block
        or specs[name] % block
        or r["embedding"] != (name == "model.embed_tokens")
        for name, r in records.items()
    ):
        raise ValueError("PRISM_TRANSFORM_BINDING")
    forward = metadata["prism.hadamard.weight_names"]
    inverse = metadata["prism.hadamard.inverse_weight_names"]
    if (
        not isinstance(forward, list)
        or len(forward) != len(specs) - 1
        or any(not isinstance(n, str) for n in forward)
        or set(forward)
        != {"language_model." + n + ".weight" for n in specs if n != "model.embed_tokens"}
        or inverse != ["language_model.model.embed_tokens.weight"]
    ):
        raise ValueError("PRISM_TRANSFORM_MANIFEST")
    widths, signs = metadata["prism.hadamard.sign_widths"], metadata["prism.hadamard.sign_values"]
    if (
        not isinstance(widths, list)
        or any(type(w) is not int for w in widths)
        or widths != sorted(set(specs.values()))
        or not isinstance(signs, list)
        or len(signs) != sum(widths)
        or any(type(v) not in {int, float} or v not in {-1, 1} for v in signs)
    ):
        raise ValueError("PRISM_SIGNS")
    offset, result = 0, {}
    for width in widths:
        result[width] = signs[offset : offset + width]
        offset += width
    return result


def validate_files(files, config):
    required = {"config.json", "tokenizer_config.json", "tokenizer.json"}
    if config["model_type"] == FAMILY:
        required |= {"hadamard.json", "generation_config.json"}
    if not required <= set(files):
        raise ValueError("PRISM_METADATA_MISSING")
    weights = [name for name in files if name.endswith(".safetensors")]
    if not weights or any(
        not re.fullmatch(r"model(?:-\d{5}-of-\d{5})?\.safetensors", name) for name in weights
    ):
        raise ValueError("PRISM_GRAPH_ARTIFACTS")
    if any(
        name.startswith(("causal/", "crystal_runtime/")) or name in {"bundle.json", "crystal.json"}
        for name in files
    ):
        raise ValueError("PRISM_GRAPH_ARTIFACTS")


def native_model(config):
    from mlx_vlm.models.qwen3_5 import Model, ModelConfig

    validate_config(config)
    return Model(ModelConfig.from_dict(native_config(config)))


def packed_modules(model, config):
    import mlx.nn as nn

    return {
        name: module
        for name, module in model.named_modules()
        if name.startswith("language_model.")
        and isinstance(module, (nn.Linear, nn.Embedding))
        and not (config["model_type"] == FAMILY and name.endswith((".in_proj_a", ".in_proj_b")))
    }


def validate_layout(model, config, shapes):
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    expected = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
    selected = packed_modules(model, config)
    records = {"language_model." + r["path"]: r for r in config.get("modules", [])}
    if config["model_type"] == FAMILY and set(records) != set(selected):
        raise ValueError("PRISM_MODULE_BINDING")
    bits = config["quantization"]["bits"]
    for name, module in selected.items():
        rows, cols = expected[name + ".weight"]
        if cols % 128:
            raise ValueError("PRISM_PACKED_WIDTH")
        expected[name + ".weight"] = (rows, cols * bits // 32)
        expected[name + ".scales"] = expected[name + ".biases"] = (rows, cols // 128)
        if name in records:
            record = records[name]
            if record["embedding"] != isinstance(module, nn.Embedding) or cols % record["block"]:
                raise ValueError("PRISM_TRANSFORM_BINDING")
            expected[name + ".signs"] = (cols,)
    if expected != shapes:
        raise ValueError("PRISM_WEIGHTS_INCOMPLETE")
    return selected, records


def widen_binary(weight):
    """Lossless 1-bit lanes to native 2-bit lanes, without dequantizing values."""
    import mlx.core as mx

    def spread(x):
        x = (x | (x << 8)) & 0x00FF00FF
        x = (x | (x << 4)) & 0x0F0F0F0F
        x = (x | (x << 2)) & 0x33333333
        return (x | (x << 1)) & 0x55555555

    return mx.stack((spread(weight & 0xFFFF), spread(weight >> 16)), axis=-1).reshape(
        weight.shape[0], weight.shape[1] * 2
    )


def packed_class():
    import math

    import mlx.core as mx
    import mlx.nn as nn

    class Packed(nn.Module):
        def __init__(self, weight, scales, biases, *, embedding=False, block=0, signs=None):
            super().__init__()
            self.weight, self.scales, self.biases = weight, scales, biases
            self.embedding, self.block = embedding, block
            if signs is not None:
                self.signs = signs

        def transform(self, x, inverse=False):
            if not self.block:
                return x
            shape, dtype = x.shape, x.dtype
            x = x.astype(mx.float32)
            if not inverse:
                x = x * self.signs
            x = mx.hadamard_transform(
                x.reshape(-1, self.block), scale=1 / math.sqrt(self.block)
            ).reshape(shape)
            if inverse:
                x = x * self.signs
            return x.astype(dtype)

        def __call__(self, x):
            if self.embedding:
                ids = x.reshape(-1)
                out = mx.dequantize(
                    self.weight[ids], self.scales[ids], self.biases[ids], group_size=128, bits=2
                ).reshape(*x.shape, -1)
                return self.transform(out.astype(mx.float16), inverse=True) if self.block else out
            return mx.quantized_matmul(
                self.transform(x),
                self.weight,
                self.scales,
                self.biases,
                transpose=True,
                group_size=128,
                bits=2,
            )

    return Packed


def load_model(root, config):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten

    from oryn_engine.mlx_vision import verify_vision_build

    verify_vision_build()
    validate_config(config)
    validate_files([p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()], config)
    sign_vectors = (
        validate_hadamard(config, metadata_file(root / "hadamard.json"))
        if config["model_type"] == FAMILY
        else {}
    )
    shapes = preflight_safetensors(root)
    model = native_model(config)
    selected, records = validate_layout(model, config, shapes)
    weights = {}
    for path in sorted(root.glob("*.safetensors")):
        for name, tensor in mx.load(str(path)).items():
            if name in weights:
                raise ValueError("PRISM_DUPLICATE_WEIGHT")
            packed = name.endswith(".weight") and name.removesuffix(".weight") in selected
            if packed:
                if tensor.dtype != mx.uint32:
                    raise ValueError("PRISM_PACKED_DTYPE")
            elif tensor.dtype not in {mx.float16, mx.float32, mx.bfloat16} or not bool(
                mx.all(mx.isfinite(tensor)).item()
            ):
                raise ValueError("PRISM_NONFINITE_WEIGHT")
            weights[name] = tensor
    replacements = []
    Packed = packed_class()
    for name, module in selected.items():
        w, s, b = [weights[name + "." + suffix] for suffix in ("weight", "scales", "biases")]
        if config["quantization"]["bits"] == 1:
            w = widen_binary(w)
            weights[name + ".weight"] = w
        record = records.get(name)
        block = record["block"] if record else 0
        signs = weights.get(name + ".signs")
        if block and (signs is None or not bool(mx.all((signs == 1) | (signs == -1)).item())):
            raise ValueError("PRISM_SIGNS")
        if block and not bool(mx.all(signs == mx.array(sign_vectors[signs.size])).item()):
            raise ValueError("PRISM_SIGN_BINDING")
        replacements.append(
            (
                name,
                Packed(
                    w, s, b, embedding=isinstance(module, nn.Embedding), block=block, signs=signs
                ),
            )
        )
    model.update_modules(tree_unflatten(replacements))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def validate_tokenizer_metadata(metadata):
    validate_model_config(metadata)
    if (
        metadata.get("tokenizer_class") not in {"Qwen2Tokenizer", "TokenizersBackend"}
        or metadata.get("eos_token") != "<|im_end|>"
        or metadata.get("pad_token") != "<|endoftext|>"
        or metadata.get("bos_token") is not None
        or metadata.get("add_prefix_space", False) is not False
        or metadata.get("add_bos_token", False) is not False
        or metadata.get("split_special_tokens", False) is not False
    ):
        raise ValueError("PRISM_TOKENIZER_METADATA")


def load_tokenizer(root, config):
    from tokenizers import Tokenizer
    from transformers import TokenizersBackend

    validate_tokenizer_metadata(metadata_file(root / "tokenizer_config.json"))
    path = root / "tokenizer.json"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 32 * 1024**2:
        raise ValueError("PRISM_TOKENIZER_GRAPH")
    vocab = Tokenizer.from_file(str(path)).get_vocab()
    if (
        not 0 <= config["text_config"]["vocab_size"] - len(vocab) <= 512
        or set(vocab.values()) != set(range(len(vocab)))
        or any(len(t.encode()) > 4096 for t in vocab)
        or any(t not in vocab for t in MARKERS)
    ):
        raise ValueError("PRISM_VOCABULARY")
    tokenizer = TokenizersBackend(
        tokenizer_file=str(path), eos_token=MARKERS[1], pad_token=MARKERS[2]
    )
    tokenizer.chat_template = None
    if tokenizer.get_vocab() != vocab or any(
        tokenizer.encode(t, add_special_tokens=False) != [vocab[t]] for t in MARKERS
    ):
        raise ValueError("PRISM_TOKENIZER_CONTROLS")
    text = config["text_config"]
    if (
        text.get("bos_token_id") != vocab[MARKERS[2]]
        or text.get("eos_token_id")
        != vocab[MARKERS[2] if config["model_type"] == FAMILY else MARKERS[1]]
    ):
        raise ValueError("PRISM_STOP_TOKENS")
    if config["model_type"] == FAMILY:
        generation = metadata_file(root / "generation_config.json")
        validate_generation_metadata(config, generation)
        if (
            generation.get("bos_token_id") != vocab[MARKERS[2]]
            or generation.get("pad_token_id") != vocab[MARKERS[2]]
            or generation.get("eos_token_id") != [vocab[MARKERS[1]], vocab[MARKERS[2]]]
        ):
            raise ValueError("PRISM_STOP_TOKENS")
        stops = {vocab[MARKERS[1]], vocab[MARKERS[2]]}
    else:
        if config.get("eos_token_id") != vocab[MARKERS[1]]:
            raise ValueError("PRISM_STOP_TOKENS")
        stops = {vocab[MARKERS[1]]}
    return tokenizer, stops


def validate_generation_metadata(config, generation):
    validate_model_config(generation)
    text = config["text_config"]
    stops = generation.get("eos_token_id")
    if (
        generation.get("bos_token_id") != text["bos_token_id"]
        or generation.get("pad_token_id") != text["bos_token_id"]
        or not isinstance(stops, list)
        or len(stops) != 2
        or any(type(t) is not int or not 0 <= t < text["vocab_size"] for t in stops)
        or stops[0] == stops[1]
        or stops[1] != text["eos_token_id"]
    ):
        raise ValueError("PRISM_STOP_TOKENS")


def prompt(messages):
    from oryn_engine.chat_formats import render

    if (
        not messages
        or messages[-1]["role"] != "user"
        or any(m["role"] == "system" for m in messages[1:])
        or any(
            any(t in m["content"] for t in ("<|", "|>", "<think>", "</think>")) for m in messages
        )
    ):
        raise ValueError("PRISM_RESERVED_CONTROL")
    # The publisher's preserve_thinking=False branch omits old reasoning turns.
    messages = [{"role": m["role"], "content": m["content"].strip()} for m in messages]
    return render("chatml", messages, thinking_off=True)


def text_view(model):
    import mlx.core as mx
    import mlx.nn as nn

    class TextView(nn.Module):
        def __init__(self):
            super().__init__()
            self.graph = model

        def __call__(self, inputs, cache=None):
            logits = self.graph.language_model(inputs, cache=cache).logits
            if not bool(mx.all(mx.isfinite(logits)).item()):
                raise ValueError("PRISM_NONFINITE_LOGITS")
            return logits

        @property
        def layers(self):
            return self.graph.language_model.layers

        def make_cache(self):
            return self.graph.language_model.make_cache()

    return TextView()


def generate(root, config, value, event):
    validate_config(config)
    return generate_with_loader(
        root, config, value, event, load_model=load_model, load_tokenizer=load_tokenizer
    )


def generate_with_loader(
    root,
    config,
    value,
    event,
    *,
    load_model,
    load_tokenizer,
    render_prompt=None,
    model_view=None,
):
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    if not 1 <= value["maxTokens"] <= MAX_OUTPUT:
        raise ValueError("PRISM_OUTPUT_LIMIT")
    tokenizer, stops = load_tokenizer(root, config)
    tokens = tokenizer.encode(
        (render_prompt or prompt)(value["messages"]), add_special_tokens=False
    )
    vocabulary_ids = set(tokenizer.get_vocab().values())
    if len(tokens) + value["maxTokens"] > min(
        MAX_CONTEXT, config.get("text_config", config)["max_position_embeddings"]
    ):
        raise ValueError("PRISM_CONTEXT_LIMIT")
    event("job.progress", {"stage": "loading-model", "progress": 0.05})
    model = (model_view or text_view)(load_model(root, config))
    wrapper = TokenizerWrapper(tokenizer, eos_token_ids=stops)
    output = []
    event("job.state", {"state": "running"})
    try:
        for response in stream_generate(
            model,
            wrapper,
            prompt=tokens,
            max_tokens=value["maxTokens"],
            sampler=make_sampler(temp=value["temperature"], top_p=value["topP"]),
        ):
            token = int(response.token)
            if token not in stops:
                if token not in vocabulary_ids:
                    raise ValueError("PRISM_OUTPUT_TOKEN")
                output.append(token)
        text = tokenizer.decode(output, skip_special_tokens=False)
        if (
            not text.strip()
            or len(text.encode()) > 65536
            or any(t in text for t in ("<|", "|>", "<think>", "</think>"))
        ):
            raise ValueError("PRISM_OUTPUT_TEXT")
        event("job.output.delta", {"channel": "text", "delta": text})
        return {"text": text, "backend": "mlx"}
    finally:
        del model
        mx.clear_cache()
