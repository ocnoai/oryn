"""Closed Kolibri MLX affine layout, expert routing and owned chat framing."""

from __future__ import annotations

import copy
import re

from oryn_engine import mlx_prism
from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

MAX_CONTEXT, MAX_OUTPUT = 4096, 512
METADATA = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "model.safetensors.index.json",
}
MARKERS = mlx_prism.MARKERS
NO_REASONING = (
    "# Reasoning effort\n\n"
    "Reasoning is disabled. Proceed straight to answering according to the user's instructions."
)


def supports(config):
    return config.get("model_type") == "kolibri1"


def validate_config(config):
    from oryn_engine.runtime_packs.mlx_lm_worker import validate_config as validate_mlx

    validate_mlx(config, {"kolibri1"})
    fields = {
        "architectures",
        "attention_bias",
        "attention_dropout",
        "bos_token_id",
        "dtype",
        "eos_token_id",
        "head_dim",
        "head_dtype",
        "hidden_act",
        "hidden_size",
        "layer_types",
        "max_position_embeddings",
        "model_type",
        "moe_intermediate_size",
        "norm_topk_prob",
        "num_attention_heads",
        "num_experts",
        "num_experts_per_tok",
        "num_hidden_layers",
        "num_key_value_heads",
        "pad_token_id",
        "quantization",
        "quantization_config",
        "rms_norm_eps",
        "rope_theta",
        "shared_expert_intermediate_size",
        "sliding_window",
        "tie_word_embeddings",
        "use_cache",
        "use_sliding_window",
        "vocab_size",
    }
    if set(config) != fields:
        raise ValueError("KOLIBRI_UNREVIEWED_OPTION")
    bounds = {
        "hidden_size": (64, 8192),
        "num_hidden_layers": (2, 128),
        "num_attention_heads": (1, 128),
        "num_key_value_heads": (1, 128),
        "head_dim": (16, 256),
        "num_experts": (2, 512),
        "num_experts_per_tok": (1, 32),
        "moe_intermediate_size": (64, 16384),
        "shared_expert_intermediate_size": (64, 16384),
        "vocab_size": (256, 262144),
        "sliding_window": (2, 4096),
        "max_position_embeddings": (128, 1048576),
    }
    if any(
        type(config.get(k)) is not int or not lo <= config[k] <= hi
        for k, (lo, hi) in bounds.items()
    ):
        raise ValueError("KOLIBRI_GEOMETRY")
    fixed = {
        "architectures": ["Kolibri1ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": None,
        "dtype": "bfloat16",
        "head_dtype": "float32",
        "hidden_act": "silu",
        "norm_topk_prob": False,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "use_cache": True,
        "use_sliding_window": True,
    }
    if any(
        config.get(k) != v or isinstance(config.get(k), bool) != isinstance(v, bool)
        for k, v in fixed.items()
    ):
        raise ValueError("KOLIBRI_GRAPH")
    if (
        config["num_attention_heads"] % config["num_key_value_heads"]
        or config["head_dim"] % 2
        or config["num_experts_per_tok"] > config["num_experts"]
        or config["num_attention_heads"] * config["head_dim"] % 64
        or any(
            config[k] % 64
            for k in ("hidden_size", "moe_intermediate_size", "shared_expert_intermediate_size")
        )
        or config["sliding_window"] > config["max_position_embeddings"]
        or any(
            type(config.get(k)) is not int or not 0 <= config[k] < config["vocab_size"]
            for k in ("eos_token_id", "pad_token_id")
        )
        or config["eos_token_id"] == config["pad_token_id"]
    ):
        raise ValueError("KOLIBRI_GEOMETRY")
    layers = config["layer_types"]
    if (
        not isinstance(layers, list)
        or len(layers) != config["num_hidden_layers"]
        or any(not isinstance(k, str) for k in layers)
        or set(layers) != {"sliding_attention", "full_attention"}
    ):
        raise ValueError("KOLIBRI_ATTENTION")
    overrides = {"model.embed_tokens", "lm_head"}
    for i in range(config["num_hidden_layers"]):
        overrides |= {
            f"model.layers.{i}." + part
            for part in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.shared_experts.gate_proj",
                "mlp.shared_experts.up_proj",
                "mlp.shared_experts.down_proj",
            )
        }
    expected = {"bits": 4, "group_size": 64, "mode": "affine"} | {
        name: {"bits": 8, "group_size": 64, "mode": "affine"} for name in overrides
    }
    if config["quantization"] != expected or config["quantization_config"] != expected:
        raise ValueError("KOLIBRI_QUANTIZATION")
    return config


def parameter_specs(config):
    validate_config(config)
    h, v, d, e = (config[k] for k in ("hidden_size", "vocab_size", "head_dim", "num_experts"))
    routed, shared = config["moe_intermediate_size"], config["shared_expert_intermediate_size"]
    q, kv = config["num_attention_heads"] * d, config["num_key_value_heads"] * d
    specs = {
        "model.embed_tokens.weight": (v, h),
        "lm_head.weight": (v, h),
        "model.norm.weight": (h,),
    }
    for i in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{i}."
        specs.update(
            {
                prefix + name + ".weight": (h,)
                for name in (
                    "input_layernorm",
                    "post_attn_norm",
                    "post_attention_layernorm",
                    "post_ffn_norm",
                )
            }
        )
        specs.update(
            {
                prefix + "self_attn." + name: shape
                for name, shape in {
                    "q_proj.weight": (q, h),
                    "k_proj.weight": (kv, h),
                    "v_proj.weight": (kv, h),
                    "o_proj.weight": (h, q),
                    "q_norm.weight": (d,),
                    "k_norm.weight": (d,),
                }.items()
            }
        )
        specs[prefix + "mlp.gate.weight"] = (e, h)
        specs[prefix + "mlp.expert_bias"] = (e,)
        for label, width in (("switch_mlp", routed), ("shared_experts", shared)):
            leading = (e,) if label == "switch_mlp" else ()
            for name, shape in {"gate": (width, h), "up": (width, h), "down": (h, width)}.items():
                specs[prefix + f"mlp.{label}.{name}_proj.weight"] = (*leading, *shape)
    return specs


def packed_specs(config):
    return {
        name.removesuffix(".weight"): (shape, 4 if ".switch_mlp." in name else 8)
        for name, shape in parameter_specs(config).items()
        if len(shape) in {2, 3} and not name.endswith(".mlp.gate.weight")
    }


def serialized_shapes(config):
    result = parameter_specs(config)
    for name, (shape, bits) in packed_specs(config).items():
        result[name + ".weight"] = (*shape[:-1], shape[-1] * bits // 32)
        result[name + ".scales"] = result[name + ".biases"] = (*shape[:-1], shape[-1] // 64)
    return result


def selected_files(files, config):
    validate_config(config)
    shards = {n for n in files if n.endswith(".safetensors")}
    if (
        not set(files) >= METADATA
        or not 1 <= len(shards) <= 32
        or shards
        != {f"model-{i:05d}-of-{len(shards):05d}.safetensors" for i in range(1, len(shards) + 1)}
    ):
        raise ValueError("KOLIBRI_FILES")
    return tuple(sorted(METADATA | shards))


def validate_tokenizer_metadata(metadata):
    validate_model_config(metadata)
    if metadata.get("tokenizer_class") != "PreTrainedTokenizerFast":
        raise ValueError("KOLIBRI_TOKENIZER_CLASS")
    mlx_prism.validate_tokenizer_metadata(metadata | {"tokenizer_class": "TokenizersBackend"})


def validate_generation_metadata(config, value):
    validate_model_config(value)
    if (
        value.get("eos_token_id") != [config["eos_token_id"], config["pad_token_id"]]
        or value.get("pad_token_id") != config["pad_token_id"]
    ):
        raise ValueError("KOLIBRI_STOPS")


def validate_headers(config, index, headers):
    validate_model_config(index)
    mapping = index.get("weight_map")
    expected = serialized_shapes(config)
    if (
        not isinstance(mapping, dict)
        or set(mapping) != set(expected)
        or any(
            not isinstance(n, str) or not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", n)
            for n in mapping.values()
        )
    ):
        raise ValueError("KOLIBRI_INDEX")
    packed = packed_specs(config)
    actual = {}
    for filename, header in headers.items():
        if not isinstance(header, dict) or not isinstance(header.get("tensors"), dict):
            raise ValueError("KOLIBRI_HEADER")
        for name, record in header["tensors"].items():
            if name in actual or mapping.get(name) != filename or not isinstance(record, dict):
                raise ValueError("KOLIBRI_SHARD_BINDING")
            packed_weight = name.endswith(".weight") and name.removesuffix(".weight") in packed
            dtypes = (
                {"U32"}
                if packed_weight
                else {"F32"}
                if name.endswith(".expert_bias")
                else {"F16", "BF16", "F32"}
            )
            if record.get("dtype") not in dtypes:
                raise ValueError("KOLIBRI_DTYPE")
            actual[name] = tuple(record.get("shape", []))
    if actual != expected:
        raise ValueError("KOLIBRI_WEIGHTS_INCOMPLETE")


def inspect_metadata(config, read_json, read_header, files):
    names = selected_files(files, config)
    validate_tokenizer_metadata(read_json("tokenizer_config.json"))
    validate_generation_metadata(config, read_json("generation_config.json"))
    index = read_json("model.safetensors.index.json")
    validate_headers(
        config, index, {n: read_header(n) for n in names if n.endswith(".safetensors")}
    )


def native_model(config):
    import mlx.core as mx

    from oryn_engine.native_kolibri import Model, ModelArgs

    validate_config(config)

    class CheckedModel(Model):
        def __call__(self, *args, **kwargs):
            out = super().__call__(*args, **kwargs)
            if not bool(mx.all(mx.isfinite(out)).item()):
                raise ValueError("KOLIBRI_NONFINITE_LOGITS")
            return out

    return CheckedModel(ModelArgs.from_dict(copy.deepcopy(config)))


def load_model(root, config):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from oryn_engine.runtime_packs.mlx_lm_worker import validate_quantized_layers

    files = selected_files(
        [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()], config
    )
    shapes = preflight_safetensors(root)
    if shapes != serialized_shapes(config):
        raise ValueError("KOLIBRI_WEIGHTS_INCOMPLETE")
    model = native_model(config)
    if {n: tuple(v.shape) for n, v in tree_flatten(model.parameters())} != parameter_specs(config):
        raise ValueError("KOLIBRI_NATIVE_GRAPH_CHANGED")
    validate_quantized_layers(model, config, shapes)
    packed = packed_specs(config)
    weights = {}
    for filename in files:
        if not filename.endswith(".safetensors"):
            continue
        for name, value in mx.load(str(root / filename)).items():
            if name in weights:
                raise ValueError("KOLIBRI_DUPLICATE_WEIGHT")
            is_packed = name.endswith(".weight") and name.removesuffix(".weight") in packed
            if is_packed:
                if value.dtype != mx.uint32:
                    raise ValueError("KOLIBRI_PACKED_DTYPE")
            elif value.dtype not in {mx.float16, mx.bfloat16, mx.float32} or not bool(
                mx.all(mx.isfinite(value)).item()
            ):
                raise ValueError("KOLIBRI_NONFINITE_WEIGHT")
            if name.endswith(".expert_bias") and value.dtype != mx.float32:
                raise ValueError("KOLIBRI_ROUTER_PRECISION")
            weights[name] = value
    nn.quantize(
        model,
        bits=4,
        group_size=64,
        mode="affine",
        class_predicate=lambda name, module: (
            {"bits": packed[name][1], "group_size": 64, "mode": "affine"}
            if name in packed
            else False
        ),
    )
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def load_tokenizer(root, config):
    from tokenizers import Tokenizer
    from transformers import TokenizersBackend

    validate_tokenizer_metadata(metadata_file(root / "tokenizer_config.json"))
    generation = metadata_file(root / "generation_config.json")
    validate_generation_metadata(config, generation)
    path = root / "tokenizer.json"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 32 * 1024**2:
        raise ValueError("KOLIBRI_TOKENIZER_GRAPH")
    vocab = Tokenizer.from_file(str(path)).get_vocab()
    ids = set(vocab.values())
    if (
        len(vocab) != len(ids)
        or not ids <= set(range(config["vocab_size"]))
        or not 0 <= config["vocab_size"] - len(vocab) <= 8
        or any(len(t.encode()) > 4096 for t in vocab)
        or any(t not in vocab for t in (*MARKERS, "<think>", "</think>"))
    ):
        raise ValueError("KOLIBRI_VOCABULARY")
    tok = TokenizersBackend(tokenizer_file=str(path), eos_token=MARKERS[1], pad_token=MARKERS[2])
    tok.chat_template = None
    if tok.get_vocab() != vocab or any(
        tok.encode(t, add_special_tokens=False) != [vocab[t]]
        for t in (*MARKERS, "<think>", "</think>")
    ):
        raise ValueError("KOLIBRI_TOKENIZER_CONTROLS")
    if vocab[MARKERS[1]] != config["eos_token_id"] or vocab[MARKERS[2]] != config["pad_token_id"]:
        raise ValueError("KOLIBRI_STOPS")
    return tok, set(generation["eos_token_id"])


def prompt(messages):
    # Validate the same closed role/control surface, then reproduce the original
    # template's non-thinking system instruction without interpreting its Jinja.
    mlx_prism.prompt(messages)
    if any(
        any(
            s in m["content"]
            for s in ("<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>")
        )
        for m in messages
    ):
        raise ValueError("KOLIBRI_RESERVED_CONTROL")
    messages = copy.deepcopy(messages)
    if messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + NO_REASONING
    else:
        messages.insert(0, {"role": "system", "content": NO_REASONING})
    for m in messages:
        if m["role"] == "assistant":
            m["content"] = m["content"].lstrip("\n")
    from oryn_engine.chat_formats import render

    return render("chatml", messages, thinking_off=True)


def generate(root, config, value, event):
    validate_config(config)
    return mlx_prism.generate_with_loader(
        root,
        config,
        value,
        event,
        load_model=load_model,
        load_tokenizer=load_tokenizer,
        render_prompt=prompt,
        model_view=lambda model: model,
    )
