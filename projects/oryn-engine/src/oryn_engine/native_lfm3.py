"""Oryn's closed, eager PyTorch LFM3 inference graph.

Math adapted from LiquidAI/LFM3-1.7B at
875094441ea6517efc0f44a5685f05e381b72b14 (Apache-2.0).
Copyright 2026 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0; see native_lfm3.LICENSE.
Modified by Oryn: inference only, batch one, bounded cache and default RoPE;
no Transformers dynamic classes, Hub kernels, Triton, compilation or offload.
The original repository files are recognition evidence and are never imported.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
from pathlib import Path

CONTEXT_LIMIT = 4096
OUTPUT_LIMIT = 512
ADAPTER_ID = "oryn.lfm3.torch.v1"
TOKENIZER_SHA256 = "7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701"
ARCHITECTURE = "OrynReviewedLfm3ForCausalLM"
LAYERS = [
    "full_attention" if i % 4 == 2 else "sliding_attention" if i % 4 == 0 and i else "conv"
    for i in range(24)
]
CORE = {
    "model_type": "lfm3",
    "hidden_size": 2048,
    "intermediate_size": 7680,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "vocab_size": 128000,
    "conv_L_cache": 7,
    "conv_bias": False,
    "sliding_window": 256,
    "use_sandwich_norm": True,
    "tie_word_embeddings": True,
    "norm_eps": 1e-5,
    "layer_types": LAYERS,
    "rope_parameters": {
        "full_attention": {
            "partial_rotary_factor": 0.5,
            "rope_theta": 1000000.0,
            "rope_type": "default",
        },
        "sliding_attention": {"rope_theta": 10000, "rope_type": "default"},
    },
}


def representation(config):
    quant = config.get("quantization_config") or config.get("quantization")
    if quant:
        if isinstance(quant, dict):
            method = quant.get("quant_method")
            if method == "compressed-tensors":
                return "fp8-unsupported"
            if method == "modelopt":
                return "nvfp4-unsupported"
            if quant.get("mode") == "affine" and quant.get("bits") == 4:
                return "mlx-affine4-unsupported"
        return "quantization-unsupported"
    return (
        "bf16-cpu"
        if config.get("dtype", config.get("torch_dtype")) == "bfloat16"
        else "dtype-unsupported"
    )


def validate_config(config):
    allowed = set(CORE) | {
        "architectures",
        "auto_map",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "dtype",
        "torch_dtype",
        "max_position_embeddings",
        "transformers_version",
        "use_cache",
    }
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("LFM3 configuration has unreviewed options")
    if representation(config) != "bf16-cpu":
        raise ValueError("LFM3 representation is not qualified")
    for key, value in CORE.items():
        # JSON bools must not masquerade as integer dimensions.
        if config.get(key) != value or type(config.get(key)) is not type(value):
            raise ValueError("LFM3 dimensions/operators differ from the reviewed graph")
    if config.get("architectures") not in (["Lfm3ForCausalLM"], [ARCHITECTURE]):
        raise ValueError("LFM3 architecture differs")
    if type(config.get("max_position_embeddings")) is not int or not (
        CONTEXT_LIMIT <= config["max_position_embeddings"] <= 262144
    ):
        raise ValueError("LFM3 positional bound differs")
    for key, expected in (
        ("bos_token_id", 124894),
        ("eos_token_id", 124900),
        ("pad_token_id", 124893),
    ):
        if type(config.get(key)) is not int or config[key] != expected:
            raise ValueError("LFM3 token identities differ")
    if config.get("use_cache", True) is not True:
        raise ValueError("LFM3 cache policy differs")
    return config


def expected_shapes(config):
    h, m, d = config["hidden_size"], config["intermediate_size"], config["head_dim"]
    heads, kv = config["num_attention_heads"], config["num_key_value_heads"]
    shapes = {
        "model.embed_tokens.weight": (config["vocab_size"], h),
        "model.embedding_norm.weight": (h,),
    }
    for i, kind in enumerate(config["layer_types"]):
        prefix = f"model.layers.{i}."
        for norm in ("operator_norm", "ffn_norm", "post_operator_norm", "post_ffn_norm"):
            shapes[prefix + norm + ".weight"] = (h,)
        for name, shape in {"w1": (m, h), "w3": (m, h), "w2": (h, m)}.items():
            shapes[prefix + f"feed_forward.{name}.weight"] = shape
        if kind == "conv":
            for name, shape in {
                "conv": (h, 1, config["conv_L_cache"]),
                "in_proj": (3 * h, h),
                "out_proj": (h, h),
            }.items():
                shapes[prefix + f"conv.{name}.weight"] = shape
        else:
            for name, shape in {
                "q_proj": (heads * d, h),
                "k_proj": (kv * d, h),
                "v_proj": (kv * d, h),
                "o_proj": (h, heads * d),
                "q_norm": (d,),
                "k_norm": (d,),
                "gate_proj": (heads, h),
            }.items():
                shapes[prefix + f"self_attn.{name}.weight"] = shape
    return shapes


def load_weights(root, config):
    from oryn_engine.model_metadata import parse_metadata, preflight_safetensors

    # Initial qualification is one BF16 file. Shards/other dtypes are separate
    # representation profiles rather than an optimistic loader fallback.
    path = root / "model.safetensors"
    if not path.is_file() or path.stat().st_size > 4 * 1024**3:
        raise ValueError("LFM3 weight layout is outside the reviewed profile")
    shapes = preflight_safetensors(root, weight_names=("model.safetensors",))
    expected = expected_shapes(config)
    if shapes != expected:
        raise ValueError("LFM3 tensor set or shapes differ from the reviewed graph")
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = parse_metadata(stream.read(length), limit=8 * 1024**2)
    if any(v.get("dtype") != "BF16" for k, v in header.items() if k != "__metadata__"):
        raise ValueError("LFM3 requires actual BF16 tensors")
    import torch

    # Preserve checkpoint BF16, matching the reference eager inference dtype.
    # Stream validated spans into fixed-size CPU buffers; no whole-file mapping.
    weights = {}
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        identity = os.fstat(stream.fileno())
        for name, shape in expected.items():
            start, end = header[name]["data_offsets"]
            buffer = bytearray(end - start)
            stream.seek(8 + length + start)
            if stream.readinto(buffer) != len(buffer):
                raise ValueError("LFM3 tensor data changed or is truncated")
            weights[name] = torch.frombuffer(buffer, dtype=torch.bfloat16).reshape(shape)
        final = os.fstat(stream.fileno())
        if (identity.st_ino, identity.st_size, identity.st_mtime_ns, identity.st_ctime_ns) != (
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        ):
            raise ValueError("LFM3 tensor file changed during loading")
    return weights


class Graph:
    """Single-request cache; no tensors or prompts survive into another job."""

    def __init__(self, config, weights):
        self.config, self.weights = config, weights
        if {k: tuple(v.shape) for k, v in weights.items()} != expected_shapes(config):
            raise ValueError("Incomplete graph")
        self.cache = {}
        self.position = 0

    def forward(self, tokens):
        import torch
        import torch.nn.functional as F

        c, w = self.config, self.weights
        if (
            not isinstance(tokens, list)
            or not tokens
            or self.position + len(tokens) > CONTEXT_LIMIT
            or any(type(t) is not int or not 0 <= t < c["vocab_size"] for t in tokens)
        ):
            raise ValueError("Token/context bounds")
        length, start = len(tokens), self.position
        x = F.embedding(torch.tensor([tokens], dtype=torch.long), w["model.embed_tokens.weight"])

        def norm(value, key):
            normalized = value.float() * torch.rsqrt(
                value.float().square().mean(-1, keepdim=True) + c["norm_eps"]
            )
            return normalized.to(value.dtype) * w[key]

        def linear(value, key):
            return F.linear(value, w[key + ".weight"])

        for i, kind in enumerate(c["layer_types"]):
            prefix = f"model.layers.{i}."
            u = norm(x, prefix + "operator_norm.weight")
            if kind == "conv":
                p = prefix + "conv."
                b, gate, z = linear(u, p + "in_proj").transpose(1, 2).chunk(3, dim=1)
                current = b * z
                previous = self.cache.get(i)
                if previous is None:
                    previous = current.new_zeros((1, c["hidden_size"], c["conv_L_cache"]))
                joined = torch.cat((previous, current), dim=-1)
                self.cache[i] = joined[..., -c["conv_L_cache"] :].clone()
                convolved = F.conv1d(joined, w[p + "conv.weight"], groups=c["hidden_size"])
                y = linear((gate * convolved[..., -length:]).transpose(1, 2), p + "out_proj")
            else:
                p = prefix + "self_attn."
                heads, kv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
                q = norm(
                    linear(u, p + "q_proj").view(1, length, heads, d), p + "q_norm.weight"
                ).transpose(1, 2)
                k = norm(
                    linear(u, p + "k_proj").view(1, length, kv, d), p + "k_norm.weight"
                ).transpose(1, 2)
                v = linear(u, p + "v_proj").view(1, length, kv, d).transpose(1, 2)
                rope = c["rope_parameters"][kind]
                angles = int(rope.get("partial_rotary_factor", 1.0) * d) // 2
                inverse = 1.0 / (rope["rope_theta"] ** (torch.arange(0, 2 * angles, 2).float() / d))
                inverse = F.pad(inverse, (0, d // 2 - angles))
                frequencies = torch.arange(start, start + length).float()[:, None] * inverse
                frequencies = torch.cat((frequencies, frequencies), dim=-1)[None, None]
                cos, sin = frequencies.cos().to(x.dtype), frequencies.sin().to(x.dtype)

                def rotate(value, cos=cos, sin=sin):
                    left, right = value.float().chunk(2, dim=-1)
                    rotated = (
                        value.float() * cos.float()
                        + torch.cat((-right, left), dim=-1) * sin.float()
                    )
                    return rotated.to(value.dtype)

                q, k = rotate(q), rotate(k)
                old = self.cache.get(i)
                key_start = 0
                if old is not None:
                    old_k, old_v, key_start = old
                    k, v = torch.cat((old_k, k), dim=2), torch.cat((old_v, v), dim=2)
                positions = torch.arange(key_start, key_start + k.shape[2])
                queries = torch.arange(start, start + length)[:, None]
                mask = positions <= queries
                if kind == "sliding_attention":
                    mask &= positions > queries - c["sliding_window"]
                    retained = min(k.shape[2], c["sliding_window"])
                else:
                    retained = k.shape[2]
                self.cache[i] = (
                    k[:, :, -retained:].clone(),
                    v[:, :, -retained:].clone(),
                    key_start + k.shape[2] - retained,
                )
                repeated_k = k.repeat_interleave(heads // kv, dim=1)
                repeated_v = v.repeat_interleave(heads // kv, dim=1)
                y = F.scaled_dot_product_attention(q, repeated_k, repeated_v, attn_mask=mask)
                y = y.transpose(1, 2) * torch.sigmoid(linear(u, p + "gate_proj")).unsqueeze(-1)
                y = linear(y.reshape(1, length, heads * d), p + "o_proj")
            x = x + norm(y, prefix + "post_operator_norm.weight")
            u = norm(x, prefix + "ffn_norm.weight")
            y = linear(
                F.silu(linear(u, prefix + "feed_forward.w1"))
                * linear(u, prefix + "feed_forward.w3"),
                prefix + "feed_forward.w2",
            )
            x = x + norm(y, prefix + "post_ffn_norm.weight")
        self.position += length
        x = norm(x[:, -1], "model.embedding_norm.weight")
        logits = F.linear(x, w["model.embed_tokens.weight"])[0]
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Non-finite LFM3 output")
        return logits


def generate(root: Path, input_value, *, event, profile_id, config_digest):
    import torch
    from tokenizers import Tokenizer

    from oryn_engine.model_metadata import parse_metadata
    from oryn_engine.reviewed_implementations import installed_index
    from oryn_engine.runtime_routing import canonical_digest

    profile = installed_index().profiles.get(profile_id)
    if (
        profile is None
        or profile["state"] != "active"
        or profile["adapter"] != ADAPTER_ID
        or "bf16-cpu" not in profile["representations"]
    ):
        raise ValueError("Reviewed implementation is unavailable or revoked")
    path = root / "config.json"
    with path.open("rb") as stream:
        config = validate_config(parse_metadata(stream.read(1024 * 1024 + 1)))
    if canonical_digest(config) != config_digest:
        raise ValueError("LFM3 configuration changed after recognition")
    if not isinstance(input_value, dict) or set(input_value) - {
        "messages",
        "maxTokens",
        "temperature",
        "topP",
        "seed",
        "stop",
    }:
        raise ValueError("Invalid LFM3 input")
    temperature, top_p = input_value.get("temperature", 0.7), input_value.get("topP", 0.9)
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or not 0 <= temperature <= 2
        or type(top_p) not in (int, float)
        or not math.isfinite(top_p)
        or not 0 < top_p <= 1
        or input_value.get("stop")
        or "seed" in input_value
    ):
        raise ValueError("LFM3 generation settings are outside the reviewed profile")
    maximum = input_value.get("maxTokens", 128)
    messages = input_value.get("messages")
    if type(maximum) is not int or not 1 <= maximum <= OUTPUT_LIMIT:
        raise ValueError("LFM3 output bound")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 128:
        raise ValueError("LFM3 message bound")
    total = 0
    for message in messages:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in {"system", "user", "assistant"}
            or not isinstance(message["content"], str)
        ):
            raise ValueError("LFM3 message shape")
        content = message["content"]
        total += len(content.encode())
        if total > 64 * 1024:
            raise ValueError("LFM3 text bound")
        if "<|startoftext|>" in content:
            raise ValueError("LFM3 message contains a reserved framing token")
    from oryn_engine.chat_formats import render

    prompt = render("lfm3", messages, bos="<|startoftext|>")
    tokenizer_file = root / "tokenizer.json"
    with tokenizer_file.open("rb") as stream:
        tokenizer_bytes = stream.read(32 * 1024 * 1024 + 1)
    if hashlib.sha256(tokenizer_bytes).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Tokenizer data is outside the reviewed LFM3 profile")
    # Construct from the exact verified bytes, never reopen a mutable pathname.
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    # The checkpoint pads embeddings to 128,000; the reviewed tokenizer has
    # 125,033 actual entries (including its 140 added tokens).
    if not 0 < tokenizer.get_vocab_size() <= config["vocab_size"]:
        raise ValueError("LFM3 tokenizer vocabulary differs")
    tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not tokens or len(tokens) + maximum > CONTEXT_LIMIT:
        raise ValueError("LFM3 context bound")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    graph = Graph(config, load_weights(root, config))
    generated, text, finish = [], "", "length"
    with torch.inference_mode():
        logits = graph.forward(tokens).float()
        for _ in range(maximum):
            if temperature == 0:
                token = int(logits.argmax())
            else:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                ordered, indices = probabilities.sort(descending=True)
                ordered[(ordered.cumsum(-1) - ordered) > top_p] = 0
                token = int(indices[torch.multinomial(ordered, 1)])
            if token == config["eos_token_id"]:
                finish = "stop"
                break
            generated.append(token)
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            # Byte fallback may temporarily end in U+FFFD; defer that suffix.
            stable = decoded.rstrip("\ufffd")
            if not stable.startswith(text):
                raise ValueError("Tokenizer stream is not prefix-stable")
            delta = stable[len(text) :]
            if delta:
                event("job.output.delta", {"channel": "text", "delta": delta})
                text = stable
            if len(generated) < maximum:
                logits = graph.forward([token]).float()
    final = tokenizer.decode(generated, skip_special_tokens=True)
    if final.startswith(text) and final != text:
        event("job.output.delta", {"channel": "text", "delta": final[len(text) :]})
        text = final
    return {
        "text": text,
        "finishReason": finish,
        "backend": "cpu",
        "usage": {"inputTokens": len(tokens), "outputTokens": len(generated)},
    }
