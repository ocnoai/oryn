"""Independent eager equations for offline numerical qualification, not a loader.

Derived from the Apache-2.0 LFM3 reference equations (HuggingFace Team, 2026).
No upstream module is imported. Run only in the contained qualification harness.
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from oryn_engine.native_lfm3 import Graph, load_weights, validate_config


def reference(config, weights, tokens):
    def norm(x, name):
        normalized = x.float() / torch.sqrt(
            x.float().square().mean(-1, keepdim=True) + config["norm_eps"]
        )
        return normalized.to(x.dtype) * weights[name + ".weight"]

    def project(x, name):
        return x @ weights[name + ".weight"].T

    hidden = weights["model.embed_tokens.weight"][torch.tensor([tokens])]
    length = len(tokens)
    for layer, kind in enumerate(config["layer_types"]):
        prefix = f"model.layers.{layer}."
        x = norm(hidden, prefix + "operator_norm")
        if kind == "conv":
            b, c, z = project(x, prefix + "conv.in_proj").transpose(1, 2).chunk(3, dim=1)
            z = b * z
            # Independent causal tap accumulation instead of conv1d/cache code.
            convolved = torch.zeros_like(z)
            kernel = weights[prefix + "conv.conv.weight"][:, 0]
            for tap in range(config["conv_L_cache"]):
                delay = config["conv_L_cache"] - 1 - tap
                if delay < length:
                    convolved[:, :, delay:] += (
                        z[:, :, : length - delay] * kernel[:, tap][None, :, None]
                    )
            operated = project((c * convolved).transpose(1, 2), prefix + "conv.out_proj")
        else:
            heads, kv, dim = (
                config["num_attention_heads"],
                config["num_key_value_heads"],
                config["head_dim"],
            )
            p = prefix + "self_attn."
            q = norm(
                project(x, p + "q_proj").reshape(1, length, heads, dim), p + "q_norm"
            ).transpose(1, 2)
            k = norm(project(x, p + "k_proj").reshape(1, length, kv, dim), p + "k_norm").transpose(
                1, 2
            )
            v = project(x, p + "v_proj").reshape(1, length, kv, dim).transpose(1, 2)
            rope = config["rope_parameters"][kind]
            angles = int(rope.get("partial_rotary_factor", 1) * dim) // 2
            inverse = torch.zeros(dim // 2)
            inverse[:angles] = rope["rope_theta"] ** (-torch.arange(angles).float() * 2 / dim)
            phase = torch.arange(length).float()[:, None] * inverse
            phase = torch.cat((phase, phase), dim=-1)[None, None]
            cosine, sine = phase.cos().to(x.dtype).float(), phase.sin().to(x.dtype).float()

            def rotate(t, cosine=cosine, sine=sine):
                a, b = t.float().chunk(2, dim=-1)
                return (t.float() * cosine + torch.cat((-b, a), dim=-1) * sine).to(t.dtype)

            q, k = rotate(q), rotate(k)
            k, v = k.repeat_interleave(heads // kv, dim=1), v.repeat_interleave(heads // kv, dim=1)
            scores = q @ k.transpose(-1, -2) / math.sqrt(dim)
            row, col = torch.arange(length)[:, None], torch.arange(length)[None, :]
            allowed = col <= row
            if kind == "sliding_attention":
                allowed &= col > row - config["sliding_window"]
            scores = scores.masked_fill(~allowed, -float("inf"))
            attention = torch.softmax(scores.float(), dim=-1).to(q.dtype) @ v
            attention = attention.transpose(1, 2) * torch.sigmoid(
                project(x, p + "gate_proj")
            ).unsqueeze(-1)
            operated = project(attention.reshape(1, length, heads * dim), p + "o_proj")
        hidden = hidden + norm(operated, prefix + "post_operator_norm")
        x = norm(hidden, prefix + "ffn_norm")
        feed = project(
            F.silu(project(x, prefix + "feed_forward.w1")) * project(x, prefix + "feed_forward.w3"),
            prefix + "feed_forward.w2",
        )
        hidden = hidden + norm(feed, prefix + "post_ffn_norm")
    final = norm(hidden, "model.embedding_norm")
    return (final @ weights["model.embed_tokens.weight"].T)[0, -1]


def main():
    root = Path(sys.argv[1])
    config = validate_config(json.loads((root / "config.json").read_text()))
    torch.set_num_threads(1)
    weights = {name: value.float() for name, value in load_weights(root, config).items()}
    tokens = [124894, 124899, 882, 198, 3923, 374, 220, 17, 489, 220, 17, 30]
    with torch.inference_mode():
        actual = Graph(config, weights).forward(tokens)
        expected = reference(config, weights, tokens)
        difference = float((actual - expected).abs().max())
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.002)
        print(
            json.dumps(
                {
                    "referenceMaxAbsoluteLogitDifference": difference,
                    "ownedArgmax": int(actual.argmax()),
                    "referenceArgmax": int(expected.argmax()),
                }
            )
        )
        cache = Graph(config, weights)
        cache.forward(tokens[:-1])
        decoded = cache.forward(tokens[-1:])
        torch.testing.assert_close(actual, decoded, atol=0.002, rtol=0.002)
        print(
            json.dumps(
                {"prefillDecodeMaxAbsoluteLogitDifference": float((actual - decoded).abs().max())}
            )
        )


if __name__ == "__main__":
    main()
