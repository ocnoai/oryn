"""Complete generated Kolibri graph and independently unpacked reference weights."""

import json
from pathlib import Path

import numpy as np
import pytest

from oryn_engine import mlx_kolibri as adapter

FIXTURES = Path(__file__).parent / "fixtures/mlx_kolibri"


def original(name):
    return json.loads((FIXTURES / name).read_text())


def tiny_config():
    c = original("config.json")
    c.update(
        hidden_size=64,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        vocab_size=384,
        sliding_window=9,
        max_position_embeddings=4096,
        eos_token_id=257,
        pad_token_id=258,
        layer_types=["sliding_attention"] * 4 + ["full_attention"],
    )
    q = {"bits": 4, "group_size": 64, "mode": "affine"}
    for name in c["quantization"]:
        if name in {"bits", "group_size", "mode"}:
            continue
        if name.startswith("model.layers.") and int(name.split(".")[2]) >= 5:
            continue
        q[name] = {"bits": 8, "group_size": 64, "mode": "affine"}
    c["quantization"] = q
    c["quantization_config"] = json.loads(json.dumps(q))
    return c


def tiny_fixture(root, dtype="float32"):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    import torch
    from mlx.utils import tree_flatten
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

    root.mkdir(parents=True, exist_ok=True)
    c = tiny_config()
    vocab = {s: i for i, s in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    markers = [*adapter.MARKERS, "<think>", "</think>"]
    vocab.update({s: 256 + i for i, s in enumerate(markers)})
    # Interior holes exercise sparse IDs too; the original's two holes are at the tail.
    vocab.update({f"[reserved-{i}]": i for i in range(261, 384) if i not in {376, 380}})
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_tokens([AddedToken(s, special=False, normalized=False) for s in markers[3:]])
    tok.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in markers[:3]])
    tok.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(json.dumps(original("tokenizer_config.json")))
    (root / "generation_config.json").write_text(
        json.dumps(dict(eos_token_id=[257, 258], pad_token_id=258))
    )
    mx.random.seed(6105)
    model = adapter.native_model(c)
    model.set_dtype(getattr(mx, dtype))
    for i, layer in enumerate(model.layers):
        layer.mlp.expert_bias = mx.array([-0.12, 0.01, 0.07, 0.13], dtype=mx.float32) + i * 0.001
        if dtype == "bfloat16":
            # Avoid a near-tied top-k boundary when comparing rounded BF16
            # activations to the independent F32 reference. The unconstrained
            # F32 cases and adversarial router case exercise varying experts.
            layer.mlp.gate.weight *= 0.02
            layer.mlp.expert_bias = mx.array([-0.75, -0.25, 0.25, 0.75])
    mx.eval(model.parameters())
    weights = dict(tree_flatten(model.parameters()))
    dense = {n: np.asarray(v.astype(mx.float32)).copy() for n, v in weights.items()}
    rng = np.random.default_rng(6105)
    for name, (shape, bits) in adapter.packed_specs(c).items():
        q = rng.integers(0, 2**bits, shape, dtype=np.uint32)
        lanes = 32 // bits
        words = np.bitwise_or.reduce(
            q.reshape(*shape[:-1], -1, lanes) << (np.arange(lanes, dtype=np.uint32) * bits), axis=-1
        )
        scale = mx.full(
            (*shape[:-1], shape[-1] // 64), 0.08 / (2**bits - 1), dtype=getattr(mx, dtype)
        )
        bias = mx.full(scale.shape, -0.04, dtype=getattr(mx, dtype))
        weights[name + ".weight"] = mx.array(words)
        weights[name + ".scales"], weights[name + ".biases"] = scale, bias
        dense[name + ".weight"] = q.astype(np.float32) * np.repeat(
            np.asarray(scale.astype(mx.float32)), 64, -1
        ) + np.repeat(np.asarray(bias.astype(mx.float32)), 64, -1)
    filename = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(root / filename), weights)
    (root / "config.json").write_text(json.dumps(c))
    (root / "model.safetensors.index.json").write_text(
        json.dumps(dict(metadata={}, weight_map={n: filename for n in weights}))
    )
    reference = {}
    for name, value in dense.items():
        if ".switch_mlp." in name:
            for expert, matrix in enumerate(value):
                reference[name.replace(".switch_mlp.", f".experts.{expert}.")] = torch.from_numpy(
                    matrix.copy()
                )
        else:
            name = name.replace(".mlp.expert_bias", ".moe.router.expert_bias")
            reference[name] = torch.from_numpy(value.copy())
    return c, reference, weights
