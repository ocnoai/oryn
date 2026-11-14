"""Complete random native graph; independent NumPy affine/Hadamard reference."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from oryn_engine import mlx_bolding as adapter

FIXTURES = Path(__file__).parent / "fixtures/mlx_bolding"


def original(name):
    return json.loads((FIXTURES / name).read_text())


def sylvester(n):
    h = np.ones((1, 1), np.float32)
    while len(h) < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def tiny_fixture(root):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    pytest.importorskip("mlx_vlm")
    from mlx.utils import tree_flatten, tree_unflatten
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

    root.mkdir(parents=True, exist_ok=True)
    c = original("config.json")
    t = c["text_config"]
    t.update(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        full_attention_interval=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=384,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=64,
        max_position_embeddings=4096,
        layer_types=["linear_attention", "full_attention"],
        bos_token_id=258,
        eos_token_id=258,
    )
    t["rope_parameters"]["mrope_section"] = [3, 3, 2]
    vocab = {s: i for i, s in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab.update({s: 256 + i for i, s in enumerate(adapter.mlx_prism.MARKERS)})
    vocab.update({f"[unused-{i}]": i for i in range(259, 384)})
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(
        [AddedToken(s, normalized=False, special=True) for s in adapter.mlx_prism.MARKERS]
    )
    tok.add_tokens(
        [AddedToken(f"[unused-{i}]", normalized=False, special=False) for i in range(381, 384)]
    )
    tok.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(json.dumps(original("tokenizer_config.json")))
    (root / "generation_config.json").write_text(
        json.dumps(dict(bos_token_id=258, pad_token_id=258, eos_token_id=[257, 258]))
    )
    mx.random.seed(6104)
    ref = adapter.native_model(c)
    mx.eval(ref.parameters())
    weights = dict(tree_flatten(ref.parameters()))
    rng = np.random.default_rng(6104)
    packed = adapter.packed_specs(c)
    widths = sorted({shape[1] for shape, bits, _ in packed.values() if bits == 2})
    signs = {w: rng.choice([-1, 1], w).astype(np.float32) for w in widths}

    class Reference(nn.Module):
        def __init__(self, dense, embedding, sign=None):
            super().__init__()
            if sign is not None:
                dense = (dense.reshape(-1, 128) @ sylvester(128)).reshape(dense.shape) * sign
            self.weight = mx.array(dense.astype(np.float32))
            self.embedding = embedding

        def __call__(self, x):
            return self.weight[x] if self.embedding else x @ self.weight.T

    replacements = []
    for name, ((rows, cols), bits, group) in packed.items():
        q = rng.integers(0, 2**bits, (rows, cols), dtype=np.uint32)
        lanes = 32 // bits
        words = np.bitwise_or.reduce(
            q.reshape(rows, -1, lanes) << (np.arange(lanes, dtype=np.uint32) * bits), axis=-1
        )
        scale = np.full((rows, cols // group), 0.06 / (2**bits - 1), np.float32)
        bias = np.full_like(scale, -0.03)
        dense = q.astype(np.float32) * np.repeat(scale, group, 1) + np.repeat(bias, group, 1)
        weights[name + ".weight"] = mx.array(words)
        weights[name + ".scales"], weights[name + ".biases"] = mx.array(scale), mx.array(bias)
        replacements.append(
            (
                name,
                Reference(
                    dense, name.endswith(".embed_tokens"), signs[cols] if bits == 2 else None
                ),
            )
        )
    ref.update_modules(tree_unflatten(replacements))
    ref.eval()
    _, norms = adapter.parameter_specs(c)
    for name in weights:
        if name in norms:
            weights[name] = weights[name] - 1
        elif name.endswith(".conv1d.weight"):
            weights[name] = mx.moveaxis(weights[name], 1, 2)
    mx.save_safetensors(str(root / "model-00001.safetensors"), weights)
    mx.save_safetensors(
        str(root / adapter.SIGN_FILE), {f"sign_{w}": mx.array(a) for w, a in signs.items()}
    )
    (root / "config.json").write_text(json.dumps(c))
    manifest, mapping = original("model-manifest.json"), original("tensor-map.json")
    original_shapes = adapter.parameter_specs(original("config.json"))[0]
    shapes = adapter.parameter_specs(c)[0]

    # Keep publisher declaration semantics, shrink only dimensions/layer indices.
    def source_name(name):
        return name.replace("model.layers.1.", "model.layers.3.")

    records = {r["runtime_module_path"]: r for r in manifest["quantized_modules"]}
    manifest["quantized_modules"] = []
    for name, (shape, bits, _) in packed.items():
        if bits != 2:
            continue
        record = json.loads(json.dumps(records[source_name(name)]))
        record.update(
            runtime_module_path=name,
            runtime_parameter_path=name + ".weight",
            sign_width=shape[1],
            target_shape=list(shape),
            source_model_shape=list(shape),
        )
        manifest["quantized_modules"].append(record)
    manifest["hadamard"].update(
        block_size=128,
        sign_widths=widths,
        folded_module_count=len(manifest["quantized_modules"]) - 1,
        sign_vector_sha256={
            str(w): hashlib.sha256(s.tobytes()).hexdigest() for w, s in signs.items()
        },
    )
    mapping["hadamard"].update(
        {k: manifest["hadamard"][k] for k in mapping["hadamard"] if k in manifest["hadamard"]}
    )
    mapped = {r["runtime_parameter_path"]: r for r in mapping["mappings"]}
    mapping["mappings"] = []
    for name, shape in shapes.items():
        source = source_name(name)
        assert source in original_shapes
        record = json.loads(json.dumps(mapped[source]))
        record.update(
            runtime_parameter_path=name,
            runtime_module_path=name.removesuffix(".weight"),
            target_shape=list(shape),
            runtime_shape=list(shape),
        )
        mapping["mappings"].append(record)
    mapping.update(
        base_layer_count=2,
        vocabulary=dict(
            model_vocab_size=384,
            source_vocab_size=384,
            unused_token_count=3,
            unused_token_id_range=[381, 384],
        ),
    )
    index = dict(metadata={}, weight_map={n: "model-00001.safetensors" for n in weights})
    for name, value in [
        ("model-manifest.json", manifest),
        ("tensor-map.json", mapping),
        ("model.safetensors.index.json", index),
    ]:
        (root / name).write_text(json.dumps(value))
    return c, ref, weights
