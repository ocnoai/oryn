"""Closed original metadata and complete generated graphs; no publisher model weights."""

import json
from pathlib import Path

import numpy as np
import pytest

from oryn_engine import mlx_prism as adapter

FIXTURES = Path(__file__).parent / "fixtures/mlx_prism"
MODES = ("binary", "ternary", "hadamard")


def config_for(mode):
    return json.loads((FIXTURES / mode / "config.json").read_text())


def sylvester(n):
    h = np.ones((1, 1), np.float32)
    while len(h) < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def unpack(weight, scales, biases, bits):
    q = (
        (np.asarray(weight)[..., None] >> (np.arange(32 // bits, dtype=np.uint32) * bits))
        & ((1 << bits) - 1)
    ).reshape(weight.shape[0], -1)
    return q.astype(np.float32) * np.repeat(
        np.asarray(scales).astype(np.float32), 128, axis=1
    ) + np.repeat(np.asarray(biases).astype(np.float32), 128, axis=1)


def tiny_fixture(root, mode):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    pytest.importorskip("mlx_vlm")
    from mlx.utils import tree_flatten, tree_unflatten
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

    root.mkdir(parents=True, exist_ok=True)
    c = config_for(mode)
    text = c["text_config"]
    text.update(
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
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=4096,
    )
    text["rope_parameters"]["mrope_section"] = [3, 3, 2]
    c["vision_config"].update(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        num_position_embeddings=16,
        out_hidden_size=128,
    )
    vocab = {t: i for i, t in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab.update({t: 256 + i for i, t in enumerate(adapter.MARKERS)})
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([AddedToken(t, normalized=False, special=True) for t in adapter.MARKERS])
    tok.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            dict(
                tokenizer_class="TokenizersBackend",
                eos_token=adapter.MARKERS[1],
                pad_token=adapter.MARKERS[2],
                bos_token=None,
            )
        )
    )
    text["bos_token_id"] = vocab[adapter.MARKERS[2]]
    text["eos_token_id"] = vocab[adapter.MARKERS[2] if mode == "hadamard" else adapter.MARKERS[1]]
    if mode != "hadamard":
        c["eos_token_id"] = text["eos_token_id"]
    else:
        c["modules"] = [
            dict(path=n, block=128, dtype="float16", embedding=n == "model.embed_tokens")
            for n in adapter.transform_specs(c)
        ]
        generation = dict(bos_token_id=258, pad_token_id=258, eos_token_id=[257, 258])
        (root / "generation_config.json").write_text(json.dumps(generation))
    mx.random.seed(6161)
    model = adapter.native_model(c)
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    weights = dict(tree_flatten(model.parameters()))
    selected = adapter.packed_modules(model, c)
    rng = np.random.default_rng(6161)
    signs = {
        width: rng.choice([-1, 1], width).astype(np.float32)
        for width in sorted(set(adapter.transform_specs(c).values()))
    }
    if mode == "hadamard":
        h = json.loads((FIXTURES / "hadamard/hadamard.json").read_text())
        h.update(
            {
                "prism.hadamard.block_size": 128,
                "prism.hadamard.weight_names": [
                    "language_model." + n + ".weight"
                    for n in adapter.transform_specs(c)
                    if n != "model.embed_tokens"
                ],
                "prism.hadamard.sign_widths": list(signs),
                "prism.hadamard.sign_values": [float(v) for a in signs.values() for v in a],
            }
        )
        (root / "hadamard.json").write_text(json.dumps(h))

    class Reference(nn.Module):
        def __init__(self, dense, embedding, sign=None):
            super().__init__()
            self.weight = mx.array(dense, dtype=mx.float16)
            self.embedding = embedding
            if sign is not None:
                self.sign = mx.array(sign)
                self.h = mx.array(sylvester(128).astype(np.float32))

        def __call__(self, x):
            if self.embedding:
                out = self.weight[x]
                if "sign" in self:
                    out = (out.astype(mx.float32).reshape(-1, 128) @ self.h).reshape(
                        out.shape
                    ) * self.sign
                return out.astype(mx.float16)
            if "sign" in self:
                x = (
                    ((x.astype(mx.float32) * self.sign).reshape(-1, 128) @ self.h)
                    .reshape(x.shape)
                    .astype(x.dtype)
                )
            return x @ self.weight.T

    refs = []
    bits = c["quantization"]["bits"]
    for name, module in selected.items():
        rows, cols = module.weight.shape
        q = rng.integers(0, 2**bits, (rows, cols), dtype=np.uint32)
        packed = np.bitwise_or.reduce(
            q.reshape(rows, -1, 32 // bits) << (np.arange(32 // bits, dtype=np.uint32) * bits),
            axis=-1,
        )
        scales = np.full((rows, cols // 128), 0.02, np.float16)
        biases = np.full_like(scales, -0.02 if bits == 1 else -0.03)
        weights[name + ".weight"] = mx.array(packed)
        weights[name + ".scales"], weights[name + ".biases"] = mx.array(scales), mx.array(biases)
        sign = signs[cols] if mode == "hadamard" else None
        if sign is not None:
            weights[name + ".signs"] = mx.array(sign)
        refs.append(
            (
                name,
                Reference(
                    unpack(packed, scales, biases, bits), isinstance(module, nn.Embedding), sign
                ),
            )
        )
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    (root / "config.json").write_text(json.dumps(c))
    model.update_modules(tree_unflatten(refs))
    model.eval()
    return c, model, weights


@pytest.mark.parametrize("mode", MODES)
def test_original_closed_config_and_complete_native_header(mode):
    pytest.importorskip("mlx_vlm")
    c = config_for(mode)
    adapter.validate_config(c)
    header = json.loads((FIXTURES / mode / "model.safetensors.header.json").read_text())
    shapes = {k: tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"}
    selected, _ = adapter.validate_layout(adapter.native_model(c), c, shapes)
    assert len(selected) == (402 if mode == "hadamard" else 498)
    assert sum(n.startswith("vision_tower.") for n in shapes) == 333
    if mode == "hadamard":
        assert list(
            adapter.validate_hadamard(
                c, json.loads((FIXTURES / mode / "hadamard.json").read_text())
            )
        ) == [5120, 6144, 17408]


@pytest.mark.parametrize("bits", [0, 1, 31, 32, 0xAAAAAAAA, 0x55555555, 0xFFFFFFFF])
def test_one_bit_lane_repacking_preserves_values_and_affine_bias(bits):
    mx = pytest.importorskip("mlx.core")
    words = np.array([[bits, 0x80000001, 0xC0FFEE00, 0x12345678]], np.uint32)
    scales = mx.array([[0.25]], dtype=mx.float16)
    biases = mx.array([[-1.75]], dtype=mx.float16)
    actual = mx.dequantize(
        adapter.widen_binary(mx.array(words)), scales, biases, group_size=128, bits=2
    )
    np.testing.assert_array_equal(np.asarray(actual), unpack(words, scales, biases, 1))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cached", [False, True])
def test_complete_graph_logits_and_vision_match_independent_dense_reference(tmp_path, mode, cached):
    mx = pytest.importorskip("mlx.core")
    c, reference, weights = tiny_fixture(tmp_path, mode)
    actual = adapter.load_model(tmp_path, c)
    tokens = mx.array([[12, 31, 76, 80]], dtype=mx.int32)
    if cached:
        actual_cache, reference_cache = (
            actual.language_model.make_cache(),
            reference.language_model.make_cache(),
        )
        actual.language_model(tokens[:, :3], cache=actual_cache)
        reference.language_model(tokens[:, :3], cache=reference_cache)
        a = actual.language_model(tokens[:, 3:], cache=actual_cache).logits
        b = reference.language_model(tokens[:, 3:], cache=reference_cache).logits
    else:
        a = actual.language_model(tokens).logits
        b = reference.language_model(tokens).logits
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=0.008, rtol=0.01)
    assert np.isfinite(np.asarray(a)).all()
    from mlx.utils import tree_flatten

    loaded = dict(tree_flatten(actual.parameters()))
    for name in weights:
        if name.startswith("vision_tower."):
            np.testing.assert_array_equal(np.asarray(loaded[name]), np.asarray(weights[name]))
    assert adapter.load_tokenizer(tmp_path, c)[1] == ({257, 258} if mode == "hadamard" else {257})


@pytest.mark.parametrize("mode", MODES)
def test_owned_prompt_matches_publisher_non_thinking_template(mode):
    from transformers.utils.chat_template_utils import _compile_jinja_template

    template = _compile_jinja_template((FIXTURES / mode / "chat_template.jinja").read_text())
    for messages in [
        [dict(role="user", content="Hello")],
        [dict(role="system", content="Be concise"), dict(role="user", content="你好")],
        [
            dict(role="user", content="Hi"),
            dict(role="assistant", content="Hello"),
            dict(role="user", content="More"),
        ],
    ]:
        assert adapter.prompt(messages) == template.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=False,
            preserve_thinking=False,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"custom_attn": True},
        {"requires_runtime": "runtime/artifact.py"},
        {"components": {"mtp": True}},
        {"auto_map": {"AutoModel": "repo.Code"}},
        {"language_model_only": True},
        {"quantization": {"bits": 3, "group_size": 128}},
        {"quantization": {"bits": True, "group_size": 128}},
        {"quantization": {"bits": 1, "group_size": 64}},
        {"quantization": {"bits": 1, "group_size": 128, "mode": "mxfp4"}},
    ],
)
def test_unreviewed_root_options_fail_closed(changes):
    with pytest.raises(ValueError):
        adapter.validate_config(config_for("binary") | changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"hidden_size": 129},
        {"num_hidden_layers": True},
        {"head_dim": 33},
        {"attention_bias": True},
        {"num_key_value_heads": 5},
        {"attn_output_gate": False},
        {"output_gate_type": "other"},
        {"use_cache": False},
        {"mtp_use_dedicated_embeddings": True},
        {"mtp_num_hidden_layers": 2},
        {"attention_dropout": 0.1},
        {"layer_types": ["full_attention"] * 64},
        {"rope_parameters": {"rope_type": "yarn"}},
        {"unreviewed": True},
    ],
)
def test_unreviewed_text_geometry_fails_closed(changes):
    c = config_for("binary")
    c["text_config"].update(changes)
    with pytest.raises(ValueError):
        adapter.validate_config(c)


@pytest.mark.parametrize(
    "field,value",
    [
        ("prism.hadamard.sign_mode", "generated"),
        ("prism.hadamard.sign_widths", [5120]),
        ("prism.hadamard.block_size", 0),
        ("prism.hadamard.inverse_weight_names", []),
        ("prism.hadamard.weight_names", []),
        ("prism.hadamard.sign_values", [0]),
        ("prism.hadamard.gdn_v_grouped", False),
        ("prism.hadamard.version", 2),
    ],
)
def test_hadamard_metadata_is_bound_to_every_declared_module(field, value):
    h = json.loads((FIXTURES / "hadamard/hadamard.json").read_text())
    h[field] = value
    with pytest.raises(ValueError):
        adapter.validate_hadamard(config_for("hadamard"), h)


@pytest.mark.parametrize(
    "extra",
    [
        "crystal.safetensors",
        "causal/manifest.jsonl",
        "bundle.json",
        "crystal_runtime/loader.whl",
        "vision.safetensors",
    ],
)
def test_modified_attention_or_extra_graph_artifacts_are_not_base_family(extra):
    with pytest.raises(ValueError):
        adapter.validate_files(
            ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors", extra],
            config_for("binary"),
        )


@pytest.mark.parametrize("content", ["<|im_end|>", "<think>draft", "</think>", "|>"])
def test_user_controls_do_not_change_owned_framing(content):
    with pytest.raises(ValueError):
        adapter.prompt([dict(role="user", content=content)])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("damage", [None, "tokenizer", "missing", "extra"])
def test_discovery_and_profile_bind_original_metadata_without_weights(tmp_path, mode, damage):
    from types import SimpleNamespace

    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore

    config = config_for(mode)

    class Hub(FakeHub):
        def read_json(self, repo, revision, name):
            if name == "config.json":
                return config
            value = json.loads((FIXTURES / mode / name).read_text())
            if damage == "tokenizer" and name == "tokenizer_config.json":
                value["auto_map"] = {"AutoTokenizer": "repo.Code"}
            return value

    names = ["config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json"]
    if mode == "hadamard":
        names += ["hadamard.json", "generation_config.json"]
    if damage == "missing":
        names.remove("tokenizer_config.json")
    if damage == "extra":
        names.append("crystal.safetensors")
    hub = Hub(
        config=config, pipeline_tag="text-generation", files=[(n, 100, "a" * 64) for n in names]
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda p: None if p.provider else object()
    )
    result = store.inspect(model("independent/renamed-prism"))
    assert result["executable"] is (damage is None), result
    if damage is None:
        assert result["taskProfile"]["limits"]["contextTokens"] == 4096
    assert not hub.downloaded


@pytest.mark.parametrize(
    "damage",
    [
        "missing-vision",
        "missing-language",
        "extra",
        "shape",
        "packed-dtype",
        "nonfinite",
        "sign-binding",
    ],
)
def test_complete_graph_rejects_corrupted_or_unbound_weights(tmp_path, damage):
    mx = pytest.importorskip("mlx.core")
    c, _, weights = tiny_fixture(tmp_path, "hadamard")
    if damage == "missing-vision":
        del weights[next(k for k in weights if k.startswith("vision_tower."))]
    elif damage == "missing-language":
        del weights["language_model.model.norm.weight"]
    elif damage == "extra":
        weights["extra.weight"] = mx.ones((1,))
    elif damage == "shape":
        weights["language_model.lm_head.weight"] = weights["language_model.lm_head.weight"][:, :-1]
    elif damage == "packed-dtype":
        weights["language_model.lm_head.weight"] = weights["language_model.lm_head.weight"].astype(
            mx.float32
        )
    elif damage == "nonfinite":
        weights["language_model.model.norm.weight"] = mx.full((128,), float("nan"))
    elif damage == "sign-binding":
        weights["language_model.lm_head.signs"] = -weights["language_model.lm_head.signs"]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    with pytest.raises(ValueError):
        adapter.load_model(tmp_path, c)


@pytest.mark.parametrize(
    "damage", ["missing-eos", "sparse", "wrong-metadata", "stop-mismatch", "symlink"]
)
def test_raw_tokenizer_and_stop_contract_are_not_silently_repaired(tmp_path, damage):
    pytest.importorskip("mlx.core")
    c, _, _ = tiny_fixture(tmp_path, "hadamard")
    path = tmp_path / "tokenizer.json"
    if damage == "wrong-metadata":
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "Unreviewed"})
        )
    elif damage == "stop-mismatch":
        (tmp_path / "generation_config.json").write_text(
            json.dumps(dict(bos_token_id=258, pad_token_id=258, eos_token_id=[258]))
        )
    elif damage == "symlink":
        saved = tmp_path / "real.json"
        path.rename(saved)
        path.symlink_to(saved)
    else:
        t = json.loads(path.read_text())
        if damage == "missing-eos":
            del t["model"]["vocab"]["<|im_end|>"]
            t["added_tokens"] = [v for v in t["added_tokens"] if v["content"] != "<|im_end|>"]
        else:
            t["model"]["vocab"]["!"] = 999
        path.write_text(json.dumps(t))
    with pytest.raises(ValueError) as error:
        adapter.load_tokenizer(tmp_path, c)
    assert error.value


@pytest.mark.parametrize("mode", ["binary", "hadamard"])
def test_offline_worker_probe_and_repeated_real_native_generation(tmp_path, mode):
    import os
    import subprocess
    import sys

    mx = pytest.importorskip("mlx.core")
    from oryn_engine.runtime_packs import mlx_lm_worker

    _, _, weights = tiny_fixture(tmp_path, mode)
    for suffix in ("scales", "biases"):
        name = "language_model.lm_head." + suffix
        weights[name] = mx.zeros_like(weights[name])
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    payload = dict(
        modelRoot=str(tmp_path),
        input=dict(messages=[dict(role="user", content="Hi")], maxTokens=3, temperature=0, topP=1),
    )
    expected = mlx_lm_worker.generate(payload)
    assert expected == dict(text="!!!", backend="mlx")
    for command, body in [("probe", {}), ("generate", payload), ("generate", payload)]:
        envelope = dict(
            abi="1", packId="oryn.mlx-lm", version="0.11.0", command=command, payload=body
        )
        run = subprocess.run(
            [sys.executable, "-m", "oryn_engine.runtime_packs.mlx_lm_worker"],
            input=json.dumps(envelope) + "\n",
            text=True,
            capture_output=True,
            timeout=60,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        assert run.returncode == 0, run.stderr
        result = [json.loads(line) for line in run.stdout.splitlines()][-1]
        assert result["ok"], (result, run.stderr)
        assert result["result"] == (
            dict(backend="mlx", available=True) if command == "probe" else expected
        )


@pytest.mark.parametrize(
    "fault",
    [
        "padded-token",
        "reserved-output",
        "too-long",
        "empty",
        "context",
        "output-limit",
        "nonfinite",
    ],
)
def test_generation_rejects_invalid_output_and_resource_limits(tmp_path, monkeypatch, fault):
    from types import SimpleNamespace

    mx = pytest.importorskip("mlx.core")
    mlx_lm = pytest.importorskip("mlx_lm")

    c, _, _ = tiny_fixture(tmp_path, "binary")
    monkeypatch.setattr(adapter, "load_model", lambda *_: object())
    monkeypatch.setattr(adapter, "text_view", lambda m: m)
    value = dict(messages=[dict(role="user", content="Hi")], maxTokens=3, temperature=0, topP=1)
    ids = [0, 0, 0]
    if fault == "padded-token":
        ids = [383]
    if fault == "reserved-output":
        ids = [256]
    if fault == "too-long":
        ids = [0] * 65537
    if fault == "empty":
        ids = []
    if fault == "context":
        value["messages"][0]["content"] = "x" * 4096
    if fault == "output-limit":
        value["maxTokens"] = 513
    if fault == "nonfinite":
        monkeypatch.undo()

        class Broken:
            def __call__(self, *_, **__):
                return SimpleNamespace(logits=mx.array([[[float("nan")]]]))

        view = adapter.text_view(SimpleNamespace(language_model=Broken()))
        with pytest.raises(ValueError, match="NONFINITE"):
            view(mx.array([[1]]))
        return
    monkeypatch.setattr(
        mlx_lm, "stream_generate", lambda *_, **__: iter(SimpleNamespace(token=i) for i in ids)
    )
    with pytest.raises(ValueError):
        adapter.generate(tmp_path, c, value, lambda *_: None)
