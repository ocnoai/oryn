"""Frozen metadata plus complete small graphs, without original learned weights."""

import json

import numpy as np
import pytest
from bolding_fixture import original, tiny_fixture

from oryn_engine import mlx_bolding as adapter


def test_every_original_array_and_manifest_matches_without_native_imports():
    config = original("config.json")
    headers = original("headers.json")
    files = list(adapter.METADATA) + list(headers) + ["model.py", "tokenization_bolding_mtp.py"]
    adapter.inspect_metadata(config, original, headers.__getitem__, files)
    assert len(adapter.serialized_shapes(config)) == 1686
    assert len(adapter.packed_specs(config)) == 410
    assert sum(bits == 8 for _, bits, _ in adapter.packed_specs(config).values()) == 8
    assert (
        len(
            adapter.parameter_specs(config)[1]
            & {n for n in adapter.serialized_shapes(config) if n.startswith(adapter.MTP)}
        )
        == 7
    )
    assert not any(n.endswith(".py") for n in adapter.selected_files(files, config))


def test_original_metadata_geometry_equals_complete_pinned_native_graph():
    pytest.importorskip("mlx_vlm")
    from mlx.utils import tree_flatten

    c = original("config.json")
    before = json.loads(json.dumps(c))
    shapes, _ = adapter.parameter_specs(c)
    expected = {
        n: (s[0], s[2], s[1]) if n.endswith(".conv1d.weight") else s for n, s in shapes.items()
    }
    actual = {n: tuple(t.shape) for n, t in tree_flatten(adapter.native_model(c).parameters())}
    assert actual == expected  # Lazy arrays; no original-size allocation/evaluation.
    assert c == before  # Native config constructors must not mutate frozen metadata.


@pytest.mark.parametrize(
    "changes",
    [
        {"model_file": "other.py"},
        {"auto_map": {"AutoModel": "repo.Code"}},
        {"vision_config": {}},
        {"language_model_only": False},
        {"unreviewed": 1},
        {"oryn_packed_layout": adapter.LAYOUT},
        {"source_sha256": "../source"},
        {"architectures": ["Unreviewed"]},
        {"tie_word_embeddings": True},
    ],
)
def test_closed_root_binding_rejects_changed_graph_or_executable_options(changes):
    with pytest.raises(ValueError):
        adapter.validate_config(original("config.json") | changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"mtp_num_hidden_layers": 0},
        {"mtp_num_hidden_layers": True},
        {"mtp_use_dedicated_embeddings": True},
        {"linear_num_key_heads": 5},
        {"hidden_size": 129},
        {"linear_value_head_dim": 33},
        {"head_dim": 63},
        {"layer_types": ["full_attention"] * 64},
        {"output_gate_type": "other"},
        {"attn_output_gate": False},
        {"rope_parameters": {"rope_type": "yarn"}},
    ],
)
def test_closed_text_geometry(changes):
    c = original("config.json")
    c["text_config"].update(changes)
    with pytest.raises(ValueError):
        adapter.validate_config(c)


@pytest.mark.parametrize(
    "damage", ["missing-draft", "backbone-bits", "draft-bits", "draft-group", "unknown-module"]
)
def test_mixed_precision_is_bound_per_module(damage):
    c = original("config.json")
    q = c["quantization"]
    if damage == "missing-draft":
        del q[adapter.MTP + "fc"]
    elif damage == "backbone-bits":
        q["bits"] = 4
    elif damage == "draft-bits":
        q[adapter.MTP + "fc"]["bits"] = 2
    elif damage == "draft-group":
        q[adapter.MTP + "fc"]["group_size"] = 128
    else:
        q["language_model.model.layers.0.mlp.gate_proj"] = {"bits": 8, "group_size": 32}
    with pytest.raises(ValueError):
        adapter.validate_config(c)


@pytest.mark.parametrize(
    "damage", ["missing-draft", "extra", "shape", "dtype", "wrong-shard", "duplicate"]
)
def test_all_original_header_arrays_and_shard_ownership_are_required(damage):
    h = original("headers.json")
    h.pop(adapter.SIGN_FILE)
    index = original("model.safetensors.index.json")
    name = adapter.MTP + "fc.weight"
    shard = index["weight_map"][name]
    if damage == "missing-draft":
        del h[shard]["tensors"][name]
    elif damage == "extra":
        h[shard]["tensors"]["unbound.weight"] = dict(shape=[1], dtype="F32")
    elif damage == "shape":
        h[shard]["tensors"][name]["shape"][-1] -= 1
    elif damage == "dtype":
        h[shard]["tensors"][name]["dtype"] = "F32"
    elif damage == "wrong-shard":
        index["weight_map"][name] = "model-00009.safetensors"
    else:
        h["model-00009.safetensors"]["tensors"][name] = h[shard]["tensors"][name]
    with pytest.raises(ValueError):
        adapter.validate_headers(original("config.json"), index, h)


@pytest.mark.parametrize(
    "damage",
    [
        "inverse-embedding",
        "module-missing",
        "duplicate",
        "sign-width",
        "block",
        "grouped",
        "norm-offset",
        "conv-layout",
        "unknown-transform",
        "conflicting-hadamard",
        "malformed-manifest",
        "unused-stops",
        "vocabulary",
        "index",
    ],
)
def test_manifest_cannot_omit_or_reinterpret_transforms(damage):
    m, t, i = [
        original(n)
        for n in ("model-manifest.json", "tensor-map.json", "model.safetensors.index.json")
    ]
    if damage == "inverse-embedding":
        m["quantized_modules"][-1]["hadamard_role"] = "folded_weight_input_transform"
    elif damage == "module-missing":
        m["quantized_modules"].pop()
    elif damage == "duplicate":
        m["quantized_modules"][-1] = m["quantized_modules"][0]
    elif damage == "sign-width":
        m["hadamard"]["sign_widths"] = [5120]
    elif damage == "block":
        m["hadamard"]["block_size"] = 0
    elif damage == "grouped":
        m["hadamard"]["gdn_v_grouped"] = False
    elif damage == "norm-offset":
        next(
            r for r in t["mappings"] if r["runtime_parameter_path"] == adapter.MTP + "norm.weight"
        )["transformations"] = []
    elif damage == "conv-layout":
        next(r for r in t["mappings"] if r["runtime_parameter_path"].endswith("conv1d.weight"))[
            "transformations"
        ] = []
    elif damage == "unknown-transform":
        t["mappings"][0]["transformations"].append("unreviewed_permutation")
    elif damage == "conflicting-hadamard":
        t["hadamard"]["block_size"] = 128
    elif damage == "malformed-manifest":
        m["hadamard"] = []
    elif damage == "unused-stops":
        t["vocabulary"].update(unused_token_id_range=[248044, 248320], unused_token_count=276)
    elif damage == "vocabulary":
        t["vocabulary"]["unused_token_id_range"][0] -= 1
    else:
        del i["weight_map"][adapter.MTP + "norm.weight"]
    with pytest.raises(ValueError):
        adapter.validate_manifest(original("config.json"), m, t, i)


@pytest.mark.parametrize(
    "damage", [None, "metadata", "draft", "sign", "extra", "tokenizer", "index"]
)
def test_discovery_uses_metadata_only_and_exposes_bounded_chat(tmp_path, damage):
    from types import SimpleNamespace

    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore

    config, headers = original("config.json"), original("headers.json")
    names = list(adapter.METADATA) + list(headers) + ["model.py", "tokenization_bolding_mtp.py"]
    if damage == "metadata":
        names.remove("tensor-map.json")
    if damage == "extra":
        names.append("other.safetensors")

    class Hub(FakeHub):
        def read_json(self, repo, revision, name):
            value = original(name)
            if name == "tokenizer_config.json" and damage == "tokenizer":
                value["auto_map"] = {"AutoTokenizer": [None, "unreviewed.Code"]}
            if name == "model.safetensors.index.json" and damage == "index":
                value["weight_map"][adapter.MTP + "fc.weight"] = "../model.safetensors"
            return value

        def read_tensor_structure(self, repo, revision, name):
            value = headers[name]
            if damage == "sign" and name == adapter.SIGN_FILE:
                value["tensors"] = {}
            if damage == "draft":
                value["tensors"].pop(adapter.MTP + "fc.weight", None)
            return value

    hub = Hub(
        config=config, pipeline_tag="text-generation", files=[(n, 100, "a" * 64) for n in names]
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda p: None if p.provider else object()
    )
    result = store.inspect(model("independent/renamed-mtp"))
    assert result["executable"] is (damage is None), result
    if damage is None:
        assert result["taskProfile"]["limits"]["contextTokens"] == 4096
        assert result["runtimePack"]["packId"] == "oryn.mlx-lm"
        assert not any(n["name"].endswith(".py") for n in result["selectedFiles"])
    assert not hub.downloaded


@pytest.mark.parametrize("cached", [False, True])
def test_complete_backbone_and_draft_logits_match_independent_dense_reference(tmp_path, cached):
    mx = pytest.importorskip("mlx.core")
    c, ref, weights = tiny_fixture(tmp_path)
    actual = adapter.load_model(tmp_path, c)
    tokens = mx.array([[12, 31, 76, 80]], dtype=mx.int32)
    a_cache, b_cache = actual.language_model.make_cache(), ref.language_model.make_cache()
    if cached:
        actual.language_model(tokens[:, :3], cache=a_cache)
        ref.language_model(tokens[:, :3], cache=b_cache)
        a = actual.language_model(tokens[:, 3:], cache=a_cache).logits
        b = ref.language_model(tokens[:, 3:], cache=b_cache).logits
    else:
        a, b = actual.language_model(tokens).logits, ref.language_model(tokens).logits
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=2e-5, rtol=2e-4)
    if cached:
        full = actual.language_model(tokens).logits[:, -1:]
        np.testing.assert_allclose(np.asarray(a), np.asarray(full), atol=2e-5, rtol=2e-4)
    # Exercise all seven MTP norms, shared inverse embedding, Q8 fc/attention/MLP,
    # position offsets, draft cache, and the same PQ2 output head.
    draft_a, draft_b = actual.language_model.mtp, ref.language_model.mtp
    draft_a.bind(actual)
    draft_b.bind(ref)
    assert draft_a._input_embed is actual.language_model.model.embed_tokens
    hidden = mx.random.normal((1, 4, 128))
    results = []
    for draft in (draft_a, draft_b):
        cache = draft.make_cache() if cached else None
        embed = draft._input_embed(tokens)
        if cached:
            draft._forward_hidden(embed[:, :3], hidden[:, :3], cache, mx.array([[0, 1, 2]]))
            out = draft._forward_hidden(embed[:, 3:], hidden[:, 3:], cache, mx.array([[3]]))
        else:
            out = draft._forward_hidden(embed, hidden, None, mx.array([[0, 1, 2, 3]]))
        results.append(np.asarray(draft._lm_head_fn(out)))
    np.testing.assert_allclose(*results, atol=2e-5, rtol=2e-4)
    assert np.isfinite(results[0]).all()
    assert actual.language_model.model.embed_tokens(mx.array([1])).dtype == mx.float32
    from mlx.utils import tree_flatten

    loaded = dict(tree_flatten(actual.parameters()))
    for n in adapter.parameter_specs(c)[1]:
        np.testing.assert_array_equal(np.asarray(loaded[n]), np.asarray(weights[n] + 1))


def test_owned_prompt_matches_original_non_thinking_template():
    from transformers.utils.chat_template_utils import _compile_jinja_template

    template = _compile_jinja_template(original("tokenizer_config.json")["chat_template"])
    for messages in [
        [dict(role="user", content="Hello")],
        [dict(role="system", content="Be brief"), dict(role="user", content="你好")],
        [
            dict(role="user", content="Hi"),
            dict(role="assistant", content="Hello"),
            dict(role="user", content="More"),
        ],
    ]:
        assert adapter.mlx_prism.prompt(messages) == template.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=False,
            preserve_thinking=False,
        )


def test_full_vocabulary_preserves_encoding_but_unused_ids_render_empty(tmp_path):
    pytest.importorskip("mlx.core")
    c, _, _ = tiny_fixture(tmp_path)
    tok, stops = adapter.load_tokenizer(tmp_path, c)
    assert len(tok) == 384 and stops == {257, 258}
    assert tok.convert_ids_to_tokens(381) == ""
    assert tok.convert_ids_to_tokens([381, 382, 383]) == ["", "", ""]
    assert tok.decode(np.array([381, 382, 383])) == ""
    for text in ["Hello", "你好 👋", "a\nb", "[unused-381]"]:
        encoded = tok.encode(text, add_special_tokens=False)
        assert tok.decode([381, *encoded, 383]) == tok.decode(encoded)
    assert tok.encode("[unused-381]", add_special_tokens=False) == [381]


@pytest.mark.parametrize(
    "override",
    [
        {"auto_map": {"AutoTokenizer": [None, "unreviewed.Tokenizer"]}},
        {"trust_remote_code": True},
        {"tokenizer_file": "../unapproved.json"},
        {"nested": {"vocab_file": "/unapproved/vocab.txt"}},
    ],
)
def test_owned_tokenizer_binding_never_grants_repository_file_or_code_authority(override):
    with pytest.raises(ValueError):
        adapter.validate_tokenizer_metadata(original("tokenizer_config.json") | override)


@pytest.mark.parametrize(
    "damage", ["missing-draft", "nonfinite", "sign", "dtype", "norm", "index", "extra"]
)
def test_worker_load_rejects_damaged_payloads(tmp_path, damage):
    mx = pytest.importorskip("mlx.core")
    c, _, weights = tiny_fixture(tmp_path)
    if damage == "missing-draft":
        del weights[adapter.MTP + "fc.scales"]
    elif damage == "nonfinite":
        weights[adapter.MTP + "fc.scales"] = mx.full_like(
            weights[adapter.MTP + "fc.scales"], float("nan")
        )
    elif damage == "sign":
        path = tmp_path / adapter.SIGN_FILE
        signs = mx.load(str(path))
        signs["sign_128"] = -signs["sign_128"]
        mx.save_safetensors(str(path), signs)
    elif damage == "dtype":
        weights[adapter.MTP + "fc.weight"] = weights[adapter.MTP + "fc.weight"].astype(mx.float32)
    elif damage == "norm":
        weights[adapter.MTP + "norm.weight"] = mx.zeros((127,))
    elif damage == "index":
        p = tmp_path / "model.safetensors.index.json"
        value = json.loads(p.read_text())
        value["weight_map"][adapter.MTP + "fc.weight"] = "model-00002.safetensors"
        p.write_text(json.dumps(value))
    else:
        weights["unexpected"] = mx.zeros((1,))
    mx.save_safetensors(str(tmp_path / "model-00001.safetensors"), weights)
    with pytest.raises(ValueError):
        adapter.load_model(tmp_path, c)


def test_offline_worker_repeated_native_generation(tmp_path):
    import os
    import subprocess
    import sys

    mx = pytest.importorskip("mlx.core")
    c, _, weights = tiny_fixture(tmp_path)
    for suffix in ("scales", "biases"):
        name = "language_model.lm_head." + suffix
        weights[name] = mx.zeros_like(weights[name])
    mx.save_safetensors(str(tmp_path / "model-00001.safetensors"), weights)
    payload = dict(
        modelRoot=str(tmp_path),
        input=dict(messages=[dict(role="user", content="Hi")], maxTokens=3, temperature=0, topP=1),
    )
    from oryn_engine.runtime_packs import mlx_lm_worker

    assert mlx_lm_worker.generate(payload) == dict(text="!!!", backend="mlx")
    for _ in range(2):
        envelope = dict(
            abi="1", packId="oryn.mlx-lm", version="0.10.0", command="generate", payload=payload
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
        assert result["ok"], result
        assert result["result"] == dict(text="!!!", backend="mlx")
