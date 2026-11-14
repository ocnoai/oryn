"""Original metadata and complete small Kolibri graphs; no learned checkpoints."""

import json

import numpy as np
import pytest
from kolibri_fixture import original, tiny_config, tiny_fixture

from oryn_engine import mlx_kolibri as adapter


def test_original_metadata_binds_every_expert_and_tensor_without_native_imports():
    c, headers = original("config.json"), original("headers.json")
    files = [*adapter.METADATA, *headers, "untrusted.py"]
    adapter.inspect_metadata(c, original, headers.__getitem__, files)
    assert len(adapter.parameter_specs(c)) == 903
    assert len(adapter.serialized_shapes(c)) == 1907
    assert len(adapter.packed_specs(c)) == 502
    assert sum(bits == 4 for _, bits in adapter.packed_specs(c).values()) == 150
    assert "untrusted.py" not in adapter.selected_files(files, c)


def test_original_shapes_equal_the_complete_lazy_native_graph():
    pytest.importorskip("mlx_lm")
    from mlx.utils import tree_flatten

    c = original("config.json")
    before = json.loads(json.dumps(c))
    graph = adapter.native_model(c)
    assert {
        n: tuple(v.shape) for n, v in tree_flatten(graph.parameters())
    } == adapter.parameter_specs(c)
    assert c == before


@pytest.mark.parametrize(
    "changes",
    [
        {"auto_map": {"AutoModel": "repo.Code"}},
        {"model_file": "model.py"},
        {"custom_attention": True},
        {"head_dim": 33},
        {"hidden_size": 65},
        {"num_experts_per_tok": 385},
        {"num_experts": True},
        {"attention_bias": True},
        {"norm_topk_prob": True},
        {"tie_word_embeddings": True},
        {"layer_types": ["full_attention"] * 50},
        {"use_sliding_window": False},
        {"head_dtype": "bfloat16"},
        {"rope_theta": 1000000},
        {"eos_token_id": -1},
        {"sliding_window": 0},
        {"moe_intermediate_size": 65},
        {"bos_token_id": 0},
    ],
)
def test_unreviewed_or_inconsistent_graph_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        adapter.validate_config(original("config.json") | changes)


@pytest.mark.parametrize(
    "damage", ["router", "missing-override", "expert", "bits", "group", "duplicate-config"]
)
def test_mixed_precision_map_preserves_routing_and_all_module_identities(damage):
    c = original("config.json")
    q = c["quantization"]
    if damage == "router":
        q["model.layers.0.mlp.gate"] = {"bits": 8, "group_size": 64, "mode": "affine"}
    elif damage == "missing-override":
        del q["lm_head"]
    elif damage == "expert":
        q["model.layers.0.mlp.switch_mlp.up_proj"] = {"bits": 8, "group_size": 64, "mode": "affine"}
    elif damage == "bits":
        q["bits"] = 2
    elif damage == "group":
        q["group_size"] = 128
    else:
        c["quantization_config"] = {}
    with pytest.raises(ValueError):
        adapter.validate_config(c)


@pytest.mark.parametrize(
    "damage",
    [
        "missing-expert",
        "missing-norm",
        "missing-scale",
        "router-precision",
        "shape",
        "dtype",
        "extra",
        "index",
    ],
)
def test_every_original_header_array_and_shard_is_bound(damage):
    c, headers, index = (
        original("config.json"),
        original("headers.json"),
        original("model.safetensors.index.json"),
    )
    tensor = "model.layers.0.mlp.switch_mlp.up_proj.weight"
    if damage == "missing-norm":
        tensor = "model.layers.0.post_ffn_norm.weight"
    if damage == "missing-scale":
        tensor = "model.layers.0.mlp.switch_mlp.up_proj.scales"
    if damage == "router-precision":
        tensor = "model.layers.0.mlp.expert_bias"
    filename = index["weight_map"][tensor]
    if damage.startswith("missing-"):
        del headers[filename]["tensors"][tensor]
    elif damage == "shape":
        headers[filename]["tensors"][tensor]["shape"][0] -= 1
    elif damage in {"dtype", "router-precision"}:
        headers[filename]["tensors"][tensor]["dtype"] = "BF16"
    elif damage == "extra":
        headers[filename]["tensors"]["unbound.weight"] = dict(shape=[1], dtype="F32")
    else:
        index["weight_map"][tensor] = "../model.safetensors"
    with pytest.raises(ValueError):
        adapter.validate_headers(c, index, headers)


@pytest.mark.parametrize("damage", [None, "tokenizer", "stops", "header", "missing", "extra"])
def test_discovery_is_metadata_only_and_preserves_original_identity(tmp_path, damage):
    from types import SimpleNamespace

    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore

    c, headers = original("config.json"), original("headers.json")
    names = [*adapter.METADATA, *headers, "repo.py"]
    if damage == "missing":
        names.remove("model-00009-of-00009.safetensors")
    if damage == "extra":
        names.append("auxiliary.safetensors")

    class Hub(FakeHub):
        def read_json(self, repo, revision, name):
            value = original(name)
            if name == "tokenizer_config.json" and damage == "tokenizer":
                value["auto_map"] = {"AutoTokenizer": "repo.Code"}
            if name == "generation_config.json" and damage == "stops":
                value["eos_token_id"] = [127906]
            return value

        def read_tensor_structure(self, repo, revision, name):
            value = headers[name]
            if damage == "header":
                value["tensors"].pop("model.layers.0.post_ffn_norm.weight", None)
            return value

    hub = Hub(config=c, pipeline_tag="text-generation", files=[(n, 100, "a" * 64) for n in names])
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda p: None if p.provider else object()
    )
    result = store.inspect(model("independent/renamed-kolibri"))
    assert result["executable"] is (damage is None), result
    if damage is None:
        assert result["runtimePack"]["packId"] == "oryn.mlx-lm"
        assert result["taskProfile"]["limits"]["contextTokens"] == 4096
        assert not any(f["name"].endswith(".py") for f in result["selectedFiles"])
    assert not hub.downloaded


@pytest.mark.parametrize(
    "dtype,length",
    [
        ("float32", 9),
        ("float32", 10),
        ("float32", 23),
        ("float32", 520),
        ("bfloat16", 10),
        ("bfloat16", 23),
    ],
)
def test_complete_mixed_graph_and_rotating_cache_match_independent_torch_reference(
    tmp_path, dtype, length
):
    mx = pytest.importorskip("mlx.core")
    import torch
    from kolibri_reference import forward

    c, weights, _ = tiny_fixture(tmp_path, dtype)
    if length == 520:
        c["sliding_window"] = 513
        (tmp_path / "config.json").write_text(json.dumps(c))
    model = adapter.load_model(tmp_path, c)
    ids = (np.arange(length, dtype=np.int32) * 13 + 7) % 256
    with torch.no_grad():
        expected = forward(torch.from_numpy(ids.astype(np.int64)), weights, c).numpy()
    actual = model(mx.array(ids[None]))
    tolerance = 4e-5 if dtype == "float32" else 0.035
    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32))[0], expected, atol=tolerance, rtol=tolerance
    )
    cache = model.make_cache()
    # Chunked prefill crosses a rotation boundary; subsequent one-token calls
    # must keep absolute RoPE positions and retain exactly the sliding band.
    split = min(length - 2, c["sliding_window"] + 1)
    model(mx.array(ids[None, :split]), cache=cache)
    outputs = []
    for i in range(split, length):
        outputs.append(model(mx.array(ids[None, i : i + 1]), cache=cache))
    cached = np.asarray(mx.concatenate(outputs, axis=1).astype(mx.float32))[0]
    np.testing.assert_allclose(cached, expected[split:], atol=tolerance, rtol=tolerance)
    assert np.isfinite(cached).all()


def test_router_selects_by_biased_logits_but_weights_by_unbiased_sigmoid():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from oryn_engine.native_kolibri import Kolibri1SparseMoeBlock, ModelArgs

    c = tiny_config()
    block = Kolibri1SparseMoeBlock(ModelArgs.from_dict(c))
    # This choice gives different experts if bias is added after sigmoid.
    logits = np.array([2.0, 1.0, -1.0, -2.0], np.float32)
    bias = np.array([0.0, 0.0, 2.5, 0.0], np.float32)
    block.gate.weight = mx.zeros_like(block.gate.weight)
    block.gate.weight[:, 0] = mx.array(logits)
    block.expert_bias = mx.array(bias)

    class Experts:
        def __call__(self, x, indices):
            self.ids = np.asarray(indices).copy()
            return mx.broadcast_to(
                mx.array([1.0, 2.0, 4.0, 8.0])[indices][..., None], (*indices.shape, x.shape[-1])
            )

    experts = Experts()
    block.switch_mlp = experts
    block.shared_experts = lambda x: mx.zeros_like(x)
    x = mx.zeros((1, 1, 64))
    x[..., 0] = 1
    actual = np.asarray(block(x))
    assert set(experts.ids.ravel()) == {0, 2}
    expected = 1 / (1 + np.exp(-2.0)) + 4 / (1 + np.exp(1.0))
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize(
    "messages",
    [
        [dict(role="user", content="Hello")],
        [dict(role="system", content="Be concise"), dict(role="user", content="Grüße")],
        [
            dict(role="user", content="Hi"),
            dict(role="assistant", content="Hello"),
            dict(role="user", content="More"),
        ],
        [dict(role="user", content="  Keep spacing\n")],
    ],
)
def test_owned_non_thinking_prompt_matches_original_template(messages):
    from transformers.utils.chat_template_utils import _compile_jinja_template

    template = _compile_jinja_template(original("tokenizer_config.json")["chat_template"])
    assert adapter.prompt(messages) == template.render(
        messages=messages,
        enable_thinking=False,
        add_generation_prompt=True,
        preserve_thinking=False,
    )


@pytest.mark.parametrize(
    "control", ["<|im_start|>", "<think>", "</think>", "<tool_call>", "<tool_response>"]
)
def test_owned_prompt_rejects_reserved_controls(control):
    with pytest.raises(ValueError):
        adapter.prompt([dict(role="user", content=control)])


@pytest.mark.parametrize("token", [376, 380, 384, -1])
def test_sparse_vocabulary_holes_and_out_of_bounds_output_fail(tmp_path, monkeypatch, token):
    from types import SimpleNamespace

    mlx_lm = pytest.importorskip("mlx_lm")
    c, _, _ = tiny_fixture(tmp_path)
    monkeypatch.setattr(adapter, "load_model", lambda *_: object())
    monkeypatch.setattr(
        mlx_lm, "stream_generate", lambda *_, **__: iter([SimpleNamespace(token=token)])
    )
    with pytest.raises(ValueError, match="OUTPUT_TOKEN"):
        adapter.generate(
            tmp_path,
            c,
            dict(messages=[dict(role="user", content="Hi")], maxTokens=1, temperature=0, topP=1),
            lambda *_: None,
        )


def test_high_valid_sparse_vocabulary_token_is_preserved(tmp_path, monkeypatch):
    from types import SimpleNamespace

    mlx_lm = pytest.importorskip("mlx_lm")
    c, _, _ = tiny_fixture(tmp_path)
    tok, _ = adapter.load_tokenizer(tmp_path, c)
    assert len(tok) == 382
    monkeypatch.setattr(adapter, "load_model", lambda *_: object())
    monkeypatch.setattr(
        mlx_lm, "stream_generate", lambda *_, **__: iter([SimpleNamespace(token=383)])
    )
    out = adapter.generate(
        tmp_path,
        c,
        dict(messages=[dict(role="user", content="Hi")], maxTokens=1, temperature=0, topP=1),
        lambda *_: None,
    )
    assert out["text"] == tok.decode([383])


@pytest.mark.parametrize("damage", ["expert", "norm", "nonfinite", "dtype", "router", "extra"])
def test_native_loader_rejects_incomplete_or_corrupt_payloads(tmp_path, damage):
    mx = pytest.importorskip("mlx.core")
    c, _, w = tiny_fixture(tmp_path)
    key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
    if damage == "expert":
        w[key] = w[key][:-1]
    elif damage == "norm":
        del w["model.layers.0.post_ffn_norm.weight"]
    elif damage == "nonfinite":
        w["model.layers.0.mlp.gate.weight"] = mx.full((4, 64), float("nan"))
    elif damage == "dtype":
        w[key] = w[key].astype(mx.float32)
    elif damage == "router":
        w["model.layers.0.mlp.expert_bias"] = w["model.layers.0.mlp.expert_bias"].astype(
            mx.bfloat16
        )
    else:
        w["unbound.weight"] = mx.ones((1,))
    mx.save_safetensors(str(tmp_path / "model-00001-of-00001.safetensors"), w)
    with pytest.raises(ValueError):
        adapter.load_model(tmp_path, c)


def test_repeated_offline_worker_generation(tmp_path):
    import os
    import subprocess
    import sys

    mx = pytest.importorskip("mlx.core")
    _, _, w = tiny_fixture(tmp_path)
    for suffix in ("scales", "biases"):
        name = "lm_head." + suffix
        w[name] = mx.zeros_like(w[name])
    mx.save_safetensors(str(tmp_path / "model-00001-of-00001.safetensors"), w)
    payload = dict(
        modelRoot=str(tmp_path),
        input=dict(messages=[dict(role="user", content="Hi")], maxTokens=3, temperature=0, topP=1),
    )
    from oryn_engine.runtime_packs import mlx_lm_worker

    assert mlx_lm_worker.generate(payload) == dict(text="!!!", backend="mlx")
    for _ in range(2):
        req = dict(
            abi="1", packId="oryn.mlx-lm", version="0.10.0", command="generate", payload=payload
        )
        run = subprocess.run(
            [sys.executable, "-m", "oryn_engine.runtime_packs.mlx_lm_worker"],
            input=json.dumps(req) + "\n",
            text=True,
            capture_output=True,
            timeout=60,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        assert run.returncode == 0, run.stderr
        result = [json.loads(line) for line in run.stdout.splitlines()][-1]
        assert result["ok"], result
        assert result["result"] == dict(text="!!!", backend="mlx")
