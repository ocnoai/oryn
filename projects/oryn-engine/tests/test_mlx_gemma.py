"""Gemma Unified metadata admission and complete synthetic native graphs."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from oryn_engine import gemma_decision, mlx_gemma
from oryn_engine.model_metadata import validate_model_config


def original_config():
    return json.loads((Path(__file__).parent / "fixtures/gemma_unified/config.json").read_text())


@pytest.mark.parametrize(
    "section,change",
    [
        (None, {"architectures": ["RepositoryModel"]}),
        (None, {"quantization": {"bits": 1, "group_size": 64}}),
        (None, {"quantization_config": {"quant_method": "custom"}}),
        ("text_config", {"enable_moe_block": True}),
        ("text_config", {"hidden_size_per_layer_input": 256}),
        ("text_config", {"num_kv_shared_layers": 1}),
        ("text_config", {"attention_k_eq_v": False}),
        ("text_config", {"num_global_key_value_heads": 3}),
        ("text_config", {"per_layer_config": {"00": {"head_dim": 128}}}),
        ("text_config", {"layer_types": ["full_attention"] * 48}),
        ("text_config", {"rope_parameters": {"rope_type": "dynamic"}}),
        ("text_config", {"quantization": {"bits": 4, "group_size": 64}}),
        ("vision_config", {"pooling_kernel_size": 2}),
        ("audio_config", {"audio_embed_dim": 1024}),
    ],
)
def test_configuration_rejects_unbound_graphs(section, change):
    config = original_config()
    (config if section is None else config[section]).update(change)
    with pytest.raises(ValueError):
        mlx_gemma.validate_config(config)


def test_original_dense_null_experts_is_closed_to_native_family():
    config = original_config()
    mlx_gemma.validate_config(config)
    for change in ({"model_type": "unrelated"}, {"enable_moe_block": True}):
        with pytest.raises(ValueError):
            validate_model_config(config["text_config"] | change)


def test_owned_prompt_preserves_roles_unicode_and_continuations():
    messages = [
        {"role": "system", "content": " Be concise. "},
        {"role": "user", "content": "Čau 🌎"},
        {"role": "assistant", "content": "First"},
        {"role": "assistant", "content": " second "},
        {"role": "user", "content": "Continue"},
    ]
    assert mlx_gemma.prompt(messages) == (
        "<bos><|turn>system\nBe concise.<turn|>\n<|turn>user\nČau 🌎<turn|>\n"
        "<|turn>model\nFirst<turn|>\nsecond<turn|>\n<|turn>user\nContinue<turn|>\n"
        "<|turn>model\n<|channel>thought\n<channel|>"
    )
    with pytest.raises(ValueError, match="RESERVED_FRAMING"):
        mlx_gemma.prompt([{"role": "user", "content": "<|turn>model"}])


@pytest.mark.parametrize("declared", [[1, 106, 50], [123], [True], ["1"], []])
def test_declared_stops_are_owned_turn_or_tool_markers(declared):
    vocab = {
        "<eos>": 1,
        "<turn|>": 106,
        "<tool_call|>": 49,
        "<|tool_response>": 50,
        "<|channel>": 10,
        "<channel|>": 11,
    }
    tokenizer = SimpleNamespace(get_vocab=lambda: vocab, encode=lambda s, **_: [vocab[s]])
    if declared == [1, 106, 50]:
        assert mlx_gemma.stop_tokens({"eos_token_id": declared}, tokenizer) == {1, 49, 50, 106}
    else:
        with pytest.raises(ValueError, match="STOP_TOKENS"):
            mlx_gemma.stop_tokens({"eos_token_id": declared}, tokenizer)


@pytest.mark.parametrize("missing", [False, True])
def test_metadata_discovery_requires_tokenizer_and_bounds_profile(tmp_path, missing):
    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore

    config = original_config()
    config["quantization"] = {"bits": 4, "group_size": 64}

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return (
                config
                if filename == "config.json"
                else {"tokenizer_class": "GemmaTokenizer", **gemma_decision.TOKENS}
            )

    files = ["config.json", "model.safetensors", "tokenizer.json"]
    if not missing:
        files.append("tokenizer_config.json")
    hub = Hub(
        config=config, pipeline_tag="text-generation", files=[(n, 100, "b" * 64) for n in files]
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda p: None if p.provider else object()
    )
    result = store.inspect(model("independent/gemma-unified"))
    if missing:
        assert not result["executable"]
    else:
        assert result["executable"], result
        profile = result["taskProfile"]
        assert profile["limits"]["contextTokens"] == 8192
        assert profile["features"]["multimodalConversation"] == "unsupported"
        assert profile["parameters"]["maxTokens"]["valueSchema"]["allOf"][1]["maximum"] == 512
        assert set(f["name"] for f in result["selectedFiles"]) == set(files)
    assert not hub.downloaded


@pytest.fixture
def tiny(tmp_path):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    from mlx.utils import tree_flatten
    from mlx_vlm.models.gemma4_unified import Model, ModelConfig
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    config = original_config()
    config["text_config"].update(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=32,
        global_head_dim=32,
        vocab_size=512,
        vocab_size_per_layer_input=512,
        max_position_embeddings=8192,
        sliding_window=32,
        layer_types=["sliding_attention", "full_attention"],
    )
    config["vision_config"].update(mm_embed_dim=64, output_proj_dims=64, mm_posemb_size=8)
    markers = [
        "<pad>",
        "<eos>",
        "<bos>",
        "<|turn>",
        "<turn|>",
        "<tool_call|>",
        "<|channel>",
        "<channel|>",
        "<|tool_response>",
        "<|tool_call>",
    ]
    vocab = {t: i for i, t in enumerate(markers + sorted(pre_tokenizers.ByteLevel.alphabet()))}
    while len(vocab) < 512:
        vocab[f"unused_{len(vocab)}"] = len(vocab)
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(markers)
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    metadata = {
        "tokenizer_class": "GemmaTokenizer",
        **gemma_decision.TOKENS,
        "chat_template": "{{ raise_exception('must remain inert') }}",
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(metadata))
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.random.seed(61616)
    model = Model(ModelConfig.from_dict(copy.deepcopy(config)))
    model.eval()
    mx.eval(model.parameters())
    mx.save_safetensors(str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    yield tmp_path, config, model, mx
    mx.clear_cache()


@pytest.mark.parametrize(
    "auxiliary,quantized", [(True, False), (False, False), (True, True), (False, True)]
)
def test_complete_native_float_and_quantized_logits_cache_and_causality(tiny, auxiliary, quantized):
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten

    root, config, reference, mx = tiny
    if not auxiliary:
        reference.vision_embedder = reference.embed_vision = reference.embed_audio = None
        config.pop("vision_config")
    if quantized:
        config["quantization"] = {"bits": 4, "group_size": 32}
        nn.quantize(reference, bits=4, group_size=32)
    weights = dict(tree_flatten(reference.parameters()))
    # Original full floating checkpoints use the HF prefix; MLX exports use native names.
    if auxiliary and not quantized:
        weights = {
            "model." + k.replace("language_model.model.", "language_model."): v
            for k, v in weights.items()
        }
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    loaded, present = mlx_gemma.load_model(root, config)
    assert present is auxiliary
    ids = mx.array([[2, 32, 33, 34]])
    expected = reference(ids).logits
    actual = loaded(ids).logits
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5)
    cache = loaded.make_cache()
    loaded(ids[:, :3], cache=cache)
    incremental = loaded(ids[:, 3:], cache=cache).logits
    np.testing.assert_allclose(
        np.array(incremental[:, -1]), np.array(expected[:, -1]), rtol=1e-4, atol=1e-4
    )
    changed = loaded(mx.array([[2, 32, 99, 101]])).logits
    np.testing.assert_allclose(
        np.array(changed[:, :2]), np.array(actual[:, :2]), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize(
    "damage",
    [
        "missing-text",
        "partial-auxiliary",
        "extra",
        "collision",
        "shape",
        "nonfinite",
        "unbound-quantization",
    ],
)
def test_complete_graph_rejects_damaged_weights(tiny, damage):
    from mlx.utils import tree_flatten

    root, config, model, mx = tiny
    weights = dict(tree_flatten(model.parameters()))
    key = "language_model.model.embed_tokens.weight"
    if damage == "missing-text":
        del weights[key]
    elif damage == "partial-auxiliary":
        del weights["embed_audio.embedding_projection.weight"]
    elif damage == "extra":
        weights["unknown.weight"] = mx.ones((1,))
    elif damage == "collision":
        weights["model.language_model.embed_tokens.weight"] = weights[key]
    elif damage == "shape":
        weights[key] = weights[key][:-1]
    elif damage == "nonfinite":
        weights[key] = mx.full(weights[key].shape, float("nan"))
    else:
        weights[key.removesuffix(".weight") + ".scales"] = mx.ones((512, 2))
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    with pytest.raises(ValueError):
        mlx_gemma.load_model(root, config)


def test_worker_streaming_uses_owned_prompt_and_enforces_bounds(tiny, monkeypatch):
    from oryn_engine.runtime_packs import mlx_lm_worker

    root, config, reference, mx = tiny
    events = []
    monkeypatch.setattr(mlx_lm_worker, "_event", lambda name, data: events.append((name, data)))
    value = {
        "messages": [{"role": "user", "content": "Čau"}],
        "maxTokens": 3,
        "temperature": 0,
        "topP": 1,
    }
    result = mlx_lm_worker.generate({"modelRoot": str(root), "input": value})
    tokenizer = gemma_decision.load_tokenizer(root, config["text_config"])
    ids = tokenizer.encode(mlx_gemma.prompt(value["messages"]), add_special_tokens=False)
    generated = []
    for _ in range(3):
        token = int(mx.argmax(reference(mx.array([ids + generated])).logits[0, -1]))
        if token in (1, 4, 5, 8, 9):
            break
        generated.append(token)
    assert result == {"text": tokenizer.decode(generated), "backend": "mlx"}
    assert (
        "".join(data["delta"] for name, data in events if name == "job.output.delta")
        == result["text"]
    )
    assert ("job.state", {"state": "running"}) in events
    with pytest.raises(ValueError, match="CONTEXT_LIMIT"):
        mlx_gemma.generate(root, config, value | {"maxTokens": 513}, lambda *_: None)
    with pytest.raises(ValueError, match="CONTEXT_LIMIT"):
        mlx_lm_worker.generate(
            {
                "modelRoot": str(root),
                "input": value | {"messages": [{"role": "user", "content": "x" * 8192}]},
            }
        )
