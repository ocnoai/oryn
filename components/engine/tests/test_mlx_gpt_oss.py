"""Original metadata and generated complete graphs for mixed native quantization."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from oryn_engine import mlx_gpt_oss as adapter
from oryn_engine.runtime_packs import mlx_lm_worker

FIXTURES = Path(__file__).parent / "fixtures/mlx_gpt_oss"
ORIGINAL = json.loads((FIXTURES / "original.json").read_text())


def tokenizer(root):
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

    vocab = {t: i for i, t in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    for token in (*adapter.MARKERS, "<|startoftext|>", "<|endoftext|>"):
        vocab[token] = len(vocab)
    native = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    native.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    native.decoder = decoders.ByteLevel()
    native.add_special_tokens(
        [AddedToken(t, normalized=False, special=True) for t in vocab if t.startswith("<|")]
    )
    native.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<|startoftext|>",
                "eos_token": "<|return|>",
                "pad_token": "<|endoftext|>",
                "tokenizer_class": "PreTrainedTokenizer",
                "chat_template": "{{ never_execute() }}",
            }
        )
    )
    config = {
        "vocab_size": 288,
        "eos_token_id": vocab["<|return|>"],
        "pad_token_id": vocab["<|endoftext|>"],
    }
    return adapter.load_tokenizer(root, config), config


def native_fixture(root, mode="mixed"):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx.utils import tree_flatten

    native = pytest.importorskip("mlx_lm.models.gpt_oss")
    Model, ModelArgs = native.Model, native.ModelArgs

    root.mkdir(exist_ok=True, parents=True)
    tok, token_config = tokenizer(root)
    config = copy.deepcopy(ORIGINAL["config"])
    config.update(
        token_config,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        experts_per_token=2,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        rope_scaling=None,
        max_position_embeddings=4096,
    )
    config.pop("quantization_config")
    config.pop("quantization")
    mx.random.seed(6149)
    model = Model(ModelArgs.from_dict(config))
    if mode != "float":
        quant = {
            "bits": 4 if mode == "mixed" else 8,
            "group_size": 32,
            "mode": "mxfp4" if mode == "mixed" else "affine",
        }
        if mode == "mixed":
            for name, module in model.named_modules():
                if hasattr(module, "to_quantized") and ".experts." not in name:
                    # Match the original router's implicit affine default too.
                    quant[name] = {"bits": 8, "group_size": 32}
        config["quantization"] = copy.deepcopy(quant)
        config["quantization_config"] = copy.deepcopy(quant)
        nn.quantize(
            model,
            bits=quant["bits"],
            group_size=32,
            mode=quant["mode"],
            class_predicate=lambda name, module: quant.get(name, hasattr(module, "to_quantized")),
        )
    mx.eval(model.parameters())
    weights = dict(tree_flatten(model.parameters()))
    names = sorted(weights)
    middle = len(names) // 2
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    index = {}
    for shard, keys in zip(shards, (names[:middle], names[middle:]), strict=True):
        mx.save_safetensors(str(root / shard), {k: weights[k] for k in keys})
        index.update({k: shard for k in keys})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    (root / "config.json").write_text(json.dumps(config))
    return model, tok, config, weights


def test_original_metadata_config_admits_exact_mixed_quantization():
    adapter.validate_config(ORIGINAL["config"])
    assert len(ORIGINAL["tensors"]) == 775 and ORIGINAL["headerBytesRead"] == 88400
    assert ORIGINAL["originalPayloadBytes"] == 0


def test_original_complete_tensor_graph_matches_native_without_payload():
    pytest.importorskip("mlx_lm.models.gpt_oss")
    adapter.validate_weight_shapes(
        ORIGINAL["config"], {k: tuple(t["shape"]) for k, t in ORIGINAL["tensors"].items()}
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"architectures": ["OtherModel"]},
        {"hidden_act": "gelu"},
        {"attention_bias": False},
        {"tie_word_embeddings": True},
        {"swiglu_limit": 8},
        {"layer_types": ["full_attention"] * 24},
        {"num_experts_per_tok": 33},
        {"experts_per_token": 1},
        {"head_dim": 63},
        {"attention_dropout": 0.1},
        {"auto_map": {"AutoModel": "repo.Code"}},
        {"max_position_embeddings": True},
        {"rope_scaling": {"rope_type": "dynamic"}},
    ],
)
def test_unsupported_graph_options_fail_closed(changes):
    with pytest.raises(ValueError):
        adapter.validate_config(ORIGINAL["config"] | changes)


@pytest.mark.parametrize(
    "settings",
    [
        {"bits": 8, "group_size": 32, "mode": "mxfp4"},
        {"bits": 4, "group_size": 64, "mode": "mxfp4"},
        {"bits": 4, "group_size": 32, "mode": "mxfp8"},
        {"bits": 4, "group_size": 32, "mode": "nvfp4"},
        {"bits": True, "group_size": 32, "mode": "mxfp4"},
    ],
)
def test_only_reviewed_native_mxfp4_geometry_is_admitted(settings):
    config = copy.deepcopy(ORIGINAL["config"])
    config["quantization"].update(settings)
    config["quantization_config"] = config["quantization"]
    with pytest.raises(ValueError):
        adapter.validate_config(config)


def test_mxfp4_does_not_expand_unreviewed_architectures():
    config = copy.deepcopy(ORIGINAL["config"])
    config["model_type"] = "qwen2"
    with pytest.raises(ValueError):
        mlx_lm_worker.validate_config(config, {"qwen2"})


def test_owned_prompt_matches_reviewed_original_template_references():
    reference = json.loads((FIXTURES / "prompt-reference.json").read_text())
    for case in reference["cases"]:
        assert adapter.prompt(case["messages"], date=case["date"]) == case["expected"]


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "<|start|>"}],
        [{"role": "user", "content": "|>"}],
        [{"role": "user", "content": "Hi"}, {"role": "system", "content": "Late"}],
        [{"role": "tool", "content": "unsupported"}],
    ],
)
def test_owned_harmony_rejects_controls_and_unbound_roles(messages):
    with pytest.raises(ValueError):
        adapter.prompt(messages)


@pytest.mark.parametrize("analysis", ["", "A hidden draft.", "日本語 {{ literal }}"])
@pytest.mark.parametrize("finish", ["", "<|return|>"])
def test_only_final_text_is_decoded_and_partial_final_is_bounded(tmp_path, analysis, finish):
    tok, _ = tokenizer(tmp_path)
    generated = (
        "<|channel|>analysis<|message|>" + analysis + "<|end|><|start|>assistant"
        if analysis
        else ""
    )
    generated += "<|channel|>final<|message|>Hello 🙂 日本語" + finish
    assert (
        adapter.final_text(tok, tok.encode(generated, add_special_tokens=False))
        == "Hello 🙂 日本語"
    )


@pytest.mark.parametrize(
    "generated",
    [
        "analysis only",
        "<|channel|>analysis<|message|>unfinished",
        "<|channel|>commentary<|message|>tool call",
        " to=functions.fake<|message|>bad<|call|>",
        "<|channel|>final<|message|>",
        "<|channel|>final<|message|>bad<|call|>",
        "<|channel|>analysis<|message|><|return|>",
        "<|channel|>final<|message|>bad<|end|><|start|>user<|message|>bad",
    ],
)
def test_malformed_or_nonfinal_harmony_is_not_emitted(tmp_path, generated):
    tok, _ = tokenizer(tmp_path)
    with pytest.raises(ValueError, match="FINAL_CHANNEL"):
        adapter.final_text(tok, tok.encode(generated, add_special_tokens=False))


def test_unused_model_vocabulary_rows_are_not_silently_decoded(tmp_path):
    tok, _ = tokenizer(tmp_path)
    with pytest.raises(ValueError, match="OUTPUT_TOKENS"):
        adapter.final_text(tok, [len(tok.get_vocab())])


@pytest.mark.parametrize(
    "damage", ["eos", "pad", "auto-map", "vocabulary", "missing-marker", "long-token"]
)
def test_tokenizer_identity_and_marker_graph_are_required(tmp_path, damage):
    _, config = tokenizer(tmp_path)
    path = tmp_path / "tokenizer_config.json"
    metadata = json.loads(path.read_text())
    if damage == "eos":
        config["eos_token_id"] = 1
    elif damage == "pad":
        metadata["pad_token"] = "<|return|>"
    elif damage == "auto-map":
        metadata["auto_map"] = {"AutoTokenizer": "repo.Code"}
    elif damage == "vocabulary":
        config["vocab_size"] = 32
    elif damage == "long-token":
        token_path = tmp_path / "tokenizer.json"
        graph = json.loads(token_path.read_text())
        graph["model"]["vocab"]["x" * 4097] = graph["model"]["vocab"].pop("!")
        token_path.write_text(json.dumps(graph))
    else:
        token_path = tmp_path / "tokenizer.json"
        token_path.write_text(token_path.read_text().replace("<|channel|>", "<|altered|>"))
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        adapter.load_tokenizer(tmp_path, config)


@pytest.mark.parametrize("backend", ["gpu"])
def test_worker_executes_full_native_graph_and_emits_only_final_channel(
    tmp_path, monkeypatch, backend
):
    mx = pytest.importorskip("mlx.core")
    sample_utils = pytest.importorskip("mlx_lm.sample_utils")

    with mx.stream(getattr(mx, backend)):
        _, tok, config, _ = native_fixture(tmp_path)
        response = (
            "<|channel|>analysis<|message|>Private draft.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Hello 🙂<|return|>"
        )
        ids = iter(tok.encode(response, add_special_tokens=False))
        calls = []

        def make_sampler(**options):
            assert options == {"temp": 0.0, "top_p": 1.0}

            def sample(logprobs):
                assert bool(mx.all(mx.isfinite(logprobs)))
                calls.append(logprobs.shape)
                return mx.array([next(ids, tok.eos_token_id)])

            return sample

        monkeypatch.setattr(sample_utils, "make_sampler", make_sampler)
        events = []
        monkeypatch.setattr(mlx_lm_worker, "_event", lambda name, data: events.append((name, data)))
        result = mlx_lm_worker.generate(
            {
                "modelRoot": str(tmp_path),
                "input": {
                    "messages": [{"role": "user", "content": "Hello €"}],
                    "maxTokens": 128,
                    "temperature": 0,
                    "topP": 1,
                },
            }
        )
        assert result == {"text": "Hello 🙂", "backend": "mlx"}
        assert len(calls) >= 20 and all(shape[-1] == config["vocab_size"] for shape in calls)
        assert [data["delta"] for name, data in events if name == "job.output.delta"] == [
            "Hello 🙂"
        ]


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("mode", ["float", "affine", "mixed"])
def test_complete_native_graph_and_cache_preserve_serialized_values(tmp_path, backend, mode):
    mx = pytest.importorskip("mlx.core")
    from mlx.utils import tree_flatten

    with mx.stream(getattr(mx, backend)):
        original, _, config, weights = native_fixture(tmp_path, mode)
        loaded = adapter.load_model(tmp_path, config)
        actual = dict(tree_flatten(loaded.parameters()))
        assert actual.keys() == weights.keys()
        for name in weights:
            assert actual[name].dtype == weights[name].dtype
            assert bool(mx.array_equal(actual[name], weights[name]))
        ids = mx.array([[1, 5, 11, 29, 18], [2, 8, 3, 20, 21]])
        expected, result = original(ids), loaded(ids)
        mx.eval(expected, result)
        assert bool(mx.allclose(expected, result, atol=2e-6, rtol=2e-6))
        caches = loaded.make_cache()
        loaded(ids[:, :3], cache=caches)
        cached = loaded(ids[:, 3:], cache=caches)
        mx.eval(cached)
        assert bool(mx.allclose(cached, result[:, 3:], atol=1e-5, rtol=1e-5))


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "extra",
        "missing-scale",
        "affine-bias",
        "mx-bias",
        "shape",
        "dtype",
        "nonfinite",
        "mx-nan",
    ],
)
def test_damaged_native_graph_is_rejected(tmp_path, damage):
    mx = pytest.importorskip("mlx.core")
    _, _, config, weights = native_fixture(tmp_path)
    weights = dict(weights)
    key = "model.layers.0.mlp.experts.down_proj"
    if damage == "missing":
        del weights[key + ".bias"]
    elif damage == "extra":
        weights["extra.weight"] = mx.ones((2, 2))
    elif damage == "missing-scale":
        del weights[key + ".scales"]
    elif damage == "affine-bias":
        del weights["lm_head.biases"]
    elif damage == "mx-bias":
        weights[key + ".biases"] = mx.ones(weights[key + ".scales"].shape)
    elif damage == "shape":
        weights[key + ".weight"] = weights[key + ".weight"][:, :, :-1]
    elif damage == "dtype":
        weights[key + ".scales"] = weights[key + ".scales"].astype(mx.float32)
    elif damage == "nonfinite":
        weights[key + ".bias"] = mx.full(weights[key + ".bias"].shape, float("nan"))
    elif damage == "mx-nan":
        weights[key + ".scales"] = mx.full(weights[key + ".scales"].shape, 255, mx.uint8)
    for path in tmp_path.glob("*.safetensors*"):
        path.unlink()
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    with pytest.raises((ValueError, KeyError)):
        adapter.load_model(tmp_path, config)


def test_metadata_discovery_uses_native_gpt_oss_without_loading_weights(tmp_path):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore

    hub = FakeHub(
        config=copy.deepcopy(ORIGINAL["config"]),
        pipeline_tag="text-generation",
        files=[
            ("model.safetensors", 1000, "a" * 64),
            ("config.json", 1000, "b" * 64),
            ("tokenizer.json", 1000, "c" * 64),
            ("tokenizer_config.json", 1000, "d" * 64),
        ],
    )
    store = ModelStore(
        tmp_path,
        hub=hub,
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 64_000_000_000),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda pack: None if pack.provider else object()
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "renamed/mixed-gpt-oss", "revision": "main"}
    )
    assert result["executable"]
    assert result["runtimePack"]["packId"] == "oryn.mlx-lm"
    assert result["taskProfile"]["limits"]["contextTokens"] == 8192
    assert hub.downloaded == []
