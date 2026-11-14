"""Complete native encoder/heads, constrained decoding and metadata-only admission."""

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from oryn_engine import bioes, pii_tasks, shared_tasks
from oryn_engine.generated_capabilities import normalized_transformers_config
from oryn_engine.model_metadata import validate_model_config


def originals():
    root = Path(__file__).parent / "fixtures/pii_tracer"
    return {
        name: json.loads((root / name).read_text())
        for name in ("config.json", "tokenizer_config.json")
    }


@pytest.mark.parametrize("index", range(12))
def test_reviewed_upstream_decoder_vectors(index):
    source = Path(__file__).parent / "fixtures/pii_tracer/decoder_reference.json"
    vector = json.loads(source.read_text())["vectors"][index]
    assert (
        bioes.decode(
            torch.tensor(vector["logits"]),
            bioes.transition_mask(pii_tasks.TYPES),
            vector["beginBias"],
            vector["endBias"],
        )
        == vector["path"]
    )


def legal(source, target):
    # Independent BIOES grammar, including prohibition of cross-type interiors.
    if source == 0 or (source - 1) % 4 in (2, 3):
        return target == 0 or (target - 1) % 4 in (0, 3)
    return target > 0 and (source - 1) // 4 == (target - 1) // 4 and (target - 1) % 4 in (1, 2)


@pytest.mark.parametrize("length", [1, 2, 3, 5])
@pytest.mark.parametrize("biases", [(0.0, 0.0), (0.8, -0.3)])
def test_viterbi_matches_exhaustive_paths(length, biases):
    torch.manual_seed(61614 + length)
    logits = torch.randn(length, 5)
    best, best_score = None, float("-inf")
    for path in itertools.product(range(5), repeat=length):
        if path[0] in (2, 3) or path[-1] in (1, 2):
            continue
        if any(not legal(a, b) for a, b in zip(path, path[1:], strict=False)):
            continue
        score = sum(float(logits[i, label]) for i, label in enumerate(path))
        score += sum(
            (biases[0] if b == 1 else 0) + (biases[1] if a == 3 else 0)
            for a, b in zip(path, path[1:], strict=False)
        )
        if score > best_score:
            best, best_score = path, score
    assert bioes.decode(logits, bioes.transition_mask(["person"]), *biases) == list(best)


@pytest.fixture
def tiny(tmp_path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import Qwen3Config, Qwen3Model, TokenizersBackend

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    vocab = {c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab["<|endoftext|>"] = len(vocab)
    graph = Tokenizer(models.BPE(vocab, []))
    graph.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    graph.decoder = decoders.ByteLevel()
    tokenizer = TokenizersBackend(
        tokenizer_object=graph, eos_token="<|endoftext|>", pad_token="<|endoftext|>"
    )
    tokenizer.save_pretrained(tmp_path)
    config = originals()["config.json"]
    config["hidden_size"] = 16
    config["backbone"].update(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["full_attention"] * 2,
        vocab_size=len(vocab),
        max_position_embeddings=4096,
        bos_token_id=256,
        eos_token_id=256,
    )
    native = Qwen3Config.from_dict(config["backbone"])
    native._attn_implementation = "eager"
    torch.manual_seed(61614)
    model = torch.nn.Module()
    model.add_module("backbone", Qwen3Model(native))
    for layer in model.backbone.layers:
        layer.self_attn.is_causal = False
    model.add_module("token_cls_head", torch.nn.Linear(16, 37))
    model.add_module("sensitivity_head", torch.nn.Linear(16, 1))
    decoder = torch.nn.Module()
    decoder.register_buffer("b_bias", torch.tensor([0.8]))
    decoder.register_buffer("e_bias", torch.tensor([-0.3]))
    decoder.register_buffer(
        "transition_mask", torch.tensor([[legal(a, b) for b in range(37)] for a in range(37)])
    )
    model.add_module("viterbi", decoder)
    model.eval()
    save_file(
        {k: v.detach().clone() for k, v in model.state_dict().items()},
        tmp_path / "model.safetensors",
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "modeling_pii_masking.py").write_text("raise RuntimeError('must remain inert')")
    try:
        yield tmp_path, model, tokenizer, config
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_native_bidirectional_encoder_complete_heads_and_unicode_labels(tiny, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    root, reference, original, config = tiny
    model, tokenizer, _ = shared_tasks.load_task(root, "token-classification", backend)
    assert tokenizer.chat_template is None
    for key, expected in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key].cpu(), expected, atol=0, rtol=0)
    text = "Čau, a blue river!"
    inputs = original(
        text, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False
    )
    ids = inputs["input_ids"]
    with torch.inference_mode():
        # Independently provide a fully bidirectional native attention mask.
        hidden = reference.backbone(
            input_ids=ids,
            attention_mask={"full_attention": torch.zeros(1, 1, ids.shape[1], ids.shape[1])},
            use_cache=False,
        ).last_hidden_state
        expected_logits = reference.token_cls_head(hidden)[0]
        captured = []
        hook = model.token_cls_head.register_forward_hook(
            lambda _m, _i, out: captured.append(out.cpu())
        )
        result = pii_tasks.execute(model, tokenizer, config, {"text": text}, backend)
        hook.remove()
        torch.testing.assert_close(captured[0][0], expected_logits, rtol=2e-3, atol=1e-4)
        changed = ids.clone()
        changed[0, -1] = (changed[0, -1] + 1) % config["backbone"]["vocab_size"]
        other = reference.backbone(
            input_ids=changed,
            attention_mask={"full_attention": torch.zeros(1, 1, ids.shape[1], ids.shape[1])},
            use_cache=False,
        ).last_hidden_state
        assert not torch.allclose(hidden[0, 0], other[0, 0], atol=1e-6, rtol=1e-6)
    labels = [row["labelId"] for row in result["tokens"]]
    assert labels[0] == 0 or (labels[0] - 1) % 4 in (0, 3)
    assert labels[-1] == 0 or (labels[-1] - 1) % 4 in (2, 3)
    assert all(legal(a, b) for a, b in zip(labels, labels[1:], strict=False))
    for i, row in enumerate(result["tokens"]):
        assert (row["start"], row["end"]) == tuple(inputs["offset_mapping"][0, i].tolist())
        assert row["label"] == pii_tasks.LABELS[row["labelId"]]
        assert row["score"] == pytest.approx(
            float(expected_logits[i, row["labelId"]].sigmoid()), abs=1e-4
        )
    shared_tasks.validate_task_output("token-classification", result)


def test_shared_worker_and_untruncated_token_limit(tiny):
    root, _, _, config = tiny
    result = shared_tasks.execute_task(
        root, "token-classification", {"text": "hello"}, "cpu", None, None, lambda *_: None
    )
    assert len(result["tokens"]) == 5
    config["max_seq_len"] = 4
    (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="PII_TOKEN_LIMIT"):
        shared_tasks.execute_task(
            root, "token-classification", {"text": "hello"}, "cpu", None, None, lambda *_: None
        )


@pytest.mark.parametrize(
    "damage",
    ["encoder", "head", "sensitivity", "extra", "shape", "nonfinite", "transition", "bias"],
)
def test_all_native_weights_and_decoder_buffers_required(tiny, damage):
    root, *_ = tiny
    path = root / "model.safetensors"
    state = load_file(path)
    if damage in {"encoder", "head", "sensitivity"}:
        del state[
            {
                "encoder": "backbone.layers.1.self_attn.q_norm.weight",
                "head": "token_cls_head.bias",
                "sensitivity": "sensitivity_head.weight",
            }[damage]
        ]
    elif damage == "extra":
        state["extra.weight"] = torch.ones(1)
    elif damage == "shape":
        state["token_cls_head.weight"] = torch.ones(1)
    elif damage == "transition":
        state["viterbi.transition_mask"][0, 2] = True
    elif damage == "bias":
        state["viterbi.e_bias"][0] = 1e10
    else:
        state["sensitivity_head.weight"][0, 0] = float("nan")
    save_file(state, path)
    with pytest.raises(ValueError, match="PII_"):
        shared_tasks.load_task(root, "token-classification", "cpu")


@pytest.mark.parametrize(
    "field,value",
    [
        ("is_causal", True),
        ("use_cache", True),
        ("attention_bias", True),
        ("sliding_window", 128),
        ("head_dim", 63),
        ("num_key_value_heads", 3),
        ("auto_map", {"AutoModel": "evil.Code"}),
        ("quantization_config", {"bits": 4}),
    ],
)
def test_only_reviewed_backbone_admitted(field, value):
    config = originals()["config.json"]
    config["backbone"][field] = value
    with pytest.raises(ValueError):
        pii_tasks.normalized_config(config, "token-classification")


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_token_labels", 38),
        ("hidden_size", 16),
        ("max_seq_len", 8192),
        ("id2label", {"0": "wrong"}),
        ("auto_map", {"AutoModel": "unknown.PiiMaskingModel"}),
    ],
)
def test_unreviewed_head_or_alias_rejected(field, value):
    with pytest.raises(ValueError):
        pii_tasks.validate_config(
            normalized_transformers_config(
                originals()["config.json"] | {field: value}, "token-classification"
            )
        )


def test_alias_does_not_enable_other_tasks():
    with pytest.raises(ValueError):
        validate_model_config(
            normalized_transformers_config(originals()["config.json"], "feature-extraction")
        )


@pytest.mark.parametrize("missing", [None, "tokenizer_config.json", "tokenizer.json"])
def test_original_metadata_routes_without_original_weights(tmp_path, missing):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    data = originals()

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return data[filename]

    files = (set(data) | {"model.safetensors", "tokenizer.json"}) - {missing}
    hub = Hub(
        config=data["config.json"],
        pipeline_tag="token-classification",
        files=[(n, 100, "b" * 64) for n in files],
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "test/pii", "revision": "main"}
    )
    assert result["executable"] is (missing is None), result
    if missing is None:
        assert result["taskProfile"]["contract"]["id"] == "token-classification-v1"
        assert result["taskProfile"]["inputModalities"] == ["text"]
    assert not hub.downloaded


def test_existing_native_token_classification_route_is_preserved(tmp_path):
    from transformers import BertConfig, BertForTokenClassification, BertTokenizer

    from oryn_engine.runners import RunnerRegistry, TransformersTaskHandle

    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=128,
        num_labels=3,
    )
    BertForTokenClassification(config).save_pretrained(tmp_path)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    vocab.update({f"t{i}": i for i in range(5, 32)})
    BertTokenizer(vocab=vocab).save_pretrained(tmp_path)
    registry = RunnerRegistry()
    registry._pack_locator = SimpleNamespace(
        resolve=lambda _: pytest.fail("Native token route changed")
    )
    record = {
        "model": {"provider": "huggingface", "repository": str(tmp_path), "revision": "a" * 40},
        "runner": "transformers",
        "cacheRoot": str(tmp_path),
        "technicalCapabilities": {
            "task": "token-classification",
            "architectures": ["BertForTokenClassification"],
        },
        "_runtimeCandidate": {"backend": "cpu"},
    }
    handle = registry._load_single(record)
    assert isinstance(handle, TransformersTaskHandle) and handle.task == "token-classification"
