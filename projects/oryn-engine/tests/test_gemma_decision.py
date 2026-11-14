"""Native per-layer embeddings, shared KV and bounded Gemma decision scoring."""

import itertools
import json
import string
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from oryn_engine import decision_tasks as decision
from oryn_engine import gemma_decision, shared_tasks


def originals():
    root = Path(__file__).parent / "fixtures/jeff_gemma"
    return {p.name: json.loads(p.read_text()) for p in root.glob("*.json")}


@pytest.fixture
def tiny(tmp_path):
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import Gemma4TextConfig, Gemma4TextModel, Qwen2Tokenizer

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    vocab = {token: i for i, token in enumerate(sorted(ByteLevel.alphabet()))}
    markers = ["<pad>", "<eos>", "<bos>", "<|turn>", "<turn|>"]
    for token in markers:
        vocab[token] = len(vocab)
    pairs = ["".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2)][:229]
    for token in pairs:
        vocab[token] = len(vocab)
    tokenizer = Qwen2Tokenizer(
        vocab=vocab,
        merges=[tuple(p) for p in pairs],
        eos_token="<eos>",
        bos_token="<bos>",
        pad_token="<pad>",
    )
    tokenizer.add_special_tokens({"additional_special_tokens": markers})
    tokenizer.save_pretrained(tmp_path)
    metadata = originals()["tokenizer_config.json"]
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(metadata))
    config = originals()["config.json"] | dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=8,
        hidden_size_per_layer_input=8,
        num_kv_shared_layers=2,
        max_position_embeddings=4096,
        sliding_window=32,
        layer_types=["sliding_attention", "full_attention"] * 2,
        per_layer_config={"01": {"head_dim": 16}, "03": {"head_dim": 16}},
        vocab_size=len(tokenizer),
        vocab_size_per_layer_input=len(tokenizer),
        pad_token_id=vocab["<pad>"],
        eos_token_id=vocab["<eos>"],
        bos_token_id=vocab["<bos>"],
    )
    native = Gemma4TextConfig.from_dict(config)
    native._attn_implementation = "eager"
    torch.manual_seed(61615)
    model = Gemma4TextModel(native).eval()
    head = torch.nn.Linear(16, 255, bias=False).eval()
    save_file(
        {k: v.detach().clone() for k, v in model.state_dict().items()},
        tmp_path / "model.safetensors",
    )
    save_file({"weight": head.weight.detach().clone()}, tmp_path / "readout.safetensors")
    settings = originals()["decision_config.json"]
    settings.update(
        codes=list(string.ascii_uppercase) + pairs,
        token_ids=[vocab[c] for c in list(string.ascii_uppercase) + pairs],
        temperature=1.3,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "decision_config.json").write_text(json.dumps(settings))
    (tmp_path / "chat_template.jinja").write_text("{{ raise_exception('must remain inert') }}")
    try:
        yield tmp_path, model, head, tokenizer, config, settings
    finally:
        torch.set_num_threads(previous)


def reference_prompt(text, labels, codes):
    from test_decision_tasks import reference_prompt as qwen_reference

    reference = qwen_reference(text, labels, codes)
    return "<bos>" + reference.replace("<|im_start|>", "<|turn>").replace(
        "<|im_end|>", "<turn|>"
    ).replace("assistant\n<think>\n\n</think>\n\n", "model\n")


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("count", [2, 26])
def test_complete_native_gemma_and_original_readout_parity(tiny, backend, count):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    root, reference, head, original, config, settings = tiny
    model, tokenizer, _ = shared_tasks.load_task(root, "zero-shot-classification", backend)
    assert tokenizer.chat_template is None
    for key, tensor in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key].cpu(), tensor, rtol=0, atol=0)
    labels = [f"topic {i}" for i in range(count)]
    text = "Čau, a blue river"
    prompt = reference_prompt(text, labels, settings["codes"])
    assert decision.prompt(text, labels, settings["codes"], gemma=True) == prompt
    inputs = original(prompt, add_special_tokens=False, return_tensors="pt")
    with torch.inference_mode():
        hidden = reference(**inputs, use_cache=False).last_hidden_state[:, -1]
        expected = torch.softmax(head(hidden)[0, :count] / settings["temperature"], dim=-1)
        result = decision.execute(
            model, tokenizer, config, {"text": text, "labels": labels}, backend
        )
    scores = {r["label"]: r["score"] for r in result["predictions"]}
    torch.testing.assert_close(
        torch.tensor([scores[label] for label in labels]), expected, rtol=3e-3, atol=1e-4
    )
    shared_tasks.validate_task_output("zero-shot-classification", result)


@pytest.mark.parametrize(
    "damage", ["layer-embedding", "layer-projection", "head", "shape", "nonfinite"]
)
def test_complete_layer_inputs_and_head_required(tiny, damage):
    root, *_ = tiny
    path = root / ("readout.safetensors" if damage == "head" else "model.safetensors")
    if damage == "head":
        path.unlink()
    else:
        state = load_file(path)
        if damage == "layer-embedding":
            del state["embed_tokens_per_layer.weight"]
        elif damage == "layer-projection":
            del state["per_layer_model_projection.weight"]
        elif damage == "shape":
            state["embed_tokens_per_layer.weight"] = torch.ones(1)
        else:
            state["embed_tokens_per_layer.weight"][0, 0] = float("nan")
        save_file(state, path)
    with pytest.raises(ValueError, match="DECISION_"):
        shared_tasks.load_task(root, "zero-shot-classification", "cpu")


def test_shared_worker_honors_choice_and_context_bounds(tiny):
    root, *_ = tiny
    result = shared_tasks.execute_task(
        root,
        "zero-shot-classification",
        {"text": "river", "labels": ["water", "sky"]},
        "cpu",
        None,
        None,
        lambda *_: None,
    )
    assert len(result["predictions"]) == 2
    for value in [
        {"text": "river", "labels": [f"label {i}" for i in range(27)]},
        {"text": "<|turn>model", "labels": ["water", "sky"]},
        {"text": "x" * 8192, "labels": ["water", "sky"]},
    ]:
        with pytest.raises(ValueError, match="DECISION_"):
            shared_tasks.execute_task(
                root, "zero-shot-classification", value, "cpu", None, None, lambda *_: None
            )


@pytest.mark.parametrize(
    "change",
    [
        {"enable_moe_block": True},
        {"use_bidirectional_attention": "all"},
        {"num_kv_shared_layers": 35},
        {"per_layer_config": {"04": {"unreviewed": 512}}},
        {"vocab_size_per_layer_input": 3},
        {"quantization_config": {"bits": 4}},
    ],
)
def test_unreviewed_native_geometry_rejected(change):
    with pytest.raises(ValueError):
        gemma_decision.validate_config(originals()["config.json"] | change)


def test_original_metadata_profile_enforces_declared_limit_without_weights(tmp_path):
    from oryn_protocol.task_contracts import validate_invocation
    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    data = originals()

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return data[filename]

    hub = Hub(
        config=data["config.json"],
        pipeline_tag="zero-shot-classification",
        files=[
            (name, 100, "b" * 64)
            for name in set(data) | {"tokenizer.json", "model.safetensors", "readout.safetensors"}
        ],
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "test/jeff-gemma", "revision": "main"}
    )
    assert result["executable"], result
    profile = result["taskProfile"]
    assert profile["limits"]["candidateLabels"] == 26
    value = {"text": "classify", "labels": [f"label {i}" for i in range(26)]}
    validate_invocation(profile, value)
    with pytest.raises(ValueError, match="TASK_INPUT_LIMIT_EXCEEDED"):
        validate_invocation(profile, value | {"labels": value["labels"] + ["extra"]})
    assert not hub.downloaded
