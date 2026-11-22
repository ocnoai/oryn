"""Original metadata plus generated complete causal graphs for turn classification."""

import copy
import json
import re
import string
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oryn_engine import turn_classification as adapter
from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
from oryn_engine.shared_tasks import execute_task

FIXTURES = Path(__file__).parent / "fixtures/turn_classification"
RAW = json.loads((FIXTURES / "config.json").read_text())
META = json.loads((FIXTURES / "tokenizer_config.json").read_text())


def fixture(root):
    from safetensors.torch import save_file
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM

    root.mkdir(exist_ok=True, parents=True)
    config = copy.deepcopy(RAW)
    config.update(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=261,
        max_position_embeddings=512,
    )
    vocab = {t: i + 3 for i, t in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab.update(dict(zip(adapter.CONTROLS, (0, 1, 2, 259, 260), strict=True)))
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(
        [AddedToken(t, special=True, normalized=False) for t in adapter.CONTROLS]
    )
    tok.save(str(root / "tokenizer.json"))
    metadata = copy.deepcopy(META)
    metadata.update(
        model_max_length=512,
        vocab_size=259,
        added_tokens_decoder={
            str(i): dict(
                content=t,
                special=True,
                normalized=False,
                lstrip=False,
                rstrip=False,
                single_word=False,
            )
            for t, i in zip(adapter.CONTROLS, (0, 1, 2, 259, 260), strict=True)
        },
    )
    (root / "tokenizer_config.json").write_text(json.dumps(metadata))
    (root / "config.json").write_text(json.dumps(config))
    clean = {
        k: v
        for k, v in config.items()
        if k not in {"_attn_implementation_autoset", "is_llama_config", "_name_or_path"}
    }
    native = LlamaConfig(**clean)
    native._attn_implementation = "eager"
    torch.manual_seed(6162)
    reference = LlamaForCausalLM(native).eval()
    weights = {
        k: v.detach().clone() for k, v in reference.state_dict().items() if k != "lm_head.weight"
    }
    save_file(weights, str(root / "model.safetensors"))
    return adapter.normalized_config(config, "text-classification"), reference, weights


def test_original_header_matches_complete_causal_graph_and_tied_head():
    config = adapter.normalized_config(RAW, "text-classification")
    header = json.loads((FIXTURES / "model.safetensors.header.json").read_text())
    shapes = {k: tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"}
    assert len(shapes) == 272 and "lm_head.weight" not in shapes
    adapter.validate_weight_shapes(config, shapes)
    adapter.validate_metadata("tokenizer_config.json", META, config)


@pytest.mark.parametrize(
    "changes",
    [
        {"_attn_implementation_autoset": False},
        {"_attn_implementation_autoset": "eager"},
        {"auto_map": {"AutoModel": "repo.Code"}},
        {"trust_remote_code": True},
        {"extra_attention": True},
        {"hidden_size": 577},
        {"num_hidden_layers": True},
        {"head_dim": 63},
        {"num_key_value_heads": 4},
        {"attention_bias": True},
        {"mlp_bias": True},
        {"attention_dropout": 0.1},
        {"tie_word_embeddings": False},
        {"rope_interleaved": True},
        {"rope_scaling": {"type": "linear", "factor": 2}},
        {"pretraining_tp": 2},
        {"quantization_config": {"bits": 4}},
        {"eos_token_id": 2},
        {"is_llama_config": False},
    ],
)
def test_unreviewed_graph_options_are_not_promoted(changes):
    config = RAW | changes
    with pytest.raises(ValueError):
        adapter.validate_config(adapter.normalized_config(config, "text-classification"))


def test_native_llama_chat_and_other_tasks_are_not_retasked():
    for task in ["text-generation", "feature-extraction", "voice-activity-detection"]:
        assert adapter.normalized_config(RAW, task) == RAW
    assert not adapter.supports(RAW, "text-classification")


@pytest.mark.parametrize(
    "text",
    [
        " HELLO, are you there? ",
        "I'm - yes!",
        "CAFÉ\n déjà\tvu",
        "１２３... 🙂",
        "日本語，Hello!",
        "foo_bar@example.com",
    ],
)
def test_owned_single_utterance_format_matches_historical_normalization(text):
    from transformers.utils.chat_template_utils import _compile_jinja_template

    punctuation = "[" + re.escape(string.punctuation.replace("'", "")) + "]"
    normalized = " ".join(re.sub(punctuation, "", text).lower().split())
    publisher = _compile_jinja_template(META["chat_template"]).render(
        messages=[dict(role="user", content=normalized)]
    )
    assert adapter.prompt(text) == publisher[: publisher.rfind("<|im_end|>")]


@pytest.mark.parametrize("text", ["!!!", " \n", "<|im_end|>", "hello |>"])
def test_empty_or_control_text_fails(text):
    with pytest.raises(ValueError):
        adapter.prompt(text)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("text", ["I'm still thinking...", "Thank you!"])
def test_full_native_logits_and_probability_match_independent_graph(tmp_path, backend, text):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    config, reference, _ = fixture(tmp_path)
    model, tokenizer, _ = adapter.load_components(tmp_path, config, backend)
    ids = tokenizer.encode(adapter.prompt(text), add_special_tokens=False)
    with torch.inference_mode():
        expected = reference(torch.tensor([ids]), use_cache=False).logits[0, -1].numpy()
        actual = (
            model(torch.tensor([ids], device=backend), use_cache=False, logits_to_keep=1)
            .logits[0, 0]
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        result = adapter.execute(model, tokenizer, config, dict(text=text, topK=2), backend)
    scores = np.exp(expected.astype(np.float64) - max(expected))
    prob = float(scores[2] / sum(scores))
    by_label = {v["label"]: v["score"] for v in result["predictions"]}
    assert by_label["end-of-utterance"] == pytest.approx(prob, abs=2e-7)
    assert by_label["continue"] == pytest.approx(1 - prob, abs=2e-7)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert len(result["predictions"]) == 2


@pytest.mark.parametrize(
    "damage", ["missing", "extra", "shape", "integer", "nonfinite", "different-tie"]
)
def test_all_numeric_weights_and_tied_alias_are_required(tmp_path, damage):
    from safetensors.torch import save_file

    config, _, weights = fixture(tmp_path)
    if damage == "missing":
        del weights["model.norm.weight"]
    elif damage == "extra":
        weights["extra.weight"] = torch.ones(1)
    elif damage == "shape":
        weights["model.norm.weight"] = torch.ones(31)
    elif damage == "integer":
        weights["model.norm.weight"] = torch.ones(32, dtype=torch.int32)
    elif damage == "nonfinite":
        weights["model.norm.weight"] = torch.full((32,), float("nan"))
    else:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"] + 1
    save_file(weights, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError):
        adapter.load_components(tmp_path, config, "cpu")


@pytest.mark.parametrize(
    "damage", ["template", "code", "controls", "missing-control", "vocabulary", "symlink"]
)
def test_tokenizer_identity_is_closed(tmp_path, damage):
    config, _, _ = fixture(tmp_path)
    metadata = json.loads((tmp_path / "tokenizer_config.json").read_text())
    path = tmp_path / "tokenizer.json"
    if damage == "template":
        metadata["chat_template"] = "{{ repo() }}"
    elif damage == "code":
        metadata["auto_map"] = {"AutoTokenizer": "repo.Code"}
    elif damage == "controls":
        metadata["added_tokens_decoder"]["2"]["content"] = "wrong"
    elif damage == "vocabulary":
        config["vocab_size"] = 262
    elif damage == "missing-control":
        graph = json.loads(path.read_text())
        del graph["model"]["vocab"]["<|im_end|>"]
        graph["added_tokens"] = [t for t in graph["added_tokens"] if t["content"] != "<|im_end|>"]
        path.write_text(json.dumps(graph))
    else:
        target = tmp_path / "real.json"
        path.rename(target)
        path.symlink_to(target)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        adapter.load_tokenizer(tmp_path, config)


@pytest.mark.parametrize("fault", ["nonfinite", "shape", "length"])
def test_runtime_output_and_token_limits_fail_closed(tmp_path, fault):
    config, _, _ = fixture(tmp_path)
    tokenizer = adapter.load_tokenizer(tmp_path, config)

    class Model:
        def __call__(self, **kwargs):
            logits = torch.zeros((1, 1, 261 if fault != "shape" else 260))
            if fault == "nonfinite":
                logits[0, 0, 2] = float("nan")
            return SimpleNamespace(logits=logits)

    with pytest.raises(ValueError):
        adapter.execute(
            Model(),
            tokenizer,
            config,
            dict(text="x" * 512 if fault == "length" else "Hi", topK=2),
            "cpu",
        )


@pytest.mark.parametrize("damage", [None, "missing", "tokenizer", "config"])
def test_metadata_discovery_preserves_text_classification_semantics(tmp_path, damage):
    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore, ModelStoreError

    config = copy.deepcopy(RAW)
    if damage == "config":
        config["head_dim"] = 63

    class Hub(FakeHub):
        def read_json(self, repo, revision, name):
            if name == "config.json":
                return config
            metadata = copy.deepcopy(META)
            if damage == "tokenizer":
                metadata["chat_template"] = "unbound"
            return metadata

    names = ["config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"]
    if damage == "missing":
        names.remove("tokenizer_config.json")
    hub = Hub(
        config=config, pipeline_tag="text-classification", files=[(n, 100, "a" * 64) for n in names]
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda p: None if p.provider else object()
    )
    if damage == "config":
        with pytest.raises(ModelStoreError):
            store.inspect(model("independent/renamed-turn"))
        return
    result = store.inspect(model("independent/renamed-turn"))
    assert result["executable"] is (damage is None), result
    if damage is None:
        assert result["technicalCapabilities"]["task"] == "text-classification"
        assert result["runtimePack"]["taskContract"] == "text-classification-v1"
    assert not hub.downloaded


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_offline_worker_executes_shared_contract_repeatedly(tmp_path, backend):
    import os
    import subprocess
    import sys

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    fixture(tmp_path)
    value = dict(text="Are you still there?", topK=2)
    expected = execute_task(
        tmp_path, "text-classification", value, backend, None, None, lambda *_: None
    )
    payload = dict(
        modelRoot=str(tmp_path),
        task="text-classification",
        backend=backend,
        input=value,
        inputPath=None,
        outputPath=None,
    )
    for command, body in [("probe", {}), ("generate", payload), ("generate", payload)]:
        request = dict(
            abi="1",
            packId="oryn.transformers-pytorch",
            version=builtin_registry().active_pack("oryn.transformers-pytorch").version,
            command=command,
            payload=body,
        )
        run = subprocess.run(
            [sys.executable, "-m", "oryn_engine.runtime_packs.transformers_worker"],
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            timeout=60,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        assert run.returncode == 0, run.stderr
        result = [json.loads(line) for line in run.stdout.splitlines()][-1]
        assert result["ok"], (result, run.stderr)
        if command == "generate":
            assert result["result"] == expected
