"""Whole native graph, contextual semantics and metadata-only discovery."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from oryn_engine import contextual_embeddings as binding
from oryn_engine import shared_tasks
from oryn_engine.generated_capabilities import normalized_transformers_config

FIXTURES = Path(__file__).parent / "fixtures/contextual_embeddings"


def original(name="config.json"):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def tiny(tmp_path):
    from test_qwen_translation import fixture
    from transformers import Qwen3_5Model

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    _, tokenizer, config, _ = fixture(tmp_path, mtp=0)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[Q] ", "[D] "]})
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.save_pretrained(tmp_path)
    config.update(
        model_type=binding.FAMILY,
        architectures=["PplxContextualModel"],
        embedding_dim=16,
        query_prefix="[Q] ",
        document_prefix="[D] ",
        boundary_marker="<|chunk_sep|>",
        query_length=1024,
        document_length=1024,
    )
    config["text_config"].update(is_causal=False, use_cache=False, vocab_size=512)
    native_config = binding.qwen_translation.native_config(binding.graph_config(config))
    reference = Qwen3_5Model(native_config).float().eval()
    projection = torch.nn.Linear(32, 16, bias=False)
    state = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    state["contextual_projection.weight"] = projection.weight.detach().clone()
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(config))
    yield tmp_path, config, reference, projection, tokenizer
    torch.set_num_threads(old)


def independent_reference(reference, projection, tokenizer, config, value):
    """Pinned algorithm: explicit token overlap, direct means and int8-valued floats.

    Source: perplexity-ai/pplx-embed-v2-context-9b-preview at
    b667039ee8b438a6350fbc91bbcecd86f9d363ba/modeling_pplx_contextual.py.
    The engine uses vectorized prefix sums; this reference pools each slice.
    """
    kind = value["inputType"]
    rows = [value["texts"]] if value["contextual"] else [[t] for t in value["texts"]]
    texts, character_spans = [], []
    for chunks in rows:
        text = "" if kind == "query" else "[D] "
        spans = []
        for index, chunk in enumerate(chunks):
            if index:
                text += "<|chunk_sep|>"
            spans.append((len(text), len(text) + len(chunk)))
            text += chunk
        texts.append(text)
        character_spans.append(spans)
    values = tokenizer(
        texts,
        padding=True,
        truncation=False,
        split_special_tokens=True,
        return_tensors="pt",
        return_token_type_ids=False,
        return_offsets_mapping=True,
    )
    offsets = values.pop("offset_mapping")
    if kind == "query":
        prefix = torch.full((len(rows), 1), tokenizer.convert_tokens_to_ids("[Q] "))
        values["input_ids"] = torch.cat((prefix, values["input_ids"]), -1)
        values["attention_mask"] = torch.cat(
            (torch.ones_like(prefix), values["attention_mask"]), -1
        )
    with torch.inference_mode():
        hidden = reference(**values, use_cache=False).last_hidden_state.float()
        pooled = []
        for row, spans in enumerate(character_spans):
            if kind == "query":
                pooled.append(hidden[row, values["attention_mask"][row].bool()].mean(0))
            else:
                for start, end in spans:
                    indexes = [
                        i
                        for i, (a, b) in enumerate(offsets[row].tolist())
                        if b > a and a < end and b > start and values["attention_mask"][row, i]
                    ]
                    pooled.append(hidden[row, indexes[0] : indexes[-1] + 1].mean(0))
        vectors = torch.round(torch.tanh(projection(torch.stack(pooled))) * 127).clamp(-128, 127)
        if value["normalize"]:
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
    return values, vectors


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize(
    "kind,contextual", [("query", False), ("document", False), ("document", True)]
)
@pytest.mark.parametrize("normalize", [False, True])
def test_complete_native_context_query_and_document_parity(
    tiny, backend, kind, contextual, normalize
):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    root, config, reference, projection, tokenizer = tiny
    loaded, owned, selected = shared_tasks.load_task(root, "feature-extraction", backend)
    assert selected == config and owned.chat_template is None
    assert owned.eos_token_id != config["text_config"]["eos_token_id"]
    for key, tensor in reference.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key].cpu(), tensor, rtol=0, atol=0)
    torch.testing.assert_close(loaded.contextual_projection.weight.cpu(), projection.weight)
    value = {
        "texts": ["Short text.", "Žltý 日本語 <|im_end|> [Q] longer chunk"],
        "inputType": kind,
        "contextual": contextual,
        "normalize": normalize,
    }
    expected_inputs, expected = independent_reference(
        reference, projection, tokenizer, config, value
    )
    actual_inputs = binding.prepare_inputs(owned, config, value)
    for key, tensor in expected_inputs.items():
        torch.testing.assert_close(actual_inputs[key], tensor, rtol=0, atol=0)
    actual = binding.execute(loaded, owned, config, value, backend)
    torch.testing.assert_close(torch.tensor(actual["embeddings"]), expected, rtol=1e-5, atol=1e-6)
    assert actual["dimensions"] == 16 and actual["normalized"] == normalize
    if normalize:
        torch.testing.assert_close(expected.norm(dim=-1), torch.ones(2))
    else:
        assert (expected == expected.round()).all() and expected.abs().max() <= 127
    assert actual == shared_tasks.execute_task(
        root, "feature-extraction", value, backend, None, None, lambda *_: None
    )


def test_context_changes_earlier_chunk_and_padding_does_not(tiny):
    root, config, _, _, _ = tiny
    model, tokenizer, _ = shared_tasks.load_task(root, "feature-extraction", "cpu")
    value = {
        "texts": ["The first chunk.", "A different much longer second chunk."],
        "inputType": "document",
        "contextual": False,
        "normalize": False,
    }
    separate = binding.execute(model, tokenizer, config, value, "cpu")["embeddings"]
    first = binding.execute(model, tokenizer, config, value | {"texts": value["texts"][:1]}, "cpu")[
        "embeddings"
    ]
    together = binding.execute(model, tokenizer, config, value | {"contextual": True}, "cpu")[
        "embeddings"
    ]
    query = binding.execute(model, tokenizer, config, value | {"inputType": "query"}, "cpu")[
        "embeddings"
    ]
    assert separate[0] == first[0]
    assert together[0] != separate[0] and query != separate


@pytest.mark.parametrize(
    "damage",
    [
        "vision",
        "linear",
        "attention",
        "projection",
        "embedding",
        "shape",
        "extra",
        "nan",
        "integer",
    ],
)
def test_complete_numeric_graph_required(tiny, damage):
    root, _, _, _, _ = tiny
    state = load_file(root / "model.safetensors")
    fragment = {
        "vision": "visual.patch_embed",
        "linear": "linear_attn.conv1d",
        "attention": "self_attn.q_proj",
        "projection": "contextual_projection",
        "embedding": "embed_tokens",
    }.get(damage, "contextual_projection")
    key = next(k for k in state if fragment in k)
    if damage in {"vision", "linear", "attention", "projection", "embedding"}:
        del state[key]
    elif damage == "shape":
        state[key] = torch.zeros(3, 3)
    elif damage == "extra":
        state["unreviewed.weight"] = torch.ones(3)
    elif damage == "nan":
        state[key].flatten()[0] = float("nan")
    else:
        state[key] = state[key].to(torch.int32)
    save_file(state, root / "model.safetensors")
    with pytest.raises(ValueError):
        shared_tasks.load_task(root, "feature-extraction", "cpu")


@pytest.mark.parametrize("damage", [None, "index", "duplicate"])
def test_separate_projection_shard_is_strictly_bound(tiny, damage):
    root, config, _, _, _ = tiny
    state = load_file(root / "model.safetensors")
    head = {"contextual_projection.weight": state.pop("contextual_projection.weight")}
    save_file(state, root / "model.safetensors")
    save_file(head, root / "contextual_head.safetensors")
    mapping = {k: "model.safetensors" for k in state} | {
        k: "contextual_head.safetensors" for k in head
    }
    if damage == "index":
        mapping.pop("contextual_projection.weight")
    if damage == "duplicate":
        save_file(head, root / "extra.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    if damage:
        with pytest.raises(ValueError):
            binding.load_components(root, config, "cpu")
    else:
        loaded, _, _ = binding.load_components(root, config, "cpu")
        torch.testing.assert_close(
            loaded.contextual_projection.weight, head["contextual_projection.weight"]
        )


@pytest.mark.parametrize(
    "damage", ["remote", "prefix", "causal", "cache", "vision", "quantized", "dimension"]
)
def test_closed_original_configuration(damage):
    config = original()
    if damage == "remote":
        config["auto_map"]["AutoModel"] = "unknown.RemoteModel"
    elif damage == "prefix":
        config["document_prefix"] = "passage: "
    elif damage == "causal":
        config["text_config"]["is_causal"] = True
    elif damage == "cache":
        config["text_config"]["use_cache"] = True
    elif damage == "vision":
        config["vision_config"]["out_hidden_size"] = 17
    elif damage == "quantized":
        config["quantization_config"] = {"quant_method": "bitsandbytes"}
    else:
        config["embedding_dim"] = True
    with pytest.raises(ValueError):
        normalized_transformers_config(config, "feature-extraction")


def test_configuration_validation_does_not_import_native_framework():
    script = f"""
import json,sys
from pathlib import Path
from oryn_engine.generated_capabilities import normalized_transformers_config
from oryn_engine.contextual_embeddings import validate_metadata
root=Path({str(FIXTURES)!r})
config=normalized_transformers_config(json.loads((root/'config.json').read_text()),'feature-extraction')
metadata=json.loads((root/'tokenizer_config.json').read_text())
validate_metadata('tokenizer_config.json', metadata, config)
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
assert config['text_config']['vocab_size']==248320
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize(
    "damage", ["vocabulary", "prefix", "remote-code", "tokenizer-class", "missing-marker"]
)
def test_tokenizer_and_vocabulary_are_bound(tiny, damage):
    root, config, _, _, _ = tiny
    if damage == "vocabulary":
        config = copy.deepcopy(config)
        config["text_config"]["vocab_size"] = 513
    elif damage == "prefix":
        config = config | {"query_prefix": "unknown prefix"}
    elif damage == "missing-marker":
        path = root / "tokenizer.json"
        path.write_text(path.read_text().replace("<|im_end|>", "<|renamed|>"))
    else:
        path = root / "tokenizer_config.json"
        value = json.loads(path.read_text())
        value.update(
            {"auto_map": {"AutoTokenizer": "remote.Tokenizer"}}
            if damage == "remote-code"
            else {"tokenizer_class": "Unknown"}
        )
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        binding.tokenizer(root, config)


def test_public_profile_modes_limits_and_defaults(tiny):
    from oryn_protocol.task_contracts import effective_input, validate_invocation

    from oryn_engine.task_profiles import profile_for_execution

    root, config, _, _, _ = tiny
    profile = profile_for_execution(
        {"runtimePack": {"runnerId": "transformers.shared-tasks", "taskContract": "embeddings-v1"}},
        config,
    )
    effective = effective_input(profile, {"texts": ["hello", "world"]})
    assert effective == {
        "texts": ["hello", "world"],
        "normalize": True,
        "inputType": "document",
        "contextual": False,
    }
    for mode in ({"contextual": True}, {"inputType": "query"}, {"normalize": False}):
        validate_invocation(profile, effective | mode)
    for change in (
        {"inputType": "query", "contextual": True},
        {"texts": ["x"] * 5},
        {"texts": [""]},
        {"texts": ["a" * 8193]},
        {"contextual": "yes"},
    ):
        with pytest.raises(ValueError):
            validate_invocation(profile, effective | change)
    tokenizer = binding.tokenizer(root, config)
    with pytest.raises(ValueError, match="TOKEN_LIMIT"):
        binding.prepare_inputs(tokenizer, config, effective | {"texts": ["x " * 1000]})
    with pytest.raises(ValueError, match="INPUT_MODE"):
        binding.prepare_inputs(
            tokenizer, config, effective | {"inputType": "query", "contextual": True}
        )


def test_ordinary_embedding_models_cannot_ignore_mode_controls(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
    with pytest.raises(ValueError, match="EMBEDDING_INPUT_MODE_UNSUPPORTED"):
        shared_tasks.execute_task(
            tmp_path,
            "feature-extraction",
            {"texts": ["hello"], "inputType": "query"},
            "cpu",
            None,
            None,
            lambda *_: None,
        )


@pytest.mark.parametrize(
    "damage", [None, "tokenizer", "missing-tokenizer", "missing-weights", "remote-code"]
)
def test_original_metadata_discovery(tmp_path, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore, ModelStoreError
    from oryn_engine.runtime_packs.execution import PackRuntime

    config = original()
    if damage == "remote-code":
        config["auto_map"]["AutoModel"] = "remote.Model"
    files = [("config.json", 100, "a" * 64), ("tokenizer.json", 100, "b" * 64)]
    if damage != "missing-tokenizer":
        files.append(("tokenizer_config.json", 100, "b" * 64))
    if damage != "missing-weights":
        files.append(("model.safetensors", 100, "b" * 64))

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            if filename == "config.json":
                return config
            if filename == "tokenizer_config.json":
                value = original(filename)
                if damage == "tokenizer":
                    value["tokenizer_class"] = "Unknown"
                return value
            raise OSError(filename)

    hub = Hub(config=config, files=files, pipeline_tag="feature-extraction")
    store = ModelStore(tmp_path / "store", hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    request = {"provider": "huggingface", "repository": "test/contextual", "revision": "main"}
    if damage == "remote-code":
        with pytest.raises(ModelStoreError):
            store.inspect(request)
        return
    result = store.inspect(request)
    assert not hub.downloaded
    assert result["executable"] == (damage is None), result
    if not damage:
        assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
