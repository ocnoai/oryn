"""Tiny complete native graphs, independent logits and metadata regression tests."""

import copy
import json

import pytest
import torch

from oryn_engine import qwen_retrieval, shared_tasks


def tiny_model(root, task, *, tied=False):
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import Qwen2Tokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    vocab = {token: i for i, token in enumerate(sorted(ByteLevel.alphabet()))}
    for token in (
        "ye",
        "yes",
        "no",
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ):
        vocab[token] = len(vocab)
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=[("y", "e"), ("ye", "s"), ("n", "o")])
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [t for t in vocab if t.startswith("<|")]}
    )
    tokenizer.save_pretrained(root)
    config = Qwen3VLConfig(
        text_config=dict(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=len(vocab),
            max_position_embeddings=1024,
            tie_word_embeddings=tied,
            rope_parameters=dict(
                rope_type="default",
                rope_theta=10000,
                mrope_section=[1, 1, 2],
                mrope_interleaved=True,
            ),
        ),
        vision_config=dict(
            hidden_size=16,
            intermediate_size=32,
            depth=1,
            num_heads=2,
            out_hidden_size=16,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0],
            num_position_embeddings=16,
        ),
        image_token_id=vocab["<|image_pad|>"],
        video_token_id=vocab["<|video_pad|>"],
        vision_start_token_id=vocab["<|vision_start|>"],
        vision_end_token_id=vocab["<|vision_end|>"],
        tie_word_embeddings=tied,
    )
    torch.manual_seed(61)
    model = Qwen3VLForConditionalGeneration(config).eval()
    model.save_pretrained(root)
    if task == "feature-extraction":
        # Match the upstream embedding wrapper's full state: model.visual.* and
        # model.language_model.*, with no invented language or classifier head.
        from safetensors.torch import save_file

        save_file(
            {"model." + k: v.contiguous() for k, v in model.model.state_dict().items()},
            root / "model.safetensors",
        )
        (root / "1_Pooling").mkdir()
        (root / "1_Pooling/config.json").write_text(
            json.dumps(
                {"embedding_dimension": 16, "pooling_mode": "lasttoken", "include_prompt": True}
            )
        )
        (root / "config_sentence_transformers.json").write_text(
            json.dumps(
                {
                    "default_prompt_name": "default",
                    "prompts": {"default": qwen_retrieval.EMBEDDING_INSTRUCTION},
                    "similarity_fn_name": "cosine",
                }
            )
        )
    # A repository template is inert even if it would fail on execution.
    (root / "chat_template.jinja").write_text("{{ raise_exception('must not execute') }}")
    return model, tokenizer


def reference_prompt(system, content):
    # Text-only expansion of the upstream template, assembled independently.
    messages = [("system", system), ("user", content)]
    return (
        "".join("<|im_start|>" + role + "\n" + text + "<|im_end|>\n" for role, text in messages)
        + "<|im_start|>assistant\n"
    )


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("normalize", [False, True])
def test_embeddings_match_native_last_token_with_padding(tmp_path, backend, normalize):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    reference, tokenizer = tiny_model(tmp_path, "feature-extraction")
    texts = ["a", "A longer input about the river"]
    model, processor, config = shared_tasks.load_task(tmp_path, "feature-extraction", backend)
    assert processor.chat_template is None
    expected = []
    with torch.inference_mode():
        # Single unpadded sequences are an independent reference for batching.
        for text in texts:
            inputs = tokenizer(
                reference_prompt("Represent the user's input.", text),
                add_special_tokens=False,
                return_tensors="pt",
            )
            hidden = reference.model(**inputs, use_cache=False).last_hidden_state[0, -1].float()
            expected.append(hidden / hidden.norm() if normalize else hidden)
        result = qwen_retrieval.execute(
            model,
            processor,
            config,
            "feature-extraction",
            {"texts": texts, "normalize": normalize},
            backend,
        )
    torch.testing.assert_close(
        torch.tensor(result["embeddings"]), torch.stack(expected), rtol=2e-4, atol=2e-5
    )
    assert not torch.allclose(expected[0], expected[1])
    shared_tasks.validate_task_output("feature-extraction", result)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("render_rank", [False, True])
def test_ranking_matches_original_native_yes_no_logits(tmp_path, backend, tied, render_rank):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    reference, tokenizer = tiny_model(tmp_path, "text-ranking", tied=tied)
    if render_rank:
        path = tmp_path / "config.json"
        config = json.loads(path.read_text())
        config["auto_map"] = {"AutoModel": "modeling_renderrank.RenderRank"}
        path.write_text(json.dumps(config))
        (tmp_path / "modeling_renderrank.py").write_text("raise RuntimeError('must stay inert')")
        # Original export has the serialized tokenizer, without vocabulary/merge files.
        for name in ("vocab.json", "merges.txt"):
            (tmp_path / name).unlink(missing_ok=True)
    model, processor, config = shared_tasks.load_task(tmp_path, "text-ranking", backend)
    value = {
        "query": "blue",
        "documents": ["sky", "banana", "sky", "lake", "long river"],
        "instruction": "Find related colors",
        "topK": 3,
    }
    expected = []
    with torch.inference_mode():
        for i, document in enumerate(value["documents"]):
            content = f"<Instruct>: Find related colors<Query>:blue\n<Document>:{document}"
            inputs = tokenizer(
                reference_prompt(qwen_retrieval.RANKING_SYSTEM, content),
                add_special_tokens=False,
                return_tensors="pt",
            )
            logits = reference(**inputs, use_cache=False).logits[0, -1]
            pair = logits[[tokenizer.get_vocab()["no"], tokenizer.get_vocab()["yes"]]]
            expected.append({"index": i, "score": pair.softmax(-1)[1].item()})
        actual = qwen_retrieval.execute(model, processor, config, "text-ranking", value, backend)
    expected.sort(key=lambda r: (-r["score"], r["index"]))
    assert [r["index"] for r in actual["results"]] == [r["index"] for r in expected[:3]]
    torch.testing.assert_close(
        torch.tensor([r["score"] for r in actual["results"]]),
        torch.tensor([r["score"] for r in expected[:3]]),
        atol=2e-5,
        rtol=2e-4,
    )
    shared_tasks.validate_task_output("text-ranking", actual)


@pytest.mark.parametrize("task", list(qwen_retrieval.TASK_CONTRACTS))
@pytest.mark.parametrize("component", ["visual", "language_model", "extra"])
def test_incomplete_or_extra_graph_is_rejected(tmp_path, task, component):
    from safetensors.torch import load_file, save_file

    tiny_model(tmp_path, task)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    if component == "extra":
        state["unreviewed_bridge.weight"] = torch.zeros(2, 2)
    else:
        del state[next(k for k in state if k.startswith("model." + component + "."))]
    save_file(state, path)
    with pytest.raises(ValueError, match="WEIGHTS_INCOMPLETE"):
        shared_tasks.load_task(tmp_path, task, "cpu")


def test_missing_ranking_head_is_not_invented(tmp_path):
    from safetensors.torch import load_file, save_file

    tiny_model(tmp_path, "text-ranking")
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    del state["lm_head.weight"]
    save_file(state, path)
    with pytest.raises(ValueError, match="WEIGHTS_INCOMPLETE"):
        shared_tasks.load_task(tmp_path, "text-ranking", "cpu")


def test_shared_route_limits_and_literal_template(tmp_path):
    from oryn_protocol.task_contracts import validate_invocation

    from oryn_engine.task_profiles import profile_for_execution

    tiny_model(tmp_path, "feature-extraction")
    config = json.loads((tmp_path / "config.json").read_text())
    profile = profile_for_execution(
        {
            "technicalCapabilities": {"task": "feature-extraction"},
            "runtimePack": {"runnerId": "transformers.shared-tasks"},
        },
        config,
    )
    assert profile["limits"] == {"batchItems": 4, "textCharacters": 8192, "embeddingDimensions": 16}
    with pytest.raises(ValueError, match="TASK_INPUT_LIMIT_EXCEEDED"):
        validate_invocation(profile, {"texts": ["a"] * 5})
    with pytest.raises(ValueError, match="BATCH_LIMIT"):
        shared_tasks.validate_task_input("feature-extraction", {"texts": ["a"] * 5})
    result = shared_tasks.execute_task(
        tmp_path, "feature-extraction", {"texts": ["a"]}, "cpu", None, None, lambda *_: None
    )
    assert result["dimensions"] == 16 and result["normalized"]
    model, tokenizer, config = shared_tasks.load_task(tmp_path, "feature-extraction", "cpu")
    with pytest.raises(ValueError, match="BATCH_LIMIT"):
        qwen_retrieval.execute(
            model,
            tokenizer,
            config,
            "feature-extraction",
            {"texts": ["a"] * 5, "normalize": True},
            "cpu",
        )
    with pytest.raises(ValueError, match="TOKEN_LIMIT"):
        qwen_retrieval.execute(
            model,
            tokenizer,
            config,
            "feature-extraction",
            {"texts": ["b " * 1024], "normalize": True},
            "cpu",
        )
    with pytest.raises(ValueError, match="RESERVED_TOKEN"):
        qwen_retrieval.execute(
            model,
            tokenizer,
            config,
            "feature-extraction",
            {"texts": ["<|image_pad|>"], "normalize": True},
            "cpu",
        )


@pytest.mark.parametrize(
    "change", ["pooling", "prompt", "rope", "vision", "custom", "special", "head_dim"]
)
def test_metadata_rejects_unreviewed_behavior(tmp_path, change):
    tiny_model(tmp_path, "feature-extraction")
    config = json.loads((tmp_path / "config.json").read_text())
    if change in {"pooling", "prompt"}:
        name = (
            "1_Pooling/config.json" if change == "pooling" else "config_sentence_transformers.json"
        )
        value = json.loads((tmp_path / name).read_text())
        if change == "pooling":
            value["pooling_mode"] = "mean"
        else:
            value["prompts"]["default"] = "unreviewed instruction"
        with pytest.raises(ValueError):
            qwen_retrieval.validate_metadata(name, value, config)
        return
    invalid = copy.deepcopy(config)
    if change == "rope":
        invalid["text_config"]["rope_parameters"]["mrope_section"] = [1, 1, 1]
    elif change == "vision":
        invalid["vision_config"]["deepstack_visual_indexes"] = [10]
    elif change == "custom":
        invalid["text_config"]["auto_map"] = {"AutoModel": "repository.Model"}
    elif change == "special":
        invalid["image_token_id"] = True
    else:
        invalid["text_config"]["head_dim"] = 9
    with pytest.raises(ValueError):
        qwen_retrieval.validate_config(invalid)


def test_metadata_layout_and_legacy_routes(tmp_path):
    from oryn_engine.model_layouts import assess_layout
    from oryn_engine.model_support import transformers_workload

    tiny_model(tmp_path, "feature-extraction")
    config = json.loads((tmp_path / "config.json").read_text())
    files = [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()]
    layout = assess_layout(files, config, "feature-extraction")
    assert not layout.artifact_reasons and not layout.runtime_reasons
    assert "1_Pooling/config.json" in layout.selected_files
    missing = assess_layout(
        [f for f in files if f != "1_Pooling/config.json"], config, "feature-extraction"
    )
    assert "MODEL_PROCESSOR_MISSING" in missing.artifact_reasons
    assert transformers_workload(["BertModel"], "feature-extraction") == "encoder-embedding"
    assert (
        transformers_workload(["Qwen3VLForConditionalGeneration"], "feature-extraction")
        == "feature-extraction"
    )
    assert (
        transformers_workload(["Qwen3VLForConditionalGeneration"], "text-ranking") == "text-ranking"
    )
    assert (
        transformers_workload(["Qwen3VLForConditionalGeneration"], "image-text-to-text")
        == "multimodal-chat"
    )
