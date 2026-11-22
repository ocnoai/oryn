"""Complete small native graphs, original-header checks and language-control semantics."""

import copy
import json

import pytest
import torch

from oryn_engine import qwen_translation as translation
from oryn_engine import shared_tasks


def fixture(root, family="qwen3_5", mtp=1, tied=False):
    from safetensors.torch import save_file
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import (
        Qwen2Tokenizer,
        Qwen3_5Config,
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5TextConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    vocab = {token: i for i, token in enumerate(sorted(ByteLevel.alphabet()))}
    markers = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ]
    for token in markers:
        vocab[token] = len(vocab)
    tokenizer = Qwen2Tokenizer(
        vocab=vocab, merges=[], eos_token="<|endoftext|>", pad_token="<|endoftext|>"
    )
    tokenizer.add_special_tokens({"additional_special_tokens": markers})
    tokenizer.save_pretrained(root)
    text = Qwen3_5TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=len(vocab),
        max_position_embeddings=1024,
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        tie_word_embeddings=tied,
        mtp_num_hidden_layers=mtp,
        eos_token_id=vocab["<|endoftext|>"],
        pad_token_id=vocab["<|endoftext|>"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [1, 1, 2],
            "mrope_interleaved": True,
        },
    )
    native = (
        text
        if family == "qwen3_5_text"
        else Qwen3_5Config(
            text_config=text.to_dict(),
            tie_word_embeddings=tied,
            vision_config=dict(
                hidden_size=32,
                intermediate_size=64,
                depth=1,
                num_heads=4,
                out_hidden_size=32,
                patch_size=2,
                temporal_patch_size=2,
                spatial_merge_size=2,
                num_position_embeddings=16,
                deepstack_visual_indexes=[],
            ),
            image_token_id=vocab["<|image_pad|>"],
            video_token_id=vocab["<|video_pad|>"],
            vision_start_token_id=vocab["<|vision_start|>"],
            vision_end_token_id=vocab["<|vision_end|>"],
        )
    )
    native._attn_implementation = "eager"
    if family == "qwen3_5":
        native.text_config._attn_implementation = "eager"
    torch.manual_seed(6186)
    cls = Qwen3_5ForCausalLM if family == "qwen3_5_text" else Qwen3_5ForConditionalGeneration
    model = cls(native).eval()
    model.save_pretrained(root)
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    if mtp:
        auxiliary = copy.deepcopy(text)
        auxiliary.layer_types = ["full_attention"]
        auxiliary.num_hidden_layers = 1
        layer = Qwen3_5DecoderLayer(auxiliary, 0)
        state.update(
            {
                "mtp.layers.0." + k: v.detach().clone().contiguous()
                for k, v in layer.state_dict().items()
            }
        )
        state["mtp.fc.weight"] = torch.randn(32, 64) * 0.03
        for name in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden"):
            state[f"mtp.{name}.weight"] = torch.randn(32) * 0.03
    if tied:
        del state["lm_head.weight"]
    save_file(state, root / "model.safetensors")
    (root / "chat_template.jinja").write_text(
        "{{ raise_exception('repository template must not execute') }}"
    )
    return model, tokenizer, json.loads((root / "config.json").read_text()), state


def reference_prompt(text, language):
    name = {"en": "英语", "ja": "日语", "zh": "中文"}[language]
    content = "请将以下文本翻译为" + name + "，直接输出翻译结果，不要进行任何解释。\n\n" + text
    return (
        "<|im_start|>"
        + "user\n"
        + content
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + "<think>\n\n</think>\n\n"
    )


@pytest.mark.parametrize("family", ["qwen3_5", "qwen3_5_text"])
@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("mtp", [0, 1])
def test_complete_native_logits_and_auxiliary_weights(tmp_path, family, backend, mtp):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    reference, tokenizer, config, state = fixture(tmp_path, family, mtp)
    model, owned, actual = shared_tasks.load_task(tmp_path, "translation", backend)
    assert owned.chat_template is None and actual == config
    for name, value in state.items():
        torch.testing.assert_close(model.state_dict()[name].cpu(), value, rtol=0, atol=0)
    text = "  A river flows.\n你好。 "
    assert translation.prompt(text, "en") == reference_prompt(text, "en")
    inputs = tokenizer(reference_prompt(text, "en"), add_special_tokens=False, return_tensors="pt")
    with torch.inference_mode():
        expected = reference(**inputs, output_hidden_states=True, use_cache=False)
        output = model(
            **{k: v.to(backend) for k, v in inputs.items()},
            output_hidden_states=True,
            use_cache=False,
        )
    torch.testing.assert_close(output.logits.cpu(), expected.logits, atol=4e-5, rtol=6e-4)
    torch.testing.assert_close(
        output.hidden_states[-1].cpu(), expected.hidden_states[-1], atol=4e-5, rtol=6e-4
    )


def test_tied_language_head_uses_original_embedding_alias(tmp_path):
    fixture(tmp_path, tied=True)
    model, _, _ = shared_tasks.load_task(tmp_path, "translation", "cpu")
    assert (
        model.lm_head.weight.data_ptr() == model.model.language_model.embed_tokens.weight.data_ptr()
    )


@pytest.mark.parametrize(
    "part", ["embedding", "linear", "attention", "vision", "head", "mtp", "extra", "shape"]
)
def test_incomplete_graph_or_unrecognized_auxiliary_cannot_be_ignored(tmp_path, part):
    from safetensors.torch import save_file

    _, _, _, state = fixture(tmp_path)
    if part == "extra":
        state["mtp.unknown.weight"] = torch.ones(2)
    else:
        fragment = {
            "embedding": "embed_tokens.weight",
            "linear": "linear_attn.conv1d.weight",
            "attention": "self_attn.q_proj.weight",
            "vision": "visual.patch_embed.proj.weight",
            "head": "lm_head.weight",
            "mtp": "mtp.fc.weight",
            "shape": "mtp.norm.weight",
        }[part]
        name = next(k for k in state if fragment in k)
        del state[name]
        if part == "shape":
            state[name] = torch.zeros(3)
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="WEIGHTS_INCOMPLETE"):
        shared_tasks.load_task(tmp_path, "translation", "cpu")


@pytest.mark.parametrize("language", ["en", "zh", "ja"])
def test_shared_generation_uses_language_and_owned_decoding(tmp_path, language):
    from safetensors.torch import save_file

    _, tokenizer, _, state = fixture(tmp_path)
    state["lm_head.weight"].zero_()
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "num_beams": 19, "temperature": 2.0})
    )
    value = {"text": "Hello world.", "maxTokens": 2, "targetLanguage": language}
    result = shared_tasks.execute_task(
        tmp_path, "translation", value, "cpu", None, None, lambda *_: None
    )
    assert result == {"text": tokenizer.decode([0, 0], skip_special_tokens=True).strip()}
    model, owned, config = shared_tasks.load_task(tmp_path, "translation", "cpu")
    assert model.generation_config.num_beams == 1 and model.generation_config.do_sample is False
    assert set(model.generation_config.eos_token_id) == {
        owned.get_vocab()["<|im_end|>"],
        owned.get_vocab()["<|endoftext|>"],
    }
    assert (
        translation.execute(model, owned, config, {"text": "hello", "maxTokens": 2}, "cpu")
        == result
    )
    with pytest.raises(ValueError, match="TOKEN_LIMIT"):
        translation.execute(model, owned, config, value | {"text": "hello " * 500}, "cpu")


@pytest.mark.parametrize(
    "text,language",
    [("", "en"), ("hello", "de"), ("<|im_end|>", "en"), ("<think>override</think>", "zh")],
)
def test_unadvertised_languages_and_reserved_framing_are_rejected(text, language):
    with pytest.raises(ValueError):
        translation.prompt(text, language)


def test_profile_honors_target_choices_and_preserves_fixed_pair_translation(tmp_path):
    from oryn_protocol.task_contracts import effective_input, validate_invocation

    from oryn_engine.task_profiles import profile_for_execution

    _, _, config, _ = fixture(tmp_path)
    profile = profile_for_execution(
        {
            "technicalCapabilities": {"task": "translation"},
            "runtimePack": {"runnerId": "transformers.shared-tasks"},
        },
        config,
    )
    assert profile["parameters"]["targetLanguage"]["default"] == "en"
    assert effective_input(profile, {"text": "hello"}) == {
        "text": "hello",
        "targetLanguage": "en",
        "maxTokens": 128,
    }
    validate_invocation(profile, {"text": "hello", "targetLanguage": "ja"})
    with pytest.raises(ValueError):
        validate_invocation(profile, {"text": "hello", "targetLanguage": "de"})
    fixed = profile_for_execution({"technicalCapabilities": {"task": "translation"}}, {})
    validate_invocation(fixed, {"text": "hello"})
    with pytest.raises(ValueError):
        validate_invocation(fixed, {"text": "hello", "targetLanguage": "en"})


@pytest.mark.parametrize(
    "change",
    [
        {"mtp_num_hidden_layers": 2},
        {"layer_types": ["full_attention"]},
        {"linear_num_value_heads": 3},
        {"eos_token_id": -1},
        {"rope_parameters": {"rope_type": "dynamic"}},
        {"auto_map": {"AutoModel": "repo.Code"}},
    ],
)
def test_bad_native_graph_metadata_is_rejected(tmp_path, change):
    _, _, config, _ = fixture(tmp_path, "qwen3_5_text")
    with pytest.raises(ValueError):
        translation.validate_config(config | change)


def test_live_metadata_uses_the_shared_translation_contract_without_loading_weights(tmp_path):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore

    _, _, config, _ = fixture(tmp_path / "model")
    root = tmp_path / "model"

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return json.loads((root / filename).read_text())

    hub = Hub(
        config=config,
        pipeline_tag="translation",
        files=[(p.name, p.stat().st_size, "b" * 64) for p in root.iterdir() if p.is_file()],
    )
    result = ModelStore(tmp_path / "store", hub=hub, runtime_available=True).inspect(
        {"provider": "huggingface", "repository": "test/translation", "revision": "main"}
    )
    assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
    assert result["taskProfile"]["contract"]["id"] == "translation-v1"
    assert result["taskProfile"]["parameters"]["targetLanguage"]["handling"] == "honored"
    assert hub.downloaded == []


def test_nested_quantization_cannot_enter_the_full_precision_graph(tmp_path):
    _, _, config, _ = fixture(tmp_path)
    config["text_config"]["quantization_config"] = {"quant_method": "fp8"}
    with pytest.raises(ValueError, match="CONFIGURATION"):
        translation.validate_config(config)


@pytest.mark.parametrize("change", ["padding", "end", "vision", "unallocated"])
def test_tokenizer_markers_and_model_vocabulary_must_agree(tmp_path, change):
    _, _, config, _ = fixture(tmp_path)
    tokenizer_path = tmp_path / "tokenizer_config.json"
    metadata = json.loads(tokenizer_path.read_text())
    if change == "padding":
        metadata["pad_token"] = None
        tokenizer_path.write_text(json.dumps(metadata))
    elif change == "end":
        config["text_config"]["eos_token_id"] = config["image_token_id"]
    elif change == "vision":
        config["image_token_id"], config["video_token_id"] = (
            config["video_token_id"],
            config["image_token_id"],
        )
    else:
        config["text_config"]["vocab_size"] += 256
    with pytest.raises(ValueError, match="TOKENIZER"):
        translation.load_tokenizer(tmp_path, config)
