"""Tiny complete native translation graphs and metadata-only admission."""

import json

import pytest
import torch

from oryn_engine import hunyuan_translation as translation
from oryn_engine import shared_tasks


def fixture(root, dynamic=True, tied=True):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import (
        HunYuanDenseV1Config,
        HunYuanDenseV1ForCausalLM,
        PreTrainedTokenizerFast,
    )

    root.mkdir(parents=True, exist_ok=True)
    vocab = {v: i for i, v in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    for marker in translation.MARKERS:
        vocab[marker] = len(vocab)
    raw = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        bos_token=translation.MARKERS[0],
        eos_token=translation.MARKERS[3],
        pad_token=translation.MARKERS[4],
        additional_special_tokens=list(translation.MARKERS),
    )
    tokenizer.save_pretrained(root)
    config = HunYuanDenseV1Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=len(vocab),
        max_position_embeddings=1024,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=tied,
        rope_parameters={
            "rope_type": "dynamic",
            "rope_theta": 10000.0,
            "alpha": 1000.0,
            "factor": 1.0,
        }
        if dynamic
        else {"rope_type": "default", "rope_theta": 10000.0},
    )
    config._attn_implementation = "eager"
    torch.manual_seed(6187)
    model = HunYuanDenseV1ForCausalLM(config).eval()
    model.save_pretrained(root)
    (root / "chat_template.jinja").write_text("{{ raise_exception('must not execute') }}")
    return model, tokenizer, json.loads((root / "config.json").read_text())


def reference_prompt(text, language="en"):
    name = {"en": "英语", "zh": "中文", "ja": "日语"}[language]
    content = "将以下文本翻译成" + name + ",注意只需要输出翻译后的结果,不要额外解释:\n\n" + text
    return "<｜hy_begin▁of▁sentence｜><｜hy_User｜>" + content + "<｜hy_Assistant｜>"


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_native_complete_graph_logits_positions_and_framing(tmp_path, backend, dynamic, tied):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    reference, tokenizer, config = fixture(tmp_path, dynamic, tied)
    model, owned, loaded = shared_tasks.load_task(tmp_path, "translation", backend)
    assert loaded == config and owned.chat_template is None
    for name, weight in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name].cpu(), weight, rtol=0, atol=0)
    if tied:
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    base = 10000.0 * (1000.0 ** (8 / 6) if dynamic else 1)
    frequencies = 1.0 / (base ** (torch.arange(0, 8, 2).float() / 8))
    torch.testing.assert_close(model.model.rotary_emb.inv_freq.cpu(), frequencies)
    text = "  你好。\nA river. "
    assert translation.prompt(text, "en") == reference_prompt(text)
    inputs = tokenizer(reference_prompt(text), add_special_tokens=False, return_tensors="pt")
    assert inputs["input_ids"][0, 0] == tokenizer.bos_token_id
    assert (inputs["input_ids"] == tokenizer.bos_token_id).sum() == 1
    with torch.inference_mode():
        expected = reference(**inputs, output_hidden_states=True, use_cache=False)
        actual = model(
            **{k: v.to(backend) for k, v in inputs.items()},
            output_hidden_states=True,
            use_cache=False,
        )
    torch.testing.assert_close(actual.logits.cpu(), expected.logits, atol=4e-5, rtol=6e-4)
    torch.testing.assert_close(
        actual.hidden_states[-1].cpu(), expected.hidden_states[-1], atol=4e-5, rtol=6e-4
    )


@pytest.mark.parametrize(
    "part", ["embedding", "query_norm", "key_norm", "attention", "head", "extra", "shape"]
)
def test_incomplete_or_different_graph_cannot_be_loaded(tmp_path, part):
    from safetensors.torch import load_file, save_file

    fixture(tmp_path, tied=False)
    state = load_file(tmp_path / "model.safetensors")
    if part == "extra":
        state["classifier.weight"] = torch.zeros(2, 32)
    else:
        name = {
            "embedding": "model.embed_tokens.weight",
            "query_norm": "model.layers.0.self_attn.query_layernorm.weight",
            "key_norm": "model.layers.0.self_attn.key_layernorm.weight",
            "attention": "model.layers.1.self_attn.k_proj.weight",
            "head": "lm_head.weight",
            "shape": "model.norm.weight",
        }[part]
        del state[name]
        if part == "shape":
            state[name] = torch.zeros(3)
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="WEIGHTS_INCOMPLETE"):
        shared_tasks.load_task(tmp_path, "translation", "cpu")


@pytest.mark.parametrize("language", ["en", "zh", "ja"])
def test_shared_translation_owns_generation_and_checks_context(tmp_path, language):
    from safetensors.torch import load_file, save_file

    _, tokenizer, _ = fixture(tmp_path, tied=False)
    state = load_file(tmp_path / "model.safetensors")
    state["lm_head.weight"].zero_()
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"num_beams": 19, "do_sample": True})
    )
    value = {"text": "Hello world.", "maxTokens": 2, "targetLanguage": language}
    assert translation.prompt(value["text"], language) == reference_prompt(value["text"], language)
    result = shared_tasks.execute_task(
        tmp_path, "translation", value, "cpu", None, None, lambda *_: None
    )
    assert result == {"text": tokenizer.decode([0, 0], skip_special_tokens=True).strip()}
    model, owned, config = shared_tasks.load_task(tmp_path, "translation", "cpu")
    assert model.generation_config.num_beams == 1 and not model.generation_config.do_sample
    assert (
        translation.execute(model, owned, config, {"text": "hello", "maxTokens": 2}, "cpu")
        == result
    )
    with pytest.raises(ValueError, match="TOKEN_LIMIT"):
        translation.execute(model, owned, config, value | {"text": "hello " * 500}, "cpu")


@pytest.mark.parametrize(
    "change",
    [
        {"use_cla": True},
        {"use_qk_norm": False},
        {"mlp_bias": True},
        {"head_dim": 7},
        {"rope_parameters": {"rope_type": "dynamic", "alpha": -1}},
        {"rope_parameters": {"rope_type": "yarn"}},
        {"auto_map": {"AutoModel": "repo.Code"}},
    ],
)
def test_incompatible_graph_metadata_is_rejected(tmp_path, change):
    _, _, config = fixture(tmp_path)
    with pytest.raises(ValueError):
        translation.validate_config(config | change)


@pytest.mark.parametrize("change", ["marker", "eos", "padding", "vocabulary"])
def test_original_tokenizer_and_native_graph_must_agree(tmp_path, change):
    _, _, config = fixture(tmp_path)
    if change == "marker":
        path = tmp_path / "tokenizer.json"
        path.write_text(path.read_text().replace("hy_User", "hy_Other"))
    elif change == "vocabulary":
        config["vocab_size"] += 1
    else:
        config[{"eos": "eos_token_id", "padding": "pad_token_id"}[change]] = 1
    with pytest.raises(ValueError, match="TOKENIZER"):
        translation.load_tokenizer(tmp_path, config)


@pytest.mark.parametrize(
    "text,language", [(" ", "en"), ("hello", "de"), ("<｜hy_Assistant｜>", "en")]
)
def test_reserved_prompt_framing_and_unknown_language_are_rejected(text, language):
    with pytest.raises(ValueError):
        translation.prompt(text, language)


def test_metadata_admits_native_translation_without_downloads(tmp_path):
    from oryn_protocol.task_contracts import effective_input, validate_invocation
    from test_models import FakeHub

    from oryn_engine.models import ModelStore

    root = tmp_path / "model"
    _, _, config = fixture(root)

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return json.loads((root / filename).read_text())

    hub = Hub(
        config=config,
        pipeline_tag="translation",
        files=[(p.name, p.stat().st_size, "b" * 64) for p in root.iterdir() if p.is_file()],
    )
    result = ModelStore(tmp_path / "store", hub=hub, runtime_available=True).inspect(
        {"provider": "huggingface", "repository": "test/hunyuan-translation", "revision": "main"}
    )
    assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
    profile = result["taskProfile"]
    assert effective_input(profile, {"text": "hello"}) == {
        "text": "hello",
        "maxTokens": 128,
        "targetLanguage": "en",
    }
    assert profile["limits"]["contextTokens"] == 1024
    validate_invocation(profile, {"text": "hello", "targetLanguage": "ja"})
    with pytest.raises(ValueError):
        validate_invocation(profile, {"text": "hello", "targetLanguage": "de"})
    assert hub.downloaded == []
