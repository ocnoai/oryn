"""Reviewed Qwen3 encoder prompt rendering; repository templates remain inert."""

from __future__ import annotations


def binding_for(definitions):
    if len(definitions) != 1:
        return None
    definition = definitions[0]
    if (
        definition.name not in {"qwen3", "z_image"}
        or definition.encoder_class.__name__ != "LanguageTokenizer"
        or definition.tokenizer_class not in {"AutoTokenizer", "Qwen2TokenizerFast"}
        or not definition.use_chat_template
        or set(definition.chat_template_kwargs or {}) != {"enable_thinking"}
        or type(definition.chat_template_kwargs["enable_thinking"]) is not bool
    ):
        return None
    return (
        "qwen3-think-v1"
        if definition.chat_template_kwargs["enable_thinking"]
        else "qwen3-no-think-v1"
    )


def render(
    messages, *, binding, tokenize=False, add_generation_prompt=True, enable_thinking=True, **kwargs
):
    expected = binding == "qwen3-think-v1"
    if (
        binding not in {"qwen3-think-v1", "qwen3-no-think-v1"}
        or tokenize is not False
        or add_generation_prompt is not True
        or enable_thinking is not expected
        or kwargs
        or not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], dict)
        or set(messages[0]) != {"role", "content"}
        or messages[0]["role"] != "user"
        or not isinstance(messages[0]["content"], str)
        or len(messages[0]["content"]) > 4096
    ):
        raise ValueError("Native encoder prompt binding changed")
    return (
        "<|im_start|>user\n"
        + messages[0]["content"]
        + "<|im_end|>\n<|im_start|>assistant\n"
        + ("" if expected else "<think>\n\n</think>\n\n")
    )
