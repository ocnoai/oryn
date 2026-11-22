"""Reviewed basic text-chat framing for the newly admitted GGUF families."""

import json
from pathlib import Path

import pytest

from oryn_engine import chat_formats


@pytest.fixture
def reviewed_formats(monkeypatch):
    table = json.loads((Path(__file__).parents[1] / "build-inputs/chat-formats.json").read_text())
    formats = chat_formats.validate_table(table)
    monkeypatch.setattr(chat_formats, "formats", lambda: formats)
    return formats


def test_spark_basic_chat_has_default_system_and_nonthinking_turn_prefixes(reviewed_formats):
    # The owned subset fixes enable_thinking=false and omits tools. An explicit
    # system instruction is a subsequent supported system block rather than
    # running the publisher's conditional Jinja to merge its initial block.
    messages = [
        {"role": "system", "content": "Use short answers."},
        {"role": "user", "content": "Name a color."},
        {"role": "assistant", "content": "Blue."},
        {"role": "user", "content": "Name another."},
    ]
    assert chat_formats.render("spark", messages) == (
        "<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant.<｜end▁of▁sentence｜>"
        "<｜start▁of▁sentence｜><|System|>\nUse short answers.<｜end▁of▁sentence｜>"
        "<｜start▁of▁sentence｜><|User|>Name a color.<｜end▁of▁sentence｜>"
        "<｜start▁of▁sentence｜><|Bot|></think>Blue.<｜end▁of▁sentence｜>"
        "<｜start▁of▁sentence｜><|User|>Name another.<｜end▁of▁sentence｜>"
        "<｜start▁of▁sentence｜><|Bot|></think>"
    )


def test_spark_chat_without_explicit_system_keeps_publisher_default(reviewed_formats):
    assert chat_formats.render("spark", [{"role": "user", "content": "Hi"}]) == (
        "<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant.<｜end▁of▁sentence｜>"
        "<｜start▁of▁sentence｜><|User|>Hi<｜end▁of▁sentence｜>"
        "<｜start▁of▁sentence｜><|Bot|></think>"
    )


def test_flint_basic_chat_matches_literal_publisher_role_framing(reviewed_formats):
    # Flint's template emits adjacent roles, with no added per-turn end token.
    # The template emits no literal BOS; adding another beginning-of-text
    # marker here would change the reviewed basic chat sequence.
    messages = [
        {"role": "system", "content": "Keep whitespace."},
        {"role": "user", "content": "  red\nblue  "},
        {"role": "assistant", "content": "Two colors."},
        {"role": "user", "content": "Continue."},
    ]
    assert chat_formats.render("flint", messages) == (
        "<|system|>Keep whitespace."
        "<|user|>  red\nblue  "
        "<|assistant|>Two colors."
        "<|user|>Continue."
        "<|assistant|>"
    )


def test_lfm3_nonthinking_prefix_is_present_in_history_and_generation(reviewed_formats):
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Name a color."},
        {"role": "assistant", "content": "Blue."},
        {"role": "user", "content": "Name another."},
    ]
    assert chat_formats.render("lfm3", messages) == (
        "<|im_start|>system\nBe brief.<|im_end|>\n"
        "<|im_start|>user\nName a color.<|im_end|>\n"
        "<|im_start|>assistant\n<think></think>Blue.<|im_end|>\n"
        "<|im_start|>user\nName another.<|im_end|>\n"
        "<|im_start|>assistant\n<think></think>"
    )


def test_lfm3_template_is_selected_only_with_its_family_scope(reviewed_formats):
    template = "<|im_start|><|im_end|>CONTINUE_FINAL_MESSAGE_TAG {{ arbitrary_repository_code() }}"
    # Its ChatML framing is shared, so unscoped selection remains fail-closed.
    # Native GGUF policy supplies this architecture's allowed renderer list.
    assert chat_formats.select(template) is None
    assert chat_formats.select(template, allowed=["lfm3"]) == "lfm3"
    assert chat_formats.select("<|im_start|><|im_end|>", allowed=["lfm3"]) is None
    assert chat_formats.select("<|im_start|><|im_end|>") == "chatml"


def test_lfm3_generated_reasoning_tags_can_be_replayed_in_assistant_history(reviewed_formats):
    # The native model can emit reasoning delimiters even with the fixed
    # non-thinking prefix. Its actual output must remain valid next-turn
    # history; these tags do not create a system/user/assistant boundary.
    messages = [
        {"role": "user", "content": "What is France's capital?"},
        {"role": "assistant", "content": "<think>Paris</think>Paris"},
        {"role": "user", "content": "What is 3 + 4?"},
    ]
    assert chat_formats.render("lfm3", messages) == (
        "<|im_start|>user\nWhat is France's capital?<|im_end|>\n"
        "<|im_start|>assistant\n<think></think><think>Paris</think>Paris<|im_end|>\n"
        "<|im_start|>user\nWhat is 3 + 4?<|im_end|>\n"
        "<|im_start|>assistant\n<think></think>"
    )


def test_flint_generated_reasoning_tags_can_be_replayed_in_assistant_history(reviewed_formats):
    # Reasoning tags are generated text, not message-role boundaries. Preserve
    # them when replaying a previous answer through the generic renderer.
    messages = [
        {"role": "user", "content": "What is France's capital?"},
        {"role": "assistant", "content": "<think>The capital is Paris.</think>Paris"},
        {"role": "user", "content": "Which country is that in?"},
    ]
    assert chat_formats.render("flint", messages) == (
        "<|user|>What is France's capital?"
        "<|assistant|><think>The capital is Paris.</think>Paris"
        "<|user|>Which country is that in?"
        "<|assistant|>"
    )


@pytest.mark.parametrize(
    ("name", "template"),
    [
        (
            "spark",
            "<｜start▁of▁sentence｜><|System|><|User|><|Bot|><｜end▁of▁sentence｜>"
            "{{ arbitrary_repository_code() }}",
        ),
        (
            "flint",
            "<|begin_of_text|><|end_of_text|><|system|><|user|><|assistant|>"
            "{{ arbitrary_repository_code() }}",
        ),
    ],
)
def test_new_formats_select_uniquely_without_executing_template(reviewed_formats, name, template):
    assert chat_formats.select(template) == name
    assert chat_formats.select(template, allowed=[name]) == name
    assert chat_formats.select(template, allowed=["chatml"]) is None


@pytest.mark.parametrize("name", ["flint", "spark", "lfm3"])
def test_all_new_family_control_markers_remain_reserved(reviewed_formats, name):
    for marker in chat_formats.reserved_markers(name):
        with pytest.raises(ValueError, match="CHAT_CONTENT_RESERVED_FRAMING"):
            chat_formats.render(name, [{"role": "user", "content": "hello " + marker + " world"}])


@pytest.mark.parametrize("name", ["flint", "spark", "lfm3"])
@pytest.mark.parametrize("role", ["tool", "developer"])
def test_basic_formats_do_not_advertise_extra_message_roles(reviewed_formats, name, role):
    with pytest.raises(ValueError, match="Chat messages are invalid"):
        chat_formats.render(name, [{"role": role, "content": "unnegotiated"}])
