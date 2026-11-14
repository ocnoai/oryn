"""Owned instruction translation over the pinned Qwen3.5 GGUF graphs."""

from oryn_engine.qwen_translation import LANGUAGES as LANGUAGES
from oryn_engine.qwen_translation import MAX_CONTEXT, prompt

ARCHITECTURES = frozenset({"qwen35", "qwen35moe"})
MAX_OUTPUT_BYTES = 16384
CONTRACT = "translation-v1"


def validate_input(value):
    # Keep the isolated native pack independent of the Transformers/Protocol
    # environments. Public invocations also pass the common task schema.
    if not isinstance(value, dict) or not {"text"} <= value.keys() <= {
        "text",
        "targetLanguage",
        "maxTokens",
    }:
        raise ValueError("GGUF_TRANSLATION_INPUT")
    text = value["text"]
    maximum = value.get("maxTokens", 128)
    language = value.get("targetLanguage", "en")
    if (
        not isinstance(text, str)
        or not 1 <= len(text) <= 32768
        or type(maximum) is not int
        or not 1 <= maximum <= 512
        or not isinstance(language, str)
        or language not in LANGUAGES
    ):
        raise ValueError("GGUF_TRANSLATION_INPUT")
    value = {"text": text, "maxTokens": maximum, "targetLanguage": language}
    prompt(value["text"], value["targetLanguage"])
    return value


def validate_binding(architecture, template, projector=None):
    if architecture not in ARCHITECTURES or template != "chatml" or projector is not None:
        raise ValueError("GGUF_TRANSLATION_BINDING_INVALID")


def render(value, context_length):
    value = validate_input(value)
    rendered = prompt(value["text"], value["targetLanguage"])
    # The admitted GPT-2 byte tokenizer cannot emit more tokens than input bytes.
    # Reserve extra space for native BOS handling; never truncate source text or
    # let the native context-shift mechanism discard an earlier document segment.
    if len(rendered.encode("utf-8")) + value["maxTokens"] + 8 > min(MAX_CONTEXT, context_length):
        raise ValueError("GGUF_TRANSLATION_CONTEXT_LIMIT")
    return rendered


def validate_output(text):
    if not isinstance(text, str) or not text.strip() or len(text.encode()) > MAX_OUTPUT_BYTES:
        raise ValueError("GGUF_TRANSLATION_OUTPUT_LIMIT")
    return {"text": text.strip()}
