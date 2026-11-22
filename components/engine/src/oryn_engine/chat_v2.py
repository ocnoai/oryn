"""Reviewed chat-v2 execution policy and bounded answer/reasoning framing.

Repository templates are evidence for selecting an owned renderer, never executed.
Thinking metadata reports configuration, not an inference from generated prose.
"""

from __future__ import annotations

MAX_TOKENS = 8192
DEFAULT_TOKENS = 4096
MAX_OUTPUT_BYTES = 196608


def native_worker_supported(pack_id):
    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
    from oryn_engine.runtime_packs.execution import PackRuntimeLocator

    pack = builtin_registry().active_pack(pack_id)
    runtime = PackRuntimeLocator().resolve(pack)
    return runtime is not None and runtime.chat_v2


def thinking_support(model_type=None, template="", *, fixed=None):
    model_type = {"qwen3moe": "qwen3_moe", "qwen35": "qwen3_5", "qwen35moe": "qwen3_5_moe"}.get(
        model_type, model_type
    )
    if type(fixed) is bool:
        return {"support": "fixed", "default": fixed}
    # The reviewed original hybrid Qwen3 template has a real enable_thinking
    # branch. Thinking-only/instruct-only templates without it remain unknown.
    if (
        model_type in {"qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe"}
        and "enable_thinking" in template
    ):
        return {"support": "switchable", "default": True}
    return {"support": "unknown", "default": None}


def resolve_thinking(requested, support):
    if requested is not None and type(requested) is not bool:
        raise ValueError("Thinking must be a boolean when provided")
    mode = support.get("support", "unknown")
    default = support.get("default")
    if requested is None:
        applied, effective, reason = False, default, "model_default"
    elif mode == "switchable":
        applied, effective, reason = True, requested, "applied"
    elif mode == "fixed":
        applied, effective = requested == default, default
        reason = "already_fixed" if applied else "fixed_mode"
    else:
        applied, effective = False, default
        reason = "unsupported_model" if mode == "unsupported" else "unknown_support"
    return {
        "requested": requested,
        "applied": applied,
        "effective": effective,
        "reason": reason,
        "source": "engine",
    }


def reviewed_control(model_type, template, format_name, requested):
    support = thinking_support(model_type, template)
    # An unreviewed format must not acquire framing authority from repository text.
    if format_name != "chatml":
        support = {"support": "unknown", "default": None}
    metadata = resolve_thinking(requested, support)
    return metadata, metadata["applied"] and metadata["effective"] is False


def thinking_suffix(metadata):
    # The reviewed hybrid Qwen templates use an assistant <think> prefill for
    # explicit reasoning. This is owned constant framing, never repository Jinja.
    return "<think>\n" if metadata and metadata.get("effective") is True else ""


def reasoning_prefix(format_name, thinking_off=False, *, thinking_on=False):
    from oryn_engine.chat_formats import formats

    entry = formats()[format_name]
    reply = (
        entry["reply"]
        + (entry.get("thinkingOff", "") if thinking_off else "")
        + ("<think>\n" if thinking_on else "")
    )
    return reply.rfind("<think>") > reply.rfind("</think>")


class ReasoningStream:
    """Recognize only a leading reasoning block; later literal markup stays prose.

    Only a marker-sized suffix is retained between chunks. Full output storage is
    separately byte bounded. Incomplete opening/closing markers never leak from
    a recognized reasoning prefix into the final-answer channel.
    """

    opening = "<think>"
    closing = "</think>"

    def __init__(self, *, initial_reasoning=False):
        self.mode = "reasoning" if initial_reasoning else "prefix"
        self.pending = ""
        self.answer = []
        self.reasoning = []
        self.size = 0

    def _publish(self, channel, text):
        if not text:
            return []
        (self.answer if channel == "text" else self.reasoning).append(text)
        return [(channel, text)]

    def feed(self, text):
        if not isinstance(text, str):
            raise ValueError("Chat output delta must be text")
        self.size += len(text.encode("utf-8"))
        if self.size > MAX_OUTPUT_BYTES:
            raise ValueError("Chat output exceeds its bounded size")
        self.pending += text
        emitted = []
        if self.mode == "prefix":
            candidate = self.pending.lstrip()
            whitespace = len(self.pending) - len(candidate)
            if whitespace > 64:
                # Flush only excess whitespace; retain prefix detection across
                # chunks instead of treating a later reasoning opener as prose.
                emitted += self._publish("text", self.pending[: whitespace - 64])
                self.pending = self.pending[whitespace - 64 :]
            if self.opening.startswith(candidate):
                if candidate != self.opening:
                    return emitted
            elif not candidate.startswith(self.opening):
                self.mode = "answer"
            if candidate.startswith(self.opening):
                self.pending = candidate[len(self.opening) :]
                self.mode = "reasoning"
        if self.mode == "answer":
            ready, self.pending = self.pending, ""
            return emitted + self._publish("text", ready)
        if self.mode == "reasoning":
            position = self.pending.find(self.closing)
            if position >= 0:
                reasoning = self.pending[:position]
                answer = self.pending[position + len(self.closing) :]
                self.pending = ""
                self.mode = "answer"
                return (
                    emitted + self._publish("reasoning", reasoning) + self._publish("text", answer)
                )
            keep = 0
            for length in range(1, min(len(self.closing), len(self.pending) + 1)):
                if self.pending.endswith(self.closing[:length]):
                    keep = length
            ready = self.pending[:-keep] if keep else self.pending
            self.pending = self.pending[-keep:] if keep else ""
            return emitted + self._publish("reasoning", ready)
        return emitted

    def finish(self):
        if self.mode == "prefix":
            # An incomplete initial <think delimiter is not a completed answer.
            candidate = self.pending.lstrip()
            if candidate and self.opening.startswith(candidate):
                self.mode = "reasoning"
        channel = "reasoning" if self.mode == "reasoning" else "text"
        ready, self.pending = self.pending, ""
        return self._publish(channel, ready)

    @property
    def incomplete(self):
        return self.mode == "reasoning"


def result_value(raw, parser, requested, support):
    metadata = raw.get("metadata", {}).get("thinking")
    if not isinstance(metadata, dict):
        metadata = resolve_thinking(requested, support)
    finish = raw.get("finishReason", "unknown")
    if finish not in {"stop", "length", "unknown"}:
        finish = "unknown"
    if parser.incomplete and finish == "stop":
        finish = "unknown"
    usage = raw.get("usage", {})
    return {
        "text": "".join(parser.answer),
        "reasoning": "".join(parser.reasoning),
        "finishReason": finish,
        "usage": {
            key: usage.get(key)
            for key in ("inputTokens", "outputTokens", "reasoningTokens", "answerTokens")
        },
        "metadata": {"thinking": metadata},
    }
