import threading
from types import SimpleNamespace

import pytest
from oryn_protocol.task_contracts import make_profile, validate_invocation, validate_result

from oryn_engine.chat_v2 import (
    MAX_OUTPUT_BYTES,
    ReasoningStream,
    resolve_thinking,
    result_value,
    reviewed_control,
)
from oryn_engine.jobs import JobExecutionError
from oryn_engine.runners import RunnerRegistry, _chat_input
from oryn_engine.task_profiles import profile_for_execution


@pytest.mark.parametrize("split", range(1, 40))
def test_reasoning_boundaries_never_enter_answer(split):
    raw = " <think>private reasoning</think>Final answer."
    parser = ReasoningStream()
    events = []
    for position in range(0, len(raw), split):
        events.extend(parser.feed(raw[position : position + split]))
        assert len(parser.pending) <= len(parser.closing)
    events.extend(parser.finish())
    assert "".join(text for channel, text in events if channel == "text") == "Final answer."
    assert "".join(parser.reasoning) == "private reasoning"
    assert not parser.incomplete


def test_known_open_prompt_prefix_and_incomplete_reasoning():
    parser = ReasoningStream(initial_reasoning=True)
    parser.feed("reasoning already opened by reviewed prompt</thi")
    parser.feed("nk>Answer")
    parser.finish()
    assert "".join(parser.answer) == "Answer"
    parser = ReasoningStream()
    parser.feed("<think>not finished</thi")
    parser.finish()
    value = result_value({"finishReason": "stop"}, parser, False, {"support": "unknown"})
    assert value["text"] == ""
    assert value["finishReason"] == "unknown"
    assert value["reasoning"] == "not finished</thi"


def test_literal_markers_in_normal_article_are_preserved_and_output_is_bounded():
    text = "An article can discuss <think> as a literal XML-style tag."
    parser = ReasoningStream()
    for char in text:
        parser.feed(char)
    parser.finish()
    assert "".join(parser.answer) == text
    assert parser.reasoning == []
    with pytest.raises(ValueError, match="bounded"):
        ReasoningStream().feed("x" * (MAX_OUTPUT_BYTES + 1))


@pytest.mark.parametrize(
    "chunks",
    [
        [" " * 65 + "<think>secret</think>Answer"],
        [" " * 65, "<think>secret</think>Answer"],
        [" " * 10000, "<thi", "nk>secret"],
    ],
)
def test_long_leading_whitespace_cannot_turn_reasoning_into_answer(chunks):
    parser = ReasoningStream()
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
        assert len(parser.pending) <= 64 + len(parser.opening)
    events.extend(parser.finish())
    answer = "".join(text for channel, text in events if channel == "text")
    assert "secret" not in answer and "<think>" not in answer
    assert "secret" in "".join(parser.reasoning)


@pytest.mark.parametrize("requested", [None, True, False])
@pytest.mark.parametrize(
    "support,default",
    [
        ("switchable", True),
        ("fixed", True),
        ("fixed", False),
        ("unknown", None),
        ("unsupported", None),
    ],
)
def test_best_effort_metadata_never_claims_unsupported_request_applied(requested, support, default):
    result = resolve_thinking(requested, {"support": support, "default": default})
    assert result["requested"] is requested
    if result["applied"]:
        assert requested is not None and result["effective"] is requested
        assert result["reason"] in {"applied", "already_fixed"}
    if requested is None:
        assert result["reason"] == "model_default" and result["applied"] is False


def test_reviewed_qwen_control_happens_before_generation_not_as_output_hiding():
    template = "<|im_start|>{{ enable_thinking }}<|im_end|>"
    metadata, framing = reviewed_control("qwen3", template, "chatml", False)
    assert framing and metadata["effective"] is False and metadata["applied"]
    metadata, framing = reviewed_control("qwen3", template, "chatml", True)
    assert not framing and metadata["effective"] is True and metadata["applied"]
    metadata, framing = reviewed_control("qwen2", template, "chatml", False)
    assert not framing and metadata["reason"] == "unknown_support"


def test_legacy_input_is_unchanged_and_v2_bool_budget_are_validated():
    messages = [{"role": "user", "content": "Hello"}]
    with pytest.raises(JobExecutionError):
        _chat_input({"messages": messages, "thinking": False})
    with pytest.raises(JobExecutionError):
        _chat_input({"messages": messages, "maxTokens": 513})
    assert _chat_input({"messages": messages, "thinking": False}, chat_v2=True)["maxTokens"] == 4096
    with pytest.raises(JobExecutionError):
        _chat_input({"messages": messages, "thinking": None}, chat_v2=True)
    with pytest.raises(JobExecutionError):
        _chat_input({"messages": messages, "maxTokens": 8193}, chat_v2=True)


def test_explicit_v2_profile_accepts_both_modes_and_keeps_legacy_default(monkeypatch):
    monkeypatch.setattr("oryn_engine.chat_v2.native_worker_supported", lambda _: True)
    native = {
        "runtimePack": {"packId": "oryn.llama-cpp", "taskContract": "chat-v1"},
        "technicalCapabilities": {"task": "text-generation", "inputModalities": ["text"]},
    }
    legacy = profile_for_execution(native)
    modern = profile_for_execution(native, contract_id="chat-v2")
    assert legacy["contract"]["id"] == "chat-v1"
    assert modern["parameters"]["maxTokens"]["default"] == 4096
    assert "thinking" not in modern["parameters"]
    for flag in [False, True]:
        validate_invocation(
            modern, {"messages": [{"role": "user", "content": "Hello"}], "thinking": flag}
        )
    assert profile_for_execution({**native, "taskProfile": modern})["contract"]["id"] == "chat-v1"
    large = {
        **native,
        "technicalCapabilities": {**native["technicalCapabilities"], "contextLength": 32768},
    }
    assert profile_for_execution(large, contract_id="chat-v2")["limits"]["contextTokens"] == 8192


def test_old_native_worker_stays_usable_without_claiming_new_controls(monkeypatch):
    monkeypatch.setattr("oryn_engine.chat_v2.native_worker_supported", lambda _: False)
    profile = profile_for_execution(
        {
            "runtimePack": {"packId": "oryn.llama-cpp", "taskContract": "chat-v1"},
            "technicalCapabilities": {"task": "text-generation", "inputModalities": ["text"]},
            "_chatThinkingSupport": {"support": "switchable", "default": True},
        },
        contract_id="chat-v2",
    )
    assert profile["parameters"]["maxTokens"]["default"] == 512
    assert profile["thinking"] == {"support": "unknown", "default": None}
    validate_invocation(
        profile, {"messages": [{"role": "user", "content": "Hello"}], "thinking": False}
    )


def test_new_discovery_field_requires_explicit_peer_opt_in():
    from oryn_engine.runtime.capabilities import runtime_capabilities

    assert "taskContracts" not in runtime_capabilities("engine_test", 1048576)
    modern = runtime_capabilities("engine_test", 1048576, chat_v2=True)
    assert {"id": "chat-v2", "revision": 1} in modern["taskContracts"]


def test_runner_reconciles_missing_suffix_and_rejects_disagreeing_final(monkeypatch):
    runner = object.__new__(RunnerRegistry)
    job = SimpleNamespace(
        task_profile=make_profile("chat-v2"), input={"thinking": False}, chat_reasoning_prefix=False
    )
    emitted = []

    def raw(job, handle, cancellation, emit):
        emit("job.output.delta", {"channel": "text", "delta": "<think>reason</think>Ans"})
        return {
            "kind": "structured",
            "value": {"text": "<think>reason</think>Answer", "finishReason": "stop"},
        }

    monkeypatch.setattr(runner, "_execute_raw", raw)
    result = runner._execute_single(
        job, object(), threading.Event(), lambda name, data: emitted.append(data)
    )
    assert result["value"]["text"] == "Answer"
    assert "".join(event["delta"] for event in emitted if event["channel"] == "text") == "Answer"
    validate_result(job.task_profile, result, job.input)

    def disagreement(job, handle, cancellation, emit):
        emit("job.output.delta", {"channel": "text", "delta": "prefix"})
        return {"kind": "structured", "value": {"text": "different"}}

    monkeypatch.setattr(runner, "_execute_raw", disagreement)
    with pytest.raises(JobExecutionError, match="disagree"):
        runner._execute_single(job, object(), threading.Event(), lambda *_: None)


def test_v2_inspection_promotes_only_chat_and_does_not_publish_internal_hint(tmp_path):
    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore

    store = ModelStore(
        tmp_path,
        hub=FakeHub(
            config={"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"},
            pipeline_tag="text-generation",
        ),
        runtime_available=True,
    )
    result = store.inspect(model(), {"contract": {"id": "chat-v2", "revision": 1}})
    assert result["executable"], result
    assert result["taskProfile"]["contract"]["id"] == "chat-v2"
    assert "_chatThinkingSupport" not in result
    assert (
        profile_for_execution(
            {"taskProfile": make_profile("embeddings-v1")}, contract_id="chat-v2"
        )["contract"]["id"]
        == "embeddings-v1"
    )
