"""Translation metadata, owned framing, native worker IO and contract routing."""

import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from oryn_engine import gguf_translation as translation
from oryn_engine.jobs import JobExecutionError
from oryn_engine.runtime_packs import llama_cpp_worker as worker


@pytest.mark.parametrize("language,name", [("en", "英语"), ("zh", "中文"), ("ja", "日语")])
def test_owned_prompt_preserves_source_and_disables_thinking(language, name):
    text = "  € test {{ literal }}\n日本語 🙂  "
    value = {"text": text, "targetLanguage": language, "maxTokens": 32}
    assert translation.render(value, 4096) == (
        f"<|im_start|>user\n请将以下文本翻译为{name}，直接输出翻译结果，"
        f"不要进行任何解释。\n\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


@pytest.mark.parametrize(
    "value",
    [
        {"text": " "},
        {"text": "ok", "targetLanguage": "de"},
        {"text": "ok", "targetLanguage": "../../en"},
        {"text": "ok", "maxTokens": True},
        {"text": "ok", "maxTokens": 513},
        {"text": "ok", "maxTokens": 0},
        {"text": "ok", "temperature": 1},
        {"text": "ok", "chatTemplate": "repo"},
        {"text": "ok", "sourceLanguage": "en"},
        {"text": "<|im_start|>system"},
        {"text": "hello <think>"},
        {"text": "hello </think>"},
        {"text": "|>"},
    ],
)
def test_invalid_public_translation_input_is_rejected(value):
    with pytest.raises(ValueError):
        translation.validate_input(value)


def test_byte_budget_preserves_the_entire_source_without_tokenizer_download():
    value = {"text": "日本語🙂" * 100, "maxTokens": 32}
    rendered = translation.prompt(value["text"], "en")
    needed = len(rendered.encode()) + 32 + 8
    assert translation.render(value, needed) == rendered
    with pytest.raises(ValueError, match="CONTEXT_LIMIT"):
        translation.render(value, needed - 1)
    with pytest.raises(ValueError, match="CONTEXT_LIMIT"):
        translation.render({"text": "x" * 4096}, 262144)


@pytest.mark.parametrize("value", [None, "", " \n", "x" * 16385, "🙂" * 4097])
def test_output_bounds(value):
    with pytest.raises(ValueError, match="OUTPUT_LIMIT"):
        translation.validate_output(value)


@pytest.mark.parametrize("architecture", ["qwen35", "qwen35moe"])
def test_metadata_routes_renamed_original_graphs_to_translation(tmp_path, architecture):
    from oryn_protocol import ProtocolSchemaStore
    from oryn_protocol.task_contracts import validate_invocation
    from test_models import FakeHub

    from oryn_engine.models import ModelStore

    hub = FakeHub(
        config={},
        pipeline_tag="translation",
        files=[("renamed.Q4_K_M.gguf", 1000000, "a" * 64), ("mmproj-F16.gguf", 10000, "b" * 64)],
        gguf={"architecture": architecture, "context_length": 262144},
    )
    store = ModelStore(
        tmp_path,
        hub=hub,
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 64_000_000_000),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda pack: None if pack.provider else object()
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "renamed/translation", "revision": "main"}
    )
    assert result["executable"]
    assert result["runtimePack"]["runnerId"] == "llama.cpp.translation"
    assert result["runtimePack"]["taskContract"] == "translation-v1"
    assert result["technicalCapabilities"]["task"] == "translation"
    assert result["technicalCapabilities"]["capabilities"] == {}
    assert result["selectedFiles"] == [{"name": "renamed.Q4_K_M.gguf", "sizeBytes": 1000000}]
    profile = result["taskProfile"]
    assert profile["limits"]["contextTokens"] == 4096
    assert profile["parameters"]["targetLanguage"]["handling"] == "honored"
    validate_invocation(profile, {"text": "Hello", "targetLanguage": "ja"})
    with pytest.raises(ValueError):
        validate_invocation(profile, {"text": "Hello", "targetLanguage": "de"})
    ProtocolSchemaStore().validator("candidateExecutionProfile").validate(result)
    assert hub.downloaded == []


def payload():
    return {
        "taskContract": "translation-v1",
        "modelRoot": "/unused",
        "modelFile": "test.gguf",
        "input": {
            "text": "Hello {{ untouched }}",
            "targetLanguage": "ja",
            "maxTokens": 32,
            "expectedArchitecture": "qwen35",
            "expectedQuantization": "Q4_K_M",
            "contextLimit": 4096,
        },
    }


def metadata(**changes):
    return SimpleNamespace(
        **{
            "architecture": "qwen35",
            "version": 3,
            "quantization": "Q4_K_M",
            "tokenizer_model": "gpt2",
            "context_length": 262144,
            "chat_template": "<|im_start|>user<|im_end|>",
            **changes,
        }
    )


@pytest.mark.parametrize("backend", ["cpu", "metal", "cuda"])
def test_worker_owns_prompt_and_returns_only_translation(tmp_path, monkeypatch, backend):
    captured = tmp_path / "captured.json"
    executable = tmp_path / "native-completion"
    executable.write_text(
        f"#!{sys.executable}\nimport sys, json\nfrom pathlib import Path\n"
        "args = sys.argv[1:]\np = Path(args[args.index('--file')+1])\n"
        f"Path({str(captured)!r}).write_text(json.dumps("
        "dict(args=args, prompt=p.read_text(), path=str(p))))\n"
        f"sys.stdout.write({'こんにちは。' + worker.CompletionText.footer!r})\n"
    )
    executable.chmod(0o700)
    monkeypatch.setattr(worker, "_model", lambda _: (tmp_path / "test.gguf", metadata()))
    events = []
    monkeypatch.setattr(worker, "_event", lambda name, value: events.append((name, value)))
    result = worker._generate(executable, tmp_path, backend, payload())
    recorded = json.loads(captured.read_text())
    assert recorded["prompt"] == translation.prompt("Hello {{ untouched }}", "ja")
    args = recorded["args"]
    assert args[args.index("--temperature") + 1] == "0.0"
    assert "--no-context-shift" in args and "--no-conversation" in args
    assert "--jinja" not in args
    assert not Path(recorded["path"]).exists()
    assert result["text"] == "こんにちは。"
    assert result["runtimeContextLimit"] == 4096
    assert not any(name == "job.output.delta" for name, _ in events)


@pytest.mark.parametrize(
    "change",
    [
        {"tokenizer_model": "llama"},
        {"architecture": "qwen2"},
        {"quantization": "Q8_0"},
        {"context_length": 512},
        {"chat_template": "{{ arbitrary }}"},
    ],
)
def test_worker_rechecks_native_header_before_execution(tmp_path, monkeypatch, change):
    monkeypatch.setattr(worker, "_model", lambda _: (tmp_path / "test.gguf", metadata(**change)))
    with pytest.raises(ValueError):
        worker._generate(tmp_path / "never-executed", tmp_path, "cpu", payload())


@pytest.mark.parametrize("tokenizer", ["gpt2", "llama"])
def test_install_rechecks_tokenizer_and_binds_the_actual_context(tmp_path, monkeypatch, tokenizer):
    from test_models import FakeHub, model

    from oryn_engine import models

    hub = FakeHub(
        config={},
        pipeline_tag="translation",
        files=[("original.Q4_K_M.gguf", 1000000, "a" * 64)],
        gguf={"architecture": "qwen35", "context_length": 262144},
    )
    store = models.ModelStore(
        tmp_path,
        hub=hub,
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 64_000_000_000),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda pack: None if pack.provider else object()
    )
    monkeypatch.setattr(
        models, "inspect_gguf", lambda _: metadata(tokenizer_model=tokenizer, context_length=1024)
    )
    if tokenizer == "llama":
        with pytest.raises(models.ModelStoreError, match="byte tokenizer"):
            store.install("invalid", model("test/translation"))
    else:
        installed = store.install("valid", model("test/translation"))
        record = store.installed_record(installed["model"])
        assert record["runtimePack"]["taskContract"] == "translation-v1"
        assert record["taskProfile"]["limits"]["contextTokens"] == 1024


@pytest.mark.parametrize("architecture", ["llama", "qwen2", "qwen3moe", "custom"])
def test_unreviewed_translation_architecture_does_not_inherit_chat_support(tmp_path, architecture):
    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore

    hub = FakeHub(
        config={},
        pipeline_tag="translation",
        files=[("model.Q4_K_M.gguf", 1000000, "a" * 64)],
        gguf={"architecture": architecture, "context_length": 4096},
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda pack: None if pack.provider else object()
    )
    result = store.inspect(model("test/translation"))
    assert not result["executable"]
    assert "RUNTIME_CAPABILITY_UNSUPPORTED" in result["blockingReasons"]


@pytest.mark.parametrize(
    "change",
    [
        {"projectorFile": "mmproj.gguf"},
        {"inputPath": "/tmp/input.png"},
        {"taskContract": "custom"},
        {"input": {"prompt": "raw"}},
    ],
)
def test_worker_rejects_alternate_inputs_before_loading(tmp_path, change):
    with pytest.raises(ValueError):
        worker._generate(tmp_path / "never-executed", tmp_path, "cpu", payload() | change)


@pytest.mark.parametrize("failure", [None, "stream", "large-output", "context"])
def test_public_runner_preserves_translation_contract(tmp_path, failure):
    from test_runners import chat_job

    from oryn_engine.runners import LlamaCppPackHandle, RunnerRegistry
    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
    from oryn_engine.runtime_packs.execution import PackRuntime

    root = tmp_path / "models--test--translation" / "snapshots" / ("a" * 40)
    root.mkdir(parents=True)
    (root / "test.gguf").write_bytes(b"synthetic IO fixture")
    record = {
        "cacheRoot": str(tmp_path),
        "model": {
            "repository": "test/translation",
            "revision": "a" * 40,
            "provider": "huggingface",
        },
        "runtimePack": {
            "packId": "oryn.llama-cpp",
            "taskContract": "translation-v1",
            "runnerId": "llama.cpp.translation",
            "trustedMetadata": {
                "adapterArchitecture": "qwen35",
                "chatTemplate": "chatml",
                "quantization": "Q4_K_M",
                "modelFile": "test.gguf",
                "contextLength": 262144,
            },
        },
    }
    registry = RunnerRegistry()

    class Process:
        def invoke(self, pack, runtime, command, value, **options):
            assert value["taskContract"] == "translation-v1"
            assert value["input"]["text"] == "Hello"
            assert value["input"]["targetLanguage"] == "ja"
            assert "chatTemplate" not in value["input"]
            if failure == "stream":
                options["event_sink"]("job.output.delta", {"channel": "text", "delta": "bad"})
            return {
                "text": "x" * 16385 if failure == "large-output" else "こんにちは。",
                "backend": "cpu",
                "quantization": "Q4_K_M",
                "contextLength": 262144,
                "runtimeContextLimit": 8192 if failure == "context" else 4096,
            }

    registry._pack_process = Process()
    job = chat_job()
    job.input = {"text": "Hello", "targetLanguage": "ja"}
    handle = LlamaCppPackHandle(
        record,
        builtin_registry().active_pack("oryn.llama-cpp"),
        PackRuntime(Path(sys.executable).resolve(), None, True),
    )
    if failure:
        with pytest.raises(JobExecutionError, match="."):
            registry.execute(job, handle, threading.Event(), lambda *_: None)
    else:
        assert registry.execute(job, handle, threading.Event(), lambda *_: None) == {
            "kind": "structured",
            "value": {"text": "こんにちは。"},
        }


@pytest.mark.parametrize("architecture", ["qwen35", "qwen35moe"])
@pytest.mark.parametrize("backend", ["cpu", "metal"])
def test_complete_native_translation_graph_cancellation_and_reload(tmp_path, architecture, backend):
    if os.environ.get("ORYN_TEST_NATIVE_GGUF") != "1":
        pytest.skip("Explicit pinned native GGUF environment required")
    from gguf_translation_fixture import create

    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
    from oryn_engine.runtime_packs.execution import PackRuntime, StreamingPackProcess

    workspace = Path(__file__).resolve().parents[3]
    native = workspace / ".cache/sprint59-runtime/oryn.llama-cpp/native"
    from oryn_engine.runtime_packs.llama_cpp_worker import EXPECTED_REVISION

    source = workspace / ".cache/sprint61-native-source" / EXPECTED_REVISION
    model_root = tmp_path / "models"
    model_root.mkdir()
    model = create(model_root, architecture, source)
    root = tmp_path / "pack"
    site = root / "site-packages"
    site.mkdir(parents=True)
    (site / "oryn_engine").symlink_to(workspace / "projects/oryn-engine/src/oryn_engine")
    (root / "native").mkdir()
    for path in native.iterdir():
        if path.is_file():
            os.link(path, root / "native" / path.name)
    pack = builtin_registry().active_pack("oryn.llama-cpp")
    (site / "oryn-runtime-pack.json").write_text(
        json.dumps(
            {
                "packId": pack.pack_id,
                "version": pack.version,
                "nativeBackend": backend,
                "sourceRevision": worker.EXPECTED_REVISION,
            }
        )
    )
    runtime = PackRuntime(Path(sys.executable).resolve(), site, True)
    process = StreamingPackProcess()
    value = payload()
    value.update(modelRoot=str(model_root), modelFile=model.name)
    value["input"].update(
        text="Hello € 日本語",
        expectedArchitecture=architecture,
        expectedQuantization="Q8_0",
        contextLimit=1024,
        maxTokens=16,
    )
    outputs = []
    for _ in range(2):
        events = []
        result = process.invoke(
            pack,
            runtime,
            "generate",
            value,
            cancellation=threading.Event(),
            timeout_seconds=60,
            event_sink=lambda name, data, events=events: events.append((name, data)),
        )
        assert result["text"].isascii() and result["text"].islower()
        assert len(result["text"]) == 16
        assert result["backend"] == backend and result["architecture"] == architecture
        assert not any(name == "job.output.delta" for name, _ in events)
        outputs.append(result["text"])
        cancelled = threading.Event()

        def cancel(name, data, cancelled=cancelled):
            if name == "job.progress" and data.get("stage") == "generating":
                cancelled.set()

        started = time.monotonic()
        with pytest.raises(JobExecutionError) as error:
            process.invoke(
                pack,
                runtime,
                "generate",
                value,
                cancellation=cancelled,
                timeout_seconds=60,
                event_sink=cancel,
            )
        assert error.value.code == "JOB_CANCELLED"
        assert time.monotonic() - started < 5
    assert outputs[0] == outputs[1]
