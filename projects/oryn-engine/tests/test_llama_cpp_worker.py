from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from oryn_engine.runtime_packs import llama_cpp_worker as worker


@pytest.mark.parametrize("backend", ["cpu", "metal", "cuda"])
@pytest.mark.parametrize("vision", [False, True])
def test_generation_keeps_cpu_operations_on_cpu_and_streams_only_model_text(
    tmp_path, monkeypatch, backend, vision
):
    """Exercise subprocess IO and prompt-file lifetime, not just an argument helper."""
    captured = tmp_path / "captured.json"
    executable = tmp_path / "native-completion"
    if vision:
        (tmp_path / "native").mkdir()
        executable = tmp_path / "native/llama-mtmd-cli"
    output = "Test response." + ("" if vision else worker.CompletionText.footer)
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\nfrom pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "prompt_path = Path(args[args.index('--file') + 1])\n"
        f"Path({str(captured)!r}).write_text(json.dumps(dict(args=args, "
        "prompt=prompt_path.read_text(), promptPath=str(prompt_path))))\n"
        f"sys.stdout.write({output!r})\n"
        "sys.stdout.flush()\n"
    )
    executable.chmod(0o700)
    model = tmp_path / "fixture.gguf"
    metadata = SimpleNamespace(
        architecture="qwen3moe",
        version=3,
        quantization="Q8_0",
        tokenizer_model="gpt2",
        context_length=512,
        chat_template="<|im_start|>user<|im_end|>",
    )
    monkeypatch.setattr(worker, "_model", lambda _: (model, metadata))
    extra = {}
    if vision:
        from PIL import Image

        from oryn_engine import generated_capabilities
        from oryn_engine.runtime_packs import gguf

        library = tmp_path / "native" / worker.VISION_LIBRARY
        library.write_bytes(b"synthetic test library")
        projector = tmp_path / "mmproj-test-f16.gguf"
        projector.write_bytes(b"fixture")
        image = tmp_path / "input.png"
        Image.new("RGB", (32, 32), "red").save(image)
        monkeypatch.setattr(gguf, "inspect_projector", lambda *_: None)
        artifact = generated_capabilities.generated_artifact()
        monkeypatch.setattr(
            generated_capabilities,
            "generated_artifact",
            lambda: {
                **artifact,
                "buildInputs": {
                    "llamaVisionBinarySha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                    "llamaVisionLibrarySha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                },
            },
        )
        extra = {
            "projectorFile": projector.name,
            "inputPath": str(image),
            "modelRoot": str(tmp_path),
        }
    events = []
    monkeypatch.setattr(worker, "_event", lambda event, data: events.append((event, data)))
    value = worker._generate(
        executable,
        tmp_path,
        backend,
        {
            **extra,
            "input": {
                "messages": ([] if vision else [{"role": "system", "content": "Be concise."}])
                + [{"role": "user", "content": "Hello {{ untouched }}"}],
                "maxTokens": 32,
                "temperature": 0,
                "topP": 1,
                "chatTemplate": "chatml",
                "expectedArchitecture": "qwen3moe",
                "expectedQuantization": "Q8_0",
                "contextLimit": 512,
            },
        },
    )
    recorded = json.loads(captured.read_text())
    args = recorded["args"]
    assert args[args.index("--gpu-layers") + 1] == ("0" if backend == "cpu" else "all")
    for flag in ("--no-kv-offload", "--no-op-offload", "--device"):
        assert (flag in args) == (backend == "cpu")
    if backend == "cpu":
        assert args[args.index("--device") + 1] == "none"
    assert ("--no-mmproj-offload" in args) == (vision and backend == "cpu")
    assert recorded["prompt"] == (
        "Hello {{ untouched }}"
        if vision
        else (
            "<|im_start|>system\nBe concise.<|im_end|>\n"
            "<|im_start|>user\nHello {{ untouched }}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    )
    assert not Path(recorded["promptPath"]).exists()
    assert value["text"] == "Test response."
    assert (
        "".join(data["delta"] for event, data in events if event == "job.output.delta")
        == value["text"]
    )
    assert value["backend"] == backend


@pytest.mark.parametrize("architecture", ["qwen3moe", "qwen35moe"])
def test_moe_admission_requires_a_reviewed_template_and_quantization(architecture):
    from oryn_engine.native_capability_sources import validate_gguf_policy

    metadata = dict(
        architecture=architecture,
        version=3,
        quantization="Q8_0",
        tokenizer_model="gpt2",
        chat_template="<|im_start|>user<|im_end|>",
    )
    validate_gguf_policy(SimpleNamespace(**metadata), "chatml")
    for field, invalid in (
        ("architecture", "qwen3next"),
        ("quantization", "TQ1_0"),
        ("chat_template", "{{ arbitrary_repository_template }}"),
        ("tokenizer_model", "custom"),
    ):
        with pytest.raises(ValueError):
            validate_gguf_policy(SimpleNamespace(**{**metadata, field: invalid}), "chatml")


def test_gemma4_sealed_turns_and_policy():
    from oryn_engine.native_capability_sources import llama_cpp_chat_template, validate_gguf_policy

    messages = [
        {"role": "system", "content": " Be brief. "},
        {"role": "user", "content": "Hi {{ untouched }}"},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "Name?"},
    ]
    assert worker._render_prompt(messages, "gemma4") == (
        "<|turn>system\nBe brief.<turn|>\n"
        "<|turn>user\nHi {{ untouched }}<turn|>\n"
        "<|turn>model\nHello.<turn|>\n"
        "<|turn>user\nName?<turn|>\n"
        "<|turn>model\n"
    )
    embedded = "{%- macro x() -%}{%- endmacro -%}<|turn>user\n{{ m }}<turn|>"
    assert llama_cpp_chat_template("gemma4", embedded) == "gemma4"
    # Gemma 3 framing is not Gemma 4 framing, and neither selects the other.
    assert llama_cpp_chat_template("gemma4", "<start_of_turn>user<end_of_turn>") is None
    assert llama_cpp_chat_template("gemma3", embedded) is None
    metadata = dict(
        architecture="gemma4",
        version=3,
        quantization="IQ4_XS",
        tokenizer_model="gemma4",
        chat_template=embedded,
    )
    validate_gguf_policy(SimpleNamespace(**metadata), "gemma4")
    for field, invalid in (
        ("tokenizer_model", "gpt2"),
        ("chat_template", "{{ arbitrary_repository_template }}"),
        ("quantization", "TQ1_0"),
    ):
        with pytest.raises(ValueError):
            validate_gguf_policy(SimpleNamespace(**{**metadata, field: invalid}), "gemma4")
    with pytest.raises(ValueError):
        validate_gguf_policy(SimpleNamespace(**metadata), "chatml")
    # Other families keep their original tokenizer constraint.
    with pytest.raises(ValueError):
        validate_gguf_policy(
            SimpleNamespace(
                **{**metadata, "architecture": "qwen3", "chat_template": "<|im_start|><|im_end|>"}
            ),
            "chatml",
        )


@pytest.mark.parametrize("template", ["chatml", "gemma4", "gemma3"])
def test_vision_uses_only_sealed_templates(tmp_path, monkeypatch, template):
    """Gemma 4 vision passes the engine-owned constant; the repository template is never used."""
    import hashlib

    from PIL import Image

    from oryn_engine import generated_capabilities
    from oryn_engine.runtime_packs import gguf

    captured = tmp_path / "captured.json"
    (tmp_path / "native").mkdir()
    executable = tmp_path / "native/llama-mtmd-cli"
    executable.write_text(
        f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
        f"Path({str(captured)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.stdout.write('Red')\n"
    )
    executable.chmod(0o700)
    library = tmp_path / "native" / worker.VISION_LIBRARY
    library.write_bytes(b"synthetic test library")
    projector = tmp_path / "mmproj-test-f16.gguf"
    projector.write_bytes(b"fixture")
    image = tmp_path / "input.png"
    Image.new("RGB", (32, 32), "red").save(image)
    architecture = "gemma4" if template != "chatml" else "qwen35"
    metadata = SimpleNamespace(
        architecture=architecture,
        version=3,
        quantization="IQ4_XS",
        tokenizer_model="gemma4" if architecture == "gemma4" else "gpt2",
        context_length=4096,
        chat_template="<|turn>user\n<turn|>"
        if architecture == "gemma4"
        else "<|im_start|><|im_end|>",
    )
    monkeypatch.setattr(worker, "_model", lambda _: (tmp_path / "fixture.gguf", metadata))
    monkeypatch.setattr(gguf, "inspect_projector", lambda *_: None)
    monkeypatch.setattr(worker, "_event", lambda *_: None)
    artifact = generated_capabilities.generated_artifact()
    monkeypatch.setattr(
        generated_capabilities,
        "generated_artifact",
        lambda: {
            **artifact,
            "buildInputs": {
                "llamaVisionBinarySha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                "llamaVisionLibrarySha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            },
        },
    )
    payload = {
        "projectorFile": projector.name,
        "inputPath": str(image),
        "modelRoot": str(tmp_path),
        "input": {
            "messages": [{"role": "user", "content": "Color?"}],
            "maxTokens": 8,
            "temperature": 0,
            "topP": 1,
            "chatTemplate": template,
            "expectedArchitecture": architecture,
            "expectedQuantization": "IQ4_XS",
            "contextLimit": 4096,
        },
    }
    if template == "gemma3":
        with pytest.raises(ValueError):
            worker._generate(executable, tmp_path, "metal", payload)
        return
    assert worker._generate(executable, tmp_path, "metal", payload)["text"] == "Red"
    args = json.loads(captured.read_text())
    value = args[args.index("--chat-template") + 1]
    if template == "gemma4":
        assert "--jinja" in args and "--no-jinja" not in args
        assert value == GEMMA4_VISION_TEMPLATE and metadata.chat_template not in value
    else:
        assert value == "chatml" and "--no-jinja" in args and "--jinja" not in args


# The engine template reviewed and run on Metal before chat formats moved to signed data.
GEMMA4_VISION_TEMPLATE = (
    "{{- bos_token -}}"
    "{%- for message in messages -%}"
    "<|turn>{{ 'model' if message['role'] == 'assistant' else message['role'] }}\n"
    "{{ message['content'] | trim }}<turn|>\n"
    "{% endfor -%}"
    "{%- if add_generation_prompt -%}<|turn>model\n{% endif -%}"
)


def test_every_admitted_template_is_allowed_by_the_public_protocol():
    """The browser validates inspection results; an unknown template fails inspection."""
    import importlib.resources

    import oryn_protocol

    schema = json.loads(
        (
            importlib.resources.files(oryn_protocol) / "schemas/v1/protocol-v1.schema.json"
        ).read_text()
    )
    allowed: set[str] = set()

    def collect(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "chatTemplate" and isinstance(value, dict) and "enum" in value:
                    allowed.update(value["enum"])
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(schema)
    policy = json.loads(
        (Path(__file__).parents[1] / "build-inputs/llama-chat-policy.json").read_text()
    )
    admitted = {name for names in policy["admitted"].values() for name in names}
    assert allowed and admitted <= allowed, admitted - allowed
    # The Protocol list is generated from the signed chat-format table (scripts/bootstrap).
    table = json.loads((Path(__file__).parents[1] / "build-inputs/chat-formats.json").read_text())
    assert allowed == set(table["formats"]), "run scripts/bootstrap to regenerate the Protocol"


@pytest.mark.parametrize(
    ("line", "valid"),
    [
        (f"version: 0.8.0 (build {worker.EXPECTED_BUILD}, commit 8642541b2)", True),
        ("version: 0.8.0 (build 11710, commit 8642541b2)", False),
        (f"version: 0.8.0 (build {worker.EXPECTED_BUILD}, commit de6999570)", False),
        (f"version: {worker.EXPECTED_BUILD} (8642541b2)", False),
    ],
)
def test_probe_accepts_only_the_pinned_native_identity(tmp_path, line, valid):
    (tmp_path / "native").mkdir()
    executable = tmp_path / "native/llama-completion"
    executable.write_text(f"#!/bin/sh\necho '{line}'\necho 'built with AppleClang'\n")
    executable.chmod(0o755)
    if valid:
        assert worker._probe(executable, tmp_path, "metal")["build"] == worker.EXPECTED_BUILD
    else:
        with pytest.raises(ValueError, match="identity"):
            worker._probe(executable, tmp_path, "metal")


@pytest.mark.parametrize("name", __import__("oryn_engine.chat_formats", fromlist=["names"]).names())
def test_worker_accepts_every_reviewed_format_and_nothing_else(name):
    value = {
        "messages": [{"role": "user", "content": "Hi"}],
        "maxTokens": 8,
        "temperature": 0,
        "topP": 1,
        "chatTemplate": name,
        "expectedArchitecture": "qwen2",
        "expectedQuantization": "Q4_K_M",
        "contextLimit": 4096,
    }
    assert worker._input(value)["chatTemplate"] == name
    with pytest.raises(ValueError):
        worker._input({**value, "chatTemplate": "unreviewed"})


def test_worker_reads_a_split_set_inside_one_quantization_folder(tmp_path, monkeypatch):
    folder = tmp_path / "Q4_K_M"
    folder.mkdir()
    for index in (1, 2):
        (folder / f"Model-Q4_K_M-0000{index}-of-00002.gguf").write_bytes(b"x")
    (tmp_path / "Model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"decoy at the root")
    seen = {}

    def fake_inspect(paths, names):
        seen.update(paths=paths, names=names)
        return SimpleNamespace(architecture="qwen35")

    monkeypatch.setattr(worker, "inspect_gguf_set", fake_inspect)
    from oryn_engine import native_capability_sources

    monkeypatch.setattr(native_capability_sources, "validate_gguf_policy", lambda _m: None)
    path, _ = worker._model(
        {"modelRoot": str(tmp_path), "modelFile": "Q4_K_M/Model-Q4_K_M-00001-of-00002.gguf"}
    )
    assert path == (folder / "Model-Q4_K_M-00001-of-00002.gguf").resolve()
    assert seen["names"] == [
        "Q4_K_M/Model-Q4_K_M-00001-of-00002.gguf",
        "Q4_K_M/Model-Q4_K_M-00002-of-00002.gguf",
    ]
    assert seen["paths"] == [tmp_path / n for n in seen["names"]]


@pytest.mark.parametrize("model_file", ["../x.gguf", "a/b/x.gguf", "/x.gguf", "linked/x.gguf"])
def test_worker_rejects_paths_outside_the_repository_cache(tmp_path, model_file):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.gguf").write_bytes(b"x")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        worker._model({"modelRoot": str(root), "modelFile": model_file})
