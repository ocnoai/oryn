from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

import oryn_engine.models as models_module
from oryn_engine.models import HuggingFaceClient, InstallCancelled, ModelStore, ModelStoreError


def test_hub_headers_do_not_probe_agent_or_send_environment_telemetry(monkeypatch):
    from huggingface_hub import constants
    from huggingface_hub.utils import _headers

    def forbidden_probe():
        raise AssertionError("Model requests must not probe the agent registry")

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_TELEMETRY", False)
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
    monkeypatch.setattr(_headers, "detect_agent", forbidden_probe)
    HuggingFaceClient()
    headers = _headers.build_hf_headers(token=False)
    assert "agent/" not in headers["user-agent"]
    assert "torch/" not in headers["user-agent"]
    assert "authorization" not in headers


SHA = "a" * 40
TINY_QWEN_SHA = "f56c2fbf7fabc03ffff3494d3c2f8d06c774493d"


@pytest.mark.parametrize("filename", ["model_index.json", "config.json"])
def test_hub_root_metadata_replaces_derived_summary(monkeypatch, filename):
    client = HuggingFaceClient()
    root = {"_class_name": "Flux2KleinPipeline"}
    info = SimpleNamespace(
        sha=SHA,
        gated=False,
        private=False,
        siblings=[SimpleNamespace(rfilename=filename)],
        config={"diffusers": {"_class_name": "StalePipeline"}},
    )
    monkeypatch.setattr(client._api, "model_info", lambda **kwargs: info)

    def read_json(repository, revision, name):
        assert (repository, revision, name) == ("org/model", SHA, filename)
        return root

    monkeypatch.setattr(client, "read_json", read_json)
    assert client.model_info("org/model", "main").config == root


def test_generated_diffusion_preserves_legacy_image_editing(tmp_path):
    from oryn_engine.models import ModelFile

    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    profile = store._inspect_generated_pack(
        "org/image",
        SHA,
        [ModelFile("model_index.json", 10, "1" * 64)],
        {"_class_name": "StableDiffusionPipeline"},
        "text-to-image",
        True,
        source_kind="diffusers",
    )
    assert profile["technicalCapabilities"]["capabilities"]["imageEditing"] is True
    assert profile["technicalCapabilities"]["inputModalities"] == ["text", "image"]
    assert profile["runtimePack"]["taskContract"] == "text-to-image-v1"


@pytest.mark.parametrize("size", [32, [64, 128]])
def test_unconditional_inspection_crosses_protocol_with_fixed_native_geometry(tmp_path, size):
    from oryn_protocol import ProtocolSchemaStore
    from oryn_protocol.task_contracts import validate_invocation

    from oryn_engine.models import ModelFile
    from oryn_engine.task_profiles import profile_for_execution

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            assert filename == "config.json"
            return {"sample_size": size, "in_channels": 3, "out_channels": 3}

    store = ModelStore(tmp_path, hub=Hub(), runtime_available=True)
    profile = store._inspect_generated_pack(
        "org/unconditional",
        SHA,
        [
            ModelFile(name, 10, "1" * 64)
            for name in (
                "model_index.json",
                "config.json",
                "scheduler_config.json",
                "diffusion_pytorch_model.safetensors",
            )
        ],
        {
            "_class_name": "DDPMPipeline",
            "_module": "modeling_ddpm.py",
            "unet": ["diffusers", "UNet2DModel"],
            "scheduler": ["diffusers", "DDPMScheduler"],
        },
        "unconditional-image-generation",
        True,
        source_kind="diffusers",
    )
    profile["taskProfile"] = profile_for_execution(profile)
    ProtocolSchemaStore().validator("candidateExecutionProfile").validate(profile)
    height, width = [size, size] if isinstance(size, int) else size
    task = profile["taskProfile"]
    assert task["parameters"]["width"]["default"] == width
    assert task["parameters"]["height"]["default"] == height
    assert not task["parameters"]["width"]["editable"]
    assert not task["parameters"]["height"]["editable"]
    validate_invocation(task, {"width": width, "height": height, "steps": 20})
    with pytest.raises(ValueError):
        validate_invocation(task, {"width": width + 8, "height": height})


def test_generated_native_image_plan_records_actual_bfloat16_precision(tmp_path):
    from oryn_engine.diffusion_tasks import IMAGE_PIPELINES
    from oryn_engine.models import ModelFile

    family = "Flux2KleinPipeline"
    config = {
        "_class_name": family,
        **{k: list(v) for k, v in IMAGE_PIPELINES[family].items()},
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "tokenizer": ["transformers", "Qwen2TokenizerFast"],
    }
    files = ["model_index.json", "scheduler/scheduler_config.json", "tokenizer/tokenizer.json"]
    for component in IMAGE_PIPELINES[family]:
        files.extend([f"{component}/config.json", f"{component}/model.safetensors"])
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _pack: None if _pack.provider else object()
    )
    profile = store._inspect_generated_pack(
        "org/image",
        SHA,
        [ModelFile(f, 10, "1" * 64) for f in files],
        config,
        "image-to-image",
        True,
        source_kind="diffusers",
    )
    selected = profile["runtimePlan"]["selected"]
    assert selected["precision"] == ("bf16" if selected["backend"] in {"mps", "cuda"} else "fp32")
    assert profile["runtimePack"]["taskContract"] == "image-editing-v1"


class FakeHub:
    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        files: list[tuple[str, int, str]] | None = None,
        sha: str = SHA,
        block_download: bool = False,
        pipeline_tag: str | None = "automatic-speech-recognition",
        gated: bool | str = False,
        private: bool = False,
        gguf: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or {
            "architectures": ["WhisperForConditionalGeneration"],
            "num_parameters": 1000,
        }
        self.files = files or [
            ("config.json", 10, "b" * 64),
            ("model.safetensors", 100, "c" * 64),
            ("tokenizer.json", 0, "e" * 64),
            ("preprocessor_config.json", 0, "f" * 64),
        ]
        self.sha = sha
        self.block_download = block_download
        self.pipeline_tag = pipeline_tag
        self.gated = gated
        self.private = private
        self.gguf = gguf
        self.downloaded: list[str] = []
        self.download_started = threading.Event()
        self.release_download = threading.Event()

    def model_info(self, repo_id: str, revision: str, files_metadata: bool = True) -> Any:
        del repo_id, revision, files_metadata
        siblings = [
            SimpleNamespace(
                rfilename=name,
                size=size,
                lfs=SimpleNamespace(
                    sha256=hashlib.sha256(blob[:1].encode() * size).hexdigest()
                    if size < 2_000_000
                    else blob
                ),
            )
            for name, size, blob in self.files
        ]
        return SimpleNamespace(
            sha=self.sha,
            config=self.config,
            pipeline_tag=self.pipeline_tag,
            siblings=siblings,
            gated=self.gated,
            private=self.private,
            gguf=self.gguf,
        )

    def download(
        self,
        repo_id: str,
        filename: str,
        revision: str,
        cache_dir: str,
        progress: models_module.DownloadProgress | None = None,
    ) -> str:
        del repo_id, revision
        self.downloaded.append(filename)
        if self.block_download:
            self.download_started.set()
            self.release_download.wait(timeout=5)
            self.block_download = False
        target = Path(cache_dir) / "snapshots" / SHA / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        _, size, blob = next(item for item in self.files if item[0] == filename)
        target.write_bytes(blob[:1].encode() * size)
        if progress is not None:
            progress(len(target.read_bytes()), None)
        return str(target)


def model(repository: str = "openai/whisper-small", revision: str = "main") -> dict[str, str]:
    return {"provider": "huggingface", "repository": repository, "revision": revision}


def test_mutable_revision_resolves_to_stable_exact_commit_and_capabilities(tmp_path: Path) -> None:
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    first = store.inspect(model())
    second = store.inspect(model())

    assert first["exactRevision"] == SHA
    assert first["model"]["revision"] == SHA
    assert second["exactRevision"] == first["exactRevision"]
    assert store.source_url(first["model"]) == (
        f"https://huggingface.co/openai/whisper-small/tree/{SHA}"
    )
    assert first["technicalCapabilities"]["inputModalities"] == ["audio"]
    assert first["technicalCapabilities"]["outputModalities"] == ["text"]
    assert first["technicalCapabilities"]["capabilities"]["speechRecognition"] is True


def test_musicgen_is_an_explicit_text_to_audio_runner_profile(tmp_path: Path) -> None:
    profile = ModelStore(
        tmp_path,
        hub=FakeHub(
            config={"architectures": ["MusicgenForConditionalGeneration"]},
            pipeline_tag="text-to-audio",
        ),
        runtime_available=True,
    ).inspect(model("org/musicgen-model"))

    assert profile["executable"] is True
    assert profile["runner"] == "transformers"
    assert profile["technicalCapabilities"] == {
        "architectures": ["MusicgenForConditionalGeneration"],
        "task": "text-to-audio",
        "inputModalities": ["text"],
        "outputModalities": ["audio"],
        "capabilities": {"audioOutput": True, "streaming": False},
    }


def test_qwen_forced_aligner_exposes_audio_text_to_structured_contract(tmp_path: Path) -> None:
    profile = ModelStore(
        tmp_path,
        hub=FakeHub(
            config={"architectures": ["Qwen3ASRForTokenClassification"]},
            pipeline_tag="token-classification",
        ),
        runtime_available=True,
    ).inspect(model("Qwen/Qwen3-ForcedAligner-0.6B-hf"))

    assert profile["executable"] is True
    assert profile["runner"] == "transformers"
    assert profile["technicalCapabilities"] == {
        "architectures": ["Qwen3ASRForTokenClassification"],
        "task": "forced-alignment",
        "inputModalities": ["audio", "text"],
        "outputModalities": ["structured"],
        "capabilities": {"structuredOutput": True},
    }


def test_qwen35_conditional_generation_is_executable_for_text_chat(tmp_path: Path) -> None:
    from oryn_protocol import ProtocolSchemaStore

    profile = ModelStore(
        tmp_path,
        hub=FakeHub(
            config={
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "tokenizer_config": {"chat_template": "{{ messages }}"},
            },
            pipeline_tag="image-text-to-text",
        ),
        runtime_available=True,
    ).inspect(model("Qwen/Qwen3.5-0.8B"))

    assert profile["executable"] is True
    assert profile["runner"] == "transformers"
    assert profile["technicalCapabilities"]["task"] == "text-generation"
    assert profile["technicalCapabilities"]["inputModalities"] == ["text", "image"]
    assert profile["technicalCapabilities"]["outputModalities"] == ["text"]
    assert profile["technicalCapabilities"]["capabilities"]["chat"] is True
    assert profile["technicalCapabilities"]["capabilities"]["multimodalConversation"] is True
    ProtocolSchemaStore().validator("candidateExecutionProfile").validate(profile)


@pytest.mark.parametrize("task", ["conversational", "image-text-to-text"])
@pytest.mark.parametrize("architecture", ["qwen35", "qwen35moe"])
def test_generic_qwen35_gguf_uses_native_architecture_metadata_and_llama_cpp(
    tmp_path: Path,
    task,
    architecture,
) -> None:
    hub = FakeHub(
        config={
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
        },
        files=[
            ("Qwen3.8-27B-UD-Q4_K_M.gguf", 16_500_000_000, "b" * 64),
            ("Qwen3.8-27B-Q8_0.gguf", 29_000_000_000, "c" * 64),
            ("mmproj-F16.gguf", 900_000_000, "d" * 64),
            ("MTP/mtp-Qwen3.8-27B-Q4_0.gguf", 1_300_000_000, "e" * 64),
        ],
        pipeline_tag=task,
        gguf={"architecture": architecture, "context_length": 262_144},
    )
    store = ModelStore(
        tmp_path,
        hub=hub,
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 64_000_000_000),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _pack: None if _pack.provider else object()
    )

    profile = store.inspect(model("unsloth/Qwen3.8-27B-GGUF"))

    assert profile["executable"] is True
    assert profile["runner"] == "runtime-pack"
    assert profile["loadDisposition"] == "trust-required"
    assert profile["technicalCapabilities"]["architectures"] == [architecture]
    assert profile["selectedFiles"] == [
        {"name": "Qwen3.8-27B-UD-Q4_K_M.gguf", "sizeBytes": 16_500_000_000}
    ] + (
        [{"name": "mmproj-F16.gguf", "sizeBytes": 900_000_000}]
        if task == "image-text-to-text"
        else []
    )
    assert profile["runtimePack"]["packId"] == "oryn.llama-cpp"
    assert profile["runtimePack"]["trustedMetadata"]["adapterArchitecture"] == architecture
    assert profile["runtimePack"]["trustedMetadata"]["quantization"] == "Q4_K_M"
    if task == "image-text-to-text":
        assert profile["runtimePack"]["trustedMetadata"]["projectorFile"] == "mmproj-F16.gguf"
        assert profile["technicalCapabilities"]["inputModalities"] == ["text", "image"]


def test_generic_gguf_reports_unknown_embedded_architecture_on_runtime_axis(
    tmp_path: Path,
) -> None:
    hub = FakeHub(
        config={"architectures": ["FutureForCausalLM"]},
        files=[("future-Q4_K_M.gguf", 1_000_000, "b" * 64)],
        pipeline_tag="text-generation",
        gguf={"architecture": "future", "context_length": 4096},
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _pack: None if _pack.provider else object()
    )

    profile = store.inspect(model("org/future-GGUF"))

    assert profile["executable"] is False
    assert profile["eligibility"]["runtime"]["state"] == "unsupported"
    assert "RUNTIME_CAPABILITY_UNSUPPORTED" in profile["blockingReasons"]


def test_generic_gguf_install_revalidates_header_before_persisting_execution_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub(
        config={"architectures": ["Qwen3_5ForConditionalGeneration"]},
        files=[("qwen35-Q4_K_M.gguf", 1_000_000, "b" * 64)],
        pipeline_tag="text-generation",
        gguf={"architecture": "qwen35", "context_length": 262_144},
    )
    store = ModelStore(
        tmp_path,
        hub=hub,
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 8_000_000_000),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _pack: None if _pack.provider else object()
    )
    monkeypatch.setattr(
        models_module,
        "inspect_gguf",
        lambda _path: SimpleNamespace(
            architecture="qwen35",
            quantization="Q4_K_M",
            context_length=262_144,
            version=3,
            tokenizer_model="gpt2",
            chat_template="<|im_start|>user<|im_end|>",
        ),
    )

    installed = store.install("op_generic_gguf", model("org/qwen35-GGUF"))
    record = store.installed_record(installed["model"])

    assert record["runner"] == "runtime-pack"
    assert record["files"][0]["name"] == "qwen35-Q4_K_M.gguf"
    assert record["runtimePack"]["trustedMetadata"] == {
        "license": "Repository-defined",
        "quantization": "Q4_K_M",
        "contextLength": 262_144,
        "chatTemplate": "chatml",
        "modelFile": "qwen35-Q4_K_M.gguf",
        "adapterArchitecture": "qwen35",
        "conformance": {
            "status": "fixture-only",
            "platforms": ["macos-arm64-metal", "cpu"],
        },
    }


def test_inspection_does_not_claim_runners_that_execution_cannot_dispatch(tmp_path: Path) -> None:
    for index, (architecture, task) in enumerate(
        [("UnsupportedEncoder", "feature-extraction"), ("UnknownAudioModel", "text-to-audio")]
    ):
        profile = ModelStore(
            tmp_path / str(index),
            hub=FakeHub(config={"architectures": [architecture]}, pipeline_tag=task),
            runtime_available=True,
        ).inspect(model(f"org/unsupported-{index}"))
        assert profile["executable"] is False
        assert "RUNNER_UNSUPPORTED" in profile["blockingReasons"]


def test_custom_code_unsafe_weights_malformed_metadata_and_traversal_are_blocked(
    tmp_path: Path,
) -> None:
    custom = ModelStore(
        tmp_path / "custom",
        hub=FakeHub(config={"architectures": ["AutoModelForCausalLM"], "auto_map": {"x": "y"}}),
        runtime_available=True,
    ).inspect(model())
    assert custom["requiresCustomCode"] is True
    assert "CUSTOM_CODE_REQUIRED" in custom["blockingReasons"]

    unsafe = ModelStore(
        tmp_path / "unsafe",
        hub=FakeHub(files=[("pytorch_model.bin", 100, "d" * 64)]),
        runtime_available=True,
    ).inspect(model())
    assert "UNSAFE_MODEL_FORMAT" in unsafe["blockingReasons"]

    with pytest.raises(ModelStoreError, match="architectures"):
        ModelStore(
            tmp_path / "malformed",
            hub=FakeHub(config={"architectures": "not-an-array"}),
            runtime_available=True,
        ).inspect(model())
    with pytest.raises(ModelStoreError, match="file name"):
        ModelStore(
            tmp_path / "traversal",
            hub=FakeHub(files=[("../escape.safetensors", 100, "e" * 64)]),
            runtime_available=True,
        ).inspect(model())


def test_unselected_repository_python_is_inert_but_auto_map_remains_blocked(tmp_path: Path) -> None:
    hub = FakeHub(
        config={"architectures": ["Qwen2ForCausalLM"]},
        files=[
            ("config.json", 10, "a" * 64),
            ("model.safetensors", 100, "b" * 64),
            ("tokenizer.json", 20, "c" * 64),
            ("training/train.py", 50, "d" * 64),
        ],
        pipeline_tag="text-generation",
    )
    profile = ModelStore(tmp_path / "inert", hub=hub, runtime_available=True).inspect(model())
    assert profile["executable"] is True
    assert profile["requiresCustomCode"] is False
    assert profile["unselectedExecutableFiles"] == ["training/train.py"]
    assert all(not item["name"].endswith(".py") for item in profile["selectedFiles"])
    assert profile["descriptorSchemaVersion"] == "oryn-model-descriptor/1"
    assert len(profile["capabilityRegistryDigest"]) == 64
    assert profile["loadDisposition"] == "trust-required"
    assert profile["eligibility"]["runtime"]["state"] == "supported"
    assert profile["matchedCapability"]["taskAdapterId"] == "oryn.task.chat-v1"

    blocked = ModelStore(
        tmp_path / "auto-map",
        hub=FakeHub(
            config={
                "architectures": ["Qwen2ForCausalLM"],
                "auto_map": {"AutoModel": "modeling.Model"},
            },
            files=hub.files,
            pipeline_tag="text-generation",
        ),
        runtime_available=True,
    ).inspect(model())
    assert blocked["executable"] is False
    assert "CUSTOM_CODE_REQUIRED" in blocked["blockingReasons"]


def test_gated_or_private_model_is_not_advertised_as_executable(tmp_path: Path) -> None:
    for index, hub in enumerate([FakeHub(gated="manual"), FakeHub(private=True)]):
        store = ModelStore(tmp_path / str(index), hub=hub, runtime_available=True)
        profile = store.inspect(model())

        assert profile["executable"] is False
        assert "MODEL_ACCESS_RESTRICTED" in profile["blockingReasons"]
        with pytest.raises(ModelStoreError) as restricted:
            store.install(f"operation_{index}", profile["model"])
        assert restricted.value.code == "MODEL_ACCESS_RESTRICTED"
        assert hub.downloaded == []


def test_gated_model_is_executable_only_after_explicit_credential_access_check(
    tmp_path: Path,
) -> None:
    class AuthorizedHub(FakeHub):
        def has_access(self, repo_id: str) -> bool:
            del repo_id
            return True

    profile = ModelStore(
        tmp_path,
        hub=AuthorizedHub(gated="auto"),
        runtime_available=True,
    ).inspect(model())

    assert "MODEL_ACCESS_RESTRICTED" not in profile.get("blockingReasons", [])
    assert profile["executable"] is True


@pytest.mark.parametrize(
    ("error_type", "status"),
    [(GatedRepoError, 403), (HfHubHTTPError, 401), (HfHubHTTPError, 403)],
)
def test_host_authentication_download_failures_have_a_stable_model_error(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[HfHubHTTPError],
    status: int,
) -> None:
    response = httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/model"))

    def fail_download(**_kwargs: Any) -> str:
        raise error_type("access denied", response=response)

    monkeypatch.setattr(models_module, "hf_hub_download", fail_download)

    with pytest.raises(ModelStoreError) as restricted:
        HuggingFaceClient().download("org/model", "config.json", SHA, "/tmp/cache")
    assert restricted.value.code == "MODEL_ACCESS_RESTRICTED"


def test_hugging_face_client_uses_only_the_supervised_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[str | bool | None] = []

    def fake_download(**kwargs: Any) -> str:
        received.append(kwargs.get("token"))
        return "/tmp/cache/model.safetensors"

    monkeypatch.setattr(models_module, "hf_hub_download", fake_download)
    token = f"hf_{'z' * 32}"

    HuggingFaceClient(lambda: token).download("org/model", "model.safetensors", SHA, "/tmp/cache")
    HuggingFaceClient().download("org/model", "model.safetensors", SHA, "/tmp/cache")

    assert received == [token, False]


def test_hugging_face_download_reports_streaming_byte_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reported: list[tuple[int, int | None]] = []

    def fake_download(**kwargs: Any) -> str:
        progress_type = kwargs["tqdm_class"]
        with progress_type(total=30, initial=0, disable=True) as progress:
            progress.update(10)
            progress.update(20)
        return "/tmp/cache/model.safetensors"

    monkeypatch.setattr(models_module, "hf_hub_download", fake_download)

    result = HuggingFaceClient().download(
        "org/model",
        "model.safetensors",
        SHA,
        "/tmp/cache",
        lambda downloaded, total: reported.append((downloaded, total)),
    )

    assert result == "/tmp/cache/model.safetensors"
    assert reported == [(0, 30), (10, 30), (30, 30)]


def test_install_forwards_in_file_download_progress(tmp_path: Path) -> None:
    class StreamingFakeHub(FakeHub):
        def download(
            self,
            repo_id: str,
            filename: str,
            revision: str,
            cache_dir: str,
            progress: models_module.DownloadProgress | None = None,
        ) -> str:
            if progress is not None:
                progress(25, 100)
                progress(75, 100)
            return super().download(repo_id, filename, revision, cache_dir, progress)

    progress_events: list[dict[str, Any]] = []
    store = ModelStore(
        tmp_path,
        event_sink=lambda name, data: (
            progress_events.append(data.copy())
            if name == "model.install.progress" and data["stage"] == "downloading"
            else None
        ),
        hub=StreamingFakeHub(
            files=[
                ("model.safetensors", 100, "c" * 64),
                ("config.json", 0, "b" * 64),
                ("tokenizer.json", 0, "e" * 64),
                ("preprocessor_config.json", 0, "f" * 64),
            ]
        ),
        runtime_available=True,
    )

    store.install("op_streaming_progress", model())

    downloaded = [event["downloadedBytes"] for event in progress_events]
    assert 25 in downloaded
    assert 75 in downloaded
    assert downloaded[-1] == 100


def test_install_reports_required_and_available_disk_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    available = 256 * 1024 * 1024
    monkeypatch.setattr(
        models_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=available),
    )
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)

    with pytest.raises(ModelStoreError) as failure:
        store.install("op_insufficient_disk", model())

    assert failure.value.code == "DISK_SPACE_INSUFFICIENT"
    assert failure.value.details == {
        "requiredBytes": 512 * 1024 * 1024 + 110,
        "availableBytes": available,
    }


@pytest.mark.parametrize("tamper", [False, True])
def test_reprepare_reuses_verified_installed_bytes_with_only_scratch_space(
    tmp_path, monkeypatch, tamper
):
    hub = FakeHub()
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store.install("initial", model())
    hub.downloaded.clear()
    record = store._records[0]
    record["runtimeBuildDigest"] = "0" * 64
    if tamper:
        entry = next(f for f in record["files"] if f["name"] == "config.json")
        (store.cache_root / entry["cache_path"]).write_bytes(b"x" * entry["size"])
    monkeypatch.setattr(
        models_module.shutil, "disk_usage", lambda _: SimpleNamespace(free=512 * 1024**2)
    )
    if tamper:
        with pytest.raises(ModelStoreError) as error:
            store.install("reprepare", model())
        assert error.value.code == "MODEL_INTEGRITY_FAILED"
    else:
        store.install("reprepare", model())
        assert not hub.downloaded


@pytest.mark.parametrize("change", [None, "digest", "size", "escape", "unidentified"])
def test_uninstalled_cache_requires_exact_integrity_before_disk_credit(
    tmp_path, monkeypatch, change
):
    class CachedHub(FakeHub):
        def cached_file(self, repo_id, filename, revision, cache_dir):
            assert repo_id == "openai/whisper-small" and revision == SHA
            return str(Path(cache_dir) / "snapshots" / revision / filename)

        def model_info(self, *args, **kwargs):
            info = super().model_info(*args, **kwargs)
            if change == "unidentified":
                for sibling in info.siblings:
                    sibling.lfs = None
            return info

    hub = CachedHub()
    store = ModelStore(tmp_path / "store", hub=hub, runtime_available=True)
    for name, _, _ in hub.files:
        hub.download("openai/whisper-small", name, SHA, str(store.cache_root))
    hub.downloaded.clear()
    weight = store.cache_root / "snapshots" / SHA / "model.safetensors"
    if change == "digest":
        weight.write_bytes(b"x" * weight.stat().st_size)
    elif change == "size":
        weight.write_bytes(b"x")
    elif change == "escape":
        outside = tmp_path / "outside.safetensors"
        weight.rename(outside)
        weight.symlink_to(outside)
    monkeypatch.setattr(
        models_module.shutil, "disk_usage", lambda _: SimpleNamespace(free=512 * 1024**2)
    )
    assert store._records == []  # Cache bytes do not grant installation or trust.
    if change:
        with pytest.raises(ModelStoreError) as error:
            store.install("cached-original", model())
        expected = {
            "escape": "MODEL_SECURITY_BLOCKED",
            "unidentified": "DISK_SPACE_INSUFFICIENT",
        }.get(change, "MODEL_INTEGRITY_FAILED")
        assert error.value.code == expected
        assert store._records == [] and not hub.downloaded
    else:
        store.install("cached-original", model())
        assert not hub.downloaded
        assert len(store._records) == 1
        assert all(len(entry["content_sha256"]) == 64 for entry in store._records[0]["files"])


def test_cached_hub_lookup_is_exact_and_network_free(monkeypatch, tmp_path):
    import huggingface_hub

    calls = []

    def lookup(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(tmp_path / filename) if filename == "model.safetensors" else object()

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lookup)
    client = HuggingFaceClient()
    assert client.cached_file("org/model", "model.safetensors", SHA, str(tmp_path))
    assert client.cached_file("org/model", "missing.safetensors", SHA, str(tmp_path)) is None
    assert calls[0] == (
        "org/model",
        "model.safetensors",
        {"revision": SHA, "cache_dir": str(tmp_path)},
    )


def test_oversized_metadata_and_revision_identity_changes_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelStoreError) as oversized_config:
        ModelStore(
            tmp_path / "oversized-config",
            hub=FakeHub(config={"architectures": ["x"], "payload": "x" * 65_537}),
            runtime_available=True,
        ).inspect(model())
    assert oversized_config.value.code == "MODEL_METADATA_INVALID"

    huge_files = [(f"weights/{index}.safetensors", 1, "a" * 64) for index in range(10_001)]
    with pytest.raises(ModelStoreError) as oversized_files:
        ModelStore(
            tmp_path / "oversized-files",
            hub=FakeHub(files=huge_files),
            runtime_available=True,
        ).inspect(model())
    assert oversized_files.value.code == "MODEL_METADATA_INVALID"

    with pytest.raises(ModelStoreError) as oversized_file_size:
        ModelStore(
            tmp_path / "oversized-file-size",
            hub=FakeHub(files=[("model.safetensors", 2**63, "a" * 64)]),
            runtime_available=True,
        ).inspect(model())
    assert oversized_file_size.value.code == "MODEL_METADATA_INVALID"

    class MutableIdentityHub(FakeHub):
        calls = 0

        def model_info(self, repo_id: str, revision: str, files_metadata: bool = True) -> Any:
            self.calls += 1
            result = super().model_info(repo_id, revision, files_metadata)
            if self.calls > 1:
                result.sha = "b" * 40
            return result

    mutable = ModelStore(
        tmp_path / "mutable-identity", hub=MutableIdentityHub(), runtime_available=True
    )
    with pytest.raises(ModelStoreError) as changed:
        mutable.install("install_mutable_identity", model())
    assert changed.value.code == "MODEL_METADATA_INVALID"
    assert mutable.inventory() == {"models": []}


def test_install_is_idempotent_inventory_persists_and_preview_accounts_blobs(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    store = ModelStore(
        tmp_path,
        event_sink=lambda name, _data: events.append(name),
        hub=FakeHub(),
        runtime_available=True,
    )
    installed = store.install("op_install_one", model())
    duplicate = store.install("op_install_two", model(revision=SHA))

    assert duplicate["installationId"] == installed["installationId"]
    assert "model.install.progress" in events
    assert "model.inventory.changed" in events

    restarted = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    inventory = restarted.inventory()["models"]
    assert len(inventory) == 1
    assert inventory[0]["model"]["revision"] == SHA
    preview = restarted.remove_preview(installed["installationId"])
    assert preview["reclaimableBytes"] == 110
    assert preview["sharedBytesRetained"] == 0


def test_installed_exact_revision_can_be_inspected_offline(tmp_path: Path) -> None:
    hub = FakeHub()
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store.install("op_install_offline", model())

    def offline(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("installed exact model inspection must not contact the host")

    hub.model_info = offline  # type: ignore[method-assign]
    profile = store.inspect(model(revision=SHA))

    assert profile["model"]["revision"] == SHA
    assert profile["executable"] is True
    assert profile["runner"] == "transformers"
    assert profile["technicalCapabilities"]["task"] == "automatic-speech-recognition"


@pytest.mark.parametrize("available_memory", [1, 1_000_000])
def test_installed_inspection_preserves_eligibility_reasons_when_memory_changes(
    tmp_path: Path, available_memory: int
) -> None:
    current_memory = 1_000_000
    store = ModelStore(
        tmp_path,
        hub=FakeHub(),
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", current_memory),
    )
    store.install("op_install_projection", model())
    record = store._records[0]
    record.update(
        estimatedResidentBytes=1000,
        descriptorSchemaVersion="oryn-model-descriptor/1",
        capabilityRegistryDigest=store._capability_registry.digest,
        loadDisposition="trust-required",
        eligibility={
            "access": {"state": "available", "reasons": []},
            "artifacts": {"state": "safe", "reasons": []},
            "runtime": {"state": "supported", "reasons": []},
            "task": {"state": "supported", "reasons": []},
            "resources": {"state": "admissible", "reasons": []},
            "conformance": {"state": "untested", "reasons": ["CONFORMANCE_REQUIRED"]},
        },
    )

    current_memory = available_memory
    profile = store.inspect(model(revision=SHA))

    assert "CONFORMANCE_REQUIRED" in profile.get("blockingReasons", [])
    assert set(profile["blockingReasons"]) == {
        reason for axis in profile["eligibility"].values() for reason in axis["reasons"]
    }
    if available_memory < 1000:
        assert profile["loadDisposition"] == "not-admissible-on-this-device"
        assert profile["executable"] is False
        assert "MODEL_MEMORY_INSUFFICIENT" in profile["blockingReasons"]
    else:
        assert profile["loadDisposition"] == "trust-required"
        assert profile["executable"] is True
    assert record["eligibility"]["resources"]["state"] == "admissible"


def test_installed_builtin_runtime_pack_remains_executable_after_restart(
    tmp_path: Path,
) -> None:
    hub = FakeHub(
        config={"architectures": ["Qwen2ForCausalLM"]},
        files=[
            ("config.json", 10, "b" * 64),
            ("model.safetensors", 100, "c" * 64),
            ("tokenizer.json", 20, "d" * 64),
        ],
        sha=TINY_QWEN_SHA,
        pipeline_tag="text-generation",
    )
    exact_model = model("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", TINY_QWEN_SHA)
    ModelStore(tmp_path, hub=hub, runtime_available=True).install(
        "op_install_builtin_pack", exact_model
    )

    restarted = ModelStore(tmp_path, hub=hub, runtime_available=True)
    profile = restarted.inspect(exact_model)

    assert profile["executable"] is True
    assert profile["blockingReasons"] == ["CONFORMANCE_REQUIRED"]
    assert profile["runtimePack"]["packId"] == "oryn.transformers-pytorch"


def test_installed_model_runtime_override_is_engine_admitted_persisted_and_reversible(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from oryn_engine.runtime.hardware import HardwareInfo

    monkeypatch.setattr(
        ModelStore,
        "_hardware_info",
        lambda self: HardwareInfo(
            "darwin",
            "arm64",
            [{"id": "cpu", "deviceClass": "cpu"}, {"id": "mps", "deviceClass": "mps"}],
            "mps",
            True,
        ),
    )
    hub = FakeHub(
        config={"architectures": ["Qwen2ForCausalLM"]},
        files=[
            ("config.json", 10, "b" * 64),
            ("model.safetensors", 100, "c" * 64),
            ("tokenizer.json", 20, "d" * 64),
        ],
        sha=TINY_QWEN_SHA,
        pipeline_tag="text-generation",
    )
    exact_model = model("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", TINY_QWEN_SHA)
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store.install("op_install_override", exact_model)
    automatic = store.installed_record(exact_model)["runtimePlan"]
    assert automatic["fallbacks"]
    alternate = automatic["fallbacks"][0]["candidateId"]

    overridden = store.set_runtime_override(exact_model, alternate)
    assert overridden["selected"]["candidateId"] == alternate
    assert overridden["reasonCodes"][0] == "USER_OVERRIDE"
    assert (
        ModelStore(tmp_path, hub=hub, runtime_available=True).installed_record(exact_model)[
            "runtimePlan"
        ]["selected"]["candidateId"]
        == alternate
    )

    restored = store.set_runtime_override(exact_model, None)
    assert restored == automatic
    with pytest.raises(ModelStoreError) as invalid:
        store.set_runtime_override(exact_model, "candidate-" + "f" * 24)
    assert invalid.value.code == "RUNTIME_OVERRIDE_UNAVAILABLE"


def test_cancelled_install_removes_partial_and_never_enters_inventory(tmp_path: Path) -> None:
    hub = FakeHub(block_download=True)
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    failures: list[BaseException] = []

    def install() -> None:
        try:
            store.install("op_cancel_test", model())
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=install)
    worker.start()
    assert hub.download_started.wait(timeout=5)
    assert store.inventory()["models"][0]["installationState"] == "downloading"
    with pytest.raises(ModelStoreError) as duplicate:
        store.install("op_duplicate_install", model())
    assert duplicate.value.code == "MODEL_INSTALL_IN_PROGRESS"
    assert store.cancel_install("op_cancel_test") is True
    hub.release_download.set()
    worker.join(timeout=5)

    assert len(failures) == 1
    assert isinstance(failures[0], InstallCancelled)
    assert store.inventory() == {"models": []}
    assert not (tmp_path / "partial" / "op_cancel_test").exists()


def test_remove_preview_retains_blobs_referenced_by_another_installation(tmp_path: Path) -> None:
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    first = store.install("op_shared_one", model("org/first-model"))
    store.install("op_shared_two", model("org/second-model"))

    preview = store.remove_preview(first["installationId"])
    assert preview["reclaimableBytes"] == 0
    assert preview["sharedBytesRetained"] == 110

    first_path = tmp_path / "cache" / "snapshots" / SHA / "model.safetensors"
    assert first_path.exists()
    store.mark_removing("remove_shared_one", first["installationId"])
    removed = store.complete_remove(first["installationId"])
    assert removed["reclaimedBytes"] == 0
    assert removed["sharedBytesRetained"] == 110
    assert first_path.exists()
    assert len(store.inventory()["models"]) == 1

    second = store.inventory()["models"][0]
    store.mark_removing("remove_shared_two", second["installationId"])
    removed_second = store.complete_remove(second["installationId"])
    assert removed_second["reclaimedBytes"] == 110
    assert store.inventory() == {"models": []}
    assert not first_path.exists()

    redownloaded = store.install("op_redownload", model("org/first-model"))
    assert redownloaded["model"]["repository"] == "org/first-model"
    assert len(store.inventory()["models"]) == 1
    assert first_path.exists()


def test_install_downloads_only_declarative_assets_and_safe_weights(tmp_path: Path) -> None:
    hub = FakeHub(
        files=[
            ("config.json", 10, "b" * 64),
            ("tokenizer.json", 0, "1" * 64),
            ("preprocessor_config.json", 0, "2" * 64),
            ("chat_template.jinja", 5, "a" * 64),
            ("model.safetensors", 100, "c" * 64),
            ("pytorch_model.bin", 100, "d" * 64),
            ("onnx/model.onnx", 100, "e" * 64),
            ("README.md", 100, "f" * 64),
        ]
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)

    profile = store.inspect(model())
    store.install("op_safe_assets", model())

    assert profile["estimatedDiskBytes"] == 115
    assert profile["selectedFiles"] == [
        {"name": "config.json", "sizeBytes": 10},
        {"name": "tokenizer.json", "sizeBytes": 0},
        {"name": "preprocessor_config.json", "sizeBytes": 0},
        {"name": "chat_template.jinja", "sizeBytes": 5},
        {"name": "model.safetensors", "sizeBytes": 100},
    ]
    assert hub.downloaded == [
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "chat_template.jinja",
        "model.safetensors",
    ]


def test_diffusers_single_file_checkpoint_is_selected_instead_of_pickle_tree(
    tmp_path: Path,
) -> None:
    files = [
        ("model_index.json", 10, "1" * 64),
        ("scheduler/scheduler_config.json", 10, "2" * 64),
        ("text_encoder/config.json", 10, "3" * 64),
        ("text_encoder/model.bin", 4_000, "4" * 64),
        ("text_encoder_2/config.json", 10, "5" * 64),
        ("tokenizer/tokenizer_config.json", 10, "6" * 64),
        ("tokenizer/vocab.json", 10, "7" * 64),
        ("tokenizer_2/tokenizer_config.json", 10, "8" * 64),
        ("tokenizer_2/vocab.json", 10, "9" * 64),
        ("unet/config.json", 10, "a" * 64),
        ("unet/diffusion_pytorch_model.bin", 10_000, "b" * 64),
        ("vae/config.json", 10, "c" * 64),
        ("Juggernaut-XI.safetensors", 7_110, "d" * 64),
    ]
    hub = FakeHub(config={}, files=files, pipeline_tag="text-to-image")
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _pack: None if _pack.provider else object()
    )

    profile = store.inspect(model("RunDiffusion/Juggernaut-XI-v11"))

    assert profile["executable"] is True
    assert "blockingReasons" not in profile
    assert profile["runtimePack"]["packId"] == "oryn.diffusers-pytorch"
    selected = [item["name"] for item in profile["selectedFiles"]]
    assert "Juggernaut-XI.safetensors" in selected
    assert "unet/diffusion_pytorch_model.bin" not in selected
    assert "text_encoder/model.bin" not in selected

    store.install("op_safe_diffusers_checkpoint", model("RunDiffusion/Juggernaut-XI-v11"))
    assert "Juggernaut-XI.safetensors" in hub.downloaded
    assert not any(name.endswith(".bin") for name in hub.downloaded)


def test_audited_ace_step_manifest_selects_exact_data_subset_without_repository_code(
    tmp_path: Path,
) -> None:
    revision = "19671f406d603126926c1b7e2adc169acbcade22"
    required = [
        ("config.json", 1, "a" * 64),
        ("acestep-v15-turbo/config.json", 1, "b" * 64),
        ("acestep-v15-turbo/model.safetensors", 10, "c" * 64),
        ("acestep-v15-turbo/silence_latent.pt", 2, "d" * 64),
        ("vae/config.json", 1, "e" * 64),
        ("vae/diffusion_pytorch_model.safetensors", 5, "f" * 64),
        ("Qwen3-Embedding-0.6B/config.json", 1, "1" * 64),
        ("Qwen3-Embedding-0.6B/tokenizer.json", 2, "2" * 64),
        ("Qwen3-Embedding-0.6B/chat_template.jinja", 1, "3" * 64),
        ("Qwen3-Embedding-0.6B/merges.txt", 1, "4" * 64),
        ("Qwen3-Embedding-0.6B/model.safetensors", 4, "5" * 64),
    ]
    ignored = [
        ("acestep-v15-turbo/modeling_acestep_v15_turbo.py", 500, "6" * 64),
        ("acestep-v15-turbo/configuration_acestep_v15.py", 500, "7" * 64),
        ("acestep-5Hz-lm-1.7B/model.safetensors", 5_000, "8" * 64),
    ]
    hub = FakeHub(
        sha=revision,
        config={
            "architectures": ["AceStepConditionGenerationModel"],
            "auto_map": {"AutoModel": "modeling_acestep_v15_turbo.Model"},
        },
        pipeline_tag="text-to-audio",
        files=required + ignored,
    )
    store = ModelStore(
        tmp_path,
        hub=hub,
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 64_000_000_000),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _pack: None if _pack.provider else object()
    )

    profile = store.inspect(model("ACE-Step/Ace-Step1.5", revision))
    assert profile["executable"] is True
    assert profile["requiresCustomCode"] is True
    assert profile["runner"] == "runtime-pack"
    assert profile["runtimePack"]["compatibilityLevel"] == "audited-adapter"
    assert profile["estimatedDiskBytes"] == sum(item[1] for item in required)
    assert profile["selectedFiles"] == [
        {"name": name, "sizeBytes": size} for name, size, _digest in required
    ]

    store.install("install_ace_step", model("ACE-Step/Ace-Step1.5", revision))
    assert hub.downloaded == [item[0] for item in required]
    assert not any(name.endswith(".py") for name in hub.downloaded)
    assert not any(name.startswith("acestep-5Hz-lm") for name in hub.downloaded)


def test_registry_migration_adds_canonical_runtime_plan_without_changing_model_data(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    original = {
        "installationId": "install_existing",
        "model": {
            "provider": "huggingface",
            "repository": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "revision": TINY_QWEN_SHA,
        },
        "installationState": "installed",
        "residencyState": "loaded",
        "logicalSizeBytes": 123,
        "estimatedResidentBytes": 456,
        "runner": "runtime-pack",
        "executableBackends": ["mps", "cpu"],
        "technicalCapabilities": {"task": "text-generation"},
        "runtimePack": {
            "packId": "oryn.transformers-pytorch",
            "packVersion": "0.1.0",
            "runnerId": "transformers.chat",
            "taskContract": "chat-v1",
        },
        "installedAt": 1,
        "files": [{"name": "model.safetensors", "size": 123, "blob_id": "a" * 64}],
    }
    (model_root / "installations.json").write_text(
        json.dumps({"version": 1, "models": [original]}), encoding="utf-8"
    )

    store = ModelStore(model_root, hub=FakeHub(), runtime_available=True)
    migrated = store.inventory()["models"][0]
    persisted = json.loads((model_root / "installations.json").read_text(encoding="utf-8"))[
        "models"
    ][0]

    assert migrated["runtimePlan"]["selected"]["foundation"]["id"] == ("oryn.runtime.pytorch")
    assert migrated["runtimePlan"]["selected"]["adapter"]["id"] == ("oryn.adapter.transformers")
    assert persisted["model"] == original["model"]
    assert persisted["files"] == original["files"]
    assert persisted["installationId"] == original["installationId"]
    assert persisted["residencyState"] == "unloaded"


def test_registry_migration_recovers_generic_diffusers_plan_and_active_pack_metadata(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    revision = "9" * 40
    original = {
        "installationId": "install_existing_diffusers",
        "model": {
            "provider": "huggingface",
            "repository": "krea/Krea-2-Turbo",
            "revision": revision,
        },
        "installationState": "installed",
        "residencyState": "unloaded",
        "logicalSizeBytes": 35_000_000_000,
        "estimatedResidentBytes": 43_750_000_000,
        "runner": "runtime-pack",
        "executableBackends": ["mps", "cpu"],
        "technicalCapabilities": {
            "task": "text-to-image",
            "architectures": [],
            "inputModalities": ["text"],
            "outputModalities": ["image"],
        },
        "runtimePack": {
            "packId": "oryn.diffusers-pytorch",
            "packVersion": "0.1.0",
            "runnerId": "diffusers.pipeline",
            "taskContract": "text-to-image-v1",
            "compatibilityLevel": "native-declarative",
            "manifestDigest": "0" * 64,
        },
        "installedAt": 1,
        "files": [
            {"name": "model_index.json", "size": 1, "blob_id": "1" * 64},
            {"name": "text_encoder/config.json", "size": 1, "blob_id": "2" * 64},
            {
                "name": "text_encoder/model.safetensors",
                "size": 1,
                "blob_id": "3" * 64,
            },
            {"name": "tokenizer/tokenizer.json", "size": 1, "blob_id": "4" * 64},
        ],
    }
    (model_root / "installations.json").write_text(
        json.dumps({"version": 1, "models": [original]}), encoding="utf-8"
    )

    store = ModelStore(model_root, hub=FakeHub(), runtime_available=True)
    migrated = store.installed_record(original["model"])

    assert migrated["runtimePlan"]["selected"]["variantId"] == (
        "oryn.compat.generic-safe-diffusers-image"
    )
    assert migrated["runtimePlan"]["selected"]["adapter"]["id"] == ("oryn.adapter.diffusers")
    assert migrated["runtimePlan"]["selected"]["precision"] == "fp16"
    assert migrated["runtimePack"]["packVersion"] == "0.11.0"
    assert migrated["runtimePack"]["manifestDigest"] != "0" * 64
    assert migrated["generationProfile"]["profileId"] == "runtime.generic.fallback"
    assert migrated["generationProfile"]["recommended"] is False


def test_installed_diffusers_bitsandbytes_nf4_metadata_corrects_runtime_plan_and_restart(
    tmp_path: Path,
) -> None:
    class QuantizedDiffusersHub(FakeHub):
        def model_info(self, repo_id: str, revision: str, files_metadata: bool = True) -> Any:
            info = super().model_info(repo_id, revision, files_metadata)
            for item in info.siblings:
                data = self.metadata_bytes(item.rfilename)
                if data is not None:
                    item.size = len(data)
                    item.lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest())
            return info

        @staticmethod
        def metadata_bytes(filename: str) -> bytes | None:
            if filename == "model_index.json":
                return json.dumps({"_class_name": "Krea2Pipeline", "is_distilled": True}).encode()
            if filename.endswith("/config.json"):
                return json.dumps(
                    {
                        "quantization_config": {
                            "quant_method": "bitsandbytes",
                            "load_in_4bit": True,
                            "bnb_4bit_quant_type": "nf4",
                        }
                    }
                ).encode()
            return None

        def download(
            self,
            repo_id: str,
            filename: str,
            revision: str,
            cache_dir: str,
            progress: models_module.DownloadProgress | None = None,
        ) -> str:
            target = Path(super().download(repo_id, filename, revision, cache_dir, progress))
            if filename == "model_index.json":
                target.write_text(
                    json.dumps({"_class_name": "Krea2Pipeline", "is_distilled": True}),
                    encoding="utf-8",
                )
            if filename.endswith("/config.json"):
                target.write_text(
                    json.dumps(
                        {
                            "quantization_config": {
                                "quant_method": "bitsandbytes",
                                "load_in_4bit": True,
                                "bnb_4bit_quant_type": "nf4",
                            }
                        }
                    ),
                    encoding="utf-8",
                )
            return str(target)

    files = [
        ("model_index.json", 10, "1" * 64),
        ("text_encoder/config.json", 100, "2" * 64),
        ("text_encoder/model.safetensors", 100, "3" * 64),
        ("transformer/config.json", 100, "4" * 64),
        ("transformer/diffusion_pytorch_model.safetensors", 100, "5" * 64),
        ("tokenizer/tokenizer.json", 100, "8" * 64),
        ("vae/config.json", 100, "6" * 64),
        ("vae/diffusion_pytorch_model.safetensors", 100, "7" * 64),
    ]
    exact_model = model("ovedrive/Krea-2-Turbo-4bit", SHA)
    store = ModelStore(
        tmp_path,
        hub=QuantizedDiffusersHub(
            config={"architectures": []}, files=files, pipeline_tag="text-to-image"
        ),
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 64 * 1024**3),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _pack: None if _pack.provider else object()
    )
    store.install("install_bnb_nf4", exact_model)

    installed = store.installed_record(exact_model)
    assert installed["runtimePlan"]["selected"]["precision"] == "quantized"
    assert installed["runtimePlan"]["selected"]["quantization"] == "bitsandbytes-nf4"
    assert installed["runtimePack"]["packVersion"] == "0.11.0"
    assert installed["generationProfile"]["profileId"] == "diffusers.krea2.distilled"
    assert installed["generationProfile"]["parameters"]["steps"]["default"] == 8
    guidance = installed["generationProfile"]["parameters"]["guidanceScale"]
    assert guidance["default"] == 0.0
    assert guidance["supported"] is True
    assert guidance["editable"] is True
    assert "generationProfiles" not in installed["runtimePack"].get("trustedMetadata", {})

    registry = json.loads((tmp_path / "installations.json").read_text(encoding="utf-8"))
    stale = registry["models"][0]
    stale["runtimePlan"]["selected"]["precision"] = "fp16"
    stale["runtimePlan"]["selected"].pop("quantization")
    stale["runtimePack"]["packVersion"] = "0.1.0"
    (tmp_path / "installations.json").write_text(json.dumps(registry), encoding="utf-8")

    restarted = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    migrated = restarted.installed_record(exact_model)
    assert migrated["runtimePlan"]["selected"]["precision"] == "quantized"
    assert migrated["runtimePlan"]["selected"]["quantization"] == "bitsandbytes-nf4"
    assert migrated["runtimePack"]["packVersion"] == "0.11.0"
    assert migrated["generationProfile"]["profileId"] == "diffusers.krea2.distilled"


def test_nonchat_safetensors_are_not_shadowed_by_an_optional_gguf_export(tmp_path):
    hub = FakeHub(
        config={"architectures": ["BertModel"], "model_type": "bert"},
        pipeline_tag="feature-extraction",
        files=[
            ("config.json", 10, "b" * 64),
            ("model.safetensors", 100, "c" * 64),
            ("tokenizer.json", 10, "d" * 64),
            ("export/model.gguf", 100, "e" * 64),
        ],
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    result = store.inspect(model("org/encoder-with-export"))
    assert result["executable"] is True
    assert result["runner"] == "transformers"
    assert all(not item["name"].endswith(".gguf") for item in result["selectedFiles"])


@pytest.mark.parametrize(
    "projectors,expected",
    [
        (["mmproj-BF16.gguf", "mmproj-F16.gguf"], "mmproj-F16.gguf"),
        (["mmproj-BF16.gguf"], "mmproj-BF16.gguf"),
        (["mmproj-F16.gguf", "Q4_K_M/mmproj-Model-F16.gguf"], "Q4_K_M/mmproj-Model-F16.gguf"),
        (["other/mmproj-F16.gguf"], None),
    ],
)
def test_folder_variants_find_their_projector_next_to_them_or_at_the_root(
    tmp_path: Path, projectors, expected
) -> None:
    hub = FakeHub(
        config={"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"},
        files=[
            ("Q4_K_M/Model-Q4_K_M-00001-of-00002.gguf", 8_000_000_000, "b" * 64),
            ("Q4_K_M/Model-Q4_K_M-00002-of-00002.gguf", 8_000_000_000, "c" * 64),
            *[(name, 900_000_000, "d" * 64) for name in projectors],
        ],
        pipeline_tag="image-text-to-text",
        gguf={"architecture": "qwen35", "context_length": 262_144},
    )
    store = ModelStore(
        tmp_path,
        hub=hub,
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 64_000_000_000),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _pack: None if _pack.provider else object()
    )
    profile = store.inspect(model("test/folder-gguf"))
    trusted = profile["runtimePack"]["trustedMetadata"]
    assert trusted["modelFile"] == "Q4_K_M/Model-Q4_K_M-00001-of-00002.gguf"
    names = {item["name"] for item in profile["selectedFiles"]}
    assert {
        "Q4_K_M/Model-Q4_K_M-00001-of-00002.gguf",
        "Q4_K_M/Model-Q4_K_M-00002-of-00002.gguf",
    } <= names
    if expected is None:
        assert "projectorFile" not in trusted
        assert "GGUF_PROJECTOR_MISSING_OR_AMBIGUOUS" in profile["blockingReasons"]
    else:
        from oryn_protocol import ProtocolSchemaStore

        assert trusted["projectorFile"] == expected and expected in names
        ProtocolSchemaStore().validator("candidateExecutionProfile").validate(profile)
