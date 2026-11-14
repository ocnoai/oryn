from __future__ import annotations

import copy
import json
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from oryn_engine.runtime_packs.builtin_catalog import (
    DEVELOPMENT_PUBLISHER,
    PACK,
    SPRINT_30_DEVELOPMENT_PUBLISHER,
    SPRINT_31_DEVELOPMENT_PUBLISHER,
    SPRINT_32_DEVELOPMENT_PUBLISHER,
    builtin_registry,
)
from oryn_engine.runtime_packs.builtin_foundations import builtin_foundation_registry
from oryn_engine.runtime_packs.diffusers_worker import (
    _pipeline,
    _supported_options,
    _validate_model,
)
from oryn_engine.runtime_packs.execution import (
    PackRuntime,
    PackRuntimeLocator,
    StreamingPackProcess,
)
from oryn_engine.runtime_packs.intake import ModelIntakeHarness
from oryn_engine.runtime_packs.manifests import ManifestError, SignedManifestVerifier
from oryn_engine.runtime_packs.supervisor import PackProcessError, PackProcessSupervisor


def test_builtin_pack_and_compatibility_manifests_are_signature_and_digest_verified() -> None:
    registry = builtin_registry()
    assert [(pack.pack_id, pack.version) for pack in registry.packs] == [
        ("oryn.ace-step-mlx", "0.1.0"),
        ("oryn.diffusers-pytorch", "0.10.0"),
        ("oryn.kokoro-onnx", "0.3.0"),
        ("oryn.llama-cpp", "0.4.0"),
        ("oryn.mflux-mlx", "0.7.0"),
        ("oryn.mlx-lm", "0.10.0"),
        ("oryn.onnx-runtime", "0.5.0"),
        ("oryn.qwen-tts-pytorch", "0.1.0"),
        ("oryn.transformers-pytorch", "0.32.0"),
    ]
    assert {item.manifest_id for item in registry.compatibility} == {
        "oryn.compat.ace-step-1.5",
        "oryn.compat.tiny-qwen2",
        "oryn.compat.tiny-whisper",
        "oryn.compat.tiny-musicgen",
        "oryn.compat.qwen3-embedding-0.6b",
        "oryn.compat.qwen3-reranker-0.6b",
        "oryn.compat.kokoro-82m-v1-onnx-q8",
        "oryn.compat.segmind-vega-fp16",
        "oryn.compat.generic-safe-diffusers-image",
        "oryn.compat.tiny-random-sdxl-safetensors",
        "oryn.compat.tiny-random-wan2.1-t2v",
        "oryn.compat.tiny-gemma3-gguf-q8-0",
        "oryn.compat.qwen2.5-0.5b-instruct-gguf-q4-k-m",
        "oryn.compat.krea-2-turbo-mflux-q4",
    }

    tampered = copy.deepcopy(PACK)
    tampered["payload"]["version"] = "0.1.1"
    verifier = SignedManifestVerifier(
        {
            DEVELOPMENT_PUBLISHER.key_id: DEVELOPMENT_PUBLISHER,
            SPRINT_30_DEVELOPMENT_PUBLISHER.key_id: SPRINT_30_DEVELOPMENT_PUBLISHER,
            SPRINT_31_DEVELOPMENT_PUBLISHER.key_id: SPRINT_31_DEVELOPMENT_PUBLISHER,
            SPRINT_32_DEVELOPMENT_PUBLISHER.key_id: SPRINT_32_DEVELOPMENT_PUBLISHER,
        }
    )
    with pytest.raises(ManifestError) as invalid:
        verifier.verify(tampered)
    assert invalid.value.code == "PACK_DIGEST_INVALID"


def test_shared_transformers_worker_accepts_the_active_signed_pack_version() -> None:
    pack = builtin_registry().active_pack("oryn.transformers-pytorch")
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-m", pack.entrypoint],
        input=json.dumps(
            {
                "abi": pack.abi,
                "packId": pack.pack_id,
                "version": pack.version,
                "command": "probe",
                "payload": {},
            }
        )
        + "\n",
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    response = json.loads(result.stdout)
    assert response["ok"] is True
    expected = next(
        r.task_contracts for r in pack.runners if r.runner_id == "transformers.shared-tasks"
    )
    assert set(response["result"]["taskContracts"]) == set(expected)


def test_dependency_incompatible_packs_run_in_distinct_supervised_processes() -> None:
    pack = replace(
        builtin_registry().active_pack("oryn.transformers-pytorch"),
        entrypoint="oryn_engine.runtime_packs.fixture_worker",
    )
    alpha = replace(pack, pack_id="oryn.fixture-alpha", dependency_closure="fixture-a==1")
    beta = replace(pack, pack_id="oryn.fixture-beta", dependency_closure="fixture-a==2")
    supervisor = PackProcessSupervisor()

    first = supervisor.invoke(alpha, "conformance", {"value": "alpha"})
    second = supervisor.invoke(beta, "conformance", {"value": "beta"})

    assert first["pid"] != second["pid"]
    assert first["dependencyClosure"] == "fixture-a==1"
    assert second["dependencyClosure"] == "fixture-a==2"
    assert first["networkPolicy"] == second["networkPolicy"] == "deny"


def test_pack_crash_is_contained_and_a_clean_process_still_runs() -> None:
    pack = replace(
        builtin_registry().active_pack("oryn.transformers-pytorch"),
        entrypoint="oryn_engine.runtime_packs.fixture_worker",
    )
    supervisor = PackProcessSupervisor()
    with pytest.raises(PackProcessError) as crashed:
        supervisor.invoke(pack, "crash")
    assert crashed.value.code == "PACK_CRASHED"
    assert supervisor.invoke(pack, "probe")["packId"] == pack.pack_id


def test_signed_catalog_supports_capability_matching_and_static_intake() -> None:
    registry = builtin_registry()
    assert registry.catalog(task="text-generation", output_modalities=["text"]) == (
        {
            "provider": "huggingface",
            "repository": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "revision": "f56c2fbf7fabc03ffff3494d3c2f8d06c774493d",
        },
        {
            "provider": "huggingface",
            "repository": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "revision": "9217f5db79a29953eb74d5343926648285ec7e67",
        },
    )
    assert registry.catalog(task="text-to-audio", output_modalities=["audio"]) == (
        {
            "provider": "huggingface",
            "repository": "ACE-Step/Ace-Step1.5",
            "revision": "19671f406d603126926c1b7e2adc169acbcade22",
        },
    )
    assert (
        registry.catalog(task="feature-extraction", output_modalities=["embedding"])[0][
            "repository"
        ]
        == "Qwen/Qwen3-Embedding-0.6B"
    )
    assert (
        registry.catalog(task="text-ranking", output_modalities=["structured"])[0]["repository"]
        == "Qwen/Qwen3-Reranker-0.6B"
    )
    speech = registry.catalog(task="text-to-speech", output_modalities=["audio"])
    assert speech[0]["repository"] == "onnx-community/Kokoro-82M-v1.0-ONNX"
    kokoro = next(
        item
        for item in registry.compatibility
        if item.manifest_id == "oryn.compat.kokoro-82m-v1-onnx-q8"
    )
    assert kokoro.trusted_metadata["license"] == "Apache-2.0"
    assert {voice["language"] for voice in kokoro.trusted_metadata["voices"]} == {
        "en-us",
        "en-gb",
    }
    assert registry.catalog(task="text-to-image", output_modalities=["image"]) == (
        {
            "provider": "huggingface",
            "repository": "mflux-community/krea-2-turbo-mflux-q4",
            "revision": "f30071b91ba672767e49e83276fe44cc29bc6a8e",
        },
        {
            "provider": "huggingface",
            "repository": "segmind/Segmind-Vega",
            "revision": "7714c4363e5856ff974a4f4b068e8691f26d0b40",
        },
    )
    assert registry.catalog(task="text-to-video", output_modalities=["video"]) == ()
    result = ModelIntakeHarness(registry).inspect(
        {
            "repository": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "revision": "f56c2fbf7fabc03ffff3494d3c2f8d06c774493d",
            "architectures": ["Qwen2ForCausalLM"],
            "task": "text-generation",
            "files": ["config.json", "model.safetensors"],
            "requiresCustomCode": False,
        }
    )
    assert result.state == "static-approved"
    assert result.matches == ("oryn.compat.tiny-qwen2",)
    assert result.blocking_reasons == ()


def test_repository_custom_code_never_becomes_pack_authority() -> None:
    result = ModelIntakeHarness(builtin_registry()).inspect(
        {
            "repository": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "revision": "f56c2fbf7fabc03ffff3494d3c2f8d06c774493d",
            "architectures": ["Qwen2ForCausalLM"],
            "task": "text-generation",
            "files": ["modeling_custom.py", "model.safetensors"],
            "requiresCustomCode": True,
            "entrypoint": "modeling_custom.CustomModel",
        }
    )
    assert result.state == "static-rejected"
    assert "CUSTOM_CODE_REQUIRED" in result.blocking_reasons


def test_ace_step_worker_probe_uses_the_typed_streaming_abi() -> None:
    pack = builtin_registry().active_pack("oryn.ace-step-mlx")
    source_root = Path(__file__).parents[1] / "src"
    result = StreamingPackProcess().invoke(
        pack,
        PackRuntime(Path(sys.executable).resolve(), source_root.resolve()),
        "probe",
        {},
        cancellation=threading.Event(),
        event_sink=lambda _event, _data: None,
        timeout_seconds=10,
    )
    assert result == {"packId": "oryn.ace-step-mlx", "abi": "1", "networkPolicy": "deny"}


def test_ace_step_adapter_never_enables_hugging_face_remote_code() -> None:
    worker = (
        Path(__file__).parents[1] / "src/oryn_engine/runtime_packs/ace_step_worker.py"
    ).read_text()
    assert "trust_remote_code" not in worker
    assert "AutoModel.from_pretrained" not in worker
    assert "modeling_acestep_v15_turbo" in worker


def test_diffusers_adapter_is_data_only_and_disables_remote_code() -> None:
    worker = (
        Path(__file__).parents[1] / "src/oryn_engine/runtime_packs/diffusers_worker.py"
    ).read_text()
    assert "trust_remote_code=False" in worker
    assert "local_files_only=True" in worker
    assert "use_safetensors=True" in worker
    assert "custom_pipeline" not in worker
    assert "subprocess" not in worker


def test_diffusers_task_contract_adapts_to_pipeline_signatures_without_model_branches() -> None:
    class MinimalPipeline:
        def __call__(self, prompt: str, width: int) -> None:
            del prompt, width

    options = {"prompt": "hello", "width": 512, "guidance_scale": 7.5}
    assert _supported_options(MinimalPipeline(), options, {"prompt"}) == {
        "prompt": "hello",
        "width": 512,
    }
    with pytest.raises(ValueError, match="task contract"):
        _supported_options(MinimalPipeline(), options, {"image"})


def test_generation_profile_resolver_contains_no_model_family_branches() -> None:
    resolver = (
        Path(__file__).parents[1] / "src/oryn_engine/runtime_packs/generation_profiles.py"
    ).read_text()
    assert "Krea" not in resolver
    assert "Flux" not in resolver
    assert "StableDiffusion" not in resolver


def test_mflux_adapter_is_data_only_and_repository_identity_never_selects_code() -> None:
    worker = (
        Path(__file__).parents[1] / "src/oryn_engine/runtime_packs/mflux_image_worker.py"
    ).read_text()
    assert "trust_remote_code" not in worker
    assert "snapshot_download" not in worker
    assert "mflux-community" not in worker
    assert "krea/Krea" not in worker
    assert "subprocess" not in worker
    assert "oryn_mflux_provider.worker" in worker
    assert "SEALED_ARCHITECTURES" not in worker
    pack = builtin_registry().active_pack("oryn.mflux-mlx")
    assert pack.entrypoint == "oryn_mflux_provider.worker"
    assert pack.provider["abi"] == "1"


def test_mflux_compatibility_pins_q4_provenance_and_shared_mlx_adapter() -> None:
    registry = builtin_registry()
    compatibility = next(
        item
        for item in registry.compatibility
        if item.manifest_id == "oryn.compat.krea-2-turbo-mflux-q4"
    )
    assert compatibility.repository == "mflux-community/krea-2-turbo-mflux-q4"
    assert compatibility.revision == "f30071b91ba672767e49e83276fe44cc29bc6a8e"
    assert compatibility.trusted_metadata["quantization"] == "Q4"
    assert compatibility.trusted_metadata["adapterArchitecture"] == "krea2-mflux-v1"
    assert compatibility.trusted_metadata["conformance"]["status"] == "passed"
    assert compatibility.trusted_metadata["inferenceDefaults"] == {
        "width": 512,
        "height": 512,
        "steps": 8,
        "guidanceScale": 1.0,
    }
    assert compatibility.trusted_metadata["source"] == {
        "provider": "huggingface",
        "repository": "krea/Krea-2-Turbo",
        "revision": "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7",
    }
    foundation_registry = builtin_foundation_registry()
    adapter = foundation_registry.adapter_for_legacy_pack("oryn.mflux-mlx")
    ace_adapter = foundation_registry.adapter_for_legacy_pack("oryn.ace-step-mlx")
    assert adapter is not None and ace_adapter is not None
    assert adapter.foundation_id == ace_adapter.foundation_id == "oryn.runtime.mlx"


def test_safe_declarative_text_to_image_models_can_use_the_bundled_diffusers_pack() -> None:
    matches = builtin_registry().match(
        repository="krea/Krea-2-Turbo",
        revision="98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7",
        architectures=[],
        task="text-to-image",
    )

    assert matches[0].manifest_id == "oryn.compat.generic-safe-diffusers-image"
    assert matches[0].trusted_metadata["conformance"]["status"] == "fixture-only"
    assert matches[0].required_files == (
        "model_index.json",
        "*/*.json",
        "*.safetensors",
        "tokenizer*/*",
    )


def test_llama_cpp_adapter_executes_only_the_packaged_native_binary() -> None:
    worker = (
        Path(__file__).parents[1] / "src/oryn_engine/runtime_packs/llama_cpp_worker.py"
    ).read_text()
    assert 'root / "native" / "llama-completion"' in worker
    assert "shell=True" not in worker
    assert "trust_remote_code" not in worker
    assert "--model" in worker
    assert "--file" in worker
    assert "stderr=subprocess.DEVNULL" in worker


def test_diffusers_accepts_standard_scalar_and_disabled_component_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "StableDiffusionXLPipeline",
                "force_zeros_for_empty_prompt": True,
                "image_encoder": [None, None],
                "unet": ["diffusers", "UNet2DConditionModel"],
            }
        ),
        encoding="utf-8",
    )

    _validate_model(tmp_path)


def test_diffusers_accepts_bounded_declarative_array_metadata(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Krea2Pipeline",
                "text_encoder_select_layers": [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
                "text_encoder": ["transformers", "Qwen3VLModel"],
            }
        ),
        encoding="utf-8",
    )

    _validate_model(tmp_path)


def test_streaming_pack_accepts_only_closed_safe_worker_error_codes() -> None:
    error_code = StreamingPackProcess._worker_error_code

    assert error_code({"code": "MODEL_MEMORY_INSUFFICIENT"}) == "MODEL_MEMORY_INSUFFICIENT"
    assert error_code({"code": "MODEL_RUNTIME_INCOMPATIBLE"}) == "MODEL_RUNTIME_INCOMPATIBLE"
    assert error_code({"code": "ARBITRARY_WORKER_CODE"}) == "PACK_EXECUTION_FAILED"
    assert error_code({"code": "MODEL_MEMORY_INSUFFICIENT", "path": "/private"}) == (
        "PACK_EXECUTION_FAILED"
    )


def test_diffusers_loads_a_safe_root_checkpoint_without_repository_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline"}), encoding="utf-8"
    )
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"safe checkpoint")
    calls: list[tuple[str, dict[str, object]]] = []

    class FakePipeline:
        @classmethod
        def from_single_file(cls, path: str, **kwargs: object) -> FakePipeline:
            calls.append((path, kwargs))
            return cls()

        def to(self, _backend: str) -> FakePipeline:
            return self

        def enable_attention_slicing(self) -> None:
            pass

        def enable_vae_slicing(self) -> None:
            pass

    fake_diffusers = SimpleNamespace(
        StableDiffusionXLPipeline=FakePipeline,
        AutoPipelineForImage2Image=FakePipeline,
        AutoPipelineForText2Image=FakePipeline,
        DiffusionPipeline=FakePipeline,
    )
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        float16="float16",
        float32="float32",
    )

    _pipe, backend = _pipeline(tmp_path, "text-to-image-v1", torch)

    assert backend == "cpu"
    assert calls == [
        (
            str(checkpoint),
            {
                "config": str(tmp_path),
                "local_files_only": True,
                "trust_remote_code": False,
                "use_safetensors": True,
                "torch_dtype": "float32",
            },
        )
    ]


def test_runtime_pack_marker_must_match_the_signed_pack_identity(tmp_path: Path) -> None:
    pack = builtin_registry().active_pack("oryn.diffusers-pytorch")
    python = Path(sys.executable).resolve()
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    marker = site_packages / "oryn-runtime-pack.json"
    marker.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "packId": pack.pack_id,
                "version": pack.version,
                "dependencyClosure": pack.dependency_closure,
                "inheritBaseRuntime": True,
            }
        ),
        encoding="utf-8",
    )

    runtime = PackRuntimeLocator._validated(pack, python, site_packages.resolve())
    assert runtime is not None and runtime.inherit_base_runtime

    marker.write_text(
        marker.read_text(encoding="utf-8").replace(pack.version, "9.9.9"),
        encoding="utf-8",
    )
    assert PackRuntimeLocator._validated(pack, python, site_packages.resolve()) is None


def test_mlx_adapters_resolve_one_independently_verified_foundation(tmp_path: Path) -> None:
    foundation_registry = builtin_foundation_registry()
    foundation = foundation_registry.active_foundation("oryn.runtime.mlx")
    python = Path(sys.executable).resolve()
    foundation_site = tmp_path / "runtime-foundations/oryn.runtime.mlx/site-packages"
    foundation_site.mkdir(parents=True)
    (foundation_site / "oryn-runtime-foundation.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "foundationId": foundation.foundation_id,
                "version": foundation.version,
                "dependencyClosure": foundation.dependency_closure,
            }
        ),
        encoding="utf-8",
    )
    resolved_foundations = set()
    for pack_id in ("oryn.ace-step-mlx", "oryn.mflux-mlx"):
        pack = builtin_registry().active_pack(pack_id)
        site = tmp_path / f"runtime-packs/{pack_id}/site-packages"
        site.mkdir(parents=True)
        (site / "oryn-runtime-pack.json").write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "packId": pack.pack_id,
                    "version": pack.version,
                    "dependencyClosure": pack.dependency_closure,
                    "foundation": {
                        "id": foundation.foundation_id,
                        "version": foundation.version,
                        "dependencyClosure": foundation.dependency_closure,
                    },
                }
            ),
            encoding="utf-8",
        )
        runtime = PackRuntimeLocator._validated(
            pack, python, site.resolve(), foundation, foundation_site.resolve()
        )
        assert runtime is not None
        resolved_foundations.add(runtime.foundation_site_packages)
    assert resolved_foundations == {foundation_site.resolve()}


def test_kokoro_admits_cpu_without_claiming_coreml_compiler_access():
    from oryn_engine.runtime_packs.builtin_foundations import builtin_foundation_registry

    pack = builtin_registry().active_pack("oryn.kokoro-onnx")
    assert pack.runner("kokoro.speech-synthesis").backends == ("cpu",)
    adapter = builtin_foundation_registry().adapter_for_legacy_pack(pack.pack_id)
    assert adapter.backends == ("cpu",)
