"""MLX-Audio joins the existing metadata and task-negotiation boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from oryn_protocol.schema import ProtocolSchemaStore
from oryn_protocol.task_contracts import effective_input, validate_invocation
from test_models import FakeHub

from oryn_engine.capability_sources import framework_registry
from oryn_engine.model_inspection import inspect_descriptor
from oryn_engine.model_layouts import assess_layout
from oryn_engine.models import ModelStore
from oryn_engine.runtime.hardware import HardwareInfo
from oryn_engine.runtime_packs.execution import PackRuntime
from oryn_engine.runtime_routing import generated_pack_id
from oryn_engine.task_profiles import profile_for_execution

SHA = "a" * 40
FISH_FILES = [
    "config.json",
    "model-00001.safetensors",
    "model-00002.safetensors",
    "codec.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]


def inspect(
    tmp_path,
    monkeypatch,
    *,
    family="fish_drama3",
    task="text-to-speech",
    files=None,
    extra=None,
    metadata=None,
    runtime=True,
    unified_memory=True,
):
    config = {"model_type": family, "architectures": ["FishOmniModel"], **(extra or {})}
    names = list(FISH_FILES if files is None else files)

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            assert revision == SHA
            return config if filename == "config.json" else (metadata or {}).get(filename, {})

    hub = Hub(config=config, pipeline_tag=task, files=[(name, 100, "b" * 64) for name in names])
    store = ModelStore(
        tmp_path,
        hub=hub,
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 128 * 1024**3),
    )
    monkeypatch.setattr(
        store,
        "_hardware_info",
        lambda: HardwareInfo(
            "darwin", "arm64", [{"id": "pytorch-mps", "deviceClass": "mps"}], "mps", unified_memory
        ),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None) if runtime else None
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "arbitrary/publisher-name", "revision": "main"}
    )
    assert hub.downloaded == []
    return result


@pytest.mark.parametrize(
    "family,task,contract",
    [
        ("fish_drama3", "text-to-speech", "speech-synthesis-v1"),
        ("soprano", "text-to-speech", "speech-synthesis-v1"),
        ("qwen3_asr", "automatic-speech-recognition", "transcription-v1"),
    ],
)
def test_live_discovery_routes_multiple_declared_families_without_repository_aliases(
    tmp_path, monkeypatch, family, task, contract
):
    result = inspect(tmp_path, monkeypatch, family=family, task=task)
    assert result["executable"], result
    assert result["runtimePack"]["packId"] == "oryn.mlx-audio"
    assert generated_pack_id(result["runtimePlan"]["selected"]["variantId"]) == "oryn.mlx-audio"
    assert result["taskProfile"]["contract"]["id"] == contract
    assert result["exactRevision"] == SHA
    ProtocolSchemaStore().validator("candidateExecutionProfile").validate(result)


def test_live_discovery_infers_audio_task_from_declared_family_without_hub_label(
    tmp_path, monkeypatch
):
    result = inspect(tmp_path, monkeypatch, task=None)
    assert result["executable"], result
    assert result["runtimePack"]["packId"] == "oryn.mlx-audio"
    assert result["taskProfile"]["contract"]["id"] == "speech-synthesis-v1"


def test_qwen3_tts_mlx_admission_accepts_sharded_speech_tokenizer(tmp_path, monkeypatch):
    files = [
        "config.json",
        "model-00001.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "speech_tokenizer/config.json",
        "speech_tokenizer/weights-00001.safetensors",
        "speech_tokenizer/weights-00002.safetensors",
    ]
    result = inspect(
        tmp_path,
        monkeypatch,
        family="qwen3_tts",
        files=files,
        extra={"architectures": ["Qwen3TTSForConditionalGeneration"], "tts_model_type": "base"},
        metadata={"speech_tokenizer/config.json": {"decode_upsample_rate": 1920}},
    )
    assert result["executable"], result
    assert result["runtimePack"]["packId"] == "oryn.mlx-audio"
    assert {item["name"] for item in result["selectedFiles"]} == set(files)


def test_speech_reference_transcript_is_negotiated_and_legacy_controls_stay_fixed(
    tmp_path, monkeypatch
):
    profile = inspect(tmp_path, monkeypatch)["taskProfile"]
    assert profile["features"]["referenceAudio"] == "supported"
    value = effective_input(profile, {"text": "Hello."})
    assert value == {"text": "Hello.", "voice": "default", "language": "auto", "speed": 1}
    validate_invocation(profile, value)
    reference = value | {
        "voice": "reference",
        "audioArtifactId": "artifact_" + "a" * 24,
        "referenceText": "Words in the reference recording.",
    }
    validate_invocation(profile, reference)
    for change in (
        {"voice": "unknown"},
        {"speed": 1.1},
        {"language": "invented"},
        {"referenceText": "Transcript without reference audio."},
    ):
        with pytest.raises(ValueError):
            validate_invocation(profile, value | change)


@pytest.mark.parametrize("damage", ["codec", "repository-code", "runtime", "hardware"])
def test_live_admission_retains_missing_data_runtime_and_hardware_failures(
    tmp_path, monkeypatch, damage
):
    result = inspect(
        tmp_path,
        monkeypatch,
        files=[n for n in FISH_FILES if n != "codec.safetensors"] if damage == "codec" else None,
        extra={"auto_map": {"AutoModel": "modeling_custom.Model"}}
        if damage == "repository-code"
        else None,
        runtime=damage != "runtime",
        unified_memory=damage != "hardware",
    )
    assert not result["executable"]
    assert result["blockingReasons"]


def test_layout_selects_nested_safe_data_and_excludes_code_and_pickle():
    config = {"model_type": "fish_drama3"}
    extra = [
        "speech_tokenizer/config.json",
        "speech_tokenizer/model.safetensors",
        "training.py",
        "codec.pth",
        "unsafe.pkl",
        "README.md",
    ]
    layout = assess_layout([*FISH_FILES, *extra], config, "text-to-speech", source_kind="mlx-audio")
    assert not layout.artifact_reasons
    assert set(extra[:2]) <= set(layout.selected_files)
    assert not set(extra[2:]) & set(layout.selected_files)
    assert assess_layout(
        [*FISH_FILES, "../escape.json"], config, "text-to-speech", source_kind="mlx-audio"
    ).artifact_reasons


def test_shared_descriptor_enforces_the_same_model_and_file_admission():
    args = dict(
        repository="org/arbitrary",
        revision=SHA,
        files=FISH_FILES,
        config={"model_type": "fish_drama3"},
        task="text-to-speech",
        registry=framework_registry(),
        access="available",
        source_kind="mlx-audio",
        estimated_memory=1024,
        available_memory=128 * 1024**3,
    )
    accepted = inspect_descriptor(**args)
    assert accepted.eligibility.disposition == "trust-required"
    for change in (
        {"config": {"model_type": "unknown"}},
        {"config": {"model_type": "fish_drama3", "auto_map": {"AutoModel": "custom.Model"}}},
        {"files": [n for n in FISH_FILES if n != "codec.safetensors"]},
    ):
        assert inspect_descriptor(**(args | change)).eligibility.disposition != "trust-required"


def test_native_literal_voice_default_and_transcript_support_remain_family_specific(monkeypatch):
    from oryn_engine import mlx_audio_tasks

    monkeypatch.setattr(
        mlx_audio_tasks,
        "capability",
        lambda *_: {
            "family": "example",
            "category": "tts",
            "parameters": ["text", "voice"],
            "defaults": {"voice": "native-voice"},
        },
    )
    profile = profile_for_execution(
        {
            "technicalCapabilities": {"task": "text-to-speech"},
            "runtimePack": {"packId": "oryn.mlx-audio", "taskContract": "speech-synthesis-v1"},
        }
    )
    assert effective_input(profile, {"text": "Hello."})["voice"] == "native-voice"
    assert profile["parameters"]["voice"]["handling"] == "fixed"
    assert "referenceText" not in profile["parameters"]
    assert profile["features"]["referenceAudio"] == "unsupported"


@pytest.mark.parametrize(
    "family,required", [("kitten_tts", "voices.npz"), ("soprano", "tokenizer.json")]
)
def test_reviewed_companion_files_are_required_by_shared_layout(family, required):
    from oryn_engine import mlx_audio_tasks

    config = {"model_type": family}
    files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "voices.npz",
    ]
    selected = mlx_audio_tasks.selected_files(files, config, "text-to-speech")
    assert required in selected
    with pytest.raises(ValueError, match="companion"):
        mlx_audio_tasks.selected_files(
            [name for name in files if name != required], config, "text-to-speech"
        )
