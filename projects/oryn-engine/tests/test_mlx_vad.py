import copy
import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from oryn_engine.mlx_vad import CONFIG, SHAPES, load_weights, validate_config
from oryn_engine.model_layouts import assess_layout


@pytest.mark.parametrize(
    "change",
    [
        {"model_type": "silero_vad_v6"},
        {"sample_rate": 8000},
        {"lstm_num_layers": True},
        {"encoder_strides": [True, 2, 2, 1]},
        {"auto_map": {"AutoModel": "custom.Model"}},
        {"quantization": {"bits": 4}},
        {"extra": None},
    ],
)
def test_silero_vad_rejects_unimplemented_graphs_and_dispatch(change):
    with pytest.raises(ValueError, match="CONFIGURATION"):
        validate_config(CONFIG | change)


def test_silero_layout_needs_no_tokenizer_and_selects_only_native_data():
    names = ["config.json", "model.safetensors", "model.py", "model.pt", "tokenizer.json"]
    result = assess_layout(names, CONFIG, "voice-activity-detection", source_kind="mlx")
    assert result.selected_files == ("config.json", "model.safetensors")
    assert not result.artifact_reasons and not result.runtime_reasons
    # Neither the audio exception nor its graph can satisfy a chat task.
    chat = assess_layout(names[:2], CONFIG, "text-generation", source_kind="mlx")
    assert "MODEL_TOKENIZER_MISSING" in chat.artifact_reasons
    bad = assess_layout(
        names, CONFIG | {"chunk_size": 1024}, "voice-activity-detection", source_kind="mlx"
    )
    assert "TASK_CONFIGURATION_UNSUPPORTED" in bad.runtime_reasons


def test_silero_weight_graph_rejects_missing_extra_quantized_and_nonfinite_tensors(tmp_path):
    pytest.importorskip("mlx.core")
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    weights = {key: np.zeros(shape, dtype=np.float32) for key, shape in SHAPES.items()}
    path = tmp_path / "model.safetensors"
    save_file(weights, path)
    assert set(load_weights(tmp_path)) == set(weights)
    changes = []
    missing = weights.copy()
    del missing["lstm.bias"]
    changes.append(missing)
    changes.append(weights | {"extra": np.zeros((1,), dtype=np.float32)})
    changes.append(weights | {"lstm.bias": np.zeros((512,), dtype=np.float16)})
    changes.append(weights | {"lstm.bias": np.full((512,), np.nan, dtype=np.float32)})
    changes.append(weights | {"lstm.bias": np.zeros((128,), dtype=np.float32)})
    for changed in changes:
        save_file(changed, path)
        with pytest.raises(ValueError, match="WEIGHTS"):
            load_weights(tmp_path)


def test_silero_capability_is_audio_only_and_requires_exact_configuration():
    from oryn_engine.compatibility_plane import compile_registry
    from oryn_engine.generated_capabilities import generated_sources
    from oryn_engine.model_inspection import inspect_descriptor

    args = dict(
        repository="test/native-vad",
        revision="a" * 40,
        files=["config.json", "model.safetensors"],
        config=copy.deepcopy(CONFIG),
        task="voice-activity-detection",
        registry=compile_registry(generated_sources()),
        access="available",
        source_kind="mlx",
        estimated_memory=1024,
        available_memory=1024**3,
    )
    assessment = inspect_descriptor(**args)
    assert assessment.eligibility.disposition == "trust-required"
    for changes in (
        {"task": "text-generation"},
        {"config": CONFIG | {"model_type": "silero_vad_v6"}},
    ):
        assert inspect_descriptor(**(args | changes)).eligibility.disposition != "trust-required"


def test_silero_execution_profile_crosses_protocol_without_chat_or_private_capabilities(tmp_path):
    from types import SimpleNamespace

    from oryn_protocol import ProtocolSchemaStore

    from oryn_engine.models import ModelFile, ModelStore
    from oryn_engine.task_profiles import profile_for_execution

    store = ModelStore(tmp_path, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(resolve=lambda _: object())
    profile = store._inspect_generated_pack(
        "test/native-vad",
        "a" * 40,
        [ModelFile(name, 1024, "b" * 64) for name in ("config.json", "model.safetensors")],
        CONFIG,
        "voice-activity-detection",
        True,
        source_kind="mlx",
    )
    profile["taskProfile"] = profile_for_execution(profile, CONFIG)
    ProtocolSchemaStore().validator("candidateExecutionProfile").validate(profile)
    assert profile["runtimePack"]["runnerId"] == "mlx.vad"
    assert profile["runtimePack"]["taskContract"] == "voice-activity-detection-v1"
    assert profile["technicalCapabilities"]["inputModalities"] == ["audio"]
    assert profile["taskProfile"]["limits"]["audioDurationSeconds"] == 30
