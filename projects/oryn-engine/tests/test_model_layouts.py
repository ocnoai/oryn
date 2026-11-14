import pytest
from test_compatibility_plane import descriptor

from oryn_engine.capability_sources import framework_registry
from oryn_engine.compatibility_plane import evaluate_eligibility
from oryn_engine.model_layouts import assess_layout, validate_weight_index


@pytest.mark.parametrize("count", [2, 4])
def test_native_qwen_shard_names_use_the_declared_index(count):
    shards = [f"model.safetensors-{i:05d}-of-{count:05d}.safetensors" for i in range(1, count + 1)]
    names = [
        "config.json",
        "tokenizer.json",
        "processor_config.json",
        "model.safetensors.index.json",
        *shards,
    ]
    layout = assess_layout(names, {}, "image-text-to-text")
    assert not layout.artifact_reasons
    index = {"weight_map": {f"tensor-{i}": name for i, name in enumerate(shards)}}
    validate_weight_index(index, set(names), expected_shards=set(shards))
    with pytest.raises(ValueError, match="complete"):
        validate_weight_index(
            {"weight_map": {"tensor": shards[0]}}, set(names), expected_shards=set(shards)
        )
    assert (
        "MODEL_SUBSET_INCOMPLETE"
        in assess_layout(names[:-1], {}, "image-text-to-text").artifact_reasons
    )
    wrong_index = [
        name.replace("model.safetensors.index.json", "model.safetensors.safetensors.index.json")
        for name in names
    ]
    assert (
        "MODEL_SUBSET_INCOMPLETE"
        in assess_layout(wrong_index, {}, "image-text-to-text").artifact_reasons
    )


def test_whisper_selects_only_default_graph_with_its_processor_assets():
    default = ["config.json", "model.safetensors", "tokenizer.json", "preprocessor_config.json"]
    alternative = [
        "model.fp32-00001-of-00002.safetensors",
        "model.fp32-00002-of-00002.safetensors",
        "model.safetensors.index.fp32.json",
    ]
    layout = assess_layout([*default, *alternative], {}, "automatic-speech-recognition")
    assert set(layout.selected_files) == set(default)
    assert not layout.artifact_reasons
    alternative_layout = assess_layout(
        [*default[:1], *default[2:], *alternative], {}, "automatic-speech-recognition"
    )
    assert not alternative_layout.artifact_reasons


def test_conflicting_index_conventions_fail_closed():
    names = [
        "config.json",
        "tokenizer.json",
        "model.fp32-00001-of-00001.safetensors",
        "model.fp32.safetensors.index.json",
        "model.safetensors.index.fp32.json",
    ]
    assert "MODEL_SUBSET_INCOMPLETE" in assess_layout(names, {}, "text-generation").artifact_reasons


def test_index_may_include_selected_safe_companion_weights():
    shards = {"model-00001-of-00001.safetensors"}
    selected = shards | {"model-vision.safetensors"}
    index = {"weight_map": {"language": next(iter(shards)), "vision": "model-vision.safetensors"}}
    validate_weight_index(index, selected, expected_shards=shards)
    with pytest.raises(ValueError, match="unselected"):
        validate_weight_index(index, shards, expected_shards=shards)


def test_component_index_cannot_hide_missing_or_conflicting_shards():
    first = "diffusion_pytorch_model-00001-of-00002.safetensors"
    with pytest.raises(ValueError, match="complete declared"):
        validate_weight_index({"weight_map": {"tensor": first}}, {first})
    conflicting = "diffusion_pytorch_model-00002-of-00003.safetensors"
    with pytest.raises(ValueError, match="inconsistent"):
        validate_weight_index({"weight_map": {"a": first, "b": conflicting}}, {first, conflicting})


def test_adapter_weights_cannot_impersonate_a_complete_model():
    layout = assess_layout(
        ["config.json", "tokenizer.json", "adapter_model.safetensors"], {}, "text-generation"
    )
    assert "MODEL_SUBSET_INCOMPLETE" in layout.artifact_reasons


def test_alternate_exports_and_repository_code_do_not_select_a_loader():
    layout = assess_layout(
        [
            "config.json",
            "tokenizer.json",
            "model.safetensors",
            "onnx/model.onnx",
            "onnx/config.json",
            "training.py",
        ],
        {},
        "feature-extraction",
    )
    assert layout.selected_files == ("config.json", "model.safetensors", "tokenizer.json")
    assert not layout.runtime_reasons


def test_missing_shards_and_unknown_quantization_are_blocked():
    result = assess_layout(
        [
            "config.json",
            "tokenizer.json",
            "model-00001-of-00002.safetensors",
            "model.safetensors.index.json",
        ],
        {"quantization_config": {"quant_method": "arbitrary"}},
        "text-generation",
    )
    assert "MODEL_SUBSET_INCOMPLETE" in result.artifact_reasons
    assert "QUANTIZATION_UNSUPPORTED" in result.runtime_reasons


def test_index_cannot_select_code_or_external_files():
    for path in ("../model.safetensors", "weights.bin", "model.py", "missing.safetensors"):
        with pytest.raises(ValueError):
            validate_weight_index({"weight_map": {"tensor": path}}, {"model.safetensors"})


def test_static_audit_has_no_machine_admission_or_load_action():
    result = evaluate_eligibility(descriptor(), framework_registry(), resource_admissible=None)
    assert result.axes["resources"].state == "not-evaluated"
    assert result.disposition is None


def test_family_metadata_cannot_certify_an_unknown_whisper_revision():
    result = evaluate_eligibility(
        descriptor(
            architectures=("WhisperForConditionalGeneration",), task="automatic-speech-recognition"
        ),
        framework_registry(),
        resource_admissible=True,
    )
    assert result.disposition == "trust-required"
    assert result.axes["conformance"].state == "untested"
