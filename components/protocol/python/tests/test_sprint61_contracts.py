import copy
import json
from pathlib import Path

import pytest
from oryn_protocol.task_contracts import (
    effective_input, make_profile, validate_invocation, validate_result,
    validate_requirements, requirement_failures,
)

CASES = json.loads((Path(__file__).parents[2] / "fixtures/shared-tasks/inputs.json").read_text())
OUTPUTS = json.loads((Path(__file__).parents[2] / "fixtures/shared-tasks/outputs.json").read_text())
NUMERIC = ["tabular-classification-v1", "tabular-regression-v1", "time-series-forecasting-v1"]


def test_labeled_pose_preserves_person_and_joint_geometry():
    p = make_profile("pose-estimation-v1")
    source = CASES["pose-estimation-v1"]
    result = {"kind": "structured", "value": copy.deepcopy(OUTPUTS["pose-estimation-v1"])}
    validate_invocation(p, source)
    validate_result(p, result, source)
    for boxes in ([[0, 0, 0, 1]], [[0.5, 0, 0.2, 1]], [[0, 0, 1, 2]]):
        with pytest.raises(ValueError):
            validate_invocation(p, source | {"boxes": boxes})
    for change in ({"box": [0, 0, 31, 32]}, {"keypoints": result["value"]["poses"][0]["keypoints"] * 2}):
        bad = copy.deepcopy(result)
        bad["value"]["poses"][0].update(change)
        with pytest.raises(ValueError):
            validate_result(p, bad, source)
    bad = copy.deepcopy(result)
    bad["value"]["poses"][0]["keypoints"][0]["visible"] = False
    with pytest.raises(ValueError):
        validate_result(p, bad, source)


@pytest.mark.parametrize("contract", NUMERIC)
def test_numeric_results_bind_to_query_count_and_context(contract):
    p = make_profile(contract)
    source = effective_input(p, CASES[contract])
    result = {"kind": "structured", "value": copy.deepcopy(OUTPUTS[contract])}
    validate_invocation(p, source)
    validate_result(p, result, source)
    result["value"]["predictions"].append(result["value"]["predictions"][0])
    with pytest.raises(ValueError, match="TASK_OUTPUT_INVALID"):
        validate_result(p, result, source)


@pytest.mark.parametrize("change", [
    {"targets": ["cold"]}, {"targets": ["cold", "cold"]},
    {"queryRows": [["not numeric"]]}, {"queryRows": [[True]]},
    {"queryRows": [[1, 2]]}, {"queryTargets": ["cold"]},
    {"columns": [{"name": "x", "type": "numeric"}, {"name": " x ", "type": "numeric"}]},
])
def test_tabular_rejects_leakage_misalignment_and_mislabeled_features(change):
    value = CASES["tabular-classification-v1"] | change
    with pytest.raises(ValueError):
        validate_invocation(make_profile("tabular-classification-v1"), value)


@pytest.mark.parametrize("change", [
    {"timestamps": [0, 2, 1], "frequencyMilliseconds": 1},
    {"timestamps": [0, 1, 3], "frequencyMilliseconds": 1},
    {"timestamps": [0, 1, 2]}, {"frequencyMilliseconds": 1},
    {"quantiles": [0.9, 0.1]}, {"quantiles": [0.5, 0.5]},
    {"pastCovariates": [[1], [2], [3]]},
    {"pastCovariates": [[1], [2], [3]], "futureCovariates": [[1, 2], [3, 4]]},
    {"futureTargets": [4, 5]},
])
def test_forecast_rejects_lookahead_and_invalid_time_alignment(change):
    with pytest.raises(ValueError):
        validate_invocation(make_profile("time-series-forecasting-v1"), CASES["time-series-forecasting-v1"] | change)


def test_forecast_preserves_native_quantiles_and_future_timestamps():
    p = make_profile("time-series-forecasting-v1")
    source = CASES[p["contract"]["id"]] | {"quantiles": [0.1, 0.9], "timestamps": [0, 10, 20], "frequencyMilliseconds": 10}
    result = {"kind": "structured", "value": {"predictions": [4, 5], "timestamps": [30, 40],
        "quantiles": [{"level": 0.1, "values": [3, 4]}, {"level": 0.9, "values": [5, 6]}]}}
    validate_invocation(p, source)
    validate_result(p, result, source)
    for change in ({"timestamps": [20, 30]}, {"quantiles": []},
                   {"quantiles": [{"level": 0.1, "values": [7, 4]}, {"level": 0.9, "values": [5, 6]}]}):
        with pytest.raises(ValueError):
            validate_result(p, {"kind": "structured", "value": result["value"] | change}, source)


def test_explicit_operation_and_output_mode_cannot_be_silently_ignored():
    p = make_profile("audio-transformation-v1", constraints={"operation": {"const": "enhance"}})
    with pytest.raises(ValueError):
        validate_invocation(p, {"operation": "decode", "codes": [[1]]})
    with pytest.raises(ValueError):
        validate_invocation(p, {"operation": "enhance", "audioArtifactId": CASES["audio-transformation-v1"]["audioArtifactId"], "codes": [[1]]})
    omni = make_profile("multimodal-generation-v1", constraints={"outputMode": {"const": "text"}},
                        features={"audioOutput": "unsupported"}, input_modalities=["text", "image"], output_modalities=["text"])
    assert requirement_failures(omni, {"contract": omni["contract"], "requiredFeatures": ["audioOutput"]})
    with pytest.raises(ValueError):
        validate_invocation(omni, {"instruction": "describe"})
    with pytest.raises(ValueError):
        validate_requirements({"contract": {"id": "multimodal-generation-v2", "revision": 2}})


def test_unconditional_generation_rejects_hidden_prompt():
    with pytest.raises(ValueError):
        validate_invocation(make_profile("unconditional-image-generation-v1"), {"prompt": ""})


@pytest.mark.parametrize("mode,sources", [("image", {"imageArtifactId"}), ("audio", {"audioArtifactId"}),
    ("image-and-audio", {"imageArtifactId", "audioArtifactId"})])
def test_omni_modes_bind_exact_sources_and_cannot_infer_joint_support(mode, sources):
    value = {"instruction": "Describe the input.", "inputMode": mode, **{k: "artifact_aaaaaaaaaaaaaaaaaaaaaaaa" for k in sources}}
    validate_invocation(make_profile("multimodal-generation-v1"), value)
    image_only = make_profile("multimodal-generation-v1", constraints={"inputMode": {"const": "image"}}, input_modalities=["text", "image"], output_modalities=["text"])
    if mode == "image":
        validate_invocation(image_only, value)
        with pytest.raises(ValueError):
            validate_invocation(image_only, value | {"outputMode": "audio"})
        with pytest.raises(ValueError):
            validate_invocation(image_only, value | {"audioArtifactId": "artifact_bbbbbbbbbbbbbbbbbbbbbbbb"})
    else:
        with pytest.raises(ValueError):
            validate_invocation(image_only, value)


@pytest.mark.parametrize("contract", ["unconditional-image-generation-v1", "audio-transformation-v1", "image-text-to-video-v1", "video-editing-v1"])
def test_new_media_contracts_reject_wrong_artifact_media(contract):
    result = copy.deepcopy(OUTPUTS[contract])
    result["artifact"]["mimeType"] = "text/plain"
    with pytest.raises(ValueError):
        validate_result(make_profile(contract), result)
