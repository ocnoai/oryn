import json
from pathlib import Path

import pytest
from oryn_protocol.task_contracts import (
    TASK_CONTRACTS,
    effective_input,
    make_profile,
    requirement_failures,
    validate_invocation,
    validate_profile,
    validate_requirements,
    validate_result,
)

FIXTURES = json.loads(
    (Path(__file__).parents[2] / "fixtures/task-contracts/negotiation.json").read_text()
)


def test_audio_classifier_duration_limit_is_public_and_closed_for_older_registries(monkeypatch):
    profile = json.loads((Path(__file__).parents[2] / "fixtures/task-contracts/audio-duration.json").read_text())
    assert validate_profile(profile)["limits"] == {"audioDurationSeconds": 12}
    validate_invocation(profile, {"audioArtifactId": "artifact_" + "a" * 24})
    validate_profile(make_profile("audio-classification-v1"))
    monkeypatch.setitem(TASK_CONTRACTS["audio-classification-v1"], "limitNames", [])
    with pytest.raises(ValueError):
        validate_profile(profile)


def test_audio_head_selection_requires_explicit_adapter_support():
    profiles = json.loads((Path(__file__).parents[2] / "fixtures/task-contracts/audio-heads.json").read_text())
    value = {"audioArtifactId": "artifact_" + "a" * 24}
    legacy, multihead = profiles["legacy"], profiles["multihead"]
    assert set(legacy["parameters"]) == {"topK"}
    validate_invocation(legacy, value)
    for head in ("language", "gender"):
        validate_invocation(multihead, value | {"classificationHead": head})
        with pytest.raises(ValueError):
            validate_invocation(legacy, value | {"classificationHead": head})
    assert effective_input(multihead, value)["classificationHead"] == "language"
    with pytest.raises(ValueError):
        validate_invocation(multihead, value | {"classificationHead": "other"})


def test_older_audio_registry_rejects_new_head_profile(monkeypatch):
    profile = make_profile("audio-classification-v1", defaults={"classificationHead": "language"})
    monkeypatch.setitem(TASK_CONTRACTS["audio-classification-v1"], "parameterNames", ["topK"])
    with pytest.raises(ValueError):
        validate_profile(profile)


def test_embedding_modes_require_explicit_adapter_support():
    legacy = make_profile("embeddings-v1")
    assert set(legacy["parameters"]) == {"normalize"}
    value = {"texts": ["first chunk", "second chunk"]}
    validate_invocation(legacy, value)
    for extra in ({"inputType": "document"}, {"inputType": "query"}, {"contextual": False}):
        with pytest.raises(ValueError):
            validate_invocation(legacy, value | extra)
    declared = make_profile("embeddings-v1", defaults={"inputType": "document", "contextual": False})
    for extra in ({"inputType": "query"}, {"inputType": "document", "contextual": True}):
        validate_invocation(declared, value | extra)
    with pytest.raises(ValueError):
        validate_invocation(declared, value | {"inputType": "query", "contextual": True})
    assert effective_input(declared, value) == value | {
        "normalize": True, "inputType": "document", "contextual": False
    }


def test_forecast_declared_context_horizon_and_covariates_are_enforced():
    profile = make_profile("time-series-forecasting-v1", defaults={"horizon": 8},
                           limits={"contextLength": 16, "forecastHorizon": 8, "covariateColumns": 2})
    value = {"context": [1.0] * 16, "horizon": 8}
    validate_invocation(profile, value)
    validate_invocation(profile, {"context": value["context"]})
    validate_invocation(profile, value | {"pastCovariates": [[1.0, 2.0]] * 16,
                                         "futureCovariates": [[1.0, 2.0]] * 8})
    for extra in ({"context": [1.0] * 17}, {"horizon": 9},
                  {"pastCovariates": [[1.0] * 3] * 16, "futureCovariates": [[1.0] * 3] * 8}):
        with pytest.raises(ValueError, match="TASK_INPUT_LIMIT_EXCEEDED"):
            validate_invocation(profile, value | extra)


def test_old_embedding_registry_rejects_new_controls(monkeypatch):
    declared = make_profile("embeddings-v1", defaults={"inputType": "document", "contextual": False})
    monkeypatch.setitem(TASK_CONTRACTS["embeddings-v1"], "parameterNames", ["normalize"])
    with pytest.raises(ValueError):
        validate_profile(declared)


def test_zero_shot_candidate_limit_is_public_and_enforced():
    profile = make_profile("zero-shot-classification-v1", limits={"candidateLabels": 26})
    value = {"text": "classify", "labels": [f"label {i}" for i in range(26)]}
    validate_invocation(profile, value)
    with pytest.raises(ValueError, match="TASK_INPUT_LIMIT_EXCEEDED"):
        validate_invocation(profile, value | {"labels": value["labels"] + ["extra"]})
    validate_invocation(make_profile("zero-shot-classification-v1"), value | {"labels": value["labels"] + ["extra"]})


def test_older_registry_rejects_unrecognized_candidate_limit(monkeypatch):
    profile = make_profile("zero-shot-classification-v1", limits={"candidateLabels": 26})
    monkeypatch.setitem(TASK_CONTRACTS["zero-shot-classification-v1"], "limitNames", [])
    with pytest.raises(ValueError):
        validate_profile(profile)


@pytest.mark.parametrize("contract", TASK_CONTRACTS)
def test_every_existing_task_has_a_valid_common_profile(contract):
    profile = make_profile(contract)
    assert validate_profile(profile)["contract"] == {"id": contract, "revision": 1}


@pytest.mark.parametrize("case", FIXTURES, ids=lambda c: c["name"])
def test_cross_language_negotiation(case):
    profile = case["profile"]
    requirements = case["requirements"]
    if case.get("requirementsInvalid"):
        with pytest.raises(ValueError):
            validate_requirements(requirements)
        return
    assert requirement_failures(profile, requirements) == case["failures"]
    if "input" in case:
        if case["inputValid"]:
            validate_invocation(profile, case["input"], requirements)
        else:
            with pytest.raises(ValueError):
                validate_invocation(profile, case["input"], requirements)


def test_defaults_are_resolved_once_and_unsupported_controls_are_not_injected():
    p = make_profile("text-to-music-v1", unsupported=["seed", "lyrics"])
    assert effective_input(p, {"prompt": "music"}) == {
        "prompt": "music",
        "durationSeconds": 10,
    }
    with pytest.raises(ValueError):
        validate_invocation(p, {"prompt": "music", "seed": 0})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(providerOptions={"path": "/private"}),
        lambda p: p["parameters"]["timestamps"].update(
            valueSchema={"$ref": "https://example.com/evil.json"}
        ),
        lambda p: p["parameters"]["timestamps"].update(default="yes"),
        lambda p: p["features"].update(wordTimestamps=True),
        lambda p: p["contract"].update(revision=2),
        lambda p: p.update(inputSchemaRef="taskInput.chat-v1"),
        lambda p: p["parameters"]["timestamps"].update(inputPointer="/backend"),
        lambda p: p["extensions"].update({"repo.loader": {"module": "repo"}}),
    ],
)
def test_profile_rejects_unsafe_or_inconsistent_declarations(mutate):
    p = make_profile("transcription-v1")
    mutate(p)
    with pytest.raises((ValueError, TypeError)):
        validate_profile(p)


@pytest.mark.parametrize(
    "contract,value",
    [
        (
            "voice-activity-detection-v1",
            {"segments": [{"start": 0, "end": 1.1}], "durationSeconds": 1},
        ),
        (
            "voice-activity-detection-v1",
            {
                "segments": [{"start": 0, "end": 0.7}, {"start": 0.5, "end": 1}],
                "durationSeconds": 1,
            },
        ),
        (
            "video-question-answering-v1",
            {
                "text": "red then blue",
                "sampledTimestamps": [1, 0],
                "durationSeconds": 2,
            },
        ),
        (
            "keypoint-detection-v1",
            {"width": 10, "height": 10, "keypoints": [{"x": 11, "y": 1, "score": 1}]},
        ),
        (
            "document-question-answering-v1",
            {
                "answer": "a",
                "score": 1,
                "wordStart": 2,
                "wordEnd": 1,
                "box": [0, 0, 1, 1],
            },
        ),
    ],
)
def test_output_geometry_and_times_cannot_escape_their_document_or_media(
    contract, value
):
    with pytest.raises(ValueError, match="TASK_OUTPUT_INVALID"):
        validate_result(make_profile(contract), {"kind": "structured", "value": value})


@pytest.mark.parametrize("scoring,score", [("sum-max-cosine", 2.0), ("mean-max-cosine", 1.0)])
def test_visual_retrieval_scores_and_normalization_match_returned_vectors(scoring, score):
    vector = [1.0] + [0.0] * 127
    value = {"dimensions": 128, "normalized": True, "scoring": scoring,
             "documentVectors": [vector], "queries": [{"vectors": [vector[:], vector[:]], "score": score}]}
    profile = make_profile("visual-document-retrieval-v1")
    validate_result(profile, {"kind": "structured", "value": value})
    value["queries"][0]["score"] = 0.5
    with pytest.raises(ValueError, match="TASK_OUTPUT_INVALID"):
        validate_result(profile, {"kind": "structured", "value": value})
    value["queries"][0]["score"] = score
    vector[0] = 0.5
    with pytest.raises(ValueError, match="TASK_OUTPUT_INVALID"):
        validate_result(profile, {"kind": "structured", "value": value})


def test_mask_metadata_rejects_duplicate_ids_wrong_totals_and_nonbinary_ids():
    import copy
    artifact = {"kind": "artifact", "artifact": {"outputId": "output_fixture",
        "internalPath": "/controlled/output/result.png", "mimeType": "image/png", "sizeBytes": 128}}
    metadata = {"width": 2, "height": 2, "encoding": "binary-8bit", "segments": [
        {"id": 0, "label": "background", "pixelCount": 1},
        {"id": 255, "label": "foreground", "pixelCount": 3}]}
    value = {"kind": "multi", "outputs": [artifact, {"kind": "structured", "value": metadata}]}
    profile = make_profile("mask-generation-v1")
    validate_result(profile, value)
    for field, replacement in [("id", 0), ("id", 1), ("pixelCount", 4)]:
        bad = copy.deepcopy(value)
        bad["outputs"][1]["value"]["segments"][1][field] = replacement
        with pytest.raises(ValueError, match="TASK_OUTPUT_INVALID"):
            validate_result(profile, bad)


@pytest.mark.parametrize("dimensions", [128, 320, 4096])
def test_visual_retrieval_keeps_declared_projection_width(dimensions):
    import copy
    vector = [1.0] + [0.0] * (dimensions - 1)
    value = {"dimensions": dimensions, "normalized": True, "scoring": "sum-max-cosine",
             "documentVectors": [vector], "queries": [{"vectors": [vector], "score": 1.0}]}
    profile = make_profile("visual-document-retrieval-v1")
    validate_result(profile, {"kind": "structured", "value": value})
    for location in ("page", "query", "declaration"):
        bad = copy.deepcopy(value)
        if location == "page":
            bad["documentVectors"][0].append(0.0)
        elif location == "query":
            bad["queries"][0]["vectors"][0].pop()
        else:
            bad["dimensions"] = 320 if dimensions == 128 else 128
        with pytest.raises(ValueError, match="TASK_OUTPUT_INVALID"):
            validate_result(profile, {"kind": "structured", "value": bad})


def test_visual_retrieval_dense_projection_keeps_wire_size_bound():
    vector = [round(1 / (320 ** 0.5), 6)] * 320
    value = {"dimensions": 320, "normalized": True, "scoring": "sum-max-cosine",
             "documentVectors": [vector] * 144,
             "queries": [{"vectors": [vector] * 48, "score": 48 * sum(x*x for x in vector)}] * 2}
    profile = make_profile("visual-document-retrieval-v1")
    validate_result(profile, {"kind": "structured", "value": value})
    dense = [1 / (320 ** 0.5)] * 320
    value["documentVectors"] = [dense] * 256
    value["queries"] = [{"vectors": [dense] * 64, "score": 64.0}] * 2
    with pytest.raises(ValueError, match="TASK_OUTPUT_INVALID"):
        validate_result(profile, {"kind": "structured", "value": value})


def test_speech_reference_requires_explicit_profile_and_authorized_artifact():
    profiles = json.loads((Path(__file__).parents[2] / 'fixtures/task-contracts/speech-reference.json').read_text())
    reference, legacy = profiles['reference'], profiles['legacy']
    value = {'text': 'Hello.', 'audioArtifactId': 'artifact_' + 'a' * 24}
    assert set(legacy['parameters']) == {'voice', 'language', 'speed'}
    assert legacy['inputModalities'] == ['text']
    validate_invocation(legacy, {'text': 'Hello.'})
    validate_invocation(reference, value)
    for bad in [dict(text='Hello.'), value | {'voice':'af_heart'}, value | {'speed':1.5},
                value | {'language':'zz'}, value | {'text':'a'*301}, value | {'audioArtifactId':'bad'}]:
        with pytest.raises(ValueError): validate_invocation(reference, bad)
    with pytest.raises(ValueError): validate_invocation(legacy,value)


def test_speech_reference_is_rejected_by_older_registry():
    from oryn_protocol.task_contracts import TASK_CONTRACTS
    profile=json.loads((Path(__file__).parents[2] / 'fixtures/task-contracts/speech-reference.json').read_text())['reference']
    c=TASK_CONTRACTS['speech-synthesis-v1'];previous=c['parameterNames']
    try:
        c['parameterNames']=['voice','language','speed']
        with pytest.raises(ValueError):validate_profile(profile)
    finally:c['parameterNames']=previous


def test_video_frame_capacity_preserves_adapter_bounds():
    expanded = make_profile('text-to-video-v1', constraints={'frames': {'enum': [124, 141, 345]}, 'fps': {'const': 24}}, defaults={'frames': 124, 'fps': 24})
    legacy = make_profile('text-to-video-v1', constraints={'frames': {'maximum': 120}})
    for frames in (124, 141, 345):
        validate_invocation(expanded, {'prompt': 'Birds', 'frames': frames})
        with pytest.raises(ValueError):
            validate_invocation(legacy, {'prompt': 'Birds', 'frames': frames})
    for frames in (123, 125, 360, 361):
        with pytest.raises(ValueError):
            validate_invocation(expanded, {'prompt': 'Birds', 'frames': frames})
    validate_invocation(make_profile('text-to-video-v1'), {'prompt': 'Birds', 'frames': 360})
    with pytest.raises(ValueError):
        validate_invocation(make_profile('text-to-video-v1'), {'prompt': 'Birds', 'frames': 361})


def test_native_image_video_preserves_more_than_129_frames():
    profile = make_profile('image-text-to-video-v1', constraints={'frames': {'enum': [124, 141, 345]}, 'fps': {'const': 24}}, defaults={'frames': 124, 'fps': 24})
    value = {'prompt': 'Birds', 'imageArtifactId': 'artifact_' + 'a' * 24}
    for frames in (124, 141, 345):
        validate_invocation(profile, value | {'frames': frames})
    for extra in ({'frames': 129}, {'frames': 360}, {'frames': 361}, {'fps': 30}):
        with pytest.raises(ValueError):
            validate_invocation(profile, value | extra)


def test_wide_retrieval_preserves_native_vectors_and_bounds_total_tokens():
    vector = [1 / 64] * 4096
    value = dict(dimensions=4096, normalized=True, scoring="sum-max-cosine",
                 documentVectors=[vector] * 144, queries=[dict(vectors=[vector] * 48, score=48)])
    assert len(json.dumps(value).encode()) > 1024 * 1024
    profile = make_profile("visual-document-retrieval-v1")
    validate_result(profile, dict(kind="structured", value=value))
    for bad in [value | dict(documentVectors=[vector] * 145),
                value | dict(queries=[dict(vectors=[vector] * 49, score=49)])]:
        with pytest.raises(ValueError):
            validate_result(profile, dict(kind="structured", value=bad))
