"""Canonical task negotiation. No model data selects validators or executable code."""

from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from importlib.resources import files

from jsonschema import Draft202012Validator

from oryn_protocol.schema import ProtocolSchemaStore

TASK_CONTRACTS = json.loads(
    (files("oryn_protocol") / "schemas/task-contracts/registry.json").read_text()
)["contracts"]
REGISTRY_DIGEST = hashlib.sha256(
    json.dumps(TASK_CONTRACTS, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


@lru_cache(maxsize=1)
def _store():
    return ProtocolSchemaStore()


def _bounded(value, maximum_bytes=262144):
    if len(json.dumps(value, allow_nan=False).encode()) > maximum_bytes:
        raise ValueError("TASK_CONTRACT_INVALID")

    def check(v, depth=0):
        if depth > 16:
            raise ValueError("TASK_CONTRACT_INVALID")
        if isinstance(v, dict):
            if set(v) & {"__proto__", "constructor", "prototype"}:
                raise ValueError("TASK_CONTRACT_INVALID")
            for x in v.values():
                check(x, depth + 1)
        if isinstance(v, list):
            for x in v:
                check(x, depth + 1)

    check(value)


def validate_requirements(value):
    _bounded(value)
    if not _store().is_valid("taskRequirements", value):
        raise ValueError("TASK_REQUIREMENTS_INVALID")
    ref = value.get("contract")
    c = TASK_CONTRACTS.get(ref["id"]) if ref else None
    if ref and (
        not c
        or c["revision"] != ref["revision"]
        or (
            value.get("task")
            and value["task"] not in [c["task"], *c.get("taskAliases", [])]
        )
    ):
        raise ValueError("TASK_CONTRACT_UNSUPPORTED")
    if (value.get("requiredFeatures") or value.get("parameterValues")) and not c:
        raise ValueError("TASK_CONTRACT_REQUIRED")
    if c:
        if any(
            f not in c["featureDefinitions"] for f in value.get("requiredFeatures", [])
        ):
            raise ValueError("TASK_FEATURE_UNKNOWN")
        for k, v in value.get("parameterValues", {}).items():
            if k not in c["parameterNames"] or not Draft202012Validator(
                c["inputSchema"]["properties"][k]
            ).is_valid(v):
                raise ValueError("TASK_PARAMETER_INVALID")
        for f in value.get("requiredFeatures", []):
            for k, v in c["featureDefinitions"][f].get("requiredInput", {}).items():
                if (
                    k in value.get("parameterValues", {})
                    and value["parameterValues"][k] != v
                ):
                    raise ValueError("TASK_REQUIREMENTS_CONTRADICTORY")
    return value


def _safe_constraint(schema, depth=0):
    allowed = {
        "type",
        "enum",
        "const",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        "items",
        "allOf",
        "default",
        "pattern",
    }
    if not isinstance(schema, dict) or depth > 8 or set(schema) - allowed:
        raise ValueError("TASK_PROFILE_INVALID")
    if "allOf" in schema:
        if not isinstance(schema["allOf"], list) or len(schema["allOf"]) > 4:
            raise ValueError("TASK_PROFILE_INVALID")
        for child in schema["allOf"]:
            _safe_constraint(child, depth + 1)
    if "items" in schema:
        _safe_constraint(schema["items"], depth + 1)
    if "pattern" in schema and schema["pattern"] not in {r"\S", r"^[a-z]{2,3}(?:-[A-Z]{2})?$", r"^artifact_[A-Za-z0-9_-]{20,80}$"}:
        raise ValueError("TASK_PROFILE_INVALID")
    Draft202012Validator.check_schema(schema)


def validate_profile(value):
    _bounded(value)
    if not _store().is_valid("taskProfile", value):
        raise ValueError("TASK_PROFILE_INVALID")
    c = TASK_CONTRACTS.get(value["contract"]["id"])
    if (
        not c
        or value["contract"]["revision"] != c["revision"]
        or value["task"] not in [c["task"], *c.get("taskAliases", [])]
    ):
        raise ValueError("TASK_PROFILE_INVALID")
    if (
        value["inputSchemaRef"] != "taskInput." + c["id"]
        or value["resultSchemaRef"] != "taskResult." + c["id"]
    ):
        raise ValueError("TASK_PROFILE_INVALID")
    if (
        set(value["features"]) - c["featureDefinitions"].keys()
        or set(value["limits"]) - set(c["limitNames"])
        or set(value["streamingChannels"]) - set(c["streamingChannels"])
    ):
        raise ValueError("TASK_PROFILE_INVALID")
    if not value["inputModalities"] and c["inputModalities"]:
        raise ValueError("TASK_PROFILE_INVALID")
    if "thinking" in value and c["id"] != "chat-v2":
        raise ValueError("TASK_PROFILE_INVALID")
    for k, v in value["parameters"].items():
        if k not in c["parameterNames"] or v["inputPointer"] != "/" + k:
            raise ValueError("TASK_PROFILE_INVALID")
        _safe_constraint(v["valueSchema"])
        base = copy.deepcopy(c["inputSchema"]["properties"][k])
        base.pop("default", None)
        effective = v["valueSchema"]
        narrowed = (
            effective == base
            or (
                isinstance(effective.get("allOf"), list)
                and len(effective["allOf"]) >= 2
                and effective["allOf"][0] == base
            )
            or (
                "const" in effective
                and effective == base | {"const": effective["const"]}
            )
        )
        if not narrowed:
            raise ValueError("TASK_PROFILE_WIDENS_CONTRACT")
        if "default" in v and (
            not Draft202012Validator(v["valueSchema"]).is_valid(v["default"])
            or not Draft202012Validator(c["inputSchema"]["properties"][k]).is_valid(
                v["default"]
            )
        ):
            raise ValueError("TASK_PROFILE_INVALID")
    return value


def requirement_failures(profile, requirements):
    validate_requirements(requirements)
    if profile is None:
        return (
            ["TASK_PROFILE_UNAVAILABLE"]
            if requirements.get("contract")
            or requirements.get("requiredFeatures")
            or requirements.get("parameterValues")
            else []
        )
    p = validate_profile(profile)
    c = TASK_CONTRACTS[p["contract"]["id"]]
    failures = []
    if requirements.get("contract") and requirements["contract"] != p["contract"]:
        failures.append("TASK_CONTRACT_MISMATCH")
    if requirements.get("task") and requirements["task"] not in [
        p["task"],
        *c.get("taskAliases", []),
    ]:
        failures.append("TASK_MISMATCH")
    for field in ("inputModalities", "outputModalities"):
        if set(requirements.get(field, [])) - set(p[field]):
            failures.append("TASK_MODALITY_MISMATCH")
    for k in requirements.get("requiredFeatures", []):
        if p["features"].get(k) != "supported":
            failures.append("TASK_FEATURE_UNSUPPORTED:" + k)
    for k, v in requirements.get("parameterValues", {}).items():
        control = p["parameters"].get(k)
        if (
            not control
            or control["handling"] not in {"honored", "fixed"}
            or not Draft202012Validator(control["valueSchema"]).is_valid(v)
            or not Draft202012Validator(c["inputSchema"]["properties"][k]).is_valid(v)
        ):
            failures.append("TASK_PARAMETER_UNSUPPORTED:" + k)
    return failures


def validate_invocation(profile, value, requirements=None):
    _bounded(value, 16 * 1024 * 1024)
    p = validate_profile(profile)
    r = requirements or {}
    c = TASK_CONTRACTS[p["contract"]["id"]]
    if requirement_failures(p, r) or not _store().is_valid(p["inputSchemaRef"], value):
        raise ValueError("TASK_INPUT_INVALID")
    validate_input_relations(c["id"], value)
    if c["id"] == "speech-synthesis-v1":
        voice = value.get("voice", p["parameters"].get("voice", {}).get("default"))
        reference = "audioArtifactId" in value
        if ((voice == "reference") != reference
            or reference and "audio" not in p["inputModalities"]):
            raise ValueError("TASK_INPUT_INVALID")
        if len(value["text"]) > p["limits"].get("textCharacters", 350):
            raise ValueError("TASK_INPUT_LIMIT_EXCEEDED")
    if c["id"] == "time-series-forecasting-v1":
        horizon = value.get("horizon", p["parameters"].get("horizon", {}).get("default", 16))
        columns = len(value["pastCovariates"][0]) if "pastCovariates" in value else 0
        if (len(value["context"]) > p["limits"].get("contextLength", 2048)
            or horizon > p["limits"].get("forecastHorizon", 512)
            or columns > p["limits"].get("covariateColumns", 32)):
            raise ValueError("TASK_INPUT_LIMIT_EXCEEDED")
    if c["id"] == "zero-shot-classification-v1" and len(value["labels"]) > p["limits"].get("candidateLabels", 32):
        raise ValueError("TASK_INPUT_LIMIT_EXCEEDED")
    if c["id"] == "embeddings-v1":
        if (len(value["texts"]) > p["limits"].get("batchItems", 64)
            or any(len(text) > p["limits"].get("textCharacters", 8192) for text in value["texts"])):
            raise ValueError("TASK_INPUT_LIMIT_EXCEEDED")
    if c["id"] == "table-question-answering-v1":
        if (len(value["rows"]) > p["limits"].get("tableRows", 128)
            or len(value["columns"]) > p["limits"].get("tableColumns", 32)):
            raise ValueError("TASK_INPUT_LIMIT_EXCEEDED")
    if c["id"] in {"tabular-classification-v1", "tabular-regression-v1"}:
        counts = {"trainingRows": len(value["trainingRows"]),
                  "queryRows": len(value["queryRows"]), "featureColumns": len(value["columns"])}
        if c["id"] == "tabular-classification-v1":
            counts["classes"] = len(set(value["targets"]))
        if any(count > p["limits"].get(key, count) for key, count in counts.items()):
            raise ValueError("TASK_INPUT_LIMIT_EXCEEDED")
    if c["id"] in {"multimodal-generation-v1", "audio-question-answering-v1"}:
        for field, modality in (("imageArtifactId", "image"), ("audioArtifactId", "audio")):
            if field in value and modality not in p["inputModalities"]:
                raise ValueError("TASK_INPUT_INVALID")
    if c["id"] == "multimodal-generation-v1":
        mode = value.get("outputMode", p["parameters"].get("outputMode", {}).get("default", "text"))
        required = {"text": {"text"}, "audio": {"audio"}, "text-and-audio": {"text", "audio"}}[mode]
        if not required <= set(p["outputModalities"]):
            raise ValueError("TASK_INPUT_INVALID")
    for k in c["parameterNames"]:
        if k not in value:
            continue
        control = p["parameters"].get(k)
        if (
            not control
            or control["handling"] not in {"honored", "fixed"}
            or not Draft202012Validator(control["valueSchema"]).is_valid(value[k])
        ):
            raise ValueError("TASK_PARAMETER_UNSUPPORTED")

    if c["id"] == "image-editing-v1":
        count = 1 + len(value.get("referenceImageArtifactIds", []))
        if not p["limits"].get("minReferenceImages", 1) <= count <= p["limits"].get("maxReferenceImages", 1):
            raise ValueError("TASK_INPUT_LIMIT_EXCEEDED")

    def effective(k):
        return value[k] if k in value else p["parameters"].get(k, {}).get("default")

    if p["contract"]["id"] in {"chat-v1", "chat-v2"} and len(value["messages"]) > p["limits"].get("maxMessages", 128):
        raise ValueError("TASK_INPUT_LIMIT_EXCEEDED")

    for k, v in r.get("parameterValues", {}).items():
        if effective(k) != v:
            raise ValueError("TASK_SESSION_PARAMETER_MISMATCH")
    for f in r.get("requiredFeatures", []):
        for k, v in c["featureDefinitions"][f].get("requiredInput", {}).items():
            if effective(k) != v:
                raise ValueError("TASK_FEATURE_INPUT_REQUIRED")


def validate_input_relations(contract, value):
    """Relational constraints that JSON Schema cannot express without extensions."""
    from oryn_protocol.sprint61_relations import valid_input

    if not valid_input(contract, value):
        raise ValueError("TASK_INPUT_INVALID")
    if contract == "table-question-answering-v1":
        columns = value["columns"]
        if len(set(v.strip() for v in columns)) != len(columns) or any(
            len(row) != len(columns) for row in value["rows"]
        ):
            raise ValueError("TASK_INPUT_INVALID")
    if contract == "document-question-answering-v1" and (
        len(value["words"]) != len(value["boxes"])
        or any(x0 > x1 or y0 > y1 for x0, y0, x1, y1 in value["boxes"])
    ):
        raise ValueError("TASK_INPUT_INVALID")
    if contract in {"video-classification-v1", "video-question-answering-v1"}:
        start = value.get("startSeconds", 0)
        if start >= 30 or start + value.get("durationSeconds", 0) > 30:
            raise ValueError("TASK_INPUT_INVALID")


def make_profile(
    contract_id,
    *,
    task=None,
    features=None,
    constraints=None,
    defaults=None,
    unsupported=(),
    limits=None,
    channels=(),
    input_modalities=None,
    output_modalities=None,
    thinking=None,
):
    c = TASK_CONTRACTS[contract_id]
    parameters = {}
    for k in c["parameterNames"]:
        # New optional controls require an explicit adapter declaration. Older
        # profiles retain their exact parameter set and reject these inputs.
        if (
            k in c.get("declarationRequiredParameters", ())
            and k not in (constraints or {})
            and k not in (defaults or {})
            and k not in unsupported
        ):
            continue
        base = copy.deepcopy(c["inputSchema"]["properties"][k])
        default = base.pop("default", None)
        constraint = (constraints or {}).get(k)
        value_schema = {"allOf": [base, constraint]} if constraint else base
        fixed = constraint is not None and "const" in constraint
        if fixed:
            value_schema = base | {"const": constraint["const"]}
            default = constraint["const"]
        if defaults and k in defaults:
            default = defaults[k]
        parameter = {
            "inputPointer": "/" + k,
            "valueSchema": value_schema,
            "handling": "unsupported"
            if k in unsupported
            else "fixed"
            if fixed
            else "honored",
            "editable": k not in unsupported and not fixed,
            "affects": ["result"],
        }
        if k not in unsupported and default is not None:
            parameter["default"] = default
        parameters[k] = parameter
    p = {
        "kind": "task-profile",
        "schemaVersion": 1,
        "profileId": contract_id,
        "contract": {"id": contract_id, "revision": c["revision"]},
        "task": task or c["task"],
        "inputModalities": input_modalities or (["text"] if contract_id == "speech-synthesis-v1" and "audioArtifactId" not in parameters else c["inputModalities"]),
        "outputModalities": output_modalities or c["outputModalities"],
        "inputSchemaRef": "taskInput." + contract_id,
        "resultSchemaRef": "taskResult." + contract_id,
        "features": {
            k: (features or {}).get(k, "unknown") for k in c["featureDefinitions"]
        },
        "parameters": parameters,
        "limits": limits or {},
        "streamingChannels": list(channels),
        "extensions": {},
        "provenance": "adapter-derived",
    }
    if thinking is not None:
        p["thinking"] = copy.deepcopy(thinking)
    validate_profile(p)
    p["profileId"] = (
        contract_id
        + "."
        + hashlib.sha256(
            json.dumps(p, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )
    return p


def effective_input(profile, value):
    """Resolve negotiated defaults once, so a backend fallback cannot change them."""
    result = copy.deepcopy(value)
    for name, control in profile["parameters"].items():
        if (
            name not in result
            and "default" in control
            and control["handling"] in {"honored", "fixed"}
        ):
            result[name] = copy.deepcopy(control["default"])
    return result


def validate_result(profile, value, input_value=None):
    # Artifact ownership and semantic geometry/timing checks remain with native adapters.
    serialized = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    contract = profile["contract"]["id"]
    limit = (900 if contract in {"visual-document-retrieval-v1", "pose-estimation-v1"} else 256) * 1024
    if contract == "visual-document-retrieval-v1" and isinstance(value.get("value"), dict) and value["value"].get("dimensions") == 4096:
        limit = 12 * 1024 * 1024
    if len(serialized) > limit:
        raise ValueError("TASK_OUTPUT_INVALID")
    if not _store().is_valid(profile["resultSchemaRef"], value):
        raise ValueError("TASK_OUTPUT_INVALID")
    from oryn_protocol.sprint61_relations import valid_result_media

    if not valid_result_media(profile["contract"]["id"], value):
        raise ValueError("TASK_OUTPUT_INVALID")
    if value.get("kind") == "structured":
        validate_output_relations(profile["contract"]["id"], value["value"])
    elif profile["contract"]["id"] in {"mask-generation-v1", "image-segmentation-v1"}:
        validate_output_relations(profile["contract"]["id"], value)
    if input_value is not None:
        from oryn_protocol.sprint61_relations import valid_result_for_input

        if not valid_result_for_input(profile["contract"]["id"], value, input_value):
            raise ValueError("TASK_OUTPUT_INVALID")
        if contract == "chat-v2":
            result = value["value"]
            if result["metadata"]["thinking"]["requested"] != input_value.get("thinking"):
                raise ValueError("TASK_OUTPUT_INVALID")
            output_tokens = result["usage"]["outputTokens"]
            budget = input_value.get("maxTokens", profile["parameters"].get("maxTokens", {}).get("default", 4096))
            if output_tokens is not None and output_tokens > budget:
                raise ValueError("TASK_OUTPUT_INVALID")


def validate_output_relations(contract, value):
    from oryn_protocol.sprint61_relations import valid_output

    valid = valid_output(contract, value)
    if contract == "chat-v2":
        mode = value["metadata"]["thinking"]
        if mode["applied"] and (mode["requested"] is None or mode["effective"] != mode["requested"] or mode["reason"] not in {"applied", "already_fixed"}):
            valid = False
        if not mode["applied"] and mode["reason"] in {"applied", "already_fixed"}:
            valid = False
        if (mode["requested"] is None) != (mode["reason"] == "model_default"):
            valid = False
        if mode["reason"] == "fixed_mode" and (mode["effective"] is None or mode["effective"] == mode["requested"]):
            valid = False
        usage = value["usage"]
        if usage["outputTokens"] is not None:
            if any(usage[key] is not None and usage[key] > usage["outputTokens"] for key in ("reasoningTokens", "answerTokens")):
                valid = False
            if usage["reasoningTokens"] is not None and usage["answerTokens"] is not None and usage["reasoningTokens"] + usage["answerTokens"] > usage["outputTokens"]:
                valid = False
    if contract == "visual-document-retrieval-v1":
        document, queries = value["documentVectors"], value["queries"]
        valid = all(len(v) == value["dimensions"] and abs(sum(x*x for x in v) - 1) <= 0.0001
                    for v in [*document, *(v for q in queries for v in q["vectors"])])
        valid = valid and all(abs(q["score"] - sum(
            max(sum(x*y for x, y in zip(v, d, strict=True)) for d in document)
            for v in q["vectors"]) / (len(q["vectors"]) if value["scoring"] == "mean-max-cosine" else 1)) <= 0.001 for q in queries)
    if contract in {"mask-generation-v1", "image-segmentation-v1"} and value.get("kind") == "multi":
        metadata = value["outputs"][1]["value"]
        segments = metadata["segments"]
        valid = len({s["id"] for s in segments}) == len(segments) and sum(
            s["pixelCount"] for s in segments
        ) == metadata["width"] * metadata["height"]
        if contract == "mask-generation-v1":
            valid = valid and all(s["id"] in {0, 255} for s in segments)
    elif contract == "document-question-answering-v1":
        x0, y0, x1, y1 = value["box"]
        valid = value["wordStart"] <= value["wordEnd"] and x0 <= x1 and y0 <= y1
    elif contract == "keypoint-detection-v1":
        valid = all(
            p["x"] <= value["width"] and p["y"] <= value["height"]
            for p in value["keypoints"]
        )
    elif contract in {"video-classification-v1", "video-question-answering-v1"}:
        times = value["sampledTimestamps"]
        valid = (
            len(set(times)) >= 2
            and times == sorted(times)
            and all(t < value["durationSeconds"] for t in times)
        )
    elif contract == "voice-activity-detection-v1":
        segments = value["segments"]
        valid = all(
            s["start"] < s["end"] <= value["durationSeconds"] for s in segments
        ) and all(
            a["end"] <= b["start"]
            for a, b in zip(segments[:-1], segments[1:], strict=True)
        )
    if not valid:
        raise ValueError("TASK_OUTPUT_INVALID")
