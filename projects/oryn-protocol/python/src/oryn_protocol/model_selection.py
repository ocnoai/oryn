"""Website intent validation, matching the TypeScript selection contract."""

import re

from oryn_protocol.schema import ProtocolSchemaStore


def validate_selection(value: object, preferred: dict | None = None) -> dict:
    if not ProtocolSchemaStore().is_valid("modelSelection", value):
        raise ValueError("SELECTION_INVALID")
    if value["intent"] == "exact-artifact":
        if not preferred or not preferred.get("files"):
            raise ValueError("EXACT_ARTIFACT_REQUIRED")
        validate_reference(preferred)
    elif value["intent"] == "preferred-variant" and not preferred:
        raise ValueError("PREFERRED_ARTIFACT_REQUIRED")
    if "sourceCheckpoint" in value:
        validate_reference(value["sourceCheckpoint"])
        if "files" in value["sourceCheckpoint"]:
            raise ValueError("CHECKPOINT_FILES_INVALID")
    return value


def validate_reference(value: object) -> dict:
    if not ProtocolSchemaStore().is_valid("requestModelRef", value):
        raise ValueError("MODEL_REFERENCE_INVALID")
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", value.get("revision", "main")
    ) or ".." in value.get("revision", "main").split("/"):
        raise ValueError("MODEL_REFERENCE_INVALID")
    if any(
        any(p in {"", ".", ".."} for p in f.split("/")) for f in value.get("files", [])
    ):
        raise ValueError("ARTIFACT_IDENTITY_INVALID")
    return value
