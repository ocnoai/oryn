from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def packaged_schema_path() -> Path:
    return Path(str(files("oryn_protocol") / "schemas" / "v1" / "protocol-v1.schema.json"))


class ProtocolSchemaStore:
    def __init__(self, schema_path: Path | None = None) -> None:
        self.schema_path = schema_path or packaged_schema_path()
        with self.schema_path.open(encoding="utf-8") as schema_file:
            self.schema: dict[str, Any] = json.load(schema_file)

        Draft202012Validator.check_schema(self.schema)
        self._validators: dict[str, Draft202012Validator] = {}

    def validator(self, definition: str) -> Draft202012Validator:
        validator = self._validators.get(definition)
        if validator is None:
            validator = Draft202012Validator(
                {
                    "$schema": self.schema["$schema"],
                    "$ref": f"#/$defs/{definition}",
                    "$defs": self.schema["$defs"],
                }
            )
            self._validators[definition] = validator
        return validator

    def is_valid(self, definition: str, value: object) -> bool:
        return self.validator(definition).is_valid(value)
