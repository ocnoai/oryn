import copy
import json
from pathlib import Path

import pytest
from oryn_protocol import ProtocolSchemaStore

CASES = json.loads((Path(__file__).parents[2] / "fixtures/v1/model-compositions.json").read_text())


@pytest.mark.parametrize("value", CASES)
def test_pinned_composition_descriptor(value):
    schema = ProtocolSchemaStore()
    assert schema.is_valid("modelComposition", value)
    for key, replacement in {
        "base": {**value["base"], "revision": "main"},
        "model": {**value["model"], "repository": "https://other.example/model"},
        "adapterFile": "../adapter.safetensors",
        "scale": 0,
        "family": "RepositoryPipeline",
        "overrides": ["../config.json"],
        "trust_remote_code": True,
    }.items():
        changed = copy.deepcopy(value)
        changed[key] = replacement
        assert not schema.is_valid("modelComposition", changed)
