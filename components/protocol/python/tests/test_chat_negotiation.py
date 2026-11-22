import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from oryn_protocol import ProtocolSchemaStore

ROOT = Path(__file__).parents[2]
FIXTURE = json.loads((ROOT / 'fixtures/task-contracts/chat-negotiation.json').read_text())


def test_existing_negotiation_accepts_v2_and_old_peer_supported_subset():
    store = ProtocolSchemaStore()
    for name in ('legacyRequest', 'currentRequest'):
        store.validator('taskContractsRequest').validate(FIXTURE[name])
    for name in ('legacyResult', 'currentResult'):
        store.validator('taskContractsResult').validate(FIXTURE[name])


def test_legacy_capabilities_shape_needs_no_new_field():
    wire = json.loads((ROOT / 'schemas/v1/protocol-v1.schema.json').read_text())
    legacy = copy.deepcopy(wire)
    del legacy['$defs']['runtimeCapabilitiesResult']['properties']['taskContracts']
    old = Draft202012Validator({'$ref': '#/$defs/runtimeCapabilitiesResult', '$defs': legacy['$defs']})
    baseline = json.loads((ROOT / 'fixtures/v1/runtime-capabilities.json').read_text())['result']
    assert old.is_valid(baseline)
    assert not old.is_valid(baseline | {'taskContracts': []})
