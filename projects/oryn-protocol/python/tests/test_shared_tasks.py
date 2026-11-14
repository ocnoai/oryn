import json
from pathlib import Path
import pytest
from oryn_protocol import ProtocolSchemaStore

CASES = json.loads((Path(__file__).resolve().parents[2] / "fixtures/shared-tasks/inputs.json").read_text())

@pytest.mark.parametrize("contract,value", CASES.items())
def test_shared_golden_contract(contract, value):
    schemas = ProtocolSchemaStore()
    assert schemas.is_valid(f"taskInput.{contract}", value)
    for field in ("backend", "modelRoot", "trust_remote_code", "runtimeOptions", "path"):
        assert not schemas.is_valid(f"taskInput.{contract}", {**value, field: "forbidden"})

@pytest.mark.parametrize("contracts,valid", [([], True), (["image-embeddings-v1"], True), (["future-task-v9"], True), (["image-embeddings-v1"] * 2, False), (["../import"], False)])
def test_negotiation_closed_schema(contracts, valid):
    assert ProtocolSchemaStore().is_valid("taskContractsParams", {"contracts": contracts}) is valid

OUTPUTS = json.loads((Path(__file__).resolve().parents[2] / "fixtures/shared-tasks/outputs.json").read_text())
@pytest.mark.parametrize("contract,value", OUTPUTS.items())
def test_closed_output_golden_contract(contract, value):
    schemas = ProtocolSchemaStore()
    assert schemas.is_valid(f"taskOutput.{contract}", value)
    assert not schemas.is_valid(f"taskOutput.{contract}", {**value, "backend": "mps"})

@pytest.mark.parametrize("count,valid", [(308,True),(512,True),(513,False)])
def test_dense_pose_joint_cardinality(count, valid):
    points = [{"id":i,"label":f"joint_{i}","x":1,"y":1,"confidence":0.5,"visible":True} for i in range(count)]
    value = {"width":20,"height":20,"poses":[{"box":[0,0,20,20],"keypoints":points}]}
    assert ProtocolSchemaStore().is_valid("taskOutput.pose-estimation-v1",value) is valid
