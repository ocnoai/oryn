import json
from pathlib import Path

import pytest
from oryn_protocol.task_contracts import validate_invocation

PROFILES = json.loads((Path(__file__).parents[2] / "fixtures/task-contracts/image-references.json").read_text())
IDS = ["artifact_" + c * 24 for c in "abcd"]
BASE = {"imageArtifactId": IDS[0], "prompt": "Use image 1 and image 2"}

@pytest.mark.parametrize("name,counts", [("legacy", [0]), ("optional", [0, 1, 2]), ("pair", [1])])
def test_reference_counts_follow_negotiated_profile(name, counts):
    for count in range(4):
        value = BASE | ({"referenceImageArtifactIds": IDS[1:1+count]} if count else {})
        if count in counts:
            validate_invocation(PROFILES[name], value)
        else:
            with pytest.raises(ValueError):
                validate_invocation(PROFILES[name], value)

@pytest.mark.parametrize("references", [[IDS[0]], [IDS[1], IDS[1]], ["/tmp/image.png"], "bad", None, [3]])
def test_reference_identity_and_shape_are_closed(references):
    with pytest.raises(ValueError):
        validate_invocation(PROFILES["optional"], BASE | {"referenceImageArtifactIds": references})
