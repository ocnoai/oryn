import json
from pathlib import Path

import pytest
from oryn_protocol.model_selection import validate_selection

FIXTURES = json.loads(
    (Path(__file__).parents[2] / "fixtures/model-selection/golden.json").read_text()
)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["name"])
def test_shared_model_selection_fixture(fixture):
    if fixture["valid"]:
        validate_selection(fixture["selection"], fixture.get("preferred"))
    else:
        with pytest.raises(ValueError):
            validate_selection(fixture["selection"], fixture.get("preferred"))
