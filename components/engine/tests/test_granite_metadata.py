import json
import subprocess
import sys
from pathlib import Path

import pytest

from oryn_engine.model_metadata import FLOAT32_MAX, parse_metadata, validate_model_config


def original():
    return json.loads((Path(__file__).parent / "fixtures/granite_title.json").read_text())


def test_original_inactive_clip_parses_without_importing_native_runtimes():
    config = original()
    raw = json.dumps(config).encode()
    assert parse_metadata(raw) == config
    validate_model_config(config)
    code = (
        "import sys; from oryn_engine.model_metadata import parse_metadata; "
        "c=parse_metadata(sys.stdin.buffer.read()); "
        "assert c['time_step_limit'][1] == 3.4028234663852886e38; "
        "assert not {'torch','transformers','mlx.core'}.intersection(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], input=raw, check=True, capture_output=True)


@pytest.mark.parametrize(
    "change",
    [
        {"model_type": "llama"},
        {"architectures": ["UnknownModel"]},
        {"num_hidden_layers": 27},
        {"num_hidden_layers": True},
        {"layer_types": ["mamba"] * 28},
        {"layer_types": []},
        {"hidden_size": FLOAT32_MAX},
        {"unknown_option": FLOAT32_MAX},
        {"time_step_limit": [0.0, float("inf")]},
        {"time_step_limit": [0.0, float("nan")]},
        {"time_step_limit": [False, FLOAT32_MAX]},
        {"time_step_limit": [0.0, 1e30]},
        {"time_step_limit": [FLOAT32_MAX, 0.0]},
        {"nested": {"time_step_limit": [0.0, FLOAT32_MAX]}},
    ],
)
def test_exception_does_not_relax_other_numeric_or_graph_bounds(change):
    with pytest.raises(ValueError, match="number"):
        parse_metadata(json.dumps(original() | change).encode())


def test_nested_graph_cannot_inherit_root_clip_exception():
    with pytest.raises(ValueError, match="number"):
        parse_metadata(json.dumps({"nested": original()}).encode())


def test_inert_clip_does_not_waive_repository_code_rejection():
    with pytest.raises(ValueError, match="Repository code"):
        validate_model_config(original() | {"auto_map": {"AutoModel": "unreviewed.Code"}})


@pytest.mark.parametrize(
    "change",
    [
        {"num_local_experts": False},
        {"num_local_experts": -1},
        {"num_experts_per_tok": False},
        {"num_experts_per_tok": 1},
        {"model_type": "llama"},
    ],
)
def test_dense_zero_experts_do_not_relax_other_dimension_checks(change):
    config = original() | {"time_step_limit": [0.0, 100.0]} | change
    with pytest.raises(ValueError, match="dimensions"):
        validate_model_config(config)
