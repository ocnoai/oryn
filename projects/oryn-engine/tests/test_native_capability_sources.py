from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

from oryn_engine.compatibility_plane import CompatibilityPlaneError, compile_registry
from oryn_engine.generated_capabilities import generated_artifact
from oryn_engine.native_capability_sources import (
    NativeDescriptorError,
    native_registry,
    native_sources,
    validate_onnx_selected_files,
)


def test_signed_diffusers_mapping_digest_matches_the_installed_source_file() -> None:
    package = importlib.util.find_spec("diffusers")
    assert package is not None and package.origin
    mapping = Path(package.origin).parent / "pipelines/auto_pipeline.py"
    artifact = generated_artifact()
    digest = artifact["buildInputs"]["diffusersMappingSha256"]
    assert digest == hashlib.sha256(mapping.read_bytes()).hexdigest()
    # The foundation also binds Oryn's task adapters. Its composite identity
    # must not be mistaken for the library mapping file's verification hash.
    source = next(s for s in artifact["sources"] if s["provider"]["id"] == "oryn.source.diffusers")
    assert source["foundation"]["digest"] != digest


def test_native_sources_compile_with_the_frozen_provider_neutral_schema() -> None:
    registry = native_registry()
    assert {value.source_kind for value in registry.tuples} >= {"gguf", "onnx"}
    assert any(value.architecture == "qwen35" for value in registry.tuples)
    assert any(value.task == "text-to-speech" for value in registry.tuples)


def test_native_provider_cannot_widen_the_common_envelope() -> None:
    source = copy.deepcopy(native_sources()[0])
    source["compiledBackends"] = ["metal"]
    with pytest.raises(CompatibilityPlaneError):
        compile_registry((source,))


def test_onnx_file_membership_is_bounded_and_path_safe() -> None:
    files = validate_onnx_selected_files(
        [
            ("model/model.onnx", 100, "a" * 64),
            ("model/model.onnx.data", 200, "b" * 64),
            ("tokenizer.json", 20, "c" * 64),
        ]
    )
    assert files == ("model/model.onnx", "model/model.onnx.data", "tokenizer.json")


@pytest.mark.parametrize(
    "files",
    [
        [("../model.onnx", 100, "a" * 64)],
        [("model.onnx", 100, "a" * 64), ("kernel.dylib", 20, "b" * 64)],
        [("first.onnx", 100, "a" * 64), ("second.onnx", 100, "b" * 64)],
        [("weights.data", 100, "a" * 64)],
    ],
)
def test_onnx_rejects_traversal_code_ambiguous_graphs_and_missing_graph(
    files: list[tuple[str, int, str]],
) -> None:
    with pytest.raises(NativeDescriptorError):
        validate_onnx_selected_files(files)
