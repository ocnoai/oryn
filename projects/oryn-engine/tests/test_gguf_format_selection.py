"""Format fallback cannot change established model choice or grant trust."""

from types import SimpleNamespace

import pytest
from test_models import FakeHub, model

from oryn_engine.models import ModelFile, ModelStore


@pytest.mark.parametrize("quantization", ["F16", "Q4_K_S"])
def test_new_native_format_is_metadata_compatible_and_still_requires_trust(tmp_path, quantization):
    hub = FakeHub(
        config={"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5"},
        files=[(f"model-{quantization}.gguf", 1000000, "b" * 64)],
        pipeline_tag="text-generation",
        gguf={"architecture": "qwen35", "context_length": 4096},
    )
    store = ModelStore(
        tmp_path,
        hub=hub,
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 64000000000),
    )
    store._pack_runtime_locator = SimpleNamespace(resolve=lambda _: object())
    result = store.inspect(model("test/native-format"))
    assert result["executable"] is True
    assert result["loadDisposition"] == "trust-required"
    assert result["runtimePack"]["trustedMetadata"]["quantization"] == quantization
    assert hub.downloaded == []


@pytest.mark.parametrize("preferred", ["IQ4_XS", "Q4_K_M", "Q5_K_M", "Q8_0"])
def test_existing_format_preference_is_preserved(preferred):
    files = [
        ModelFile(name=f"model-{q}.gguf", size=size, blob_id="b" * 64)
        for q, size in [("F16", 1), ("Q4_K_S", 2), (preferred, 3)]
    ]
    chosen, quantization = ModelStore._select_generic_gguf_file(files)
    assert chosen.name == f"model-{preferred}.gguf" and quantization == preferred


def test_fallback_prefers_compact_format_and_excludes_unreviewed_or_auxiliary_files():
    files = [
        ModelFile(name=name, size=1000, blob_id="b" * 64)
        for name in [
            "model-PQ2_0.gguf",
            "model-PTQ1_0.gguf",
            "mmproj-F16.gguf",
            "mtp-F16.gguf",
            "model-F16.gguf",
            "model-Q4_K_S.gguf",
        ]
    ]
    chosen, quantization = ModelStore._select_generic_gguf_file(files)
    assert chosen.name == "model-Q4_K_S.gguf" and quantization == "Q4_K_S"
    assert ModelStore._select_generic_gguf_file(files[:4]) == (None, None)


def _files(*names):
    return [ModelFile(name=name, size=1000 + i, blob_id="b" * 64) for i, name in enumerate(names)]


def test_quantization_folders_are_variants_and_keep_the_existing_preference():
    files = _files(
        "BF16/Model-BF16-00001-of-00002.gguf",
        "BF16/Model-BF16-00002-of-00002.gguf",
        "Q4_K_M/Model-Q4_K_M-00001-of-00002.gguf",
        "Q4_K_M/Model-Q4_K_M-00002-of-00002.gguf",
        "Q6_K/Model-Q6_K.gguf",
    )
    chosen, quantization = ModelStore._select_generic_gguf_file(files)
    assert chosen.name == "Q4_K_M/Model-Q4_K_M-00001-of-00002.gguf" and quantization == "Q4_K_M"


def test_deeper_or_unsafe_paths_are_never_variants():
    files = _files(
        "a/b/Model-Q4_K_M.gguf",
        "../Model-Q4_K_M.gguf",
        ".hidden/Model-Q4_K_M.gguf",
        "MTP/mtp-Model-Q4_K_M.gguf",
        "Q4_K_M/mmproj-Model-F16.gguf",
    )
    assert ModelStore._select_generic_gguf_file(files) == (None, None)


@pytest.mark.parametrize(
    "names,expected",
    [
        (["m-Q6_K.gguf", "m-Q3_K_M.gguf"], "Q6_K"),
        (["m-Q6_K.gguf", "m-Q4_K_S.gguf"], "Q4_K_S"),
        (["m-IQ3_S.gguf"], "IQ3_S"),
        (["m-BF16.gguf", "m-Q2_K.gguf"], "Q2_K"),
    ],
)
def test_new_levels_are_fallbacks_after_the_historical_six(names, expected):
    chosen, quantization = ModelStore._select_generic_gguf_file(_files(*names))
    assert quantization == expected and chosen.name == f"m-{expected}.gguf"
