"""GGUF variants are chosen by what their headers declare; names only order candidates."""

from types import SimpleNamespace

import pytest
from test_models import FakeHub, model

from oryn_engine.models import ModelStore, ModelStoreError

CHATML = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n"
)
GB = 1_000_000_000


def language(architecture="qwen35", quantization="Q4_K_M", template=CHATML, width=4096):
    return {
        "version": 3,
        "architecture": architecture,
        "quantization": quantization,
        "fileType": 15,
        "tokenizerModel": "gpt2",
        "chatTemplate": template,
        "contextLength": 262_144,
        "embeddingLength": width,
        "projectorType": "",
        "projectionLength": 0,
        "tensorTypes": [0, 12],
    }


def projector(kind, width=4096, tensor_types=(0, 30)):
    return {
        "version": 3,
        "architecture": "clip",
        "quantization": "TYPE_39",
        "fileType": 39,
        "tokenizerModel": "projector",
        "chatTemplate": "none",
        "contextLength": 128,
        "embeddingLength": 0,
        "projectorType": kind,
        "projectionLength": width,
        "tensorTypes": list(tensor_types),
    }


class HeaderHub(FakeHub):
    """A Hub whose GGUF headers are known; reads are recorded."""

    def __init__(self, headers, files, **kwargs):
        super().__init__(files=files, **kwargs)
        self.headers = headers
        self.reads = []

    def read_gguf_metadata(self, repository, revision, filename, size, count=1, projector=False):
        self.reads.append(filename)
        value = self.headers[filename]
        if isinstance(value, Exception):
            raise value
        return value


def inspect(tmp_path, hub, task="text-generation", budget=48 * 1024**3):
    hub.pipeline_tag = task
    store = ModelStore(
        tmp_path,
        hub=hub,
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 2**60),
    )
    store._gguf_memory_budget = lambda: budget
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda pack: None if pack.provider else object()
    )
    return store.inspect(model("test/header-gguf"))


def test_a_file_name_without_a_known_level_is_accepted_by_its_header(tmp_path):
    hub = HeaderHub(
        {"Model-IQ2-mix.gguf": language(quantization="IQ2_XS")},
        [("Model-IQ2-mix.gguf", 8 * GB, "b" * 64)],
        config={},
        gguf={"architecture": "qwen35"},
    )
    result = inspect(tmp_path, hub)
    trusted = result["runtimePack"]["trustedMetadata"]
    assert result["executable"] is True and result["loadDisposition"] == "trust-required"
    assert trusted["modelFile"] == "Model-IQ2-mix.gguf"
    assert trusted["quantization"] == "IQ2_XS" and trusted["chatTemplate"] == "chatml"
    assert trusted["contextLength"] == 262_144


def test_drafters_and_unknown_tensor_types_are_skipped_whatever_their_names_say(tmp_path):
    files = [
        ("Bonsai-DFlash2-Q8_0.gguf", 1 * GB, "b" * 64),
        ("Bonsai-PQ2_0-MTP-Q8_0.gguf", 2 * GB, "c" * 64),
        ("Bonsai-Q2_0.gguf", 2 * GB, "d" * 64),
        ("Bonsai-PTQ1_0.gguf", 2 * GB, "e" * 64),
    ]
    hub = HeaderHub(
        {
            "Bonsai-DFlash2-Q8_0.gguf": language(architecture="dflash", quantization="Q8_0"),
            "Bonsai-PQ2_0-MTP-Q8_0.gguf": ModelStoreError("MODEL_METADATA_INVALID", "type 142"),
            "Bonsai-Q2_0.gguf": language(quantization="Q2_0"),
            "Bonsai-PTQ1_0.gguf": ModelStoreError("MODEL_METADATA_INVALID", "type 143"),
        },
        files,
        config={},
        # The Hub summary describes the drafter, not the model.
        gguf={"architecture": "dflash"},
    )
    result = inspect(tmp_path, hub)
    trusted = result["runtimePack"]["trustedMetadata"]
    assert result["executable"] is True
    assert trusted["modelFile"] == "Bonsai-Q2_0.gguf" and trusted["quantization"] == "Q2_0"
    assert trusted["adapterArchitecture"] == "qwen35"
    assert "Bonsai-PTQ1_0.gguf" not in hub.reads  # stops at the first accepted header


@pytest.mark.parametrize("budget,expected", [(48 * 1024**3, "IQ2_XXS"), (2**60, "Q8_0")])
def test_a_variant_that_fits_the_memory_budget_is_preferred(tmp_path, budget, expected):
    files = [
        ("Q8_0/Model-Q8_0-00001-of-00002.gguf", 40 * GB, "b" * 64),
        ("Q8_0/Model-Q8_0-00002-of-00002.gguf", 39 * GB, "c" * 64),
        ("Model-IQ2_XXS.gguf", 22 * GB, "d" * 64),
    ]
    hub = HeaderHub(
        {
            "Q8_0/Model-Q8_0-00001-of-00002.gguf": language(quantization="Q8_0"),
            "Model-IQ2_XXS.gguf": language(quantization="IQ2_XXS"),
        },
        files,
        config={},
        gguf={"architecture": "qwen35"},
    )
    result = inspect(tmp_path, hub, budget=budget)
    assert result["runtimePack"]["trustedMetadata"]["quantization"] == expected
    assert result["executable"] is True


def test_projectors_are_chosen_by_header_even_without_a_float_label(tmp_path):
    files = [
        ("Model-24GB.gguf", 22 * GB, "b" * 64),
        ("mmproj-Model-24GB.gguf", 2 * GB, "c" * 64),
        ("mmproj-other-Q8_0.gguf", 1 * GB, "d" * 64),
    ]
    hub = HeaderHub(
        {
            "Model-24GB.gguf": language(quantization="IQ2_XXS"),
            "mmproj-Model-24GB.gguf": projector("qwen3vl_merger"),
            "mmproj-other-Q8_0.gguf": projector("qwen3vl_merger", tensor_types=(0, 8)),
        },
        files,
        config={},
        gguf={"architecture": "qwen35"},
    )
    result = inspect(tmp_path, hub, task="image-text-to-text")
    trusted = result["runtimePack"]["trustedMetadata"]
    assert result["executable"] is True
    assert result["technicalCapabilities"]["inputModalities"] == ["text", "image"]
    assert trusted["modelFile"] == "Model-24GB.gguf"
    assert trusted["projectorFile"] == "mmproj-Model-24GB.gguf"


@pytest.mark.parametrize(
    "bad",
    [
        projector("nemotron_omni"),  # a projector type not reviewed for this language graph
        projector("qwen3vl_merger", width=2048),  # a width that does not match
        projector("qwen3vl_merger", tensor_types=(0, 8)),  # compressed projector tensors
    ],
)
def test_a_projector_whose_header_does_not_match_is_never_selected(tmp_path, bad):
    hub = HeaderHub(
        {"Model-IQ2_XXS.gguf": language(quantization="IQ2_XXS"), "mmproj-F16.gguf": bad},
        [("Model-IQ2_XXS.gguf", 22 * GB, "b" * 64), ("mmproj-F16.gguf", 2 * GB, "c" * 64)],
        config={},
        gguf={"architecture": "qwen35"},
    )
    result = inspect(tmp_path, hub, task="image-text-to-text")
    assert result["executable"] is False
    assert "GGUF_PROJECTOR_MISSING_OR_AMBIGUOUS" in result["blockingReasons"]


@pytest.mark.parametrize(
    "header,reason",
    [
        (language("dflash", "Q4_K_M"), "RUNTIME_CAPABILITY_UNSUPPORTED"),
        # Withheld: 9 experts per token have no Metal mul_mm_id kernel at b11711.
        (language("nemotron_h_moe", "IQ1_S"), "RUNTIME_CAPABILITY_UNSUPPORTED"),
        (language("qwen35", "TYPE_143"), "GGUF_VARIANT_UNSUPPORTED"),
        (language("qwen35", "Q4_K_M", template="{{ prompt }}"), "GGUF_VARIANT_UNSUPPORTED"),
    ],
)
def test_headers_that_were_read_and_rejected_never_become_executable(tmp_path, header, reason):
    hub = HeaderHub(
        {"Model-Q4_K_M.gguf": header},
        [("Model-Q4_K_M.gguf", 5 * GB, "b" * 64)],
        config={},
        gguf={"architecture": "qwen35"},
    )
    result = inspect(tmp_path, hub)
    assert result["executable"] is False
    assert reason in result["blockingReasons"]
    assert "GGUF_VARIANT_UNSUPPORTED" in result["blockingReasons"]
    assert result["technicalCapabilities"]["architectures"] == [header["architecture"]]


def test_unreadable_headers_keep_the_provisional_name_choice(tmp_path):
    hub = HeaderHub(
        {"Model-Q4_K_M.gguf": OSError("network")},
        [("Model-Q4_K_M.gguf", 5 * GB, "b" * 64)],
        config={},
        gguf={"architecture": "qwen35", "context_length": 4096},
    )
    result = inspect(tmp_path, hub)
    trusted = result["runtimePack"]["trustedMetadata"]
    assert result["executable"] is True
    assert trusted["quantization"] == "Q4_K_M" and trusted["contextLength"] == 4096


def test_header_reads_per_inspection_are_capped(tmp_path):
    names = [f"Model-part{i}.gguf" for i in range(10)]
    hub = HeaderHub(
        {name: language("dflash") for name in names},
        [(name, GB, "b" * 64) for name in names],
        config={},
        gguf={"architecture": "dflash"},
    )
    inspect(tmp_path, hub)
    assert len(hub.reads) == 6
