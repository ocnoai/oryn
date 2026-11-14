import json
from types import SimpleNamespace

import pytest
from test_metadata_audit import row, snapshot

from oryn_engine.audit_profiles import TARGETS, profile_provenance
from oryn_engine.metadata_audit import inspect_row, requirements_for


@pytest.mark.parametrize("platform", ["linux-x64-cuda", "windows-x64-cuda"])
def test_mac_metadata_candidate_does_not_become_a_nvidia_claim(tmp_path, platform):
    result = inspect_row(row(), snapshot(tmp_path), target_platform=platform)
    assert result["status"] == "metadata-incompatible"
    assert "RUNTIME_PLATFORM_UNDECLARED" in result["reasons"]
    assert not result["passingProfiles"]
    assert not result["executionVerified"]


def test_union_retains_all_profiles_and_static_resource_state(tmp_path):
    result = inspect_row(row(), snapshot(tmp_path), target_platform="mac-nvidia-union")
    assert result["status"] == "metadata-compatible"
    assert result["passingProfiles"] == ["macos-arm64"]
    assert set(result["profileResults"]) == set(TARGETS)
    profile = result["inspectorProfile"]
    assert profile["loadDisposition"] is None
    assert profile["eligibility"]["resources"]["state"] == "not-evaluated"
    assert "runtimePlan" not in profile


def test_union_without_an_eligible_profile_fails(tmp_path):
    result = inspect_row(
        row(task="unknown-task"), snapshot(tmp_path), target_platform="mac-nvidia-union"
    )
    assert result["status"] == "metadata-incompatible"
    assert result["passingProfiles"] == []


def test_profile_backend_is_intersection_of_signed_pack_adapter_and_foundation():
    pack = SimpleNamespace(runners=[SimpleNamespace(backends=("mps", "cpu", "cuda"))])
    adapter = SimpleNamespace(backends=("cuda",))
    foundation = SimpleNamespace(platforms=("linux-x64",), backends=("cpu",))
    assert not TARGETS["linux-x64-cuda"].eligible_backends(pack, adapter, foundation)
    foundation.backends = ("cuda", "cpu")
    assert TARGETS["linux-x64-cuda"].eligible_backends(pack, adapter, foundation) == ("cuda",)
    assert not TARGETS["macos-arm64"].eligible_backends(pack, adapter, foundation)
    assert (
        profile_provenance("macos-arm64")["digest"]
        != profile_provenance("mac-nvidia-union")["digest"]
    )
    value = profile_provenance("mac-nvidia-union")
    assert json.loads(json.dumps(value)) == value


@pytest.mark.parametrize(
    "family,task,contract",
    [
        ("Flux2KleinPipeline", "image-to-image", "image-editing-v1"),
        ("QwenImageEditPlusPipeline", "image-to-image", "image-editing-v1"),
        ("QwenImageLayeredPipeline", "image-text-to-image", "image-layers-v1"),
    ],
)
def test_native_pipeline_contract_keeps_original_task(family, task, contract):
    requirements = requirements_for(task, {"_class_name": family})
    assert requirements["contract"]["id"] == contract
    assert requirements["task"] == task
    if contract == "image-layers-v1":
        assert requirements["outputModalities"] == ["image", "structured"]


def test_repository_custom_pipeline_declaration_cannot_choose_a_contract():
    config = {"_class_name": ["modeling_ddpm", "CustomPipeline"]}
    assert (
        requirements_for("unconditional-image-generation", config)["contract"]["id"]
        == "unconditional-image-generation-v1"
    )
    assert requirements_for("image-to-image", config)["contract"]["id"] == "image-to-image-v1"


def test_onnx_variant_transport_failure_preserves_existing_complete_offer(tmp_path):
    import httpx
    from test_onnx_tasks import encoder, scan

    from oryn_engine.audit_metadata import CatalogInspectionStore
    from oryn_engine.models import ModelFile
    from oryn_engine.onnx_inspection import inspect_onnx

    path = encoder(tmp_path / "native")
    graph = scan(path)
    config = json.loads((path.parent / "config.json").read_text())

    class Hub:
        def read_onnx_metadata(self, repository, revision, name, sizes):
            if name == "z.onnx":
                raise httpx.ConnectTimeout("private endpoint details")
            return graph

    store = CatalogInspectionStore(tmp_path / "store", Hub())
    files = [ModelFile(p.name, p.stat().st_size, "a" * 64) for p in path.parent.iterdir()]
    files.append(ModelFile("z.onnx", 100, "b" * 64))
    offers = []
    inspect_onnx(store, "org/model", "a" * 40, files, config, "feature-extraction", True, offers)
    assert any(profile["executable"] for candidate, profile, plan in offers)
    assert any(
        "ONNX_METADATA_UNAVAILABLE" in profile["blockingReasons"]
        for candidate, profile, plan in offers
    )
    assert "private endpoint" not in str(offers)


@pytest.mark.parametrize(
    "pooling,reason",
    [
        ({"pooling_mode_cls_token": True, "word_embedding_dimension": 3}, None),
        (
            {"pooling_mode_max_tokens": True, "word_embedding_dimension": 3},
            "ONNX_POOLING_UNSUPPORTED",
        ),
        (OSError("missing snapshot"), "ONNX_METADATA_UNAVAILABLE"),
    ],
)
def test_onnx_inspection_binds_pooling_before_offering_execution(tmp_path, pooling, reason):
    from test_onnx_tasks import encoder, scan

    from oryn_engine.audit_metadata import CatalogInspectionStore
    from oryn_engine.models import ModelFile
    from oryn_engine.onnx_inspection import inspect_onnx

    path = encoder(tmp_path / "native", output="token_embeddings")
    graph = scan(path)
    config = json.loads((path.parent / "config.json").read_text())

    class Hub:
        def read_json(self, *_):
            if isinstance(pooling, Exception):
                raise pooling
            return pooling

        def read_onnx_metadata(self, *_):
            return graph

    store = CatalogInspectionStore(tmp_path / "store", Hub())
    files = [ModelFile(p.name, p.stat().st_size, "a" * 64) for p in path.parent.iterdir()]
    files.append(ModelFile("1_Pooling/config.json", 100, "b" * 64))
    offers = []
    inspect_onnx(store, "org/model", "a" * 40, files, config, "feature-extraction", True, offers)
    cpu = next(profile for candidate, profile, _ in offers if candidate.identity.backend == "cpu")
    assert cpu["executable"] is (reason is None)
    if reason:
        assert reason in cpu["blockingReasons"]


def test_omitted_architecture_uses_only_unambiguous_signed_native_mapping():
    from oryn_engine.generated_capabilities import normalized_transformers_config

    config = {"model_type": "siglip", "vision_config": {}, "text_config": {}}
    result = normalized_transformers_config(config, "zero-shot-image-classification")
    assert result["architectures"] == ["SiglipModel"]
    assert "architectures" not in config
    for invalid in ([], ["RepositoryCustomModel"], None):
        declared = config | {"architectures": invalid}
        assert (
            normalized_transformers_config(declared, "zero-shot-image-classification") == declared
        )
    assert normalized_transformers_config(config, "text-classification") == config
    assert normalized_transformers_config({"model_type": "unknown"}, "feature-extraction") == {
        "model_type": "unknown"
    }


def test_historical_detection_class_spelling_requires_exact_native_family_and_task():
    from oryn_engine.generated_capabilities import normalized_transformers_config
    from oryn_engine.model_support import transformers_workload

    config = {"model_type": "rt_detr_v2", "architectures": ["RtDetrV2ForObjectDetection"]}
    normalized = normalized_transformers_config(config, "object-detection")
    assert normalized["architectures"] == ["RTDetrV2ForObjectDetection"]
    assert (
        transformers_workload(normalized["architectures"], "object-detection") == "object-detection"
    )
    assert config["architectures"] == ["RtDetrV2ForObjectDetection"]
    for altered in (
        config | {"model_type": "rt_detr"},
        config | {"architectures": ["CustomDetector"]},
    ):
        assert normalized_transformers_config(altered, "object-detection") == altered
    assert normalized_transformers_config(config, "image-classification") == config
