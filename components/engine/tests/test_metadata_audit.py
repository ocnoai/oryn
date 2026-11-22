from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from xml.etree import ElementTree as ET

import httpx
import pytest
from test_models import FakeHub

from oryn_engine.audit_metadata import (
    BoundedAuditClient,
    CatalogInspectionStore,
    HubApiGate,
    MetadataSnapshot,
    save_json,
)
from oryn_engine.metadata_audit import (
    classify,
    inspect_row,
    main,
    parse_reference,
    read_rows,
    requirements_for,
)
from oryn_engine.runtime_routing import canonical_digest

sys.path.insert(0, str(Path(__file__).parents[1] / "providers/mflux/src"))
SHA = "a" * 40


def test_audio_audit_requires_playable_audio_without_optional_codec_tokens():
    requirements = requirements_for("audio-to-audio")
    assert requirements["contract"] == {"id": "audio-transformation-v1", "revision": 1}
    assert requirements["inputModalities"] == ["audio"]
    assert requirements["outputModalities"] == ["audio"]


class MetadataClient:
    def __init__(self, **kwargs):
        self.hub = FakeHub(sha=SHA, **kwargs)
        self._api = SimpleNamespace(model_info=Mock(side_effect=self.info))
        self._token = Mock(return_value=False)
        self.has_access = Mock(return_value=False)
        self.read_json = Mock(side_effect=self.json)
        self.download = Mock(side_effect=AssertionError("No weights may be downloaded"))

    def info(self, repo_id, revision, **kwargs):
        return self.hub.model_info(repo_id, revision)

    def json(self, repository, revision, filename):
        assert revision == SHA
        return self.hub.config if filename == "config.json" else {}


def snapshot(tmp_path, **kwargs):
    return MetadataSnapshot(
        tmp_path / "snapshot.json", "org/model", "main", client=MetadataClient(**kwargs)
    )


def row(**kwargs):
    return {"row": 1, "input": "https://huggingface.co/org/model", **kwargs}


@pytest.mark.parametrize(
    "link,revision,expected",
    [
        ("org/model", "", "main"),
        ("https://huggingface.co/org/model?x=y#usage", "", "main"),
        (f"https://huggingface.co/org/model/tree/{SHA}", "", SHA),
        ("https://huggingface.co/org/model/tree/refs%2Fpr%2F1", "", "refs/pr/1"),
        ("org/model", SHA, SHA),
    ],
)
def test_reference_normalization(link, revision, expected):
    assert parse_reference(link, revision) == ("org/model", expected)


@pytest.mark.parametrize(
    "link,revision",
    [
        ("https://example.com/org/model", ""),
        ("https://huggingface.co@evil.test/org/model", ""),
        ("https://huggingface.co/datasets/org/model", ""),
        ("https://huggingface.co/org/model/blob/main/config.json", ""),
        ("https://huggingface.co/org/model/tree/main/subfolder", ""),
        ("https://huggingface.co/org/model/tree/main", SHA),
        ("org/..", ""),
    ],
)
def test_ambiguous_or_non_model_links_fail(link, revision):
    with pytest.raises(ValueError):
        parse_reference(link, revision)


def test_csv_bom_quotes_duplicates_and_optional_columns(tmp_path):
    path = tmp_path / "models.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["url", "task", "files"])
        writer.writerow(["org/model", "text-generation", '["a.gguf", "b.gguf"]'])
        writer.writerow(["org/model", "feature-extraction", ""])
    rows = read_rows(path)
    assert len(rows) == 2 and rows[0]["files"] == '["a.gguf", "b.gguf"]'
    path.write_text("org/model\norg/second\n")
    assert [r["input"] for r in read_rows(path)] == ["org/model", "org/second"]


def test_uses_production_inspector_and_pins_revision_without_weights(tmp_path):
    hub = snapshot(tmp_path)
    result = inspect_row(row(), hub)
    assert result["status"] == "metadata-compatible", result
    assert result["revision"] == SHA
    assert result["taskProfile"]["contract"]["id"] == "transcription-v1"
    assert result["executionVerified"] is False
    assert result["selectedFiles"] and all(f["blobId"] for f in result["selectedFiles"])
    hub.client.download.assert_not_called()
    assert not list(tmp_path.rglob("*.safetensors"))
    with pytest.raises(RuntimeError, match="WEIGHT_DOWNLOAD_FORBIDDEN"):
        hub.download()


def test_snapshot_offline_replay_never_uses_network_and_matches_verdict(tmp_path):
    hub = snapshot(tmp_path)
    first = inspect_row(row(), hub)
    bomb = Mock()
    bomb._api.model_info.side_effect = AssertionError("No network in offline mode")
    bomb.read_json.side_effect = AssertionError("No network in offline mode")
    replay = MetadataSnapshot(hub.path, "org/model", "main", offline=True, client=bomb)
    second = inspect_row(row(), replay)
    assert second["status"] == first["status"] == "metadata-compatible"
    assert second["snapshotDigest"] == first["snapshotDigest"]
    assert second["identity"] == first["identity"]
    assert second["selectedFiles"] == first["selectedFiles"]
    assert not bomb.mock_calls


def test_missing_root_metadata_is_incomplete_not_supported(tmp_path):
    hub = snapshot(tmp_path)
    hub.client.read_json.side_effect = TimeoutError("credential must not appear in report")
    result = inspect_row(row(), hub)
    assert result["status"] == "metadata-incomplete"
    assert "credential" not in json.dumps(result)
    assert "config.json" in result["metadataReadFailures"]


def test_tensor_structure_has_its_own_frozen_cache_and_offline_replay(tmp_path):
    hub = snapshot(tmp_path)
    hub.model_info("org/model", "main")
    name = next(
        f["name"] for f in hub.data["inventory"]["files"] if f["name"].endswith(".safetensors")
    )
    summary = {"metadata": {}, "dtypes": ["F32"]}
    structure = summary | {
        "tensors": {"weight": {"shape": [2, 2], "dtype": "F32"}},
        "headerBytes": 80,
        "dataBytes": 16,
    }
    hub.client.read_tensor_metadata = Mock(return_value=summary)
    hub.client.read_tensor_structure = Mock(return_value=structure)
    assert hub.read_tensor_metadata("org/model", SHA, name) == summary
    replay = MetadataSnapshot(hub.path, "org/model", "main", offline=True)
    with pytest.raises(OSError, match="SNAPSHOT_METADATA_MISSING"):
        replay.read_tensor_structure("org/model", SHA, name)
    assert hub.read_tensor_structure("org/model", SHA, name) == structure
    replay = MetadataSnapshot(hub.path, "org/model", "main", offline=True, client=Mock())
    assert replay.read_tensor_structure("org/model", SHA, name) == structure
    assert replay.read_tensor_metadata("org/model", SHA, name) == summary
    assert not replay.client.mock_calls
    hub.client.download.assert_not_called()


def test_missing_offline_snapshot_is_incomplete(tmp_path):
    hub = MetadataSnapshot(tmp_path / "missing.json", "org/model", "main", offline=True)
    assert inspect_row(row(), hub)["reasons"] == ["SNAPSHOT_MISSING"]


def test_tampered_snapshot_is_rejected(tmp_path):
    hub = snapshot(tmp_path)
    hub.model_info("org/model", "main")
    envelope = json.loads(hub.path.read_text())
    envelope["payload"]["revision"] = "b" * 40
    hub.path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="DIGEST_MISMATCH"):
        MetadataSnapshot(hub.path, "org/model", "main", offline=True)


def test_exact_revision_mismatch_is_never_silently_replaced(tmp_path):
    hub = MetadataSnapshot(
        tmp_path / "snapshot.json", "org/model", "b" * 40, client=MetadataClient()
    )
    assert inspect_row(row(), hub)["status"] == "error"


def test_gated_metadata_is_access_required_without_weight_or_config_read(tmp_path):
    hub = snapshot(tmp_path, gated="auto")
    result = inspect_row(row(), hub)
    assert result["status"] == "access-required"
    hub.client.read_json.assert_not_called()
    hub.client.download.assert_not_called()


@pytest.mark.parametrize(
    "task,status,reason",
    [
        (None, "metadata-incomplete", "TASK_UNSPECIFIED"),
        ("unimplemented-task", "metadata-incompatible", "TASK_CONTRACT_MISSING"),
    ],
)
def test_missing_and_unknown_tasks_are_not_replaced(tmp_path, task, status, reason):
    result = inspect_row(row(), snapshot(tmp_path, pipeline_tag=task))
    assert result["status"] == status
    assert result["reasons"] == [reason]


def test_explicit_task_and_files_are_enforced(tmp_path):
    hub = snapshot(tmp_path)
    wrong_task = inspect_row(row(task="text-generation"), hub)
    assert wrong_task["status"] != "metadata-compatible"
    wrong_files = inspect_row(row(files='["missing.safetensors"]'), hub)
    assert wrong_files["status"] != "metadata-compatible"
    assert "MODEL_SUBSET_INCOMPLETE" in wrong_files["reasons"]


def test_custom_code_is_rejected_by_production_rule(tmp_path):
    hub = snapshot(
        tmp_path,
        config={
            "architectures": ["WhisperForConditionalGeneration"],
            "auto_map": {"AutoModel": "modeling.Custom"},
        },
    )
    result = inspect_row(row(), hub)
    assert result["status"] == "metadata-incompatible", result


def test_invalid_native_forecast_profile_remains_a_causal_blocker(tmp_path):
    hub = snapshot(
        tmp_path,
        pipeline_tag="time-series-forecasting",
        config={
            "model_type": "timesfm",
            "architectures": ["TimesFmModelForPrediction"],
            "context_length": 512,
            "horizon_length": 128,
            "patch_length": 32,
            "quantiles": [0.9, 0.1],
        },
    )
    result = inspect_row(row(task="time-series-forecasting"), hub)
    assert result["status"] in {"metadata-incompatible", "metadata-incomplete"}, result
    assert "TASK_CONFIGURATION_UNSUPPORTED" in result["reasons"]
    assert result["task"] == "time-series-forecasting"
    hub.client.download.assert_not_called()


def test_declined_alternative_does_not_invalidate_selected_route():
    profile = {
        "executable": True,
        "selectedFiles": [{"name": "config.json"}],
        "taskProfile": {"task": "text-generation"},
        "providerDiagnostics": [
            {"providerId": "alternative", "code": "PROVIDER_INSPECTION_FAILED"}
        ],
        "blockingReasons": ["CONFORMANCE_REQUIRED"],
    }
    status, _, missing = classify(profile, {"unused/config.json": "METADATA_NETWORK_ERROR"})
    assert status == "metadata-compatible" and not missing
    assert classify(profile, {"config.json": "METADATA_NETWORK_ERROR"})[0] == "metadata-incomplete"
    profile["executable"] = False
    assert classify(profile, {})[0] == "metadata-incomplete"


def test_rejected_metadata_is_incompatible_instead_of_missing():
    profile = {
        "executable": False,
        "selectedFiles": [{"name": "config.json"}],
        "blockingReasons": ["MODEL_METADATA_INVALID"],
    }
    status, reasons, missing = classify(profile, {"config.json": "MODEL_METADATA_INVALID"})
    assert status == "metadata-incompatible" and not missing
    assert reasons == ["MODEL_METADATA_INVALID"]
    status, _, _ = classify(profile, {"config.json": "MODEL_METADATA_UNAVAILABLE"})
    assert status == "metadata-incomplete"


def test_only_content_rejections_replay_with_same_parser(tmp_path, monkeypatch):
    from oryn_engine import audit_metadata
    from oryn_engine.models import ModelStoreError

    hub = snapshot(tmp_path)
    hub.model_info("org/model", "main")
    hub.data["metadata"].clear()
    evidence = {"sha256": "b" * 64, "sizeBytes": 42}
    hub.client.read_json.side_effect = ModelStoreError(
        "MODEL_METADATA_INVALID", "Invalid JSON", details={"metadataRejection": evidence}
    )
    with pytest.raises(ModelStoreError):
        hub.read_json("org/model", SHA, "config.json")
    replay = MetadataSnapshot(hub.path, "org/model", "main", offline=True, client=hub.client)
    hub.client.read_json.reset_mock()
    with pytest.raises(ModelStoreError) as exc:
        replay.read_json("org/model", SHA, "config.json")
    assert exc.value.code == "MODEL_METADATA_INVALID"
    hub.client.read_json.assert_not_called()
    monkeypatch.setattr(audit_metadata, "json_parser_digest", lambda: "changed-parser")
    changed = MetadataSnapshot(hub.path, "org/model", "main", offline=True, client=hub.client)
    with pytest.raises(OSError, match="SNAPSHOT_METADATA_MISSING"):
        changed.read_json("org/model", SHA, "config.json")
    live = MetadataSnapshot(hub.path, "org/model", "main", client=hub.client)
    hub.client.read_json.side_effect = None
    hub.client.read_json.return_value = {"model_type": "bert"}
    assert live.read_json("org/model", SHA, "config.json") == {"model_type": "bert"}
    assert not live.data["metadataRejections"]


def test_metadata_transient_failures_are_not_frozen_as_rejections(tmp_path):
    from oryn_engine.models import ModelStoreError

    hub = snapshot(tmp_path)
    hub.model_info("org/model", "main")
    hub.data["metadata"].clear()
    hub.client.read_json.side_effect = ModelStoreError("MODEL_METADATA_UNAVAILABLE", "Timed out")
    with pytest.raises(ModelStoreError):
        hub.read_json("org/model", SHA, "config.json")
    assert not hub.data.get("metadataRejections")


def test_chat_requirements_keep_visual_input_and_task_alias():
    assert requirements_for("image-text-to-text")["inputModalities"] == ["text", "image"]
    assert requirements_for("sentence-similarity")["contract"]["id"] == "embeddings-v1"


def test_cli_csv_json_junit_and_unique_counts_offline(tmp_path):
    cache = tmp_path / "snapshots"
    key = canonical_digest(["org/model", "main", "anonymous"])
    hub = MetadataSnapshot(cache / f"{key}.json", "org/model", "main", client=MetadataClient())
    inspect_row(row(), hub)
    source = tmp_path / "models.csv"
    source.write_text("url\norg/model\nhttps://huggingface.co/org/model\n")
    output = tmp_path / "report"
    assert main([str(source), "--output", str(output), "--cache", str(cache), "--offline"]) == 0
    report = json.loads((output / "results.json").read_text())
    assert report["summary"]["inputRows"] == 2
    assert report["summary"]["uniqueIdentities"] == 1
    assert report["summary"]["executedModels"] == 0
    assert report["assumptions"]["loading"] == "not-tested"
    assert len(list(csv.DictReader((output / "results.csv").open()))) == 2
    suite = ET.parse(output / "junit.xml").getroot()
    assert suite.get("tests") == "2" and suite.get("failures") == "0"


def test_cli_fail_policy_and_expected_negative_case(tmp_path):
    cache = tmp_path / "snapshots"
    key = canonical_digest(["org/model", "main", "anonymous"])
    hub = MetadataSnapshot(cache / f"{key}.json", "org/model", "main", client=MetadataClient())
    hub.model_info("org/model", "main")
    source = tmp_path / "models.csv"
    source.write_text("url,task,expected\norg/model,unknown,metadata-incompatible\n")
    argv = [str(source), "--output", str(tmp_path / "out"), "--cache", str(cache), "--offline"]
    assert main(argv) == 1
    assert main([*argv, "--fail-on", "unexpected"]) == 0
    assert main([*argv, "--fail-on", "none"]) == 0
    source.write_text("url\nhttps://evil.test/org/model\n")
    assert main([*argv, "--fail-on", "none"]) == 2


def test_cli_refresh_resolves_mutable_ref_once_for_duplicate_rows(tmp_path, monkeypatch):
    client = MetadataClient()
    monkeypatch.setattr("oryn_engine.audit_metadata.BoundedAuditClient", lambda **_: client)
    source = tmp_path / "models.csv"
    source.write_text("url\norg/model\norg/model\n")
    argv = [str(source), "--output", str(tmp_path / "out")]
    assert main(argv) == 0
    assert main(argv) == 0
    assert client._api.model_info.call_count == 1
    assert main([*argv, "--refresh"]) == 0
    assert client._api.model_info.call_count == 2


def test_optional_provider_is_inspection_only(tmp_path):
    hub = snapshot(tmp_path)
    store = CatalogInspectionStore(tmp_path / "store", hub)
    with pytest.raises(RuntimeError, match="EXECUTION_FORBIDDEN"):
        store._provider_process.invoke(
            SimpleNamespace(pack_id="oryn.mflux-mlx"), None, "prepare", {}
        )


def test_metadata_read_budget_and_inventory_gate(tmp_path):
    hub = snapshot(tmp_path)
    hub.model_info("org/model", "main")
    hub.reads = 64
    with pytest.raises(OSError):
        hub.read_json("org/model", SHA, "tokenizer.json")
    with pytest.raises(OSError):
        hub.read_json("org/model", SHA, "../outside.json")
    assert hub.client.read_json.call_count == 1


def test_offline_missing_selected_config_does_not_use_hub_summary(tmp_path):
    hub = snapshot(tmp_path)
    hub.model_info("org/model", "main")
    hub.data["metadata"] = {}
    save_json(hub.path, {"schemaVersion": 1, "sha256": hub.digest, "payload": hub.data})
    offline = MetadataSnapshot(hub.path, "org/model", "main", offline=True)
    assert inspect_row(row(), offline)["status"] == "metadata-incomplete"


def test_inventory_timeout_is_unknown_not_incompatibility(tmp_path):
    hub = snapshot(tmp_path)
    hub.client._api.model_info.side_effect = TimeoutError("private-token")
    result = inspect_row(row(), hub)
    assert result["status"] == "metadata-incomplete"
    assert result["reasons"] == ["METADATA_NETWORK_ERROR"]
    assert "private-token" not in json.dumps(result)


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, "access-required"),
        (403, "access-required"),
        (404, "metadata-incomplete"),
        (429, "metadata-incomplete"),
    ],
)
def test_http_failures_are_not_counted_as_unsupported(tmp_path, status, expected):
    hub = snapshot(tmp_path)
    response = httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co"))
    hub.client._api.model_info.side_effect = httpx.HTTPStatusError(
        "secret",
        request=response.request,
        response=response,
    )
    result = inspect_row(row(), hub)
    assert result["status"] == expected
    assert "secret" not in json.dumps(result)


def test_ambient_token_is_not_used_without_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    hub = MetadataSnapshot(tmp_path / "snapshot.json", "org/model", "main")
    assert hub.client._token() is False
    assert hub.data["accessMode"] == "anonymous"


def test_onnx_graph_budget_bounds_all_variants_without_fetching_full_files(monkeypatch):
    import oryn_engine.onnx_graph as graph

    client = BoundedAuditClient()
    client.MAX_GRAPH_REQUESTS = 1
    requests = []

    def serve(request):
        requests.append(request)
        assert request.headers["Range"] == "bytes=0-3"
        return httpx.Response(206, content=b"data", headers={"content-range": "bytes 0-3/100"})

    transport_client = httpx.Client(transport=httpx.MockTransport(serve))
    monkeypatch.setattr("oryn_engine.audit_metadata.httpx.Client", lambda **_: transport_client)

    def scan(read_range, *_):
        assert read_range(0, 4) == b"data"
        read_range(4, 4)

    monkeypatch.setattr(graph, "scan_graph", scan)
    with pytest.raises(OSError, match="BUDGET_EXCEEDED"):
        client.read_onnx_metadata("org/model", SHA, "model.onnx", {"model.onnx": 100})
    assert len(requests) == 1


def test_onnx_server_ignoring_range_is_rejected(monkeypatch):
    import oryn_engine.onnx_graph as graph

    client = BoundedAuditClient()
    transport_client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    monkeypatch.setattr("oryn_engine.audit_metadata.httpx.Client", lambda **_: transport_client)
    monkeypatch.setattr(graph, "scan_graph", lambda read, *_: read(0, 4))
    with pytest.raises(ValueError, match="ONNX_RANGE_UNAVAILABLE"):
        client.read_onnx_metadata("org/model", SHA, "model.onnx", {"model.onnx": 100})


def test_mflux_metadata_provider_is_actually_evaluated(tmp_path):
    from test_providers import Checkpoint

    checkpoint = Checkpoint()
    inventory = checkpoint.inventory
    client = MetadataClient()
    client._api.model_info.side_effect = lambda *_, **__: SimpleNamespace(
        sha=inventory["revision"],
        config=checkpoint.config,
        pipeline_tag="text-to-image",
        library_name="mflux",
        card_data={},
        gated=False,
        private=False,
        siblings=[
            SimpleNamespace(rfilename=f["name"], size=f["sizeBytes"], blob_id=f["blobId"])
            for f in inventory["files"]
        ],
    )
    client.read_json.side_effect = lambda _r, _v, f: json.loads(checkpoint.data[f])
    client.read_tensor_metadata = lambda _r, _v, f: checkpoint.headers[f]
    hub = MetadataSnapshot(tmp_path / "mflux.json", checkpoint.repository, "main", client=client)
    result = inspect_row(row(), hub)
    assert result["status"] == "metadata-compatible", result
    assert result["runtime"]["packId"] == "oryn.mflux-mlx"


def test_qwen_turbo_sampling_metadata_is_not_custom_code(tmp_path):
    fixture = json.loads((Path(__file__).parent / "fixtures/qwen-image-2.1-turbo.json").read_text())
    client = MetadataClient()
    client._api.model_info.side_effect = lambda *_, **__: SimpleNamespace(
        sha=fixture["revision"],
        config={},
        pipeline_tag="text-to-image",
        library_name="diffusers",
        gated=False,
        private=False,
        card_data={},
        siblings=[
            SimpleNamespace(rfilename=f["name"], size=f["size"], blob_id="a" * 40)
            for f in fixture["files"]
        ],
    )

    def read_fixture(_repository, _revision, filename):
        if filename == "model_index.json":
            return fixture["modelIndex"]
        if ".safetensors.index" in filename:
            parent = str(Path(filename).parent)
            return {
                "weight_map": {
                    f"tensor-{i}": Path(f["name"]).name
                    for i, f in enumerate(fixture["files"])
                    if str(Path(f["name"]).parent) == parent and f["name"].endswith(".safetensors")
                }
            }
        return {}

    client.read_json.side_effect = read_fixture
    hub = MetadataSnapshot(
        tmp_path / "turbo.json", fixture["repository"], fixture["revision"], client=client
    )
    result = inspect_row(row(), hub)
    assert result["status"] == "metadata-compatible", result
    assert result["executionVerified"] is False
    assert result["nextLayerReasons"] == ["CONFORMANCE_REQUIRED"]
    assert not result["inspectorProfile"]["requiresCustomCode"]


def test_inventory_rate_limit_retries_share_server_cooldown(monkeypatch):
    gate = Mock()
    monkeypatch.setattr("oryn_engine.audit_metadata.API_GATE", gate)
    client = BoundedAuditClient()
    response = httpx.Response(
        429,
        headers={"ratelimit": '"api";r=0;t=47'},
        request=httpx.Request("GET", "https://huggingface.co/api/models/org/model"),
    )
    error = httpx.HTTPStatusError("rate limited", request=response.request, response=response)
    client._api.model_info = Mock(side_effect=[error, SimpleNamespace(sha=SHA)])
    assert client.inventory_info("org/model", revision="main").sha == SHA
    assert gate.wait.call_count == 2
    gate.limited.assert_called_once_with(response.headers)
    client._api.model_info = Mock(side_effect=error)
    with pytest.raises(httpx.HTTPStatusError):
        client.inventory_info("org/model", revision="main")
    assert client._api.model_info.call_count == 3


def test_hub_gate_paces_workers_and_honors_reset_header(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("oryn_engine.audit_metadata.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "oryn_engine.audit_metadata.time.sleep",
        lambda delay: clock.__setitem__(0, clock[0] + delay),
    )
    gate = HubApiGate()
    gate.wait()
    gate.wait()
    assert clock[0] == pytest.approx(100.7)
    gate.limited({"ratelimit": '"api";r=0;t=47'})
    gate.wait()
    assert clock[0] == pytest.approx(148.7)


def test_metadata_projection_does_not_report_device_memory_failure(tmp_path, monkeypatch):
    from oryn_engine.models import ModelStore

    profile = {
        "executable": False,
        "blockingReasons": ["MODEL_CONFIG_MISSING", "MODEL_MEMORY_INSUFFICIENT"],
        "loadDisposition": "not-admissible-on-this-device",
        "runtimePlan": {"selected": {}},
        "eligibility": {
            "resources": {"state": "not-admissible", "reasons": ["MODEL_MEMORY_INSUFFICIENT"]}
        },
    }
    monkeypatch.setattr(ModelStore, "inspect", lambda *args, **kwargs: profile)
    store = CatalogInspectionStore(tmp_path / "store", snapshot(tmp_path))
    result = store.inspect({"provider": "huggingface", "repository": "org/model"})
    assert result["blockingReasons"] == ["MODEL_CONFIG_MISSING"]
    assert result["eligibility"]["resources"] == {"state": "not-evaluated", "reasons": []}
    assert result["executable"] is False and result["loadDisposition"] is None
    assert "runtimePlan" not in result


def test_json_rejection_version_tracks_parser_not_unrelated_model_routing(monkeypatch):
    from oryn_engine.audit_metadata import json_parser_digest
    from oryn_engine.models import HuggingFaceClient

    json_parser_digest.cache_clear()
    original = json_parser_digest()
    monkeypatch.setattr(HuggingFaceClient, "model_info", lambda *_: None)
    json_parser_digest.cache_clear()
    assert json_parser_digest() == original

    def different_parser(*_):
        return {}

    monkeypatch.setattr(HuggingFaceClient, "read_json", different_parser)
    json_parser_digest.cache_clear()
    assert json_parser_digest() != original
    json_parser_digest.cache_clear()
