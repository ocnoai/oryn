from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save

from oryn_engine.jobs import JobExecutionError
from oryn_engine.models import ModelStore
from oryn_engine.providers import (
    ProviderRegistry,
    ProviderResult,
    bind_plan,
    validate_candidate,
    verify_catalog,
    verify_plan,
)
from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

PROVIDER_ROOT = Path(__file__).parents[1] / "providers/mflux/src"
sys.path.insert(0, str(PROVIDER_ROOT))
from oryn_mflux_provider.inspection import catalog, inspect_inventory  # noqa: E402
from oryn_mflux_provider.worker import prepare  # noqa: E402

REVISION = "d" * 40


class Checkpoint:
    """A real bounded SafeTensors checkpoint graph, with synthetic tensors."""

    def __init__(self, family="qwen-image-2.1", bits=8, repository="someone/renamed-copy"):
        self.entry = next(e for e in catalog()[0]["entries"] if e["id"] == family)
        self.repository = repository
        self.data = {}
        self.headers = {}
        for component in self.entry["components"]:
            prefix = component["directory"] + "/" if component["directory"] else ""
            component_bits = bits.get(component["name"]) if isinstance(bits, dict) else bits
            if component["skipQuantization"] and not isinstance(bits, dict):
                component_bits = None
            metadata = {"quantization_level": str(component_bits), "mflux_version": "0.20.0"}
            self.data[prefix + "0.safetensors"] = save(
                {"weight": np.zeros((2, 2), dtype=np.float32)}, metadata=metadata
            )
            self.headers[prefix + "0.safetensors"] = {"metadata": metadata, "dtypes": ["F32"]}
            self.data[prefix + "model.safetensors.index.json"] = json.dumps(
                {"weight_map": {"weight": "0.safetensors"}}
            ).encode()
        for tokenizer in self.entry["tokenizers"]:
            self.data[tokenizer["directory"] + "/tokenizer_config.json"] = b"{}"
            self.data[tokenizer["directory"] + "/tokenizer.json"] = b"{}"
        self.data["README.md"] = b"This repository name has no architecture or quantization hints."
        self.config = {"base_model": family}

    @property
    def inventory(self):
        return {
            "repository": self.repository,
            "revision": REVISION,
            "task": "text-to-image",
            "library": "mflux",
            "config": self.config,
            "card": {},
            "files": [
                {"name": n, "sizeBytes": len(v), "blobId": hashlib.sha256(v).hexdigest()}
                for n, v in self.data.items()
            ],
        }

    @property
    def metadata(self):
        return self.headers | {
            n: json.loads(v) for n, v in self.data.items() if n.endswith(".json")
        }

    def model_info(self, *_, **__):
        return SimpleNamespace(
            sha=REVISION,
            config=self.config,
            library_name="mflux",
            card_data={},
            gated=False,
            private=False,
            pipeline_tag="text-to-image",
            siblings=[
                SimpleNamespace(
                    rfilename=n,
                    size=len(v),
                    lfs=SimpleNamespace(sha256=hashlib.sha256(v).hexdigest()),
                )
                for n, v in self.data.items()
            ],
        )

    def read_json(self, repo, rev, name):
        assert (repo, rev) == (self.repository, REVISION)
        return json.loads(self.data[name])

    def read_tensor_metadata(self, repo, rev, name):
        assert (repo, rev) == (self.repository, REVISION)
        return self.headers[name]

    def download(self, repo, filename, revision, cache_dir, progress=None):
        path = (
            Path(cache_dir)
            / ("models--" + repo.replace("/", "--"))
            / "snapshots"
            / revision
            / filename
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.data[filename])
        return str(path)


class InProcessProvider:
    def __init__(self):
        self.commands = []

    def invoke(self, pack, runtime, command, payload, **_):
        self.commands.append(command)
        if command == "inspect":
            return inspect_inventory(payload["inventory"], payload["metadata"])
        if command == "prepare":
            prepared = prepare(payload)
            return {
                "ready": True,
                "catalogDigest": prepared["plan"]["catalogDigest"],
                "estimatedMemoryBytes": prepared["plan"]["estimatedMemoryBytes"],
            }
        raise AssertionError(command)


def provider_runtime():
    return SimpleNamespace(site_packages=PROVIDER_ROOT)


def setup_store(tmp_path, checkpoint):
    store = ModelStore(
        tmp_path,
        hub=checkpoint,
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 128 * 1024**3),
    )
    store._provider_process = InProcessProvider()
    store._pack_runtime_locator = SimpleNamespace(resolve=lambda _: provider_runtime())
    store._available_pack_backends = lambda backends: list(backends)
    store._executable_backends = lambda: ["mlx"]
    return store


@pytest.mark.parametrize("family", ["krea-2", "qwen-image", "qwen-image-2.1"])
@pytest.mark.parametrize("bits", [None, 4, 8])
def test_unlisted_checkpoint_and_stored_precision_share_provider(family, bits):
    checkpoint = Checkpoint(family, bits)
    pack = builtin_registry().active_pack("oryn.mflux-mlx")
    process = InProcessProvider()
    results = ProviderRegistry(
        builtin_registry(), SimpleNamespace(resolve=lambda _: provider_runtime()), process
    ).inspect(checkpoint.inventory, checkpoint, access=True)
    assert len(results) == 1
    candidate = results[0].candidate
    assert not candidate["diagnostics"]
    assert candidate["architecture"] == checkpoint.entry["architecture"]
    assert candidate["quantization"] == (None if bits is None else f"INT{bits}")
    assert "README.md" not in candidate["selectedFiles"]
    assert "inspect" in process.commands
    assert candidate["plan"]["catalogDigest"] == pack.provider["catalogSha256"]


def test_mixed_component_precision_is_not_inferred_from_repository_name():
    checkpoint = Checkpoint(
        bits={"transformer": 8, "text_encoder": 4}, repository="x/wrong-q2-label"
    )
    value = inspect_inventory(checkpoint.inventory, checkpoint.metadata)["candidates"][0]
    assert value["quantization"] == "INT4+INT8"
    assert {c["quantizationBits"] for c in value["components"]} == {None, 4, 8}


@pytest.mark.parametrize(
    "change,reason",
    [
        ("missing-shard", "MODEL_SUBSET_INCOMPLETE"),
        ("escaping-index", "MODEL_SHARDS_INVALID"),
        ("unknown-precision", "QUANTIZATION_UNSUPPORTED"),
        ("missing-header", "TENSOR_METADATA_UNAVAILABLE"),
    ],
)
def test_incompatible_graph_has_specific_diagnostic(change, reason):
    checkpoint = Checkpoint()
    weights = next(iter(checkpoint.headers))
    index = str(Path(weights).with_name("model.safetensors.index.json"))
    metadata = checkpoint.metadata
    if change == "missing-shard":
        del checkpoint.data[weights]
    elif change == "escaping-index":
        metadata[index] = {"weight_map": {"x": "../outside.safetensors"}}
    elif change == "unknown-precision":
        metadata[weights]["metadata"]["quantization_level"] = "1"
    else:
        metadata[weights] = {"unavailable": True}
    result = inspect_inventory(checkpoint.inventory, metadata)
    assert reason in result["candidates"][0]["diagnostics"]


def test_unknown_family_is_not_guessed_from_unrelated_architecture():
    checkpoint = Checkpoint()
    checkpoint.config = {"architectures": ["SomethingNew"]}
    result = inspect_inventory(checkpoint.inventory, checkpoint.metadata)
    assert result["candidates"] == []
    assert result["diagnostics"] == ["MODEL_ARCHITECTURE_UNRESOLVED"]


def test_catalog_and_saved_plan_are_bound_to_runtime_and_exact_files(tmp_path):
    checkpoint = Checkpoint()
    pack = builtin_registry().active_pack("oryn.mflux-mlx")
    candidate = inspect_inventory(checkpoint.inventory, checkpoint.metadata)["candidates"][0]
    bound = bind_plan(ProviderResult(pack, provider_runtime(), candidate), checkpoint.inventory)
    record = {
        "model": bound["model"],
        "files": [
            {"name": f["name"], "size": f["sizeBytes"], "blob_id": f["blobId"]}
            for f in bound["files"]
        ],
    }
    assert verify_plan(bound, pack, provider_runtime(), record) == bound
    changed = copy.deepcopy(record)
    changed["files"][0]["blob_id"] = "a" * 64
    with pytest.raises(JobExecutionError, match="artifact selection"):
        verify_plan(bound, pack, provider_runtime(), changed)
    changed = copy.deepcopy(bound)
    changed["providerPlan"]["mode"] = "unbounded"
    with pytest.raises(JobExecutionError, match="plan or runtime changed"):
        verify_plan(changed, pack, provider_runtime(), record)
    root = tmp_path / "oryn_mflux_provider"
    root.mkdir()
    (root / "catalog.json").write_text("{}")
    with pytest.raises(ValueError, match="signed build"):
        verify_catalog(pack, SimpleNamespace(site_packages=tmp_path))


def test_provider_cannot_select_files_or_tasks_outside_inventory():
    checkpoint = Checkpoint()
    pack = builtin_registry().active_pack("oryn.mflux-mlx")
    candidate = inspect_inventory(checkpoint.inventory, checkpoint.metadata)["candidates"][0]
    validate_candidate(candidate, checkpoint.inventory, pack)
    candidate["selectedFiles"].append("../code.py")
    with pytest.raises(ValueError, match="undeclared files"):
        validate_candidate(candidate, checkpoint.inventory, pack)


def test_install_prepares_actual_tensors_and_preserves_provider_plan_after_restart(tmp_path):
    checkpoint = Checkpoint()
    store = setup_store(tmp_path, checkpoint)
    model = {"provider": "huggingface", "repository": checkpoint.repository, "revision": REVISION}
    profile = store.inspect(model)
    assert profile["executable"]
    store.install("install_provider", model)
    assert "prepare" in store._provider_process.commands
    record = store._records[0]
    selected = record["runtimePlan"]["selected"]
    assert selected["quantization"] == "INT8"
    assert record["providerPlans"][selected["candidateId"]]["model"] == model
    assert record["generationProfile"] == profile["generationProfile"]
    store._assert_candidate_files_installed(record, selected)
    store._apply_plan_runtime_pack(record, record["runtimePlan"])
    restarted = setup_store(tmp_path, checkpoint)
    assert restarted._records[0]["providerPlans"] == record["providerPlans"]
    assert restarted._records[0]["generationProfile"] == profile["generationProfile"]


def test_preparation_rejects_changed_component_quantization(tmp_path):
    checkpoint = Checkpoint()
    candidate = inspect_inventory(checkpoint.inventory, checkpoint.metadata)["candidates"][0]
    for name, data in checkpoint.data.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    weights = next(iter(checkpoint.headers))
    (tmp_path / weights).write_bytes(
        save({"weight": np.zeros((2, 2), np.float32)}, metadata={"quantization_level": "4"})
    )
    with pytest.raises(ValueError, match="Stored quantization differs"):
        prepare(
            {"modelRoot": str(tmp_path), "plan": candidate["plan"], "input": {"prompt": "test"}}
        )


def legacy_checkpoint():
    checkpoint = Checkpoint(family="schnell", bits=4)
    first = next(iter(checkpoint.headers))
    index = str(Path(first).with_name("model.safetensors.index.json"))
    del checkpoint.data[index]
    second = str(Path(first).with_name("1.safetensors"))
    header = copy.deepcopy(checkpoint.headers[first])
    header["metadata"]["mflux_version"] = "0.6.2"
    checkpoint.headers[first] = copy.deepcopy(header)
    checkpoint.headers[second] = header
    for name, key in ((first, "weight"), (second, "another_weight")):
        checkpoint.data[name] = save(
            {key: np.zeros((2, 2), np.float32)}, metadata=header["metadata"]
        )
    return checkpoint, first, second


def test_legacy_numbered_shards_require_all_headers_and_prepare_safe_data(tmp_path):
    checkpoint, first, second = legacy_checkpoint()
    metadata = checkpoint.metadata
    del metadata[second]
    pending = inspect_inventory(checkpoint.inventory, metadata)
    assert {"file": second, "kind": "safetensors-header"} in pending["metadataNeeded"]
    candidate = inspect_inventory(checkpoint.inventory, checkpoint.metadata)["candidates"][0]
    assert not candidate["diagnostics"]
    component = next(c for c in candidate["components"] if first in c["files"])
    assert component["files"] == [first, second]
    assert component["metadataCoverage"] == "all-shards"
    for name, data in checkpoint.data.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    payload = {"modelRoot": str(tmp_path), "plan": candidate["plan"], "input": {"prompt": "test"}}
    prepare(payload)
    (tmp_path / second).write_bytes(
        save(
            {"another_weight": np.zeros((2, 2), np.float32)},
            metadata={"quantization_level": "4", "mflux_version": "0.20.0"},
        )
    )
    with pytest.raises(ValueError, match="inconsistent"):
        prepare(payload)


def test_legacy_family_uses_pinned_original_model_metadata_without_name_guessing():
    checkpoint, _, _ = legacy_checkpoint()
    checkpoint.config = {"model_type": "flux-rectified-flow"}
    checkpoint.data["config.json"] = json.dumps(
        {
            "model_type": "flux-rectified-flow",
            "original_model": "black-forest-labs/FLUX.1-dev",
        }
    ).encode()
    metadata = checkpoint.metadata
    del metadata["config.json"]
    result = inspect_inventory(checkpoint.inventory, metadata)
    assert result["metadataNeeded"] == [{"file": "config.json", "kind": "json"}]
    candidate = inspect_inventory(checkpoint.inventory, checkpoint.metadata)["candidates"][0]
    assert candidate["plan"]["entryId"] == "dev"
    assert "config.json" in candidate["selectedFiles"]
    assert not candidate["diagnostics"]


@pytest.mark.parametrize(
    "mutation", ["gap", "arbitrary", "version", "missing-version", "quantization"]
)
def test_legacy_shards_reject_ambiguous_or_inconsistent_metadata(mutation):
    checkpoint, _, second = legacy_checkpoint()
    if mutation in {"gap", "arbitrary"}:
        changed = str(
            Path(second).with_name("2.safetensors" if mutation == "gap" else "other.safetensors")
        )
        checkpoint.data[changed] = checkpoint.data.pop(second)
        checkpoint.headers[changed] = checkpoint.headers.pop(second)
    elif mutation == "missing-version":
        del checkpoint.headers[second]["metadata"]["mflux_version"]
    else:
        key = "mflux_version" if mutation == "version" else "quantization_level"
        checkpoint.headers[second]["metadata"][key] = "0.20.0" if mutation == "version" else "8"
    result = inspect_inventory(checkpoint.inventory, checkpoint.metadata)
    if mutation in {"gap", "arbitrary"}:
        # Alternate unindexed names require a complete graph even with a family
        # hint; these header-only malformed fixtures cannot establish that graph.
        assert not result["candidates"]
        assert result["diagnostics"] == ["MODEL_ARCHITECTURE_UNRESOLVED"]
    else:
        assert {"MODEL_SHARDS_INVALID", "MODEL_SUBSET_INCOMPLETE"} & set(
            result["candidates"][0]["diagnostics"]
        )


@pytest.mark.parametrize("bits", [None, 4, 8])
def test_native_mflux_legacy_shards_match_logits_and_reject_partial_graph(tmp_path, bits):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mflux")
    import mlx.nn as nn
    from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mlx.utils import tree_flatten, tree_unflatten
    from oryn_mflux_provider.worker import validated_weights

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(128, 64)

        def __call__(self, value):
            return self.projection(value)

    source = Tiny()
    if bits is not None:
        nn.quantize(source, bits=bits, group_size=64)
    flat = tree_flatten(source.parameters())
    for i, part in enumerate((flat[:1], flat[1:])):
        mx.save_safetensors(
            str(tmp_path / f"{i}.safetensors"),
            dict(part),
            metadata={"quantization_level": str(bits), "mflux_version": "0.6.2"},
        )
    tree, stored_bits, version = WeightLoader._try_load_mflux_format(tmp_path)
    assert (stored_bits, version) == (bits, "0.6.2")
    weights = LoadedWeights({"transformer": tree}, MetaData(stored_bits, version))
    loaded = Tiny()
    component = ComponentDefinition(name="transformer", hf_subdir="transformer")
    original = nn.Module.update
    with validated_weights():
        WeightApplier.apply_and_quantize_single(weights, loaded, component, None)
        values = mx.arange(256, dtype=mx.float32).reshape(2, 128) / 256
        assert bool(mx.allclose(source(values), loaded(values), atol=1e-6).item())
        # The bulk component path also validates after its quantization setup.
        WeightApplier._set_weights(weights, {"transformer": loaded})
        for bad in (
            flat[1:],
            flat + [("unreviewed", mx.zeros((1,)))],
            [(flat[0][0], mx.zeros((1,))), *flat[1:]],
        ):
            invalid = LoadedWeights({"transformer": tree_unflatten(bad)}, weights.meta_data)
            with pytest.raises(ValueError, match="weight graph"):
                WeightApplier._set_weights(invalid, {"transformer": loaded})
        with pytest.raises(ValueError, match="missing"):
            WeightApplier._set_weights(
                LoadedWeights({}, weights.meta_data), {"transformer": loaded}
            )
    assert nn.Module.update is original


def test_installed_provider_reinspection_reuses_cache_after_runtime_change(tmp_path):
    checkpoint = Checkpoint()
    store = setup_store(tmp_path, checkpoint)
    model = {"provider": "huggingface", "repository": checkpoint.repository, "revision": REVISION}
    store.install("install_original", model)
    record = store._records[0]
    original_id = record["installationId"]
    record["runtimeBuildDigest"] = "0" * 64
    record["providerPlans"] = {}
    profile = store.inspect(model)
    assert profile["executable"]
    store.installed_record(model)
    assert len(store._records) == 1
    assert store._records[0]["installationId"] == original_id
    assert store._records[0]["providerPlans"]


def test_provider_profile_conforms_to_public_protocol(tmp_path):
    from jsonschema import Draft202012Validator

    checkpoint = Checkpoint()
    store = setup_store(tmp_path, checkpoint)
    profile = store.inspect(
        {"provider": "huggingface", "repository": checkpoint.repository, "revision": REVISION}
    )
    schema_path = Path(__file__).parents[2] / "protocol/schemas/v1/protocol-v1.schema.json"
    schema = json.loads(schema_path.read_text())
    matching = [
        name
        for name, value in schema["$defs"].items()
        if "exactRevision" in value.get("properties", {})
        and "selectedFiles" in value.get("properties", {})
    ]
    assert len(matching) == 1
    Draft202012Validator({"$ref": "#/$defs/" + matching[0], "$defs": schema["$defs"]}).validate(
        profile
    )


@pytest.mark.parametrize("ignore_range", [False, True])
def test_tensor_header_fetch_is_bounded_when_server_ignores_range(monkeypatch, ignore_range):
    import httpx

    from oryn_engine.models import HuggingFaceClient

    checkpoint = Checkpoint()
    data = next(v for k, v in checkpoint.data.items() if k.endswith(".safetensors"))
    data += b"x" * 100000
    requests = []

    def respond(request):
        start, end = map(int, request.headers["range"].removeprefix("bytes=").split("-"))
        requests.append((start, end))
        if ignore_range:
            return httpx.Response(200, content=data)
        return httpx.Response(
            206,
            content=data[start : end + 1],
            headers={"content-range": f"bytes {start}-{end}/{len(data)}"},
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(httpx, "stream", client.stream)
    metadata = HuggingFaceClient().read_tensor_metadata("someone/model", REVISION, "0.safetensors")
    assert metadata["metadata"]["mflux_version"] == "0.20.0"
    assert len(requests) == 2
    assert sum(end - start + 1 for start, end in requests) < 1024


def test_reviewed_qwen3_encoder_prompt_is_literal_and_closed():
    from oryn_mflux_provider.prompt_binding import render

    text = "{{ unsafe() }} <|im_end|>"
    messages = [{"role": "user", "content": text}]
    prefix = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"
    assert render(messages, binding="qwen3-think-v1", enable_thinking=True) == prefix
    assert (
        render(messages, binding="qwen3-no-think-v1", enable_thinking=False)
        == prefix + "<think>\n\n</think>\n\n"
    )
    for options in ({"tokenize": True}, {"tools": []}, {"enable_thinking": False}):
        with pytest.raises(ValueError, match="binding changed"):
            render(messages, binding="qwen3-think-v1", **options)
    with pytest.raises(ValueError, match="binding changed"):
        render([{"role": "system", "content": text}], binding="qwen3-think-v1")


def test_flux2_editing_requires_complete_native_graph_and_owned_image(tmp_path):
    checkpoint = Checkpoint("flux2-klein-4b", 4)
    inventory = checkpoint.inventory | {"task": "image-to-image"}
    candidate = inspect_inventory(inventory, checkpoint.metadata)["candidates"][0]
    assert candidate["taskContract"] == "image-editing-v1"
    assert candidate["inputModalities"] == ["text", "image"]
    assert candidate["plan"]["entryId"] == "flux2-klein-4b.edit"
    validate_candidate(candidate, inventory, builtin_registry().active_pack("oryn.mflux-mlx"))
    for name, data in checkpoint.data.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    payload = {
        "modelRoot": str(tmp_path),
        "plan": candidate["plan"],
        "input": {"prompt": "Make it blue"},
    }
    assert prepare(payload)["entry"]["taskContract"] == "image-editing-v1"
    with pytest.raises(JobExecutionError):
        prepare(payload, generating=True)
    payload["input"]["imageArtifactId"] = "artifact_aaaaaaaaaaaaaaaaaaaaaaaa"
    assert prepare(payload, generating=True)["input"]["imageArtifactId"]
    incomplete = inventory | {
        "files": [f for f in inventory["files"] if f["name"] not in checkpoint.headers]
    }
    blocked = inspect_inventory(incomplete, checkpoint.metadata)["candidates"][0]
    assert "MODEL_SUBSET_INCOMPLETE" in blocked["diagnostics"]


@pytest.mark.parametrize(
    "status,code",
    [
        (401, "MODEL_ACCESS_RESTRICTED"),
        (403, "MODEL_ACCESS_RESTRICTED"),
        (404, "MODEL_NOT_FOUND"),
        (503, "MODEL_METADATA_UNAVAILABLE"),
    ],
)
def test_tensor_metadata_access_failures_do_not_become_format_failures(monkeypatch, status, code):
    import httpx

    from oryn_engine.models import HuggingFaceClient, ModelStoreError

    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(status)))
    monkeypatch.setattr(httpx, "stream", client.stream)
    with pytest.raises(ModelStoreError) as error:
        HuggingFaceClient().read_tensor_metadata("someone/model", REVISION, "model.safetensors")
    assert error.value.code == code


def test_null_safetensors_metadata_matches_native_absent_metadata(monkeypatch):
    import struct

    import httpx

    from oryn_engine.models import HuggingFaceClient

    header = json.dumps(
        {"__metadata__": None, "x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    data = struct.pack("<Q", len(header)) + header + b"\0" * 4

    def respond(request):
        start, end = map(int, request.headers["range"].removeprefix("bytes=").split("-"))
        return httpx.Response(
            206,
            content=data[start : end + 1],
            headers={"content-range": f"bytes {start}-{end}/{len(data)}"},
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(httpx, "stream", client.stream)
    assert HuggingFaceClient().read_tensor_metadata(
        "someone/model", REVISION, "model.safetensors"
    ) == {"metadata": {}, "dtypes": ["F32"]}


def test_final_metadata_fetch_is_evaluated_before_returning_offer():
    checkpoint = Checkpoint("qwen-image", 8)
    checkpoint.data["config.json"] = json.dumps(
        {"original_model": checkpoint.entry["aliases"][0]}
    ).encode()
    checkpoint.config = {}
    for component in checkpoint.entry["components"]:
        prefix = component["directory"] + "/"
        first, selected = prefix + "0.safetensors", prefix + "1.safetensors"
        checkpoint.data[selected] = checkpoint.data[first]
        checkpoint.headers[selected] = checkpoint.headers[first]
        checkpoint.data[prefix + "model.safetensors.index.json"] = json.dumps(
            {"weight_map": {"weight": "1.safetensors"}}
        ).encode()
    process = InProcessProvider()
    result = ProviderRegistry(
        builtin_registry(), SimpleNamespace(resolve=lambda _: provider_runtime()), process
    ).inspect(checkpoint.inventory, checkpoint, access=True)
    assert process.commands == ["inspect"] * 4
    assert len(result) == 1 and not result[0].candidate["diagnostics"]
    assert not any(name.endswith("/0.safetensors") for name in result[0].candidate["selectedFiles"])
    assert all(
        component["files"][0].endswith("/1.safetensors")
        for component in result[0].candidate["components"]
    )


def test_repeated_metadata_request_without_progress_is_rejected():
    checkpoint = Checkpoint()
    name = next(n for n in checkpoint.data if n.endswith("model.safetensors.index.json"))

    class RepeatingProvider:
        def invoke(self, *args, **kwargs):
            return {"candidates": [], "metadataNeeded": [{"file": name, "kind": "json"}]}

    registry = ProviderRegistry(
        builtin_registry(),
        SimpleNamespace(resolve=lambda _: provider_runtime()),
        RepeatingProvider(),
    )
    with pytest.raises(ValueError, match="made no progress"):
        registry.inspect(checkpoint.inventory, checkpoint, access=True)


def test_structural_metadata_crosses_provider_boundary_without_payload_reads():
    fixture = (
        Path(__file__).parent / "fixtures/mflux_structure/mlx-community--shuttle-jaguar-MLX-Q8.json"
    )
    data = json.loads(fixture.read_text())
    reads = []

    def read_structure(repo, revision, name):
        assert (repo, revision) == (data["inventory"]["repository"], data["inventory"]["revision"])
        reads.append(name)
        return data["metadata"][name]

    process = InProcessProvider()
    results = ProviderRegistry(
        builtin_registry(), SimpleNamespace(resolve=lambda _: provider_runtime()), process
    ).inspect(data["inventory"], SimpleNamespace(read_tensor_structure=read_structure), access=True)
    assert process.commands == ["inspect", "inspect"]
    assert set(reads) == set(data["metadata"])
    assert len(results) == 1 and not results[0].candidate["diagnostics"]
    assert results[0].candidate["plan"]["entryId"] == "schnell"


def test_final_evaluation_does_not_expand_metadata_fetch_budget(monkeypatch):
    checkpoint = Checkpoint()
    names = [n for n in checkpoint.data if n.endswith(".json")][:4]
    assert len(names) == 4
    reads = []
    original = checkpoint.read_json

    def read_json(repo, revision, name):
        reads.append(name)
        return original(repo, revision, name)

    monkeypatch.setattr(checkpoint, "read_json", read_json)

    class ExpandingProvider:
        calls = 0

        def invoke(self, *args, **kwargs):
            name = names[self.calls]
            self.calls += 1
            return {"candidates": [], "metadataNeeded": [{"file": name, "kind": "json"}]}

    process = ExpandingProvider()
    registry = ProviderRegistry(
        builtin_registry(), SimpleNamespace(resolve=lambda _: provider_runtime()), process
    )
    with pytest.raises(ValueError, match="did not converge"):
        registry.inspect(checkpoint.inventory, checkpoint, access=True)
    assert process.calls == 4
    assert reads == names[:3]
