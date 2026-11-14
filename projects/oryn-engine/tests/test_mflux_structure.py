from __future__ import annotations

import copy
import hashlib
import json
import struct
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
from safetensors.numpy import save

from oryn_engine.models import HuggingFaceClient, ModelStoreError

sys.path.insert(0, str(Path(__file__).parents[1] / "providers/mflux/src"))
from oryn_mflux_provider import inspection, worker  # noqa: E402
from oryn_mflux_provider.structure import matches_graph  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures/mflux_structure"
REVISION = "e" * 40


@pytest.fixture
def original():
    return json.loads((FIXTURES / "mlx-community--shuttle-jaguar-MLX-Q8.json").read_text())


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.stem)
def test_original_complete_headers_resolve_without_repository_hints(path):
    data = json.loads(path.read_text())
    inventory = data["inventory"]
    inventory.update(repository="someone/completely-renamed", card={}, config={})
    result = inspection.inspect_inventory(inventory, data["metadata"])
    (candidate,) = result["candidates"]
    assert candidate["plan"]["entryId"] == ("dev" if "lite" in path.name else "schnell")
    assert not candidate["diagnostics"]
    assert all(c["metadataCoverage"] == "complete-graph" for c in candidate["components"])
    assert set(n for c in candidate["components"] for n in c["files"]) == set(data["metadata"])
    # The committed fixture contains only original headers and their provenance.
    assert (
        len(json.dumps({"inventory": inventory, "metadata": data["metadata"]}).encode())
        < worker.MAX_MESSAGE_BYTES
    )
    assert {f["revision"] for f in data["provenance"]} == {inventory["revision"]}
    assert all(f["headerBytesRead"] < f["originalFileBytes"] for f in data["provenance"])


def test_complete_resolution_requests_every_header_in_one_bounded_round(original):
    result = inspection.inspect_inventory(original["inventory"], {})
    assert 1 < len(result["metadataNeeded"]) <= 24
    assert {r["file"] for r in result["metadataNeeded"]} == set(original["metadata"])
    assert all(r["kind"] == "safetensors-structure" for r in result["metadataNeeded"])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-component",
        "missing-shard",
        "extra-tensor",
        "missing-tensor",
        "shape",
        "dtype",
        "missing-scale",
        "scale-shape",
        "scale-dtype",
        "missing-affine-biases",
        "packed-shape",
        "wrong-bits",
        "inconsistent-version",
        "swift-version",
        "duplicate-tensor",
        "file-size",
        "layer-gap",
        "excess-layers",
        "indexed-subset",
        "wrong-task",
        "too-many-shards",
        "missing-output-bias-shape",
    ],
)
def test_incomplete_or_different_graph_is_not_inferred(original, mutation):
    inventory, metadata = original["inventory"], original["metadata"]
    first = next(n for n in metadata if n.startswith("transformer/"))
    header = metadata[first]
    tensors = header["tensors"]
    key = next(k for k, v in tensors.items() if v["dtype"] == "U32")
    if mutation in {"missing-component", "missing-shard"}:
        removed = (
            {n for n in metadata if n.startswith("vae/")}
            if mutation == "missing-component"
            else {first}
        )
        inventory["files"] = [f for f in inventory["files"] if f["name"] not in removed]
        for name in removed:
            metadata.pop(name)
    elif mutation == "extra-tensor":
        tensors["unreviewed.weight"] = {"shape": [1], "dtype": "F32"}
    elif mutation == "missing-tensor":
        tensors.pop(key)
    elif mutation in {"shape", "packed-shape"}:
        tensors[key]["shape"][-1] += 1
    elif mutation == "dtype":
        tensors[key]["dtype"] = "I32"
    elif mutation in {"missing-scale", "scale-shape", "scale-dtype", "missing-affine-biases"}:
        suffix = ".biases" if mutation == "missing-affine-biases" else ".scales"
        companion = key.removesuffix(".weight") + suffix
        owner = next(h["tensors"] for h in metadata.values() if companion in h["tensors"])
        if mutation.startswith("missing"):
            owner.pop(companion)
        elif mutation == "scale-shape":
            owner[companion]["shape"][-1] += 1
        else:
            owner[companion]["dtype"] = "U32"
    elif mutation == "wrong-bits":
        for h in metadata.values():
            h["metadata"]["quantization_level"] = "4"
    elif mutation == "inconsistent-version":
        header["metadata"]["mflux_version"] = "0.20.0"
    elif mutation == "swift-version":
        for h in metadata.values():
            h["metadata"]["flux_swift_version"] = h["metadata"].pop("mflux_version")
    elif mutation == "duplicate-tensor":
        second = next(h for n, h in metadata.items() if n.startswith("transformer/") and n != first)
        duplicate = next(iter(second["tensors"]))
        tensors[duplicate] = second["tensors"][duplicate]
    elif mutation == "file-size":
        next(f for f in inventory["files"] if f["name"] == first)["sizeBytes"] -= 1
    elif mutation in {"layer-gap", "excess-layers"}:
        for h in metadata.values():
            for name in list(h["tensors"]):
                if name.startswith("transformer_blocks.0."):
                    value = h["tensors"][name]
                    h["tensors"][name.replace(".0.", ".19.", 1)] = value
                    if mutation == "layer-gap":
                        del h["tensors"][name]
    elif mutation == "indexed-subset":
        inventory["files"].append(
            {"name": "transformer/model.safetensors.index.json", "sizeBytes": 100}
        )
    elif mutation == "wrong-task":
        inventory["task"] = "image-to-image"
    elif mutation == "too-many-shards":
        inventory["files"].extend(
            {"name": f"transformer/extra-{i}.safetensors", "sizeBytes": 100} for i in range(25)
        )
    elif mutation == "missing-output-bias-shape":
        owner = next(
            h["tensors"] for h in metadata.values() if "norm_out.linear.bias" in h["tensors"]
        )
        owner["norm_out.linear.bias"]["shape"] = [1]
    result = inspection.inspect_inventory(inventory, metadata)
    assert not result["candidates"]
    assert result["diagnostics"] == ["MODEL_ARCHITECTURE_UNRESOLVED"]


def tensor_header(raw):
    length = struct.unpack("<Q", raw[:8])[0]
    h = json.loads(raw[8 : 8 + length])
    tensors = {
        k: {f: v[f] for f in ("dtype", "shape")} for k, v in h.items() if k != "__metadata__"
    }
    return {
        "metadata": h.get("__metadata__", {}),
        "dtypes": sorted({v["dtype"] for v in tensors.values()}),
        "tensors": tensors,
        "headerBytes": 8 + length,
        "dataBytes": len(raw) - 8 - length,
    }


@pytest.fixture
def small_checkpoint(tmp_path, monkeypatch):
    catalog, _ = inspection.catalog()
    catalog = copy.deepcopy(catalog)
    graphs = catalog["nativeFluxGraphs"]
    graph = {"projection.weight": {"shape": [2, 2], "quantizable": True}}
    graphs["components"] = {d: copy.deepcopy(graph) for d in graphs["components"]}
    graphs["transformers"]["schnell"] = {
        "transformer_blocks.0.projection.weight": graph["projection.weight"],
        "single_transformer_blocks.0.projection.weight": graph["projection.weight"],
        "norm_out.linear.weight": {"shape": [4, 2], "quantizable": True},
    }
    graphs["transformers"]["dev"] = graphs["transformers"]["schnell"] | {
        "guidance.weight": graph["projection.weight"]
    }
    digest = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
    monkeypatch.setattr(inspection, "catalog", lambda: (catalog, digest))
    monkeypatch.setattr(worker, "catalog", lambda: (catalog, digest))
    files, metadata = [], {}
    for directory, graph in [
        *graphs["components"].items(),
        ("transformer", graphs["transformers"]["schnell"]),
    ]:
        parts = list(graph.items())
        for index, part in enumerate((parts[:1], parts[1:]) if len(parts) > 1 else (parts,)):
            name = f"{directory}/arbitrary-{index}.safetensors"
            raw = save(
                {k: np.zeros(v["shape"], np.float32) for k, v in part},
                metadata={"mflux_version": "0.7.1", "quantization_level": "None"},
            )
            path = tmp_path / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(raw)
            files.append({"name": name, "sizeBytes": len(raw)})
            metadata[name] = tensor_header(raw)
    for directory in ("tokenizer", "tokenizer_2"):
        path = tmp_path / directory / "tokenizer_config.json"
        path.parent.mkdir()
        path.write_text("{}")
        files.append({"name": path.relative_to(tmp_path).as_posix(), "sizeBytes": 2})
    inventory = {
        "repository": "someone/renamed",
        "revision": REVISION,
        "task": "text-to-image",
        "library": "mlx",
        "files": files,
        "config": {},
        "card": {},
    }
    (candidate,) = inspection.inspect_inventory(inventory, metadata)["candidates"]
    assert not candidate["diagnostics"]
    return {"modelRoot": str(tmp_path), "plan": candidate["plan"], "input": {"prompt": "example"}}


def test_prepare_rechecks_complete_graph_from_actual_headers(small_checkpoint):
    assert worker.prepare(small_checkpoint)["entry"]["id"] == "schnell"


@pytest.mark.parametrize(
    "mutation", ["shape", "extra-file", "changed-family", "changed-marker", "missing-file"]
)
def test_prepare_cannot_reuse_a_structural_offer_after_artifacts_change(small_checkpoint, mutation):
    payload = small_checkpoint
    root = Path(payload["modelRoot"])
    if mutation == "changed-family":
        payload["plan"]["entryId"] = "dev"
    elif mutation == "changed-marker":
        payload["plan"]["components"][0]["metadataCoverage"] = "first-shard"
    elif mutation == "extra-file":
        (root / "vae/additional.safetensors").write_bytes(
            save({"new.weight": np.zeros((2, 2), np.float32)})
        )
    elif mutation == "missing-file":
        next((root / "vae").glob("*.safetensors")).unlink()
    else:
        next((root / "vae").glob("*.safetensors")).write_bytes(
            save(
                {"projection.weight": np.zeros((3, 2), np.float32)},
                metadata={"mflux_version": "0.7.1", "quantization_level": "None"},
            )
        )
    with pytest.raises(ValueError):
        worker.prepare(payload)


@pytest.mark.parametrize("ignore_range", [False, True])
def test_structure_reader_never_reads_tensor_payload(monkeypatch, ignore_range):
    raw = save(
        {"projection.weight": np.zeros((128, 64), np.float32)}, metadata={"mflux_version": "0.7.1"}
    )
    requested = []

    def respond(request):
        start, end = map(int, request.headers["range"].removeprefix("bytes=").split("-"))
        requested.append((start, end))
        return httpx.Response(
            200 if ignore_range else 206,
            content=raw if ignore_range else raw[start : end + 1],
            headers={"content-range": f"bytes {start}-{end}/{len(raw)}"},
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(httpx, "stream", client.stream)
        result = HuggingFaceClient().read_tensor_structure(
            "someone/model", REVISION, "transformer/0.safetensors"
        )
    assert result == tensor_header(raw)
    assert requested == [(0, 7), (8, result["headerBytes"] - 1)]


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "shape-bool",
        "shape-zero",
        "dtype",
        "dtype-object",
        "overlap",
        "size",
        "extra-field",
        "invalid-name",
        "empty",
    ],
)
def test_structure_reader_rejects_malformed_tensor_declarations(monkeypatch, mutation):
    header = {"weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}
    tensor = header["weight"]
    if mutation == "shape-bool":
        tensor["shape"] = [True, 4]
    elif mutation == "shape-zero":
        tensor["shape"] = [0, 4]
    elif mutation == "dtype":
        tensor["dtype"] = "F8"
    elif mutation == "dtype-object":
        tensor["dtype"] = {}
    elif mutation == "overlap":
        header["another"] = copy.deepcopy(tensor)
    elif mutation == "size":
        tensor["data_offsets"] = [0, 15]
    elif mutation == "extra-field":
        tensor["execute"] = "never"
    elif mutation == "invalid-name":
        header["../weight"] = header.pop("weight")
    elif mutation == "empty":
        header = {}
    encoded = json.dumps(header).encode()
    if mutation == "duplicate":
        encoded = b'{"weight":{},"weight":{}}'
    raw = struct.pack("<Q", len(encoded)) + encoded

    def respond(request):
        start, end = map(int, request.headers["range"].removeprefix("bytes=").split("-"))
        return httpx.Response(
            206,
            content=raw[start : end + 1],
            headers={"content-range": f"bytes {start}-{end}/{len(raw)}"},
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(httpx, "stream", client.stream)
        with pytest.raises(ModelStoreError, match="Tensor metadata could not be read"):
            HuggingFaceClient().read_tensor_structure("someone/model", REVISION, "0.safetensors")


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("bits,group", [(None, 64), (4, 32), (4, 64), (8, 128)])
def test_native_loader_preserves_nonzero_output_bias_and_quantized_geometry(device, bits, group):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mflux")
    import mlx.nn as nn
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
    from mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous import (
        AdaLayerNormContinuous,
    )
    from mflux.models.flux.model.flux_transformer.transformer import Transformer
    from mlx.utils import tree_flatten

    def model():
        # Exercise the actual native output module in an isolated component.
        value = Transformer.__new__(Transformer)
        nn.Module.__init__(value)
        value.norm_out = AdaLayerNormContinuous(64, 128)
        return value

    with mx.stream(getattr(mx, device)):
        mx.random.seed(61282)
        source, loaded = model(), model()
        source.norm_out.linear.bias = mx.linspace(-0.3, 0.4, 128)
        expected = {
            name: {"shape": list(value.shape), "quantizable": name.endswith(".weight")}
            for name, value in tree_flatten(source.parameters())
        }
        if bits is not None:
            nn.quantize(source, bits=bits, group_size=group)
        dtype_names = {mx.float32: "F32", mx.float16: "F16", mx.bfloat16: "BF16", mx.uint32: "U32"}
        structure = {
            name: {"shape": list(value.shape), "dtype": dtype_names[value.dtype]}
            for name, value in tree_flatten(source.parameters())
        }
        assert matches_graph(structure, expected, bits)
        weights = LoadedWeights({"transformer": source.parameters()}, MetaData(bits, "0.7.1"))
        with worker.validated_weights():
            WeightApplier.apply_and_quantize_single(
                weights,
                loaded,
                ComponentDefinition(name="transformer", hf_subdir="transformer"),
                None,
            )
        x, context = mx.random.normal((1, 3, 64)), mx.random.normal((1, 128))
        reference = source.norm_out(x, context)
        actual = loaded.norm_out(x, context)
        assert bool(mx.allclose(actual, reference, atol=1e-6).item())
        # Independently evaluate layer normalization and affine modulation.
        linear = source.norm_out.linear
        weight = (
            linear.weight
            if bits is None
            else mx.dequantize(
                linear.weight, linear.scales, linear.biases, bits=bits, group_size=group
            )
        )
        modulation = nn.silu(context).astype(ModelConfig.precision) @ weight.T + linear.bias
        scale, shift = mx.split(modulation, 2, axis=-1)
        normalized = (x - mx.mean(x, axis=-1, keepdims=True)) * mx.rsqrt(
            mx.var(x, axis=-1, keepdims=True) + 1e-6
        )
        manual = normalized * (1 + scale[:, None]) + shift[:, None]
        assert bool(mx.allclose(actual, manual, atol=0.015).item())
        loaded.norm_out.linear.bias = mx.zeros((128,))
        assert float(mx.max(mx.abs(loaded.norm_out(x, context) - actual)).item()) > 0.1
