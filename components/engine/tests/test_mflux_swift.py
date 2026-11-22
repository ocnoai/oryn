"""Complete original metadata and native numerical checks for Swift MLX saves."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from test_mflux_structure import small_checkpoint  # noqa: F401

sys.path.insert(0, str(Path(__file__).parents[1] / "providers/mflux/src"))
from oryn_mflux_provider import inspection, worker  # noqa: E402
from oryn_mflux_provider.swift_layout import (  # noqa: E402
    remap_swift_tensors,
    swift_metadata,
)

FIXTURES = Path(__file__).parent / "fixtures/mflux_swift"


@pytest.fixture
def original():
    return json.loads((FIXTURES / "mzbac--flux1.schnell.4bit.mlx.json").read_text())


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.stem)
@pytest.mark.parametrize("renamed", [False, True])
def test_original_graph_and_every_companion_are_bound(path, renamed):
    data = json.loads(path.read_text())
    if renamed:
        data["inventory"]["repository"] = "someone/no-family-hint"
    (candidate,) = inspection.inspect_inventory(data["inventory"], data["metadata"])["candidates"]
    assert candidate["plan"]["entryId"] == ("dev" if ".dev." in path.name else "schnell")
    assert not candidate["diagnostics"]
    assert all(c["metadataCoverage"] == "complete-graph" for c in candidate["components"])
    assert {n for c in candidate["components"] for n in c["files"]} == set(data["metadata"])
    assert len(json.dumps(data).encode()) < worker.MAX_MESSAGE_BYTES


def test_named_schnell_requests_complete_headers_without_hint_bypass(original):
    result = inspection.inspect_inventory(original["inventory"], {})
    assert {r["file"] for r in result["metadataNeeded"]} == set(original["metadata"])
    assert {r["kind"] for r in result["metadataNeeded"]} == {"safetensors-structure"}


@pytest.mark.parametrize("value", [None, "flux_swift_version", {"flux_swift_version": "unknown"}])
def test_serializer_identity_is_read_only_from_tensor_headers(original, value):
    original["metadata"] = {"unrelated.json": {"metadata": value}, **original["metadata"]}
    (candidate,) = inspection.inspect_inventory(original["inventory"], original["metadata"])[
        "candidates"
    ]
    assert not candidate["diagnostics"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("flux_swift_version", "2.0.0"),
        ("flux_swift_version", None),
        ("quantization_level", "3"),
        ("quantization_level", "8"),
        ("group_size", "32"),
        ("group_size", "128"),
        ("quantization_method", "remote-code"),
        ("model_type", "vae"),
        ("chunk_index", "1"),
        ("total_chunks", "999"),
        ("chunk_name", "different"),
        ("quantization_date", None),
        ("custom_code", "run"),
    ],
)
def test_serialization_declarations_cannot_override_graph(original, field, value):
    name = next(n for n in original["metadata"] if n.startswith("transformer/"))
    original["metadata"][name]["metadata"][field] = value
    result = inspection.inspect_inventory(original["inventory"], original["metadata"])
    assert not result["candidates"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-bias",
        "missing-scale",
        "bad-convolution-layout",
        "missing-relative-bias",
        "duplicate-alias",
        "extra-weight",
        "missing-shard",
        "duplicate-tensor",
        "wrong-file-size",
        "mixed-layout",
        "wrong-relative-shape",
    ],
)
def test_graph_aliases_never_drop_or_invent_weights(original, mutation):
    metadata = original["metadata"]

    def owner(key):
        return next(h["tensors"] for h in metadata.values() if key in h["tensors"])

    if mutation in {"missing-bias", "missing-scale"}:
        suffix = ".biases" if mutation == "missing-bias" else ".scales"
        tensors = next(
            h["tensors"] for h in metadata.values() if any(k.endswith(suffix) for k in h["tensors"])
        )
        tensors.pop(next(k for k in tensors if k.endswith(suffix)))
    elif mutation == "bad-convolution-layout":
        owner("decoder.conv_in.weight")["decoder.conv_in.weight"]["shape"] = [512, 16, 3, 3]
    elif mutation in {"missing-relative-bias", "wrong-relative-shape"}:
        tensors = owner("relative_attention_bias.weight")
        if mutation == "missing-relative-bias":
            tensors.pop("relative_attention_bias.weight")
        else:
            tensors["relative_attention_bias.weight"]["shape"] = [16, 64]
    elif mutation == "duplicate-alias":
        tensors = owner("decoder.conv_in.weight")
        tensors["decoder.conv_in.conv2d.weight"] = tensors["decoder.conv_in.weight"]
    elif mutation == "extra-weight":
        tensors = next(iter(metadata.values()))["tensors"]
        tensors["unrelated.weight"] = {"shape": [1], "dtype": "F32"}
    elif mutation == "missing-shard":
        name = next(n for n in metadata if n.startswith("transformer/"))
        metadata.pop(name)
        original["inventory"]["files"] = [
            f for f in original["inventory"]["files"] if f["name"] != name
        ]
    elif mutation == "duplicate-tensor":
        headers = [h for n, h in metadata.items() if n.startswith("transformer/")]
        key = next(iter(headers[1]["tensors"]))
        headers[0]["tensors"][key] = headers[1]["tensors"][key]
    elif mutation == "wrong-file-size":
        next(f for f in original["inventory"]["files"] if f["name"] in metadata)["sizeBytes"] += 1
    elif mutation == "mixed-layout":
        header = metadata[next(n for n in metadata if n.startswith("vae/"))]
        header["metadata"] = {"mflux_version": "0.7.1", "quantization_level": "4"}
    assert not inspection.inspect_inventory(original["inventory"], metadata)["candidates"]


def test_shared_position_table_aliases_retain_the_same_array():
    aliases = inspection.catalog()[0]["swiftFluxAliases"]
    value = np.arange(32 * 64, dtype=np.float32).reshape(32, 64)
    result = remap_swift_tensors(
        {"relative_attention_bias.weight": value}, "text_encoder_2", aliases
    )
    assert len(result) == 24
    assert all(v is value for v in result.values())
    assert {k.split(".")[1] for k in result} == {str(i) for i in range(24)}
    with pytest.raises(ValueError, match="Colliding"):
        remap_swift_tensors(
            {"relative_attention_bias.weight": value, next(iter(result)): value},
            "text_encoder_2",
            aliases,
        )


def declaration(directory, bits, group):
    return {
        "chunk_index": "0",
        "chunk_name": "model",
        "flux_swift_version": "1.0.0",
        "group_size": str(group),
        "model_type": directory,
        "quantization_date": "2025-07-04T00:15:04Z",
        "quantization_level": str(bits),
        "quantization_method": "mlx-nn-quantize",
        "total_chunks": "1",
    }


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("bits,group", [(4, 64), (8, 128)])
@pytest.mark.parametrize(
    "directory,source,destination",
    [
        (
            "text_encoder_2",
            "encoder.block.0.layer.0.SelfAttention.q",
            "t5_blocks.0.attention.SelfAttention.q",
        ),
        ("transformer", "transformer_blocks.0.ff.net.0.proj", "transformer_blocks.0.ff.linear1"),
        (
            "text_encoder",
            "text_model.encoder.layers.0.self_attn.q_proj",
            "text_model.encoder.layers.0.self_attn.q_proj",
        ),
    ],
)
def test_worker_native_quantized_loading_preserves_every_value(
    tmp_path, device, bits, group, directory, source, destination
):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mflux")
    import mlx.nn as nn
    from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mlx.utils import tree_flatten

    with mx.stream(getattr(mx, device)):
        mx.random.seed(61285)
        reference = nn.QuantizedLinear.from_linear(
            nn.Linear(128, 64, bias=False), group_size=group, bits=bits
        )
        raw = {source + "." + k: v for k, v in tree_flatten(reference.parameters())}
        path = tmp_path / directory
        path.mkdir()
        mx.save_safetensors(
            str(path / "model.safetensors"), raw, declaration(directory, bits, group)
        )
        original_loader = WeightLoader._try_load_mflux_format
        with worker.validated_weights():
            parameters, stored_bits, _ = WeightLoader._try_load_mflux_format(path)
            flat = dict(tree_flatten(parameters))
            assert set(flat) == {
                destination + "." + k for k, _ in tree_flatten(reference.parameters())
            }
            for key, value in tree_flatten(reference.parameters()):
                np.testing.assert_array_equal(
                    np.array(value), np.array(flat[destination + "." + key])
                )
            # Apply native quantization reconstruction and strict loading to the
            # selected real linear operator after the component-level rename.
            loaded = nn.Module()
            loaded.linear = nn.Linear(128, 64, bias=False)
            local = {k.removeprefix(destination + "."): v for k, v in flat.items()}
            from mlx.utils import tree_unflatten

            weights = LoadedWeights(
                {"unit": {"linear": tree_unflatten(list(local.items()))}},
                MetaData(stored_bits, None),
            )
            WeightApplier.apply_and_quantize_single(
                weights, loaded, ComponentDefinition(name="unit", hf_subdir=directory), None
            )
            values = mx.random.normal((2, 128))
            np.testing.assert_allclose(
                np.array(loaded.linear(values)), np.array(reference(values)), atol=1e-6, rtol=1e-6
            )
        assert WeightLoader._try_load_mflux_format is original_loader


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_native_vae_convolution_layout_and_t5_position_bias(tmp_path, device):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mflux")
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_self_attention import (
        T5SelfAttention,
    )
    from mflux.models.flux.model.flux_vae.decoder.conv_in import ConvIn
    from mlx.utils import tree_flatten

    with mx.stream(getattr(mx, device)):
        mx.random.seed(61286)
        source, loaded = ConvIn(), ConvIn()
        path = tmp_path / "vae"
        path.mkdir()
        mx.save_safetensors(
            str(path / "model.safetensors"),
            {
                "decoder.conv_in." + k.removeprefix("conv2d."): v
                for k, v in tree_flatten(source.parameters())
            },
            declaration("vae", 4, 64),
        )
        with worker.validated_weights():
            values, _, _ = WeightLoader._try_load_mflux_format(path)
        loaded.update(values["decoder"]["conv_in"], strict=True)
        inputs = mx.random.normal((1, 16, 4, 4))
        np.testing.assert_allclose(
            np.array(loaded(inputs)), np.array(source(inputs)), atol=1e-6, rtol=1e-6
        )
        bias = mx.random.normal((32, 64))
        mapped = remap_swift_tensors(
            {"relative_attention_bias.weight": bias},
            "text_encoder_2",
            inspection.catalog()[0]["swiftFluxAliases"],
        )
        reference = T5SelfAttention()
        reference.relative_attention_bias.weight = bias
        for value in mapped.values():
            attention = T5SelfAttention()
            attention.relative_attention_bias.weight = value
            np.testing.assert_array_equal(
                np.array(attention._compute_bias(17)), np.array(reference._compute_bias(17))
            )


def test_empty_or_unsupported_serialization_is_rejected():
    assert swift_metadata([], "vae") is None
    assert (
        swift_metadata([{"metadata": declaration("vae", 4, 64) | {"group_size": "0"}}], "vae")
        is None
    )


@pytest.fixture
def swift_checkpoint(small_checkpoint, monkeypatch):  # noqa: F811 - imported pytest fixture
    payload = small_checkpoint
    root = Path(payload["modelRoot"])
    data, digest = inspection.catalog()
    # Keep the same generated complete graph; give it the reviewed serializer's
    # declarations to test prepare against actual files, including all shards.
    aliases = {}
    metadata = {}
    for component in payload["plan"]["components"]:
        directory = component["directory"]
        files = sorted((root / directory).glob("*.safetensors"))
        aliases[directory] = {}
        for index, file in enumerate(files):
            tensors = load_file(file)
            aliases[directory].update({k: [k] for k in tensors})
            header = declaration(directory, 4, 64) | {
                "chunk_name": file.stem,
                "chunk_index": str(index),
                "total_chunks": str(len(files)),
            }
            save_file(tensors, file, header)
            from test_mflux_structure import tensor_header

            metadata[file.relative_to(root).as_posix()] = tensor_header(file.read_bytes())
    data["swiftFluxAliases"] = aliases
    names = payload["plan"]["selectedFiles"]
    inventory = {
        "repository": "someone/schnell",
        "revision": "e" * 40,
        "task": "text-to-image",
        "library": "mlx",
        "config": {},
        "card": {},
        "files": [{"name": n, "sizeBytes": (root / n).stat().st_size} for n in names],
    }
    (candidate,) = inspection.inspect_inventory(inventory, metadata)["candidates"]
    assert not candidate["diagnostics"]
    payload["plan"] = candidate["plan"]
    return payload


def test_prepare_rechecks_actual_swift_shards(swift_checkpoint):
    assert worker.prepare(swift_checkpoint)["entry"]["id"] == "schnell"


@pytest.mark.parametrize("mutation", ["header", "shape", "extra", "file"])
def test_prepare_rejects_changed_swift_graph(swift_checkpoint, mutation):
    root = Path(swift_checkpoint["modelRoot"])
    path = next((root / "vae").glob("*.safetensors"))
    tensors = load_file(path)
    header = declaration("vae", 4, 64) | {"chunk_name": path.stem}
    if mutation == "header":
        header["flux_swift_version"] = "9.0.0"
    elif mutation == "shape":
        tensors[next(iter(tensors))] = np.zeros((3, 2), np.float32)
    elif mutation == "extra":
        tensors["unrelated.weight"] = np.zeros(1, np.float32)
    else:
        save_file(tensors, path.with_name("unexpected.safetensors"), header)
    save_file(tensors, path, header)
    with pytest.raises(ValueError):
        worker.prepare(swift_checkpoint)
