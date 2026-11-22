"""Indexed native Krea metadata, complete generated components and preparation."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from test_mflux_structure import tensor_header

sys.path.insert(0, str(Path(__file__).parents[1] / "providers/mflux/src"))
from oryn_mflux_provider import inspection, worker  # noqa: E402
from oryn_mflux_provider.krea_layout import (  # noqa: E402
    DERIVED_FREQUENCY,
    inspect_saved_krea,
    normalized_shapes,
)
from oryn_mflux_provider.structure import matches_graph  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures/krea_layout/mh73772--krea-2-turbo-mflux-q8-4step.json"


@pytest.fixture
def original():
    return json.loads(FIXTURE.read_text())


def test_all_original_graphs_match_without_weight_payload(original):
    catalog, _ = inspection.catalog()
    assert len(original["provenance"]) == 68
    assert sum(x["tensorCount"] for x in original["provenance"]) == 2052
    for component, graph in catalog["nativeKreaGraphs"].items():
        tensors = {
            k: v
            for n, h in original["metadata"].items()
            if n.startswith(component + "/")
            for k, v in h.get("tensors", {}).items()
        }
        assert matches_graph(normalized_shapes(tensors, graph, component), graph, 8)
    original["inventory"].update(
        repository="someone/renamed", card={"base_model": "krea/Krea-2-Turbo"}
    )
    (candidate,) = inspection.inspect_inventory(original["inventory"], original["metadata"])[
        "candidates"
    ]
    assert not candidate["diagnostics"]
    assert all(c["metadataCoverage"] == "complete-index-graph" for c in candidate["components"])


def test_index_inspection_stays_within_one_bounded_metadata_round(original):
    result = inspection.inspect_inventory(original["inventory"], {})
    requests = result["metadataNeeded"]
    assert len(requests) == 6
    assert {r["kind"] for r in requests} == {"json", "safetensors-structure"}
    selected = {r["file"]: original["metadata"][r["file"]] for r in requests}
    (candidate,) = inspection.inspect_inventory(original["inventory"], selected)["candidates"]
    assert not candidate["diagnostics"]


@pytest.mark.parametrize(
    "mutation",
    [
        "index-missing-key",
        "index-extra-key",
        "index-missing-scale",
        "index-escape",
        "index-wrong-shard",
        "missing-shard",
        "extra-shard",
        "root-ambiguity",
        "missing-index",
        "wrong-task",
        "too-many-shards",
        "shape",
        "dtype",
        "scale-shape",
        "version",
        "bits",
        "file-size",
        "duplicate-key",
        "changed-normalization",
    ],
)
def test_missing_or_altered_graph_has_no_candidate(original, mutation):
    inv, metadata = original["inventory"], original["metadata"]
    index = metadata["transformer/model.safetensors.index.json"]["weight_map"]
    key = next(k for k in index if k.endswith(".scales"))
    first = next(
        n for n in sorted(metadata) if n.startswith("transformer/") and n.endswith(".safetensors")
    )
    header = metadata[first]
    if mutation == "index-missing-key":
        index.pop(next(iter(index)))
    elif mutation == "index-extra-key":
        index["unreviewed.weight"] = next(iter(index.values()))
    elif mutation == "index-missing-scale":
        index.pop(key)
    elif mutation == "index-escape":
        index[key] = "../escaped.safetensors"
    elif mutation == "index-wrong-shard":
        index[key] = next(v for v in index.values() if v != index[key])
    elif mutation in {"missing-shard", "missing-index"}:
        name = first if mutation == "missing-shard" else "text_encoder/model.safetensors.index.json"
        inv["files"] = [f for f in inv["files"] if f["name"] != name]
        metadata.pop(name)
    elif mutation in {"extra-shard", "root-ambiguity", "too-many-shards"}:
        names = (
            [f"transformer/extra-{i}.safetensors" for i in range(129)]
            if mutation == "too-many-shards"
            else [
                "extra.safetensors"
                if mutation == "root-ambiguity"
                else "transformer/extra.safetensors"
            ]
        )
        inv["files"].extend({"name": n, "sizeBytes": 100} for n in names)
    elif mutation == "wrong-task":
        inv["task"] = "image-to-image"
    elif mutation in {"shape", "dtype", "scale-shape"}:
        tensors = header["tensors"]
        name = next(
            k
            for k, v in tensors.items()
            if (k.endswith(".scales") if mutation == "scale-shape" else v["dtype"] == "U32")
        )
        if mutation == "dtype":
            tensors[name]["dtype"] = "I32"
        else:
            tensors[name]["shape"][-1] += 1
    elif mutation in {"version", "bits"}:
        header["metadata"]["mflux_version" if mutation == "version" else "quantization_level"] = (
            "unknown"
        )
    elif mutation == "file-size":
        next(f for f in inv["files"] if f["name"] == first)["sizeBytes"] -= 1
    elif mutation == "duplicate-key":
        other = next(
            h
            for n, h in metadata.items()
            if n != first and n.startswith("transformer/") and "tensors" in h
        )
        name = next(iter(other["tensors"]))
        header["tensors"][name] = other["tensors"][name]
    elif mutation == "changed-normalization":
        vae = next(
            h["tensors"] for n, h in metadata.items() if n.startswith("vae/") and "tensors" in h
        )
        name = next(k for k in vae if k.endswith("norm1.weight"))
        vae[name]["shape"] = [vae[name]["shape"][0], 2]
    assert inspect_saved_krea(inv, metadata, inspection.catalog()[0])[0] is None


def save_native(root, directory, tensors, bits):
    mx = pytest.importorskip("mlx.core")
    path = root / directory
    path.mkdir()
    names = sorted(tensors)
    groups = [names[: len(names) // 2], names[len(names) // 2 :]]
    index = {}
    for i, keys in enumerate(groups):
        name = f"part-{i}.safetensors"
        mx.save_safetensors(
            str(path / name),
            {k: tensors[k] for k in keys},
            {"mflux_version": "16gb-bench", "quantization_level": str(bits)},
        )
        index.update({k: name for k in keys})
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("component", ["transformer", "text_encoder"])
def test_complete_native_components_load_from_indexes_and_match_outputs(
    tmp_path, device, bits, component
):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mflux")
    import mlx.nn as nn
    from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder
    from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
    from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
    from mlx.utils import tree_flatten

    def model():
        if component == "transformer":
            return Krea2Transformer(
                features=128,
                tdim=32,
                txtdim=128,
                heads=8,
                kvheads=2,
                layers=2,
                txtlayers=12,
                txtheads=8,
                txtkvheads=2,
            )
        return Krea2TextEncoder(
            vocab_size=256,
            hidden_size=128,
            num_hidden_layers=36,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            head_dim=16,
        )

    with mx.stream(getattr(mx, device)):
        mx.random.seed(61287)
        source, loaded = model(), model()
        nn.quantize(
            source,
            group_size=64,
            bits=bits,
            class_predicate=Krea2WeightDefinition.quantization_predicate,
        )
        parameters = dict(tree_flatten(source.parameters()))
        parameters.pop(DERIVED_FREQUENCY, None)
        save_native(tmp_path, component, parameters, bits)
        definition = next(c for c in Krea2WeightDefinition.get_components() if c.name == component)
        previous = WeightLoader._load_component
        with worker.validated_weights():
            values, stored, version = WeightLoader._load_component(
                tmp_path, definition, {}, definition.hf_subdir
            )
            weights = LoadedWeights({component: values}, MetaData(stored, version))
            WeightApplier.apply_and_quantize(
                weights, {component: loaded}, None, Krea2WeightDefinition
            )
        assert WeightLoader._load_component is previous
        actual = dict(tree_flatten(loaded.parameters()))
        assert set(actual) == set(dict(tree_flatten(source.parameters())))
        for k, v in parameters.items():
            np.testing.assert_array_equal(np.array(actual[k]), np.array(v))
        if component == "transformer":
            inputs = mx.random.normal((1, 16, 4, 4))
            time = mx.array([0.6])
            context = mx.random.normal((1, 3, 12 * 128))
            expected = source(inputs, time, context)
            output = loaded(inputs, time, context)
        else:
            inputs = mx.array([[1, 2, 3, 4], [5, 6, 0, 0]])
            mask = mx.array([[1, 1, 1, 1], [1, 1, 0, 0]])
            expected = source.get_prompt_embeds(inputs, mask)
            output = loaded.get_prompt_embeds(inputs, mask)
            assert output.shape == (2, 4, 12 * 128)
        assert np.isfinite(np.array(output)).all()
        np.testing.assert_allclose(np.array(output), np.array(expected), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("images", [True, False])
def test_native_vae_channel_scales_preserve_values_and_broadcasting(device, images):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mflux")
    import mlx.nn as nn
    from mflux.models.qwen.model.qwen_vae.qwen_image_rms_norm import QwenImageRMSNorm
    from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
    from oryn_mflux_provider.krea_layout import native_parameters

    with mx.stream(getattr(mx, device)):
        model = QwenVAE.__new__(QwenVAE)
        nn.Module.__init__(model)
        model.encoder = {"norm_out": QwenImageRMSNorm(8, images=images)}
        weight = mx.linspace(-0.4, 0.7, 8)
        values = native_parameters(model, {"encoder": {"norm_out": {"weight": weight}}})
        model.update(values, strict=True)
        inputs = (
            mx.arange(2 * 8 * 3 * 3 * (1 if images else 2), dtype=mx.float32).reshape(
                (2, 8, 3, 3) if images else (2, 8, 2, 3, 3)
            )
            / 100
        )
        x = np.array(inputs)
        w = np.array(weight).reshape((1, 8) + ((1, 1) if images else (1, 1, 1)))
        expected = (
            x / np.maximum(np.sqrt(np.square(x).sum(axis=1, keepdims=True)), 1e-12) * np.sqrt(8) * w
        )
        np.testing.assert_allclose(
            np.array(model.encoder["norm_out"](inputs)), expected, atol=2e-6, rtol=2e-5
        )


@pytest.mark.parametrize("damage", ["missing-learned", "extra", "frequency", "frequency-shape"])
def test_derived_frequency_cannot_hide_missing_or_changed_parameters(damage):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mflux")
    from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder
    from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
    from mlx.utils import tree_flatten, tree_unflatten

    model = Krea2TextEncoder(
        vocab_size=32,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        head_dim=16,
    )
    values = dict(tree_flatten(model.parameters()))
    values.pop(DERIVED_FREQUENCY)
    if damage == "missing-learned":
        values.pop("norm.weight")
    elif damage == "extra":
        values["extra.weight"] = mx.zeros(1)
    elif damage == "frequency":
        values[DERIVED_FREQUENCY] = mx.zeros(8)
    else:
        values[DERIVED_FREQUENCY] = mx.zeros(3)
    with worker.validated_weights(), pytest.raises(ValueError):
        WeightApplier.apply_and_quantize(
            LoadedWeights(
                {"text_encoder": tree_unflatten(list(values.items()))}, MetaData(None, None)
            ),
            {"text_encoder": model},
            None,
            Krea2WeightDefinition,
        )


@pytest.fixture
def small_checkpoint(tmp_path, monkeypatch):
    data, _ = inspection.catalog()
    data = copy.deepcopy(data)
    graphs = {
        "transformer": {
            "block0.weight": {"shape": [2, 2], "quantizable": False},
            "block1.weight": {"shape": [2, 2], "quantizable": False},
        },
        "text_encoder": {
            "embed_tokens.weight": {"shape": [2, 2], "quantizable": False},
            "norm.weight": {"shape": [2], "quantizable": False},
            DERIVED_FREQUENCY: {"shape": [1], "quantizable": False},
        },
        "vae": {"encoder.norm_out.weight": {"shape": [2, 1, 1, 1], "quantizable": False}},
    }
    data["nativeKreaGraphs"] = graphs
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    monkeypatch.setattr(inspection, "catalog", lambda: (data, digest))
    monkeypatch.setattr(worker, "catalog", lambda: (data, digest))
    files = []
    metadata = {}
    for directory, graph in graphs.items():
        path = tmp_path / directory
        path.mkdir()
        index = {}
        for i, (name, description) in enumerate(
            (k, v) for k, v in graph.items() if k != DERIVED_FREQUENCY
        ):
            file = path / f"part-{i}.safetensors"
            shape = [2] if directory == "vae" else description["shape"]
            save_file(
                {name: np.ones(shape, np.float32)},
                file,
                {"mflux_version": "16gb-bench", "quantization_level": "8"},
            )
            index[name] = file.name
            relative = file.relative_to(tmp_path).as_posix()
            files.append({"name": relative, "sizeBytes": file.stat().st_size})
            metadata[relative] = tensor_header(file.read_bytes())
        file = path / "model.safetensors.index.json"
        file.write_text(json.dumps({"weight_map": index}))
        relative = file.relative_to(tmp_path).as_posix()
        files.append({"name": relative, "sizeBytes": file.stat().st_size})
        metadata[relative] = {"weight_map": index}
    file = tmp_path / "tokenizer/tokenizer_config.json"
    file.parent.mkdir()
    file.write_text("{}")
    files.append({"name": "tokenizer/tokenizer_config.json", "sizeBytes": 2})
    inv = {
        "repository": "someone/krea-2",
        "revision": "e" * 40,
        "task": "text-to-image",
        "library": "mflux",
        "config": {},
        "card": {},
        "files": files,
    }
    (candidate,) = inspection.inspect_inventory(inv, metadata)["candidates"]
    assert not candidate["diagnostics"]
    return {"modelRoot": str(tmp_path), "plan": candidate["plan"], "input": {"prompt": "example"}}


def test_worker_checks_every_installed_header_and_index(small_checkpoint):
    assert worker.prepare(small_checkpoint)["entry"]["id"] == "krea-2"


@pytest.mark.parametrize(
    "damage", ["unseen-shape", "unseen-dtype", "index-key", "index-shard", "extra-shard", "marker"]
)
def test_worker_rejects_changes_beyond_the_sampled_metadata(small_checkpoint, damage):
    root = Path(small_checkpoint["modelRoot"])
    path = root / "transformer/part-1.safetensors"
    if damage in {"unseen-shape", "unseen-dtype", "extra-shard"}:
        values = load_file(path)
        key = next(iter(values))
        values[key] = np.ones(
            (3, 2) if damage == "unseen-shape" else (2, 2),
            np.int32 if damage == "unseen-dtype" else np.float32,
        )
        if damage == "extra-shard":
            path = path.with_name("extra.safetensors")
        save_file(values, path, {"mflux_version": "16gb-bench", "quantization_level": "8"})
    elif damage in {"index-key", "index-shard"}:
        path = root / "transformer/model.safetensors.index.json"
        index = json.loads(path.read_text())
        key = next(iter(index["weight_map"]))
        if damage == "index-key":
            index["weight_map"].pop(key)
        else:
            index["weight_map"][key] = "part-1.safetensors"
        path.write_text(json.dumps(index))
    else:
        small_checkpoint["plan"]["components"][0]["metadataCoverage"] = "first-shard"
    with pytest.raises(ValueError):
        worker.prepare(small_checkpoint)
