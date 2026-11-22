"""Independent SegFormer encoder/head parity and shared segmentation regression."""

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from oryn_engine import hdf5_segformer, hdf5_tasks, shared_tasks
from oryn_engine.model_layouts import assess_layout

FIXTURE = Path(__file__).parent / "fixtures/hdf5/segformer"
TASK = "image-segmentation"


def fixture(root):
    from transformers import SegformerImageProcessor

    shutil.copytree(FIXTURE, root, dirs_exist_ok=True)
    SegformerImageProcessor(
        size={"height": 32, "width": 32}, do_reduce_labels=True
    ).save_pretrained(root)
    return json.loads((root / "config.json").read_text())


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_encoder_and_segmentation_head_match_independent_tf(tmp_path, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    fixture(tmp_path)
    model, _, _ = shared_tasks.load_task(tmp_path, TASK, backend)
    assert int(model.decode_head.batch_norm.num_batches_tracked) == 0
    with np.load(tmp_path / "reference.npz", allow_pickle=False) as ref, torch.inference_mode():
        output = model(
            pixel_values=torch.tensor(ref["pixel_values"], device=backend),
            output_hidden_states=True,
        )
        for i, value in enumerate(output.hidden_states):
            torch.testing.assert_close(
                value.cpu(), torch.tensor(ref[f"encoder_{i}"]), atol=4e-5, rtol=4e-4
            )
        torch.testing.assert_close(
            output.logits.cpu(), torch.tensor(ref["logits"]), atol=4e-5, rtol=4e-4
        )


@pytest.mark.parametrize(
    "part",
    [
        "embedding",
        "reduction",
        "depthwise",
        "projection",
        "head",
        "statistics",
        "negative_variance",
        "extra",
        "shape",
    ],
)
def test_no_missing_or_invalid_graph_component_is_admitted(tmp_path, part):
    fixture(tmp_path)
    with h5py.File(tmp_path / "tf_model.h5", "r+") as f:
        names = []
        f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
        if part == "extra":
            f.create_dataset("rogue/kernel:0", data=np.zeros((2, 3), np.float32))
        else:
            fragment = {
                "embedding": "patch_embeddings.0/proj/kernel",
                "reduction": "attention/self/sr/kernel",
                "depthwise": "dwconv/dwconv/kernel",
                "projection": "linear_c.0/proj/kernel",
                "head": "classifier/kernel",
                "statistics": "moving_mean",
                "negative_variance": "moving_variance",
                "shape": "classifier/bias",
            }[part]
            name = next(n for n in names if fragment in n)
            if part == "negative_variance":
                f[name][...] = -1
            else:
                del f[name]
                if part == "shape":
                    f.create_dataset(name, data=np.zeros(9, np.float32))
    with pytest.raises(ValueError, match="HDF5_WEIGHT_"):
        hdf5_tasks.load_native(tmp_path, TASK)


def test_shared_segmentation_retains_original_pixel_coordinates_and_labels(tmp_path):
    config = fixture(tmp_path)
    image_path = tmp_path / "input.png"
    Image.new("RGB", (24, 19), (20, 90, 150)).save(image_path)
    output_path = tmp_path / "mask.png"
    result = shared_tasks.execute_task(
        tmp_path,
        TASK,
        {"imageArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa"},
        "cpu",
        image_path,
        output_path,
        lambda *_: None,
    )
    assert (result["width"], result["height"]) == (24, 19)
    assert sum(s["pixelCount"] for s in result["metadata"]["segments"]) == 24 * 19
    mask = np.asarray(Image.open(output_path))
    assert mask.shape == (19, 24)
    for segment in result["metadata"]["segments"]:
        assert segment["label"] == config["id2label"][str(segment["id"])]
        assert int((mask == segment["id"]).sum()) == segment["pixelCount"]
    layout = assess_layout([p.name for p in tmp_path.iterdir()], config, TASK)
    assert not layout.artifact_reasons and not layout.runtime_reasons
    assert "tf_model.h5" in layout.selected_files


@pytest.mark.parametrize(
    "change",
    [
        {"size": 16},
        {"size": 10000},
        {"resample": 3},
        {"reduce_labels": False},
        {"image_std": [1, 0, 1]},
    ],
)
def test_closed_processor_rejects_bad_geometry_and_conflicting_options(tmp_path, change):
    config = fixture(tmp_path)
    raw = json.loads((tmp_path / "preprocessor_config.json").read_text())
    with pytest.raises(ValueError, match="PROCESSOR|preprocessing dimensions"):
        hdf5_segformer.processor_options(raw | change, config)


@pytest.mark.parametrize(
    "change",
    [
        {"hidden_sizes": [8, 16]},
        {"num_attention_heads": [3, 2, 3, 4]},
        {"id2label": {"0": "only"}},
        {"reshape_last_stage": False},
    ],
)
def test_unsupported_model_geometry_is_rejected(tmp_path, change):
    config = fixture(tmp_path)
    with pytest.raises(ValueError):
        hdf5_tasks.validate_config(config | change, TASK)
