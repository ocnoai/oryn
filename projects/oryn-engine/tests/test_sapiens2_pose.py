"""Native tiny Sapiens2 graphs with the full 308-joint output vocabulary."""

import copy
import json

import pytest
import torch
from PIL import Image

from oryn_engine import pose_tasks, sapiens2_pose, shared_tasks


def tiny_model(root):
    from transformers import Sapiens2Config, Sapiens2ForPoseEstimation, Sapiens2ImageProcessor

    config = Sapiens2Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_register_tokens=2,
        num_key_value_heads_per_layer=[1],
        image_size=[64, 64],
        patch_size=16,
        num_labels=308,
        flip_pairs=sapiens2_pose.joint_metadata()["flipPairs"],
        head_config={
            "upsample_out_channels": [8, 8],
            "upsample_kernel_sizes": [4, 4],
            "conv_out_channels": [8, 8, 8],
            "conv_kernel_sizes": [1, 1, 1],
        },
    )
    torch.manual_seed(61)
    model = Sapiens2ForPoseEstimation(config).eval()
    with torch.no_grad():
        model.decode_head.predictor.bias.fill_(0.2)
    model.save_pretrained(root)
    processor = Sapiens2ImageProcessor(size={"height": 64, "width": 64})
    processor.save_pretrained(root)
    return model, processor


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_native_heatmaps_and_decoding(tmp_path, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    native, original_processor = tiny_model(tmp_path)
    model, processor, config = shared_tasks.load_task(tmp_path, "pose-estimation", backend)
    image = Image.new("RGB", (160, 120), "red")
    boxes = [[[16, 12, 128, 96]]]
    inputs = original_processor(images=image, boxes=boxes, return_tensors="pt")
    with torch.inference_mode():
        reference = native(**inputs)
        actual = model(pixel_values=inputs["pixel_values"].to(backend))
        torch.testing.assert_close(actual.heatmaps.cpu(), reference.heatmaps, atol=2e-5, rtol=2e-4)
        decoded = original_processor.post_process_pose_estimation(reference, boxes=boxes)[0][0]
        result = pose_tasks.execute(
            model, processor, config, image, {"boxes": [[0.1, 0.1, 0.9, 0.9]]}, backend
        )
    assert result["width"] == 160 and result["height"] == 120
    points = result["poses"][0]["keypoints"]
    assert len(points) == 308 and [p["id"] for p in points] == list(range(308))
    assert [p["label"] for p in points] == sapiens2_pose.joint_metadata()["names"]
    # CPU directly matches the native postprocessor. Heatmap parity above is the
    # cross-device comparison, avoiding unstable derivatives of random heatmaps.
    if backend == "cpu":
        torch.testing.assert_close(
            torch.tensor([[p["x"], p["y"]] for p in points]),
            decoded["keypoints"],
            atol=1e-5,
            rtol=1e-5,
        )
    shared_tasks.validate_task_output("pose-estimation", result)


def test_original_sociopticon_order_and_flip_semantics():
    metadata = sapiens2_pose.joint_metadata()
    assert len(metadata["names"]) == len(set(metadata["names"])) == 308
    assert metadata["names"][:7] == [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
    ]
    assert all(not name.startswith("teeth_") for name in metadata["names"])
    assert metadata["names"][220] == "l_top_end_of_inferior_crus"
    assert [1, 2] in metadata["flipPairs"] and [3, 4] in metadata["flipPairs"]


def test_shared_execution_contract_and_profile(tmp_path):
    from oryn_protocol.task_contracts import validate_result

    from oryn_engine.model_inspection import effective_task
    from oryn_engine.task_profiles import profile_for_execution

    tiny_model(tmp_path)
    image = tmp_path / "image.png"
    Image.new("RGB", (80, 120), "blue").save(image)
    value = {
        "imageArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa",
        "boxes": [[0, 0, 1, 1], [0.2, 0.3, 0.8, 0.9]] * 4,
    }
    result = shared_tasks.execute_task(
        tmp_path, "pose-estimation", value, "cpu", image, None, lambda *_: None
    )
    config = json.loads((tmp_path / "config.json").read_text())
    assert effective_task("transformers", "keypoint-detection", config) == "pose-estimation"
    profile = profile_for_execution(
        {
            "technicalCapabilities": {"task": "pose-estimation"},
            "runtimePack": {"runnerId": "transformers.shared-tasks"},
        },
        config,
    )
    assert profile["limits"] == {"poses": 8, "joints": 308}
    validate_result(profile, {"kind": "structured", "value": result}, value)
    assert len(result["poses"]) == 8
    assert 256 * 1024 < len(json.dumps(result).encode()) < 900 * 1024


@pytest.mark.parametrize("component", ["backbone", "head", "extra"])
def test_incomplete_graph_never_initializes_weights(tmp_path, component):
    from safetensors.torch import load_file, save_file

    tiny_model(tmp_path)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    if component == "extra":
        state["unreviewed_bridge.weight"] = torch.zeros(2, 2)
    else:
        prefix = "backbone." if component == "backbone" else "decode_head."
        del state[next(key for key in state if key.startswith(prefix))]
    save_file(state, path)
    with pytest.raises(ValueError, match="incomplete"):
        shared_tasks.load_task(tmp_path, "pose-estimation", "cpu")


@pytest.mark.parametrize(
    "change", ["labels", "flip_pairs", "head", "attention", "geometry", "custom"]
)
def test_metadata_rejects_unreviewed_joint_and_graph_variants(tmp_path, change):
    tiny_model(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    bad = copy.deepcopy(config)
    if change == "labels":
        bad["id2label"]["0"] = "wrong_joint"
    elif change == "flip_pairs":
        bad["flip_pairs"][0] = [0, 1]
    elif change == "head":
        bad["head_config"]["use_pixel_shuffle"] = True
    elif change == "attention":
        bad["num_key_value_heads_per_layer"] = [3]
    elif change == "geometry":
        bad["image_size"] = [2048, 2048]
    else:
        bad["head_config"]["auto_map"] = {"AutoModel": "repository.Head"}
    with pytest.raises(ValueError):
        sapiens2_pose.validate_config(bad)


@pytest.mark.parametrize(
    "change",
    [
        {"do_center_crop": True},
        {"do_pad": True},
        {"image_mean": [0, 0, 0]},
        {"size": {"height": 1024, "width": 768}},
    ],
)
def test_processor_rejects_altered_geometry_and_normalization(tmp_path, change):
    tiny_model(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    metadata = json.loads((tmp_path / "preprocessor_config.json").read_text())
    with pytest.raises(ValueError, match="PROCESSOR"):
        sapiens2_pose.processor_options(metadata | change, config)


def test_metadata_uses_pose_contract_and_one_complete_native_export(tmp_path):
    from oryn_engine.metadata_audit import requirements_for
    from oryn_engine.model_layouts import assess_layout

    tiny_model(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert requirements_for("keypoint-detection", config)["contract"]["id"] == "pose-estimation-v1"
    files = [
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "sapiens2_1b_pose.safetensors",
    ]
    layout = assess_layout(files, config, "pose-estimation")
    assert not layout.artifact_reasons and not layout.runtime_reasons
    assert set(layout.selected_files) == set(files) - {"sapiens2_1b_pose.safetensors"}
