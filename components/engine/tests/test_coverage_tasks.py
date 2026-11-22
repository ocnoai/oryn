from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from oryn_protocol.task_contracts import make_profile, validate_invocation

from oryn_engine.coverage_tasks import sample_video
from oryn_engine.shared_tasks import validate_task_config, validate_task_input

ARTIFACT = "artifact_aaaaaaaaaaaaaaaaaaaaaaaa"


def test_table_shape_and_document_geometry_are_bound_before_execution():
    table = {
        "question": "Which city?",
        "columns": ["City", "Country"],
        "rows": [["Paris", "France"]],
    }
    validate_task_input("table-question-answering", table)
    for change in ({"columns": ["City", " City "]}, {"rows": [["Paris"]]}):
        with pytest.raises(ValueError):
            validate_task_input("table-question-answering", table | change)
    page = {
        "imageArtifactId": ARTIFACT,
        "question": "Total?",
        "words": ["42"],
        "boxes": [[0.2, 0.1, 0.3, 0.2]],
    }
    validate_task_input("document-question-answering", page)
    for boxes in ([], [[0.3, 0.1, 0.2, 0.2]], [[0, 0, 1.1, 1]]):
        with pytest.raises(ValueError):
            validate_task_input("document-question-answering", page | {"boxes": boxes})


def test_video_profiles_reject_impossible_clip_bounds():
    profile = make_profile("video-classification-v1")
    for value in ({"startSeconds": 30}, {"startSeconds": 25, "durationSeconds": 6}):
        with pytest.raises(ValueError):
            validate_invocation(profile, {"videoArtifactId": ARTIFACT} | value)


def video(path: Path, *, seconds=2, reverse=False):
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=16)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(seconds * 16):
            color = [255, 0, 0] if (i < seconds * 8) != reverse else [0, 255, 0]
            pixels = np.full((48, 64, 3), color, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_native_video_decoder_retains_order_timestamps_and_bounded_geometry(tmp_path):
    path = tmp_path / "clip.mp4"
    for reverse in (False, True):
        video(path, reverse=reverse)
        frames, times, duration = sample_video(path, 16, 0, None)
        assert len(frames) == len(times) == 16
        assert times == sorted(times) and len(set(times)) == 16
        assert duration == pytest.approx(2)
        assert all(frame.size == (64, 48) for frame in frames)
        assert np.asarray(frames[0]).mean((0, 1)).argmax() == (1 if reverse else 0)
        assert np.asarray(frames[-1]).mean((0, 1)).argmax() == (0 if reverse else 1)


def test_long_corrupt_and_unowned_video_are_rejected(tmp_path):
    from av.error import InvalidDataError

    path = tmp_path / "clip.mp4"
    video(path, seconds=31)
    with pytest.raises(ValueError, match="too long|duration"):
        sample_video(path, 16, 0, None)
    path.write_bytes(b"not a video")
    with pytest.raises(InvalidDataError):
        sample_video(path, 16, 0, None)
    link = tmp_path / "link.mp4"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="artifact"):
        sample_video(link, 16, 0, None)


def test_video_pretraining_and_unknown_aggregation_do_not_borrow_task_support():
    for config in (
        {"model_type": "videomae", "num_frames": 16},
        {"model_type": "videomae", "num_frames": 128, "id2label": {"0": "a", "1": "b"}},
    ):
        with pytest.raises(ValueError):
            validate_task_config("video-classification", config)
    with pytest.raises(ValueError):
        validate_task_config(
            "table-question-answering",
            {
                "model_type": "tapas",
                "aggregation_labels": {"0": "EXECUTE"},
                "num_aggregation_labels": 1,
            },
        )


def test_legacy_videomae_bias_layout_preserves_native_logits_and_rejects_missing_head(tmp_path):
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification, VideoMAEImageProcessor

    from oryn_engine.shared_tasks import load_task

    config = VideoMAEConfig(
        image_size=16,
        patch_size=8,
        num_frames=4,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        id2label={0: "first", 1: "second"},
    )
    original = VideoMAEForVideoClassification(config).eval()
    key = original.videomae.encoder.layer[0].attention.attention.key
    if key.bias is not None:  # Transformers >= 5.20 builds VideoMAE keys without a bias.
        with torch.no_grad():
            key.bias.zero_()
    original.save_pretrained(tmp_path, safe_serialization=True)
    VideoMAEImageProcessor(
        size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
    ).save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    weights = load_file(path)
    for name in list(weights):
        if ".attention.attention." in name:
            if name.endswith(".key.bias"):
                del weights[name]
            elif name.endswith((".query.bias", ".value.bias")):
                legacy = name.replace(".query.bias", ".q_bias").replace(".value.bias", ".v_bias")
                weights[legacy] = weights.pop(name)
    save_file(weights, path)
    loaded, _, _ = load_task(tmp_path, "video-classification", "cpu")
    pixels = torch.randn(1, 4, 3, 16, 16)
    with torch.inference_mode():
        assert torch.allclose(original(pixels).logits, loaded(pixels).logits, atol=1e-6)
    del weights["classifier.weight"]
    save_file(weights, path)
    with pytest.raises(ValueError, match="incomplete"):
        load_task(tmp_path, "video-classification", "cpu")


def test_native_vjepa_classifier_preserves_temporal_logits_and_requires_the_head(tmp_path):
    import torch
    from PIL import Image
    from safetensors.torch import load_file, save_file
    from transformers import VJEPA2Config, VJEPA2ForVideoClassification, VJEPA2VideoProcessor

    from oryn_engine.model_layouts import assess_layout
    from oryn_engine.shared_tasks import load_task

    config = VJEPA2Config(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        frames_per_clip=4,
        image_size=32,
        crop_size=32,
        patch_size=16,
        pred_hidden_size=32,
        pred_num_attention_heads=4,
        pred_num_hidden_layers=1,
        num_pooler_layers=1,
        id2label={0: "first", 1: "second"},
    )
    original = VJEPA2ForVideoClassification(config).eval()
    original.save_pretrained(tmp_path, safe_serialization=True)
    processor = VJEPA2VideoProcessor(crop_size=32)
    processor.save_pretrained(tmp_path)
    from oryn_engine.model_metadata import metadata_file

    layout = assess_layout(
        [p.name for p in tmp_path.iterdir()],
        metadata_file(tmp_path / "config.json"),
        "video-classification",
    )
    assert not layout.artifact_reasons and not layout.runtime_reasons
    loaded, native_processor, _ = load_task(tmp_path, "video-classification", "cpu")
    frames = [Image.new("RGB", (32, 32), color) for color in ["red", "blue", "green", "red"]]
    pixels = native_processor(videos=[frames], do_sample_frames=False, return_tensors="pt")
    assert pixels["pixel_values_videos"].shape == (1, 4, 3, 32, 32)
    with torch.inference_mode():
        assert torch.allclose(original(**pixels).logits, loaded(**pixels).logits, atol=1e-6)
    path = tmp_path / "model.safetensors"
    weights = load_file(path)
    del weights["classifier.weight"]
    save_file(weights, path)
    with pytest.raises(ValueError, match="incomplete"):
        load_task(tmp_path, "video-classification", "cpu")


@pytest.mark.parametrize("frames,tubelet", [(64, 2), (3, 2), (True, 2), (4, 0)])
def test_vjepa_frame_geometry_is_bounded(frames, tubelet):
    with pytest.raises(ValueError):
        validate_task_config(
            "video-classification",
            {
                "model_type": "vjepa2",
                "frames_per_clip": frames,
                "tubelet_size": tubelet,
                "id2label": {"0": "first", "1": "second"},
            },
        )
