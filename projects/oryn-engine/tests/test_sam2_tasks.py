from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from oryn_engine.sam2_tasks import predict_mask
from oryn_engine.shared_tasks import load_task


def tiny_sam2():
    from transformers import Sam2VideoConfig, Sam2VideoMaskDecoderConfig, Sam2VideoModel

    config = Sam2VideoConfig(
        image_size=64,
        vision_config={
            "model_type": "sam2_vision_model",
            "fpn_hidden_size": 32,
            "backbone_channel_list": [256, 128, 64, 32],
            "backbone_feature_sizes": [[16, 16], [8, 8], [4, 4]],
            "backbone_config": {
                "model_type": "sam2_hiera_det_model",
                "hidden_size": 32,
                "image_size": [64, 64],
                "blocks_per_stage": [1, 1, 1, 1],
                "embed_dim_per_stage": [32, 64, 128, 256],
                "num_attention_heads_per_stage": [1, 2, 4, 8],
                "window_size_per_stage": [4, 2, 2, 2],
                "global_attention_blocks": [],
                "window_positional_embedding_background_size": [4, 4],
            },
        },
        prompt_encoder_config={"hidden_size": 32, "image_size": 64},
        mask_decoder_config=Sam2VideoMaskDecoderConfig(
            **{
                "hidden_size": 32,
                "mlp_dim": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "iou_head_hidden_dim": 32,
            }
        ),
        memory_attention_hidden_size=32,
        memory_attention_num_layers=1,
        memory_attention_feed_forward_hidden_size=64,
        memory_attention_rope_feat_sizes=[4, 4],
        memory_encoder_hidden_size=32,
        memory_encoder_output_channels=8,
        mask_downsampler_embed_dim=32,
        memory_fuser_num_layers=1,
        memory_fuser_embed_dim=32,
        memory_fuser_intermediate_dim=64,
        memory_fuser_kernel_size=3,
        memory_fuser_padding=1,
    )
    return Sam2VideoModel(config).eval()


def test_sam2_complete_graph_load_and_first_frame_masks_match_native(tmp_path):
    from safetensors.torch import load_file, save_file
    from transformers import Sam2ImageProcessor, Sam2VideoProcessor, Sam2VideoVideoProcessor

    torch.manual_seed(61)
    original = tiny_sam2()
    processor = Sam2VideoProcessor(
        image_processor=Sam2ImageProcessor(size={"height": 64, "width": 64}),
        video_processor=Sam2VideoVideoProcessor(size={"height": 64, "width": 64}),
    )
    original.save_pretrained(tmp_path, safe_serialization=True)
    processor.save_pretrained(tmp_path)
    loaded, loaded_processor, _ = load_task(tmp_path, "mask-generation", "cpu")
    assert type(loaded) is type(original)
    pixels = np.zeros((48, 64, 3), dtype=np.uint8)
    pixels[10:35, 15:45, 0] = 255
    image = Image.fromarray(pixels)
    points, labels = [[[30, 20], [2, 2]]], [[1, 0]]
    with torch.inference_mode():
        session = processor.init_video_session(video=[image], inference_device="cpu")
        processor.add_inputs_to_inference_session(
            session, frame_idx=0, obj_ids=1, input_points=[points], input_labels=[labels]
        )
        output = original(inference_session=session, frame_idx=0)
        reference = processor.post_process_masks([output.pred_masks], [[48, 64]])[0][0, 0]
        actual = predict_mask(loaded, loaded_processor, image, points, labels, "cpu")
        assert actual.dtype == torch.bool and actual.shape == (48, 64)
        assert torch.equal(actual, reference)
        # Another invocation must start from fresh state, with no prior prompt/mask memory.
        assert torch.equal(
            actual, predict_mask(loaded, loaded_processor, image, points, labels, "cpu")
        )
    path = tmp_path / "model.safetensors"
    weights = load_file(path)
    del weights[next(name for name in weights if "memory_encoder" in name)]
    save_file(weights, path)
    with pytest.raises(ValueError, match="incomplete"):
        load_task(tmp_path, "mask-generation", "cpu")
