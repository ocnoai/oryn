"""Point-guided image masks through the complete native SAM 2 video graph."""

from __future__ import annotations


def validate_config(config):
    if config.get("model_type") != "sam2_video":
        raise ValueError("SAM 2 graph identity is unavailable")
    for key, maximum, default in (
        ("image_size", 1024, 1024),
        ("num_maskmem", 16, 7),
        ("memory_attention_num_layers", 16, 4),
        ("memory_attention_hidden_size", 4096, 256),
        ("memory_encoder_hidden_size", 4096, 256),
        ("memory_encoder_output_channels", 1024, 64),
    ):
        value = config.get(key, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("SAM 2 dimensions exceed the image binding")
    if any(
        not isinstance(config.get(key), dict)
        for key in (
            "vision_config",
            "prompt_encoder_config",
            "mask_decoder_config",
        )
    ):
        raise ValueError("SAM 2 component configuration is incomplete")


def native_config(config):
    from copy import deepcopy

    from transformers import Sam2VideoConfig, Sam2VideoMaskDecoderConfig

    validate_config(config)
    data = deepcopy(config)
    # Transformers 5.17's dict constructor mistakenly uses its prompt-encoder
    # class for the decoder. Supply the actual native decoder config explicitly.
    data["mask_decoder_config"] = Sam2VideoMaskDecoderConfig(**data["mask_decoder_config"])
    return Sam2VideoConfig(**data)


def predict_mask(model, processor, image, points, labels, backend):
    import torch

    # A fresh one-frame session gives the native first-frame segmentation
    # semantics and bounds all temporal state. No URL or video decoder is used.
    session = processor.init_video_session(
        video=[image],
        inference_device=backend,
        inference_state_device=backend,
        processing_device="cpu",
        video_storage_device="cpu",
        max_vision_features_cache_size=1,
        dtype=torch.float32,
    )
    try:
        processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=0,
            obj_ids=1,
            input_points=[points],
            input_labels=[labels],
        )
        output = model(inference_session=session, frame_idx=0)
        if (
            output.object_ids != [1]
            or output.frame_idx != 0
            or output.pred_masks.ndim != 4
            or tuple(output.pred_masks.shape[:2]) != (1, 1)
            or output.pred_masks.numel() > 1024**2
            or not bool(torch.isfinite(output.pred_masks).all())
            or not bool(torch.isfinite(output.object_score_logits).all())
        ):
            raise ValueError("SAM 2 mask output is invalid")
        masks = processor.post_process_masks(
            [output.pred_masks.cpu()],
            original_sizes=[[image.height, image.width]],
        )[0]
        if tuple(masks.shape) != (1, 1, image.height, image.width) or masks.dtype != torch.bool:
            raise ValueError("SAM 2 mask geometry is invalid")
        return masks[0, 0]
    finally:
        del session
