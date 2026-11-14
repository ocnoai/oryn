from copy import deepcopy

import pytest

from oryn_engine.diffusion_tasks import VIDEO_PIPELINES, validate_pipeline_index, video_options
from oryn_engine.runtime_packs.diffusers_worker import _supported_options


def test_native_video_index_and_bounds_do_not_admit_other_graphs():
    index = {
        "_class_name": "CogVideoXPipeline",
        **{k: list(v) for k, v in VIDEO_PIPELINES["CogVideoXPipeline"].items()},
        "scheduler": ["diffusers", "CogVideoXDDIMScheduler"],
        "tokenizer": ["transformers", "T5Tokenizer"],
    }
    validate_pipeline_index(index)
    bad = deepcopy(index)
    bad["transformer"] = ["custom_module", "RemoteTransformer"]
    with pytest.raises(ValueError):
        validate_pipeline_index(bad)
    value = {"width": 256, "height": 256, "frames": 17, "steps": 8}
    assert (
        video_options("CogVideoXPipeline", "text-to-video-v1", value, {})["max_sequence_length"]
        == 226
    )
    for override in (
        {"frames": 16},
        {"frames": 49},
        {"width": 520},
        {"height": 264},
        {"steps": 31},
    ):
        with pytest.raises(ValueError):
            video_options("CogVideoXPipeline", "text-to-video-v1", {**value, **override}, {})
    with pytest.raises(ValueError):
        video_options("CogVideoXPipeline", "image-to-video-v1", value, {})


def test_video_controls_cannot_be_silently_dropped():
    class MissingFrames:
        def __call__(self, prompt, width, height):
            pass

    options = {"prompt": "A moving ball", "width": 256, "height": 256, "num_frames": 17}
    with pytest.raises(ValueError):
        _supported_options(MissingFrames(), options, set(options))


def test_native_video_vae_tiles_preserve_requested_geometry():
    from types import SimpleNamespace

    import torch

    from oryn_engine.runtime_packs.diffusers_worker import _enable_video_vae_tiling

    AutoencoderKLCogVideoX = pytest.importorskip("diffusers").AutoencoderKLCogVideoX
    vae = AutoencoderKLCogVideoX(
        block_out_channels=(4, 4, 4, 4),
        latent_channels=4,
        layers_per_block=1,
        norm_num_groups=1,
        sample_height=64,
        sample_width=256,
    ).eval()
    # No pipeline-level tiling method: the real CogVideoX VAE owns the API.
    _enable_video_vae_tiling(SimpleNamespace(vae=vae))
    assert vae.use_tiling and vae.use_slicing
    with torch.no_grad():
        output = vae.decode(torch.zeros(1, 4, 1, 8, 32)).sample
    assert output.shape == (1, 3, 1, 64, 256)
    assert torch.isfinite(output).all()
