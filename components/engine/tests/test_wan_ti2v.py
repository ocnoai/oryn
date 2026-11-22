"""Native expanded-timestep video graph, geometry and data-only inspection."""

import json
from copy import deepcopy

import pytest
import torch

from oryn_engine import wan_video
from oryn_engine.diffusion_tasks import validate_pipeline_index, video_options
from oryn_engine.runtime_packs.diffusers_worker import _pipeline, _validate_model


def fixture(root):
    from diffusers import (
        AutoencoderKLWan,
        UniPCMultistepScheduler,
        WanPipeline,
        WanTransformer3DModel,
    )
    from transformers import T5Tokenizer, UMT5Config, UMT5EncoderModel

    torch.manual_seed(6188)
    tokenizer = T5Tokenizer(
        vocab=[("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), ("▁", -2.0)]
        + [(c, -3.0) for c in "abcdefghijklmnopqrstuvwxyz.!?"],
        extra_ids=0,
    )
    text = UMT5EncoderModel(
        UMT5Config(
            vocab_size=len(tokenizer),
            d_model=32,
            d_ff=64,
            d_kv=8,
            num_heads=4,
            num_layers=1,
            num_decoder_layers=1,
            feed_forward_proj="gated-gelu",
            dropout_rate=0.0,
        )
    ).eval()
    transformer = WanTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=4,
        out_channels=4,
        text_dim=32,
        freq_dim=16,
        ffn_dim=32,
        num_layers=1,
        rope_max_seq_len=32,
    ).eval()
    vae = AutoencoderKLWan(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=4,
        num_res_blocks=1,
        is_residual=True,
        in_channels=12,
        out_channels=12,
        patch_size=2,
        scale_factor_spatial=16,
        latents_mean=[0.1, -0.2, 0.3, -0.4],
        latents_std=[0.5, 1.5, 1.0, 2.0],
    ).eval()
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5.0
    )
    pipe = WanPipeline(
        tokenizer, text, vae, scheduler, transformer, boundary_ratio=None, expand_timesteps=True
    )
    pipe.save_pretrained(root, safe_serialization=True)
    return pipe


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_native_components_and_expanded_step_parity(tmp_path, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    reference = fixture(tmp_path)
    _validate_model(tmp_path)
    loaded, selected = _pipeline(tmp_path, "text-to-video-v1", torch, backend_request=backend)
    assert selected == backend and loaded.config.expand_timesteps is True
    assert loaded.transformer_2 is None and loaded.config.boundary_ratio is None
    dtype = loaded.transformer.dtype
    reference.to(dtype=dtype)
    for name in wan_video.COMPONENTS:
        actual = getattr(loaded, name).state_dict()
        for key, weight in getattr(reference, name).state_dict().items():
            torch.testing.assert_close(actual[key].cpu(), weight, rtol=0, atol=0)
    ids = torch.tensor([[4, 5, 6, 1]])
    mask = torch.ones_like(ids)
    latent = torch.linspace(-0.4, 0.4, 4 * 2 * 4 * 4).reshape(1, 4, 2, 4, 4).to(dtype)
    # One time value per spatial/temporal patch is the new native TI2V path.
    times = torch.tensor([[900, 800, 700, 600, 500, 400, 300, 200]], dtype=dtype)
    with torch.inference_mode():
        expected_text = reference.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        actual_text = loaded.text_encoder(
            input_ids=ids.to(backend), attention_mask=mask.to(backend)
        ).last_hidden_state
        expected = reference.transformer(latent, times, expected_text).sample
        actual = loaded.transformer(latent.to(backend), times.to(backend), actual_text).sample
        reference.vae.clear_cache()
        loaded.vae.clear_cache()
        expected_pixels = reference.vae.decode(latent[:, :, :1]).sample
        actual_pixels = loaded.vae.decode(latent[:, :, :1].to(backend)).sample
    atol, rtol = (2e-3, 5e-3) if backend == "mps" else (1e-6, 1e-5)
    torch.testing.assert_close(actual_text.cpu(), expected_text, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual.cpu(), expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_pixels.cpu(), expected_pixels, atol=atol, rtol=rtol)
    assert actual_pixels.shape == (1, 3, 1, 64, 64)


def test_native_pipeline_returns_exact_requested_frames_and_pixels(tmp_path):
    fixture(tmp_path)
    pipe, _ = _pipeline(tmp_path, "text-to-video-v1", torch, backend_request="cpu")
    value = {"width": 64, "height": 64, "frames": 5, "steps": 1}
    options = video_options("WanPipeline", "text-to-video-v1", value, {}, expanded_wan=True)
    with torch.inference_mode():
        output = pipe(
            prompt="a river",
            negative_prompt="",
            width=64,
            height=64,
            num_frames=5,
            num_inference_steps=1,
            guidance_scale=1.0,
            generator=torch.Generator().manual_seed(61),
            output_type="np",
            **options,
        ).frames
    assert output.shape == (1, 5, 64, 64, 3)
    assert torch.isfinite(torch.from_numpy(output)).all()


@pytest.mark.parametrize("component", wan_video.COMPONENTS)
@pytest.mark.parametrize("fault", ["missing", "extra", "shape"])
def test_complete_component_weights_are_required_before_native_loading(tmp_path, component, fault):
    from safetensors.torch import load_file, save_file

    fixture(tmp_path)
    path = next((tmp_path / component).glob("*.safetensors"))
    state = load_file(path)
    name = next(iter(state))
    if fault == "extra":
        state["unrecognized.weight"] = torch.zeros(2)
    elif fault == "shape":
        state[name] = torch.zeros(3)
    else:
        del state[name]
    save_file(state, path)
    with pytest.raises(ValueError, match="WEIGHTS_INCOMPLETE"):
        _pipeline(tmp_path, "text-to-video-v1", torch, backend_request="cpu")


@pytest.mark.parametrize(
    "change",
    [
        {"boundary_ratio": 0.5},
        {"transformer_2": ["diffusers", "WanTransformer3DModel"]},
        {"expand_timesteps": 1},
        {"custom_option": True},
    ],
)
def test_unknown_graph_variants_do_not_inherit_the_native_binding(tmp_path, change):
    fixture(tmp_path)
    index = json.loads((tmp_path / "model_index.json").read_text())
    with pytest.raises(ValueError):
        validate_pipeline_index(index | change)


@pytest.mark.parametrize(
    "component,change",
    [
        ("transformer", {"in_channels": 8}),
        ("transformer", {"patch_size": [1, 1, 1]}),
        ("vae", {"scale_factor_spatial": 8}),
        ("vae", {"latents_std": [1.0, 1.0, 0.0, 1.0]}),
        ("text_encoder", {"d_model": 64}),
        ("text_encoder", {"auto_map": {"AutoModel": "repo.Custom"}}),
    ],
)
def test_incompatible_native_component_metadata_is_rejected(tmp_path, component, change):
    fixture(tmp_path)
    configs = wan_video.read_configs(tmp_path)
    configs[component].update(change)
    with pytest.raises(ValueError):
        wan_video.validate_configs(configs)


def test_metadata_profile_and_worker_preserve_exact_raster_geometry(tmp_path):
    from oryn_protocol.task_contracts import effective_input, validate_invocation
    from test_models import FakeHub

    from oryn_engine.models import ModelStore

    root = tmp_path / "model"
    fixture(root)
    index = json.loads((root / "model_index.json").read_text())

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return json.loads((root / filename).read_text())

    hub = Hub(
        config=index,
        pipeline_tag="text-to-video",
        files=[
            (p.relative_to(root).as_posix(), p.stat().st_size, "b" * 64)
            for p in root.rglob("*")
            if p.is_file()
        ],
    )
    result = ModelStore(tmp_path / "store", hub=hub, runtime_available=True).inspect(
        {"provider": "huggingface", "repository": "test/wan-ti2v", "revision": "main"}
    )
    profile = result["taskProfile"]
    value = effective_input(profile, {"prompt": "a river", "width": 64, "height": 64, "frames": 5})
    validate_invocation(profile, value)
    video_options("WanPipeline", "text-to-video-v1", value, {}, expanded_wan=True)
    with pytest.raises(ValueError):
        validate_invocation(profile, value | {"width": 80})
    with pytest.raises(ValueError):
        video_options(
            "WanPipeline", "text-to-video-v1", value | {"width": 80}, {}, expanded_wan=True
        )
    assert hub.downloaded == []
    original = deepcopy(index)
    original["expand_timesteps"] = False
    validate_pipeline_index(original)
