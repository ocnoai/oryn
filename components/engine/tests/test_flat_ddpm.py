"""Closed legacy declarations, exact native tensors and deterministic image output."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oryn_engine import flat_ddpm as adapter
from oryn_engine.diffusion_tasks import validate_pipeline_index

FIXTURES = Path(__file__).parent / "fixtures/flat_ddpm"
UNET = json.loads((FIXTURES / "config.json").read_text())
SCHEDULER = json.loads((FIXTURES / "scheduler_config.json").read_text())
INDEX = json.loads((FIXTURES / "model_index.json").read_text())


def fixture(root, precision="float32"):
    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
    from safetensors.torch import save_file

    root.mkdir(exist_ok=True, parents=True)
    config = copy.deepcopy(UNET)
    config.update(
        block_out_channels=[8, 16],
        layers_per_block=1,
        norm_num_groups=4,
        down_block_types=["DownBlock2D", "AttnDownBlock2D"],
        up_block_types=["AttnUpBlock2D", "UpBlock2D"],
    )
    scheduler = SCHEDULER | {"num_train_timesteps": 16}
    for name, value in [
        ("config.json", config),
        ("scheduler_config.json", scheduler),
        ("model_index.json", INDEX),
    ]:
        (root / name).write_text(json.dumps(value))
    torch.manual_seed(6115)
    # Independent native construction; retain quantization to the stored precision.
    unet = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        norm_num_groups=4,
        norm_eps=1e-6,
        attention_head_dim=None,
        act_fn="silu",
        center_input_sample=False,
        downsample_padding=0,
        flip_sin_to_cos=False,
        freq_shift=1,
        mid_block_scale_factor=1,
        time_embedding_type="positional",
    ).eval()

    def old_name(k):
        for new, old in [
            (".to_q.", ".query."),
            (".to_k.", ".key."),
            (".to_v.", ".value."),
            (".to_out.0.", ".proj_attn."),
        ]:
            k = k.replace(new, old)
        return k

    native = {
        k: v.detach().to(getattr(torch, precision)).clone() for k, v in unet.state_dict().items()
    }
    state = {old_name(k): v for k, v in native.items()}
    save_file(state, str(root / "diffusion_pytorch_model.safetensors"))
    unet.load_state_dict({k: v.float() for k, v in native.items()})
    reference = DDPMPipeline(
        unet,
        DDPMScheduler(
            num_train_timesteps=16,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            variance_type="fixed_large",
            clip_sample=True,
        ),
    )
    reference.set_progress_bar_config(disable=True)
    return config, reference, state


def test_original_complete_graph_headers_and_legacy_names_match():
    adapter.validate_configs(UNET, SCHEDULER)
    index = adapter.normalized_index(INDEX)
    assert adapter.supports(index)
    validate_pipeline_index(index)
    h = json.loads((FIXTURES / "original-header.json").read_text())
    shapes = {k: tuple(v["shape"]) for k, v in h.items() if k != "__metadata__"}
    assert len(shapes) == 328
    assert sum(adapter.native_name(k) != k for k in shapes) == 48
    adapter.validate_weight_shapes(UNET, shapes)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("precision", ["float16", "float32"])
def test_complete_native_denoising_and_seeded_pipeline_parity(tmp_path, backend, precision):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    _, reference, _ = fixture(tmp_path, precision)
    loaded = adapter.load_pipeline(tmp_path).to(backend)
    loaded.set_progress_bar_config(disable=True)
    reference.to(backend)
    noise = torch.randn(1, 3, 32, 32, generator=torch.Generator().manual_seed(42)).to(backend)
    with torch.inference_mode():
        for step in [0, 8, 15]:
            torch.testing.assert_close(
                loaded.unet(noise, step).sample,
                reference.unet(noise, step).sample,
                atol=2e-5,
                rtol=2e-5,
            )
        actual = loaded(
            generator=torch.Generator().manual_seed(61), num_inference_steps=4, output_type="np"
        ).images
        expected = reference(
            generator=torch.Generator().manual_seed(61), num_inference_steps=4, output_type="np"
        ).images
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
    assert actual.shape == (1, 32, 32, 3) and np.isfinite(actual).all() and np.std(actual) > 0.01


@pytest.mark.parametrize(
    "change",
    [
        {"scheduler": ["diffusers", "DDIMScheduler"]},
        {"unet": ["custom", "UNet2DModel"]},
        {"_module": "modeling_ddpm.py"},
        {"text_encoder": ["transformers", "BertModel"]},
        {"trust_remote_code": True},
    ],
)
def test_custom_pipeline_cannot_add_components_or_flags(change):
    with pytest.raises(ValueError):
        adapter.normalized_index(INDEX | change)


@pytest.mark.parametrize(
    "identity", [["other", "CustomPipeline"], ["modeling_ddpm", "Other"], [], None]
)
def test_other_repository_classes_are_not_aliased(identity):
    value = INDEX | {"_class_name": identity}
    assert not adapter.supports(adapter.normalized_index(value))


@pytest.mark.parametrize(
    "change",
    [
        {"attention_head_dim": 8},
        {"in_channels": 4},
        {"sample_size": 31},
        {"layers_per_block": True},
        {"block_out_channels": [7]},
        {"norm_num_groups": 3},
        {"down_block_types": ["CrossAttnDownBlock2D"]},
        {"class_embed_type": "identity"},
        {"center_input_sample": 0},
        {"freq_shift": 0},
        {"time_embedding_type": "fourier"},
    ],
)
def test_closed_unet_config_preserves_original_graph(change):
    with pytest.raises(ValueError):
        adapter.validate_configs(UNET | change, SCHEDULER)


@pytest.mark.parametrize(
    "change",
    [
        {"beta_schedule": "scaled_linear"},
        {"variance_type": "learned"},
        {"prediction_type": "v_prediction"},
        {"clip_sample": False},
        {"num_train_timesteps": True},
        {"beta_start": 0.5},
        {"trained_betas": [0.1, 0.2]},
    ],
)
def test_closed_original_scheduler(change):
    with pytest.raises(ValueError):
        adapter.validate_configs(UNET, SCHEDULER | change)


def test_scheduler_that_underflows_native_precision_is_rejected():
    with pytest.raises(ValueError, match="FLAT_DDPM_SCHEDULER_PRECISION"):
        adapter.validate_configs(UNET, SCHEDULER | {"beta_start": 0.99, "beta_end": 0.999})


@pytest.mark.parametrize("damage", ["nonfinite", "shape"])
def test_invalid_native_denoiser_outputs_fail_before_image_conversion(tmp_path, damage):
    fixture(tmp_path)
    pipe = adapter.load_pipeline(tmp_path)
    pipe.unet.forward = lambda *args, **kwargs: SimpleNamespace(
        sample=torch.full(
            (1, 3, 32, 32 if damage == "nonfinite" else 16),
            float("nan") if damage == "nonfinite" else 0.0,
        )
    )
    with pytest.raises(ValueError, match="FLAT_DDPM_DENOISER_OUTPUT"):
        pipe(num_inference_steps=2, generator=torch.Generator().manual_seed(61))


@pytest.mark.parametrize(
    "damage", ["missing", "extra", "shape", "integer", "nan", "alias", "symlink", "nested"]
)
def test_complete_finite_numeric_graph_is_required(tmp_path, damage):
    from safetensors.torch import save_file

    _, _, state = fixture(tmp_path)
    key = next(k for k in state if ".query.weight" in k)
    if damage == "missing":
        del state[key]
    elif damage == "extra":
        state["unreviewed.weight"] = torch.ones(2)
    elif damage == "shape":
        state[key] = state[key][:1]
    elif damage == "integer":
        state[key] = state[key].to(torch.int32)
    elif damage == "nan":
        state[key][0, 0] = float("nan")
    elif damage == "alias":
        state[adapter.native_name(key)] = state[key].clone()
    path = tmp_path / "diffusion_pytorch_model.safetensors"
    save_file(state, str(path))
    if damage == "symlink":
        path.rename(tmp_path / "outside.data")
        path.symlink_to(tmp_path / "outside.data")
    elif damage == "nested":
        (tmp_path / "extra").mkdir()
        save_file({"weight": torch.ones(2)}, str(tmp_path / "extra/model.safetensors"))
    with pytest.raises(ValueError):
        adapter.load_pipeline(tmp_path)


@pytest.mark.parametrize("damage", [None, "scheduler", "missing", "unet", "custom"])
def test_original_metadata_routes_to_owned_native_loader(tmp_path, damage):
    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore

    data = {
        "model_index.json": copy.deepcopy(INDEX),
        "config.json": copy.deepcopy(UNET),
        "scheduler_config.json": copy.deepcopy(SCHEDULER),
    }
    names = {
        "model_index.json",
        "config.json",
        "scheduler_config.json",
        "diffusion_pytorch_model.safetensors",
        "modeling_ddpm.py",
    }
    if damage == "scheduler":
        data["scheduler_config.json"]["variance_type"] = "learned"
    elif damage == "missing":
        names.remove("scheduler_config.json")
    elif damage == "unet":
        data["config.json"]["class_embed_type"] = "identity"
    elif damage == "custom":
        data["model_index.json"]["unet"] = ["repo", "Code"]

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return data[filename]

    hub = Hub(
        config=data["model_index.json"],
        pipeline_tag="unconditional-image-generation",
        files=[(n, 100, "b" * 64) for n in names],
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda p: None if p.provider else object()
    )
    result = store.inspect(model("someone/flat-export"))
    assert result["executable"] is (damage is None), result
    if damage is None:
        assert result["runtimePack"]["taskContract"] == "unconditional-image-generation-v1"
        assert result["generationProfile"]["parameters"]["width"]["default"] == 32
        assert "modeling_ddpm.py" not in {f["name"] for f in result["selectedFiles"]}
    assert not hub.downloaded


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_owned_worker_loads_original_declaration_and_produces_repeatable_png(tmp_path, backend):
    from PIL import Image

    from oryn_engine.runtime_packs.diffusers_worker import WorkerFailure, _generate

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    fixture(tmp_path)
    output = tmp_path / "image.png"
    payload = dict(
        modelRoot=str(tmp_path),
        backend=backend,
        taskContract="unconditional-image-generation-v1",
        input=dict(width=32, height=32, steps=4, seed=61),
        outputPath=str(output),
        previewPath=None,
        inputPath=None,
    )
    result = _generate(payload)
    assert result["mediaType"] == "image/png" and result["width"] == result["height"] == 32
    first = np.asarray(Image.open(output)).copy()
    _generate(payload)
    assert np.array_equal(first, np.asarray(Image.open(output)))
    _generate(payload | {"input": payload["input"] | {"seed": 62}})
    assert not np.array_equal(first, np.asarray(Image.open(output)))
    with pytest.raises(WorkerFailure, match="MODEL_GENERATION_FAILED"):
        _generate(payload | {"input": payload["input"] | {"width": 64}})
