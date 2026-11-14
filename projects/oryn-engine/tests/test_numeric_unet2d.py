"""Exact native UNet2D graphs for numeric-archive DDPM and NCSN++ exports."""

import copy
import json
import pickle
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oryn_engine import numeric_unet2d as adapter
from oryn_engine.diffusion_tasks import validate_pipeline_index

FIXTURES = Path(__file__).parent / "fixtures/numeric_unet2d"
ORIGINALS = ["ddpm-church-256", "ncsnpp-celebahq-256"]


def original(name):
    root = FIXTURES / name
    return {
        part: json.loads((root / f"{part}.json").read_text())
        for part in ["model_index", "config", "scheduler_config", "header"]
    }


def archive_metadata(shapes, dtype="F32"):
    return {
        "format": "torch-numeric-zip-v1",
        "byteOrder": "little",
        "tensors": {k: {"shape": list(v), "dtype": dtype} for k, v in shapes.items()},
    }


def native_shapes(config):
    from diffusers import UNet2DModel

    with torch.device("meta"):
        model = UNet2DModel(**{k: v for k, v in config.items() if not k.startswith("_")})
    return {k: list(v.shape) for k, v in model.state_dict().items()}


def old_name(name):
    for new, old in [(".to_q.", ".query."), (".to_k.", ".key."), (".to_v.", ".value.")]:
        name = name.replace(new, old)
    return name.replace(".to_out.0.", ".proj_attn.")


def tiny(family):
    data = original("ncsnpp-celebahq-256" if family == "sde" else "ddpm-church-256")
    config = copy.deepcopy(data["config"])
    if family == "sde":
        config.update(
            block_out_channels=[32, 64],
            layers_per_block=1,
            sample_size=16,
            down_block_types=["SkipDownBlock2D", "AttnSkipDownBlock2D"],
            up_block_types=["AttnSkipUpBlock2D", "SkipUpBlock2D"],
        )
        scheduler = data["scheduler_config"] | {"num_train_timesteps": 20}
    else:
        config.update(
            block_out_channels=[8, 16],
            layers_per_block=1,
            norm_num_groups=4,
            sample_size=16,
            down_block_types=["DownBlock2D", "AttnDownBlock2D"],
            up_block_types=["AttnUpBlock2D", "UpBlock2D"],
        )
        scheduler = data["scheduler_config"] | {"num_train_timesteps": 16}
    return data["model_index"], config, scheduler


def fixture(root, family="ddpm", precision="float32", legacy=True):
    import diffusers

    index, config, scheduler = tiny(family)
    root.mkdir(exist_ok=True, parents=True)
    for name, value in [
        ("model_index.json", index),
        ("config.json", config),
        ("scheduler_config.json", scheduler),
    ]:
        (root / name).write_text(json.dumps(value))
    torch.manual_seed(6116)
    unet = diffusers.UNet2DModel(**{k: v for k, v in config.items() if not k.startswith("_")})
    native = {
        k: v.detach().to(getattr(torch, precision)).clone() for k, v in unet.state_dict().items()
    }
    state = {old_name(k) if legacy else k: v for k, v in native.items()}
    if family == "sde":
        # Original NCSN++ exports retain the Fourier weight under both names.
        state[adapter.FOURIER_ALIAS] = state["time_proj.weight"].clone()
    torch.save(state, root / adapter.WEIGHTS)
    unet.load_state_dict({k: v.float() for k, v in native.items()})
    schedule_class = adapter.SCHEDULERS[index["_class_name"]][0]
    reference = getattr(diffusers, index["_class_name"])(
        unet=unet.eval(),
        scheduler=getattr(diffusers, schedule_class)(
            **{k: v for k, v in scheduler.items() if not k.startswith("_")}
        ),
    )
    reference.set_progress_bar_config(disable=True)
    return config, reference, state


@pytest.mark.parametrize("name", ORIGINALS)
def test_original_headers_map_onto_complete_native_graph(name):
    data = original(name)
    validate_pipeline_index(data["model_index"])
    adapter.validate_configs(data["model_index"], data["config"], data["scheduler_config"])
    expected = adapter.expected_shapes(data["config"])
    assert expected == native_shapes(data["config"])
    shapes, native = adapter.validate_archive(data["config"], archive_metadata(data["header"]))
    assert set(native) == set(expected)
    renamed = sum(source != target for target, source in native.items())
    assert renamed == 8 * sum(".query.weight" in k for k in data["header"]) > 0
    assert ("time_proj.W" in shapes) is name.startswith("ncsnpp")


@pytest.mark.parametrize("family", ["ddpm", "sde"])
@pytest.mark.parametrize(
    "blocks",
    [
        None,
        ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        ("AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
    ],
)
def test_pure_derivation_matches_native_construction(family, blocks):
    _, config, _ = tiny(family)
    if blocks:
        config["block_out_channels"] = [32, 64, 64] if family == "sde" else [8, 16, 16]
        config["layers_per_block"] = 2
        config["down_block_types"] = [
            ("AttnSkipDownBlock2D" if "Attn" in b else "SkipDownBlock2D") if family == "sde" else b
            for b in blocks
        ]
        config["up_block_types"] = [
            (
                ("AttnSkipUpBlock2D" if "Attn" in b else "SkipUpBlock2D")
                if family == "sde"
                else b.replace("Down", "Up")
            )
            for b in reversed(blocks)
        ]
    assert adapter.expected_shapes(config) == native_shapes(config)


@pytest.mark.parametrize("family", ["ddpm", "sde"])
@pytest.mark.parametrize("legacy", [True, False])
def test_seeded_native_sampling_parity(tmp_path, family, legacy):
    _, reference, _ = fixture(tmp_path, family, legacy=legacy)
    loaded = adapter.load_pipeline(tmp_path)
    loaded.set_progress_bar_config(disable=True)
    noise = torch.randn(1, 3, 16, 16, generator=torch.Generator().manual_seed(42))
    with torch.inference_mode():
        for step in [1, 7, 15]:
            torch.testing.assert_close(
                loaded.unet(noise, step).sample, reference.unet(noise, step).sample, atol=0, rtol=0
            )
        actual = loaded(
            generator=torch.Generator().manual_seed(61), num_inference_steps=3, output_type="np"
        ).images
        expected = reference(
            generator=torch.Generator().manual_seed(61), num_inference_steps=3, output_type="np"
        ).images
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (1, 16, 16, 3) and np.isfinite(actual).all()


def test_float16_archive_loads_to_float32_native_graph(tmp_path):
    fixture(tmp_path, precision="float16")
    loaded = adapter.load_pipeline(tmp_path)
    assert {p.dtype for p in loaded.unet.parameters()} == {torch.float32}


@pytest.mark.parametrize(
    "change",
    [
        {"attention_head_dim": 8},
        {"in_channels": 4},
        {"sample_size": 31},
        {"layers_per_block": True},
        {"block_out_channels": [7, 16]},
        {"norm_num_groups": 3},
        {"down_block_types": ["CrossAttnDownBlock2D", "DownBlock2D"]},
        {"down_block_types": ["SkipDownBlock2D", "DownBlock2D"]},
        {"class_embed_type": "identity"},
        {"center_input_sample": 0},
        {"freq_shift": 2},
        {"time_embedding_type": "fourier"},
        {"mid_block_scale_factor": "1"},
        {"resnet_time_scale_shift": "scale_shift"},
    ],
)
def test_closed_ddpm_unet_configuration(change):
    index, config, scheduler = tiny("ddpm")
    with pytest.raises(ValueError):
        adapter.validate_configs(index, config | change, scheduler)


@pytest.mark.parametrize(
    "change",
    [
        {"norm_num_groups": 32},
        {"block_out_channels": [48, 64]},
        {"time_embedding_type": "positional"},
        {"up_block_types": ["AttnUpBlock2D", "UpBlock2D"]},
    ],
)
def test_closed_ncsnpp_unet_configuration(change):
    index, config, scheduler = tiny("sde")
    with pytest.raises(ValueError):
        adapter.validate_configs(index, config | change, scheduler)


@pytest.mark.parametrize(
    "family,change",
    [
        ("ddpm", {"beta_schedule": "scaled_linear"}),
        ("ddpm", {"variance_type": "learned"}),
        ("ddpm", {"prediction_type": "v_prediction"}),
        ("ddpm", {"trained_betas": [0.1, 0.2]}),
        ("ddpm", {"num_train_timesteps": True}),
        ("ddpm", {"_class_name": "DDIMScheduler"}),
        ("sde", {"sigma_min": 500}),
        ("sde", {"snr": 0}),
        ("sde", {"correct_steps": 9}),
        ("sde", {"_class_name": "DDPMScheduler"}),
    ],
)
def test_closed_scheduler_configuration(family, change):
    index, config, scheduler = tiny(family)
    with pytest.raises(ValueError):
        adapter.validate_configs(index, config, scheduler | change)


def test_pipeline_family_must_match_its_scheduler():
    index, config, scheduler = tiny("ddpm")
    sde_index, sde_config, sde_scheduler = tiny("sde")
    with pytest.raises(ValueError):
        adapter.validate_configs(sde_index, config, scheduler)
    with pytest.raises(ValueError):
        adapter.validate_configs(index, sde_config, sde_scheduler)


@pytest.mark.parametrize(
    "damage", ["missing", "extra", "shape", "integer", "duplicate", "alias-shape"]
)
def test_archive_headers_must_be_the_complete_floating_graph(damage):
    data = original("ncsnpp-celebahq-256")
    header = dict(data["header"])
    dtype = "F32"
    key = next(k for k in header if ".query.weight" in k)
    if damage == "missing":
        del header[key]
    elif damage == "extra":
        header["unreviewed.weight"] = [2]
    elif damage == "shape":
        header[key] = [1, 1]
    elif damage == "integer":
        dtype = "I64"
    elif damage == "duplicate":
        header[adapter.native_name(key)] = header[key]
    elif damage == "alias-shape":
        header["time_proj.W"] = [64]
    with pytest.raises(ValueError):
        adapter.validate_archive(data["config"], archive_metadata(header, dtype))


@pytest.mark.parametrize(
    "files,admitted",
    [
        (["model_index.json", "config.json", "scheduler_config.json", adapter.WEIGHTS], True),
        (["model_index.json", "config.json", adapter.WEIGHTS], False),
        (
            [
                "model_index.json",
                "config.json",
                "scheduler_config.json",
                adapter.WEIGHTS,
                "diffusion_pytorch_model.safetensors",
            ],
            False,
        ),
        (
            [
                "model_index.json",
                "config.json",
                "scheduler_config.json",
                adapter.WEIGHTS,
                "ema.pt",
            ],
            False,
        ),
    ],
)
def test_only_the_unambiguous_numeric_layout_is_selected(files, admitted):
    index, _, _ = tiny("ddpm")
    assert adapter.selected(index, files) is admitted
    assert not adapter.selected(index | {"_class_name": "LDMPipeline"}, files)


@pytest.mark.parametrize("damage", ["reducer", "nan", "alias", "symlink", "compressed"])
def test_executable_or_damaged_archives_are_rejected(tmp_path, damage):
    _, _, state = fixture(tmp_path, "sde")
    path = tmp_path / adapter.WEIGHTS
    if damage == "reducer":

        class Payload:
            def __reduce__(self):
                return (print, ("executed",))

        state["conv_in.bias"] = Payload()
        torch.save(state, path)
    elif damage == "nan":
        state["conv_in.bias"][0] = float("nan")
        torch.save(state, path)
    elif damage == "alias":
        state[adapter.FOURIER_ALIAS] = state[adapter.FOURIER_ALIAS] + 1
        torch.save(state, path)
    elif damage == "symlink":
        path.rename(tmp_path / "outside.data")
        path.symlink_to(tmp_path / "outside.data")
    elif damage == "compressed":
        data = path.read_bytes()
        source = tmp_path / "plain.zip"
        source.write_bytes(data)
        with (
            zipfile.ZipFile(source) as plain,
            zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as out,
        ):
            for item in plain.infolist():
                out.writestr(item.filename, plain.read(item))
    with pytest.raises(ValueError):
        adapter.load_pipeline(tmp_path)


def test_plain_pickle_is_never_loaded(tmp_path):
    fixture(tmp_path)
    (tmp_path / adapter.WEIGHTS).write_bytes(pickle.dumps({"conv_in.bias": [0.0]}))
    with pytest.raises(ValueError):
        adapter.load_pipeline(tmp_path)


def test_invalid_denoiser_output_fails_before_image_conversion(tmp_path):
    fixture(tmp_path)
    pipe = adapter.load_pipeline(tmp_path)
    pipe.unet.forward = lambda *args, **kwargs: SimpleNamespace(
        sample=torch.full((1, 3, 16, 16), float("nan"))
    )
    with pytest.raises(ValueError, match="NUMERIC_UNET2D_DENOISER_OUTPUT"):
        pipe(num_inference_steps=2, generator=torch.Generator().manual_seed(61))


@pytest.mark.parametrize("name", ORIGINALS)
@pytest.mark.parametrize("damage", [None, "header", "scheduler", "safetensors", "python"])
def test_original_metadata_routes_to_numeric_loader(tmp_path, name, damage):
    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore

    data = original(name)
    names = {"model_index.json", "config.json", "scheduler_config.json", adapter.WEIGHTS}
    header = dict(data["header"])
    if damage == "header":
        header.popitem()
    elif damage == "scheduler":
        data["scheduler_config"]["trained_betas"] = [0.1]
    elif damage == "safetensors":
        names.add("diffusion_pytorch_model.safetensors")
    elif damage == "python":
        data["model_index"]["unet"] = ["repo", "Code"]

    class Hub(FakeHub):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.archive_reads = []

        def read_json(self, repository, revision, filename):
            return data[filename.removesuffix(".json")]

        def read_numeric_archive(self, repository, revision, filename, size):
            self.archive_reads.append(filename)
            return archive_metadata(header)

    hub = Hub(
        config=data["model_index"],
        pipeline_tag="unconditional-image-generation",
        files=[(n, 100, "b" * 64) for n in names],
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda p: None if p.provider else object()
    )
    result = store.inspect(model("someone/numeric-unet"))
    selected = {f["name"] for f in result["selectedFiles"]}
    if damage == "safetensors":
        # The safe native export keeps precedence; the archive is neither selected nor read.
        assert adapter.WEIGHTS not in selected and not hub.archive_reads
        return
    assert result["executable"] is (damage is None), result
    if damage is None:
        assert result["runtimePack"]["taskContract"] == "unconditional-image-generation-v1"
        assert result["generationProfile"]["parameters"]["width"]["default"] == 256
        assert adapter.WEIGHTS in selected
        assert hub.archive_reads == [adapter.WEIGHTS]
    assert not hub.downloaded


@pytest.mark.parametrize("family", ["ddpm", "sde"])
def test_owned_worker_produces_repeatable_png(tmp_path, family):
    from PIL import Image

    from oryn_engine.runtime_packs.diffusers_worker import WorkerFailure, _generate

    fixture(tmp_path, family)
    output = tmp_path / "image.png"
    payload = dict(
        modelRoot=str(tmp_path),
        backend="cpu",
        taskContract="unconditional-image-generation-v1",
        input=dict(width=16, height=16, steps=3, seed=61),
        outputPath=str(output),
        previewPath=None,
        inputPath=None,
    )
    result = _generate(payload)
    assert result["mediaType"] == "image/png" and result["width"] == result["height"] == 16
    first = np.asarray(Image.open(output)).copy()
    _generate(payload)
    assert np.array_equal(first, np.asarray(Image.open(output)))
    with pytest.raises(WorkerFailure, match="MODEL_GENERATION_FAILED"):
        _generate(payload | {"input": payload["input"] | {"width": 32}})


def test_worker_still_rejects_other_pickle_suffixes(tmp_path):
    from oryn_engine.runtime_packs.diffusers_worker import _validate_model

    fixture(tmp_path)
    _validate_model(tmp_path)
    (tmp_path / "ema.pt").write_bytes(b"x")
    with pytest.raises(ValueError):
        _validate_model(tmp_path)
