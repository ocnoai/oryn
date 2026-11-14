"""Native-library integration, run with Oryn's pinned source-build Diffusers wheel."""

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from oryn_engine.runtime_packs.diffusers_worker import _generate_image, _pipeline, _validate_model


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "QwenImage21Pipeline"):
        pytest.skip("requires Oryn's pinned Qwen Image 2.1 Diffusers wheel")
    from qwen_image_21_fixture import write_checkpoint

    return write_checkpoint(tmp_path_factory.mktemp("qwen21-native"))


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_native_checkpoint_loads_and_generates_rgba_without_repository_template(
    checkpoint, tmp_path, monkeypatch, backend
):
    import torch

    from oryn_engine.runtime_packs import diffusers_worker

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS requires Apple Silicon")
    monkeypatch.setattr(diffusers_worker, "_backend", lambda _: (backend, torch.float32))
    _validate_model(checkpoint)
    pipe, selected = _pipeline(checkpoint, "text-to-image-v1", torch)
    assert selected == backend
    value = {
        "prompt": "A red circle",
        "negativePrompt": "",
        "width": 64,
        "height": 64,
        "steps": 2,
        "guidanceScale": 1.0,
        "seed": 7,
    }
    target = tmp_path / "image.png"
    _generate_image(pipe, "text-to-image-v1", value, None, target, None, torch)
    with Image.open(target) as image:
        assert image.mode == "RGBA" and image.size == (64, 64)
        assert image.getchannel("A").getextrema()[0] < 255


def test_native_single_image_edit_preserves_alpha(checkpoint, tmp_path, monkeypatch):
    import torch

    from oryn_engine.runtime_packs import diffusers_worker

    monkeypatch.setattr(diffusers_worker, "_backend", lambda _: ("cpu", torch.float32))
    pipe, _ = _pipeline(checkpoint, "image-editing-v1", torch)
    source = tmp_path / "source.png"
    Image.new("RGBA", (64, 64), (220, 40, 10, 80)).save(source)
    value = {
        "prompt": "Make it blue",
        "negativePrompt": "",
        "width": 64,
        "height": 64,
        "steps": 2,
        "guidanceScale": 1.0,
        "seed": 7,
    }
    target = tmp_path / "image.png"
    _generate_image(pipe, "image-editing-v1", value, source, target, None, torch)
    with Image.open(target) as image:
        assert image.mode == "RGBA" and image.size == (64, 64)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("contract", ["text-to-image-v1", "image-editing-v1"])
def test_native_turbo_grid_and_nested_processor_reach_generation(
    checkpoint, tmp_path, monkeypatch, backend, contract
):
    import torch

    from oryn_engine.runtime_packs import diffusers_worker

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS requires Apple Silicon")
    metadata = json.loads(
        (Path(__file__).parent / "fixtures/qwen-image-2.1-turbo.json").read_text()
    )
    model = tmp_path / "turbo"
    shutil.copytree(checkpoint, model)
    index = json.loads((model / "model_index.json").read_text())
    sigmas = metadata["modelIndex"]["sample_sigmas"]
    (model / "model_index.json").write_text(json.dumps({**index, "sample_sigmas": sigmas}))
    (model / "processor/preprocessor_config.json").unlink()
    (model / "processor/processor_config.json").write_text(json.dumps(metadata["processor"]))
    (model / "scheduler/scheduler_config.json").write_text(json.dumps(metadata["scheduler"]))
    monkeypatch.setattr(diffusers_worker, "_backend", lambda _: (backend, torch.float32))
    _validate_model(model)
    pipe, selected = _pipeline(model, contract, torch)
    assert selected == backend
    seen = []

    def observe(pipe, step, timestep, values):
        seen.append((step, float(timestep)))
        return values

    monkeypatch.setattr(diffusers_worker, "_preview_callback", lambda *_: observe)
    source = tmp_path / "source.png"
    Image.new("RGBA", (64, 64), (220, 40, 10, 80)).save(source)
    value = {
        "prompt": "A blue circle",
        "negativePrompt": "",
        "width": 64,
        "height": 64,
        "steps": 8,
        "guidanceScale": 1.0,
        "seed": 7,
    }
    target = tmp_path / "image.png"
    with pytest.raises(ValueError, match="requires 8 denoising steps"):
        _generate_image(pipe, contract, {**value, "steps": 40}, source, target, None, torch)
    assert not seen and not target.exists()
    _generate_image(pipe, contract, value, source, target, None, torch)
    expected = torch.tensor(sigmas + [0.0], device=pipe.scheduler.sigmas.device)
    assert torch.equal(pipe.scheduler.sigmas, expected)
    assert [step for step, _ in seen] == list(range(8))
    assert [timestep for _, timestep in seen] == pytest.approx(
        [sigma * 1000 for sigma in sigmas], rel=1e-6
    )
    with Image.open(target) as image:
        assert image.mode == "RGBA" and image.size == (64, 64)
