"""Ordered multi-image contracts and complete synthetic native image-editing graphs."""

import json
from pathlib import Path

import pytest
from oryn_protocol.task_contracts import validate_invocation
from PIL import Image

from oryn_engine.diffusion_tasks import validate_media_input
from oryn_engine.model_composition import binding
from oryn_engine.runtime_packs import diffusers_worker as worker
from oryn_engine.task_profiles import profile_for_execution

IDS = ["artifact_" + c * 24 for c in "abcd"]
BASE = {"imageArtifactId": IDS[0], "prompt": "Use image 1 and image 2"}
OUTFIT = binding(
    "ausboss/Qwen-Image-2.1-Outfit-Swap-Consistency-LoRA",
    "a4ec9064268e413f6cf79b3c037963c67760e044",
)


@pytest.mark.parametrize(
    "family", ["QwenImage21Pipeline", "QwenImageEditPlusPipeline", "Flux2KleinPipeline"]
)
def test_profiles_advertise_only_bounded_native_references(family):
    profile = profile_for_execution(
        {
            "runtimePack": {"packId": "oryn.diffusers-pytorch", "taskContract": "image-editing-v1"},
            "technicalCapabilities": {"architectures": [family]},
        }
    )
    assert profile["features"]["multipleReferenceImages"] == "supported"
    assert profile["limits"] == {"minReferenceImages": 1, "maxReferenceImages": 3}
    for count in range(3):
        validate_invocation(profile, BASE | {"referenceImageArtifactIds": IDS[1 : count + 1]})
    with pytest.raises(ValueError):
        validate_invocation(profile, BASE | {"referenceImageArtifactIds": IDS[1:]})


def test_outfit_requires_exactly_two_references_before_loading():
    profile = profile_for_execution(
        {
            "runtimePack": {
                "packId": "oryn.diffusers-pytorch",
                "taskContract": "image-editing-v1",
                "trustedMetadata": {"composition": OUTFIT},
            },
            "technicalCapabilities": {"architectures": ["QwenImage21Pipeline"]},
        }
    )
    validate_invocation(profile, BASE | {"referenceImageArtifactIds": [IDS[1]]})
    for value in (
        BASE,
        BASE | {"referenceImageArtifactIds": []},
        BASE | {"referenceImageArtifactIds": IDS[1:3]},
    ):
        with pytest.raises(ValueError):
            validate_invocation(profile, value)


@pytest.mark.parametrize(
    "references", [[IDS[0]], [IDS[1], IDS[1]], IDS[1:], ["/tmp/image.png"], None, 4]
)
def test_media_validator_rejects_invalid_reference_identity(references):
    with pytest.raises(ValueError):
        validate_media_input("image-editing-v1", BASE | {"referenceImageArtifactIds": references})


@pytest.mark.parametrize(
    "problem",
    ["missing", "unrequested", "relative", "absent", "type", "too-many", "wrong-contract"],
)
def test_worker_requires_exactly_bound_reference_paths(tmp_path, problem):
    path = tmp_path / "image.png"
    Image.new("RGB", (64, 64)).save(path)
    value = BASE | {"referenceImageArtifactIds": [IDS[1]]}
    paths, contract = [str(path)], "image-editing-v1"
    if problem == "missing":
        paths = []
    if problem == "unrequested":
        value = BASE
    if problem == "relative":
        paths = ["image.png"]
    if problem == "absent":
        paths = [str(tmp_path / "missing.png")]
    if problem == "type":
        paths = [5]
    if problem == "too-many":
        paths = [str(path)] * 3
    if problem == "wrong-contract":
        contract = "image-layers-v1"
    with pytest.raises(ValueError):
        worker._reference_paths({"referenceImagePaths": paths}, value, contract)


@pytest.fixture
def checkpoints(tmp_path):
    from qwen_image_21_fixture import write_checkpoint

    root = write_checkpoint(tmp_path / "qwen")
    try:
        yield root
    finally:
        for path in tmp_path.rglob("*.safetensors"):
            path.unlink()


@pytest.mark.parametrize("family", ["qwen21", "edit2511", "klein"])
@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_native_graph_uses_each_reference_in_order(checkpoints, tmp_path, family, backend):
    import numpy as np
    import torch

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Metal")
    torch.set_num_threads(1)
    root = checkpoints
    if family == "edit2511":
        from qwen_image_edit_fixture import write_checkpoint

        root = write_checkpoint(tmp_path / "edit", checkpoints / "processor")
    if family == "klein":
        from flux_klein_fixture import write_checkpoint

        root = write_checkpoint(tmp_path / "klein", checkpoints / "processor")
    worker._validate_model(root)
    pipe, selected = worker._pipeline(root, "image-editing-v1", torch, backend_request=backend)
    assert selected == backend
    paths = []
    for i, color in enumerate([(200, 20, 60), (10, 200, 20), (20, 30, 230)]):
        path = tmp_path / f"ref{i}.png"
        Image.new("RGB", (64, 64), color).save(path)
        paths.append(path)
    value = validate_media_input(
        "image-editing-v1",
        BASE
        | {"referenceImageArtifactIds": IDS[1:3], "width": 64, "height": 64, "steps": 2, "seed": 7},
    )
    # Observe the real complete transformer inputs as well as the rendered output.
    seen = []

    def capture(module, args, kwargs):
        seen.append(kwargs["hidden_states"].detach().float().cpu().clone())

    hook = pipe.transformer.register_forward_pre_hook(capture, with_kwargs=True)
    results = []
    inputs = []
    for order in [(0, 1, 2), (0, 2, 1), (0, 1, 0)]:
        seen.clear()
        output = tmp_path / ("output" + "".join(map(str, order)) + ".png")
        worker._generate_image(
            pipe,
            "image-editing-v1",
            value,
            paths[order[0]],
            output,
            None,
            torch,
            reference_paths=tuple(paths[i] for i in order[1:]),
        )
        assert Image.open(output).size == (64, 64)
        results.append(np.asarray(Image.open(output)))
        inputs.append(seen[0])
    hook.remove()
    assert not torch.equal(inputs[0], inputs[1])
    assert not torch.equal(inputs[0], inputs[2])
    # The miniature model's conditioning must reach its output, not just preprocessing.
    assert np.any(results[0] != results[2])


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_original_outfit_composition_runs_two_references_in_offline_worker(
    checkpoints, tmp_path, backend
):
    import os
    import subprocess
    import sys

    import torch
    from diffusers import QwenImage21Transformer2DModel
    from safetensors.torch import save_file
    from test_diffusion_lora import arrays

    from oryn_engine.model_composition import adapter_name

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Metal")
    model = QwenImage21Transformer2DModel.from_pretrained(
        checkpoints / "transformer", local_files_only=True
    )
    (checkpoints / "oryn_adapter").mkdir()
    save_file(arrays(model, fused=True), checkpoints / adapter_name(OUTFIT))
    paths = []
    for index, color in enumerate(["red", "blue"]):
        path = tmp_path / f"source{index}.png"
        Image.new("RGB", (64, 64), color).save(path)
        paths.append(path)
    output = tmp_path / "image.png"
    request = {
        "abi": "1",
        "packId": "oryn.diffusers-pytorch",
        "version": "0.7.0",
        "command": "generate",
        "payload": {
            "modelRoot": str(checkpoints),
            "outputPath": str(output),
            "inputPath": str(paths[0]),
            "referenceImagePaths": [str(paths[1])],
            "taskContract": "image-editing-v1",
            "backend": backend,
            "composition": OUTFIT,
            "input": BASE
            | {
                "referenceImageArtifactIds": [IDS[1]],
                "width": 64,
                "height": 64,
                "steps": 2,
                "seed": 7,
            },
        },
    }
    run = subprocess.run(
        [sys.executable, "-m", "oryn_engine.runtime_packs.diffusers_worker"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        timeout=90,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "ORYN_PACK_NETWORK_POLICY": "deny",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert run.returncode == 0, run.stderr
    response = json.loads(run.stdout.splitlines()[-1])
    assert response.get("ok") is True, (response, run.stderr)
    assert Image.open(output).size == (64, 64)
    pipe, _ = worker._pipeline(
        checkpoints, "image-editing-v1", torch, backend_request=backend, composition=OUTFIT
    )
    with pytest.raises(ValueError, match="reference images"):
        worker._generate_image(
            pipe,
            "image-editing-v1",
            validate_media_input("image-editing-v1", BASE),
            paths[0],
            output,
            None,
            torch,
        )


@pytest.mark.parametrize("missing", [False, True])
def test_runner_binds_ordered_images_by_id_and_rejects_absent_leases(tmp_path, missing):
    import sys
    import threading

    from oryn_engine.artifacts import ArtifactOutputStaging
    from oryn_engine.jobs import EngineJob
    from oryn_engine.runners import DiffusersPackHandle, JobExecutionError, RunnerRegistry
    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
    from oryn_engine.runtime_packs.execution import PackRuntime

    root = tmp_path / "cache" / "models--fixture--model" / "snapshots" / ("a" * 40)
    root.mkdir(parents=True)
    paths = []
    bindings = []
    for index in range(3):
        path = tmp_path / f"ref{index}.png"
        Image.new("RGB", (64, 64)).save(path)
        paths.append(path)
        bindings.append(
            {
                "artifactId": IDS[index],
                "internalPath": str(path),
                "mimeType": "image/png",
                "sizeBytes": path.stat().st_size,
            }
        )
    record = {
        "model": {"provider": "huggingface", "repository": "fixture/model", "revision": "a" * 40},
        "cacheRoot": str(tmp_path / "cache"),
        "runtimePack": {"packId": "oryn.diffusers-pytorch", "taskContract": "image-editing-v1"},
        "_runtimeCandidate": {"backend": "cpu"},
        "technicalCapabilities": {"architectures": ["QwenImage21Pipeline"]},
    }
    value = BASE | {
        "referenceImageArtifactIds": [IDS[2], IDS[1]],
        "width": 64,
        "height": 64,
        "steps": 2,
    }
    job = EngineJob(
        job_id="job_images",
        session_id="session_images",
        model=record["model"],
        input=value,
        input_artifacts=tuple(bindings[:2] if missing else reversed(bindings)),
        invocation_digest="digest",
        state="running",
        enqueued_at=0,
    )
    registry = RunnerRegistry(ArtifactOutputStaging(tmp_path / "outputs", "engine_images"))

    class Process:
        def invoke(self, pack, runtime, command, payload, **kwargs):
            assert payload["inputPath"] == str(paths[0])
            assert payload["referenceImagePaths"] == [str(paths[2]), str(paths[1])]
            Image.new("RGB", (64, 64)).save(payload["outputPath"])
            return {
                "relativeOutput": "image.png",
                "mediaType": "image/png",
                "backend": "cpu",
                "width": 64,
                "height": 64,
            }

    registry._pack_process = Process()
    handle = DiffusersPackHandle(
        record,
        builtin_registry().active_pack("oryn.diffusers-pytorch"),
        PackRuntime(Path(sys.executable).resolve(), None, inherit_base_runtime=True),
    )
    if missing:
        with pytest.raises(JobExecutionError):
            registry.execute(job, handle, threading.Event(), lambda *_: None)
    else:
        result = registry.execute(job, handle, threading.Event(), lambda *_: None)
        assert result["artifact"]["mimeType"] == "image/png"
