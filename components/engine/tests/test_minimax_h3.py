"""Metadata-only original review and complete generated H3 workflow regression tests."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from minimax_h3_fixture import FIXTURES, original, tiny_configs, tiny_fixture

from oryn_engine import minimax_h3 as adapter


def headers():
    return {
        p.relative_to(FIXTURES).as_posix().removesuffix(".header.json"): json.loads(p.read_text())
        for p in FIXTURES.rglob("*.header.json")
    }


def original_names():
    return [
        adapter.INDEX,
        *adapter.CONFIGS.values(),
        "tokenizer/tokenizer.json",
        *headers(),
        *(p.relative_to(FIXTURES).as_posix() for p in FIXTURES.rglob("*.index.json")),
    ]


def test_original_metadata_matches_every_native_t2va_parameter():
    c = adapter.inspect_metadata(
        original(adapter.INDEX), original, headers().__getitem__, original_names()
    )
    assert {k: len(v) for k, v in adapter.parameter_specs(c).items()} == dict(
        text_encoder=1058, transformer=638, vae=703, audio_vae=1087
    )


def test_original_metadata_matches_complete_native_meta_graphs():
    import torch

    c = {k: original(n) for k, n in adapter.CONFIGS.items()}
    with torch.device("meta"):
        components = adapter.native_components(c)
    assert {
        k: {n: tuple(t.shape) for n, t in m.state_dict().items()} for k, m in components.items()
    } == adapter.parameter_specs(c)


@pytest.mark.parametrize(
    "damage",
    ["class", "block", "library", "subfolder", "path", "variant", "revision", "extra", "code"],
)
def test_closed_component_binding(damage):
    index = original(adapter.INDEX)
    if damage == "class":
        index["_class_name"] = "Unknown"
    elif damage == "block":
        index["_blocks_class_name"] = "Repository"
    elif damage == "library":
        index["transformer"][0] = "repository"
    elif damage == "subfolder":
        index["transformer"][2]["subfolder"] = "../other"
    elif damage == "path":
        index["transformer"][2]["pretrained_model_name_or_path"] = "/tmp/model"
    elif damage == "variant":
        index["transformer"][2]["variant"] = "fp8"
    elif damage == "revision":
        index["transformer"][2]["revision"] = "main"
    elif damage == "extra":
        index["new_branch"] = ["diffusers", "AutoencoderKL"]
    else:
        index["auto_map"] = {"AutoPipeline": "repo.Code"}
    with pytest.raises(ValueError):
        adapter.validate_index(index)


@pytest.mark.parametrize(
    "component,key,value",
    [
        ("text_encoder", "tie_word_embeddings", True),
        ("text_encoder", "image_token_id", 3),
        ("transformer", "text_dim", 32),
        ("transformer", "rope_freq_dim", 4),
        ("transformer", "audio_in_channels", 8),
        ("transformer", "patch_size", [1, 4, 4]),
        ("transformer", "quantization_config", {"quant_method": "fp8"}),
        ("vae", "latent_channels", 8),
        ("vae", "temporal_downsample_factors", [1] * 6),
        ("vae", "clip_length", 16),
        ("vae", "token_drop", 4),
        ("vae", "latents_std", [0] * 4),
        ("vae", "decoder_attention_head_dim", 20),
        ("audio_vae", "sampling_rate", 44100),
        ("audio_vae", "decoder_rates", [8, 8, 4, 2]),
        ("audio_vae", "latent_dim", 24),
        ("scheduler", "shift", 3),
        ("audio_scheduler", "shift", 12),
        ("tokenizer", "add_prefix_space", True),
        ("processor", "patch_size", 14),
        ("video_processor", "size", {"longest_edge": 2**32, "shortest_edge": 64}),
    ],
)
def test_cross_component_and_numeric_geometry(component, key, value):
    c = tiny_configs()
    c[component][key] = value
    # 24 is legal dimensional arithmetic; force an incompatible channel divisor.
    if component == "audio_vae" and key == "latent_dim":
        c[component]["latent_channels"] = 10
    with pytest.raises(ValueError):
        adapter.validate_configs(c)


def test_native_hidden_state_fifty_cannot_be_replaced_by_a_final_normalized_state():
    c = tiny_configs()
    c["text_encoder"]["text_config"]["num_hidden_layers"] = 50
    with pytest.raises(ValueError, match="CONDITIONER_LAYER"):
        adapter.validate_configs(c)


@pytest.mark.parametrize("damage", ["missing", "shape", "dtype", "duplicate", "index", "extra"])
def test_all_tensor_headers_and_shard_ownership_are_required(damage):
    hs = headers()
    filename = next(n for n in hs if n.startswith("transformer/"))
    key = next(iter(hs[filename]["tensors"]))
    index = original("transformer/diffusion_pytorch_model.safetensors.index.json")
    if damage == "missing":
        del hs[filename]["tensors"][key]
    elif damage == "shape":
        hs[filename]["tensors"][key]["shape"][0] += 1
    elif damage == "dtype":
        hs[filename]["tensors"][key]["dtype"] = "I64"
    elif damage == "extra":
        hs[filename]["tensors"]["unbound.weight"] = dict(dtype="F32", shape=[1])
    elif damage == "index":
        index["weight_map"][key] = "../outside.safetensors"
    else:
        other = next(n for n in hs if n.startswith("transformer/") and n != filename)
        hs[other]["tensors"][key] = hs[filename]["tensors"][key]

    def read(n):
        return (
            index
            if n == "transformer/diffusion_pytorch_model.safetensors.index.json"
            else original(n)
        )

    with pytest.raises(ValueError):
        adapter.inspect_metadata(original(adapter.INDEX), read, hs.__getitem__, original_names())


@pytest.mark.parametrize(
    "damage", [None, "missing-tokenizer", "missing-audio", "header", "binding"]
)
@pytest.mark.parametrize("task", ["text-to-video", "image-text-to-video"])
def test_metadata_discovery_is_repository_independent_and_preserves_audio(tmp_path, damage, task):
    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore

    index, hs = original(adapter.INDEX), headers()
    names = original_names() + [
        "repository.py",
        "T2VA/unused.safetensors",
        "transformer_ref/diffusion_pytorch_model.safetensors",
    ]
    if damage == "missing-tokenizer":
        names.remove("tokenizer/tokenizer.json")
    if damage == "missing-audio":
        names = [n for n in names if not n.startswith("audio_vae/")]
    if damage == "binding":
        index["transformer"][2]["subfolder"] = "transformer_ref"

    class Hub(FakeHub):
        def read_json(self, repo, revision, name):
            return index if name == adapter.INDEX else original(name)

        def read_tensor_structure(self, repo, revision, name):
            value = hs[name]
            if damage == "header" and name.startswith("audio_vae/"):
                value["tensors"].pop("mean_proj.weight", None)
            return value

    hub = Hub(config=index, pipeline_tag=task, files=[(n, 100, "a" * 64) for n in names])
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda p: None if p.provider else object()
    )
    result = store.inspect(model("independent/renamed-video"))
    assert result["executable"] is (damage is None), result
    assert not hub.downloaded
    if damage is None:
        assert result["runtimePack"]["taskContract"] == task + "-v1"
        assert result["taskProfile"]["parameters"]["frames"]["default"] == 124
        assert result["taskProfile"]["parameters"]["fps"]["default"] == 24
        if task == "text-to-video":
            assert result["taskProfile"]["features"]["negativePrompt"] == "unsupported"
        assert not any(
            n["name"].startswith(("transformer_ref/", "T2VA/")) or n["name"].endswith(".py")
            for n in result["selectedFiles"]
        )
        assert any(n["name"].startswith("audio_vae/") for n in result["selectedFiles"])


@pytest.mark.parametrize(
    "backend,count,storage",
    [
        ("cpu", 124, "float32"),
        ("mps", 124, "float32"),
        ("cpu", 158, "float32"),
        ("cpu", 345, "float32"),
        ("cpu", 124, "float16"),
        ("cpu", 124, "bfloat16"),
    ],
)
def test_complete_graph_matches_native_video_and_soundtrack(
    tmp_path, monkeypatch, backend, count, storage
):
    import torch
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
    from transformers import Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

    from oryn_engine import video_audio_artifacts as media

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    c, components = tiny_fixture(tmp_path, weight_dtype=getattr(torch, storage))
    tok = adapter.load_tokenizer(tmp_path)
    pipe = MiniMaxH3ModularPipeline(blocks=MiniMaxH3Blocks().get_workflow("t2va"))
    pipe.update_components(
        **components,
        tokenizer=tok,
        processor=Qwen3VLProcessor(
            tokenizer=tok,
            image_processor=Qwen2VLImageProcessor(
                patch_size=16, temporal_patch_size=2, merge_size=2
            ),
            video_processor=Qwen3VLVideoProcessor(
                patch_size=16, temporal_patch_size=2, merge_size=2
            ),
        ),
        scheduler=MiniMaxH3Scheduler(shift=12),
        audio_scheduler=MiniMaxH3Scheduler(shift=3),
    )
    pipe.to(backend)
    with torch.inference_mode():
        expected = pipe(
            prompt="A red bird sings.",
            width=64,
            height=64,
            num_frames=count,
            num_inference_steps=3,
            generator=torch.Generator().manual_seed(7),
            output_type="np",
            output=["videos", "audio", "sampling_rate"],
        )
    write = media.write_video_audio
    seen = []

    def check(path, frames, audio, **kwargs):
        np.testing.assert_allclose(frames, expected["videos"][0], atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(
            audio, expected["audio"][0].detach().cpu().numpy(), atol=1e-6, rtol=1e-5
        )
        seen.append(True)
        return write(path, frames, audio, **kwargs)

    monkeypatch.setattr(media, "write_video_audio", check)
    v = adapter.validate_input(
        dict(prompt="A red bird sings.", width=64, height=64, frames=count, steps=2, seed=7)
    )
    events = []
    output = tmp_path / "video.mp4"
    result = adapter.generate(tmp_path, output, backend, v, lambda s, p: events.append((s, p)))
    media.validate_video_audio(output, result, v)
    assert seen == [True] and result["sampleRate"] == 32000 and result["channels"] == 2
    assert all(a[1] <= b[1] for a, b in zip(events, events[1:], strict=False))


def test_repeatability_and_both_output_branches_respond_to_prompt_and_seed(tmp_path):
    import torch

    c, components = tiny_fixture(tmp_path)
    pipe = adapter.native_pipeline(c, components, adapter.load_tokenizer(tmp_path))

    def run(prompt="A red bird.", seed=1):
        with torch.inference_mode():
            return pipe(
                prompt=prompt,
                height=64,
                width=64,
                num_frames=124,
                num_inference_steps=3,
                generator=torch.Generator().manual_seed(seed),
                output_type="np",
                output=["videos", "audio"],
            )

    first = run()
    again = run()
    np.testing.assert_array_equal(first["videos"], again["videos"])
    torch.testing.assert_close(first["audio"], again["audio"], rtol=0, atol=0)
    for other in [run(prompt="A loud blue train."), run(seed=2)]:
        assert np.max(np.abs(first["videos"] - other["videos"])) > 1e-6
        assert torch.max(torch.abs(first["audio"] - other["audio"])).item() > 1e-6


@pytest.mark.parametrize("damage", ["missing", "shape", "dtype", "nonfinite", "extra", "tokenizer"])
def test_loader_rejects_incomplete_numeric_graphs(tmp_path, damage):
    import torch
    from safetensors.torch import load_file, save_file

    tiny_fixture(tmp_path)
    path = tmp_path / "audio_vae/diffusion_pytorch_model.safetensors"
    w = load_file(path)
    if damage == "missing":
        del w["mean_proj.weight"]
    elif damage == "shape":
        w["mean_proj.weight"] = w["mean_proj.weight"][:-1]
    elif damage == "dtype":
        w["mean_proj.weight"] = w["mean_proj.weight"].long()
    elif damage == "nonfinite":
        w["mean_proj.weight"][0, 0, 0] = float("nan")
    elif damage == "extra":
        w["unbound.weight"] = torch.zeros(1)
    else:
        (tmp_path / "tokenizer/tokenizer.json").write_text("{}")
    save_file(w, path)
    with pytest.raises(ValueError):
        adapter.load_pipeline(tmp_path, "cpu")


@pytest.mark.parametrize(
    "update",
    [
        dict(frames=120),
        dict(frames=125),
        dict(frames=360),
        dict(fps=30),
        dict(seed=-1),
        dict(seed=True),
        dict(width=72),
        dict(width=1024, height=1024),
        dict(guidanceScale=7.5),
        dict(negativePrompt="blur"),
        dict(prompt=""),
        dict(prompt="<|video_pad|>"),
        dict(other=True),
        dict(width=1024, height=64),
    ],
)
def test_request_geometry_and_unsupported_controls_are_rejected(update):
    with pytest.raises(ValueError):
        adapter.validate_input(dict(prompt="Bird") | update)


@pytest.mark.parametrize("conditioned", [False, True])
def test_repeated_offline_workers_publish_both_streams(tmp_path, conditioned):
    import os
    import subprocess
    import sys

    from oryn_engine.video_audio_artifacts import validate_video_audio

    tiny_fixture(tmp_path)
    v = dict(prompt="A red bird sings.", width=64, height=64, frames=124, steps=1, seed=7)
    payload = dict(
        modelRoot=str(tmp_path),
        outputPath=str(tmp_path / "video.mp4"),
        taskContract=adapter.CONTRACT,
        backend="cpu",
        input=v,
    )
    contract = adapter.IMAGE_CONTRACT if conditioned else adapter.CONTRACT
    payload["taskContract"] = contract
    if conditioned:
        from PIL import Image

        source = tmp_path / "keyframe.png"
        Image.new("RGB", (96, 64), "red").save(source)
        payload["inputPath"] = str(source)
        v["imageArtifactId"] = "artifact_" + "a" * 24
    outputs = []
    for _ in range(2):
        request = dict(
            abi="1",
            packId="oryn.diffusers-pytorch",
            version="0.11.0",
            command="generate",
            payload=payload,
        )
        run = subprocess.run(
            [sys.executable, "-m", "oryn_engine.runtime_packs.diffusers_worker"],
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            timeout=90,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        assert run.returncode == 0, run.stderr
        result = [json.loads(line) for line in run.stdout.splitlines()][-1]
        assert result["ok"], result
        validate_video_audio(
            tmp_path / "video.mp4", result["result"], adapter.validate_input(v, contract)
        )
        outputs.append((tmp_path / "video.mp4").read_bytes())
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("stage", ["generating-video-audio", "encoding-video-audio"])
def test_native_cancellation_removes_partial_artifacts(tmp_path, stage):
    tiny_fixture(tmp_path)
    count = 0

    def event(name, progress):
        nonlocal count
        if name == stage:
            count += 1
            if count == 2:
                raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError, match="cancelled"):
        adapter.generate(
            tmp_path,
            tmp_path / "video.mp4",
            "cpu",
            dict(prompt="Bird", width=64, height=64, steps=2),
            event,
        )
    assert not (tmp_path / "video.mp4").exists()


@pytest.mark.parametrize(
    "damage", [None, "rate", "frames", "samples", "truncated", "silent-video", "nonfinite"]
)
def test_artifact_requires_both_complete_tracks_and_matching_metadata(tmp_path, damage):
    import av

    from oryn_engine.video_audio_artifacts import validate_video_audio, write_video_audio

    path = tmp_path / "video.mp4"
    value = adapter.validate_input(dict(prompt="Bird", width=64, height=64))
    frames = np.zeros((124, 64, 64, 3), dtype=np.uint8)
    audio = np.zeros((2, 165600), dtype=np.float32)
    if damage == "nonfinite":
        audio[0, 0] = float("nan")
        with pytest.raises(ValueError):
            write_video_audio(path, frames, audio, fps=24, sample_rate=32000, event=lambda *_: None)
        assert not path.exists()
        return
    samples = write_video_audio(
        path, frames, audio, fps=24, sample_rate=32000, event=lambda *_: None
    )
    result = dict(
        relativeOutput=path.name,
        mediaType="video/mp4",
        sampleRate=32000,
        channels=2,
        samples=samples,
        **{k: value[k] for k in ("width", "height", "frames", "fps")},
    )
    if damage == "rate":
        result["sampleRate"] = 44100
    elif damage == "frames":
        result["frames"] = 123
    elif damage == "samples":
        result["samples"] = 165000
    elif damage == "truncated":
        path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 2])
    elif damage == "silent-video":
        other = tmp_path / "no-audio.mp4"
        with av.open(str(path)) as source, av.open(str(other), "w") as target:
            stream = target.add_stream_from_template(source.streams.video[0])
            for packet in source.demux(video=0):
                if packet.dts is not None:
                    packet.stream = stream
                    target.mux(packet)
        other.replace(path)
    if damage:
        with pytest.raises(ValueError):
            validate_video_audio(path, result, value)
    else:
        validate_video_audio(path, result, value)


@pytest.mark.parametrize("damage", [None, "audio", "backend", "cancel"])
@pytest.mark.parametrize("contract", [adapter.CONTRACT, adapter.IMAGE_CONTRACT])
def test_shared_runner_checks_video_soundtrack_and_ownership(tmp_path, damage, contract):
    import sys
    import threading
    from pathlib import Path

    from oryn_engine.artifacts import ArtifactOutputStaging
    from oryn_engine.jobs import EngineJob, JobExecutionError
    from oryn_engine.runners import DiffusersPackHandle, RunnerRegistry
    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
    from oryn_engine.runtime_packs.execution import PackRuntime
    from oryn_engine.video_audio_artifacts import write_video_audio

    revision = "a" * 40
    cache = tmp_path / "models"
    root = cache / "models--fixture--video" / "snapshots" / revision
    root.mkdir(parents=True)
    record = dict(
        model=dict(provider="huggingface", repository="fixture/video", revision=revision),
        cacheRoot=str(cache),
        runtimePack=dict(packId="oryn.diffusers-pytorch", taskContract=contract),
        _runtimeCandidate=dict(backend="cpu"),
        technicalCapabilities=dict(architectures=[adapter.FAMILY]),
    )
    outputs = ArtifactOutputStaging(tmp_path / "outputs", "engine_h3")
    registry = RunnerRegistry(outputs)
    cancel = threading.Event()
    input_value = dict(prompt="Bird", width=64, height=64)
    bindings = ()
    if contract == adapter.IMAGE_CONTRACT:
        from PIL import Image

        image_path = tmp_path / "leased-keyframe.png"
        Image.new("RGB", (64, 64), "red").save(image_path)
        input_value["imageArtifactId"] = "artifact_" + "a" * 24
        bindings = (
            dict(
                artifactId=input_value["imageArtifactId"],
                mimeType="image/png",
                internalPath=str(image_path),
                sizeBytes=image_path.stat().st_size,
            ),
        )

    class Process:
        def invoke(self, pack, runtime, command, payload, **kwargs):
            assert payload["taskContract"] == contract and payload["backend"] == "cpu"
            v = payload["input"]
            assert v["fps"] == 24 and v["frames"] == 124
            if contract == adapter.CONTRACT:
                assert v["negativePrompt"] == "" and payload.get("inputPath") is None
            else:
                assert payload["inputPath"] == bindings[0]["internalPath"]
            path = Path(payload["outputPath"])
            samples = write_video_audio(
                path,
                np.zeros((124, 64, 64, 3), dtype=np.uint8),
                np.zeros((2, 165600), dtype=np.float32),
                fps=24,
                sample_rate=32000,
                event=lambda *_: None,
            )
            if damage == "cancel":
                cancel.set()
            return dict(
                relativeOutput=path.name,
                mediaType="video/mp4",
                backend="mps" if damage == "backend" else "cpu",
                sampleRate=16000 if damage == "audio" else 32000,
                channels=2,
                samples=samples,
                **{k: v[k] for k in ("width", "height", "frames", "fps")},
            )

    registry._pack_process = Process()
    handle = DiffusersPackHandle(
        record,
        builtin_registry().active_pack("oryn.diffusers-pytorch"),
        PackRuntime(Path(sys.executable).resolve(), None, inherit_base_runtime=True),
    )
    job = EngineJob(
        job_id="job_h3",
        session_id="session_h3",
        model=record["model"],
        input=input_value,
        input_artifacts=bindings,
        invocation_digest="digest",
        state="running",
        enqueued_at=0,
    )
    if damage:
        with pytest.raises(JobExecutionError) as exc:
            registry.execute(job, handle, cancel, lambda *_: None)
        assert exc.value.code == ("JOB_CANCELLED" if damage == "cancel" else "PACK_OUTPUT_INVALID")
        assert not outputs.job_root(job.job_id).exists()
    else:
        result = registry.execute(job, handle, cancel, lambda *_: None)
        assert (
            result["artifact"]["mimeType"] == "video/mp4"
            and Path(result["artifact"]["internalPath"]).is_file()
        )


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_keyframe_conditioning_matches_native_vision_encoder_and_latent_anchor(
    tmp_path, monkeypatch, backend
):
    import torch
    from PIL import Image

    from oryn_engine import video_audio_artifacts as media

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    c, components = tiny_fixture(tmp_path)
    pipe = adapter.native_pipeline(
        c, components, adapter.load_tokenizer(tmp_path), adapter.IMAGE_CONTRACT
    )
    pipe.to(backend)
    image = Image.new("RGB", (96, 64), "red")
    source = tmp_path / "keyframe.png"
    image.save(source)

    def native(im):
        with torch.inference_mode():
            return pipe(
                prompt="A red bird sings.",
                image=im,
                width=64,
                height=64,
                num_frames=124,
                num_inference_steps=3,
                generator=torch.Generator().manual_seed(7),
                output_type="np",
                output=["videos", "audio", "sampling_rate"],
            )

    expected = native(image)
    other = native(Image.new("RGB", (96, 64), "blue"))
    assert np.max(np.abs(expected["videos"] - other["videos"])) > 1e-6
    assert torch.max(torch.abs(expected["audio"] - other["audio"])).item() > 1e-6
    assert tuple(pipe.processor.image_processor.image_mean) == (0.5,) * 3
    assert pipe.processor.image_processor.size == c["processor"]["size"]
    writer = media.write_video_audio

    def check(path, frames, audio, **kwargs):
        np.testing.assert_allclose(frames, expected["videos"][0], atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(
            audio, expected["audio"][0].detach().cpu().numpy(), atol=1e-6, rtol=1e-5
        )
        return writer(path, frames, audio, **kwargs)

    monkeypatch.setattr(media, "write_video_audio", check)
    v = adapter.validate_input(
        dict(
            prompt="A red bird sings.",
            width=64,
            height=64,
            frames=124,
            steps=2,
            seed=7,
            imageArtifactId="artifact_" + "a" * 24,
        ),
        adapter.IMAGE_CONTRACT,
    )
    output = tmp_path / "video.mp4"
    result = adapter.generate(
        tmp_path,
        output,
        backend,
        v,
        lambda *_: None,
        contract=adapter.IMAGE_CONTRACT,
        input_path=source,
    )
    media.validate_video_audio(output, result, v)
