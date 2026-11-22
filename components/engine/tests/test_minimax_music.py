"""Pinned metadata and complete generated music graphs, with no learned downloads."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from minimax_music_fixture import native_pipeline, original, tiny_configs, tiny_fixture

from oryn_engine import minimax_music as adapter


def original_names():
    return [
        adapter.INDEX,
        *adapter.CONFIGS.values(),
        "tokenizer/tokenizer.json",
        *original("headers.json"),
        "language_model/model.safetensors.index.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    ]


def test_original_component_graph_is_complete_without_importing_a_native_backend():
    headers = original("headers.json")
    configs = adapter.inspect_metadata(
        original(adapter.INDEX), original, headers.__getitem__, original_names()
    )
    specs = adapter.parameter_specs(configs)
    assert {n: len(v) for n, v in specs.items()} == dict(
        language_model=399, rvq_depth_decoder=47, condition_encoder=4, transformer=441, vocoder=121
    )


def test_all_original_shapes_match_complete_native_meta_graphs():
    import torch

    configs = {k: original(n) for k, n in adapter.CONFIGS.items()}
    with torch.device("meta"):
        components = adapter.native_components(configs)
    assert {
        k: {n: tuple(t.shape) for n, t in m.state_dict().items()} for k, m in components.items()
    } == adapter.parameter_specs(configs)


@pytest.mark.parametrize(
    "damage",
    ["class", "block", "library", "subfolder", "path", "variant", "revision", "extra", "code"],
)
def test_unreviewed_pipeline_and_component_bindings_are_rejected(damage):
    index = original(adapter.INDEX)
    if damage == "class":
        index["_class_name"] = "UnknownPipeline"
    elif damage == "block":
        index["_blocks_class_name"] = "RepositoryBlocks"
    elif damage == "library":
        index["transformer"][0] = "repository"
    elif damage == "subfolder":
        index["transformer"][2]["subfolder"] = "../elsewhere"
    elif damage == "path":
        index["transformer"][2]["pretrained_model_name_or_path"] = "/tmp/elsewhere"
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
        ("language_model", "vocab_size", 151675),
        ("language_model", "attention_bias", True),
        ("language_model", "tie_word_embeddings", True),
        ("language_model", "head_dim", 7),
        ("language_model", "quantization_config", {"quant_method": "fp8"}),
        ("rvq_depth_decoder", "hidden_size", 32),
        ("rvq_depth_decoder", "num_codebooks", 4),
        ("rvq_depth_decoder", "audio_vocab_size", 512),
        ("condition_encoder", "output_sampling_rate", 32000),
        ("condition_encoder", "input_hop_length", 480),
        ("transformer", "condition_dim", 32),
        ("transformer", "rotary_dim", 16),
        ("vocoder", "latent_channels", 8),
        ("vocoder", "upsampling_ratios", [8, 8, 2, 2]),
        ("scheduler", "invert_sigmas", False),
        ("tokenizer", "tokenizer_class", "RepositoryTokenizer"),
    ],
)
def test_component_geometry_and_cross_component_contracts_are_closed(component, key, value):
    configs = tiny_configs()
    configs[component][key] = value
    with pytest.raises(ValueError):
        adapter.validate_configs(configs)


@pytest.mark.parametrize("damage", ["missing", "shape", "dtype", "duplicate", "index", "extra"])
def test_headers_bind_every_component_parameter_and_shard(damage):
    headers = original("headers.json")
    filename = next(n for n in headers if n.startswith("transformer/"))
    name = next(iter(headers[filename]["tensors"]))
    index = original("transformer/diffusion_pytorch_model.safetensors.index.json")
    if damage == "missing":
        del headers[filename]["tensors"][name]
    elif damage == "shape":
        headers[filename]["tensors"][name]["shape"][0] += 1
    elif damage == "dtype":
        headers[filename]["tensors"][name]["dtype"] = "I64"
    elif damage == "extra":
        headers[filename]["tensors"]["unbound.weight"] = dict(dtype="F32", shape=[1])
    elif damage == "index":
        index["weight_map"][name] = "../outside.safetensors"
    else:
        other = next(n for n in headers if n.startswith("transformer/") and n != filename)
        headers[other]["tensors"][name] = headers[filename]["tensors"][name]

    def read(n):
        return (
            index
            if n == "transformer/diffusion_pytorch_model.safetensors.index.json"
            else original(n)
        )

    with pytest.raises(ValueError):
        adapter.inspect_metadata(
            original(adapter.INDEX), read, headers.__getitem__, original_names()
        )


@pytest.mark.parametrize(
    "damage", [None, "missing-tokenizer", "missing-component", "header", "binding"]
)
def test_discovery_uses_original_metadata_and_preserves_renamed_repository_identity(
    tmp_path, damage
):
    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore

    index, headers = original(adapter.INDEX), original("headers.json")
    names = original_names() + ["config.json", "repository.py", "qwen_7B/unused.safetensors"]
    if damage == "missing-tokenizer":
        names.remove("tokenizer/tokenizer.json")
    if damage == "missing-component":
        names = [n for n in names if not n.startswith("vocoder/")]
    if damage == "binding":
        index["transformer"][2]["subfolder"] = "other"

    class Hub(FakeHub):
        def read_json(self, repo, revision, name):
            return index if name == adapter.INDEX else original(name)

        def read_tensor_structure(self, repo, revision, name):
            value = headers[name]
            if damage == "header" and name.startswith("vocoder/"):
                value["tensors"].pop("snake_out.alpha", None)
            return value

    hub = Hub(config=index, pipeline_tag="text-to-audio", files=[(n, 100, "a" * 64) for n in names])
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda p: None if p.provider else object()
    )
    result = store.inspect(model("independent/renamed-music"))
    assert result["executable"] is (damage is None), result
    assert not hub.downloaded
    if damage is None:
        assert result["runtimePack"]["packId"] == "oryn.diffusers-pytorch"
        assert result["runtimePack"]["taskContract"] == "text-to-music-v1"
        assert result["technicalCapabilities"]["outputModalities"] == ["audio"]
        assert result["taskProfile"]["features"]["lyrics"] == "supported"
        assert result["taskProfile"]["features"]["deterministicSeed"] == "supported"
        assert not any(
            n["name"].endswith(".py") or n["name"].startswith("qwen_7B/")
            for n in result["selectedFiles"]
        )


@pytest.mark.parametrize("backend,seconds", [("cpu", 1), ("mps", 1), ("cpu", 9)])
def test_complete_owned_graph_matches_native_pipeline_and_chunk_boundaries(
    tmp_path, backend, seconds
):
    import soundfile as sf
    import torch

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    configs, components = tiny_fixture(tmp_path)
    native = native_pipeline(tmp_path, configs, components, backend)
    with torch.inference_mode():
        expected = native(
            prompt="Warm piano melody",
            lyrics="[Instrumental]",
            audio_duration=seconds,
            num_inference_steps=30,
            generator=torch.Generator("cpu").manual_seed(7),
            output_type="np",
            output="audios",
        )
    events = []
    output = tmp_path / "audio.wav"
    result = adapter.generate(
        tmp_path,
        output,
        backend,
        dict(prompt="Warm piano melody", lyrics="[Instrumental]", durationSeconds=seconds, seed=7),
        lambda s, p: events.append((s, p)),
    )
    adapter.validate_audio_artifact(output, result, seconds)
    actual, rate = sf.read(output, dtype="float32", always_2d=True)
    np.testing.assert_allclose(
        actual, expected[0, :, : seconds * adapter.SAMPLE_RATE].T, atol=1 / 32768, rtol=0
    )
    assert rate == adapter.SAMPLE_RATE and result["channels"] == 2
    assert events[0][0] == "loading-music-components" and events[-1][0] == "encoding-audio"
    assert all(a[1] <= b[1] for a, b in zip(events, events[1:], strict=False))


def test_seed_repeatability_and_prompt_lyrics_conditioning(tmp_path):
    import torch

    configs, components = tiny_fixture(tmp_path)
    pipe = native_pipeline(tmp_path, configs, components)

    def run(prompt="Soft piano", lyrics="[Instrumental]", seed=7):
        with torch.inference_mode():
            return pipe(
                prompt=prompt,
                lyrics=lyrics,
                audio_duration=1,
                num_inference_steps=30,
                generator=torch.Generator("cpu").manual_seed(seed),
                output_type="np",
                output="audios",
            )

    first = run()
    np.testing.assert_array_equal(first, run())
    for other in [run(prompt="Loud drums"), run(lyrics="[verse]\nHello melody"), run(seed=8)]:
        assert np.max(np.abs(first - other)) > 1e-6


@pytest.mark.parametrize("damage", ["missing", "shape", "dtype", "nonfinite", "extra", "tokenizer"])
def test_worker_loader_rejects_incomplete_or_corrupt_graph(tmp_path, damage):
    import torch
    from safetensors.torch import load_file, save_file

    tiny_fixture(tmp_path)
    path = tmp_path / "condition_encoder/diffusion_pytorch_model.safetensors"
    weights = load_file(path)
    if damage == "missing":
        del weights["layer_scale"]
    elif damage == "shape":
        weights["proj.weight"] = weights["proj.weight"][:-1]
    elif damage == "dtype":
        weights["proj.weight"] = weights["proj.weight"].long()
    elif damage == "nonfinite":
        weights["proj.weight"][0, 0, 0] = float("nan")
    elif damage == "extra":
        weights["unbound.weight"] = torch.zeros(1)
    else:
        (tmp_path / "tokenizer/tokenizer.json").write_text("{}")
    save_file(weights, path)
    with pytest.raises(ValueError):
        adapter.load_pipeline(tmp_path, "cpu")


@pytest.mark.parametrize(
    "update",
    [
        dict(durationSeconds=0),
        dict(durationSeconds=31),
        dict(seed=True),
        dict(seed=-1),
        dict(prompt=""),
        dict(prompt="<|audio_start|>"),
        dict(lyrics=""),
        dict(other=True),
    ],
)
def test_bounded_music_request_rejects_unsupported_input(update):
    with pytest.raises(ValueError):
        adapter.validate_input(dict(prompt="Piano") | update)


def test_repeated_offline_worker_runs(tmp_path):
    import os
    import subprocess
    import sys

    tiny_fixture(tmp_path)
    payload = dict(
        modelRoot=str(tmp_path),
        outputPath=str(tmp_path / "audio.wav"),
        taskContract=adapter.CONTRACT,
        backend="cpu",
        input=dict(prompt="Warm piano melody", durationSeconds=1, seed=7),
    )
    outputs = []
    for _ in range(2):
        request = dict(
            abi="1",
            packId="oryn.diffusers-pytorch",
            version="0.8.0",
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
        adapter.validate_audio_artifact(tmp_path / "audio.wav", result["result"], 1)
        outputs.append((tmp_path / "audio.wav").read_bytes())
    assert outputs[0] == outputs[1]


def test_native_generation_can_be_interrupted_before_artifact_creation(tmp_path):
    tiny_fixture(tmp_path)
    events = []

    def event(stage, progress):
        events.append(stage)
        if len(events) == 5:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError, match="cancelled"):
        adapter.generate(
            tmp_path, tmp_path / "audio.wav", "cpu", dict(prompt="Piano", durationSeconds=1), event
        )
    assert events[-1] == "generating-music" and not (tmp_path / "audio.wav").exists()


@pytest.mark.parametrize("damage", [None, "rate", "truncated", "backend", "cancel"])
def test_shared_runner_stages_only_valid_music_and_honors_cancellation(tmp_path, damage):
    import sys
    import threading
    import wave
    from pathlib import Path

    from oryn_engine.artifacts import ArtifactOutputStaging
    from oryn_engine.jobs import EngineJob, JobExecutionError
    from oryn_engine.runners import DiffusersPackHandle, RunnerRegistry
    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
    from oryn_engine.runtime_packs.execution import PackRuntime

    revision = "a" * 40
    cache = tmp_path / "models"
    root = cache / "models--fixture--music" / "snapshots" / revision
    root.mkdir(parents=True)
    record = dict(
        model=dict(provider="huggingface", repository="fixture/music", revision=revision),
        cacheRoot=str(cache),
        runtimePack=dict(packId="oryn.diffusers-pytorch", taskContract=adapter.CONTRACT),
        _runtimeCandidate=dict(backend="cpu"),
    )
    outputs = ArtifactOutputStaging(tmp_path / "outputs", "engine_music")
    registry = RunnerRegistry(outputs)
    cancellation = threading.Event()

    class Process:
        def invoke(self, pack, runtime, command, payload, **kwargs):
            assert payload["taskContract"] == adapter.CONTRACT and payload["backend"] == "cpu"
            assert payload["input"] == dict(
                prompt="Piano", durationSeconds=1, lyrics="[Instrumental]", seed=0
            )
            assert payload["inputPath"] is None and payload["previewPath"] is None
            path = Path(payload["outputPath"])
            with wave.open(str(path), "wb") as f:
                f.setparams(
                    (
                        2,
                        2,
                        32000 if damage == "rate" else adapter.SAMPLE_RATE,
                        100,
                        "NONE",
                        "not compressed",
                    )
                )
                f.writeframes(b"\0" * 400)
            if damage == "truncated":
                path.write_bytes(path.read_bytes()[:-4])
            if damage == "cancel":
                cancellation.set()
            return dict(
                relativeOutput=path.name,
                mediaType="audio/wav",
                sampleRate=adapter.SAMPLE_RATE,
                channels=2,
                samples=100,
                backend="mps" if damage == "backend" else "cpu",
            )

    registry._pack_process = Process()
    handle = DiffusersPackHandle(
        record,
        builtin_registry().active_pack("oryn.diffusers-pytorch"),
        PackRuntime(Path(sys.executable).resolve(), None, inherit_base_runtime=True),
    )
    job = EngineJob(
        job_id="job_music",
        session_id="session_music",
        model=record["model"],
        input=dict(prompt="Piano", durationSeconds=1),
        input_artifacts=(),
        invocation_digest="digest",
        state="running",
        enqueued_at=0,
    )
    if damage:
        with pytest.raises(JobExecutionError) as error:
            registry.execute(job, handle, cancellation, lambda *_: None)
        assert error.value.code == (
            "JOB_CANCELLED" if damage == "cancel" else "PACK_OUTPUT_INVALID"
        )
        assert not outputs.job_root(job.job_id).exists()
    else:
        result = registry.execute(job, handle, cancellation, lambda *_: None)
        assert result["artifact"]["mimeType"] == "audio/wav"
        assert Path(result["artifact"]["internalPath"]).is_file()
