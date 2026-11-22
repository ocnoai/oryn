"""Complete three-stem TIGER graphs and authorized multi-WAV result staging."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from oryn_engine import tiger_tasks as m


def config(tiny=True):
    c = json.loads((m.DATA / "config.json").read_text())
    if tiny:
        c.update(out_channels=12, in_channels=16, num_blocks=2, upsampling_depth=2, att_hid_chan=2)
    return m.normalized_config(c, m.TASK)


def fixture(root, precision="float32"):
    root.mkdir(exist_ok=True)
    c = config()
    raw = {k: v for k, v in c.items() if k not in {"model_type", "architectures"}}
    (root / "config.json").write_text(json.dumps(raw))
    torch.manual_seed(6187)
    model = m.native_model(c)
    state = {k: v.to(getattr(torch, precision)).contiguous() for k, v in model.state_dict().items()}
    save_file(state, root / "model.safetensors")
    model.load_state_dict({k: v.float() for k, v in state.items()})
    return c, model


def wav(path, samples=2049, rate=44100, channels=1):
    pcm = (np.sin(np.arange(samples) * 0.07) * 5000).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())


def stem_result(directory, source):
    count, _ = m.audio_geometry(source)
    stems = []
    for name in m.STEMS:
        wav(directory / (name + ".wav"), count)
        stems.append(
            dict(
                relativeOutput=name + ".wav",
                mediaType="audio/wav",
                sampleRate=44100,
                channels=1,
                frames=count,
                durationSeconds=count / 44100,
            )
        )
    return dict(stems=stems, backend="cpu")


def test_original_all_three_complete_tensor_inventories():
    h = json.loads(
        (Path(__file__).parent / "fixtures/tiger/model.safetensors.header.json").read_text()
    )
    shapes = {k: v["shape"] for k, v in h.items() if k != "__metadata__"}
    m.validate_weight_shapes(config(False), shapes)
    assert len(shapes) == 2304 and all(
        sum(k.startswith(name + ".") for k in shapes) == 768 for name in m.STEMS
    )


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("precision", ["float32", "float16", "bfloat16"])
def test_complete_stored_branches_match_native_waveform(tmp_path, backend, precision):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    c, reference = fixture(tmp_path, precision)
    actual, _, _ = m.load_components(tmp_path, c, backend)
    audio = torch.sin(torch.arange(2049) * 0.07)[None, None] * 0.15
    with torch.inference_mode():
        for name in m.STEMS:
            a = getattr(actual, name)(audio)
            z = getattr(reference, name)(audio)
            torch.testing.assert_close(a, z, atol=1e-4, rtol=1e-4)
            assert a.shape == (1, 3, 2049) and bool(torch.isfinite(a).all())
            torch.testing.assert_close(a.sum(1), audio[:, 0], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "key,value",
    [
        ("sample_rate", 48000),
        ("win", 1024),
        ("num_sources", 2),
        ("out_channels", 16),
        ("in_channels", 0),
        ("upsampling_depth", 1),
        ("num_blocks", 9),
        ("att_hid_chan", True),
        ("att_kernel_size", 4),
        ("att_stride", 2),
    ],
)
def test_closed_original_configuration(key, value):
    c = config()
    c[key] = value
    with pytest.raises(ValueError, match="TIGER"):
        m.validate_config(c)


def test_normalization_cannot_replace_explicit_architecture():
    c = config()
    c["architectures"] = ["OtherGraph"]
    with pytest.raises(ValueError, match="TIGER"):
        m.normalized_config(c, m.TASK)


def test_original_metadata_discovery_without_processor_or_tokenizer(tmp_path):
    from test_metadata_audit import row, snapshot

    from oryn_engine.metadata_audit import inspect_row

    raw = json.loads((m.DATA / "config.json").read_text())
    hub = snapshot(
        tmp_path, config=raw, files=[(name, 100, "b" * 64) for name in m.FILES], pipeline_tag=m.TASK
    )
    result = inspect_row(row(task=m.TASK), hub)
    assert result["status"] == "metadata-compatible", result
    assert {f["name"] for f in result["selectedFiles"]} == set(m.FILES) and not result[
        "executionVerified"
    ]
    hub.client.download.assert_not_called()


@pytest.mark.parametrize("missing", [None, *m.FILES])
def test_original_complete_subset(missing):
    from oryn_engine.model_layouts import assess_layout

    files = {*m.FILES, "model.py", "generator.pt"}
    if missing:
        files.remove(missing)
    result = assess_layout(sorted(files), config(), m.TASK)
    if missing:
        assert result.artifact_reasons
    else:
        assert (
            set(result.selected_files) == set(m.FILES)
            and not result.runtime_reasons
            and not result.artifact_reasons
        )


@pytest.mark.parametrize(
    "change",
    [
        "dialog",
        "effect",
        "music",
        "bottleneck",
        "attention",
        "mask",
        "extra",
        "shape",
        "integer",
        "nonfinite",
    ],
)
def test_every_branch_head_and_array_is_required(tmp_path, change):
    c, _ = fixture(tmp_path)
    p = tmp_path / "model.safetensors"
    state = load_file(p)
    key = next(iter(state))
    prefixes = {
        **{k: k + "." for k in m.STEMS},
        "bottleneck": "dialog.BN.",
        "attention": "effect.separator.freq_path.1.",
        "mask": "music.mask.",
    }
    if change in prefixes:
        del state[next(k for k in state if k.startswith(prefixes[change]))]
    elif change == "extra":
        state["extra.weight"] = torch.ones(1)
    elif change == "shape":
        state[key] = state[key][:-1]
    elif change == "integer":
        state[key] = state[key].to(torch.int32)
    else:
        state[key].view(-1)[0] = float("inf")
    save_file(state, p)
    with pytest.raises(ValueError, match="TIGER"):
        m.load_components(tmp_path, c, "cpu")


@pytest.mark.parametrize(
    "samples,rate,channels",
    [(2047, 44100, 1), (2049, 16000, 1), (4098, 44100, 2), (529201, 44100, 1)],
)
def test_native_audio_bounds(tmp_path, samples, rate, channels):
    p = tmp_path / "input.wav"
    wav(p, samples, rate, channels)
    with pytest.raises(ValueError, match="TIGER"):
        m.read_audio(p)


@pytest.mark.parametrize("name", m.FILES)
def test_checkpoint_symlinks_rejected(tmp_path, name):
    c, _ = fixture(tmp_path)
    p = tmp_path / name
    other = tmp_path / "outside"
    p.rename(other)
    p.symlink_to(other)
    with pytest.raises(ValueError, match="TIGER"):
        m.load_components(tmp_path, c, "cpu")


@pytest.mark.parametrize("samples", [2048, 176399, 176400, 529200])
def test_native_overlap_add_preserves_edges_and_hop_clock(samples):
    model = m.native_model(config())
    audio = torch.sin(torch.arange(samples) * 0.01)[None, None]
    # Independent identity separator isolates the publisher's chunk/pad/overlap
    # arithmetic, including the first and final samples at exact hop boundaries.
    with torch.inference_mode():
        actual = model.wav_chunk_inference(lambda x: x[:, 0, None].expand(-1, 3, -1), audio)
    torch.testing.assert_close(actual, audio[0].expand(3, -1).unsqueeze(1), atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("failure", ["mask", "gate"])
def test_nonfinite_values_rejected_before_sigmoid(failure):
    model = m.native_model(config())
    layer = (
        model.dialog.mask[0][1]
        if failure == "mask"
        else model.dialog.separator.freq_path[0].loc_glo_fus[0].act
    )
    layer.register_forward_pre_hook(
        lambda module, args: (torch.full_like(args[0], float("inf")),), prepend=True
    )
    with torch.inference_mode(), pytest.raises(ValueError, match="NONFINITE"):
        model.dialog(torch.rand(1, 1, 2049))


def test_profile_exposes_bounded_separation():
    from oryn_protocol.task_contracts import effective_input, validate_invocation

    from oryn_engine.task_profiles import profile_for_execution

    profile = profile_for_execution(
        {
            "runtimePack": {
                "packId": "oryn.transformers-pytorch",
                "taskContract": "audio-transformation-v1",
            }
        },
        config(),
    )
    value = effective_input(profile, {"audioArtifactId": "artifact_" + "a" * 24})
    assert (
        value["sampleRate"] == 44100
        and value["operation"] == "separate"
        and profile["limits"]["audioDurationSeconds"] == 12
    )
    validate_invocation(profile, value)
    for change in [{"operation": "enhance"}, {"sampleRate": 16000}]:
        with pytest.raises(ValueError):
            validate_invocation(profile, {**value, **change})


@pytest.mark.parametrize(
    "change",
    [
        "count",
        "path",
        "duplicate",
        "frames",
        "channels",
        "rate",
        "duration",
        "backend",
        "extra",
        "truncated",
        "symlink",
    ],
)
def test_host_validates_all_stems_before_any_staging(tmp_path, change):
    from oryn_engine.runners import JobExecutionError, _validate_tiger_audio

    source = tmp_path / "input.wav"
    wav(source)
    result = stem_result(tmp_path, source)
    assert [p.name for p in _validate_tiger_audio(tmp_path, result, source, "cpu")] == [
        "dialog.wav",
        "effect.wav",
        "music.wav",
    ]
    stem = result["stems"][1]
    if change == "count":
        result["stems"].pop()
    elif change == "path":
        stem["relativeOutput"] = "../outside.wav"
    elif change == "duplicate":
        stem["relativeOutput"] = "dialog.wav"
    elif change == "frames":
        stem["frames"] = 2048
    elif change == "channels":
        stem["channels"] = True
    elif change == "rate":
        stem["sampleRate"] = 16000
    elif change == "duration":
        stem["durationSeconds"] = 1
    elif change == "backend":
        result["backend"] = "mps"
    elif change == "extra":
        stem["extra"] = True
    elif change == "truncated":
        p = tmp_path / "music.wav"
        p.write_bytes(p.read_bytes()[:-2])
    else:
        p = tmp_path / "music.wav"
        p.unlink()
        p.symlink_to(source)
    with pytest.raises(JobExecutionError):
        _validate_tiger_audio(tmp_path, result, source, "cpu")


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_offline_worker_produces_three_complete_native_stems(tmp_path, backend):
    import os
    import subprocess
    import sys

    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    fixture(tmp_path)
    source = tmp_path / "input.wav"
    wav(source)
    output = tmp_path / "result.wav"
    request = dict(
        abi="1",
        packId="oryn.transformers-pytorch",
        version=builtin_registry().active_pack("oryn.transformers-pytorch").version,
        command="generate",
        payload=dict(
            modelRoot=str(tmp_path),
            task=m.TASK,
            backend=backend,
            input={
                "operation": "separate",
                "sampleRate": 44100,
                "audioArtifactId": "artifact_" + "a" * 24,
            },
            inputPath=str(source),
            outputPath=str(output),
        ),
    )
    run = subprocess.run(
        [sys.executable, "-m", "oryn_engine.runtime_packs.transformers_worker"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        timeout=90,
        env=os.environ
        | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1"},
    )
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout.splitlines()[-1])
    assert result["ok"], (result, run.stderr)
    assert (
        len(m.validate_outputs(tmp_path, result["result"], source, backend)) == 3
        and not output.exists()
    )


def test_host_returns_three_named_artifacts(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    from oryn_engine.artifacts import ArtifactOutputStaging
    from oryn_engine.runners import RunnerRegistry

    source = tmp_path / "input.wav"
    wav(source)
    root = tmp_path / "models--fixture--tiger/snapshots" / ("a" * 40)
    root.mkdir(parents=True)
    registry = RunnerRegistry(ArtifactOutputStaging(tmp_path / "outputs", "engine_tiger"))

    def invoke(record, pack, runtime, command, payload, **options):
        assert command == "generate" and Path(payload["inputPath"]) == source
        assert (
            payload["input"]["operation"] == "separate"
            and Path(payload["outputPath"]).name == "result.wav"
        )
        return stem_result(Path(payload["outputPath"]).parent, source)

    monkeypatch.setattr(registry, "_invoke_pack", invoke)
    identifier = "artifact_" + "a" * 24
    job = SimpleNamespace(
        job_id="job_tiger",
        input={"operation": "separate", "sampleRate": 44100, "audioArtifactId": identifier},
        input_artifacts=[
            dict(
                artifactId=identifier,
                internalPath=str(source),
                mimeType="audio/wav",
                sizeBytes=source.stat().st_size,
            )
        ],
    )
    handle = SimpleNamespace(
        record={
            "cacheRoot": str(tmp_path),
            "model": {"repository": "fixture/tiger", "revision": "a" * 40},
        },
        task=m.TASK,
        backend="cpu",
        runtime=None,
        pack=SimpleNamespace(pack_id="oryn.transformers-pytorch"),
    )
    result = registry._run_shared_task(job, handle, threading.Event(), lambda *args: None)
    assert result["kind"] == "multi" and [x["artifact"]["name"] for x in result["outputs"]] == [
        "dialog.wav",
        "effect.wav",
        "music.wav",
    ]
    assert all(
        Path(x["artifact"]["internalPath"]).is_file() and x["artifact"]["mimeType"] == "audio/wav"
        for x in result["outputs"]
    )


def test_truncated_and_symlink_input_rejected(tmp_path):
    p = tmp_path / "input.wav"
    wav(p)
    p.write_bytes(p.read_bytes()[:-2])
    with pytest.raises(ValueError, match="TRUNCATED"):
        m.read_audio(p)
    other = tmp_path / "outside.wav"
    p.rename(other)
    p.symlink_to(other)
    with pytest.raises(ValueError, match="SCOPE"):
        m.read_audio(p)


@pytest.mark.parametrize(
    "change", [{"operation": "enhance"}, {"codes": [[1]]}, {"sampleRate": 24000}]
)
def test_only_native_separation_inputs_admitted(tmp_path, change):
    _, model = fixture(tmp_path)
    source = tmp_path / "input.wav"
    wav(source)
    with torch.inference_mode(), pytest.raises(ValueError, match="TIGER_INPUT"):
        m.execute(
            model,
            {"operation": "separate", "audioArtifactId": "artifact_" + "a" * 24, **change},
            "cpu",
            source,
            tmp_path / "result.wav",
        )
