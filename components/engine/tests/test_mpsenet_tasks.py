"""Complete native MP-SENet graphs, spectral processing and original discovery."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from oryn_engine import mpsenet_tasks as m


def fixture(root, precision="float32"):
    root.mkdir(exist_ok=True)
    raw = json.loads((m.DATA / "config.json").read_text())
    raw["h"]["dense_channel"] = 8
    raw["num_tsblocks"] = 2
    (root / "config.json").write_text(json.dumps(raw))
    c = m.normalized_config(raw, m.TASK)
    torch.manual_seed(6182)
    model = m.native_model(c)
    state = {k: v.to(getattr(torch, precision)).contiguous() for k, v in model.state_dict().items()}
    save_file(state, root / "model.safetensors")
    model.load_state_dict({k: v.float() for k, v in state.items()})
    return c, model


def wav(path, samples=1703, rate=16000, channels=1):
    pcm = (np.sin(np.arange(samples) * 0.07) * 5000).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())


def test_original_complete_tensor_geometry():
    c = m.normalized_config(json.loads((m.DATA / "config.json").read_text()), m.TASK)
    h = json.loads(
        (Path(__file__).parent / "fixtures/mpsenet/model.safetensors.header.json").read_text()
    )
    shapes = {k: v["shape"] for k, v in h.items() if k != "__metadata__"}
    m.validate_weight_shapes(c, shapes)
    assert len(shapes) == 247


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("precision", ["float32", "float16", "bfloat16"])
def test_complete_stored_graph_and_waveform_parity(tmp_path, backend, precision):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    c, reference = fixture(tmp_path, precision)
    model, _, _ = m.load_components(tmp_path, c, backend)
    reference.to(backend)
    path = tmp_path / "input.wav"
    wav(path)
    audio = m.read_audio(path)
    with torch.inference_mode():
        actual = m.enhance(model, audio, backend)
        expected = m.enhance(reference, audio, backend)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        result = m.execute(
            model,
            {"operation": "enhance", "audioArtifactId": "artifact_" + "a" * 24},
            backend,
            path,
            tmp_path / "result.wav",
        )
    assert (
        result["frames"] == 1700 and actual.shape == (1700,) and bool(torch.isfinite(actual).all())
    )
    m.validate_output(tmp_path / "result.wav", result, path, backend)


@pytest.mark.parametrize(
    "key,value",
    [
        ("dense_channel", 10),
        ("n_fft", 512),
        ("sampling_rate", 48000),
        ("segment_size", 16000),
        ("compress_factor", 0.5),
        ("beta", True),
        ("num_tsconformers", 8),
    ],
)
def test_unreviewed_geometry_is_rejected(key, value):
    c = m.normalized_config(json.loads((m.DATA / "config.json").read_text()), m.TASK)
    c["h"][key] = value
    with pytest.raises(ValueError, match="MPSENET"):
        m.validate_config(c)


def test_normalization_does_not_override_declared_architecture():
    raw = json.loads((m.DATA / "config.json").read_text())
    c = m.normalized_config(raw, m.TASK)
    assert set(c) - set(raw) == {"model_type", "architectures"}
    with pytest.raises(ValueError, match="MPSENET"):
        m.normalized_config({**c, "architectures": ["OtherGraph"]}, m.TASK)


def test_original_metadata_binds_runtime_without_text_or_processor_files(tmp_path):
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

    c = m.normalized_config(json.loads((m.DATA / "config.json").read_text()), m.TASK)
    files = {*m.FILES, "generator.pt", "model.py"}
    if missing:
        files.remove(missing)
    result = assess_layout(sorted(files), c, m.TASK)
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
    ["encoder", "attention", "gru", "magnitude", "phase", "extra", "shape", "integer", "nonfinite"],
)
def test_every_native_array_required_and_validated(tmp_path, change):
    from safetensors.torch import load_file

    c, _ = fixture(tmp_path)
    p = tmp_path / "model.safetensors"
    state = load_file(p)
    key = "mask_decoder.lsigmoid.slope"
    prefixes = {
        "encoder": "dense_encoder.",
        "attention": "TSTransformer.0.time_transformer.attention.",
        "gru": "TSTransformer.0.time_transformer.ffn.gru.",
        "magnitude": "mask_decoder.",
        "phase": "phase_decoder.",
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
        state[key][0, 0] = float("inf")
    save_file(state, p)
    with pytest.raises(ValueError, match="MPSENET"):
        m.load_components(tmp_path, c, "cpu")


@pytest.mark.parametrize(
    "samples,rate,channels",
    [(399, 16000, 1), (1600, 8000, 1), (1600, 16000, 2), (480001, 16000, 1)],
)
def test_native_audio_bounds(tmp_path, samples, rate, channels):
    p = tmp_path / "input.wav"
    wav(p, samples, rate, channels)
    with pytest.raises(ValueError, match="MPSENET"):
        m.read_audio(p)


@pytest.mark.parametrize("name", m.FILES)
def test_symlink_checkpoint_rejected(tmp_path, name):
    c, _ = fixture(tmp_path)
    p = tmp_path / name
    other = tmp_path / "outside"
    p.rename(other)
    p.symlink_to(other)
    with pytest.raises(ValueError, match="MPSENET"):
        m.load_components(tmp_path, c, "cpu")


@pytest.mark.parametrize("samples", [400, 479, 32000, 32017, 48000, 48017, 64000])
def test_original_chunk_boundaries_and_native_hop_clock(tmp_path, samples):
    _, model = fixture(tmp_path)
    p = tmp_path / "input.wav"
    wav(p, samples)
    audio = m.read_audio(p)
    with torch.inference_mode():
        result = m.enhance(model, audio, "cpu")
    assert result.shape == (samples // 100 * 100,) and bool(torch.isfinite(result).all())


@pytest.mark.parametrize("failure", ["silence", "mask", "phase"])
def test_nonfinite_values_are_rejected_before_bounded_heads(tmp_path, failure):
    _, model = fixture(tmp_path)
    p = tmp_path / "input.wav"
    wav(p)
    audio = m.read_audio(p)
    if failure == "silence":
        audio.zero_()
    elif failure == "mask":
        model.mask_decoder.lsigmoid.register_forward_pre_hook(
            lambda module, args: (torch.full_like(args[0], float("inf")),)
        )
    else:
        model.phase_decoder.phase_conv_r.register_forward_pre_hook(
            lambda module, args: (torch.full_like(args[0], float("inf")),)
        )
    with torch.inference_mode(), pytest.raises(ValueError, match="NONFINITE"):
        m.enhance(model, audio, "cpu")


def test_profile_and_host_validation_preserve_native_frame_count(tmp_path):
    from oryn_protocol.task_contracts import effective_input, validate_invocation

    from oryn_engine.runners import JobExecutionError, _validate_mpsenet_audio
    from oryn_engine.task_profiles import profile_for_execution

    c, model = fixture(tmp_path)
    profile = profile_for_execution(
        {
            "runtimePack": {
                "packId": "oryn.transformers-pytorch",
                "taskContract": "audio-transformation-v1",
            }
        },
        c,
    )
    value = effective_input(profile, {"audioArtifactId": "artifact_" + "a" * 24})
    assert value["sampleRate"] == 16000 and value["operation"] == "enhance"
    validate_invocation(profile, value)
    for change in [{"operation": "encode"}, {"sampleRate": 48000}]:
        with pytest.raises(ValueError):
            validate_invocation(profile, {**value, **change})
    source = tmp_path / "input.wav"
    wav(source)
    output = tmp_path / "result.wav"
    with torch.inference_mode():
        result = m.execute(model, value, "cpu", source, output)
    _validate_mpsenet_audio(output, result, source, "cpu")
    for change in [
        {"frames": 1703},
        {"channels": True},
        {"sampleRate": 48000},
        {"durationSeconds": 1},
        {"backend": "mps"},
        {"extra": True},
    ]:
        with pytest.raises(JobExecutionError):
            _validate_mpsenet_audio(output, {**result, **change}, source, "cpu")


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_offline_shared_worker_enhances_original_native_graph(tmp_path, backend):
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
            input={"operation": "enhance", "audioArtifactId": "artifact_" + "a" * 24},
            inputPath=str(source),
            outputPath=str(output),
        ),
    )
    run = subprocess.run(
        [sys.executable, "-m", "oryn_engine.runtime_packs.transformers_worker"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout.splitlines()[-1])
    assert result["ok"], (result, run.stderr)
    m.validate_output(output, result["result"], source, backend)


def test_truncated_and_symlink_audio_rejected(tmp_path):
    p = tmp_path / "input.wav"
    wav(p)
    p.write_bytes(p.read_bytes()[:-2])
    with pytest.raises(ValueError, match="TRUNCATED"):
        m.read_audio(p)
    other = tmp_path / "audio.wav"
    p.rename(other)
    p.symlink_to(other)
    with pytest.raises(ValueError, match="SCOPE"):
        m.read_audio(p)


def test_pcm16_conversion_matches_publisher_soundfile_writer(tmp_path):
    import soundfile as sf

    _, model = fixture(tmp_path)
    source = tmp_path / "input.wav"
    wav(source)
    with torch.inference_mode():
        samples = m.enhance(model, m.read_audio(source), "cpu")
        m.execute(
            model,
            {"operation": "enhance", "audioArtifactId": "artifact_" + "a" * 24},
            "cpu",
            source,
            tmp_path / "result.wav",
        )
    sf.write(tmp_path / "reference.wav", samples.numpy(), 16000, subtype="PCM_16")
    with (
        wave.open(str(tmp_path / "result.wav"), "rb") as a,
        wave.open(str(tmp_path / "reference.wav"), "rb") as z,
    ):
        assert a.readframes(a.getnframes()) == z.readframes(z.getnframes())


@pytest.mark.parametrize(
    "change", [{"operation": "encode"}, {"codes": [[1] * 16]}, {"sampleRate": 24000}]
)
def test_only_native_enhancement_inputs_are_admitted(tmp_path, change):
    _, model = fixture(tmp_path)
    source = tmp_path / "input.wav"
    wav(source)
    with torch.inference_mode(), pytest.raises(ValueError, match="MPSENET_INPUT"):
        m.execute(
            model,
            {"operation": "enhance", "audioArtifactId": "artifact_" + "a" * 24, **change},
            "cpu",
            source,
            tmp_path / "result.wav",
        )
