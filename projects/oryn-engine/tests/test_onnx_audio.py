import copy
import wave

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from oryn_engine.onnx_audio import analysis, enhance, read_audio, synthesis, validate_graph
from oryn_engine.onnx_tasks import execute, input_value, validate_config
from oryn_engine.runners import _validate_enhanced_audio


def write_wave(path, samples, rate=48000):
    with wave.open(str(path), "wb") as target:
        target.setnchannels(samples.shape[1])
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(np.rint(samples * 32767).astype("<i2").tobytes())


def geometry():
    return {
        "inputs": [
            {"name": "spec", "dtype": 1, "shape": [1, 1, 200, 481, 2]},
            {"name": "feat_erb", "dtype": 1, "shape": [1, 1, 200, 32]},
            {"name": "feat_spec", "dtype": 1, "shape": [1, 1, 200, 96, 2]},
        ],
        "outputs": [{"name": "spec_enhanced", "dtype": 1, "shape": [1, 1, 200, 481, 2]}],
    }


def test_audio_binding_rejects_different_geometry_and_unimplemented_operations():
    validate_graph(geometry())
    for kind in ("inputs", "outputs"):
        for member in range(len(geometry()[kind])):
            bad = copy.deepcopy(geometry())
            bad[kind][member]["shape"][2] = "frames"
            with pytest.raises(ValueError, match="BINDING_UNSUPPORTED"):
                validate_graph(bad)
            bad = copy.deepcopy(geometry())
            bad[kind][member]["dtype"] = 10
            with pytest.raises(ValueError, match="BINDING_UNSUPPORTED"):
                validate_graph(bad)
    good = {"operation": "enhance", "audioArtifactId": "artifact_abcdefghijklmnopqrstuvwxyz"}
    assert input_value("audio-to-audio", good) == good
    for change in ({"operation": "separate"}, {"sampleRate": 16000}, {"prompt": "ignore"}):
        with pytest.raises(ValueError):
            input_value("audio-to-audio", good | change)
    for config in ({"auto_map": {"AutoModel": "repo.Model"}}, {"sample_rate": 16000}):
        with pytest.raises(ValueError):
            validate_config(config, "audio-to-audio")


@pytest.mark.parametrize("length", [1, 479, 480, 481, 96001, 192123])
def test_audio_overlap_add_preserves_samples_and_channel_state(length):
    signal = np.random.default_rng(61).normal(0, 0.1, length).astype(np.float32)
    spec, erb, features = analysis(signal)
    np.testing.assert_allclose(synthesis(spec)[:length], signal, atol=2e-7)
    # A second request must not reuse the first request's normalization history.
    silence = analysis(np.zeros(length, dtype=np.float32))
    repeated = analysis(signal)
    for first, second in zip((spec, erb, features), repeated, strict=True):
        np.testing.assert_array_equal(first, second)
    assert not np.any(silence[0]) and not np.any(silence[2])


@pytest.mark.parametrize("rate,channels", [(8000, 1), (16000, 2), (44100, 2), (48000, 1)])
def test_audio_resampling_preserves_duration_and_channels(tmp_path, rate, channels):
    signal = np.zeros((rate + 13, channels), dtype=np.float32)
    signal[:, 0] = 0.1 * np.sin(np.arange(len(signal)) * 2 * np.pi * 440 / rate)
    path = tmp_path / "input.wav"
    write_wave(path, signal, rate)
    actual = read_audio(path)
    assert actual.shape == ((len(signal) * 48000 + rate - 1) // rate, channels)
    if channels == 2:
        assert not np.any(actual[:, 1])
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="TRUNCATED"):
        read_audio(path)


def test_audio_rejects_symlinks_unsupported_pcm_and_long_inputs(tmp_path):
    path = tmp_path / "input.wav"
    for channels, rate, frames in ((3, 48000, 12), (1, 96000, 12), (1, 8000, 240001)):
        write_wave(path, np.zeros((frames, channels), dtype=np.float32), rate)
        with pytest.raises(ValueError, match="FORMAT_UNSUPPORTED"):
            read_audio(path)
    link = tmp_path / "link.wav"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="AUDIO_LIMIT"):
        read_audio(link)


def test_audio_worker_runs_native_graph_and_output_validation_rejects_mismatch(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    description = geometry()
    graph = helper.make_graph(
        [helper.make_node("Identity", ["spec"], ["spec_enhanced"])],
        "audio-roundtrip",
        [
            helper.make_tensor_value_info(x["name"], TensorProto.FLOAT, x["shape"])
            for x in description["inputs"]
        ],
        [helper.make_tensor_value_info("spec_enhanced", TensorProto.FLOAT, [1, 1, 200, 481, 2])],
    )
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9),
        root / "model.onnx",
    )
    signal = np.zeros((96123, 2), dtype=np.float32)
    signal[:, 0] = 0.2 * np.sin(np.arange(len(signal)) * 2 * np.pi * 440 / 48000)
    source, target = tmp_path / "input.wav", tmp_path / "result.wav"
    write_wave(source, signal)
    value = {"operation": "enhance", "audioArtifactId": "artifact_abcdefghijklmnopqrstuvwxyz"}
    result = execute(root, "audio-to-audio", value, "cpu", lambda *_: None, source, target)
    _validate_enhanced_audio(target, result, source)
    np.testing.assert_allclose(read_audio(target), read_audio(source), atol=1 / 32768)
    from oryn_engine.jobs import JobExecutionError

    for bad in (
        result | {"channels": 1},
        result | {"frames": len(signal) + 1},
        result | {"sampleRate": True},
        result | {"relativeOutput": "../result.wav"},
    ):
        with pytest.raises(JobExecutionError, match="Enhanced audio"):
            _validate_enhanced_audio(target, bad, source)
    target.write_bytes(target.read_bytes()[:-2])
    with pytest.raises(JobExecutionError, match="Enhanced audio"):
        _validate_enhanced_audio(target, result, source)


def test_audio_enhancement_rejects_nonfinite_worker_output():
    class BadSession:
        def run(self, *_):
            return [np.full((1, 1, 200, 481, 2), np.nan, dtype=np.float32)]

    with pytest.raises(ValueError, match="OUTPUT_INVALID"):
        enhance(BadSession(), np.zeros(480, dtype=np.float32), lambda *_: None)


def test_audio_task_profile_exposes_only_the_implemented_operation():
    from oryn_protocol.task_contracts import effective_input, validate_invocation

    from oryn_engine.task_profiles import profile_for_execution

    profile = profile_for_execution(
        {"runtimePack": {"packId": "oryn.onnx-runtime", "taskContract": "audio-transformation-v1"}}
    )
    value = effective_input(profile, {"audioArtifactId": "artifact_abcdefghijklmnopqrstuvwxyz"})
    assert value["operation"] == "enhance" and value["sampleRate"] == 48000
    validate_invocation(profile, value)
    for change in ({"operation": "voice-conversion"}, {"sampleRate": 16000}):
        with pytest.raises(ValueError):
            validate_invocation(profile, value | change)
