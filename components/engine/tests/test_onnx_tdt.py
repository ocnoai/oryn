"""Complete native transducer composition and bounded original ONNX metadata."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from oryn_engine import onnx_tdt as adapter
from oryn_engine.onnx_graph import scan_graph

FIXTURE = Path(__file__).parent / "fixtures/onnx_tdt"


def config():
    return json.loads((FIXTURE / "config.json").read_text())


def frontend_graph(path, include_terminal=True):
    import onnx
    from onnx import TensorProto as T
    from onnx import helper as h
    from onnx import numpy_helper as nh
    from transformers.audio_utils import mel_filter_bank

    nodes, arrays = [], []

    def constant(name, value, dtype=np.int64):
        arrays.append(nh.from_array(np.array(value, dtype=dtype), name))
        return name

    def op(kind, inputs, output, **attributes):
        nodes.append(h.make_node(kind, inputs, [output], **attributes))
        return output

    zero = constant("zero", 0)
    one = constant("one", 1)
    hop = constant("hop", 160)
    end = constant("end", [2**31 - 1])
    axis = constant("axis", [1])
    start = constant("start", [0])
    onevec = constant("onevec", [1])
    minusone = constant("minusone", [-1])
    twoaxes = constant("twoaxes", [1, 2])
    zfloat = constant("zfloat", 0, np.float32)
    ofloat = constant("ofloat", 1, np.float32)
    pre = constant("preemphasis", 0.97, np.float32)
    constant(
        "window", np.pad(np.hanning(400), (56, 56)), np.float64 if include_terminal else np.float32
    )
    constant(
        "mel",
        mel_filter_bank(257, 128, 0, 8000, 16000, norm="slaney", mel_scale="slaney"),
        np.float32,
    )
    first = op("Slice", ["waveforms", start, onevec, axis], "first")
    rest = op("Slice", ["waveforms", onevec, end, axis], "rest")
    previous = op("Slice", ["waveforms", start, minusone, axis], "previous")
    scaled = op("Mul", [previous, pre], "scaled")
    changed = op("Sub", [rest, scaled], "changed")
    emphasized = op("Concat", [first, changed], "emphasized", axis=1)
    padded = op(
        "Pad",
        [emphasized, constant("pads", [0, 256, 0, 256])],
        "padded",
        mode="reflect" if include_terminal else "constant",
    )
    if include_terminal:
        padded = op("Cast", [padded], "double_padded", to=T.DOUBLE)
    stft = op("STFT", [padded, hop, "window"], "stft")
    power = op("ReduceSumSquare", [stft], "power", axes=[-1], keepdims=0)
    if include_terminal:
        power = op("Cast", [power], "float_power", to=T.FLOAT)
    mel = op("MatMul", [power, "mel"], "mel_values")
    guarded = op("Add", [mel, constant("guard", 2**-24, np.float32)], "guarded")
    logs = op("Log", [guarded], "logs")
    features = op("Transpose", [logs], "unscaled_features", perm=[0, 2, 1])
    base = op("Div", ["waveforms_lens", hop], "base_length")
    count = op("Add", [base, one], "count")
    lengths = op("Identity", [count if include_terminal else base], "features_lens")
    scalar_count = op("Squeeze", [count, constant("axiszero", [0])], "scalar_count")
    indices = op("Range", [zero, scalar_count, one], "indices")
    lens3 = op("Unsqueeze", [lengths, twoaxes], "lens3")
    mask = op("Less", [indices, lens3], "mask")
    lensf = op("Cast", [lens3], "lensf", to=T.FLOAT)
    kept = op("Where", [mask, features, zfloat], "kept")
    sums = op("ReduceSum", [kept, minusone], "sums", keepdims=1)
    mean = op("Div", [sums, lensf], "mean")
    centered = op("Sub", [features, mean], "centered")
    masked = op("Where", [mask, centered, zfloat], "masked")
    sumsq = op("ReduceSumSquare", [masked], "sumsq", axes=[-1], keepdims=1)
    denom = op("Sub", [lensf, ofloat], "denom")
    variance = op("Div", [sumsq, denom], "variance")
    std = op("Sqrt", [variance], "std")
    safe = op("Add", [std, constant("epsilon", 1e-5, np.float32)], "safe")
    normalized = op("Div", [centered, safe], "normalized")
    op("Where", [mask, normalized, zfloat], "features")
    model = h.make_model(
        h.make_graph(
            nodes,
            "owned-nemo128",
            [
                h.make_tensor_value_info("waveforms", T.FLOAT, [1, "samples"]),
                h.make_tensor_value_info("waveforms_lens", T.INT64, [1]),
            ],
            [
                h.make_tensor_value_info("features", T.FLOAT, [1, 128, "frames"]),
                h.make_tensor_value_info("features_lens", T.INT64, [1]),
            ],
            arrays,
        ),
        opset_imports=[h.make_opsetid("", 17)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def native_fixture(root, mode="seeded", include_terminal=True):
    from transformers import ParakeetForTDT, ParakeetTDTConfig

    c = json.loads((Path(__file__).parent / "fixtures/parakeet/config.json").read_text())
    c.update(vocab_size=32, blank_token_id=31, decoder_hidden_size=32)
    c["encoder_config"].update(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        subsampling_conv_channels=8,
        max_position_embeddings=512,
    )
    native = ParakeetTDTConfig.from_dict(c)
    native._attn_implementation = native.encoder_config._attn_implementation = "sdpa"
    torch.manual_seed(617)
    model = ParakeetForTDT(native).eval()
    if mode != "seeded":
        with torch.no_grad():
            model.joint.head.weight.zero_()
            model.joint.head.bias.fill_(-10)
            model.joint.head.bias[31 if mode.startswith("blank") else 1] = 10
            duration = {"blank": 0, "blank-two": 2, "repeat": 0, "repeat-two": 2, "overshoot": 4}[
                mode
            ]
            model.joint.head.bias[32 + duration] = 10

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = model.encoder

        def forward(self, audio_signal, length):
            mask = torch.arange(audio_signal.shape[-1])[None, :] < length[:, None]
            output = self.encoder(
                audio_signal.transpose(1, 2), attention_mask=mask, output_attention_mask=True
            )
            return output.last_hidden_state.transpose(1, 2), output.attention_mask.sum(-1)

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = model.decoder
            self.joint = model.joint
            self.projector = model.encoder_projector

        def forward(self, encoder_outputs, targets, target_length, input_states_1, input_states_2):
            y, (h, c) = self.decoder.lstm(
                self.decoder.embedding(targets.long()), (input_states_1, input_states_2)
            )
            y = self.decoder.decoder_projector(y)
            logits = self.joint(y, self.projector(encoder_outputs.transpose(1, 2)))
            return logits[:, None], target_length, h, c

    encoder, decoder = Encoder().eval(), Decoder().eval()
    for name, module, args, ins, outs in [
        (
            "encoder-model",
            encoder,
            (torch.randn(1, 128, 21), torch.tensor([20])),
            ["audio_signal", "length"],
            ["outputs", "encoded_lengths"],
        ),
        (
            "decoder_joint-model",
            decoder,
            (
                torch.randn(1, 128, 1),
                torch.tensor([[31]], dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
                torch.zeros(2, 1, 32),
                torch.zeros(2, 1, 32),
            ),
            ["encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"],
            ["outputs", "prednet_lengths", "output_states_1", "output_states_2"],
        ),
    ]:
        dyn = {x: {0: "batch"} for x in ins + outs if "states" not in x}
        if name == "encoder-model":
            dyn["audio_signal"][2] = "frames"
            dyn["outputs"][2] = "encoded_frames"
        torch.onnx.export(
            module,
            args,
            str(root / (name + ".onnx")),
            dynamo=False,
            opset_version=17,
            input_names=ins,
            output_names=outs,
            dynamic_axes=dyn,
        )
    frontend_graph(root / "nemo128.onnx", include_terminal)
    (root / "config.json").write_text(json.dumps(config()))
    words = (
        ["<unk>", "a", "<pad>", "▁hello", "▁world"] + [f"w{i}" for i in range(5, 31)] + ["<blk>"]
    )
    (root / "vocab.txt").write_text("".join(f"{w} {i}\n" for i, w in enumerate(words)))
    return model, encoder, decoder, words


def inspect(root):
    sizes = {p.name: p.stat().st_size for p in root.iterdir() if p.is_file()}
    members = []
    for name in adapter.GRAPH_NAMES:
        with (root / name).open("rb") as f:

            def read(start, length):
                f.seek(start)
                return f.read(length)

            members.append(scan_graph(read, sizes[name], name, sizes))
    return adapter.compose(members)


def sessions(root):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return tuple(
        ort.InferenceSession(
            str(root / n), sess_options=options, providers=["CPUExecutionProvider"]
        )
        for n in adapter.GRAPH_NAMES
    )


def test_original_graph_composition_and_vocabulary():
    g = json.loads((FIXTURE / "original-graphs.json").read_text())
    assert adapter.validate_graphs(g, config()) == {
        "encoder": 1024,
        "layers": 2,
        "hidden": 640,
        "vocab": 8193,
    }
    assert len(adapter.vocabulary(FIXTURE, 8193)) == 8193
    assert adapter.graph_groups(adapter.GRAPH_NAMES) == [adapter.GRAPH_NAMES]
    assert not adapter.graph_groups(adapter.GRAPH_NAMES[:-1])


@pytest.mark.parametrize(
    "mode", ["seeded", "blank", "blank-two", "repeat", "repeat-two", "overshoot"]
)
@pytest.mark.parametrize(
    "include_terminal,length", [(True, 1280), (True, 4800), (False, 1600), (False, 4800)]
)
def test_complete_native_graphs_and_independent_full_history_decode(
    tmp_path, mode, length, include_terminal
):
    model, encoder, decoder, words = native_fixture(tmp_path, mode, include_terminal)
    graph = inspect(tmp_path)
    geometry = adapter.validate_graphs(graph, config())
    native = sessions(tmp_path)
    audio = (np.sin(np.arange(length) * 0.13) * 0.2).astype(np.float32)
    features, lens = native[0].run(
        None, {"waveforms": audio[None], "waveforms_lens": np.array([length], np.int64)}
    )
    actual, valid = native[1].run(None, {"audio_signal": features, "length": lens})
    with torch.inference_mode():
        expected, expected_lens = encoder(torch.from_numpy(features), torch.from_numpy(lens))
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(valid, expected_lens)
        history = [31]
        frame = 0
        emitted = 0
        while frame < int(valid[0]):
            # Recompute the whole LSTM prefix independently of the ONNX cache loop.
            prediction = model.decoder(torch.tensor([history]))[:, -1:]
            logits = model.joint(
                prediction,
                model.encoder_projector(expected[:, :, frame : frame + 1].transpose(1, 2)),
            )[0, 0]
            token = int(logits[:32].argmax())
            duration = int(logits[32:].argmax())
            if token != 31:
                history.append(token)
                emitted += 1
            if duration > 0:
                frame += duration
                emitted = 0
            elif token == 31 or emitted == 10:
                frame += 1
                emitted = 0
        expected_text = adapter.SPACE.sub(
            lambda m: " " if m.group(1) else "",
            "".join(words[i].replace("▁", " ") for i in history[1:]),
        )
        # Compare every decoder output, including state, on a nonzero recurrent input.
        args = (
            torch.from_numpy(actual[:, :, :1]),
            torch.tensor([[1]], dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            torch.randn(2, 1, 32),
            torch.randn(2, 1, 32),
        )
        reference = decoder(*args)
        outputs = native[2].run(
            None,
            dict(
                zip(
                    [
                        "encoder_outputs",
                        "targets",
                        "target_length",
                        "input_states_1",
                        "input_states_2",
                    ],
                    [a.numpy() for a in args],
                    strict=True,
                )
            ),
        )
        for value, ref in zip(outputs, reference, strict=True):
            np.testing.assert_allclose(value, ref, atol=2e-5, rtol=2e-5)
    result = adapter.transcribe(native, geometry, words, audio, lambda *_: None)
    assert result == {"text": expected_text, "language": None}
    if mode.startswith("blank"):
        assert result["text"] == ""
    if mode == "repeat":
        assert result["text"] == "a" * (10 * int(valid[0]))


@pytest.mark.parametrize(
    "damage",
    ["missing", "dtype", "width", "state", "vocab", "extra", "static-time", "duplicate", "config"],
)
def test_unbound_graphs_rejected(damage):
    graph = json.loads((FIXTURE / "original-graphs.json").read_text())
    c = config()
    if damage == "missing":
        del graph["frontend"]
    elif damage == "dtype":
        graph["decoder"]["inputs"][1]["dtype"] = 7
    elif damage == "width":
        graph["encoder"]["outputs"][0]["shape"][1] = 1023
    elif damage == "state":
        graph["decoder"]["outputs"][2]["shape"][2] = 639
    elif damage == "vocab":
        graph["decoder"]["outputs"][0]["shape"][-1] = "unknown"
    elif damage == "extra":
        graph["encoder"]["inputs"].append({"name": "custom", "dtype": 1, "shape": [1]})
    elif damage == "static-time":
        graph["frontend"]["inputs"][0]["shape"][1] = 480000
    elif damage == "duplicate":
        graph["encoder"]["inputs"].append(graph["encoder"]["inputs"][0])
    else:
        c["subsampling_factor"] = 4
    with pytest.raises(ValueError):
        adapter.validate_graphs(graph, c)


@pytest.mark.parametrize("damage", ["duplicate", "missing", "blank", "id", "symlink", "unicode"])
def test_original_vocabulary_identity(tmp_path, damage):
    text = (FIXTURE / "vocab.txt").read_text()
    lines = text.splitlines()
    if damage == "duplicate":
        lines[2] = "<unk> 2"
    elif damage == "missing":
        lines.pop()
    elif damage == "blank":
        lines[-1] = "other 8192"
    elif damage == "id":
        lines[1] = "<|nospeech|> 0"
    elif damage == "unicode":
        lines[10] = "▁čučoriedka 10"
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(lines) + "\n")
    if damage == "symlink":
        path.rename(tmp_path / "real.txt")
        path.symlink_to(tmp_path / "real.txt")
    if damage == "unicode":
        assert adapter.vocabulary(tmp_path, 8193)[10] == "▁čučoriedka"
    else:
        with pytest.raises(ValueError):
            adapter.vocabulary(tmp_path, 8193)


def numpy_features(audio, include_terminal=True):
    from transformers.audio_utils import mel_filter_bank

    emphasized = np.concatenate([audio[:1], audio[1:] - np.float32(0.97) * audio[:-1]])
    padded = np.pad(emphasized, (256, 256), mode="reflect" if include_terminal else "constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 512)[::160]
    window = np.pad(np.hanning(400), (56, 56)).astype(
        np.float64 if include_terminal else np.float32
    )
    power = (np.abs(np.fft.rfft(windows * window)) ** 2).astype(np.float32)
    mel = power @ mel_filter_bank(
        257, 128, 0, 8000, 16000, norm="slaney", mel_scale="slaney"
    ).astype(np.float32)
    logs = np.log(mel + np.float32(2**-24)).T
    frames = len(audio) // 160 + int(include_terminal)
    valid = logs[:, :frames]
    normalized = (valid - valid.mean(-1, keepdims=True)) / (
        valid.std(-1, ddof=1, keepdims=True) + 1e-5
    )
    return np.pad(normalized, ((0, 0), (0, int(not include_terminal))))[None].astype(np.float32)


@pytest.mark.parametrize("length", [320, 1599, 1600, 1601, 480000])
@pytest.mark.parametrize("include_terminal", [True, False])
def test_complete_frontend_matches_independent_numpy(tmp_path, length, include_terminal):
    import onnxruntime as ort

    frontend_graph(tmp_path / "nemo128.onnx", include_terminal)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.log_severity_level = 3
    native = ort.InferenceSession(
        str(tmp_path / "nemo128.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    audio = np.random.default_rng(617).normal(0, 0.15, length).astype(np.float32)
    features, lens = native.run(
        None, {"waveforms": audio[None], "waveforms_lens": np.array([length], np.int64)}
    )
    np.testing.assert_allclose(
        features, numpy_features(audio, include_terminal), atol=4e-4, rtol=4e-4
    )
    assert lens.tolist() == [length // 160 + int(include_terminal)]
    if not include_terminal:
        assert np.all(features[:, :, -1] == 0)


@pytest.mark.parametrize(
    "damage", [None, "missing", "state", "operator", "external", "config", "vocab"]
)
def test_discovery_composes_exact_original_members(tmp_path, damage):
    from types import SimpleNamespace

    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore

    c = config()
    if damage == "config":
        c["subsampling_factor"] = 4
    names = [*adapter.GRAPH_NAMES, "encoder-model.onnx.data", "config.json", "vocab.txt"]
    if damage == "missing":
        names.remove("nemo128.onnx")
    if damage == "external":
        names.remove("encoder-model.onnx.data")
    if damage == "vocab":
        names.remove("vocab.txt")

    class Hub(FakeHub):
        def read_onnx_metadata(self, repo, rev, name, sizes):
            g = json.loads((FIXTURE / "original-graphs.json").read_text())[
                dict(zip(adapter.GRAPH_NAMES, adapter.ROLES, strict=True))[name]
            ]
            if damage == "state" and name == "decoder_joint-model.onnx":
                g["outputs"][2]["shape"][2] = 639
            if damage == "operator":
                g["operators"].append("UnknownCustom")
            return g

    hub = Hub(
        config=c,
        pipeline_tag="automatic-speech-recognition",
        files=[(n, 100, "a" * 64) for n in names],
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda p: object() if p.pack_id == "oryn.onnx-runtime" else None
    )
    result = store.inspect(model("renamed/transducer-export"))
    assert result["executable"] is (damage is None), json.dumps(result, indent=2)
    if damage is None:
        assert result["executableBackends"] == ["cpu"]
        assert {f["name"] for f in result["selectedFiles"]} == set(names)
        assert result["runtimePack"]["packId"] == "oryn.onnx-runtime"
        assert result["taskProfile"]["features"]["languageDetection"] == "unsupported"
    assert not hub.downloaded


@pytest.mark.parametrize(
    "damage",
    [
        "frontend-length",
        "frontend-nan",
        "encoder-length",
        "encoder-nan",
        "decoder-nan",
        "state-nan",
        "state-shape",
        "prednet-length",
        "short",
        "long",
        "nan",
    ],
)
def test_runtime_validates_every_composed_output(tmp_path, damage):
    _, _, _, words = native_fixture(tmp_path, "repeat-two")
    native = list(sessions(tmp_path))
    geometry = adapter.validate_graphs(inspect(tmp_path), config())
    index = 0 if damage.startswith("frontend") else 1 if damage.startswith("encoder") else 2

    class Changed:
        def __init__(self, real):
            self.real = real

        def run(self, *args):
            values = self.real.run(*args)
            if damage in {"frontend-length", "encoder-length", "prednet-length"}:
                values[1][0] += 1
            elif damage in {"frontend-nan", "encoder-nan", "decoder-nan"}:
                values[0].flat[0] = np.nan
            elif damage == "state-nan":
                values[2].flat[0] = np.nan
            elif damage == "state-shape":
                values[2] = values[2][:, :, :-1]
            return values

    native[index] = Changed(native[index])
    audio = np.zeros(319 if damage == "short" else 480001 if damage == "long" else 1600, np.float32)
    if damage == "nan":
        audio[0] = np.nan
    with pytest.raises(ValueError):
        adapter.transcribe(native, geometry, words, audio, lambda *_: None)


def test_offline_worker_prepares_all_three_graphs_and_repeats(tmp_path):
    import os
    import subprocess
    import sys
    import wave

    from oryn_engine.onnx_tasks import execute, prepare
    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    native_fixture(tmp_path, "repeat-two")
    path = tmp_path / "input.wav"
    with wave.open(str(path), "wb") as f:
        f.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        f.writeframes(bytes(3200))
    # A checkpoint generation file is not a control channel for the owned loop.
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"max_new_tokens": 1, "do_sample": True})
    )
    value = {"audioArtifactId": "artifact_" + "a" * 24, "language": "auto", "timestamps": False}
    expected = execute(
        tmp_path, "automatic-speech-recognition", value, "cpu", lambda *_: None, path
    )
    assert expected == {"text": "a", "language": None}
    with pytest.raises(ValueError):
        prepare(tmp_path, "automatic-speech-recognition", "coreml")
    pack = builtin_registry().active_pack("oryn.onnx-runtime")
    for command in ["prepare", "generate", "generate"]:
        payload = {
            "modelRoot": str(tmp_path),
            "task": "automatic-speech-recognition",
            "backend": "cpu",
        }
        payload.update(input=value, inputPath=str(path), outputPath=None)
        request = {
            "abi": pack.abi,
            "packId": pack.pack_id,
            "version": pack.version,
            "command": command,
            "payload": payload,
        }
        run = subprocess.run(
            [sys.executable, "-m", "oryn_engine.runtime_packs.onnx_worker"],
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            timeout=60,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        assert run.returncode == 0, run.stderr
        result = json.loads(run.stdout.splitlines()[-1])
        assert result["ok"], (result, run.stderr)
        if command == "generate":
            assert result["result"] == expected


def test_exported_all_masked_attention_cannot_return_a_transcript(tmp_path):
    # Legacy Torch ONNX export (before 2.14) lowered SDPA to Softmax without its
    # all-masked safeguard, so original graphs exported that way can emit NaN at a
    # stride boundary. Reject invalid graph output explicitly; never change the
    # original graph or report a transcript from NaN logits.
    _, encoder, _, words = native_fixture(tmp_path, include_terminal=False)
    native = sessions(tmp_path)
    geometry = adapter.validate_graphs(inspect(tmp_path), config())
    audio = (np.sin(np.arange(1280) * 0.13) * 0.2).astype(np.float32)
    features, lens = native[0].run(
        None, {"waveforms": audio[None], "waveforms_lens": np.array([1280], np.int64)}
    )
    with torch.inference_mode():
        expected, _ = encoder(torch.from_numpy(features), torch.from_numpy(lens))
    assert np.isfinite(expected.numpy()).all()

    class LegacyExportEncoder:
        """The defective legacy export's output, independent of the local exporter."""

        def __init__(self, session):
            self.session = session

        def __getattr__(self, name):
            return getattr(self.session, name)

        def run(self, *args, **kwargs):
            encoded, lengths = self.session.run(*args, **kwargs)
            encoded = encoded.copy()
            encoded[..., -1, :] = np.nan
            return encoded, lengths

    with pytest.raises(ValueError, match="ONNX_TDT_ENCODER_OUTPUT"):
        adapter.transcribe(
            (native[0], LegacyExportEncoder(native[1]), native[2]),
            geometry,
            words,
            audio,
            lambda *_: None,
        )


def test_original_terminal_frame_at_thirty_seconds(tmp_path):
    _, _, _, words = native_fixture(tmp_path, "blank")
    native = sessions(tmp_path)
    geometry = adapter.validate_graphs(inspect(tmp_path), config())
    audio = np.random.default_rng(617).normal(0, 0.1, 480000).astype(np.float32)
    features, lens = native[0].run(
        None, {"waveforms": audio[None], "waveforms_lens": np.array([480000], np.int64)}
    )
    _, encoded_lens = native[1].run(None, {"audio_signal": features, "length": lens})
    assert lens.tolist() == [3001]
    assert encoded_lens.tolist() == [376]
    assert adapter.transcribe(native, geometry, words, audio, lambda *_: None) == {
        "text": "",
        "language": None,
    }
