"""Complete NeMo TDT frontend/encoder/decoder composition on the ONNX CPU runtime."""

from __future__ import annotations

import re

from oryn_engine.model_metadata import validate_model_config

FAMILY = "nemo-conformer-tdt"
GRAPH_NAMES = ("nemo128.onnx", "encoder-model.onnx", "decoder_joint-model.onnx")
ROLES = ("frontend", "encoder", "decoder")
SPACE = re.compile(r"\A\s|\s\B|(\s)\b")


def supports(config, task):
    return task == "automatic-speech-recognition" and config.get("model_type") == FAMILY


def validate_config(config):
    validate_model_config(config)
    if (
        set(config) != {"model_type", "features_size", "subsampling_factor"}
        or config["model_type"] != FAMILY
        or type(config["features_size"]) is not int
        or config["features_size"] != 128
        or type(config["subsampling_factor"]) is not int
        or config["subsampling_factor"] != 8
    ):
        raise ValueError("ONNX_TDT_CONFIGURATION")


def graph_groups(names):
    return [GRAPH_NAMES] if set(GRAPH_NAMES) <= set(names) else []


def compose(members):
    return dict(zip(ROLES, members, strict=True)) | {
        "operators": sorted({op for m in members for op in m["operators"]})
    }


def validate_graphs(graph, config):
    validate_config(config)
    try:
        maps = {}
        for role in ROLES:
            for key in ("inputs", "outputs"):
                values = graph[role][key]
                maps[role, key] = {v["name"]: v for v in values}
                if len(values) != len(maps[role, key]):
                    raise ValueError("ONNX_TDT_BINDING")
        encoder = maps["encoder", "outputs"]["outputs"]["shape"][1]
        state = maps["decoder", "inputs"]["input_states_1"]["shape"]
        layers, hidden = state[0], state[2]
        vocab = maps["decoder", "outputs"]["outputs"]["shape"][-1] - 5
        if any(type(v) is not int for v in (encoder, layers, hidden, vocab)) or not (
            8 <= encoder <= 2048
            and 1 <= layers <= 4
            and 8 <= hidden <= 2048
            and 8 <= vocab <= 131072
        ):
            raise ValueError("ONNX_TDT_GEOMETRY")
        expected = {
            ("frontend", "inputs"): {"waveforms": (1, [1, None]), "waveforms_lens": (7, [1])},
            ("frontend", "outputs"): {"features": (1, [1, 128, None]), "features_lens": (7, [1])},
            ("encoder", "inputs"): {"audio_signal": (1, [1, 128, None]), "length": (7, [1])},
            ("encoder", "outputs"): {
                "outputs": (1, [1, encoder, None]),
                "encoded_lengths": (7, [1]),
            },
            ("decoder", "inputs"): {
                "encoder_outputs": (1, [1, encoder, 1]),
                "targets": (6, [1, 1]),
                "target_length": (6, [1]),
                "input_states_1": (1, [layers, 1, hidden]),
                "input_states_2": (1, [layers, 1, hidden]),
            },
            ("decoder", "outputs"): {
                "outputs": (1, [1, 1, 1, vocab + 5]),
                "prednet_lengths": (6, [1]),
                "output_states_1": (1, [layers, 1, hidden]),
                "output_states_2": (1, [layers, 1, hidden]),
            },
        }
        for key, fields in expected.items():
            if maps[key].keys() != fields.keys():
                raise ValueError("ONNX_TDT_BINDING")
            for name, (dtype, shape) in fields.items():
                actual = maps[key][name]
                if actual["dtype"] != dtype or len(actual["shape"]) != len(shape):
                    raise ValueError("ONNX_TDT_BINDING")
                for value, bound in zip(actual["shape"], shape, strict=True):
                    if type(value) is int and (bound is None or value != bound):
                        raise ValueError("ONNX_TDT_BINDING")
        # Learned and state axes must be explicit, not inferred from arbitrary
        # symbolic strings. Other dynamic axes are bounded again on each output.
        for key, name, axes in (
            (("frontend", "outputs"), "features", {1: 128}),
            (("encoder", "inputs"), "audio_signal", {1: 128}),
            (("encoder", "outputs"), "outputs", {1: encoder}),
            (("decoder", "inputs"), "encoder_outputs", {1: encoder}),
            (("decoder", "outputs"), "outputs", {3: vocab + 5}),
            *(
                (
                    (("decoder", direction), name, {0: layers, 2: hidden})
                    for direction, prefix in (("inputs", "input"), ("outputs", "output"))
                    for name in (prefix + "_states_1", prefix + "_states_2")
                )
            ),
        ):
            if any(maps[key][name]["shape"][i] != v for i, v in axes.items()):
                raise ValueError("ONNX_TDT_GEOMETRY")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("ONNX_TDT_BINDING") from exc
    return {"encoder": encoder, "layers": layers, "hidden": hidden, "vocab": vocab}


def vocabulary(root, count):
    path = root / "vocab.txt"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 4 * 1024**2:
        raise ValueError("ONNX_TDT_VOCABULARY")
    tokens = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        parts = line.rsplit(" ", 1)
        if len(parts) != 2 or parts[1] != str(index) or not 1 <= len(parts[0].encode()) <= 512:
            raise ValueError("ONNX_TDT_VOCABULARY")
        tokens.append(parts[0])
    if len(tokens) != count or len(set(tokens)) != count or tokens[-1] != "<blk>":
        raise ValueError("ONNX_TDT_VOCABULARY")
    return tokens


def transcribe(sessions, geometry, tokens, audio, event):
    import numpy as np

    frontend, encoder, decoder = sessions
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not 320 <= audio.size <= 480000 or not np.isfinite(audio).all():
        raise ValueError("ONNX_TDT_AUDIO_BOUND")
    frames = audio.size // 160
    features, lengths = frontend.run(
        ["features", "features_lens"],
        {"waveforms": audio[None], "waveforms_lens": np.array([audio.size], np.int64)},
    )
    if (
        features.dtype != np.float32
        or features.shape != (1, 128, frames + 1)
        or not np.isfinite(features).all()
        or lengths.dtype != np.int64
        or lengths.shape != (1,)
        or lengths[0] not in {frames, frames + 1}
    ):
        raise ValueError("ONNX_TDT_FRONTEND_OUTPUT")
    encoded, encoded_lens = encoder.run(
        ["outputs", "encoded_lengths"],
        {"audio_signal": features, "length": lengths},
    )
    # The frozen frontend counts its terminal STFT frame; newer reviewed
    # exports mask it. Preserve the actual bounded length emitted by the graph.
    valid = (int(lengths[0]) + 7) // 8
    if (
        encoded.dtype != np.float32
        or encoded.ndim != 3
        or encoded.shape[:2] != (1, geometry["encoder"])
        or not valid <= encoded.shape[2] <= valid + 1
        or not np.isfinite(encoded).all()
        or encoded_lens.dtype != np.int64
        or encoded_lens.shape != (1,)
        or encoded_lens[0] != valid
    ):
        raise ValueError("ONNX_TDT_ENCODER_OUTPUT")
    blank = geometry["vocab"] - 1
    shape = (geometry["layers"], 1, geometry["hidden"])
    state = [np.zeros(shape, np.float32), np.zeros(shape, np.float32)]
    ids, frame, emitted, steps = [], 0, 0, 0
    while frame < valid:
        steps += 1
        if steps > valid * 10:
            raise ValueError("ONNX_TDT_STEP_BOUND")
        outputs, predlen, h, c = decoder.run(
            ["outputs", "prednet_lengths", "output_states_1", "output_states_2"],
            {
                "encoder_outputs": np.ascontiguousarray(encoded[:, :, frame : frame + 1]),
                "targets": np.array([[ids[-1] if ids else blank]], np.int32),
                "target_length": np.ones(1, np.int32),
                "input_states_1": state[0],
                "input_states_2": state[1],
            },
        )
        if (
            outputs.dtype != np.float32
            or outputs.shape != (1, 1, 1, geometry["vocab"] + 5)
            or not np.isfinite(outputs).all()
            or predlen.dtype != np.int32
            or predlen.shape != (1,)
            or predlen[0] != 1
            or any(
                v.dtype != np.float32 or v.shape != shape or not np.isfinite(v).all()
                for v in (h, c)
            )
        ):
            raise ValueError("ONNX_TDT_DECODER_OUTPUT")
        logits = outputs[0, 0, 0]
        token, duration = (
            int(logits[: geometry["vocab"]].argmax()),
            int(logits[geometry["vocab"] :].argmax()),
        )
        if token != blank:
            ids.append(token)
            state = [h, c]
            emitted += 1
        if duration > 0:
            frame += duration
            emitted = 0
        elif token == blank or emitted == 10:
            frame += 1
            emitted = 0
        if steps % 64 == 0:
            event(
                "job.progress",
                {"stage": "transcribing", "progress": 0.3 + 0.6 * min(frame / valid, 1)},
            )
    text = SPACE.sub(
        lambda m: " " if m.group(1) else "", "".join(tokens[i].replace("▁", " ") for i in ids)
    )
    if len(text.encode()) > 16384:
        raise ValueError("ONNX_TDT_OUTPUT_BOUND")
    return {"text": text, "language": None}


def execute(sessions, root, config, value, input_path, event):
    from oryn_engine.vad_tasks import read_audio

    if value.get("timestamps", False) or value.get("language", "auto") != "auto":
        raise ValueError("ONNX_TDT_PARAMETERS")
    geometry = config["_orynTdtGeometry"]
    tokens = vocabulary(root, geometry["vocab"])
    audio, _ = read_audio(input_path)
    return transcribe(sessions, geometry, tokens, audio, event)
