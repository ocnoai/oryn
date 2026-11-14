import copy

import pytest

from oryn_engine.onnx_whisper import graph_pairs, validate_config, validate_graphs


def config():
    return dict(
        model_type="whisper",
        d_model=384,
        decoder_layers=4,
        decoder_attention_heads=6,
        encoder_layers=4,
        encoder_attention_heads=6,
        num_mel_bins=80,
        vocab_size=51864,
        max_source_positions=1500,
        max_target_positions=448,
        decoder_start_token_id=50257,
        eos_token_id=50256,
        pad_token_id=50256,
        suppress_tokens=[1, 2],
        begin_suppress_tokens=[220, 50256],
    )


def io(name, dtype, shape):
    return dict(name=name, dtype=dtype, shape=shape)


def graphs():
    return {
        "encoder": {
            "inputs": [io("input_features", 1, ["batch", 80, 3000])],
            "outputs": [io("last_hidden_state", 1, ["batch", 1500, 384])],
        },
        "decoder": {
            "inputs": [
                io("input_ids", 7, ["batch", "sequence"]),
                io("encoder_hidden_states", 1, ["batch", 1500, 384]),
            ],
            "outputs": [io("logits", 1, ["batch", "sequence", 51864])],
        },
    }


def test_whisper_graph_selection_requires_a_matching_complete_variant():
    names = [
        "onnx/encoder_model.onnx",
        "onnx/decoder_model.onnx",
        "onnx/encoder_model_quantized.onnx",
        "onnx/decoder_model_quantized.onnx",
        "onnx/encoder_model_uint8.onnx",
        "onnx/decoder_model_merged_uint8.onnx",
        "onnx/decoder_with_past_model_uint8.onnx",
        "other/decoder_model_uint8.onnx",
    ]
    assert graph_pairs(names) == [tuple(names[:2]), tuple(names[2:4])]
    assert graph_pairs(["encoder_model.onnx", "decoder_model_quantized.onnx"]) == []


def test_whisper_graph_identity_binds_both_members_and_rejects_task_borrowing():
    from oryn_engine.onnx_tasks import variant_id

    pair = ["onnx/encoder_model.onnx", "onnx/decoder_model.onnx"]
    identity = variant_id(pair, "automatic-speech-recognition")
    assert identity == variant_id(list(reversed(pair)), "automatic-speech-recognition")
    assert identity != variant_id(
        [name.replace(".onnx", "_quantized.onnx") for name in pair], "automatic-speech-recognition"
    )
    assert identity != variant_id(pair[:1], "automatic-speech-recognition")
    for names, task in [
        (pair, "feature-extraction"),
        (pair + ["extra.onnx"], "automatic-speech-recognition"),
        ([pair[0], "other/decoder_model.onnx"], "automatic-speech-recognition"),
    ]:
        with pytest.raises(ValueError, match="GRAPH_AMBIGUOUS"):
            variant_id(names, task)


@pytest.mark.parametrize(
    "changes",
    [
        {"decoder_layers": 0},
        {"d_model": 385},
        {"vocab_size": 1000000},
        {"max_source_positions": 5000},
        {"num_mel_bins": 81},
        {"suppress_tokens": [-1]},
        {"decoder_start_token_id": True},
        {"eos_token_id": 51864},
    ],
)
def test_whisper_geometry_and_generation_controls_are_bounded(changes):
    with pytest.raises(ValueError, match="CONFIGURATION"):
        validate_config(config() | changes)


def test_whisper_does_not_borrow_a_ctc_or_cache_decoder_binding():
    value = graphs()
    validate_graphs(value, config())
    for member, section, index, field, bad in (
        ("encoder", "inputs", 0, "name", "input_values"),
        ("encoder", "outputs", 0, "shape", [1, 1500, 768]),
        ("decoder", "inputs", 0, "shape", [1, 1]),
        ("decoder", "inputs", 1, "name", "past_key_values.0.encoder.key"),
        ("decoder", "outputs", 0, "shape", [1, "sequence", 51865]),
        ("decoder", "outputs", 0, "dtype", 10),
    ):
        changed = copy.deepcopy(value)
        changed[member][section][index][field] = bad
        with pytest.raises(ValueError, match="BINDING"):
            validate_graphs(changed, config())
    value["decoder"]["outputs"].append(io("present.0.encoder.key", 1, [1, 6, 1500, 64]))
    validate_graphs(value, config())
    value["decoder"]["outputs"][-1]["name"] = "present.99.encoder.key"
    with pytest.raises(ValueError, match="BINDING"):
        validate_graphs(value, config())


def test_beam_decoder_recovers_the_best_complete_sequence_and_masks_timestamps():
    import numpy as np

    from oryn_engine.onnx_whisper import decode_beams

    # Greedy chooses A first, but B -> EOS has the highest complete score.
    # Timestamp and initial EOS logits deliberately dominate before suppression.
    def logits(ids):
        if len(ids) == 1:
            return np.log([0.1, 0.4, 0.3, 0.2, 100.0])
        if ids[-1] == 2:
            return np.log([0.001, 0.001, 0.001, 0.997, 100.0])
        return np.log([0.3, 0.3, 0.3, 0.1, 100.0])

    settings = dict(vocab_size=5, max_target_positions=3, eos_token_id=3, begin_suppress_tokens=[3])
    result = decode_beams(logits, [0], settings, 3, lambda *_: None)
    assert result == [2]
    assert decode_beams(logits, [0], settings, 3, lambda *_: None) == result


@pytest.mark.parametrize("failure", ["nonfinite", "suppressed", "no-end"])
def test_beam_decoder_rejects_invalid_scores_and_unterminated_output(failure):
    import numpy as np

    from oryn_engine.onnx_whisper import decode_beams

    settings = dict(vocab_size=3, max_target_positions=4, eos_token_id=2)
    if failure == "suppressed":
        settings["suppress_tokens"] = [0, 1, 2]
    elif failure == "no-end":
        settings["suppress_tokens"] = [2]
    scores = [0.0, 1.0, np.nan if failure == "nonfinite" else -100.0]
    with pytest.raises(ValueError, match="TOKEN_LIMIT|DECODER_OUTPUT_INVALID"):
        decode_beams(lambda _: np.array(scores), [0], settings, 2, lambda *_: None)
