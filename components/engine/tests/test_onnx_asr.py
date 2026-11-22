import copy
import json
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from oryn_engine.onnx_asr import execute, validate_config, validate_graph


def test_ctc_graph_does_not_borrow_encoder_or_decoder_bindings():
    config = {"model_type": "wav2vec2", "vocab_size": 6, "pad_token_id": 0}
    validate_config(config)
    graph = {
        "inputs": [{"name": "input_values", "dtype": 1, "shape": [1, None]}],
        "outputs": [{"name": "logits", "dtype": 1, "shape": [1, None, 6]}],
    }
    validate_graph(graph, config)
    for section, key, value in (
        ("inputs", "name", "input_ids"),
        ("inputs", "shape", [1, 16000]),
        ("inputs", "dtype", 7),
        ("outputs", "shape", [1, None, 7]),
        ("outputs", "name", "last_hidden_state"),
    ):
        bad = copy.deepcopy(graph)
        bad[section][0][key] = value
        with pytest.raises(ValueError, match="CTC_BINDING"):
            validate_graph(bad, config)


def test_ctc_collapse_preserves_blank_separated_repeats_and_has_no_language_claim(tmp_path):
    from transformers import Wav2Vec2CTCTokenizer

    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "A": 5}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    tokenizer = Wav2Vec2CTCTokenizer(tmp_path / "vocab.json")
    tokenizer.save_pretrained(tmp_path)
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "feature_extractor_type": "Wav2Vec2FeatureExtractor",
                "sampling_rate": 16000,
                "feature_size": 1,
                "do_normalize": True,
                "padding_value": 0,
            }
        )
    )
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes((1000 * np.sin(np.arange(1600) / 10)).astype("<i2").tobytes())
    config = {"model_type": "wav2vec2", "vocab_size": 6, "pad_token_id": 0}
    ids = [5, 5, 0, 5, 4, 5]
    logits = np.full((1, len(ids), 6), -10, dtype=np.float32)
    logits[0, np.arange(len(ids)), ids] = 10

    def run(names, inputs):
        assert names == ["logits"]
        assert inputs["input_values"].shape == (1, 1600)
        assert inputs["input_values"].dtype == np.float32
        assert np.isfinite(inputs["input_values"]).all()
        return [logits]

    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="input_values")], run=run)
    assert execute(session, tmp_path, config, {}, audio, lambda *_: None) == {
        "text": "AA A",
        "language": None,
    }
    for value in ({"language": "fr"}, {"timestamps": True}):
        with pytest.raises(ValueError, match="PARAMETER_UNSUPPORTED"):
            execute(session, tmp_path, config, value, audio, lambda *_: None)
    silence = tmp_path / "silence.wav"
    with wave.open(str(silence), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(np.zeros(1600, dtype="<i2").tobytes())
    assert execute(session, tmp_path, config, {}, silence, lambda *_: None)["text"] == ""
    logits[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="OUTPUT_INVALID"):
        execute(session, tmp_path, config, {}, audio, lambda *_: None)
    config["vocab_size"] = 7
    with pytest.raises(ValueError, match="VOCABULARY_MISMATCH"):
        execute(session, tmp_path, config, {}, audio, lambda *_: None)


def test_kokoro_rejects_unknown_voice_language_mismatch_and_timestamps():
    import pytest

    from oryn_engine.runtime_packs.kokoro_onnx_worker import _input

    valid = {"text": "Hello", "voice": "af_heart", "language": "en-us", "speed": 1.0}
    assert _input({"input": valid}) == valid
    for change in (
        {"voice": "unknown"},
        {"language": "en-gb"},
        {"timestamps": True},
        {"speed": 3.0},
    ):
        with pytest.raises(ValueError, match="speech input"):
            _input({"input": valid | change})


def test_ctc_large_character_vocabulary_preserves_unicode_and_blank_separated_repeats(tmp_path):
    from transformers import Wav2Vec2CTCTokenizer

    from oryn_engine.onnx_asr import load_components, normalize_logits

    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4}
    vocab.update({chr(0x4E00 + i): i + 5 for i in range(2336)})
    (tmp_path / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))
    Wav2Vec2CTCTokenizer(str(tmp_path / "vocab.json")).save_pretrained(tmp_path)
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "feature_extractor_type": "Wav2Vec2FeatureExtractor",
                "sampling_rate": 16000,
                "feature_size": 1,
                "do_normalize": True,
                "padding_value": 0,
            }
        )
    )
    config = {"model_type": "wav2vec2", "vocab_size": 2341, "pad_token_id": 0}
    _, tokenizer = load_components(tmp_path, config)
    # Older native CTC exports predate the processor-class annotation. The
    # selected family still determines one closed feature extractor.
    metadata_path = tmp_path / "preprocessor_config.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["feature_extractor_type"]
    metadata_path.write_text(json.dumps(metadata))
    _, legacy_tokenizer = load_components(tmp_path, config)
    assert legacy_tokenizer.get_vocab() == tokenizer.get_vocab()
    metadata_path.write_text(
        json.dumps({**metadata, "feature_extractor_type": "UnreviewedExtractor"})
    )
    with pytest.raises(ValueError, match="PREPROCESSOR_UNSUPPORTED"):
        load_components(tmp_path, config)
    metadata_path.write_text(json.dumps(metadata))
    logits = np.full((1, 4, 2341), -10, dtype=np.float32)
    logits[0, np.arange(4), [2339, 2339, 0, 2339]] = 10
    text = chr(0x4E00 + 2334)
    assert normalize_logits(logits, tokenizer, config) == {"text": text * 2, "language": None}
    validate_config(config | {"vocab_size": 4096})
    for size in (0, 4097, True):
        with pytest.raises(ValueError, match="CONFIGURATION"):
            validate_config(config | {"vocab_size": size})
    with pytest.raises(ValueError, match="VOCABULARY_MISMATCH"):
        load_components(tmp_path, config | {"vocab_size": 2340})
