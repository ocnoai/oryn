"""Shared native audio bindings use synthetic graphs and bounded metadata only."""

import json
import wave

import numpy as np
import pytest
import torch

from oryn_engine import audio_understanding as audio
from oryn_engine.shared_tasks import load_task, validate_processor_config


@pytest.fixture
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def processor_metadata(family):
    return {
        "processor_class": "MusicFlamingoProcessor"
        if family == "musicflamingo"
        else "AudioFlamingo3Processor",
        "audio_token": "<sound>",
        "feature_extractor": {
            "feature_extractor_type": "WhisperFeatureExtractor",
            "feature_size": 128,
            "sampling_rate": 16000,
            "chunk_length": 30,
            "hop_length": 160,
            "n_fft": 400,
        },
    }


@pytest.mark.parametrize("family", ["audioflamingo3", "musicflamingo"])
def test_native_flamingo_graph_and_processor_use_one_owned_audio_chunk(
    tmp_path, single_thread, family
):
    import transformers as tf
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {
        token: index
        for index, token in enumerate(
            [
                "<unk>",
                "<pad>",
                "<sound>",
                "<|sound_bos|>",
                "<|sound_eos|>",
                "<|im_start|>",
                "<|im_end|>",
                "Describe",
                "sound",
                "user",
                "assistant",
            ]
        )
    }
    vocab.update({f"word{i}": i for i in range(len(vocab), 32)})
    native = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    native.pre_tokenizer = Whitespace()
    tokenizer = tf.PreTrainedTokenizerFast(
        tokenizer_object=native,
        unk_token="<unk>",
        pad_token="<pad>",
        model_max_length=4096,
        additional_special_tokens=list(vocab)[2:7],
    )
    tokenizer.save_pretrained(tmp_path)
    (tmp_path / "processor_config.json").write_text(json.dumps(processor_metadata(family)))
    encoder = tf.AudioFlamingo3EncoderConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_mel_bins=128,
        max_source_positions=1500,
    )
    text = tf.Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        pad_token_id=1,
        eos_token_id=6,
    )
    config_class, model_class = (
        (tf.MusicFlamingoConfig, tf.MusicFlamingoForConditionalGeneration)
        if family == "musicflamingo"
        else (tf.AudioFlamingo3Config, tf.AudioFlamingo3ForConditionalGeneration)
    )
    options = (
        {
            "audio_bos_token_id": 3,
            "audio_eos_token_id": 4,
            "head_dim": 16,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 1200,
                "partial_rotary_factor": 0.25,
            },
        }
        if family == "musicflamingo"
        else {}
    )
    config = config_class(
        audio_config=encoder.to_dict(), text_config=text.to_dict(), audio_token_id=2, **options
    )
    torch.manual_seed(61)
    model_class(config).save_pretrained(tmp_path, safe_serialization=True)
    model, processor, loaded = load_task(tmp_path, "audio-text-to-text", "cpu")
    assert loaded["model_type"] == family and processor.max_audio_len == 30
    assert not processor.chat_template and not processor.tokenizer.chat_template
    path = tmp_path / "sound.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes((1000 * np.sin(np.arange(1600) / 10)).astype("<i2").tobytes())
    seen = []
    original = model.generate

    def capture(**kwargs):
        seen.append(kwargs)
        return original(**kwargs)

    model.generate = capture
    value = {"instruction": "Describe sound", "maxTokens": 2}
    try:
        audio.execute(model, processor, value, "cpu", path)
    except ValueError as error:
        # Random fixture weights can generate only special tokens; the real
        # native joint audio/text forward and decoding still execute.
        assert "answer is empty" in str(error)
    assert len(seen) == 1 and tuple(seen[0]["input_features"].shape) == (1, 128, 3000)
    assert tuple(seen[0]["input_features_mask"].shape) == (1, 3000)
    assert seen[0]["max_new_tokens"] == 2 and seen[0]["do_sample"] is False
    ids = seen[0]["input_ids"][0].tolist()
    assert ids.count(2) > 1 and ids.count(5) == 3 and ids.count(6) == 2
    if family == "musicflamingo":
        assert ids.count(3) == ids.count(4) == 1
    with pytest.raises(ValueError, match="reserved native framing"):
        audio.execute(model, processor, value | {"instruction": "<sound>"}, "cpu", path)
    config_dict = json.loads((tmp_path / "config.json").read_text())
    config_dict["audio_token_id"] = 8
    with pytest.raises(ValueError, match="identities disagree"):
        audio.load_flamingo(
            tmp_path, config_dict, {"local_files_only": True, "trust_remote_code": False}
        )


@pytest.mark.parametrize(
    "change",
    [
        {"sampling_rate": 48000},
        {"hop_length": 320},
        {"feature_size": 80},
        {"n_fft": 1024},
        {"chunk_length": 120},
        {"return_attention_mask": 1},
        {"auto_map": {"AutoFeatureExtractor": "repo.Code"}},
        {"device": "cuda"},
    ],
)
def test_flamingo_rejects_unbound_frontends(change):
    metadata = processor_metadata("audioflamingo3")
    metadata["feature_extractor"].update(change)
    with pytest.raises(ValueError):
        validate_processor_config(metadata, "audioflamingo3")


def test_flamingo_repository_window_cannot_expand_runtime_budget():
    metadata = processor_metadata("musicflamingo") | {"max_audio_len": 1200}
    assert audio.flamingo_processor_options(metadata, "musicflamingo")["chunk_length"] == 30
    for change in (
        {"max_audio_len": -1},
        {"audio_token": "custom"},
        {"processor_class": "RepoProcessor"},
    ):
        with pytest.raises(ValueError):
            audio.flamingo_processor_options(metadata | change, "musicflamingo")
