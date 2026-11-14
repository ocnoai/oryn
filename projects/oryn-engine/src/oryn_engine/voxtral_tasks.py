"""Native Voxtral audio instruction graph with bounded, build-owned Tekken framing."""

from __future__ import annotations

from oryn_engine.model_metadata import metadata_file, validate_model_config

FEATURES = {
    "feature_size": 128,
    "sampling_rate": 16000,
    "chunk_length": 30,
    "hop_length": 160,
    "n_fft": 400,
    "n_samples": 480000,
    "nb_max_frames": 3000,
    "padding_side": "right",
    "padding_value": 0.0,
    "return_attention_mask": False,
    "dither": 0.0,
}


def validate_config(config):
    validate_model_config(config)
    audio, text = config.get("audio_config"), config.get("text_config")
    if (
        config.get("model_type") != "voxtral"
        or config.get("architectures") != ["VoxtralForConditionalGeneration"]
        or not isinstance(audio, dict)
        or not isinstance(text, dict)
        or audio.get("model_type") != "voxtral_encoder"
        or text.get("model_type") != "llama"
        or config.get("audio_token_id") != 24
        or config.get("projector_hidden_act", "gelu") != "gelu"
        or any(
            c.get(k) is not None
            for c in (config, audio, text)
            for k in ("quantization", "quantization_config")
        )
    ):
        raise ValueError("VOXTRAL_CONFIGURATION_UNSUPPORTED")
    for part, bounds in (
        (
            audio,
            {
                "hidden_size": (8, 2048),
                "intermediate_size": (32, 8192),
                "num_attention_heads": (1, 32),
                "num_hidden_layers": (1, 48),
            },
        ),
        (
            text,
            {
                "hidden_size": (8, 8192),
                "intermediate_size": (8, 65536),
                "num_attention_heads": (1, 64),
                "num_key_value_heads": (1, 64),
                "num_hidden_layers": (1, 80),
                "head_dim": (4, 256),
                "vocab_size": (1256, 262144),
                "max_position_embeddings": (4224, 1048576),
            },
        ),
    ):
        for key, (low, high) in bounds.items():
            if type(part.get(key)) is not int or not low <= part[key] <= high:
                raise ValueError("VOXTRAL_CONFIGURATION_LIMIT")
    for part, expected in (
        (
            audio,
            {
                "num_mel_bins": 128,
                "max_source_positions": 1500,
                "scale_embedding": False,
                "activation_function": "gelu",
                "attention_dropout": 0.0,
                "activation_dropout": 0.0,
                "dropout": 0.0,
                "layerdrop": 0.0,
            },
        ),
        (
            text,
            {
                "attention_bias": False,
                "mlp_bias": False,
                "hidden_act": "silu",
                "attention_dropout": 0.0,
                "pretraining_tp": 1,
                "sliding_window": None,
                "rope_scaling": None,
                "use_cache": True,
            },
        ),
    ):
        if any(part.get(k, v) != v for k, v in expected.items()):
            raise ValueError("VOXTRAL_CONFIGURATION_UNSUPPORTED")
    rope = text.get("rope_parameters")
    if rope is not None and (
        not isinstance(rope, dict)
        or set(rope) - {"rope_type", "rope_theta"}
        or rope.get("rope_type", "default") != "default"
    ):
        raise ValueError("VOXTRAL_ROTARY_CONFIGURATION")
    if (
        audio["hidden_size"] % audio["num_attention_heads"]
        or audio["intermediate_size"] != 4 * audio["hidden_size"]
        or text["num_attention_heads"] % text["num_key_value_heads"]
        or text["head_dim"] % 2
        or config.get("vocab_size", text["vocab_size"]) != text["vocab_size"]
        or config.get("hidden_size", text["hidden_size"]) != text["hidden_size"]
    ):
        raise ValueError("VOXTRAL_GRAPH_GEOMETRY")


def processor_options(metadata):
    validate_model_config(metadata)
    expected = FEATURES | {
        "feature_extractor_type": "WhisperFeatureExtractor",
        "processor_class": "VoxtralProcessor",
    }
    if (
        set(metadata) - set(expected)
        or metadata.get("feature_extractor_type") != "WhisperFeatureExtractor"
        or any(
            type(metadata[k]) is not type(v) or metadata[k] != v
            for k, v in expected.items()
            if k in metadata
        )
    ):
        raise ValueError("VOXTRAL_PROCESSOR_UNSUPPORTED")
    return dict(FEATURES)


class Processor:
    def __init__(self, root, config):
        from transformers import WhisperFeatureExtractor

        from oryn_engine.tekken_tokenizer import TekkenV7

        self.tokenizer = TekkenV7(root / "tekken.json", config["text_config"]["vocab_size"])
        self.extractor = WhisperFeatureExtractor(
            **processor_options(metadata_file(root / "preprocessor_config.json"))
        )

    def prepare(self, audio, instruction):
        import numpy as np
        import torch

        if (
            audio.ndim != 1
            or not 80 <= audio.size <= 480000
            or not np.isfinite(audio).all()
            or not isinstance(instruction, str)
            or not 1 <= len(instruction) <= 16384
        ):
            raise ValueError("VOXTRAL_INPUT_LIMIT")
        # User text is encoded as ordinary bytes: literal control strings cannot
        # introduce another turn or change the fixed audio-placeholder count.
        ids = [1, 3, 25] + [24] * 375 + self.tokenizer.encode(instruction) + [4]
        if len(ids) > 4096:
            raise ValueError("VOXTRAL_CONTEXT_LIMIT")
        features = self.extractor(audio, sampling_rate=16000, return_tensors="pt")["input_features"]
        if tuple(features.shape) != (1, 128, 3000) or not bool(torch.isfinite(features).all()):
            raise ValueError("VOXTRAL_FEATURE_GEOMETRY")
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            "input_features": features,
        }


def execute(model, processor, audio, value, backend):
    from transformers import GenerationConfig

    from oryn_engine.shared_tasks import _move

    limit = value.get("maxTokens", 128)
    if type(limit) is not int or not 1 <= limit <= 128:
        raise ValueError("VOXTRAL_OUTPUT_LIMIT")
    inputs = _move(processor.prepare(audio, value["instruction"]), backend)
    generated = model.generate(
        **inputs,
        generation_config=GenerationConfig(
            bos_token_id=1, eos_token_id=2, pad_token_id=11, do_sample=False, max_new_tokens=limit
        ),
    )
    text = processor.tokenizer.decode(
        generated[0, inputs["input_ids"].shape[-1] :].tolist()
    ).strip()
    if not text or len(text.encode()) > 16384:
        raise ValueError("Grounded audio answer is empty or oversized")
    return {"text": text}
