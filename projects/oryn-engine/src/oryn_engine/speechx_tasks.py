"""Closed native Whisper encoder with separately selected classification heads."""

from __future__ import annotations

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

FAMILY = "oryn_speechx"
ARCHITECTURE = "OrynWhisperMultiHeadClassifier"
TASK = "audio-classification"
ALIASES = {
    "AutoConfig": "configuration_speechx.SpeechXConfig",
    "AutoModel": "modeling_speechx.SpeechXModel",
}
BASE_ENCODER = dict(
    d_model=512,
    encoder_layers=6,
    encoder_attention_heads=8,
    encoder_ffn_dim=2048,
    max_source_positions=1500,
    num_mel_bins=80,
)
FIELDS = {
    "model_type",
    "architectures",
    "base_model",
    "d_model",
    "dropout",
    "dtype",
    "gender2id",
    "id2gender",
    "lang2id",
    "id2lang",
    "num_genders",
    "num_langs",
    "transformers_version",
    "encoder_config",
}


def supports(config, task):
    return task == TASK and config.get("model_type") == FAMILY


def normalized_config(config, task):
    if task == TASK and config.get("model_type") == "speechx":
        if (
            config.get("architectures") != ["SpeechXModel"]
            or config.get("auto_map") != ALIASES
            or config.get("base_model") != "openai/whisper-base"
            or config.get("d_model") != 512
            or "encoder_config" in config
        ):
            raise ValueError("SPEECHX_NATIVE_IDENTITY")
        config = {k: v for k, v in config.items() if k != "auto_map"}
        config.update(
            model_type=FAMILY, architectures=[ARCHITECTURE], encoder_config=dict(BASE_ENCODER)
        )
    if supports(config, task):
        validate_config(config)
    return config


def validate_config(config):
    validate_model_config(config)
    if (
        not supports(config, TASK)
        or set(config) - FIELDS
        or config.get("architectures") != [ARCHITECTURE]
        or config.get("base_model") != "openai/whisper-base"
        or config.get("dtype") not in {"float16", "float32"}
        or type(config.get("dropout")) not in {int, float}
        or not 0 <= config["dropout"] < 1
    ):
        raise ValueError("SPEECHX_CONFIGURATION")
    native = config.get("encoder_config")
    if not isinstance(native, dict) or set(native) != set(BASE_ENCODER):
        raise ValueError("SPEECHX_ENCODER")
    for key, low, high in [
        ("d_model", 16, 512),
        ("encoder_layers", 1, 6),
        ("encoder_attention_heads", 1, 8),
        ("encoder_ffn_dim", 16, 2048),
        ("max_source_positions", 50, 1500),
        ("num_mel_bins", 80, 80),
    ]:
        if type(native.get(key)) is not int or not low <= native[key] <= high:
            raise ValueError("SPEECHX_ENCODER")
    if (
        type(config.get("d_model")) is not int
        or config["d_model"] != native["d_model"]
        or native["d_model"] % native["encoder_attention_heads"]
        or native["d_model"] % 2
        or native["max_source_positions"] % 50
    ):
        raise ValueError("SPEECHX_ENCODER")
    for name, maximum in [("gender", 2), ("lang", 256)]:
        size = config.get("num_" + name + "s")
        labels, reverse = config.get("id2" + name), config.get(name + "2id")
        if (
            type(size) is not int
            or not 2 <= size <= maximum
            or not isinstance(labels, dict)
            or set(labels) != {str(i) for i in range(size)}
            or any(
                not isinstance(v, str) or not 1 <= len(v) <= 64 or v.strip() != v
                for v in labels.values()
            )
            or len(set(labels.values())) != size
            or not isinstance(reverse, dict)
            or any(type(v) is not int for v in reverse.values())
            or reverse != {label: int(i) for i, label in labels.items()}
        ):
            raise ValueError("SPEECHX_LABELS")
    if config["id2gender"] != {"0": "female", "1": "male"}:
        raise ValueError("SPEECHX_LABELS")


def duration_limit(config):
    return config["encoder_config"]["max_source_positions"] / 50


def validate_processor(metadata, config=None):
    seconds = metadata.get("chunk_length")
    if type(seconds) is not int or not 1 <= seconds <= 30:
        raise ValueError("SPEECHX_PREPROCESSOR")
    expected = dict(
        chunk_length=seconds,
        dither=0.0,
        feature_extractor_type="WhisperFeatureExtractor",
        feature_size=80,
        hop_length=160,
        n_fft=400,
        n_samples=seconds * 16000,
        nb_max_frames=seconds * 100,
        padding_side="right",
        padding_value=0.0,
        processor_class="WhisperProcessor",
        return_attention_mask=False,
        sampling_rate=16000,
    )
    if (
        set(metadata) != set(expected)
        or any(type(metadata[k]) is not type(v) or metadata[k] != v for k, v in expected.items())
        or config is not None
        and duration_limit(config) != seconds
    ):
        raise ValueError("SPEECHX_PREPROCESSOR")


def native_model(config):
    from torch import nn
    from transformers import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    validate_config(config)
    native = WhisperConfig(
        **config["encoder_config"],
        activation_function="gelu",
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        encoder_layerdrop=0.0,
        scale_embedding=False,
    )
    native._attn_implementation = "eager"

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = WhisperEncoder(native)
            self.dropout = nn.Dropout(config["dropout"])
            self.gender_head = nn.Linear(config["d_model"], config["num_genders"])
            self.lang_head = nn.Linear(config["d_model"], config["num_langs"])

        def forward(self, input_features):
            # Publisher pooling includes every fixed-window frame, including padding.
            hidden = self.dropout(self.encoder(input_features).last_hidden_state.mean(dim=1))
            return {"gender": self.gender_head(hidden), "language": self.lang_head(hidden)}

    return Classifier().eval()


def validate_weight_shapes(config, shapes):
    import torch

    with torch.device("meta"):
        model = native_model(config)
    if {k: tuple(v) for k, v in shapes.items()} != {
        k: tuple(v.shape) for k, v in model.state_dict().items()
    }:
        raise ValueError("SPEECHX_WEIGHTS_INCOMPLETE")
    return model


def load_components(root, config, backend):
    import torch
    from safetensors.torch import load_file
    from transformers import WhisperFeatureExtractor

    metadata = metadata_file(root / "preprocessor_config.json")
    validate_processor(metadata, config)
    model = validate_weight_shapes(config, preflight_safetensors(root))
    state = {}
    for path in sorted(root.glob("*.safetensors")):
        for name, value in load_file(path).items():
            if name in state:
                raise ValueError("SPEECHX_DUPLICATE_WEIGHTS")
            if not value.is_floating_point() or not bool(torch.isfinite(value.float()).all()):
                raise ValueError("SPEECHX_WEIGHT_VALUES")
            state[name] = value.float()
    model.load_state_dict(state, strict=True, assign=True)
    processor = WhisperFeatureExtractor(
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        n_fft=400,
        chunk_length=metadata["chunk_length"],
        padding_value=0.0,
        padding_side="right",
        return_attention_mask=False,
        dither=0.0,
    )
    return model.eval().to(backend), processor, config


def execute(model, processor, config, value, backend, input_path):
    import torch

    from oryn_engine.vad_tasks import read_audio

    head = value.get("classificationHead", "language")
    if head not in {"language", "gender"}:
        raise ValueError("SPEECHX_HEAD")
    if input_path is None:
        raise ValueError("SPEECHX_AUDIO_MISSING")
    audio, duration = read_audio(input_path)
    if len(audio) < 320 or duration > duration_limit(config):
        raise ValueError("SPEECHX_AUDIO_LIMIT")
    features = processor(
        audio, sampling_rate=16000, return_tensors="pt", truncation=False
    ).input_features
    if features.shape != (1, 80, 2 * config["encoder_config"]["max_source_positions"]) or not bool(
        torch.isfinite(features).all()
    ):
        raise ValueError("SPEECHX_FEATURES")
    outputs = model(features.to(backend))
    for name, count in [("gender", config["num_genders"]), ("language", config["num_langs"])]:
        if outputs[name].shape != (1, count) or not bool(torch.isfinite(outputs[name]).all()):
            raise ValueError("SPEECHX_OUTPUT")
    labels = config["id2lang" if head == "language" else "id2gender"]
    scores = outputs[head][0].float().softmax(dim=-1).cpu().tolist()
    indices = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[: value.get("topK", 5)]
    return {
        "predictions": [
            {"labelId": i, "label": labels[str(i)], "score": scores[i]} for i in indices
        ]
    }
