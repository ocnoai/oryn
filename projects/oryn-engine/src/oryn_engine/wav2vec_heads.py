"""Closed Wav2Vec2 classifier heads over complete native audio backbones."""

from __future__ import annotations

from copy import deepcopy

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

TASK = "audio-classification"
AGE = "oryn_wav2vec2_age_gender"
DUAL = "oryn_wav2vec2_dual_hypersphere"
FAMILIES = {AGE, DUAL}
ARCHITECTURES = {
    AGE: "Wav2Vec2AgeGenderModel",
    DUAL: "Wav2Vec2DualHypersphereForAudioClassification",
}
ALIASES = {
    "AutoConfig": "configuration.Wav2Vec2DualHypersphereConfig",
    "AutoModelForAudioClassification": "modeling.Wav2Vec2DualHypersphereForAudioClassification",
}
LABELS = {
    AGE: {"0": "female", "1": "male", "2": "child"},
    DUAL: {"0": "AIVoice", "1": "HumanVoice"},
}
NATIVE_KEYS = {
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "hidden_act",
    "hidden_dropout",
    "attention_dropout",
    "activation_dropout",
    "layer_norm_eps",
    "conv_dim",
    "conv_stride",
    "conv_kernel",
    "conv_bias",
    "feat_extract_norm",
    "feat_extract_activation",
    "feat_proj_dropout",
    "num_conv_pos_embeddings",
    "num_conv_pos_embedding_groups",
    "do_stable_layer_norm",
    "apply_spec_augment",
    "mask_time_prob",
    "mask_time_length",
    "mask_time_min_masks",
    "mask_feature_prob",
    "mask_feature_length",
    "mask_feature_min_masks",
    "layerdrop",
    "initializer_range",
}
# Training/export metadata remains inert and cannot select additional graph branches.
INERT_KEYS = {
    "_name_or_path",
    "transformers_version",
    "torch_dtype",
    "dtype",
    "model_type",
    "architectures",
    "adapter_kernel_size",
    "adapter_stride",
    "add_adapter",
    "adapter_attn_dim",
    "num_adapter_layers",
    "use_weighted_layer_sum",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "vocab_size",
    "classifier_proj_size",
    "final_dropout",
    "id2label",
    "label2id",
    "num_labels",
    "problem_type",
    "codevector_dim",
    "contrastive_logits_temperature",
    "ctc_loss_reduction",
    "ctc_zero_infinity",
    "diversity_loss_weight",
    "feat_quantizer_dropout",
    "feat_extract_dropout",
    "gradient_checkpointing",
    "hidden_dropout_prob",
    "finetuning_task",
    "data",
    "num_feat_extract_layers",
    "num_codevector_groups",
    "num_codevectors_per_group",
    "num_negatives",
    "output_hidden_size",
    "proj_codevector_dim",
    "sampling_rate",
    "tdnn_dilation",
    "tdnn_dim",
    "tdnn_kernel",
    "xvector_output_dim",
    "freeze_feat_extract_train",
    "mask_channel_length",
    "mask_channel_min_space",
    "mask_channel_other",
    "mask_channel_prob",
    "mask_channel_selection",
    "mask_time_min_space",
    "mask_time_other",
    "mask_time_selection",
    "no_mask_channel_overlap",
    "no_mask_time_overlap",
}


def supports(config, task):
    return task == TASK and config.get("model_type") in FAMILIES


def normalized_config(config, task):
    if task != TASK:
        return config
    if (
        config.get("model_type") == "wav2vec2"
        and config.get("architectures") == ["Model"]
        and config.get("finetuning_task") == "age-gender"
    ):
        config = config | {"model_type": AGE, "architectures": [ARCHITECTURES[AGE]]}
    elif config.get("model_type") == "wav2vec2-dual-hypersphere" and config.get(
        "architectures"
    ) == [ARCHITECTURES[DUAL]]:
        config = config | {"model_type": DUAL}
        if config.get("auto_map") == ALIASES:
            config = {k: v for k, v in config.items() if k != "auto_map"}
    if supports(config, task):
        validate_config(config)
    return config


def validate_config(config):
    validate_model_config(config)
    family = config.get("model_type")
    if family not in FAMILIES or set(config) - (NATIVE_KEYS | INERT_KEYS):
        raise ValueError("WAV2VEC_HEAD_CONFIGURATION")
    if (
        config.get("architectures") != [ARCHITECTURES[family]]
        or config.get("id2label") != LABELS[family]
        or config.get("num_labels", len(LABELS[family])) != len(LABELS[family])
        or config.get("problem_type") not in {None, "single_label_classification"}
        or config.get("add_adapter", False) is not False
        or config.get("adapter_attn_dim") is not None
        or config.get("use_weighted_layer_sum", False) is not False
        or config.get("output_hidden_size", config.get("hidden_size")) != config.get("hidden_size")
        or config.get("sampling_rate", 16000) != 16000
        or config.get("hidden_act") != "gelu"
        or config.get("feat_extract_activation") != "gelu"
        or config.get("feat_extract_norm") != ("layer" if family == AGE else "group")
        or config.get("do_stable_layer_norm") is not (family == AGE)
        or config.get("conv_bias") is not (family == AGE)
        or type(config.get("apply_spec_augment")) is not bool
    ):
        raise ValueError("WAV2VEC_HEAD_CONFIGURATION")
    inverse = {label: str(index) for index, label in config["id2label"].items()}
    if (
        not isinstance(config.get("label2id"), dict)
        or {k: str(v) for k, v in config["label2id"].items()} != inverse
    ):
        raise ValueError("WAV2VEC_HEAD_LABELS")
    for name, lo, hi in (
        ("hidden_size", 8, 1024),
        ("num_hidden_layers", 1, 24),
        ("num_attention_heads", 1, 16),
        ("intermediate_size", 8, 8192),
        ("num_conv_pos_embeddings", 2, 128),
        ("num_conv_pos_embedding_groups", 1, 16),
        ("classifier_proj_size", 1, 1024),
    ):
        if type(config.get(name)) is not int or not lo <= config[name] <= hi:
            raise ValueError("WAV2VEC_HEAD_GEOMETRY")
    if (
        config["hidden_size"] % config["num_attention_heads"]
        or config["hidden_size"] % config["num_conv_pos_embedding_groups"]
        or config["num_conv_pos_embeddings"] % 2
    ):
        raise ValueError("WAV2VEC_HEAD_GEOMETRY")
    dims, kernels, strides = (config.get(k) for k in ("conv_dim", "conv_kernel", "conv_stride"))
    if not all(isinstance(v, list) and 1 <= len(v) <= 7 for v in (dims, kernels, strides)):
        raise ValueError("WAV2VEC_HEAD_CONVOLUTION")
    if (
        len(dims) != len(kernels)
        or len(dims) != len(strides)
        or config.get("num_feat_extract_layers", len(dims)) != len(dims)
    ):
        raise ValueError("WAV2VEC_HEAD_CONVOLUTION")
    for values, hi in ((dims, 512), (kernels, 16), (strides, 8)):
        if any(type(v) is not int or not 1 <= v <= hi for v in values):
            raise ValueError("WAV2VEC_HEAD_CONVOLUTION")
    if minimum_samples(config) > 16000:
        raise ValueError("WAV2VEC_HEAD_CONVOLUTION")
    for name in (
        "hidden_dropout",
        "attention_dropout",
        "activation_dropout",
        "feat_proj_dropout",
        "final_dropout",
        "layerdrop",
        "mask_time_prob",
        "mask_feature_prob",
    ):
        value = config.get(name)
        if type(value) not in (float, int) or not 0 <= value < 1:
            raise ValueError("WAV2VEC_HEAD_CONFIGURATION")
    if (
        type(config.get("layer_norm_eps")) not in (float, int)
        or not 1e-8 <= config["layer_norm_eps"] <= 1e-3
    ):
        raise ValueError("WAV2VEC_HEAD_CONFIGURATION")


def minimum_samples(config):
    count = 1
    for kernel, stride in reversed(
        list(zip(config["conv_kernel"], config["conv_stride"], strict=True))
    ):
        count = (count - 1) * stride + kernel
    return count


def duration_limit(config):
    return 12 if config["model_type"] == DUAL else 30


def validate_processor(metadata, family):
    expected = {
        "feature_extractor_type": "Wav2Vec2FeatureExtractor",
        "sampling_rate": 16000,
        "feature_size": 1,
        "do_normalize": True,
        "padding_side": "right",
        "padding_value": 0.0,
        "return_attention_mask": True,
    }
    if family == DUAL:
        expected.update(max_length=192000, truncation=True)
    validate_model_config(metadata)
    if set(metadata) != set(expected) or any(
        type(metadata[k]) is not type(v) or metadata[k] != v for k, v in expected.items()
    ):
        raise ValueError("WAV2VEC_HEAD_PREPROCESSOR")


def native_model(config):
    import torch
    from torch import nn
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    validate_config(config)
    config = deepcopy(config)
    native = Wav2Vec2Config(**{k: deepcopy(config[k]) for k in NATIVE_KEYS if k in config})
    native._attn_implementation = "eager"

    class Head(nn.Module):
        def __init__(self, labels):
            super().__init__()
            self.dense = nn.Linear(native.hidden_size, native.hidden_size)
            self.out_proj = nn.Linear(native.hidden_size, labels)

        def forward(self, hidden):
            return self.out_proj(torch.tanh(self.dense(hidden)))

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.wav2vec2 = Wav2Vec2Model(native)
            if config["model_type"] == AGE:
                self.age, self.gender = Head(1), Head(3)
            else:
                self.projector = nn.Linear(native.hidden_size, config["classifier_proj_size"])
                self.classifier = nn.Linear(config["classifier_proj_size"], 2)

        def forward(self, input_values, attention_mask=None):
            if config["model_type"] == AGE:
                # The publisher's age/gender forward explicitly omits attention masks.
                pooled = self.wav2vec2(input_values).last_hidden_state.mean(dim=1)
                return {"logits": self.gender(pooled), "age": self.age(pooled), "pooled": pooled}
            hidden = self.wav2vec2(input_values, attention_mask=attention_mask).last_hidden_state
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = self.wav2vec2._get_feature_vector_attention_mask(
                    hidden.shape[1], attention_mask
                ).unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            # The pinned implementation uses L2 in both projections, despite the
            # model card's L1 wording. Preserve the actual published graph.
            normalized = nn.functional.normalize(pooled, p=2, dim=-1)
            projected = nn.functional.normalize(self.projector(normalized), p=2, dim=-1)
            return {"logits": self.classifier(projected), "pooled": pooled}

    return Classifier().eval()


def native_name(name):
    prefix = "wav2vec2.encoder.pos_conv_embed.conv."
    if name in {prefix + "weight_g", prefix + "weight_v"}:
        return prefix + "parametrizations.weight.original" + ("0" if name.endswith("_g") else "1")
    return name


def validate_weight_shapes(config, shapes):
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    mapped = {native_name(k): tuple(v) for k, v in shapes.items()}
    if len(mapped) != len(shapes) or mapped != {
        k: tuple(v.shape) for k, v in model.state_dict().items()
    }:
        raise ValueError("WAV2VEC_HEAD_WEIGHTS_INCOMPLETE")
    return model


def load_components(root, config, backend):
    import torch
    from safetensors.torch import load_file
    from transformers import Wav2Vec2FeatureExtractor

    processor_config = metadata_file(root / "preprocessor_config.json")
    validate_processor(processor_config, config["model_type"])
    model = validate_weight_shapes(config, preflight_safetensors(root))
    state = {}
    for path in sorted(root.glob("*.safetensors")):
        for name, value in load_file(path).items():
            target = native_name(name)
            if target in state:
                raise ValueError("WAV2VEC_HEAD_DUPLICATE_WEIGHTS")
            if not value.is_floating_point() or not bool(torch.isfinite(value.float()).all()):
                raise ValueError("WAV2VEC_HEAD_WEIGHT_VALUES")
            state[target] = value.float()
    direction = state["wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1"]
    if not bool((direction.square().sum(dim=(0, 1)) > 0).all()):
        raise ValueError("WAV2VEC_HEAD_POSITION_NORM")
    model.load_state_dict(state, strict=True, assign=True)
    # Explicit native construction never imports a repository tokenizer or Auto class.
    processor = Wav2Vec2FeatureExtractor(
        **{k: v for k, v in processor_config.items() if k != "feature_extractor_type"}
    )
    return model.eval().to(backend), processor, config


def execute(model, processor, config, value, backend, input_path):
    import torch

    from oryn_engine.vad_tasks import read_audio

    if input_path is None:
        raise ValueError("WAV2VEC_HEAD_AUDIO_MISSING")
    audio, duration = read_audio(input_path)
    if len(audio) < minimum_samples(config) or duration > duration_limit(config):
        raise ValueError("WAV2VEC_HEAD_AUDIO_LIMIT")
    # Match native single-clip pipeline preprocessing; reject overlong input
    # instead of silently truncating it according to an unused exporter setting.
    processed = processor(audio, sampling_rate=16000, return_tensors="pt", truncation=False)
    output = model(**{k: v.to(backend) for k, v in processed.items()})
    logits = output["logits"]
    if logits.shape != (1, len(LABELS[config["model_type"]])) or not bool(
        torch.isfinite(logits).all()
    ):
        raise ValueError("WAV2VEC_HEAD_OUTPUT")
    if "age" in output and not bool(torch.isfinite(output["age"]).all()):
        raise ValueError("WAV2VEC_HEAD_OUTPUT")
    scores = logits[0].float().softmax(dim=-1).cpu().tolist()
    indices = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[: value.get("topK", 5)]
    return {
        "predictions": [
            {"labelId": i, "label": config["id2label"][str(i)], "score": scores[i]} for i in indices
        ]
    }
