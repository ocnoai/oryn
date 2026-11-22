"""Native Nemotron RNN-T transcription with complete graphs and bounded decoding."""

from __future__ import annotations

import copy

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

FAMILY = "nemotron3_5_asr"
ARCHITECTURE = "Nemotron3_5AsrForRNNT"
METADATA_NAMES = ("processor_config.json", "tokenizer_config.json")
FEATURES = {
    "feature_extractor_type": "NemotronAsrStreamingFeatureExtractor",
    "feature_size": 128,
    "hop_length": 160,
    "n_fft": 512,
    "padding_side": "right",
    "padding_value": 0.0,
    "preemphasis": 0.97,
    "return_attention_mask": True,
    "sampling_rate": 16000,
    "win_length": 400,
}
MAX_OUTPUT_TOKENS = 4096


def supports(config, task):
    return task == "automatic-speech-recognition" and config.get("model_type") == FAMILY


def validate_config(config):
    validate_model_config(config)
    encoder = config.get("encoder_config")
    if (
        config.get("model_type") != FAMILY
        or config.get("architectures") != [ARCHITECTURE]
        or not isinstance(encoder, dict)
        or encoder.get("model_type") != "nemotron_asr_streaming_encoder"
        or any(config.get(k) is not None for k in ("quantization", "quantization_config"))
    ):
        raise ValueError("RNNT_CONFIGURATION")
    for part, limits in (
        (
            config,
            {
                "vocab_size": (16, 32768),
                "decoder_hidden_size": (16, 2048),
                "num_decoder_layers": (1, 4),
                "prompt_intermediate_size": (16, 8192),
            },
        ),
        (
            encoder,
            {
                "hidden_size": (16, 2048),
                "intermediate_size": (16, 8192),
                "num_hidden_layers": (1, 48),
                "num_attention_heads": (1, 32),
                "num_key_value_heads": (1, 32),
                "subsampling_conv_channels": (8, 512),
                "max_position_embeddings": (64, 8192),
            },
        ),
    ):
        for key, (low, high) in limits.items():
            if type(part.get(key)) is not int or not low <= part[key] <= high:
                raise ValueError("RNNT_GEOMETRY")
    fixed = {
        "hidden_act": "relu",
        "is_encoder_decoder": True,
        "durations": [],
        "max_symbols_per_step": 10,
        "num_prompts": 128,
        "default_prompt_id": 101,
    }
    enc_fixed = {
        "hidden_act": "silu",
        "attention_bias": False,
        "convolution_bias": False,
        "num_mel_bins": 128,
        "conv_kernel_size": 9,
        "scale_input": False,
        "subsampling_conv_kernel_size": 3,
        "subsampling_conv_stride": 2,
        "subsampling_factor": 8,
        "sliding_window": 57,
        "supported_num_lookahead_tokens": [3, 0, 6, 13],
        "default_num_lookahead_tokens": 3,
    }
    if (
        any(config.get(k) != v for k, v in fixed.items())
        or any(encoder.get(k) != v for k, v in enc_fixed.items())
        or encoder["num_key_value_heads"] != encoder["num_attention_heads"]
        or encoder["hidden_size"] % encoder["num_attention_heads"]
        or encoder["hidden_size"] % 2
        or config.get("blank_token_id") != config["vocab_size"] - 1
        or type(config.get("pad_token_id")) is not int
        or not 0 <= config["pad_token_id"] < config["vocab_size"] - 1
    ):
        raise ValueError("RNNT_GRAPH")
    for key in (
        "dropout",
        "attention_dropout",
        "activation_dropout",
        "layerdrop",
        "dropout_positions",
    ):
        if type(encoder.get(key)) not in {int, float} or not 0 <= encoder[key] <= 0.5:
            raise ValueError("RNNT_DROPOUT")
    return config


def validate_metadata(name, metadata, config):
    validate_model_config(metadata)
    if name == "processor_config.json":
        if (
            set(metadata)
            != {
                "blank_token",
                "default_num_lookahead_tokens",
                "feature_extractor",
                "num_prompts",
                "processor_class",
                "prompt_dictionary",
                "supported_num_lookahead_tokens",
            }
            or metadata["processor_class"] != "Nemotron3_5AsrProcessor"
            or metadata["blank_token"] != "<blank>"
            or metadata["feature_extractor"] != FEATURES
            or metadata["num_prompts"] != 128
            or metadata["default_num_lookahead_tokens"] != 3
            or metadata["supported_num_lookahead_tokens"] != [3, 0, 6, 13]
        ):
            raise ValueError("RNNT_PROCESSOR")
        prompts = metadata["prompt_dictionary"]
        if (
            not isinstance(prompts, dict)
            or not 1 <= len(prompts) <= 256
            or prompts.get("auto") != config["default_prompt_id"]
            or any(
                not isinstance(k, str)
                or not 1 <= len(k) <= 32
                or type(v) is not int
                or not 0 <= v < config["num_prompts"]
                for k, v in prompts.items()
            )
        ):
            raise ValueError("RNNT_PROMPTS")
    elif name == "tokenizer_config.json":
        if (
            metadata.get("tokenizer_class") != "ParakeetTokenizer"
            or metadata.get("processor_class") != "Nemotron3_5AsrProcessor"
            or metadata.get("backend") != "tokenizers"
            or metadata.get("pad_token") != "<pad>"
            or metadata.get("unk_token") != "<unk>"
            or metadata.get("clean_up_tokenization_spaces") is not False
            or not isinstance(metadata.get("extra_special_tokens"), list)
            or len(metadata["extra_special_tokens"]) > 256
            or any(
                not isinstance(t, str) or not 1 <= len(t) <= 64
                for t in metadata["extra_special_tokens"]
            )
        ):
            raise ValueError("RNNT_TOKENIZER")
    else:
        raise ValueError("RNNT_METADATA")


def components(root, config):
    from transformers import NemotronAsrStreamingFeatureExtractor, ParakeetTokenizer

    for name in METADATA_NAMES:
        validate_metadata(name, metadata_file(root / name), config)
    metadata = metadata_file(root / "tokenizer_config.json")
    path = root / "tokenizer.json"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 4 * 1024**2:
        raise ValueError("RNNT_TOKENIZER_FILE")
    tokenizer = ParakeetTokenizer(
        tokenizer_file=str(path),
        pad_token="<pad>",
        unk_token="<unk>",
        extra_special_tokens=metadata["extra_special_tokens"],
        clean_up_tokenization_spaces=False,
    )
    tokenizer.chat_template = None
    vocab = tokenizer.get_vocab()
    aligned = (
        len(vocab) == config["vocab_size"]
        and tokenizer.pad_token_id == config["pad_token_id"]
        and vocab.get("<blank>") == config["blank_token_id"]
    )
    # The original export appends a tokenizer-only <blank> after its output head.
    # Its numeric blank row is named <pad>; its pad row is <unk>. Audio input
    # needs no text encoding. Retain all tokenizer rows, drive the decoder by
    # model IDs and remove those numeric control rows when decoding the result.
    terminal_blank = (
        len(vocab) == config["vocab_size"] + 1
        and tokenizer.pad_token_id == config["blank_token_id"]
        and vocab.get("<blank>") == config["vocab_size"]
        and tokenizer.unk_token_id == config["pad_token_id"]
    )
    if (
        not (aligned or terminal_blank)
        or set(vocab.values()) != set(range(len(vocab)))
        or tokenizer.encode("<blank>", add_special_tokens=False) != [vocab.get("<blank>")]
    ):
        raise ValueError("RNNT_VOCABULARY")
    return NemotronAsrStreamingFeatureExtractor(
        **{k: v for k, v in FEATURES.items() if k != "feature_extractor_type"}
    ), tokenizer


def native_model(config):
    from transformers import Nemotron3_5AsrConfig, Nemotron3_5AsrForRNNT

    native = Nemotron3_5AsrConfig.from_dict(copy.deepcopy(config))
    native._attn_implementation = "sdpa"
    native.encoder_config._attn_implementation = "sdpa"
    return Nemotron3_5AsrForRNNT(native)


def load_components(root, config, backend):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    validate_config(config)
    processor = components(root, config)
    shapes = preflight_safetensors(root)
    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    expected = model.state_dict()
    if shapes != {k: tuple(v.shape) for k, v in expected.items()}:
        raise ValueError("RNNT_WEIGHTS_INCOMPLETE")
    weights = {}
    for path in sorted(root.glob("*.safetensors")):
        values = load_file(path, device="cpu")
        if weights.keys() & values.keys():
            raise ValueError("RNNT_WEIGHTS_DUPLICATE")
        for key, value in values.items():
            if expected[key].is_floating_point():
                if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                    raise ValueError("RNNT_WEIGHTS_NUMERIC")
                value = value.float()
            elif value.dtype != expected[key].dtype or bool((value < 0).any()):
                raise ValueError("RNNT_WEIGHTS_NUMERIC")
            weights[key] = value
    model.load_state_dict(weights, assign=True, strict=True)
    model.eval().to(backend)
    return model, processor, config


def prepare_inputs(feature, audio, config, backend):
    import numpy as np
    import torch

    audio = np.asarray(audio)
    if audio.ndim != 1 or not 512 <= audio.size <= 30 * 16000 or not np.isfinite(audio).all():
        raise ValueError("RNNT_AUDIO_BOUND")
    inputs = dict(feature([audio], sampling_rate=16000, center=True, return_tensors="pt"))
    if (
        set(inputs) != {"input_features", "attention_mask"}
        or inputs["input_features"].ndim != 3
        or inputs["input_features"].shape[0] != 1
        or inputs["input_features"].shape[2] != 128
        or not 1 <= inputs["input_features"].shape[1] <= 3008
        or not bool(torch.isfinite(inputs["input_features"]).all())
    ):
        raise ValueError("RNNT_FEATURES")
    inputs = {k: v.to(backend) for k, v in inputs.items()}
    inputs.update(
        prompt_ids=torch.tensor([config["default_prompt_id"]], device=backend),
        num_lookahead_tokens=3,
    )
    return inputs


def transcribe(model, processor, config, audio, backend):
    import torch
    from transformers import GenerationConfig

    feature, tokenizer = processor
    inputs = prepare_inputs(feature, audio, config, backend)
    frames = int(
        model.encoder._get_subsampling_output_length(inputs["attention_mask"].sum(-1)).item()
    )
    limit = frames * config["max_symbols_per_step"]
    if not 1 <= limit <= MAX_OUTPUT_TOKENS:
        raise ValueError("RNNT_TOKEN_BOUND")
    generation = GenerationConfig(
        do_sample=False,
        num_beams=1,
        use_cache=True,
        return_dict_in_generate=True,
        pad_token_id=config["pad_token_id"],
        decoder_start_token_id=config["blank_token_id"],
        bos_token_id=None,
        eos_token_id=None,
        max_new_tokens=limit,
    )
    # No repository generation defaults can alter greedy decoding or its bound.
    model.generation_config = generation

    def check_logits(_module, _arguments, output):
        if not bool(torch.isfinite(output.logits).all()):
            raise ValueError("RNNT_LOGITS_NONFINITE")

    hook = model.register_forward_hook(check_logits)
    try:
        with torch.inference_mode():
            output = model.generate(**inputs, generation_config=generation)
    finally:
        hook.remove()
    return decode_output(output, tokenizer, config, frames, backend)


def decode_output(output, tokenizer, config, frames, backend):
    import torch

    limit = frames * config["max_symbols_per_step"]
    ids = output.sequences
    durations = output.durations
    if (
        ids.ndim != 2
        or ids.shape[0] != 1
        or not 1 <= ids.shape[1] <= limit + 1
        or ids.dtype != torch.int64
        or durations.ndim != 2
        or durations.shape[0] != 1
        or durations.dtype != torch.int64
        or bool((ids < 0).any())
        or bool((ids >= config["vocab_size"]).any())
        or bool((durations < 0).any())
        or bool((durations > 1).any())
    ):
        raise ValueError("RNNT_DECODING_INCOMPLETE")
    # Transformers 5.17's MPS DeferredStopCheck rolls back one speculative
    # token, but Parakeet's separate duration list retains that step. Reconcile
    # that single sidecar entry only; validate the retained frame path below.
    if backend == "mps" and durations.shape[1] == ids.shape[1] + 1:
        durations = durations[:, :-1]
    tokens = ids[0].tolist()
    expected, symbols = [0], 0
    for token in tokens[1:]:
        symbols = 0 if token == config["blank_token_id"] else symbols + 1
        advance = token == config["blank_token_id"] or symbols == config["max_symbols_per_step"]
        expected.append(int(advance))
        if advance:
            symbols = 0
    if (
        durations.shape != ids.shape
        or tokens[0] != config["blank_token_id"]
        or durations[0].tolist() != expected
        or sum(expected) != frames
        or expected[-1] != 1
    ):
        raise ValueError("RNNT_DECODING_INCOMPLETE")
    tokens = [i for i in tokens if i not in {config["pad_token_id"], config["blank_token_id"]}]
    text = tokenizer.decode(
        tokens, group_tokens=False, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    if len(text.encode()) > 16384:
        raise ValueError("RNNT_OUTPUT_BOUND")
    return {"text": text, "language": None}


def execute(model, processor, config, value, backend, input_path, emit):
    from oryn_engine.vad_tasks import read_audio

    if value.get("timestamps", False) or value.get("language", "auto") != "auto":
        raise ValueError("RNNT_PARAMETERS")
    audio, _ = read_audio(input_path)
    emit("job.progress", {"stage": "transcribing", "progress": 0.3})
    return transcribe(model, processor, config, audio, backend)
