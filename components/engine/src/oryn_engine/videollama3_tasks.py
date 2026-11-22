# Preprocessing geometry follows VideoLLaMA3 (Apache-2.0).
# Copyright 2024 The Qwen team, Alibaba Group and HuggingFace Inc.
# See videollama3_assets/LICENSE and provenance.json.
"""Original VideoLLaMA3 exports bound to complete native Qwen2/NaViT graphs.

Repository Python and chat templates are never executed. The owned video profile
uses eight bounded, uniformly sampled frames and the publisher's patch order,
bilinear spatial merge, temporal compression and single-turn token framing.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from oryn_engine.model_metadata import (
    MAX_JSON_BYTES,
    metadata_file,
    parse_metadata,
    preflight_safetensors,
    validate_model_config,
)

FAMILY = "videollama3_qwen2"
ARCHITECTURE = "Videollama3Qwen2ForCausalLM"
TASK = "video-text-to-text"
DATA = Path(__file__).parent / "videollama3_assets"
METADATA_NAMES = (
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)
REQUIRED_FILES = ("config.json", "vocab.json", "merges.txt", *METADATA_NAMES)
EMBEDDING = "model.language_model.embed_tokens.weight"


def template(name):
    return json.loads((DATA / name).read_text())


def supports(config, task):
    return task == TASK and config.get("model_type") == FAMILY


def closed(value, expected, flexible=()):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("VIDEOLLAMA3_CONFIGURATION")
    for k, v in expected.items():
        if k not in flexible and (type(value[k]) is not type(v) or value[k] != v):
            raise ValueError("VIDEOLLAMA3_CONFIGURATION:" + k)


def normalized_config(config, task):
    if not supports(config, task):
        return config
    c = copy.deepcopy(config)
    if "auto_map" in c and c.pop("auto_map") != template("config.json")["auto_map"]:
        raise ValueError("VIDEOLLAMA3_REMOTE_MAPPING")
    validate_config(c)
    return c


def validate_config(c):
    validate_model_config(c)
    expected = template("config.json")
    del expected["auto_map"]
    dimensions = {
        "hidden_size": (16, 3584),
        "intermediate_size": (32, 18944),
        "num_hidden_layers": (1, 28),
        "num_attention_heads": (1, 28),
        "num_key_value_heads": (1, 4),
        "max_window_layers": (1, 28),
        "vocab_size": (281, 152064),
        "max_position_embeddings": (256, 32768),
        "bos_token_id": (256, 151643),
        "eos_token_id": (258, 151645),
        "image_token_index": (278, 151665),
    }
    closed(c, expected, {*dimensions, "tie_word_embeddings", "use_cache", "vision_encoder_config"})
    for k, (low, high) in dimensions.items():
        if type(c[k]) is not int or not low <= c[k] <= high:
            raise ValueError("VIDEOLLAMA3_TEXT_GEOMETRY")
    v = c["vision_encoder_config"]
    vision = {
        "hidden_size": (16, 1152),
        "intermediate_size": (32, 4304),
        "num_hidden_layers": (1, 27),
        "num_attention_heads": (1, 16),
    }
    closed(v, expected["vision_encoder_config"], vision)
    if any(type(v[k]) is not int or not low <= v[k] <= high for k, (low, high) in vision.items()):
        raise ValueError("VIDEOLLAMA3_VISION_GEOMETRY")
    if (
        type(c["tie_word_embeddings"]) is not bool
        or type(c["use_cache"]) is not bool
        or c["hidden_size"] % c["num_attention_heads"]
        or c["num_attention_heads"] % c["num_key_value_heads"]
        or (c["hidden_size"] // c["num_attention_heads"]) % 2
        or v["hidden_size"] % v["num_attention_heads"]
        or (v["hidden_size"] // v["num_attention_heads"]) % 4
        or c["bos_token_id"] + 2 != c["eos_token_id"]
        or c["bos_token_id"] + 22 != c["image_token_index"]
        or c["image_token_index"] + 2 >= c["vocab_size"]
    ):
        raise ValueError("VIDEOLLAMA3_GEOMETRY")
    return c


def validate_metadata(name, value, config):
    if name not in METADATA_NAMES:
        raise ValueError("VIDEOLLAMA3_METADATA")
    expected = template(name)
    offset = config["bos_token_id"] - 151643
    if name == "added_tokens.json":
        expected = {k: v + offset for k, v in expected.items()}
    if name == "tokenizer_config.json":
        expected["added_tokens_decoder"] = {
            str(int(k) + offset): v for k, v in expected["added_tokens_decoder"].items()
        }
    closed(value, expected)


def native_config(c):
    from transformers import Qwen2Config, VideoLlama3Config, VideoLlama3VisionConfig

    validate_config(c)
    excluded = {
        "architectures",
        "image_token_index",
        "mm_projector_type",
        "model_type",
        "vision_encoder",
        "vision_encoder_config",
        "use_token_compression",
        "transformers_version",
        "torch_dtype",
    }
    text = Qwen2Config(**{k: v for k, v in c.items() if k not in excluded})
    vision = VideoLlama3VisionConfig(
        **{k: v for k, v in c["vision_encoder_config"].items() if k != "model_type"}
    )
    result = VideoLlama3Config(
        text_config=text,
        vision_config=vision,
        image_token_id=c["image_token_index"],
        video_token_id=c["image_token_index"],
        tie_word_embeddings=c["tie_word_embeddings"],
    )
    for part in (result, text, vision):
        part._attn_implementation = "eager"
    return result


def native_key(k):
    for old, new in (
        ("model.vision_encoder.", "model.vision_model."),
        ("model.mm_projector.", "model.projector."),
        ("model.", "model.language_model."),
    ):
        if k.startswith(old):
            return new + k[len(old) :]
    return k


def validate_weight_shapes(config, shapes):
    import torch
    from transformers import VideoLlama3ForConditionalGeneration

    with torch.device("meta"):
        model = VideoLlama3ForConditionalGeneration(native_config(config))
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    actual = {native_key(k): tuple(v) for k, v in shapes.items()}
    if len(actual) != len(shapes):
        raise ValueError("VIDEOLLAMA3_WEIGHT_ALIAS")
    if config["tie_word_embeddings"] and "lm_head.weight" not in actual and EMBEDDING in actual:
        actual["lm_head.weight"] = actual[EMBEDDING]
    if actual != expected:
        raise ValueError("VIDEOLLAMA3_WEIGHTS_INCOMPLETE")


def finite(value):
    import torch

    if not bool(torch.isfinite(value).all()):
        raise ValueError("VIDEOLLAMA3_NONFINITE")
    return value


def tokenizer(root, config):
    from oryn_engine.native_qwen_bpe import load_qwen_bpe

    # Match the publisher's complete, closed declaration without dispatching its
    # repository tokenizer. Only the validated token table reaches the native BPE.
    path = root / "tokenizer_config.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("VIDEOLLAMA3_TOKENIZER_METADATA")
    with path.open("rb") as stream:
        tokenizer_metadata = parse_metadata(stream.read(MAX_JSON_BYTES + 1), redundant_keys=True)
    for name in METADATA_NAMES:
        value = (
            tokenizer_metadata if name == "tokenizer_config.json" else metadata_file(root / name)
        )
        validate_metadata(name, value, config)
    return load_qwen_bpe(
        root,
        config["bos_token_id"],
        config["vocab_size"],
        tokenizer_metadata["added_tokens_decoder"],
    )


def load_components(root, config, backend):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import VideoLlama3ForConditionalGeneration

    if backend not in {"cpu", "mps"} or root.is_symlink():
        raise ValueError("VIDEOLLAMA3_SCOPE")
    for name in REQUIRED_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024**2:
            raise ValueError("VIDEOLLAMA3_EXPORT_INCOMPLETE")
    validate_weight_shapes(config, preflight_safetensors(root))
    processor = tokenizer(root, config)
    state = {}
    for path in sorted(root.glob("*.safetensors")):
        for k, v in load_file(path).items():
            if v.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
                raise ValueError("VIDEOLLAMA3_NUMERIC_WEIGHT")
            state[native_key(k)] = finite(v).float()
    if config["tie_word_embeddings"]:
        if "lm_head.weight" in state and not torch.equal(state["lm_head.weight"], state[EMBEDDING]):
            raise ValueError("VIDEOLLAMA3_TIED_WEIGHT")
        state["lm_head.weight"] = state[EMBEDDING]
    with init_empty_weights(include_buffers=False):
        model = VideoLlama3ForConditionalGeneration(native_config(config))
    model.load_state_dict(state, strict=True, assign=True)
    model.lm_head.register_forward_hook(lambda module, args, output: finite(output))
    return model.to(backend).eval(), processor, config


def video_pixels(frames):
    import numpy as np
    import torch
    from transformers.image_transforms import normalize, rescale, resize

    if len(frames) != 8 or any(f.mode != "RGB" or f.size != frames[0].size for f in frames):
        raise ValueError("VIDEOLLAMA3_FRAMES")
    width, height = frames[0].size
    if not min(width, height) >= 1 or max(width, height) > 512:
        raise ValueError("VIDEOLLAMA3_FRAME_SIZE")
    h, w = round(height / 28) * 28, round(width / 28) * 28
    if h * w < 16 * 28**2:
        beta = math.sqrt(16 * 28**2 / (height * width))
        h, w = math.ceil(height * beta / 28) * 28, math.ceil(width * beta / 28) * 28
    if not 1 <= h * w <= 512**2 or 8 * h * w // 28**2 > 3000:
        raise ValueError("VIDEOLLAMA3_PIXEL_BUDGET")
    patches = []
    for frame in frames:
        value = resize(np.asarray(frame), (h, w), resample=3, input_data_format="channels_last")
        value = rescale(value, 1 / 255, input_data_format="channels_last")
        value = normalize(value, [0.5] * 3, [0.5] * 3, input_data_format="channels_last")
        patches.append(value.transpose(2, 0, 1))
    gh, gw = h // 14, w // 14
    pixels = np.asarray(patches).reshape(8, 3, gh // 2, 2, 14, gw // 2, 2, 14)
    pixels = pixels.transpose(0, 2, 5, 3, 6, 1, 4, 7).reshape(8 * gh * gw, 3 * 14**2)
    return torch.from_numpy(pixels), torch.tensor([[8, gh, gw]]), torch.tensor([2])


def compression_mask(pixels, grid):
    import torch

    t, h, w = grid[0].tolist()
    values = finite(pixels).reshape(t, h * w // 4, -1)
    mask = torch.ones(values.shape[:2], dtype=torch.bool)
    mask[1:] = (values[1:] - values[:-1]).abs().mean(-1) * 255 > 0.1
    mask[mask.sum(1) == 0, 0] = True
    return mask.flatten()


def prepare_inputs(processor, config, frames, times, question, max_tokens):
    import torch

    if (
        not isinstance(question, str)
        or not question.strip()
        or len(question) > 4096
        or any(word in question for word in template("added_tokens.json"))
        or len(times) != 8
        or any(
            type(t) not in {int, float} or not math.isfinite(t) or not 0 <= t <= 30 for t in times
        )
        or any(a > b for a, b in zip(times[:-1], times[1:], strict=True))
        or type(max_tokens) is not int
        or not 1 <= max_tokens <= 256
    ):
        raise ValueError("VIDEOLLAMA3_INPUT")
    pixels, grid, merge = video_pixels(frames)
    mask = compression_mask(pixels, grid)
    per_frame = int(grid[0, 1] * grid[0, 2]) // 4
    # Publisher template with add_system_prompt=False (its default). Build data
    # directly; no repository-supplied Jinja or processor code is interpreted.
    content = "".join(
        f"Time {round(t, 1)}s:" + "<image>" * per_frame + ("," if i < 7 else "\n")
        for i, t in enumerate(times)
    )
    prompt = f"<|im_start|>user\n{content}{question}<|im_end|>\n<|im_start|>assistant\n"
    ids = processor(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if int((ids == config["image_token_index"]).sum()) != len(mask):
        raise ValueError("VIDEOLLAMA3_IMAGE_TOKENS")
    select = torch.ones_like(ids, dtype=torch.bool)
    select[ids == config["image_token_index"]] = mask
    ids = ids[select][None]
    if ids.shape[-1] + max_tokens > min(4096, config["max_position_embeddings"]):
        raise ValueError("VIDEOLLAMA3_TOKEN_BUDGET")
    return dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        pixel_values_videos=pixels,
        video_grid_thw=grid,
        video_merge_sizes=merge,
        video_compression_mask=mask,
    )


def execute(model, processor, config, value, backend, path):
    from transformers import GenerationConfig

    from oryn_engine.coverage_tasks import sample_video
    from oryn_engine.shared_tasks import _move

    frames, times, duration = sample_video(
        path, 8, value["startSeconds"], value.get("durationSeconds")
    )
    inputs = _move(
        prepare_inputs(processor, config, frames, times, value["question"], value["maxTokens"]),
        backend,
    )
    generated = model.generate(
        **inputs,
        generation_config=GenerationConfig(
            max_new_tokens=value["maxTokens"],
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=config["eos_token_id"],
            pad_token_id=config["bos_token_id"],
        ),
    )
    tokens = generated[0, inputs["input_ids"].shape[-1] :].tolist()
    if not tokens or tokens[-1] != config["eos_token_id"]:
        raise ValueError("VIDEOLLAMA3_OUTPUT_INCOMPLETE")
    answer = processor.decode(tokens, skip_special_tokens=True).strip()
    if not answer or len(answer.encode()) > 16384:
        raise ValueError("VIDEOLLAMA3_OUTPUT_INVALID")
    return dict(text=answer, sampledTimestamps=times, durationSeconds=duration)
