"""Complete original LLaVA-Video Qwen2 export in the shared video environment.

The publisher exports a 26-layer SigLIP tower, its retained post norm, the full
projector and a learned grid-row newline. No external base vision model is needed.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config
from oryn_engine.videollama3_tasks import closed, finite

FAMILY = "oryn_llava_video_qwen2"
ARCHITECTURE = "LlavaQwenForCausalLM"
TASK = "video-text-to-text"
DATA = Path(__file__).parent / "llava_video_qwen_assets"
METADATA_NAMES = (
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)
REQUIRED_FILES = ("config.json", "vocab.json", "merges.txt", *METADATA_NAMES)
VISION = dict(
    hidden_size=1152,
    intermediate_size=4304,
    num_hidden_layers=26,
    num_attention_heads=16,
    image_size=384,
    patch_size=14,
    hidden_act="gelu_pytorch_tanh",
    layer_norm_eps=1e-6,
    attention_dropout=0.0,
    num_channels=3,
    vision_use_head=False,
)


def template(name):
    return json.loads((DATA / name).read_text())


def supports(c, task):
    return task == TASK and (
        c.get("model_type") == FAMILY
        or c.get("model_type") == "llava"
        and c.get("architectures") == [ARCHITECTURE]
    )


def normalized_config(config, task):
    if not supports(config, task):
        return config
    c = copy.deepcopy(config)
    c["model_type"] = FAMILY
    # Make the source wrapper's implicit SigLIP defaults explicit in the owned
    # layout. The last layer and pooling head are absent in the original export.
    c.setdefault("vision_config", dict(VISION))
    validate_config(c)
    return c


def validate_config(c):
    validate_model_config(c)
    expected = template("config.json") | {"model_type": FAMILY, "vision_config": VISION}
    bounds = {
        "hidden_size": (16, 3584),
        "intermediate_size": (32, 18944),
        "num_hidden_layers": (1, 28),
        "num_attention_heads": (1, 28),
        "num_key_value_heads": (1, 4),
        "vocab_size": (260, 152064),
        "max_position_embeddings": (256, 32768),
        "tokenizer_model_max_length": (256, 32768),
        "bos_token_id": (256, 151643),
        "eos_token_id": (258, 151645),
        "image_token_index": (259, 151646),
        "mm_hidden_size": (16, 1152),
    }
    closed(c, expected, {*bounds, "vision_config"})
    for k, (low, high) in bounds.items():
        if type(c[k]) is not int or not low <= c[k] <= high:
            raise ValueError("LLAVA_VIDEO_QWEN_GEOMETRY")
    vision_bounds = {
        "hidden_size": (16, 1152),
        "intermediate_size": (32, 4304),
        "num_hidden_layers": (1, 26),
        "num_attention_heads": (1, 16),
        "image_size": (28, 384),
    }
    v = c["vision_config"]
    closed(v, VISION, vision_bounds)
    if (
        any(
            type(v[k]) is not int or not low <= v[k] <= high
            for k, (low, high) in vision_bounds.items()
        )
        or c["mm_hidden_size"] != v["hidden_size"]
        or c["hidden_size"] % c["num_attention_heads"]
        or c["num_attention_heads"] % c["num_key_value_heads"]
        or c["hidden_size"] // c["num_attention_heads"] % 2
        or v["hidden_size"] % v["num_attention_heads"]
        or c["bos_token_id"] + 2 != c["eos_token_id"]
        or c["bos_token_id"] + 3 != c["image_token_index"]
        or c["image_token_index"] >= c["vocab_size"]
    ):
        raise ValueError("LLAVA_VIDEO_QWEN_GEOMETRY")
    return c


def validate_metadata(name, value, config):
    if name not in METADATA_NAMES:
        raise ValueError("LLAVA_VIDEO_QWEN_METADATA")
    expected = template(name)
    offset = config["bos_token_id"] - 151643
    if name == "added_tokens.json":
        expected = {k: v + offset for k, v in expected.items()}
    if name == "tokenizer_config.json":
        expected["added_tokens_decoder"] = {
            str(int(k) + offset): v for k, v in expected["added_tokens_decoder"].items()
        }
    if name == "preprocessor_config.json":
        expected["size"] = dict(
            height=config["vision_config"]["image_size"],
            width=config["vision_config"]["image_size"],
        )
    closed(value, expected)


def native_model(c):
    import torch
    from torch import nn
    from transformers import Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel

    validate_config(c)
    keys = {
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "bos_token_id",
        "eos_token_id",
        "attention_dropout",
        "hidden_act",
        "initializer_range",
        "rms_norm_eps",
        "rope_scaling",
        "rope_theta",
        "tie_word_embeddings",
        "use_sliding_window",
        "use_cache",
    }
    text = Qwen2Config(**{k: c[k] for k in keys})
    vision = SiglipVisionConfig(**c["vision_config"])
    text._attn_implementation = vision._attn_implementation = "eager"
    model = Qwen2ForCausalLM(text)
    model.model.vision_tower = nn.Module()
    model.model.vision_tower.vision_tower = SiglipVisionModel(vision)
    model.model.mm_projector = nn.Sequential(
        nn.Linear(vision.hidden_size, text.hidden_size),
        nn.GELU(),
        nn.Linear(text.hidden_size, text.hidden_size),
    )
    model.model.image_newline = nn.Parameter(torch.zeros(text.hidden_size))
    model.lm_head.register_forward_hook(lambda module, args, output: finite(output))
    return model.eval()


def native_key(k):
    prefix = "model.vision_tower.vision_tower.vision_model."
    return "model.vision_tower.vision_tower." + k[len(prefix) :] if k.startswith(prefix) else k


def validate_weight_shapes(c, shapes):
    import torch

    with torch.device("meta"):
        model = native_model(c)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    actual = {native_key(k): tuple(v) for k, v in shapes.items()}
    if len(actual) != len(shapes) or expected != actual:
        raise ValueError("LLAVA_VIDEO_QWEN_WEIGHTS_INCOMPLETE")


def load_components(root, c, backend):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    from oryn_engine.native_qwen_bpe import load_qwen_bpe

    if backend not in {"cpu", "mps"} or root.is_symlink():
        raise ValueError("LLAVA_VIDEO_QWEN_SCOPE")
    for name in REQUIRED_FILES:
        p = root / name
        if p.is_symlink() or not p.is_file() or p.stat().st_size > 4 * 1024**2:
            raise ValueError("LLAVA_VIDEO_QWEN_EXPORT_INCOMPLETE")
    validate_weight_shapes(c, preflight_safetensors(root))
    for name in METADATA_NAMES:
        validate_metadata(name, metadata_file(root / name), c)
    processor = load_qwen_bpe(
        root,
        c["bos_token_id"],
        c["vocab_size"],
        metadata_file(root / "tokenizer_config.json")["added_tokens_decoder"],
    )
    state = {}
    for path in sorted(root.glob("*.safetensors")):
        for k, v in load_file(path).items():
            if v.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
                raise ValueError("LLAVA_VIDEO_QWEN_NUMERIC_WEIGHT")
            state[native_key(k)] = finite(v).float()
    with init_empty_weights(include_buffers=False):
        model = native_model(c)
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(backend).eval(), processor, c


def video_pixels(c, frames):
    import numpy as np
    import torch
    from transformers.image_transforms import normalize, rescale, resize

    if len(frames) != 8 or any(
        f.mode != "RGB" or not min(f.size) >= 1 or max(f.size) > 512 for f in frames
    ):
        raise ValueError("LLAVA_VIDEO_QWEN_FRAMES")
    size = c["vision_config"]["image_size"]
    result = []
    for frame in frames:
        value = resize(
            np.asarray(frame), (size, size), resample=3, input_data_format="channels_last"
        )
        value = rescale(value, 1 / 255, input_data_format="channels_last")
        value = normalize(value, [0.5] * 3, [0.5] * 3, input_data_format="channels_last")
        result.append(value.transpose(2, 0, 1))
    return torch.from_numpy(np.asarray(result))


def visual_features(model, c, pixels):
    import torch
    from torch.nn.functional import interpolate

    # Original wrapper uses the encoder's final hidden state BEFORE post norm;
    # the retained post norm is still required and validated in the checkpoint.
    values = model.model.vision_tower.vision_tower(pixels, output_hidden_states=True).hidden_states[
        -1
    ]
    values = finite(model.model.mm_projector(finite(values)))
    side = c["vision_config"]["image_size"] // 14
    width = c["hidden_size"]
    if tuple(values.shape) != (8, side**2, width):
        raise ValueError("LLAVA_VIDEO_QWEN_FEATURES")
    pooled = interpolate(
        values.reshape(8, side, side, width).permute(0, 3, 1, 2),
        size=(math.ceil(side / 2),) * 2,
        mode="bilinear",
        align_corners=False,
    )
    pooled = pooled.permute(0, 2, 3, 1)
    newline = model.model.image_newline.expand(8, pooled.shape[1], 1, width)
    return finite(torch.cat((pooled, newline), dim=2).reshape(-1, width))


def prompt_tokens(processor, c, times, duration, question, maximum):
    if (
        len(times) != 8
        or any(
            type(t) not in {int, float} or not math.isfinite(t) or not 0 <= t <= 30 for t in times
        )
        or any(a > b for a, b in zip(times[:-1], times[1:], strict=True))
        or type(duration) not in {int, float}
        or not math.isfinite(duration)
        or not times[-1] <= duration <= 30.000001
        or not isinstance(question, str)
        or not question.strip()
        or len(question) > 4096
        or any(token in question for token in template("added_tokens.json"))
        or type(maximum) is not int
        or not 1 <= maximum <= 256
    ):
        raise ValueError("LLAVA_VIDEO_QWEN_INPUT")
    prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    time_string = ",".join(f"{t:.2f}s" for t in times)
    suffix = (
        f"\nThe video lasts for {duration:.2f} seconds, "
        "and 8 frames are uniformly sampled from it. "
        f"These frames are located at {time_string}.Please answer the following questions "
        f"related to this video.\n{question}<|im_end|>\n<|im_start|>assistant\n"
    )
    parts = [processor.encode(s, add_special_tokens=False) for s in (prefix, suffix)]
    side = math.ceil((c["vision_config"]["image_size"] // 14) / 2)
    if sum(map(len, parts)) + 8 * side * (side + 1) + maximum > min(
        4096, c["max_position_embeddings"], c["tokenizer_model_max_length"]
    ):
        raise ValueError("LLAVA_VIDEO_QWEN_TOKEN_BUDGET")
    return parts


def execute(model, processor, config, value, backend, path):
    import torch
    from transformers import GenerationConfig

    from oryn_engine.coverage_tasks import sample_video

    frames, times, duration = sample_video(
        path, 8, value["startSeconds"], value.get("durationSeconds")
    )
    parts = prompt_tokens(processor, config, times, duration, value["question"], value["maxTokens"])
    pixels = video_pixels(config, frames).to(backend)
    visual = visual_features(model, config, pixels)
    text = [model.model.embed_tokens(torch.tensor(part, device=backend)) for part in parts]
    embeddings = finite(torch.cat((text[0], visual, text[1])))[None]
    output = model.generate(
        inputs_embeds=embeddings,
        attention_mask=torch.ones(embeddings.shape[:2], device=backend, dtype=torch.long),
        generation_config=GenerationConfig(
            max_new_tokens=value["maxTokens"],
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=config["eos_token_id"],
            pad_token_id=config["bos_token_id"],
        ),
    )
    tokens = output[0].tolist()
    if not tokens or tokens[-1] != config["eos_token_id"]:
        raise ValueError("LLAVA_VIDEO_QWEN_OUTPUT_INCOMPLETE")
    answer = processor.decode(tokens, skip_special_tokens=True).strip()
    if not answer or len(answer.encode()) > 16384:
        raise ValueError("LLAVA_VIDEO_QWEN_OUTPUT_INVALID")
    return dict(text=answer, sampledTimestamps=times, durationSeconds=duration)
