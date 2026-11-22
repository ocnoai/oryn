"""Owned MiniMax H3 text-to-video/audio workflow over complete native components."""

from __future__ import annotations

import copy
import math

from oryn_engine import modular_components as bundle
from oryn_engine.model_metadata import metadata_file, validate_model_config

FAMILY = "MiniMaxH3ModularPipeline"
INDEX = "model_index.json"
CONTRACT = "text-to-video-v1"
IMAGE_CONTRACT = "image-text-to-video-v1"
CONTRACTS = {CONTRACT, IMAGE_CONTRACT}
CLASSES = {
    "text_encoder": ("transformers", "Qwen3VLForConditionalGeneration"),
    "transformer": ("diffusers", "MiniMaxH3Transformer3DModel"),
    "transformer_ref": ("diffusers", "MiniMaxH3Transformer3DModel"),
    "vae": ("diffusers", "AutoencoderKLMiniMaxH3"),
    "audio_vae": ("diffusers", "AutoencoderKLMiniMaxH3Audio"),
    "scheduler": ("diffusers", "MiniMaxH3Scheduler"),
    "audio_scheduler": ("diffusers", "MiniMaxH3Scheduler"),
    "tokenizer": ("transformers", "Qwen2TokenizerFast"),
    "processor": ("transformers", "Qwen3VLProcessor"),
}
# Native get_workflow('t2va') excludes transformer_ref: it is an alternative
# checkpoint partition, not a dropped branch of the generated video/audio graph.
WEIGHT_COMPONENTS = ("text_encoder", "transformer", "vae", "audio_vae")
CONFIGS = {k: f"{k}/config.json" for k in WEIGHT_COMPONENTS} | {
    "scheduler": "scheduler/scheduler_config.json",
    "audio_scheduler": "audio_scheduler/scheduler_config.json",
    "tokenizer": "tokenizer/tokenizer_config.json",
    "processor": "processor/preprocessor_config.json",
    "video_processor": "processor/video_preprocessor_config.json",
}
CONTROL_IDS = {
    "<|endoftext|>": 151643,
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|vision_pad|>": 151654,
    "<|image_pad|>": 151655,
    "<|video_pad|>": 151656,
}
FPS, SAMPLE_RATE = 24, 32000
FRAME_COUNTS = tuple(range(124, 346, 17))


def supports(index):
    return index.get("_class_name") == FAMILY


def validate_index(index):
    bundle.validate_index(index, family=FAMILY, blocks="MiniMaxH3Blocks", classes=CLASSES)


def selected_files(files, index):
    validate_index(index)
    return bundle.select_files(
        files,
        required={INDEX, *CONFIGS.values(), "tokenizer/tokenizer.json"},
        weight_stems={
            k: "model" if k == "text_encoder" else "diffusion_pytorch_model"
            for k in WEIGHT_COMPONENTS
        },
    )


def _integer(c, name, lo, hi, multiple=1):
    x = c.get(name)
    if type(x) is not int or not lo <= x <= hi or x % multiple:
        raise ValueError("H3_GEOMETRY:" + name)
    return x


def _options(c, variable, fixed):
    if set(c) != set(variable) | set(fixed) or any(c.get(k) != v for k, v in fixed.items()):
        raise ValueError("H3_CONFIGURATION")


def _normalization(c):
    for key in ("latents_mean", "latents_std"):
        x = c.get(key)
        if (
            not isinstance(x, list)
            or len(x) != c["latent_channels"]
            or any(
                type(v) not in {int, float}
                or not math.isfinite(v)
                or abs(v) > 100
                or key == "latents_std"
                and v <= 0
                for v in x
            )
        ):
            raise ValueError("H3_LATENT_NORMALIZATION")


def validate_configs(configs):
    from oryn_engine.qwen_retrieval import validate_config as validate_qwen

    if set(configs) != set(CONFIGS):
        raise ValueError("H3_COMPONENTS")
    for name, c in configs.items():
        # T2VA uses no pixels from this processor. Its original area budgets
        # are checked exactly below and never become request allocation limits.
        validate_model_config(
            {k: v for k, v in c.items() if k != "size"}
            if name in {"processor", "video_processor"}
            else c
        )
    q, t, v, a = (configs[k] for k in WEIGHT_COMPONENTS)
    validate_qwen(q)
    _options(
        q,
        ("text_config", "vision_config", "transformers_version"),
        dict(
            architectures=["Qwen3VLForConditionalGeneration"],
            model_type="qwen3_vl",
            tie_word_embeddings=False,
            image_token_id=151655,
            video_token_id=151656,
            vision_start_token_id=151652,
            vision_end_token_id=151653,
        ),
    )
    text, vision = q["text_config"], q["vision_config"]
    _options(
        text,
        (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "rope_scaling",
        ),
        dict(
            model_type="qwen3_vl_text",
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151645,
            dtype="bfloat16",
            hidden_act="silu",
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            rope_theta=5000000,
            use_cache=True,
            vocab_size=151936,
        ),
    )
    if text["num_hidden_layers"] <= 50:
        raise ValueError("H3_CONDITIONER_LAYER")
    _options(
        vision,
        (
            "depth",
            "hidden_size",
            "intermediate_size",
            "num_heads",
            "out_hidden_size",
            "deepstack_visual_indexes",
            "num_position_embeddings",
        ),
        dict(
            model_type="qwen3_vl",
            hidden_act="gelu_pytorch_tanh",
            in_channels=3,
            initializer_range=0.02,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
        ),
    )
    _options(
        t,
        (
            "_diffusers_version",
            "hidden_size",
            "num_attention_heads",
            "attention_head_dim",
            "num_layers",
            "num_refiner_layers",
            "ffn_dim",
            "in_channels",
            "audio_in_channels",
            "text_dim",
            "freq_dim",
            "time_embed_hidden_dim",
            "time_embed_dim",
            "rope_freq_dim",
        ),
        dict(
            _class_name="MiniMaxH3Transformer3DModel",
            patch_size=[1, 2, 2],
            rope_theta=10000.0,
            norm_eps=1e-5,
            qk_norm_eps=1e-5,
            final_norm_eps=1e-5,
        ),
    )
    for name, lo, hi, mul in [
        ("hidden_size", 16, 8192, 8),
        ("num_attention_heads", 1, 64, 1),
        ("attention_head_dim", 8, 256, 2),
        ("num_layers", 1, 64, 1),
        ("num_refiner_layers", 1, 4, 1),
        ("ffn_dim", 16, 32768, 8),
        ("in_channels", 4, 128, 2),
        ("audio_in_channels", 4, 128, 2),
        ("text_dim", 8, 8192, 8),
        ("freq_dim", 8, 1024, 2),
        ("time_embed_hidden_dim", 16, 8192, 8),
        ("time_embed_dim", 8, 8192, 8),
        ("rope_freq_dim", 1, 32, 1),
    ]:
        _integer(t, name, lo, hi, mul)
    if t["rope_freq_dim"] * 6 > t["attention_head_dim"] or t["text_dim"] != text["hidden_size"]:
        raise ValueError("H3_TRANSFORMER_GEOMETRY")
    _options(
        v,
        (
            "_diffusers_version",
            "latent_channels",
            "block_out_channels",
            "layers_per_block",
            "norm_num_groups",
            "decoder_num_layers",
            "decoder_num_attention_heads",
            "decoder_attention_head_dim",
            "decoder_num_register_tokens",
            "decoder_ffn_mult",
            "latents_mean",
            "latents_std",
        ),
        dict(
            _class_name="AutoencoderKLMiniMaxH3",
            in_channels=3,
            out_channels=3,
            spatial_downsample_factors=[2, 2, 2, 2, 1, 1],
            temporal_downsample_factors=[1, 2, 2, 1, 1, 1],
            norm_eps=1e-6,
            spatial_padding_mode="reflect",
            decoder_rope_theta=100.0,
            decoder_rope_dim_ratio=0.75,
            decoder_norm_eps=1e-5,
            clip_length=17,
            token_drop=3,
        ),
    )
    for name, lo, hi, mul in [
        ("latent_channels", 4, 128, 2),
        ("layers_per_block", 1, 4, 1),
        ("norm_num_groups", 1, 32, 1),
        ("decoder_num_layers", 1, 48, 1),
        ("decoder_num_attention_heads", 1, 32, 1),
        ("decoder_attention_head_dim", 16, 128, 8),
        ("decoder_num_register_tokens", 1, 16, 1),
        ("decoder_ffn_mult", 1, 8, 1),
    ]:
        _integer(v, name, lo, hi, mul)
    channels = v["block_out_channels"]
    if (
        not isinstance(channels, list)
        or len(channels) != 6
        or any(
            type(x) is not int or not 4 <= x <= 2048 or x % v["norm_num_groups"] for x in channels
        )
    ):
        raise ValueError("H3_VIDEO_ENCODER")
    if int(v["decoder_attention_head_dim"] * 0.75) % 6 or v["latent_channels"] != t["in_channels"]:
        raise ValueError("H3_VIDEO_GEOMETRY")
    _normalization(v)
    _options(
        a,
        (
            "_diffusers_version",
            "encoder_dim",
            "latent_dim",
            "latent_channels",
            "decoder_dim",
            "num_attention_heads",
            "latents_mean",
            "latents_std",
        ),
        dict(
            _class_name="AutoencoderKLMiniMaxH3Audio",
            encoder_rates=[2, 4, 4, 5, 5],
            decoder_rates=[5, 5, 2, 2, 2, 2, 2],
            decoder_kernel_sizes=[9, 9, 4, 4, 4, 4, 4],
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5]] * 3,
            sampling_rate=SAMPLE_RATE,
        ),
    )
    for name, lo, hi, mul in [
        ("encoder_dim", 2, 128, 2),
        ("latent_dim", 8, 4096, 8),
        ("latent_channels", 4, 128, 2),
        ("decoder_dim", 128, 2048, 128),
        ("num_attention_heads", 1, 16, 1),
    ]:
        _integer(a, name, lo, hi, mul)
    if (
        a["latent_channels"] != t["audio_in_channels"]
        or a["latent_dim"] % a["latent_channels"]
        or a["latent_dim"] % a["num_attention_heads"]
    ):
        raise ValueError("H3_AUDIO_GEOMETRY")
    _normalization(a)
    for name, shift in [("scheduler", 12.0), ("audio_scheduler", 3.0)]:
        _options(
            configs[name],
            ("_diffusers_version",),
            dict(_class_name="MiniMaxH3Scheduler", shift=shift),
        )
    tok = configs["tokenizer"]
    if any(
        tok.get(k) != x
        for k, x in dict(
            tokenizer_class="Qwen2Tokenizer",
            add_bos_token=False,
            add_prefix_space=False,
            eos_token="<|im_end|>",
            pad_token="<|endoftext|>",
            clean_up_tokenization_spaces=False,
            split_special_tokens=False,
        ).items()
    ):
        raise ValueError("H3_TOKENIZER")
    for name, size, kind in [
        ("processor", dict(longest_edge=16777216, shortest_edge=65536), "image"),
        ("video_processor", dict(longest_edge=25165824, shortest_edge=4096), "video"),
    ]:
        _options(
            configs[name],
            (),
            dict(
                size=size,
                patch_size=16,
                temporal_patch_size=2,
                merge_size=2,
                image_mean=[0.5] * 3,
                image_std=[0.5] * 3,
                processor_class="Qwen3VLProcessor",
                **{
                    kind + "_processor_type": "Qwen2VLImageProcessorFast"
                    if kind == "image"
                    else "Qwen3VLVideoProcessor"
                },
            ),
        )
    return configs


def parameter_specs(configs):
    validate_configs(configs)
    result = {name: {} for name in WEIGHT_COMPONENTS}
    s = result["text_encoder"]

    def linear(name, inputs, outputs, bias=True):
        s[name + ".weight"] = (outputs, inputs)
        if bias:
            s[name + ".bias"] = (outputs,)

    def norm(name, dim, bias=True):
        s[name + ".weight"] = (dim,)
        if bias:
            s[name + ".bias"] = (dim,)

    def conv(name, inputs, outputs, kernel, bias=True):
        s[name + ".weight"] = (outputs, inputs, *kernel)
        if bias:
            s[name + ".bias"] = (outputs,)

    q = configs["text_encoder"]
    t = q["text_config"]
    v = q["vision_config"]
    h, f, d, n, kv = (
        t[k]
        for k in (
            "hidden_size",
            "intermediate_size",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
        )
    )
    linear("model.language_model.embed_tokens", h, t["vocab_size"], False)
    linear("lm_head", h, t["vocab_size"], False)
    norm("model.language_model.norm", h, False)
    for i in range(t["num_hidden_layers"]):
        p = f"model.language_model.layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            norm(p + name, h, False)
        for name, inp, out in [
            ("mlp.gate_proj", h, f),
            ("mlp.up_proj", h, f),
            ("mlp.down_proj", f, h),
            ("self_attn.q_proj", h, n * d),
            ("self_attn.k_proj", h, kv * d),
            ("self_attn.v_proj", h, kv * d),
            ("self_attn.o_proj", n * d, h),
        ]:
            linear(p + name, inp, out, False)
        for name in ("q_norm", "k_norm"):
            norm(p + "self_attn." + name, d, False)
    h = v["hidden_size"]
    f = v["intermediate_size"]
    m = h * v["spatial_merge_size"] ** 2
    conv("model.visual.patch_embed.proj", 3, h, (2, 16, 16))
    linear("model.visual.pos_embed", h, v["num_position_embeddings"], False)
    for i in range(v["depth"]):
        p = f"model.visual.blocks.{i}."
        for name in ("norm1", "norm2"):
            norm(p + name, h)
        for name, inp, out in [
            ("attn.qkv", h, 3 * h),
            ("attn.proj", h, h),
            ("mlp.linear_fc1", h, f),
            ("mlp.linear_fc2", f, h),
        ]:
            linear(p + name, inp, out)
    for name, dim in [("merger", h)] + [
        (f"deepstack_merger_list.{i}", m) for i in range(len(v["deepstack_visual_indexes"]))
    ]:
        p = "model.visual." + name + "."
        norm(p + "norm", dim)
        linear(p + "linear_fc1", m, m)
        linear(p + "linear_fc2", m, v["out_hidden_size"])
    s = result["transformer"]
    t = configs["transformer"]
    h = t["hidden_size"]
    d = t["attention_head_dim"]
    inner = d * t["num_attention_heads"]
    f = t["ffn_dim"]
    for name, inp, out in [
        ("proj_in", t["in_channels"] * 4, h),
        ("audio_proj_in", t["audio_in_channels"], h),
        ("context_embedder", t["text_dim"], h),
        ("time_embedder.linear_1", t["freq_dim"], t["time_embed_hidden_dim"]),
        ("time_embedder.linear_2", t["time_embed_hidden_dim"], t["time_embed_dim"]),
        ("norm_out.linear", t["time_embed_dim"], 2 * h),
        ("proj_out", h, t["in_channels"] * 4),
        ("audio_proj_out", h, t["audio_in_channels"]),
    ]:
        linear(name, inp, out)
    norm("norm_out.norm", h, False)
    norm("token_refiner.final_norm", h, False)
    for group, count in [
        ("token_refiner.refiner_blocks", t["num_refiner_layers"]),
        ("transformer_blocks", t["num_layers"]),
    ]:
        for i in range(count):
            p = f"{group}.{i}."
            for name in ("norm1", "norm2"):
                norm(p + name, h, False)
            for name in ("q", "k", "v"):
                linear(p + "attn.to_" + name, h, inner, False)
            linear(p + "attn.to_out.0", inner, h, False)
            for name in ("q", "k"):
                norm(p + "attn.norm_" + name, d, False)
            linear(p + "ff.net.0.proj", h, 2 * f, False)
            linear(p + "ff.net.2", f, h, False)
            if group == "transformer_blocks":
                linear(p + "adaln_proj.linear", t["time_embed_dim"], 18 * h)
    s = result["vae"]
    v = configs["vae"]
    channels = v["block_out_channels"]
    c = v["latent_channels"]
    conv("encoder.conv_in", 3, channels[0], (3, 3, 3))
    before = channels[0]
    for i, out in enumerate(channels):
        for j in range(v["layers_per_block"]):
            p = f"encoder.down_blocks.{i}.resnets.{j}."
            inp = before if j == 0 else out
            norm(p + "norm1", inp)
            norm(p + "norm2", out)
            conv(p + "conv1", inp, out, (3, 3, 3))
            conv(p + "conv2", out, out, (3, 3, 3))
            if inp != out:
                conv(p + "conv_shortcut", inp, out, (1, 1, 1))
        if v["spatial_downsample_factors"][i] * v["temporal_downsample_factors"][i] > 1:
            conv(f"encoder.down_blocks.{i}.downsamplers.0.conv", out, out, (3, 3, 3))
        before = out
    norm("encoder.norm_out", before)
    conv("encoder.conv_out", before, 2 * c, (3, 3, 3))
    conv("quant_conv", 2 * c, 2 * c, (1, 1, 1))
    conv("post_quant_conv", c, c, (1, 1, 1))
    h = v["decoder_num_attention_heads"] * v["decoder_attention_head_dim"]
    f = h * v["decoder_ffn_mult"]
    s["decoder.register_tokens"] = (1, v["decoder_num_register_tokens"], h)
    linear("decoder.proj_in", c, h)
    norm("decoder.norm_out", h)
    linear("decoder.proj_out", h, 3 * 4 * 16 * 16)
    for i in range(v["decoder_num_layers"]):
        p = f"decoder.transformer_blocks.{i}."
        for name in ("norm1", "norm2"):
            norm(p + name, h, False)
        for name in ("scale1", "scale2"):
            s[p + name] = (h,)
        for name in ("to_q", "to_k", "to_v", "to_out.0"):
            linear(p + "attn." + name, h, h)
        linear(p + "ff.net.0.proj", h, 2 * f)
        linear(p + "ff.net.2", f, h)
    s = result["audio_vae"]
    a = configs["audio_vae"]
    h = a["encoder_dim"]
    d = a["latent_dim"]
    c = a["latent_channels"]

    def wn(name, inp, out, width, transpose=False, bias=True):
        s[name + ".weight_g"] = (inp if transpose else out, 1, 1)
        s[name + ".weight_v"] = (inp, out, width) if transpose else (out, inp, width)
        if bias:
            s[name + ".bias"] = (out,)

    def activation(name, dim):
        for key in ("alpha", "beta"):
            s[name + ".act." + key] = (dim,)
        for key in ("upsample.filter", "downsample.lowpass.filter"):
            s[name + "." + key] = (1, 1, 12)

    wn("encoder.block.0", 1, h, 7)
    for i, stride in enumerate(a["encoder_rates"], 1):
        p = f"encoder.block.{i}.block."
        for j in range(3):
            q = p + f"{j}.block."
            for k in (0, 2):
                s[q + str(k) + ".alpha"] = (1, h, 1)
            wn(q + "1", h, h, 7)
            wn(q + "3", h, h, 1)
        s[p + "3.alpha"] = (1, h, 1)
        wn(p + "4", h, 2 * h, 2 * stride)
        h *= 2
    s["encoder.block.6.alpha"] = (1, h, 1)
    wn("encoder.block.7", h, d, 3)
    for name, dim in [("norm1", d), ("norm3", d), ("norm2", c), ("mlp.norm", c)]:
        norm("pre_block." + name, dim)
    linear("pre_block.attn.qkv", d, 3 * d, False)
    for name in ("q_bias", "v_bias", "zero_k_bias"):
        s["pre_block.attn." + name] = (d,)
    for name, inp, out in [
        ("attn.proj", c, c),
        ("proj", d, c),
        ("mlp.w0", c, 2 * c),
        ("mlp.w1", c, 2 * c),
        ("mlp.w2", 2 * c, c),
    ]:
        linear("pre_block." + name, inp, out)
    for name in ("mean_proj", "logs_proj"):
        conv(name, c, c, (1,))
    conv("dec_in_proj", c, d, (1,))
    h = a["decoder_dim"]
    wn("decoder.conv_pre", d, h, 7)
    for i, (_rate, kernel) in enumerate(
        zip(a["decoder_rates"], a["decoder_kernel_sizes"], strict=True)
    ):
        out = h // 2
        wn(f"decoder.ups.{i}.0", h, out, kernel, True)
        for j, k in enumerate(a["resblock_kernel_sizes"]):
            p = f"decoder.resblocks.{3 * i + j}."
            for unit in range(3):
                wn(p + f"convs1.{unit}", out, out, k)
                wn(p + f"convs2.{unit}", out, out, k)
            for unit in range(6):
                activation(p + f"activations.{unit}", out)
        h = out
    activation("decoder.activation_post", h)
    wn("decoder.conv_post", h, 1, 7, bias=False)
    return result


def inspect_metadata(index, read_json, read_header, files):
    selected = selected_files(files, index)
    configs = validate_configs({k: read_json(n) for k, n in CONFIGS.items()})
    bundle.validate_headers(selected, parameter_specs(configs), read_header, read_json)
    return configs


def native_components(configs):
    import diffusers
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    validate_configs(configs)
    return {
        name: Qwen3VLForConditionalGeneration(Qwen3VLConfig.from_dict(copy.deepcopy(configs[name])))
        if name == "text_encoder"
        else getattr(diffusers, CLASSES[name][1]).from_config(copy.deepcopy(configs[name]))
        for name in WEIGHT_COMPONENTS
    }


def load_tokenizer(root):
    from tokenizers import Tokenizer
    from transformers import Qwen2Tokenizer

    path = root / "tokenizer/tokenizer.json"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 32 * 1024**2:
        raise ValueError("H3_TOKENIZER_GRAPH")
    try:
        vocab = Tokenizer.from_file(str(path)).get_vocab()
    except Exception as exc:
        raise ValueError("H3_TOKENIZER_GRAPH") from exc
    if (
        len(vocab) != 151669
        or set(vocab.values()) != set(range(151669))
        or any(vocab.get(k) != v for k, v in CONTROL_IDS.items())
    ):
        raise ValueError("H3_TOKENIZER_IDS")
    tok = Qwen2Tokenizer(
        tokenizer_file=str(path),
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        add_prefix_space=False,
        clean_up_tokenization_spaces=False,
    )
    tok.chat_template = None
    if tok.get_vocab() != vocab or any(
        tok.encode(k, add_special_tokens=False) != [v] for k, v in CONTROL_IDS.items()
    ):
        raise ValueError("H3_TOKENIZER_CONTROLS")
    return tok


def native_pipeline(configs, components, tokenizer, contract=CONTRACT):
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
    from transformers import Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

    if contract not in CONTRACTS:
        raise ValueError("H3_CONTRACT")
    # T2VA uses token types; FL2VA also encodes the bounded keyframe canvas.
    # Preserve the native pixel convention without executing repository Jinja.
    processor = Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            size=configs["processor"]["size"],
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
        ),
        video_processor=Qwen3VLVideoProcessor(patch_size=16, temporal_patch_size=2, merge_size=2),
    )
    pipe = MiniMaxH3ModularPipeline(
        blocks=MiniMaxH3Blocks().get_workflow("fl2va" if contract == IMAGE_CONTRACT else "t2va")
    )
    pipe.update_components(
        **components,
        tokenizer=tokenizer,
        processor=processor,
        scheduler=MiniMaxH3Scheduler.from_config(configs["scheduler"]),
        audio_scheduler=MiniMaxH3Scheduler.from_config(configs["audio_scheduler"]),
    )
    return pipe


def load_pipeline(root, backend, contract=CONTRACT):
    index = metadata_file(root / INDEX)
    selected = selected_files(
        [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()], index
    )
    configs = validate_configs({k: metadata_file(root / n) for k, n in CONFIGS.items()})
    components = bundle.load_components(
        root, selected, parameter_specs(configs), lambda: native_components(configs), backend
    )
    pipe = native_pipeline(configs, components, load_tokenizer(root), contract)
    pipe.to(backend)
    return pipe


def task_profile(contract=CONTRACT):
    from oryn_protocol.task_contracts import make_profile

    if contract not in CONTRACTS:
        raise ValueError("H3_CONTRACT")
    constraints = dict(
        width={"maximum": 1024, "multipleOf": 32},
        height={"maximum": 1024, "multipleOf": 32},
        frames={"enum": list(FRAME_COUNTS)},
        fps={"const": FPS},
        steps={"minimum": 1, "maximum": 100},
        guidanceScale={"const": 1},
        seed={"minimum": 0},
    )
    defaults = dict(width=512, height=512, frames=124, fps=FPS, steps=20, guidanceScale=1, seed=0)
    features = {"deterministicSeed": "supported"} if contract == CONTRACT else {}
    if contract == CONTRACT:
        constraints["negativePrompt"] = {"const": ""}
        defaults["negativePrompt"] = ""
        features["negativePrompt"] = "unsupported"
    return make_profile(contract, features=features, constraints=constraints, defaults=defaults)


def validate_input(value, contract=CONTRACT):
    from oryn_protocol.task_contracts import effective_input, validate_invocation

    profile = task_profile(contract)
    validate_invocation(profile, value)
    result = effective_input(profile, value)
    if (
        not result["prompt"].strip()
        or any(
            x in result["prompt"] for x in ("<|", "|>", "<tool_", "</tool_", "<think>", "</think>")
        )
        or result["width"] * result["height"] * result["frames"] > 96 * 1024 * 1024
        or not 0.25 <= result["width"] / result["height"] <= 4
    ):
        raise ValueError("H3_INPUT")
    return result


def generate(root, output_path, backend, value, event, *, contract=CONTRACT, input_path=None):
    import numpy as np
    import torch

    from oryn_engine.video_audio_artifacts import write_video_audio

    value = validate_input(value, contract)
    event("loading-video-audio-components", 0.02)
    pipe = load_pipeline(root, backend, contract)
    tokens = pipe.tokenizer.encode(value["prompt"], add_special_tokens=False)
    if not 1 <= len(tokens) <= 512:
        raise ValueError("H3_PROMPT_LIMIT")
    image_options = {}
    if contract == IMAGE_CONTRACT:
        from PIL import Image, ImageOps

        if (
            input_path is None
            or input_path.is_symlink()
            or not input_path.is_file()
            or not 0 < input_path.stat().st_size <= 20 * 1024**2
        ):
            raise ValueError("H3_KEYFRAME")
        with Image.open(input_path) as image:
            if image.width * image.height > 4 * 1024 * 1024 or getattr(image, "n_frames", 1) != 1:
                raise ValueError("H3_KEYFRAME")
            image_options["image"] = ImageOps.exif_transpose(image).convert("RGB").copy()
    elif input_path is not None:
        raise ValueError("H3_KEYFRAME")
    calls = 0

    def progress(module, inputs):
        nonlocal calls
        calls += 1
        event("generating-video-audio", min(0.85, 0.05 + 0.8 * calls / (value["steps"] + 1)))

    hooks = [
        pipe.text_encoder.model.register_forward_pre_hook(progress),
        pipe.transformer.register_forward_pre_hook(progress),
    ]
    try:
        with torch.inference_mode():
            result = pipe(
                prompt=value["prompt"],
                height=value["height"],
                width=value["width"],
                num_frames=value["frames"],
                # H3 counts the terminal zero in the grid. The public steps
                # parameter counts model evaluations, as for other pipelines.
                num_inference_steps=value["steps"] + 1,
                generator=torch.Generator("cpu").manual_seed(value["seed"]),
                output_type="np",
                output=["videos", "audio", "sampling_rate"],
                **image_options,
            )
    finally:
        for hook in hooks:
            hook.remove()
    frames = result["videos"]
    audio = result["audio"]
    if (
        not isinstance(frames, np.ndarray)
        or frames.shape != (1, value["frames"], value["height"], value["width"], 3)
        or not np.isfinite(frames).all()
        or frames.min() < 0
        or frames.max() > 1
        or not isinstance(audio, torch.Tensor)
        or audio.ndim != 3
        or audio.shape[:2] != (1, 2)
        or result["sampling_rate"] != SAMPLE_RATE
    ):
        raise ValueError("H3_OUTPUT")
    try:
        samples = write_video_audio(
            output_path,
            frames[0],
            audio[0].detach().cpu().numpy(),
            fps=FPS,
            sample_rate=SAMPLE_RATE,
            event=event,
        )
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return dict(
        relativeOutput=output_path.name,
        mediaType="video/mp4",
        backend=backend,
        width=value["width"],
        height=value["height"],
        frames=value["frames"],
        fps=FPS,
        sampleRate=SAMPLE_RATE,
        channels=2,
        samples=samples,
    )
