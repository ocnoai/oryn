"""Complete DPT/BiT depth graphs and owned declarative image preprocessing."""

from oryn_engine.model_metadata import validate_model_config

ARCHITECTURES = {("depth-estimation", "dpt"): "DPTForDepthEstimation"}


def supports(config, task):
    return task == "depth-estimation" and config.get("model_type") == "dpt"


def validate_config(c):
    validate_model_config(c)
    if (
        c.get("architectures") != ["DPTForDepthEstimation"]
        or c.get("auto_map")
        or c.get("quantization_config")
        or c.get("quantization")
        or c.get("backbone") is not None
        or c.get("backbone_kwargs")
        or c.get("use_pretrained_backbone", False)
        or c.get("use_timm_backbone", False)
    ):
        raise ValueError("NUMERIC_DPT_CONFIGURATION")
    for k, lo, hi in [
        ("hidden_size", 8, 2048),
        ("intermediate_size", 8, 16384),
        ("num_hidden_layers", 4, 48),
        ("num_attention_heads", 1, 32),
        ("image_size", 32, 512),
        ("fusion_hidden_size", 8, 512),
    ]:
        if type(c.get(k)) is not int or not lo <= c[k] <= hi:
            raise ValueError("NUMERIC_DPT_CONFIGURATION_LIMIT")
    if (
        type(c.get("patch_size", 16)) is not int
        or c.get("patch_size", 16) != 16
        or c["image_size"] % 32
        or c["hidden_size"] % c["num_attention_heads"]
        or c["fusion_hidden_size"] % 2
        or type(c.get("num_channels", 3)) is not int
        or c.get("num_channels", 3) != 3
        or c.get("use_batch_norm_in_fusion_residual", False) is not False
        or c.get("add_projection", False) is not False
        or type(c.get("head_in_index", -1)) is not int
        or c.get("head_in_index", -1) != -1
        or c.get("hidden_act", "gelu") != "gelu"
        or c.get("readout_type", "project") != "project"
    ):
        raise ValueError("NUMERIC_DPT_CONFIGURATION")
    for k in ["qkv_bias", "is_hybrid"]:
        if k in c and type(c[k]) is not bool:
            raise ValueError("NUMERIC_DPT_CONFIGURATION")
    indices = c.get("backbone_out_indices")
    if (
        not isinstance(indices, list)
        or len(indices) != 4
        or any(type(i) is not int or not 0 <= i < c["num_hidden_layers"] for i in indices)
        or indices != sorted(set(indices))
    ):
        raise ValueError("NUMERIC_DPT_CONFIGURATION")
    widths = c.get("neck_hidden_sizes")
    if (
        not isinstance(widths, list)
        or len(widths) != 4
        or any(type(i) is not int or not 8 <= i <= 2048 for i in widths)
    ):
        raise ValueError("NUMERIC_DPT_CONFIGURATION")
    hybrid = c.get("is_hybrid", False)
    if c.get("neck_ignore_stages", []) != ([0, 1] if hybrid else []):
        raise ValueError("NUMERIC_DPT_CONFIGURATION")
    factors = c.get("reassemble_factors")
    if factors != ([1, 1, 1, 0.5] if hybrid else [4, 2, 1, 0.5]) or any(
        type(f) is not int for f in factors[:3]
    ):
        raise ValueError("NUMERIC_DPT_CONFIGURATION")
    b = c.get("backbone_config")
    if not hybrid:
        if b is not None:
            raise ValueError("NUMERIC_DPT_CONFIGURATION")
        return
    if (
        not isinstance(b, dict)
        or b.get("model_type") != "bit"
        or b.get("layer_type", "preactivation") != "bottleneck"
        or b.get("width_factor", 1) != 1
        or b.get("hidden_act", "relu") != "relu"
        or type(b.get("num_channels", 3)) is not int
        or b.get("num_channels", 3) != 3
        or b.get("output_stride", 32) != 32
        or b.get("global_padding") != "SAME"
        or b.get("embedding_dynamic_padding", False) is not True
        or b.get("out_features") != ["stage1", "stage2", "stage3"]
        or b.get("out_indices", [1, 2, 3]) != [1, 2, 3]
        or b.get("stage_names", ["stem", "stage1", "stage2", "stage3"])
        != ["stem", "stage1", "stage2", "stage3"]
        or b.get("auto_map")
    ):
        raise ValueError("NUMERIC_DPT_BACKBONE")
    depths = b.get("depths")
    widths = b.get("hidden_sizes")
    embedding = b.get("embedding_size")
    groups = b.get("num_groups", 32)
    if (
        not isinstance(depths, list)
        or len(depths) != 3
        or any(type(d) is not int or not 1 <= d <= 16 for d in depths)
        or not isinstance(widths, list)
        or not 3 <= len(widths) <= 4
        or any(type(d) is not int or not 16 <= d <= 2048 or d % 8 for d in widths)
        or type(embedding) is not int
        or not 8 <= embedding <= 256
        or type(groups) is not int
        or not 1 <= groups <= 64
    ):
        raise ValueError("NUMERIC_DPT_BACKBONE")
    mids = [max(8, int(w / 4 + 4) // 8 * 8) for w in widths[:3]]
    mids = [x + 8 if x < 0.9 * (w / 4) else x for x, w in zip(mids, widths[:3], strict=True)]
    if (
        any(x % groups for x in [embedding, *widths[:3], *mids])
        or c["neck_hidden_sizes"][:2] != widths[:2]
        or c.get("backbone_featmap_shape")
        != [1, widths[2], c["image_size"] // 16, c["image_size"] // 16]
        or any(type(x) is not int for x in c["backbone_featmap_shape"])
    ):
        raise ValueError("NUMERIC_DPT_BACKBONE")


def processor_options(c, m):
    validate_model_config(m)
    if (
        m.get("image_processor_type") not in {"DPTImageProcessor", "DPTImageProcessorFast"}
        or m.get("auto_map")
        or m.get("do_resize", True) is not True
        or m.get("do_rescale", True) is not True
        or m.get("do_normalize", True) is not True
        or m.get("keep_aspect_ratio", False) is not False
        or m.get("do_pad", False)
        or m.get("do_center_crop", False)
    ):
        raise ValueError("NUMERIC_DPT_PROCESSOR")
    size = m.get("size")
    target = {"height": c["image_size"], "width": c["image_size"]}
    if type(size) is int:
        size = {"height": size, "width": size}
    if (
        size != target
        or any(type(x) is not int for x in size.values())
        or type(m.get("resample", 3)) is not int
        or m.get("resample", 3) not in {2, 3}
        or type(m.get("ensure_multiple_of", 1)) is not int
        or m.get("ensure_multiple_of", 1) not in {1, 16, 32}
        or m.get("rescale_factor", 1 / 255) != 1 / 255
        or m.get("image_mean", [0.5] * 3) != [0.5] * 3
        or m.get("image_std", [0.5] * 3) != [0.5] * 3
    ):
        raise ValueError("NUMERIC_DPT_PROCESSOR")
    return dict(
        size=target,
        resample=m.get("resample", 3),
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
        keep_aspect_ratio=False,
        do_pad=False,
        ensure_multiple_of=m.get("ensure_multiple_of", 1),
        rescale_factor=1 / 255,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )


def execute(model, processor, config, image, backend):
    import torch

    if image is None:
        raise ValueError("NUMERIC_DPT_IMAGE")
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    pixels = inputs["pixel_values"]
    if tuple(pixels.shape) != (1, 3, config["image_size"], config["image_size"]) or not bool(
        torch.isfinite(pixels).all()
    ):
        raise ValueError("NUMERIC_DPT_PIXELS")
    output = model(pixel_values=pixels.to(backend)).predicted_depth
    if (
        tuple(output.shape) != (1, config["image_size"], config["image_size"])
        or not bool(torch.isfinite(output).all())
        or bool((output < 0).any())
    ):
        raise ValueError("NUMERIC_DPT_OUTPUT")
    depth = output[0].float().cpu()
    return {
        "width": depth.shape[1],
        "height": depth.shape[0],
        "minimum": float(depth.min()),
        "maximum": float(depth.max()),
        "mean": float(depth.mean()),
    }


def expected_shapes(c):
    out = {}
    h = c["hidden_size"]
    hybrid = c.get("is_hybrid", False)

    def linear(n, r, columns, bias=True):
        out[n + ".weight"] = [r, columns]
        if bias:
            out[n + ".bias"] = [r]

    def conv(n, r, columns, k, bias=True):
        out[n + ".weight"] = [r, columns, k, k]
        if bias:
            out[n + ".bias"] = [r]

    def norm(n, d):
        out[n + ".weight"] = [d]
        out[n + ".bias"] = [d]

    out["dpt.embeddings.cls_token"] = [1, 1, h]
    out["dpt.embeddings.position_embeddings"] = [
        1,
        (c["image_size"] // c["patch_size"]) ** 2 + 1,
        h,
    ]
    if hybrid:
        b = c["backbone_config"]
        prefix = "dpt.embeddings.backbone.bit"
        previous = b["embedding_size"]
        conv(prefix + ".embedder.convolution", previous, 3, 7, False)
        norm(prefix + ".embedder.norm", previous)
        for stage, depth in enumerate(b["depths"]):
            width = b["hidden_sizes"][stage]
            mid = max(8, int(width / 4 + 4) // 8 * 8)
            if mid < 0.9 * (width / 4):
                mid += 8
            for layer in range(depth):
                n = f"{prefix}.encoder.stages.{stage}.layers.{layer}"
                if layer == 0:
                    conv(n + ".downsample.conv", width, previous, 1, False)
                    norm(n + ".downsample.norm", width)
                conv(n + ".conv1", mid, previous, 1, False)
                norm(n + ".norm1", mid)
                conv(n + ".conv2", mid, mid, 3, False)
                norm(n + ".norm2", mid)
                conv(n + ".conv3", width, mid, 1, False)
                norm(n + ".norm3", width)
                previous = width
        conv("dpt.embeddings.projection", h, previous, 1)
    else:
        conv("dpt.embeddings.patch_embeddings.projection", h, 3, c["patch_size"])
    for i in range(c["num_hidden_layers"]):
        n = f"dpt.encoder.layer.{i}"
        for leaf in ["query", "key", "value"]:
            linear(n + ".attention.attention." + leaf, h, h, c.get("qkv_bias", True))
        linear(n + ".attention.output.dense", h, h)
        linear(n + ".intermediate.dense", c["intermediate_size"], h)
        linear(n + ".output.dense", h, c["intermediate_size"])
        norm(n + ".layernorm_before", h)
        norm(n + ".layernorm_after", h)
    norm("dpt.layernorm", h)
    fusion = c["fusion_hidden_size"]
    for i, width in enumerate(c["neck_hidden_sizes"]):
        if not hybrid or i >= 2:
            n = f"neck.reassemble_stage.layers.{i}"
            conv(n + ".projection", width, h, 1)
            factor = c["reassemble_factors"][i]
            if factor != 1:
                conv(n + ".resize", width, width, int(factor) if factor > 1 else 3)
            if c.get("readout_type", "project") == "project":
                linear(f"neck.reassemble_stage.readout_projects.{i}.0", h, 2 * h)
        conv(f"neck.convs.{i}", fusion, width, 3, False)
        n = f"neck.fusion_stage.layers.{i}"
        conv(n + ".projection", fusion, fusion, 1)
        for r in [1, 2]:
            for v in [1, 2]:
                conv(f"{n}.residual_layer{r}.convolution{v}", fusion, fusion, 3)
    conv("head.head.0", fusion // 2, fusion, 3)
    conv("head.head.2", 32, fusion // 2, 3)
    conv("head.head.4", 1, 32, 1)
    return out
