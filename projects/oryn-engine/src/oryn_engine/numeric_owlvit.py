"""Complete native OwlViT detection with bounded numeric archive metadata."""

from oryn_engine.model_metadata import validate_model_config


def validate_config(c):
    validate_model_config(c)
    if (
        c.get("model_type") != "owlvit"
        or c.get("architectures") != ["OwlViTForObjectDetection"]
        or c.get("auto_map")
        or c.get("quantization_config")
        or c.get("quantization")
        or c.get("text_config_dict")
        or c.get("vision_config_dict")
    ):
        raise ValueError("NUMERIC_OWLVIT_CONFIGURATION")
    for side in ["text", "vision"]:
        x = c.get(side + "_config")
        if (
            not isinstance(x, dict)
            or x.get("model_type", f"owlvit_{side}_model") != f"owlvit_{side}_model"
            or x.get("auto_map")
            or x.get("quantization_config")
            or x.get("quantization")
            or x.get("is_decoder", False)
            or x.get("add_cross_attention", False)
            or x.get("pruned_heads")
        ):
            raise ValueError("NUMERIC_OWLVIT_CONFIGURATION")
        for k, lo, hi in [
            ("hidden_size", 8, 2048),
            ("intermediate_size", 8, 16384),
            ("num_hidden_layers", 1, 48),
            ("num_attention_heads", 1, 32),
        ]:
            if type(x.get(k)) is not int or not lo <= x[k] <= hi:
                raise ValueError("NUMERIC_OWLVIT_CONFIGURATION_LIMIT")
        if x["hidden_size"] % x["num_attention_heads"] or x.get("hidden_act", "quick_gelu") not in [
            "quick_gelu",
            "gelu",
        ]:
            raise ValueError("NUMERIC_OWLVIT_CONFIGURATION")
    for obj, k, lo, hi in [
        (c, "projection_dim", 8, 2048),
        (c["text_config"], "vocab_size", 8, 100000),
        (c["text_config"], "max_position_embeddings", 4, 512),
        (c["vision_config"], "image_size", 32, 1024),
        (c["vision_config"], "patch_size", 2, 64),
    ]:
        if type(obj.get(k)) is not int or not lo <= obj[k] <= hi:
            raise ValueError("NUMERIC_OWLVIT_CONFIGURATION_LIMIT")
    text, vision = c["text_config"], c["vision_config"]
    if c["projection_dim"] != text["hidden_size"]:
        raise ValueError("NUMERIC_OWLVIT_CONFIGURATION")
    if (
        vision.get("num_channels", 3) != 3
        or type(vision.get("num_channels", 3)) is not int
        or vision["image_size"] % vision["patch_size"]
        or (vision["image_size"] // vision["patch_size"]) ** 2 > 4096
    ):
        raise ValueError("NUMERIC_OWLVIT_CONFIGURATION")
    for k in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        if type(text.get(k)) is not int or not 0 <= text[k] < text["vocab_size"]:
            raise ValueError("NUMERIC_OWLVIT_TOKENIZER")


def expected_shapes(c):
    validate_config(c)
    text, vision = c["text_config"], c["vision_config"]
    h = vision["hidden_size"]
    d = text["hidden_size"]
    projection = c["projection_dim"]
    out = {"owlvit.logit_scale": []}

    def linear(name, rows, cols, bias=True):
        out[name + ".weight"] = [rows, cols]
        if bias:
            out[name + ".bias"] = [rows]

    def norm(name, width):
        out[name + ".weight"] = [width]
        out[name + ".bias"] = [width]

    out["owlvit.text_model.embeddings.token_embedding.weight"] = [text["vocab_size"], d]
    out["owlvit.text_model.embeddings.position_embedding.weight"] = [
        text["max_position_embeddings"],
        d,
    ]
    out["owlvit.vision_model.embeddings.class_embedding"] = [h]
    out["owlvit.vision_model.embeddings.patch_embedding.weight"] = [
        h,
        3,
        vision["patch_size"],
        vision["patch_size"],
    ]
    out["owlvit.vision_model.embeddings.position_embedding.weight"] = [
        (vision["image_size"] // vision["patch_size"]) ** 2 + 1,
        h,
    ]
    norm("owlvit.text_model.final_layer_norm", d)
    norm("owlvit.vision_model.pre_layernorm", h)
    norm("owlvit.vision_model.post_layernorm", h)
    linear("owlvit.text_projection", projection, d, False)
    linear("owlvit.visual_projection", projection, h, False)
    for side, x in [("text", text), ("vision", vision)]:
        width = x["hidden_size"]
        for i in range(x["num_hidden_layers"]):
            prefix = f"owlvit.{side}_model.encoder.layers.{i}"
            for name in ["k_proj", "v_proj", "q_proj", "out_proj"]:
                linear(prefix + ".self_attn." + name, width, width)
            for name in ["layer_norm1", "layer_norm2"]:
                norm(prefix + "." + name, width)
            linear(prefix + ".mlp.fc1", x["intermediate_size"], width)
            linear(prefix + ".mlp.fc2", width, x["intermediate_size"])
    linear("class_head.dense0", d, h)
    linear("class_head.logit_shift", 1, h)
    linear("class_head.logit_scale", 1, h)
    for name, rows in [("dense0", h), ("dense1", h), ("dense2", 4)]:
        linear("box_head." + name, rows, h)
    norm("layer_norm", h)
    return out


def processor_options(c, p):
    validate_model_config(p)
    size = c["vision_config"]["image_size"]
    shape = p.get("size")
    dimensions = list(shape.values()) if isinstance(shape, dict) else shape
    if (
        p.get("processor_class", "OwlViTProcessor") != "OwlViTProcessor"
        or p.get("image_processor_type", p.get("feature_extractor_type", "OwlViTImageProcessor"))
        not in ["OwlViTImageProcessor", "OwlViTImageProcessorFast", "OwlViTFeatureExtractor"]
        or p.get("auto_map")
        or shape not in ([size, size], {"height": size, "width": size})
        or not isinstance(dimensions, list)
        or any(type(dimension) is not int for dimension in dimensions)
        or any(
            p.get(k, True) is not True
            for k in ["do_resize", "do_normalize", "do_rescale", "do_convert_rgb"]
        )
        or p.get("rescale", True) is not True
        or p.get("do_center_crop", False) is not False
        or p.get("do_pad", False) is not False
        or p.get("resample", 3) != 3
        or type(p.get("resample", 3)) is not int
        or p.get("rescale_factor", 1 / 255) != 1 / 255
        or p.get("image_mean") != [0.48145466, 0.4578275, 0.40821073]
        or p.get("image_std") != [0.26862954, 0.26130258, 0.27577711]
    ):
        raise ValueError("NUMERIC_OWLVIT_PROCESSOR")
    return dict(
        size={"height": size, "width": size},
        do_resize=True,
        do_center_crop=False,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        do_convert_rgb=True,
        resample=3,
        image_mean=p["image_mean"],
        image_std=p["image_std"],
    )


def prepare_tokenizer(c, tokenizer):
    import transformers as tf

    text = c["text_config"]
    if (
        not isinstance(tokenizer, tf.CLIPTokenizer)
        or not tokenizer.is_fast
        or len(tokenizer) > text["vocab_size"]
    ):
        raise ValueError("NUMERIC_OWLVIT_TOKENIZER")
    if text["eos_token_id"] == 2:
        # Legacy CLIP pools the highest token ID, which must be its EOS marker.
        if (
            tokenizer.eos_token_id != len(tokenizer) - 1
            or tokenizer.bos_token_id != len(tokenizer) - 2
        ):
            raise ValueError("NUMERIC_OWLVIT_TOKENIZER")
    elif tokenizer.eos_token_id != text["eos_token_id"]:
        raise ValueError("NUMERIC_OWLVIT_TOKENIZER")
    tokenizer.model_max_length = text["max_position_embeddings"]
    tokenizer.chat_template = None


def execute(model, processor, c, image, value, backend):
    import torch

    labels = value["labels"]
    encoded = processor(
        text=[labels], images=image, return_tensors="pt", padding=True, truncation=False
    )
    size = c["vision_config"]["image_size"]
    positions = c["text_config"]["max_position_embeddings"]
    if (
        encoded["input_ids"].shape[0] != len(labels)
        or not 1 <= encoded["input_ids"].shape[1] <= positions
        or list(encoded["pixel_values"].shape) != [1, 3, size, size]
    ):
        raise ValueError("NUMERIC_OWLVIT_INPUT_LIMIT")
    output = model(**{k: v.to(backend) for k, v in encoded.items()})
    count = (size // c["vision_config"]["patch_size"]) ** 2
    if (
        list(output.logits.shape) != [1, count, len(labels)]
        or list(output.pred_boxes.shape) != [1, count, 4]
        or not torch.isfinite(output.logits).all()
        or not torch.isfinite(output.pred_boxes).all()
    ):
        raise ValueError("NUMERIC_OWLVIT_OUTPUT")
    result = processor.post_process_grounded_object_detection(
        output,
        threshold=value.get("threshold", 0.3),
        target_sizes=[(image.height, image.width)],
        text_labels=[labels],
    )[0]
    detections = []
    for score, label, box in zip(
        result["scores"], result["text_labels"], result["boxes"], strict=True
    ):
        if len(detections) == 256:
            break
        x0, y0, x1, y1 = box.cpu().tolist()
        if not 0 <= float(score) <= 1 or x1 < x0 or y1 < y0 or label not in labels:
            raise ValueError("NUMERIC_OWLVIT_OUTPUT")
        detections.append(
            dict(
                label=label,
                score=float(score),
                box=[
                    max(0, min(image.width, x0)),
                    max(0, min(image.height, y0)),
                    max(0, min(image.width, x1)),
                    max(0, min(image.height, y1)),
                ],
            )
        )
    return dict(width=image.width, height=image.height, detections=detections)


ARCHITECTURES = {("zero-shot-object-detection", "owlvit"): "OwlViTForObjectDetection"}


def supports(config, task):
    return (task, config.get("model_type")) in ARCHITECTURES
