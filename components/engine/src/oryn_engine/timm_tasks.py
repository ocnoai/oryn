"""Sealed raw timm graph binding through the pinned native Transformers wrapper."""

from __future__ import annotations

import copy
import math

from oryn_engine.model_metadata import validate_model_config


def normalized_config(config, task):
    if (
        task not in {"image-classification", "image-feature-extraction"}
        or not isinstance(config.get("architecture"), str)
        or not isinstance(config.get("pretrained_cfg"), dict)
    ):
        return config
    from oryn_engine.generated_capabilities import generated_artifact

    validate_model_config(config)
    architecture = config["architecture"]
    if architecture not in generated_artifact()["buildInputs"].get("timmArchitectures", []):
        raise ValueError("Native timm architecture is not in the signed build")
    if config["pretrained_cfg"].get("custom_load"):
        raise ValueError("Custom timm load behavior is not admitted")
    metadata = config["pretrained_cfg"]
    size, mean, std = metadata.get("input_size"), metadata.get("mean"), metadata.get("std")
    if (
        not isinstance(size, list)
        or len(size) != 3
        or size[0] != 3
        or any(type(x) is not int or not 8 <= x <= 1024 for x in size[1:])
        or any(
            not isinstance(v, list)
            or len(v) != 3
            or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in v
            )
            for v in (mean, std)
        )
        or any(abs(x) > 10 for x in mean)
        or any(not 0 < x <= 10 for x in std)
        or metadata.get("interpolation") not in {"bilinear", "bicubic", "nearest", "lanczos"}
        or metadata.get("crop_mode", "center") != "center"
        or isinstance(metadata.get("crop_pct"), bool)
        or not isinstance(metadata.get("crop_pct"), (int, float))
        or not 0.5 <= metadata["crop_pct"] <= 1
    ):
        raise ValueError("Native timm preprocessing exceeds reviewed bounds")
    arguments = config.get("model_args") or {}
    if not isinstance(arguments, dict) or set(arguments) - {
        "img_size",
        "window_size",
        "act_layer",
        "drop_path_rate",
    }:
        raise ValueError("Custom timm constructor behavior is not admitted")
    if arguments and not architecture.startswith("swinv2_"):
        raise ValueError("Native constructor overrides have no reviewed family binding")
    if (
        "img_size" in arguments
        and (
            type(arguments["img_size"]) is not int
            or size[1:] != [arguments["img_size"], arguments["img_size"]]
        )
        or "window_size" in arguments
        and (type(arguments["window_size"]) is not int or not 1 <= arguments["window_size"] <= 32)
        or "act_layer" in arguments
        and arguments["act_layer"] not in {"gelu", "gelu_tanh"}
        or "drop_path_rate" in arguments
        and (
            isinstance(arguments["drop_path_rate"], bool)
            or not isinstance(arguments["drop_path_rate"], (int, float))
            or not 0 <= arguments["drop_path_rate"] <= 0.5
        )
    ):
        raise ValueError("Native SwinV2 constructor values exceed reviewed bounds")
    if task == "image-feature-extraction" and (
        not architecture.startswith("vit_")
        or "_dinov2" not in architecture
        or config.get("num_classes") != 0
        or config.get("global_pool", "token") != "token"
    ):
        raise ValueError("This timm graph does not implement native CLS embeddings")
    from transformers.models.timm_wrapper.configuration_timm_wrapper import TimmWrapperConfig

    # Preserve the native explicit label names or native ImageNet label mapping;
    # generic LABEL_n placeholders cannot establish task semantics.
    tags = config.get("tags")
    tag_classifier = task == "image-classification" and isinstance(tags, list)
    if tag_classifier and (
        len(tags) != config.get("num_classes")
        or not 1 <= len(tags) <= 20000
        or len(set(tags)) != len(tags)
        or any(not isinstance(v, str) or not 1 <= len(v) <= 512 for v in tags)
    ):
        raise ValueError("Native tag labels are invalid")
    source = copy.deepcopy(config)
    if tag_classifier:
        source["label_names"] = list(tags)
    native = TimmWrapperConfig.from_dict(source)
    result = {
        **copy.deepcopy(config),
        "model_type": "timm_wrapper",
        "architectures": [
            "TimmWrapperForImageClassification"
            if task == "image-classification"
            else "TimmWrapperModel"
        ],
    }
    if task == "image-classification":
        labels = list(native.id2label.values())
        if (
            type(config.get("num_classes")) is not int
            or not 1 <= config["num_classes"] <= (20000 if tag_classifier else 4096)
            or len(labels) != config["num_classes"]
            or any(
                not isinstance(v, str) or not 1 <= len(v) <= 1024 or v.startswith("LABEL_")
                for v in labels
            )
        ):
            raise ValueError("Native timm classification labels are unavailable")
        result["label_names"] = labels
        result["_orynTimmTagClassifier"] = tag_classifier
    return result


def load_components(root, config, task):
    import transformers as tf

    # Only preprocessing data reaches this constructor. Location/loader/URL
    # declarations remain inert, and pretrained native fetching is disabled.
    metadata = {
        k: config["pretrained_cfg"][k]
        for k in ("input_size", "interpolation", "mean", "std", "crop_pct", "crop_mode")
        if k in config["pretrained_cfg"]
    }
    processor = tf.TimmWrapperImageProcessor(pretrained_cfg=metadata)
    if config.get("_orynTimmTagClassifier"):
        from oryn_engine.model_metadata import metadata_file

        processor = tag_processor(metadata_file(root / "preprocess.json"), config)
    loader = (
        tf.TimmWrapperForImageClassification
        if task == "image-classification"
        else tf.TimmWrapperModel
    )
    native = tf.TimmWrapperConfig.from_dict(copy.deepcopy(config))
    return processor, loader, native


def classify(model, processor, image, config, value, backend):
    import torch

    from oryn_engine.shared_tasks import _move

    output = model(**_move(dict(processor(images=image, return_tensors="pt")), backend))
    logits = output.logits[0].float()
    labels = config["label_names"]
    if logits.numel() != len(labels) or not bool(torch.isfinite(logits).all()):
        raise ValueError("Native timm classification head is invalid")
    probabilities = (
        torch.sigmoid(logits)
        if config.get("_orynTimmTagClassifier")
        else torch.softmax(logits, dim=-1)
    )
    scores, indices = probabilities.topk(min(value.get("topK", 5), len(labels)))
    return {
        "predictions": [
            {"label": labels[int(i)], "score": float(s)}
            for s, i in zip(scores, indices, strict=True)
        ]
    }


def load_model(root, loader, config):
    from safetensors.torch import load_file

    # Native SwinV2 derives nonpersistent position buffers in its constructor.
    # A meta-device skeleton does not materialize them when checkpoint weights
    # load. Construct on CPU, then strictly bind the original safe tensors.
    model = loader(config)
    state = {}
    for path in sorted(root.glob("*.safetensors")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Native timm weight path is invalid")
        for name, tensor in load_file(path, device="cpu").items():
            target = name if name.startswith("timm_model.") else "timm_model." + name
            if target in state:
                raise ValueError("Native timm tensor names collide")
            state[target] = tensor
    model.load_state_dict(state, strict=True)
    return model


def tag_processor(metadata, config):
    """Bind one closed five-operation data recipe, without a plugin registry."""
    import torch
    from PIL import Image
    from torchvision import transforms as tv

    validate_model_config(metadata)
    recipe = metadata.get("test")
    kinds = ["pad_to_size", "resize", "center_crop", "maybe_to_tensor", "normalize"]
    keys = [
        {"type", "size", "interpolation", "background_color"},
        {"type", "size", "interpolation", "max_size", "antialias"},
        {"type", "size"},
        {"type"},
        {"type", "mean", "std"},
    ]
    if (
        not set(metadata) <= {"pre", "test", "val", "train"}
        or not isinstance(recipe, list)
        or len(recipe) != 5
        or any(
            not isinstance(op, dict) or op.get("type") != kind or set(op) != fields
            for op, kind, fields in zip(recipe, kinds, keys, strict=True)
        )
    ):
        raise ValueError("Tag preprocessing recipe is outside the closed binding")
    pad, resize, crop, _, norm = recipe
    size = pad["size"]
    target = config["pretrained_cfg"]["input_size"][1:]
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(v) is not int or not 8 <= v <= 1024 for v in size)
        or pad["background_color"] not in {"white", "black"}
        or pad["interpolation"] != "bilinear"
        or resize["interpolation"] != "bicubic"
        or resize["antialias"] is not True
        or resize["max_size"] is not None
        or crop["size"] != target
        or resize["size"] not in (target[0], target)
    ):
        raise ValueError("Tag geometry or interpolation is not admitted")
    for field in ("mean", "std"):
        values = norm[field]
        if (
            not isinstance(values, list)
            or len(values) != 3
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or field == "std"
                and not 0 < v <= 10
                or abs(v - config["pretrained_cfg"][field][i]) > 1e-5
                for i, v in enumerate(values)
            )
        ):
            raise ValueError("Tag normalization differs from the native graph")

    def pad_image(image):
        # Native PadToSize resizes to fit and then centers on the declared canvas.
        width, height = size
        ratio = min(width / image.width, height / image.height)
        resized = image.resize(
            (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
            Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (width, height), color=pad["background_color"])
        canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
        return canvas

    pipeline = tv.Compose(
        [
            pad_image,
            tv.Resize(resize["size"], interpolation=tv.InterpolationMode.BICUBIC, antialias=True),
            tv.CenterCrop(crop["size"]),
            tv.ToTensor(),
            tv.Normalize(norm["mean"], norm["std"]),
        ]
    )

    class Processor:
        def __call__(self, *, images, return_tensors="pt"):
            if return_tensors != "pt" or not isinstance(images, Image.Image):
                raise ValueError("Tag input must be one bounded owned image")
            return {"pixel_values": torch.stack([pipeline(images)])}

    return Processor()
