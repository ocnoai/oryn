"""Closed foreground segmentation over complete native BiRefNet graphs."""

from __future__ import annotations

from oryn_engine.model_metadata import preflight_safetensors, validate_model_config

FAMILY = "birefnet"
ARCHITECTURE = "BiRefNet"
IMAGE_SIZE = 1024
LABELS = {"0": "background", "1": "foreground"}
ALIASES = {
    "AutoConfig": "BiRefNet_config.BiRefNetConfig",
    "AutoModelForImageSegmentation": "birefnet.BiRefNet",
}
PIPELINES = {
    "image-segmentation": {"pt": ["AutoModelForImageSegmentation"], "tf": [], "type": "image"}
}


def normalized_config(config, task):
    if task == "image-segmentation" and config.get("architectures") == [ARCHITECTURE]:
        for name, expected in (("auto_map", ALIASES), ("custom_pipelines", PIPELINES)):
            if config.get(name) == expected:
                config = {k: v for k, v in config.items() if k != name}
        if "model_type" not in config:
            config = config | {"model_type": FAMILY}
        if "id2label" not in config:
            config = config | {"id2label": LABELS}
        validate_config(config)
    return config


def validate_config(config):
    validate_model_config(config)
    if (
        set(config) - {"_name_or_path", "architectures", "bb_pretrained", "model_type", "id2label"}
        or config.get("architectures") != [ARCHITECTURE]
        or config.get("model_type") != FAMILY
        or config.get("bb_pretrained") is not False
        or config.get("id2label") != LABELS
        or "_name_or_path" in config
        and not isinstance(config["_name_or_path"], str)
    ):
        raise ValueError("BIREFNET_CONFIGURATION")


def native_model(variant):
    from oryn_engine.native_birefnet.model import BiRefNet

    return BiRefNet(variant)


def backbone_variant(shapes):
    variant = {(192, 3, 4, 4): "swin_v1_l", (96, 3, 4, 4): "swin_v1_t"}.get(
        shapes.get("bb.patch_embed.proj.weight")
    )
    if variant is None:
        raise ValueError("BIREFNET_BACKBONE")
    return variant


def validate_weight_shapes(config, shapes):
    from accelerate import init_empty_weights

    validate_config(config)
    variant = backbone_variant(shapes)
    with init_empty_weights(include_buffers=False):
        model = native_model(variant)
    if shapes != {k: tuple(v.shape) for k, v in model.state_dict().items()}:
        raise ValueError("BIREFNET_WEIGHTS_INCOMPLETE")
    return model


def load(root, config):
    import torch
    from safetensors.torch import load_file

    model = validate_weight_shapes(config, preflight_safetensors(root))
    expected = model.state_dict()
    state = {}
    for path in sorted(root.glob("*.safetensors")):
        values = load_file(path)
        if state.keys() & values.keys():
            raise ValueError("BIREFNET_DUPLICATE_WEIGHTS")
        state.update(values)
    for name, value in state.items():
        if name.endswith(".relative_position_index"):
            if value.dtype != torch.int64 or not torch.equal(value, expected[name]):
                raise ValueError("BIREFNET_POSITION_INDEX")
        elif name.endswith(".num_batches_tracked"):
            if value.dtype != torch.int64 or bool((value < 0).any()):
                raise ValueError("BIREFNET_BATCH_COUNTER")
        else:
            if not value.is_floating_point() or not bool(torch.isfinite(value.float()).all()):
                raise ValueError("BIREFNET_WEIGHT_VALUES")
            if name.endswith(".running_var") and bool((value < 0).any()):
                raise ValueError("BIREFNET_BATCH_VARIANCE")
            state[name] = value.float()
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


def preprocess(image):
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor

    return Compose(
        [
            Resize((IMAGE_SIZE, IMAGE_SIZE)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )(image.convert("RGB"))


def segmentation(model, image, backend):
    import numpy as np
    import torch
    from torchvision.transforms.functional import to_pil_image

    prediction = model(preprocess(image).unsqueeze(0).to(backend))[-1]
    if prediction.ndim != 4 or tuple(prediction.shape) != (1, 1, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("BIREFNET_OUTPUT_SHAPE")
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("BIREFNET_OUTPUT_VALUES")
    # Preserve the publisher's sigmoid -> 8-bit alpha -> original-size resize.
    # The shared semantic-mask contract then assigns the 0.5 foreground boundary.
    alpha = to_pil_image(prediction.sigmoid()[0].float().cpu()).resize(image.size)
    return torch.from_numpy((np.asarray(alpha) >= 128).astype(np.int64))
