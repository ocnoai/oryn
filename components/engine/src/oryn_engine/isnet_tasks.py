"""Complete IS-Net foreground graphs and the declared RMBG image pipeline."""

from __future__ import annotations

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

FAMILY = "oryn_isnet"
ARCHITECTURE = "ISNetDIS"
IMAGE_SIZE = 1024
LABELS = {"0": "background", "1": "foreground"}
ALIASES = {
    "AutoConfig": "MyConfig.RMBGConfig",
    "AutoModelForImageSegmentation": "briarmbg.BriaRMBG",
}
PIPELINES = {
    "image-segmentation": {
        "impl": "MyPipe.RMBGPipe",
        "pt": ["AutoModelForImageSegmentation"],
        "tf": [],
        "type": "image",
    }
}


def normalized_config(config, task):
    if task != "image-segmentation":
        return config
    if config.get("architectures") == ["BriaRMBG"]:
        if config.get("model_type") != "SegformerForSemanticSegmentation":
            raise ValueError("ISNET_CONFIGURATION")
        config = config | {"model_type": FAMILY, "architectures": [ARCHITECTURE]}
        for name, expected in (("auto_map", ALIASES), ("custom_pipelines", PIPELINES)):
            if config.get(name) == expected:
                config = {k: v for k, v in config.items() if k != name}
    if config.get("model_type") == FAMILY:
        if "id2label" not in config:
            config = config | {"id2label": LABELS}
        validate_config(config)
    return config


def validate_config(config):
    validate_model_config(config)
    fixed = {
        "model_type": FAMILY,
        "architectures": [ARCHITECTURE],
        "in_ch": 3,
        "out_ch": 1,
        "torch_dtype": "float32",
        "id2label": LABELS,
    }
    optional = {"_name_or_path", "transformers_version"}
    if (
        set(config) - (fixed.keys() | optional)
        or any(type(config.get(k)) is not type(v) or config[k] != v for k, v in fixed.items())
        or any(k in config and not isinstance(config[k], str) for k in optional)
    ):
        raise ValueError("ISNET_CONFIGURATION")


def validate_processor(metadata):
    expected = {
        "do_normalize": True,
        "do_pad": False,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [1, 1, 1],
        "feature_extractor_type": "ImageFeatureExtractor",
        "resample": 2,
        "rescale_factor": 1 / 255,
        "size": {"height": 1024, "width": 1024},
    }
    validate_model_config(metadata)
    if set(metadata) != set(expected) or any(
        type(metadata[k]) is not type(v) or metadata[k] != v for k, v in expected.items()
    ):
        raise ValueError("ISNET_PREPROCESSOR")


def native_model():
    from oryn_engine.native_isnet.model import ISNetDIS

    return ISNetDIS(in_ch=3, out_ch=1)


def validate_weight_shapes(config, shapes):
    from accelerate import init_empty_weights

    validate_config(config)
    with init_empty_weights(include_buffers=False):
        model = native_model()
    if shapes != {k: tuple(v.shape) for k, v in model.state_dict().items()}:
        raise ValueError("ISNET_WEIGHTS_INCOMPLETE")
    return model


def load(root, config):
    import torch
    from safetensors.torch import load_file

    validate_processor(metadata_file(root / "preprocessor_config.json"))
    model = validate_weight_shapes(config, preflight_safetensors(root))
    state = {}
    for path in sorted(root.glob("*.safetensors")):
        values = load_file(path)
        if state.keys() & values.keys():
            raise ValueError("ISNET_DUPLICATE_WEIGHTS")
        state.update(values)
    for name, value in state.items():
        if name.endswith(".num_batches_tracked"):
            if value.dtype != torch.int64 or bool((value < 0).any()):
                raise ValueError("ISNET_BATCH_COUNTER")
        else:
            if not value.is_floating_point() or not bool(torch.isfinite(value.float()).all()):
                raise ValueError("ISNET_WEIGHT_VALUES")
            if name.endswith(".running_var") and bool((value < 0).any()):
                raise ValueError("ISNET_BATCH_VARIANCE")
            state[name] = value.float()
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


def preprocess(image):
    import numpy as np
    import torch
    from torch.nn.functional import interpolate
    from torchvision.transforms.functional import normalize

    pixels = torch.tensor(np.asarray(image.convert("RGB")), dtype=torch.float32).permute(2, 0, 1)
    pixels = interpolate(pixels[None], size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear")
    # The declared MyPipe pipeline retains float pixels after interpolation.
    return normalize(pixels / 255, [0.5, 0.5, 0.5], [1, 1, 1])


def segmentation(model, image, backend):
    import torch
    from torch.nn.functional import interpolate

    prediction = model(preprocess(image).to(backend))[0][0]
    if prediction.ndim != 4 or tuple(prediction.shape) != (1, 1, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("ISNET_OUTPUT_SHAPE")
    if not bool(torch.isfinite(prediction).all()) or bool(
        ((prediction < 0) | (prediction > 1)).any()
    ):
        raise ValueError("ISNET_OUTPUT_VALUES")
    # Bilinear interpolation can introduce rounding noise into a constant map;
    # min/max normalization must not amplify that noise into a foreground mask.
    if not bool(prediction.max() > prediction.min()):
        raise ValueError("ISNET_OUTPUT_RANGE")
    alpha = interpolate(prediction, size=(image.height, image.width), mode="bilinear")
    minimum, maximum = alpha.min(), alpha.max()
    if not bool(maximum > minimum):
        raise ValueError("ISNET_OUTPUT_RANGE")
    alpha = ((alpha - minimum) / (maximum - minimum) * 255).to(torch.uint8)
    return (alpha[0, 0].cpu() >= 128).long()
