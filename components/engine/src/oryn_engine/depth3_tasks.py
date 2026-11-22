"""Closed construction of complete native DA3 depth graphs and bounded image inputs."""

from __future__ import annotations

import copy

from oryn_engine.model_metadata import preflight_safetensors, validate_model_config

FAMILY = "depth_anything_3"
ARCHITECTURES = {"DepthAnything3Net", "NestedDepthAnything3Net"}
BACKBONES = {"vits": (384, 12), "vitb": (768, 12), "vitl": (1024, 24), "vitg": (1536, 40)}


def _component(value, name, path, keys):
    if (
        not isinstance(value, dict)
        or set(value) != {"__object__", *keys}
        or value["__object__"]
        != {"args": "as_params", "name": name, "path": "depth_anything_3.model." + path}
    ):
        raise ValueError("DEPTH3_COMPONENT_UNSUPPORTED")
    return {k: v for k, v in value.items() if k != "__object__"}


def _single(config):
    if not isinstance(config, dict):
        raise ValueError("DEPTH3_CONFIGURATION")
    dual = "cam_dec" in config or "cam_enc" in config
    gaussian = "gs_head" in config or "gs_adapter" in config
    keys = {"net", "head"} | ({"cam_dec", "cam_enc"} if dual else set())
    keys |= {"gs_head", "gs_adapter"} if gaussian else set()
    _component(config, "DepthAnything3Net", "da3", keys)
    net = _component(
        config["net"],
        "DinoV2",
        "dinov2.dinov2",
        {"name", "out_layers", "alt_start", "qknorm_start", "rope_start", "cat_token"},
    )
    if not isinstance(net["name"], str) or net["name"] not in BACKBONES:
        raise ValueError("DEPTH3_BACKBONE_UNSUPPORTED")
    width, layers = BACKBONES[net["name"]]
    selected = net["out_layers"]
    if (
        not isinstance(selected, list)
        or len(selected) != 4
        or any(type(i) is not int or not 0 <= i < layers for i in selected)
        or selected != sorted(set(selected))
        or type(net["alt_start"]) is not int
        or not -1 <= net["alt_start"] < layers
        or type(net["qknorm_start"]) is not int
        or type(net["rope_start"]) is not int
        or net["qknorm_start"] != net["alt_start"]
        or net["rope_start"] != net["alt_start"]
        or type(net["cat_token"]) is not bool
        or net["cat_token"] != dual
        or dual != (net["alt_start"] >= 0)
    ):
        raise ValueError("DEPTH3_ATTENTION_GEOMETRY")
    dimension = width * (2 if dual else 1)
    for key, identity, path, output in (
        ("head", "DualDPT" if dual else "DPT", "dualdpt" if dual else "dpt", 2 if dual else 1),
        *(([("gs_head", "GSDPT", "gsdpt", 38)]) if gaussian else []),
    ):
        head = _component(
            config[key], identity, path, {"dim_in", "features", "out_channels", "output_dim"}
        )
        if (
            type(head["dim_in"]) is not int
            or head["dim_in"] != dimension
            or type(head["features"]) is not int
            or not 16 <= head["features"] <= 512
            or type(head["output_dim"]) is not int
            or head["output_dim"] != output
            or not isinstance(head["out_channels"], list)
            or len(head["out_channels"]) != 4
            or any(type(i) is not int or not 16 <= i <= 2048 for i in head["out_channels"])
        ):
            raise ValueError("DEPTH3_HEAD_GEOMETRY")
    if dual:
        dec = _component(config["cam_dec"], "CameraDec", "cam_dec", {"dim_in"})
        enc = _component(config["cam_enc"], "CameraEnc", "cam_enc", {"dim_out"})
        if dec != {"dim_in": dimension} or enc != {"dim_out": width}:
            raise ValueError("DEPTH3_CAMERA_GEOMETRY")
    if gaussian:
        adapter = _component(
            config["gs_adapter"],
            "GaussianAdapter",
            "gs_adapter",
            {
                "gaussian_scale_max",
                "gaussian_scale_min",
                "pred_color",
                "pred_offset_depth",
                "pred_offset_xy",
                "sh_degree",
            },
        )
        if not dual or adapter != {
            "gaussian_scale_max": 30.0,
            "gaussian_scale_min": 1e-5,
            "pred_color": False,
            "pred_offset_depth": True,
            "pred_offset_xy": True,
            "sh_degree": 2,
        }:
            raise ValueError("DEPTH3_AUXILIARY_CONFIGURATION")


def normalized_config(config, task):
    declaration = (
        config.get("config", {}).get("__object__", {})
        if isinstance(config.get("config"), dict)
        else {}
    )
    if (
        task == "depth-estimation"
        and isinstance(declaration, dict)
        and declaration.get("path") == "depth_anything_3.model.da3"
        and isinstance(declaration.get("name"), str)
        and declaration.get("name") in ARCHITECTURES
    ):
        result = config | {"model_type": FAMILY, "architectures": [declaration["name"]]}
        validate_config(result)
        return result
    return config


def validate_config(config):
    validate_model_config(config)
    if (
        config.get("model_type") != FAMILY
        or config.get("architectures") not in [[name] for name in ARCHITECTURES]
        or set(config) != {"config", "model_name", "model_type", "architectures"}
        or not isinstance(config.get("model_name"), str)
        or not 1 <= len(config["model_name"]) <= 128
    ):
        raise ValueError("DEPTH3_CONFIGURATION")
    if config["architectures"] == ["NestedDepthAnything3Net"]:
        _component(config["config"], "NestedDepthAnything3Net", "da3", {"anyview", "metric"})
        for name in ("anyview", "metric"):
            _single(config["config"][name])
        if "cam_dec" not in config["config"]["anyview"] or "cam_dec" in config["config"]["metric"]:
            raise ValueError("DEPTH3_NESTED_CONFIGURATION")
    else:
        _single(config["config"])


def native_model(config):
    import torch.nn as nn
    from depth_anything_3.model.cam_dec import CameraDec
    from depth_anything_3.model.cam_enc import CameraEnc
    from depth_anything_3.model.da3 import DepthAnything3Net, NestedDepthAnything3Net
    from depth_anything_3.model.dinov2.dinov2 import DinoV2
    from depth_anything_3.model.dpt import DPT
    from depth_anything_3.model.dualdpt import DualDPT
    from depth_anything_3.model.gsdpt import GSDPT

    validate_config(config)

    def options(value):
        return {k: copy.deepcopy(v) for k, v in value.items() if k != "__object__"}

    def single(value):
        dual = "cam_dec" in value
        model = DepthAnything3Net(
            net=DinoV2(**options(value["net"])),
            head=(DualDPT if dual else DPT)(**options(value["head"])),
            cam_dec=CameraDec(**options(value["cam_dec"])) if dual else None,
            cam_enc=CameraEnc(**options(value["cam_enc"])) if dual else None,
        )
        if "gs_head" in value:
            # Preserve every learned auxiliary tensor. The non-learned Gaussian
            # renderer is outside the depth contract and cannot be invoked.
            model.gs_head = GSDPT(**options(value["gs_head"]))
        return model

    class BoundNested(NestedDepthAnything3Net):
        def __init__(self, value):
            nn.Module.__init__(self)
            self.da3 = single(value["anyview"])
            self.da3_metric = single(value["metric"])

    return (
        BoundNested(config["config"])
        if config["architectures"] == ["NestedDepthAnything3Net"]
        else single(config["config"])
    )


def _weights(config, shapes):
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    names = {}
    for key, shape in shapes.items():
        target = key.removeprefix("model.")
        if target in names or target not in expected or shape != expected[target]:
            raise ValueError("DEPTH3_WEIGHT_GEOMETRY")
        names[target] = key
    groups = {}
    for name, value in model.named_parameters(remove_duplicate=False):
        if name in expected:
            groups.setdefault(id(value), []).append(name)
    for name, value in model.named_buffers(remove_duplicate=False):
        if name in expected:
            groups.setdefault(id(value), []).append(name)
    aliases = {}
    for group in groups.values():
        present = sorted(set(group) & names.keys())
        if not present:
            raise ValueError("DEPTH3_WEIGHTS_INCOMPLETE")
        for name in group:
            if name not in names:
                aliases[name] = present[0]
    if names.keys() | aliases.keys() != expected.keys():
        raise ValueError("DEPTH3_WEIGHTS_INCOMPLETE")
    return model, names, aliases, list(groups.values())


def validate_weight_shapes(config, shapes):
    _weights(config, shapes)


def load(root, config):
    import torch
    from safetensors.torch import load_file

    model, names, aliases, groups = _weights(config, preflight_safetensors(root))
    raw = {}
    for path in sorted(root.glob("*.safetensors")):
        state = load_file(path, device="cpu")
        if raw.keys() & state.keys() or any(
            not value.is_floating_point() or not bool(torch.isfinite(value).all())
            for value in state.values()
        ):
            raise ValueError("DEPTH3_WEIGHT_VALUES")
        raw.update(state)
    weights = {name: raw[key].float() for name, key in names.items()}
    for group in groups:
        present = [name for name in group if name in weights]
        if any(not torch.equal(weights[name], weights[present[0]]) for name in present[1:]):
            raise ValueError("DEPTH3_ALIASED_WEIGHT_CONFLICT")
    weights.update({name: weights[source] for name, source in aliases.items()})
    model.load_state_dict(weights, strict=True, assign=True)
    return model.eval()


def pixels(image):
    import cv2
    import numpy as np
    import torch

    # Reviewed upper_bound_resize path: OpenCV area/cubic resize, nearest
    # patch multiple (ties up), then native ImageNet channel normalization.
    image = image.convert("RGB")
    width, height = image.size
    scale = 504 / max(width, height)
    resized = max(1, round(width * scale)), max(1, round(height * scale))
    raw = np.asarray(image)
    if resized != (width, height):
        raw = cv2.resize(
            raw, resized, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        )
    multiple = tuple((axis // 14 + int(axis % 14 >= 7)) * 14 for axis in resized)
    if min(multiple) < 14 or max(multiple) > 504:
        raise ValueError("DEPTH3_IMAGE_GEOMETRY")
    if multiple != resized:
        interpolation = (
            cv2.INTER_CUBIC
            if any(a > b for a, b in zip(multiple, resized, strict=True))
            else cv2.INTER_AREA
        )
        raw = cv2.resize(raw, multiple, interpolation=interpolation)
    tensor = torch.from_numpy(raw.copy()).permute(2, 0, 1).float() / 255
    mean, std = tensor.new_tensor([0.485, 0.456, 0.406]), tensor.new_tensor([0.229, 0.224, 0.225])
    return ((tensor - mean[:, None, None]) / std[:, None, None])[None, None]


def execute(model, image, backend):
    import torch

    with torch.inference_mode():
        output = model(pixels(image).to(backend), infer_gs=False, use_ray_pose=False)
    depth = output.depth
    if (
        depth.ndim != 4
        or depth.shape[:2] != (1, 1)
        or not bool(torch.isfinite(depth).all())
        or bool((depth < 0).any())
    ):
        raise ValueError("DEPTH3_OUTPUT_INVALID")
    depth = depth[0, 0].float().cpu()
    return {
        "width": int(depth.shape[-1]),
        "height": int(depth.shape[-2]),
        "minimum": float(depth.min()),
        "maximum": float(depth.max()),
        "mean": float(depth.mean()),
    }
