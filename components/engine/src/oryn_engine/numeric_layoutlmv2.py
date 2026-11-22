"""Complete native LayoutLMv2 document QA over the closed numeric archive reader.

The pinned Transformers graph builds its visual backbone through detectron2,
which is not part of any runtime pack. This binding owns that ResNeXt-FPN
backbone in plain PyTorch with the exact reviewed detectron2 configuration and
parameter names, and reuses every other native LayoutLMv2 module and forward.
"""

from __future__ import annotations

import math

from oryn_engine.model_metadata import validate_model_config

TASK = "document-question-answering"
ARCHITECTURES = {(TASK, "layoutlmv2"): "LayoutLMv2ForQuestionAnswering"}
# The complete reviewed visual-backbone declaration; any other detectron2
# meta-architecture, depth, normalization or feature set is not admitted.
DETECTRON2 = {
    "MODEL.ANCHOR_GENERATOR.SIZES": [[32], [64], [128], [256], [512]],
    "MODEL.BACKBONE.NAME": "build_resnet_fpn_backbone",
    "MODEL.FPN.IN_FEATURES": ["res2", "res3", "res4", "res5"],
    "MODEL.MASK_ON": True,
    "MODEL.PIXEL_STD": [57.375, 57.12, 58.395],
    "MODEL.POST_NMS_TOPK_TEST": 1000,
    "MODEL.RESNETS.ASPECT_RATIOS": [[0.5, 1.0, 2.0]],
    "MODEL.RESNETS.DEPTH": 101,
    "MODEL.RESNETS.NUM_GROUPS": 32,
    "MODEL.RESNETS.OUT_FEATURES": ["res2", "res3", "res4", "res5"],
    "MODEL.RESNETS.SIZES": [[32], [64], [128], [256], [512]],
    "MODEL.RESNETS.STRIDE_IN_1X1": False,
    "MODEL.RESNETS.WIDTH_PER_GROUP": 8,
    "MODEL.ROI_BOX_HEAD.NAME": "FastRCNNConvFCHead",
    "MODEL.ROI_BOX_HEAD.NUM_FC": 2,
    "MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION": 14,
    "MODEL.ROI_HEADS.IN_FEATURES": ["p2", "p3", "p4", "p5"],
    "MODEL.ROI_HEADS.NAME": "StandardROIHeads",
    "MODEL.ROI_HEADS.NUM_CLASSES": 5,
    "MODEL.ROI_MASK_HEAD.NAME": "MaskRCNNConvUpsampleHead",
    "MODEL.ROI_MASK_HEAD.NUM_CONV": 4,
    "MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION": 7,
    "MODEL.RPN.IN_FEATURES": ["p2", "p3", "p4", "p5", "p6"],
    "MODEL.RPN.POST_NMS_TOPK_TRAIN": 1000,
    "MODEL.RPN.PRE_NMS_TOPK_TEST": 1000,
    "MODEL.RPN.PRE_NMS_TOPK_TRAIN": 2000,
}
# detectron2 defaults not overridden above (BGR mean, stem, stage widths, FPN).
PIXEL_MEAN = [103.53, 116.28, 123.675]
STAGE_BLOCKS = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}
STEM, RES2, FPN = 64, 256, 256
FROZEN_BN_EPS = 1e-5
POSITIONS = "layoutlmv2.embeddings.position_ids"
PIXELS = ("layoutlmv2.visual.pixel_mean", "layoutlmv2.visual.pixel_std")


def supports(config, task):
    return (task, config.get("model_type")) in ARCHITECTURES


def visual(config):
    """Reviewed backbone geometry: (stage blocks, groups, width per group)."""
    args = config.get("detectron2_config_args")
    if not isinstance(args, dict):
        raise ValueError("NUMERIC_LAYOUTLMV2_BACKBONE")
    variable = {"MODEL.RESNETS.DEPTH", "MODEL.RESNETS.NUM_GROUPS", "MODEL.RESNETS.WIDTH_PER_GROUP"}
    if set(args) != set(DETECTRON2) or any(
        args[k] != v or type(args[k]) is not type(v)
        for k, v in DETECTRON2.items()
        if k not in variable
    ):
        raise ValueError("NUMERIC_LAYOUTLMV2_BACKBONE")
    depth, groups, width = (args[k] for k in sorted(variable))
    if (
        type(depth) is not int
        or depth not in STAGE_BLOCKS
        or type(groups) is not int
        or not 1 <= groups <= 64
        or type(width) is not int
        or not 1 <= width <= 64
    ):
        raise ValueError("NUMERIC_LAYOUTLMV2_BACKBONE")
    return STAGE_BLOCKS[depth], groups, width


def validate_config(c):
    validate_model_config(c)
    if (
        c.get("model_type") != "layoutlmv2"
        or c.get("architectures") != ["LayoutLMv2ForQuestionAnswering"]
        or c.get("auto_map")
        or c.get("quantization")
        or c.get("quantization_config")
        or c.get("is_decoder", False)
        or c.get("add_cross_attention", False)
    ):
        raise ValueError("NUMERIC_LAYOUTLMV2_CONFIGURATION")
    for key, low, high in (
        ("hidden_size", 8, 2048),
        ("intermediate_size", 8, 8192),
        ("num_hidden_layers", 1, 48),
        ("num_attention_heads", 1, 32),
        ("vocab_size", 8, 300000),
        ("max_position_embeddings", 8, 4096),
        ("max_2d_position_embeddings", 8, 4096),
        ("coordinate_size", 1, 1024),
        ("shape_size", 1, 1024),
        ("rel_pos_bins", 2, 256),
        ("rel_2d_pos_bins", 2, 256),
        ("max_rel_pos", 2, 4096),
        ("max_rel_2d_pos", 2, 4096),
        ("type_vocab_size", 1, 16),
    ):
        if type(c.get(key)) is not int or not low <= c[key] <= high:
            raise ValueError("NUMERIC_LAYOUTLMV2_CONFIGURATION_LIMIT")
    pool = c.get("image_feature_pool_shape")
    if (
        c["hidden_size"] % c["num_attention_heads"]
        # Left/upper/right/lower coordinates plus height/width form the layout embedding.
        or 4 * c["coordinate_size"] + 2 * c["shape_size"] != c["hidden_size"]
        or c.get("hidden_act", "gelu") != "gelu"
        or type(c.get("fast_qkv", True)) is not bool
        or any(
            c.get(k, True) is not True
            for k in (
                "has_relative_attention_bias",
                "has_spatial_attention_bias",
                "has_visual_segment_embedding",
            )
        )
        or type(c.get("convert_sync_batchnorm", True)) is not bool
        or c.get("num_labels", 2) != 2
        or not isinstance(pool, list)
        or pool != [7, 7, FPN]
        or type(c.get("pad_token_id", 0)) is not int
        or not 0 <= c.get("pad_token_id", 0) < c["vocab_size"]
    ):
        raise ValueError("NUMERIC_LAYOUTLMV2_CONFIGURATION")
    visual(c)


def backbone_shapes(config, prefix="layoutlmv2.visual.backbone"):
    blocks, groups, width = visual(config)
    out = {}

    def conv_norm(name, rows, columns, kernel):
        out[name + ".weight"] = [rows, columns, kernel, kernel]
        for part in ("weight", "bias", "running_mean", "running_var"):
            out[f"{name}.norm.{part}"] = [rows]

    conv_norm(prefix + ".bottom_up.stem.conv1", STEM, 3, 7)
    source, target, bottleneck = STEM, RES2, groups * width
    for stage, count in enumerate(blocks, 2):
        for index in range(count):
            name = f"{prefix}.bottom_up.res{stage}.{index}"
            if source != target:
                conv_norm(name + ".shortcut", target, source, 1)
            conv_norm(name + ".conv1", bottleneck, source, 1)
            conv_norm(name + ".conv2", bottleneck, bottleneck // groups, 3)
            conv_norm(name + ".conv3", target, bottleneck, 1)
            source = target
        target, bottleneck = target * 2, bottleneck * 2
    for stage, channels in zip(range(2, 6), (RES2, RES2 * 2, RES2 * 4, RES2 * 8), strict=True):
        out[f"{prefix}.fpn_lateral{stage}.weight"] = [FPN, channels, 1, 1]
        out[f"{prefix}.fpn_lateral{stage}.bias"] = [FPN]
        out[f"{prefix}.fpn_output{stage}.weight"] = [FPN, FPN, 3, 3]
        out[f"{prefix}.fpn_output{stage}.bias"] = [FPN]
    return out


def expected_shapes(c):
    validate_config(c)
    h, heads = c["hidden_size"], c["num_attention_heads"]
    out = {}

    def linear(name, rows, columns, bias=True):
        out[name + ".weight"] = [rows, columns]
        if bias:
            out[name + ".bias"] = [rows]

    def norm(name):
        out[name + ".weight"] = [h]
        out[name + ".bias"] = [h]

    p = "layoutlmv2"
    out[p + ".visual_segment_embedding"] = [h]
    out[p + ".embeddings.word_embeddings.weight"] = [c["vocab_size"], h]
    out[p + ".embeddings.position_embeddings.weight"] = [c["max_position_embeddings"], h]
    for axis, width in (
        ("x", "coordinate_size"),
        ("y", "coordinate_size"),
        ("h", "shape_size"),
        ("w", "shape_size"),
    ):
        out[f"{p}.embeddings.{axis}_position_embeddings.weight"] = [
            c["max_2d_position_embeddings"],
            c[width],
        ]
    out[p + ".embeddings.token_type_embeddings.weight"] = [c["type_vocab_size"], h]
    norm(p + ".embeddings.LayerNorm")
    out.update(backbone_shapes(c))
    linear(p + ".visual_proj", h, FPN)
    norm(p + ".visual_LayerNorm")
    for i in range(c["num_hidden_layers"]):
        prefix = f"{p}.encoder.layer.{i}"
        if c.get("fast_qkv", True):
            out[prefix + ".attention.self.q_bias"] = [1, 1, h]
            out[prefix + ".attention.self.v_bias"] = [1, 1, h]
            linear(prefix + ".attention.self.qkv_linear", 3 * h, h, bias=False)
        else:
            for part in ("query", "key", "value"):
                linear(f"{prefix}.attention.self.{part}", h, h)
        linear(prefix + ".attention.output.dense", h, h)
        norm(prefix + ".attention.output.LayerNorm")
        linear(prefix + ".intermediate.dense", c["intermediate_size"], h)
        linear(prefix + ".output.dense", h, c["intermediate_size"])
        norm(prefix + ".output.LayerNorm")
    out[p + ".encoder.rel_pos_bias.weight"] = [heads, c["rel_pos_bins"]]
    out[p + ".encoder.rel_pos_x_bias.weight"] = [heads, c["rel_2d_pos_bins"]]
    out[p + ".encoder.rel_pos_y_bias.weight"] = [heads, c["rel_2d_pos_bins"]]
    linear(p + ".pooler.dense", h, h)
    linear("qa_outputs", 2, h)
    return out


def validate_archive(config, metadata):
    """Exact archive headers: the native graph plus fixed historical buffers."""
    tensors = metadata.get("tensors") if isinstance(metadata, dict) else None
    if (
        not isinstance(tensors, dict)
        or metadata.get("format") != "torch-numeric-zip-v1"
        or metadata.get("byteOrder") != "little"
    ):
        raise ValueError("NUMERIC_ARCHIVE_METADATA")
    shapes = expected_shapes(config)
    legacy = {POSITIONS: [1, config["max_position_embeddings"]]}
    legacy.update({name: [3, 1, 1] for name in PIXELS})
    if not set(shapes) <= set(tensors) <= set(shapes) | set(legacy):
        raise ValueError("NUMERIC_ARCHIVE_GRAPH_INCOMPLETE")
    expected = {**shapes, **{k: v for k, v in legacy.items() if k in tensors}}
    for name, value in tensors.items():
        dtypes = {"I64"} if name == POSITIONS else {"F32", "F16", "BF16"}
        if (
            not isinstance(value, dict)
            or value.get("shape") != expected[name]
            or value.get("dtype") not in dtypes
        ):
            raise ValueError("NUMERIC_ARCHIVE_GRAPH_INCOMPLETE")
    return expected


def native_model(config):
    """Native LayoutLMv2 QA graph with the owned ResNeXt-FPN visual backbone."""
    import torch
    import torch.nn.functional as F
    import transformers as tf
    from torch import nn
    from transformers.models.layoutlmv2 import modeling_layoutlmv2 as native

    blocks, groups, width = visual(config)

    class FrozenBatchNorm(nn.Module):
        def __init__(self, channels):
            super().__init__()
            for name in ("weight", "running_var"):
                self.register_buffer(name, torch.ones(channels))
            for name in ("bias", "running_mean"):
                self.register_buffer(name, torch.zeros(channels))

        def forward(self, x):
            scale = self.weight * (self.running_var + FROZEN_BN_EPS).rsqrt()
            shift = self.bias - self.running_mean * scale
            return x * scale.reshape(1, -1, 1, 1) + shift.reshape(1, -1, 1, 1)

    class ConvNorm(nn.Conv2d):
        def __init__(self, source, target, kernel, stride=1, groups=1):
            super().__init__(source, target, kernel, stride, kernel // 2, groups=groups, bias=False)
            self.norm = FrozenBatchNorm(target)

        def forward(self, x):
            return self.norm(super().forward(x))

    class Bottleneck(nn.Module):
        def __init__(self, source, target, bottleneck, stride):
            super().__init__()
            self.shortcut = ConvNorm(source, target, 1, stride) if source != target else None
            # STRIDE_IN_1X1=False: the grouped 3x3 convolution carries the stride.
            self.conv1 = ConvNorm(source, bottleneck, 1)
            self.conv2 = ConvNorm(bottleneck, bottleneck, 3, stride, groups)
            self.conv3 = ConvNorm(bottleneck, target, 1)

        def forward(self, x):
            out = F.relu_(self.conv1(x))
            out = F.relu_(self.conv2(out))
            out = self.conv3(out)
            return F.relu_(out + (x if self.shortcut is None else self.shortcut(x)))

    class Stem(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ConvNorm(3, STEM, 7, 2)

        def forward(self, x):
            return F.max_pool2d(F.relu_(self.conv1(x)), kernel_size=3, stride=2, padding=1)

    class BottomUp(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = Stem()
            source, target, bottleneck = STEM, RES2, groups * width
            self.stages = []
            for stage, count in enumerate(blocks, 2):
                layers = []
                for index in range(count):
                    stride = 2 if index == 0 and stage > 2 else 1
                    layers.append(Bottleneck(source, target, bottleneck, stride))
                    source = target
                self.add_module(f"res{stage}", nn.Sequential(*layers))
                self.stages.append(f"res{stage}")
                target, bottleneck = target * 2, bottleneck * 2

        def forward(self, x):
            x = self.stem(x)
            features = []
            for name in self.stages:
                x = getattr(self, name)(x)
                features.append(x)
            return features

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.bottom_up = BottomUp()
            for stage, channels in zip(
                range(2, 6), (RES2, RES2 * 2, RES2 * 4, RES2 * 8), strict=True
            ):
                self.add_module(f"fpn_lateral{stage}", nn.Conv2d(channels, FPN, 1))
                self.add_module(f"fpn_output{stage}", nn.Conv2d(FPN, FPN, 3, padding=1))

        def forward(self, x):
            # detectron2 FPN top-down pathway (sum fusion, nearest 2x); returns p2.
            features = self.bottom_up(x)
            previous = self.fpn_lateral5(features[3])
            for stage in (4, 3, 2):
                top_down = F.interpolate(previous, scale_factor=2.0, mode="nearest")
                previous = getattr(self, f"fpn_lateral{stage}")(features[stage - 2]) + top_down
            return self.fpn_output2(previous)

    class VisualBackbone(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.backbone = Backbone()
            mean = torch.tensor(PIXEL_MEAN).view(3, 1, 1)
            std = torch.tensor(config.detectron2_config_args["MODEL.PIXEL_STD"]).view(3, 1, 1)
            self.register_buffer("pixel_mean", mean, persistent=False)
            self.register_buffer("pixel_std", std, persistent=False)
            self.pool = nn.AdaptiveAvgPool2d(config.image_feature_pool_shape[:2])

        def forward(self, images):
            features = self.backbone((images - self.pixel_mean) / self.pixel_std)
            return self.pool(features).flatten(start_dim=2).transpose(1, 2).contiguous()

    def initialize(module, config):
        # Native initialization references detectron2; loading replaces every value.
        if isinstance(module, nn.Linear):
            module.weight.normal_(0.0, config.initializer_range)
            if module.bias is not None:
                module.bias.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.normal_(0.0, config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.weight.fill_(1.0)
            module.bias.zero_()
        elif isinstance(module, native.LayoutLMv2Embeddings):
            module.position_ids.copy_(torch.arange(module.position_ids.shape[-1]).expand(1, -1))

    class Model(native.LayoutLMv2Model):
        def __init__(self, config):
            native.LayoutLMv2PreTrainedModel.__init__(self, config)
            self.config = config
            self.has_visual_segment_embedding = config.has_visual_segment_embedding
            self.embeddings = native.LayoutLMv2Embeddings(config)
            self.visual = VisualBackbone(config)
            self.visual_proj = nn.Linear(config.image_feature_pool_shape[-1], config.hidden_size)
            self.visual_segment_embedding = nn.Parameter(torch.zeros(config.hidden_size))
            self.visual_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.visual_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.encoder = native.LayoutLMv2Encoder(config)
            self.pooler = native.LayoutLMv2Pooler(config)
            self.post_init()

        @torch.no_grad()
        def _init_weights(self, module):
            initialize(module, self.config)

    class QuestionAnswering(native.LayoutLMv2ForQuestionAnswering):
        def __init__(self, config):
            native.LayoutLMv2PreTrainedModel.__init__(self, config)
            self.num_labels = config.num_labels
            config.has_visual_segment_embedding = True
            self.layoutlmv2 = Model(config)
            self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
            self.post_init()

        @torch.no_grad()
        def _init_weights(self, module):
            initialize(module, self.config)

    native_config = tf.LayoutLMv2Config(**{k: v for k, v in config.items() if k != "model_type"})
    native_config._attn_implementation = "eager"
    if not math.isclose(native_config.layer_norm_eps, config.get("layer_norm_eps", 1e-12)):
        raise ValueError("NUMERIC_LAYOUTLMV2_CONFIGURATION")
    return QuestionAnswering(native_config)


def load_native(root, config):
    import torch

    from oryn_engine.numeric_torch_archive import inspect_archive, read_weights

    validate_config(config)
    path = root / "pytorch_model.bin"
    if path.is_symlink() or not path.is_file():
        raise ValueError("NUMERIC_ARCHIVE_SCOPE")
    with path.open("rb") as stream:

        def read_range(start, length):
            stream.seek(start)
            return stream.read(length)

        metadata = inspect_archive(read_range, path.stat().st_size)
    expected = validate_archive(config, metadata)
    model = native_model(config).float()
    if {k: list(v.shape) for k, v in model.state_dict().items()} != expected_shapes(config):
        raise ValueError("NUMERIC_ARCHIVE_NATIVE_GRAPH_CHANGED")
    state = read_weights(path, expected_shapes=expected, maximum_bytes=path.stat().st_size)
    if POSITIONS in state and not torch.equal(
        state.pop(POSITIONS), torch.arange(config["max_position_embeddings"]).expand(1, -1)
    ):
        raise ValueError("NUMERIC_ARCHIVE_POSITION_BUFFER")
    for name, reference in zip(
        PIXELS, (model.layoutlmv2.visual.pixel_mean, model.layoutlmv2.visual.pixel_std), strict=True
    ):
        # Historical exports persisted the fixed normalization; it must equal the config.
        if name in state and not torch.allclose(state.pop(name).float(), reference, atol=1e-4):
            raise ValueError("NUMERIC_LAYOUTLMV2_PIXEL_NORMALIZATION")
    for name, value in state.items():
        state[name] = value.float().contiguous()
        if not bool(torch.isfinite(state[name]).all()):
            raise ValueError("NUMERIC_ARCHIVE_NONFINITE")
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


def load_processor(root, config):
    import transformers as tf

    common = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = tf.AutoTokenizer.from_pretrained(root, **common)
    if not tokenizer.is_fast or len(tokenizer) > config["vocab_size"]:
        raise ValueError("NUMERIC_LAYOUTLMV2_TOKENIZER")
    image_processor = tf.LayoutLMv2ImageProcessor.from_pretrained(root, apply_ocr=False, **common)
    if image_processor.size != {"height": 224, "width": 224}:
        raise ValueError("NUMERIC_LAYOUTLMV2_PROCESSOR")
    return tokenizer, image_processor
