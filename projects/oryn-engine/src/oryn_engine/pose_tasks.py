"""Native labeled pose estimators over caller-supplied person boxes."""

from __future__ import annotations

TASK_CONTRACTS = {"pose-estimation": "pose-estimation-v1"}
TASK_FAMILIES = {"pose-estimation": {"vitpose", "sapiens2"}}


def validate_config(config):
    if config.get("model_type") == "sapiens2":
        from oryn_engine.sapiens2_pose import validate_config as validate_sapiens2

        validate_sapiens2(config)
        return
    labels = config.get("id2label", {})
    backbone = config.get("backbone_config", {})
    if (
        not isinstance(labels, dict)
        or not 1 <= len(labels) <= 64
        or set(labels) != {str(i) for i in range(len(labels))}
        or len(set(labels.values())) != len(labels)
        or any(
            not isinstance(v, str) or not 1 <= len(v) <= 128 or v.startswith("LABEL_")
            for v in labels.values()
        )
        or backbone.get("model_type") != "vitpose_backbone"
        or config.get("backbone") is not None
        or config.get("backbone_kwargs") is not None
        or config.get("use_pretrained_backbone")
        or config.get("use_timm_backbone")
        or type(backbone.get("num_experts", 1)) is not int
        or not 1 <= backbone.get("num_experts", 1) <= 6
    ):
        raise ValueError("Native pose labels/backbone are outside the reviewed binding")


def load_components(root, config, common):
    if config.get("model_type") == "sapiens2":
        from oryn_engine.sapiens2_pose import load_components as load_sapiens2

        return load_sapiens2(root, config, common)
    from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor

    from oryn_engine.model_metadata import metadata_file, validate_model_config

    metadata = metadata_file(root / "preprocessor_config.json")
    validate_model_config(metadata)
    if metadata.get("image_processor_type") != "VitPoseImageProcessor":
        raise ValueError("Pose preprocessing identity is not admitted")
    size = metadata.get("size", {"height": 256, "width": 192})
    if size != {"height": 256, "width": 192}:
        raise ValueError("Pose crop geometry is not admitted")
    processor = VitPoseImageProcessor.from_pretrained(root, **common)
    return processor, VitPoseForPoseEstimation


def execute(model, processor, config, image, value, backend):
    import torch

    from oryn_engine.shared_tasks import _move

    width, height = image.size
    boxes = [
        [b[0] * width, b[1] * height, (b[2] - b[0]) * width, (b[3] - b[1]) * height]
        for b in value["boxes"]
    ]
    labels = config["id2label"]
    if config.get("model_type") == "sapiens2":
        from oryn_engine import sapiens2_pose

        predictions = sapiens2_pose.predictions(model, processor, config, image, boxes, backend)
        labels = sapiens2_pose.labels(config)
    else:
        inputs = _move(dict(processor(images=image, boxes=[boxes], return_tensors="pt")), backend)
        if inputs["pixel_values"].shape != (len(boxes), 3, 256, 192):
            raise ValueError("Native person crop shape is invalid")
        if config["backbone_config"].get("num_experts", 1) > 1:
            inputs["dataset_index"] = torch.zeros(len(boxes), dtype=torch.long, device=backend)
        output = model(**inputs)
        if output.heatmaps.shape[0:2] != (len(boxes), len(labels)) or not bool(
            torch.isfinite(output.heatmaps).all()
        ):
            raise ValueError("Native pose heatmaps are invalid")
        predictions = processor.post_process_pose_estimation(output, boxes=[boxes])[0]
    if len(predictions) != len(boxes):
        raise ValueError("Native pose count is invalid")
    poses = []
    for source, prediction in zip(value["boxes"], predictions, strict=True):
        points = []
        for xy, label, score in zip(
            prediction["keypoints"], prediction["labels"], prediction["scores"], strict=True
        ):
            x, y = map(float, xy)
            identifier = int(label)
            points.append(
                {
                    "id": identifier,
                    "label": labels[str(identifier)],
                    "x": x,
                    "y": y,
                    "confidence": float(score),
                    "visible": 0 <= x < width and 0 <= y < height,
                }
            )
        poses.append(
            {
                "box": [
                    source[0] * width,
                    source[1] * height,
                    source[2] * width,
                    source[3] * height,
                ],
                "keypoints": points,
            }
        )
    return {"width": width, "height": height, "poses": poses}
