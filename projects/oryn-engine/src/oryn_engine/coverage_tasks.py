"""Bounded table, document, interest-point and temporal task behavior.

Family identities are matched against generated pinned Auto mappings by shared_tasks.
Repository configuration never chooses an import, decoder, command or remote processor.
"""

from __future__ import annotations

import math
from pathlib import Path

TASK_CONTRACTS = {
    "visual-document-retrieval": "visual-document-retrieval-v1",
    "table-question-answering": "table-question-answering-v1",
    "document-question-answering": "document-question-answering-v1",
    "keypoint-detection": "keypoint-detection-v1",
    "video-classification": "video-classification-v1",
    "video-text-to-text": "video-question-answering-v1",
}
TASK_FAMILIES = {
    "visual-document-retrieval": {
        "colqwen2",
        "colqwen3",
        "neomme",
        "qwen3_5",
        "jina_embeddings_v4",
    },
    "table-question-answering": {"tapas", "bart"},
    "document-question-answering": {"layoutlm", "layoutlmv3"},
    "keypoint-detection": {"superpoint"},
    "video-classification": {"videomae", "timesformer", "vivit", "vjepa2"},
    "video-text-to-text": {
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_5",
        "llava_next_video",
        "videollama3_qwen2",
        "oryn_llava_video_qwen2",
    },
}
VIDEO_TASKS = frozenset({"video-classification", "video-text-to-text"})
IMAGE_TASKS = frozenset({"document-question-answering", "keypoint-detection"})


def validate_relations(task, value):
    if task == "table-question-answering":
        columns = value["columns"]
        if len(set(c.strip() for c in columns)) != len(columns) or any(
            len(row) != len(columns) for row in value["rows"]
        ):
            raise ValueError("Table columns must be distinct and rows rectangular")
    if task == "document-question-answering" and (
        len(value["words"]) != len(value["boxes"])
        or any(x0 > x1 or y0 > y1 for x0, y0, x1, y1 in value["boxes"])
    ):
        raise ValueError("Each OCR word requires an ordered normalized page box")
    if task in VIDEO_TASKS:
        start = value.setdefault("startSeconds", 0)
        if start >= 30 or start + value.get("durationSeconds", 0) > 30:
            raise ValueError("Selected clip exceeds video duration limit")
    for key, default in {
        "keypoint-detection": {"threshold": 0.005, "maxKeypoints": 256},
        "video-classification": {"topK": 5},
        "video-text-to-text": {"maxTokens": 128},
    }.get(task, {}).items():
        value.setdefault(key, default)


def validate_config(task, config):
    if task == "video-text-to-text" and config.get("model_type") == "qwen3_vl":
        from oryn_engine.qwen_retrieval import validate_geometry

        validate_geometry(config)
    if task == "video-text-to-text" and config.get("model_type") == "llava_next_video":
        from oryn_engine.llava_video import validate_config as validate_llava

        validate_llava(config)
    if task == "visual-document-retrieval":
        if config.get("model_type") == "jina_embeddings_v4":
            from oryn_engine.jina4_tasks import validate_config as validate_retrieval
        elif config.get("model_type") == "qwen3_5":
            from oryn_engine.colqwen35_tasks import validate_config as validate_retrieval
        elif config.get("model_type") == "colqwen3":
            from oryn_engine.colqwen3_tasks import native_config as validate_retrieval
        elif config.get("model_type") == "neomme":
            from oryn_engine.neomme_retrieval import validate_config as validate_retrieval
        else:
            from oryn_engine.visual_retrieval import validate_config as validate_retrieval

        validate_retrieval(config)
    if task == "table-question-answering" and config["model_type"] == "tapas":
        table_shape_limits(config)
        labels = config.get("aggregation_labels", {})
        if labels and (
            not isinstance(labels, dict)
            or len(labels) != config.get("num_aggregation_labels")
            or set(labels) != {str(i) for i in range(len(labels))}
            or not set(labels.values()) <= {"NONE", "SUM", "AVERAGE", "COUNT"}
            or labels.get("0") != "NONE"
        ):
            raise ValueError("Table aggregation semantics are unavailable")
    if task == "video-classification":
        video_frame_count(config)
        labels = config.get("id2label")
        if not isinstance(labels, dict) or not 2 <= len(labels) <= 4096:
            raise ValueError("Video classification labels are unavailable")
        if set(labels) != {str(i) for i in range(len(labels))} or any(
            not isinstance(v, str) or not v.strip() or len(v) > 1024 for v in labels.values()
        ):
            raise ValueError("Video classification label mapping is invalid")


def tapex_options(metadata):
    """The reviewed TAPEX table serializer binding declared by tokenizer metadata."""
    if (
        not isinstance(metadata, dict)
        or metadata.get("tokenizer_class") != "TapexTokenizer"
        or type(metadata.get("do_lower_case", True)) is not bool
        or type(metadata.get("max_cell_length", 15)) is not int
        or not 1 <= metadata.get("max_cell_length", 15) <= 64
    ):
        raise ValueError("TAPEX preprocessing binding is unavailable")
    return metadata.get("do_lower_case", True), metadata.get("max_cell_length", 15)


def table_shape_limits(config):
    types = config.get("type_vocab_sizes", [3, 256, 256, 2, 256, 256, 10])
    if not isinstance(types, list) or len(types) != 7:
        raise ValueError("TAPAS table geometry is invalid")
    values = (config.get("max_num_rows", 64), config.get("max_num_columns", 32), types[2], types[1])
    if any(type(v) is not int or not 2 <= v <= 4096 for v in values):
        raise ValueError("TAPAS table geometry is invalid")
    # Zero denotes question/header positions. Native reductions clamp out-of-range
    # row/column IDs, which would silently merge distinct input cells.
    return min(128, values[0] - 1, values[2] - 1), min(32, values[1] - 1, values[3] - 1)


def video_frame_count(config):
    key = "frames_per_clip" if config.get("model_type") == "vjepa2" else "num_frames"
    count = config.get(key)
    if type(count) is not int or not 2 <= count <= 32:
        raise ValueError("Native temporal frame count exceeds contract")
    if config.get("model_type") == "vjepa2":
        tubelet = config.get("tubelet_size", 2)
        if type(tubelet) is not int or not 1 <= tubelet <= 4 or count % tubelet:
            raise ValueError("Native temporal tubelet geometry is invalid")
    return count


def load_components(root, task, config, common):
    import transformers as tf

    family = config["model_type"]
    if task == "visual-document-retrieval":
        if family == "neomme":
            from oryn_engine.neomme_retrieval import load_components

            return load_components(root, config, common)
        from oryn_engine.visual_retrieval import load_processor

        return load_processor(root, config, common), tf.ColQwen2ForRetrieval
    if task == "table-question-answering":
        if family == "tapas":
            return tf.TapasTokenizer.from_pretrained(
                root, **common
            ), tf.AutoModelForTableQuestionAnswering
        from oryn_engine.model_metadata import metadata_file

        lower, cell_length = tapex_options(metadata_file(root / "tokenizer_config.json"))
        # Transformers 5 consolidated BART's byte BPE into RobertaTokenizer and
        # removed TAPEX's table wrapper. The reviewed serializer below owns the
        # table behavior; its native byte tokenizer remains in the shared build.
        tokenizer = tf.RobertaTokenizer.from_pretrained(root, **common)
        tokenizer._oryn_lower_case = lower
        tokenizer._oryn_max_cell_length = cell_length
        return tokenizer, tf.AutoModelForSeq2SeqLM
    if task == "document-question-answering":
        tokenizer = tf.AutoTokenizer.from_pretrained(root, **common)
        if not tokenizer.is_fast:
            raise ValueError("Document word grounding requires a fast tokenizer")
        image_processor = (
            tf.AutoImageProcessor.from_pretrained(root, apply_ocr=False, **common)
            if family == "layoutlmv3"
            else None
        )
        return (tokenizer, image_processor), tf.AutoModelForDocumentQuestionAnswering
    if task == "keypoint-detection":
        return tf.AutoImageProcessor.from_pretrained(
            root, **common
        ), tf.AutoModelForKeypointDetection
    if task == "video-classification":
        if family == "vjepa2":
            return tf.AutoVideoProcessor.from_pretrained(
                root, **common
            ), tf.AutoModelForVideoClassification
        return tf.AutoImageProcessor.from_pretrained(
            root, **common
        ), tf.AutoModelForVideoClassification
    raise ValueError("Task behavior is unavailable")


def sample_video(path: Path, count: int, start: float, duration: float | None):
    """Decode only a local bounded container; uniformly sample presentation time.

    Two passes retain at most count RGB frames, independently cap decoded work, and
    reject dishonest duration/frame-count metadata. The isolated worker is killable.
    """
    import av
    import numpy as np

    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 64 * 1024**2:
        raise ValueError("Video artifact exceeds byte bounds")
    if not 2 <= count <= 32:
        raise ValueError("Temporal sample count exceeds bounds")

    def frames():
        with path.open("rb") as source, av.open(source, mode="r") as container:
            if not set(container.format.name.split(",")) & {"mov", "mp4", "matroska", "webm"}:
                raise ValueError("Video container is not admitted")
            if len(container.streams.video) != 1:
                raise ValueError("Exactly one video stream is required")
            stream = container.streams.video[0]
            stream.thread_type = "NONE"
            stream.codec_context.thread_count = 1
            if not 1 <= stream.width <= 1920 or not 1 <= stream.height <= 1080:
                raise ValueError("Encoded video dimensions exceed bounds")
            if stream.codec_context.name not in {"h264", "hevc", "vp8", "vp9", "av1"}:
                raise ValueError("Video codec is not admitted")
            first = previous = None
            pixels = 0
            for index, frame in enumerate(container.decode(stream)):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError("Video presentation timestamps are missing")
                timestamp = float(frame.pts * frame.time_base)
                if first is None:
                    first = timestamp
                timestamp -= first
                if (
                    not math.isfinite(timestamp)
                    or not 0 <= timestamp <= 30
                    or (previous is not None and timestamp <= previous)
                ):
                    raise ValueError("Video timestamps are invalid or too long")
                pixels += frame.width * frame.height
                if (
                    index >= 1800
                    or pixels > 1_000_000_000
                    or (frame.width != stream.width or frame.height != stream.height)
                ):
                    raise ValueError("Decoded video work exceeds bounds")
                previous = timestamp
                yield timestamp, frame

    times = []
    actual_duration = 0
    for timestamp, frame in frames():
        times.append(timestamp)
        interval = float(frame.duration * frame.time_base)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Frame presentation duration is unavailable")
        actual_duration = timestamp + interval
    if actual_duration > 30.000001:
        raise ValueError("Video duration exceeds bound")
    if len(times) < 2:
        raise ValueError("Temporal tasks require at least two distinct frames")
    # Include the final frame's declared presentation interval in clip duration,
    # while sampling only times with an actual decoded frame.
    end = min(times[-1], start + duration) if duration is not None else times[-1]
    if start >= end or start < 0:
        raise ValueError("Selected video interval contains no temporal sequence")
    targets = np.linspace(start, end, count)
    indices = [min(range(len(times)), key=lambda i: abs(times[i] - t)) for t in targets]
    if len(set(indices)) < 2:
        raise ValueError("Selected interval contains fewer than two distinct frames")
    selected = {}
    for index, (_, frame) in enumerate(frames()):
        if index in indices:
            image = frame.to_image().convert("RGB")
            image.thumbnail((512, 512))
            selected[index] = image
    return [selected[i] for i in indices], [times[i] for i in indices], actual_duration


def load_qwen3_video(root, config, shapes):
    """Strict native loading, including duplicated tied vocabulary weights."""
    from copy import deepcopy

    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from oryn_engine.qwen_retrieval import validate_geometry

    validate_geometry(config)
    native_config = Qwen3VLConfig.from_dict(deepcopy(config))
    with init_empty_weights(include_buffers=False):
        model = Qwen3VLForConditionalGeneration(native_config)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    embedding, head = "model.language_model.embed_tokens.weight", "lm_head.weight"
    tied = config.get("tie_word_embeddings") is True
    if tied and head not in shapes:
        expected.pop(head)
    if expected != shapes:
        raise ValueError("Qwen video complete tensor graph disagrees")
    weights = {}
    for path in sorted(root.glob("*.safetensors")):
        state = load_file(path)
        if set(weights) & set(state) or any(
            not value.is_floating_point() or not bool(torch.isfinite(value).all())
            for value in state.values()
        ):
            raise ValueError("Qwen video numeric weights are invalid")
        weights.update(state)
    if tied:
        if head in weights and not torch.equal(weights[head], weights[embedding]):
            raise ValueError("Qwen video tied vocabulary weights disagree")
        weights[head] = weights[embedding]
    model.load_state_dict(weights, strict=True, assign=True)
    model.tie_weights()
    return model.float()


def qwen3_video_inputs(processor, config, frames, times, question, max_tokens):
    """Native video pixels and owned timestamp framing, without repository Jinja."""
    import math

    if (
        config.get("model_type") not in {"qwen3_vl", "qwen3_5"}
        or len(frames) != 8
        or len(times) != 8
        or any(
            type(t) not in {int, float} or not math.isfinite(t) or not 0 <= t <= 30 for t in times
        )
        or any(b <= a for a, b in zip(times[:-1], times[1:], strict=True))
        or not isinstance(question, str)
        or not question.strip()
        or len(question) > 8192
        or "<|" in question
        or "|>" in question
        or type(max_tokens) is not int
        or not 1 <= max_tokens <= 512
    ):
        raise ValueError("Qwen video input or framing is invalid")
    for token, field in (
        ("<|vision_start|>", "vision_start_token_id"),
        ("<|vision_end|>", "vision_end_token_id"),
        ("<|video_pad|>", "video_token_id"),
    ):
        if processor.tokenizer.encode(token, add_special_tokens=False) != [config.get(field)]:
            raise ValueError("Qwen video marker identity disagrees")
    pixels = dict(
        processor.video_processor(
            videos=[frames], do_sample_frames=False, return_metadata=False, return_tensors="pt"
        )
    )
    grid = pixels["video_grid_thw"]
    if grid.shape != (1, 3) or grid[0, 0].item() != 4:
        raise ValueError("Qwen video grid is invalid")
    frame_tokens = int(grid[0, 1:].prod()) // 4
    if not 1 <= frame_tokens <= 1024:
        raise ValueError("Qwen video token budget exceeded")
    # The original processor expands each temporal pair into one timestamp and
    # one vision block. Use decoded source timestamps, preserving a clip's offset.
    content = "".join(
        f"<{(times[i] + times[i + 1]) / 2:.1f} seconds>"
        + "<|vision_start|>"
        + "<|video_pad|>" * frame_tokens
        + "<|vision_end|>"
        for i in range(0, 8, 2)
    )
    text = "<|im_start|>user\n" + content + question + "<|im_end|>\n<|im_start|>assistant\n"
    tokens = dict(
        processor.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            return_token_type_ids=False,
            truncation=False,
        )
    )
    limit = min(4096, config.get("text_config", {}).get("max_position_embeddings", 4096))
    if tokens["input_ids"].shape[-1] + max_tokens > limit:
        raise ValueError("Qwen video token budget exceeded")
    tokens["mm_token_type_ids"] = (tokens["input_ids"] == config["video_token_id"]).long() * 2
    return tokens | pixels


def video_answer(model, processor, config, value, backend, path):
    if config["model_type"] == "oryn_llava_video_qwen2":
        from oryn_engine.llava_video_qwen import execute

        return execute(model, processor, config, value, backend, path)
    if config["model_type"] == "videollama3_qwen2":
        from oryn_engine.videollama3_tasks import execute

        return execute(model, processor, config, value, backend, path)
    if config["model_type"] == "llava_next_video":
        from oryn_engine.llava_video import execute

        return execute(model, processor, config, value, backend, path)
    from oryn_engine.shared_tasks import _move

    frames, times, duration = sample_video(
        path, 8, value["startSeconds"], value.get("durationSeconds")
    )
    interval = (times[-1] - times[0]) / (len(times) - 1)
    if interval <= 0 or any(
        abs((b - a) - interval) > interval * 0.15
        for a, b in zip(times[:-1], times[1:], strict=True)
    ):
        raise ValueError(
            "Native temporal positions require eight approximately uniform distinct frames"
        )
    prompt = (
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
        + value["question"]
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    metadata = {
        "total_num_frames": len(frames),
        "fps": 1 / interval,
        "frames_indices": list(range(len(frames))),
        "duration": times[-1] - times[0] + interval,
        "height": frames[0].height,
        "width": frames[0].width,
    }
    inputs = _move(
        qwen3_video_inputs(processor, config, frames, times, value["question"], value["maxTokens"])
        if config["model_type"] in {"qwen3_vl", "qwen3_5"}
        else dict(
            processor(
                text=prompt,
                videos=[frames],
                video_metadata=[metadata],
                do_sample_frames=False,
                return_tensors="pt",
            )
        ),
        backend,
    )
    if inputs["input_ids"].shape[-1] > 4096 or inputs["video_grid_thw"].shape != (1, 3):
        raise ValueError("Temporal prompt exceeds the native token bound")
    # One forward/generation call attends jointly to the ordered video tokens.
    from transformers import GenerationConfig

    text_config = config.get("text_config", config)
    generated = model.generate(
        **inputs,
        max_new_tokens=value["maxTokens"],
        generation_config=GenerationConfig(
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=text_config.get("eos_token_id", config.get("eos_token_id")),
            pad_token_id=config.get("pad_token_id", processor.tokenizer.pad_token_id),
        ),
    )
    answer = processor.batch_decode(
        generated[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )[0].strip()
    if not answer or len(answer.encode()) > 16384:
        raise ValueError("Video answer is empty or oversized")
    return {"text": answer, "sampledTimestamps": times, "durationSeconds": duration}


def table_answer(model, processor, config, value, backend):
    import pandas as pd
    import torch

    from oryn_engine.shared_tasks import _move

    if config["model_type"] == "tapas":
        rows, columns = table_shape_limits(config)
        if len(value["rows"]) > rows or len(value["columns"]) > columns:
            raise ValueError("TAPAS table exceeds native row/column limits")
    if config["model_type"] == "bart":
        for row in value["rows"]:
            if any(len(processor.tokenize(cell)) > processor._oryn_max_cell_length for cell in row):
                raise ValueError("Table cell exceeds TAPEX's native cell token limit")
        text = value["question"] + " col : " + " | ".join(value["columns"])
        for index, row in enumerate(value["rows"], 1):
            text += f" row {index} : " + " | ".join(row)
        if processor._oryn_lower_case:
            text = text.lower()
        inputs = processor(text, return_tensors="pt", return_token_type_ids=False, truncation=False)
    else:
        table = pd.DataFrame(value["rows"], columns=value["columns"])
        inputs = processor(
            table=table, queries=value["question"], return_tensors="pt", truncation=False
        )
    if inputs["input_ids"].shape[-1] > min(1024, config.get("max_position_embeddings", 512)):
        raise ValueError("Table and question exceed native token limit")
    if config["model_type"] == "bart":
        output = model.generate(**_move(inputs, backend), max_new_tokens=128, do_sample=False)
        return {
            "answer": processor.batch_decode(output, skip_special_tokens=True)[0].strip(),
            "cells": [],
            "aggregation": "generated",
        }
    output = model(**_move(inputs, backend))
    if not bool(torch.isfinite(output.logits).all()):
        raise ValueError("Table cell scores are invalid")
    aggregation_logits = output.logits_aggregation
    args = [inputs, output.logits.cpu()]
    if aggregation_logits is not None:
        if not bool(torch.isfinite(aggregation_logits).all()):
            raise ValueError("Table aggregation scores are invalid")
        args.append(aggregation_logits.cpu())
    prediction = processor.convert_logits_to_predictions(*args)
    coordinates = prediction[0][0]
    if len(coordinates) > 128:
        raise ValueError("Table answer exceeds cell bound")
    cells = [{"row": int(r), "column": int(c), "text": value["rows"][r][c]} for r, c in coordinates]
    operation = (
        config.get("aggregation_labels", {}).get(str(prediction[1][0]), "NONE")
        if aggregation_logits is not None
        else "NONE"
    )
    operation = operation.lower()
    if operation == "none":
        answer = ", ".join(c["text"] for c in cells)
    elif operation == "count":
        answer = str(len(cells))
    else:
        numbers = [float(c["text"].replace(",", "")) for c in cells]
        if not numbers or any(not math.isfinite(v) for v in numbers):
            raise ValueError("Table aggregation requires finite numeric cells")
        number = sum(numbers) / (len(numbers) if operation == "average" else 1)
        answer = format(number, ".15g")
    return {"answer": answer, "cells": cells, "aggregation": operation}


def document_answer(model, processor, config, image, value, backend):
    import torch

    from oryn_engine import layoutlm_tasks

    if layoutlm_tasks.supports(config, "document-question-answering"):
        return layoutlm_tasks.answer(model, processor, config, value, backend)

    from oryn_engine.shared_tasks import _move

    tokenizer, image_processor = processor
    boxes = [[round(c * 1000) for c in box] for box in value["boxes"]]
    if config["model_type"] == "layoutlm":
        inputs = tokenizer(
            value["question"].split(),
            value["words"],
            is_split_into_words=True,
            return_tensors="pt",
            truncation=False,
        )
    else:
        inputs = tokenizer(
            value["question"], value["words"], boxes=boxes, return_tensors="pt", truncation=False
        )
    words, sequence = inputs.word_ids(0), inputs.sequence_ids(0)
    if len(words) > 512:
        raise ValueError("Document and question exceed native token limit")
    valid = torch.tensor([s == 1 and w is not None for s, w in zip(sequence, words, strict=True)])
    if not bool(valid.any()):
        raise ValueError("Document has no grounded tokens")
    if "bbox" not in inputs:
        inputs["bbox"] = torch.tensor(
            [
                [
                    boxes[w] if s == 1 and w is not None else [0, 0, 0, 0]
                    for s, w in zip(sequence, words, strict=True)
                ]
            ]
        )
    if image_processor is not None:
        inputs.update(image_processor(images=image, apply_ocr=False, return_tensors="pt"))
    if config["model_type"] == "layoutlmv2":
        # The native LayoutLMv2 graph names its BGR page input "image".
        inputs["image"] = inputs.pop("pixel_values")
    output = model(**_move(inputs, backend))
    start, end = output.start_logits[0].float().cpu(), output.end_logits[0].float().cpu()
    if config["model_type"] == "layoutlmv3":
        # The native QA head also returns visual patch positions. Those positions
        # have no OCR word/box and cannot be used as grounded answer endpoints.
        height, width = inputs["pixel_values"].shape[-2:]
        patch = config.get("patch_size", 16)
        if type(patch) is not int or patch < 1 or height % patch or width % patch:
            raise ValueError("Document visual patch geometry is invalid")
        expected = len(words) + (height // patch) * (width // patch) + 1
        if start.shape != (expected,) or end.shape != (expected,):
            raise ValueError("Document visual/text positions disagree")
        if not (bool(torch.isfinite(start).all()) and bool(torch.isfinite(end).all())):
            raise ValueError("Document span scores are invalid")
        start, end = start[: len(words)].clone(), end[: len(words)].clone()
    if (
        start.shape != valid.shape
        or end.shape != valid.shape
        or not (bool(torch.isfinite(start).all()) and bool(torch.isfinite(end).all()))
    ):
        raise ValueError("Document span scores are invalid")
    start[~valid] = end[~valid] = -torch.inf
    scores = torch.softmax(start, 0)[:, None] * torch.softmax(end, 0)[None, :]
    positions = torch.arange(len(words))
    allowed = (positions[:, None] <= positions[None, :]) & (
        positions[None, :] - positions[:, None] < 32
    )
    scores[~allowed] = 0
    a, b = divmod(int(scores.argmax()), len(words))
    first, last = words[a], words[b]
    if first is None or last is None or first > last:
        raise ValueError("Document grounding is invalid")
    selected = value["boxes"][first : last + 1]
    return {
        "answer": " ".join(value["words"][first : last + 1]),
        "score": float(scores[a, b]),
        "wordStart": first,
        "wordEnd": last,
        "box": [
            min(b[0] for b in selected),
            min(b[1] for b in selected),
            max(b[2] for b in selected),
            max(b[3] for b in selected),
        ],
    }


def execute(task, model, processor, config, value, backend, input_path, image, original):
    import torch

    from oryn_engine.shared_tasks import _move

    if task == "visual-document-retrieval":
        if config["model_type"] == "jina_embeddings_v4":
            from oryn_engine.jina4_tasks import retrieve

            return retrieve(model, processor, config, image, value, backend)
        if config["model_type"] == "neomme":
            from oryn_engine.neomme_retrieval import retrieve

            return retrieve(model, processor, config, image, value, backend)
        if config["model_type"] == "qwen3_5":
            from oryn_engine.colqwen35_tasks import retrieve

            return retrieve(model, processor, config, image, value, backend)
        if config["model_type"] == "colqwen3":
            from oryn_engine.colqwen3_tasks import retrieve

            return retrieve(model, processor, config, image, value, backend)
        from oryn_engine.visual_retrieval import retrieve

        return retrieve(model, processor, image, value, backend)

    if task == "table-question-answering":
        return table_answer(model, processor, config, value, backend)
    if task == "document-question-answering":
        return document_answer(model, processor, config, image, value, backend)
    if task == "video-text-to-text":
        return video_answer(model, processor, config, value, backend, input_path)
    if task == "keypoint-detection":
        output = model(**_move(processor(images=image, return_tensors="pt"), backend))
        result = processor.post_process_keypoint_detection(output, [(original[1], original[0])])[0]
        points = []
        for i in result["scores"].argsort(descending=True).tolist():
            score = float(result["scores"][i])
            x, y = result["keypoints"][i].tolist()
            if not all(math.isfinite(v) for v in (score, x, y)):
                raise ValueError("Keypoint output contains non-finite values")
            if not 0 <= x <= original[0] or not 0 <= y <= original[1]:
                raise ValueError("Keypoint is outside original image")
            if score >= value["threshold"]:
                points.append({"x": float(x), "y": float(y), "score": score})
            if len(points) >= value["maxKeypoints"]:
                break
        return {"keypoints": points, "width": original[0], "height": original[1]}
    frames, times, duration = sample_video(
        input_path, video_frame_count(config), value["startSeconds"], value.get("durationSeconds")
    )
    processed = (
        processor(videos=[frames], do_sample_frames=False, return_tensors="pt")
        if config["model_type"] == "vjepa2"
        else processor(frames, return_tensors="pt")
    )
    output = model(**_move(processed, backend))
    logits = output.logits[0].float()
    labels = config["id2label"]
    if logits.numel() != len(labels) or not bool(torch.isfinite(logits).all()):
        raise ValueError("Video logits do not match the declared labels")
    scores, indices = torch.softmax(logits, 0).topk(min(value["topK"], len(labels)))
    return {
        "predictions": [
            {"label": labels[str(int(i))], "score": float(s)}
            for s, i in zip(scores, indices, strict=True)
        ],
        "sampledTimestamps": times,
        "durationSeconds": duration,
    }
