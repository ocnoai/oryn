"""Native page/query multi-vectors with bounded, reproducible late interaction."""

from __future__ import annotations

from oryn_engine.model_metadata import metadata_file


def validate_config(config):
    vlm = config.get("vlm_config", {})
    if (
        config.get("model_type") != "colqwen2"
        or config.get("embedding_dim", 128) != 128
        or not isinstance(vlm, dict)
        or vlm.get("model_type") != "qwen2_vl"
        or not isinstance(vlm.get("vision_config"), dict)
        or vlm.get("quantization_config")
        or vlm.get("auto_map")
    ):
        raise ValueError("Native visual retrieval graph is unsupported")


def load_processor(root, config, common):
    import transformers as tf

    from oryn_engine.shared_tasks import qwen_image_options

    metadata = metadata_file(root / "preprocessor_config.json")
    options = qwen_image_options(metadata, config["vlm_config"]["vision_config"])
    options["size"] = {"shortest_edge": 56**2, "longest_edge": 196 * 28**2}
    tokenizer = tf.AutoTokenizer.from_pretrained(root, **common)
    if tokenizer.pad_token != "<|endoftext|>" or tokenizer.convert_tokens_to_ids(
        "<|image_pad|>"
    ) != config["vlm_config"].get("image_token_id"):
        raise ValueError("Visual retrieval tokenizer binding is unsupported")
    # Sealed native prefix and query augmentation; no repository Jinja or code.
    return tf.ColQwen2Processor(
        image_processor=tf.Qwen2VLImageProcessor(**options), tokenizer=tokenizer
    )


def legacy_weight_mapping(shapes):
    """Rename the native pre-composite Qwen2 serialization, without changing tensors."""
    legacy = "vlm.model.embed_tokens.weight" in shapes
    if not legacy:
        return None
    if any(
        name.startswith(("vlm.language_model.", "vlm.model.language_model.", "vlm.model.visual."))
        for name in shapes
    ):
        raise ValueError("Mixed visual retrieval weight layouts are not admitted")
    if not any(name.startswith("vlm.visual.") for name in shapes):
        raise ValueError("Visual retrieval vision weights are missing")
    return {r"^vlm\.model\.": "vlm.language_model."}


def vectors(output, mask, limit, *, dimensions=128):
    import torch

    matrix = output.embeddings
    if (
        matrix.ndim != 3
        or matrix.shape[0] != 1
        or dimensions not in {128, 320, 4096}
        or matrix.shape[2] != dimensions
        or matrix.shape[:2] != mask.shape
        or not bool(torch.isfinite(matrix).all())
    ):
        raise ValueError("Visual retrieval vector shape is invalid")
    matrix = matrix[0][mask[0].bool()].float().cpu()
    if not 1 <= len(matrix) <= limit or not bool(
        torch.allclose(matrix.norm(dim=-1), torch.ones(len(matrix)), atol=1e-4, rtol=0)
    ):
        raise ValueError("Visual retrieval vectors are not bounded unit vectors")
    return [[round(float(x), 6) for x in row] for row in matrix]


def late_interaction(query, document):
    import numpy as np

    scores = np.asarray(query, dtype=np.float64) @ np.asarray(document, dtype=np.float64).T
    return float(scores.max(axis=1).sum())


def retrieve(model, processor, image, value, backend):
    from oryn_engine.shared_tasks import _move

    page = _move(dict(processor.process_images(images=[image], return_tensors="pt")), backend)
    queries = [
        _move(
            dict(processor.process_queries(text=[q], return_tensors="pt", truncation=False)),
            backend,
        )
        for q in value["queries"]
    ]
    # Validate every sequence before running the first forward pass. Never silently
    # truncate a query or alter native query-augmentation token semantics.
    if page["input_ids"].shape[1] > 256 or any(q["input_ids"].shape[1] > 64 for q in queries):
        raise ValueError("Visual retrieval token limit exceeded")
    document = vectors(model(**page), page["attention_mask"], 256)
    results = []
    for q in queries:
        query = vectors(model(**q), q["attention_mask"], 64)
        results.append({"vectors": query, "score": late_interaction(query, document)})
    return {
        "dimensions": 128,
        "normalized": True,
        "scoring": "sum-max-cosine",
        "documentVectors": document,
        "queries": results,
    }
