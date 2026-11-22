from types import SimpleNamespace

import pytest
import torch

from oryn_engine.visual_retrieval import late_interaction, validate_config, vectors


def test_late_interaction_retains_each_query_token_instead_of_mean_pooling():
    assert late_interaction([[1, 0], [0, 1]], [[1, 0], [0, 1]]) == 2
    assert late_interaction([[1, 0], [0, 1]], [[-1, 0], [0, -1]]) == 0
    assert late_interaction([[1, 0], [1, 0]], [[1, 0]]) == 2


def test_vectors_remove_only_padding_and_reject_nonfinite_unnormalized_or_oversized_output():
    matrix = torch.zeros((1, 3, 128))
    matrix[0, :2, 0] = 1
    mask = torch.tensor([[1, 1, 0]])
    output = SimpleNamespace(embeddings=matrix)
    assert len(vectors(output, mask, 2)) == 2
    with pytest.raises(ValueError, match="bounded unit"):
        vectors(output, mask, 1)
    matrix[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="bounded unit"):
        vectors(output, mask, 2)
    matrix[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="shape is invalid"):
        vectors(output, mask, 2)


def test_retrieval_requires_projection_and_native_vision_config():
    config = {
        "model_type": "colqwen2",
        "embedding_dim": 128,
        "vlm_config": {"model_type": "qwen2_vl", "vision_config": {}},
    }
    validate_config(config)
    for changes in (
        {"embedding_dim": 256},
        {"model_type": "qwen2_vl"},
        {"vlm_config": {"model_type": "qwen2_vl"}},
    ):
        with pytest.raises(ValueError, match="graph is unsupported"):
            validate_config({**config, **changes})


def test_native_legacy_retrieval_layout_cannot_mix_serializations():
    import re

    from oryn_engine.visual_retrieval import legacy_weight_mapping

    shapes = {
        "vlm.model.embed_tokens.weight": (151936, 1536),
        "vlm.model.layers.0.self_attn.q_proj.weight": (1536, 1536),
        "vlm.visual.patch_embed.proj.weight": (1280, 3, 2, 14, 14),
        "embedding_proj_layer.weight": (128, 1536),
    }
    mapping = legacy_weight_mapping(shapes)
    source, target = next(iter(mapping.items()))
    assert (
        re.sub(source, target, "vlm.model.layers.0.self_attn.q_proj.weight")
        == "vlm.language_model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        re.sub(source, target, "vlm.visual.patch_embed.proj.weight")
        == "vlm.visual.patch_embed.proj.weight"
    )
    with pytest.raises(ValueError, match="Mixed"):
        legacy_weight_mapping(
            shapes | {"vlm.model.language_model.embed_tokens.weight": (151936, 1536)}
        )
