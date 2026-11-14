import copy

import pytest

from oryn_engine.native_lfm3 import CORE, Graph, expected_shapes, representation, validate_config


def config():
    return copy.deepcopy(CORE) | {
        "architectures": ["Lfm3ForCausalLM"],
        "dtype": "bfloat16",
        "max_position_embeddings": 262144,
        "bos_token_id": 124894,
        "eos_token_id": 124900,
        "pad_token_id": 124893,
    }


def test_closed_representation_and_dimensions():
    assert validate_config(config())
    for patch in (
        {"hidden_size": True},
        {"hidden_size": 4096},
        {"attn_implementation": "author"},
        {"quantization_config": {"mode": "affine", "bits": 4}},
    ):
        with pytest.raises(ValueError):
            validate_config(config() | patch)
    assert (
        representation(config() | {"quantization_config": {"mode": "affine", "bits": 4}})
        == "mlx-affine4-unsupported"
    )
    assert (
        representation(config() | {"quantization_config": {"quant_method": "compressed-tensors"}})
        == "fp8-unsupported"
    )
    assert (
        representation(config() | {"quantization_config": {"quant_method": "modelopt"}})
        == "nvfp4-unsupported"
    )


def test_prefill_decode_chunking_and_sliding_cache_equivalence():
    import torch

    torch.set_num_threads(1)
    c = config() | {
        "hidden_size": 16,
        "intermediate_size": 24,
        "head_dim": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 32,
        "num_hidden_layers": 3,
        "layer_types": ["conv", "full_attention", "sliding_attention"],
        "sliding_window": 3,
    }
    generator = torch.Generator().manual_seed(63)
    weights = {
        name: torch.randn(shape, generator=generator) * 0.1
        for name, shape in expected_shapes(c).items()
    }
    for name, value in weights.items():
        if "norm.weight" in name:
            value.fill_(1)
    tokens = [1, 4, 2, 6, 3, 7, 5, 8, 9]
    full = Graph(c, weights).forward(tokens)
    incremental = Graph(c, weights)
    for token in tokens:
        last = incremental.forward([token])
    torch.testing.assert_close(full, last, atol=2e-6, rtol=2e-5)
    chunked = Graph(c, weights)
    chunked.forward(tokens[:4])
    last = chunked.forward(tokens[4:])
    torch.testing.assert_close(full, last, atol=2e-6, rtol=2e-5)
    assert incremental.cache[2][0].shape[2] == 3
    assert incremental.cache[0].shape[-1] == 7
    with pytest.raises(ValueError):
        incremental.forward([True])
