"""Complete generated native Music3 components; no publisher weights."""

import json
from pathlib import Path

from oryn_engine import minimax_music as adapter

FIXTURES = Path(__file__).parent / "fixtures/minimax_music"


def original(name):
    return json.loads((FIXTURES / name).read_text())


def tiny_configs():
    configs = {k: original(n) for k, n in adapter.CONFIGS.items()}
    configs["language_model"].update(
        hidden_size=16,
        head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        num_hidden_layers=1,
        layer_types=["full_attention"],
    )
    configs["rvq_depth_decoder"].update(
        hidden_size=16, num_layers=1, num_attention_heads=2, intermediate_size=32
    )
    configs["condition_encoder"].update(condition_hidden_dim=16, out_dim=16)
    configs["transformer"].update(
        attention_head_dim=8,
        num_attention_heads=2,
        ff_inner_dim=32,
        condition_dim=16,
        in_channels=4,
        fourier_embedding_dim=8,
        num_layers=1,
        rotary_dim=4,
    )
    configs["vocoder"].update(decoder_hidden_dim=32, decoder_input_dim=8, latent_channels=4)
    return configs


def tiny_fixture(root):
    import torch
    from safetensors.torch import save_file
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

    root.mkdir(parents=True, exist_ok=True)
    configs = tiny_configs()
    for name, filename in adapter.CONFIGS.items():
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(configs[name]))
    (root / adapter.INDEX).write_text(json.dumps(original(adapter.INDEX)))
    torch.manual_seed(6106)
    components = adapter.native_components(configs)
    for name, model in components.items():
        model.eval()
        filename = (
            "model.safetensors"
            if name == "language_model"
            else "diffusion_pytorch_model.safetensors"
        )
        save_file(
            {n: t.detach().contiguous() for n, t in model.state_dict().items()},
            root / name / filename,
        )
    vocab = {s: i for i, s in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab.update({f"[unused-{i}]": i for i in range(256, 151675)})
    for token, index in adapter.CONTROL_IDS.items():
        del vocab[f"[unused-{index}]"]
        vocab[token] = index
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(
        [AddedToken(s, normalized=False, special=True) for s in adapter.CONTROL_IDS]
    )
    tok.save(str(root / "tokenizer/tokenizer.json"))
    return configs, components


def native_pipeline(root, configs, components, backend="cpu"):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.modular_pipelines.minimax_music3.modular_blocks_minimax_music3 import (
        MiniMaxMusic3Blocks,
    )
    from diffusers.modular_pipelines.minimax_music3.modular_pipeline import (
        MiniMaxMusic3ModularPipeline,
    )

    pipe = MiniMaxMusic3ModularPipeline(blocks=MiniMaxMusic3Blocks())
    pipe.update_components(
        **components,
        tokenizer=adapter.load_tokenizer(root),
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(configs["scheduler"]),
    )
    pipe.to(backend)
    return pipe
