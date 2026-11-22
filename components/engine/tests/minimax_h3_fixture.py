"""Complete reduced native H3 graph and deterministic generated weights."""

import json
from pathlib import Path

from oryn_engine import minimax_h3 as adapter

FIXTURES = Path(__file__).parent / "fixtures/minimax_h3"


def original(name):
    return json.loads((FIXTURES / name).read_text())


def tiny_configs():
    c = {k: original(n) for k, n in adapter.CONFIGS.items()}
    t = c["text_encoder"]
    t["text_config"].update(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=51,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    t["text_config"]["rope_scaling"]["mrope_section"] = [2, 1, 1]
    t["vision_config"].update(
        hidden_size=16,
        intermediate_size=32,
        depth=3,
        num_heads=2,
        out_hidden_size=16,
        deepstack_visual_indexes=[0, 1, 2],
        num_position_embeddings=16,
    )
    c["transformer"].update(
        hidden_size=16,
        num_attention_heads=2,
        attention_head_dim=8,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=4,
        audio_in_channels=4,
        text_dim=16,
        freq_dim=8,
        time_embed_hidden_dim=16,
        time_embed_dim=8,
        rope_freq_dim=1,
    )
    c["vae"].update(
        latent_channels=4,
        block_out_channels=[4] * 6,
        layers_per_block=1,
        norm_num_groups=2,
        decoder_num_layers=1,
        decoder_num_attention_heads=1,
        decoder_attention_head_dim=16,
        decoder_num_register_tokens=2,
        decoder_ffn_mult=2,
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
    )
    c["audio_vae"].update(
        encoder_dim=2,
        latent_dim=16,
        latent_channels=4,
        decoder_dim=128,
        num_attention_heads=2,
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
    )
    return c


def tiny_fixture(root, weight_dtype=None):
    import torch
    from safetensors.torch import save_file
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

    root.mkdir(exist_ok=True, parents=True)
    configs = tiny_configs()
    for k, n in adapter.CONFIGS.items():
        path = root / n
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(json.dumps(configs[k]))
    (root / adapter.INDEX).write_text(json.dumps(original(adapter.INDEX)))
    torch.manual_seed(6107)
    components = adapter.native_components(configs)
    for name, model in components.items():
        # The adapter computes in float32, including the Qwen text submodel
        # whose constructor otherwise honors the checkpoint's bfloat16 config.
        model.to(dtype=torch.float32).eval()
        stem = "model" if name == "text_encoder" else "diffusion_pytorch_model"
        weights = {
            n: t.detach().to(weight_dtype or torch.float32).contiguous()
            for n, t in model.state_dict().items()
        }
        save_file(
            weights,
            root / name / (stem + ".safetensors"),
        )
        model.load_state_dict(weights, strict=True, assign=True)
        model.to(dtype=torch.float32).eval()
    vocab = {s: i for i, s in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab.update({f"[unused-{i}]": i for i in range(256, 151669)})
    for s, i in adapter.CONTROL_IDS.items():
        del vocab[f"[unused-{i}]"]
        vocab[s] = i
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(
        [AddedToken(s, normalized=False, special=True) for s in adapter.CONTROL_IDS]
    )
    tok.save(str(root / "tokenizer/tokenizer.json"))
    return configs, components
