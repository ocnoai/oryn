# Derived from Sopro 2.2.0 (Apache-2.0); see LICENSE-Sopro.txt and provenance.json.
from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from oryn_engine.native_sopro.config import ModelConfig
from oryn_engine.native_sopro.nn.layers import apply_rotary, rotary_cos_sin, sinusoidal_time_embedding

def _expand_time(x: torch.Tensor, target_steps: int) -> torch.Tensor:
    src = int(x.shape[-1])
    idx = torch.div(torch.arange(int(target_steps), device=x.device) * src, int(target_steps), rounding_mode='floor').clamp_max(src - 1)
    return x.index_select(dim=2, index=idx)

def _row_has_signal(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(int(x.shape[0]), -1).abs().amax(dim=-1) > 0.0

def build_time_grid(steps: int, sway_coef: float, device: torch.device) -> torch.Tensor:
    times = torch.linspace(0.0, 1.0, steps=int(steps) + 1, device=device, dtype=torch.float32)
    if abs(float(sway_coef)) <= 1e-08:
        return times
    return times + float(sway_coef) * (torch.cos(0.5 * math.pi * times) - 1.0 + times)

def build_chunk_mask(seq_len: int, chunk_size: int, num_left_chunks: int, device: torch.device) -> Optional[torch.Tensor]:
    if int(chunk_size) <= 0:
        return None
    positions = torch.arange(int(seq_len), device=device)
    chunk_index = torch.div(positions, int(chunk_size), rounding_mode='floor')
    chunk_end = ((chunk_index + 1) * int(chunk_size)).clamp_max(int(seq_len))
    if int(num_left_chunks) < 0:
        chunk_start = torch.zeros_like(chunk_end)
    else:
        chunk_start = ((chunk_index - int(num_left_chunks)).clamp_min(0) * int(chunk_size)).clamp_max(int(seq_len))
    key_pos = positions.unsqueeze(0)
    return ((key_pos >= chunk_start.unsqueeze(1)) & (key_pos < chunk_end.unsqueeze(1))).unsqueeze(0)

class CausalConv1d(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        self.left_context = int(kernel_size) - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=int(kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_context, 0)))

class LearnedCausalUpsampler(nn.Module):

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        hidden = max(8, int(channels))
        self.in_proj = nn.Conv1d(int(channels), hidden, kernel_size=1)
        self.mix = CausalConv1d(hidden, hidden, kernel_size=int(kernel_size))
        self.out_proj = nn.Conv1d(hidden, int(channels), kernel_size=1)

    def forward(self, x: torch.Tensor, target_steps: int) -> torch.Tensor:
        x_rep = _expand_time(x, target_steps)
        h = F.silu(self.in_proj(x_rep))
        h = F.silu(self.mix(h))
        return x_rep + self.out_proj(h)

class PreLookahead(nn.Module):

    def __init__(self, channels: int, lookahead: int) -> None:
        super().__init__()
        self.lookahead = int(lookahead)
        self.conv1 = nn.Conv1d(int(channels), int(channels), kernel_size=self.lookahead + 1)
        self.conv2 = nn.Conv1d(int(channels), int(channels), kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x, (0, self.lookahead))
        y = F.leaky_relu(self.conv1(y), negative_slope=0.1)
        y = F.pad(y, (2, 0))
        return x + self.conv2(y)

class CausalConvPositionEmbedding(nn.Module):

    def __init__(self, dim: int, kernel_size: int, groups: int=16) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv1 = nn.Conv1d(int(dim), int(dim), self.kernel_size, groups=int(groups))
        self.conv2 = nn.Conv1d(int(dim), int(dim), self.kernel_size, groups=int(groups))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2).contiguous()
        y = F.mish(self.conv1(F.pad(y, (self.kernel_size - 1, 0))))
        y = F.mish(self.conv2(F.pad(y, (self.kernel_size - 1, 0))))
        return y.transpose(1, 2).contiguous()

class AdaLayerNormZero(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim), elementwise_affine=False, eps=1e-06)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(int(dim), int(dim) * 6))

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.mod(emb).chunk(6, dim=-1)
        return (self.norm(x) * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :], gate_msa, shift_mlp, scale_mlp, gate_mlp)

class AdaLayerNormFinal(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim), elementwise_affine=False, eps=1e-06)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(int(dim), int(dim) * 2))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.mod(emb).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale[:, None, :]) + shift[:, None, :]

class DiTSelfAttention(nn.Module):

    def __init__(self, dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        inner = self.heads * self.dim_head
        self.to_q = nn.Linear(int(dim), inner)
        self.to_k = nn.Linear(int(dim), inner)
        self.to_v = nn.Linear(int(dim), inner)
        self.to_out = nn.Sequential(nn.Linear(inner, int(dim)), nn.Identity())

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.to_q(x).view(bsz, seq_len, self.heads, self.dim_head).transpose(1, 2).contiguous()
        k = self.to_k(x).view(bsz, seq_len, self.heads, self.dim_head).transpose(1, 2).contiguous()
        v = self.to_v(x).view(bsz, seq_len, self.heads, self.dim_head).transpose(1, 2).contiguous()
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None if attn_mask is None else attn_mask[:, None, :, :])
        return self.to_out(y.transpose(1, 2).contiguous().view(bsz, seq_len, self.heads * self.dim_head))

class DiTBlock(nn.Module):

    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: float) -> None:
        super().__init__()
        self.attn_norm = AdaLayerNormZero(int(dim))
        self.attn = DiTSelfAttention(int(dim), int(heads), int(dim_head))
        self.ff_norm = nn.LayerNorm(int(dim), elementwise_affine=False, eps=1e-06)
        hidden = max(1, int(round(float(ff_mult) * int(dim))))
        self.ff = nn.Sequential(nn.Linear(int(dim), hidden), nn.GELU(approximate='tanh'), nn.Identity(), nn.Linear(hidden, int(dim)))

    def forward(self, x: torch.Tensor, emb: torch.Tensor, attn_mask: Optional[torch.Tensor], cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb)
        x = x + gate_msa[:, None, :] * self.attn(h, attn_mask, cos, sin)
        h_ff = self.ff_norm(x) * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        return x + gate_mlp[:, None, :] * self.ff(h_ff)

class InputEmbedding(nn.Module):

    def __init__(self, mel_dim: int, mu_dim: int, spk_dim: int, out_dim: int, pos_kernel_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(mel_dim) * 2 + int(mu_dim) + int(spk_dim), int(out_dim))
        self.pos = CausalConvPositionEmbedding(int(out_dim), kernel_size=int(pos_kernel_size))
        self.cond_mask_proj = nn.Linear(1, int(out_dim), bias=False)

    def forward(self, x_t: torch.Tensor, cond_mel: torch.Tensor, cond_mask: torch.Tensor, mu: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        x_btc = x_t.transpose(1, 2).contiguous()
        cond_btc = cond_mel.transpose(1, 2).contiguous()
        cond_mask_btc = cond_mask.transpose(1, 2).contiguous().to(dtype=x_btc.dtype)
        mu_btc = mu.transpose(1, 2).contiguous()
        spk_btc = spk[:, None, :].expand(-1, x_btc.shape[1], -1)
        h = self.proj(torch.cat([x_btc, cond_btc, mu_btc, spk_btc], dim=-1))
        h = h + self.cond_mask_proj(cond_mask_btc)
        return h + self.pos(h)

class AcousticHead(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.semantic_dim = int(cfg.latent_dim)
        self.target_dim = int(cfg.acoustic_mel_n_mels)
        self.model_dim = int(cfg.acoustic_dit_dim)
        self.time_embed_dim = int(cfg.acoustic_time_embed_dim)
        self.mu_dim = int(cfg.acoustic_mu_dim)
        self.spk_dim = int(cfg.acoustic_spk_dim)
        self.dim_head = int(cfg.acoustic_dit_dim_head)
        self.semantic_token_emb = nn.Embedding(int(cfg.semantic_vocab_size), self.semantic_dim)
        self.semantic_prelook = PreLookahead(self.semantic_dim, int(cfg.acoustic_pre_lookahead_frames))
        self.semantic_upsampler = LearnedCausalUpsampler(self.semantic_dim, int(cfg.acoustic_upsampler_kernel_size))
        self.mu_proj = nn.Conv1d(self.semantic_dim, self.mu_dim, kernel_size=1)
        self.time_mlp = nn.Sequential(nn.Linear(self.time_embed_dim, self.model_dim), nn.SiLU(), nn.Linear(self.model_dim, self.model_dim))
        self.spk_proj = nn.Linear(int(cfg.cond_hidden_dim), self.spk_dim)
        self.input_embed = InputEmbedding(self.target_dim, self.mu_dim, self.spk_dim, self.model_dim, int(cfg.acoustic_pos_kernel_size))
        self.blocks = nn.ModuleList([DiTBlock(self.model_dim, int(cfg.acoustic_dit_heads), self.dim_head, float(cfg.acoustic_dit_ff_mult)) for _ in range(int(cfg.acoustic_dit_depth))])
        self.out_norm = AdaLayerNormFinal(self.model_dim)
        self.out_proj = nn.Linear(self.model_dim, self.target_dim)

    def semantic_latents(self, semantic_tokens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.semantic_token_emb(semantic_tokens.to(dtype=torch.long)).transpose(1, 2).contiguous().to(dtype=dtype)

    def _time_embedding(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        emb = sinusoidal_time_embedding(t.to(dtype=torch.float32), self.time_embed_dim)
        return self.time_mlp(emb.to(dtype=self.time_mlp[0].weight.dtype)).to(dtype=dtype)

    def _speaker_embedding(self, cond_vec: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        cond = F.normalize(cond_vec.to(dtype=self.spk_proj.weight.dtype), dim=-1)
        spk = self.spk_proj(cond)
        return (spk * _row_has_signal(cond_vec).to(dtype=spk.dtype)[:, None]).to(dtype=dtype)

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor, semantic_latents: torch.Tensor, cond_vec: torch.Tensor, cond_mel: torch.Tensor, cond_mask: torch.Tensor, attn_mask: Optional[torch.Tensor], cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        mu = self.mu_proj(self.semantic_upsampler(self.semantic_prelook(semantic_latents), int(x_t.shape[-1])))
        mu = mu * _row_has_signal(semantic_latents)[:, None, None].to(dtype=mu.dtype)
        spk = self._speaker_embedding(cond_vec, x_t.dtype)
        emb = self._time_embedding(t, x_t.dtype)
        h = self.input_embed(x_t, cond_mel, cond_mask, mu, spk)
        for block in self.blocks:
            h = block(h, emb, attn_mask, cos, sin)
        h = self.out_norm(h, emb)
        return self.out_proj(h).transpose(1, 2).contiguous()

    @torch.no_grad()
    def solve(self, x_init: torch.Tensor, semantic_tokens: torch.Tensor, cond_vec: torch.Tensor, cond_mel: torch.Tensor, cond_mask: torch.Tensor, steps: int, chunk_size: int=0) -> torch.Tensor:
        x = x_init
        seq_len = int(x.shape[-1])
        cond_mask = cond_mask.to(dtype=x.dtype).clamp(0.0, 1.0)
        cond_mel = cond_mel.to(dtype=x.dtype) * cond_mask
        sigma_min = float(self.cfg.acoustic_sigma_min)
        grid = build_time_grid(int(steps), float(self.cfg.acoustic_sway_sampling_coef), x.device)
        latents = self.semantic_latents(semantic_tokens, x.dtype)
        attn_mask = build_chunk_mask(seq_len, int(chunk_size), int(self.cfg.acoustic_num_left_chunks), x.device)
        cos, sin = rotary_cos_sin(torch.arange(seq_len, device=x.device), self.dim_head, x.dtype)
        batch = int(x.shape[0])
        for i in range(int(steps)):
            t0, t1 = (grid[i], grid[i + 1])
            v = self.velocity(x, t0.expand(batch), latents, cond_vec, cond_mel, cond_mask, attn_mask, cos, sin)
            x = x + (t1 - t0) * v
            x_prompt = (1.0 - (1.0 - sigma_min) * t1) * x_init + t1 * cond_mel
            x = cond_mask * x_prompt + (1.0 - cond_mask) * x
        return cond_mask * cond_mel + (1.0 - cond_mask) * x
