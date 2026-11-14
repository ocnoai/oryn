# Derived from Sopro 2.2.0 (Apache-2.0); see LICENSE-Sopro.txt and provenance.json.
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from oryn_engine.native_sopro import audio_math as torchaudio
from oryn_engine.native_sopro.config import VocoderConfig

class MelFeatures(nn.Module):

    def __init__(self, cfg: VocoderConfig) -> None:
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=int(cfg.sample_rate), n_fft=int(cfg.n_fft), hop_length=int(cfg.hop_length), n_mels=int(cfg.n_mels), center=True, power=1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clip(self.mel_spec(audio), min=1e-07))

def _conv_causal(x: torch.Tensor, conv: nn.Conv1d, lookahead: int) -> torch.Tensor:
    context = int(conv.kernel_size[0]) - 1
    return F.conv1d(F.pad(x, (context - int(lookahead), int(lookahead))), conv.weight, conv.bias, groups=conv.groups)

class ConvNeXtBlock(nn.Module):

    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init_value: float, causal: bool, lookahead: int=0) -> None:
        super().__init__()
        self.causal = bool(causal)
        self.lookahead = int(lookahead)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=0 if causal else 3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-06)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def _pointwise(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x.transpose(1, 2)))))
        return (self.gamma * x).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = _conv_causal(x, self.dwconv, self.lookahead) if self.causal else self.dwconv(x)
        return x + self._pointwise(h)

class Backbone(nn.Module):

    def __init__(self, cfg: VocoderConfig) -> None:
        super().__init__()
        self.causal = bool(cfg.causal)
        self.lookahead_frames = int(cfg.lookahead_frames) if self.causal else 0
        block_lookaheads = list(cfg.block_lookaheads) if self.causal and cfg.block_lookaheads else [0] * int(cfg.num_layers)
        self.block_lookaheads = [int(v) for v in block_lookaheads]
        self.embed = nn.Conv1d(int(cfg.n_mels), int(cfg.dim), kernel_size=7, padding=0 if self.causal else 3)
        self.norm = nn.LayerNorm(int(cfg.dim), eps=1e-06)
        self.convnext = nn.ModuleList([ConvNeXtBlock(int(cfg.dim), int(cfg.intermediate_dim), 1.0 / int(cfg.num_layers), self.causal, self.block_lookaheads[i]) for i in range(int(cfg.num_layers))])
        self.final_layer_norm = nn.LayerNorm(int(cfg.dim), eps=1e-06)

    @property
    def total_lookahead(self) -> int:
        return self.lookahead_frames + sum(self.block_lookaheads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _conv_causal(x, self.embed, self.lookahead_frames) if self.causal else self.embed(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x.transpose(1, 2))

class ISTFT(nn.Module):

    def __init__(self, n_fft: int, hop_length: int) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.register_buffer('window', torch.hann_window(self.n_fft))

    @property
    def pad(self) -> int:
        return self.n_fft // 2

    def _overlap_add(self, spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, t = spec.shape
        if t <= 0:
            return (spec.real.new_zeros(b, 0), self.window.new_zeros(0))
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm='backward') * self.window[None, :, None]
        output_size = (t - 1) * self.hop_length + self.n_fft
        y = F.fold(ifft, output_size=(1, output_size), kernel_size=(1, self.n_fft), stride=(1, self.hop_length))[:, 0, 0, :]
        window_sq = self.window.square().view(1, self.n_fft, 1).expand(1, self.n_fft, t)
        env = F.fold(window_sq, output_size=(1, output_size), kernel_size=(1, self.n_fft), stride=(1, self.hop_length))[0, 0, 0, :]
        return (y, env)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        y, env = self._overlap_add(spec)
        y = y[:, self.pad:-self.pad]
        env = env[self.pad:-self.pad]
        return y / env.unsqueeze(0)

class ISTFTHead(nn.Module):

    def __init__(self, cfg: VocoderConfig) -> None:
        super().__init__()
        self.out = nn.Linear(int(cfg.dim), int(cfg.n_fft) + 2)
        self.istft = ISTFT(int(cfg.n_fft), int(cfg.hop_length))
        self.log_max_magnitude = math.log(float(cfg.max_magnitude))

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        mag, p = self.out(x).transpose(1, 2).chunk(2, dim=1)
        mag = torch.exp(torch.clamp(mag, max=self.log_max_magnitude))
        return mag * (torch.cos(p) + 1j * torch.sin(p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.istft(self.spectrogram(x))

class Vocoder(nn.Module):

    def __init__(self, cfg: VocoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = MelFeatures(cfg)
        self.backbone = Backbone(cfg)
        self.head = ISTFTHead(cfg)

    @property
    def hop_length(self) -> int:
        return int(self.cfg.hop_length)

    @torch.no_grad()
    def mel(self, audio: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(audio.float())

    @torch.no_grad()
    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(mel.float()))
