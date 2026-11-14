# Derived from Sopro 2.2.0 (Apache-2.0); see LICENSE-Sopro.txt and provenance.json.
from __future__ import annotations
from typing import Iterator, Optional
import torch
import torch.nn as nn
from oryn_engine.native_sopro.config import ModelConfig
from oryn_engine.native_sopro.nn.acoustic import AcousticHead
from oryn_engine.native_sopro.nn.ar import SemanticLM, StylePrefixEncoder
from oryn_engine.native_sopro.nn.decode import EagerDecoder
from oryn_engine.native_sopro.sampling import sample_next_token
GRAPH_CAPACITY = 2048

class SoproModel(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cond_proj = nn.Sequential(nn.Linear(int(cfg.cond_in_dim), int(cfg.cond_hidden_dim)), nn.SiLU(), nn.Identity(), nn.Linear(int(cfg.cond_hidden_dim), int(cfg.cond_hidden_dim)))
        self.text_tok_emb = nn.Embedding(int(cfg.text_vocab_size), int(cfg.ar_model_dim))
        self.semantic_tok_emb = nn.Embedding(int(cfg.semantic_vocab_size) + 2, int(cfg.latent_dim))
        self.sem_in_proj = nn.Linear(int(cfg.latent_dim), int(cfg.ar_model_dim))
        self.style_prefix = StylePrefixEncoder(cfg)
        self.ar_prior = SemanticLM(cfg)
        self.acoustic_head = AcousticHead(cfg)
        self._graph_decoder = None

    @property
    def device(self) -> torch.device:
        return self.text_tok_emb.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.text_tok_emb.weight.dtype

    def build_condition(self, id_emb: torch.Tensor, style_emb: torch.Tensor, style_ctrl: torch.Tensor) -> torch.Tensor:
        return self.cond_proj(torch.cat([id_emb, style_emb, style_ctrl], dim=-1))

    def embed_semantic(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.sem_in_proj(self.semantic_tok_emb(tokens.to(dtype=torch.long)))

    def build_prefix(self, text_ids: torch.Tensor, style_tokens: torch.Tensor, prompt_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        text_ids = text_ids[:, :int(self.cfg.max_text_len)]
        bos = torch.full((1, 1), int(self.cfg.semantic_bos_id), device=text_ids.device, dtype=torch.long)
        bos_emb = self.embed_semantic(bos)
        parts = [self.style_prefix(self.embed_semantic(style_tokens)).to(dtype=bos_emb.dtype), self.text_tok_emb(text_ids).to(dtype=bos_emb.dtype)]
        if prompt_tokens is not None and int(prompt_tokens.shape[-1]) > 0:
            parts.append(self.embed_semantic(prompt_tokens).to(dtype=bos_emb.dtype))
        parts.append(bos_emb)
        return torch.cat(parts, dim=1)
