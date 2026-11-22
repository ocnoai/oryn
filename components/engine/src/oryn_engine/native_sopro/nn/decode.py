# Derived from Sopro 2.2.0 (Apache-2.0); see LICENSE-Sopro.txt and provenance.json.
from __future__ import annotations
from typing import Dict, Tuple
import torch
from oryn_engine.native_sopro.nn.ar import KVCache
GRAPH_BUCKET = 256

class EagerDecoder:

    def __init__(self, model, capacity: int) -> None:
        self.model = model
        self.capacity = int(capacity)
        self.cache: KVCache = model.ar_prior.new_cache(1, self.capacity, model.device, model.dtype)

    def prefill(self, prefix: torch.Tensor) -> torch.Tensor:
        self.cache.length = 0
        return self.model.ar_prior(prefix, self.cache)

    def step(self, tok: torch.Tensor) -> torch.Tensor:
        return self.model.ar_prior(self.model.embed_semantic(tok.view(1, 1)), self.cache)
