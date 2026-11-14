"""Complete Breeze native topology, with owned bounded inference operations.

Adapted from Breeze 6eba901d1845353802652c708a4740fdfc5baa75 under Apache-2.0;
see BREEZE-LICENSE.txt and provenance.json. No repository code is imported.
"""
from copy import deepcopy

import torch
from torch import nn
from transformers import LlamaConfig, LlamaModel, MimiConfig, Qwen3Config, Qwen3Model
from transformers.models.t5gemma2.configuration_t5gemma2 import T5Gemma2TextConfig
from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2TextEncoder
from .mimi import CheckpointMimiModel as MimiModel


def finite(x):
    if not bool(torch.isfinite(x).all()):
        raise ValueError("BREEZE_NONFINITE_OUTPUT")
    return x


class AudioEmbeddings(nn.Module):
    def __init__(self, config, embedding):
        super().__init__()
        self.embed_audio_tokens = embedding
        self.register_buffer("audio_tokens_offsets", torch.arange(config['num_codebooks']) * config['vocab_size'], persistent=False)

    def forward(self, codes):
        return self.embed_audio_tokens(codes + self.audio_tokens_offsets).sum(dim=2)


class DepthDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = LlamaConfig(**deepcopy(config))
        cfg._attn_implementation = "eager"
        self.model = LlamaModel(cfg)
        self.model.embed_tokens = nn.Embedding(config['num_codebooks'] * config['vocab_size'], config['audio_embed_size'])
        self.model.inputs_embeds_projector = nn.Linear(config['audio_embed_size'], config['hidden_size'], bias=False)
        self.codebooks_head = nn.Module()
        self.codebooks_head.weight = nn.Parameter(torch.empty(config['num_codebooks'] - 1, config['hidden_size'], config['vocab_size']))
        nn.init.normal_(self.codebooks_head.weight, std=config['initializer_range'])
        self.vocab_size = config['vocab_size']

    def step(self, ids, backbone_hidden, cache=None, *, use_cache=True):
        start = cache.get_seq_length() if cache is not None else 0
        positions = torch.arange(start, start + ids.shape[1], device=ids.device)
        embeds = self.model.embed_tokens(ids + (positions - 1).clamp(min=0) * self.vocab_size)
        if start == 0:
            embeds[:, 0] = backbone_hidden
        output = self.model(inputs_embeds=self.model.inputs_embeds_projector(embeds), past_key_values=cache, use_cache=use_cache)
        head = int(positions[-1]) - 1
        if not 0 <= head < self.codebooks_head.weight.shape[0]:
            raise ValueError("BREEZE_DEPTH_POSITION")
        logits = nn.functional.linear(output.last_hidden_state[:, -1], self.codebooks_head.weight[head].T)
        return finite(logits), output.past_key_values


class Breeze(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        q = Qwen3Config(**deepcopy(config['backbone_config']))
        q._attn_implementation = "eager"
        self.backbone_model = Qwen3Model(q)
        self.depth_decoder = DepthDecoder(config['depth_decoder_config'])
        self.backbone_model.embed_tokens = AudioEmbeddings(config, self.depth_decoder.model.embed_tokens)
        self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'] + 1, bias=False)
        self.embed_text_tokens = nn.Embedding(config['text_vocab_size'], config['hidden_size'])
        m = MimiConfig(**deepcopy(config['codec_config']))
        m._attn_implementation = "eager"
        self.codec_model = MimiModel(m)
        t = T5Gemma2TextConfig(**deepcopy(config['text_encoder_config']))
        t._attn_implementation = "eager"
        self.text_encoder = T5Gemma2TextEncoder(t, eoi_token_index=config['text_encoder_config']['eoi_token_index'])
        self.text_encoder_proj = nn.Linear(t.hidden_size, config['hidden_size'], bias=False)

    @property
    def device(self):
        return self.lm_head.weight.device

    def text_embeddings(self, ids):
        return finite(self.text_encoder_proj(self.text_encoder(input_ids=ids).last_hidden_state))

    def backbone_step(self, embeds, cache=None, *, use_cache=True):
        result = self.backbone_model(inputs_embeds=embeds, past_key_values=cache, use_cache=use_cache)
        hidden = finite(result.last_hidden_state[:, -1])
        return finite(self.lm_head(hidden)), hidden, result.past_key_values

    def audio_embeddings(self, codes):
        return self.backbone_model.embed_tokens(codes)


def sample(logits, history=None, *, eos=False):
    scores = finite(logits.float()).clone()
    if history:
        ids = torch.tensor(sorted(set(history)), dtype=torch.long, device=scores.device)
        values = scores[:, ids]
        scores[:, ids] = torch.where(values < 0, values * 1.1, values / 1.1)
    scores = finite(scores / 0.9)
    scores[:, 2048:2051] = -torch.inf
    if scores.shape[-1] != (2052 if eos else 2051):
        raise ValueError("BREEZE_LOGITS_SHAPE")
    threshold = scores.topk(50, dim=-1).values[:, -1:]
    scores.masked_fill_(scores < threshold, -torch.inf)
    probs = finite(scores.softmax(-1))
    if not bool((probs.sum(-1) > 0).all()):
        raise ValueError("BREEZE_EMPTY_DISTRIBUTION")
    return torch.multinomial(probs, 1)


def depth_frame(model, first, hidden):
    prefix = torch.cat((torch.zeros_like(first), first), dim=1)
    cache = None
    codes = [first]
    for _ in range(15):
        logits, cache = model.depth_decoder.step(prefix, hidden, cache)
        prefix = sample(logits)
        codes.append(prefix)
    return torch.cat(codes, dim=1)


def generate_codes(model, ids, max_frames):
    embeds = model.text_embeddings(ids)
    cache = None
    frames, history = [], []
    for index in range(max_frames + 1):
        logits, hidden, cache = model.backbone_step(embeds, cache)
        first = sample(logits, history, eos=True)
        ident = int(first.item())
        if ident == 2051:
            if not frames:
                raise ValueError("BREEZE_EMPTY_GENERATION")
            return torch.stack(frames, dim=1)
        if index == max_frames:
            raise ValueError("BREEZE_EOS_LIMIT")
        if not 0 <= ident < 2048:
            raise ValueError("BREEZE_AUDIO_CODE")
        frame = depth_frame(model, first, hidden)
        frames.append(frame)
        history.append(ident)
        embeds = model.audio_embeddings(frame.unsqueeze(1))
    raise ValueError("BREEZE_EOS_LIMIT")
