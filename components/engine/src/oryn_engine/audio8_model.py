"""Audio8's owned delay modulation around the pinned native Qwen2 decoder.

Adapted from Edge0-AI/Audio8-ASR-Infinite, Apache-2.0,
c8ba8eea829be0339e8d7757f8ca52dac06e1e32. See third_party/Audio8-LICENSE.
"""

import copy

import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.voxtral_realtime.configuration_voxtral_realtime import (
    VoxtralRealtimeEncoderConfig,
)
from transformers.models.voxtral_realtime.modeling_voxtral_realtime import (
    VoxtralRealtimeEncoder,
    VoxtralRealtimeTextAdaRmsNorm,
    VoxtralRealtimeTimeEmbedding,
)


class DelayDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.ada_rms_norm = VoxtralRealtimeTextAdaRmsNorm(config)

    def forward(self, hidden_states, t_cond=None, **kwargs):
        if t_cond is None:
            raise ValueError("AUDIO8_DELAY_REQUIRED")
        residual = hidden_states
        hidden_states, _ = self.self_attn(self.input_layernorm(hidden_states), **kwargs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states * (1 + self.ada_rms_norm(t_cond))
        return residual + self.mlp(hidden_states)


class MaxFrameProjector(nn.Module):
    def __init__(self, audio_hidden, text_hidden):
        super().__init__()
        self.linear_1 = nn.Linear(audio_hidden * 8, text_hidden, bias=False)
        self.linear_2 = nn.Linear(text_hidden, text_hidden, bias=False)

    def forward(self, inputs):
        return self.linear_2(torch.nn.functional.gelu(self.linear_1(inputs)))


class Audio8Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        audio = VoxtralRealtimeEncoderConfig(**config["audio_config"])
        text = Qwen2Config(**config["text_config"])
        audio._attn_implementation = text._attn_implementation = "sdpa"
        self.audio_tower = VoxtralRealtimeEncoder(audio)
        self.language_model = Qwen2ForCausalLM(text)
        self.language_model.model.layers = nn.ModuleList(
            DelayDecoderLayer(text, i) for i in range(text.num_hidden_layers)
        )
        self.multi_modal_projector = MaxFrameProjector(audio.hidden_size, text.hidden_size)
        self.time_embedding = VoxtralRealtimeTimeEmbedding(text.hidden_size)
        self.frame_len_embedding = nn.Embedding(3, text.hidden_size)
        self.semantic_vad_heads = nn.ModuleList(nn.Linear(text.hidden_size, 8) for _ in range(4))

    def condition(self):
        reference = self.frame_len_embedding.weight
        phase = self.time_embedding.inv_freq.to(reference) * 6.0
        return (torch.cat((phase.cos(), phase.sin())) + reference[0])[None, None, :]

    def project(self, hidden, count):
        grouped = hidden.reshape(1, count, 4, self.config["audio_config"]["hidden_size"])
        grouped = torch.nn.functional.pad(grouped, (0, 0, 0, 4))
        return self.multi_modal_projector(grouped.flatten(2))
