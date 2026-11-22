# Copyright 2026 Xiaomi Corp.; Copyright 2025 Boson AI and HuggingFace.
# Adapted under Apache-2.0; see LICENSE.txt and provenance.json.
from __future__ import annotations
import logging
import math
import re
from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3Model
from .duration import RuleDurationEstimator
logger = logging.getLogger(__name__)


@dataclass
class OmniVoiceGenerationConfig:
    num_step: int = 32
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    denoise: bool = True
    preprocess_prompt: bool = True
    postprocess_output: bool = True
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0
    pad_duration: float = 0.1
    fade_duration: float = 0.1

    @classmethod
    def from_dict(cls, kwargs_dict):
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs_dict.items() if k in valid_keys}
        return cls(**filtered)

@dataclass
class GenerationTask:
    batch_size: int
    texts: List[str]
    target_lens: List[int]
    langs: List[Optional[str]]
    instructs: List[Optional[str]]
    ref_texts: List[Optional[str]]
    ref_audio_tokens: List[Optional[torch.Tensor]]
    ref_rms: List[Optional[float]]
    speed: Optional[List[float]] = None

    def get_indices(self, config: OmniVoiceGenerationConfig, frame_rate: int):
        threshold = int(config.audio_chunk_threshold * frame_rate)
        short_idx = [i for i, l in enumerate(self.target_lens) if l <= threshold]
        long_idx = [i for i, l in enumerate(self.target_lens) if l > threshold]
        return (short_idx, long_idx)

    def slice_task(self, indices: List[int]):
        if not indices:
            return None
        return GenerationTask(batch_size=len(indices), texts=[self.texts[i] for i in indices], target_lens=[self.target_lens[i] for i in indices], langs=[self.langs[i] for i in indices], instructs=[self.instructs[i] for i in indices], ref_texts=[self.ref_texts[i] for i in indices], ref_audio_tokens=[self.ref_audio_tokens[i] for i in indices], ref_rms=[self.ref_rms[i] for i in indices], speed=[self.speed[i] for i in indices] if self.speed else None)

def _filter_top_k(logits: torch.Tensor, ratio: float=0.1) -> torch.Tensor:
    k = math.ceil(ratio * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

def _gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled_logits = logits / temperature
    u = torch.rand_like(scaled_logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    return scaled_logits + gumbel_noise

def _get_time_steps(t_start: float=0.0, t_end: float=1.0, num_step: int=10, t_shift: float=1.0, device: torch.device=torch.device('cpu')) -> torch.Tensor:
    timesteps = torch.linspace(t_start, t_end, num_step + 1).to(device)
    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)
    return timesteps
_NONVERBAL_PATTERN = re.compile('\\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|surprise-yo|dissatisfaction-hnn)\\]')

def _tokenize_with_nonverbal_tags(text: str, tokenizer) -> torch.Tensor:
    """Tokenize text containing non-verbal tags, handling each tag independently.

    Non-verbal tags are tokenized standalone to guarantee consistent token
    IDs regardless of surrounding language context (Chinese, English, etc.).

    Args:
        text: Full text string potentially containing non-verbal tags.
        tokenizer: HuggingFace text tokenizer instance.
    Returns:
        Token IDs tensor of shape (1, seq_len).
    """
    parts = []
    last_end = 0
    for m in _NONVERBAL_PATTERN.finditer(text):
        if m.start() > last_end:
            segment = text[last_end:m.start()]
            ids = tokenizer(segment, add_special_tokens=False).input_ids
            if ids:
                parts.append(ids)
        tag_ids = tokenizer(m.group(), add_special_tokens=False).input_ids
        if tag_ids:
            parts.append(tag_ids)
        last_end = m.end()
    if last_end < len(text):
        segment = text[last_end:]
        ids = tokenizer(segment, add_special_tokens=False).input_ids
        if ids:
            parts.append(ids)
    if not parts:
        result = tokenizer(text, return_tensors='pt').input_ids
    else:
        combined = []
        for p in parts:
            combined.extend(p)
        result = torch.tensor([combined], dtype=torch.long)
    return result

def _combine_text(text, ref_text: Optional[str]=None) -> str:
    if ref_text:
        full_text = ref_text.strip() + ' ' + text.strip()
    else:
        full_text = text.strip()
    full_text = re.sub('[\\r\\n]+', '', full_text)
    full_text = full_text.replace('（', '(').replace('）', ')')
    full_text = re.sub('[ \\t]+', ' ', full_text)
    chinese_range = '[\\u4e00-\\u9fff]'
    pattern = f'(?<={chinese_range})\\s+|\\s+(?={chinese_range})'
    full_text = re.sub(pattern, '', full_text)
    return full_text

class OmniVoice(nn.Module):

    def __init__(self, config, tokenizer=None, *, attention='sdpa'):
        super().__init__()
        self.config = SimpleNamespace(**config)
        qconfig = Qwen3Config(**config['llm_config'])
        qconfig._attn_implementation = attention
        self.llm = Qwen3Model(qconfig)
        self.audio_embeddings = nn.Embedding(config['num_audio_codebook'] * config['audio_vocab_size'], qconfig.hidden_size)
        self.audio_heads = nn.Linear(qconfig.hidden_size, config['num_audio_codebook'] * config['audio_vocab_size'], bias=False)
        self.register_buffer('codebook_layer_offsets', torch.arange(config['num_audio_codebook']) * config['audio_vocab_size'])
        self.text_tokenizer = tokenizer
        self.duration_estimator = RuleDurationEstimator()

    @property
    def device(self):
        return self.audio_heads.weight.device

    def forward(self, input_ids, audio_mask, attention_mask):
        embeds = self._prepare_embed_inputs(input_ids, audio_mask)
        if self.llm.config._attn_implementation == 'eager':
            attention_mask = torch.zeros_like(attention_mask, dtype=embeds.dtype).masked_fill(~attention_mask, -torch.inf)
        hidden = self.llm(inputs_embeds=embeds, attention_mask=attention_mask, use_cache=False, return_dict=True)[0]
        logits = self.audio_heads(hidden).view(input_ids.shape[0], input_ids.shape[-1], self.config.num_audio_codebook, self.config.audio_vocab_size).permute(0, 2, 1, 3)
        if not torch.isfinite(logits).all():
            raise ValueError('OMNIVOICE_NONFINITE_LOGITS')
        return SimpleNamespace(logits=logits)

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def _prepare_embed_inputs(self, input_ids: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        """
        Prepares embeddings from input_ids of shape (batch_size, layers, seq_length).
        Embedding shape is (batch_size, seq_length, hidden_size).
        """
        text_embeds = self.get_input_embeddings()(input_ids[:, 0, :])
        shifted_ids = input_ids * audio_mask.unsqueeze(1) + self.codebook_layer_offsets.view(1, -1, 1)
        audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)
        return torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

    def _estimate_target_tokens(self, text, ref_text, num_ref_audio_tokens, speed=1.0):
        """Estimate number of target audio tokens."""
        if num_ref_audio_tokens is None or ref_text is None or len(ref_text) == 0:
            ref_text = 'Nice to meet you.'
            num_ref_audio_tokens = 25
        est = self.duration_estimator.estimate_duration(text, ref_text, num_ref_audio_tokens)
        if speed > 0 and speed != 1.0:
            est = est / speed
        return max(1, int(est))

    def _prepare_inference_inputs(self, text: str, num_target_tokens: int, ref_text: Optional[str]=None, ref_audio_tokens: Optional[torch.Tensor]=None, lang: Optional[str]=None, instruct: Optional[str]=None, denoise: bool=True):
        """Prepare input_ids and audio masks for inference.
        Args:
            text: Target text to generate.
            num_target_tokens: Number of audio tokens to generate.
            ref_text: Optional reference text for voice cloning.
            ref_audio_tokens: Optional reference audio tokens for voice cloning.
                with shape (C, T).
            lang: Optional language ID.
            instruct: Optional style instruction for voice design.
            denoise: Whether to include the <|denoise|> token.
        """
        style_text = ''
        if denoise and ref_audio_tokens is not None:
            style_text += '<|denoise|>'
        lang_str = lang if lang else 'None'
        instruct_str = instruct if instruct else 'None'
        style_text += f'<|lang_start|>{lang_str}<|lang_end|>'
        style_text += f'<|instruct_start|>{instruct_str}<|instruct_end|>'
        style_tokens = self.text_tokenizer(style_text, return_tensors='pt').input_ids.repeat(self.config.num_audio_codebook, 1).unsqueeze(0).to(self.device)
        full_text = _combine_text(ref_text=ref_text, text=text)
        wrapped_text = f'<|text_start|>{full_text}<|text_end|>'
        text_tokens = _tokenize_with_nonverbal_tags(wrapped_text, self.text_tokenizer).repeat(self.config.num_audio_codebook, 1).unsqueeze(0).to(self.device)
        target_audio_tokens = torch.full((1, self.config.num_audio_codebook, num_target_tokens), self.config.audio_mask_id, dtype=torch.long, device=self.device)
        parts = [style_tokens, text_tokens]
        if ref_audio_tokens is not None:
            parts.append(ref_audio_tokens.unsqueeze(0).to(self.device))
        parts.append(target_audio_tokens)
        cond_input_ids = torch.cat(parts, dim=2)
        cond_total_length = cond_input_ids.shape[2]
        cond_audio_start_idx = cond_total_length - num_target_tokens
        if ref_audio_tokens is not None:
            cond_audio_start_idx -= ref_audio_tokens.size(-1)
        cond_audio_mask = torch.zeros(1, cond_total_length, dtype=torch.bool, device=self.device)
        cond_audio_mask[0, cond_audio_start_idx:] = True
        return {'input_ids': cond_input_ids, 'audio_mask': cond_audio_mask}

    def _generate_iterative(self, task: GenerationTask, gen_config: OmniVoiceGenerationConfig) -> List[torch.Tensor]:
        """N-step iterative unmasked decoding.

        Args:
            task: A :class:`GenerationTask` containing batch texts, target
                lengths, languages, instructions, and optional reference data.
            gen_config: A :class:`OmniVoiceGenerationConfig` controlling
                decoding steps, guidance, temperatures, etc.
        Returns:
            List of generated audio token tensors of shape (C, T) (one per
            input text).
        """
        B = task.batch_size
        for i in range(B):
            logger.debug('Item %d — text: %s | ref_text: %s | instruct: %s | lang: %s | target_tokens: %d', i, task.texts[i], task.ref_texts[i], task.instructs[i], task.langs[i], task.target_lens[i])
        inputs_list = [self._prepare_inference_inputs(task.texts[i], task.target_lens[i], task.ref_texts[i], task.ref_audio_tokens[i], task.langs[i], task.instructs[i], gen_config.denoise) for i in range(B)]
        c_lens = [inp['input_ids'].size(2) for inp in inputs_list]
        max_c_len = max(c_lens)
        pad_id = self.config.audio_mask_id
        batch_input_ids = torch.full((2 * B, self.config.num_audio_codebook, max_c_len), pad_id, dtype=torch.long, device=self.device)
        batch_audio_mask = torch.zeros((2 * B, max_c_len), dtype=torch.bool, device=self.device)
        batch_attention_mask = torch.zeros((2 * B, 1, max_c_len, max_c_len), dtype=torch.bool, device=self.device)
        for i, inp in enumerate(inputs_list):
            c_len, u_len = (c_lens[i], task.target_lens[i])
            batch_input_ids[i, :, :c_len] = inp['input_ids']
            batch_audio_mask[i, :c_len] = inp['audio_mask']
            batch_attention_mask[i, :, :c_len, :c_len] = True
            batch_input_ids[B + i, :, :u_len] = inp['input_ids'][..., -u_len:]
            batch_audio_mask[B + i, :u_len] = inp['audio_mask'][..., -u_len:]
            batch_attention_mask[B + i, :, :u_len, :u_len] = True
            if max_c_len > u_len:
                pad_diag = torch.arange(u_len, max_c_len, device=self.device)
                batch_attention_mask[B + i, :, pad_diag, pad_diag] = True
        tokens = torch.full((B, self.config.num_audio_codebook, max(task.target_lens)), self.config.audio_mask_id, dtype=torch.long, device=self.device)
        timesteps = _get_time_steps(t_start=0.0, t_end=1.0, num_step=gen_config.num_step, t_shift=gen_config.t_shift).tolist()
        schedules = []
        for t_len in task.target_lens:
            total_mask = t_len * self.config.num_audio_codebook
            rem = total_mask
            sched = []
            for step in range(gen_config.num_step):
                num = rem if step == gen_config.num_step - 1 else min(math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem)
                sched.append(int(num))
                rem -= int(num)
            schedules.append(sched)
        layer_ids = torch.arange(self.config.num_audio_codebook, device=self.device).view(1, -1, 1)
        for step in range(gen_config.num_step):
            batch_logits = self(input_ids=batch_input_ids, audio_mask=batch_audio_mask, attention_mask=batch_attention_mask).logits.to(torch.float32)
            for i in range(B):
                k = schedules[i][step]
                if k <= 0:
                    continue
                c_len, t_len = (c_lens[i], task.target_lens[i])
                c_logits = batch_logits[i:i + 1, :, c_len - t_len:c_len, :]
                u_logits = batch_logits[B + i:B + i + 1, :, :t_len, :]
                pred_tokens, scores = self._predict_tokens_with_scoring(c_logits, u_logits, gen_config)
                scores = scores - layer_ids * gen_config.layer_penalty_factor
                if gen_config.position_temperature > 0.0:
                    scores = _gumbel_sample(scores, gen_config.position_temperature)
                sample_tokens = tokens[i:i + 1, :, :t_len]
                scores.masked_fill_(sample_tokens != self.config.audio_mask_id, -float('inf'))
                _, topk_idx = torch.topk(scores.flatten(), k)
                flat_tokens = sample_tokens.flatten()
                flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
                sample_tokens.copy_(flat_tokens.view_as(sample_tokens))
                tokens[i:i + 1, :, :t_len] = sample_tokens
                batch_input_ids[i:i + 1, :, c_len - t_len:c_len] = sample_tokens
                batch_input_ids[B + i:B + i + 1, :, :t_len] = sample_tokens
        return [tokens[i, :, :task.target_lens[i]] for i in range(B)]

    def _predict_tokens_with_scoring(self, c_logits, u_logits, gen_config):
        if gen_config.guidance_scale != 0:
            c_log_probs = F.log_softmax(c_logits, dim=-1)
            u_log_probs = F.log_softmax(u_logits, dim=-1)
            log_probs = torch.log_softmax(c_log_probs + gen_config.guidance_scale * (c_log_probs - u_log_probs), dim=-1)
        else:
            log_probs = F.log_softmax(c_logits, dim=-1)
        if not torch.isfinite(log_probs).all():
            raise ValueError('OMNIVOICE_NONFINITE_SCORES')
        log_probs[..., self.config.audio_mask_id] = -float('inf')
        if gen_config.class_temperature > 0.0:
            filtered_probs = _filter_top_k(log_probs, ratio=0.1)
            pred_tokens = _gumbel_sample(filtered_probs, gen_config.class_temperature).argmax(dim=-1)
        else:
            pred_tokens = log_probs.argmax(dim=-1)
        confidence_scores = log_probs.max(dim=-1)[0]
        return (pred_tokens, confidence_scores)
