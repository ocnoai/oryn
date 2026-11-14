"""Keep Breeze's published tensor names across Transformers' Mimi module split."""

import torch
from transformers import MimiModel

_PREFIXES = (
    ("encoder.", "encoder.encoder_seanet."),
    ("encoder_transformer.", "encoder.encoder_transformer."),
    ("downsample.", "encoder.downsample."),
    ("decoder.", "decoder.decoder_seanet."),
    ("decoder_transformer.", "decoder.decoder_transformer."),
    ("upsample.", "decoder.upsample."),
)


def _rename(state, prefix, *, loading):
    # Stage replacements so overlapping encoder/decoder prefixes cannot cause
    # a second rewrite, and preserve strict state-dict validation for all keys.
    changed = {}
    for key in list(state):
        for checkpoint, native in _PREFIXES:
            before, after = (checkpoint, native) if loading else (native, checkpoint)
            if key.startswith(prefix + before):
                target = prefix + after + key[len(prefix + before) :]
                if target in state or target in changed:
                    raise ValueError("BREEZE_MIMI_DUPLICATE_WEIGHT")
                changed[target] = state.pop(key)
                break
    state.update(changed)


def _save(module, state, prefix, metadata):
    _rename(state, prefix, loading=False)


def _load(module, state, prefix, metadata, strict, missing, unexpected, errors):
    _rename(state, prefix, loading=True)


class CheckpointMimiModel(MimiModel):
    def __init__(self, config):
        # AutoModel now constructs the split halves using their checkpoint dtype.
        # Preserve the original constructor's uniform dtype; the validated loader
        # applies checkpoint precision to the complete graph after loading.
        config.dtype = torch.get_default_dtype()
        super().__init__(config)
        self.register_state_dict_post_hook(_save)
        self.register_load_state_dict_pre_hook(_load)
