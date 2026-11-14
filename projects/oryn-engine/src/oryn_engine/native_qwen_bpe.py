"""Bounded original Qwen byte BPE data shared by owned video adapters."""

import json


def load_qwen_bpe(root, base_vocab_size, max_vocab_size, decoder):
    from tokenizers import AddedToken
    from transformers import Qwen2Tokenizer

    if (
        root.is_symlink()
        or not 256 <= base_vocab_size <= 151643
        or not base_vocab_size < max_vocab_size <= 152064
    ):
        raise ValueError("QWEN_BPE_SCOPE")
    for name in ("vocab.json", "merges.txt"):
        path = root / name
        if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 4 * 1024**2:
            raise ValueError("QWEN_BPE_SCOPE")
    if sorted(int(k) for k in decoder) != list(
        range(base_vocab_size, base_vocab_size + len(decoder))
    ):
        raise ValueError("QWEN_BPE_TOKEN_IDS")

    def unique(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("QWEN_BPE_VOCABULARY")
        return result

    # Vocabulary exceeds the general config object limit; parse as a bounded,
    # flat token-to-integer table without relaxing configuration validation.
    vocab = json.loads((root / "vocab.json").read_bytes(), object_pairs_hook=unique)
    if (
        not isinstance(vocab, dict)
        or len(vocab) != base_vocab_size
        or any(not isinstance(k, str) or not k or len(k) > 1024 for k in vocab)
        or any(type(v) is not int for v in vocab.values())
        or set(vocab.values()) != set(range(len(vocab)))
    ):
        raise ValueError("QWEN_BPE_VOCABULARY")
    lines = (root / "merges.txt").read_text().splitlines()
    if not lines or lines[0] != "#version: 0.2" or len(lines) > 200000:
        raise ValueError("QWEN_BPE_MERGES")
    merges = [tuple(line.split()) for line in lines[1:] if line]
    if any(
        len(pair) != 2 or any(x not in vocab for x in (*pair, "".join(pair))) for pair in merges
    ):
        raise ValueError("QWEN_BPE_MERGES")
    result = Qwen2Tokenizer(
        vocab=vocab,
        merges=merges,
        unk_token=None,
        bos_token=None,
        eos_token=None,
        pad_token=None,
        clean_up_tokenization_spaces=False,
    )
    for index, description in sorted(decoder.items(), key=lambda pair: int(pair[0])):
        result.add_tokens([AddedToken(**description)])
        if result.convert_tokens_to_ids(description["content"]) != int(index):
            raise ValueError("QWEN_BPE_TOKEN_IDS")
    result.eos_token = "<|im_end|>"
    result.pad_token = "<|endoftext|>"
    result.chat_template = None
    if len(result) > max_vocab_size:
        raise ValueError("QWEN_BPE_TOKEN_IDS")
    return result
