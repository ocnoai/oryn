"""Bounded Tekken-v7 byte vocabulary with build-owned framing and regex."""

from __future__ import annotations

import base64
import itertools
import json

PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
AUDIO = {
    "sampling_rate": 16000,
    "frame_rate": 12.5,
    "audio_encoding_config": {"num_mel_bins": 128, "hop_length": 160, "window_size": 400},
    "chunk_length_s": 30.0,
}
SPECIALS = {
    0: "<unk>",
    1: "<s>",
    2: "</s>",
    3: "[INST]",
    4: "[/INST]",
    11: "<pad>",
    24: "[AUDIO]",
    25: "[BEGIN_AUDIO]",
}


def read_vocabulary(path, model_vocab_size):
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 32 * 1024**2:
        raise ValueError("TEKKEN_FILE_UNAVAILABLE")

    def unique(pairs):
        values = {}
        for key, value in pairs:
            if key in values:
                raise ValueError("TEKKEN_DUPLICATE_KEY")
            values[key] = value
        return values

    try:
        data = json.loads(path.read_bytes(), object_pairs_hook=unique)
    except (RecursionError, UnicodeError) as exc:
        raise ValueError("TEKKEN_JSON_INVALID") from exc
    if not isinstance(data, dict) or set(data) != {"config", "audio", "vocab", "special_tokens"}:
        raise ValueError("TEKKEN_STRUCTURE_UNSUPPORTED")
    config = data["config"]
    if (
        not isinstance(config, dict)
        or set(config)
        != {
            "pattern",
            "num_vocab_tokens",
            "default_vocab_size",
            "default_num_special_tokens",
            "version",
        }
        or config["pattern"] != PATTERN
        or config["version"] != "v7"
        or type(config["default_num_special_tokens"]) is not int
        or config["default_num_special_tokens"] != 1000
        or type(config["default_vocab_size"]) is not int
        or config["default_vocab_size"] != model_vocab_size
        or not 1256 <= model_vocab_size <= 262144
        or type(config["num_vocab_tokens"]) is not int
        or not model_vocab_size - 1000 <= config["num_vocab_tokens"] <= 262144
        or json.dumps(data["audio"], sort_keys=True) != json.dumps(AUDIO, sort_keys=True)
    ):
        raise ValueError("TEKKEN_CONFIGURATION_UNSUPPORTED")
    specials = data["special_tokens"]
    if not isinstance(specials, list) or len(specials) != 1000:
        raise ValueError("TEKKEN_SPECIAL_TOKENS")
    seen_specials = set()
    for index, item in enumerate(specials):
        if (
            not isinstance(item, dict)
            or set(item) != {"rank", "token_str", "is_control"}
            or type(item["rank"]) is not int
            or item["rank"] != index
            or type(item["is_control"]) is not bool
            or not isinstance(item["token_str"], str)
            or not 1 <= len(item["token_str"]) <= 128
            or item["token_str"] in seen_specials
            or index in SPECIALS
            and (item["token_str"] != SPECIALS[index] or not item["is_control"])
        ):
            raise ValueError("TEKKEN_SPECIAL_TOKENS")
        seen_specials.add(item["token_str"])
    vocab = data["vocab"]
    if not isinstance(vocab, list) or len(vocab) != config["num_vocab_tokens"]:
        raise ValueError("TEKKEN_VOCABULARY_INVALID")
    ranks = {}
    seen = set()
    for index, item in enumerate(vocab):
        if (
            not isinstance(item, dict)
            or set(item) != {"rank", "token_bytes", "token_str"}
            or type(item["rank"]) is not int
            or item["rank"] != index
            or not isinstance(item["token_bytes"], str)
            or len(item["token_bytes"]) > 1368
            or item["token_str"] is not None
            and (not isinstance(item["token_str"], str) or len(item["token_str"]) > 1024)
        ):
            raise ValueError("TEKKEN_VOCABULARY_INVALID")
        token = base64.b64decode(item["token_bytes"], validate=True)
        if not 1 <= len(token) <= 1024 or token in seen or index < 256 and token != bytes([index]):
            raise ValueError("TEKKEN_VOCABULARY_INVALID")
        try:
            text = token.decode("utf-8")
        except UnicodeError:
            text = None
        if text != item["token_str"]:
            raise ValueError("TEKKEN_VOCABULARY_INVALID")
        seen.add(token)
        if index < model_vocab_size - 1000:
            ranks[token] = index
    return ranks


class TekkenV7:
    def __init__(self, path, vocab_size):
        import tiktoken

        ranks = read_vocabulary(path, vocab_size)
        self.vocab_size = vocab_size
        self.encoding = tiktoken.Encoding(
            name="oryn-tekken-v7", pat_str=PATTERN, mergeable_ranks=ranks, special_tokens={}
        )

    def encode(self, text):
        return [token + 1000 for token in self.encoding.encode_ordinary(text)]

    def decode(self, tokens):
        if any(type(token) is not int or not 0 <= token < self.vocab_size for token in tokens):
            raise ValueError("TEKKEN_OUTPUT_TOKEN_INVALID")
        return "".join(
            self.encoding.decode([token - 1000 for token in group])
            for special, group in itertools.groupby(tokens, lambda token: token < 1000)
            if not special
        )
