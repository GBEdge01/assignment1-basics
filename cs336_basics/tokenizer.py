from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from functools import lru_cache

import regex as re


GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.pretoken_pattern = re.compile(GPT2_PRETOKEN_PATTERN)
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)

        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                token_id = len(self.vocab)
                self.vocab[token_id] = token_bytes
                self.token_to_id[token_bytes] = token_id

        if self.special_tokens:
            special_pattern = "|".join(re.escape(token) for token in self.special_tokens)
            self.special_pattern = re.compile(f"({special_pattern})")
            self.special_token_set = set(self.special_tokens)
        else:
            self.special_pattern = None
            self.special_token_set = set()

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        byte_decoder = {v: k for k, v in _gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            gpt2_vocab = json.load(vocab_file)

        vocab = {
            token_id: bytes(byte_decoder[char] for char in token)
            for token, token_id in gpt2_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as merges_file:
            for line in merges_file:
                parts = line.rstrip().split(" ")
                if len(parts) == 2:
                    merges.append(
                        (
                            bytes(byte_decoder[char] for char in parts[0]),
                            bytes(byte_decoder[char] for char in parts[1]),
                        )
                    )

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for chunk in self._split_on_special_tokens(text):
            if chunk in self.special_token_set:
                token_ids.append(self.token_to_id[chunk.encode("utf-8")])
            else:
                token_ids.extend(self._encode_ordinary(chunk))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: Iterable[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")

    def _split_on_special_tokens(self, text: str) -> list[str]:
        if self.special_pattern is None:
            return [text]
        return [part for part in self.special_pattern.split(text) if part]

    def _encode_ordinary(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for match in self.pretoken_pattern.finditer(text):
            pretoken = match.group(0).encode("utf-8")
            token_ids.extend(self.token_to_id[token] for token in self._bpe(pretoken))
        return token_ids

    @lru_cache(maxsize=100_000)
    def _bpe(self, pretoken: bytes) -> tuple[bytes, ...]:
        if len(pretoken) <= 1:
            return (pretoken,) if pretoken else ()

        parts = tuple(bytes([byte]) for byte in pretoken)
        while len(parts) > 1:
            ranked_pairs = [
                (self.merge_ranks[pair], pair) for pair in zip(parts, parts[1:]) if pair in self.merge_ranks
            ]
            if not ranked_pairs:
                break

            _, pair_to_merge = min(ranked_pairs)
            merged_parts: list[bytes] = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == pair_to_merge[0] and parts[i + 1] == pair_to_merge[1]:
                    merged_parts.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged_parts.append(parts[i])
                    i += 1
            parts = tuple(merged_parts)

        return parts


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for byte in range(2**8):
        if byte not in bs:
            bs.append(byte)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(codepoint) for codepoint in cs)))
