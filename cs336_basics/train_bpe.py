from __future__ import annotations

import os
from collections import Counter, defaultdict

import regex as re


GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_pretokenization_chunks = kwargs.get("num_pretokenization_chunks", 128)
    vocab: dict[int, bytes] = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)}
    byte_offset = len(vocab)
    for byte in range(256):
        vocab[byte_offset + byte] = bytes([byte])

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return {i: vocab[i] for i in range(vocab_size)}, []

    word_counts = _pretoken_counts_from_file(input_path, special_tokens, num_pretokenization_chunks)
    pair_counts, pair_to_words = _initial_pair_stats(word_counts)

    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        _merge_pair_in_word_counts(best_pair, word_counts, pair_counts, pair_to_words)

    return vocab, merges


def _pretoken_counts_from_file(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    desired_num_chunks: int,
) -> Counter[tuple[bytes, ...]]:
    word_counts: Counter[tuple[bytes, ...]] = Counter()
    split_special_token = special_tokens[0].encode("utf-8") if special_tokens else b""

    with open(input_path, "rb") as file:
        if split_special_token:
            boundaries = _find_chunk_boundaries(file, desired_num_chunks, split_special_token)
        else:
            file.seek(0, os.SEEK_END)
            boundaries = [0, file.tell()]

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")
            chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
            word_counts.update(_pretoken_counts(chunk, special_tokens))

    return word_counts


def _find_chunk_boundaries(
    file,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1 or file_size == 0:
        return [0, file_size]

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for boundary_index in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[boundary_index]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[boundary_index] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[boundary_index] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    pattern = re.compile(GPT2_PRETOKEN_PATTERN)
    word_counts: Counter[tuple[bytes, ...]] = Counter()

    for chunk in _ordinary_chunks(text, special_tokens):
        for match in pattern.finditer(chunk):
            token_bytes = match.group(0).encode("utf-8")
            if token_bytes:
                word_counts[tuple(bytes([byte]) for byte in token_bytes)] += 1

    return word_counts


def _ordinary_chunks(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    special_pattern = "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    return [chunk for chunk in re.split(special_pattern, text) if chunk]


def _initial_pair_stats(
    word_counts: Counter[tuple[bytes, ...]],
) -> tuple[Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    for word, count in word_counts.items():
        for pair, pair_count in Counter(zip(word, word[1:])).items():
            pair_counts[pair] += pair_count * count
            pair_to_words[pair].add(word)

    return pair_counts, pair_to_words


def _merge_pair_in_word_counts(
    pair_to_merge: tuple[bytes, bytes],
    word_counts: Counter[tuple[bytes, ...]],
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    affected_words = list(pair_to_words.get(pair_to_merge, set()))
    merged_token = pair_to_merge[0] + pair_to_merge[1]

    for old_word in affected_words:
        count = word_counts.pop(old_word, 0)
        if count == 0:
            continue

        _remove_word_pair_stats(old_word, count, pair_counts, pair_to_words)
        new_word = _merge_word(old_word, pair_to_merge, merged_token)
        word_counts[new_word] += count
        _add_word_pair_stats(new_word, count, pair_counts, pair_to_words)


def _remove_word_pair_stats(
    word: tuple[bytes, ...],
    count: int,
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    for pair, pair_count in Counter(zip(word, word[1:])).items():
        pair_counts[pair] -= pair_count * count
        if pair_counts[pair] <= 0:
            del pair_counts[pair]
        pair_to_words[pair].discard(word)


def _add_word_pair_stats(
    word: tuple[bytes, ...],
    count: int,
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    for pair, pair_count in Counter(zip(word, word[1:])).items():
        pair_counts[pair] += pair_count * count
        pair_to_words[pair].add(word)


def _merge_word(
    word: tuple[bytes, ...],
    pair_to_merge: tuple[bytes, bytes],
    merged_token: bytes,
) -> tuple[bytes, ...]:
    merged: list[bytes] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair_to_merge[0] and word[i + 1] == pair_to_merge[1]:
            merged.append(merged_token)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)
