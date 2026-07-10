"""Shared text normalization and tokenization helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

DEFAULT_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "am",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "but",
        "by",
        "can",
        "do",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "him",
        "his",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "just",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "who",
        "will",
        "with",
        "you",
        "your",
    }
)


def normalize_text(value: object, *, lowercase: bool = True) -> str:
    """Return text as a normalized string without changing message content aggressively."""

    text = "" if value is None else str(value).strip()
    if lowercase:
        return text.lower()
    return text


def tokenize_text(
    value: object,
    *,
    lowercase: bool = True,
    remove_stopwords: bool = False,
    min_token_length: int = 1,
    filter_numeric_only: bool = True,
    stopwords: Iterable[str] = DEFAULT_STOPWORDS,
) -> list[str]:
    """Tokenize one SMS into deterministic word tokens."""

    if min_token_length < 1:
        raise ValueError("min_token_length must be at least 1")

    stopword_set = {word.lower() for word in stopwords}
    text = normalize_text(value, lowercase=lowercase)
    tokens = TOKEN_PATTERN.findall(text)

    filtered = []
    for token in tokens:
        if len(token) < min_token_length:
            continue
        if filter_numeric_only and token.isdigit():
            continue
        if remove_stopwords and token in stopword_set:
            continue
        filtered.append(token)
    return filtered


def make_ngrams(tokens: list[str], n: int) -> list[str]:
    """Return space-joined n-grams for a token sequence."""

    if n < 1:
        raise ValueError("n must be at least 1")
    if len(tokens) < n:
        return []
    return [" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
