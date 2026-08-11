"""Deterministic local embeddings for tests and mock-provider benchmarks.

These are NOT a substitute for `text-embedding-3-small`. They make exact
and lightly paraphrased prompts similar via canonicalization + token hashing,
which is enough to exercise cache hit/miss paths without paid APIs.
"""

from __future__ import annotations

import hashlib
import math
import re

_FILLER = re.compile(r"\b(to me|please|in simple terms|simply|can you|could you)\b")
_PREFIXES = (
    "what is",
    "what's",
    "whats",
    "what are",
    "explain",
    "describe",
    "tell me about",
    "define",
    "give me an overview of",
    "overview of",
)


def canonicalize_for_mock(text: str) -> str:
    parts: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        for role in ("user: ", "assistant: ", "system: "):
            if lowered.startswith(role):
                stripped = stripped[len(role) :]
                break
        parts.append(stripped)
    value = " ".join(parts).lower().strip().rstrip("?.!")
    value = re.sub(r"\s+", " ", value)
    value = _FILLER.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    for prefix in sorted(_PREFIXES, key=len, reverse=True):
        if value == prefix or value.startswith(prefix + " "):
            rest = value[len(prefix) :].strip()
            value = f"explain {rest}".strip()
            break
    return value


def hashed_embedding(text: str, dimensions: int) -> list[float]:
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    cleaned = canonicalize_for_mock(text)
    if not cleaned:
        raise ValueError("Cannot embed empty input")

    vector = [0.0] * dimensions
    tokens = re.findall(r"[a-z0-9]+", cleaned)
    if not tokens:
        tokens = ["empty"]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0
        # A second hash bucket reduces accidental collisions for short prompts.
        index2 = int.from_bytes(digest[4:8], "big") % dimensions
        vector[index2] += 0.5

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class MockEmbeddingService:
    def __init__(self, dimensions: int = 1536) -> None:
        self._dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        return hashed_embedding(text, self._dimensions)
