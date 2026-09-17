"""Deterministic hashing encoder — development and CI only.

**Not a semantic model.** It maps tokens to fixed positions by hash, so
it captures lexical overlap and nothing else: "terminate the agreement"
and "end the contract" are near-orthogonal to it. It must never be used
to serve real retrieval.

It exists for three concrete reasons:

1. **The test suite must not download a model.** A 90 MB fetch on every
   CI run is slow, and it fails outright in an air-gapped or
   egress-restricted environment.
2. **It is a second implementation of the interface.** Every dimension,
   normalisation and batching contract in ``base.py`` is exercised by
   two encoders, so the interface is proven rather than assumed.
3. **It keeps the service startable.** When a model cannot be loaded,
   falling back here with a loud warning beats refusing to boot — the
   health endpoint reports the degradation and no query silently
   returns nonsense without it being visible.

The dimension is configurable precisely so that tests can run at widths
other than 768 and catch any hard-coded assumption.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Sequence

from app.embeddings.base import EmbeddingModel

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class DeterministicEmbedding(EmbeddingModel):
    """Signed hashing (a "hashing trick") bag-of-words encoder."""

    name = "deterministic"

    def __init__(self, dimension: int = 384, normalize: bool = True) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self._dimension = int(dimension)
        self._normalize = bool(normalize)
        self.model_id = f"deterministic-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normalized(self) -> bool:
        return self._normalize

    def _vector(self, text: str) -> List[float]:
        vector = [0.0] * self._dimension
        tokens = _TOKEN_RE.findall((text or "").lower())
        if not tokens:
            return vector

        for position, token in enumerate(tokens):
            # Unigram plus bigram: bigrams give word order a little
            # weight, so "notice of termination" and "termination of
            # notice" are not identical vectors.
            grams = [token]
            if position:
                grams.append(f"{tokens[position - 1]}~{token}")
            for gram in grams:
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self._dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign

        # Sub-linear damping, the same idea as log term frequency: a word
        # repeated 40 times should not dominate the vector.
        vector = [math.copysign(math.log1p(abs(v)), v) for v in vector]

        if self._normalize:
            norm = math.sqrt(sum(v * v for v in vector))
            if norm > 1e-12:
                vector = [v / norm for v in vector]
        return vector

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._vector(text) for text in texts]


__all__ = ["DeterministicEmbedding"]
