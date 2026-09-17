"""Embedding model interface.

One rule dominates this module: **the dimension is read from the model,
never assumed.** A hard-coded 768 is the single most common way to break
a vector pipeline — it is right for ``bert-base`` and wrong for
``all-MiniLM-L6-v2`` (384), ``all-mpnet-base-v2`` (768),
``bge-large-en`` (1024), ``text-embedding-3-large`` (3072) and most
multilingual models. Every consumer asks the model.

The interface is deliberately small. An implementation must provide:

* ``dimension``            the real output width, discovered at load time
* ``embed_documents``      batch encoding for indexing
* ``embed_query``          single encoding for search
* ``name`` / ``model_id``  identification, recorded with the index

``embed_query`` is separate from ``embed_documents`` because several
modern embedding models are asymmetric — they expect an instruction
prefix on the query side and not on the document side. Collapsing the
two would make those models silently under-perform.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence


class EmbeddingError(RuntimeError):
    """Raised when a model cannot be loaded or an encode call fails."""


class EmbeddingModel(ABC):
    """A text-to-vector encoder."""

    #: Short provider name, e.g. "sentence_transformers".
    name: str = "embedding_model"
    #: The specific model, e.g. "sentence-transformers/all-MiniLM-L6-v2".
    model_id: str = ""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output width, determined from the loaded model."""

    @property
    def max_sequence_length(self) -> int:
        """Tokens the model accepts before truncating. 0 when unknown.

        Exposed so the indexer can warn when a chunk will be silently
        cut short — losing the tail of a clause without telling anyone
        is worse than a slow index.
        """
        return 0

    @property
    def normalized(self) -> bool:
        """True when vectors are unit length.

        With unit vectors, cosine similarity and inner product agree,
        which lets a backend use whichever it implements faster.
        """
        return False

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode texts for indexing. Order is preserved."""

    def embed_query(self, text: str) -> List[float]:
        """Encode one search query. Symmetric models reuse the doc path."""
        vectors = self.embed_documents([text])
        if not vectors:
            raise EmbeddingError("The model returned no vector for the query")
        return vectors[0]

    def describe(self) -> dict:
        return {
            "provider": self.name,
            "model": self.model_id,
            "dimension": self.dimension,
            "max_sequence_length": self.max_sequence_length,
            "normalized": self.normalized,
        }


__all__ = ["EmbeddingModel", "EmbeddingError"]
