"""Embedding models (Stage 4).

The dimension is always read from the loaded model — never configured,
never assumed.
"""

from app.embeddings.base import EmbeddingError, EmbeddingModel
from app.embeddings.deterministic import DeterministicEmbedding
from app.embeddings.factory import (
    build_embedding_model,
    embed_in_batches,
    get_embedding_model,
    set_embedding_model,
)

__all__ = [
    "EmbeddingModel",
    "EmbeddingError",
    "DeterministicEmbedding",
    "build_embedding_model",
    "get_embedding_model",
    "set_embedding_model",
    "embed_in_batches",
]
