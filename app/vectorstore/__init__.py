"""Vector storage (Stage 4).

The RAG pipeline depends on ``VectorStore``, ``SearchFilter`` and
``SearchHit`` — never on a backend client. Swapping ChromaDB for
Pinecone means writing one adapter and registering it.
"""

from app.vectorstore.base import (
    BackendUnavailableError,
    DimensionMismatchError,
    SearchFilter,
    SearchHit,
    StoreStats,
    TenantScopeError,
    VectorStore,
    VectorStoreError,
)
from app.vectorstore.chroma import ChromaVectorStore
from app.vectorstore.factory import (
    available_backends,
    build_vector_store,
    get_vector_store,
    register_backend,
    set_vector_store,
)
from app.vectorstore.memory import InMemoryVectorStore

__all__ = [
    "VectorStore",
    "SearchFilter",
    "SearchHit",
    "StoreStats",
    "VectorStoreError",
    "TenantScopeError",
    "DimensionMismatchError",
    "BackendUnavailableError",
    "ChromaVectorStore",
    "InMemoryVectorStore",
    "build_vector_store",
    "get_vector_store",
    "set_vector_store",
    "register_backend",
    "available_backends",
]
