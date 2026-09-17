"""Vector-store backend selection.

``VECTOR_BACKEND`` names a registered backend; :func:`get_vector_store`
builds it at the *embedding model's actual dimension*, never a constant.
That coupling is the point of the factory: the store cannot be created
without asking the model how wide its vectors are.

Adding Pinecone later is a new module implementing :class:`VectorStore`
plus one ``register_backend`` call. No caller changes, because no caller
names a backend.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional

from app.config import Settings, get_settings
from app.embeddings.base import EmbeddingModel
from app.embeddings.factory import get_embedding_model
from app.logging_config import get_logger
from app.vectorstore.base import VectorStore, VectorStoreError
from app.vectorstore.chroma import ChromaVectorStore
from app.vectorstore.memory import InMemoryVectorStore

logger = get_logger(__name__)

#: name -> (settings, model) -> VectorStore
Builder = Callable[[Settings, EmbeddingModel], VectorStore]

_BACKENDS: Dict[str, Builder] = {}
_store: Optional[VectorStore] = None
_lock = threading.Lock()


def register_backend(name: str, builder: Builder) -> None:
    _BACKENDS[name.strip().lower()] = builder


def available_backends() -> List[str]:
    return sorted(_BACKENDS)


def _build_chroma(settings: Settings, model: EmbeddingModel) -> VectorStore:
    return ChromaVectorStore(
        persist_directory=settings.vector_persist_dir,
        collection_name=settings.vector_collection,
        dimension=model.dimension,
        metric=settings.vector_metric,
        embedding_model_id=model.model_id,
    )


def _build_memory(settings: Settings, model: EmbeddingModel) -> VectorStore:
    return InMemoryVectorStore(
        dimension=model.dimension,
        metric=settings.vector_metric,
        embedding_model_id=model.model_id,
        collection_name=settings.vector_collection,
    )


register_backend("chroma", _build_chroma)
register_backend("memory", _build_memory)


def build_vector_store(
    settings: Optional[Settings] = None,
    model: Optional[EmbeddingModel] = None,
) -> VectorStore:
    settings = settings or get_settings()
    model = model or get_embedding_model(settings)

    name = (settings.vector_backend or "chroma").strip().lower()
    builder = _BACKENDS.get(name)
    if builder is None:
        raise VectorStoreError(
            f"Unknown VECTOR_BACKEND '{name}'. Available: "
            f"{', '.join(available_backends())}"
        )

    store = builder(settings, model)
    logger.info(
        "Vector store: backend=%s collection=%s dimension=%d metric=%s "
        "model=%s",
        store.name,
        settings.vector_collection,
        model.dimension,
        settings.vector_metric,
        model.model_id,
    )
    return store


def get_vector_store(settings: Optional[Settings] = None) -> VectorStore:
    """The process-wide store singleton."""
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = build_vector_store(settings)
    return _store


def set_vector_store(store: Optional[VectorStore]) -> None:
    """Install a specific store, or ``None`` to rebuild from settings."""
    global _store
    with _lock:
        _store = store


__all__ = [
    "register_backend",
    "available_backends",
    "build_vector_store",
    "get_vector_store",
    "set_vector_store",
    "Builder",
]
