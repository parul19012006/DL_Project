"""Embedding provider selection and batching.

Two things live here:

**The factory.** ``EMBEDDING_PROVIDER`` picks an implementation. Loading
a model costs seconds and hundreds of megabytes of RAM, so the result is
a process-wide singleton — building one per request would make the
service unusable.

**The batching helper.** Sentence-Transformers batches internally, but
the *outer* batch matters too: handing it 40,000 chunks in one call
materialises 40,000 vectors in memory before a single one is written.
:func:`embed_in_batches` walks a long list in bounded slices and reports
progress, so a 500-document ingest has a flat memory profile.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Iterable, Iterator, List, Optional, Sequence

from app.config import Settings, get_settings
from app.embeddings.base import EmbeddingError, EmbeddingModel
from app.embeddings.deterministic import DeterministicEmbedding
from app.logging_config import get_logger

logger = get_logger(__name__)

_model: Optional[EmbeddingModel] = None
_lock = threading.Lock()


def build_embedding_model(settings: Optional[Settings] = None) -> EmbeddingModel:
    """Construct the configured encoder.

    A failure to load the real model falls back to the deterministic
    encoder with a loud error rather than refusing to start: the service
    stays reachable, ``/health`` reports ``degraded``, and the operator
    sees why. Set ``EMBEDDING_STRICT=true`` to make a load failure fatal
    instead — the right choice for production, where serving nonsense
    vectors is worse than being down.
    """
    settings = settings or get_settings()
    provider = (settings.embedding_provider or "").strip().lower()

    if provider in ("deterministic", "hash", "test"):
        return DeterministicEmbedding(
            dimension=settings.deterministic_embedding_dim,
            normalize=settings.normalize_embeddings,
        )

    try:
        from app.embeddings.sentence_transformer import (
            SentenceTransformerEmbedding,
        )

        return SentenceTransformerEmbedding(
            model_id=settings.embedding_model,
            device=settings.embedding_device,
            normalize=settings.normalize_embeddings,
            batch_size=settings.embedding_batch_size,
            trust_remote_code=settings.embedding_trust_remote_code,
            cache_dir=settings.embedding_cache_dir,
        )
    except Exception as exc:
        if settings.embedding_strict:
            raise EmbeddingError(
                f"Could not load embedding model "
                f"'{settings.embedding_model}': {exc}"
            ) from exc
        logger.error(
            "Could not load embedding model '%s' (%s). Falling back to the "
            "deterministic encoder — retrieval quality will be poor and "
            "/health will report degraded. Set EMBEDDING_STRICT=true to make "
            "this fatal instead.",
            settings.embedding_model,
            exc,
        )
        return DeterministicEmbedding(
            dimension=settings.deterministic_embedding_dim,
            normalize=settings.normalize_embeddings,
        )


def get_embedding_model(settings: Optional[Settings] = None) -> EmbeddingModel:
    """The process-wide encoder singleton."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = build_embedding_model(settings)
    return _model


def set_embedding_model(model: Optional[EmbeddingModel]) -> None:
    """Install a specific model, or ``None`` to rebuild from settings."""
    global _model
    with _lock:
        _model = model


def embed_in_batches(
    model: EmbeddingModel,
    texts: Sequence[str],
    batch_size: int = 32,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[List[float]]:
    """Encode ``texts`` in bounded slices, preserving order.

    Order preservation matters: the caller zips the result back against
    its chunks, and a reordering here would attach every vector to the
    wrong passage — a bug that produces plausible-looking nonsense
    rather than an error.
    """
    if not texts:
        return []

    size = max(1, int(batch_size))
    out: List[List[float]] = []
    started = time.perf_counter()

    for start in range(0, len(texts), size):
        window = list(texts[start : start + size])
        vectors = model.embed_documents(window)
        if len(vectors) != len(window):
            raise EmbeddingError(
                f"The encoder returned {len(vectors)} vectors for "
                f"{len(window)} inputs; order cannot be trusted"
            )
        out.extend(vectors)
        if on_progress:
            on_progress(min(start + size, len(texts)), len(texts))

    elapsed = time.perf_counter() - started
    if len(texts) > 100:
        logger.info(
            "Embedded %d texts in %.1fs (%.0f/s) with %s",
            len(texts),
            elapsed,
            len(texts) / elapsed if elapsed else 0.0,
            model.model_id,
        )
    return out


def iter_batches(items: Sequence, size: int) -> Iterator[List]:
    """Yield ``items`` in lists of at most ``size``."""
    size = max(1, int(size))
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


__all__ = [
    "build_embedding_model",
    "get_embedding_model",
    "set_embedding_model",
    "embed_in_batches",
    "iter_batches",
]
