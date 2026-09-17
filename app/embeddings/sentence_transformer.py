"""Sentence-Transformers embedding model — the production encoder.

Any Sentence-Transformers model can be selected with ``EMBEDDING_MODEL``,
including a local directory path for air-gapped deployments.

Dimension discovery, in order:

1. ``model.get_sentence_embedding_dimension()`` — the supported API
2. the pooling module's ``get_sentence_embedding_dimension()``
3. a probe encode, measuring the width of a real output vector

The probe is the backstop that cannot be fooled: whatever the model
config claims, the vector it actually returns is the vector the index
has to store. A model whose three sources disagree is refused rather
than indexed at the wrong width.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Sequence

from app.embeddings.base import EmbeddingError, EmbeddingModel
from app.logging_config import get_logger

logger = get_logger(__name__)

#: Text used to probe the model's true output width.
PROBE_TEXT = "dimension probe"


class SentenceTransformerEmbedding(EmbeddingModel):
    """Wraps a ``SentenceTransformer`` with real dimension detection."""

    name = "sentence_transformers"

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise EmbeddingError(
                "sentence-transformers is not installed. Install it, or set "
                "EMBEDDING_PROVIDER=deterministic for a dependency-free "
                "development encoder."
            ) from exc

        self.model_id = model_id
        self._normalize = bool(normalize)
        self._batch_size = max(1, int(batch_size))
        self._lock = threading.Lock()

        logger.info("Loading embedding model '%s'", model_id)
        try:
            kwargs = {"device": device}
            if trust_remote_code:
                kwargs["trust_remote_code"] = True
            if cache_dir:
                kwargs["cache_folder"] = cache_dir
            self._model = SentenceTransformer(model_id, **kwargs)
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load embedding model '{model_id}': {exc}"
            ) from exc

        self._dimension = self._discover_dimension()
        self._max_seq = self._discover_max_sequence_length()
        logger.info(
            "Embedding model '%s' ready (dimension=%d, max_seq=%s, "
            "normalized=%s)",
            model_id,
            self._dimension,
            self._max_seq or "unknown",
            self._normalize,
        )

    # -- discovery ----------------------------------------------------

    #: Sentence-Transformers 6.x renamed the accessor and deprecated the
    #: old name. Both are tried, newest first, so the wrapper works
    #: across versions without emitting a deprecation warning on 6.x.
    DIMENSION_ACCESSORS = (
        "get_embedding_dimension",
        "get_sentence_embedding_dimension",
    )

    def _discover_dimension(self) -> int:
        """Determine the true output width. Never assumed, never guessed."""
        declared: Optional[int] = None

        for accessor in self.DIMENSION_ACCESSORS:
            getter = getattr(self._model, accessor, None)
            if not callable(getter):
                continue
            try:
                value = getter()
                if value:
                    declared = int(value)
                    break
            except Exception as exc:  # pragma: no cover - model dependent
                logger.debug("%s failed: %s", accessor, exc)

        if declared is None:
            declared = self._dimension_from_modules()

        probed = self._probe_dimension()

        if declared is not None and declared != probed:
            # The config and the model disagree. Indexing at the wrong
            # width produces a collection that silently rejects or
            # corrupts later writes, so refuse rather than pick one.
            raise EmbeddingError(
                f"Model '{self.model_id}' reports dimension {declared} but "
                f"produces vectors of width {probed}. Refusing to index at "
                "an uncertain dimension."
            )
        return probed

    def _dimension_from_modules(self) -> Optional[int]:
        """Ask the pooling module, for models without the top-level API."""
        try:
            for module in reversed(list(self._model.modules())):
                for accessor in self.DIMENSION_ACCESSORS:
                    getter = getattr(module, accessor, None)
                    if callable(getter):
                        value = getter()
                        if value:
                            return int(value)
        except Exception:  # pragma: no cover - model dependent
            pass
        return None

    def _probe_dimension(self) -> int:
        """Encode one short string and measure the vector it returns."""
        try:
            vector = self._model.encode(
                PROBE_TEXT, convert_to_numpy=True, show_progress_bar=False
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Model '{self.model_id}' failed to encode a probe: {exc}"
            ) from exc

        width = int(getattr(vector, "shape", (len(vector),))[-1])
        if width <= 0:
            raise EmbeddingError(
                f"Model '{self.model_id}' produced an empty vector"
            )
        return width

    def _discover_max_sequence_length(self) -> int:
        try:
            value = getattr(self._model, "max_seq_length", 0)
            return int(value) if value else 0
        except Exception:  # pragma: no cover
            return 0

    # -- interface ----------------------------------------------------

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_sequence_length(self) -> int:
        return self._max_seq

    @property
    def normalized(self) -> bool:
        return self._normalize

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            # SentenceTransformer.encode is not documented as thread-safe
            # and the service may handle concurrent requests.
            with self._lock:
                vectors = self._model.encode(
                    list(texts),
                    batch_size=self._batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=self._normalize,
                    show_progress_bar=False,
                )
        except Exception as exc:
            raise EmbeddingError(f"Encoding failed: {exc}") from exc

        out = [[float(value) for value in row] for row in vectors]
        # Cheap invariant: a silent width change would corrupt the index.
        if out and len(out[0]) != self._dimension:
            raise EmbeddingError(
                f"Model '{self.model_id}' returned {len(out[0])}-dimensional "
                f"vectors but was loaded at {self._dimension}"
            )
        return out


__all__ = ["SentenceTransformerEmbedding", "PROBE_TEXT"]
