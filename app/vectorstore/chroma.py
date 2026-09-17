"""ChromaDB adapter.

Chroma is the right first backend: it runs embedded with no server to
operate, persists to a local directory, and supports metadata filtering
— everything Stage 5 needs, with zero infrastructure. When scale or
multi-replica writes demand it, a Pinecone adapter implements the same
:class:`VectorStore` interface and nothing upstream changes.

Three Chroma-specific details worth knowing:

**``embedding_function=None`` is mandatory here.** Left unset, Chroma
falls back to its bundled ONNX model and tries to download it on first
use — a network call at startup, and a second, contradictory encoder in
a service that already has one. This adapter always supplies its own
vectors.

**The collection records its dimension and model.** Chroma infers width
from the first insert and then rejects mismatches with an opaque error.
Recording both in collection metadata lets this adapter fail early with
a message that says what to do.

**Cosine distance is converted to a similarity.** Chroma returns
distance (lower is better); the interface promises "higher is better".
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Sequence

from app.logging_config import get_logger
from app.models.chunk import Chunk
from app.models.document import DocumentType
from app.vectorstore.base import (
    BackendUnavailableError,
    DimensionMismatchError,
    dedupe_by_chunk_id,
    SearchFilter,
    SearchHit,
    StoreStats,
    VectorStore,
    VectorStoreError,
    VersionTracker,
)

logger = get_logger(__name__)

DIMENSION_KEY = "embedding_dimension"
MODEL_KEY = "embedding_model"
METRIC_KEY = "hnsw:space"

#: Chroma requires 3-512 characters, starting and ending alphanumeric.
COLLECTION_NAME_HINT = (
    "Collection names must be 3-512 characters of [a-zA-Z0-9._-], "
    "starting and ending with a letter or digit."
)

#: Chroma accepts a bounded batch; larger upserts are chunked.
MAX_UPSERT_BATCH = 1000


class ChromaVectorStore(VectorStore):
    """Persistent embedded ChromaDB."""

    name = "chroma"

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        dimension: int,
        metric: str = "cosine",
        embedding_model_id: str = "",
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._dimension = int(dimension)
        self.metric = (metric or "cosine").lower()
        self.embedding_model_id = embedding_model_id

        self._client = None
        self._collection = None
        # Reentrant: the collection property reaches back into client.
        self._lock = threading.RLock()
        self._versions = VersionTracker()

    # -- lifecycle ----------------------------------------------------

    @property
    def dimension(self) -> int:
        return self._dimension

    def ensure_ready(self) -> None:
        self._get_collection()

    @property
    def client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._make_client()
        return self._client

    def _make_client(self):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailableError(
                "chromadb is not installed"
            ) from exc

        os.makedirs(self.persist_directory, exist_ok=True)
        logger.info("Opening Chroma store at %s", self.persist_directory)
        try:
            return chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False, allow_reset=True
                ),
            )
        except Exception as exc:
            raise BackendUnavailableError(
                f"Could not open the Chroma store: {exc}"
            ) from exc

    def _get_collection(self):
        if self._collection is None:
            with self._lock:
                if self._collection is None:
                    self._collection = self._open_collection()
        return self._collection

    def _open_collection(self):
        metadata = {
            METRIC_KEY: self.metric,
            DIMENSION_KEY: self._dimension,
            MODEL_KEY: self.embedding_model_id or "unknown",
        }
        try:
            # embedding_function=None: this service always supplies its
            # own vectors; the default would download an ONNX model.
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=metadata,
                embedding_function=None,
            )
        except TypeError:  # pragma: no cover - older chromadb
            collection = self.client.get_or_create_collection(
                name=self.collection_name, metadata=metadata
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Could not open collection '{self.collection_name}': {exc}. "
                f"{COLLECTION_NAME_HINT}"
            ) from exc

        self._verify_dimension(collection)
        return collection

    def _verify_dimension(self, collection) -> None:
        """Refuse to write into a collection built at another width."""
        existing = collection.metadata or {}
        recorded = existing.get(DIMENSION_KEY)
        if recorded is None:
            return
        if int(recorded) != self._dimension:
            raise DimensionMismatchError(
                f"Collection '{self.collection_name}' was created for "
                f"{recorded}-dimensional vectors (model "
                f"'{existing.get(MODEL_KEY, 'unknown')}') but the configured "
                f"model produces {self._dimension}. Changing the embedding "
                "model requires re-indexing into a new collection: set "
                "VECTOR_COLLECTION to a new name, or delete the old one."
            )

    # -- writes -------------------------------------------------------

    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"{len(chunks)} chunks but {len(vectors)} vectors"
            )
        for index, vector in enumerate(vectors):
            if len(vector) != self._dimension:
                raise DimensionMismatchError(
                    f"Vector {index} has width {len(vector)}, expected "
                    f"{self._dimension}"
                )

        # Chroma rejects a duplicate id inside a single call; normalise
        # to last-wins so every backend behaves identically.
        original = len(chunks)
        chunks, vectors = dedupe_by_chunk_id(chunks, vectors)
        if len(chunks) != original:
            logger.warning(
                "Collapsed %d duplicate chunk id(s) in one upsert batch",
                original - len(chunks),
            )

        collection = self._get_collection()
        total = 0
        for start in range(0, len(chunks), MAX_UPSERT_BATCH):
            window = chunks[start : start + MAX_UPSERT_BATCH]
            window_vectors = vectors[start : start + MAX_UPSERT_BATCH]
            try:
                collection.upsert(
                    ids=[c.chunk_id for c in window],
                    documents=[c.text for c in window],
                    embeddings=[list(map(float, v)) for v in window_vectors],
                    metadatas=[c.metadata() for c in window],
                )
            except Exception as exc:
                raise VectorStoreError(f"Upsert failed: {exc}") from exc
            total += len(window)
        self._versions.bump_chunks(chunks)
        return total

    def delete_document(self, document_id: str, tenant_id: str) -> int:
        self._versions.bump(tenant_id)
        filters = SearchFilter(tenant_id=tenant_id, document_ids=[document_id])
        return self._delete_where(self._where(filters))

    def delete_tenant(self, tenant_id: str) -> int:
        self._versions.bump(tenant_id)
        return self._delete_where(self._where(SearchFilter(tenant_id=tenant_id)))

    def _delete_where(self, where: Dict[str, Any]) -> int:
        collection = self._get_collection()
        try:
            existing = collection.get(where=where, include=[])
            ids = list(existing.get("ids") or [])
            if ids:
                collection.delete(ids=ids)
            return len(ids)
        except Exception as exc:
            raise VectorStoreError(f"Delete failed: {exc}") from exc

    # -- reads --------------------------------------------------------

    def search(
        self,
        query_vector: Sequence[float],
        filters: SearchFilter,
        top_k: int = 10,
    ) -> List[SearchHit]:
        if len(query_vector) != self._dimension:
            raise DimensionMismatchError(
                f"Query vector has width {len(query_vector)}, expected "
                f"{self._dimension}"
            )

        collection = self._get_collection()
        try:
            raw = collection.query(
                query_embeddings=[list(map(float, query_vector))],
                n_results=max(1, int(top_k)),
                where=self._where(filters),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Search failed: {exc}") from exc

        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        hits: List[SearchHit] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            # Defence in depth. The where clause already scopes by
            # tenant; this catches a backend bug or a hand-edited store
            # before another tenant's text reaches a user.
            if str(metadata.get("tenant_id")) != filters.tenant_id:
                logger.error(
                    "Dropping chunk %s: tenant mismatch (%r != %r)",
                    metadata.get("chunk_id"),
                    metadata.get("tenant_id"),
                    filters.tenant_id,
                )
                continue
            hits.append(
                SearchHit(
                    chunk=chunk_from_metadata(text or "", metadata),
                    score=self._to_similarity(distance),
                    raw_score=float(distance) if distance is not None else None,
                )
            )

        for rank, hit in enumerate(hits, start=1):
            hit.rank = rank
        return hits

    def fetch(
        self, filters: SearchFilter, limit: Optional[int] = None
    ) -> List[Chunk]:
        collection = self._get_collection()
        kwargs: Dict[str, Any] = {
            "where": self._where(filters),
            "include": ["documents", "metadatas"],
        }
        if limit:
            kwargs["limit"] = int(limit)
        try:
            raw = collection.get(**kwargs)
        except Exception as exc:
            raise VectorStoreError(f"Fetch failed: {exc}") from exc

        out: List[Chunk] = []
        for text, metadata in zip(
            raw.get("documents") or [], raw.get("metadatas") or []
        ):
            metadata = metadata or {}
            if str(metadata.get("tenant_id")) != filters.tenant_id:
                continue
            out.append(chunk_from_metadata(text or "", metadata))
        # Chroma's get() has no ordering guarantee; sort so callers see a
        # stable sequence.
        out.sort(key=lambda c: (c.document_id, c.chunk_index))
        return out

    def count(self, tenant_id: Optional[str] = None) -> int:
        collection = self._get_collection()
        try:
            if tenant_id:
                where = self._where(SearchFilter(tenant_id=tenant_id))
                return len(collection.get(where=where, include=[]).get("ids") or [])
            return int(collection.count())
        except Exception as exc:
            raise VectorStoreError(f"Count failed: {exc}") from exc

    def document_chunk_count(self, document_id: str, tenant_id: str) -> int:
        filters = SearchFilter(tenant_id=tenant_id, document_ids=[document_id])
        collection = self._get_collection()
        try:
            raw = collection.get(where=self._where(filters), include=[])
            return len(raw.get("ids") or [])
        except Exception as exc:
            raise VectorStoreError(f"Count failed: {exc}") from exc

    # -- introspection ------------------------------------------------

    def stats(self) -> StoreStats:
        return StoreStats(
            backend=self.name,
            collection=self.collection_name,
            dimension=self._dimension,
            total_chunks=self.count(),
            metric=self.metric,
            embedding_model=self.embedding_model_id,
        )

    def version(self, tenant_id: str) -> int:
        return self._versions.get(tenant_id)

    def reset(self) -> None:
        self._versions.reset()
        with self._lock:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:  # pragma: no cover - already absent
                pass
            self._collection = None

    # -- filter translation -------------------------------------------

    _OPS = {"eq": "$eq", "in": "$in", "gte": "$gte", "lte": "$lte"}

    def _where(self, filters: SearchFilter) -> Dict[str, Any]:
        """Translate the portable clause list into Chroma's dialect."""
        conditions = [
            {clause["field"]: {self._OPS[clause["op"]]: clause["value"]}}
            for clause in filters.to_clauses()
        ]
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _to_similarity(self, distance: Optional[float]) -> float:
        """Convert Chroma's distance to a higher-is-better score."""
        if distance is None:
            return 0.0
        value = float(distance)
        if self.metric == "cosine":
            # Chroma cosine distance is 1 - cosine similarity, in [0, 2].
            return max(0.0, min(1.0, 1.0 - value))
        if self.metric == "ip":
            return -value
        # l2 and anything else: map [0, inf) onto (0, 1].
        return 1.0 / (1.0 + value)


# ---------------------------------------------------------------------
# Metadata -> Chunk
# ---------------------------------------------------------------------

_KNOWN_KEYS = {
    "chunk_id", "chunk_index", "document_id", "tenant_id", "filename",
    "page_number", "page_end", "section", "document_type", "token_count",
    "char_count", "ocr",
}


def chunk_from_metadata(text: str, metadata: Dict[str, Any]) -> Chunk:
    """Rebuild a :class:`Chunk` from stored metadata.

    Shared by every backend adapter, so a hit looks identical whichever
    store produced it.
    """
    try:
        document_type = DocumentType(str(metadata.get("document_type", "unknown")))
    except ValueError:
        document_type = DocumentType.UNKNOWN

    # An absent or zero page number means the page is genuinely unknown,
    # and it is left that way rather than coerced to 1. Since Stage 7 it
    # matters: a citation is the part a user acts on, and "page 1" for a
    # passage whose pagination was never established is a fabricated
    # reference that looks checkable. Stage 6's formatter renders it as
    # "(unknown)" and the citation layer emits null.
    page_number = int(metadata.get("page_number") or 0)
    return Chunk(
        chunk_id=str(metadata.get("chunk_id", "")),
        chunk_index=int(metadata.get("chunk_index", 0) or 0),
        text=text,
        document_id=str(metadata.get("document_id", "")),
        tenant_id=str(metadata.get("tenant_id", "")),
        filename=str(metadata.get("filename", "")),
        page_number=page_number,
        page_end=int(metadata.get("page_end", page_number) or page_number),
        section=(metadata.get("section") or None),
        document_type=document_type,
        token_count=int(metadata.get("token_count", 0) or 0),
        char_count=int(metadata.get("char_count", 0) or 0),
        ocr=bool(metadata.get("ocr", False)),
        extra={k: v for k, v in metadata.items() if k not in _KNOWN_KEYS},
    )


__all__ = ["ChromaVectorStore", "chunk_from_metadata", "MAX_UPSERT_BATCH"]
