"""Vector store interface.

This is the seam that lets ChromaDB be replaced by Pinecone, Qdrant,
pgvector or anything else without the RAG pipeline noticing. The
pipeline depends on :class:`VectorStore` and on the two dataclasses
below — never on a backend client, a backend's filter syntax, or a
backend's result shape.

Design decisions and the reasons for them:

**Tenant scoping is a required argument, not an optional filter.**
:class:`SearchFilter` refuses to construct without a ``tenant_id``, so
no code path can express an unscoped query. In a legal product the worst
possible bug is one tenant retrieving another's contracts; making it
unrepresentable beats remembering to add a filter.

**Filters are a small, portable vocabulary**, not a passthrough dict.
Chroma wants ``{"$and": [{"k": {"$eq": v}}]}``; Pinecone wants
``{"k": {"$eq": v}}``; pgvector wants SQL. Each backend translates
:class:`SearchFilter` into its own dialect. Exposing a raw filter dict
would leak Chroma's syntax into the pipeline and defeat the abstraction.

**Similarity is normalised to "higher is better", in [0, 1] where the
backend permits.** Chroma returns cosine *distance*, Pinecone returns
cosine *similarity*, FAISS returns L2. Ranking code should not have to
know which.

**Upsert is the only write.** Chunk ids are deterministic (Stage 3), so
re-indexing an unchanged document is a no-op rather than a duplicate.
There is no separate "add" that could double-insert.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.models.chunk import Chunk


# =====================================================================
# Errors
# =====================================================================


class VectorStoreError(RuntimeError):
    """Base class for vector-store failures."""


class TenantScopeError(VectorStoreError):
    """A query was attempted without tenant scoping."""


class DimensionMismatchError(VectorStoreError):
    """Vectors do not match the width the collection was created with."""


class BackendUnavailableError(VectorStoreError):
    """The backend could not be reached or its client is not installed."""


# =====================================================================
# Data structures
# =====================================================================


@dataclass
class SearchFilter:
    """Metadata scoping for a query.

    ``tenant_id`` is mandatory and validated on construction. The other
    fields are optional narrowings.
    """

    tenant_id: str
    document_ids: Optional[List[str]] = None
    document_type: Optional[str] = None
    section: Optional[str] = None
    #: Restrict to a page range, e.g. ``(10, 20)``. Inclusive.
    page_range: Optional[tuple] = None
    #: Additional exact-match metadata constraints (scalars only).
    equals: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tenant = (self.tenant_id or "").strip()
        if not tenant:
            raise TenantScopeError(
                "tenant_id is required: an unscoped vector query could "
                "return another tenant's documents"
            )
        self.tenant_id = tenant

        if self.document_ids:
            self.document_ids = [d for d in self.document_ids if d]
            if not self.document_ids:
                self.document_ids = None

        if self.page_range is not None:
            low, high = self.page_range
            self.page_range = (int(min(low, high)), int(max(low, high)))

        self.equals = {
            key: value
            for key, value in (self.equals or {}).items()
            if isinstance(value, (str, int, float, bool))
        }

    def to_clauses(self) -> List[Dict[str, Any]]:
        """Backend-neutral clause list: ``[{field, op, value}, ...]``.

        Every backend adapter translates from this. ``tenant_id`` is
        always the first clause.
        """
        clauses: List[Dict[str, Any]] = [
            {"field": "tenant_id", "op": "eq", "value": self.tenant_id}
        ]
        if self.document_ids:
            clauses.append(
                {"field": "document_id", "op": "in", "value": list(self.document_ids)}
            )
        if self.document_type:
            clauses.append(
                {"field": "document_type", "op": "eq", "value": self.document_type}
            )
        if self.section:
            clauses.append({"field": "section", "op": "eq", "value": self.section})
        if self.page_range:
            low, high = self.page_range
            clauses.append({"field": "page_number", "op": "gte", "value": low})
            clauses.append({"field": "page_number", "op": "lte", "value": high})
        for key, value in self.equals.items():
            clauses.append({"field": key, "op": "eq", "value": value})
        return clauses

    @property
    def is_tenant_only(self) -> bool:
        """True when nothing narrows this beyond the tenant.

        The common case, and the only one worth caching a derived index
        for: a filtered query already looks at a smaller corpus, and
        caching every filter combination would be unbounded.
        """
        return len(self.to_clauses()) == 1

    def describe(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "document_ids": self.document_ids,
            "document_type": self.document_type,
            "section": self.section,
            "page_range": list(self.page_range) if self.page_range else None,
            "equals": dict(self.equals) or None,
        }


@dataclass
class SearchHit:
    """One retrieved chunk with its similarity score.

    ``score`` is always "higher is better". ``chunk`` is a real
    :class:`Chunk`, reconstructed from stored metadata, so downstream
    code handles the same type whether the chunk came from the chunker
    or from the index.
    """

    chunk: Chunk
    score: float
    #: The backend's raw number, kept for debugging (a distance for
    #: Chroma, a similarity for Pinecone).
    raw_score: Optional[float] = None
    rank: int = 0

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    def to_dict(self, include_text: bool = True) -> Dict[str, Any]:
        data = self.chunk.to_dict(include_text=include_text)
        data["score"] = round(float(self.score), 6)
        data["rank"] = self.rank
        return data


@dataclass
class UpsertResult:
    upserted: int = 0
    replaced_document: bool = False
    deleted_before: int = 0
    duration_ms: int = 0


@dataclass
class StoreStats:
    backend: str = ""
    collection: str = ""
    dimension: int = 0
    total_chunks: int = 0
    metric: str = ""
    embedding_model: str = ""


# =====================================================================
# Interface
# =====================================================================


class VectorStore(ABC):
    """Persistence and similarity search over chunk vectors."""

    #: Short backend name, e.g. "chroma".
    name: str = "vector_store"

    # -- lifecycle ----------------------------------------------------

    @abstractmethod
    def ensure_ready(self) -> None:
        """Open the collection, creating it at the model's dimension.

        Called at startup so a misconfiguration surfaces there rather
        than on the first user query.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector width this store is configured for."""

    # -- writes -------------------------------------------------------

    @abstractmethod
    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> int:
        """Insert or replace chunks by ``chunk_id``. Returns the count."""

    @abstractmethod
    def delete_document(self, document_id: str, tenant_id: str) -> int:
        """Delete every chunk of one document. Tenant-scoped."""

    @abstractmethod
    def delete_tenant(self, tenant_id: str) -> int:
        """Delete everything belonging to a tenant (account deletion)."""

    # -- reads --------------------------------------------------------

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        filters: SearchFilter,
        top_k: int = 10,
    ) -> List[SearchHit]:
        """Nearest neighbours within the filter. Ordered best-first."""

    @abstractmethod
    def fetch(
        self, filters: SearchFilter, limit: Optional[int] = None
    ) -> List[Chunk]:
        """Return chunks matching the filter, without a vector query.

        Needed by keyword retrieval in Stage 5, and by any operation
        that has to enumerate a tenant's corpus.
        """

    @abstractmethod
    def count(self, tenant_id: Optional[str] = None) -> int:
        """Chunk count, overall or for one tenant."""

    # -- change tracking ----------------------------------------------

    def version(self, tenant_id: str) -> int:
        """A counter that changes whenever this tenant's chunks change.

        Exists so a derived structure — the BM25 index in Stage 5 — can
        be cached and invalidated *exactly*, rather than on a timer.
        Every write path in an adapter bumps it, so a cache keyed on
        (tenant, version) cannot serve a stale corpus.

        The counter lives in this process. A second process writing to
        the same store does not bump it, so a cache built on it is
        correct for a single writer and no more — which matches what the
        default ChromaDB local client supports anyway. The base
        implementation returns 0, which disables caching for an adapter
        that has not opted in.
        """
        return 0

    @abstractmethod
    def document_chunk_count(self, document_id: str, tenant_id: str) -> int:
        """How many chunks one document currently has indexed."""

    # -- introspection ------------------------------------------------

    @abstractmethod
    def stats(self) -> StoreStats:
        """Backend, collection, dimension and size — for /health."""

    def reset(self) -> None:  # pragma: no cover - optional
        """Drop everything. Maintenance and tests only."""
        raise NotImplementedError(
            f"{self.name} does not support reset()"
        )


class VersionTracker:
    """Per-tenant write counters, shared by the backend adapters."""

    def __init__(self) -> None:
        self._versions: Dict[str, int] = {}
        self._lock = threading.RLock()

    def bump(self, *tenant_ids: Optional[str]) -> None:
        with self._lock:
            for tenant_id in tenant_ids:
                if tenant_id:
                    self._versions[tenant_id] = self._versions.get(tenant_id, 0) + 1

    def bump_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Bump every tenant represented in a write."""
        self.bump(*{chunk.tenant_id for chunk in chunks})

    def get(self, tenant_id: str) -> int:
        with self._lock:
            return self._versions.get(tenant_id, 0)

    def reset(self) -> None:
        """A full store reset invalidates everything, so bump all of it."""
        with self._lock:
            for tenant_id in list(self._versions):
                self._versions[tenant_id] += 1


def dedupe_by_chunk_id(
    chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
):
    """Collapse repeated chunk ids within one write, keeping the last.

    Backends disagree here: Chroma raises on a duplicate id inside a
    single ``upsert`` call, while a dict-backed store silently takes the
    last one. Normalising in one place means a caller behaves the same
    on every backend — and last-wins matches what a second ``upsert``
    call would do anyway, so the semantics are consistent whether the
    duplicate arrives in one batch or two.

    Deterministic chunk ids make this rare, but a batch that includes two
    versions of a document would otherwise fail the whole write.
    """
    seen: Dict[str, int] = {}
    for position, chunk in enumerate(chunks):
        seen[chunk.chunk_id] = position
    if len(seen) == len(chunks):
        return list(chunks), list(vectors)

    keep = sorted(seen.values())
    return [chunks[i] for i in keep], [vectors[i] for i in keep]


__all__ = [
    "VectorStore",
    "VersionTracker",
    "dedupe_by_chunk_id",
    "SearchFilter",
    "SearchHit",
    "UpsertResult",
    "StoreStats",
    "VectorStoreError",
    "TenantScopeError",
    "DimensionMismatchError",
    "BackendUnavailableError",
]
