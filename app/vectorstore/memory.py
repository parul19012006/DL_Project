"""In-memory vector store.

**Why this exists, given the instruction to avoid unnecessary
infrastructure.** It is ~150 lines with no dependency, and it earns its
place twice over:

1. **It proves the abstraction.** An interface with exactly one
   implementation is an unverified claim. The requirement is that
   ChromaDB can be swapped for Pinecone "without rewriting the RAG
   pipeline" — the cheapest honest evidence for that is a second
   backend passing the same conformance suite. When the Pinecone
   adapter is written, that suite already exists and it is the
   specification.
2. **It makes the test suite fast and hermetic.** Chroma spins up a
   persistent client and writes to disk per test collection; this
   backend does neither. The conformance tests run against *both*, so
   Chroma-specific behaviour is still covered.

It is not for production: everything lives in a dict, nothing is
persisted, and search is a linear scan. ``/health`` says so when it is
selected.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.logging_config import get_logger
from app.models.chunk import Chunk
from app.vectorstore.base import (
    DimensionMismatchError,
    dedupe_by_chunk_id,
    SearchFilter,
    SearchHit,
    StoreStats,
    VectorStore,
    VectorStoreError,
    VersionTracker,
)
from app.vectorstore.chroma import chunk_from_metadata

logger = get_logger(__name__)


@dataclass
class _Record:
    chunk_id: str
    text: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class InMemoryVectorStore(VectorStore):
    """Dict-backed store with exhaustive cosine search."""

    name = "memory"

    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
        embedding_model_id: str = "",
        collection_name: str = "memory",
    ) -> None:
        self._dimension = int(dimension)
        self.metric = (metric or "cosine").lower()
        self.embedding_model_id = embedding_model_id
        self.collection_name = collection_name
        self._records: Dict[str, _Record] = {}
        self._lock = threading.RLock()
        self._versions = VersionTracker()

    # -- lifecycle ----------------------------------------------------

    def ensure_ready(self) -> None:
        logger.warning(
            "Using the in-memory vector store: nothing is persisted and "
            "every restart loses the index. Set VECTOR_BACKEND=chroma for "
            "anything but tests."
        )

    @property
    def dimension(self) -> int:
        return self._dimension

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
        # Same normalisation as every other backend, applied explicitly
        # rather than relying on dict assignment happening to be
        # last-wins.
        chunks, vectors = dedupe_by_chunk_id(chunks, vectors)

        with self._lock:
            for chunk, vector in zip(chunks, vectors):
                if len(vector) != self._dimension:
                    raise DimensionMismatchError(
                        f"Vector for chunk {chunk.chunk_id} has width "
                        f"{len(vector)}, expected {self._dimension}"
                    )
                # Keyed by chunk_id, so upsert is idempotent by
                # construction — the same semantics Chroma gives.
                self._records[chunk.chunk_id] = _Record(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    vector=[float(v) for v in vector],
                    metadata=chunk.metadata(),
                )
        self._versions.bump_chunks(chunks)
        return len(chunks)

    def delete_document(self, document_id: str, tenant_id: str) -> int:
        self._versions.bump(tenant_id)
        return self._delete(
            lambda m: m.get("tenant_id") == tenant_id
            and m.get("document_id") == document_id
        )

    def delete_tenant(self, tenant_id: str) -> int:
        self._versions.bump(tenant_id)
        return self._delete(lambda m: m.get("tenant_id") == tenant_id)

    def _delete(self, predicate) -> int:
        with self._lock:
            doomed = [
                key
                for key, record in self._records.items()
                if predicate(record.metadata)
            ]
            for key in doomed:
                del self._records[key]
        return len(doomed)

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

        query = [float(v) for v in query_vector]
        with self._lock:
            candidates = [
                record
                for record in self._records.values()
                if self._matches(record.metadata, filters)
            ]

        scored = [
            (self._similarity(query, record.vector), record)
            for record in candidates
        ]
        # Sort by score, then chunk_id: a deterministic order for ties,
        # which keeps tests and reranking reproducible.
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))

        hits: List[SearchHit] = []
        for rank, (score, record) in enumerate(scored[: max(1, int(top_k))], 1):
            hits.append(
                SearchHit(
                    chunk=chunk_from_metadata(record.text, record.metadata),
                    score=float(score),
                    raw_score=float(score),
                    rank=rank,
                )
            )
        return hits

    def fetch(
        self, filters: SearchFilter, limit: Optional[int] = None
    ) -> List[Chunk]:
        with self._lock:
            matching = [
                chunk_from_metadata(r.text, r.metadata)
                for r in self._records.values()
                if self._matches(r.metadata, filters)
            ]
        matching.sort(key=lambda c: (c.document_id, c.chunk_index))
        return matching[:limit] if limit else matching

    def count(self, tenant_id: Optional[str] = None) -> int:
        with self._lock:
            if tenant_id is None:
                return len(self._records)
            return sum(
                1
                for r in self._records.values()
                if r.metadata.get("tenant_id") == tenant_id
            )

    def document_chunk_count(self, document_id: str, tenant_id: str) -> int:
        with self._lock:
            return sum(
                1
                for r in self._records.values()
                if r.metadata.get("tenant_id") == tenant_id
                and r.metadata.get("document_id") == document_id
            )

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
        with self._lock:
            self._records.clear()
        self._versions.reset()

    # -- helpers ------------------------------------------------------

    @staticmethod
    def _matches(metadata: Dict[str, Any], filters: SearchFilter) -> bool:
        """Apply the portable clause list — the same one Chroma translates."""
        for clause in filters.to_clauses():
            value = metadata.get(clause["field"])
            op, expected = clause["op"], clause["value"]
            if op == "eq":
                if value != expected:
                    return False
            elif op == "in":
                if value not in expected:
                    return False
            elif op == "gte":
                if value is None or float(value) < float(expected):
                    return False
            elif op == "lte":
                if value is None or float(value) > float(expected):
                    return False
            else:  # pragma: no cover - guarded by SearchFilter
                raise VectorStoreError(f"Unsupported filter operator: {op}")
        return True

    def _similarity(self, a: List[float], b: List[float]) -> float:
        if self.metric == "ip":
            return sum(x * y for x, y in zip(a, b))
        if self.metric == "l2":
            distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
            return 1.0 / (1.0 + distance)

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        # Clamp to [0, 1] to match the Chroma adapter's contract.
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))


__all__ = ["InMemoryVectorStore"]
