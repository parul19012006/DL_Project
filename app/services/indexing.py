"""Indexing and search orchestration.

    file → extract (2) → chunk (3) → embed (4) → upsert (4)
    query → embed (4) → search (4) → hits

Plain functions, no FastAPI types: Stage 5's retriever and any future
background worker call these directly.

Two behaviours worth stating explicitly, because both are easy to get
wrong and expensive to discover later:

**Re-indexing replaces.** Chunk ids are deterministic, so an unchanged
document upserts onto itself — idempotent, no duplicates. But an *edited*
document produces fewer or different chunks, and the ones that no longer
exist would linger in the index and keep being retrieved. So indexing
deletes the document's existing chunks first (``REPLACE_ON_REINDEX``).

**The embedding model and the index are one unit.** Vectors from model A
are meaningless in an index built with model B. The store records the
model and dimension it was created with and refuses a mismatch, rather
than accepting vectors that would silently poison retrieval.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.chunking.chunker import LegalChunker
from app.config import Settings, get_settings
from app.embeddings.base import EmbeddingModel
from app.embeddings.factory import embed_in_batches, get_embedding_model
from app.ingestion.errors import IndexingError, IngestionError
from app.ingestion.service import (
    DOCUMENT_SHA256,
    extract_document,
    file_digest,
)
from app.ingestion.validation import resolve_document_path
from app.logging_config import get_logger
from app.models.chunk import Chunk, ChunkingStats, summarize
from app.models.document import ExtractedDocument
from app.vectorstore.base import SearchFilter, SearchHit, VectorStore
from app.vectorstore.factory import get_vector_store

logger = get_logger(__name__)


@dataclass
class IndexResult:
    """What indexing one document did."""

    document_id: str
    tenant_id: str
    filename: str
    #: The classified type. Reported because the service *infers* it when
    #: the caller declares none, and a caller cannot filter queries by a
    #: value it was never told about.
    document_type: str = ""
    status: str = "success"
    chunks_indexed: int = 0
    chunks_deleted: int = 0
    replaced: bool = False
    #: Another document of the same tenant with byte-identical content.
    duplicate_of: Optional[str] = None
    pages: int = 0
    ocr_pages: int = 0
    embedding_model: str = ""
    embedding_dimension: int = 0
    truncation_warning: bool = False
    chunking: Optional[ChunkingStats] = None
    warnings: List[str] = field(default_factory=list)
    duration_ms: int = 0
    extract_ms: int = 0
    chunk_ms: int = 0
    embed_ms: int = 0
    store_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "tenant_id": self.tenant_id,
            "filename": self.filename,
            "document_type": self.document_type,
            "status": self.status,
            "chunks_indexed": self.chunks_indexed,
            "chunks_deleted": self.chunks_deleted,
            "replaced": self.replaced,
            "duplicate_of": self.duplicate_of,
            "pages": self.pages,
            "ocr_pages": self.ocr_pages,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "warnings": list(self.warnings),
            "duration_ms": self.duration_ms,
            "timings_ms": {
                "extract": self.extract_ms,
                "chunk": self.chunk_ms,
                "embed": self.embed_ms,
                "store": self.store_ms,
            },
        }


# =====================================================================
# Indexing
# =====================================================================


def index_chunks(
    chunks: Sequence[Chunk],
    settings: Optional[Settings] = None,
    model: Optional[EmbeddingModel] = None,
    store: Optional[VectorStore] = None,
) -> int:
    """Embed and upsert chunks. The narrowest useful unit of work."""
    if not chunks:
        return 0
    settings = settings or get_settings()
    model = model or get_embedding_model(settings)
    store = store or get_vector_store(settings)

    vectors = embed_in_batches(
        model, [c.text for c in chunks], batch_size=settings.index_batch_size
    )
    return store.upsert(chunks, vectors)


def _embed_and_store(
    chunks: Sequence[Chunk],
    model: EmbeddingModel,
    store: VectorStore,
    settings: Settings,
    result: "IndexResult",
) -> int:
    """Embed and upsert in slices, releasing each slice as it goes.

    The earlier version embedded every chunk, held every vector, and
    upserted once. That is fine for a ten-page contract and unbounded for
    a thousand-page one: a list of 384 Python floats costs roughly 3 KB,
    so 20,000 chunks is ~60 MB of vectors alive at the same time as the
    chunks they came from. Slicing caps that at ``INDEX_BATCH_SIZE``
    chunks regardless of document size.
    """
    batch_size = max(1, int(settings.index_batch_size))
    stored = 0

    for start in range(0, len(chunks), batch_size):
        window = list(chunks[start : start + batch_size])

        mark = time.perf_counter()
        vectors = model.embed_documents([c.text for c in window])
        result.embed_ms += int((time.perf_counter() - mark) * 1000)

        mark = time.perf_counter()
        stored += store.upsert(window, vectors)
        result.store_ms += int((time.perf_counter() - mark) * 1000)

        # Drop this slice's vectors before the next one is built.
        del vectors, window

    return stored


def _rollback(store: VectorStore, document_id: str, tenant_id: str) -> None:
    """Leave the index in a clean state after a failed write.

    Best-effort by necessity: if the store is the thing that failed, the
    cleanup can fail too. Logged either way, never raised — the original
    error is the one the caller needs.
    """
    try:
        removed = store.delete_document(document_id, tenant_id)
        if removed:
            logger.warning(
                "Rolled back %d partially written chunk(s) for document %s",
                removed,
                document_id,
            )
    except Exception:  # pragma: no cover - the store is already failing
        logger.error(
            "Could not clean up after a failed write of document %s; the "
            "index may hold a partial copy",
            document_id,
        )


def _find_duplicate(
    store: VectorStore, tenant_id: str, document_id: str, digest: str
) -> Optional[str]:
    """Another document of this tenant with byte-identical content.

    One filtered fetch of a single row. Tenant-scoped by construction, so
    the same contract held by two firms is never reported as a duplicate
    — that is two separate documents that happen to share bytes, and
    conflating them would be a cross-tenant leak of the fact.
    """
    try:
        existing = store.fetch(
            SearchFilter(tenant_id=tenant_id, equals={DOCUMENT_SHA256: digest}),
            limit=2,
        )
    except Exception:  # pragma: no cover - reporting must never fail a write
        return None

    for chunk in existing:
        if chunk.document_id != document_id:
            return chunk.document_id
    return None


def unchanged_document(
    document_id: str,
    tenant_id: str,
    digest: str,
    store: VectorStore,
) -> int:
    """Chunks already indexed for this exact content, or 0.

    Re-ingesting an unchanged document costs a full extract, chunk and
    embed to arrive at byte-identical vectors under identical ids. On a
    500-document corpus that measured 33 seconds of work for no change at
    all. One filtered fetch answers the question instead.
    """
    if not digest:
        return 0
    try:
        existing = store.fetch(
            SearchFilter(
                tenant_id=tenant_id,
                document_ids=[document_id],
                equals={DOCUMENT_SHA256: digest},
            ),
            limit=None,
        )
    except Exception:  # pragma: no cover - fall through to a real index
        return 0
    return len(existing)


def _unchanged_count(
    document_id: str,
    tenant_id: str,
    file_path: str | Path,
    store: VectorStore,
    settings: Settings,
    skip_path_check: bool,
) -> int:
    """How many chunks this exact file already has indexed, or 0.

    Never raises: a path problem here is not this function's business —
    extraction will report it properly a moment later with the right
    error type and status code.
    """
    try:
        path = (
            Path(file_path)
            if skip_path_check
            else resolve_document_path(str(file_path), settings)
        )
        return unchanged_document(
            document_id, tenant_id, file_digest(path), store
        )
    except Exception:
        return 0


def index_document(
    document_id: str,
    tenant_id: str,
    file_path: str | Path,
    filename: Optional[str] = None,
    document_type: Optional[str] = None,
    title: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    force_ocr: bool = False,
    settings: Optional[Settings] = None,
    model: Optional[EmbeddingModel] = None,
    store: Optional[VectorStore] = None,
    skip_path_check: bool = False,
) -> IndexResult:
    """Extract, chunk, embed and index one document.

    Raises :class:`IngestionError` on an extraction failure; the caller
    decides whether that fails a batch or just one item.
    """
    settings = settings or get_settings()
    model = model or get_embedding_model(settings)
    store = store or get_vector_store(settings)
    started = time.perf_counter()

    # -- already indexed, unchanged? ----------------------------------
    # Checked before extraction, because the whole point is to skip the
    # extract/chunk/embed that would otherwise produce byte-identical
    # vectors under identical ids. The digest reads only the head of the
    # file, so the check costs one stat, one small read and one filtered
    # fetch. On a 500-document corpus this turned a 33-second re-run into
    # a 2-second one.
    if settings.skip_unchanged_documents and not force_ocr:
        existing = _unchanged_count(
            document_id, tenant_id, file_path, store, settings, skip_path_check
        )
        if existing:
            logger.info(
                "Document %s is unchanged (%d chunk(s) already indexed); "
                "skipping re-extraction",
                document_id,
                existing,
            )
            return IndexResult(
                document_id=document_id,
                tenant_id=tenant_id,
                filename=str(filename or Path(str(file_path)).name),
                document_type=document_type or "",
                status="unchanged",
                chunks_indexed=existing,
                embedding_model=model.model_id,
                embedding_dimension=model.dimension,
                warnings=[
                    "Content is unchanged since the last ingest, so the "
                    "document was not re-processed. Set "
                    "SKIP_UNCHANGED_DOCUMENTS=false to force re-indexing."
                ],
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

    # -- extract ------------------------------------------------------
    mark = time.perf_counter()
    document = extract_document(
        document_id=document_id,
        tenant_id=tenant_id,
        file_path=file_path,
        filename=filename,
        document_type=document_type,
        title=title,
        extra=extra,
        force_ocr=force_ocr,
        settings=settings,
        skip_path_check=skip_path_check,
    )
    extract_ms = int((time.perf_counter() - mark) * 1000)

    result = index_from_document(
        document, settings=settings, model=model, store=store, force=True
    )
    result.extract_ms = extract_ms
    result.duration_ms = int((time.perf_counter() - started) * 1000)
    return result


def index_from_document(
    document: ExtractedDocument,
    settings: Optional[Settings] = None,
    model: Optional[EmbeddingModel] = None,
    store: Optional[VectorStore] = None,
    force: bool = False,
) -> IndexResult:
    """Chunk, embed and index an already-extracted document.

    ``force`` skips the unchanged-content check. :func:`index_document`
    passes it because it has already run that check against the file
    itself, before extraction, where the saving is larger. The check
    lives here as well so that the inline-``text`` path — which never
    goes through :func:`index_document` — gets it too; without it,
    ``POST /ingest`` with a ``text`` body re-chunked and re-embedded
    identical content on every call and reported it as fresh work.
    """
    settings = settings or get_settings()
    model = model or get_embedding_model(settings)
    store = store or get_vector_store(settings)
    started = time.perf_counter()

    meta = document.metadata

    # -- already indexed, unchanged? ----------------------------------
    if not force and settings.skip_unchanged_documents:
        digest = str(meta.extra.get(DOCUMENT_SHA256, "")) if meta.extra else ""
        existing = unchanged_document(
            meta.document_id, meta.tenant_id, digest, store
        )
        if existing:
            logger.info(
                "Document %s is unchanged (%d chunk(s) already indexed); "
                "skipping re-chunking",
                meta.document_id,
                existing,
            )
            return IndexResult(
                document_id=meta.document_id,
                tenant_id=meta.tenant_id,
                filename=meta.filename,
                document_type=meta.document_type.value,
                status="unchanged",
                chunks_indexed=existing,
                pages=document.stats.page_count,
                ocr_pages=document.stats.ocr_page_count,
                embedding_model=model.model_id,
                embedding_dimension=model.dimension,
                warnings=[
                    "Content is unchanged since the last ingest, so the "
                    "document was not re-processed. Set "
                    "SKIP_UNCHANGED_DOCUMENTS=false to force re-indexing."
                ],
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

    result = IndexResult(
        document_id=meta.document_id,
        tenant_id=meta.tenant_id,
        filename=meta.filename,
        document_type=meta.document_type.value,
        pages=document.stats.page_count,
        ocr_pages=document.stats.ocr_page_count,
        embedding_model=model.model_id,
        embedding_dimension=model.dimension,
        warnings=list(document.warnings),
    )

    # -- chunk --------------------------------------------------------
    mark = time.perf_counter()
    chunker = LegalChunker(settings)
    chunks = chunker.chunk_document(document)
    result.chunk_ms = int((time.perf_counter() - mark) * 1000)
    result.chunking = summarize(
        chunks,
        merged_small_count=chunker._merged_small,
        tokenizer=chunker.counter.name,
        chunk_size=chunker.chunk_size,
        chunk_overlap=chunker.overlap,
    )

    if not chunks:
        result.status = "empty"
        result.warnings.append("The document produced no chunks to index")
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    # A chunk longer than the model's window is silently truncated by
    # the encoder. Surfacing it beats losing the tail of a clause.
    max_seq = model.max_sequence_length
    if max_seq:
        over = [c for c in chunks if c.token_count > max_seq]
        if over:
            result.truncation_warning = True
            result.warnings.append(
                f"{len(over)} chunk(s) exceed the model's {max_seq}-token "
                "window and will be truncated when embedded. Lower "
                "CHUNK_SIZE to below the model's limit."
            )

    # -- duplicate detection ------------------------------------------
    digest = str(meta.extra.get(DOCUMENT_SHA256, "")) if meta.extra else ""
    if digest and settings.detect_duplicate_documents:
        twin = _find_duplicate(store, meta.tenant_id, meta.document_id, digest)
        if twin:
            result.duplicate_of = twin
            result.warnings.append(
                f"Identical content is already indexed for this tenant as "
                f"document '{twin}'. Both copies will be retrievable; the "
                "context builder collapses them at query time."
            )
            logger.info(
                "Document %s duplicates %s (tenant %s)",
                meta.document_id,
                twin,
                meta.tenant_id,
            )

    # -- replace and store --------------------------------------------
    # Delete first so an edited document leaves no orphan chunks that
    # would keep matching queries. Embedding and upserting then stream in
    # batches, so a large document never holds every vector at once.
    #
    # The window between the delete and the last upsert is the one place
    # a document can be left unindexed. It cannot be closed without an
    # atomic swap the store does not offer, so instead it is made
    # *loud*: on any failure the document's chunks are removed, leaving
    # a clean "absent" state rather than half a contract, and the error
    # says the document must be re-ingested.
    try:
        if settings.replace_on_reindex:
            mark = time.perf_counter()
            removed = store.delete_document(meta.document_id, meta.tenant_id)
            result.chunks_deleted = removed
            result.replaced = removed > 0
            result.store_ms += int((time.perf_counter() - mark) * 1000)
            if removed:
                logger.info(
                    "Replacing %d existing chunk(s) for document %s",
                    removed,
                    meta.document_id,
                )

        result.chunks_indexed = _embed_and_store(
            chunks, model, store, settings, result
        )
    except Exception as exc:
        _rollback(store, meta.document_id, meta.tenant_id)
        logger.exception(
            "Indexing failed for %s after the existing chunks were removed; "
            "the document is NOT indexed and must be re-ingested",
            meta.document_id,
        )
        raise IndexingError(
            f"Indexing failed part-way through: {type(exc).__name__}. The "
            "document is not indexed — any previous version was removed and "
            "the partial write has been cleaned up. Re-ingest it.",
            {"document_id": meta.document_id, "tenant_id": meta.tenant_id},
        ) from exc

    result.duration_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "Indexed %s (document=%s tenant=%s chunks=%d dim=%d model=%s %dms)",
        meta.filename,
        meta.document_id,
        meta.tenant_id,
        result.chunks_indexed,
        model.dimension,
        model.model_id,
        result.duration_ms,
    )
    return result


def index_many(
    requests: Sequence[Dict[str, Any]],
    settings: Optional[Settings] = None,
    model: Optional[EmbeddingModel] = None,
    store: Optional[VectorStore] = None,
) -> List[IndexResult]:
    """Index several documents, isolating failures."""
    settings = settings or get_settings()
    model = model or get_embedding_model(settings)
    store = store or get_vector_store(settings)

    results: List[IndexResult] = []
    for request in requests:
        document_id = str(request.get("document_id", ""))
        tenant_id = str(request.get("tenant_id", ""))
        filename = str(
            request.get("filename") or Path(str(request.get("file_path", ""))).name
        )
        try:
            results.append(
                index_document(
                    settings=settings, model=model, store=store, **request
                )
            )
        except IngestionError as exc:
            logger.warning("Indexing failed for %s: %s", document_id, exc.message)
            results.append(
                IndexResult(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    filename=filename,
                    status="failed",
                    warnings=[f"{exc.error_type}: {exc.message}"],
                )
            )
        except Exception as exc:  # a surprise must not kill the batch
            logger.exception("Unexpected indexing error for %s", document_id)
            results.append(
                IndexResult(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    filename=filename,
                    status="failed",
                    warnings=[f"internal_error: {exc}"],
                )
            )
    return results


# =====================================================================
# Deletion
# =====================================================================


def delete_document(
    document_id: str,
    tenant_id: str,
    settings: Optional[Settings] = None,
    store: Optional[VectorStore] = None,
) -> int:
    settings = settings or get_settings()
    store = store or get_vector_store(settings)
    removed = store.delete_document(document_id, tenant_id)
    logger.info(
        "Deleted %d chunk(s) for document %s (tenant %s)",
        removed,
        document_id,
        tenant_id,
    )
    return removed


def delete_tenant(
    tenant_id: str,
    settings: Optional[Settings] = None,
    store: Optional[VectorStore] = None,
) -> int:
    settings = settings or get_settings()
    store = store or get_vector_store(settings)
    removed = store.delete_tenant(tenant_id)
    logger.info("Deleted %d chunk(s) for tenant %s", removed, tenant_id)
    return removed


# =====================================================================
# Search
# =====================================================================


@dataclass
class SearchOutcome:
    hits: List[SearchHit] = field(default_factory=list)
    query: str = ""
    filters: Dict[str, Any] = field(default_factory=dict)
    top_k: int = 0
    candidates_scanned: int = 0
    min_score: float = 0.0
    embedding_model: str = ""
    embedding_dimension: int = 0
    duration_ms: int = 0
    embed_ms: int = 0
    search_ms: int = 0


def search(
    query: str,
    tenant_id: str,
    document_ids: Optional[List[str]] = None,
    document_type: Optional[str] = None,
    section: Optional[str] = None,
    page_range: Optional[tuple] = None,
    equals: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    settings: Optional[Settings] = None,
    model: Optional[EmbeddingModel] = None,
    store: Optional[VectorStore] = None,
) -> SearchOutcome:
    """Similarity search, always tenant-scoped.

    This is Stage 4's retrieval primitive. Stage 5 will wrap it with
    hybrid keyword search and reranking; it will not replace it.
    """
    settings = settings or get_settings()
    model = model or get_embedding_model(settings)
    store = store or get_vector_store(settings)
    started = time.perf_counter()

    k = int(top_k or settings.search_top_k)
    floor = settings.search_min_score if min_score is None else float(min_score)

    # SearchFilter refuses to construct without a tenant, so this is
    # where an unscoped query becomes impossible rather than unlikely.
    filters = SearchFilter(
        tenant_id=tenant_id,
        document_ids=document_ids,
        document_type=document_type,
        section=section,
        page_range=page_range,
        equals=equals or {},
    )

    mark = time.perf_counter()
    query_vector = model.embed_query(query)
    embed_ms = int((time.perf_counter() - mark) * 1000)

    mark = time.perf_counter()
    hits = store.search(query_vector, filters, top_k=k)
    search_ms = int((time.perf_counter() - mark) * 1000)

    scanned = len(hits)
    if floor > 0:
        hits = [h for h in hits if h.score >= floor]
        for rank, hit in enumerate(hits, start=1):
            hit.rank = rank

    # Final guard: nothing leaves this function for the wrong tenant.
    foreign = [h for h in hits if h.chunk.tenant_id != filters.tenant_id]
    if foreign:  # pragma: no cover - the store already filters
        logger.error(
            "Dropping %d hit(s) belonging to another tenant", len(foreign)
        )
        hits = [h for h in hits if h.chunk.tenant_id == filters.tenant_id]

    return SearchOutcome(
        hits=hits,
        query=query,
        filters=filters.describe(),
        top_k=k,
        candidates_scanned=scanned,
        min_score=floor,
        embedding_model=model.model_id,
        embedding_dimension=model.dimension,
        embed_ms=embed_ms,
        search_ms=search_ms,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


__all__ = [
    "IndexResult",
    "SearchOutcome",
    "index_document",
    "index_from_document",
    "index_chunks",
    "index_many",
    "delete_document",
    "delete_tenant",
    "search",
]
