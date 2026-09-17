"""The integration API — the five endpoints the MERN backend consumes.

    POST   /ingest                     one document in
    POST   /ingest/batch               ~500 documents in, as a job
    GET    /ingest/batch/{job_id}      that job's progress
    POST   /query                      question out, answer + citations in
    DELETE /documents/{document_id}    remove a document's vector data
    GET    /health                     (in app/api/health.py)

Everything underneath — extraction, chunking, embeddings, hybrid
retrieval, reranking, context construction, generation, citation
validation — is reached through these. The per-stage endpoints
(``/index``, ``/search``, ``/retrieve``, ``/context``, ``/answer``) stay
mounted and are useful for debugging a pipeline stage in isolation, but
they are **not** the integration contract and Express should not build on
them.

Four properties this layer holds to, because they are what makes it
consumable from another codebase:

**One error envelope, always.** Every non-2xx response is
``{"error": {"type", "message", "details"}, "request_id"}``. Express
branches on ``error.type``, a stable string, never on a message.

**A request id on every response.** Set from the caller's
``X-Request-ID`` when present, generated when not, echoed in the response
header *and* in the body, and stamped on every log line this service
emits while handling the request. One id ties an Express log line to the
retrieval that produced a citation.

**Tenant isolation is required, never inferred.** Every endpoint that
touches stored data takes an explicit ``tenant_id``, and
:class:`~app.vectorstore.base.SearchFilter` cannot be constructed without
one. There is no "default tenant" and no way to ask for everything.

**Status codes mean what they say.** 200 for work done, 202 for work
accepted, 400/404/413/422 for the caller's problem, 429 when the service
is saturated, 500/503 for ours. The two judgement calls — a delete that
matched nothing, and a generation failure after a successful retrieval —
are documented on their handlers rather than left to be discovered.
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Path, Query, Response, status

from app.api.deps import get_config, verify_service_token
from app.config import Settings
from app.context.base import ContextSource
from app.exceptions import (
    NotFoundError,
    PayloadTooLargeError,
    TooManyRequestsError,
    ValidationError,
)
from app.generation.service import AnswerResult, get_answer_service
from app.ingestion.validation import validate_identifier
from app.logging_config import get_logger, get_request_id
from app.models.schemas import (
    BatchIngestAccepted,
    BatchIngestItemResult,
    BatchIngestRequest,
    BatchIngestStatus,
    CitationModel,
    CitationValidationModel,
    DeleteDocumentResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    QuerySourceModel,
)
from app.services import indexing
from app.services.jobs import get_job_registry, run_batch

logger = get_logger(__name__)

router = APIRouter(tags=["pipeline"], dependencies=[Depends(verify_service_token)])


# =====================================================================
# POST /ingest
# =====================================================================


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest one document: extract, chunk, embed, index",
    description=(
        "Supply **either** `file_path` (a path under `DOCUMENT_ROOT`) or "
        "`text` (content the backend already holds). Returns when the "
        "document is searchable.\n\n"
        "Re-ingesting the same `document_id` **replaces** its chunks, so "
        "an edited document leaves no orphans behind and a retry after a "
        "timeout is safe.\n\n"
        "Extraction failures are HTTP errors with a stable `error.type`: "
        "`unsupported_file_type` (415), `file_too_large` (413), "
        "`unsafe_path` (400), `document_not_found` (404), "
        "`corrupt_document` / `empty_document` (422)."
    ),
)
async def ingest(
    request: IngestRequest, settings: Settings = Depends(get_config)
) -> IngestResponse:
    started = time.perf_counter()

    if request.text is not None and request.text.strip():
        if len(request.text) > settings.ingest_max_text_chars:
            raise ValidationError(
                "The supplied text is larger than this service accepts inline. "
                "Write it to the shared volume and send 'file_path' instead.",
                {
                    "characters": len(request.text),
                    "limit": settings.ingest_max_text_chars,
                },
            )
        from app.ingestion.service import document_from_text

        document = document_from_text(
            document_id=request.document_id,
            tenant_id=request.tenant_id,
            text=request.text,
            filename=request.filename,
            document_type=request.document_type,
            title=request.title,
            extra=request.metadata,
            settings=settings,
        )
        result = indexing.index_from_document(document, settings=settings)
        result.duration_ms = int((time.perf_counter() - started) * 1000)
    else:
        result = indexing.index_document(
            document_id=request.document_id,
            tenant_id=request.tenant_id,
            file_path=request.file_path,
            filename=request.filename,
            document_type=request.document_type,
            title=request.title,
            extra=request.metadata,
            force_ocr=request.force_ocr,
            settings=settings,
        )

    return IngestResponse(
        status=result.status,
        document_id=result.document_id,
        tenant_id=result.tenant_id,
        filename=result.filename,
        document_type=result.document_type,
        duplicate_of=result.duplicate_of,
        chunks_indexed=result.chunks_indexed,
        chunks_replaced=result.chunks_deleted,
        pages=result.pages,
        ocr_pages=result.ocr_pages,
        embedding_model=result.embedding_model,
        embedding_dimension=result.embedding_dimension,
        warnings=list(result.warnings),
        duration_ms=result.duration_ms,
        request_id=get_request_id(),
    )


# =====================================================================
# POST /ingest/batch
# =====================================================================


@router.post(
    "/ingest/batch",
    response_model=BatchIngestAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest up to ~500 documents in the background",
    description=(
        "Accepts the batch and returns **202 Accepted** with a `job_id` "
        "immediately; poll `poll_url` for progress and per-document "
        "results.\n\n"
        "**Why not synchronous.** 500 documents take minutes to hours to "
        "extract, chunk and embed. Every proxy between Express and this "
        "service would time out first, and a 504 after forty minutes of "
        "real work — with no way to learn what got indexed — is the worst "
        "available outcome.\n\n"
        "**Memory.** Items are *references*, never contents: there is no "
        "field in which 500 documents could arrive. The worker then "
        "processes them strictly one at a time, so peak memory is one "
        "document regardless of batch size.\n\n"
        "A failure on one document never stops the batch; it is recorded "
        "against its own id and the rest continue."
    ),
)
async def ingest_batch(
    request: BatchIngestRequest,
    background: BackgroundTasks,
    settings: Settings = Depends(get_config),
) -> BatchIngestAccepted:
    count = len(request.documents)
    if count > settings.ingest_batch_max_documents:
        # 413 rather than 422: the request is well-formed, there is
        # simply too much of it.
        raise PayloadTooLargeError(
            f"A batch may contain at most "
            f"{settings.ingest_batch_max_documents} documents; {count} were "
            "sent. Split it.",
            {"documents": count, "limit": settings.ingest_batch_max_documents},
        )

    registry = get_job_registry(settings)
    if registry.active_count() >= settings.ingest_max_concurrent_jobs:
        raise TooManyRequestsError(
            f"{settings.ingest_max_concurrent_jobs} batch job(s) are already "
            "running. Embedding is CPU-bound, so a second concurrent batch "
            "finishes neither sooner. Retry when the running job completes.",
            {"max_concurrent_jobs": settings.ingest_max_concurrent_jobs},
        )

    # Duplicate ids inside one batch would race each other to replace the
    # same document. Caught here, where the message can name them.
    seen = set()
    duplicates = []
    for item in request.documents:
        key = (item.tenant_id, item.document_id)
        if key in seen:
            duplicates.append(item.document_id)
        seen.add(key)
    if duplicates:
        raise ValidationError(
            "The batch contains the same document more than once; the "
            "copies would race to replace each other.",
            {"duplicate_document_ids": sorted(set(duplicates))[:20]},
        )

    items = [
        {
            "document_id": item.document_id,
            "tenant_id": item.tenant_id,
            "file_path": item.file_path,
            "filename": item.filename,
            "document_type": item.document_type,
            "title": item.title,
            "extra": item.metadata,
            "force_ocr": item.force_ocr,
        }
        for item in request.documents
    ]

    job = registry.create(total=count, request_id=get_request_id())
    background.add_task(run_batch, job, items, settings)

    logger.info(
        "Accepted batch ingest job %s: %d document(s)", job.job_id, count
    )
    return BatchIngestAccepted(
        job_id=job.job_id,
        status=job.status,
        total=job.total,
        poll_url=f"/ingest/batch/{job.job_id}",
        accepted_at=job.created_at.isoformat(),
        request_id=get_request_id(),
    )


@router.get(
    "/ingest/batch/{job_id}",
    response_model=BatchIngestStatus,
    summary="Progress and results for a batch job",
    description=(
        "Poll this after a 202 from `/ingest/batch`. `status` moves "
        "`queued` → `running` → `completed` | `failed`; `results` carries "
        "one record per document as it finishes.\n\n"
        "Job records are in-memory and per-process: a restart loses job "
        "*status*, never indexed data. Documents are committed as they "
        "complete and indexing is idempotent, so re-submitting a batch "
        "after a restart re-indexes onto the same chunk ids rather than "
        "duplicating."
    ),
)
async def ingest_batch_status(
    job_id: str = Path(..., min_length=8, max_length=64),
    include_results: bool = Query(
        True, description="False returns counters only — lighter for frequent polling"
    ),
    settings: Settings = Depends(get_config),
) -> BatchIngestStatus:
    job = get_job_registry(settings).get(job_id)
    if job is None:
        raise NotFoundError(
            "No such ingest job. Job records are kept in memory and are "
            "lost on restart; documents already indexed are unaffected.",
            {"job_id": job_id},
        )

    data = job.to_dict(include_results=include_results)
    return BatchIngestStatus(
        **{k: v for k, v in data.items() if k != "results"},
        results=[
            BatchIngestItemResult(**r) for r in data.get("results", [])
        ],
    )


# =====================================================================
# POST /query
# =====================================================================


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question; get a grounded answer with citations",
    description=(
        "Runs the whole pipeline — hybrid retrieval, reranking, context "
        "construction, generation, citation validation — scoped to one "
        "tenant.\n\n"
        "Every citation is checked against the passages actually "
        "retrieved; one naming a source that was not retrieved is dropped "
        "and reported. `grounded: false` means the answer could not be "
        "fully checked against the evidence, and is worth surfacing.\n\n"
        "**Branch on `status`.** A generation failure returns 200 with "
        "`answer: null` and — when `include_sources` is set — the "
        "retrieved passages intact: retrieval succeeded, and discarding "
        "it because the model call failed serves the user worse than "
        "showing them the relevant clauses."
    ),
)
async def query(
    request: QueryRequest, settings: Settings = Depends(get_config)
) -> QueryResponse:
    result: AnswerResult = get_answer_service(settings).answer(
        question=request.query,
        tenant_id=request.tenant_id,
        retrieval_options={
            "document_ids": request.document_ids,
            "document_type": request.document_type,
            "section": request.section,
            "page_range": tuple(request.page_range) if request.page_range else None,
            "equals": request.metadata,
        },
        context_options={"max_sources": request.top_k},
        temperature=request.temperature,
    )

    return QueryResponse(
        status=result.status,
        answer=result.answer,
        citations=[CitationModel(**c.to_dict()) for c in result.citations.citations],
        interpretation=result.interpretation,
        insufficient_evidence=result.insufficient_evidence,
        grounded=result.citations.grounded,
        query=request.query,
        tenant_id=request.tenant_id,
        sources=(
            [_source_model(s) for s in (result.context.sources if result.context else [])]
            if request.include_sources
            else []
        ),
        context=(
            result.context.text
            if (request.include_context and result.context)
            else None
        ),
        validation=CitationValidationModel(
            grounded=result.citations.grounded,
            valid_citations=len(result.citations.citations),
            rejected=[
                {"reason": r.reason, "claimed_source": r.claimed_source,
                 "detail": r.detail}
                for r in result.citations.rejected
            ],
            verified_quotes=result.citations.verified_quotes,
            unverified_quotes=list(result.citations.unverified_quotes),
        ),
        provider=result.provider,
        model=result.model,
        is_language_model=result.is_language_model,
        warnings=list(result.warnings),
        duration_ms=result.duration_ms,
        timings_ms=dict(result.timings_ms),
        request_id=get_request_id(),
    )


def _source_model(source: ContextSource) -> QuerySourceModel:
    chunk = source.chunk
    return QuerySourceModel(
        source=source.number,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        filename=chunk.filename,
        page=chunk.page_number if chunk.page_number and chunk.page_number > 0 else None,
        page_range=chunk.page_range,
        section=chunk.section,
        document_type=chunk.document_type.value,
        text=source.text,
        score=round(float(source.score), 6),
        retrieval_rank=source.retrieval_rank,
        truncated=source.truncated,
    )


# =====================================================================
# DELETE /documents/{document_id}
# =====================================================================


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="Remove every vector and chunk belonging to a document",
    description=(
        "Tenant-scoped: `tenant_id` is required and deleting another "
        "tenant's `document_id` removes nothing. Call this whenever the "
        "backend deletes a document, so the index cannot keep citing "
        "something the user has removed.\n\n"
        "**A document with nothing indexed returns 200 with "
        "`deleted: false`, not 404.** Delete is idempotent, and the "
        "no-match case is legitimate — an empty or scanned PDF that "
        "produced no chunks, or a retry of a delete that already "
        "succeeded. Turning either into an error would make Express treat "
        "a correct outcome as a failure."
    ),
)
async def delete_document(
    response: Response,
    document_id: str = Path(..., min_length=1, max_length=128),
    tenant_id: str = Query(
        ..., min_length=1, max_length=128, description="Required: the owning tenant"
    ),
    settings: Settings = Depends(get_config),
) -> DeleteDocumentResponse:
    document_id = validate_identifier(document_id, "document_id")
    tenant_id = validate_identifier(tenant_id, "tenant_id")

    removed = indexing.delete_document(document_id, tenant_id, settings=settings)

    if not removed:
        logger.info(
            "Delete matched nothing for document %s (tenant %s)",
            document_id,
            tenant_id,
        )
    response.headers["X-Chunks-Deleted"] = str(removed)

    return DeleteDocumentResponse(
        deleted=removed > 0,
        document_id=document_id,
        tenant_id=tenant_id,
        chunks_deleted=removed,
        request_id=get_request_id(),
    )


__all__ = ["router"]
