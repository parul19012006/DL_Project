"""Indexing and search endpoints (Stage 4).

This is where documents stop being transient. `POST /index` runs the
whole pipeline — extract, chunk, embed, store — and `POST /search`
returns the passages nearest a query.

`POST /search` is **retrieval only**: it returns chunks with scores, not
an answer. Hybrid retrieval and reranking arrive in Stage 5, grounded
generation in Stage 6. Exposing it now lets the MERN developer integrate
and lets retrieval quality be judged before an LLM can paper over it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import get_config, verify_service_token
from app.config import Settings
from app.embeddings.factory import get_embedding_model
from app.ingestion.errors import IngestionError
from app.ingestion.tempfiles import managed_upload
from app.ingestion.validation import validate_identifier, validate_upload_name
from app.logging_config import get_logger
from app.models.schemas import (
    BatchIndexRequest,
    BatchIndexResponse,
    ChunkingStatsModel,
    DeleteResponse,
    ErrorDetail,
    IndexRequest,
    IndexResponse,
    SearchHitModel,
    SearchRequest,
    SearchResponse,
    StoreStatsResponse,
)
from app.services import indexing
from app.vectorstore.factory import get_vector_store

logger = get_logger(__name__)

router = APIRouter(tags=["index"], dependencies=[Depends(verify_service_token)])


def _to_response(result: indexing.IndexResult) -> IndexResponse:
    return IndexResponse(
        status=result.status,
        document_id=result.document_id,
        tenant_id=result.tenant_id,
        filename=result.filename,
        chunks_indexed=result.chunks_indexed,
        chunks_deleted=result.chunks_deleted,
        replaced=result.replaced,
        pages=result.pages,
        ocr_pages=result.ocr_pages,
        embedding_model=result.embedding_model,
        embedding_dimension=result.embedding_dimension,
        chunking=(
            ChunkingStatsModel(**result.chunking.to_dict())
            if result.chunking
            else None
        ),
        warnings=list(result.warnings),
        duration_ms=result.duration_ms,
        timings_ms=result.to_dict()["timings_ms"],
    )


@router.post(
    "/index",
    response_model=IndexResponse,
    summary="Extract, chunk, embed and index one document",
    description=(
        "Runs the full pipeline and stores the chunks in the vector "
        "database. Re-indexing the same document_id replaces its existing "
        "chunks, so an edited document leaves no orphans behind."
    ),
)
async def index(
    request: IndexRequest, settings: Settings = Depends(get_config)
) -> IndexResponse:
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
    return _to_response(result)


@router.post(
    "/index/batch",
    response_model=BatchIndexResponse,
    summary="Index several documents, isolating failures",
    description=(
        "Each document is processed independently and returns HTTP 200 "
        "with per-document status; one damaged file fails only itself."
    ),
)
async def index_batch(
    request: BatchIndexRequest, settings: Settings = Depends(get_config)
) -> BatchIndexResponse:
    started = time.perf_counter()
    results = indexing.index_many(
        [
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
        ],
        settings=settings,
    )
    responses = [_to_response(r) for r in results]
    succeeded = sum(1 for r in responses if r.status == "success")
    return BatchIndexResponse(
        total=len(responses),
        succeeded=succeeded,
        failed=len(responses) - succeeded,
        chunks_indexed=sum(r.chunks_indexed for r in responses),
        results=responses,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


@router.post(
    "/index/upload",
    response_model=IndexResponse,
    summary="Index a streamed upload",
    description=(
        "For deployments where the Node.js backend and this service do not "
        "share a filesystem. The file is streamed to a private temporary "
        "file, indexed, and deleted."
    ),
)
async def index_upload(
    file: UploadFile = File(..., description="PDF or DOCX"),
    document_id: str = Form(...),
    tenant_id: str = Form(...),
    document_type: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    force_ocr: bool = Form(False),
    settings: Settings = Depends(get_config),
) -> IndexResponse:
    document_id = validate_identifier(document_id, "document_id")
    tenant_id = validate_identifier(tenant_id, "tenant_id")

    original_name = Path(file.filename or "upload").name
    extension = validate_upload_name(original_name, settings)

    with managed_upload(file.file, suffix=extension, settings=settings) as path:
        result = indexing.index_document(
            document_id=document_id,
            tenant_id=tenant_id,
            file_path=path,
            filename=original_name,
            document_type=document_type,
            title=title,
            force_ocr=force_ocr,
            settings=settings,
            skip_path_check=True,
        )
    return _to_response(result)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Similarity search over a tenant's indexed chunks",
    description=(
        "Vector search with metadata filtering. Returns passages and "
        "scores, **not** an answer — reranking arrives in Stage 5 and "
        "grounded generation in Stage 6. Always scoped to one tenant."
    ),
)
async def search(
    request: SearchRequest, settings: Settings = Depends(get_config)
) -> SearchResponse:
    outcome = indexing.search(
        query=request.query,
        tenant_id=request.tenant_id,
        document_ids=request.document_ids,
        document_type=request.document_type,
        section=request.section,
        page_range=tuple(request.page_range) if request.page_range else None,
        equals=request.metadata_equals,
        top_k=request.top_k,
        min_score=request.min_score,
        settings=settings,
    )

    return SearchResponse(
        query=outcome.query,
        tenant_id=request.tenant_id,
        hits=[
            SearchHitModel(
                rank=hit.rank,
                score=round(hit.score, 6),
                chunk_id=hit.chunk.chunk_id,
                chunk_index=hit.chunk.chunk_index,
                document_id=hit.chunk.document_id,
                filename=hit.chunk.filename,
                page_number=hit.chunk.page_number,
                page_end=hit.chunk.page_end or hit.chunk.page_number,
                section=hit.chunk.section,
                document_type=hit.chunk.document_type.value,
                text=hit.chunk.text if request.include_text else "",
                token_count=hit.chunk.token_count,
                ocr=hit.chunk.ocr,
            )
            for hit in outcome.hits
        ],
        filters=outcome.filters,
        top_k=outcome.top_k,
        min_score=outcome.min_score,
        embedding_model=outcome.embedding_model,
        embedding_dimension=outcome.embedding_dimension,
        duration_ms=outcome.duration_ms,
        timings_ms={"embed": outcome.embed_ms, "search": outcome.search_ms},
    )


@router.delete(
    "/documents/{document_id}/index",
    response_model=DeleteResponse,
    summary="Delete a document's indexed chunks",
    description=(
        "Tenant-scoped: deleting another tenant's document_id removes "
        "nothing. Call this whenever the backend deletes a document, so "
        "the index stays in step with MongoDB."
    ),
)
async def delete_document_index(
    document_id: str, tenant_id: str, settings: Settings = Depends(get_config)
) -> DeleteResponse:
    document_id = validate_identifier(document_id, "document_id")
    tenant_id = validate_identifier(tenant_id, "tenant_id")
    removed = indexing.delete_document(document_id, tenant_id, settings=settings)
    return DeleteResponse(
        deleted=removed > 0,
        document_id=document_id,
        tenant_id=tenant_id,
        chunks_deleted=removed,
    )


@router.delete(
    "/tenants/{tenant_id}/index",
    response_model=DeleteResponse,
    summary="Delete everything belonging to a tenant",
    description="For account deletion and data-subject erasure requests.",
)
async def delete_tenant_index(
    tenant_id: str, settings: Settings = Depends(get_config)
) -> DeleteResponse:
    tenant_id = validate_identifier(tenant_id, "tenant_id")
    removed = indexing.delete_tenant(tenant_id, settings=settings)
    return DeleteResponse(
        deleted=removed > 0, tenant_id=tenant_id, chunks_deleted=removed
    )


@router.get(
    "/index/stats",
    response_model=StoreStatsResponse,
    summary="Vector store and embedding model status",
)
async def index_stats(
    tenant_id: Optional[str] = None, settings: Settings = Depends(get_config)
) -> StoreStatsResponse:
    store = get_vector_store(settings)
    model = get_embedding_model(settings)
    stats = store.stats()
    return StoreStatsResponse(
        backend=stats.backend,
        collection=stats.collection,
        metric=stats.metric,
        dimension=stats.dimension,
        total_chunks=stats.total_chunks,
        tenant_chunks=store.count(tenant_id) if tenant_id else None,
        embedding_provider=model.name,
        embedding_model=model.model_id,
        embedding_dimension=model.dimension,
        max_sequence_length=model.max_sequence_length,
        normalized=model.normalized,
    )


__all__ = ["router"]
