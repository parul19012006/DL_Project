"""Document extraction endpoints (Stage 2).

These expose ingestion over HTTP so the MERN developer can integrate and
verify extraction now, before chunking and retrieval exist. They return
extracted text and metadata; **nothing is indexed or stored** — there is
no vector store until Stage 4. The endpoint that ingests-and-indexes
(``POST /ingest``) arrives with the stage that has something to index
into.

Two transports, because deployments differ:

``POST /documents/extract``          the backend shares a filesystem with
                                     this service and sends a path
``POST /documents/extract/upload``   no shared filesystem; the file is
                                     streamed here as multipart

Failures come back through the Stage 1 error envelope automatically:
every ingestion error subclasses ``GenAIServiceError``, so there is no
try/except in these handlers except where a *batch* has to keep going.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import get_config, verify_service_token
from app.chunking.chunker import LegalChunker
from app.config import Settings
from app.ingestion import ocr as ocr_module
from app.ingestion.errors import IngestionError
from app.ingestion.service import extract_document
from app.ingestion.tempfiles import managed_upload
from app.ingestion.validation import validate_identifier, validate_upload_name
from app.logging_config import get_logger
from app.models.chunk import Chunk, summarize
from app.models.document import ExtractedDocument
from app.models.schemas import (
    BatchExtractItem,
    BatchExtractRequest,
    BatchExtractResponse,
    BlockModel,
    DocumentMetadataModel,
    ErrorDetail,
    ExtractionStatsModel,
    ExtractionStatus,
    ExtractRequest,
    ExtractResponse,
    ChunkingStatsModel,
    ChunkModel,
    ChunkRequest,
    ChunkResponse,
    PageModel,
    SupportedFormatsResponse,
)

logger = get_logger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(verify_service_token)],
)


# ---------------------------------------------------------------------
# Projection: internal dataclasses -> HTTP models
# ---------------------------------------------------------------------


def to_response(
    document: ExtractedDocument,
    include_blocks: bool = False,
    include_text: bool = True,
) -> ExtractResponse:
    """Project the internal document onto the wire format.

    The internal representation stays free of HTTP concerns; this is the
    only place the two meet.
    """
    meta = document.metadata
    pages = [
        PageModel(
            page_number=page.page_number,
            text=page.text if include_text else "",
            method=page.method.value,
            char_count=page.char_count,
            synthetic=page.synthetic,
            sections=page.sections,
            blocks=(
                [
                    BlockModel(
                        block_id=block.block_id(),
                        text=block.text,
                        page_number=block.page_number,
                        block_index=block.block_index,
                        kind=block.kind.value,
                        section=block.section,
                        ocr=block.ocr,
                    )
                    for block in page.blocks
                ]
                if include_blocks
                else None
            ),
        )
        for page in document.pages
    ]

    return ExtractResponse(
        status=ExtractionStatus.SUCCESS,
        metadata=DocumentMetadataModel(
            document_id=meta.document_id,
            tenant_id=meta.tenant_id,
            filename=meta.filename,
            document_type=meta.document_type.value,
            title=meta.title,
            content_sha256=meta.content_sha256,
            size_bytes=meta.size_bytes,
            ingested_at=meta.ingested_at.isoformat(),
        ),
        stats=ExtractionStatsModel(**document.stats.to_dict()),
        sections=document.sections,
        pages=pages,
        warnings=list(document.warnings),
    )


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------


@router.post(
    "/extract",
    response_model=ExtractResponse,
    summary="Extract one document by path",
    description=(
        "Extract text, page structure and metadata from a PDF or DOCX that "
        "is readable by this service under DOCUMENT_ROOT. Nothing is "
        "indexed — indexing arrives in a later stage."
    ),
)
async def extract(
    request: ExtractRequest, settings: Settings = Depends(get_config)
) -> ExtractResponse:
    document = extract_document(
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
    return to_response(document, request.include_blocks, request.include_text)


@router.post(
    "/extract/batch",
    response_model=BatchExtractResponse,
    summary="Extract several documents, isolating failures",
    description=(
        "Each document is processed independently: one damaged file fails "
        "only itself and returns HTTP 200 with status 'failed'."
    ),
)
async def extract_batch(
    request: BatchExtractRequest, settings: Settings = Depends(get_config)
) -> BatchExtractResponse:
    started = time.perf_counter()
    results: List[BatchExtractItem] = []

    for item in request.documents:
        filename = item.filename or Path(item.file_path).name
        try:
            document = extract_document(
                document_id=item.document_id,
                tenant_id=item.tenant_id,
                file_path=item.file_path,
                filename=item.filename,
                document_type=item.document_type,
                title=item.title,
                extra=item.metadata,
                force_ocr=item.force_ocr,
                settings=settings,
            )
            results.append(
                BatchExtractItem(
                    document_id=item.document_id,
                    status=ExtractionStatus.SUCCESS,
                    filename=document.metadata.filename,
                    stats=ExtractionStatsModel(**document.stats.to_dict()),
                    warnings=list(document.warnings),
                )
            )
        except IngestionError as exc:
            results.append(
                BatchExtractItem(
                    document_id=item.document_id,
                    status=ExtractionStatus.FAILED,
                    filename=filename,
                    error=ErrorDetail(
                        type=exc.error_type, message=exc.message, details=exc.details
                    ),
                )
            )
        except Exception as exc:  # never let one document kill the batch
            logger.exception("Unexpected error extracting %s", item.document_id)
            results.append(
                BatchExtractItem(
                    document_id=item.document_id,
                    status=ExtractionStatus.FAILED,
                    filename=filename,
                    error=ErrorDetail(
                        type="internal_error",
                        message="An internal error occurred for this document",
                    ),
                )
            )

    succeeded = sum(1 for r in results if r.status is ExtractionStatus.SUCCESS)
    return BatchExtractResponse(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


@router.post(
    "/extract/upload",
    response_model=ExtractResponse,
    summary="Extract a streamed upload",
    description=(
        "For deployments where the Node.js backend and this service do not "
        "share a filesystem. The file is streamed to a private temporary "
        "file, extracted, and deleted — it is never retained."
    ),
)
async def extract_upload(
    file: UploadFile = File(..., description="PDF or DOCX"),
    document_id: str = Form(...),
    tenant_id: str = Form(...),
    document_type: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    force_ocr: bool = Form(False),
    include_blocks: bool = Form(False),
    settings: Settings = Depends(get_config),
) -> ExtractResponse:
    document_id = validate_identifier(document_id, "document_id")
    tenant_id = validate_identifier(tenant_id, "tenant_id")

    original_name = Path(file.filename or "upload").name
    extension = validate_upload_name(original_name, settings)

    # managed_upload streams in bounded blocks, enforces the size limit
    # and deletes the temp file on every exit path.
    with managed_upload(file.file, suffix=extension, settings=settings) as temp_path:
        document = extract_document(
            document_id=document_id,
            tenant_id=tenant_id,
            file_path=temp_path,
            filename=original_name,
            document_type=document_type,
            title=title,
            force_ocr=force_ocr,
            settings=settings,
            # The path is one this service just created in its own temp
            # directory, so the DOCUMENT_ROOT containment check does not
            # apply — but every other validation still runs.
            skip_path_check=True,
        )
        return to_response(document, include_blocks, include_text=True)


def to_chunk_model(chunk: Chunk, include_text: bool = True) -> ChunkModel:
    """Project an internal Chunk onto the wire format."""
    return ChunkModel(
        chunk_id=chunk.chunk_id,
        chunk_index=chunk.chunk_index,
        document_id=chunk.document_id,
        tenant_id=chunk.tenant_id,
        filename=chunk.filename,
        page_number=chunk.page_number,
        page_end=chunk.page_end or chunk.page_number,
        section=chunk.section,
        document_type=chunk.document_type.value,
        text=chunk.text if include_text else "",
        token_count=chunk.token_count,
        char_count=chunk.char_count,
        overlap_tokens=chunk.overlap_tokens,
        ocr=chunk.ocr,
        split_mid_sentence=chunk.split_mid_sentence,
        block_ids=list(chunk.block_ids),
    )


@router.post(
    "/chunk",
    response_model=ChunkResponse,
    summary="Extract and chunk a document",
    description=(
        "Runs extraction (Stage 2) then boundary-aware chunking (Stage 3) "
        "and returns the chunks that Stage 4 will embed. Chunk size and "
        "overlap default to the service configuration and can be overridden "
        "per request. Nothing is indexed — there is no vector store yet."
    ),
)
async def chunk(
    request: ChunkRequest, settings: Settings = Depends(get_config)
) -> ChunkResponse:
    import time

    # Per-request chunking overrides, applied to a copy so the process
    # configuration is never mutated by a request.
    overrides = {
        key: value
        for key, value in {
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
            "min_chunk_tokens": request.min_chunk_tokens,
            "respect_sections": request.respect_sections,
        }.items()
        if value is not None
    }
    effective = (
        settings.model_copy(update=overrides) if overrides else settings
    )

    document = extract_document(
        document_id=request.document_id,
        tenant_id=request.tenant_id,
        file_path=request.file_path,
        filename=request.filename,
        document_type=request.document_type,
        title=request.title,
        extra=request.metadata,
        force_ocr=request.force_ocr,
        settings=effective,
    )

    chunker = LegalChunker(effective)
    started = time.perf_counter()
    chunks = chunker.chunk_document(document)
    stats = summarize(
        chunks,
        merged_small_count=chunker._merged_small,
        tokenizer=chunker.counter.name,
        chunk_size=chunker.chunk_size,
        chunk_overlap=chunker.overlap,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )

    extract_payload = to_response(document, include_blocks=False, include_text=False)
    return ChunkResponse(
        metadata=extract_payload.metadata,
        extraction=extract_payload.stats,
        chunking=ChunkingStatsModel(**stats.to_dict()),
        sections=document.sections,
        chunks=[
            to_chunk_model(c, request.include_chunk_text) for c in chunks
        ],
        warnings=list(document.warnings),
    )


@router.get(
    "/formats",
    response_model=SupportedFormatsResponse,
    summary="Supported formats and limits",
    description="What the backend should accept before calling this service.",
)
async def formats(
    settings: Settings = Depends(get_config),
) -> SupportedFormatsResponse:
    available = ocr_module.ocr_available(settings)
    return SupportedFormatsResponse(
        extensions=settings.allowed_extension_list,
        max_file_size_mb=settings.max_file_size_mb,
        ocr_enabled=settings.ocr_enabled,
        ocr_available=available,
        ocr_language=settings.ocr_language if available else None,
    )


__all__ = ["router", "to_response"]
