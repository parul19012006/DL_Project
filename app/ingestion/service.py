"""Ingestion orchestration.

The one public entry point for Stage 2:

    validate  ->  parse  ->  clean  ->  assign sections  ->  stats

Cleaning and statistics live here rather than in the parsers so that
every format is treated identically: a PDF and a DOCX go through the
same normalisation, the same running-header logic and the same emptiness
rule. A new format only has to produce pages.

These are plain functions with no FastAPI types in their signatures.
That is deliberate — Stage 3 will call :func:`extract_document` directly
from the chunking pipeline, and a background worker (Celery/RQ) can call
it without an HTTP layer.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.config import Settings, get_settings
from app.ingestion import cleaner as cleaning
from app.ingestion.base import get_parser
from app.ingestion.errors import EmptyDocumentError, IngestionError
from app.ingestion.validation import (
    resolve_document_path,
    validate_file,
    validate_identifier,
)
from app.logging_config import get_logger
from app.chunking.splitters import split_paragraphs
from app.ingestion.sections import normalize_section
from app.models.document import (
    BlockKind,
    DocumentMetadata,
    DocumentType,
    ExtractedDocument,
    ExtractionMethod,
    ExtractionStats,
    Page,
    TextBlock,
)

logger = get_logger(__name__)

#: Chunk-metadata key holding the source document's content digest.
#: Reserved: it is what lets the index recognise a document it already
#: holds, so a caller-supplied value of the same name is dropped.
DOCUMENT_SHA256 = "document_sha256"

#: Metadata keys the service owns. Caller-supplied values are ignored.
RESERVED_METADATA = frozenset({DOCUMENT_SHA256})

#: Digest this many bytes of the file for the content hash. The head of
#: a document is enough to spot an accidental re-upload without reading
#: a 400 MB scan twice.
HASH_SAMPLE_BYTES = 8 * 1024 * 1024


# ---------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------

_TYPE_HINTS = [
    (DocumentType.NDA, ("non-disclosure", "nondisclosure", "_nda", "nda_", " nda")),
    (DocumentType.CONTRACT, ("agreement", "contract", "msa", "sow", "addendum")),
    (DocumentType.JUDGMENT, ("judgment", "judgement", " v. ", " vs ", "tribunal")),
    (DocumentType.POLICY, ("policy", "policies", "guideline")),
    (DocumentType.INVOICE, ("invoice", "bill of ")),
    (DocumentType.LETTER, ("letter", "notice of")),
]


def guess_document_type(filename: str, sample_text: str = "") -> DocumentType:
    """Best-effort classification. Advisory only — never load-bearing."""
    haystack = f" {filename} {sample_text[:800]} ".lower()
    for document_type, needles in _TYPE_HINTS:
        if any(needle in haystack for needle in needles):
            return document_type
    return DocumentType.UNKNOWN


def file_digest(path: Path, limit: int = HASH_SAMPLE_BYTES) -> str:
    digest = hashlib.sha256()
    read = 0
    with path.open("rb") as handle:
        while read < limit:
            block = handle.read(min(1024 * 1024, limit - read))
            if not block:
                break
            read += len(block)
            digest.update(block)
    return digest.hexdigest()


def build_metadata(
    document_id: str,
    tenant_id: str,
    path: Path,
    filename: Optional[str] = None,
    document_type: Optional[str] = None,
    title: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> DocumentMetadata:
    """Assemble validated, provenance-carrying document metadata."""
    declared: DocumentType = DocumentType.UNKNOWN
    if document_type:
        try:
            declared = DocumentType(str(document_type).strip().lower())
        except ValueError:
            declared = DocumentType.UNKNOWN

    # Only scalars survive: metadata is copied onto every chunk and,
    # from Stage 4, into vector-store filters that accept scalars only.
    # Reserved keys are dropped: a caller-supplied "document_sha256"
    # would shadow the one the service computes and break the
    # unchanged-document check that depends on it.
    safe_extra = {
        key: value
        for key, value in (extra or {}).items()
        if isinstance(value, (str, int, float, bool)) and key not in RESERVED_METADATA
    }

    digest = file_digest(path)
    # Carried onto every chunk so the index itself can answer "is this
    # the same document I already have?" without a second store.
    safe_extra[DOCUMENT_SHA256] = digest

    return DocumentMetadata(
        document_id=validate_identifier(document_id, "document_id"),
        tenant_id=validate_identifier(tenant_id, "tenant_id"),
        filename=Path(filename or path.name).name,
        document_type=declared,
        title=title,
        source_path=str(path),
        content_sha256=digest,
        size_bytes=path.stat().st_size,
        extra=safe_extra,
    )


# ---------------------------------------------------------------------
# Cleaning pass
# ---------------------------------------------------------------------


def clean_pages(pages: List[Page], settings: Settings) -> int:
    """Clean every page in place. Returns the running-header count.

    Runs as a document-level pass because running-header detection needs
    to compare pages against each other — it cannot be done page by page.
    """
    if not settings.clean_text:
        return 0

    normalized = [cleaning.normalize_characters(p.text) for p in pages]

    running: set = set()
    if settings.strip_running_headers:
        running = cleaning.detect_running_lines(
            normalized,
            min_ratio=settings.header_footer_min_ratio,
            min_pages=settings.header_footer_min_pages,
        )

    removed = 0
    for page in pages:
        before = page.text
        page.text = cleaning.clean_text(
            page.text,
            running=running,
            drop_page_numbers=settings.strip_page_numbers,
        )
        removed += cleaning.count_removed_lines(before, page.text)

        kept_blocks = []
        for block in page.blocks:
            block.text = cleaning.clean_text(
                block.text,
                running=running,
                drop_page_numbers=settings.strip_page_numbers,
            )
            if block.text.strip():
                kept_blocks.append(block)
        # Re-index so block_index stays contiguous after removals.
        for index, block in enumerate(kept_blocks):
            block.block_index = index
        page.blocks = kept_blocks

    return len(running)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def extract_document(
    document_id: str,
    tenant_id: str,
    file_path: str | Path,
    filename: Optional[str] = None,
    document_type: Optional[str] = None,
    title: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    force_ocr: bool = False,
    settings: Optional[Settings] = None,
    skip_path_check: bool = False,
) -> ExtractedDocument:
    """Ingest one file into an :class:`ExtractedDocument`.

    Raises an :class:`IngestionError` subclass on any failure, so the
    caller can distinguish "damaged file" from "wrong type" from "no
    text". ``skip_path_check`` is for a file this service just wrote to
    its own temp directory (an upload), which is trusted by construction
    and lives outside ``DOCUMENT_ROOT``.
    """
    settings = settings or get_settings()
    started = time.perf_counter()

    path = (
        Path(file_path)
        if skip_path_check
        else resolve_document_path(str(file_path), settings)
    )
    extension = validate_file(path, settings)

    metadata = build_metadata(
        document_id=document_id,
        tenant_id=tenant_id,
        path=path,
        filename=filename,
        document_type=document_type,
        title=title,
        extra=extra,
    )

    parser = get_parser(extension)
    outcome = parser.parse(path, metadata, settings, force_ocr=force_ocr)

    running_headers = clean_pages(outcome.pages, settings)

    # Classify only after extraction, so the first page's text can help.
    if metadata.document_type is DocumentType.UNKNOWN:
        sample = outcome.pages[0].text if outcome.pages else ""
        metadata.document_type = guess_document_type(metadata.filename, sample)

    document = ExtractedDocument(
        metadata=metadata,
        pages=outcome.pages,
        stats=ExtractionStats(
            parser=outcome.parser,
            cleaning_applied=settings.clean_text,
            running_headers_removed=running_headers,
            duration_ms=int((time.perf_counter() - started) * 1000),
        ),
        warnings=list(outcome.warnings),
    )
    document.recompute_stats()

    if document.is_empty:
        raise EmptyDocumentError(
            _empty_reason(document, settings),
            {
                "page_count": document.page_count,
                "parser": outcome.parser,
                "warnings": document.warnings,
            },
        )

    if document.stats.empty_page_count:
        document.warnings.append(
            f"{document.stats.empty_page_count} of {document.page_count} "
            "page(s) contained no extractable text"
        )

    logger.info(
        "Extracted %s (document_id=%s tenant=%s pages=%d blocks=%d chars=%d "
        "ocr_pages=%d parser=%s %dms)",
        metadata.filename,
        metadata.document_id,
        metadata.tenant_id,
        document.stats.page_count,
        document.stats.block_count,
        document.stats.char_count,
        document.stats.ocr_page_count,
        document.stats.parser,
        document.stats.duration_ms,
    )
    return document


def document_from_text(
    document_id: str,
    tenant_id: str,
    text: str,
    filename: Optional[str] = None,
    document_type: Optional[str] = None,
    title: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    settings: Optional[Settings] = None,
) -> ExtractedDocument:
    """Build a document from text the caller already holds.

    The Node backend often has the text in hand — a paste, a webhook
    payload, a file it already parsed — and making it write that to a
    shared volume just so this service can read it back is a round trip
    that buys nothing. This path skips file validation entirely, which
    is *safer* rather than looser: there is no path to traverse, no
    magic bytes to spoof and no file to open.

    **Page numbers are zero, not one.** Pasted text has no pagination,
    and inventing "page 1" would put a page reference into a citation
    that cannot be checked against anything. Zero is carried through to
    the citation as ``null`` and rendered as "(unknown)" in the context
    block.

    This is deliberately *not* offered on the batch endpoint: a batch
    takes references, never contents, so that a 500-document request
    cannot carry 500 documents in its body.
    """
    from app.ingestion.sections import detect_section, is_heading

    settings = settings or get_settings()
    started = time.perf_counter()

    body = (text or "").strip()
    if not body:
        raise EmptyDocumentError(
            "The supplied text is empty",
            {"page_count": 0, "parser": "text"},
        )

    declared: DocumentType = DocumentType.UNKNOWN
    if document_type:
        try:
            declared = DocumentType(str(document_type).strip().lower())
        except ValueError:
            declared = DocumentType.UNKNOWN

    safe_extra = {
        key: value
        for key, value in (extra or {}).items()
        if isinstance(value, (str, int, float, bool)) and key not in RESERVED_METADATA
    }

    encoded = body.encode("utf-8")
    safe_extra[DOCUMENT_SHA256] = hashlib.sha256(encoded).hexdigest()
    metadata = DocumentMetadata(
        document_id=validate_identifier(document_id, "document_id"),
        tenant_id=validate_identifier(tenant_id, "tenant_id"),
        filename=Path(filename or f"{document_id}.txt").name,
        document_type=declared,
        title=title,
        source_path=None,
        content_sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
        extra=safe_extra,
    )

    blocks: List[TextBlock] = []
    section: Optional[str] = None
    for index, paragraph in enumerate(split_paragraphs(body)):
        first_line = paragraph.splitlines()[0] if paragraph else ""
        heading = is_heading(first_line)
        if heading:
            section = detect_section(first_line) or normalize_section(first_line)
        blocks.append(
            TextBlock(
                text=paragraph,
                page_number=0,          # no pagination; see the docstring
                block_index=index,
                kind=BlockKind.HEADING if heading else BlockKind.PARAGRAPH,
                section=section,
            )
        )

    page = Page(
        page_number=0,
        blocks=blocks,
        text="\n\n".join(block.text for block in blocks),
        method=ExtractionMethod.NATIVE,
        synthetic=True,
    )

    if metadata.document_type is DocumentType.UNKNOWN:
        metadata.document_type = guess_document_type(metadata.filename, page.text)

    document = ExtractedDocument(
        metadata=metadata,
        pages=[page],
        stats=ExtractionStats(
            parser="text",
            cleaning_applied=False,
            duration_ms=int((time.perf_counter() - started) * 1000),
        ),
        warnings=[
            "Ingested from supplied text: there is no pagination, so "
            "citations report no page number"
        ],
    )
    document.recompute_stats()

    if document.is_empty:  # pragma: no cover - guarded above
        raise EmptyDocumentError(
            "The supplied text contained no usable content",
            {"page_count": document.page_count, "parser": "text"},
        )

    logger.info(
        "Ingested text (document_id=%s tenant=%s blocks=%d chars=%d)",
        metadata.document_id,
        metadata.tenant_id,
        document.stats.block_count,
        document.stats.char_count,
    )
    return document


def _empty_reason(document: ExtractedDocument, settings: Settings) -> str:
    """Explain *why* nothing came out — the useful half of the error."""
    from app.ingestion import ocr as ocr_module

    if document.page_count == 0:
        return "The document contains no pages"
    if settings.ocr_enabled and not ocr_module.ocr_available(settings):
        return (
            "No text layer was found and OCR is unavailable, so a scanned "
            "document cannot be read. Install Tesseract or upload a "
            "text-based file."
        )
    if not settings.ocr_enabled:
        return (
            "No text layer was found and OCR is disabled (OCR_ENABLED=false). "
            "The document may be a scan."
        )
    return (
        "No extractable text was found. The document may be blank, or a scan "
        "whose image quality defeated OCR."
    )


def extract_many(
    requests: Iterable[Dict[str, Any]],
    settings: Optional[Settings] = None,
) -> List[Dict[str, Any]]:
    """Extract several documents, isolating failures.

    One damaged file in a batch of 500 fails only itself. Each result is
    ``{"document_id", "status", "document"|"error"}`` — Stage 3 will
    reuse this shape for indexing status.
    """
    settings = settings or get_settings()
    results: List[Dict[str, Any]] = []

    for request in requests:
        document_id = str(request.get("document_id", ""))
        try:
            document = extract_document(settings=settings, **request)
            results.append(
                {
                    "document_id": document_id,
                    "status": "success",
                    "document": document,
                }
            )
        except IngestionError as exc:
            logger.warning("Ingestion failed for %s: %s", document_id, exc.message)
            results.append(
                {
                    "document_id": document_id,
                    "status": "failed",
                    "error": exc,
                }
            )
        except Exception as exc:  # a surprise must not kill the batch
            logger.exception("Unexpected ingestion error for %s", document_id)
            results.append(
                {
                    "document_id": document_id,
                    "status": "failed",
                    "error": IngestionError(f"Unexpected error: {exc}"),
                }
            )
    return results


__all__ = [
    "DOCUMENT_SHA256",
    "RESERVED_METADATA",
    "extract_document",
    "document_from_text",
    "extract_many",
    "build_metadata",
    "clean_pages",
    "guess_document_type",
    "file_digest",
]
