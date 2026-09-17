"""Internal document representation.

This is the contract between ingestion (Stage 2) and every stage that
follows. It is deliberately **not** Pydantic: these objects are built and
walked in hot loops, never parsed from untrusted JSON, and the pipeline
benefits from cheap ``dataclasses`` with real methods. The HTTP-facing
Pydantic models in ``schemas.py`` are projections of these.

The hierarchy:

    ExtractedDocument            one ingested file
    ├── DocumentMetadata         ownership + provenance
    ├── ExtractionStats          what happened during extraction
    └── Page[]                   page-level preservation
        └── TextBlock[]          paragraph / heading / table / list item

Design rules that later stages depend on:

1. **Every TextBlock knows where it came from.** ``document_id``,
   ``tenant_id``, ``filename``, ``page_number`` and ``section`` are
   resolvable from any block via :meth:`TextBlock.provenance`, so a
   citation can always be produced without carrying the document around.
2. **Pages are never merged or reordered.** Page numbers are 1-based and
   contiguous; a blank page stays in the list as an empty page rather
   than being dropped, so page *n* in the output is page *n* in the file.
3. **The original text stays recoverable.** Cleaning never edits in
   place; ``Page.raw_text`` holds the pre-cleaning text.
4. **Block ids are deterministic.** Re-ingesting the same file produces
   the same ids, so Stage 3 chunk ids and Stage 5 citations stay stable
   across re-indexing.

Stage 3 (chunking) will consume ``ExtractedDocument.iter_blocks()`` and
produce ``Chunk`` objects carrying a ``chunk_id`` plus the provenance
dict returned by :meth:`TextBlock.provenance`. Nothing in this module
needs to change for that to happen — which is the point.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


class BlockKind(str, Enum):
    """What a block of text structurally is."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST_ITEM = "list_item"
    CAPTION = "caption"


class ExtractionMethod(str, Enum):
    """How a page's text was obtained."""

    NATIVE = "native"          # embedded text layer
    OCR = "ocr"                # rasterised + Tesseract
    MIXED = "mixed"            # document-level: some pages each way
    NONE = "none"              # nothing extractable


class DocumentType(str, Enum):
    """Coarse legal document class. Advisory, never load-bearing."""

    CONTRACT = "contract"
    NDA = "nda"
    POLICY = "policy"
    JUDGMENT = "judgment"
    LETTER = "letter"
    INVOICE = "invoice"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------


@dataclass
class DocumentMetadata:
    """Ownership and provenance carried by every piece of extracted text.

    ``tenant_id`` is supplied by the MERN backend and trusted. It is the
    isolation key for the whole pipeline: from Stage 4 onward no
    retrieval may cross it.
    """

    document_id: str
    tenant_id: str
    filename: str
    document_type: DocumentType = DocumentType.UNKNOWN
    title: Optional[str] = None
    source_path: Optional[str] = None
    content_sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    ingested_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    #: Scalar extras supplied by the backend (matter_id, case_number, ...).
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "document_id": self.document_id,
            "tenant_id": self.tenant_id,
            "filename": self.filename,
            "document_type": self.document_type.value,
            "title": self.title,
            "content_sha256": self.content_sha256,
            "size_bytes": self.size_bytes,
            "ingested_at": self.ingested_at.isoformat(),
        }
        data.update(self.extra or {})
        return {k: v for k, v in data.items() if v is not None}


# ---------------------------------------------------------------------
# Blocks and pages
# ---------------------------------------------------------------------


@dataclass
class TextBlock:
    """The smallest addressable unit of extracted text."""

    text: str
    page_number: int
    block_index: int
    kind: BlockKind = BlockKind.PARAGRAPH
    section: Optional[str] = None
    ocr: bool = False
    #: Page coordinates (x0, y0, x1, y1) when the parser provides them.
    bbox: Optional[tuple] = None
    #: Back-reference filled in by :meth:`ExtractedDocument.link`.
    _meta: Optional[DocumentMetadata] = field(default=None, repr=False)

    @property
    def is_heading(self) -> bool:
        return self.kind is BlockKind.HEADING

    @property
    def char_count(self) -> int:
        return len(self.text)

    def block_id(self, document_id: Optional[str] = None) -> str:
        """Deterministic id: same file in, same id out.

        Derived from the document id, the page, the block index and a
        digest of the text, so an edited document does not silently
        reuse an id that now points at different words.
        """
        doc = document_id or (self._meta.document_id if self._meta else "")
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:12]
        return f"{doc}:p{self.page_number}:b{self.block_index}:{digest}"

    def provenance(self) -> Dict[str, Any]:
        """Everything needed to cite this block.

        Stage 3 copies this onto each chunk; Stage 6 turns it into a
        citation. ``chunk_id`` is added by Stage 3 — it does not exist
        yet and is deliberately absent rather than present-and-null.
        """
        data: Dict[str, Any] = {
            "page_number": self.page_number,
            "block_index": self.block_index,
            "section": self.section,
            "kind": self.kind.value,
            "ocr": self.ocr,
        }
        if self._meta is not None:
            data.update(
                {
                    "document_id": self._meta.document_id,
                    "tenant_id": self._meta.tenant_id,
                    "filename": self._meta.filename,
                    "document_type": self._meta.document_type.value,
                    "block_id": self.block_id(),
                }
            )
        return data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id(),
            "text": self.text,
            "page_number": self.page_number,
            "block_index": self.block_index,
            "kind": self.kind.value,
            "section": self.section,
            "ocr": self.ocr,
        }


@dataclass
class Page:
    """One page of a PDF, or one synthetic page of a DOCX.

    ``page_number`` is 1-based and matches the file. A page with no
    extractable text is kept as an empty page so numbering never shifts.
    """

    page_number: int
    blocks: List[TextBlock] = field(default_factory=list)
    #: Cleaned page text (blocks joined). The canonical text of the page.
    text: str = ""
    #: Pre-cleaning text, kept when ``KEEP_RAW_TEXT`` is on.
    raw_text: Optional[str] = None
    method: ExtractionMethod = ExtractionMethod.NATIVE
    #: Set for a DOCX page, whose boundaries are synthesised, not real.
    synthetic: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def ocr_used(self) -> bool:
        return self.method is ExtractionMethod.OCR

    @property
    def sections(self) -> List[str]:
        seen: List[str] = []
        for block in self.blocks:
            if block.section and block.section not in seen:
                seen.append(block.section)
        return seen

    def to_dict(self, include_blocks: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "page_number": self.page_number,
            "text": self.text,
            "method": self.method.value,
            "char_count": self.char_count,
            "synthetic": self.synthetic,
            "sections": self.sections,
        }
        if include_blocks:
            data["blocks"] = [b.to_dict() for b in self.blocks]
        return data


# ---------------------------------------------------------------------
# Stats and warnings
# ---------------------------------------------------------------------


@dataclass
class ExtractionStats:
    """What actually happened, for observability and for /health."""

    parser: str = ""
    page_count: int = 0
    empty_page_count: int = 0
    ocr_page_count: int = 0
    block_count: int = 0
    char_count: int = 0
    duration_ms: int = 0
    cleaning_applied: bool = False
    running_headers_removed: int = 0

    @property
    def method(self) -> ExtractionMethod:
        if self.page_count == 0 or self.char_count == 0:
            return ExtractionMethod.NONE
        if self.ocr_page_count == 0:
            return ExtractionMethod.NATIVE
        if self.ocr_page_count == self.page_count:
            return ExtractionMethod.OCR
        return ExtractionMethod.MIXED

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["method"] = self.method.value
        return data


# ---------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------


@dataclass
class ExtractedDocument:
    """The output of ingestion; the input to chunking."""

    metadata: DocumentMetadata
    pages: List[Page] = field(default_factory=list)
    stats: ExtractionStats = field(default_factory=ExtractionStats)
    #: Non-fatal problems worth surfacing (OCR skipped, blank pages, ...).
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.link()

    # -- wiring -------------------------------------------------------

    def link(self) -> "ExtractedDocument":
        """Give every block a back-reference to the document metadata."""
        for page in self.pages:
            for block in page.blocks:
                block._meta = self.metadata
        return self

    # -- access -------------------------------------------------------

    def iter_blocks(self) -> Iterator[TextBlock]:
        """Every block in document order — what Stage 3 will consume."""
        for page in self.pages:
            for block in page.blocks:
                yield block

    def get_page(self, page_number: int) -> Optional[Page]:
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def full_text(self, separator: str = "\n\n") -> str:
        return separator.join(p.text for p in self.pages if p.text.strip())

    @property
    def sections(self) -> List[str]:
        seen: List[str] = []
        for block in self.iter_blocks():
            if block.section and block.section not in seen:
                seen.append(block.section)
        return seen

    @property
    def is_empty(self) -> bool:
        return not any(not p.is_empty for p in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    # -- serialisation ------------------------------------------------

    def to_dict(self, include_blocks: bool = True) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "stats": self.stats.to_dict(),
            "pages": [p.to_dict(include_blocks) for p in self.pages],
            "sections": self.sections,
            "warnings": list(self.warnings),
        }

    def recompute_stats(self) -> ExtractionStats:
        """Refresh the counters after pages or blocks change."""
        self.stats.page_count = len(self.pages)
        self.stats.empty_page_count = sum(1 for p in self.pages if p.is_empty)
        self.stats.ocr_page_count = sum(1 for p in self.pages if p.ocr_used)
        self.stats.block_count = sum(len(p.blocks) for p in self.pages)
        self.stats.char_count = sum(p.char_count for p in self.pages)
        return self.stats


__all__ = [
    "BlockKind",
    "ExtractionMethod",
    "DocumentType",
    "DocumentMetadata",
    "TextBlock",
    "Page",
    "ExtractionStats",
    "ExtractedDocument",
]
