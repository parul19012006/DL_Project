"""The Chunk — the unit that will be embedded, retrieved and cited.

A chunk is a passage of a document small enough to embed and large enough
to answer a question on its own. Everything downstream addresses a chunk:
Stage 4 embeds ``Chunk.text`` and stores ``Chunk.metadata()``, Stage 5
retrieves chunks, Stage 6 cites them.

Two properties matter more than anything else here:

**Self-sufficient provenance.** A chunk carries every field needed to
produce a citation — ``document_id``, ``tenant_id``, ``filename``,
``page_number``, ``section``, ``document_type``, ``chunk_id``,
``chunk_index`` — so a retrieval hit never has to be joined back to the
document to be explained to a user. ``tenant_id`` in particular is the
isolation key every later query must filter on.

**Flat, scalar metadata.** :meth:`Chunk.metadata` returns only
``str``/``int``/``float``/``bool`` values, because that is what vector
databases accept in a filterable metadata document. Nested structures
are kept off that path deliberately.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.models.document import DocumentType

#: Namespace for deterministic chunk ids. Fixed forever: changing it
#: would renumber every chunk in every existing index.
CHUNK_NAMESPACE = uuid.UUID("8a1f0b3e-6d54-4a17-9b2c-3f7e1d0c5a92")


@dataclass
class Chunk:
    """One retrievable passage, with full provenance."""

    # -- identity -----------------------------------------------------
    chunk_id: str
    chunk_index: int
    text: str

    # -- ownership and provenance -------------------------------------
    document_id: str
    tenant_id: str
    filename: str
    page_number: int
    document_type: DocumentType = DocumentType.UNKNOWN
    section: Optional[str] = None

    # -- span ---------------------------------------------------------
    #: Last page this chunk touches. Equals ``page_number`` unless the
    #: chunk spans a page break (a clause continuing overleaf).
    page_end: Optional[int] = None
    #: Ids of the source blocks, so a chunk can be traced to the exact
    #: paragraphs it came from.
    block_ids: List[str] = field(default_factory=list)

    # -- measurements -------------------------------------------------
    token_count: int = 0
    char_count: int = 0
    #: Tokens of leading context copied from the previous chunk.
    overlap_tokens: int = 0

    # -- flags --------------------------------------------------------
    #: True when any source block came from an OCR'd page — useful when
    #: judging how much to trust an exact quotation.
    ocr: bool = False
    #: True when this chunk had to be cut inside a sentence because a
    #: single sentence exceeded the budget.
    split_mid_sentence: bool = False

    #: True when the chunk contains nothing but heading text. A heading
    #: alone carries no obligation, so the chunker never emits one as a
    #: standalone chunk; the flag lets the merge pass recognise it.
    heading_only: bool = False

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    #: Scalar extras carried through from the document's metadata.
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- derived ------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.char_count:
            self.char_count = len(self.text)
        if self.page_end is None:
            self.page_end = self.page_number

    @property
    def spans_pages(self) -> bool:
        return (self.page_end or self.page_number) != self.page_number

    @property
    def page_range(self) -> str:
        """Human-readable page reference for a citation."""
        if self.spans_pages:
            return f"{self.page_number}-{self.page_end}"
        return str(self.page_number)

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    # -- serialisation ------------------------------------------------

    def metadata(self) -> Dict[str, Any]:
        """Flat, scalar-only metadata for the vector store (Stage 4).

        Every value here is filterable. ``tenant_id`` and ``document_id``
        are the keys retrieval will scope on.
        """
        data: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "chunk_index": int(self.chunk_index),
            "document_id": self.document_id,
            "tenant_id": self.tenant_id,
            "filename": self.filename,
            "page_number": int(self.page_number),
            "page_end": int(self.page_end or self.page_number),
            "section": self.section or "",
            "document_type": self.document_type.value,
            "token_count": int(self.token_count),
            "char_count": int(self.char_count),
            "ocr": bool(self.ocr),
        }
        for key, value in (self.extra or {}).items():
            if isinstance(value, (str, int, float, bool)) and key not in data:
                data[key] = value
        return data

    def citation(self) -> Dict[str, Any]:
        """The minimum a user needs to find this passage in the original."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page": self.page_range,
            "section": self.section,
            "chunk_id": self.chunk_id,
        }

    def to_dict(self, include_text: bool = True) -> Dict[str, Any]:
        data = self.metadata()
        data.update(
            {
                "overlap_tokens": self.overlap_tokens,
                "split_mid_sentence": self.split_mid_sentence,
                "block_ids": list(self.block_ids),
                "created_at": self.created_at.isoformat(),
            }
        )
        if include_text:
            data["text"] = self.text
        return data


def make_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    """Deterministic chunk id.

    UUIDv5 over ``document_id | index | sha256(text)``. Two consequences
    that matter operationally:

    * re-ingesting an unchanged document produces identical ids, so
      re-indexing is idempotent and existing citations keep resolving;
    * if the text at a given index changes, the id changes, so a stale
      vector cannot silently masquerade as the current passage.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return uuid.uuid5(
        CHUNK_NAMESPACE, f"{document_id}|{chunk_index}|{digest}"
    ).hex


@dataclass
class ChunkingStats:
    """What the chunker did — for observability and tuning."""

    chunk_count: int = 0
    total_tokens: int = 0
    min_tokens: int = 0
    max_tokens: int = 0
    mean_tokens: float = 0.0
    #: Chunks that respected a paragraph or section boundary.
    clean_boundary_count: int = 0
    #: Chunks that had to be cut inside a sentence (long-sentence path).
    mid_sentence_count: int = 0
    merged_small_count: int = 0
    sections_covered: int = 0
    pages_covered: int = 0
    spanning_page_count: int = 0
    tokenizer: str = ""
    chunk_size: int = 0
    chunk_overlap: int = 0
    duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def summarize(chunks: List[Chunk], **kwargs: Any) -> ChunkingStats:
    """Build :class:`ChunkingStats` from a finished chunk list."""
    stats = ChunkingStats(**kwargs)
    stats.chunk_count = len(chunks)
    if not chunks:
        return stats

    counts = [c.token_count for c in chunks]
    stats.total_tokens = sum(counts)
    stats.min_tokens = min(counts)
    stats.max_tokens = max(counts)
    stats.mean_tokens = round(sum(counts) / len(counts), 1)
    stats.mid_sentence_count = sum(1 for c in chunks if c.split_mid_sentence)
    stats.clean_boundary_count = len(chunks) - stats.mid_sentence_count
    stats.sections_covered = len({c.section for c in chunks if c.section})
    stats.pages_covered = len({c.page_number for c in chunks})
    stats.spanning_page_count = sum(1 for c in chunks if c.spans_pages)
    return stats


__all__ = ["Chunk", "ChunkingStats", "make_chunk_id", "summarize", "CHUNK_NAMESPACE"]
