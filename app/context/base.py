"""Shared types for context construction.

Stage 5 returns ranked passages. Stage 6 turns them into the block of
text that will sit inside an LLM prompt — and, just as importantly, into
a **structured record of what that block contains and what it leaves
out**.

The second half is the part that is easy to skip and expensive to skip.
A context builder that returns only a string has silently thrown away
every fact the caller needs afterwards: which documents were cited,
which passage was truncated and by how much, what was dropped for
budget, and whether the evidence was thin enough that the answer should
be hedged. In a legal product those are not diagnostics — they are the
difference between a citation the user can verify and a paragraph of
text with no provenance.

So :class:`BuiltContext` carries three things in parallel:

* ``text``      — the prompt block, exactly as the LLM will see it
* ``sources``   — one :class:`ContextSource` per rendered block, with the
                  full :class:`~app.models.chunk.Chunk` behind it
* ``omitted``   — every passage that did **not** make it, with its
                  citation and the reason

Nothing is dropped without an entry in the third list. That is the whole
of requirement 9, expressed as a data structure rather than a promise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.models.chunk import Chunk

# =====================================================================
# Placeholders for absent metadata
# =====================================================================

#: Rendered when a metadata field is genuinely unknown.
#:
#: An explicit marker rather than an empty value, because "Section:"
#: followed by nothing reads to a language model as though the section
#: were blank or unimportant, and an omitted line changes the shape of
#: the block from source to source. Saying "not detected" is honest and
#: keeps every source structurally identical.
UNKNOWN = "(not detected)"
UNKNOWN_PAGE = "(unknown)"


class ContextError(RuntimeError):
    """Context could not be constructed."""


# =====================================================================
# Why a passage did not make it
# =====================================================================


class OmissionReason:
    """Stable reason codes. The caller branches on these, not on prose."""

    BELOW_SCORE_FLOOR = "below_score_floor"
    REDUNDANT = "redundant"
    MAX_SOURCES = "max_sources"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TOO_SMALL_TO_TRUNCATE = "too_small_to_truncate"
    EMPTY_TEXT = "empty_text"

    ALL = (
        BELOW_SCORE_FLOOR,
        REDUNDANT,
        MAX_SOURCES,
        BUDGET_EXHAUSTED,
        TOO_SMALL_TO_TRUNCATE,
        EMPTY_TEXT,
    )


@dataclass
class OmittedSource:
    """A passage that was retrieved but did not reach the prompt.

    It keeps its citation. A user asking "did you look at clause 9?"
    deserves "yes, but it did not fit the budget" rather than silence.
    """

    chunk_id: str
    reason: str
    score: float = 0.0
    citation: Dict[str, Any] = field(default_factory=dict)
    #: For ``REDUNDANT``: the chunk this one duplicated.
    duplicate_of: Optional[str] = None
    #: For ``BUDGET_EXHAUSTED``: what it would have cost.
    tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "reason": self.reason,
            "score": round(float(self.score), 6),
            "citation": dict(self.citation),
            "duplicate_of": self.duplicate_of,
            "tokens": self.tokens,
        }


# =====================================================================
# A rendered source
# =====================================================================


@dataclass
class ContextSource:
    """One numbered block in the context, plus everything behind it."""

    chunk: Chunk
    #: 1-based position **as rendered**. Assigned after ordering, so
    #: "Source 3" always means the third block in the text the model
    #: sees — which is what a citation of "Source 3" has to resolve to.
    number: int = 0
    #: Retrieval's final blended score, carried through unchanged.
    score: float = 0.0
    #: Where this passage sat in the retrieval ranking. Kept because
    #: presentation order deliberately is not relevance order.
    retrieval_rank: int = 0

    #: The text actually rendered — possibly shortened.
    text: str = ""
    truncated: bool = False
    original_tokens: int = 0
    text_tokens: int = 0
    #: Tokens the whole block costs, header included.
    total_tokens: int = 0
    #: True when the section heading was removed from the body because
    #: the header already states it.
    heading_stripped: bool = False

    #: chunk_ids of near-duplicates folded into this source, and their
    #: citations — so a passage that appears in two documents cites both.
    duplicates: List[str] = field(default_factory=list)
    duplicate_citations: List[Dict[str, Any]] = field(default_factory=list)

    #: True when the previous rendered source is the immediately
    #: preceding chunk of the same document, so the two read as one
    #: continuous passage.
    continues_source: Optional[int] = None

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def omitted_tokens(self) -> int:
        return max(0, self.original_tokens - self.text_tokens)

    def citation(self) -> Dict[str, Any]:
        """What a user needs to find this passage in the original.

        ``source`` is added so a model citing "Source 2" can be resolved
        mechanically rather than by matching text.
        """
        data = self.chunk.citation()
        data["source"] = self.number
        data["page_number"] = self.chunk.page_number
        data["page_end"] = self.chunk.page_end or self.chunk.page_number
        data["document_type"] = self.chunk.document_type.value
        data["chunk_index"] = self.chunk.chunk_index
        data["truncated"] = self.truncated
        if self.duplicate_citations:
            data["also_appears_in"] = list(self.duplicate_citations)
        return data

    def to_dict(self, include_text: bool = True) -> Dict[str, Any]:
        data = {
            "source": self.number,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk.chunk_index,
            "document_id": self.chunk.document_id,
            "filename": self.chunk.filename,
            "page_number": self.chunk.page_number,
            "page_end": self.chunk.page_end or self.chunk.page_number,
            "page_range": self.chunk.page_range,
            "section": self.chunk.section,
            "document_type": self.chunk.document_type.value,
            "score": round(float(self.score), 6),
            "retrieval_rank": self.retrieval_rank,
            "tokens": self.text_tokens,
            "total_tokens": self.total_tokens,
            "truncated": self.truncated,
            "omitted_tokens": self.omitted_tokens,
            "ocr": self.chunk.ocr,
            "duplicates": list(self.duplicates),
            "continues_source": self.continues_source,
        }
        if include_text:
            data["text"] = self.text
        return data


# =====================================================================
# Statistics
# =====================================================================


@dataclass
class ContextStats:
    """What the builder did. Every number here answers a real question."""

    candidates_in: int = 0
    sources_out: int = 0
    documents: int = 0
    redundant_removed: int = 0
    below_floor_removed: int = 0
    over_max_sources: int = 0
    dropped_for_budget: int = 0
    truncated_sources: int = 0
    tokens_used: int = 0
    tokens_budget: int = 0
    text_tokens: int = 0
    overhead_tokens: int = 0
    omitted_text_tokens: int = 0
    build_ms: int = 0

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.tokens_budget - self.tokens_used)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        data = asdict(self)
        data["tokens_remaining"] = self.tokens_remaining
        return data


# =====================================================================
# The result
# =====================================================================


@dataclass
class BuiltContext:
    """The prompt block, its provenance, and what it leaves out."""

    text: str = ""
    sources: List[ContextSource] = field(default_factory=list)
    omitted: List[OmittedSource] = field(default_factory=list)
    query: str = ""
    stats: ContextStats = field(default_factory=ContextStats)
    tokenizer: str = ""
    order: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.sources

    @property
    def sufficient(self) -> bool:
        """Whether enough evidence reached the prompt to answer from.

        Deliberately a *report*, not an enforcement. This stage does not
        generate an answer and must not decide unilaterally that a
        question is unanswerable; it states what it found and lets the
        generation stage — and its prompt — decide what to do about thin
        evidence.
        """
        return bool(self.sources) and not any(
            w.startswith("Insufficient evidence") for w in self.warnings
        )

    @property
    def document_ids(self) -> List[str]:
        seen: List[str] = []
        for source in self.sources:
            if source.chunk.document_id not in seen:
                seen.append(source.chunk.document_id)
        return seen

    def citations(self) -> List[Dict[str, Any]]:
        """Citations in rendered order — the "Sources" list, ready to use.

        The caller never has to parse ``text`` to find out what was
        cited, which is the point: a regex over a prompt block is a bad
        foundation for a citation feature.
        """
        return [source.citation() for source in self.sources]

    def source_by_number(self, number: int) -> Optional[ContextSource]:
        for source in self.sources:
            if source.number == number:
                return source
        return None

    def to_dict(self, include_text: bool = True) -> Dict[str, Any]:
        return {
            "text": self.text if include_text else "",
            "sources": [s.to_dict(include_text=include_text) for s in self.sources],
            "omitted": [o.to_dict() for o in self.omitted],
            "citations": self.citations(),
            "query": self.query,
            "stats": self.stats.to_dict(),
            "tokenizer": self.tokenizer,
            "order": self.order,
            "warnings": list(self.warnings),
            "sufficient": self.sufficient,
        }


def to_candidates(items: Sequence[Any]) -> List[Any]:
    """Accept retrieval candidates *or* bare chunks.

    The builder is required to be independently testable, and a test
    that has to construct a full retrieval outcome to check a token
    budget is not an independent test. Passing a list of
    :class:`~app.models.chunk.Chunk` objects works exactly as well, they
    simply carry no scores.
    """
    from app.retrieval.base import Candidate

    out: List[Candidate] = []
    for item in items:
        if isinstance(item, Candidate):
            out.append(item)
        elif isinstance(item, Chunk):
            out.append(Candidate.from_chunk(item))
        else:
            raise ContextError(
                f"Expected Candidate or Chunk, got {type(item).__name__}"
            )
    return out


__all__ = [
    "BuiltContext",
    "ContextSource",
    "ContextStats",
    "OmittedSource",
    "OmissionReason",
    "ContextError",
    "UNKNOWN",
    "UNKNOWN_PAGE",
    "to_candidates",
]
