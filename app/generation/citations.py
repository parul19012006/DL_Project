"""Citation validation — checking the answer against what was retrieved.

A model asked to cite its sources will sometimes cite one that does not
exist, cite one it did not use, or restate a page number wrongly. In a
legal product a citation is the part the user acts on: it is how they
verify the answer, and a confident reference to "clause 7.3, page 12" of
a document that has no clause 7.3 is worse than no answer, because it
looks checkable.

So nothing the model says about a source is trusted. Four rules:

**A citation must name a source that was actually in the context.** The
context numbered its blocks 1..N; anything outside that set is dropped
and reported as ``unknown_source``. This is the requirement that
citations correspond to genuinely retrieved sources, and it is enforced
by set membership rather than by inspection.

**Metadata comes from the record, not from the reply.** Document,
filename, page, section and chunk_id are copied from the
:class:`~app.context.base.ContextSource` the number resolves to, and any
values the model supplied are discarded — kept in ``claimed`` only when
they disagree, so a model that is drifting is visible rather than
silently corrected.

**Inline references are harvested, not ignored.** A model that writes
"see Source 3" in its prose but leaves ``citations`` empty has told us
which passage it used; adding that citation is recovery, not invention.
An inline reference to a source number that does **not** exist is the
opposite — that is a fabricated reference inside the answer text, and it
is reported loudly.

**Quoted spans are checked against the evidence.** If the answer puts
words in quotation marks, those words should appear in a retrieved
passage. Spans that do not are listed in ``unverified_quotes``.

None of this makes a wrong answer impossible — a model can state
something untrue and cite a real passage that does not support it, and no
check here would catch that. What it does is make every *citation*
verifiable and every unverifiable one visible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.context.base import BuiltContext, ContextSource
from app.generation.parser import ParsedAnswer
from app.logging_config import get_logger
from app.retrieval.dedup import normalize_for_comparison

logger = get_logger(__name__)

#: "Source 3", "source #3", "[Source 3]" inside the answer prose.
_INLINE_RE = re.compile(r"\bsources?\s*#?\s*(\d{1,3})\b", re.IGNORECASE)

#: A quoted span worth verifying. Short quotes ("the Vendor") are common
#: as emphasis rather than quotation and produce noise, so a minimum
#: length applies.
_QUOTE_RE = re.compile(r"[\"“]([^\"”]{12,400})[\"”]")
MIN_QUOTE_WORDS = 4


class CitationProblem:
    """Stable reason codes for a rejected or suspect citation."""

    UNKNOWN_SOURCE = "unknown_source"
    UNKNOWN_SOURCE_IN_ANSWER = "unknown_source_in_answer"
    NO_CITATIONS = "no_citations"
    UNVERIFIED_QUOTE = "unverified_quote"
    METADATA_MISMATCH = "metadata_mismatch"


@dataclass
class Citation:
    """A validated citation. Every field comes from the retrieved record."""

    source: int
    document: str
    document_id: str
    page: Optional[int]
    page_range: str
    section: Optional[str]
    chunk_id: str
    #: True when this was recovered from the answer prose rather than
    #: supplied in the citations list.
    inline: bool = False
    #: What the model claimed, where it disagreed with the record.
    claimed: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "document": self.document,
            "page": self.page,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "document_id": self.document_id,
            "page_range": self.page_range,
        }
        if self.claimed:
            data["claimed"] = dict(self.claimed)
        return data


@dataclass
class RejectedCitation:
    """A citation that did not survive validation."""

    reason: str
    claimed_source: Optional[int] = None
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "claimed_source": self.claimed_source,
            "detail": self.detail,
        }


@dataclass
class CitationReport:
    """The outcome of validating one answer's citations."""

    citations: List[Citation] = field(default_factory=list)
    rejected: List[RejectedCitation] = field(default_factory=list)
    unverified_quotes: List[str] = field(default_factory=list)
    verified_quotes: int = 0
    #: False when the answer asserts something with no valid citation, or
    #: when it referenced a source that does not exist.
    grounded: bool = True
    warnings: List[str] = field(default_factory=list)

    @property
    def cited_sources(self) -> List[int]:
        return [c.source for c in self.citations]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citations": [c.to_dict() for c in self.citations],
            "rejected": [r.to_dict() for r in self.rejected],
            "unverified_quotes": list(self.unverified_quotes),
            "verified_quotes": self.verified_quotes,
            "grounded": self.grounded,
            "warnings": list(self.warnings),
        }


def _citation_from_source(
    source: ContextSource, claimed: Optional[Dict[str, Any]] = None, inline: bool = False
) -> Citation:
    chunk = source.chunk
    citation = Citation(
        source=source.number,
        document=chunk.filename or "",
        document_id=chunk.document_id or "",
        page=chunk.page_number if chunk.page_number and chunk.page_number > 0 else None,
        page_range=chunk.page_range,
        section=chunk.section,
        chunk_id=chunk.chunk_id,
        inline=inline,
    )

    # Record a disagreement rather than quietly overwriting it: a model
    # consistently naming the wrong page is a signal about the prompt or
    # the model, and it is invisible if the correction is silent.
    if claimed:
        mismatches: Dict[str, Any] = {}
        for key, actual in (
            ("document", citation.document),
            ("section", citation.section),
            ("chunk_id", citation.chunk_id),
        ):
            value = claimed.get(key)
            if isinstance(value, str) and value.strip() and value.strip() != (actual or ""):
                mismatches[key] = value.strip()
        claimed_page = claimed.get("page")
        if isinstance(claimed_page, int) and claimed_page != citation.page:
            mismatches["page"] = claimed_page
        if mismatches:
            citation.claimed = mismatches

    return citation


def _quotes(text: str) -> List[str]:
    out: List[str] = []
    for match in _QUOTE_RE.finditer(text or ""):
        span = match.group(1).strip()
        if len(span.split()) >= MIN_QUOTE_WORDS:
            out.append(span)
    return out


def verify_quotes(
    answer: str, sources: Sequence[ContextSource]
) -> "tuple[int, List[str]]":
    """``(verified count, spans not found in any source)``.

    Compared on normalised text — lowercased, punctuation flattened — so
    a quote differing only in a curly apostrophe or a line break still
    matches. A span the model shortened with an ellipsis will not match
    and is reported; that is the correct outcome, since a quotation mark
    is a claim that the words are exact.
    """
    spans = _quotes(answer)
    if not spans:
        return 0, []

    haystack = " ".join(normalize_for_comparison(s.text) for s in sources)
    verified = 0
    unverified: List[str] = []
    for span in spans:
        if normalize_for_comparison(span) and normalize_for_comparison(span) in haystack:
            verified += 1
        else:
            unverified.append(span)
    return verified, unverified


def validate_citations(
    parsed: ParsedAnswer,
    context: BuiltContext,
    require_citations: bool = True,
    harvest_inline: bool = True,
    check_quotes: bool = True,
) -> CitationReport:
    """Check an answer's citations against the sources actually retrieved."""
    report = CitationReport()
    by_number: Dict[int, ContextSource] = {s.number: s for s in context.sources}
    claimed_by_number: Dict[int, Dict[str, Any]] = {}

    for entry in parsed.raw_citations:
        if not isinstance(entry, dict):
            continue
        number = entry.get("source")
        if isinstance(number, int) and number not in claimed_by_number:
            claimed_by_number[number] = entry

    seen: set = set()

    # -- citations the model listed ------------------------------------
    for number in parsed.cited_sources:
        source = by_number.get(number)
        if source is None:
            report.rejected.append(
                RejectedCitation(
                    reason=CitationProblem.UNKNOWN_SOURCE,
                    claimed_source=number,
                    detail=(
                        f"The answer cited Source {number}, which was not in "
                        f"the context ({len(by_number)} source(s) were "
                        "supplied). The citation was dropped."
                    ),
                )
            )
            continue
        if number in seen:
            continue
        seen.add(number)
        report.citations.append(
            _citation_from_source(source, claimed_by_number.get(number))
        )

    # -- references the model wrote into the prose ---------------------
    if harvest_inline:
        for match in _INLINE_RE.finditer(parsed.answer):
            number = int(match.group(1))
            if number in seen:
                continue
            source = by_number.get(number)
            if source is None:
                report.rejected.append(
                    RejectedCitation(
                        reason=CitationProblem.UNKNOWN_SOURCE_IN_ANSWER,
                        claimed_source=number,
                        detail=(
                            f"The answer refers to Source {number}, which does "
                            "not exist. The reference is in the answer text "
                            "and could not be removed."
                        ),
                    )
                )
                report.grounded = False
                continue
            seen.add(number)
            report.citations.append(_citation_from_source(source, inline=True))

    report.citations.sort(key=lambda c: c.source)

    # -- quotes --------------------------------------------------------
    if check_quotes and parsed.answer:
        report.verified_quotes, report.unverified_quotes = verify_quotes(
            parsed.answer, context.sources
        )
        if report.unverified_quotes:
            report.grounded = False
            report.warnings.append(
                f"{len(report.unverified_quotes)} quoted passage(s) in the "
                "answer could not be found in the retrieved evidence. Treat "
                "them as unverified."
            )

    # -- an answer with nothing to stand on ----------------------------
    asserts_something = bool(parsed.answer) and not parsed.insufficient_evidence
    if require_citations and asserts_something and not report.citations:
        report.grounded = False
        report.rejected.append(
            RejectedCitation(
                reason=CitationProblem.NO_CITATIONS,
                detail=(
                    "The answer makes assertions but cites no retrieved "
                    "source, so none of it could be checked against the "
                    "evidence."
                ),
            )
        )
        report.warnings.append(
            "The answer cites no source. It could not be checked against the "
            "retrieved evidence and should not be relied on."
        )

    dropped = [
        r for r in report.rejected if r.reason == CitationProblem.UNKNOWN_SOURCE
    ]
    if dropped:
        report.warnings.append(
            f"{len(dropped)} citation(s) referred to a source that was not "
            "retrieved and were dropped."
        )

    mismatched = [c for c in report.citations if c.claimed]
    if mismatched:
        report.warnings.append(
            f"{len(mismatched)} citation(s) disagreed with the retrieved "
            "record; the record was used. See 'claimed' on each."
        )

    if report.rejected or report.unverified_quotes:
        logger.warning(
            "Citation validation: %d valid, %d rejected, %d unverified quote(s)",
            len(report.citations),
            len(report.rejected),
            len(report.unverified_quotes),
        )

    return report


__all__ = [
    "Citation",
    "RejectedCitation",
    "CitationReport",
    "CitationProblem",
    "validate_citations",
    "verify_quotes",
]
