"""The token budget — what fits, what gets shortened, what gets dropped.

A context window is a hard limit, and the cost of getting this wrong is
asymmetric. Under-fill and the answer is thinner than it needed to be.
Over-fill and the provider truncates the prompt from one end — usually
silently, usually taking the instructions or the last source with it —
and the model answers from a context nobody inspected.

Three rules make that not happen:

**The header is part of the cost.** Each source's Document / Page /
Section / Chunk lines cost ~35 tokens. Counting only the passage text
under-counts a six-source context by more than 200 tokens. Every block is
priced by rendering it and counting the result, not by estimating.

**A passage is never shortened below the point where it still means
something.** ``CONTEXT_MIN_SOURCE_TOKENS`` is a floor: below it the
passage is dropped instead of truncated. Half a termination clause reads
as a complete obligation with the wrong conditions attached, which is a
worse input to a language model than no passage at all. This is
requirement 8 made operational — "enough context to understand the
clause" means a clause or nothing.

**Truncation is at a sentence boundary and always declared.** Cutting
mid-sentence produces text that reads as finished when it is not; the
model has no way to know the qualifier it needed was in the next clause.
So the cut lands on a sentence end where one exists, and the rendered
block says so and says how much was removed.

Passages are fitted in **relevance order** — the strongest evidence gets
first call on the budget — and re-ordered for presentation afterwards.

One subtlety worth stating, because it looks like a bug and is not:
fitting happens before the final numbering, using each passage's position
in relevance order as its provisional source number. That is exact for
budgeting, because whatever the final order turns out to be, the set of
numbers rendered is the same ``{1..n}`` — reordering permutes which
number lands on which block, and the total cost of the headers is
unchanged. Per-source costs are recomputed after numbering so the
reported figures match the text.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.chunking.splitters import split_sentences
from app.chunking.tokenizer import TokenCounter, truncate_to_tokens
from app.context.base import (
    ContextSource,
    OmissionReason,
    OmittedSource,
)
from app.context.formatter import SEPARATOR, render_header, render_source
from app.logging_config import get_logger
from app.retrieval.base import Candidate

logger = get_logger(__name__)

#: Bounded re-fit attempts when declaring a truncation changes the
#: header size (the note line costs tokens too).
_REFIT_ATTEMPTS = 3

#: How much of the budget a sentence-aligned cut must retain before it is
#: preferred over a word-aligned one.
#:
#: Sentence alignment is worth having — a mid-sentence cut reads as
#: finished when it is not — but only while it is cheap. Legal sentences
#: are long and uneven, so "the last whole sentence that fits" can land
#: far short of the budget: the fixture contract's termination clause
#: sentence-aligns to 38 tokens of a 60-token allowance, losing 37% of
#: the space and, with it, the second half of the clause. Worse, the
#: shortfall can then drop the passage below ``min_source_tokens`` and
#: discard it altogether — the most relevant clause in the corpus thrown
#: away by the mechanism meant to fit it.
#:
#: So: keep the sentence boundary when it costs a quarter of the budget
#: or less, and otherwise fill the budget on a word boundary. Either way
#: the rendered block declares that the passage was shortened, so a
#: mid-sentence cut is never undeclared.
SENTENCE_ALIGN_MIN_RATIO = 0.75


def strip_repeated_heading(text: str, section: Optional[str]) -> Tuple[str, bool]:
    """Drop a leading section heading already stated in the header.

    Stage 3 attaches the heading to the chunk body on purpose — it
    sharpens the embedding and it keeps the passage self-describing in
    the index. In the *prompt* it is redundant: the header line above
    already says ``Section: 4. TERMINATION``. Removing it saves a few
    tokens per source and, more usefully, stops the model seeing the same
    string twice and treating the repetition as emphasis.

    Conservative by construction: only an exact leading match of the
    first line is removed, and only when text remains afterwards.
    """
    if not text or not section:
        return text, False

    stripped = text.lstrip()
    heading = section.strip()
    if not stripped.lower().startswith(heading.lower()):
        return text, False

    remainder = stripped[len(heading):].lstrip("\n\r \t:.-")
    if not remainder.strip():
        # The chunk is nothing but its heading. Keep it — dropping the
        # body entirely would render a source with no text at all.
        return text, False
    return remainder, True


def truncate_at_sentence(
    text: str, max_tokens: int, counter: TokenCounter
) -> str:
    """Trim to ``max_tokens``, preferring a sentence boundary.

    Falls back to a word boundary in two cases: when a single sentence
    already exceeds the budget — legal drafting produces 300-word
    sentences often enough that this is a normal path, not an edge case —
    and when stopping at the last whole sentence would waste more than
    ``SENTENCE_ALIGN_MIN_RATIO`` of the space available.
    """
    if max_tokens <= 0:
        return ""
    if counter.count(text) <= max_tokens:
        return text

    sentences = split_sentences(text)
    if len(sentences) > 1:
        kept: List[str] = []
        used = 0
        for sentence in sentences:
            cost = counter.count(sentence if not kept else " " + sentence)
            if used + cost > max_tokens:
                break
            kept.append(sentence)
            used += cost
        if kept:
            aligned = " ".join(kept)
            if counter.count(aligned) >= max_tokens * SENTENCE_ALIGN_MIN_RATIO:
                return aligned

    return truncate_to_tokens(text, max_tokens, counter)


def _omit(candidate: Candidate, reason: str, tokens: int = 0) -> OmittedSource:
    return OmittedSource(
        chunk_id=candidate.chunk_id,
        reason=reason,
        score=float(candidate.final_score),
        citation=candidate.chunk.citation(),
        tokens=tokens,
    )


def fit_to_budget(
    candidates: Sequence[Candidate],
    counter: TokenCounter,
    max_tokens: int,
    max_source_tokens: int = 0,
    min_source_tokens: int = 60,
    truncate: bool = True,
    strip_heading: bool = True,
    duplicate_citations: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Tuple[List[ContextSource], List[OmittedSource]]:
    """Turn selected candidates into sources that fit the budget.

    ``candidates`` must already be in relevance order. Returns
    ``(sources, omitted)``; the sources carry provisional numbers that
    the builder reassigns after ordering.
    """
    sources: List[ContextSource] = []
    omitted: List[OmittedSource] = []
    remaining = max(0, int(max_tokens))

    for position, candidate in enumerate(candidates, start=1):
        chunk = candidate.chunk

        body, heading_stripped = (
            strip_repeated_heading(chunk.text, chunk.section)
            if strip_heading
            else (chunk.text, False)
        )
        original_tokens = counter.count(body)

        source = ContextSource(
            chunk=chunk,
            number=position,
            score=float(candidate.final_score),
            retrieval_rank=candidate.rank or position,
            text=body,
            original_tokens=original_tokens,
            text_tokens=original_tokens,
            heading_stripped=heading_stripped,
            duplicates=list(candidate.duplicates),
            duplicate_citations=list(
                (duplicate_citations or {}).get(candidate.chunk_id, [])
            ),
        )

        separator_cost = counter.count(SEPARATOR) if sources else 0
        overhead = counter.count(render_header(source))
        available = remaining - separator_cost - overhead

        # A per-source ceiling, independent of the overall budget: one
        # 2000-token schedule must not consume the space five clauses
        # would have used.
        if max_source_tokens and max_source_tokens > 0:
            available = min(available, int(max_source_tokens))

        if available < min_source_tokens:
            # Not enough room left for a meaningful passage. Stop rather
            # than continue — later candidates are weaker, so a smaller
            # one squeezing in ahead of a stronger one that did not fit
            # would be arbitrary.
            omitted.append(
                _omit(
                    candidate,
                    OmissionReason.BUDGET_EXHAUSTED,
                    tokens=original_tokens + overhead,
                )
            )
            continue

        if original_tokens > available:
            if not truncate:
                omitted.append(
                    _omit(
                        candidate,
                        OmissionReason.BUDGET_EXHAUSTED,
                        tokens=original_tokens + overhead,
                    )
                )
                continue

            fitted = _truncate_and_refit(
                source, counter, available, remaining, separator_cost
            )
            if fitted is None:
                omitted.append(
                    _omit(
                        candidate,
                        OmissionReason.TOO_SMALL_TO_TRUNCATE,
                        tokens=original_tokens + overhead,
                    )
                )
                continue
            if source.text_tokens < min_source_tokens:
                # Shortened past the point of meaning something.
                omitted.append(
                    _omit(
                        candidate,
                        OmissionReason.TOO_SMALL_TO_TRUNCATE,
                        tokens=original_tokens + overhead,
                    )
                )
                continue
            total = fitted
        else:
            total = separator_cost + counter.count(render_source(source))

        source.total_tokens = total
        sources.append(source)
        remaining -= total

    return sources, omitted


def _truncate_and_refit(
    source: ContextSource,
    counter: TokenCounter,
    available: int,
    remaining: int,
    separator_cost: int,
) -> Optional[int]:
    """Shorten a passage until the whole rendered block fits.

    Declaring the truncation costs tokens of its own — the ``Note:`` line
    says how much was removed — so the first attempt can still overflow
    by a handful. Rather than pad with a guessed reserve, the block is
    re-measured and the body shortened by the overflow, at most a few
    times. Returns the block's total cost, or ``None`` if it cannot be
    made to fit.
    """
    target = available
    for _ in range(_REFIT_ATTEMPTS):
        if target <= 0:
            return None
        source.text = truncate_at_sentence(source.text, target, counter)
        source.text_tokens = counter.count(source.text)
        source.truncated = source.text_tokens < source.original_tokens

        total = separator_cost + counter.count(render_source(source))
        if total <= remaining:
            return total

        overflow = total - remaining
        target = source.text_tokens - overflow - 1

    return None


__all__ = [
    "fit_to_budget",
    "truncate_at_sentence",
    "strip_repeated_heading",
]
