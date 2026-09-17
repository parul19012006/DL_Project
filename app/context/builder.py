"""Context construction — the orchestration, and nothing else.

    retrieved passages
          │
          ▼
    1. SELECT      strongest evidence, redundancy removed   (selection.py)
          │
          ▼
    2. FIT         token budget, per-source cap, truncation (budget.py)
          │
          ▼
    3. ORDER       document order by default                (ordering.py)
          │
          ▼
    4. NUMBER      Source 1..N, continuations marked
          │
          ▼
    5. RENDER      uniform labelled blocks                  (formatter.py)
          │
          ▼
    BuiltContext:  text + sources + omitted + stats

Two orderings are in play and the sequence above is deliberate.
Selection and budgeting run in **relevance order**, so the strongest
evidence wins the space. Presentation runs in **document order**, so the
model reads the clauses in the sequence the contract states them.
Numbering happens after ordering, so ``Source 3`` always denotes the
third block in the text the model actually sees.

This module holds no policy of its own. Every decision — what counts as
redundant, what fits, what order to read in, how a block looks — lives in
the module named for it and can be tested, replaced or reasoned about
without the others. That is requirement 10, and it is also why the
builder accepts a plain list of :class:`~app.models.chunk.Chunk` objects
as readily as a retrieval outcome: nothing here needs retrieval to have
happened.

**This stage does not generate an answer.** It does not write
instructions, a system prompt, or a question into the block either —
those are the generation stage's to own, and putting them here would fix
the prompt for every future caller. What comes back is evidence, labelled.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from app.chunking.tokenizer import TokenCounter, get_token_counter
from app.config import Settings, get_settings
from app.context import formatter, ordering
from app.context.base import (
    BuiltContext,
    ContextSource,
    ContextStats,
    OmissionReason,
    OmittedSource,
    to_candidates,
)
from app.context.budget import fit_to_budget
from app.context.selection import select_evidence
from app.logging_config import get_logger

logger = get_logger(__name__)


class ContextBuilder:
    """Builds an LLM-ready context block from retrieved passages."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        counter: Optional[TokenCounter] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.counter = counter or get_token_counter(self.settings)

    # -- main ----------------------------------------------------------

    def build(
        self,
        items: Sequence[Any],
        query: str = "",
        max_tokens: Optional[int] = None,
        max_sources: Optional[int] = None,
        min_sources: Optional[int] = None,
        min_score: Optional[float] = None,
        order: Optional[str] = None,
        dedupe: Optional[bool] = None,
        truncate: Optional[bool] = None,
        max_source_tokens: Optional[int] = None,
        min_source_tokens: Optional[int] = None,
        strip_heading: Optional[bool] = None,
    ) -> BuiltContext:
        """Build a context from candidates or chunks.

        Every knob is overridable per call. Retrieval settings tuned for
        a chat feature are rarely right for a summarisation job over one
        document, and a redeploy is not an acceptable way to change that.
        """
        settings = self.settings
        started = time.perf_counter()

        budget = int(
            max_tokens if max_tokens is not None else settings.context_max_tokens
        )
        cap = int(
            max_sources if max_sources is not None else settings.context_max_sources
        )
        floor_sources = int(
            min_sources if min_sources is not None else settings.context_min_sources
        )
        floor_score = float(
            min_score if min_score is not None else settings.context_min_score
        )
        chosen_order = (order or settings.context_order).strip().lower()
        do_dedupe = (
            settings.context_dedupe_enabled if dedupe is None else bool(dedupe)
        )
        do_truncate = (
            settings.context_truncate_long_sources
            if truncate is None
            else bool(truncate)
        )
        source_cap = int(
            max_source_tokens
            if max_source_tokens is not None
            else settings.context_max_source_tokens
        )
        source_floor = int(
            min_source_tokens
            if min_source_tokens is not None
            else settings.context_min_source_tokens
        )
        do_strip = (
            settings.context_strip_repeated_heading
            if strip_heading is None
            else bool(strip_heading)
        )

        context = BuiltContext(
            query=query,
            tokenizer=getattr(self.counter, "name", "unknown"),
            order=chosen_order,
        )
        context.stats.tokens_budget = budget

        candidates = to_candidates(items)
        context.stats.candidates_in = len(candidates)
        if not candidates:
            context.warnings.append(
                "Insufficient evidence: retrieval returned no passages, so "
                "there is nothing to ground an answer in"
            )
            context.stats.build_ms = int((time.perf_counter() - started) * 1000)
            return context

        # -- 1. select --------------------------------------------------
        selected, omitted = select_evidence(
            candidates,
            max_sources=cap,
            min_score=floor_score,
            dedupe=do_dedupe,
            redundancy_threshold=settings.context_redundancy_threshold,
            shingle_size=settings.dedupe_shingle_size,
        )

        # A passage collapsed as redundant *here* still has its chunk in
        # hand, so its citation can travel with the survivor. (Duplicates
        # already collapsed upstream by Stage 5 arrive as bare ids; those
        # are carried in ``ContextSource.duplicates`` rather than
        # rendered, because an id is not a citation a reader can use.)
        duplicate_citations = _duplicate_citation_map(omitted)

        # -- 2. fit -----------------------------------------------------
        sources, budget_omitted = fit_to_budget(
            selected,
            counter=self.counter,
            max_tokens=budget,
            max_source_tokens=source_cap,
            min_source_tokens=source_floor,
            truncate=do_truncate,
            strip_heading=do_strip,
            duplicate_citations=duplicate_citations,
        )
        omitted.extend(budget_omitted)

        # -- 3. order and 4. number -------------------------------------
        sources = _reorder(sources, chosen_order)
        _number(sources)

        # Header contents changed when the numbers and continuation lines
        # were assigned, so per-source costs are re-measured against the
        # text that will actually be sent. The total is unchanged in
        # aggregate (see the note in budget.py) but the per-source
        # figures must match the block or they are not worth reporting.
        for position, source in enumerate(sources):
            separator = (
                self.counter.count(formatter.SEPARATOR) if position else 0
            )
            source.total_tokens = separator + self.counter.count(
                formatter.render_source(source)
            )

        # -- 5. render --------------------------------------------------
        context.sources = sources
        context.omitted = omitted
        context.text = formatter.render(sources)
        context.stats = _summarise(
            context, omitted, budget, self.counter, started
        )

        _add_warnings(context, floor_sources, budget)

        logger.info(
            "Context: %d source(s) from %d document(s), %d/%d tokens, "
            "%d omitted (%d redundant, %d for budget), %d truncated",
            len(sources),
            context.stats.documents,
            context.stats.tokens_used,
            budget,
            len(omitted),
            context.stats.redundant_removed,
            context.stats.dropped_for_budget,
            context.stats.truncated_sources,
        )
        return context

    # -- convenience ---------------------------------------------------

    def from_outcome(self, outcome, **kwargs) -> BuiltContext:
        """Build directly from a Stage 5 :class:`RetrievalOutcome`."""
        return self.build(
            outcome.results,
            query=(outcome.query.raw if outcome.query else ""),
            **kwargs,
        )


# =====================================================================
# Helpers
# =====================================================================


def _duplicate_citation_map(
    omitted: Sequence[OmittedSource],
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for entry in omitted:
        if entry.reason == OmissionReason.REDUNDANT and entry.duplicate_of:
            out.setdefault(entry.duplicate_of, []).append(dict(entry.citation))
    return out


def _reorder(sources: List[ContextSource], order: str) -> List[ContextSource]:
    """Apply the presentation order to already-fitted sources.

    :func:`app.context.ordering.order_sources` works on candidates, so
    the sources are keyed back to it by chunk_id rather than duplicating
    the ordering rules here — there must be exactly one place that knows
    what "document order" means.
    """
    from app.retrieval.base import Candidate

    proxies = []
    by_id: Dict[str, ContextSource] = {}
    for source in sources:
        proxy = Candidate(chunk=source.chunk)
        proxy.final_score = source.score
        proxies.append(proxy)
        by_id[source.chunk_id] = source

    ordered = ordering.order_sources(proxies, order)
    return [by_id[p.chunk_id] for p in ordered]


def _number(sources: List[ContextSource]) -> None:
    """Assign 1..N and mark continuations, in that order.

    Continuations are computed after numbering because the label they
    render ("Continues: Source 2") refers to a number that does not exist
    until now.
    """
    for position, source in enumerate(sources, start=1):
        source.number = position
        source.continues_source = None

    for position in range(1, len(sources)):
        previous, current = sources[position - 1], sources[position]
        same_document = (
            previous.chunk.document_id == current.chunk.document_id
        )
        if same_document and (
            current.chunk.chunk_index == previous.chunk.chunk_index + 1
        ):
            current.continues_source = previous.number


def _summarise(
    context: BuiltContext,
    omitted: Sequence[OmittedSource],
    budget: int,
    counter: TokenCounter,
    started: float,
) -> ContextStats:
    stats = ContextStats(
        candidates_in=context.stats.candidates_in,
        sources_out=len(context.sources),
        tokens_budget=budget,
    )
    stats.documents = len({s.chunk.document_id for s in context.sources})
    stats.text_tokens = sum(s.text_tokens for s in context.sources)
    stats.tokens_used = counter.count(context.text)
    stats.overhead_tokens = max(0, stats.tokens_used - stats.text_tokens)
    stats.truncated_sources = sum(1 for s in context.sources if s.truncated)
    stats.omitted_text_tokens = sum(s.omitted_tokens for s in context.sources)

    for entry in omitted:
        if entry.reason == OmissionReason.REDUNDANT:
            stats.redundant_removed += 1
        elif entry.reason == OmissionReason.BELOW_SCORE_FLOOR:
            stats.below_floor_removed += 1
        elif entry.reason == OmissionReason.MAX_SOURCES:
            stats.over_max_sources += 1
        elif entry.reason in (
            OmissionReason.BUDGET_EXHAUSTED,
            OmissionReason.TOO_SMALL_TO_TRUNCATE,
        ):
            stats.dropped_for_budget += 1

    stats.build_ms = int((time.perf_counter() - started) * 1000)
    return stats


def _add_warnings(context: BuiltContext, min_sources: int, budget: int) -> None:
    """State what a caller needs to know before prompting with this.

    Warnings are reported, never enforced. This stage does not decide
    that a question is unanswerable — it says what the evidence looks
    like and leaves the judgement to the stage that writes the prompt.
    """
    count = len(context.sources)

    if count == 0:
        context.warnings.append(
            "Insufficient evidence: nothing survived selection, so there is "
            "nothing to ground an answer in"
        )
        return

    if min_sources and count < min_sources:
        context.warnings.append(
            f"Insufficient evidence: {count} source(s) reached the context "
            f"but CONTEXT_MIN_SOURCES is {min_sources}. Any answer will rest "
            "on thin evidence and should be hedged accordingly"
        )

    if context.stats.truncated_sources:
        context.warnings.append(
            f"{context.stats.truncated_sources} source(s) were shortened to "
            f"fit the {budget}-token budget; "
            f"{context.stats.omitted_text_tokens} tokens of passage text are "
            "not in the context"
        )

    if context.stats.dropped_for_budget:
        context.warnings.append(
            f"{context.stats.dropped_for_budget} retrieved passage(s) did not "
            "fit the token budget and were left out; their citations are in "
            "'omitted'"
        )

    if any(s.chunk.ocr for s in context.sources):
        context.warnings.append(
            "At least one source came from an OCR'd page; exact quotations "
            "from it may contain recognition errors"
        )


# =====================================================================
# Module-level convenience
# =====================================================================

_builder: Optional[ContextBuilder] = None


def get_context_builder(settings: Optional[Settings] = None) -> ContextBuilder:
    global _builder
    if _builder is None or (
        settings is not None and _builder.settings is not settings
    ):
        _builder = ContextBuilder(settings)
    return _builder


def set_context_builder(builder: Optional[ContextBuilder]) -> None:
    global _builder
    _builder = builder


def build_context(
    items: Sequence[Any],
    query: str = "",
    settings: Optional[Settings] = None,
    **kwargs,
) -> BuiltContext:
    """One-call context construction."""
    return get_context_builder(settings).build(items, query=query, **kwargs)


__all__ = [
    "ContextBuilder",
    "build_context",
    "get_context_builder",
    "set_context_builder",
]
