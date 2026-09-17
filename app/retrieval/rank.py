"""Candidate ranking — cheap, explainable signals fusion cannot see.

Fusion knows only positions and similarity. It has no idea that the user
asked about *clause 7.3* and that one candidate's section actually is
7.3, or that a quoted phrase appears verbatim in one passage and nowhere
else. Those are strong, nearly free signals, and they are applied here —
before the reranker, so that a passage the reranker would never see
because it sat at position 19 can be lifted into view.

Every adjustment is additive, bounded, named and recorded in
``Candidate.boosts``. That is a deliberate constraint: a multiplicative
or learned reweighting would be harder to justify to a user asking why
this clause was cited, and "why was this retrieved" is a question a
legal product has to be able to answer.

The boosts:

``reference``     the query named a clause/section and this chunk's
                  section or text carries that number. The single most
                  reliable signal in the whole pipeline when it fires,
                  because the user has told us where to look. When the
                  match is on the chunk's *section* — this chunk **is**
                  the clause the user named — the candidate is also
                  *pinned*, and no later stage may demote it.
``phrase``        a quoted phrase from the query occurs verbatim.
``agreement``     both retrievers returned this passage independently.
``heading``       a query term appears in the section heading, which in
                  legal drafting is a strong statement of topic.

A penalty also applies: chunks flagged ``heading_only`` — a bare heading
with no body — carry no obligation and should never outrank the clause
they introduce. The chunker already merges these away, so this is a
belt-and-braces guard for indexes built by earlier versions.
"""

from __future__ import annotations

import re
from typing import List, Sequence

from app.logging_config import get_logger
from app.retrieval.base import Candidate
from app.retrieval.query import ProcessedQuery, tokenize

logger = get_logger(__name__)


def _contains_reference(text: str, number: str) -> bool:
    """Does ``text`` refer to clause ``number`` as a number, not a digit?

    Three cases have to come out right, and a naive word-boundary match
    gets two of them wrong:

    * ``"4. TERMINATION"`` **is** clause 4 — the trailing full stop is
      how section headings are written, so it must not block the match;
    * ``"14.25"`` is **not** clause 4 — hence the left guard;
    * ``"USD 4.20"`` is **not** clause 4 — hence ``(?!\\.\\d)``.
    """
    pattern = rf"(?<![\w.]){re.escape(number)}(?!\w)(?!\.\d)"
    return re.search(pattern, text) is not None


def rank_candidates(
    candidates: Sequence[Candidate],
    query: ProcessedQuery,
    reference_boost: float = 0.25,
    phrase_boost: float = 0.15,
    agreement_boost: float = 0.05,
    heading_boost: float = 0.05,
    heading_only_penalty: float = 0.30,
    limit: int = 0,
) -> List[Candidate]:
    """Apply metadata boosts, order, and assign ranks.

    Returns a new ordered list; the candidates themselves are mutated so
    that every score they collected stays with them.
    """
    query_terms = set(query.terms)
    references = [number for _, number in query.references]

    for candidate in candidates:
        chunk = candidate.chunk
        section = (chunk.section or "").strip()
        lowered_text = chunk.text.lower()
        score = candidate.fused_score

        # -- explicit clause reference --------------------------------
        if references:
            in_section = any(
                _contains_reference(section, number) for number in references
            )
            in_text = any(
                _contains_reference(lowered_text, number) for number in references
            )
            if in_section:
                candidate.boosts["reference"] = reference_boost
                score += reference_boost
                # A named clause is a lookup, not a ranking problem. The
                # user said "clause 8"; this chunk *is* clause 8. Pinning
                # it is not a bigger boost — it is a statement that no
                # later stage may overrule the citation the user gave.
                #
                # This is not hypothetical tidiness. Asked "what does
                # clause 8 say", a relevance model scores the clause that
                # *mentions* "Clause 8" above the clause itself, because
                # the mentioning passage contains the literal string and
                # clause 8's own text does not. Without pinning, the one
                # passage the user explicitly asked for is the one that
                # gets demoted.
                candidate.pinned = True
            elif in_text:
                # Mentioning a clause is weaker evidence than being it.
                candidate.boosts["reference"] = reference_boost / 2
                score += reference_boost / 2

        # -- exact quoted phrase --------------------------------------
        if query.phrases:
            matched = sum(1 for p in query.phrases if p and p in lowered_text)
            if matched:
                value = phrase_boost * min(matched, 2)
                candidate.boosts["phrase"] = value
                score += value

        # -- both retrievers agreed -----------------------------------
        if candidate.matched_both:
            candidate.boosts["agreement"] = agreement_boost
            score += agreement_boost

        # -- query term in the section heading ------------------------
        if section and query_terms:
            # tokenize() already applies the term-family mapping, so the
            # heading's terms and the query's terms are comparable.
            if set(tokenize(section)) & query_terms:
                candidate.boosts["heading"] = heading_boost
                score += heading_boost

        # -- heading with no body -------------------------------------
        if chunk.heading_only:
            candidate.boosts["heading_only"] = -heading_only_penalty
            score -= heading_only_penalty

        candidate.rank_score = max(0.0, score)
        candidate.final_score = candidate.rank_score

    ordered = sorted(
        candidates, key=lambda c: (not c.pinned, -c.rank_score, c.chunk_id)
    )
    if limit and limit > 0:
        ordered = ordered[:limit]
    for position, candidate in enumerate(ordered, start=1):
        candidate.rank = position
    return ordered


def assign_ranks(candidates: Sequence[Candidate]) -> List[Candidate]:
    """Re-number an already-ordered list, 1-based."""
    out = list(candidates)
    for position, candidate in enumerate(out, start=1):
        candidate.rank = position
    return out


__all__ = ["rank_candidates", "assign_ranks"]
