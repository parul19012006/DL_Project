"""Evidence selection — deciding which passages are worth the prompt.

Retrieval hands over ~6 ranked passages. Not all of them belong in the
context, and the reason is not tidiness: **every irrelevant passage in a
prompt is an active liability.** It competes for the model's attention,
it gives a wrong answer something to be grounded in, and in a legal
product a plausible-looking citation to the wrong clause is worse than
no answer. Tokens spent on a weak passage are also tokens not spent on a
strong one.

Three filters run here, in this order, and each is separately
configurable because each has a different failure mode:

1. **Score floor** (``CONTEXT_MIN_SCORE``). Off by default. A floor is
   the bluntest instrument available and the easiest to set wrongly —
   scores are relative to a query's own candidate set, so a floor tuned
   on one corpus silently empties the context on another. It exists for
   callers who have measured their own distribution.
2. **Redundancy removal.** Stage 5 already deduplicates its candidates,
   so this is usually a no-op there — but the builder must work on
   passages that did *not* come through that pipeline, and running it
   here is what makes it independently correct rather than dependent on
   an upstream promise.
3. **A hard cap on source count** (``CONTEXT_MAX_SOURCES``). Independent
   of the token budget: ten short passages can fit a budget comfortably
   and still be worse than the best five, because attention is finite
   even when context is not.

Selection works in **relevance order** — strongest evidence wins the
budget. Presentation order is a separate decision, made in
:mod:`app.context.ordering` after this has finished.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from app.logging_config import get_logger
from app.context.base import OmissionReason, OmittedSource
from app.retrieval.base import Candidate
from app.retrieval.dedup import jaccard, normalize_for_comparison, shingles

logger = get_logger(__name__)


def _omit(candidate: Candidate, reason: str, **extra) -> OmittedSource:
    return OmittedSource(
        chunk_id=candidate.chunk_id,
        reason=reason,
        score=float(candidate.final_score),
        citation=candidate.chunk.citation(),
        **extra,
    )


def remove_redundant(
    candidates: Sequence[Candidate],
    threshold: float = 0.85,
    shingle_size: int = 5,
) -> Tuple[List[Candidate], List[OmittedSource]]:
    """Collapse passages that say the same thing.

    Input is assumed strongest-first, so the copy retrieval preferred is
    the survivor. The duplicate's ``chunk_id`` is recorded on it, and the
    duplicate itself is returned in the omitted list with its citation —
    a passage that appears in two documents is cited to both, not
    silently reduced to one.

    Adjacent chunks of the same document are exempt from *near*-duplicate
    comparison, exactly as in Stage 5: their overlap is deliberate and
    collapsing it would undo the mechanism that makes a boundary-spanning
    clause retrievable. Byte-identical text is collapsed regardless.
    """
    kept: List[Candidate] = []
    kept_shingles = []
    exact = {}
    omitted: List[OmittedSource] = []

    for candidate in candidates:
        normalised = normalize_for_comparison(candidate.text)

        twin = exact.get(normalised) if normalised else None
        if twin is not None:
            twin.duplicates.append(candidate.chunk_id)
            omitted.append(
                _omit(
                    candidate,
                    OmissionReason.REDUNDANT,
                    duplicate_of=twin.chunk_id,
                )
            )
            continue

        fingerprint = shingles(candidate.text, shingle_size)
        duplicate_of = None
        for existing, existing_shingles in zip(kept, kept_shingles):
            same_document = (
                existing.chunk.document_id == candidate.chunk.document_id
            )
            adjacent = (
                same_document
                and abs(existing.chunk.chunk_index - candidate.chunk.chunk_index) == 1
            )
            if adjacent:
                continue
            if jaccard(existing_shingles, fingerprint) >= threshold:
                duplicate_of = existing
                break

        if duplicate_of is not None:
            duplicate_of.duplicates.append(candidate.chunk_id)
            omitted.append(
                _omit(
                    candidate,
                    OmissionReason.REDUNDANT,
                    duplicate_of=duplicate_of.chunk_id,
                )
            )
            continue

        kept.append(candidate)
        kept_shingles.append(fingerprint)
        if normalised:
            exact[normalised] = candidate

    return kept, omitted


def select_evidence(
    candidates: Sequence[Candidate],
    max_sources: int = 8,
    min_score: float = 0.0,
    dedupe: bool = True,
    redundancy_threshold: float = 0.85,
    shingle_size: int = 5,
) -> Tuple[List[Candidate], List[OmittedSource]]:
    """Filter retrieved passages down to the ones worth prompting with.

    Returns ``(selected, omitted)`` — both, always. The second list is
    not a diagnostic afterthought: it is how the caller answers "what
    about clause 9?" without re-running retrieval.
    """
    omitted: List[OmittedSource] = []

    # Empty passages can reach here from an index built by an earlier
    # version, or from a chunk whose text was whitespace after cleaning.
    # They cost header tokens and carry nothing.
    present: List[Candidate] = []
    for candidate in candidates:
        if candidate.text and candidate.text.strip():
            present.append(candidate)
        else:
            omitted.append(_omit(candidate, OmissionReason.EMPTY_TEXT))

    # Work strongest-first regardless of the order handed in, so the
    # budget and the source cap are spent on the best evidence. Ties
    # break on chunk_id, keeping the result reproducible.
    ranked = sorted(present, key=lambda c: (-c.final_score, c.chunk_id))

    if min_score > 0:
        above = []
        for candidate in ranked:
            if candidate.final_score >= min_score:
                above.append(candidate)
            else:
                omitted.append(_omit(candidate, OmissionReason.BELOW_SCORE_FLOOR))
        ranked = above

    if dedupe:
        ranked, redundant = remove_redundant(
            ranked, threshold=redundancy_threshold, shingle_size=shingle_size
        )
        omitted.extend(redundant)

    if max_sources and len(ranked) > max_sources:
        for candidate in ranked[max_sources:]:
            omitted.append(_omit(candidate, OmissionReason.MAX_SOURCES))
        ranked = ranked[:max_sources]

    logger.debug(
        "Context selection: %d in, %d selected, %d omitted",
        len(candidates),
        len(ranked),
        len(omitted),
    )
    return ranked, omitted


__all__ = ["select_evidence", "remove_redundant"]
