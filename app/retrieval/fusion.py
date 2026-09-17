"""Hybrid merging — combining the semantic and keyword result lists.

The two retrievers return overlapping lists scored on incompatible
scales: cosine similarity lives in [0, 1] and is roughly comparable
across queries, BM25 is unbounded and means nothing except relative to
the rest of *this* corpus. Merging them is therefore not a matter of
adding numbers.

Two strategies are implemented, and the default is the rank-based one.

**Reciprocal Rank Fusion (default).** Each candidate scores
``weight / (k + rank)`` in each list it appears in, and the scores are
summed. Only *positions* are used, so the scales never have to be
reconciled and no normalisation assumption can be wrong. A passage
ranked 2nd by vectors and 3rd by BM25 beats one ranked 1st by vectors
and absent from BM25 — which is the behaviour we want, because agreement
between two independent signals is stronger evidence than a single
signal's confidence. ``k`` (default 60, the value from the original
paper) damps the difference between the top ranks, so rank 1 does not
overwhelm rank 2.

**Weighted score fusion.** ``w_semantic * semantic + w_keyword * keyword``
on normalised scores. Available because it is more controllable when you
know your corpus, and because it makes the weights mean something
obvious. Its weakness is the reason it is not the default: a candidate
absent from one list contributes 0 for that signal, which punishes a
strong semantic hit merely for falling outside the keyword cut-off.

Whichever runs, the output is scaled so the best candidate sits at 1.0
(:func:`app.retrieval.base.scale_to_max`) — otherwise the ranking stage's
additive boosts would not be on speaking terms with an RRF sum of 0.016.
Scaling by the maximum rather than min-maxing the range is deliberate and
load-bearing: min-max stretches every run to fill [0, 1] however close
the raw scores were, so two candidates separated in the fourth decimal
would come out a full point apart and no boost could ever move them.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from app.logging_config import get_logger
from app.retrieval.base import (
    SOURCE_KEYWORD,
    SOURCE_SEMANTIC,
    Candidate,
    RetrievalError,
    scale_to_max,
)

logger = get_logger(__name__)

RRF = "rrf"
WEIGHTED = "weighted"
METHODS = (RRF, WEIGHTED)


def _collect(runs: Sequence[Sequence[Candidate]]) -> Dict[str, Candidate]:
    """One candidate per chunk_id, carrying every run's signals."""
    merged: Dict[str, Candidate] = {}
    for run in runs:
        for candidate in run:
            existing = merged.get(candidate.chunk_id)
            if existing is None:
                merged[candidate.chunk_id] = candidate
            else:
                existing.merge(candidate)
    return merged


def reciprocal_rank_fusion(
    semantic: Sequence[Candidate],
    keyword: Sequence[Candidate],
    k: int = 60,
    semantic_weight: float = 1.0,
    keyword_weight: float = 1.0,
) -> List[Candidate]:
    """Rank-based fusion. Equal weights give textbook RRF."""
    merged = _collect([semantic, keyword])

    raw: List[float] = []
    candidates = list(merged.values())
    for candidate in candidates:
        score = 0.0
        if candidate.semantic_rank:
            score += semantic_weight / (k + candidate.semantic_rank)
        if candidate.keyword_rank:
            score += keyword_weight / (k + candidate.keyword_rank)
        raw.append(score)

    for candidate, normalised, value in zip(
        candidates, scale_to_max(raw), raw
    ):
        candidate.fused_score = float(normalised)
        candidate.boosts["rrf_raw"] = round(value, 8)

    return _ordered(candidates)


def weighted_fusion(
    semantic: Sequence[Candidate],
    keyword: Sequence[Candidate],
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.4,
) -> List[Candidate]:
    """Score-based fusion over normalised signals."""
    merged = _collect([semantic, keyword])
    candidates = list(merged.values())

    total = semantic_weight + keyword_weight
    if total <= 0:
        raise RetrievalError(
            "SEMANTIC_WEIGHT and KEYWORD_WEIGHT cannot both be zero"
        )
    w_semantic = semantic_weight / total
    w_keyword = keyword_weight / total

    raw = [
        w_semantic * candidate.semantic_score
        + w_keyword * candidate.keyword_score
        for candidate in candidates
    ]
    for candidate, normalised in zip(candidates, scale_to_max(raw)):
        candidate.fused_score = float(normalised)

    return _ordered(candidates)


def _ordered(candidates: List[Candidate]) -> List[Candidate]:
    """Best first; ties broken on chunk_id so the order is reproducible."""
    candidates.sort(key=lambda c: (-c.fused_score, c.chunk_id))
    return candidates


def fuse(
    semantic: Sequence[Candidate],
    keyword: Sequence[Candidate],
    method: str = RRF,
    rrf_k: int = 60,
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.4,
) -> List[Candidate]:
    """Merge two result lists into one ordered candidate list.

    Either list may be empty — a query with no content terms produces no
    keyword run, and a tenant with an empty index produces neither. The
    surviving list passes through with its own ordering preserved, so a
    single-signal query still works.
    """
    chosen = (method or RRF).strip().lower()
    if chosen not in METHODS:
        raise RetrievalError(
            f"Unknown fusion method '{method}'. Available: {', '.join(METHODS)}"
        )

    if chosen == WEIGHTED:
        return weighted_fusion(
            semantic, keyword, semantic_weight, keyword_weight
        )
    return reciprocal_rank_fusion(
        semantic,
        keyword,
        k=rrf_k,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
    )


def agreement_ratio(candidates: Sequence[Candidate]) -> float:
    """Fraction of candidates both retrievers found.

    A useful health signal when tuning: near zero means the two signals
    are looking at different things and fusion is doing all the work;
    near one means the keyword run is adding little.
    """
    if not candidates:
        return 0.0
    both = sum(
        1
        for c in candidates
        if SOURCE_SEMANTIC in c.sources and SOURCE_KEYWORD in c.sources
    )
    return round(both / len(candidates), 4)


__all__ = [
    "fuse",
    "reciprocal_rank_fusion",
    "weighted_fusion",
    "agreement_ratio",
    "RRF",
    "WEIGHTED",
    "METHODS",
]
