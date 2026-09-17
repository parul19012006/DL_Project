"""Shared retrieval types.

Stage 5 turns a query into a short list of passages. Several components
contribute a *signal* along the way — vector similarity, BM25, metadata
boosts, a cross-encoder — and every one of them has to be able to speak
about the same passage without losing what the others already said.

:class:`Candidate` is that carrier. One chunk, every score it has
collected, and the retrievers that produced it. Nothing is overwritten:
``semantic_score``, ``keyword_score``, ``fused_score``, ``rank_score``
and ``rerank_score`` all survive to the end of the pipeline, which is
what makes a result explainable ("this came back third on vectors,
first on keywords, and the reranker agreed") instead of a bare number.

The alternative — passing tuples, or re-using
:class:`~app.vectorstore.base.SearchHit` with a single mutable
``score`` — loses exactly the information needed to debug a bad answer,
and a bad answer in a legal product is the expensive kind.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from app.models.chunk import Chunk
from app.vectorstore.base import SearchFilter, SearchHit

#: Retriever names used in ``Candidate.sources``.
SOURCE_SEMANTIC = "semantic"
SOURCE_KEYWORD = "keyword"


class RetrievalError(RuntimeError):
    """A retrieval stage could not complete."""


class RerankerUnavailableError(RetrievalError):
    """The configured reranker could not be loaded."""


@dataclass
class Candidate:
    """One passage plus every score any stage has given it."""

    chunk: Chunk

    # -- per-retriever signals ----------------------------------------
    #: Vector similarity, higher-is-better, as normalised by the store.
    semantic_score: float = 0.0
    #: BM25 score, normalised to [0, 1] across the keyword run.
    keyword_score: float = 0.0
    #: Raw, un-normalised BM25 — kept because the normalised value is
    #: only meaningful relative to the other candidates in one query.
    keyword_score_raw: float = 0.0
    #: 1-based position within each retriever's own result list.
    semantic_rank: Optional[int] = None
    keyword_rank: Optional[int] = None

    # -- pipeline stages ----------------------------------------------
    #: Output of hybrid merging (RRF or weighted).
    fused_score: float = 0.0
    #: Fused score after metadata/agreement boosts — the candidate
    #: ranking stage.
    rank_score: float = 0.0
    #: Cross-encoder (or fallback) relevance, ``None`` when not run.
    rerank_score: Optional[float] = None
    #: What the caller sorts on. Set by the last stage that ran.
    final_score: float = 0.0

    # -- provenance ---------------------------------------------------
    sources: Set[str] = field(default_factory=set)
    #: Named boosts applied during ranking, for explainability.
    boosts: Dict[str, float] = field(default_factory=dict)
    #: chunk_ids of near-duplicates collapsed into this candidate. The
    #: passages are dropped, the citations are not.
    duplicates: List[str] = field(default_factory=list)
    #: 1-based final position.
    rank: int = 0

    #: The user named this exact clause. A pinned candidate sorts above
    #: unpinned ones at every later stage, including after reranking —
    #: see :func:`app.retrieval.rank.rank_candidates` for why a relevance
    #: model does not get to overrule an explicit citation.
    pinned: bool = False

    # -- derived ------------------------------------------------------

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def matched_both(self) -> bool:
        """Found by vectors *and* by keywords — the strongest cheap signal."""
        return SOURCE_SEMANTIC in self.sources and SOURCE_KEYWORD in self.sources

    @classmethod
    def from_hit(cls, hit: SearchHit) -> "Candidate":
        return cls(
            chunk=hit.chunk,
            semantic_score=float(hit.score),
            semantic_rank=hit.rank or None,
            sources={SOURCE_SEMANTIC},
        )

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "Candidate":
        return cls(chunk=chunk)

    def merge(self, other: "Candidate") -> None:
        """Absorb another view of the same chunk.

        Called when two retrievers return the same passage. Scores are
        taken per-signal rather than combined, because each signal has
        its own scale; combining happens in :mod:`app.retrieval.fusion`.
        """
        if other.chunk_id != self.chunk_id:  # pragma: no cover - guarded
            raise RetrievalError(
                f"Cannot merge candidate {other.chunk_id} into {self.chunk_id}"
            )
        self.sources |= other.sources
        if other.semantic_rank is not None and self.semantic_rank is None:
            self.semantic_rank = other.semantic_rank
            self.semantic_score = other.semantic_score
        if other.keyword_rank is not None and self.keyword_rank is None:
            self.keyword_rank = other.keyword_rank
            self.keyword_score = other.keyword_score
            self.keyword_score_raw = other.keyword_score_raw

    # -- serialisation -------------------------------------------------

    def citation(self) -> Dict[str, Any]:
        return self.chunk.citation()

    def scores(self) -> Dict[str, Any]:
        """Every signal, for the ``explain`` view and for debugging."""
        return {
            "semantic": round(self.semantic_score, 6),
            "keyword": round(self.keyword_score, 6),
            "keyword_raw": round(self.keyword_score_raw, 6),
            "fused": round(self.fused_score, 6),
            "ranked": round(self.rank_score, 6),
            "rerank": (
                None if self.rerank_score is None else round(self.rerank_score, 6)
            ),
            "final": round(self.final_score, 6),
            "semantic_rank": self.semantic_rank,
            "keyword_rank": self.keyword_rank,
            "sources": sorted(self.sources),
            "boosts": {k: round(v, 6) for k, v in self.boosts.items()},
            "duplicates_removed": len(self.duplicates),
            "pinned": self.pinned,
        }

    def to_dict(self, include_text: bool = True, explain: bool = False):
        data = self.chunk.to_dict(include_text=include_text)
        data["rank"] = self.rank
        data["score"] = round(float(self.final_score), 6)
        if explain:
            data["scores"] = self.scores()
        return data


@dataclass
class StageCounts:
    """How many candidates each stage produced. Pure observability, but
    the cheapest way to answer "why did I only get two results?"."""

    semantic: int = 0
    keyword: int = 0
    fused: int = 0
    after_dedupe: int = 0
    reranked: int = 0
    returned: int = 0
    duplicates_removed: int = 0
    corpus_size: int = 0

    def to_dict(self) -> Dict[str, int]:
        from dataclasses import asdict

        return asdict(self)


class Retriever(ABC):
    """One retrieval signal.

    Keeping semantic and keyword retrieval behind the same tiny
    interface is what lets the pipeline treat them uniformly — and what
    would let a third signal (a title index, a clause-number lookup) be
    added without touching fusion or reranking.
    """

    name: str = "retriever"

    @abstractmethod
    def retrieve(
        self, query: Any, filters: SearchFilter, limit: int
    ) -> List[Candidate]:
        """Return up to ``limit`` candidates, best first, rank assigned."""


def index_candidates(candidates: Sequence[Candidate]) -> Dict[str, Candidate]:
    """chunk_id -> candidate, merging repeats."""
    out: Dict[str, Candidate] = {}
    for candidate in candidates:
        existing = out.get(candidate.chunk_id)
        if existing is None:
            out[candidate.chunk_id] = candidate
        else:
            existing.merge(candidate)
    return out


def scale_to_max(values: Sequence[float]) -> List[float]:
    """Scale a non-negative score run so the best is 1.0.

    Every signal in this pipeline lives on its own scale — cosine
    similarity is bounded, BM25 is unbounded and corpus-relative, RRF
    sums to a fraction of 1/k, a cross-encoder emits probabilities — so
    they must be brought to a common range before they can be weighed
    against one another or added to a boost.

    **Dividing by the maximum, not min-maxing between the minimum and the
    maximum.** Min-max normalisation always stretches the run to fill
    [0, 1] no matter how close the raw values were: two candidates whose
    RRF scores differ in the fourth decimal come out 1.0 apart, which
    destroys the very information the score carried and makes every
    downstream adjustment — a clause-reference boost, the retrieval half
    of the rerank blend — either irrelevant or decisive by accident.
    Scaling by the maximum preserves the *ratios*: near-ties stay near
    ties, and a genuine gap stays a gap.

    Values are assumed non-negative, which holds for every signal here
    (the IDF cannot go negative, RRF terms are positive, probabilities
    and similarities are bounded below by zero). A run that is entirely
    zero — or empty — stays zero rather than inventing an ordering.
    """
    if not values:
        return []
    high = max(values)
    if high <= 1e-12:
        return [0.0 for _ in values]
    return [max(0.0, float(v)) / high for v in values]


__all__ = [
    "Candidate",
    "Retriever",
    "StageCounts",
    "RetrievalError",
    "RerankerUnavailableError",
    "SOURCE_SEMANTIC",
    "SOURCE_KEYWORD",
    "index_candidates",
    "scale_to_max",
]
