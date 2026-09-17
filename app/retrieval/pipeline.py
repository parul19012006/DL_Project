"""Retrieval orchestration — the stage that owns the order of the others.

    query
      -> preprocess            (app.retrieval.query)
      -> semantic retrieval    (app.retrieval.semantic)     ~18 candidates
      -> keyword retrieval     (app.retrieval.keyword)      ~18 candidates
      -> hybrid merging        (app.retrieval.fusion)
      -> candidate ranking     (app.retrieval.rank)         -> pool of ~18
      -> duplicate removal     (app.retrieval.dedup)
      -> reranking             (app.retrieval.rerank)       -> final ~6
      -> relevant chunks

This module contains no retrieval logic of its own, on purpose. Every
stage above is independently constructible and independently testable,
and swapping one — a different fusion rule, a different reranker, a
third retrieval signal — is a change in one file plus a line here.

Three invariants hold for every path through this function:

**Tenant scoping is structural.** Both retrievers receive the same
:class:`SearchFilter`, which cannot be constructed without a tenant, and
a final assertion drops anything that somehow carries a foreign
``tenant_id``. Two independent barriers, because a cross-tenant leak in a
legal product is the failure that ends the product.

**Metadata survives end to end.** Every stage moves
:class:`~app.models.chunk.Chunk` objects, never bare strings, so
``document_id``, ``page_number``, ``section`` and ``chunk_id`` are
present at the output by construction rather than by being copied
carefully at each step.

**Every stage is measured.** ``RetrievalOutcome.timings_ms`` reports each
stage separately. A pipeline where you cannot see which stage is slow is
a pipeline nobody will tune.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import Settings, get_settings
from app.embeddings.base import EmbeddingModel
from app.logging_config import get_logger
from app.retrieval import fusion as fusion_module
from app.retrieval.base import Candidate, RetrievalError, StageCounts
from app.retrieval.dedup import remove_duplicates
from app.retrieval.keyword import KeywordRetriever
from app.retrieval.query import ProcessedQuery, preprocess_query
from app.retrieval.rank import assign_ranks, rank_candidates
from app.retrieval.rerank import Reranker, apply_reranking, get_reranker
from app.retrieval.semantic import SemanticRetriever
from app.vectorstore.base import SearchFilter, VectorStore

logger = get_logger(__name__)

HYBRID = "hybrid"
SEMANTIC_ONLY = "semantic"
KEYWORD_ONLY = "keyword"
MODES = (HYBRID, SEMANTIC_ONLY, KEYWORD_ONLY)


@dataclass
class RetrievalOutcome:
    """Everything one retrieval produced, and how it got there."""

    results: List[Candidate] = field(default_factory=list)
    query: Optional[ProcessedQuery] = None
    filters: Dict[str, Any] = field(default_factory=dict)

    mode: str = HYBRID
    fusion_method: str = fusion_module.RRF
    top_k: int = 0
    candidate_pool: int = 0
    min_score: float = 0.0

    counts: StageCounts = field(default_factory=StageCounts)
    agreement: float = 0.0

    embedding_model: str = ""
    embedding_dimension: int = 0
    reranker: str = ""
    reranked: bool = False
    #: False when the lexical fallback ran instead of a real
    #: cross-encoder — never allowed to be invisible.
    rerank_is_cross_encoder: bool = False

    warnings: List[str] = field(default_factory=list)
    duration_ms: int = 0
    timings_ms: Dict[str, int] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.results

    def to_dict(self, include_text: bool = True, explain: bool = False):
        return {
            "query": self.query.to_dict() if self.query else {},
            "results": [
                c.to_dict(include_text=include_text, explain=explain)
                for c in self.results
            ],
            "filters": self.filters,
            "mode": self.mode,
            "top_k": self.top_k,
            "candidate_pool": self.candidate_pool,
            "counts": self.counts.to_dict(),
            "warnings": list(self.warnings),
            "duration_ms": self.duration_ms,
            "timings_ms": dict(self.timings_ms),
        }


class RetrievalPipeline:
    """Runs the stages in order and records what each one did."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model: Optional[EmbeddingModel] = None,
        store: Optional[VectorStore] = None,
        reranker: Optional[Reranker] = None,
        semantic: Optional[SemanticRetriever] = None,
        keyword: Optional[KeywordRetriever] = None,
    ) -> None:
        from app.embeddings.factory import get_embedding_model
        from app.vectorstore.factory import get_vector_store

        self.settings = settings or get_settings()
        self.model = model or get_embedding_model(self.settings)
        self.store = store or get_vector_store(self.settings)
        self.semantic = semantic or SemanticRetriever(self.model, self.store)
        self.keyword = keyword or KeywordRetriever(self.store, self.settings)
        #: Resolved lazily so constructing the pipeline never loads a model.
        self._reranker = reranker

    # -- reranker ------------------------------------------------------

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = get_reranker(self.settings)
        return self._reranker

    # -- main ----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        tenant_id: str,
        document_ids: Optional[List[str]] = None,
        document_type: Optional[str] = None,
        section: Optional[str] = None,
        page_range: Optional[tuple] = None,
        equals: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        candidate_pool: Optional[int] = None,
        mode: Optional[str] = None,
        rerank: Optional[bool] = None,
        min_score: Optional[float] = None,
        dedupe: Optional[bool] = None,
    ) -> RetrievalOutcome:
        settings = self.settings
        started = time.perf_counter()
        timings: Dict[str, int] = {}

        chosen_mode = (mode or settings.retrieval_mode).strip().lower()
        if chosen_mode not in MODES:
            raise RetrievalError(
                f"Unknown retrieval mode '{chosen_mode}'. "
                f"Available: {', '.join(MODES)}"
            )

        final_k = int(top_k or settings.retrieval_top_k)
        pool = int(candidate_pool or settings.retrieval_candidates)
        # The pool must be at least the final size, or reranking would be
        # choosing from fewer passages than it is asked to return.
        pool = max(pool, final_k)
        floor = (
            settings.retrieval_min_score if min_score is None else float(min_score)
        )
        do_rerank = settings.rerank_enabled if rerank is None else bool(rerank)
        do_dedupe = settings.dedupe_enabled if dedupe is None else bool(dedupe)

        outcome = RetrievalOutcome(
            mode=chosen_mode,
            fusion_method=settings.fusion_method,
            top_k=final_k,
            candidate_pool=pool,
            min_score=floor,
            embedding_model=self.model.model_id,
            embedding_dimension=self.model.dimension,
        )

        # -- 1. query preprocessing ------------------------------------
        mark = time.perf_counter()
        processed = preprocess_query(query, settings)
        timings["preprocess"] = int((time.perf_counter() - mark) * 1000)
        outcome.query = processed

        # A filter is built even for an empty query so the response
        # echoes back what was asked, and so tenant validation happens
        # on every path rather than only the ones that retrieve.
        filters = SearchFilter(
            tenant_id=tenant_id,
            document_ids=document_ids,
            document_type=document_type,
            section=section,
            page_range=page_range,
            equals=equals or {},
        )
        outcome.filters = filters.describe()

        if processed.is_empty:
            outcome.warnings.append("The query was empty after normalisation")
            outcome.duration_ms = int((time.perf_counter() - started) * 1000)
            outcome.timings_ms = timings
            return outcome

        # -- 2. semantic retrieval -------------------------------------
        semantic_hits: List[Candidate] = []
        if chosen_mode in (HYBRID, SEMANTIC_ONLY):
            mark = time.perf_counter()
            semantic_hits = self.semantic.retrieve(processed, filters, pool)
            timings["semantic"] = int((time.perf_counter() - mark) * 1000)
            timings["embed"] = self.semantic.last_embed_ms
        outcome.counts.semantic = len(semantic_hits)

        # -- 3. keyword retrieval --------------------------------------
        keyword_hits: List[Candidate] = []
        if chosen_mode in (HYBRID, KEYWORD_ONLY):
            mark = time.perf_counter()
            keyword_hits = self.keyword.retrieve(processed, filters, pool)
            timings["keyword"] = int((time.perf_counter() - mark) * 1000)
            outcome.counts.corpus_size = self.keyword.last_corpus_size
            if self.keyword.last_corpus_truncated:
                outcome.warnings.append(
                    "The keyword corpus hit KEYWORD_CORPUS_LIMIT; BM25 scored "
                    "only part of this tenant's chunks"
                )
            if not processed.has_terms:
                outcome.warnings.append(
                    "The query had no content terms; keyword retrieval was "
                    "skipped and results are semantic only"
                )
        outcome.counts.keyword = len(keyword_hits)

        # -- 4. hybrid merging -----------------------------------------
        mark = time.perf_counter()
        merged = fusion_module.fuse(
            semantic_hits,
            keyword_hits,
            method=settings.fusion_method,
            rrf_k=settings.rrf_k,
            semantic_weight=settings.semantic_weight,
            keyword_weight=settings.keyword_weight,
        )
        timings["fusion"] = int((time.perf_counter() - mark) * 1000)
        outcome.counts.fused = len(merged)
        outcome.agreement = fusion_module.agreement_ratio(merged)

        if not merged:
            outcome.duration_ms = int((time.perf_counter() - started) * 1000)
            outcome.timings_ms = timings
            logger.info(
                "Retrieval returned nothing for tenant %s (corpus=%d)",
                filters.tenant_id,
                outcome.counts.corpus_size,
            )
            return outcome

        # -- 5. candidate ranking --------------------------------------
        mark = time.perf_counter()
        ranked = rank_candidates(
            merged,
            processed,
            reference_boost=settings.reference_boost,
            phrase_boost=settings.phrase_boost,
            agreement_boost=settings.agreement_boost,
            heading_boost=settings.heading_boost,
            limit=pool,
        )
        timings["rank"] = int((time.perf_counter() - mark) * 1000)

        # -- 6. duplicate removal --------------------------------------
        # After ranking, so the copy retrieval preferred is the survivor;
        # before reranking, so the reranker is not spent on repeats.
        if do_dedupe:
            mark = time.perf_counter()
            ranked, removed = remove_duplicates(
                ranked,
                threshold=settings.dedupe_threshold,
                shingle_size=settings.dedupe_shingle_size,
                across_documents=settings.dedupe_across_documents,
            )
            timings["dedupe"] = int((time.perf_counter() - mark) * 1000)
            outcome.counts.duplicates_removed = removed
            ranked = assign_ranks(ranked)
        outcome.counts.after_dedupe = len(ranked)

        # -- 7. reranking ----------------------------------------------
        if do_rerank:
            reranker = self.reranker
            outcome.reranker = reranker.describe()
            outcome.rerank_is_cross_encoder = reranker.is_cross_encoder
            mark = time.perf_counter()
            try:
                final = apply_reranking(
                    ranked,
                    processed,
                    reranker,
                    top_k=final_k,
                    weight=settings.rerank_weight,
                )
                outcome.reranked = True
            except Exception as exc:
                # A reranker failing mid-request must not lose the
                # retrieval that already succeeded. Degrade to the
                # retrieval ordering and say so.
                logger.error("Reranking failed (%s); returning retrieval order", exc)
                outcome.warnings.append(
                    f"Reranking failed ({type(exc).__name__}); results are in "
                    "retrieval order"
                )
                final = assign_ranks(ranked[:final_k])
            timings["rerank"] = int((time.perf_counter() - mark) * 1000)
            outcome.counts.reranked = len(final)
        else:
            final = assign_ranks(ranked[:final_k])
            outcome.reranker = "disabled"

        # -- 8. score floor --------------------------------------------
        if floor > 0:
            final = [c for c in final if c.final_score >= floor]
            final = assign_ranks(final)

        # -- 9. tenant guard -------------------------------------------
        # The store filters and SearchFilter enforces scoping; this is the
        # last line of defence and it is cheap.
        foreign = [c for c in final if c.chunk.tenant_id != filters.tenant_id]
        if foreign:  # pragma: no cover - unreachable via the stores
            logger.error(
                "Dropping %d retrieved chunk(s) belonging to another tenant",
                len(foreign),
            )
            final = assign_ranks(
                [c for c in final if c.chunk.tenant_id == filters.tenant_id]
            )

        outcome.results = final
        outcome.counts.returned = len(final)
        outcome.duration_ms = int((time.perf_counter() - started) * 1000)
        outcome.timings_ms = timings

        logger.info(
            "Retrieval: tenant=%s mode=%s semantic=%d keyword=%d fused=%d "
            "dupes=%d returned=%d rerank=%s (%d ms)",
            filters.tenant_id,
            chosen_mode,
            outcome.counts.semantic,
            outcome.counts.keyword,
            outcome.counts.fused,
            outcome.counts.duplicates_removed,
            outcome.counts.returned,
            outcome.reranker,
            outcome.duration_ms,
        )
        return outcome


# =====================================================================
# Module-level convenience
# =====================================================================

_pipeline: Optional[RetrievalPipeline] = None


def build_pipeline(settings: Optional[Settings] = None) -> RetrievalPipeline:
    return RetrievalPipeline(settings=settings)


def get_pipeline(settings: Optional[Settings] = None) -> RetrievalPipeline:
    """Process-wide pipeline.

    Rebuilt when the caller passes a *different* Settings object — an app
    created by :func:`app.main.create_app` with custom settings must not
    be served by a pipeline wired to the global ones. Rebuilding is
    cheap: the expensive parts (embedding model, vector store, reranker)
    are their own singletons and are not reloaded.
    """
    global _pipeline
    if _pipeline is None or (settings is not None and _pipeline.settings is not settings):
        _pipeline = build_pipeline(settings)
    return _pipeline


def set_pipeline(pipeline: Optional[RetrievalPipeline]) -> None:
    global _pipeline
    _pipeline = pipeline


def retrieve(
    query: str,
    tenant_id: str,
    settings: Optional[Settings] = None,
    **kwargs,
) -> RetrievalOutcome:
    """One-call retrieval, mirroring :func:`app.services.indexing.search`."""
    pipeline = get_pipeline(settings)
    return pipeline.retrieve(query=query, tenant_id=tenant_id, **kwargs)


__all__ = [
    "RetrievalPipeline",
    "RetrievalOutcome",
    "build_pipeline",
    "get_pipeline",
    "set_pipeline",
    "retrieve",
    "HYBRID",
    "SEMANTIC_ONLY",
    "KEYWORD_ONLY",
    "MODES",
]
