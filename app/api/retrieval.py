"""Retrieval endpoints (Stage 5).

``POST /retrieve`` is the endpoint the MERN backend should call for
"find the relevant passages". ``POST /search`` (Stage 4) stays exactly as
it was — a single vector query, useful for debugging retrieval quality
and for callers that want raw similarity — but it is not the one to build
a chat feature on.

Both return passages. **Neither returns an answer.** Grounded generation
with citations is Stage 6, and nothing here should be described to a user
as an answer produced by the service.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_config, verify_service_token
from app.config import Settings
from app.logging_config import get_logger
from app.models.schemas import (
    CandidateScoresModel,
    RetrievalConfigResponse,
    RetrievalCountsModel,
    RetrievedChunkModel,
    RetrieveRequest,
    RetrieveResponse,
)
from app.retrieval.base import Candidate
from app.retrieval.pipeline import RetrievalOutcome, get_pipeline

logger = get_logger(__name__)

router = APIRouter(tags=["retrieval"], dependencies=[Depends(verify_service_token)])


def _to_model(
    candidate: Candidate, include_text: bool, explain: bool
) -> RetrievedChunkModel:
    chunk = candidate.chunk
    return RetrievedChunkModel(
        rank=candidate.rank,
        score=round(candidate.final_score, 6),
        chunk_id=chunk.chunk_id,
        chunk_index=chunk.chunk_index,
        document_id=chunk.document_id,
        filename=chunk.filename,
        page_number=chunk.page_number,
        page_end=chunk.page_end or chunk.page_number,
        page_range=chunk.page_range,
        section=chunk.section,
        document_type=chunk.document_type.value,
        text=chunk.text if include_text else "",
        token_count=chunk.token_count,
        ocr=chunk.ocr,
        duplicates=list(candidate.duplicates),
        scores=(
            CandidateScoresModel(**candidate.scores()) if explain else None
        ),
    )


def _to_response(
    outcome: RetrievalOutcome, request: RetrieveRequest
) -> RetrieveResponse:
    return RetrieveResponse(
        query=request.query,
        normalized_query=outcome.query.normalized if outcome.query else "",
        tenant_id=request.tenant_id,
        mode=outcome.mode,
        results=[
            _to_model(c, request.include_text, request.explain)
            for c in outcome.results
        ],
        counts=RetrievalCountsModel(**outcome.counts.to_dict()),
        filters=outcome.filters,
        top_k=outcome.top_k,
        candidate_pool=outcome.candidate_pool,
        min_score=outcome.min_score,
        fusion_method=outcome.fusion_method,
        agreement=outcome.agreement,
        references=(
            [f"{kind} {number}" for kind, number in outcome.query.references]
            if outcome.query
            else []
        ),
        embedding_model=outcome.embedding_model,
        embedding_dimension=outcome.embedding_dimension,
        reranker=outcome.reranker,
        reranked=outcome.reranked,
        rerank_is_cross_encoder=outcome.rerank_is_cross_encoder,
        warnings=list(outcome.warnings),
        duration_ms=outcome.duration_ms,
        timings_ms=dict(outcome.timings_ms),
    )


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Hybrid retrieval: semantic + BM25, fused, deduplicated, reranked",
    description=(
        "Runs the full Stage 5 pipeline and returns the passages most "
        "relevant to the query, with full provenance (document, page, "
        "section, chunk id) on every result.\n\n"
        "Returns **passages, not an answer** — grounded generation arrives "
        "in Stage 6. Always scoped to one tenant.\n\n"
        "Set `explain=true` to see every per-stage score and boost."
    ),
)
async def retrieve(
    request: RetrieveRequest, settings: Settings = Depends(get_config)
) -> RetrieveResponse:
    pipeline = get_pipeline(settings)
    outcome = pipeline.retrieve(
        query=request.query,
        tenant_id=request.tenant_id,
        document_ids=request.document_ids,
        document_type=request.document_type,
        section=request.section,
        page_range=tuple(request.page_range) if request.page_range else None,
        equals=request.metadata_equals,
        top_k=request.top_k,
        candidate_pool=request.candidate_pool,
        mode=request.mode,
        rerank=request.rerank,
        dedupe=request.dedupe,
        min_score=request.min_score,
    )
    return _to_response(outcome, request)


@router.get(
    "/retrieve/config",
    response_model=RetrievalConfigResponse,
    summary="The live retrieval configuration",
    description=(
        "What the pipeline is actually doing right now, including which "
        "reranker is in use — a deployment running the lexical fallback "
        "instead of the cross-encoder can be identified from here."
    ),
)
async def retrieval_config(
    settings: Settings = Depends(get_config),
) -> RetrievalConfigResponse:
    pipeline = get_pipeline(settings)
    reranker = pipeline.reranker
    return RetrievalConfigResponse(
        mode=settings.retrieval_mode,
        candidate_pool=settings.retrieval_candidates,
        top_k=settings.retrieval_top_k,
        fusion_method=settings.fusion_method,
        rrf_k=settings.rrf_k,
        semantic_weight=settings.semantic_weight,
        keyword_weight=settings.keyword_weight,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
        keyword_corpus_limit=settings.keyword_corpus_limit,
        dedupe_enabled=settings.dedupe_enabled,
        dedupe_threshold=settings.dedupe_threshold,
        rerank_enabled=settings.rerank_enabled,
        reranker_provider=settings.reranker_provider,
        reranker_model=settings.reranker_model,
        reranker_active=reranker.describe(),
        rerank_is_cross_encoder=reranker.is_cross_encoder,
        rerank_weight=settings.rerank_weight,
        min_score=settings.retrieval_min_score,
    )


__all__ = ["router"]
