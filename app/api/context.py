"""Context endpoints (Stage 6).

``POST /context`` is what a grounded-answer feature calls: it retrieves,
builds the evidence block, and hands back the block together with the
citations and an account of everything that was left out.

It returns **evidence, not an answer.** The LLM call, the prompt template
and the instructions that go with it are the next stage's, and are
deliberately not here — writing a prompt into this endpoint would fix it
for every future caller.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from app.api.deps import get_config, verify_service_token
from app.config import Settings
from app.context.base import BuiltContext
from app.context.builder import get_context_builder
from app.logging_config import get_logger
from app.models.schemas import (
    ContextConfigResponse,
    ContextRequest,
    ContextResponse,
    ContextSourceModel,
    ContextStatsModel,
    OmittedSourceModel,
    RetrievalCountsModel,
)
from app.retrieval.pipeline import RetrievalOutcome, get_pipeline

logger = get_logger(__name__)

router = APIRouter(tags=["context"], dependencies=[Depends(verify_service_token)])


def _to_response(
    context: BuiltContext,
    outcome: RetrievalOutcome,
    request: ContextRequest,
    duration_ms: int,
    timings: dict,
) -> ContextResponse:
    return ContextResponse(
        context=context.text if request.include_context_text else "",
        query=request.query,
        tenant_id=request.tenant_id,
        sources=[
            ContextSourceModel(**source.to_dict(include_text=request.include_text))
            for source in context.sources
        ],
        citations=context.citations(),
        omitted=[OmittedSourceModel(**o.to_dict()) for o in context.omitted],
        stats=ContextStatsModel(**context.stats.to_dict()),
        retrieval=RetrievalCountsModel(**outcome.counts.to_dict()),
        order=context.order,
        tokenizer=context.tokenizer,
        sufficient=context.sufficient,
        warnings=list(outcome.warnings) + list(context.warnings),
        reranker=outcome.reranker,
        rerank_is_cross_encoder=outcome.rerank_is_cross_encoder,
        duration_ms=duration_ms,
        timings_ms=timings,
    )


@router.post(
    "/context",
    response_model=ContextResponse,
    summary="Retrieve and build an LLM-ready context block",
    description=(
        "Runs hybrid retrieval, then selects the strongest evidence, "
        "removes redundancy, fits a token budget and renders uniform, "
        "numbered `Source N:` blocks with full provenance.\n\n"
        "Returns **evidence, not an answer** — grounded generation is the "
        "next stage. Every passage that was left out is reported in "
        "`omitted` with its citation and the reason, so nothing is "
        "discarded silently."
    ),
)
async def build_context(
    request: ContextRequest, settings: Settings = Depends(get_config)
) -> ContextResponse:
    started = time.perf_counter()

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

    mark = time.perf_counter()
    context = get_context_builder(settings).build(
        outcome.results,
        query=request.query,
        max_tokens=request.context_max_tokens,
        max_sources=request.context_max_sources,
        min_sources=request.context_min_sources,
        min_score=request.context_min_score,
        max_source_tokens=request.context_max_source_tokens,
        order=request.context_order,
        dedupe=request.context_dedupe,
        truncate=request.context_truncate,
    )
    context_ms = int((time.perf_counter() - mark) * 1000)

    timings = dict(outcome.timings_ms)
    timings["retrieval"] = outcome.duration_ms
    timings["context"] = context_ms
    duration_ms = int((time.perf_counter() - started) * 1000)

    return _to_response(context, outcome, request, duration_ms, timings)


@router.get(
    "/context/config",
    response_model=ContextConfigResponse,
    summary="The live context-builder configuration",
)
async def context_config(
    settings: Settings = Depends(get_config),
) -> ContextConfigResponse:
    builder = get_context_builder(settings)
    return ContextConfigResponse(
        max_tokens=settings.context_max_tokens,
        max_sources=settings.context_max_sources,
        min_sources=settings.context_min_sources,
        min_score=settings.context_min_score,
        max_source_tokens=settings.context_max_source_tokens,
        min_source_tokens=settings.context_min_source_tokens,
        truncate_long_sources=settings.context_truncate_long_sources,
        order=settings.context_order,
        dedupe_enabled=settings.context_dedupe_enabled,
        redundancy_threshold=settings.context_redundancy_threshold,
        strip_repeated_heading=settings.context_strip_repeated_heading,
        tokenizer=getattr(builder.counter, "name", "unknown"),
    )


__all__ = ["router"]
