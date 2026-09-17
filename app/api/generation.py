"""Answer endpoints (Stage 7).

``POST /answer`` is the end of the pipeline: a question in, a grounded
answer with validated citations out.

**Why a failed generation is HTTP 200.** Retrieval, context building and
citation validation are separate pieces of work from the model call, and
when the model call fails the rest of it succeeded. Returning 503 would
throw away passages the user can read right now and reduce the response
to an error envelope. So the response is 200 with ``status`` naming the
failure, ``answer: null``, and the sources intact — the backend renders
"we could not generate an answer, here are the relevant clauses". The
failure is logged at error level and ``/health`` reports the provider, so
nothing is hidden by the choice; only the user's view of it is better.

A malformed *request* is still a 4xx, and an unconfigured provider in
strict mode still fails at startup.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_config, verify_service_token
from app.config import Settings
from app.generation.factory import available_providers, get_llm_provider
from app.generation.service import AnswerResult, get_answer_service
from app.logging_config import get_logger
from app.models.schemas import (
    AnswerRequest,
    AnswerResponse,
    CitationModel,
    CitationValidationModel,
    ContextSourceModel,
    ContextStatsModel,
    GenerationConfigResponse,
    RejectedCitationModel,
    RetrievalCountsModel,
)

logger = get_logger(__name__)

router = APIRouter(tags=["generation"], dependencies=[Depends(verify_service_token)])


def _retrieval_options(request: AnswerRequest) -> dict:
    return {
        "document_ids": request.document_ids,
        "document_type": request.document_type,
        "section": request.section,
        "page_range": tuple(request.page_range) if request.page_range else None,
        "equals": request.metadata_equals,
        "top_k": request.top_k,
        "candidate_pool": request.candidate_pool,
        "mode": request.mode,
        "rerank": request.rerank,
        "dedupe": request.dedupe,
        "min_score": request.min_score,
    }


def _context_options(request: AnswerRequest) -> dict:
    return {
        "max_tokens": request.context_max_tokens,
        "max_sources": request.context_max_sources,
        "min_sources": request.context_min_sources,
        "min_score": request.context_min_score,
        "max_source_tokens": request.context_max_source_tokens,
        "order": request.context_order,
        "dedupe": request.context_dedupe,
        "truncate": request.context_truncate,
    }


def _to_response(result: AnswerResult, request: AnswerRequest) -> AnswerResponse:
    context = result.context
    report = result.citations

    return AnswerResponse(
        status=result.status,
        question=request.query,
        tenant_id=request.tenant_id,
        answer=result.answer,
        citations=[CitationModel(**c.to_dict()) for c in report.citations],
        interpretation=result.interpretation,
        insufficient_evidence=result.insufficient_evidence,
        grounded=report.grounded,
        validation=CitationValidationModel(
            grounded=report.grounded,
            valid_citations=len(report.citations),
            rejected=[RejectedCitationModel(**r.to_dict()) for r in report.rejected],
            verified_quotes=report.verified_quotes,
            unverified_quotes=list(report.unverified_quotes),
        ),
        sources=(
            [
                ContextSourceModel(
                    **source.to_dict(include_text=request.include_text)
                )
                for source in (context.sources if context else [])
            ]
            if request.include_sources
            else []
        ),
        context=(context.text if (request.include_context and context) else None),
        context_stats=(
            ContextStatsModel(**context.stats.to_dict())
            if context
            else ContextStatsModel()
        ),
        retrieval=(
            RetrievalCountsModel(**result.retrieval.counts.to_dict())
            if result.retrieval
            else RetrievalCountsModel()
        ),
        provider=result.provider,
        model=result.model,
        is_language_model=result.is_language_model,
        usage=dict(result.usage),
        attempts=result.attempts,
        error=result.error,
        raw_excerpt=result.raw_excerpt,
        warnings=list(result.warnings),
        duration_ms=result.duration_ms,
        timings_ms=dict(result.timings_ms),
    )


@router.post(
    "/answer",
    response_model=AnswerResponse,
    summary="Answer a question from the tenant's documents, with citations",
    description=(
        "Runs the whole pipeline: hybrid retrieval, reranking, context "
        "construction, generation, then citation validation.\n\n"
        "The answer is generated only from passages retrieved for this "
        "tenant, and **every citation is checked against those passages** — "
        "a citation to a source that was not retrieved is dropped and "
        "reported. `grounded: false` means the answer could not be fully "
        "checked against the evidence.\n\n"
        "Branch on `status`. A model failure returns 200 with "
        "`answer: null` and the retrieved sources intact, so a failed "
        "generation does not discard the retrieval."
    ),
)
async def answer(
    request: AnswerRequest, settings: Settings = Depends(get_config)
) -> AnswerResponse:
    result = get_answer_service(settings).answer(
        question=request.query,
        tenant_id=request.tenant_id,
        retrieval_options=_retrieval_options(request),
        context_options=_context_options(request),
        temperature=request.temperature,
        max_output_tokens=request.max_output_tokens,
    )
    return _to_response(result, request)


@router.get(
    "/answer/config",
    response_model=GenerationConfigResponse,
    summary="The live generation configuration",
    description=(
        "What the pipeline is actually doing, including whether a real "
        "language model is answering. Never returns a credential — only "
        "whether one is present."
    ),
)
async def generation_config(
    settings: Settings = Depends(get_config),
) -> GenerationConfigResponse:
    provider = get_llm_provider(settings)

    def _has(value) -> bool:
        getter = getattr(value, "get_secret_value", None)
        return bool((getter() if callable(getter) else str(value or "")).strip())

    configured = {
        "openai": _has(settings.openai_api_key),
        "anthropic": _has(settings.anthropic_api_key),
    }.get((settings.llm_provider or "").strip().lower(), True)

    return GenerationConfigResponse(
        provider=settings.llm_provider,
        provider_active=provider.describe(),
        model=provider.model,
        is_language_model=provider.is_language_model,
        available_providers=available_providers(),
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        strict=settings.llm_strict,
        require_citations=settings.generation_require_citations,
        harvest_inline_citations=settings.generation_harvest_inline_citations,
        verify_quotes=settings.generation_verify_quotes,
        api_key_configured=configured,
    )


__all__ = ["router"]
