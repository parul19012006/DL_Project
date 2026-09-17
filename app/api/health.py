"""Health endpoints.

Three deliberately distinct probes:

``GET /health``        full status — what a human or a dashboard reads
``GET /health/live``   liveness — is the process up? (container restarts)
``GET /health/ready``  readiness — should traffic be routed here?

None of them require the service token: an orchestrator has to be able to
probe the service without credentials.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from app.api.deps import get_config, get_uptime_seconds
from app.config import Settings
from app.logging_config import get_logger
from app.models.schemas import (
    ComponentStatus,
    HealthResponse,
    HealthStatus,
    LivenessResponse,
    ReadinessResponse,
)

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


def _check_components(settings: Settings) -> list[ComponentStatus]:
    """Collect per-component health.

    Stage 1 has one component. Later stages append the vector store, the
    embedding model and the LLM provider to this list — the response
    shape the backend parses does not change.
    """
    components = [
        ComponentStatus(
            name="configuration",
            status=HealthStatus.OK,
            detail=f"loaded ({settings.environment})",
        )
    ]

    # A production deployment with no shared secret is reachable but not
    # safely configured, so it reports degraded rather than ok.
    if settings.is_production and not settings.auth_enabled:
        components.append(
            ComponentStatus(
                name="service_token",
                status=HealthStatus.DEGRADED,
                detail="SERVICE_TOKEN is not set in a production environment",
            )
        )

    components.extend(_ingestion_components(settings))
    components.append(_chunking_component(settings))
    components.extend(_index_components(settings))
    components.extend(_retrieval_components(settings))
    return components


def _chunking_component(settings: Settings) -> ComponentStatus:
    """Stage 3: report the live chunking configuration.

    An overlap at or above half the chunk size is reported as degraded:
    the chunker clamps it, but the operator should know their setting is
    not being honoured.
    """
    from app.chunking.tokenizer import get_token_counter

    counter = get_token_counter(settings)
    clamped = settings.chunk_overlap > settings.chunk_size // 2
    return ComponentStatus(
        name="chunking",
        status=HealthStatus.DEGRADED if clamped else HealthStatus.OK,
        detail=(
            f"size={settings.chunk_size} overlap={settings.chunk_overlap} "
            f"min={settings.min_chunk_tokens} tokenizer={counter.name}"
            + (" (overlap clamped to half the chunk size)" if clamped else "")
        ),
    )


def _index_components(settings: Settings) -> list[ComponentStatus]:
    """Stage 4: the embedding model and the vector store.

    The embedding model is reported as degraded when the deterministic
    encoder is in use but the real provider was requested — that means
    the model failed to load and retrieval is running on a lexical
    hashing encoder, which must never be invisible.

    The vector store is reported as *unavailable* when it cannot be
    opened: without it, search cannot answer at all, so the service is
    genuinely not ready for traffic.
    """
    from app.embeddings.factory import get_embedding_model
    from app.vectorstore.factory import get_vector_store

    out: list[ComponentStatus] = []

    try:
        model = get_embedding_model(settings)
        fell_back = (
            settings.embedding_provider == "sentence_transformers"
            and model.name == "deterministic"
        )
        out.append(
            ComponentStatus(
                name="embedding_model",
                status=HealthStatus.DEGRADED if fell_back else HealthStatus.OK,
                detail=(
                    f"'{settings.embedding_model}' could not be loaded; "
                    f"running on the deterministic encoder "
                    f"(dimension={model.dimension}) — retrieval quality is "
                    "poor and results are not semantic"
                    if fell_back
                    else f"{model.name}:{model.model_id} dimension="
                    f"{model.dimension}"
                ),
            )
        )
    except Exception as exc:
        logger.error("Embedding model health check failed: %s", exc)
        out.append(
            ComponentStatus(
                name="embedding_model",
                status=HealthStatus.UNAVAILABLE,
                detail="the embedding model could not be loaded",
            )
        )
        return out

    try:
        store = get_vector_store(settings)
        store.ensure_ready()
        stats = store.stats()
        transient = store.name == "memory"
        out.append(
            ComponentStatus(
                name="vector_store",
                status=HealthStatus.DEGRADED if transient else HealthStatus.OK,
                detail=(
                    f"{stats.backend}:{stats.collection} "
                    f"dimension={stats.dimension} metric={stats.metric} "
                    f"chunks={stats.total_chunks}"
                    + (" (in-memory: nothing is persisted)" if transient else "")
                ),
            )
        )
    except Exception as exc:
        logger.error("Vector store health check failed: %s", exc)
        out.append(
            ComponentStatus(
                name="vector_store",
                status=HealthStatus.UNAVAILABLE,
                detail=f"the vector store is not usable: {type(exc).__name__}",
            )
        )

    return out


def _retrieval_components(settings: Settings) -> list[ComponentStatus]:
    """Stage 5: the retrieval pipeline and the reranker.

    The reranker gets its own component for one reason: when the
    cross-encoder cannot be loaded the service still answers, using a
    lexical fallback that is measurably worse at ordering. That
    substitution must be visible somewhere an operator looks, or a
    deployment quietly serves second-rate retrieval forever.
    """
    from app.retrieval.rerank import CROSS_ENCODER, get_reranker

    out: list[ComponentStatus] = [
        ComponentStatus(
            name="retrieval",
            status=HealthStatus.OK,
            detail=(
                f"mode={settings.retrieval_mode} "
                f"fusion={settings.fusion_method} "
                f"candidates={settings.retrieval_candidates} "
                f"top_k={settings.retrieval_top_k} "
                f"dedupe={'on' if settings.dedupe_enabled else 'off'}"
            ),
        )
    ]

    if not settings.rerank_enabled:
        out.append(
            ComponentStatus(
                name="reranker",
                status=HealthStatus.OK,
                detail=(
                    "disabled by configuration; results are in fused "
                    "retrieval order"
                ),
            )
        )
        return out

    try:
        reranker = get_reranker(settings)
    except Exception as exc:
        logger.error("Reranker health check failed: %s", exc)
        out.append(
            ComponentStatus(
                name="reranker",
                status=HealthStatus.UNAVAILABLE,
                detail=(
                    f"RERANK_STRICT is set and '{settings.reranker_model}' "
                    "could not be loaded"
                ),
            )
        )
        return out

    fell_back = (
        settings.reranker_provider == CROSS_ENCODER
        and not reranker.is_cross_encoder
    )
    out.append(
        ComponentStatus(
            name="reranker",
            status=HealthStatus.DEGRADED if fell_back else HealthStatus.OK,
            detail=(
                f"'{settings.reranker_model}' could not be loaded; running "
                "the lexical fallback — retrieval still works but ordering "
                "quality is lower. Set RERANK_STRICT=true to make this fatal"
                if fell_back
                else reranker.describe()
            ),
        )
    )
    out.append(_context_component(settings))
    out.append(_llm_component(settings))
    return out


def _llm_component(settings: Settings) -> ComponentStatus:
    """Stage 7: which provider is answering, and whether it is a model.

    Degraded when a real provider was configured and the deterministic
    extractive one is answering instead — that substitution changes what
    the service *is*, from a system that writes answers to one that
    selects sentences, and it must never be invisible.

    The check is configuration-only: it does not call the vendor. A
    health endpoint that bills per probe is one nobody leaves enabled.
    """
    from app.generation.factory import EXTRACTIVE, get_llm_provider

    configured = (settings.llm_provider or EXTRACTIVE).strip().lower()

    try:
        provider = get_llm_provider(settings)
    except Exception as exc:
        logger.error("LLM provider health check failed: %s", exc)
        return ComponentStatus(
            name="llm",
            status=HealthStatus.UNAVAILABLE,
            detail=(
                f"LLM_STRICT is set and provider '{configured}' could not be "
                "built. Check the provider name and its API key environment "
                "variable."
            ),
        )

    fell_back = configured != EXTRACTIVE and not provider.is_language_model
    if fell_back:
        return ComponentStatus(
            name="llm",
            status=HealthStatus.DEGRADED,
            detail=(
                f"provider '{configured}' could not be built; answering with "
                "the deterministic extractive provider, which selects "
                "sentences from the evidence and is NOT a language model. "
                "Set LLM_STRICT=true to make this fatal"
            ),
        )

    return ComponentStatus(
        name="llm",
        status=HealthStatus.OK,
        detail=(
            f"{provider.describe()} temperature={settings.llm_temperature} "
            f"max_output={settings.llm_max_output_tokens} "
            f"citations={'required' if settings.generation_require_citations else 'optional'}"
            + (
                ""
                if provider.is_language_model
                else " (no language model is configured: answers are selected "
                "sentences, not generated prose)"
            )
        ),
    )


def _context_component(settings: Settings) -> ComponentStatus:
    """Stage 6: the context budget and which token counter prices it.

    Reported as **ok** even on the heuristic counter, because that
    counter is the configured default and the service is working as
    specified — degraded is reserved for something not behaving as
    configured, and a status that is always degraded teaches operators to
    ignore it.

    The approximation is still stated, because it matters more here than
    it did at chunking: a chunk budgeted 15% wrong is a slightly odd
    chunk, while a *context* budgeted 15% wrong is a prompt the provider
    truncates from one end — usually silently, usually taking the last
    source with it. The remedy is in the detail line.
    """
    from app.chunking.tokenizer import get_token_counter

    counter = get_token_counter(settings)
    approximate = counter.name == "heuristic"
    return ComponentStatus(
        name="context",
        status=HealthStatus.OK,
        detail=(
            f"budget={settings.context_max_tokens} tokens "
            f"max_sources={settings.context_max_sources} "
            f"order={settings.context_order} tokenizer={counter.name}"
            + (
                " (counts are estimates — keep the budget clear of the "
                "model's real window, or set TOKENIZER=tiktoken for exact "
                "counts)"
                if approximate
                else ""
            )
        ),
    )


def _ingestion_components(settings: Settings) -> list[ComponentStatus]:
    """Stage 2 components: the parsers and OCR."""
    from app.ingestion import ocr as ocr_module
    from app.ingestion.base import registered_extensions

    out: list[ComponentStatus] = []

    extensions = registered_extensions()
    missing = [e for e in settings.allowed_extension_list if e not in extensions]
    out.append(
        ComponentStatus(
            name="ingestion",
            status=HealthStatus.DEGRADED if missing else HealthStatus.OK,
            detail=(
                f"no parser registered for {', '.join(missing)}"
                if missing
                else f"parsers: {', '.join(extensions)}"
            ),
        )
    )

    # OCR is an enhancement, not a requirement: text PDFs and DOCX files
    # ingest fine without it, so its absence is degraded, not
    # unavailable — and it is reported so the gap is never a surprise.
    if settings.ocr_enabled:
        available = ocr_module.ocr_available(settings)
        out.append(
            ComponentStatus(
                name="ocr",
                status=HealthStatus.OK if available else HealthStatus.DEGRADED,
                detail=(
                    f"tesseract {ocr_module.ocr_version(settings)} "
                    f"({settings.ocr_language})"
                    if available
                    else "tesseract is unavailable; scanned pages cannot be read"
                ),
            )
        )
    else:
        out.append(
            ComponentStatus(
                name="ocr", status=HealthStatus.OK, detail="disabled by configuration"
            )
        )

    return out


def _aggregate(components: list[ComponentStatus]) -> HealthStatus:
    statuses = {c.status for c in components}
    if HealthStatus.UNAVAILABLE in statuses:
        return HealthStatus.UNAVAILABLE
    if HealthStatus.DEGRADED in statuses:
        return HealthStatus.DEGRADED
    return HealthStatus.OK


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health",
    description=(
        "Full service status: version, environment, uptime and per-component "
        "health. Returns 503 when the aggregate status is 'unavailable'."
    ),
)
async def health(
    response: Response, settings: Settings = Depends(get_config)
) -> HealthResponse:
    components = _check_components(settings)
    overall = _aggregate(components)

    if overall is HealthStatus.UNAVAILABLE:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status=overall,
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=round(get_uptime_seconds(), 3),
        components=components,
    )


@router.get(
    "/health/live",
    response_model=LivenessResponse,
    summary="Liveness probe",
    description="Cheapest possible check — touches no configuration or dependency.",
)
async def liveness() -> LivenessResponse:
    return LivenessResponse(status="alive")


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description=(
        "Whether the service should receive traffic. Returns 503 with a "
        "reason when it should not."
    ),
)
async def readiness(
    response: Response, settings: Settings = Depends(get_config)
) -> ReadinessResponse:
    components = _check_components(settings)
    unavailable = [c for c in components if c.status is HealthStatus.UNAVAILABLE]

    if unavailable:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        names = ", ".join(c.name for c in unavailable)
        return ReadinessResponse(ready=False, detail=f"unavailable: {names}")

    return ReadinessResponse(ready=True, detail=None)


__all__ = ["router"]
