"""FastAPI application for the LegalDocAI GenAI service.

Stage 7: the foundation, ingestion, chunking, embeddings, vector storage,
hybrid retrieval with reranking, RAG context construction and grounded
generation with citation validation.

This process is an independent AI microservice consumed by the MERN
backend over HTTP. It contains no React, Node.js, Express or MongoDB
code, and never will.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    context,
    documents,
    generation,
    health,
    index,
    pipeline,
    retrieval,
)
from app.api.deps import mark_started
from app.config import Settings, get_settings
from app.exceptions import ERROR_RESPONSES, register_exception_handlers
from app.logging_config import (
    get_logger,
    reset_request_id,
    set_request_id,
    setup_logging,
)
from app.models.schemas import ServiceInfoResponse

logger = get_logger(__name__)

STAGE = (
    "9 - integrated pipeline (ingestion, chunking, embeddings, vector "
    "index, hybrid retrieval with reranking, context construction, "
    "grounded generation with validated citations) exposed as POST "
    "/ingest, POST /ingest/batch, POST /query, DELETE /documents/{id} "
    "and GET /health, hardened and measured at ~500 documents"
)


def _is_probe(path: str) -> bool:
    """Health probes are high-frequency and low-information."""
    return path.startswith("/health")


SCOPE = (
    "GenAI / NLP / RAG only. Ingestion, chunking, embeddings, vector "
    "search, hybrid retrieval with reranking, context construction and "
    "grounded generation with validated citations. Answers are produced "
    "only from passages retrieved from the caller's own documents, and "
    "every citation is checked against those passages. The React "
    "frontend, Express API, MongoDB and authentication are owned by the "
    "MERN backend."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown."""
    settings: Settings = app.state.settings
    mark_started()
    logger.info(
        "%s v%s starting (environment=%s, docs=%s, auth=%s)",
        settings.app_name,
        settings.app_version,
        settings.environment,
        "on" if settings.enable_docs else "off",
        "on" if settings.auth_enabled else "off",
    )
    if settings.is_production and not settings.auth_enabled:
        logger.warning(
            "Running in production with no SERVICE_TOKEN — the service is "
            "unauthenticated. Set SERVICE_TOKEN before exposing it."
        )
    # Tidy up after any previous process that was killed mid-request.
    # Bounded by age and confined to TEMP_DIR; never fatal.
    try:
        from app.ingestion.tempfiles import sweep_stale_temp_files

        sweep_stale_temp_files(settings)
    except Exception:  # pragma: no cover - startup must not depend on it
        logger.warning("Stale temp-file sweep failed", exc_info=True)

    logger.info("Stage %s", STAGE)
    yield
    logger.info("%s shutting down", settings.app_name)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory.

    Building the app through a factory (rather than at import time) keeps
    it constructible with arbitrary settings in tests.
    """
    settings = settings or get_settings()
    setup_logging(level=settings.log_level, fmt=settings.log_format, force=True)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Independent Python AI microservice for LegalDocAI, consumed by "
            "the MERN backend over HTTP.\n\n"
            f"**Current build stage:** {STAGE}"
        ),
        lifespan=lifespan,
        docs_url=settings.docs_url,
        redoc_url=None,
        openapi_url=settings.openapi_url,
    )
    app.state.settings = settings

    # -- Middleware ---------------------------------------------------

    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Assign a request id, log the request, expose timing."""
        incoming = request.headers.get("X-Request-ID", "").strip()
        request_id = incoming or uuid.uuid4().hex[:16]
        token = set_request_id(request_id)
        started = time.perf_counter()
        # The id must stay bound until the last log line for this request
        # has been emitted — releasing it earlier (e.g. in a `finally`
        # that runs before the response is logged) silently strips the
        # correlation id from exactly the line you need when debugging.
        try:
            try:
                response = await call_next(request)
            except Exception:
                # The registered handlers produce the body; this only
                # records timing for a request that blew up mid-flight.
                elapsed = (time.perf_counter() - started) * 1000
                logger.error(
                    "%s %s failed after %.1f ms",
                    request.method,
                    request.url.path,
                    elapsed,
                )
                raise

            elapsed = (time.perf_counter() - started) * 1000
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-ms"] = f"{elapsed:.1f}"
            # Health probes fire constantly — keep them out of INFO.
            level = logging.DEBUG if _is_probe(request.url.path) else logging.INFO
            logger.log(
                level,
                "%s %s -> %d (%.1f ms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed,
            )
            return response
        finally:
            reset_request_id(token)

    # -- Routers ------------------------------------------------------

    # The integration contract first, so it heads the OpenAPI document
    # the MERN developer reads. The per-stage routers below it are
    # mounted for diagnosis, not for Express to build on.
    app.include_router(pipeline.router, responses=ERROR_RESPONSES)
    app.include_router(health.router, responses=ERROR_RESPONSES)
    app.include_router(documents.router, responses=ERROR_RESPONSES)
    app.include_router(index.router, responses=ERROR_RESPONSES)
    app.include_router(retrieval.router, responses=ERROR_RESPONSES)
    app.include_router(context.router, responses=ERROR_RESPONSES)
    app.include_router(generation.router, responses=ERROR_RESPONSES)

    # -- Error handling -----------------------------------------------

    register_exception_handlers(app)

    # -- Root ---------------------------------------------------------

    @app.get(
        "/",
        response_model=ServiceInfoResponse,
        tags=["meta"],
        summary="Service information",
        responses=ERROR_RESPONSES,
    )
    async def root() -> ServiceInfoResponse:
        return ServiceInfoResponse(
            service=settings.app_name,
            version=settings.app_version,
            environment=settings.environment,
            scope=SCOPE,
            docs_url=settings.docs_url,
            stage=STAGE,
        )

    return app


app = create_app()

__all__ = ["app", "create_app", "STAGE", "SCOPE"]
