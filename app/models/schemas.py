"""Pydantic request/response schemas — the HTTP contract.

These models are the single source of truth for what the MERN backend
sends and receives; FastAPI derives the OpenAPI document from them, so
``/openapi.json`` is always in sync with this file.

Stage 1 defines the envelope every stage will reuse (health, errors,
service info). Ingestion and query schemas arrive with their stages.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class HealthStatus(str, Enum):
    """Overall service state."""

    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ComponentStatus(BaseModel):
    """Health of one internal component.

    Stage 1 has only the configuration component. Later stages register
    the vector store, the embedding model and the LLM provider here
    without changing the response shape the backend already parses.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {"name": "configuration", "status": "ok", "detail": "loaded"}
    })

    name: str = Field(..., description="Component identifier")
    status: HealthStatus = Field(..., description="Component state")
    detail: Optional[str] = Field(
        None, description="Short human-readable note; never a stack trace"
    )


class HealthResponse(BaseModel):
    """Response of ``GET /health``."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "ok",
            "service": "LegalDocAI GenAI Service",
            "version": "0.1.0",
            "environment": "development",
            "uptime_seconds": 12.48,
            "timestamp": "2026-09-12T06:30:00Z",
            "components": [
                {"name": "configuration", "status": "ok", "detail": "loaded"}
            ],
        }
    })

    status: HealthStatus = Field(..., description="Aggregate service state")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    environment: str = Field(..., description="development | staging | production")
    uptime_seconds: float = Field(
        ..., ge=0, description="Seconds since the application finished starting"
    )
    timestamp: datetime = Field(
        default_factory=_utc_now, description="Server time, UTC"
    )
    components: List[ComponentStatus] = Field(
        default_factory=list, description="Per-component health"
    )


class LivenessResponse(BaseModel):
    """Response of ``GET /health/live`` — a cheap container probe."""

    status: str = Field("alive", description="Always 'alive' when reachable")


class ReadinessResponse(BaseModel):
    """Response of ``GET /health/ready`` — is the service ready for traffic?"""

    ready: bool = Field(..., description="False while a dependency is not usable")
    detail: Optional[str] = Field(None, description="Why the service is not ready")


class ServiceInfoResponse(BaseModel):
    """Response of ``GET /`` — orientation for a human or a smoke test."""

    service: str
    version: str
    environment: str
    scope: str = Field(
        ..., description="What this service does and does not own"
    )
    docs_url: Optional[str] = Field(None, description="Null when docs are disabled")
    stage: str = Field(..., description="Which build stage is implemented")


class ErrorDetail(BaseModel):
    """The body of an error response."""

    type: str = Field(..., description="Stable machine-readable error code")
    message: str = Field(..., description="Human-readable description")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Structured context, e.g. failed fields"
    )


class ErrorResponse(BaseModel):
    """Every non-2xx response from this service uses this shape."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "error": {
                "type": "validation_error",
                "message": "Request validation failed",
                "details": {
                    "fields": [
                        {"field": "tenant_id", "message": "Field required",
                         "type": "missing"}
                    ]
                },
            },
            "request_id": "0f9c1a2b3c4d5e6f",
        }
    })

    error: ErrorDetail
    request_id: str = Field(
        "-", description="Correlates the response with the service logs"
    )


# =====================================================================
# Stage 2 — document extraction
# =====================================================================


class ExtractionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class ExtractRequest(BaseModel):
    """Extract a document the AI service can already read.

    The MERN backend writes the upload to the shared volume rooted at
    ``DOCUMENT_ROOT`` and sends the path. ``tenant_id`` is trusted: the
    backend has already authenticated the user.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "document_id": "665f1c9e2ab4d1f0a1b2c3d4",
            "tenant_id": "665e02b41f9c8d7e6f5a4b3c",
            "file_path": "tenant-a/vendor_contract_2024.pdf",
            "filename": "vendor_contract_2024.pdf",
            "document_type": "contract",
        }
    })

    document_id: str = Field(..., description="The backend's id for this document")
    tenant_id: str = Field(..., description="Trusted tenant/user id")
    file_path: str = Field(..., description="Path relative to DOCUMENT_ROOT")
    filename: Optional[str] = Field(
        None, description="Original filename; defaults to the path basename"
    )
    document_type: Optional[str] = Field(
        None, description="contract | nda | policy | judgment | letter | invoice"
    )
    title: Optional[str] = None
    force_ocr: bool = Field(
        False, description="OCR every page, even ones with a text layer"
    )
    include_blocks: bool = Field(
        False, description="Return per-block detail as well as page text"
    )
    include_text: bool = Field(
        True, description="Return page text; false returns metadata and stats only"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Extra scalar fields to carry through"
    )


class BlockModel(BaseModel):
    block_id: str
    text: str
    page_number: int
    block_index: int
    kind: str
    section: Optional[str] = None
    ocr: bool = False


class PageModel(BaseModel):
    page_number: int
    text: str = ""
    method: str = Field(..., description="native | ocr | none")
    char_count: int = 0
    synthetic: bool = Field(
        False, description="True for DOCX, whose page boundaries are inferred"
    )
    sections: List[str] = Field(default_factory=list)
    blocks: Optional[List[BlockModel]] = None


class DocumentMetadataModel(BaseModel):
    document_id: str
    tenant_id: str
    filename: str
    document_type: str
    title: Optional[str] = None
    content_sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    ingested_at: Optional[str] = None


class ExtractionStatsModel(BaseModel):
    parser: str
    method: str = Field(..., description="native | ocr | mixed | none")
    page_count: int
    empty_page_count: int
    ocr_page_count: int
    block_count: int
    char_count: int
    duration_ms: int
    cleaning_applied: bool
    running_headers_removed: int


class ExtractResponse(BaseModel):
    """A successfully extracted document."""

    status: ExtractionStatus = ExtractionStatus.SUCCESS
    metadata: DocumentMetadataModel
    stats: ExtractionStatsModel
    sections: List[str] = Field(default_factory=list)
    pages: List[PageModel] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class BatchExtractRequest(BaseModel):
    documents: List[ExtractRequest] = Field(..., min_length=1, max_length=100)


class BatchExtractItem(BaseModel):
    """One document's outcome. A failure here is not an HTTP error."""

    document_id: str
    status: ExtractionStatus
    filename: Optional[str] = None
    stats: Optional[ExtractionStatsModel] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[ErrorDetail] = None


class BatchExtractResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: List[BatchExtractItem]
    duration_ms: int = 0


# =====================================================================
# Stage 3 — chunking
# =====================================================================


class ChunkModel(BaseModel):
    """One retrievable passage. This is what Stage 4 will embed."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "chunk_id": "3f1c8a2b4d5e6f708192a3b4c5d6e7f8",
            "chunk_index": 4,
            "document_id": "665f1c9e2ab4d1f0a1b2c3d4",
            "tenant_id": "665e02b41f9c8d7e6f5a4b3c",
            "filename": "vendor_contract_2024.pdf",
            "page_number": 17,
            "page_end": 17,
            "section": "4. TERMINATION",
            "document_type": "contract",
            "text": "4. TERMINATION\n\nEither party may terminate...",
            "token_count": 124,
            "overlap_tokens": 48,
        }
    })

    chunk_id: str = Field(..., description="Deterministic; stable across re-ingestion")
    chunk_index: int = Field(..., ge=0, description="Position within the document")
    document_id: str
    tenant_id: str = Field(..., description="Isolation key for all later retrieval")
    filename: str
    page_number: int = Field(..., description="First page this chunk touches")
    page_end: int = Field(..., description="Last page; differs when a clause spans a break")
    section: Optional[str] = Field(None, description="Clause/section label, when detected")
    document_type: str
    text: str
    token_count: int
    char_count: int
    overlap_tokens: int = Field(0, description="Leading tokens copied from the previous chunk")
    ocr: bool = False
    split_mid_sentence: bool = Field(
        False, description="True when a single oversized sentence had to be cut"
    )
    block_ids: List[str] = Field(default_factory=list)


class ChunkingStatsModel(BaseModel):
    chunk_count: int
    total_tokens: int
    min_tokens: int
    max_tokens: int
    mean_tokens: float
    clean_boundary_count: int
    mid_sentence_count: int
    merged_small_count: int
    sections_covered: int
    pages_covered: int
    spanning_page_count: int
    tokenizer: str
    chunk_size: int
    chunk_overlap: int
    duration_ms: int


class ChunkRequest(ExtractRequest):
    """Extract a document and chunk it in one call.

    Chunk size and overlap default to the service configuration and can
    be overridden per request, which is what makes tuning possible
    without a redeploy.
    """

    chunk_size: Optional[int] = Field(
        None, ge=32, le=8192, description="Target tokens per chunk"
    )
    chunk_overlap: Optional[int] = Field(
        None, ge=0, le=4096, description="Tokens repeated between neighbours"
    )
    min_chunk_tokens: Optional[int] = Field(
        None, ge=1, description="Below this a chunk is merged into a neighbour"
    )
    respect_sections: Optional[bool] = Field(
        None, description="Never let a chunk span two sections"
    )
    include_chunk_text: bool = Field(
        True, description="False returns chunk metadata without the text"
    )


class ChunkResponse(BaseModel):
    status: ExtractionStatus = ExtractionStatus.SUCCESS
    metadata: DocumentMetadataModel
    extraction: ExtractionStatsModel
    chunking: ChunkingStatsModel
    sections: List[str] = Field(default_factory=list)
    chunks: List[ChunkModel] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# =====================================================================
# Stage 4 — indexing and search
# =====================================================================


class IndexRequest(ExtractRequest):
    """Extract, chunk, embed and index one document."""


class IndexResponse(BaseModel):
    status: str = Field(..., description="success | failed | empty")
    document_id: str
    tenant_id: str
    filename: str
    chunks_indexed: int = 0
    chunks_deleted: int = Field(
        0, description="Existing chunks removed before re-indexing"
    )
    replaced: bool = Field(
        False, description="True when this replaced an earlier version"
    )
    pages: int = 0
    ocr_pages: int = 0
    embedding_model: str = ""
    embedding_dimension: int = Field(
        0, description="Read from the model, never assumed"
    )
    chunking: Optional[ChunkingStatsModel] = None
    warnings: List[str] = Field(default_factory=list)
    duration_ms: int = 0
    timings_ms: Dict[str, int] = Field(default_factory=dict)


class BatchIndexRequest(BaseModel):
    documents: List[IndexRequest] = Field(..., min_length=1, max_length=100)


class BatchIndexResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    chunks_indexed: int = 0
    results: List[IndexResponse]
    duration_ms: int = 0


class SearchRequest(BaseModel):
    """Similarity search. ``tenant_id`` is mandatory and trusted."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "How much notice is required to terminate?",
            "tenant_id": "665e02b41f9c8d7e6f5a4b3c",
            "top_k": 8,
            "document_type": "contract",
        }
    })

    query: str = Field(..., min_length=1, max_length=4000)
    tenant_id: str = Field(..., description="Trusted tenant/user id")
    document_ids: Optional[List[str]] = Field(
        None, description="Restrict to these documents"
    )
    document_type: Optional[str] = None
    section: Optional[str] = None
    page_range: Optional[List[int]] = Field(
        None, min_length=2, max_length=2, description="Inclusive [from, to]"
    )
    metadata_equals: Dict[str, Any] = Field(
        default_factory=dict, description="Extra exact-match constraints"
    )
    top_k: Optional[int] = Field(None, ge=1, le=200)
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    include_text: bool = True


class SearchHitModel(BaseModel):
    rank: int
    score: float = Field(..., description="Higher is better, backend-normalised")
    chunk_id: str
    chunk_index: int
    document_id: str
    filename: str
    page_number: int
    page_end: int
    section: Optional[str] = None
    document_type: str
    text: str
    token_count: int = 0
    ocr: bool = False


class SearchResponse(BaseModel):
    query: str
    tenant_id: str
    hits: List[SearchHitModel] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = 0
    min_score: float = 0.0
    embedding_model: str = ""
    embedding_dimension: int = 0
    duration_ms: int = 0
    timings_ms: Dict[str, int] = Field(default_factory=dict)


class DeleteResponse(BaseModel):
    deleted: bool
    document_id: Optional[str] = None
    tenant_id: Optional[str] = None
    chunks_deleted: int = 0


class StoreStatsResponse(BaseModel):
    backend: str
    collection: str
    metric: str
    dimension: int
    total_chunks: int
    tenant_chunks: Optional[int] = None
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    max_sequence_length: int = 0
    normalized: bool = False


# =====================================================================
# Stage 5 — hybrid retrieval
# =====================================================================


class RetrieveRequest(BaseModel):
    """Hybrid retrieval over one tenant's indexed chunks.

    The difference from ``/search``: that endpoint is a single vector
    query, this one runs the whole pipeline — preprocessing, semantic and
    keyword retrieval, fusion, ranking, duplicate removal and reranking —
    and returns a short list fit to put in front of an LLM.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "How much notice is required to terminate for convenience?",
            "tenant_id": "665e02b41f9c8d7e6f5a4b3c",
            "top_k": 6,
            "document_type": "contract",
            "explain": True,
        }
    })

    query: str = Field(..., min_length=1, max_length=4000)
    tenant_id: str = Field(..., description="Trusted tenant/user id")
    document_ids: Optional[List[str]] = Field(
        None, description="Restrict to these documents"
    )
    document_type: Optional[str] = None
    section: Optional[str] = None
    page_range: Optional[List[int]] = Field(
        None, min_length=2, max_length=2, description="Inclusive [from, to]"
    )
    metadata_equals: Dict[str, Any] = Field(
        default_factory=dict, description="Extra exact-match constraints"
    )
    top_k: Optional[int] = Field(
        None, ge=1, le=100, description="Passages returned after reranking"
    )
    candidate_pool: Optional[int] = Field(
        None,
        ge=1,
        le=500,
        description=(
            "Candidates retrieved before reranking. Raised only when recall "
            "matters more than latency; the default is ~18."
        ),
    )
    # Validated here rather than in the pipeline so a typo from the
    # backend comes back as a 422 naming the field, not a 500 with a
    # traceback in the service log.
    mode: Optional[Literal["hybrid", "semantic", "keyword"]] = Field(
        None,
        description=(
            "hybrid (both signals, default) | semantic (vectors only) | "
            "keyword (BM25 only)"
        ),
    )
    rerank: Optional[bool] = Field(
        None, description="Override RERANK_ENABLED for this request"
    )
    dedupe: Optional[bool] = Field(
        None, description="Override DEDUPE_ENABLED for this request"
    )
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    include_text: bool = True
    explain: bool = Field(
        False,
        description=(
            "Include every per-stage score and boost for each result — "
            "how you answer 'why was this passage retrieved?'"
        ),
    )


class CandidateScoresModel(BaseModel):
    """Every signal one passage collected, kept separate on purpose."""

    semantic: float = 0.0
    keyword: float = 0.0
    keyword_raw: float = 0.0
    fused: float = 0.0
    ranked: float = 0.0
    rerank: Optional[float] = None
    final: float = 0.0
    semantic_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    sources: List[str] = Field(default_factory=list)
    boosts: Dict[str, float] = Field(default_factory=dict)
    duplicates_removed: int = 0
    pinned: bool = Field(
        False,
        description=(
            "The query named this exact clause, so no later stage was "
            "allowed to demote it"
        ),
    )


class RetrievedChunkModel(BaseModel):
    """One retrieved passage, with everything a citation needs."""

    rank: int
    score: float = Field(..., description="Final blended score, higher is better")
    chunk_id: str
    chunk_index: int
    document_id: str
    filename: str
    page_number: int
    page_end: int
    page_range: str = Field(..., description="Human-readable, e.g. '7' or '7-8'")
    section: Optional[str] = None
    document_type: str
    text: str
    token_count: int = 0
    ocr: bool = Field(
        False, description="Came from an OCR'd page — quote with care"
    )
    duplicates: List[str] = Field(
        default_factory=list,
        description="chunk_ids of near-duplicates collapsed into this result",
    )
    scores: Optional[CandidateScoresModel] = Field(
        None, description="Present when explain=true"
    )


class RetrievalCountsModel(BaseModel):
    """How many candidates survived each stage."""

    semantic: int = 0
    keyword: int = 0
    fused: int = 0
    after_dedupe: int = 0
    reranked: int = 0
    returned: int = 0
    duplicates_removed: int = 0
    corpus_size: int = Field(
        0, description="Chunks BM25 scored for this query"
    )


class RetrieveResponse(BaseModel):
    """Retrieved passages — **not** an answer.

    Grounded generation with citations is Stage 6. This response is what
    that stage will consume, and what the MERN backend can already render
    as 'sources'.
    """

    query: str
    normalized_query: str = ""
    tenant_id: str
    mode: str = "hybrid"
    results: List[RetrievedChunkModel] = Field(default_factory=list)
    counts: RetrievalCountsModel = Field(default_factory=RetrievalCountsModel)
    filters: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = 0
    candidate_pool: int = 0
    min_score: float = 0.0
    fusion_method: str = ""
    agreement: float = Field(
        0.0, description="Fraction of candidates both retrievers found"
    )
    references: List[str] = Field(
        default_factory=list,
        description="Clause/section references detected in the query",
    )
    embedding_model: str = ""
    embedding_dimension: int = 0
    reranker: str = ""
    reranked: bool = False
    rerank_is_cross_encoder: bool = Field(
        False,
        description=(
            "False when the lexical fallback ran instead of a real "
            "cross-encoder — ordering quality is lower"
        ),
    )
    warnings: List[str] = Field(default_factory=list)
    duration_ms: int = 0
    timings_ms: Dict[str, int] = Field(default_factory=dict)


class RetrievalConfigResponse(BaseModel):
    """The live retrieval configuration — for tuning and for support."""

    mode: str
    candidate_pool: int
    top_k: int
    fusion_method: str
    rrf_k: int
    semantic_weight: float
    keyword_weight: float
    bm25_k1: float
    bm25_b: float
    keyword_corpus_limit: int
    dedupe_enabled: bool
    dedupe_threshold: float
    rerank_enabled: bool
    reranker_provider: str
    reranker_model: str
    reranker_active: str
    rerank_is_cross_encoder: bool
    rerank_weight: float
    min_score: float


# =====================================================================
# Stage 6 — RAG context construction
# =====================================================================


class ContextRequest(RetrieveRequest):
    """Retrieve, then build an LLM-ready context from what comes back.

    Every retrieval option from ``/retrieve`` still applies; these add
    control over the context itself. **No answer is generated** — the
    response is evidence, labelled and citable.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "How much notice is required to terminate for convenience?",
            "tenant_id": "665e02b41f9c8d7e6f5a4b3c",
            "context_max_tokens": 2000,
            "context_max_sources": 5,
        }
    })

    context_max_tokens: Optional[int] = Field(
        None,
        ge=64,
        le=200_000,
        description=(
            "Token ceiling for the whole context block, headers included. "
            "Not the model's window — leave room for the prompt, the "
            "question and the answer."
        ),
    )
    context_max_sources: Optional[int] = Field(None, ge=1, le=100)
    context_min_sources: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Below this the response is flagged as thin evidence",
    )
    context_min_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    context_max_source_tokens: Optional[int] = Field(None, ge=0, le=100_000)
    context_order: Optional[Literal["document", "relevance"]] = Field(
        None,
        description=(
            "document (reading order within each document, default) | "
            "relevance (strongest passage first)"
        ),
    )
    context_dedupe: Optional[bool] = None
    context_truncate: Optional[bool] = Field(
        None,
        description=(
            "Shorten an oversized passage rather than dropping it. A "
            "passage shortened below the minimum is dropped either way."
        ),
    )
    include_context_text: bool = Field(
        True, description="False returns the structure without the prompt block"
    )


class ContextSourceModel(BaseModel):
    """One numbered block of the context, with its provenance."""

    source: int = Field(..., description="1-based, matches 'Source N:' in the text")
    chunk_id: str
    chunk_index: int
    document_id: str
    filename: str
    page_number: int
    page_end: int
    page_range: str
    section: Optional[str] = None
    document_type: str
    text: str = Field("", description="Empty when include_text=false")
    score: float = Field(0.0, description="Retrieval score, carried through")
    retrieval_rank: int = Field(
        0, description="Rank in the retrieval ordering, which is not this one"
    )
    tokens: int = Field(0, description="Tokens of passage text")
    total_tokens: int = Field(0, description="Tokens of the whole block, header included")
    truncated: bool = False
    omitted_tokens: int = Field(
        0, description="Passage tokens removed by truncation"
    )
    ocr: bool = False
    duplicates: List[str] = Field(
        default_factory=list,
        description="chunk_ids of identical passages folded into this source",
    )
    continues_source: Optional[int] = Field(
        None,
        description="Set when this is the next chunk of the previous source",
    )


class OmittedSourceModel(BaseModel):
    """A retrieved passage that did not reach the prompt — and why.

    Nothing is dropped without an entry here. "Did you look at clause 9?"
    has an answer.
    """

    chunk_id: str
    reason: str = Field(
        ...,
        description=(
            "below_score_floor | redundant | max_sources | "
            "budget_exhausted | too_small_to_truncate | empty_text"
        ),
    )
    score: float = 0.0
    citation: Dict[str, Any] = Field(default_factory=dict)
    duplicate_of: Optional[str] = None
    tokens: int = 0


class ContextStatsModel(BaseModel):
    candidates_in: int = 0
    sources_out: int = 0
    documents: int = 0
    redundant_removed: int = 0
    below_floor_removed: int = 0
    over_max_sources: int = 0
    dropped_for_budget: int = 0
    truncated_sources: int = 0
    tokens_used: int = 0
    tokens_budget: int = 0
    tokens_remaining: int = 0
    text_tokens: int = 0
    overhead_tokens: int = Field(
        0, description="Tokens spent on source headers and separators"
    )
    omitted_text_tokens: int = 0
    build_ms: int = 0


class ContextResponse(BaseModel):
    """A context block ready for an LLM prompt — **not** an answer."""

    context: str = Field(
        ..., description="The prompt block, exactly as an LLM would receive it"
    )
    query: str
    tenant_id: str
    sources: List[ContextSourceModel] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ready to render as 'Sources'; no parsing of the block needed",
    )
    omitted: List[OmittedSourceModel] = Field(default_factory=list)
    stats: ContextStatsModel = Field(default_factory=ContextStatsModel)
    retrieval: RetrievalCountsModel = Field(default_factory=RetrievalCountsModel)
    order: str = "document"
    tokenizer: str = ""
    sufficient: bool = Field(
        True,
        description=(
            "False when too little evidence reached the context for a "
            "confident answer. Reported, not enforced."
        ),
    )
    warnings: List[str] = Field(default_factory=list)
    reranker: str = ""
    rerank_is_cross_encoder: bool = False
    duration_ms: int = 0
    timings_ms: Dict[str, int] = Field(default_factory=dict)


class ContextConfigResponse(BaseModel):
    """The live context-builder configuration."""

    max_tokens: int
    max_sources: int
    min_sources: int
    min_score: float
    max_source_tokens: int
    min_source_tokens: int
    truncate_long_sources: bool
    order: str
    dedupe_enabled: bool
    redundancy_threshold: float
    strip_repeated_heading: bool
    tokenizer: str


# =====================================================================
# Stage 7 — grounded generation
# =====================================================================


class AnswerRequest(ContextRequest):
    """Ask a question and get a grounded, cited answer.

    Every retrieval and context option still applies. The answer is
    generated **only** from the passages retrieved for this tenant.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "How much notice is required to terminate for convenience?",
            "tenant_id": "665e02b41f9c8d7e6f5a4b3c",
            "include_context": False,
        }
    })

    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Overrides LLM_TEMPERATURE. 0 keeps answers reproducible.",
    )
    max_output_tokens: Optional[int] = Field(None, ge=64, le=32_000)
    include_context: bool = Field(
        False,
        description="Include the full prompt block that was sent to the model",
    )
    include_sources: bool = Field(
        True, description="Include the retrieved passages behind the answer"
    )


class CitationModel(BaseModel):
    """One validated citation.

    Every field is taken from the retrieved record, never from the
    model's reply — a wrong page number in a legal citation is worse than
    no page number, because it looks checkable.
    """

    document: str = Field(..., description="Filename of the source document")
    page: Optional[int] = Field(
        None, description="First page; null when the document has no pagination"
    )
    section: Optional[str] = Field(None, description="Clause or section label")
    chunk_id: str = Field(..., description="Exact passage this cites")
    source: int = Field(..., description="'Source N' in the context block")
    document_id: str = ""
    page_range: str = Field("", description="e.g. '7' or '7-8'")
    claimed: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Where the model's own metadata disagreed with the record. "
            "The record was used."
        ),
    )


class RejectedCitationModel(BaseModel):
    """A citation that did not survive validation, and why."""

    reason: str = Field(
        ...,
        description=(
            "unknown_source | unknown_source_in_answer | no_citations | "
            "metadata_mismatch"
        ),
    )
    claimed_source: Optional[int] = None
    detail: str = ""


class CitationValidationModel(BaseModel):
    """The outcome of checking the answer against what was retrieved."""

    grounded: bool = Field(
        ...,
        description=(
            "False when the answer cites nothing, cites a source that was "
            "not retrieved, or quotes text not found in the evidence"
        ),
    )
    valid_citations: int = 0
    rejected: List[RejectedCitationModel] = Field(default_factory=list)
    verified_quotes: int = 0
    unverified_quotes: List[str] = Field(
        default_factory=list,
        description="Quoted spans not found in any retrieved passage",
    )


class AnswerResponse(BaseModel):
    """A grounded answer with validated citations.

    `status` is the field to branch on. `answer` is null when generation
    failed; the retrieved evidence is still returned in that case, so a
    failed model call does not throw away the retrieval.
    """

    status: str = Field(
        ...,
        description=(
            "ok | insufficient_evidence | llm_unavailable | "
            "llm_configuration_error | malformed_response"
        ),
    )
    question: str
    tenant_id: str
    answer: Optional[str] = Field(
        None, description="Null when no answer could be generated"
    )
    citations: List[CitationModel] = Field(default_factory=list)
    interpretation: str = Field(
        "",
        description=(
            "The model's own analysis, kept separate from what the "
            "documents state. Empty when it offered none."
        ),
    )
    insufficient_evidence: bool = Field(
        False, description="The evidence does not answer the question"
    )
    grounded: bool = Field(
        True, description="Shorthand for validation.grounded"
    )
    validation: CitationValidationModel

    sources: List[ContextSourceModel] = Field(
        default_factory=list, description="The passages the answer was built from"
    )
    context: Optional[str] = Field(
        None, description="The prompt block, when include_context=true"
    )
    context_stats: ContextStatsModel = Field(default_factory=ContextStatsModel)
    retrieval: RetrievalCountsModel = Field(default_factory=RetrievalCountsModel)

    provider: str = ""
    model: str = ""
    is_language_model: bool = Field(
        True,
        description=(
            "False when the deterministic extractive provider answered. "
            "Its output is selected sentences, not generated prose."
        ),
    )
    usage: Dict[str, int] = Field(default_factory=dict)
    attempts: int = Field(1, description="Generation attempts made")
    error: str = Field("", description="Set when status is a failure")
    raw_excerpt: str = Field(
        "", description="Start of an unparseable reply, for diagnosis"
    )

    warnings: List[str] = Field(default_factory=list)
    duration_ms: int = 0
    timings_ms: Dict[str, int] = Field(default_factory=dict)


class GenerationConfigResponse(BaseModel):
    """The live generation configuration. Never includes a credential."""

    provider: str
    provider_active: str
    model: str
    is_language_model: bool
    available_providers: List[str]
    temperature: float
    max_output_tokens: int
    timeout_seconds: float
    max_retries: int
    strict: bool
    require_citations: bool
    harvest_inline_citations: bool
    verify_quotes: bool
    api_key_configured: bool = Field(
        ...,
        description=(
            "Whether a key is present for the selected provider. The key "
            "itself is never returned."
        ),
    )


# =====================================================================
# Stage 8 — the integration API
#
# Five endpoints, shaped for a Node/Express backend to consume: flat
# fields, stable string status codes to branch on, one error envelope,
# and a request_id on every response so a log line in Express can be
# matched to a log line here.
# =====================================================================


class IngestRequest(BaseModel):
    """Ingest one document: extract, chunk, embed and index it.

    Supply **either** `file_path` (a path under `DOCUMENT_ROOT`, the
    shared volume the Node backend writes uploads to) **or** `text` (the
    content itself, when the backend already has it in memory). Sending
    both, or neither, is a validation error.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "document_id": "665f1c9e2ab4d1f0a1b2c3d4",
            "tenant_id": "665e02b41f9c8d7e6f5a4b3c",
            "file_path": "tenant-a/vendor_contract_2024.pdf",
            "document_type": "contract",
        }
    })

    document_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="The backend's own id for this document. Re-ingesting "
        "the same id replaces its chunks rather than duplicating them.",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Trusted tenant/user id. Every later query is scoped to it.",
    )
    file_path: Optional[str] = Field(
        None, description="Path relative to DOCUMENT_ROOT. Mutually exclusive with `text`."
    )
    text: Optional[str] = Field(
        None,
        description=(
            "Document content the backend already holds. No pagination, so "
            "citations report no page number. Mutually exclusive with "
            "`file_path`."
        ),
    )
    filename: Optional[str] = Field(
        None, max_length=512, description="Shown in citations; defaults to the path basename"
    )
    document_type: Optional[str] = Field(
        None, description="contract | nda | policy | judgment | letter | invoice"
    )
    title: Optional[str] = Field(None, max_length=512)
    force_ocr: bool = Field(False, description="OCR every page of a PDF")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra scalar fields carried onto every chunk and filterable at query time",
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "IngestRequest":
        has_path = bool(self.file_path and self.file_path.strip())
        has_text = bool(self.text and self.text.strip())
        if has_path == has_text:
            raise ValueError(
                "Provide exactly one of 'file_path' or 'text'"
                + (" — both were supplied" if has_path else " — neither was supplied")
            )
        return self


class IngestResponse(BaseModel):
    """What ingesting one document did."""

    status: str = Field(
        ...,
        description=(
            "success | unchanged | empty. A failure is an HTTP error, not "
            "this field. 'unchanged' means the content digest already "
            "matched what is indexed, so nothing was re-processed."
        ),
    )
    document_id: str
    tenant_id: str
    filename: str
    document_type: str = ""
    duplicate_of: Optional[str] = Field(
        None,
        description=(
            "Another document of this tenant holding byte-identical "
            "content. Reported, never refused — two copies of one contract "
            "filed under two matters is legitimate."
        ),
    )
    chunks_indexed: int = 0
    chunks_replaced: int = Field(
        0, description="Existing chunks removed because this id was re-ingested"
    )
    pages: int = 0
    ocr_pages: int = 0
    embedding_model: str = ""
    embedding_dimension: int = 0
    warnings: List[str] = Field(default_factory=list)
    duration_ms: int = 0
    request_id: str = ""


class BatchIngestItem(BaseModel):
    """One document in a batch. A **reference**, never content.

    There is deliberately no `text` field here: a batch takes paths so
    that a 500-document request cannot carry 500 documents in its body.
    """

    document_id: str = Field(..., min_length=1, max_length=128)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    file_path: str = Field(..., min_length=1, description="Path relative to DOCUMENT_ROOT")
    filename: Optional[str] = Field(None, max_length=512)
    document_type: Optional[str] = None
    title: Optional[str] = Field(None, max_length=512)
    force_ocr: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchIngestRequest(BaseModel):
    """Up to ~500 documents, processed in the background."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "documents": [
                {
                    "document_id": "doc-1",
                    "tenant_id": "665e02b41f9c8d7e6f5a4b3c",
                    "file_path": "tenant-a/contract_001.pdf",
                },
                {
                    "document_id": "doc-2",
                    "tenant_id": "665e02b41f9c8d7e6f5a4b3c",
                    "file_path": "tenant-a/contract_002.pdf",
                },
            ]
        }
    })

    documents: List[BatchIngestItem] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description=(
            "The configured ceiling is INGEST_BATCH_MAX_DOCUMENTS (default "
            "500); beyond it the request is rejected with 413."
        ),
    )


class BatchIngestAccepted(BaseModel):
    """HTTP 202. The batch was accepted; poll `poll_url` for progress."""

    job_id: str
    status: str = Field("queued", description="queued | running | completed | failed")
    total: int
    poll_url: str = Field(..., description="GET this for progress and per-document results")
    accepted_at: str
    request_id: str = ""


class BatchIngestItemResult(BaseModel):
    """One document's outcome inside a job. A failure here is not an HTTP error."""

    document_id: str
    tenant_id: str
    status: str = Field(..., description="success | empty | failed")
    filename: str = ""
    chunks_indexed: int = 0
    pages: int = 0
    error_type: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    duration_ms: int = 0


class BatchIngestStatus(BaseModel):
    """Progress and results for one batch job."""

    job_id: str
    status: str = Field(..., description="queued | running | completed | failed")
    total: int
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    chunks_indexed: int = 0
    progress: float = Field(0.0, ge=0.0, le=1.0)
    results: List[BatchIngestItemResult] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: int = 0
    request_id: str = ""


class QueryRequest(BaseModel):
    """Ask a question against one tenant's documents."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "How much notice is required to terminate for convenience?",
            "tenant_id": "665e02b41f9c8d7e6f5a4b3c",
            "top_k": 6,
            "document_type": "contract",
            "include_sources": False,
        }
    })

    query: str = Field(..., min_length=1, max_length=4000)
    tenant_id: str = Field(..., min_length=1, max_length=128)

    # -- optional document filters ------------------------------------
    document_ids: Optional[List[str]] = Field(
        None, max_length=200, description="Restrict the search to these documents"
    )
    document_type: Optional[str] = None
    section: Optional[str] = Field(None, max_length=256)
    page_range: Optional[List[int]] = Field(
        None, min_length=2, max_length=2, description="Inclusive [from, to]"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Exact-match constraints on fields supplied at ingest time",
    )

    # -- optional tuning ----------------------------------------------
    top_k: Optional[int] = Field(
        None,
        ge=1,
        le=50,
        description="Passages the answer is built from. Defaults to CONTEXT_MAX_SOURCES.",
    )
    include_sources: bool = Field(
        False, description="Return the retrieved passages — for debugging and 'show sources' UIs"
    )
    include_context: bool = Field(
        False, description="Return the exact prompt block sent to the model — debugging only"
    )
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)


class QuerySourceModel(BaseModel):
    """A retrieved passage, returned when `include_sources` is set."""

    source: int = Field(..., description="Matches the 'source' on a citation")
    chunk_id: str
    document_id: str
    filename: str
    page: Optional[int] = None
    page_range: str = ""
    section: Optional[str] = None
    document_type: str = ""
    text: str = ""
    score: float = 0.0
    retrieval_rank: int = 0
    truncated: bool = False


class QueryResponse(BaseModel):
    """An answer, its citations, and — optionally — the evidence behind it."""

    status: str = Field(
        ...,
        description=(
            "ok | insufficient_evidence | llm_unavailable | "
            "llm_configuration_error | malformed_response"
        ),
    )
    answer: Optional[str] = Field(
        None, description="Null when no answer could be generated; `status` says why"
    )
    citations: List[CitationModel] = Field(default_factory=list)
    interpretation: str = Field(
        "", description="The model's own analysis, kept out of the answer"
    )
    insufficient_evidence: bool = False
    grounded: bool = Field(
        True, description="False when the answer could not be fully checked against the evidence"
    )

    query: str
    tenant_id: str
    sources: List[QuerySourceModel] = Field(
        default_factory=list, description="Present when include_sources=true"
    )
    context: Optional[str] = Field(
        None, description="Present when include_context=true"
    )

    validation: CitationValidationModel
    provider: str = ""
    model: str = ""
    is_language_model: bool = True
    warnings: List[str] = Field(default_factory=list)
    duration_ms: int = 0
    timings_ms: Dict[str, int] = Field(default_factory=dict)
    request_id: str = ""


class DeleteDocumentResponse(BaseModel):
    """Outcome of removing a document's vector data."""

    deleted: bool = Field(
        ..., description="False when nothing matched — not an error, see the endpoint docs"
    )
    document_id: str
    tenant_id: str
    chunks_deleted: int = 0
    request_id: str = ""


class SupportedFormatsResponse(BaseModel):
    extensions: List[str]
    max_file_size_mb: int
    ocr_enabled: bool
    ocr_available: bool
    ocr_language: Optional[str] = None


__all__ = [
    "HealthStatus",
    "ComponentStatus",
    "HealthResponse",
    "LivenessResponse",
    "ReadinessResponse",
    "ServiceInfoResponse",
    "ErrorDetail",
    "ErrorResponse",
    "ExtractionStatus",
    "ExtractRequest",
    "ExtractResponse",
    "BlockModel",
    "PageModel",
    "DocumentMetadataModel",
    "ExtractionStatsModel",
    "BatchExtractRequest",
    "BatchExtractItem",
    "BatchExtractResponse",
    "SupportedFormatsResponse",
    "ChunkModel",
    "ChunkingStatsModel",
    "ChunkRequest",
    "ChunkResponse",
    "IndexRequest",
    "IndexResponse",
    "BatchIndexRequest",
    "BatchIndexResponse",
    "SearchRequest",
    "SearchHitModel",
    "SearchResponse",
    "DeleteResponse",
    "StoreStatsResponse",
    "RetrieveRequest",
    "RetrieveResponse",
    "RetrievedChunkModel",
    "CandidateScoresModel",
    "RetrievalCountsModel",
    "RetrievalConfigResponse",
    "ContextRequest",
    "ContextResponse",
    "ContextSourceModel",
    "OmittedSourceModel",
    "ContextStatsModel",
    "ContextConfigResponse",
    "AnswerRequest",
    "AnswerResponse",
    "CitationModel",
    "RejectedCitationModel",
    "CitationValidationModel",
    "GenerationConfigResponse",
    "IngestRequest",
    "IngestResponse",
    "BatchIngestItem",
    "BatchIngestRequest",
    "BatchIngestAccepted",
    "BatchIngestItemResult",
    "BatchIngestStatus",
    "QueryRequest",
    "QuerySourceModel",
    "QueryResponse",
    "DeleteDocumentResponse",
]
