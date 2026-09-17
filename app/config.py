"""Configuration management for the LegalDocAI GenAI service.

Every setting is environment-driven (see ``.env.example``). Values are
loaded once and cached, so the rest of the application depends on a
single immutable ``Settings`` object obtained through
:func:`get_settings` — never on ``os.environ`` directly.

Stage 1 deliberately contains only the settings the foundation needs.
Ingestion, embedding, vector-store and LLM settings arrive with the
stages that introduce those components.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Any field can be overridden by an environment variable of the same
    name, case-insensitively (``LOG_LEVEL=DEBUG``, ``PORT=9000``).
    """

    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- Identity -----------------------------------------------------
    app_name: str = "LegalDocAI GenAI Service"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"

    # -- Server -------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)

    # -- Logging ------------------------------------------------------
    log_level: str = "INFO"
    # "text" for human-readable local logs, "json" for log aggregators.
    log_format: Literal["text", "json"] = "text"

    # -- Security -----------------------------------------------------
    # Shared secret the MERN backend sends as X-Service-Token.
    # Empty disables the check — acceptable in development only.
    service_token: str = ""
    # Comma-separated origins. The MERN backend calls this service
    # server-to-server, so the default is empty (no browser origins).
    cors_origins: str = ""

    # -- Docs ---------------------------------------------------------
    # Interactive docs are useful while the MERN developer integrates,
    # but should be switched off in production.
    enable_docs: bool = True

    # =================================================================
    # Stage 2 - document ingestion
    # =================================================================

    # -- File handling ------------------------------------------------
    # The only directory documents may be read from. Any referenced path
    # that resolves outside it is rejected (path-traversal protection).
    document_root: str = "./data/documents"
    # Scratch space for uploads and rasterised pages. Always cleaned up.
    temp_dir: str = "./data/tmp"
    allowed_extensions: str = ".pdf,.docx"
    max_file_size_mb: int = Field(default=50, ge=1, le=2048)

    # -- OCR ----------------------------------------------------------
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_dpi: int = Field(default=200, ge=72, le=600)
    # A page yielding fewer extractable characters than this is treated
    # as scanned and routed to OCR.
    ocr_min_chars_per_page: int = Field(default=100, ge=0)
    # OCR output replaces the native text only if it is at least this
    # many times longer - protects against OCR noise beating real text.
    ocr_min_gain_ratio: float = Field(default=1.2, ge=1.0)
    tesseract_cmd: Optional[str] = None

    # -- Cleaning -----------------------------------------------------
    clean_text: bool = True
    strip_running_headers: bool = True
    # A line must appear near the edge of at least this fraction of
    # pages before it is treated as a running header/footer...
    header_footer_min_ratio: float = Field(default=0.6, ge=0.1, le=1.0)
    # ...and the document must have at least this many pages, so a
    # two-page contract never loses a real clause to the heuristic.
    header_footer_min_pages: int = Field(default=3, ge=2)
    strip_page_numbers: bool = True
    # Keep the pre-cleaning text on every page so the original remains
    # recoverable. Costs memory on very large documents.
    keep_raw_text: bool = True
    # Delete temp files older than this at startup. Every temp file is
    # removed in a finally, so this only catches leftovers from a process
    # that was KILLED mid-request - which on a shared volume otherwise
    # accumulates until the disk fills. The age bound is what makes the
    # sweep safe to run while the service is live: a younger file may
    # belong to an in-flight request. 0 disables it.
    temp_file_max_age_hours: float = Field(default=6.0, ge=0.0, le=720.0)

    # -- Section detection --------------------------------------------
    detect_sections: bool = True

    # =================================================================
    # Stage 3 - chunking
    # =================================================================

    # Target tokens per chunk. ~500 suits legal text: large enough to
    # hold a complete clause, small enough that several fit in a prompt.
    chunk_size: int = Field(default=500, ge=32, le=8192)
    # Tokens of the previous chunk repeated at the start of the next, so
    # an answer spanning a boundary is retrievable from either side.
    chunk_overlap: int = Field(default=50, ge=0, le=4096)
    # Chunks below this are merged into a neighbour: a 12-token chunk
    # embeds to a vector that matches everything and informs nothing.
    min_chunk_tokens: int = Field(default=50, ge=1)
    # Hard ceiling as a multiple of chunk_size. Merging and heading
    # attachment may exceed the target, never this.
    chunk_size_tolerance: float = Field(default=1.35, ge=1.0, le=3.0)
    # Never let one chunk span two sections/clauses.
    respect_sections: bool = True
    # Fold undersized chunks into a same-section neighbour.
    merge_small_chunks: bool = True
    # Keep a heading with the text it introduces.
    attach_headings: bool = True

    # Token counter: "heuristic" (no dependency) or "tiktoken" (exact,
    # if installed). Stage 4 may install the embedding model's own.
    tokenizer: Literal["heuristic", "tiktoken"] = "heuristic"
    tiktoken_encoding: str = "cl100k_base"

    # =================================================================
    # Stage 4 - embeddings and vector storage
    # =================================================================

    # -- Embedding model ----------------------------------------------
    # sentence_transformers (production) | deterministic (dev/CI only)
    embedding_provider: Literal[
        "sentence_transformers", "deterministic"
    ] = "sentence_transformers"
    # Any Sentence-Transformers model id, or a local directory path.
    # The DIMENSION IS READ FROM THE MODEL - never configured, never
    # assumed. Changing this value requires re-indexing.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    embedding_device: Optional[str] = None          # cpu | cuda | mps
    embedding_cache_dir: Optional[str] = None
    # Unit-length vectors make cosine and inner product agree.
    normalize_embeddings: bool = True
    # Some models ship custom code; opt in explicitly.
    embedding_trust_remote_code: bool = False
    # True: a model that fails to load is fatal. False: fall back to the
    # deterministic encoder and report degraded health. Set true in
    # production, where serving meaningless vectors is worse than being
    # down.
    embedding_strict: bool = False
    # Width of the deterministic encoder. Exists so tests can run at
    # widths other than 768 and catch hard-coded assumptions.
    deterministic_embedding_dim: int = Field(default=384, ge=8, le=4096)

    # -- Vector store -------------------------------------------------
    # Any name in the backend registry: "chroma" (persistent, default) or
    # "memory" (tests only, not persisted). Deliberately a free string
    # rather than a Literal - registering a new backend (Pinecone,
    # Qdrant, pgvector) must not require editing this file, which is the
    # whole point of the registry. An unknown name is rejected by the
    # factory with a message listing what is available.
    vector_backend: str = "chroma"
    vector_persist_dir: str = "./data/vectors"
    vector_collection: str = "legal_chunks"
    vector_metric: Literal["cosine", "ip", "l2"] = "cosine"
    # Chunks embedded per outer batch during indexing - bounds memory on
    # a large document.
    index_batch_size: int = Field(default=64, ge=1, le=1024)
    # Re-indexing a document deletes its existing chunks first, so a
    # shortened document does not leave orphans behind.
    replace_on_reindex: bool = True
    # Skip re-processing a document whose content digest already matches
    # what is indexed under the same id. Re-ingesting an unchanged
    # 500-document corpus measured 33 seconds of extract/chunk/embed to
    # produce byte-identical vectors under identical ids; the check costs
    # one stat, one short read and one filtered fetch. Set false to force
    # a rebuild (after changing CHUNK_SIZE, say).
    skip_unchanged_documents: bool = True
    # Report when another document of the same tenant already holds
    # byte-identical content. Reported, never refused: two copies of one
    # contract filed under two matters is a legitimate thing to do.
    detect_duplicate_documents: bool = True

    # -- Search -------------------------------------------------------
    search_top_k: int = Field(default=10, ge=1, le=200)
    # Drop hits below this similarity. 0 disables the floor; a floor is
    # risky before reranking exists, so the default is off.
    search_min_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # =================================================================
    # Stage 5 - hybrid retrieval
    # =================================================================

    # -- Pipeline shape -----------------------------------------------
    # hybrid (both signals) | semantic (vectors only) | keyword (BM25 only)
    retrieval_mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"
    # Candidates each retriever is asked for BEFORE reranking. Larger is
    # better recall at a cost the cross-encoder pays linearly; ~15-20 is
    # the point where more candidates stop changing the final answer.
    retrieval_candidates: int = Field(default=18, ge=1, le=500)
    # Passages returned after reranking. ~5-8 fits a grounded-answer
    # prompt without burying the model in near-misses.
    retrieval_top_k: int = Field(default=6, ge=1, le=100)
    # Drop final results below this blended score. 0 disables the floor.
    retrieval_min_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # -- Hybrid merging -----------------------------------------------
    # rrf (rank-based, robust to incomparable score scales - default) |
    # weighted (score-based, more controllable once tuned on a corpus)
    fusion_method: Literal["rrf", "weighted"] = "rrf"
    # RRF damping constant. Higher flattens the difference between the
    # top ranks; 60 is the value from the original paper.
    rrf_k: int = Field(default=60, ge=1, le=1000)
    # Relative trust in each signal. Used by both fusion methods; equal
    # values make RRF the textbook, unweighted form.
    semantic_weight: float = Field(default=0.6, ge=0.0, le=10.0)
    keyword_weight: float = Field(default=0.4, ge=0.0, le=10.0)

    # -- Keyword retrieval (BM25) -------------------------------------
    # Chunks scored per query. The BM25 index is built per request (see
    # app/retrieval/keyword.py for why), so this bounds that cost.
    keyword_corpus_limit: int = Field(default=5000, ge=1, le=200_000)
    # Term-frequency saturation and length normalisation. The standard
    # Okapi defaults; exposed because tuning them on a real corpus is
    # legitimate and should not require a code change.
    bm25_k1: float = Field(default=1.5, ge=0.0, le=10.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    # Cache the per-tenant BM25 index instead of rebuilding it on every
    # query. Measured on 20,000 chunks: fetching the corpus and building
    # the index costs ~3.2 s per query to produce ~37 MB of structure.
    # Invalidation is EXACT - keyed on a per-tenant write counter every
    # store write path bumps - so no TTL and no staleness.
    #
    # That counter is per PROCESS. If a second writer shares this store,
    # this process will not see its writes and may serve a stale index:
    # set this false in that deployment. ChromaDB's local client does not
    # support concurrent writers anyway.
    keyword_cache_enabled: bool = True
    # Total chunks held across all cached tenants, evicting
    # least-recently-used. ~1.9 MB per 1,000 chunks, so the default is
    # roughly 95 MB. Bounded by chunks rather than entries: "four
    # tenants" says nothing when one holds 200 documents and another
    # 20,000.
    keyword_cache_max_chunks: int = Field(default=50_000, ge=0, le=5_000_000)

    # -- Candidate ranking boosts -------------------------------------
    # Additive, bounded and recorded per result, so "why was this
    # retrieved" always has an answer.
    # The query named a clause and this chunk is that clause.
    reference_boost: float = Field(default=0.25, ge=0.0, le=1.0)
    # A quoted phrase from the query occurs verbatim.
    phrase_boost: float = Field(default=0.15, ge=0.0, le=1.0)
    # Both retrievers found this passage independently.
    agreement_boost: float = Field(default=0.05, ge=0.0, le=1.0)
    # A query term appears in the section heading.
    heading_boost: float = Field(default=0.05, ge=0.0, le=1.0)

    # -- Duplicate removal --------------------------------------------
    dedupe_enabled: bool = True
    # Jaccard similarity over character shingles above which two
    # passages are the same passage. High on purpose: Stage 3's
    # deliberate chunk overlap must not be collapsed.
    dedupe_threshold: float = Field(default=0.85, ge=0.1, le=1.0)
    dedupe_shingle_size: int = Field(default=5, ge=2, le=20)
    # Collapse the same clause appearing in two different documents.
    # The dropped chunk_ids stay with the survivor, so no citation is
    # lost - set false to keep both passages in the results.
    dedupe_across_documents: bool = True

    # -- Reranking -----------------------------------------------------
    rerank_enabled: bool = True
    # cross_encoder (real reranking) | lexical (no model, fallback only)
    reranker_provider: Literal["cross_encoder", "lexical"] = "cross_encoder"
    # ~90 MB, 6 layers, CPU-friendly. Deliberately small.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = Field(default=16, ge=1, le=256)
    reranker_device: Optional[str] = None            # cpu | cuda | mps
    reranker_max_length: int = Field(default=512, ge=64, le=2048)
    # Passage characters fed to the cross-encoder. Beyond the model's
    # window the tail is truncated anyway; cutting early saves the work.
    rerank_max_chars: int = Field(default=1200, ge=64, le=20000)
    # The reranker's share of the final score; the rest stays with the
    # fused retrieval score, so two retrievers agreeing is never
    # discarded outright - and a fallback reranker cannot own the whole
    # ordering on a deployment where the real model failed to load.
    rerank_weight: float = Field(default=0.85, ge=0.0, le=1.0)
    # True: a cross-encoder that will not load is fatal. False: fall back
    # to the lexical reranker and report degraded health.
    rerank_strict: bool = False

    # =================================================================
    # Stage 6 - RAG context construction
    # =================================================================

    # -- Budget -------------------------------------------------------
    # Tokens the whole context block may occupy, headers included. This
    # is NOT the model's context window: leave room for the system
    # prompt, the question, the conversation and the answer. ~3000 of a
    # 16k window is a deliberate, conservative default.
    context_max_tokens: int = Field(default=3000, ge=64, le=200_000)
    # Hard cap on source count, independent of the budget: ten short
    # passages can fit and still be worse than the best five, because
    # attention is finite even when context is not.
    context_max_sources: int = Field(default=6, ge=1, le=100)
    # Below this the context is reported as thin evidence. A report,
    # never an enforcement - this stage does not decide that a question
    # is unanswerable.
    context_min_sources: int = Field(default=1, ge=0, le=100)
    # Drop passages scoring below this. 0 disables it. Scores are
    # relative to a query's own candidate set, so a floor tuned on one
    # corpus can silently empty the context on another - set it only
    # after measuring your own distribution.
    context_min_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # -- Per-source limits --------------------------------------------
    # Ceiling for one source, so a 2000-token schedule cannot consume
    # the space five clauses would have used. 0 disables it.
    context_max_source_tokens: int = Field(default=700, ge=0, le=100_000)
    # A passage shortened below this is DROPPED instead of truncated:
    # half a termination clause reads as a complete obligation with the
    # wrong conditions attached, which is worse input than no passage.
    context_min_source_tokens: int = Field(default=60, ge=1, le=10_000)
    # Shorten an oversized passage (at a sentence boundary, always
    # declared in the rendered block) rather than dropping it whole.
    context_truncate_long_sources: bool = True

    # -- Content ------------------------------------------------------
    # document  : group by document, strongest document first, reading
    #             order within it - legal text is sequential and
    #             self-referential, so this is the default
    # relevance : strongest passage first, regardless of document
    context_order: Literal["document", "relevance"] = "document"
    # Deduplicate again in the builder. Usually a no-op after Stage 5,
    # but it is what makes the builder correct on passages that did not
    # come through that pipeline.
    context_dedupe_enabled: bool = True
    context_redundancy_threshold: float = Field(default=0.85, ge=0.1, le=1.0)
    # Drop a leading section heading the header line already states.
    context_strip_repeated_heading: bool = True

    # =================================================================
    # Stage 7 - grounded generation
    # =================================================================

    # -- Provider -----------------------------------------------------
    # Any name in the provider registry:
    #   extractive : deterministic sentence selection, NO model, no key.
    #                Offline default. NOT a language model - it cannot
    #                reason or synthesise, and every response says so.
    #   openai     : /chat/completions. Also covers Azure OpenAI,
    #                Together, Groq, vLLM, Ollama and anything else
    #                serving that wire format - point LLM_BASE_URL at it.
    #   anthropic  : the Messages API.
    # A free string rather than a Literal, deliberately: registering a
    # new provider must not require editing this file.
    llm_provider: str = "extractive"
    # Empty uses the provider's own default model.
    llm_model: str = ""
    # Override the provider's endpoint - self-hosted or a gateway.
    llm_base_url: Optional[str] = None

    # -- Credentials --------------------------------------------------
    # READ FROM THE ENVIRONMENT ONLY. Never hard-coded, never accepted
    # from a request, never logged, never returned in a response. Held as
    # SecretStr so an accidental repr() of the settings prints a mask
    # rather than the credential.
    openai_api_key: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = SecretStr("")

    # -- Generation ---------------------------------------------------
    # 0 by default: a legal answer that changes between two identical
    # questions is not a feature, and a non-deterministic pipeline cannot
    # be regression-tested.
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_output_tokens: int = Field(default=800, ge=64, le=32_000)
    llm_timeout_seconds: float = Field(default=30.0, ge=1.0, le=600.0)
    # Extra attempts after an UNPARSEABLE reply. The retry nudges about
    # formatting only - restating the grounding rules would invite a
    # rewritten answer rather than a reformatted one.
    llm_max_retries: int = Field(default=1, ge=0, le=3)
    # True: a provider that cannot be built is fatal. False: fall back to
    # the extractive provider and report degraded health. Set true in
    # production, where serving sentence selection while believing you
    # are serving a language model is worse than being down.
    llm_strict: bool = False
    # House rules appended to the grounding prompt - a jurisdiction note,
    # a required disclaimer. Appended, never substituted.
    llm_extra_instructions: str = ""

    # -- Citation validation ------------------------------------------
    # An answer that asserts something but cites no retrieved source is
    # marked ungrounded, because none of it could be checked.
    generation_require_citations: bool = True
    # Recover a citation the model wrote into its prose ("see Source 3")
    # but left out of the citations list.
    generation_harvest_inline_citations: bool = True
    # Check quoted spans in the answer against the evidence text.
    generation_verify_quotes: bool = True

    # =================================================================
    # Stage 8 - the integration API
    # =================================================================

    # Documents one POST /ingest/batch may carry. The body holds
    # references, never contents, so 500 items is ~100 KB of JSON.
    ingest_batch_max_documents: int = Field(default=500, ge=1, le=5000)
    # Batches processed at once. 1 by default and deliberately: embedding
    # is CPU-bound, so a second concurrent job competes for the same
    # cores and finishes neither sooner. Raise only with evidence.
    ingest_max_concurrent_jobs: int = Field(default=1, ge=1, le=16)
    # Finished jobs kept for polling. In-memory and per-process: a
    # restart loses job STATUS, never indexed data.
    ingest_job_history: int = Field(default=100, ge=1, le=10_000)
    # Characters of inline text POST /ingest will accept. Large documents
    # belong on the shared volume, not in a JSON body.
    ingest_max_text_chars: int = Field(default=2_000_000, ge=1, le=50_000_000)

    # -- Validators ---------------------------------------------------

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        normalized = str(value).strip().upper()
        if normalized not in allowed:
            raise ValueError(
                f"LOG_LEVEL must be one of {sorted(allowed)}, got '{value}'"
            )
        return normalized

    @field_validator("service_token")
    @classmethod
    def _strip_token(cls, value: str) -> str:
        return (value or "").strip()

    # -- Derived ------------------------------------------------------

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def docs_url(self) -> str | None:
        return "/docs" if self.enable_docs else None

    @property
    def openapi_url(self) -> str | None:
        return "/openapi.json" if self.enable_docs else None

    @property
    def auth_enabled(self) -> bool:
        return bool(self.service_token)

    # -- Stage 2 derived ----------------------------------------------

    @property
    def allowed_extension_list(self) -> List[str]:
        return [
            e.strip().lower()
            for e in self.allowed_extensions.split(",")
            if e.strip()
        ]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def document_root_path(self) -> Path:
        return Path(self.document_root).expanduser().resolve()

    @property
    def temp_dir_path(self) -> Path:
        return Path(self.temp_dir).expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cache — used by tests after mutating the environment."""
    get_settings.cache_clear()


__all__ = ["Settings", "get_settings", "reset_settings_cache"]
