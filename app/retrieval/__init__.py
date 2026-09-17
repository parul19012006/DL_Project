"""Stage 5 — hybrid retrieval.

    query -> preprocess -> semantic + keyword -> fuse -> rank
          -> deduplicate -> rerank -> relevant chunks

Each stage is its own module so that any one of them can be replaced,
tested or reasoned about alone:

``query``     normalise the question; extract clause references and phrases
``semantic``  vector similarity over the Stage 4 index
``keyword``   Okapi BM25 over the same tenant-filtered chunks
``fusion``    reciprocal rank fusion (or weighted) over the two lists
``rank``      metadata boosts fusion cannot see, applied explainably
``dedup``     shingle/Jaccard near-duplicate collapsing
``rerank``    lightweight cross-encoder, with a lexical fallback
``pipeline``  the orchestration, and nothing else

This package retrieves passages. It does not generate answers — the LLM,
context building and citation arrive in Stage 6.
"""

from app.retrieval.base import (
    Candidate,
    RerankerUnavailableError,
    RetrievalError,
    Retriever,
    StageCounts,
)
from app.retrieval.dedup import jaccard, remove_duplicates, shingles
from app.retrieval.fusion import (
    agreement_ratio,
    fuse,
    reciprocal_rank_fusion,
    weighted_fusion,
)
from app.retrieval.keyword import BM25Index, KeywordRetriever
from app.retrieval.pipeline import (
    HYBRID,
    KEYWORD_ONLY,
    MODES,
    SEMANTIC_ONLY,
    RetrievalOutcome,
    RetrievalPipeline,
    build_pipeline,
    get_pipeline,
    retrieve,
    set_pipeline,
)
from app.retrieval.query import ProcessedQuery, preprocess_query, tokenize
from app.retrieval.rank import rank_candidates
from app.retrieval.rerank import (
    CrossEncoderReranker,
    LexicalReranker,
    Reranker,
    apply_reranking,
    build_reranker,
    get_reranker,
    set_reranker,
)
from app.retrieval.semantic import SemanticRetriever

__all__ = [
    "Candidate",
    "Retriever",
    "StageCounts",
    "RetrievalError",
    "RerankerUnavailableError",
    "ProcessedQuery",
    "preprocess_query",
    "tokenize",
    "SemanticRetriever",
    "KeywordRetriever",
    "BM25Index",
    "fuse",
    "reciprocal_rank_fusion",
    "weighted_fusion",
    "agreement_ratio",
    "rank_candidates",
    "remove_duplicates",
    "shingles",
    "jaccard",
    "Reranker",
    "CrossEncoderReranker",
    "LexicalReranker",
    "apply_reranking",
    "build_reranker",
    "get_reranker",
    "set_reranker",
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
