"""Semantic retrieval — the vector-similarity signal.

Thin by design. Stage 4 already owns embedding and the vector store;
duplicating any of that here would give the pipeline two places where
the embedding model and the index can disagree. This module's whole job
is: encode the query, ask the store, return :class:`Candidate` objects.

What it *does* add is the candidate pool size. Stage 4's ``search()``
returns ``SEARCH_TOP_K`` results because that is what a caller asking for
similarity wants. Stage 5 asks for far more than it will return —
``RETRIEVAL_CANDIDATES`` — because the reranker downstream can only
promote a passage that retrieval handed it.

Semantic retrieval finds passages that *mean* the same thing as the
query even when they share no words: "how much warning before the
contract ends" matches a clause headed TERMINATION that never uses the
word "warning". That is exactly what keyword retrieval cannot do, and
it is why both exist.
"""

from __future__ import annotations

from typing import List, Optional

from app.embeddings.base import EmbeddingModel
from app.logging_config import get_logger
from app.retrieval.base import Candidate, Retriever
from app.retrieval.query import ProcessedQuery
from app.vectorstore.base import SearchFilter, VectorStore

logger = get_logger(__name__)


class SemanticRetriever(Retriever):
    """Vector search over the tenant's indexed chunks."""

    name = "semantic"

    def __init__(self, model: EmbeddingModel, store: VectorStore) -> None:
        self.model = model
        self.store = store
        #: Milliseconds spent encoding the last query, for the response.
        self.last_embed_ms: int = 0

    def retrieve(
        self,
        query: ProcessedQuery,
        filters: SearchFilter,
        limit: int,
    ) -> List[Candidate]:
        if query.is_empty or limit <= 0:
            return []

        import time

        mark = time.perf_counter()
        # The *normalized* form, not the keyword form: encoders are
        # trained on natural language and a stripped bag of terms embeds
        # to a worse vector than the question as asked.
        vector = self.model.embed_query(query.normalized)
        self.last_embed_ms = int((time.perf_counter() - mark) * 1000)

        hits = self.store.search(vector, filters, top_k=limit)
        candidates = [Candidate.from_hit(hit) for hit in hits]
        for position, candidate in enumerate(candidates, start=1):
            # Trust our own ordering rather than the backend's rank
            # field: min_score filtering upstream can leave gaps.
            candidate.semantic_rank = position

        logger.debug(
            "Semantic retrieval: %d candidate(s) for tenant %s",
            len(candidates),
            filters.tenant_id,
        )
        return candidates


def build_semantic_retriever(
    model: Optional[EmbeddingModel] = None,
    store: Optional[VectorStore] = None,
    settings=None,
) -> SemanticRetriever:
    from app.config import get_settings
    from app.embeddings.factory import get_embedding_model
    from app.vectorstore.factory import get_vector_store

    settings = settings or get_settings()
    return SemanticRetriever(
        model=model or get_embedding_model(settings),
        store=store or get_vector_store(settings),
    )


__all__ = ["SemanticRetriever", "build_semantic_retriever"]
