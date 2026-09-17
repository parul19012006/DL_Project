"""Keyword retrieval — Okapi BM25 over the tenant's chunks.

**Why keyword retrieval at all, when we have embeddings.** Because
embeddings are lossy in exactly the places legal text is precise. A
384-dimension vector of a clause does not reliably encode that the clause
is numbered 7.3, that the cap is "USD 50,000", that the governing law is
"Karnataka", or that the defined term is "Permitted Recipient". Ask for
any of those and vector search returns passages that are *about* the
right topic while the passage containing the actual number sits at rank
40. BM25 finds it at rank 1, because it matches the literal string.

The converse is just as true — BM25 cannot match "how much warning
before the contract ends" to a clause headed TERMINATION — which is why
neither signal is dropped and both are fused.

**Why BM25 and not TF-IDF cosine.** Same cost, strictly better
behaviour: BM25 saturates term frequency (the tenth occurrence of
"Vendor" adds almost nothing over the ninth) and normalises for document
length (a 500-token clause does not outrank a 120-token one merely by
being longer). Both matter here, because chunks vary in length and legal
text repeats defined terms constantly.

**The index is cached per tenant, and invalidated exactly.** Earlier
stages rebuilt it on every query, on the reasoning that a cache with
heuristic invalidation would eventually retrieve passages that no longer
exist — citing a document the user has deleted. That reasoning was right
about the danger and wrong about the price. Measured on a 500-document
corpus:

===============  ==========  ==========  ===========
corpus (chunks)  fetch (ms)  build (ms)  index (MB)
===============  ==========  ==========  ===========
1,000                   137         116         1.9
5,000                   685         632         9.3
20,000                  669       2,580        37.2
===============  ==========  ==========  ===========

At the scale this service is built for — 500 legal documents in one
tenant, roughly 20,000 chunks — that is **3.2 seconds of the same work on
every single query**, to produce a structure costing 37 MB. Caching it is
not premature optimisation; rebuilding it was the optimisation, and it
was the wrong one.

What makes the cache safe is that invalidation is **exact, not
heuristic**: :meth:`VectorStore.version` is a counter every write path in
every adapter bumps, so an entry keyed on (tenant, version) cannot serve
a corpus that has changed. No TTL, no guessing.

Two honest bounds on that. The counter lives in *this process*, so a
second writer against the same store would not invalidate this one's
cache — which is no worse than ChromaDB's local client already supports,
and ``KEYWORD_CACHE_ENABLED=false`` turns the cache off for anyone who
needs it. And the cache holds only tenant-only queries, evicting
least-recently-used entries to stay under ``KEYWORD_CACHE_MAX_CHUNKS``;
a filtered query looks at a smaller corpus anyway and is built fresh.

**Why not the ``rank_bm25`` package.** It is a fine library, but it is
~40 lines of arithmetic wrapped around a tokenizer we would have to
override anyway — clause numbers like "4.2" must survive tokenisation,
and the term-family mapping in :mod:`app.retrieval.query` has to apply
to the corpus and the query identically. Owning the loop keeps those two
things in one place and adds no dependency.

"""

from __future__ import annotations

import math
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from app.config import Settings, get_settings
from app.logging_config import get_logger
from app.models.chunk import Chunk
from app.retrieval.base import SOURCE_KEYWORD, Candidate, Retriever, scale_to_max
from app.retrieval.query import ProcessedQuery, tokenize
from app.vectorstore.base import SearchFilter, VectorStore

logger = get_logger(__name__)


@dataclass
class BM25Index:
    """An in-memory Okapi BM25 index over a list of chunks.

    ``k1`` controls term-frequency saturation, ``b`` controls
    length normalisation. The defaults (1.5 / 0.75) are the standard
    ones and are a reasonable starting point for prose; both are
    configurable because tuning them on a real corpus is a legitimate
    thing to want to do without editing code.
    """

    chunks: List[Chunk] = field(default_factory=list)
    k1: float = 1.5
    b: float = 0.75
    #: True when the corpus hit ``KEYWORD_CORPUS_LIMIT`` and was cut.
    #: Carried on the index itself so a cached one reports it exactly as
    #: a freshly built one does — deriving it from the size afterwards
    #: gets the boundary case wrong.
    truncated: bool = False

    # -- built state ---------------------------------------------------
    doc_terms: List[Counter] = field(default_factory=list, repr=False)
    doc_lengths: List[int] = field(default_factory=list, repr=False)
    document_frequency: Dict[str, int] = field(default_factory=dict, repr=False)
    average_length: float = 0.0

    def __post_init__(self) -> None:
        self._build()

    # -- construction --------------------------------------------------

    def _build(self) -> None:
        self.doc_terms = []
        self.doc_lengths = []
        self.document_frequency = {}

        for chunk in self.chunks:
            # The section heading is prepended so a query naming a clause
            # ("termination") matches a chunk whose body is the clause
            # text under a TERMINATION heading. The heading is part of
            # what the passage is about, and it is cheap to include.
            text = f"{chunk.section or ''}\n{chunk.text}"
            terms = Counter(tokenize(text))
            self.doc_terms.append(terms)
            self.doc_lengths.append(sum(terms.values()))
            for term in terms:
                self.document_frequency[term] = (
                    self.document_frequency.get(term, 0) + 1
                )

        total = sum(self.doc_lengths)
        self.average_length = (total / len(self.doc_lengths)) if self.doc_lengths else 0.0

    # -- scoring -------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.chunks)

    def idf(self, term: str) -> float:
        """Inverse document frequency, in the form that cannot go negative.

        The textbook Robertson/Sparck-Jones IDF turns negative for a term
        appearing in more than half the corpus, which lets a very common
        term *subtract* from a score. The ``ln(1 + ...)`` variant used
        here stays non-negative — important on a small tenant corpus
        where "agreement" genuinely is in most chunks.
        """
        n = self.size
        if n == 0:
            return 0.0
        df = self.document_frequency.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score_document(self, index: int, terms: Sequence[str]) -> float:
        if not terms or self.average_length <= 0:
            return 0.0
        frequencies = self.doc_terms[index]
        length = self.doc_lengths[index]
        score = 0.0
        for term in terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (
                1.0 - self.b + self.b * (length / self.average_length)
            )
            score += self.idf(term) * (frequency * (self.k1 + 1.0)) / denominator
        return score

    def search(
        self, terms: Sequence[str], top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """``[(chunk position, score), ...]`` best first, zeros dropped.

        Ties break on the chunk's position, which is stable because
        :meth:`VectorStore.fetch` returns a deterministic order. Without
        that, two runs of the same query could return different passages.
        """
        if not terms or not self.chunks:
            return []
        # Duplicate query terms would double-count; BM25 is defined over
        # the set of query terms for our purposes.
        unique = list(dict.fromkeys(terms))
        scored = [
            (position, self.score_document(position, unique))
            for position in range(self.size)
        ]
        scored = [pair for pair in scored if pair[1] > 0.0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[: max(0, int(top_k))]


@dataclass
class _CacheEntry:
    version: int
    index: BM25Index

    @property
    def size(self) -> int:
        return self.index.size


class BM25Cache:
    """Per-tenant BM25 indexes, bounded and exactly invalidated.

    Bounded by total *chunks* rather than by entry count, because that is
    what actually predicts the memory: roughly 1.9 MB per 1,000 chunks,
    measured. A limit of "four tenants" says nothing when one holds 200
    documents and another holds 20,000.
    """

    def __init__(self, max_chunks: int = 50_000) -> None:
        self.max_chunks = max(0, int(max_chunks))
        self._entries: "OrderedDict[str, _CacheEntry]" = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    @property
    def cached_chunks(self) -> int:
        with self._lock:
            return sum(entry.size for entry in self._entries.values())

    def get(self, key: str, version: int) -> Optional[BM25Index]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.version != version:
                self.misses += 1
                if entry is not None:
                    # The corpus moved on. Drop it rather than let a
                    # stale index sit there occupying the budget.
                    del self._entries[key]
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry.index

    def put(self, key: str, version: int, index: BM25Index) -> None:
        if self.max_chunks <= 0 or index.size > self.max_chunks:
            # A single corpus larger than the whole budget is not worth
            # evicting everything else for.
            return
        with self._lock:
            self._entries[key] = _CacheEntry(version=version, index=index)
            self._entries.move_to_end(key)
            while self._entries and self.cached_chunks > self.max_chunks:
                self._entries.popitem(last=False)

    def invalidate(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key is None:
                self._entries.clear()
            else:
                self._entries.pop(key, None)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "cached_chunks": self.cached_chunks,
                "max_chunks": self.max_chunks,
                "hits": self.hits,
                "misses": self.misses,
            }


_cache: Optional[BM25Cache] = None
_cache_lock = threading.Lock()


def get_bm25_cache(settings: Optional[Settings] = None) -> BM25Cache:
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                settings = settings or get_settings()
                _cache = BM25Cache(settings.keyword_cache_max_chunks)
    return _cache


def set_bm25_cache(cache: Optional[BM25Cache]) -> None:
    global _cache
    _cache = cache


class KeywordRetriever(Retriever):
    """BM25 over the chunks the filter selects.

    The corpus comes from :meth:`VectorStore.fetch`, which applies the
    same :class:`SearchFilter` the semantic side uses — so tenant
    isolation and metadata filtering hold identically on both paths, by
    construction rather than by remembering.
    """

    name = "keyword"

    def __init__(
        self,
        store: VectorStore,
        settings: Optional[Settings] = None,
    ) -> None:
        self.store = store
        self.settings = settings or get_settings()
        #: Size of the corpus the last query scored, for observability.
        self.last_corpus_size: int = 0
        #: True when the corpus hit ``KEYWORD_CORPUS_LIMIT`` and was cut.
        self.last_corpus_truncated: bool = False
        #: True when the last query reused a cached index.
        self.last_cache_hit: bool = False

    def build_index(self, filters: SearchFilter) -> BM25Index:
        """The tenant's BM25 index, from cache when it is still valid."""
        cacheable = (
            self.settings.keyword_cache_enabled and filters.is_tenant_only
        )
        self.last_cache_hit = False

        if cacheable:
            cache = get_bm25_cache(self.settings)
            key = self._cache_key(filters.tenant_id)
            version = self.store.version(filters.tenant_id)
            cached = cache.get(key, version)
            if cached is not None:
                self.last_cache_hit = True
                self.last_corpus_size = cached.size
                self.last_corpus_truncated = cached.truncated
                return cached

            index = self._build(filters)
            cache.put(key, version, index)
            return index

        return self._build(filters)

    def _cache_key(self, tenant_id: str) -> str:
        """Tenant plus every setting that changes what the index *is*.

        The corpus limit decides how much of the tenant was indexed, and
        k1/b decide how it scores. Keying on the tenant alone would let a
        retriever configured one way serve an index built another — which
        is exactly what the test suite caught when this key was just the
        tenant id.
        """
        return (
            f"{tenant_id}|{self.settings.keyword_corpus_limit}"
            f"|{self.settings.bm25_k1}|{self.settings.bm25_b}"
        )

    def _build(self, filters: SearchFilter) -> BM25Index:
        limit = int(self.settings.keyword_corpus_limit)
        # Fetch one extra so a corpus exactly at the limit is not
        # reported as truncated.
        chunks = self.store.fetch(filters, limit=limit + 1)
        self.last_corpus_truncated = len(chunks) > limit
        if self.last_corpus_truncated:
            chunks = chunks[:limit]
            logger.warning(
                "Keyword corpus for tenant %s hit KEYWORD_CORPUS_LIMIT (%d); "
                "BM25 scored only part of the corpus. Raise the limit or "
                "narrow the query with document_ids.",
                filters.tenant_id,
                limit,
            )
        self.last_corpus_size = len(chunks)
        return BM25Index(
            chunks=chunks,
            k1=self.settings.bm25_k1,
            b=self.settings.bm25_b,
            truncated=self.last_corpus_truncated,
        )

    def retrieve(
        self,
        query: ProcessedQuery,
        filters: SearchFilter,
        limit: int,
    ) -> List[Candidate]:
        if not query.has_terms or limit <= 0:
            self.last_corpus_size = 0
            return []

        index = self.build_index(filters)
        if index.size == 0:
            return []

        results = index.search(query.terms, top_k=limit)
        if not results:
            return []

        # BM25 is unbounded and corpus-relative, so it is normalised to
        # [0, 1] across this run before it meets the semantic score.
        # ``keyword_score_raw`` keeps the real number.
        raw = [score for _, score in results]
        scaled = scale_to_max(raw)

        candidates: List[Candidate] = []
        for position, ((chunk_index, score), normalised) in enumerate(
            zip(results, scaled), start=1
        ):
            candidate = Candidate(
                chunk=index.chunks[chunk_index],
                keyword_score=float(normalised),
                keyword_score_raw=float(score),
                keyword_rank=position,
                sources={SOURCE_KEYWORD},
            )
            candidates.append(candidate)

        logger.debug(
            "Keyword retrieval: %d candidate(s) from a corpus of %d",
            len(candidates),
            index.size,
        )
        return candidates


def build_keyword_retriever(
    store: Optional[VectorStore] = None, settings: Optional[Settings] = None
) -> KeywordRetriever:
    from app.vectorstore.factory import get_vector_store

    settings = settings or get_settings()
    return KeywordRetriever(store=store or get_vector_store(settings), settings=settings)


__all__ = [
    "BM25Index",
    "BM25Cache",
    "KeywordRetriever",
    "build_keyword_retriever",
    "get_bm25_cache",
    "set_bm25_cache",
]
