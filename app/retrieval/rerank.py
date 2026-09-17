"""Reranking — the last and most discriminating stage.

Retrieval is recall-oriented and cheap: a bi-encoder compares two
vectors that were computed *without knowing about each other*, so the
query's vector cannot attend to the passage's words. That is what makes
it fast enough to search a whole index, and it is also why its ordering
inside the top twenty is unreliable.

A cross-encoder reads the query and the passage **together** in one
forward pass and emits a single relevance score. It cannot be used to
search an index — scoring every chunk against every query is
prohibitive — but scoring ~18 candidates is a handful of milliseconds on
CPU, and it is markedly better than a bi-encoder at the distinction that
matters here: is this passage *about* the question, or merely about the
same subject matter? "Either party may terminate for convenience on 60
days' notice" and "Termination shall not affect accrued rights" are
neighbours in vector space and worlds apart as answers to "how much
notice do I need to give?".

**The model.** ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — about 90 MB,
6 layers, CPU-friendly. Deliberately not a large reranker: the
instruction was an inexpensive cross-encoder, and this one costs
milliseconds per candidate rather than seconds.

**The fallback, and why it exists.** The model has to be downloaded or
baked into the image. In an environment without it — a first boot with
no network, an air-gapped deployment, this development sandbox —
:class:`LexicalReranker` takes over: a query-term coverage score with an
exact-phrase bonus. It is *not* a cross-encoder and does not pretend to
be; it exists so the pipeline keeps its shape and the service keeps
answering, and ``/health`` reports the degradation explicitly rather than
letting a silently worse ranking pass for the real thing. Setting
``RERANK_STRICT=true`` makes a missing model fatal instead, which is the
right choice in production.

**The final score is a blend, not a replacement.** ``RERANK_WEIGHT``
(default 0.85) mixes the reranker's judgement with the fused retrieval
score. Handing the ordering entirely to the reranker throws away the
agreement signal that hybrid retrieval just computed, and — more
practically — makes the fallback reranker's opinion the only one that
counts on a deployment where the real model is missing.
"""

from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from app.config import Settings, get_settings
from app.logging_config import get_logger
from app.retrieval.base import (
    Candidate,
    RerankerUnavailableError,
    scale_to_max,
)
from app.retrieval.query import ProcessedQuery, tokenize

logger = get_logger(__name__)

CROSS_ENCODER = "cross_encoder"
LEXICAL = "lexical"
PROVIDERS = (CROSS_ENCODER, LEXICAL)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
    exp = math.exp(max(value, -60.0))
    return exp / (1.0 + exp)


# =====================================================================
# Interface
# =====================================================================


class Reranker(ABC):
    """Scores (query, passage) pairs. Higher is more relevant."""

    #: Short provider name, reported by /health.
    name: str = "reranker"
    #: Model identifier, or a description for a model-free reranker.
    model_id: str = ""
    #: False for anything that is not a genuine cross-encoder, so the
    #: rest of the system can be honest about what ran.
    is_cross_encoder: bool = False

    @abstractmethod
    def score(self, query: ProcessedQuery, passages: Sequence[str]) -> List[float]:
        """One score per passage, in the same order."""

    def describe(self) -> str:
        return f"{self.name}:{self.model_id}" if self.model_id else self.name


# =====================================================================
# Cross-encoder
# =====================================================================


class CrossEncoderReranker(Reranker):
    """Sentence-Transformers ``CrossEncoder``.

    The model is loaded lazily on first use rather than at construction,
    so building the pipeline never blocks on a download and a
    misconfigured reranker degrades one request instead of preventing
    startup.
    """

    name = CROSS_ENCODER
    is_cross_encoder = True

    def __init__(
        self,
        model_id: str,
        batch_size: int = 16,
        max_chars: int = 1200,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        self.model_id = model_id
        self.batch_size = max(1, int(batch_size))
        self.max_chars = max(64, int(max_chars))
        self.device = device
        self.cache_dir = cache_dir
        self.max_length = int(max_length)
        self._model = None
        self._lock = threading.RLock()

    # -- loading -------------------------------------------------------

    def load(self):
        """Load the model, raising :class:`RerankerUnavailableError`."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import CrossEncoder
            except Exception as exc:  # pragma: no cover - import guard
                raise RerankerUnavailableError(
                    "sentence-transformers is not installed, so the "
                    "cross-encoder reranker cannot be used"
                ) from exc

            started = time.perf_counter()
            try:
                kwargs = {"max_length": self.max_length}
                if self.device:
                    kwargs["device"] = self.device
                if self.cache_dir:
                    kwargs["cache_folder"] = self.cache_dir
                self._model = CrossEncoder(self.model_id, **kwargs)
            except TypeError:
                # Older/newer releases disagree on cache_folder; retry
                # with the arguments every version accepts.
                try:
                    self._model = CrossEncoder(
                        self.model_id, max_length=self.max_length
                    )
                except Exception as exc:
                    raise RerankerUnavailableError(
                        f"Could not load cross-encoder '{self.model_id}': {exc}"
                    ) from exc
            except Exception as exc:
                raise RerankerUnavailableError(
                    f"Could not load cross-encoder '{self.model_id}': {exc}"
                ) from exc

            logger.info(
                "Cross-encoder reranker loaded: %s (%d ms)",
                self.model_id,
                int((time.perf_counter() - started) * 1000),
            )
            return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # -- scoring -------------------------------------------------------

    def score(self, query: ProcessedQuery, passages: Sequence[str]) -> List[float]:
        if not passages:
            return []
        model = self.load()

        # The query the user actually asked — a cross-encoder is a
        # language model and wants the sentence, not a term list.
        text = query.normalized
        pairs = [[text, (p or "")[: self.max_chars]] for p in passages]

        raw = model.predict(pairs, batch_size=self.batch_size)
        return self._to_probabilities(raw)

    @staticmethod
    def _to_probabilities(raw) -> List[float]:
        """Map model output to [0, 1].

        Decided by the output's *shape*, never by inspecting the values:
        a data-dependent rule would apply a sigmoid to one query and not
        the next, making scores incomparable between requests.

        * one score per pair (num_labels=1) -> sigmoid
        * two logits per pair               -> softmax, positive class
        """
        try:
            values = raw.tolist()
        except AttributeError:
            values = list(raw)

        if values and isinstance(values[0], (list, tuple)):
            out: List[float] = []
            for row in values:
                row = [float(v) for v in row]
                top = max(row)
                exps = [math.exp(v - top) for v in row]
                total = sum(exps) or 1.0
                # Convention: the last column is the positive class.
                out.append(exps[-1] / total)
            return out

        return [_sigmoid(float(v)) for v in values]


# =====================================================================
# Lexical fallback
# =====================================================================


class LexicalReranker(Reranker):
    """Model-free relevance scoring. A fallback, not a cross-encoder.

    Three components, all cheap:

    * **coverage** — the share of the query's distinct content terms that
      appear in the passage at all. Coverage rather than frequency:
      a passage mentioning every term once answers a question better than
      one repeating a single term ten times, and BM25 already rewarded
      the frequency.
    * **phrase** — a quoted phrase occurring verbatim.
    * **density** — how tightly the matched terms cluster. The terms of
      an answer tend to sit in one sentence; the terms of a coincidence
      are scattered across a page.
    """

    name = LEXICAL
    model_id = "lexical-coverage (fallback, not a cross-encoder)"
    is_cross_encoder = False

    def __init__(self, window: int = 60) -> None:
        self.window = max(10, int(window))

    def score(self, query: ProcessedQuery, passages: Sequence[str]) -> List[float]:
        terms = list(dict.fromkeys(query.terms))
        phrases = [p for p in query.phrases if p]
        if not terms and not phrases:
            return [0.0 for _ in passages]

        out: List[float] = []
        for passage in passages:
            tokens = tokenize(passage or "")
            positions = {
                term: [i for i, t in enumerate(tokens) if t == term]
                for term in terms
            }
            present = [term for term, hits in positions.items() if hits]

            coverage = (len(present) / len(terms)) if terms else 0.0
            density = self._density(positions, present)
            phrase = 0.0
            if phrases:
                lowered = (passage or "").lower()
                matched = sum(1 for p in phrases if p in lowered)
                phrase = matched / len(phrases)

            out.append(
                round(0.65 * coverage + 0.2 * density + 0.15 * phrase, 6)
            )
        return out

    def _density(self, positions, present) -> float:
        """1.0 when every matched term sits inside one window."""
        if len(present) < 2:
            return 1.0 if present else 0.0
        firsts = [positions[term][0] for term in present]
        span = max(firsts) - min(firsts)
        if span <= 0:
            return 1.0
        return max(0.0, min(1.0, self.window / span))


# =====================================================================
# Applying a reranker to candidates
# =====================================================================


def apply_reranking(
    candidates: Sequence[Candidate],
    query: ProcessedQuery,
    reranker: Reranker,
    top_k: int,
    weight: float = 0.85,
) -> List[Candidate]:
    """Score, blend with the retrieval score, order, truncate, re-rank.

    ``weight`` is the reranker's share of the final score; the remainder
    stays with the fused-and-boosted retrieval score, so an agreement
    between two independent retrievers is never discarded outright.
    """
    items = list(candidates)
    if not items:
        return []

    scores = reranker.score(query, [c.text for c in items])
    if len(scores) != len(items):  # pragma: no cover - defensive
        raise RerankerUnavailableError(
            f"{reranker.describe()} returned {len(scores)} scores for "
            f"{len(items)} candidates"
        )

    weight = max(0.0, min(1.0, float(weight)))
    # Both signals are min-maxed across this candidate set so the blend
    # is between comparable quantities: a cross-encoder's probabilities
    # can all sit in [0.9, 0.95] while the retrieval scores span [0, 1].
    rerank_norm = scale_to_max([float(s) for s in scores])
    retrieval_norm = scale_to_max([c.rank_score for c in items])

    for candidate, raw, rerank_value, retrieval_value in zip(
        items, scores, rerank_norm, retrieval_norm
    ):
        candidate.rerank_score = float(raw)
        candidate.final_score = (
            weight * rerank_value + (1.0 - weight) * retrieval_value
        )

    # Pinned candidates — the exact clause the user named — sort first
    # whatever the reranker thought. A relevance model is a good judge of
    # "is this passage about the question"; it is not entitled to
    # overrule an explicit citation, and asked for "clause 8" it will
    # happily rank the clause that *mentions* clause 8 above clause 8.
    items.sort(key=lambda c: (not c.pinned, -c.final_score, c.chunk_id))
    if top_k and top_k > 0:
        items = items[:top_k]
    for position, candidate in enumerate(items, start=1):
        candidate.rank = position
    return items


# =====================================================================
# Factory
# =====================================================================

_reranker: Optional[Reranker] = None
_factory_lock = threading.Lock()


def build_reranker(settings: Optional[Settings] = None) -> Reranker:
    """Build the configured reranker, honouring ``RERANK_STRICT``."""
    settings = settings or get_settings()
    provider = (settings.reranker_provider or LEXICAL).strip().lower()

    if provider == LEXICAL:
        return LexicalReranker()

    if provider != CROSS_ENCODER:
        raise RerankerUnavailableError(
            f"Unknown RERANKER_PROVIDER '{provider}'. "
            f"Available: {', '.join(PROVIDERS)}"
        )

    reranker = CrossEncoderReranker(
        model_id=settings.reranker_model,
        batch_size=settings.reranker_batch_size,
        max_chars=settings.rerank_max_chars,
        device=settings.reranker_device or settings.embedding_device,
        cache_dir=settings.embedding_cache_dir,
        max_length=settings.reranker_max_length,
    )

    try:
        reranker.load()
        return reranker
    except RerankerUnavailableError as exc:
        if settings.rerank_strict:
            # In production, ranking silently worse is a bigger problem
            # than failing loudly.
            raise
        logger.warning(
            "Cross-encoder '%s' is unavailable (%s); falling back to the "
            "lexical reranker. Retrieval will still work, but ordering "
            "quality is lower and /health reports this.",
            settings.reranker_model,
            exc,
        )
        return LexicalReranker()


def get_reranker(settings: Optional[Settings] = None) -> Reranker:
    """Process-wide reranker singleton — the model is loaded once."""
    global _reranker
    if _reranker is None:
        with _factory_lock:
            if _reranker is None:
                _reranker = build_reranker(settings)
    return _reranker


def set_reranker(reranker: Optional[Reranker]) -> None:
    """Install a reranker, or ``None`` to rebuild from settings."""
    global _reranker
    with _factory_lock:
        _reranker = reranker


__all__ = [
    "Reranker",
    "CrossEncoderReranker",
    "LexicalReranker",
    "apply_reranking",
    "build_reranker",
    "get_reranker",
    "set_reranker",
    "CROSS_ENCODER",
    "LEXICAL",
    "PROVIDERS",
]
