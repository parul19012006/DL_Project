"""A deterministic, model-free provider — the offline default.

**Why this exists.** The same reason the deterministic embedder and the
lexical reranker exist in earlier stages: the pipeline has to be runnable
and testable without a network, a key or a bill. It also makes the whole
Stage 7 path — prompt, provider, parse, validate, respond — exercisable
in CI on every commit, which a mocked provider would not.

**What it is.** An extractive summariser. It reads the evidence block
back out of the user message, scores each sentence against the question's
terms, and returns the best few sentences verbatim with citations to the
sources they came from. Every word it emits is copied from the evidence.

**What it is not.** It is not a language model, it does not write prose,
and it cannot reason, compare or synthesise. ``is_language_model`` is
False, ``/health`` reports it, and every answer it produces is marked
``"model": "extractive"`` in the response — no part of the system
presents its output as a generated answer.

Reading the evidence block back is deliberate rather than a shortcut: it
holds Stage 6's rendered format to being machine-readable, and it keeps
the provider interface honest, since a provider is given text and nothing
else.
"""

from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Optional, Tuple

from app.chunking.splitters import split_sentences
from app.generation.base import LLMProvider, LLMRequest, LLMResponse
from app.generation.prompts import EVIDENCE_LABEL, NO_EVIDENCE, QUESTION_LABEL
from app.logging_config import get_logger
from app.retrieval.query import tokenize

logger = get_logger(__name__)

_SOURCE_RE = re.compile(r"^Source (\d+):", re.MULTILINE)
_SEPARATOR = "\n\n---\n\n"

INSUFFICIENT = (
    "The retrieved passages do not contain anything that answers this "
    "question. No answer can be given from the available evidence."
)


class ExtractiveProvider(LLMProvider):
    """Selects the most relevant sentences from the evidence, verbatim."""

    name = "extractive"
    model = "extractive"
    is_language_model = False
    requires_key = False

    def __init__(self, max_sentences: int = 4, max_sources: int = 3) -> None:
        self.max_sentences = max(1, int(max_sentences))
        self.max_sources = max(1, int(max_sources))

    def describe(self) -> str:
        return "extractive (deterministic sentence selection, not a language model)"

    def generate(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        question, blocks = parse_user_message(request.user)
        terms = set(tokenize(question))

        scored: List[Tuple[float, int, int, str]] = []
        for number, text in blocks.items():
            for position, sentence in enumerate(split_sentences(text)):
                sentence_terms = set(tokenize(sentence))
                if not sentence_terms:
                    continue
                overlap = len(terms & sentence_terms)
                if not overlap:
                    continue
                # Coverage of the question, lightly preferring earlier
                # sentences: in a clause, the operative statement usually
                # comes before its qualifications.
                score = overlap / max(1, len(terms)) - position * 0.001
                scored.append((score, number, position, sentence))

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        chosen = self._choose(scored)

        if not chosen:
            payload = {
                "answer": INSUFFICIENT,
                "citations": [],
                "insufficient_evidence": True,
                "interpretation": "",
            }
        else:
            payload = {
                "answer": " ".join(sentence for _, sentence in chosen),
                "citations": [
                    {"source": number}
                    for number in sorted({number for number, _ in chosen})
                ],
                "insufficient_evidence": False,
                "interpretation": "",
            }

        return LLMResponse(
            text=json.dumps(payload),
            model=self.model,
            provider=self.name,
            finish_reason="stop",
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    def _choose(
        self, scored: List[Tuple[float, int, int, str]]
    ) -> List[Tuple[int, str]]:
        """Best sentences, capped per answer and by distinct source.

        Presented in source order so the result reads in the order the
        documents state it, matching Stage 6's ordering rather than
        fighting it.
        """
        picked: List[Tuple[int, int, str]] = []
        sources: List[int] = []
        for _, number, position, sentence in scored:
            if number not in sources:
                if len(sources) >= self.max_sources:
                    continue
                sources.append(number)
            picked.append((number, position, sentence))
            if len(picked) >= self.max_sentences:
                break

        picked.sort(key=lambda item: (item[0], item[1]))
        return [(number, sentence) for number, position, sentence in picked]


def parse_user_message(message: str) -> Tuple[str, Dict[int, str]]:
    """``(question, {source number: passage text})``.

    Tolerant of a missing label or an empty evidence section — a provider
    that raises on an unexpected prompt would turn a formatting change
    into an outage.
    """
    question = ""
    evidence = ""

    if QUESTION_LABEL in message:
        after = message.split(QUESTION_LABEL, 1)[1]
        question = after.split(EVIDENCE_LABEL, 1)[0].strip()
    if EVIDENCE_LABEL in message:
        evidence = message.split(EVIDENCE_LABEL, 1)[1].strip()

    blocks: Dict[int, str] = {}
    if not evidence or evidence == NO_EVIDENCE:
        return question, blocks

    for block in evidence.split(_SEPARATOR):
        match = _SOURCE_RE.search(block)
        if not match:
            continue
        if "Text:" not in block:
            continue
        body = block.split("Text:", 1)[1].strip()
        # Drop the truncation marker Stage 6 appends: it is a note about
        # the passage, not part of it, and must never be quoted back as
        # though the document said it.
        body = "\n".join(
            line for line in body.splitlines() if not line.lstrip().startswith("[")
        ).strip()
        blocks[int(match.group(1))] = body

    return question, blocks


def build_extractive_provider(settings=None) -> ExtractiveProvider:
    return ExtractiveProvider()


__all__ = ["ExtractiveProvider", "parse_user_message", "build_extractive_provider"]
