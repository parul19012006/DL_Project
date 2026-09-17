"""Grounded generation — the whole pipeline, end to end.

    question
      -> retrieval        hybrid search + reranking          (Stage 5)
      -> context builder  selected, budgeted, labelled       (Stage 6)
      -> prompt           grounding rules + evidence         (prompts.py)
      -> LLM              a configurable provider            (factory.py)
      -> parse            tolerant of packaging, strict on shape
      -> validate         citations checked against the sources
      -> structured response

Four behaviours are worth stating plainly, because each is a decision
rather than an implementation detail.

**No evidence means no model call.** When retrieval returns nothing, the
answer is fixed and the provider is never invoked. Asking a model to say
"I don't know" and hoping it complies is strictly worse than not asking:
it costs a call, it takes longer, and it has a failure mode — the model
answering anyway from its own knowledge — that this stage exists to
prevent.

**A failed generation does not discard the retrieval.** If the provider
is down or its reply is unparseable, the response still carries the
sources, the context and the citations that retrieval produced, with a
``status`` saying what went wrong. The user gets "we could not generate an
answer, here are the relevant clauses" instead of an error page, and the
work that succeeded is not thrown away because a later step failed.

**One retry, and only for shape.** An unparseable reply is retried once
with a nudge about *formatting only* — deliberately not a restatement of
the grounding rules, which would invite the model to rewrite the answer
rather than reformat it. A second failure is reported, not papered over.

**The answer is never presented as more than it is.** The provider's name
and whether it is a language model at all travel with every response, and
citation validation runs on every answer regardless of provider. The
checks here reduce ungrounded output and make every citation verifiable;
they cannot guarantee a model states only what the evidence supports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import Settings, get_settings
from app.context.base import BuiltContext
from app.context.builder import get_context_builder
from app.generation.base import (
    LLMConfigurationError,
    LLMError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    MalformedResponseError,
)
from app.generation.citations import CitationReport, validate_citations
from app.generation.factory import get_llm_provider
from app.generation.parser import ParsedAnswer, parse_answer
from app.generation.prompts import (
    REPAIR_INSTRUCTION,
    build_system_prompt,
    build_user_message,
)
from app.logging_config import get_logger
from app.retrieval.pipeline import RetrievalOutcome, get_pipeline

logger = get_logger(__name__)

#: Characters of an unparseable reply kept for diagnosis.
RAW_EXCERPT_CHARS = 500

#: Said when retrieval found nothing. Fixed text, no model involved.
NO_EVIDENCE_ANSWER = (
    "The retrieved passages do not contain information that answers this "
    "question. No answer can be given from the documents available."
)


class AnswerStatus:
    """Stable status codes. The caller branches on these, not on prose."""

    OK = "ok"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    LLM_UNAVAILABLE = "llm_unavailable"
    LLM_CONFIGURATION_ERROR = "llm_configuration_error"
    MALFORMED_RESPONSE = "malformed_response"


@dataclass
class AnswerResult:
    """A grounded answer, its evidence, and how it was produced."""

    status: str = AnswerStatus.OK
    question: str = ""
    tenant_id: str = ""
    answer: Optional[str] = None
    interpretation: str = ""
    insufficient_evidence: bool = False

    citations: CitationReport = field(default_factory=CitationReport)
    context: Optional[BuiltContext] = None
    retrieval: Optional[RetrievalOutcome] = None

    provider: str = ""
    model: str = ""
    #: False when the "provider" is the deterministic extractive one.
    is_language_model: bool = True
    usage: Dict[str, int] = field(default_factory=dict)

    attempts: int = 0
    #: Present only when the reply could not be parsed.
    raw_excerpt: str = ""
    error: str = ""

    warnings: List[str] = field(default_factory=list)
    duration_ms: int = 0
    timings_ms: Dict[str, int] = field(default_factory=dict)

    @property
    def grounded(self) -> bool:
        return self.citations.grounded

    @property
    def succeeded(self) -> bool:
        return self.status in (
            AnswerStatus.OK,
            AnswerStatus.INSUFFICIENT_EVIDENCE,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations.citations],
            "interpretation": self.interpretation,
            "insufficient_evidence": self.insufficient_evidence,
            "grounded": self.grounded,
            "validation": self.citations.to_dict(),
            "provider": self.provider,
            "model": self.model,
            "is_language_model": self.is_language_model,
            "usage": dict(self.usage),
            "warnings": list(self.warnings),
            "duration_ms": self.duration_ms,
            "timings_ms": dict(self.timings_ms),
        }


class AnswerService:
    """Question in, grounded structured answer out."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        provider: Optional[LLMProvider] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._provider = provider

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = get_llm_provider(self.settings)
        return self._provider

    # -- main ----------------------------------------------------------

    def answer(
        self,
        question: str,
        tenant_id: str,
        retrieval_options: Optional[Dict[str, Any]] = None,
        context_options: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        extra_instructions: Optional[str] = None,
    ) -> AnswerResult:
        settings = self.settings
        started = time.perf_counter()
        timings: Dict[str, int] = {}

        result = AnswerResult(question=question, tenant_id=tenant_id)

        # -- 1. retrieve ------------------------------------------------
        mark = time.perf_counter()
        outcome = get_pipeline(settings).retrieve(
            query=question, tenant_id=tenant_id, **(retrieval_options or {})
        )
        timings["retrieval"] = int((time.perf_counter() - mark) * 1000)
        result.retrieval = outcome
        result.warnings.extend(outcome.warnings)

        # -- 2. build the context ---------------------------------------
        mark = time.perf_counter()
        context = get_context_builder(settings).build(
            outcome.results, query=question, **(context_options or {})
        )
        timings["context"] = int((time.perf_counter() - mark) * 1000)
        result.context = context
        result.warnings.extend(context.warnings)

        provider = self.provider
        result.provider = provider.name
        result.model = provider.model
        result.is_language_model = provider.is_language_model

        # -- 3. nothing to ground an answer in --------------------------
        if context.is_empty:
            result.status = AnswerStatus.INSUFFICIENT_EVIDENCE
            result.answer = NO_EVIDENCE_ANSWER
            result.insufficient_evidence = True
            result.citations = CitationReport(grounded=True)
            result.duration_ms = int((time.perf_counter() - started) * 1000)
            result.timings_ms = timings
            logger.info(
                "No evidence for a question from tenant %s; the model was "
                "not called",
                tenant_id,
            )
            return result

        # -- 4. generate -------------------------------------------------
        request = LLMRequest(
            system=build_system_prompt(
                extra_instructions
                if extra_instructions is not None
                else settings.llm_extra_instructions
            ),
            user=build_user_message(question, context.text),
            temperature=(
                settings.llm_temperature if temperature is None else float(temperature)
            ),
            max_output_tokens=int(
                max_output_tokens or settings.llm_max_output_tokens
            ),
        )

        mark = time.perf_counter()
        parsed, response, failure = self._generate(request)
        timings["generation"] = int((time.perf_counter() - mark) * 1000)

        if response is not None:
            result.usage = response.usage()
            result.model = response.model or result.model
            if response.truncated:
                result.warnings.append(
                    "The model stopped at its output limit; the answer may be "
                    "incomplete. Raise LLM_MAX_OUTPUT_TOKENS."
                )

        if failure is not None:
            result.status = failure["status"]
            result.error = failure["message"]
            result.raw_excerpt = failure.get("raw", "")
            result.warnings.append(failure["warning"])
            result.attempts = failure["attempts"]
            result.duration_ms = int((time.perf_counter() - started) * 1000)
            result.timings_ms = timings
            return result

        result.attempts = 1 if not parsed.recovered else result.attempts or 1

        # -- 5. validate citations --------------------------------------
        mark = time.perf_counter()
        report = validate_citations(
            parsed,
            context,
            require_citations=settings.generation_require_citations,
            harvest_inline=settings.generation_harvest_inline_citations,
            check_quotes=settings.generation_verify_quotes,
        )
        timings["validation"] = int((time.perf_counter() - mark) * 1000)

        result.citations = report
        result.answer = parsed.answer
        result.interpretation = parsed.interpretation
        result.insufficient_evidence = parsed.insufficient_evidence
        result.warnings.extend(report.warnings)
        result.status = (
            AnswerStatus.INSUFFICIENT_EVIDENCE
            if parsed.insufficient_evidence
            else AnswerStatus.OK
        )
        if parsed.recovered:
            result.warnings.append(
                "The model's reply was not clean JSON and had to be recovered."
            )

        result.duration_ms = int((time.perf_counter() - started) * 1000)
        result.timings_ms = timings

        logger.info(
            "Answered for tenant %s: status=%s provider=%s sources=%d "
            "citations=%d grounded=%s (%d ms)",
            tenant_id,
            result.status,
            provider.describe(),
            len(context.sources),
            len(report.citations),
            report.grounded,
            result.duration_ms,
        )
        return result

    # -- generation with one format retry ------------------------------

    def _generate(self, request: LLMRequest):
        """``(parsed, response, failure)``. Exactly one is a failure."""
        attempts = max(1, int(self.settings.llm_max_retries) + 1)
        last_raw = ""

        for attempt in range(1, attempts + 1):
            try:
                response = self.provider.generate(request)
            except LLMConfigurationError as exc:
                logger.error("LLM is misconfigured: %s", exc)
                return (
                    None,
                    None,
                    {
                        "status": AnswerStatus.LLM_CONFIGURATION_ERROR,
                        "message": str(exc),
                        "warning": (
                            "The language model is not configured correctly, so "
                            "no answer was generated. The retrieved evidence "
                            "below is unaffected."
                        ),
                        "attempts": attempt,
                    },
                )
            except LLMError as exc:
                logger.error("LLM call failed: %s", exc)
                return (
                    None,
                    None,
                    {
                        "status": AnswerStatus.LLM_UNAVAILABLE,
                        "message": str(exc),
                        "warning": (
                            "The language model could not be reached, so no "
                            "answer was generated. The retrieved evidence "
                            "below is unaffected and the question can be "
                            "retried."
                        ),
                        "attempts": attempt,
                    },
                )

            last_raw = response.text or ""
            try:
                return parse_answer(response.text), response, None
            except MalformedResponseError as exc:
                logger.warning(
                    "Unparseable reply from %s on attempt %d/%d: %s",
                    self.provider.describe(),
                    attempt,
                    attempts,
                    exc,
                )
                if attempt >= attempts:
                    return (
                        None,
                        response,
                        {
                            "status": AnswerStatus.MALFORMED_RESPONSE,
                            "message": str(exc),
                            "raw": last_raw[:RAW_EXCERPT_CHARS],
                            "warning": (
                                "The model's reply could not be read as a "
                                "structured answer after "
                                f"{attempt} attempt(s), so no answer is being "
                                "returned. The retrieved evidence below is "
                                "unaffected."
                            ),
                            "attempts": attempt,
                        },
                    )
                # Retry about the format only. Restating the grounding
                # rules here would invite a rewritten answer rather than
                # a reformatted one.
                request = LLMRequest(
                    system=request.system,
                    user=request.user,
                    repair=REPAIR_INSTRUCTION,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                    json_mode=request.json_mode,
                )

        # Unreachable: the loop returns on every path.
        raise AssertionError("generation loop exited without a result")


# =====================================================================
# Module-level convenience
# =====================================================================

_service: Optional[AnswerService] = None


def get_answer_service(settings: Optional[Settings] = None) -> AnswerService:
    global _service
    if _service is None or (
        settings is not None and _service.settings is not settings
    ):
        _service = AnswerService(settings)
    return _service


def set_answer_service(service: Optional[AnswerService]) -> None:
    global _service
    _service = service


def answer_question(
    question: str,
    tenant_id: str,
    settings: Optional[Settings] = None,
    **kwargs,
) -> AnswerResult:
    return get_answer_service(settings).answer(question, tenant_id, **kwargs)


__all__ = [
    "AnswerService",
    "AnswerResult",
    "AnswerStatus",
    "answer_question",
    "get_answer_service",
    "set_answer_service",
    "NO_EVIDENCE_ANSWER",
]
