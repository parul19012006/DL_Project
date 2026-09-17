"""Stage 7 — grounded RAG generation.

    question -> retrieval -> context -> prompt -> LLM
             -> parse -> citation validation -> structured response

``base``        the provider interface and its error taxonomy
``prompts``     the grounding rules and the prompt assembly
``extractive``  a deterministic, model-free provider for offline use
``openai_provider`` / ``anthropic_provider``  hosted providers over httpx
``remote``      shared HTTP plumbing and secret hygiene
``factory``     the provider registry and selection
``parser``      reading the reply, tolerantly but not credulously
``citations``   checking every citation against the retrieved sources
``service``     the orchestration

API keys are read from the environment only. Nothing here hard-codes a
secret, logs one, or echoes one in a response.

The validation in this package reduces ungrounded output and makes every
citation checkable against a passage that was genuinely retrieved. It
does not guarantee that a model states only what the evidence supports.
"""

from app.generation.base import (
    LLMConfigurationError,
    LLMError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
    MalformedResponseError,
)
from app.generation.citations import (
    Citation,
    CitationProblem,
    CitationReport,
    RejectedCitation,
    validate_citations,
    verify_quotes,
)
from app.generation.extractive import ExtractiveProvider
from app.generation.factory import (
    available_providers,
    build_llm_provider,
    get_llm_provider,
    register_provider,
    set_llm_provider,
)
from app.generation.parser import ParsedAnswer, parse_answer
from app.generation.prompts import (
    SYSTEM_PROMPT,
    build_system_prompt,
    build_user_message,
)
from app.generation.service import (
    AnswerResult,
    AnswerService,
    AnswerStatus,
    answer_question,
    get_answer_service,
    set_answer_service,
)

__all__ = [
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LLMError",
    "LLMConfigurationError",
    "LLMUnavailableError",
    "LLMTimeoutError",
    "MalformedResponseError",
    "ExtractiveProvider",
    "register_provider",
    "available_providers",
    "build_llm_provider",
    "get_llm_provider",
    "set_llm_provider",
    "SYSTEM_PROMPT",
    "build_system_prompt",
    "build_user_message",
    "ParsedAnswer",
    "parse_answer",
    "Citation",
    "RejectedCitation",
    "CitationReport",
    "CitationProblem",
    "validate_citations",
    "verify_quotes",
    "AnswerService",
    "AnswerResult",
    "AnswerStatus",
    "answer_question",
    "get_answer_service",
    "set_answer_service",
]
