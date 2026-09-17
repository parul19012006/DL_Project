"""The LLM provider interface.

This is the seam that keeps the RAG pipeline independent of whichever
model is answering. The pipeline depends on :class:`LLMProvider`, on
:class:`LLMRequest` and on :class:`LLMResponse` — never on an SDK, a
vendor's message format, or a vendor's error classes.

Design decisions and the reasons for them:

**Providers speak in messages, not in prompts.** A single ``prompt``
string would force every provider to reinvent the system/user split, and
the system instruction is the part doing the grounding work — it must
survive the trip to any vendor intact.

**No vendor SDK.** ``openai`` and ``anthropic`` are each a dependency
tree, a release cadence and a breaking change waiting to happen, and both
are a thin wrapper over one HTTP POST. The providers here use ``httpx``,
which is already installed. Adding a vendor is one module and one
``register_provider`` call.

**API keys are read from the environment and never leave it.** They are
held as :class:`~pydantic.SecretStr`, never logged, never echoed in a
response, and never included in an error message — a provider error
carries the status code and the vendor's message, not the request.

**Every failure mode is a distinct exception.** "The key is missing",
"the vendor is down", "the request timed out" and "the model returned
something unparseable" want different handling: the first is a
configuration error an operator must fix, the second is worth retrying,
the fourth is worth retrying *differently*. One generic error would
collapse all four.

**Determinism is the default.** ``LLM_TEMPERATURE`` defaults to 0. A
legal answer that changes between two identical questions is not a
feature, and a non-deterministic pipeline cannot be regression-tested.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# =====================================================================
# Errors
# =====================================================================


class LLMError(RuntimeError):
    """Base class for generation failures."""

    #: Stable code the API surfaces so the caller can branch.
    code: str = "llm_error"


class LLMConfigurationError(LLMError):
    """The provider cannot be built — a missing key, an unknown name.

    An operator problem, not a transient one. Retrying will not help.
    """

    code = "llm_configuration_error"


class LLMUnavailableError(LLMError):
    """The provider could not be reached, or refused the request."""

    code = "llm_unavailable"


class LLMTimeoutError(LLMUnavailableError):
    """The provider did not respond in time."""

    code = "llm_timeout"


class MalformedResponseError(LLMError):
    """The model replied, but not in the shape that was asked for."""

    code = "malformed_response"

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        #: What actually came back, for diagnosis. Truncated by the caller.
        self.raw = raw


# =====================================================================
# Request and response
# =====================================================================


@dataclass
class LLMRequest:
    """One generation call, in vendor-neutral terms."""

    system: str
    user: str
    #: Nudge appended on a retry after an unparseable reply. Kept
    #: separate from ``user`` so the original question is never rewritten.
    repair: Optional[str] = None
    temperature: float = 0.0
    max_output_tokens: int = 800
    #: Ask the vendor for JSON where it supports doing so. A hint, not a
    #: guarantee — the parser never assumes it was honoured.
    json_mode: bool = True

    def messages(self) -> List[Dict[str, str]]:
        """The conversation, in the shape most vendors accept."""
        out = [{"role": "user", "content": self.user}]
        if self.repair:
            out.append({"role": "user", "content": self.repair})
        return out


@dataclass
class LLMResponse:
    """What a provider returns. Raw text — parsing happens elsewhere."""

    text: str = ""
    model: str = ""
    provider: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: int = 0
    #: Anything vendor-specific worth keeping for debugging.
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def truncated(self) -> bool:
        """The model ran out of output budget mid-answer.

        Worth surfacing: a truncated JSON reply is the most common cause
        of a parse failure, and the fix is ``LLM_MAX_OUTPUT_TOKENS``, not
        a retry.
        """
        return self.finish_reason in ("length", "max_tokens")

    def usage(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


# =====================================================================
# Interface
# =====================================================================


class LLMProvider(ABC):
    """Generates text from a system instruction and a user message."""

    #: Short provider name, reported by /health and in responses.
    name: str = "llm"
    #: Model identifier, or a description for a model-free provider.
    model: str = ""
    #: False for anything that is not a language model, so no part of the
    #: system can present its output as though a model produced it.
    is_language_model: bool = True
    #: True when the provider needs an API key and has one.
    requires_key: bool = True

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Run one completion. Raises :class:`LLMError` on failure."""

    def describe(self) -> str:
        return f"{self.name}:{self.model}" if self.model else self.name

    def check(self) -> None:
        """Cheap readiness check for ``/health``.

        Deliberately does *not* call the vendor: a health endpoint that
        bills per probe is a health endpoint nobody leaves enabled.
        Configuration problems — a missing key, an unknown model — are
        what this catches, and they are the ones that matter at startup.
        """
        return None


__all__ = [
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LLMError",
    "LLMConfigurationError",
    "LLMUnavailableError",
    "LLMTimeoutError",
    "MalformedResponseError",
]
