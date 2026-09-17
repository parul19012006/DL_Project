"""Shared plumbing for HTTP-based LLM providers.

Every hosted provider is the same three steps — build a JSON body, POST
it with an auth header, read one string out of the reply — differing only
in the shapes. That shared part lives here so a new vendor is a subclass
with three small methods rather than a fresh copy of the error handling,
the timeout policy and the secret hygiene.

Two rules are enforced here rather than left to each subclass:

**Secrets never travel with an error.** The request body and headers are
not attached to any exception, logged, or echoed in a response. An error
carries the status code and the vendor's own message, which is what a
developer needs and the only part safe to show.

**Transport is injectable.** ``httpx`` accepts a ``transport``, so the
real request-building and response-parsing code can be exercised against
``httpx.MockTransport`` in tests. A mocked provider object would test
nothing; a mocked transport tests everything except the socket.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from app.generation.base import (
    LLMConfigurationError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.logging_config import get_logger

logger = get_logger(__name__)

#: Status codes worth retrying. 429 and 5xx are transient; 400 and 401
#: are not, and retrying them just burns quota against a broken request.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class RemoteProvider(LLMProvider):
    """An LLM reached over HTTPS."""

    #: Endpoint path appended to the base URL.
    path: str = ""
    default_base_url: str = ""
    default_model: str = ""

    def __init__(
        self,
        api_key: str,
        model: str = "",
        base_url: str = "",
        timeout: float = 30.0,
        transport: Any = None,
    ) -> None:
        key = (api_key or "").strip()
        if not key:
            raise LLMConfigurationError(
                f"{self.name} requires an API key. Set it in the environment "
                f"(see .env.example); it is never read from anywhere else and "
                "must not be committed."
            )
        self._api_key = key
        self.model = (model or self.default_model).strip()
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.timeout = float(timeout)
        #: Test seam: an httpx transport. None means a real connection.
        self._transport = transport

    # -- to implement --------------------------------------------------

    def build_body(self, request: LLMRequest) -> Dict[str, Any]:
        raise NotImplementedError

    def build_headers(self) -> Dict[str, str]:
        raise NotImplementedError

    def read_response(self, payload: Dict[str, Any]) -> LLMResponse:
        raise NotImplementedError

    # -- the shared call -----------------------------------------------

    def generate(self, request: LLMRequest) -> LLMResponse:
        import httpx

        url = f"{self.base_url}{self.path}"
        started = time.perf_counter()

        try:
            client_kwargs: Dict[str, Any] = {"timeout": self.timeout}
            if self._transport is not None:
                client_kwargs["transport"] = self._transport
            with httpx.Client(**client_kwargs) as client:
                response = client.post(
                    url, json=self.build_body(request), headers=self.build_headers()
                )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"{self.describe()} did not respond within {self.timeout}s"
            ) from exc
        except httpx.HTTPError as exc:
            # Deliberately does not include the request: it carries the
            # Authorization header and the whole prompt.
            raise LLMUnavailableError(
                f"{self.describe()} could not be reached: {type(exc).__name__}"
            ) from exc

        if response.status_code >= 400:
            raise self._error_for(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMUnavailableError(
                f"{self.describe()} returned a non-JSON body "
                f"(HTTP {response.status_code})"
            ) from exc

        result = self.read_response(payload)
        result.provider = self.name
        result.model = result.model or self.model
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    def _error_for(self, response: Any) -> LLMUnavailableError:
        """Map a vendor error to ours, carrying only what is safe."""
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                error = body.get("error")
                if isinstance(error, dict):
                    detail = str(error.get("message") or error.get("type") or "")
                elif isinstance(error, str):
                    detail = error
                detail = detail or str(body.get("message") or "")
        except Exception:  # pragma: no cover - defensive
            detail = ""

        status = response.status_code
        if status in (401, 403):
            return LLMConfigurationError(
                f"{self.describe()} rejected the API key (HTTP {status}). "
                "Check the environment variable for this provider."
            )

        retryable = status in RETRYABLE_STATUS
        message = (
            f"{self.describe()} returned HTTP {status}"
            + (f": {detail[:300]}" if detail else "")
            + ("" if retryable else " (not retryable)")
        )
        logger.error("LLM provider error: %s", message)
        return LLMUnavailableError(message)

    # -- health --------------------------------------------------------

    def check(self) -> None:
        if not self.model:
            raise LLMConfigurationError(
                f"{self.name} has no model configured. Set LLM_MODEL."
            )
        if not self.base_url:  # pragma: no cover - defaults are set
            raise LLMConfigurationError(f"{self.name} has no base URL.")


def first_text(*values: Optional[str]) -> str:
    for value in values:
        if value:
            return str(value)
    return ""


__all__ = ["RemoteProvider", "RETRYABLE_STATUS", "first_text"]
