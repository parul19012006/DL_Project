"""Anthropic Messages API.

Two differences from the OpenAI shape are worth naming, because they are
exactly the sort of detail a shared abstraction hides badly if it is
written around one vendor:

* the system instruction is a **top-level field**, not a message with
  ``role: "system"``;
* the reply is a **list of content blocks**, not one string, so the text
  has to be gathered from the blocks of type ``text``.

There is no JSON mode, so ``request.json_mode`` is ignored here — the
prompt asks for JSON and the parser is tolerant of the usual deviations.
"""

from __future__ import annotations

from typing import Any, Dict

from app.generation.base import LLMRequest, LLMResponse
from app.generation.remote import RemoteProvider

#: Pinned: the API version is part of the contract, not an implicit
#: "whatever is current", so a server-side default change cannot alter
#: the response shape underneath us.
API_VERSION = "2023-06-01"


class AnthropicProvider(RemoteProvider):
    """Claude models over the Messages API."""

    name = "anthropic"
    path = "/v1/messages"
    default_base_url = "https://api.anthropic.com"
    default_model = "claude-3-5-haiku-latest"

    def build_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": API_VERSION,
            "Content-Type": "application/json",
        }

    def build_body(self, request: LLMRequest) -> Dict[str, Any]:
        return {
            "model": self.model,
            "system": request.system,
            "messages": request.messages(),
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }

    def read_response(self, payload: Dict[str, Any]) -> LLMResponse:
        blocks = payload.get("content") or []
        text = "".join(
            str(block.get("text") or "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        usage = payload.get("usage") or {}

        return LLMResponse(
            text=text,
            model=str(payload.get("model") or self.model),
            finish_reason=str(payload.get("stop_reason") or ""),
            prompt_tokens=int(usage.get("input_tokens") or 0),
            completion_tokens=int(usage.get("output_tokens") or 0),
            extra={"id": payload.get("id", "")},
        )


__all__ = ["AnthropicProvider", "API_VERSION"]
