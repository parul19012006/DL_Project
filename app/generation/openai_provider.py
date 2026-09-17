"""OpenAI-compatible chat completions.

Speaks the ``/chat/completions`` dialect, which is also what Azure
OpenAI, Together, Groq, Fireworks, vLLM, Ollama and LM Studio serve. So
this one provider covers a hosted API *and* a self-hosted model behind
the same shape — point ``LLM_BASE_URL`` at the other server and nothing
else changes.

``response_format: {"type": "json_object"}`` is requested because the
answer contract is JSON. It is a hint: older models and most
OpenAI-compatible servers ignore it, and even when honoured it does not
guarantee the *schema* — only that the output parses. The parser
downstream assumes neither.
"""

from __future__ import annotations

from typing import Any, Dict

from app.generation.base import LLMRequest, LLMResponse
from app.generation.remote import RemoteProvider


class OpenAIProvider(RemoteProvider):
    """Chat completions over the OpenAI wire format."""

    name = "openai"
    path = "/chat/completions"
    default_base_url = "https://api.openai.com/v1"
    default_model = "gpt-4o-mini"

    def build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def build_body(self, request: LLMRequest) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                *request.messages(),
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def read_response(self, payload: Dict[str, Any]) -> LLMResponse:
        choices = payload.get("choices") or []
        first = choices[0] if choices else {}
        message = first.get("message") or {}
        usage = payload.get("usage") or {}

        return LLMResponse(
            text=str(message.get("content") or ""),
            model=str(payload.get("model") or self.model),
            finish_reason=str(first.get("finish_reason") or ""),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            extra={"id": payload.get("id", "")},
        )


__all__ = ["OpenAIProvider"]
