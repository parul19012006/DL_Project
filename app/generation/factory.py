"""Provider selection.

``LLM_PROVIDER`` names a registered provider. As with the vector-store
registry, the name is a free string validated by the registry rather than
a closed enum: adding Mistral, Bedrock, Vertex or a local vLLM server is
one module plus one :func:`register_provider` call, with no edit to
``config.py`` and no change to any caller — because no caller names a
provider.

**Keys come from the environment and nowhere else.** Each builder reads
its own variable off :class:`~app.config.Settings`, which is populated
from the environment. There is no request field, no header and no config
file that can supply a key, and the value is held as a ``SecretStr`` so
an accidental ``repr`` of the settings object prints ``**********``
rather than the credential.

**A provider that cannot be built degrades, unless told not to.** With
``LLM_STRICT=false`` (the default) an unusable provider falls back to the
deterministic extractive one and ``/health`` reports it, so a missing key
in development does not stop the service. With ``LLM_STRICT=true`` it is
fatal — the right setting for production, where serving sentence
selection while believing you are serving a language model is worse than
being down.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional

from app.config import Settings, get_settings
from app.generation.anthropic_provider import AnthropicProvider
from app.generation.base import LLMConfigurationError, LLMProvider
from app.generation.extractive import ExtractiveProvider
from app.generation.openai_provider import OpenAIProvider
from app.logging_config import get_logger

logger = get_logger(__name__)

Builder = Callable[[Settings], LLMProvider]

_PROVIDERS: Dict[str, Builder] = {}
_provider: Optional[LLMProvider] = None
_lock = threading.Lock()

EXTRACTIVE = "extractive"


def register_provider(name: str, builder: Builder) -> None:
    _PROVIDERS[name.strip().lower()] = builder


def available_providers() -> List[str]:
    return sorted(_PROVIDERS)


def _secret(value) -> str:
    """Read a SecretStr (or a plain string) without logging it."""
    if value is None:
        return ""
    getter = getattr(value, "get_secret_value", None)
    return (getter() if callable(getter) else str(value)).strip()


def _build_extractive(settings: Settings) -> LLMProvider:
    return ExtractiveProvider()


def _build_openai(settings: Settings) -> LLMProvider:
    return OpenAIProvider(
        api_key=_secret(settings.openai_api_key),
        model=settings.llm_model,
        base_url=settings.llm_base_url or "",
        timeout=settings.llm_timeout_seconds,
    )


def _build_anthropic(settings: Settings) -> LLMProvider:
    return AnthropicProvider(
        api_key=_secret(settings.anthropic_api_key),
        model=settings.llm_model,
        base_url=settings.llm_base_url or "",
        timeout=settings.llm_timeout_seconds,
    )


register_provider(EXTRACTIVE, _build_extractive)
register_provider("openai", _build_openai)
register_provider("anthropic", _build_anthropic)


def build_llm_provider(settings: Optional[Settings] = None) -> LLMProvider:
    """Build the configured provider, honouring ``LLM_STRICT``."""
    settings = settings or get_settings()
    name = (settings.llm_provider or EXTRACTIVE).strip().lower()

    builder = _PROVIDERS.get(name)
    if builder is None:
        error = LLMConfigurationError(
            f"Unknown LLM_PROVIDER '{name}'. Available: "
            f"{', '.join(available_providers())}"
        )
        if settings.llm_strict:
            raise error
        logger.error("%s Falling back to the extractive provider.", error)
        return ExtractiveProvider()

    try:
        provider = builder(settings)
        provider.check()
    except LLMConfigurationError as exc:
        if settings.llm_strict:
            raise
        logger.warning(
            "LLM provider '%s' is not usable (%s); falling back to the "
            "deterministic extractive provider. It is NOT a language model "
            "— it selects sentences from the evidence — and /health reports "
            "this. Set LLM_STRICT=true to make it fatal instead.",
            name,
            exc,
        )
        return ExtractiveProvider()

    logger.info("LLM provider: %s", provider.describe())
    return provider


def get_llm_provider(settings: Optional[Settings] = None) -> LLMProvider:
    """Process-wide provider singleton."""
    global _provider
    if _provider is None:
        with _lock:
            if _provider is None:
                _provider = build_llm_provider(settings)
    return _provider


def set_llm_provider(provider: Optional[LLMProvider]) -> None:
    """Install a provider, or ``None`` to rebuild from settings."""
    global _provider
    with _lock:
        _provider = provider


__all__ = [
    "register_provider",
    "available_providers",
    "build_llm_provider",
    "get_llm_provider",
    "set_llm_provider",
    "Builder",
    "EXTRACTIVE",
]
