"""Shared API dependencies.

Routers depend on these callables rather than importing globals, which
keeps handlers testable (any dependency can be overridden with
``app.dependency_overrides``) and keeps the dependency direction one-way:

    api  ->  deps  ->  config / logging

End-user authentication is **not** performed here. JWT, sessions and RBAC
belong to the MERN backend; this service only verifies an optional shared
secret proving the caller is that backend.
"""

from __future__ import annotations

import hmac
import time
from typing import Optional

from fastapi import Depends, Header, Request

from app.config import Settings, get_settings
from app.exceptions import AuthenticationError
from app.logging_config import get_logger

logger = get_logger(__name__)

# Set once when the application finishes starting up (see app.main).
_STARTED_AT: Optional[float] = None


def mark_started(when: Optional[float] = None) -> float:
    """Record the moment the app became ready. Returns that timestamp."""
    global _STARTED_AT
    _STARTED_AT = when if when is not None else time.monotonic()
    return _STARTED_AT


def reset_started() -> None:
    """Test helper — forget the recorded start time."""
    global _STARTED_AT
    _STARTED_AT = None


def get_uptime_seconds() -> float:
    """Seconds since startup; 0.0 if startup has not been recorded."""
    if _STARTED_AT is None:
        return 0.0
    return max(0.0, time.monotonic() - _STARTED_AT)


def get_config(request: Request) -> Settings:
    """Inject the settings this application was built with.

    ``create_app(settings)`` stores its settings on ``app.state``, so an
    app constructed with custom settings really uses them — handlers must
    never reach past this into the global singleton. The singleton is the
    fallback for an app assembled without the factory.
    """
    settings = getattr(request.app.state, "settings", None)
    return settings if isinstance(settings, Settings) else get_settings()


async def verify_service_token(
    x_service_token: Optional[str] = Header(
        default=None,
        alias="X-Service-Token",
        description="Shared secret proving the caller is the MERN backend",
    ),
    settings: Settings = Depends(get_config),
) -> None:
    """Reject callers without the configured shared secret.

    A no-op when ``SERVICE_TOKEN`` is empty, which is the development
    default. Comparison is constant-time.
    """
    if not settings.auth_enabled:
        return
    provided = (x_service_token or "").strip()
    if not provided or not hmac.compare_digest(provided, settings.service_token):
        # Never log the value that was supplied.
        logger.warning("Rejected a request with a missing or invalid service token")
        raise AuthenticationError()


__all__ = [
    "get_config",
    "verify_service_token",
    "get_uptime_seconds",
    "mark_started",
    "reset_started",
]
