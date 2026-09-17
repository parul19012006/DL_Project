"""Logging setup.

One entry point, :func:`setup_logging`, called once at application
startup. Two formats are supported:

``text``  human-readable, for local development
``json``  one JSON object per line, for log aggregators

A ``request_id`` is attached to every record emitted while handling a
request (see the middleware in ``app.main``), so a single request can be
traced end to end. Uses only the standard library — no extra dependency.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any, Dict, Optional

# Set per request by RequestContextMiddleware; empty outside a request.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_CONFIGURED = False

# Record attributes that are standard; anything else was added by the
# caller via `extra=` and should be surfaced in JSON output.
_STANDARD_ATTRS = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"asctime", "message", "taskName"}


class RequestIdFilter(logging.Filter):
    """Inject the current request id into a record.

    Installed on our own handler as a safety net. The primary mechanism
    is the record factory below, which stamps ``request_id`` onto every
    record at creation time — so third-party handlers (pytest's caplog,
    a Sentry integration, a file handler) see it too.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True


def _install_record_factory() -> None:
    """Stamp request_id onto every LogRecord, whoever handles it."""
    current_factory = logging.getLogRecordFactory()
    if getattr(current_factory, "_genai_request_id", False):
        return

    def factory(*args, **kwargs):
        record = current_factory(*args, **kwargs)
        record.request_id = request_id_var.get()
        return record

    factory._genai_request_id = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and key != "request_id":
                payload[key] = value
        return json.dumps(payload, default=str)


TEXT_FORMAT = "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s | %(message)s"


def setup_logging(
    level: str = "INFO", fmt: str = "text", force: bool = False
) -> logging.Logger:
    """Configure the root logger. Idempotent unless ``force`` is set."""
    global _CONFIGURED
    _install_record_factory()
    if _CONFIGURED and not force:
        return logging.getLogger()

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(TEXT_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S")
        )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    # uvicorn installs its own handlers; route them through ours so the
    # output format stays consistent.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv = logging.getLogger(name)
        uv.handlers = []
        uv.propagate = True

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_request_id(value: str) -> Any:
    """Bind a request id to the current context; returns the reset token."""
    return request_id_var.set(value)


def get_request_id() -> str:
    return request_id_var.get()


def reset_request_id(token: Any) -> None:
    request_id_var.reset(token)


__all__ = [
    "setup_logging",
    "get_logger",
    "set_request_id",
    "get_request_id",
    "reset_request_id",
    "JsonFormatter",
    "RequestIdFilter",
]
