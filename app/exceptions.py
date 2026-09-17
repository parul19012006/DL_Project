"""Exception hierarchy and FastAPI error handlers.

Every error leaving this service — expected or not — is rendered as the
same JSON shape, so the MERN backend only ever has to parse one:

    {"error": {"type": "...", "message": "...", "details": {...}},
     "request_id": "..."}

Internal failures are logged with a traceback but never leak the
traceback, a file path or a stack frame to the caller.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logging_config import get_logger, get_request_id
from app.models.schemas import ErrorDetail, ErrorResponse

logger = get_logger(__name__)


# =====================================================================
# Exception hierarchy
# =====================================================================


class GenAIServiceError(Exception):
    """Base class for every error this service raises deliberately."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type: str = "service_error"
    message: str = "An error occurred in the GenAI service"

    def __init__(
        self,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message or self.__class__.message
        self.details = details or {}
        super().__init__(self.message)

    def to_response(self) -> ErrorResponse:
        return ErrorResponse(
            error=ErrorDetail(
                type=self.error_type, message=self.message, details=self.details
            ),
            request_id=get_request_id(),
        )


class ConfigurationError(GenAIServiceError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type = "configuration_error"
    message = "The service is misconfigured"


class ValidationError(GenAIServiceError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_type = "validation_error"
    message = "The request payload is invalid"


class AuthenticationError(GenAIServiceError):
    status_code = status.HTTP_401_UNAUTHORIZED
    error_type = "authentication_error"
    message = "Invalid or missing service token"


class NotFoundError(GenAIServiceError):
    status_code = status.HTTP_404_NOT_FOUND
    error_type = "not_found"
    message = "The requested resource does not exist"


class PayloadTooLargeError(GenAIServiceError):
    """The request is well-formed; there is simply too much of it."""

    # Starlette renamed this constant; 413 is the code either way.
    status_code = getattr(
        status, "HTTP_413_CONTENT_TOO_LARGE", 413
    )
    error_type = "payload_too_large"
    message = "The request is larger than this service accepts"


class TooManyRequestsError(GenAIServiceError):
    """The service is saturated. The caller should retry later."""

    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_type = "too_many_requests"
    message = "The service is busy; retry shortly"


class ServiceUnavailableError(GenAIServiceError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_type = "service_unavailable"
    message = "A dependency required by this service is unavailable"


# =====================================================================
# Handlers
# =====================================================================


async def genai_error_handler(
    request: Request, exc: GenAIServiceError
) -> JSONResponse:
    logger.warning(
        "%s on %s %s: %s",
        exc.error_type,
        request.method,
        request.url.path,
        exc.message,
    )
    return JSONResponse(
        status_code=exc.status_code, content=exc.to_response().model_dump()
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Normalise FastAPI/Starlette HTTPExceptions into our shape."""
    body = ErrorResponse(
        error=ErrorDetail(
            type=_type_for_status(exc.status_code),
            message=str(exc.detail),
        ),
        request_id=get_request_id(),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=body.model_dump(),
        headers=getattr(exc, "headers", None),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """422 from Pydantic — report which fields failed and why."""
    fields = [
        {
            "field": ".".join(str(p) for p in err.get("loc", ()) if p != "body"),
            "message": err.get("msg", ""),
            "type": err.get("type", ""),
        }
        for err in exc.errors()
    ]
    logger.info(
        "Request validation failed on %s %s: %d field(s)",
        request.method,
        request.url.path,
        len(fields),
    )
    body = ErrorResponse(
        error=ErrorDetail(
            type="validation_error",
            message="Request validation failed",
            details={"fields": fields},
        ),
        request_id=get_request_id(),
    )
    return JSONResponse(status_code=422, content=body.model_dump())


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Last resort. Logs the traceback, returns nothing internal."""
    logger.exception(
        "Unhandled %s on %s %s",
        type(exc).__name__,
        request.method,
        request.url.path,
    )
    body = ErrorResponse(
        error=ErrorDetail(
            type="internal_error",
            message="An internal error occurred. Contact the service owner "
            "with the request id.",
        ),
        request_id=get_request_id(),
    )
    return JSONResponse(status_code=500, content=body.model_dump())


def _type_for_status(status_code: int) -> str:
    return {
        400: "bad_request",
        401: "authentication_error",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        413: "payload_too_large",
        415: "unsupported_media_type",
        429: "rate_limited",
        503: "service_unavailable",
    }.get(status_code, "http_error")


# =====================================================================
# OpenAPI documentation of the error contract
# =====================================================================

#: Attached to every router so ``/openapi.json`` documents the error
#: envelope the MERN backend must be able to parse.
ERROR_RESPONSES: Dict[int | str, Dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Invalid request"},
    401: {"model": ErrorResponse, "description": "Missing or invalid service token"},
    404: {"model": ErrorResponse, "description": "Resource not found"},
    413: {"model": ErrorResponse, "description": "Payload too large"},
    422: {"model": ErrorResponse, "description": "Request validation failed"},
    429: {"model": ErrorResponse, "description": "Service busy; retry later"},
    500: {"model": ErrorResponse, "description": "Internal service error"},
    503: {"model": ErrorResponse, "description": "Service unavailable"},
}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every handler to the application."""
    app.add_exception_handler(GenAIServiceError, genai_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)


__all__ = [
    "GenAIServiceError",
    "ConfigurationError",
    "ValidationError",
    "AuthenticationError",
    "NotFoundError",
    "PayloadTooLargeError",
    "TooManyRequestsError",
    "ServiceUnavailableError",
    "register_exception_handlers",
    "ERROR_RESPONSES",
    "genai_error_handler",
    "http_exception_handler",
    "validation_exception_handler",
    "unhandled_exception_handler",
]
