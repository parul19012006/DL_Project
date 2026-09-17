"""Ingestion error hierarchy.

Extends the Stage 1 ``GenAIServiceError``, so every one of these already
renders through the registered handler in the standard error envelope
with the right status code — no per-endpoint try/except needed.

The distinctions matter operationally: the MERN backend should retry a
``ServiceUnavailable``, show "this file is damaged" for a
``CorruptDocumentError``, and refuse the upload outright for an
``UnsupportedFileTypeError``. Branch on ``error.type``, not on the
message.
"""

from __future__ import annotations

from fastapi import status

from app.exceptions import GenAIServiceError

# Starlette renamed two constants; support both spellings so the service
# runs on either version without emitting deprecation warnings.
HTTP_422 = getattr(
    status, "HTTP_422_UNPROCESSABLE_CONTENT", None
) or status.HTTP_422_UNPROCESSABLE_ENTITY
HTTP_413 = getattr(
    status, "HTTP_413_CONTENT_TOO_LARGE", None
) or status.HTTP_413_REQUEST_ENTITY_TOO_LARGE


class IngestionError(GenAIServiceError):
    """Base class for every ingestion failure."""

    status_code = HTTP_422
    error_type = "ingestion_error"
    message = "The document could not be ingested"


# -- Rejected before parsing -----------------------------------------


class UnsupportedFileTypeError(IngestionError):
    """Extension or magic bytes are not an accepted document type."""

    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    error_type = "unsupported_file_type"
    message = "This file type is not supported"


class FileTooLargeError(IngestionError):
    status_code = HTTP_413
    error_type = "file_too_large"
    message = "The file exceeds the configured size limit"


class FileNotFoundError_(IngestionError):
    """Named with a trailing underscore so it cannot shadow the builtin."""

    status_code = status.HTTP_404_NOT_FOUND
    error_type = "document_not_found"
    message = "The referenced file does not exist"


class UnsafePathError(IngestionError):
    """The referenced path escapes DOCUMENT_ROOT."""

    status_code = status.HTTP_400_BAD_REQUEST
    error_type = "unsafe_path"
    message = "The referenced path is outside the permitted document root"


# -- Failures during parsing -----------------------------------------


class CorruptDocumentError(IngestionError):
    error_type = "corrupt_document"
    message = "The document is damaged or not readable"


class EncryptedDocumentError(IngestionError):
    error_type = "encrypted_document"
    message = "The document is password protected"


class EmptyDocumentError(IngestionError):
    """No usable text — a blank file, or a scan with OCR unavailable.

    This is an error rather than an empty success on purpose: silently
    indexing nothing would later surface as "the AI can't find anything
    in my contract" with no explanation anywhere in the system.
    """

    error_type = "empty_document"
    message = "No extractable text was found in the document"


class IndexingError(IngestionError):
    """The document was read but could not be written to the index.

    Distinct from an extraction failure: the document itself is fine, so
    retrying is reasonable — which is exactly the opposite of what a
    corrupt file warrants.
    """

    status_code = 503
    error_type = "indexing_failed"
    message = "The document could not be written to the index"


class ParserUnavailableError(IngestionError):
    """A required parsing library is not installed."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_type = "parser_unavailable"
    message = "The parser required for this file type is not available"


__all__ = [
    "IngestionError",
    "UnsupportedFileTypeError",
    "FileTooLargeError",
    "FileNotFoundError_",
    "UnsafePathError",
    "CorruptDocumentError",
    "EncryptedDocumentError",
    "EmptyDocumentError",
    "ParserUnavailableError",
]
