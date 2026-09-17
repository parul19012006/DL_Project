"""File validation.

Three independent checks, all of which must pass before a byte is
parsed:

**Path** — a referenced path is resolved and must land inside
``DOCUMENT_ROOT``. ``../../etc/passwd``, an absolute path elsewhere, a
symlink pointing out of the root and a null byte are all rejected.

**Type** — the extension must be allow-listed *and* the leading bytes
must match it. An ``.exe`` renamed to ``.pdf`` fails the magic-byte
check; a real PDF renamed to ``.txt`` fails the extension check. Both
are required because either alone is trivially bypassed.

**Size** — enforced against the configured limit, and separately while
streaming an upload so a huge body is abandoned rather than buffered.

Identifiers (``document_id``, ``tenant_id``) are validated here too:
they flow into metadata and, from Stage 4, into vector-store filters, so
they are constrained to a conservative character set at the boundary.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from app.config import Settings, get_settings
from app.ingestion.errors import (
    FileNotFoundError_,
    FileTooLargeError,
    UnsafePathError,
    UnsupportedFileTypeError,
)
from app.logging_config import get_logger

logger = get_logger(__name__)

#: Leading bytes each accepted extension must start with.
#: DOCX is an OOXML package, i.e. a ZIP container.
MAGIC_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF",),
    ".docx": (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),
}

#: Conservative: letters, digits and a few separators. Mongo ObjectIds,
#: UUIDs and slugs all pass; path separators and quotes do not.
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")

MIME_BY_EXTENSION = {
    ".pdf": "application/pdf",
    ".docx": (
        "application/vnd.openxmlformats-officedocument."
        "wordprocessingml.document"
    ),
}


def validate_identifier(value: str, field: str) -> str:
    """Validate a tenant/document identifier supplied by the backend."""
    candidate = (value or "").strip()
    if not IDENTIFIER_RE.match(candidate):
        raise UnsafePathError(
            f"Invalid {field}: expected 1-128 characters from "
            f"[A-Za-z0-9_.:@-]",
            {"field": field},
        )
    return candidate


def normalize_extension(filename: str) -> str:
    return Path(filename or "").suffix.lower()


def is_supported_extension(
    filename: str, settings: Optional[Settings] = None
) -> bool:
    settings = settings or get_settings()
    return normalize_extension(filename) in settings.allowed_extension_list


def resolve_document_path(
    file_path: str, settings: Optional[Settings] = None
) -> Path:
    """Resolve ``file_path`` and confine it to ``DOCUMENT_ROOT``.

    Relative paths resolve against the root. Absolute paths are accepted
    only when they already point inside it.
    """
    settings = settings or get_settings()
    root = settings.document_root_path
    root.mkdir(parents=True, exist_ok=True)

    raw = (file_path or "").strip()
    if not raw:
        raise UnsafePathError("file_path must not be empty")
    if "\x00" in raw:
        raise UnsafePathError("file_path contains a null byte")

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate

    # resolve() follows symlinks, so a link pointing outside the root
    # resolves outside it and is caught by the containment check below.
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        logger.warning("Rejected a path outside DOCUMENT_ROOT: %s", raw)
        raise UnsafePathError() from exc

    if not resolved.exists():
        raise FileNotFoundError_(f"File not found: {resolved.name}")
    if not resolved.is_file():
        raise UnsafePathError("file_path does not point to a regular file")
    return resolved


def validate_magic_bytes(path: Path, extension: str) -> None:
    """Confirm the file's leading bytes match its extension."""
    signatures = MAGIC_SIGNATURES.get(extension)
    if not signatures:
        return
    try:
        with path.open("rb") as handle:
            head = handle.read(8)
    except OSError as exc:
        raise UnsupportedFileTypeError(
            f"The file could not be read: {exc}"
        ) from exc
    if not any(head.startswith(sig) for sig in signatures):
        raise UnsupportedFileTypeError(
            f"File content does not match its '{extension}' extension",
            {"extension": extension},
        )


def validate_size(size_bytes: int, settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    if size_bytes <= 0:
        raise UnsupportedFileTypeError("The file is empty (0 bytes)")
    if size_bytes > settings.max_file_size_bytes:
        raise FileTooLargeError(
            f"The file is {size_bytes / 1048576:.1f} MB; the limit is "
            f"{settings.max_file_size_mb} MB",
            {"limit_mb": settings.max_file_size_mb},
        )


def validate_file(path: Path, settings: Optional[Settings] = None) -> str:
    """Run every file check. Returns the validated extension."""
    settings = settings or get_settings()
    extension = path.suffix.lower()

    if extension not in settings.allowed_extension_list:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension or '(none)'}'. Allowed: "
            f"{', '.join(settings.allowed_extension_list)}",
            {
                "extension": extension,
                "allowed": settings.allowed_extension_list,
            },
        )

    validate_size(path.stat().st_size, settings)
    validate_magic_bytes(path, extension)
    return extension


def validate_upload_name(
    filename: str, settings: Optional[Settings] = None
) -> str:
    """Validate an uploaded filename and return its extension.

    Only the basename is trusted — a client-supplied ``../../x.pdf``
    contributes nothing but ``x.pdf``.
    """
    settings = settings or get_settings()
    base = Path(filename or "").name
    extension = Path(base).suffix.lower()
    if extension not in settings.allowed_extension_list:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension or '(none)'}'. Allowed: "
            f"{', '.join(settings.allowed_extension_list)}",
            {"extension": extension},
        )
    return extension


__all__ = [
    "MAGIC_SIGNATURES",
    "MIME_BY_EXTENSION",
    "validate_identifier",
    "normalize_extension",
    "is_supported_extension",
    "resolve_document_path",
    "validate_magic_bytes",
    "validate_size",
    "validate_file",
    "validate_upload_name",
]
