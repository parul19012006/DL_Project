"""Document ingestion (Stage 2).

Importing this package registers the bundled parsers, so
``get_parser(".pdf")`` works without the caller importing the PDF module
by name.
"""

from app.ingestion import docx_parser as _docx_parser  # noqa: F401
from app.ingestion import pdf_parser as _pdf_parser  # noqa: F401
from app.ingestion.base import get_parser, registered_extensions
from app.ingestion.service import extract_document, extract_many

__all__ = [
    "extract_document",
    "extract_many",
    "get_parser",
    "registered_extensions",
]
