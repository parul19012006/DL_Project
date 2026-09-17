"""Parser interface and registry.

Every format parser implements :class:`DocumentParser` and registers the
extensions it handles. The ingestion service looks up a parser by
extension and never imports a format module directly, so adding (say)
RTF later is one new module plus one ``register()`` call — no change to
the service, the API or the tests for existing formats.

A parser's job ends at producing raw :class:`Page` objects with their
blocks. Cleaning, section assignment and statistics are applied
uniformly afterwards by the service, so no format can drift into its own
cleaning behaviour.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from app.config import Settings
from app.ingestion.errors import UnsupportedFileTypeError
from app.models.document import DocumentMetadata, Page


class ParseOutcome:
    """A parser's raw result, before cleaning."""

    __slots__ = ("pages", "parser", "warnings")

    def __init__(
        self,
        pages: List[Page],
        parser: str,
        warnings: Optional[List[str]] = None,
    ) -> None:
        self.pages = pages
        self.parser = parser
        self.warnings = warnings or []


class DocumentParser(ABC):
    """Base class for a format parser."""

    #: Extensions this parser handles, lower-case and dot-prefixed.
    extensions: tuple = ()
    #: Short identifier recorded in ExtractionStats.parser.
    name: str = "parser"

    @abstractmethod
    def parse(
        self,
        path: Path,
        metadata: DocumentMetadata,
        settings: Settings,
        force_ocr: bool = False,
    ) -> ParseOutcome:
        """Extract pages and blocks. Raises an ``IngestionError`` on failure."""
        raise NotImplementedError


_REGISTRY: Dict[str, DocumentParser] = {}


def register(parser):
    """Register a parser for its extensions.

    Usable as a class decorator (``@register`` above a
    :class:`DocumentParser` subclass) or with an already-built instance.
    A class is instantiated here, so the registry always holds instances
    — looking one up and calling ``parse`` on a *class* would silently
    bind ``self`` to the first argument.

    The decorated class is returned unchanged, so ``@register`` does not
    interfere with subclassing or direct instantiation in tests.
    """
    instance = parser() if isinstance(parser, type) else parser
    for extension in instance.extensions:
        _REGISTRY[extension.lower()] = instance
    return parser


def get_parser(extension: str) -> DocumentParser:
    parser = _REGISTRY.get((extension or "").lower())
    if parser is None:
        raise UnsupportedFileTypeError(
            f"No parser is registered for '{extension}'",
            {"extension": extension},
        )
    return parser


def registered_extensions() -> List[str]:
    return sorted(_REGISTRY)


def clear_registry() -> None:
    """Test helper."""
    _REGISTRY.clear()


__all__ = [
    "DocumentParser",
    "ParseOutcome",
    "register",
    "get_parser",
    "registered_extensions",
    "clear_registry",
]
