"""PDF extraction with PyMuPDF, plus per-page OCR fallback.

Text is pulled as *blocks* rather than as one string per page, because
PyMuPDF's block output preserves the layout grouping a legal document
depends on: a clause heading arrives as its own block, separate from the
clause body. Blocks are sorted top-to-bottom then left-to-right, which
recovers reading order for the single-column layouts that contracts
overwhelmingly use.

Page-level preservation is strict. Every page of the file produces
exactly one :class:`Page` at its real 1-based number, including blank
ones, so page 17 of the output is page 17 of the PDF and a citation can
never drift.

Per-page failures are contained: an unreadable page becomes an empty
page plus a warning, and the rest of the document still ingests.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from app.config import Settings
from app.ingestion import ocr as ocr_module
from app.ingestion.base import DocumentParser, ParseOutcome, register
from app.ingestion.errors import (
    CorruptDocumentError,
    EncryptedDocumentError,
    ParserUnavailableError,
)
from app.ingestion.sections import detect_section
from app.logging_config import get_logger
from app.models.document import (
    BlockKind,
    DocumentMetadata,
    ExtractionMethod,
    Page,
    TextBlock,
)

logger = get_logger(__name__)


def _import_pymupdf():
    try:
        import pymupdf

        return pymupdf
    except ImportError:  # pragma: no cover - PyMuPDF < 1.24
        try:
            import fitz

            return fitz
        except ImportError as exc:
            raise ParserUnavailableError(
                "PyMuPDF (pymupdf) is required to read PDF files"
            ) from exc


def _page_blocks(page) -> List[Tuple[str, tuple]]:
    """Return ``(text, bbox)`` per text block, in reading order."""
    try:
        raw = page.get_text("blocks") or []
    except Exception as exc:  # pragma: no cover - malformed page
        logger.debug("Block extraction failed, falling back to plain text: %s", exc)
        try:
            text = page.get_text() or ""
        except Exception:
            return []
        return [(text.strip(), ())] if text.strip() else []

    # Tuples are (x0, y0, x1, y1, text, block_no, block_type);
    # block_type 0 is text, 1 is an image.
    blocks = [b for b in raw if len(b) > 4 and (len(b) < 7 or b[6] == 0)]
    # Round the vertical coordinate so that words sharing a line are not
    # reordered by sub-pixel differences.
    blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
    return [
        (str(b[4]).strip(), (b[0], b[1], b[2], b[3]))
        for b in blocks
        if str(b[4]).strip()
    ]


def _to_text_blocks(
    raw_blocks: List[Tuple[str, tuple]],
    page_number: int,
    detect: bool,
    ocr_used: bool,
    carried_section: Optional[str],
) -> Tuple[List[TextBlock], Optional[str]]:
    """Turn raw block text into TextBlocks, tracking the open section.

    ``carried_section`` enters from the previous page: a clause that
    starts on page 4 and continues on page 5 keeps its label on page 5.
    """
    out: List[TextBlock] = []
    section = carried_section
    index = 0

    for text, bbox in raw_blocks:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue

        heading = detect_section(lines[0]) if detect else None
        if heading:
            section = heading
            out.append(
                TextBlock(
                    text=lines[0],
                    page_number=page_number,
                    block_index=index,
                    kind=BlockKind.HEADING,
                    section=section,
                    ocr=ocr_used,
                    bbox=bbox or None,
                )
            )
            index += 1
            body = "\n".join(lines[1:]).strip()
            if not body:
                continue
        else:
            body = "\n".join(lines)

        out.append(
            TextBlock(
                text=body,
                page_number=page_number,
                block_index=index,
                kind=BlockKind.PARAGRAPH,
                section=section,
                ocr=ocr_used,
                bbox=bbox or None,
            )
        )
        index += 1

    return out, section


@register
class PDFParser(DocumentParser):
    extensions = (".pdf",)
    name = "pymupdf"

    def parse(
        self,
        path: Path,
        metadata: DocumentMetadata,
        settings: Settings,
        force_ocr: bool = False,
    ) -> ParseOutcome:
        fitz = _import_pymupdf()
        warnings: List[str] = []

        try:
            document = fitz.open(str(path))
        except Exception as exc:
            raise CorruptDocumentError(
                f"The PDF could not be opened: {exc}"
            ) from exc

        if getattr(document, "needs_pass", False):
            document.close()
            raise EncryptedDocumentError(
                "The PDF is password protected and cannot be read"
            )

        ocr_wanted = force_ocr or settings.ocr_enabled
        ocr_ready = ocr_module.ocr_available(settings) if ocr_wanted else False
        pages: List[Page] = []
        section: Optional[str] = None
        ocr_pages = 0
        skipped_ocr_pages = 0

        try:
            page_count = document.page_count

            # PyMuPDF is deliberately lenient: rather than refusing a
            # damaged file it repairs what it can and may report zero
            # pages. A PDF with no pages is not a valid PDF, so treat
            # that as damage — reporting it as "empty" would send the
            # user looking for missing text in a file that never parsed.
            if page_count == 0:
                repaired = bool(getattr(document, "is_repaired", False))
                raise CorruptDocumentError(
                    "The PDF is damaged: no pages could be read"
                    + (" (the file structure had to be repaired)" if repaired else ""),
                    {"repaired": repaired},
                )

            # A file that only parsed after repair still yields pages,
            # but the user should know the source was damaged.
            if getattr(document, "is_repaired", False):
                warnings.append(
                    "The PDF structure was damaged and had to be repaired; "
                    "some content may be missing"
                )

            for index in range(page_count):
                page_number = index + 1
                try:
                    pdf_page = document.load_page(index)
                except Exception as exc:
                    # One bad page must not lose the whole document.
                    logger.warning(
                        "Page %d of %s is unreadable: %s",
                        page_number,
                        metadata.filename,
                        exc,
                    )
                    warnings.append(f"Page {page_number} could not be read")
                    pages.append(
                        Page(
                            page_number=page_number,
                            method=ExtractionMethod.NONE,
                        )
                    )
                    continue

                raw_blocks = _page_blocks(pdf_page)
                native_text = "\n\n".join(t for t, _ in raw_blocks)
                method = ExtractionMethod.NATIVE
                used_ocr = False

                needs_ocr = force_ocr or ocr_module.page_needs_ocr(
                    native_text, settings
                )
                if needs_ocr and ocr_wanted:
                    if not ocr_ready:
                        skipped_ocr_pages += 1
                    else:
                        ocr_text = ocr_module.ocr_pdf_page(pdf_page, settings)
                        if ocr_module.ocr_improves(native_text, ocr_text, settings):
                            raw_blocks = [
                                (part.strip(), ())
                                for part in ocr_text.split("\n\n")
                                if part.strip()
                            ] or [(ocr_text.strip(), ())]
                            method = ExtractionMethod.OCR
                            used_ocr = True
                            ocr_pages += 1

                blocks, section = _to_text_blocks(
                    raw_blocks,
                    page_number,
                    settings.detect_sections,
                    used_ocr,
                    section,
                )
                page_text = "\n\n".join(b.text for b in blocks)
                if not page_text.strip():
                    method = ExtractionMethod.NONE

                pages.append(
                    Page(
                        page_number=page_number,
                        blocks=blocks,
                        text=page_text,
                        raw_text=page_text if settings.keep_raw_text else None,
                        method=method,
                    )
                )
        finally:
            document.close()

        if skipped_ocr_pages:
            warnings.append(
                f"{skipped_ocr_pages} page(s) appear to be scanned but OCR is "
                "unavailable; their text was not extracted"
            )
            logger.warning(
                "OCR unavailable for %d scanned page(s) of %s",
                skipped_ocr_pages,
                metadata.filename,
            )

        parser_name = "pymupdf+ocr" if ocr_pages else "pymupdf"
        return ParseOutcome(pages=pages, parser=parser_name, warnings=warnings)


__all__ = ["PDFParser"]
