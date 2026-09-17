"""OCR fallback for scanned PDF pages.

Both the ``pytesseract`` package and the ``tesseract`` binary are treated
as optional. A deployment without them still ingests text PDFs and DOCX
files normally; scanned pages simply come back empty and the document
carries a warning saying so. OCR being unavailable is a degraded mode,
never a crash.

The decision is per page, not per document — a scanned signature page
appended to a typed contract gets OCR'd while the other 30 pages take
the fast native path.

Two guards keep OCR from making things worse:

* it runs only when native extraction yielded less than
  ``OCR_MIN_CHARS_PER_PAGE`` characters
* its output replaces the native text only if it is at least
  ``OCR_MIN_GAIN_RATIO`` times longer, so OCR noise never displaces a
  short but real text layer
"""

from __future__ import annotations

import io
import shutil
from typing import Optional

from app.config import Settings, get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_AVAILABLE: Optional[bool] = None
_VERSION: Optional[str] = None


def _import_pymupdf():
    """PyMuPDF renamed its import from ``fitz`` to ``pymupdf`` in 1.24."""
    try:
        import pymupdf

        return pymupdf
    except ImportError:  # pragma: no cover - older installs
        import fitz

        return fitz


def ocr_available(settings: Optional[Settings] = None) -> bool:
    """Is a usable Tesseract installation reachable? Cached."""
    global _AVAILABLE, _VERSION
    if _AVAILABLE is not None:
        return _AVAILABLE

    settings = settings or get_settings()
    try:
        import pytesseract
    except ImportError:
        logger.warning("pytesseract is not installed - OCR is disabled")
        _AVAILABLE = False
        return False

    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    command = pytesseract.pytesseract.tesseract_cmd or "tesseract"
    if shutil.which(command) is None:
        logger.warning(
            "The tesseract binary ('%s') was not found - OCR is disabled",
            command,
        )
        _AVAILABLE = False
        return False

    try:
        _VERSION = str(pytesseract.get_tesseract_version())
    except Exception:  # pragma: no cover
        _VERSION = "unknown"
    logger.info("OCR available (tesseract %s)", _VERSION)
    _AVAILABLE = True
    return True


def ocr_version(settings: Optional[Settings] = None) -> Optional[str]:
    ocr_available(settings)
    return _VERSION


def reset_availability_cache() -> None:
    """Test helper — re-probe on the next call."""
    global _AVAILABLE, _VERSION
    _AVAILABLE = None
    _VERSION = None


def page_needs_ocr(native_text: str, settings: Optional[Settings] = None) -> bool:
    """Was native extraction thin enough to suspect a scanned page?"""
    settings = settings or get_settings()
    if not settings.ocr_enabled:
        return False
    # Count non-whitespace: a page of blank lines is not a page of text.
    return len("".join((native_text or "").split())) < settings.ocr_min_chars_per_page


def ocr_image(image, settings: Optional[Settings] = None) -> str:
    """Run Tesseract over a PIL image. Returns '' when unavailable."""
    settings = settings or get_settings()
    if not ocr_available(settings):
        return ""
    import pytesseract

    try:
        # --oem 1 = LSTM engine, --psm 3 = automatic page segmentation,
        # the right defaults for a full page of prose.
        return pytesseract.image_to_string(
            image, lang=settings.ocr_language, config="--oem 1 --psm 3"
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Tesseract failed on an image: %s", exc)
        return ""


def ocr_pdf_page(page, settings: Optional[Settings] = None) -> str:
    """Rasterise a PyMuPDF page and OCR it.

    Never raises: a failure on one page must not lose the other 200.
    """
    settings = settings or get_settings()
    if not ocr_available(settings):
        return ""
    try:
        fitz = _import_pymupdf()
        from PIL import Image

        # 72 dpi is the PDF user-space unit, so this is the zoom factor
        # needed to reach the configured DPI.
        zoom = max(1.0, settings.ocr_dpi / 72.0)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
            return ocr_image(image, settings) or ""
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning(
            "OCR failed on page %s: %s", getattr(page, "number", "?"), exc
        )
        return ""


def ocr_improves(native_text: str, ocr_text: str, settings: Optional[Settings] = None) -> bool:
    """Should OCR output replace the native text?

    Requires a real gain, not merely a longer string, so speckle
    transcribed as garbage cannot displace a short real text layer.
    """
    settings = settings or get_settings()
    ocr_len = len("".join((ocr_text or "").split()))
    native_len = len("".join((native_text or "").split()))
    if ocr_len == 0:
        return False
    if native_len == 0:
        return True
    return ocr_len >= native_len * settings.ocr_min_gain_ratio


__all__ = [
    "ocr_available",
    "ocr_version",
    "reset_availability_cache",
    "page_needs_ocr",
    "ocr_image",
    "ocr_pdf_page",
    "ocr_improves",
]
