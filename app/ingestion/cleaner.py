"""Text cleaning for legal documents.

The governing rule: **never change what the document says.** Everything
here removes artefacts of extraction — encoding damage, hard line wraps,
running headers, page numbers, OCR speckle. Wording, numbers, dates,
party names, defined terms and clause references pass through untouched,
and ``Page.raw_text`` preserves the pre-cleaning text so the original
stays recoverable.

Every heuristic is deliberately conservative, because in this domain a
false positive is far worse than a missed cleanup: dropping a line that
happened to look like a footer could delete a liability cap. Hence the
running-header detector requires a minimum page count *and* a repetition
ratio before it removes anything at all.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Set

from app.config import Settings, get_settings

# ---------------------------------------------------------------------
# Character-level repair
# ---------------------------------------------------------------------

#: Typographic ligatures PDF encoders emit as single glyphs. Left alone
#: they break search: "fi" in "confidential" would not match "fi".
LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}

#: Smart punctuation and invisible characters normalised to ASCII so
#: that a quoted clause matches whether it was typed or exported.
PUNCTUATION = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
    " ": " ", " ": " ", " ": " ", " ": " ",
    "​": "", "‌": "", "‍": "", "﻿": "",
    " ": "\n", " ": "\n",
    "­": "",  # soft hyphen
}

# Control characters except tab and newline.
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# The Unicode replacement character: evidence of a decoding failure.
REPLACEMENT_RE = re.compile("�+")

MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
SPACE_BEFORE_PUNCT_RE = re.compile(r" +([,;:.)\]])")
# "termi-\nnation" -> "termination"
HYPHEN_BREAK_RE = re.compile(r"(\w)[-‐‑]\s*\n\s*([a-z])")
# A newline that merely wraps a sentence, not one that ends it.
SOFT_WRAP_RE = re.compile(r"(?<![.:;!?•])\n(?!\s*\n)(?=[a-z(\"'])")

# ---------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------

PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*[-–—(\[]*\s*\d{1,4}\s*[-–—)\]]*\s*$"),
    re.compile(r"^\s*page\s+\d{1,4}(\s+of\s+\d{1,4})?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,4}\s*(?:/|\|)\s*\d{1,4}\s*$"),
    re.compile(r"^\s*[ivxlcdm]{1,7}\s*$", re.IGNORECASE),
]

ALNUM_RE = re.compile(r"[A-Za-z0-9]")
#: Lines near the top/bottom of a page considered for header detection.
EDGE_LINES = 3


def normalize_characters(text: str) -> str:
    """Unicode and malformed-character handling.

    NFKC folds compatibility forms (full-width digits, odd spaces) onto
    their canonical equivalents, then ligatures, smart punctuation,
    control characters and replacement characters are repaired. No word
    is altered.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in LIGATURES.items():
        text = text.replace(src, dst)
    for src, dst in PUNCTUATION.items():
        text = text.replace(src, dst)
    text = REPLACEMENT_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def repair_line_breaks(text: str) -> str:
    """Undo hard wrapping introduced by the page layout.

    Only a hyphen followed by a newline and a *lower-case* letter is
    joined, so "Anti-\\nCorruption" keeps its hyphen while "termi-\\nnation"
    is repaired.
    """
    if not text:
        return ""
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    return SOFT_WRAP_RE.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    if not text:
        return ""
    lines = [MULTI_SPACE_RE.sub(" ", line).rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return SPACE_BEFORE_PUNCT_RE.sub(r"\1", text).strip()


def is_page_number_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and any(p.match(stripped) for p in PAGE_NUMBER_PATTERNS)


def is_noise_line(line: str) -> bool:
    """Scanner speckle, rule lines and stray marks — never real text."""
    stripped = line.strip()
    if not stripped:
        return False
    if not ALNUM_RE.search(stripped):
        return True
    if len(stripped) <= 2 and not stripped.isdigit():
        return True
    alnum = len(ALNUM_RE.findall(stripped))
    return len(stripped) >= 8 and alnum / len(stripped) < 0.35


def remove_noise_lines(text: str) -> str:
    return "\n".join(ln for ln in text.split("\n") if not is_noise_line(ln))


# ---------------------------------------------------------------------
# Running headers and footers
# ---------------------------------------------------------------------


#: A line short enough, or explicitly page-labelled enough, that a digit
#: inside it is probably a page number rather than content.
PAGE_MARKER_WORDS = 6
PAGE_KEYWORD_RE = re.compile(r"\b(page|p\.|pg)\b", re.IGNORECASE)
DIGIT_RE = re.compile(r"\d")


def _fingerprint(line: str) -> str:
    """Case- and whitespace-normalised form of a line. No digit masking.

    Used for headers whose text is identical on every page.
    """
    key = " ".join(line.lower().split())
    return key if 4 <= len(key) <= 120 else ""


def _is_page_marker_like(line: str) -> bool:
    """Could the digits in this line be a page number?

    Only short lines, or lines explicitly saying "page", qualify.
    Without this guard, masking digits everywhere makes "Clause 3 of the
    Schedule applies" and "Clause 4 of the Schedule applies" look like
    the same repeated line, and real content gets deleted.
    """
    stripped = line.strip()
    if not DIGIT_RE.search(stripped):
        return False
    if PAGE_KEYWORD_RE.search(stripped):
        return True
    return len(stripped.split()) <= PAGE_MARKER_WORDS


def _masked_fingerprint(line: str) -> str:
    """Digit-masked form, so 'Page 3 of 20' and 'Page 4 of 20' collapse.

    Returns '' for any line where masking would be unsafe.
    """
    if not _is_page_marker_like(line):
        return ""
    key = re.sub(r"\d+", "#", " ".join(line.lower().split()))
    return key if 4 <= len(key) <= 120 else ""


def _edge_lines(page_text: str, window: int = EDGE_LINES) -> List[str]:
    lines = [ln.strip() for ln in page_text.split("\n") if ln.strip()]
    return lines[:window] + lines[-window:]


def detect_running_lines(
    page_texts: Sequence[str],
    min_ratio: float = 0.6,
    min_pages: int = 3,
) -> Set[str]:
    """Find lines repeating near the edge of most pages.

    Returns fingerprints, not literal lines, so a header carrying a page
    number is still matched. Returns an empty set — removing nothing —
    for documents shorter than ``min_pages``, where "it appears on every
    page" is not yet evidence of anything.
    """
    if len(page_texts) < min_pages:
        return set()

    counter: Counter = Counter()
    for text in page_texts:
        # set() so a line repeated within one page counts once.
        for line in set(_edge_lines(text)):
            # Two fingerprints per line: the exact text (a static
            # header) and, only where it is safe, a digit-masked form
            # (a header carrying a page number).
            for key in (_fingerprint(line), _masked_fingerprint(line)):
                if key:
                    counter[key] += 1

    threshold = max(min_pages - 1, int(len(page_texts) * min_ratio))
    return {key for key, count in counter.items() if count >= threshold}


def strip_furniture(
    text: str, running: Optional[Set[str]] = None, drop_page_numbers: bool = True
) -> str:
    """Remove page numbers and known running header/footer lines."""
    running = running or set()
    kept: List[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if drop_page_numbers and is_page_number_line(stripped):
            continue
        if running:
            exact = _fingerprint(stripped)
            masked = _masked_fingerprint(stripped)
            if (exact and exact in running) or (masked and masked in running):
                continue
        kept.append(line)
    return "\n".join(kept)


# ---------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------


def clean_text(
    text: str,
    running: Optional[Set[str]] = None,
    drop_noise: bool = True,
    drop_page_numbers: bool = True,
) -> str:
    """The full single-string cleaning pipeline, in dependency order."""
    text = normalize_characters(text)
    if drop_noise:
        text = remove_noise_lines(text)
    text = strip_furniture(text, running, drop_page_numbers)
    text = repair_line_breaks(text)
    return collapse_whitespace(text)


def count_removed_lines(before: str, after: str) -> int:
    """How many non-empty lines cleaning removed (for stats)."""
    def non_empty(value: str) -> int:
        return sum(1 for ln in value.split("\n") if ln.strip())

    return max(0, non_empty(before) - non_empty(after))


__all__ = [
    "LIGATURES",
    "PUNCTUATION",
    "EDGE_LINES",
    "normalize_characters",
    "repair_line_breaks",
    "collapse_whitespace",
    "is_page_number_line",
    "is_noise_line",
    "remove_noise_lines",
    "detect_running_lines",
    "strip_furniture",
    "clean_text",
    "count_removed_lines",
]
