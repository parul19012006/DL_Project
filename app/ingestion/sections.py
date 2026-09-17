"""Section and heading detection.

Legal documents are organised into numbered clauses, articles and
schedules. Recovering that structure at ingestion time pays off twice
later: Stage 3 can avoid splitting a chunk across two clauses, and
Stage 6 can cite "Clause 4.2 (Termination), page 17" instead of just a
page number.

Detection is heuristic and explicitly best-effort — ``section`` is an
``Optional[str]`` throughout the pipeline and nothing depends on it
being present. A missed heading costs a little retrieval precision; a
*false* heading would mislabel real clause text, so the patterns are
tight and a line that looks like prose is never treated as a heading.
"""

from __future__ import annotations

import re
from typing import List, Optional

#: "4." / "4.2" / "12.3.4" followed by a title
NUMBERED_RE = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,3}){0,3})[.)]?\s+([A-Z][^\n]{2,80})$"
)
#: "ARTICLE V - TERMINATION", "SCHEDULE B", "Clause 7: Liability"
LABELLED_RE = re.compile(
    r"^\s*(ARTICLE|SECTION|CLAUSE|SCHEDULE|EXHIBIT|ANNEX(?:URE)?|APPENDIX|PART|"
    r"RECITAL|PREAMBLE)\s+([IVXLCDM]+|\d{1,3}|[A-Z])\b[\s:.–-]*(.{0,80})$",
    re.IGNORECASE,
)
#: "TERMINATION FOR CONVENIENCE" — an all-caps heading line
ALLCAPS_RE = re.compile(r"^\s*([A-Z][A-Z0-9 ,.'&()/–-]{3,80})\s*$")
#: A numbered heading with no title: "4.", "12.3"
BARE_NUMBER_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,3}){1,3})\s*$")

#: Clause names common enough to recognise on their own.
SECTION_KEYWORDS = {
    "definitions", "interpretation", "term", "renewal", "termination",
    "liability", "limitation of liability", "indemnity", "indemnification",
    "confidentiality", "governing law", "jurisdiction", "payment",
    "fees", "fees and payment", "warranty", "warranties", "force majeure",
    "assignment", "dispute resolution", "arbitration", "notices",
    "severability", "entire agreement", "scope of work", "deliverables",
    "intellectual property", "data protection", "privacy", "insurance",
    "compliance", "audit", "subcontracting", "amendment", "waiver",
    "counterparts", "survival", "representations", "covenants",
}

#: Words that mark a line as page furniture, never a heading.
FURNITURE_WORDS = {"page", "continued", "confidential", "draft", "exhibit page"}

MAX_HEADING_CHARS = 120
MAX_HEADING_WORDS = 14


def is_heading(line: str) -> bool:
    """Does this line look like a legal heading?"""
    stripped = (line or "").strip()
    if not stripped or len(stripped) > MAX_HEADING_CHARS:
        return False

    words = stripped.split()
    if words[0].lower().rstrip(":.") in FURNITURE_WORDS:
        return False
    if len(words) > MAX_HEADING_WORDS:
        return False

    # Sentence-ending punctuation on a long line means prose. Short
    # numbered headings ("4. Termination.") legitimately end with a dot.
    if stripped.endswith((";", ",")) or (
        stripped.endswith(".") and len(words) > 8 and not NUMBERED_RE.match(stripped)
    ):
        return False

    if LABELLED_RE.match(stripped) or NUMBERED_RE.match(stripped):
        return True
    if BARE_NUMBER_RE.match(stripped):
        return True
    if ALLCAPS_RE.match(stripped) and len(words) <= 12:
        return True

    lowered = stripped.lower().rstrip(":.")
    if lowered in SECTION_KEYWORDS:
        return True
    # "Termination:" — a short title-case line ending in a colon
    if stripped.endswith(":") and len(words) <= 8 and stripped[0].isupper():
        return True
    return False


def normalize_section(line: str) -> str:
    """Collapse a heading line into a compact, stable label."""
    label = " ".join((line or "").split()).strip(" :.–-")
    return label[:100].rstrip() if len(label) > 100 else label


def detect_section(line: str) -> Optional[str]:
    """Return a section label if ``line`` is a heading, else ``None``."""
    if not is_heading(line):
        return None
    return normalize_section(line) or None


def assign_sections(lines: List[str]) -> List[Optional[str]]:
    """Walk lines top-down, carrying the current section forward.

    Returns one label per input line, so a caller can attach the section
    a line belongs to without re-scanning.
    """
    current: Optional[str] = None
    out: List[Optional[str]] = []
    for line in lines:
        found = detect_section(line)
        if found:
            current = found
        out.append(current)
    return out


__all__ = [
    "SECTION_KEYWORDS",
    "is_heading",
    "detect_section",
    "normalize_section",
    "assign_sections",
]
