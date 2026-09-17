"""Boundary detection — where it is safe to cut legal text.

The chunker never cuts arbitrarily. It walks a hierarchy of boundaries
and uses the strongest one that fits:

    section  >  paragraph  >  enumerated clause item  >  sentence  >  words

Only the last is a true fallback, used when a single sentence is longer
than the whole budget — which in legal drafting happens often enough to
need handling rather than assuming away.

Two legal-specific behaviours live here:

**Enumerated items.** ``(a) … (b) … (c)`` and ``(i) … (ii)`` lists are
extremely common in contracts, and splitting one mid-item produces a
fragment that reads as an obligation without its subject. When a
paragraph must be split, item boundaries are preferred over sentence
boundaries, because an item is the smaller self-contained unit.

**Abbreviations.** A naive "split on full stop" breaks "Art. 9", "No. 4",
"Inc.", "e.g.", "U.S.C." and every section cross-reference in the
document. The sentence splitter knows the common ones and requires the
next character to look like a sentence start.
"""

from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------
# Paragraphs
# ---------------------------------------------------------------------

_PARAGRAPH_RE = re.compile(r"\n\s*\n+")


def split_paragraphs(text: str) -> List[str]:
    """Split on blank lines. Always returns at least one piece for
    non-empty input."""
    if not text or not text.strip():
        return []
    parts = [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]
    return parts or [text.strip()]


# ---------------------------------------------------------------------
# Enumerated clause items
# ---------------------------------------------------------------------

#: Start of an enumerated item at the beginning of a line or after a
#: separator: "(a)", "(iv)", "(1)", "a)", "1.", "4.2", "•", "-".
_ITEM_PATTERN = (
    r"(?:"
    r"\(\s*(?:[a-zA-Z]|[ivxlcdmIVXLCDM]{1,7}|\d{1,3})\s*\)"   # (a) (iv) (12)
    r"|(?:[a-z]|[ivx]{1,6})\)"                                  # a)  iv)
    # Multi-level clause numbers need no trailing punctuation:
    # "4.2 Termination" is a heading, "4.2. Termination" is too.
    r"|\d{1,2}(?:\.\d{1,3}){1,3}[.)]?"                         # 4.2  12.3.4)
    # Single-level numbers do need it, or "4 units" would match.
    r"|\d{1,2}[.)]"                                             # 1.  12)
    r"|[•●▪‣⁃]"                       # bullets
    r")"
)
#: An item boundary: the pattern at a line start, or after "; " / ": ".
_ITEM_SPLIT_RE = re.compile(
    rf"(?:(?<=\n)|(?<=^)|(?<=;\s)|(?<=:\s))\s*(?={_ITEM_PATTERN}\s)"
)
_ITEM_START_RE = re.compile(rf"^\s*{_ITEM_PATTERN}\s")


def looks_enumerated(text: str) -> bool:
    """Does this paragraph contain an enumerated list?"""
    return len(_ITEM_SPLIT_RE.split(text)) > 1


def split_enumerated_items(text: str) -> List[str]:
    """Split a paragraph at enumerated item boundaries.

    The lead-in ("The Vendor shall:") stays as its own piece so the
    chunker can keep it attached to the first item when there is room.
    """
    if not text or not text.strip():
        return []
    parts = [p.strip() for p in _ITEM_SPLIT_RE.split(text) if p.strip()]
    return parts or [text.strip()]


def is_item_start(text: str) -> bool:
    return bool(_ITEM_START_RE.match(text or ""))


# ---------------------------------------------------------------------
# Sentences
# ---------------------------------------------------------------------

#: Abbreviations that end in a full stop but not a sentence. Kept
#: explicit rather than clever: a missed abbreviation costs one bad
#: split, an over-eager rule costs a merged pair of sentences.
_ABBREVIATIONS = [
    "Art", "Arts", "Sec", "Secs", "Cl", "Sch", "Ex", "Para", "Paras",
    "No", "Nos", "Fig", "pp", "p", "vs", "v", "cf", "al", "seq",
    "Inc", "Ltd", "LLC", "LLP", "Plc", "Co", "Corp", "Pvt", "Pte", "GmbH",
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Hon", "St", "Jr", "Sr",
    "approx", "est", "incl", "excl", "min", "max",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sept", "Sep",
    "Oct", "Nov", "Dec",
]
# The split point is the position *after* the full stop, so each
# lookbehind must include the stop itself — "(?<!\bArt)" would test the
# three characters before the position, which are "rt." and never match.
_ABBREV_LOOKBEHIND = "".join(
    rf"(?<!\b{re.escape(a)}\.)" for a in _ABBREVIATIONS
)
#: Also guard single initials ("J. Smith") and dotted acronyms ("U.S.C.").
_INITIAL_GUARD = r"(?<!\b[A-Z]\.)(?<![A-Z]\.[A-Z]\.)"

_SENTENCE_RE = re.compile(
    rf"{_ABBREV_LOOKBEHIND}{_INITIAL_GUARD}"
    r"(?<=[.!?])[\"')\]]*\s+"
    r"(?=[\"'(\[]*[A-Z0-9])"
)


def split_sentences(text: str) -> List[str]:
    """Abbreviation-aware sentence split.

    Returns the whole string as one piece when it contains no detectable
    sentence boundary — legal prose regularly runs to a single 300-word
    sentence, and pretending otherwise would cut it in the wrong place.
    """
    if not text or not text.strip():
        return []
    parts = [p.strip() for p in _SENTENCE_RE.split(text) if p and p.strip()]
    return parts or [text.strip()]


# ---------------------------------------------------------------------
# Words (last resort)
# ---------------------------------------------------------------------


def split_words(text: str, window: int, step: int) -> List[str]:
    """Sliding word window. Used only for an oversized single sentence.

    ``step`` below ``window`` produces overlapping windows, so a clause
    cut in the middle still appears whole in one of the pieces.
    """
    words = (text or "").split()
    if not words:
        return []
    window = max(1, window)
    step = max(1, min(step, window))

    out: List[str] = []
    start = 0
    while start < len(words):
        out.append(" ".join(words[start : start + window]))
        if start + window >= len(words):
            break
        start += step
    return out


def split_characters(text: str, window: int, step: int) -> List[str]:
    """Absolute fallback for text with no whitespace at all.

    A scanned table, a base64 blob, or a script that does not space its
    words would otherwise be a single unsplittable token.
    """
    stripped = (text or "").strip()
    if not stripped:
        return []
    window = max(1, window)
    step = max(1, min(step, window))

    out: List[str] = []
    start = 0
    while start < len(stripped):
        out.append(stripped[start : start + window])
        if start + window >= len(stripped):
            break
        start += step
    return out


def has_whitespace(text: str) -> bool:
    return bool(text) and any(ch.isspace() for ch in text.strip())


__all__ = [
    "split_paragraphs",
    "split_enumerated_items",
    "split_sentences",
    "split_words",
    "split_characters",
    "looks_enumerated",
    "is_item_start",
    "has_whitespace",
]
