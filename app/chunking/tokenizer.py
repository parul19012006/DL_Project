"""Token counting.

Chunk size is expressed in *tokens* because that is the unit the
embedding model and the LLM context window are measured in. But the
tokenizer that matters is the one belonging to the embedding model, and
that model is not chosen until Stage 4. So counting sits behind an
interface with two implementations:

``heuristic`` (default)
    A deterministic estimator with no dependencies and no model
    download. Counts word-pieces the way BPE roughly does — short words
    cost one token, longer words cost one per ~4 characters, punctuation
    and digits cost their own. Accurate to within roughly ±15% of
    cl100k / WordPiece on English legal prose, which is why the target
    is documented as "approximately 500 tokens".

``tiktoken``
    Exact GPT-family counts when ``tiktoken`` is installed.

Stage 4 can register the real tokenizer of whatever embedding model is
chosen via :func:`set_token_counter`, and every chunk size in the system
becomes exact without another line changing. That is the reason for the
indirection: hard-coding ``len(text.split())`` here would quietly bake a
wrong assumption into the index.
"""

from __future__ import annotations

import math
import re
import threading
from typing import List, Optional, Protocol

from app.config import Settings, get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

#: Words, numbers (including 4,50,000 and 12.3), and single punctuation.
_PIECE_RE = re.compile(r"[A-Za-z]+|\d[\d.,]*|[^\sA-Za-z\d]")

#: Characters per sub-word piece for words longer than this.
CHARS_PER_PIECE = 4
SHORT_WORD_CHARS = 4


class TokenCounter(Protocol):
    """Anything that can count tokens in a string."""

    name: str

    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """Dependency-free, deterministic token estimator.

    Deliberately *slightly* over-estimates rather than under-estimates:
    a chunk that turns out smaller than budgeted is harmless, one that
    overflows the embedding model's window gets silently truncated,
    losing text the user believes is indexed.
    """

    name = "heuristic"

    def count(self, text: str) -> int:
        if not text:
            return 0

        total = 0
        for piece in _PIECE_RE.findall(text):
            first = piece[0]
            if first.isalpha():
                # BPE keeps common short words whole and splits longer
                # ones into sub-words.
                total += (
                    1
                    if len(piece) <= SHORT_WORD_CHARS
                    else math.ceil(len(piece) / CHARS_PER_PIECE)
                )
            elif first.isdigit():
                # Digit runs tokenise densely: roughly one token per
                # 2-3 characters.
                total += max(1, math.ceil(len(piece) / 3))
            else:
                total += 1

        if total:
            return total

        # No recognisable pieces but non-empty text: scripts without
        # spaces (CJK), or pure symbols. Fall back to characters so a
        # long string can never be counted as zero tokens and slip
        # through the size guard.
        stripped = "".join(text.split())
        return math.ceil(len(stripped) / 2) if stripped else 0


class TiktokenCounter:  # pragma: no cover - optional dependency
    """Exact counts for GPT-family tokenizers."""

    name = "tiktoken"

    def __init__(self, encoding: str = "cl100k_base") -> None:
        import tiktoken

        self._encoding = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken:{encoding}"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

_counter: Optional[TokenCounter] = None
_lock = threading.Lock()


def build_token_counter(settings: Optional[Settings] = None) -> TokenCounter:
    settings = settings or get_settings()
    choice = (settings.tokenizer or "heuristic").strip().lower()

    if choice == "tiktoken":
        try:
            return TiktokenCounter(settings.tiktoken_encoding)
        except Exception as exc:
            logger.warning(
                "tiktoken is unavailable (%s); using the heuristic counter. "
                "Chunk sizes remain approximate.",
                exc,
            )
    return HeuristicTokenCounter()


def get_token_counter(settings: Optional[Settings] = None) -> TokenCounter:
    """Process-wide counter. Cheap to build, but shared for consistency."""
    global _counter
    if _counter is None:
        with _lock:
            if _counter is None:
                _counter = build_token_counter(settings)
    return _counter


def set_token_counter(counter: Optional[TokenCounter]) -> None:
    """Install a specific counter (Stage 4's embedding tokenizer, or a
    test double). ``None`` resets to the configured default."""
    global _counter
    with _lock:
        _counter = counter


def count_tokens(text: str, counter: Optional[TokenCounter] = None) -> int:
    return (counter or get_token_counter()).count(text)


def estimate_words_for_tokens(tokens: int) -> int:
    """Rough inverse, used to size a word window.

    Intentionally conservative: under-estimating the word count keeps
    the resulting window inside the token budget.
    """
    return max(1, int(tokens * 0.75))


def truncate_to_tokens(
    text: str, max_tokens: int, counter: Optional[TokenCounter] = None
) -> str:
    """Trim ``text`` so its token count fits, on a whitespace boundary."""
    counter = counter or get_token_counter()
    if max_tokens <= 0:
        return ""
    if counter.count(text) <= max_tokens:
        return text

    words = text.split()
    if not words:
        return ""

    # Binary search the longest prefix that fits.
    low, high = 0, len(words)
    while low < high:
        mid = (low + high + 1) // 2
        if counter.count(" ".join(words[:mid])) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return " ".join(words[:low])


__all__ = [
    "TokenCounter",
    "HeuristicTokenCounter",
    "TiktokenCounter",
    "build_token_counter",
    "get_token_counter",
    "set_token_counter",
    "count_tokens",
    "truncate_to_tokens",
    "estimate_words_for_tokens",
]
