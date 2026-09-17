"""Near-duplicate removal.

Duplicates reach the candidate list by three routes, and all three are
normal rather than exceptional in a legal document store:

1. **Chunk overlap.** Stage 3 deliberately repeats ~50 tokens between
   neighbouring chunks so an answer straddling a boundary is retrievable
   from either side. Two adjacent chunks are therefore *slightly*
   similar by design — and must not be collapsed, or the overlap
   mechanism would be undone by the thing meant to benefit from it.
2. **Boilerplate.** Standard clauses — notices, severability, entire
   agreement — are near-identical across every contract a firm holds.
3. **Re-uploads.** The same agreement filed twice under two document
   ids, which happens constantly in practice.

Cases 2 and 3 waste the generation budget: eight retrieved passages that
are really two distinct passages give the LLM a third of the context it
could have had.

**The method: character shingles and Jaccard similarity.** Text is
lowercased, punctuation-flattened and cut into overlapping character
n-grams; two passages are duplicates when the overlap between their
shingle sets exceeds a threshold. Character shingles (rather than word
shingles) tolerate the small differences that matter least — an OCR'd
copy of the same clause, a different party name, a reflowed line break —
while still separating genuinely different clauses.

**Why not embeddings.** Cosine similarity between two chunk vectors
*would* be a better duplicate detector, and it is available for the
semantic candidates. It is not used because the keyword candidates have
no vector in hand and fetching or computing ~18 of them per query costs
an embedding round-trip for a job that string comparison does well
enough. If duplicate detection ever needs to be subtler than this, that
is the upgrade path.

**Nothing is lost silently.** The survivor keeps the removed chunk_ids
in ``Candidate.duplicates``, so the citation for the second copy is
still available to the caller even though its text is not repeated.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Set, Tuple

from app.logging_config import get_logger
from app.retrieval.base import Candidate

logger = get_logger(__name__)

_NORMALISE_RE = re.compile(r"[^a-z0-9 ]+")
_SPACE_RE = re.compile(r"\s+")


def normalize_for_comparison(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Two copies of a clause that differ only in quotation marks, hyphens
    or line breaks compare equal after this.
    """
    lowered = (text or "").lower()
    lowered = _NORMALISE_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", lowered).strip()


def shingles(text: str, size: int = 5) -> Set[str]:
    """Overlapping character n-grams of the normalised text.

    Short passages (below one shingle) return the whole string as a
    single shingle rather than an empty set, so two identical short
    chunks still compare as identical.
    """
    normalised = normalize_for_comparison(text)
    if not normalised:
        return set()
    size = max(2, int(size))
    if len(normalised) <= size:
        return {normalised}
    return {normalised[i : i + size] for i in range(len(normalised) - size + 1)}


def jaccard(left: Set[str], right: Set[str]) -> float:
    """Set overlap in [0, 1]. Two empty sets are identical, not undefined."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / len(left | right)


def _is_adjacent_overlap(a: Candidate, b: Candidate) -> bool:
    """Neighbouring chunks of one document, whose overlap is intentional.

    Collapsing these would defeat Stage 3's overlap, so they are exempt
    from removal unless they are *exactly* identical (handled before this
    is consulted).
    """
    if a.chunk.document_id != b.chunk.document_id:
        return False
    return abs(a.chunk.chunk_index - b.chunk.chunk_index) == 1


def remove_duplicates(
    candidates: Sequence[Candidate],
    threshold: float = 0.85,
    shingle_size: int = 5,
    across_documents: bool = True,
    keep_adjacent: bool = True,
) -> Tuple[List[Candidate], int]:
    """Collapse duplicates, keeping the best-scoring copy.

    Input order is assumed to be best-first: the first occurrence of a
    passage survives and later copies fold into it, which means the copy
    retrieval ranked highest is the one that reaches the user.

    Returns ``(kept, removed_count)``.
    """
    if not candidates:
        return [], 0

    kept: List[Candidate] = []
    kept_shingles: List[Set[str]] = []
    exact: Dict[str, Candidate] = {}
    removed = 0

    for candidate in candidates:
        normalised = normalize_for_comparison(candidate.text)

        # -- exact duplicates ------------------------------------------
        # Checked first and without exemption: identical text is
        # identical text, even in adjacent chunks of one document.
        twin = exact.get(normalised) if normalised else None
        if twin is not None:
            twin.duplicates.append(candidate.chunk_id)
            removed += 1
            continue

        # -- near duplicates -------------------------------------------
        fingerprint = shingles(candidate.text, shingle_size)
        duplicate_of = None
        for existing, existing_shingles in zip(kept, kept_shingles):
            if not across_documents and (
                existing.chunk.document_id != candidate.chunk.document_id
            ):
                continue
            if keep_adjacent and _is_adjacent_overlap(existing, candidate):
                continue
            if jaccard(existing_shingles, fingerprint) >= threshold:
                duplicate_of = existing
                break

        if duplicate_of is not None:
            duplicate_of.duplicates.append(candidate.chunk_id)
            removed += 1
            continue

        kept.append(candidate)
        kept_shingles.append(fingerprint)
        if normalised:
            exact[normalised] = candidate

    if removed:
        logger.debug("Duplicate removal dropped %d candidate(s)", removed)
    return kept, removed


__all__ = [
    "remove_duplicates",
    "shingles",
    "jaccard",
    "normalize_for_comparison",
]
