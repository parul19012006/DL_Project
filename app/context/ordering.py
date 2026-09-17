"""Source ordering — the order the model reads the evidence in.

Selection decides *which* passages survive; this decides the order they
are rendered in, and the two are deliberately different decisions.
Selection is a competition, so it happens in relevance order. Reading is
comprehension, so it happens in document order.

**Why document order is the default.** Legal text is sequential and
self-referential. Clause 4 says "subject to Clause 8"; Clause 8 qualifies
Clause 4; a definition in Clause 1 governs every clause after it. Two
passages from the same agreement presented in the order the agreement
states them read as a document. Presented in retrieval-score order they
read as a pile, and a model asked to reason across them has to
reconstruct the sequence before it can use it. Worse, adjacent chunks of
one clause — which Stage 3 deliberately overlaps — can end up separated
by a passage from a different contract, which is exactly the
juxtaposition most likely to produce a confidently blended answer.

So: documents are ordered by the best score anything in them achieved
(the most relevant document comes first — relevance still drives the
top-level order), and within a document passages appear in the order
they appear in the document.

**Why relevance order remains available.** It is the right choice when
the passages are independent of each other and the model should see the
strongest evidence first — a wide search across many unrelated documents,
or a model known to weight early context heavily.

Whichever runs, numbering is assigned **after** ordering, so "Source 3"
always means the third block in the text the model was given. Numbering
before ordering would produce a context whose labels do not match its own
sequence, and every citation resolved by number would be wrong.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from app.context.base import ContextError
from app.retrieval.base import Candidate

DOCUMENT = "document"
RELEVANCE = "relevance"
ORDERS = (DOCUMENT, RELEVANCE)


def order_sources(
    candidates: Sequence[Candidate], order: str = DOCUMENT
) -> List[Candidate]:
    """Return the candidates in presentation order.

    Deterministic in every branch — ties break on ``chunk_id`` — because
    a context block that differs between two runs of the same query makes
    an LLM's output impossible to reproduce or compare.
    """
    chosen = (order or DOCUMENT).strip().lower()
    if chosen not in ORDERS:
        raise ContextError(
            f"Unknown context order '{order}'. Available: {', '.join(ORDERS)}"
        )

    items = list(candidates)
    if not items:
        return []

    if chosen == RELEVANCE:
        items.sort(key=lambda c: (-c.final_score, c.chunk_id))
        return items

    # Document order: strongest document first, then reading order.
    best_in_document: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}
    for position, candidate in enumerate(items):
        document_id = candidate.chunk.document_id
        score = float(candidate.final_score)
        if score > best_in_document.get(document_id, float("-inf")):
            best_in_document[document_id] = score
        first_seen.setdefault(document_id, position)

    items.sort(
        key=lambda c: (
            -best_in_document[c.chunk.document_id],
            # Two documents whose best passage scored identically are
            # ordered by which was seen first, so the result does not
            # depend on dictionary iteration.
            first_seen[c.chunk.document_id],
            c.chunk.document_id,
            c.chunk.page_number,
            c.chunk.chunk_index,
            c.chunk_id,
        )
    )
    return items


def find_continuations(candidates: Sequence[Candidate]) -> Dict[str, int]:
    """``chunk_id -> 1-based position of the passage it continues from``.

    Two selected chunks that are neighbours in the same document are a
    single clause split by chunking. Saying so in the rendered header
    lets the model read them as continuous rather than as two separate
    statements of overlapping obligations — which is where a "the
    contract says X in one place and Y in another" answer comes from.

    Only *immediately* preceding positions count: a gap means real text
    is missing between them, and pretending otherwise would be worse
    than saying nothing.
    """
    out: Dict[str, int] = {}
    for position, candidate in enumerate(candidates):
        if position == 0:
            continue
        previous = candidates[position - 1]
        same_document = (
            previous.chunk.document_id == candidate.chunk.document_id
        )
        if same_document and (
            candidate.chunk.chunk_index == previous.chunk.chunk_index + 1
        ):
            out[candidate.chunk_id] = position  # 1-based number of `previous`
    return out


__all__ = ["order_sources", "find_continuations", "DOCUMENT", "RELEVANCE", "ORDERS"]
