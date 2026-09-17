"""Rendering — turning selected passages into the block the LLM reads.

The structure is fixed and uniform:

    Source 1:
    Document: vendor_contract_2024.pdf (id: doc-msa-001)
    Page: 2
    Section: 4. TERMINATION
    Chunk: 3 (id: 9f2c34a1…)
    Text:
    Either party may terminate this Agreement for convenience on …

    ---

    Source 2:
    …

Four properties of that layout are doing real work:

**Every source is labelled and numbered.** A model asked to cite its
evidence needs something to cite *by*. "Source 2" is resolvable
mechanically — :meth:`BuiltContext.source_by_number` turns it straight
back into a chunk_id, a document and a page — where "the termination
clause" is resolvable only by guessing.

**Every source carries the same fields in the same order**, including
when a field is unknown, which is rendered as an explicit marker rather
than left blank. Uniform blocks are easy for a model to parse and hard
to accidentally merge; a block missing its Section line looks like a
continuation of the block above it.

**Sources are separated by an unambiguous delimiter.** ``---`` on its own
line between blank lines does not occur inside legal prose, so there is
no sequence of clause text that can be mistaken for a source boundary.
Passages run together are how a model ends up attributing one contract's
liability cap to another contract.

**Metadata comes before the text, not after.** The model reads the
provenance while it is reading the passage, rather than having to hold
the passage in mind and attach a citation retrospectively.

Notes — truncation, OCR provenance, continuation, duplicate citations —
are rendered as their own labelled lines only when they apply, and each
states a fact rather than an instruction. Instructions belong in the
prompt template, which is the generation stage's business, not this one's.
"""

from __future__ import annotations

from typing import List, Sequence

from app.context.base import UNKNOWN, UNKNOWN_PAGE, ContextSource

#: Between sources. Blank line, rule, blank line.
SEPARATOR = "\n\n---\n\n"

#: Rendered in place of text removed by truncation, so the model knows
#: it is looking at part of a clause rather than all of one.
TRUNCATION_MARKER = "[… passage shortened; {tokens} tokens omitted …]"

#: Characters of the chunk id shown in the header. The full id is always
#: in the structured output; this is for a human reading the prompt.
ID_PREVIEW = 12


def _page_line(source: ContextSource) -> str:
    chunk = source.chunk
    page = chunk.page_number
    if not page or page < 1:
        return UNKNOWN_PAGE
    end = chunk.page_end or page
    return str(page) if end == page else f"{page}-{end}"


def _note(source: ContextSource) -> str:
    """Facts about this passage the model should know while reading it."""
    notes: List[str] = []
    if source.truncated:
        notes.append(
            f"This passage was shortened to fit the context budget; "
            f"{source.omitted_tokens} tokens were removed from the end, so "
            "it may not contain the whole clause."
        )
    if source.chunk.ocr:
        notes.append(
            "Text recovered by OCR from a scanned page; wording may contain "
            "recognition errors."
        )
    if source.chunk.split_mid_sentence:
        notes.append("This passage begins or ends mid-sentence.")
    return " ".join(notes)


def render_source(source: ContextSource) -> str:
    """One numbered block, header and text."""
    chunk = source.chunk
    lines = [
        f"Source {source.number}:",
        f"Document: {chunk.filename or UNKNOWN} (id: {chunk.document_id or UNKNOWN})",
        f"Page: {_page_line(source)}",
        f"Section: {chunk.section or UNKNOWN}",
        f"Chunk: {chunk.chunk_index} (id: {(chunk.chunk_id or '')[:ID_PREVIEW]})",
    ]

    if source.continues_source:
        lines.append(f"Continues: Source {source.continues_source}")

    if source.duplicate_citations:
        also = "; ".join(
            f"{c.get('filename') or UNKNOWN} p.{c.get('page') or UNKNOWN_PAGE}"
            for c in source.duplicate_citations
        )
        lines.append(f"Also appears in: {also}")

    note = _note(source)
    if note:
        lines.append(f"Note: {note}")

    body = source.text
    if source.truncated:
        body = f"{body}\n{TRUNCATION_MARKER.format(tokens=source.omitted_tokens)}"

    lines.append("Text:")
    lines.append(body)
    return "\n".join(lines)


def render_header(source: ContextSource) -> str:
    """The block with an empty body — used to price the overhead.

    The budget has to account for the header, not just the passage. A
    builder that counts only the text overflows by ~35 tokens per source,
    which on eight sources is most of a clause.
    """
    placeholder = ContextSource(
        chunk=source.chunk,
        number=source.number,
        text="",
        truncated=source.truncated,
        original_tokens=source.original_tokens,
        text_tokens=0,
        continues_source=source.continues_source,
        duplicate_citations=list(source.duplicate_citations),
    )
    return render_source(placeholder)


def render(sources: Sequence[ContextSource]) -> str:
    """The whole context block."""
    return SEPARATOR.join(render_source(s) for s in sources)


__all__ = ["render", "render_source", "render_header", "SEPARATOR", "TRUNCATION_MARKER"]
