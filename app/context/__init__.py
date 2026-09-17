"""Stage 6 — RAG context construction.

    retrieved passages -> select -> fit to budget -> order -> render
                       -> a labelled, citable context block

Input is Stage 5's ranked passages (or a plain list of chunks — nothing
here requires retrieval to have run). Output is the block of text that
will sit in an LLM prompt, *plus* the structured record of what it
contains and what it leaves out.

``selection``  which passages are worth the prompt, and redundancy removal
``budget``     what fits, what is shortened, what is dropped
``ordering``   document order for reading, relevance order for selection
``formatter``  the uniform ``Source N:`` blocks
``builder``    the orchestration, and nothing else

**No answer is generated here, and no instructions are written into the
context.** Prompt assembly and the LLM call belong to the next stage.
"""

from app.context.base import (
    UNKNOWN,
    UNKNOWN_PAGE,
    BuiltContext,
    ContextError,
    ContextSource,
    ContextStats,
    OmissionReason,
    OmittedSource,
)
from app.context.budget import (
    fit_to_budget,
    strip_repeated_heading,
    truncate_at_sentence,
)
from app.context.builder import (
    ContextBuilder,
    build_context,
    get_context_builder,
    set_context_builder,
)
from app.context.formatter import SEPARATOR, render, render_source
from app.context.ordering import DOCUMENT, ORDERS, RELEVANCE, order_sources
from app.context.selection import remove_redundant, select_evidence

__all__ = [
    "BuiltContext",
    "ContextSource",
    "ContextStats",
    "OmittedSource",
    "OmissionReason",
    "ContextError",
    "UNKNOWN",
    "UNKNOWN_PAGE",
    "ContextBuilder",
    "build_context",
    "get_context_builder",
    "set_context_builder",
    "select_evidence",
    "remove_redundant",
    "fit_to_budget",
    "truncate_at_sentence",
    "strip_repeated_heading",
    "order_sources",
    "render",
    "render_source",
    "SEPARATOR",
    "DOCUMENT",
    "RELEVANCE",
    "ORDERS",
]
