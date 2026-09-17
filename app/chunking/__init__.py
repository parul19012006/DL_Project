"""Legal-document-aware chunking (Stage 3).

Consumes the ``ExtractedDocument`` produced by Stage 2 and produces the
``Chunk`` objects Stage 4 will embed.
"""

from app.chunking.chunker import LegalChunker, chunk_document, chunk_with_stats
from app.chunking.tokenizer import count_tokens, get_token_counter

__all__ = [
    "LegalChunker",
    "chunk_document",
    "chunk_with_stats",
    "count_tokens",
    "get_token_counter",
]
