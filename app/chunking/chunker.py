"""Legal-document-aware chunking.

Turns an :class:`ExtractedDocument` from Stage 2 into a list of
:class:`Chunk` objects for Stage 4 to embed.

The algorithm, in one paragraph: blocks are grouped into *segments* that
never span a section boundary; within a segment, blocks are packed
greedily into chunks until the token budget is reached; a block too big
to fit alone is split at the strongest available boundary (enumerated
item, then sentence, then words); each new chunk is prefixed with a
sentence-aligned tail of the previous one; and finally any chunk still
below the minimum size is merged into a neighbour from the same section.

Three properties are guaranteed and tested:

**Deterministic.** Same document and same settings produce byte-identical
chunks and identical chunk ids, every time. No randomness, no dict
ordering dependence, no wall-clock input to any decision.

**No text is lost.** Every block's content appears in at least one chunk.
Overlap duplicates text on purpose; nothing is dropped.

**No assumptions that break on unusual documents.** Sections are
optional, pages may be synthetic, a document may be a single block with
no punctuation, a "paragraph" may be 4000 tokens of one sentence, and
text may contain no whitespace at all. Each of those has an explicit
path rather than an implicit crash.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from app.chunking import splitters
from app.chunking.tokenizer import (
    TokenCounter,
    estimate_words_for_tokens,
    get_token_counter,
)
from app.config import Settings, get_settings
from app.logging_config import get_logger
from app.models.chunk import Chunk, ChunkingStats, make_chunk_id, summarize
from app.models.document import BlockKind, ExtractedDocument, TextBlock

logger = get_logger(__name__)


@dataclass
class _Piece:
    """A candidate unit of text with the provenance of its source block."""

    text: str
    tokens: int
    page_number: int
    section: Optional[str]
    block_id: str
    ocr: bool
    is_heading: bool = False
    #: True when this piece was produced by cutting inside a sentence.
    mid_sentence: bool = False


@dataclass
class _Buffer:
    """A chunk under construction."""

    pieces: List[_Piece] = field(default_factory=list)
    tokens: int = 0
    overlap_text: str = ""
    overlap_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.pieces

    @property
    def body(self) -> str:
        return "\n\n".join(p.text for p in self.pieces)

    def text(self) -> str:
        if self.overlap_text:
            return f"{self.overlap_text}\n\n{self.body}"
        return self.body

    def total_tokens(self) -> int:
        return self.tokens + self.overlap_tokens


class LegalChunker:
    """Boundary-aware chunker.

    Reusable across documents. Not safe to share across threads while
    reading ``_merged_small``, which reports on the last call only.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        counter: Optional[TokenCounter] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.counter = counter or get_token_counter(self.settings)

        self.chunk_size = max(32, int(self.settings.chunk_size))
        # Overlap can never reach the chunk size: at 100% every chunk
        # would be its predecessor and the walk would not advance.
        self.overlap = max(
            0, min(int(self.settings.chunk_overlap), self.chunk_size // 2)
        )
        self.min_tokens = max(1, int(self.settings.min_chunk_tokens))
        # Absolute ceiling. Merging and heading attachment may push a
        # chunk above the target, but never above this.
        self.max_tokens = max(
            self.chunk_size,
            int(self.chunk_size * float(self.settings.chunk_size_tolerance)),
        )
        # Overlap is carved *out of* the chunk budget, not added on top.
        # If a chunk could hold a full ``chunk_size`` of new text and
        # then have the overlap prefixed, every chunk after the first
        # would be ``chunk_size + overlap`` tokens — silently 50% over
        # budget at the maximum overlap setting, and straight past the
        # embedding model's window. So new content gets this much room
        # and the overlap fills the rest.
        self.body_budget = max(16, self.chunk_size - self.overlap)
        self.respect_sections = bool(self.settings.respect_sections)
        self.merge_small = bool(self.settings.merge_small_chunks)
        self.attach_headings = bool(self.settings.attach_headings)
        #: Number of small chunks folded away by the last call. An
        #: instance is reusable but not safe to share across threads
        #: while this is being read.
        self._merged_small = 0

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def chunk_document(self, document: ExtractedDocument) -> List[Chunk]:
        """Chunk a full extracted document."""
        started = time.perf_counter()
        blocks = [b for b in document.iter_blocks() if b.text and b.text.strip()]
        if not blocks:
            return []

        chunks: List[Chunk] = []
        for segment in self._segments(blocks):
            chunks.extend(self._chunk_segment(segment))

        if self.merge_small:
            chunks = self._merge_small(chunks)

        finalized = self._finalize(chunks, document)
        logger.debug(
            "Chunked %s into %d chunks in %d ms",
            document.metadata.filename,
            len(finalized),
            int((time.perf_counter() - started) * 1000),
        )
        return finalized

    def chunk_text(
        self,
        text: str,
        document_id: str,
        tenant_id: str,
        filename: str = "inline.txt",
        section: Optional[str] = None,
        page_number: int = 1,
    ) -> List[Chunk]:
        """Chunk a bare string. Used for tests and ad-hoc callers."""
        from app.models.document import (
            DocumentMetadata,
            ExtractedDocument as Doc,
            Page,
        )

        if not text or not text.strip():
            return []

        blocks = [
            TextBlock(
                text=paragraph,
                page_number=page_number,
                block_index=index,
                section=section,
            )
            for index, paragraph in enumerate(splitters.split_paragraphs(text))
        ]
        page = Page(page_number=page_number, blocks=blocks, text=text)
        document = Doc(
            metadata=DocumentMetadata(
                document_id=document_id, tenant_id=tenant_id, filename=filename
            ),
            pages=[page],
        )
        return self.chunk_document(document)

    # -----------------------------------------------------------------
    # Segmentation
    # -----------------------------------------------------------------

    def _segments(self, blocks: Sequence[TextBlock]) -> Iterable[List[TextBlock]]:
        """Group blocks into runs that a chunk may not span.

        With section detection on, a segment is a run of blocks sharing a
        section label — so a chunk never mixes two clauses. With it off,
        or on a document where no heading was detected, the whole
        document is one segment and packing falls back to paragraph
        boundaries. Both paths are normal; neither is an error.
        """
        if not self.respect_sections:
            yield list(blocks)
            return

        current: List[TextBlock] = []
        current_section: Optional[str] = None

        for block in blocks:
            if current and block.section != current_section:
                yield current
                current = []
            current_section = block.section
            current.append(block)

        if current:
            yield current

    # -----------------------------------------------------------------
    # Chunking one segment
    # -----------------------------------------------------------------

    def _chunk_segment(self, blocks: Sequence[TextBlock]) -> List[Chunk]:
        pieces = self._to_pieces(blocks)
        if not pieces:
            return []

        buffers: List[_Buffer] = []
        buffer = _Buffer()

        for piece in pieces:
            # A heading is not a chunk on its own: it is a label for the
            # text that follows. Attaching it keeps "4. TERMINATION"
            # with the obligation it introduces, which is exactly the
            # context an embedding needs.
            if (
                self.attach_headings
                and piece.is_heading
                and not buffer.is_empty
                and buffer.tokens + piece.tokens > self.body_budget
            ):
                buffers.append(buffer)
                buffer = self._new_buffer(buffers)

            if (
                not buffer.is_empty
                and buffer.tokens + piece.tokens > self.body_budget
            ):
                buffers.append(buffer)
                buffer = self._new_buffer(buffers)

            # Defence in depth: a piece that somehow still exceeds the
            # budget loses its overlap rather than the chunk exceeding
            # the embedding window.
            if (
                buffer.is_empty
                and buffer.overlap_tokens
                and buffer.overlap_tokens + piece.tokens > self.chunk_size
            ):
                buffer.overlap_text = ""
                buffer.overlap_tokens = 0

            buffer.pieces.append(piece)
            buffer.tokens += piece.tokens

        if not buffer.is_empty:
            buffers.append(buffer)

        return [self._buffer_to_chunk(b) for b in buffers]

    def _new_buffer(self, previous: List[_Buffer]) -> _Buffer:
        """Start a chunk carrying overlap from the one just closed."""
        buffer = _Buffer()
        if self.overlap and previous:
            tail = self._tail(previous[-1].text(), self.overlap)
            if tail:
                buffer.overlap_text = tail
                buffer.overlap_tokens = self.counter.count(tail)
        return buffer

    def _tail(self, text: str, budget: int) -> str:
        """The last ``budget`` tokens of ``text``, sentence-aligned.

        Whole trailing sentences are preferred so the overlap reads as
        language rather than a fragment starting mid-clause.

        The budget is a hard limit, never a suggestion. Taking "at least
        one sentence" regardless of size looks harmless until a clause
        is a single 400-token sentence: the overlap then becomes the
        whole previous chunk, and because each chunk's overlap is
        measured from the *previous chunk including its overlap*, the
        excess compounds with every chunk. When no whole sentence fits,
        this falls back to a word-level tail measured with the same
        counter rather than an assumed words-per-token ratio.
        """
        if budget <= 0 or not text.strip():
            return ""

        selected: List[str] = []
        total = 0
        for sentence in reversed(splitters.split_sentences(text)):
            cost = self.counter.count(sentence)
            if total + cost > budget:
                break
            selected.insert(0, sentence)
            total += cost
            if total >= budget:
                break
        if selected:
            return " ".join(selected).strip()

        # No whole sentence fits. Take trailing words, costed one by one.
        words = text.split()
        if not words:
            return ""
        tail: List[str] = []
        total = 0
        for word in reversed(words):
            cost = self.counter.count(word)
            if total + cost > budget and tail:
                break
            tail.insert(0, word)
            total += cost
            if total >= budget:
                break
        return " ".join(tail).strip()

    # -----------------------------------------------------------------
    # Blocks -> pieces (splitting oversized blocks)
    # -----------------------------------------------------------------

    def _to_pieces(self, blocks: Sequence[TextBlock]) -> List[_Piece]:
        pieces: List[_Piece] = []
        for block in blocks:
            block_id = block.block_id()
            for text, mid_sentence in self._split_block(block.text):
                if not text.strip():
                    continue
                pieces.append(
                    _Piece(
                        text=text,
                        tokens=self.counter.count(text),
                        page_number=block.page_number,
                        section=block.section,
                        block_id=block_id,
                        ocr=block.ocr,
                        is_heading=block.kind is BlockKind.HEADING,
                        mid_sentence=mid_sentence,
                    )
                )
        return pieces

    def _split_block(self, text: str) -> List[Tuple[str, bool]]:
        """Split one block down to pieces that fit the budget.

        Returns ``(text, was_cut_mid_sentence)`` pairs. The boundary
        hierarchy is applied in order, and each level is only reached
        when the level above produced something still too large.
        """
        text = (text or "").strip()
        if not text:
            return []
        if self.counter.count(text) <= self.body_budget:
            return [(text, False)]

        out: List[Tuple[str, bool]] = []

        # Level 1: paragraphs inside the block.
        paragraphs = splitters.split_paragraphs(text)
        if len(paragraphs) > 1:
            for paragraph in paragraphs:
                out.extend(self._split_block(paragraph))
            return out

        # Level 2: enumerated items - the smaller self-contained unit in
        # a contract, so preferred over sentences.
        if splitters.looks_enumerated(text):
            items = splitters.split_enumerated_items(text)
            if len(items) > 1:
                for item in items:
                    out.extend(self._split_paragraph(item))
                return out

        return self._split_paragraph(text)

    def _split_paragraph(self, text: str) -> List[Tuple[str, bool]]:
        """Sentence-level split, then the word/character fallbacks."""
        text = text.strip()
        if not text:
            return []
        if self.counter.count(text) <= self.body_budget:
            return [(text, False)]

        out: List[Tuple[str, bool]] = []
        sentences = splitters.split_sentences(text)

        if len(sentences) > 1:
            buffer: List[str] = []
            tokens = 0
            for sentence in sentences:
                cost = self.counter.count(sentence)
                if cost > self.body_budget:
                    # An oversized sentence: flush, then fall through.
                    if buffer:
                        out.append((" ".join(buffer), False))
                        buffer, tokens = [], 0
                    out.extend(self._split_oversized_sentence(sentence))
                    continue
                if tokens + cost > self.body_budget and buffer:
                    out.append((" ".join(buffer), False))
                    buffer, tokens = [], 0
                buffer.append(sentence)
                tokens += cost
            if buffer:
                out.append((" ".join(buffer), False))
            return out

        return self._split_oversized_sentence(text)

    def _split_oversized_sentence(self, text: str) -> List[Tuple[str, bool]]:
        """Last resort: a single sentence longer than the whole budget.

        Common in legal drafting — a 400-word recital with no full stop.
        These pieces are flagged ``mid_sentence`` so the statistics make
        the compromise visible.

        Words are accumulated against *measured* token costs rather than
        an assumed words-per-token ratio. A fixed ratio is wrong by a
        wide margin on dense text — identifiers, citation strings and
        tabular fragments can each cost several tokens per "word" — and
        produces pieces far over budget, which then overflow the
        embedding window. Measuring costs one counter call per word and
        is exact.
        """
        if not splitters.has_whitespace(text):
            return [(piece, True) for piece in self._split_by_characters(text)]

        words = text.split()
        if not words:
            return []
        costs = [self.counter.count(word) for word in words]

        pieces: List[str] = []
        start = 0
        total_words = len(words)

        while start < total_words:
            end = start
            budget = 0
            while end < total_words and budget + costs[end] <= self.body_budget:
                budget += costs[end]
                end += 1

            if end == start:
                # A single "word" that alone exceeds the budget — a long
                # identifier or an unbroken run. Drop to characters.
                pieces.extend(self._split_by_characters(words[start]))
                start += 1
                continue

            pieces.append(" ".join(words[start:end]))
            if end >= total_words:
                break

            # Step back by the overlap, while guaranteeing forward
            # progress: the cursor never returns to ``start``.
            cursor = end
            carried = 0
            while cursor > start + 1 and carried + costs[cursor - 1] <= self.overlap:
                cursor -= 1
                carried += costs[cursor]
            start = cursor

        return [(piece, True) for piece in pieces]

    def _split_by_characters(self, text: str) -> List[str]:
        """Character windows for text with no usable whitespace.

        The window is sized from the text's own measured character-per-
        token density, then shrunk until a window actually fits the
        budget — a fixed guess is wrong for scripts that pack several
        tokens into every character.
        """
        stripped = (text or "").strip()
        if not stripped:
            return []

        sample = stripped[:512]
        sample_tokens = max(1, self.counter.count(sample))
        chars_per_token = max(1.0, len(sample) / sample_tokens)

        window = max(1, int(self.body_budget * chars_per_token))
        guard = 0
        while (
            window > 1
            and guard < 24
            and self.counter.count(stripped[:window]) > self.body_budget
        ):
            window = max(1, int(window * 0.8))
            guard += 1

        step = max(1, window - int(self.overlap * chars_per_token))
        return splitters.split_characters(stripped, window, step)

    # -----------------------------------------------------------------
    # Buffer -> chunk
    # -----------------------------------------------------------------

    def _buffer_to_chunk(self, buffer: _Buffer) -> Chunk:
        pages = [p.page_number for p in buffer.pieces]
        sections = [p.section for p in buffer.pieces if p.section]
        text = buffer.text()

        return Chunk(
            chunk_id="",  # assigned in _finalize, once the index is known
            chunk_index=0,
            text=text,
            document_id="",
            tenant_id="",
            filename="",
            page_number=min(pages) if pages else 1,
            page_end=max(pages) if pages else 1,
            # The first section in the buffer labels the chunk: with
            # respect_sections on there is only one, and with it off the
            # opening section is the honest label for where it starts.
            section=sections[0] if sections else None,
            block_ids=list(dict.fromkeys(p.block_id for p in buffer.pieces)),
            token_count=self.counter.count(text),
            overlap_tokens=buffer.overlap_tokens,
            ocr=any(p.ocr for p in buffer.pieces),
            split_mid_sentence=any(p.mid_sentence for p in buffer.pieces),
            heading_only=all(p.is_heading for p in buffer.pieces),
        )

    # -----------------------------------------------------------------
    # Small-chunk merging
    # -----------------------------------------------------------------

    def _merge_small(self, chunks: List[Chunk]) -> List[Chunk]:
        """Fold undersized chunks into a neighbour.

        A 12-token chunk ("8. NOTICES") embeds to a vector that matches
        almost any query about notices while carrying no information —
        it pollutes retrieval. Two rules, with different strictness:

        * **Body text** merges only within the same section, so a merge
          can never join two different clauses. A short chunk that is a
          complete clause on its own is legitimate and is kept.
        * **Heading-only** chunks merge *forward across* sections,
          because a heading is a label for what follows and is never
          content in its own right. A document title or a heading that
          happened to open its own segment would otherwise survive as an
          orphan chunk.

        Neither rule may push a chunk past ``max_tokens``.
        """
        self._merged_small = 0
        if not chunks:
            return chunks

        # Pass 1: fold heading-only chunks forward into the next chunk.
        forward: List[Chunk] = []
        pending: List[Chunk] = []
        for chunk in chunks:
            if chunk.heading_only and chunk.token_count < self.min_tokens:
                pending.append(chunk)
                continue
            if pending:
                combined = self._prefix_tokens(pending) + chunk.token_count
                if combined <= self.max_tokens:
                    self._prepend(chunk, pending)
                    self._merged_small += len(pending)
                else:
                    forward.extend(pending)
                pending = []
            forward.append(chunk)
        # Headings with nothing after them stay as they are: dropping
        # them would lose text, and this module never loses text.
        forward.extend(pending)

        # Pass 2: fold small body chunks backward within their section.
        merged: List[Chunk] = []
        for chunk in forward:
            if (
                merged
                and chunk.token_count < self.min_tokens
                and merged[-1].section == chunk.section
                and merged[-1].token_count + chunk.token_count <= self.max_tokens
            ):
                self._absorb(merged[-1], chunk)
                self._merged_small += 1
                continue
            merged.append(chunk)

        # A leading small chunk has no predecessor; fold it forward.
        if (
            len(merged) > 1
            and merged[0].token_count < self.min_tokens
            and merged[0].section == merged[1].section
            and merged[0].token_count + merged[1].token_count <= self.max_tokens
        ):
            self._prepend(merged[1], [merged[0]])
            self._merged_small += 1
            merged = merged[1:]

        return merged

    def _prefix_tokens(self, chunks: Sequence[Chunk]) -> int:
        return sum(c.token_count for c in chunks)

    def _prepend(self, target: Chunk, heads: Sequence[Chunk]) -> None:
        """Put ``heads`` in front of ``target``, in place."""
        joined = "\n\n".join(c.text for c in heads)
        target.text = f"{joined}\n\n{target.text}".strip()
        target.token_count = self.counter.count(target.text)
        target.char_count = len(target.text)
        target.page_number = min(
            [target.page_number] + [c.page_number for c in heads]
        )
        target.block_ids = list(
            dict.fromkeys(
                [b for c in heads for b in c.block_ids] + target.block_ids
            )
        )
        target.ocr = target.ocr or any(c.ocr for c in heads)
        # A heading supplies the section label when the body had none.
        if target.section is None:
            for head in heads:
                if head.section:
                    target.section = head.section
                    break
        target.heading_only = False

    def _absorb(self, target: Chunk, extra: Chunk) -> None:
        """Append ``extra`` onto ``target`` in place."""
        # Drop the absorbed chunk's overlap: it is a copy of text that
        # is already immediately above it in the merged result.
        body = extra.text
        if extra.overlap_tokens and target.text.strip():
            body = self._strip_leading_overlap(body, target.text)

        target.text = f"{target.text}\n\n{body}".strip()
        target.token_count = self.counter.count(target.text)
        target.char_count = len(target.text)
        target.page_end = max(target.page_end or 0, extra.page_end or 0)
        target.block_ids = list(dict.fromkeys(target.block_ids + extra.block_ids))
        target.ocr = target.ocr or extra.ocr
        target.split_mid_sentence = (
            target.split_mid_sentence or extra.split_mid_sentence
        )

    @staticmethod
    def _strip_leading_overlap(body: str, previous: str) -> str:
        """Remove a duplicated overlap prefix when merging."""
        head, separator, rest = body.partition("\n\n")
        if separator and head and head in previous:
            return rest.strip() or body
        return body

    # -----------------------------------------------------------------
    # Finalisation
    # -----------------------------------------------------------------

    def _finalize(
        self, chunks: List[Chunk], document: ExtractedDocument
    ) -> List[Chunk]:
        """Stamp document metadata and deterministic ids.

        Ids are assigned last, from the final index and final text, so
        that merging cannot leave an id describing text that changed.
        """
        meta = document.metadata
        out: List[Chunk] = []
        index = 0

        for chunk in chunks:
            if not chunk.text.strip():
                continue
            chunk.chunk_index = index
            chunk.document_id = meta.document_id
            chunk.tenant_id = meta.tenant_id
            chunk.filename = meta.filename
            chunk.document_type = meta.document_type
            chunk.extra = dict(meta.extra or {})
            chunk.char_count = len(chunk.text)
            chunk.chunk_id = make_chunk_id(meta.document_id, index, chunk.text)
            out.append(chunk)
            index += 1

        return out


# ---------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------


def chunk_document(
    document: ExtractedDocument,
    settings: Optional[Settings] = None,
    counter: Optional[TokenCounter] = None,
) -> List[Chunk]:
    """Chunk an extracted document. The entry point Stage 4 will call."""
    return LegalChunker(settings, counter).chunk_document(document)


def chunk_with_stats(
    document: ExtractedDocument,
    settings: Optional[Settings] = None,
    counter: Optional[TokenCounter] = None,
) -> Tuple[List[Chunk], ChunkingStats]:
    """Chunk and report what happened."""
    settings = settings or get_settings()
    chunker = LegalChunker(settings, counter)
    started = time.perf_counter()
    chunks = chunker.chunk_document(document)
    stats = summarize(
        chunks,
        merged_small_count=chunker._merged_small,
        tokenizer=chunker.counter.name,
        chunk_size=chunker.chunk_size,
        chunk_overlap=chunker.overlap,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    return chunks, stats


__all__ = ["LegalChunker", "chunk_document", "chunk_with_stats"]
