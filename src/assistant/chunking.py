"""Merging passages into retrieval-sized chunks.

Chunking is the decision that quietly determines retrieval quality. Too small and
a chunk lacks the context to answer anything; too large and the embedding becomes
an average of several topics, matching everything weakly and nothing strongly.

Two constraints drive the numbers here:

* **The embedding model truncates.** `bge-small-en-v1.5` encodes at most 512
  tokens and silently discards the rest. A chunk longer than that would have a
  tail that is stored, shown to the model, and cited — while being invisible to
  the search that was supposed to find it. `TARGET_WORDS` is set well below the
  limit so this cannot happen.
* **Answers straddle boundaries.** A definition split across two chunks may be
  in neither one's embedding strongly enough to retrieve. `OVERLAP_WORDS` repeats
  the tail of each chunk at the head of the next so a straddling answer survives
  in at least one whole chunk.

Words are used rather than tokens deliberately: it avoids a tokeniser dependency
purely for splitting, and the ratio is stable enough (~1.3 tokens per English
word) that a conservative word budget keeps us safely inside the token limit.
These values are a starting point to be tuned against the evaluation set, not
tuned by intuition.
"""

from __future__ import annotations

from dataclasses import dataclass

from assistant.documents import Passage

TARGET_WORDS = 180
"""Roughly 230 tokens — comfortably inside the model's 512-token window."""

OVERLAP_WORDS = 40
"""Repeated between neighbours so an answer split across a boundary survives."""

MIN_WORDS = 25
"""Below this a chunk is too thin to answer anything; it is merged forward."""


@dataclass(frozen=True)
class Chunk:
    """A retrieval-sized unit of text that still knows where it came from."""

    text: str
    source: str
    page: int | None
    section: str | None
    index: int

    def cite(self) -> str:
        if self.page is not None:
            return f"{self.source}, p.{self.page}"
        if self.section:
            return f"{self.source} — {self.section}"
        return self.source


def _split_oversized(passage: Passage) -> list[Passage]:
    """Break a single passage that already exceeds the target on its own.

    A long unbroken paragraph cannot be merged with anything — it has to be cut.
    Overlap is applied here too, for the same reason it is applied between
    chunks: the cut may land mid-definition.
    """
    words = passage.text.split()
    if len(words) <= TARGET_WORDS:
        return [passage]

    pieces: list[Passage] = []
    start = 0
    step = TARGET_WORDS - OVERLAP_WORDS
    while start < len(words):
        window = words[start : start + TARGET_WORDS]
        pieces.append(
            Passage(
                text=" ".join(window),
                source=passage.source,
                page=passage.page,
                section=passage.section,
            )
        )
        if start + TARGET_WORDS >= len(words):
            break
        start += step
    return pieces


def chunk_passages(passages: list[Passage]) -> list[Chunk]:
    """Merge passages into chunks, never merging across documents or pages.

    A chunk that spanned two documents could not be cited honestly, and one that
    spanned two pages would have to claim a single page number. Both boundaries
    are therefore hard: provenance constrains chunking, not the other way round.
    """
    chunks: list[Chunk] = []
    buffer: list[Passage] = []
    buffer_words = 0

    def flush() -> None:
        nonlocal buffer_words
        if not buffer:
            return
        head = buffer[0]
        chunks.append(
            Chunk(
                text=" ".join(p.text for p in buffer),
                source=head.source,
                page=head.page,
                section=head.section,
                index=len(chunks),
            )
        )
        # Carry the tail forward so the next chunk overlaps this one.
        tail_words: list[str] = []
        carried: list[Passage] = []
        for passage in reversed(buffer):
            words = passage.text.split()
            if len(tail_words) + len(words) > OVERLAP_WORDS:
                break
            tail_words = words + tail_words
            carried.insert(0, passage)
        buffer.clear()
        buffer.extend(carried)
        buffer_words = len(tail_words)

    for original in passages:
        for passage in _split_oversized(original):
            boundary = buffer and (
                passage.source != buffer[-1].source or passage.page != buffer[-1].page
            )
            if boundary:
                # Flush without overlap: the carried tail belongs to a different
                # document or page and must not leak into the next chunk.
                head = buffer[0]
                chunks.append(
                    Chunk(
                        text=" ".join(p.text for p in buffer),
                        source=head.source,
                        page=head.page,
                        section=head.section,
                        index=len(chunks),
                    )
                )
                buffer.clear()
                buffer_words = 0

            words = len(passage.text.split())
            if buffer_words + words > TARGET_WORDS and buffer_words >= MIN_WORDS:
                flush()
            buffer.append(passage)
            buffer_words += words

    if buffer:
        head = buffer[0]
        chunks.append(
            Chunk(
                text=" ".join(p.text for p in buffer),
                source=head.source,
                page=head.page,
                section=head.section,
                index=len(chunks),
            )
        )

    return chunks
