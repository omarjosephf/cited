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


def _provenance_differs(left: Passage, right: Passage) -> bool:
    """Whether two passages cannot honestly share one citation.

    Document and page are the obvious cases. **Section is included for the same
    reason**, and it is the subtle one: a chunk takes the section of its first
    passage, so a chunk allowed to run past a heading gets cited under the
    heading it started in while containing text from the next. The citation is
    then confidently, specifically wrong — which is worse than no citation at
    all, because a reader who follows it finds the wrong part of the document
    and concludes the tool is untrustworthy.

    The cost is that a short section becomes a short chunk. That is an
    acceptable trade: a short section is still a complete idea and retrieves
    perfectly well, whereas a misattributed citation defeats the point of the
    project.
    """
    return (
        left.source != right.source
        or left.page != right.page
        or left.section != right.section
    )


def chunk_passages(passages: list[Passage]) -> list[Chunk]:
    """Merge passages into chunks that can each be cited by a single reference.

    Provenance constrains chunking, never the other way round: a chunk is only
    allowed to grow while every passage in it shares one citation.
    """
    chunks: list[Chunk] = []
    buffer: list[Passage] = []
    buffer_words = 0

    def emit() -> None:
        """Turn the buffer into a chunk. Does not carry any overlap forward."""
        nonlocal buffer_words
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

    def emit_with_overlap() -> None:
        """Emit, then seed the next chunk with this one's tail.

        Overlap only happens when the split was driven by *size*. A split driven
        by provenance must not carry text across, or the overlap would be
        attributed to the wrong document, page or section.
        """
        nonlocal buffer_words
        tail: list[Passage] = []
        tail_words = 0
        for passage in reversed(buffer):
            words = len(passage.text.split())
            if tail_words + words > OVERLAP_WORDS:
                break
            tail.insert(0, passage)
            tail_words += words

        emit()
        buffer.extend(tail)
        buffer_words = tail_words

    for original in passages:
        for passage in _split_oversized(original):
            if buffer and _provenance_differs(passage, buffer[-1]):
                emit()

            words = len(passage.text.split())
            if buffer_words + words > TARGET_WORDS and buffer_words >= MIN_WORDS:
                emit_with_overlap()

            buffer.append(passage)
            buffer_words += words

    if buffer:
        emit()

    return chunks
