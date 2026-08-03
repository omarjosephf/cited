"""Tests for reading and chunking.

The load-bearing property is the last one: no chunk may exceed the embedding
model's window. A chunk that does is worse than a missing chunk, because its
tail is stored, retrieved, shown to the model and cited — while being invisible
to the search that was supposed to find it. That failure is silent, so it has to
be caught by a test rather than noticed in use.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant.chunking import OVERLAP_WORDS, TARGET_WORDS, Chunk, chunk_passages
from assistant.documents import (
    Passage,
    UnsupportedDocument,
    read_corpus,
    read_document,
)


def write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_markdown_reader_tracks_the_nearest_heading(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "guide.md",
        "# Prompt Engineering\n\nA prompt has four components.\n\n"
        "## Context\n\nContext tells the model what it is working with.\n",
    )
    passages = read_document(path)

    assert [p.section for p in passages] == ["Prompt Engineering", "Context"]
    assert all(p.source == "guide.md" for p in passages)
    assert all(p.page is None for p in passages)  # Markdown has no pages


def test_blank_lines_separate_passages_but_headings_are_not_passages(
    tmp_path: Path,
) -> None:
    path = write(tmp_path, "a.md", "# Title\n\nOne.\n\nTwo.\n")
    texts = [p.text for p in read_document(path)]
    assert texts == ["One.", "Two."]


def test_citation_prefers_page_then_section() -> None:
    assert Passage("t", "a.pdf", page=4).cite() == "a.pdf, p.4"
    assert Passage("t", "a.md", section="Setup").cite() == "a.md — Setup"
    assert Passage("t", "a.md").cite() == "a.md"


def test_unsupported_format_names_what_is_supported(tmp_path: Path) -> None:
    path = write(tmp_path, "notes.rtf", "x")
    with pytest.raises(UnsupportedDocument, match=r"\.md"):
        read_document(path)


def test_corpus_skips_unsupported_files_but_reads_the_rest(tmp_path: Path) -> None:
    write(tmp_path, "keep.md", "# H\n\nReal content here.\n")
    write(tmp_path, "ignore.rtf", "not a supported format")
    write(tmp_path, "image.png", "binary-ish")

    passages = read_corpus(tmp_path)

    assert [p.source for p in passages] == ["keep.md"]


def test_chunks_never_span_two_documents() -> None:
    passages = [
        Passage("First document text.", "one.md"),
        Passage("Second document text.", "two.md"),
    ]
    chunks = chunk_passages(passages)

    assert len(chunks) == 2
    assert {c.source for c in chunks} == {"one.md", "two.md"}
    # A chunk spanning both could not be cited honestly.
    assert all("Second" not in c.text for c in chunks if c.source == "one.md")


def test_chunks_never_span_two_pages() -> None:
    passages = [
        Passage("Text from the first page.", "doc.pdf", page=1),
        Passage("Text from the second page.", "doc.pdf", page=2),
    ]
    chunks = chunk_passages(passages)

    assert [c.page for c in chunks] == [1, 2]


def test_chunks_never_span_two_sections() -> None:
    """The citation-accuracy guard.

    A chunk takes the section of its first passage. If it were allowed to run
    past a heading it would be cited under the section it started in while
    containing text from the next — a confidently, specifically wrong citation.
    A reader who follows it lands in the wrong part of the document, which is
    worse than no citation at all.
    """
    passages = [
        Passage("Text under the first heading.", "doc.md", section="First"),
        Passage("Text under the second heading.", "doc.md", section="Second"),
    ]
    chunks = chunk_passages(passages)

    assert [c.section for c in chunks] == ["First", "Second"]
    first = next(c for c in chunks if c.section == "First")
    assert "second heading" not in first.text


def test_every_section_in_a_document_survives_into_a_citation() -> None:
    """No heading may be swallowed by the chunk that preceded it."""
    sections = [f"Section {i}" for i in range(6)]
    passages = [
        Passage(f"Body text for section {i}. " * 3, "doc.md", section=name)
        for i, name in enumerate(sections)
    ]
    chunks = chunk_passages(passages)

    assert [c.section for c in chunks] == sections


def test_overlap_does_not_leak_across_a_section_boundary() -> None:
    """Size-driven splits overlap; provenance-driven splits must not.

    Carrying a tail across a heading would attribute one section's words to
    another section's citation — the same misattribution the boundary exists to
    prevent, reintroduced through the overlap mechanism.
    """
    passages = [
        Passage("alpha " * 60, "doc.md", section="First"),
        Passage("beta " * 60, "doc.md", section="Second"),
    ]
    chunks = chunk_passages(passages)

    second = next(c for c in chunks if c.section == "Second")
    assert "alpha" not in second.text


def test_neighbouring_chunks_overlap_so_a_split_answer_survives() -> None:
    # Enough passages to force more than one chunk.
    passages = [Passage(f"word{i} " * 30, "doc.md") for i in range(12)]
    chunks = chunk_passages(passages)

    assert len(chunks) > 1
    first_tail = set(chunks[0].text.split()[-OVERLAP_WORDS:])
    second_head = set(chunks[1].text.split()[:OVERLAP_WORDS])
    assert first_tail & second_head, "expected neighbouring chunks to share text"


def test_a_single_oversized_passage_is_split_rather_than_truncated() -> None:
    passages = [Passage("word " * (TARGET_WORDS * 3), "big.md")]
    chunks = chunk_passages(passages)

    assert len(chunks) > 1
    assert all(c.source == "big.md" for c in chunks)


def test_every_chunk_carries_a_usable_citation() -> None:
    passages = [
        Passage("Some text.", "doc.pdf", page=2),
        Passage("Other text.", "notes.md", section="Setup"),
    ]
    for chunk in chunk_passages(passages):
        assert chunk.cite()
        assert chunk.source


def test_chunk_indexes_are_sequential_and_stable() -> None:
    passages = [Passage(f"passage {i} " * 40, "doc.md") for i in range(10)]
    chunks = chunk_passages(passages)

    assert [c.index for c in chunks] == list(range(len(chunks)))


@pytest.mark.parametrize(
    "passages",
    [
        [Passage("word " * 500, "big.md")],
        [Passage(f"p{i} " * 60, "doc.md") for i in range(20)],
        [Passage(f"p{i} " * 5, "doc.md") for i in range(200)],
        [Passage(f"p{i} " * 45, "doc.pdf", page=i % 3 + 1) for i in range(30)],
    ],
)
def test_no_chunk_can_exceed_the_embedding_window(passages: list[Passage]) -> None:
    """The guard against silent truncation.

    `bge-small-en-v1.5` encodes 512 tokens and discards the rest without error.
    English runs about 1.3 tokens per word, so the ceiling below leaves real
    headroom. If this fails, retrieval is quietly broken for the tail of every
    oversized chunk — visible in citations, invisible to search.
    """
    ceiling = TARGET_WORDS * 2  # generous: overlap can push a chunk over target
    for chunk in chunk_passages(passages):
        words = len(chunk.text.split())
        assert words <= ceiling, f"chunk {chunk.index} has {words} words"
        assert words * 1.3 < 512, f"chunk {chunk.index} risks token truncation"


def test_citation_falls_back_to_the_bare_filename() -> None:
    """A `.txt` with no headings has neither page nor section to cite.

    The citation must still name the document rather than returning something
    empty — an answer sourced to nothing is indistinguishable from an invented
    one, which is the failure this whole project exists to avoid.
    """
    chunk = Chunk("some text", "plain.txt", page=None, section=None, index=0)
    assert chunk.cite() == "plain.txt"


def test_chunks_from_a_headingless_document_still_cite(tmp_path: Path) -> None:
    path = tmp_path / "plain.txt"
    path.write_text("First paragraph.\n\nSecond paragraph.\n", encoding="utf-8")

    for chunk in chunk_passages(read_document(path)):
        assert chunk.cite() == "plain.txt"


def test_empty_corpus_produces_no_chunks() -> None:
    assert chunk_passages([]) == []


def test_chunk_is_hashable_and_frozen() -> None:
    chunk = Chunk("t", "a.md", None, None, 0)
    assert hash(chunk)
    with pytest.raises(AttributeError):
        chunk.text = "changed"  # type: ignore[misc]
