"""The inspector must describe production ingestion without changing it."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from assistant.chunking import chunk_passages
from assistant.documents import read_corpus
from assistant.embedding import EMBEDDING_DIMENSIONS, MODEL_NAME
from assistant.inspection import CorpusProfile, inspect_corpus
from assistant.vectors import save as save_vectors


def make_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "content"
    nested = corpus / "policies"
    nested.mkdir(parents=True)
    (corpus / "guide.md").write_text(
        "# Retrieval Settings\n\n" + "A documented retrieval sentence. " * 35,
        encoding="utf-8",
    )
    (nested / "limits.txt").write_text(
        "# Boundaries\n\n" + "A boundary statement. " * 20,
        encoding="utf-8",
    )
    return corpus


def test_inspection_uses_the_production_readers_and_chunker(tmp_path: Path) -> None:
    directory = make_corpus(tmp_path)
    expected_passages = read_corpus(directory)
    expected_chunks = chunk_passages(expected_passages)

    result = inspect_corpus(CorpusProfile.create("OJ Assistant", directory))

    assert result.id == "oj-assistant"
    assert result.passage_count == len(expected_passages)
    assert len(result.chunks) == len(expected_chunks)
    assert [item.source for item in result.documents] == [
        "guide.md",
        "policies/limits.txt",
    ]
    assert all(len(item.sha256) == 64 for item in result.documents)
    assert all(item.estimated_embedding_tokens > 0 for item in result.documents)
    assert result.vectors.state == "not_supplied"


def test_estimated_tokens_cover_the_exact_indexed_string(tmp_path: Path) -> None:
    result = inspect_corpus(CorpusProfile.create("Cited", make_corpus(tmp_path)))
    chunk = result.chunks[0]

    assert chunk.indexed_text.startswith("Retrieval Settings.")
    assert chunk.indexed_words == len(chunk.indexed_text.split())
    assert chunk.estimated_embedding_tokens >= chunk.indexed_words
    assert result.summary()["chunking"]["estimated_target_tokens"] == 234
    assert result.summary()["chunking"]["estimated_overlap_tokens"] == 52
    assert (
        "not an exact tokenizer count"
        in result.summary()["embedding"]["token_measurement"]
    )


def test_serialised_results_never_disclose_the_local_corpus_path(
    tmp_path: Path,
) -> None:
    directory = make_corpus(tmp_path)
    result = inspect_corpus(CorpusProfile.create("Cited", directory))

    exposed = repr(result.detail()) + repr(result.chunks[0].as_dict(include_text=True))
    assert str(tmp_path) not in exposed


def test_matching_and_stale_vector_artifacts_are_reported(tmp_path: Path) -> None:
    directory = make_corpus(tmp_path)
    chunks = chunk_passages(read_corpus(directory))
    vectors = tmp_path / "vectors.npz"
    save_vectors(
        vectors,
        chunks,
        np.zeros((len(chunks), EMBEDDING_DIMENSIONS), dtype=np.float32),
        model=MODEL_NAME,
    )

    valid = inspect_corpus(CorpusProfile.create("Cited", directory, vectors))
    assert valid.vectors.state == "valid"

    (directory / "guide.md").write_text(
        "# Changed\n\nDifferent content now.", encoding="utf-8"
    )
    stale = inspect_corpus(CorpusProfile.create("Cited", directory, vectors))
    assert stale.vectors.state == "invalid"
