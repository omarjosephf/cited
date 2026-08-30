"""Tests for precomputed corpus vectors.

Every test here is about one question: can a stored matrix be trusted to still
describe the chunks it is being loaded for? A matrix that is merely *plausible*
— right shape, right dtype, wrong text — produces answers that cite the wrong
passages while looking entirely normal, so each check below is asserted to
*raise* rather than to cope.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from assistant.chunking import Chunk
from assistant.vectors import VectorsMismatch, chunk_digest, load, save

MODEL = "test-model"
DIMENSIONS = 4


def chunk(text: str, index: int, section: str | None = "Section") -> Chunk:
    return Chunk(text=text, source="doc.md", page=None, section=section, index=index)


@pytest.fixture
def chunks() -> list[Chunk]:
    return [chunk("first body", 0, "First"), chunk("second body", 1, "Second")]


@pytest.fixture
def matrix() -> np.ndarray:
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float32,
    )


@pytest.fixture
def stored(tmp_path: Path, chunks: list[Chunk], matrix: np.ndarray) -> Path:
    path = tmp_path / "vectors.npz"
    save(path, chunks, matrix, model=MODEL, corpus_checksum="abc123")
    return path


class TestDigest:
    def test_the_digest_covers_the_heading_not_only_the_body(self) -> None:
        """`indexed_text` is what gets embedded, so it is what must be hashed.

        Hashing `text` alone would let a heading change — which changes every
        vector — pass as unchanged.
        """
        before = chunk_digest([chunk("body", 0, "First")])
        after = chunk_digest([chunk("body", 0, "Renamed")])
        assert before != after

    def test_reordering_the_same_chunks_changes_the_digest(self) -> None:
        """Order is the whole meaning of the matrix: row i is chunk i."""
        a, b = chunk("alpha", 0), chunk("beta", 1)
        assert chunk_digest([a, b]) != chunk_digest([b, a])

    def test_identical_text_gives_an_identical_digest(self) -> None:
        assert chunk_digest([chunk("same", 0)]) == chunk_digest([chunk("same", 0)])


class TestSave:
    def test_a_round_trip_returns_the_same_vectors(
        self, stored: Path, chunks: list[Chunk], matrix: np.ndarray
    ) -> None:
        loaded = load(stored, chunks, model=MODEL, dimensions=DIMENSIONS)
        assert np.array_equal(loaded, matrix)
        assert loaded.dtype == np.float32

    def test_saving_a_mismatched_matrix_is_refused(
        self, tmp_path: Path, chunks: list[Chunk], matrix: np.ndarray
    ) -> None:
        """Caught at the point it is created, not only at the point it is read."""
        with pytest.raises(ValueError, match="refusing to save"):
            save(tmp_path / "v.npz", chunks, matrix[:1], model=MODEL)


class TestLoadRefuses:
    """Each of these is a way the matrix stops describing the corpus."""

    def test_a_missing_file(self, tmp_path: Path, chunks: list[Chunk]) -> None:
        with pytest.raises(VectorsMismatch, match="does not exist"):
            load(tmp_path / "absent.npz", chunks, model=MODEL, dimensions=DIMENSIONS)

    def test_a_file_that_is_not_a_vectors_file(
        self, tmp_path: Path, chunks: list[Chunk]
    ) -> None:
        path = tmp_path / "other.npz"
        np.savez_compressed(path, something_else=np.zeros(3))
        with pytest.raises(VectorsMismatch, match="not a corpus vectors file"):
            load(path, chunks, model=MODEL, dimensions=DIMENSIONS)

    def test_vectors_from_a_different_model(
        self, stored: Path, chunks: list[Chunk]
    ) -> None:
        """Two models put text in two unrelated geometries.

        Nothing about the file's shape says so, and the scores that come out are
        numbers in the right range — just meaningless ones.
        """
        with pytest.raises(VectorsMismatch, match="model"):
            load(stored, chunks, model="a-different-model", dimensions=DIMENSIONS)

    def test_a_width_the_query_cannot_be_compared_against(
        self, stored: Path, chunks: list[Chunk]
    ) -> None:
        with pytest.raises(VectorsMismatch, match="expected width"):
            load(stored, chunks, model=MODEL, dimensions=DIMENSIONS + 1)

    def test_more_chunks_than_rows(self, stored: Path, chunks: list[Chunk]) -> None:
        with pytest.raises(VectorsMismatch, match="vectors for"):
            load(
                stored,
                [*chunks, chunk("third body", 2, "Third")],
                model=MODEL,
                dimensions=DIMENSIONS,
            )

    def test_the_same_number_of_chunks_with_different_text(
        self, stored: Path, chunks: list[Chunk]
    ) -> None:
        """The case a row count cannot see, and the reason the digest exists.

        An edited corpus of unchanged length loads cleanly under every shape
        check and misattributes every citation it touches.
        """
        edited = [chunks[0], chunk("second body, rewritten", 1, "Second")]
        with pytest.raises(VectorsMismatch, match="different text"):
            load(stored, edited, model=MODEL, dimensions=DIMENSIONS)

    def test_an_unreadable_header(self, tmp_path: Path, chunks: list[Chunk]) -> None:
        path = tmp_path / "corrupt.npz"
        np.savez_compressed(
            path,
            vectors=np.zeros((2, DIMENSIONS), dtype=np.float32),
            metadata=np.array("{not json"),
        )
        with pytest.raises(VectorsMismatch, match="unreadable metadata"):
            load(path, chunks, model=MODEL, dimensions=DIMENSIONS)
