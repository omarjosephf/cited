"""Tests for embedding and retrieval.

These run against a deterministic bag-of-words embedder rather than the real
model. That is not a shortcut: a unit test that downloads and runs a transformer
takes seconds, and a suite slow enough to skip is a suite that stops catching
things. The real model is exercised separately, against the real corpus, where
what is being measured is retrieval *quality* rather than retrieval mechanics.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from assistant.chunking import Chunk
from assistant.embedding import QUERY_PREFIX, FastEmbedEmbedder
from assistant.retrieval import InMemoryRetriever, SearchResult

VOCAB = ["prompt", "format", "example", "context", "boxing", "nutrition"]


class BagOfWordsEmbedder:
    """A real embedder in miniature: unit vectors over a tiny fixed vocabulary.

    Similarity behaves the way the actual model's does — texts sharing words
    score higher — so retrieval assertions mean something, while staying
    instant and perfectly predictable.
    """

    def __init__(self) -> None:
        self.passage_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def dimensions(self) -> int:
        return len(VOCAB)

    def _vector(self, text: str) -> NDArray[np.float32]:
        words = text.lower().split()
        counts = np.array([words.count(term) for term in VOCAB], dtype=np.float32)
        norm = float(np.linalg.norm(counts))
        return counts / norm if norm else counts

    def embed_passages(self, texts: list[str]) -> NDArray[np.float32]:
        self.passage_calls.append(texts)
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> NDArray[np.float32]:
        self.query_calls.append(text)
        return self._vector(text)


def chunk(text: str, index: int, section: str | None = None) -> Chunk:
    return Chunk(text=text, source="doc.md", page=None, section=section, index=index)


@pytest.fixture
def retriever() -> InMemoryRetriever:
    chunks = [
        chunk("prompt prompt format", 0, "Format"),
        chunk("example example example", 1, "Examples"),
        chunk("boxing nutrition", 2, "Unrelated"),
    ]
    return InMemoryRetriever(chunks, BagOfWordsEmbedder())


def test_returns_the_closest_chunk_first(retriever: InMemoryRetriever) -> None:
    results = retriever.search("prompt format")
    assert results[0].chunk.index == 0


def test_results_are_ordered_best_first(retriever: InMemoryRetriever) -> None:
    scores = [r.score for r in retriever.search("prompt example", top_k=3)]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_the_number_returned(retriever: InMemoryRetriever) -> None:
    assert len(retriever.search("prompt", top_k=2)) == 2


def test_top_k_larger_than_the_corpus_returns_everything(
    retriever: InMemoryRetriever,
) -> None:
    assert len(retriever.search("prompt", top_k=99)) == 3


@pytest.mark.parametrize("top_k", [0, -1])
def test_non_positive_top_k_returns_nothing(
    retriever: InMemoryRetriever, top_k: int
) -> None:
    assert retriever.search("prompt", top_k=top_k) == []


def test_empty_corpus_searches_without_error() -> None:
    empty = InMemoryRetriever([], BagOfWordsEmbedder())
    assert empty.search("anything") == []
    assert len(empty) == 0


def test_scores_are_cosine_similarities(retriever: InMemoryRetriever) -> None:
    results = retriever.search("prompt format", top_k=3)
    for result in results:
        assert -1.0 <= result.score <= 1.0
    # An unrelated chunk shares no vocabulary, so it must score exactly zero.
    unrelated = next(r for r in results if r.chunk.index == 2)
    assert unrelated.score == pytest.approx(0.0)


def test_retrieval_preserves_the_citation(retriever: InMemoryRetriever) -> None:
    result = retriever.search("prompt format")[0]
    assert result.cite() == "doc.md — Format"
    assert result.chunk.source == "doc.md"


def test_a_vector_count_mismatch_fails_loudly() -> None:
    """Row i must be chunk i, or every citation silently misattributes.

    A retriever that returned the wrong chunk for a row would still produce
    fluent, confident answers — with citations pointing at documents that do not
    contain them. That is the single worst failure this system can have, so it
    is caught at construction rather than trusted.
    """

    class WrongCountEmbedder(BagOfWordsEmbedder):
        def embed_passages(self, texts: list[str]) -> NDArray[np.float32]:
            return super().embed_passages(texts)[:-1]

    with pytest.raises(ValueError, match="vectors"):
        InMemoryRetriever([chunk("a", 0), chunk("b", 1)], WrongCountEmbedder())


def test_passages_are_embedded_once_at_construction() -> None:
    embedder = BagOfWordsEmbedder()
    retriever = InMemoryRetriever([chunk("prompt", 0)], embedder)

    retriever.search("prompt")
    retriever.search("format")

    # Indexing cost is paid once; queries must not re-encode the corpus.
    assert len(embedder.passage_calls) == 1
    assert len(embedder.query_calls) == 2


def test_search_result_is_frozen() -> None:
    result = SearchResult(chunk=chunk("a", 0), score=0.5)
    with pytest.raises(AttributeError):
        result.score = 0.9  # type: ignore[misc]


class TestQueryPrefix:
    """The prefix is worth two top-1 hits on the real corpus (5/9 -> 7/9).

    It is invisible in output and produces no error when missing, so nothing but
    a test will notice if it is dropped during a refactor.
    """

    def test_the_prefix_is_applied_to_queries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[list[str]] = []
        embedder = FastEmbedEmbedder()

        def fake_encode(texts: list[str]) -> NDArray[np.float32]:
            seen.append(texts)
            return np.ones((len(texts), 384), dtype=np.float32)

        monkeypatch.setattr(embedder, "_encode", fake_encode)
        embedder.embed_query("What are the four components?")

        assert seen == [[QUERY_PREFIX + "What are the four components?"]]

    def test_the_prefix_is_never_applied_to_passages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prefixing passages too would destroy the asymmetry that helps."""
        seen: list[list[str]] = []
        embedder = FastEmbedEmbedder()

        def fake_encode(texts: list[str]) -> NDArray[np.float32]:
            seen.append(texts)
            return np.ones((len(texts), 384), dtype=np.float32)

        monkeypatch.setattr(embedder, "_encode", fake_encode)
        embedder.embed_passages(["A prompt has four components."])

        assert seen == [["A prompt has four components."]]
        assert QUERY_PREFIX not in seen[0][0]

    def test_embedding_no_passages_skips_the_model_entirely(self) -> None:
        embedder = FastEmbedEmbedder()
        result = embedder.embed_passages([])
        assert result.shape == (0, embedder.dimensions)


class FakeTextEmbedding:
    """Stands in for fastembed's model, returning deliberately un-normalised rows."""

    instances = 0

    def __init__(self, model_name: str) -> None:
        FakeTextEmbedding.instances += 1
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]:
        # Row scale varies and one row is all-zero, so both the normalisation
        # and the divide-by-zero guard are actually exercised.
        return [np.full(384, float(len(text) or 0), dtype=np.float32) for text in texts]


class TestEncoding:
    """Covers the parts of the real embedder that do not need a model download."""

    @pytest.fixture(autouse=True)
    def _fake_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import fastembed

        FakeTextEmbedding.instances = 0
        monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)

    def test_encoded_rows_are_unit_vectors(self) -> None:
        matrix = FastEmbedEmbedder().embed_passages(["short", "a much longer passage"])
        assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)

    def test_an_empty_string_does_not_produce_nan(self) -> None:
        """A zero vector divided by its own norm would poison every later score."""
        matrix = FastEmbedEmbedder().embed_passages([""])
        assert not np.isnan(matrix).any()

    def test_output_is_float32(self) -> None:
        assert FastEmbedEmbedder().embed_passages(["text"]).dtype == np.float32

    def test_the_model_is_loaded_lazily_and_only_once(self) -> None:
        embedder = FastEmbedEmbedder()
        assert FakeTextEmbedding.instances == 0, "constructing must not load a model"

        embedder.embed_passages(["one"])
        embedder.embed_query("two")

        assert FakeTextEmbedding.instances == 1
