"""Finding the chunks most likely to answer a question.

A brute-force scan over an in-memory matrix, deliberately. A corpus of this size
scores every chunk in well under a millisecond, so an approximate-nearest-
neighbour index would add a dependency, a build step and a whole class of
"the index is stale" bugs to solve a problem that does not exist yet.

`Retriever` is a `Protocol` so that swap is a contained change when a corpus
does grow large enough to need one — and so the rest of the system can be tested
against a trivial fake instead of a real index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from assistant.chunking import Chunk
from assistant.embedding import Embedder

DEFAULT_TOP_K = 4
"""How many chunks to put in front of the model.

Four is a starting point, not a tuned value. Too few and the answer is missed
when the best chunk ranks second; too many and weakly-related text crowds the
context, which makes ungrounded answers *more* likely rather than less. Tune
against the evaluation set in M4.
"""


@dataclass(frozen=True)
class SearchResult:
    """A chunk and how well it matched, ordered best-first by the retriever."""

    chunk: Chunk
    score: float

    def cite(self) -> str:
        return self.chunk.cite()


class Retriever(Protocol):
    """Returns the chunks most similar to a query, best first."""

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]: ...


class InMemoryRetriever:
    """Exhaustive cosine search over an in-memory embedding matrix.

    Embeddings are computed once at construction, or supplied. A corpus that
    changes only at deploy time need not be embedded at every start: passing
    `matrix` uses vectors built earlier — see `vectors.py`, which owns the
    checks that decide whether a stored matrix still describes these chunks.
    Computing them here stays the default, because it is the behaviour that
    cannot be wrong.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        matrix: NDArray[np.float32] | None = None,
    ) -> None:
        self._chunks = chunks
        self._embedder = embedder
        # `indexed_text`, not `text`: the heading is part of what gets searched,
        # and a matrix built from one and queried against the other would rank
        # everything slightly wrong while looking entirely normal.
        self._matrix: NDArray[np.float32] = (
            matrix
            if matrix is not None
            else embedder.embed_passages([chunk.indexed_text() for chunk in chunks])
        )

        if self._matrix.shape[0] != len(chunks):
            # A mismatch here would silently misattribute every citation: row i
            # would no longer be chunk i, so answers would cite the wrong source
            # while looking entirely normal. Fail loudly at construction instead.
            raise ValueError(
                f"embedder returned {self._matrix.shape[0]} vectors "
                f"for {len(chunks)} chunks"
            )

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        if not self._chunks or top_k <= 0:
            return []

        query_vector = self._embedder.embed_query(query)
        # Both sides are unit vectors, so the dot product *is* cosine similarity.
        scores = self._matrix @ query_vector

        take = min(top_k, len(self._chunks))
        # argpartition finds the top-k without sorting the rest; only those k are
        # then sorted. Irrelevant at this size, but it costs nothing and means
        # the hot path does not quietly become O(n log n) as a corpus grows.
        candidates = np.argpartition(-scores, take - 1)[:take]
        ordered = candidates[np.argsort(-scores[candidates])]

        return [
            SearchResult(chunk=self._chunks[int(i)], score=float(scores[i]))
            for i in ordered
        ]
