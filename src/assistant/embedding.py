"""Turning text into vectors.

The embedder sits behind a `Protocol` so that everything downstream — retrieval,
answering, the evaluation harness — can be tested without loading a model.
Unit tests that each spend seconds downloading and running a transformer stop
being run, and a test suite nobody runs protects nothing.

Embeddings are computed locally on CPU via ONNX. There is no embedding API call,
so indexing a corpus is free and works offline; the only paid call in the whole
system is answer generation.
"""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]
"""An L2-normalised embedding. Normalisation makes cosine similarity a dot product."""

MODEL_NAME = "BAAI/bge-small-en-v1.5"

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
"""Prepended to queries only — never to passages.

`bge` is trained asymmetrically: a question and the passage answering it are not
the same kind of text, and the model is told which it is looking at. fastembed's
`query_embed()` does *not* apply this (it is byte-identical to `embed()`, cosine
1.0000 on the same input), so it is applied here by hand.

Measured on the project corpus before being adopted, rather than taken on trust
from the model card: top-1 retrieval went from 5/9 to 7/9 across a fixed set of
questions, with top-3 unchanged at 8/9. Re-measure if the corpus or model
changes — the gain is a property of this pairing, not a law.
"""


class Embedder(Protocol):
    """Encodes text for retrieval.

    Two methods rather than one because passages and queries are encoded
    differently. Collapsing them into a single `embed()` is the mistake this
    interface exists to prevent: it silently drops the query prefix and costs
    real retrieval accuracy without producing an error anywhere.
    """

    @property
    def dimensions(self) -> int: ...

    def embed_passages(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode corpus text. Returns one row per input, in order."""
        ...

    def embed_query(self, text: str) -> Vector:
        """Encode a search query, with whatever instruction prefix the model wants."""
        ...


class FastEmbedEmbedder:
    """`Embedder` backed by fastembed's ONNX runtime.

    The model is loaded lazily on first use. Constructing an embedder is
    therefore cheap, which matters because the API process builds one at import
    time but should not pay a model load before it can serve a health check.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: object | None = None

    def _loaded(self) -> object:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    @property
    def dimensions(self) -> int:
        return 384  # bge-small-en-v1.5

    def _encode(self, texts: list[str]) -> NDArray[np.float32]:
        model = self._loaded()
        raw = list(model.embed(texts))  # type: ignore[attr-defined]
        matrix = np.asarray(raw, dtype=np.float32)
        # fastembed already returns unit vectors, but normalising is cheap and
        # makes the invariant local: everything downstream treats a dot product
        # as cosine similarity, and that must hold even if the backend changes.
        # The zero-guard matters because an empty string encodes to a zero
        # vector, and dividing by its norm would put NaN into the index — which
        # then propagates into every score without raising anything.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalised = matrix / np.maximum(norms, 1e-12)
        # numpy's stubs type the result of this division as `Any`, so the cast
        # restores what is actually guaranteed: `matrix` is float32 and division
        # by a float64 scalar array preserves the float32 dtype. A test asserts
        # the dtype at runtime rather than trusting the cast.
        return cast("NDArray[np.float32]", normalised.astype(np.float32, copy=False))

    def embed_passages(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return self._encode(texts)

    def embed_query(self, text: str) -> Vector:
        # Indexing a row out of the matrix is another spot where numpy's stubs
        # widen to `Any`; the row is float32 by construction of `_encode`.
        return cast("Vector", self._encode([QUERY_PREFIX + text])[0])
