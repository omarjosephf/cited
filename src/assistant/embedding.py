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

import os
import threading
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]
"""An L2-normalised embedding. Normalisation makes cosine similarity a dot product."""

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384
EMBEDDING_WINDOW_TOKENS = 512

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

    def __init__(
        self, model_name: str = MODEL_NAME, cache_dir: str | None = None
    ) -> None:
        self._model_name = model_name
        # Falls back to `EMBEDDING_CACHE_DIR`, then to fastembed's own default.
        #
        # This exists so a container image can pre-download the model at build
        # time and have the running process find it. fastembed takes the cache
        # location as a constructor argument and reads no environment variable
        # of its own — a Dockerfile that sets one and expects it to be honoured
        # downloads the model again on every cold start, which reads to a
        # visitor as the demo being broken rather than loading.
        self._cache_dir = cache_dir or os.environ.get("EMBEDDING_CACHE_DIR") or None
        self._model: object | None = None
        self._lock = threading.Lock()

    def _loaded(self) -> object:
        # Double-checked under a lock, because with precomputed vectors the
        # first load is genuinely concurrent: the API warms the model on a
        # background thread while a visitor's first question may arrive on
        # another. Unguarded, both threads see `None` and build a model, and the
        # loser's copy is discarded after the memory has already been taken —
        # a transient doubling on the one machine sized against a measured peak.
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(
                        model_name=self._model_name, cache_dir=self._cache_dir
                    )
        return self._model

    @property
    def dimensions(self) -> int:
        return EMBEDDING_DIMENSIONS

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
