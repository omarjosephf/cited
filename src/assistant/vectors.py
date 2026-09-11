"""Corpus vectors computed ahead of time, and the binding that keeps them honest.

Embedding the corpus at startup was 80% of a cold start. Measured on the
portfolio corpus (10 documents, 64 chunks), on the development machine:

    read + chunk corpus     0.21s
    import onnxruntime      0.37s
    load bge-small          1.03s
    embed the 64 chunks     7.39s      <- this
    embed one query         0.05s

On the deployment's throttled shared CPU the same work took three to four
minutes, and the portfolio's route gives up after twenty seconds — so the first
visitor after an idle period did not get a slow answer, they got "unavailable".
Computing the matrix once, at build time, on a machine that is not throttled,
removes that step from the startup path entirely: 8.37s to ready becomes 0.20s.

WHAT THIS DOES NOT REMOVE. A question still has to be embedded, and that still
needs the model. Precomputing moves the model load off startup and onto the
first question; `api.py` covers the gap by warming the embedder in the
background once it is already serving. Anyone reading this as "the service no
longer needs the model" will size the container wrong.

THE FAILURE THIS FILE EXISTS TO PREVENT. A stored matrix is a copy of a
derived thing, and a copy can go stale. Row *i* stops being chunk *i* the moment
the corpus, the chunker or `Chunk.indexed_text` changes, and nothing about a
stale matrix looks wrong: every answer still arrives, with citations, confidently
attributed to the wrong passage. `InMemoryRetriever` already refuses a matrix of
the wrong *height* for exactly this reason; a digest of the embedded strings is
the same guard extended to their *content*, which is the half a row count cannot
see.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from assistant.chunking import Chunk

FORMAT_VERSION = 1
"""Bumped when the file layout changes in a way an older reader would misread.

A version that only ever appears in a mismatch message is doing its job.
"""


class VectorsMismatch(RuntimeError):
    """A stored matrix does not describe the chunks it is being loaded for."""


def chunk_digest(chunks: list[Chunk]) -> str:
    """A fingerprint of exactly the strings that were embedded, in order.

    Order is part of the digest because order is what makes row *i* mean chunk
    *i*. The separator is a NUL, which cannot occur in the text being joined, so
    two different chunk lists cannot produce one identical joined string.
    """
    joined = "\x00".join(chunk.indexed_text() for chunk in chunks)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def save(
    path: Path,
    chunks: list[Chunk],
    matrix: NDArray[np.float32],
    *,
    model: str,
    corpus_checksum: str = "",
) -> None:
    """Write the matrix together with everything needed to reject it later.

    `corpus_checksum` is recorded for a human reading a mismatch report — it says
    *which* corpus these vectors were built from. It is deliberately not what
    `load` verifies: the checksum covers the files on disk, while the digest
    covers the strings actually embedded, and it is the second one that decides
    whether a row still means what it says.
    """
    if matrix.shape[0] != len(chunks):
        raise ValueError(
            f"{matrix.shape[0]} vectors for {len(chunks)} chunks; refusing to save"
        )

    metadata = {
        "format_version": FORMAT_VERSION,
        "model": model,
        "dimensions": int(matrix.shape[1]),
        "chunks": len(chunks),
        "chunk_digest": chunk_digest(chunks),
        "corpus_checksum": corpus_checksum,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vectors=matrix.astype(np.float32, copy=False),
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )


def load(
    path: Path, chunks: list[Chunk], *, model: str, dimensions: int
) -> NDArray[np.float32]:
    """Read the matrix for `chunks`, or raise.

    Every check below raises rather than falling back to embedding the corpus.
    A fallback would turn "these vectors are wrong" into a slow start nobody
    investigates, and the whole point of the digest is that this failure is loud.
    """
    if not path.is_file():
        raise VectorsMismatch(
            f"corpus vectors file {path} does not exist. Build it with "
            "`doc-assistant embed`, or unset CORPUS_VECTORS_FILE to embed at "
            "startup."
        )

    with np.load(path, allow_pickle=False) as data:
        if "vectors" not in data or "metadata" not in data:
            raise VectorsMismatch(f"{path} is not a corpus vectors file.")
        matrix = cast("NDArray[np.float32]", data["vectors"].astype(np.float32))
        raw = str(data["metadata"].item())

    try:
        metadata = cast(dict[str, Any], json.loads(raw))
    except json.JSONDecodeError as error:
        raise VectorsMismatch(f"{path} has unreadable metadata: {error}") from error

    _require(
        metadata.get("format_version") == FORMAT_VERSION,
        f"{path} is format version {metadata.get('format_version')!r}, "
        f"this build reads version {FORMAT_VERSION}",
    )
    _require(
        metadata.get("model") == model,
        f"{path} was built with model {metadata.get('model')!r}, "
        f"this build queries with {model!r}. Vectors from two models are not "
        "comparable, and mixing them scores every chunk against the wrong "
        "geometry",
    )
    _require(
        matrix.ndim == 2 and matrix.shape[1] == dimensions,
        f"{path} holds {matrix.shape} vectors, expected width {dimensions}",
    )
    _require(
        matrix.shape[0] == len(chunks),
        f"{path} holds {matrix.shape[0]} vectors for {len(chunks)} chunks",
    )
    expected = chunk_digest(chunks)
    _require(
        metadata.get("chunk_digest") == expected,
        f"{path} was built for different text: digest "
        f"{str(metadata.get('chunk_digest'))[:12]}, corpus now hashes to "
        f"{expected[:12]}. The corpus, the chunker or Chunk.indexed_text has "
        "changed since these vectors were built — rebuild them",
    )
    return matrix


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VectorsMismatch(message)
