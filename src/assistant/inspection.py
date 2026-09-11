"""Read-only descriptions of the corpus and the chunks retrieval actually uses.

This module deliberately stops before embedding or answering. It uses the same
document enumeration, readers and chunker as the production path, then turns
their output into immutable records suitable for a local operator interface.
No function here writes a file, loads a model or constructs a provider client.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from assistant.chunking import (
    MIN_WORDS,
    OVERLAP_WORDS,
    TARGET_WORDS,
    Chunk,
    chunk_passages,
)
from assistant.corpus_checksum import corpus_checksum, file_digests
from assistant.documents import corpus_documents, read_corpus
from assistant.embedding import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_WINDOW_TOKENS,
    MODEL_NAME,
)
from assistant.retrieval import DEFAULT_TOP_K
from assistant.vectors import VectorsMismatch
from assistant.vectors import load as load_vectors

ESTIMATED_TOKENS_PER_WORD = 1.3
"""A labelled estimate, matching the conservative chunk-window test.

The project chunks by words. FastEmbed does not expose a stable public token
counting API, so presenting this as an exact tokenizer result would be false
precision. The count includes ``Chunk.indexed_text()`` because that heading plus
body string, not the body alone, is what the embedding model receives.
"""


class InspectionError(RuntimeError):
    """The selected corpus cannot be described safely and honestly."""


def corpus_id(label: str) -> str:
    """Build a stable URL-safe identifier from an operator-supplied label."""
    value = re.sub(r"[^a-z0-9]+", "-", label.casefold()).strip("-")
    if not value:
        raise InspectionError(
            "Every corpus needs a label containing a letter or number."
        )
    return value


@dataclass(frozen=True)
class CorpusProfile:
    """One corpus selected at process startup, never from a browser request."""

    id: str
    label: str
    directory: Path
    vectors_file: Path | None = None

    @classmethod
    def create(
        cls, label: str, directory: Path, vectors_file: Path | None = None
    ) -> CorpusProfile:
        clean_label = label.strip()
        if not clean_label:
            raise InspectionError("Every corpus needs a non-empty label.")
        return cls(corpus_id(clean_label), clean_label, directory, vectors_file)


@dataclass(frozen=True)
class DocumentInspection:
    source: str
    format: str
    sha256: str
    size_bytes: int
    passage_count: int
    chunk_count: int
    estimated_embedding_tokens: int
    estimated_chunk_tokens: dict[str, int | float] | None
    pages: tuple[int, ...]
    sections: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "format": self.format,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "passage_count": self.passage_count,
            "chunk_count": self.chunk_count,
            "estimated_embedding_tokens": self.estimated_embedding_tokens,
            "estimated_chunk_tokens": self.estimated_chunk_tokens,
            "pages": list(self.pages),
            "sections": list(self.sections),
            "status": "ready" if self.passage_count else "no_extractable_text",
        }


@dataclass(frozen=True)
class ChunkInspection:
    index: int
    source: str
    page: int | None
    section: str | None
    citation: str
    body_words: int
    indexed_words: int
    estimated_embedding_tokens: int
    text: str
    indexed_text: str

    def as_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "index": self.index,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "citation": self.citation,
            "body_words": self.body_words,
            "indexed_words": self.indexed_words,
            "estimated_embedding_tokens": self.estimated_embedding_tokens,
            "embedding_window_tokens": EMBEDDING_WINDOW_TOKENS,
            "preview": _preview(self.text),
        }
        if include_text:
            result["text"] = self.text
            result["indexed_text"] = self.indexed_text
        return result


VectorState = Literal["not_supplied", "valid", "invalid"]


@dataclass(frozen=True)
class VectorInspection:
    state: VectorState
    filename: str | None
    message: str

    def as_dict(self) -> dict[str, str | None]:
        return {
            "state": self.state,
            "filename": self.filename,
            "message": self.message,
        }


@dataclass(frozen=True)
class CorpusInspection:
    id: str
    label: str
    checksum: str
    documents: tuple[DocumentInspection, ...]
    passage_count: int
    chunks: tuple[ChunkInspection, ...]
    vectors: VectorInspection

    def summary(self) -> dict[str, Any]:
        word_sizes = [chunk.body_words for chunk in self.chunks]
        token_sizes = [chunk.estimated_embedding_tokens for chunk in self.chunks]
        return {
            "id": self.id,
            "label": self.label,
            "document_count": len(self.documents),
            "passage_count": self.passage_count,
            "chunk_count": len(self.chunks),
            "checksum": self.checksum,
            "chunk_words": _range_summary(word_sizes),
            "estimated_embedding_tokens": _range_summary(token_sizes),
            "chunking": {
                "target_words": TARGET_WORDS,
                "overlap_words": OVERLAP_WORDS,
                "minimum_words": MIN_WORDS,
                "estimated_target_tokens": _estimated_tokens(TARGET_WORDS),
                "estimated_overlap_tokens": _estimated_tokens(OVERLAP_WORDS),
                "estimated_minimum_tokens": _estimated_tokens(MIN_WORDS),
                "counting_basis": (
                    "Production chunks are word-based; token values are labelled "
                    "estimates."
                ),
            },
            "embedding": {
                "model": MODEL_NAME,
                "dimensions": EMBEDDING_DIMENSIONS,
                "window_tokens": EMBEDDING_WINDOW_TOKENS,
                "token_measurement": (
                    "Estimated from indexed words at 1.3 tokens per word; not an "
                    "exact tokenizer count."
                ),
            },
            "retrieval": {"default_top_k": DEFAULT_TOP_K},
            "vectors": self.vectors.as_dict(),
        }

    def detail(self) -> dict[str, Any]:
        result = self.summary()
        result["documents"] = [document.as_dict() for document in self.documents]
        return result

    def chunk(self, index: int) -> ChunkInspection | None:
        return next((chunk for chunk in self.chunks if chunk.index == index), None)


def inspect_corpus(profile: CorpusProfile) -> CorpusInspection:
    """Describe one fixed corpus using the production ingestion pipeline."""
    if not profile.directory.is_dir():
        raise InspectionError(f"{profile.label}: corpus directory does not exist.")

    paths = corpus_documents(profile.directory)
    passages = read_corpus(profile.directory)
    chunks = chunk_passages(passages)
    if not paths or not chunks:
        raise InspectionError(
            f"{profile.label}: no extractable corpus content was found."
        )

    passage_counts = Counter(passage.source for passage in passages)
    chunk_counts = Counter(chunk.source for chunk in chunks)
    inspected_chunks = tuple(_inspect_chunk(chunk) for chunk in chunks)
    token_counts: dict[str, list[int]] = defaultdict(list)
    for chunk in inspected_chunks:
        token_counts[chunk.source].append(chunk.estimated_embedding_tokens)
    digests = dict(file_digests(profile.directory))
    pages: dict[str, set[int]] = defaultdict(set)
    sections: dict[str, set[str]] = defaultdict(set)
    for passage in passages:
        if passage.page is not None:
            pages[passage.source].add(passage.page)
        if passage.section:
            sections[passage.source].add(passage.section)

    documents = tuple(
        DocumentInspection(
            source=path.relative_to(profile.directory).as_posix(),
            format=path.suffix.lower().lstrip("."),
            sha256=digests[path.relative_to(profile.directory).as_posix()],
            size_bytes=path.stat().st_size,
            passage_count=passage_counts[
                path.relative_to(profile.directory).as_posix()
            ],
            chunk_count=chunk_counts[path.relative_to(profile.directory).as_posix()],
            estimated_embedding_tokens=sum(
                token_counts[path.relative_to(profile.directory).as_posix()]
            ),
            estimated_chunk_tokens=(
                _range_summary(
                    token_counts[path.relative_to(profile.directory).as_posix()]
                )
                if token_counts[path.relative_to(profile.directory).as_posix()]
                else None
            ),
            pages=tuple(sorted(pages[path.relative_to(profile.directory).as_posix()])),
            sections=tuple(
                sorted(sections[path.relative_to(profile.directory).as_posix()])
            ),
        )
        for path in paths
    )
    vectors = _inspect_vectors(profile, chunks)
    return CorpusInspection(
        id=profile.id,
        label=profile.label,
        checksum=corpus_checksum(profile.directory),
        documents=documents,
        passage_count=len(passages),
        chunks=inspected_chunks,
        vectors=vectors,
    )


def _inspect_chunk(chunk: Chunk) -> ChunkInspection:
    indexed_text = chunk.indexed_text()
    indexed_words = len(indexed_text.split())
    return ChunkInspection(
        index=chunk.index,
        source=chunk.source,
        page=chunk.page,
        section=chunk.section,
        citation=chunk.cite(),
        body_words=len(chunk.text.split()),
        indexed_words=indexed_words,
        estimated_embedding_tokens=math.ceil(indexed_words * ESTIMATED_TOKENS_PER_WORD),
        text=chunk.text,
        indexed_text=indexed_text,
    )


def _inspect_vectors(profile: CorpusProfile, chunks: list[Chunk]) -> VectorInspection:
    path = profile.vectors_file
    if path is None:
        return VectorInspection(
            "not_supplied",
            None,
            "No vectors file was supplied for local validation.",
        )
    try:
        load_vectors(
            path,
            chunks,
            model=MODEL_NAME,
            dimensions=EMBEDDING_DIMENSIONS,
        )
    except (OSError, ValueError, VectorsMismatch):
        return VectorInspection(
            "invalid",
            path.name,
            "The supplied vectors do not match this corpus and embedding model.",
        )
    return VectorInspection(
        "valid",
        path.name,
        "The supplied vectors match this corpus, chunk order, model and dimensions.",
    )


def _preview(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def _estimated_tokens(words: int) -> int:
    return math.ceil(words * ESTIMATED_TOKENS_PER_WORD)


def _range_summary(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    return {
        "minimum": ordered[0],
        "median": ordered[len(ordered) // 2],
        "average": round(sum(ordered) / len(ordered), 1),
        "maximum": ordered[-1],
    }
