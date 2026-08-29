"""A deterministic fingerprint of a corpus directory.

The corpus for a deployed instance is authored in one repository and served from
another. That copy is the weak seam in the arrangement: a stale, truncated,
half-copied or quietly edited corpus produces a service that answers *confidently
from the wrong content*, which is the one failure this project exists to prevent
and the one a reader cannot detect from the outside.

The checksum closes that seam by making a wrong corpus loud rather than silent.
It does not make the copy automatic — worth saying plainly, because a checksum is
easy to mistake for a synchronisation mechanism. It is a tripwire.

THE ALGORITHM IS A CROSS-LANGUAGE CONTRACT.
It is implemented here and, identically, in the authoring repository's
TypeScript. Both sides assert the same fixture digest, so a change to one that
does not reach the other fails a test rather than a deployment. Do not "tidy"
any step below without changing both and re-running both suites.

    1. Select exactly the files `read_corpus` would read, via
       `is_corpus_document`. Hashing a different set from the one that gets
       served would defeat the point.
    2. Label each by its corpus-relative POSIX path, so a Windows-built artifact
       and a Linux-built one agree.
    3. Sort by that label. Directory iteration order is not a promise.
    4. Normalise line endings for `.md` and `.txt`, hash `.pdf` and `.docx`
       byte-for-byte. Text files pass through Git's `eol` handling and can
       legitimately differ by CRLF between checkouts; binary files cannot, and
       normalising them would corrupt the hash of a valid file.
    5. Digest each file, then digest the joined `"<digest>  <path>\\n"` lines.
       Including the path means a rename is a change, which it is.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from assistant.documents import is_corpus_document

TEXT_SUFFIXES = frozenset({".md", ".txt"})
"""Formats whose line endings are normalised before hashing.

Deliberately not "every text-ish format". `.docx` is a zip archive and `.pdf`
may contain CR bytes inside binary streams; rewriting those would change a file
that is byte-identical to the one that was reviewed.
"""


def _normalise(data: bytes, suffix: str) -> bytes:
    if suffix.lower() not in TEXT_SUFFIXES:
        return data
    # CRLF first, then any surviving lone CR: doing it in the other order turns
    # every CRLF into two newlines.
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def file_digests(directory: Path) -> list[tuple[str, str]]:
    """`(relative POSIX path, sha256 hex)` for every corpus document, path-sorted.

    Exposed separately from `corpus_checksum` so a mismatch can be *localised*.
    Being told two corpora differ is much less useful than being told which file
    differs, and at the moment that matters — a failed deployment check — nobody
    wants to bisect by hand.
    """
    entries: list[tuple[str, str]] = []
    for path in directory.rglob("*"):
        if not is_corpus_document(path):
            continue
        relative = path.relative_to(directory).as_posix()
        digest = hashlib.sha256(_normalise(path.read_bytes(), path.suffix)).hexdigest()
        entries.append((relative, digest))
    entries.sort(key=lambda entry: entry[0])
    return entries


def corpus_checksum(directory: Path) -> str:
    """The single hex digest identifying this corpus.

    An empty corpus still returns a digest — the sha256 of the empty string —
    rather than raising. Whether an empty corpus is acceptable is a separate
    question, and `api.py` refuses to start on one; conflating "no documents"
    with "cannot compute a checksum" would put that decision in the wrong place.
    """
    joined = "".join(f"{digest}  {path}\n" for path, digest in file_digests(directory))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class CorpusChecksumMismatch(RuntimeError):
    """The corpus on disk is not the corpus that was reviewed and approved."""


def verify_corpus(directory: Path, expected: str) -> str:
    """Check the corpus against an expected digest, returning the actual one.

    An empty or whitespace-only `expected` means verification is not configured
    and the actual checksum is returned unchecked. That is the correct behaviour
    for local development and for the demo instance, which has no separate
    authoring repository — but it is *not* safe for a deployment whose corpus was
    built elsewhere, which is why `api.py` decides whether to require it rather
    than this function guessing.
    """
    actual = corpus_checksum(directory)
    wanted = expected.strip()
    if wanted and actual != wanted:
        raise CorpusChecksumMismatch(
            "corpus checksum mismatch: refusing to serve content that was not "
            f"the content approved. expected {wanted}, found {actual}. "
            "Re-export the corpus artifact and redeploy."
        )
    return actual
