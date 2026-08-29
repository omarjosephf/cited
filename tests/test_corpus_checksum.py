"""Tests for the corpus fingerprint.

The checksum's only job is to notice that the corpus is not the one that was
approved. So the tests worth having are the ones that would let a *wrong* corpus
pass: a file quietly edited, added, removed or renamed. Each of those has a case
below, because each is a way this control could be useless while appearing to
work.

The cross-language fixture at the bottom is the load-bearing one. The algorithm
is duplicated in TypeScript in the authoring repository, and duplicated logic
drifts. A frozen expected digest here and there turns that drift into a failing
test in whichever repository moved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant.corpus_checksum import (
    CorpusChecksumMismatch,
    corpus_checksum,
    file_digests,
    verify_corpus,
)


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    write(tmp_path, "about.md", "# About\n\nOJ builds practical products.\n")
    write(tmp_path, "projects/cited.md", "# Cited\n\nA document assistant.\n")
    return tmp_path


class TestStability:
    def test_the_same_corpus_hashes_the_same_way_twice(self, corpus: Path) -> None:
        assert corpus_checksum(corpus) == corpus_checksum(corpus)

    def test_an_empty_directory_has_a_checksum_rather_than_an_error(
        self, tmp_path: Path
    ) -> None:
        """Whether an empty corpus is servable is `api.py`'s decision, not this
        module's. Raising here would put that judgement in the wrong place."""
        assert len(corpus_checksum(tmp_path)) == 64

    def test_line_endings_do_not_change_a_text_document_digest(
        self, tmp_path: Path
    ) -> None:
        """A CRLF checkout must produce the same artifact as an LF one.

        Without this the checksum would fail on Windows for a corpus that is
        byte-identical as far as Git is concerned, and the natural fix would be
        to stop verifying — losing the control to a platform detail.
        """
        unix = tmp_path / "unix"
        windows = tmp_path / "windows"
        unix.mkdir()
        windows.mkdir()
        (unix / "a.md").write_bytes(b"line one\nline two\n")
        (windows / "a.md").write_bytes(b"line one\r\nline two\r\n")

        assert corpus_checksum(unix) == corpus_checksum(windows)

    def test_a_pdf_is_hashed_byte_for_byte(self, tmp_path: Path) -> None:
        """Binary formats are not normalised: a PDF can legitimately contain CR
        bytes inside a stream, and rewriting them would change a valid file."""
        crlf = tmp_path / "crlf"
        lf = tmp_path / "lf"
        crlf.mkdir()
        lf.mkdir()
        (crlf / "cv.pdf").write_bytes(b"%PDF-1.4\r\nbody\r\n")
        (lf / "cv.pdf").write_bytes(b"%PDF-1.4\nbody\n")

        assert corpus_checksum(crlf) != corpus_checksum(lf)


class TestSensitivity:
    def test_editing_a_document_changes_the_checksum(self, corpus: Path) -> None:
        before = corpus_checksum(corpus)
        write(corpus, "about.md", "# About\n\nOJ builds other things.\n")

        assert corpus_checksum(corpus) != before

    def test_adding_a_document_changes_the_checksum(self, corpus: Path) -> None:
        before = corpus_checksum(corpus)
        write(corpus, "services.md", "# Services\n")

        assert corpus_checksum(corpus) != before

    def test_removing_a_document_changes_the_checksum(self, corpus: Path) -> None:
        before = corpus_checksum(corpus)
        (corpus / "about.md").unlink()

        assert corpus_checksum(corpus) != before

    def test_renaming_a_document_changes_the_checksum(self, corpus: Path) -> None:
        """The path is part of the hash because it is part of the citation. A
        renamed file cites differently, so it is a different corpus."""
        before = corpus_checksum(corpus)
        (corpus / "about.md").rename(corpus / "about-oj.md")

        assert corpus_checksum(corpus) != before


class TestSelection:
    def test_only_files_the_reader_would_read_are_hashed(self, corpus: Path) -> None:
        """The hashed set must equal the served set.

        Hashing a different set is the subtle failure: the checksum would pass
        while the content served had changed, which is worse than no checksum at
        all because it carries a guarantee it does not provide.
        """
        before = corpus_checksum(corpus)
        write(corpus, "README.md", "Notes about this folder, not content.\n")
        write(corpus, "_draft.md", "Not ready.\n")
        write(corpus, "notes.rst", "Unsupported format.\n")

        assert corpus_checksum(corpus) == before

    def test_digests_are_reported_per_file_and_path_sorted(self, corpus: Path) -> None:
        """Sorted so a mismatch can be localised, and so directory iteration
        order — which is not a promise on any platform — cannot change it."""
        paths = [path for path, _ in file_digests(corpus)]

        assert paths == ["about.md", "projects/cited.md"]
        assert paths == sorted(paths)


class TestVerification:
    def test_a_matching_checksum_passes_and_returns_the_actual_value(
        self, corpus: Path
    ) -> None:
        expected = corpus_checksum(corpus)

        assert verify_corpus(corpus, expected) == expected

    def test_surrounding_whitespace_in_the_expected_value_is_tolerated(
        self, corpus: Path
    ) -> None:
        """The value arrives from a file or an environment variable, and a
        trailing newline is not a corpus mismatch."""
        expected = corpus_checksum(corpus)

        assert verify_corpus(corpus, f"  {expected}\n") == expected

    def test_a_mismatch_raises_and_names_both_values(self, corpus: Path) -> None:
        with pytest.raises(CorpusChecksumMismatch) as raised:
            verify_corpus(corpus, "f" * 64)

        message = str(raised.value)
        assert "f" * 64 in message, "the operator needs to know what was expected"
        assert corpus_checksum(corpus) in message, "and what was actually found"

    def test_an_empty_expected_value_skips_verification(self, corpus: Path) -> None:
        """Unconfigured means unverified, not failed. Local development and the
        demo instance own their own corpus and have nothing to compare against."""
        assert verify_corpus(corpus, "") == corpus_checksum(corpus)
        assert verify_corpus(corpus, "   ") == corpus_checksum(corpus)


class TestCrossLanguageContract:
    """The algorithm is duplicated in TypeScript. This is where that is enforced.

    If this fails after a change here, the TypeScript implementation and this one
    have diverged, and a corpus built by one will be rejected by the other. Fix
    both, or change the expected value in both suites deliberately.
    """

    EXPECTED = "2f5ed64d2a10043ec14c73eb2be41af3dbd949f3a4e282ac4adf27d4914dbbe3"

    def test_a_fixed_corpus_produces_the_agreed_digest(self, tmp_path: Path) -> None:
        (tmp_path / "one.md").write_bytes(b"# One\n\nAlpha.\n")
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested" / "two.txt").write_bytes(b"Beta.\n")

        assert corpus_checksum(tmp_path) == self.EXPECTED
