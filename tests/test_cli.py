"""Tests for the command line interface.

The CLI shipped broken once already — `pyproject.toml` declared an entry point
to a module that did not exist, so `pip install -e .` produced a command that
died on import. Nothing caught it because nothing exercised it. These tests
exist so that cannot recur.

The expensive path (building a real retriever, which loads the embedding model)
is deliberately avoided; what is tested here is argument handling, output, and
exit codes, which is where CLI bugs actually live.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant import cli


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    directory = tmp_path / "content"
    directory.mkdir()
    (directory / "guide.md").write_text(
        "# Topic One\n\nFirst body paragraph about a topic.\n\n"
        "# Topic Two\n\nSecond body paragraph about another topic.\n",
        encoding="utf-8",
    )
    return directory


class TestEntryPoint:
    def test_the_declared_console_script_is_importable(self) -> None:
        """`pyproject.toml` names `assistant.cli:main`.

        The previous entry point pointed at a module that did not exist. This
        asserts the target of the declaration actually resolves.
        """
        assert callable(cli.main)

    def test_no_subcommand_is_an_error_rather_than_a_crash(self) -> None:
        with pytest.raises(SystemExit) as exit_info:
            cli.main([])
        assert exit_info.value.code == 2

    def test_an_unknown_subcommand_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            cli.main(["nonsense"])


class TestIndex:
    def test_it_reports_what_retrieval_will_see(
        self, corpus: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["--corpus", str(corpus), "index"]) == 0

        output = capsys.readouterr().out
        assert "guide.md" in output
        assert "chunks" in output
        assert "words" in output

    def test_verbose_lists_every_chunk_with_its_citation(
        self, corpus: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main(["--corpus", str(corpus), "index", "--verbose"])

        output = capsys.readouterr().out
        assert "guide.md — Topic One" in output
        assert "guide.md — Topic Two" in output

    def test_an_empty_corpus_exits_non_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        assert cli.main(["--corpus", str(empty), "index"]) == 1
        assert "No documents" in capsys.readouterr().out


class TestEval:
    def test_an_invalid_question_set_exits_two_without_building_an_index(
        self, corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Validation must happen before the expensive work.

        Loading the embedding model and then failing on a typo in the question
        set would waste the user's time for no reason.
        """
        broken = tmp_path / "broken.toml"
        broken.write_text(
            '[[question]]\ntext = "Q"\nanswerable = true\n', encoding="utf-8"
        )

        code = cli.main(["--corpus", str(corpus), "eval", "--questions", str(broken)])

        assert code == 2
        assert "invalid" in capsys.readouterr().err.lower()

    def test_a_missing_question_set_exits_two(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        assert (
            cli.main(
                [
                    "--corpus",
                    str(corpus),
                    "eval",
                    "--questions",
                    str(tmp_path / "absent.toml"),
                ]
            )
            == 2
        )


class TestAsk:
    def test_it_refuses_to_run_without_a_key_and_says_what_still_works(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        code = cli.main(["ask", "a question"])

        assert code == 2
        error = capsys.readouterr().err
        assert "ANTHROPIC_API_KEY" in error
        # An error that only says "no" is worse than one that says what to do.
        assert ".env" in error
        assert "index" in error and "eval" in error
