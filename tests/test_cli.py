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


class TestInspect:
    def test_it_discovers_cited_and_deployment_corpora_and_binds_loopback(
        self,
        corpus: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        deployed = corpus.parent / "deploy" / "oj-assistant" / "content"
        deployed.mkdir(parents=True)
        (deployed / "profile.md").write_text(
            "# Profile\n\n" + "Portfolio evidence sentence. " * 30,
            encoding="utf-8",
        )
        called: dict[str, object] = {}

        def fake_run(app: object, **options: object) -> None:
            called.update(app=app, **options)

        monkeypatch.setattr("uvicorn.run", fake_run)

        result = cli.main(["--corpus", str(corpus), "inspect", "--port", "9876"])

        assert result == 0
        assert called["host"] == "127.0.0.1"
        assert called["port"] == 9876
        output = capsys.readouterr().out
        assert "Cited, OJ Assistant" in output
        assert "Read-only" in output

    def test_explicit_profiles_support_two_corpora(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        second = tmp_path / "other"
        second.mkdir()
        (second / "other.md").write_text(
            "# Other\n\n" + "Another documented sentence. " * 30,
            encoding="utf-8",
        )
        called: dict[str, object] = {}
        monkeypatch.setattr(
            "uvicorn.run", lambda app, **options: called.update(app=app, **options)
        )

        code = cli.main(
            [
                "inspect",
                "--corpus-profile",
                f"Cited={corpus}",
                "--corpus-profile",
                f"OJ Assistant={second}",
            ]
        )

        assert code == 0
        assert called["host"] == "127.0.0.1"


class TestEvalIsFreeUnlessPaidIsRequested:
    """The command-line guarantee: a key alone can never start spending.

    These are the regression tests for a real incident. A command intended as a
    dry run was executed while `ANTHROPIC_API_KEY` was set in a `.env` file; the
    old code took "a key is configured" to mean "run the paid half", and made 48
    billable calls. Nothing here relies on a key being absent, because in the
    incident it was present.
    """

    def corpus_and_questions(self, tmp_path: Path) -> tuple[Path, Path]:
        corpus = tmp_path / "content"
        corpus.mkdir()
        (corpus / "doc.md").write_text(
            "# Doc\n\n## Wanted\n\n" + ("Body text about prompts. " * 30),
            encoding="utf-8",
        )
        questions = tmp_path / "q.toml"
        questions.write_text(
            '[[question]]\ntext = "What is this about?"\n'
            'expects = "Wanted"\nanswerable = true\n',
            encoding="utf-8",
        )
        return corpus, questions

    def test_a_key_alone_does_not_trigger_a_paid_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The incident, as a test."""
        corpus, questions = self.corpus_and_questions(tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")

        # Any attempt to construct a provider client fails the test outright,
        # rather than being detected afterwards by counting calls.
        def explode(*args: object, **kwargs: object) -> object:
            raise AssertionError("a provider client was constructed without --paid")

        monkeypatch.setattr("assistant.answering.build_client", explode)

        code = cli.main(
            ["--corpus", str(corpus), "eval", "--questions", str(questions)]
        )

        assert code == 0
        out = capsys.readouterr().out
        assert "no provider call was made" in out

    def test_paid_without_a_call_ceiling_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spending authority is granted as a number, so the number is required."""
        corpus, questions = self.corpus_and_questions(tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
        monkeypatch.setattr(
            "assistant.answering.build_client",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("client built despite a missing ceiling")
            ),
        )

        code = cli.main(
            ["--corpus", str(corpus), "eval", "--questions", str(questions), "--paid"]
        )

        assert code == 2

    def test_paid_without_a_key_exits_rather_than_pretending_to_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus, questions = self.corpus_and_questions(tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        code = cli.main(
            [
                "--corpus",
                str(corpus),
                "eval",
                "--questions",
                str(questions),
                "--paid",
                "--max-paid-calls",
                "10",
            ]
        )

        assert code == 2

    def test_a_run_needing_more_calls_than_authorised_never_starts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Refused during preflight, before the first call rather than at the
        ceiling — so an over-large run costs nothing at all."""
        corpus, questions = self.corpus_and_questions(tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
        monkeypatch.setattr(
            "assistant.answering.build_client",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("client built despite an insufficient ceiling")
            ),
        )

        code = cli.main(
            [
                "--corpus",
                str(corpus),
                "eval",
                "--questions",
                str(questions),
                "--paid",
                "--max-paid-calls",
                "0",
            ]
        )

        assert code == 2
        assert "No calls were made" in capsys.readouterr().err

    def test_the_preflight_never_prints_the_key(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        corpus, questions = self.corpus_and_questions(tmp_path)
        secret = "sk-ant-verysecretvalue12345"
        monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
        monkeypatch.setattr(
            "assistant.answering.build_client",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call expected")),
        )

        cli.main(
            [
                "--corpus",
                str(corpus),
                "eval",
                "--questions",
                str(questions),
                "--paid",
                "--max-paid-calls",
                "0",
            ]
        )

        captured = capsys.readouterr()
        assert secret not in captured.out
        assert secret not in captured.err
        assert "configured (value not shown)" in captured.out
