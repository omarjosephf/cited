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

import json
from pathlib import Path

import pytest

from assistant import cli
from assistant.release_manifest import ANSWER_RUNTIME_SOURCES, AnswerConfiguration
from assistant.settings import Settings


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
    def test_it_refuses_without_the_complete_pair_and_says_what_still_works(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("OPENAI_PROJECT_ID", "")
        monkeypatch.setenv("OPENAI_ACCOUNT_VERIFIED", "false")
        monkeypatch.setenv("ENABLE_FALLBACK", "false")

        code = cli.main(["ask", "a question"])

        assert code == 2
        error = capsys.readouterr().err
        assert "complete verified primary and fallback configuration" in error
        assert "index" in error and "eval" in error


class TestInspect:
    def test_windows_launcher_uses_this_checkout_and_stays_loopback_only(
        self,
    ) -> None:
        launcher = Path(__file__).parents[1] / "Start-RAG-Management-Panel.cmd"
        contents = launcher.read_text(encoding="utf-8")
        executable_lines = [
            line.strip()
            for line in contents.splitlines()
            if line.strip() and not line.strip().casefold().startswith("echo ")
        ]

        assert 'cd /d "%~dp0"' in contents
        assert 'set "PYTHONPATH=%~dp0src"' in contents
        assert "http://127.0.0.1:8765/" in contents
        assert '"%PYTHON_EXE%" -m assistant.cli inspect' in contents
        assert "--host" not in contents
        assert not any("pip install" in line for line in executable_lines)

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
    """The command-line guarantee: configuration alone cannot start spending.

    These are the regression tests for a real incident. A command intended as a
    dry run was executed while `ANTHROPIC_API_KEY` was set in a `.env` file; the
    old code took "a key is configured" to mean "run the paid half", and made 48
    billable calls. Paid multi-provider capture is now disabled before any client
    construction; these tests preserve that boundary.
    """

    @pytest.fixture(autouse=True)
    def deterministic_retrieval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from assistant.chunking import Chunk
        from assistant.retrieval import SearchResult

        class FixedRetriever:
            def search(self, text: str, top_k: int = 4) -> list[SearchResult]:
                return [
                    SearchResult(Chunk("Body text.", "doc.md", None, "Wanted", 0), 0.9)
                ]

        monkeypatch.setattr(cli, "_build_retriever", lambda path: FixedRetriever())

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
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

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

    def test_free_capture_records_v3_answer_runtime_identity(
        self, tmp_path: Path
    ) -> None:
        corpus, questions = self.corpus_and_questions(tmp_path)
        output = tmp_path / "identity.json"

        code = cli.main(
            [
                "--corpus",
                str(corpus),
                "eval",
                "--questions",
                str(questions),
                "--output",
                str(output),
            ]
        )

        assert code == 0
        captured = json.loads(output.read_text(encoding="utf-8"))
        assert captured["answer_contract_version"] == 3
        assert tuple(captured["answer_runtime_sha256"]) == ANSWER_RUNTIME_SOURCES
        assert all(
            len(value) == 64 for value in captured["answer_runtime_sha256"].values()
        )
        # The record that used to exist only past the confirmation prompt.
        assert AnswerConfiguration.model_validate(captured["config"])

    def test_the_free_path_records_the_identity_the_paid_path_will_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The capture envelope must not reach the evidence identity.

        `PersistentBudget` pins the capture's settings to its ledger's stamped
        limits, so a capture against the 150-attempt ledger runs with these
        values set. If any of them still reached `AnswerConfiguration`, the
        free rehearsal and the paid capture would record different identities
        and only the paid one would ever find out.
        """
        bare = Settings(_env_file=None, retrieval_top_k=4)  # type: ignore[call-arg]
        live = cli._answer_configuration(bare, 4)

        for name, value in (
            ("DAILY_ANSWER_LIMIT", "150"),
            ("MONTHLY_ANSWER_LIMIT", "150"),
            ("DAILY_BUDGET_MICRO_USD", "6000000"),
            ("MONTHLY_BUDGET_MICRO_USD", "6000000"),
        ):
            monkeypatch.setenv(name, value)
        widened = Settings(_env_file=None, retrieval_top_k=4)  # type: ignore[call-arg]
        capture = cli._answer_configuration(widened, 4)

        assert capture == live
        assert not {
            "daily_attempt_limit",
            "monthly_attempt_limit",
            "daily_budget_micro_usd",
            "monthly_budget_micro_usd",
        } & set(capture)

    def test_answer_configuration_records_the_routed_behavior(self) -> None:
        settings = Settings.model_construct(
            answer_model="gemini-3.5-flash-lite",
            fallback_answer_model="gpt-5.6-luna",
            answer_effort="none",
            answer_max_tokens=1024,
            prefilter_score=0.45,
            backend_timeout_seconds=8.0,
            provider_timeout_seconds=6.0,
            primary_timeout_seconds=3.0,
            validation_margin_seconds=0.5,
        )

        assert cli._answer_configuration(settings, 4) == {
            "primary_model": "gemini-3.5-flash-lite",
            "fallback_model": "gpt-5.6-luna",
            "answer_effort": "none",
            "answer_max_tokens": 1024,
            "top_k": 4,
            "prefilter_score": 0.45,
            "backend_timeout_seconds": 8.0,
            "provider_timeout_seconds": 6.0,
            "primary_timeout_seconds": 3.0,
            "validation_margin_seconds": 0.5,
            "wire_version": 3,
            "max_attempts": 2,
            "retries": 0,
            "fallback_mode": "availability_only",
            "complete_pair_required": True,
            "max_provider_request_bytes": 32000,
            "attempt_reservation_micro_usd": 40000,
            "shared_worker_limit": 1,
            "budget_storage": "persistent_local_sqlite",
        }

    def test_paid_capture_is_disabled_without_constructing_a_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus, questions = self.corpus_and_questions(tmp_path)
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
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

    def test_paid_capture_stays_disabled_without_provider_configuration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus, questions = self.corpus_and_questions(tmp_path)
        monkeypatch.setenv("OPENAI_API_KEY", "")

        code = cli.main(
            [
                "--corpus",
                str(corpus),
                "eval",
                "--questions",
                str(questions),
                "--paid",
                "--spec-version",
                "3.0",
                "--output",
                str(tmp_path / "run.json"),
                "--max-paid-calls",
                "10",
            ]
        )

        assert code == 2

    def test_disabled_capture_does_not_reach_the_legacy_call_ceiling(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        corpus, questions = self.corpus_and_questions(tmp_path)
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
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
                "--spec-version",
                "3.0",
                "--output",
                str(tmp_path / "run.json"),
                "--max-paid-calls",
                "0",
            ]
        )

        assert code == 2
        assert "Paid evaluation capture is disabled" in capsys.readouterr().err

    def test_disabled_capture_never_prints_an_environment_key(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        corpus, questions = self.corpus_and_questions(tmp_path)
        secret = "synthetic-secret-value-12345"
        monkeypatch.setenv("OPENAI_API_KEY", secret)
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
                "--spec-version",
                "3.0",
                "--output",
                str(tmp_path / "run.json"),
                "--max-paid-calls",
                "0",
            ]
        )

        captured = capsys.readouterr()
        assert secret not in captured.out
        assert secret not in captured.err
        assert "Paid evaluation capture is disabled" in captured.err
