"""Synthetic capture evidence: zero network or provider credentials."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from assistant.answering import Answer, Answerer, Turn
from assistant.budget import BudgetExhausted
from assistant.capture import CaptureBudget, QualificationAllowance, capture_answers
from assistant.evaluation import AnswerOutcome, AnswerReport, Question
from assistant.persistent_budget import (
    BudgetLimits,
    PersistentBudget,
    initialize_ledger,
)
from assistant.runtime import ExecutionContext, ProviderAttemptStatus
from assistant.settings import Settings

IDENTITY = "c" * 64


def capture_budget(tmp_path: Path, *, carried: int = 0) -> CaptureBudget:
    service_path = tmp_path / "service.db"
    initialize_ledger(service_path, IDENTITY, BudgetLimits())
    allowance_path = tmp_path / "qualification.db"
    QualificationAllowance.initialize(allowance_path, IDENTITY, 1_000_000, carried)
    return CaptureBudget(
        PersistentBudget(service_path, IDENTITY, BudgetLimits()),
        QualificationAllowance(allowance_path, IDENTITY),
        10,
    )


class ScriptedAnswerer:
    calls = 0

    def __init__(self, fail_at: int | None = None) -> None:
        self.fail_at = fail_at

    def answer(
        self, question: str, history: tuple[Turn, ...], *, context: ExecutionContext
    ) -> Answer:
        self.calls += 1
        context.begin_provider("primary", "gemini-3.5-flash-lite")
        if self.calls == self.fail_at:
            context.finish_provider("uncertain")
            raise RuntimeError("synthetic failure; no network")
        context.finish_provider("completed", input_tokens=100, output_tokens=20)
        return Answer(
            "Synthetic answer.",
            (),
            False,
            (),
            input_tokens=100,
            output_tokens=20,
            stop_reason="end_turn",
        )


def test_lifetime_allowance_carries_prior_spend_and_never_resets(
    tmp_path: Path,
) -> None:
    budget = capture_budget(tmp_path, carried=960_000)
    assert budget.remaining == 1
    budget.spend()
    restarted = QualificationAllowance(budget.allowance.path, IDENTITY)
    assert restarted.remaining == 0
    with pytest.raises(BudgetExhausted):
        restarted.spend()
    with pytest.raises(FileExistsError):
        QualificationAllowance.initialize(budget.allowance.path, IDENTITY, 1_000_000, 0)
    assert budget.service.used == 1


def test_capture_preserves_attempts_and_refuses_to_overwrite(tmp_path: Path) -> None:
    budget = capture_budget(tmp_path)
    fake = ScriptedAnswerer()
    output = tmp_path / "run.json"
    report = capture_answers(
        cast(Answerer, fake),
        [Question("Q", True)],
        Settings.model_construct(),
        budget,
        output,
        {},
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["capture_state"] == "complete"
    assert (
        payload["runtime_cases"][0]["status"]["attempts"][0]["model"]
        == "gemini-3.5-flash-lite"
    )
    assert report.paid_calls == 1
    assert report.cost_usd == pytest.approx(0.00008)
    assert len(report.unreviewed) == 1
    with pytest.raises(FileExistsError):
        capture_answers(
            cast(Answerer, fake),
            [Question("Q", True)],
            Settings.model_construct(),
            budget,
            output,
            {},
        )
    assert fake.calls == 1


def test_failure_keeps_completed_case_and_uncertain_attempt(tmp_path: Path) -> None:
    budget = capture_budget(tmp_path)
    fake = ScriptedAnswerer(fail_at=2)
    output = tmp_path / "run.json"
    with pytest.raises(RuntimeError):
        capture_answers(
            cast(Answerer, fake),
            [Question("A", True), Question("B", True)],
            Settings.model_construct(),
            budget,
            output,
            {},
        )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["capture_state"] == "incomplete"
    assert payload["active_case"] == 1
    assert len(payload["raw_answering"]["outcomes"]) == 1
    assert (
        payload["runtime_cases"][1]["status"]["attempts"][0]["outcome"] == "uncertain"
    )
    assert budget.service.used == 2
    assert QualificationAllowance(budget.allowance.path, IDENTITY).remaining == 23
    budget.service.acquire_worker()()


def test_storage_failure_prevents_first_dispatch(tmp_path: Path) -> None:
    budget = capture_budget(tmp_path)
    fake = ScriptedAnswerer()

    def fail_save(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    with pytest.raises(OSError):
        capture_answers(
            cast(Answerer, fake),
            [Question("Q", True)],
            Settings.model_construct(),
            budget,
            tmp_path / "run.json",
            {},
            save=fail_save,
        )
    assert fake.calls == 0
    assert budget.service.used == 0


def test_routed_estimate_uses_each_model_and_keeps_unknown_primary_unknown() -> None:
    primary = ProviderAttemptStatus(
        "primary", "gemini-3.5-flash-lite", "completed", 1000, 100, 100.0
    )
    fallback = ProviderAttemptStatus(
        "fallback", "gpt-5.6-luna", "completed", 1000, 100, 100.0
    )
    outcome = AnswerOutcome(
        Question("Q", True), False, False, 0, "Text", attempts=(primary, fallback)
    )
    assert AnswerReport((outcome,), 2).cost_usd == pytest.approx(0.00087)
    unknown = replace(
        primary, outcome="uncertain", input_tokens=None, output_tokens=None
    )
    assert (
        AnswerReport((replace(outcome, attempts=(unknown, fallback)),), 2).cost_usd
        is None
    )


def test_allowance_bootstrap_creates_the_capture_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ledger creation is permanent, so it gets an entrypoint, not a snippet."""
    from assistant.capture import main

    path = tmp_path / "qualification.sqlite3"
    monkeypatch.setattr(
        "sys.argv",
        [
            "capture",
            "--path",
            str(path),
            "--ledger-id",
            IDENTITY,
            "--ceiling-micro-usd",
            "6000000",
            "--carried-micro-usd",
            "0",
        ],
    )

    main()

    assert QualificationAllowance(path, IDENTITY).remaining == 150


def test_allowance_bootstrap_refuses_a_ceiling_given_in_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--ceiling-micro-usd 150` is US$0.00015: no attempts, and no way back."""
    from assistant.capture import main

    path = tmp_path / "qualification.sqlite3"
    monkeypatch.setattr(
        "sys.argv",
        [
            "capture",
            "--path",
            str(path),
            "--ledger-id",
            IDENTITY,
            "--ceiling-micro-usd",
            "150",
            "--carried-micro-usd",
            "0",
        ],
    )

    with pytest.raises(SystemExit):
        main()

    assert not path.exists()
