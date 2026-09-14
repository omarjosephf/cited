"""Private, interrupted-run-safe capture with a non-renewing allowance.

This module never initializes spending authority implicitly. An operator must
carry earlier reservations into the separately initialized allowance. No usage
report settles or replenishes either ledger. Only frozen evaluation questions
are written here; production visitor requests never use this capture path.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing, suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

from assistant.answering import Answer, Answerer
from assistant.budget import BudgetExhausted
from assistant.evaluation import AnswerOutcome, AnswerReport, Question
from assistant.persistent_budget import (
    ATTEMPT_RESERVATION_MICRO_USD,
    BudgetUnavailable,
    PersistentBudget,
)
from assistant.runtime import BoundedAnswerExecutor, ExecutionContext
from assistant.settings import Settings


class QualificationAllowance:
    """Lifetime ceiling shared by captures; no daily/monthly rollover or refunds."""

    def __init__(self, path: Path, ledger_id: str) -> None:
        self.path = path.resolve(strict=True)
        self.ledger_id = ledger_id
        self._identity = (self.path.stat().st_dev, self.path.stat().st_ino)
        _ = self.remaining  # Validate before constructing any provider client.

    @staticmethod
    def initialize(path: Path, ledger_id: str, ceiling: int, carried: int) -> None:
        import re

        if (
            not re.fullmatch(r"[0-9a-f]{64}", ledger_id)
            or type(ceiling) is not int
            or type(carried) is not int
            or not 0 <= carried <= ceiling
            or ceiling < 1
        ):
            raise ValueError("invalid qualification allowance")
        with path.open("xb"):
            pass
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                "CREATE TABLE allowance (identity TEXT NOT NULL, "
                "ceiling INTEGER NOT NULL, "
                "carried INTEGER NOT NULL) STRICT"
            )
            connection.execute(
                "INSERT INTO allowance VALUES (?, ?, ?)", (ledger_id, ceiling, carried)
            )
            connection.execute(
                "CREATE TABLE reservations (id INTEGER PRIMARY KEY) STRICT"
            )
            connection.commit()
        with path.open("r+b") as file:
            os.fsync(file.fileno())

    def _remaining(self, reserve: bool) -> int:
        try:
            stat = self.path.stat()
            if (stat.st_dev, stat.st_ino) != self._identity:
                raise BudgetUnavailable("qualification_ledger_replaced")
            with closing(
                sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=0.1)
            ) as connection:
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("BEGIN IMMEDIATE")
                rows = connection.execute(
                    "SELECT identity, ceiling, carried FROM allowance"
                ).fetchall()
                if (
                    len(rows) != 1
                    or rows[0][0] != self.ledger_id
                    or type(rows[0][1]) is not int
                    or type(rows[0][2]) is not int
                    or not 0 <= rows[0][2] <= rows[0][1]
                ):
                    raise BudgetUnavailable("qualification_accounting")
                count = connection.execute(
                    "SELECT COUNT(*) FROM reservations"
                ).fetchone()[0]
                remaining = (
                    rows[0][1] - rows[0][2]
                ) // ATTEMPT_RESERVATION_MICRO_USD - count
                if reserve:
                    if remaining < 1:
                        raise BudgetExhausted("qualification_allowance_exhausted")
                    connection.execute("INSERT INTO reservations DEFAULT VALUES")
                connection.commit()
                return max(0, int(remaining))
        except (OSError, sqlite3.Error):
            raise BudgetUnavailable("qualification_storage") from None

    @property
    def remaining(self) -> int:
        return self._remaining(False)

    def spend(self) -> None:
        self._remaining(True)


class CaptureBudget:
    """Each attempt must fit lifetime, per-run and actual service limits."""

    def __init__(
        self, service: PersistentBudget, allowance: QualificationAllowance, maximum: int
    ) -> None:
        if type(maximum) is not int or maximum < 1:
            raise ValueError("capture requires a positive attempt ceiling")
        self.service = service
        self.allowance = allowance
        self.maximum = maximum
        self.used = 0

    @property
    def remaining(self) -> int:
        return min(
            self.maximum - self.used, self.service.remaining, self.allowance.remaining
        )

    def spend(self) -> None:
        if self.used >= self.maximum:
            raise BudgetExhausted("capture_attempt_limit")
        # Deliberately conservative if the second commit fails. Never undo the
        # first reservation when a later stage cannot establish the full state.
        self.allowance.spend()
        self.service.spend()
        self.used += 1


def save_capture(path: Path, payload: dict[str, Any], *, first: bool = False) -> None:
    """Exclusive creation, then durable replacement; never truncate prior evidence."""
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    if first:
        with path.open("xb") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
    else:
        temporary = path.with_name(path.name + ".pending")
        with temporary.open("xb") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    if os.name == "posix":
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def capture_answers(
    answerer: Answerer,
    questions: list[Question],
    settings: Settings,
    budget: CaptureBudget,
    path: Path,
    identity: dict[str, Any],
    *,
    save: Callable[..., None] = save_capture,
) -> AnswerReport:
    """Capture actual runtime jobs serially. Never auto-retry an interrupted case."""
    outcomes: list[AnswerOutcome] = []
    payload = {
        **identity,
        "capture_state": "incomplete",
        "active_case": None,
        "raw_answering": asdict(AnswerReport(())),
        "runtime_cases": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save(path, payload, first=True)  # Storage and overwrite errors precede any call.
    executor: BoundedAnswerExecutor[Any] = BoundedAnswerExecutor(
        1, acquire_shared_worker=budget.service.acquire_worker
    )
    try:
        for index, question in enumerate(questions):
            payload["active_case"] = index
            save(path, payload)
            context = ExecutionContext(
                time.monotonic() + settings.backend_timeout_seconds, budget
            )
            started = time.monotonic()
            try:

                def answer_case(
                    ctx: ExecutionContext, q: Question = question
                ) -> Answer:
                    return answerer.answer(q.text, q.history, context=ctx)

                future = executor.submit(answer_case, context)
                try:
                    answer = future.result(
                        timeout=max(0.0, context.deadline - time.monotonic())
                    )
                except TimeoutError:
                    context.cancel()
                    # Keep admission until actual work exits; no next case may overlap.
                    with suppress(Exception):
                        future.result()
                    raise
                cited = question.expects is not None and any(
                    citation.source == question.expects
                    or citation.source.endswith(" — " + question.expects)
                    for citation in answer.citations
                )
                outcomes.append(
                    AnswerOutcome(
                        question=question,
                        grounded=answer.grounded,
                        cited_expected=cited,
                        rejected_citations=answer.rejected_citations,
                        text=answer.text,
                        accepted_citations=len(answer.citations),
                        input_tokens=answer.input_tokens,
                        output_tokens=answer.output_tokens,
                        stop_reason=answer.stop_reason,
                        policy=answer.policy,
                        refused=answer.refused,
                        citations=answer.citations,
                        evidence=answer.results,
                        attempts=context.status().attempts,
                    )
                )
            finally:
                payload["runtime_cases"].append(
                    {
                        "index": index,
                        "duration_ms": (time.monotonic() - started) * 1000,
                        "status": asdict(context.status()),
                    }
                )
                payload["raw_answering"] = asdict(
                    AnswerReport(tuple(outcomes), budget.used)
                )
                save(path, payload)
        report = AnswerReport(tuple(outcomes), budget.used)
        payload["capture_state"] = "complete"
        payload["active_case"] = None
        payload["usage_estimate_usd"] = report.cost_usd
        payload["usage_note"] = (
            "Uncached token-use estimate, not an invoice or reservation settlement."
        )
        save(path, payload)
        return report
    finally:
        executor.shutdown(wait=True)


def main() -> None:
    """Bootstrap only through a deliberate operator command; never starts inference.

    The allowance ledger is permanent: the runbook forbids recreating one to
    regain authority, so a mistyped ceiling cannot be corrected afterwards.
    Before this entrypoint existed the only way to create it was an improvised
    snippet, which is the class of mistake that has no undo.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Initialize a new non-renewing qualification allowance ledger. "
            "Permanent: it cannot be recreated later to regain allowance."
        )
    )
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--ledger-id", required=True)
    parser.add_argument(
        "--ceiling-micro-usd",
        type=int,
        required=True,
        help=(
            "lifetime ceiling in micro-USD, NOT a number of attempts; one "
            f"attempt reserves {ATTEMPT_RESERVATION_MICRO_USD} micro-USD"
        ),
    )
    parser.add_argument(
        "--carried-micro-usd",
        type=int,
        required=True,
        help=(
            "reviewed prior qualification spend in micro-USD, deducted from the "
            "ceiling; use 0 only when the prior allowance is genuinely unused"
        ),
    )
    args = parser.parse_args()
    # Both operands are money. A ceiling given in attempts by mistake reads as a
    # few micro-USD and silently yields an unusable ledger that cannot be
    # replaced, so refuse it here rather than at the first capture attempt.
    attempts = (
        args.ceiling_micro_usd - args.carried_micro_usd
    ) // ATTEMPT_RESERVATION_MICRO_USD
    if attempts < 1:
        parser.error(
            f"a ceiling of {args.ceiling_micro_usd} micro-USD less "
            f"{args.carried_micro_usd} carried leaves room for no attempts at "
            f"{ATTEMPT_RESERVATION_MICRO_USD} micro-USD each. Both flags are "
            f"money, not attempt counts: 150 attempts is "
            f"{150 * ATTEMPT_RESERVATION_MICRO_USD}. No ledger was created."
        )
    print(
        f"About to create a permanent allowance ledger at {args.path}: "
        f"ceiling {args.ceiling_micro_usd} micro-USD "
        f"(US${args.ceiling_micro_usd / 1_000_000:.2f}), carried "
        f"{args.carried_micro_usd} micro-USD "
        f"(US${args.carried_micro_usd / 1_000_000:.2f}), leaving {attempts} "
        f"attempts at {ATTEMPT_RESERVATION_MICRO_USD} micro-USD each."
    )
    QualificationAllowance.initialize(
        args.path, args.ledger_id, args.ceiling_micro_usd, args.carried_micro_usd
    )
    print("Qualification allowance initialized. No provider request was made.")


if __name__ == "__main__":
    main()
