"""Offline failure-mode tests for durable spending and cross-process admission."""

from __future__ import annotations

import multiprocessing
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from assistant.budget import BudgetExhausted
from assistant.persistent_budget import (
    ATTEMPT_RESERVATION_MICRO_USD,
    BudgetCapacityError,
    BudgetLimits,
    BudgetUnavailable,
    PersistentBudget,
    initialize_ledger,
)

IDENTITY = "a" * 64


def ledger(tmp_path: Path, limits: BudgetLimits | None = None) -> PersistentBudget:
    limits = limits or BudgetLimits()
    path = tmp_path / "budget.sqlite3"
    initialize_ledger(path, IDENTITY, limits)
    return PersistentBudget(path, IDENTITY, limits)


def test_reservations_survive_restart_and_caps_combine(tmp_path: Path) -> None:
    limits = BudgetLimits(
        daily_attempts=20,
        monthly_attempts=30,
        daily_micro_usd=80_000,
        monthly_micro_usd=120_000,
    )
    first = ledger(tmp_path, limits)
    first.spend()
    restarted = PersistentBudget(first.path, IDENTITY, limits)
    assert restarted.used == 1
    assert restarted.remaining == 1
    restarted.spend()
    with pytest.raises(BudgetExhausted):
        first.spend()
    assert restarted.used == 2


def test_independent_instances_cannot_race_past_money_limit(tmp_path: Path) -> None:
    first = ledger(tmp_path)
    instances = [
        PersistentBudget(first.path, IDENTITY, first.limits) for _ in range(20)
    ]

    def spend(budget: PersistentBudget) -> bool:
        try:
            budget.spend()
            return True
        except (BudgetExhausted, BudgetUnavailable):
            return False  # Storage contention fails closed too.

    with ThreadPoolExecutor(max_workers=20) as workers:
        outcomes = list(workers.map(spend, instances))
    assert 1 <= sum(outcomes) <= 10
    assert first.used == sum(outcomes)
    assert first.remaining == 10 - sum(outcomes)


def test_utc_rollover_keeps_monthly_reservations_and_rejects_clock_reversal(
    tmp_path: Path,
) -> None:
    first = ledger(
        tmp_path,
        BudgetLimits(
            daily_attempts=2,
            monthly_attempts=3,
            daily_micro_usd=80_000,
            monthly_micro_usd=120_000,
        ),
    )
    clock = [datetime(2026, 9, 29, 23, 59, 59, tzinfo=UTC)]
    first._now = lambda: clock[0]
    first.spend()
    first.spend()
    clock[0] = datetime(2026, 9, 30, tzinfo=UTC)
    assert first.used == 0
    assert first.remaining == 1
    first.spend()
    with pytest.raises(BudgetExhausted):
        first.spend()
    clock[0] = datetime(2026, 10, 1, tzinfo=UTC)
    assert first.remaining == 2
    first.spend()
    clock[0] = datetime(2026, 9, 30, tzinfo=UTC)
    with pytest.raises(BudgetUnavailable, match="clock_regressed"):
        first.spend()


def test_unknown_calls_are_never_refunded_and_ledger_has_no_user_content(
    tmp_path: Path,
) -> None:
    first = ledger(tmp_path)
    first.spend()  # Simulates dispatch followed by timeout/process death.
    with sqlite3.connect(first.path) as connection:
        assert (
            connection.execute(
                "SELECT day, month, micro_usd FROM reservations"
            ).fetchone()[2]
            == ATTEMPT_RESERVATION_MICRO_USD
        )
        columns = connection.execute("PRAGMA table_info(reservations)").fetchall()
    assert [column[1] for column in columns] == ["id", "day", "month", "micro_usd"]
    assert PersistentBudget(first.path, IDENTITY, first.limits).used == 1


def test_runtime_never_bootstraps_missing_or_replaced_ledger(tmp_path: Path) -> None:
    missing = tmp_path / "absent.sqlite3"
    with pytest.raises((BudgetUnavailable, FileNotFoundError)):
        PersistentBudget(missing, IDENTITY, BudgetLimits())
    assert not missing.exists()
    first = ledger(tmp_path)
    first.path.rename(tmp_path / "old.sqlite3")
    with pytest.raises(BudgetUnavailable):
        first.spend()
    assert not first.path.exists()


def test_wrong_identity_changed_limits_and_partial_initialization_fail_closed(
    tmp_path: Path,
) -> None:
    first = ledger(tmp_path)
    with pytest.raises(BudgetUnavailable):
        PersistentBudget(first.path, "b" * 64, first.limits)
    with pytest.raises(BudgetUnavailable):
        PersistentBudget(first.path, IDENTITY, BudgetLimits(monthly_attempts=199))
    with pytest.raises(ValueError, match="already exist"):
        initialize_ledger(first.path, IDENTITY, first.limits)


def test_corrupt_accounting_and_unwritable_storage_prevent_reservation(
    tmp_path: Path,
) -> None:
    first = ledger(tmp_path)
    first.spend()
    with sqlite3.connect(first.path) as connection:
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute("UPDATE reservations SET micro_usd=1")
    with pytest.raises(BudgetUnavailable, match="accounting"):
        first.spend()


def test_shared_worker_is_held_until_actual_release(tmp_path: Path) -> None:
    first = ledger(tmp_path)
    second = PersistentBudget(first.path, IDENTITY, first.limits)
    release = first.acquire_worker()
    try:
        with pytest.raises(BudgetCapacityError):
            second.acquire_worker()
        first.spend()  # Worker admission does not lock out accounting.
    finally:
        release()
    second.acquire_worker()()
    assert second.used == 1


def _hold_worker(path: str, ready: object) -> None:
    budget = PersistentBudget(Path(path), IDENTITY, BudgetLimits())
    release = budget.acquire_worker()
    budget.spend()
    ready.set()  # type: ignore[attr-defined]
    import time

    try:
        time.sleep(60)
    finally:
        release()


def test_process_death_releases_worker_but_preserves_paid_attempt(
    tmp_path: Path,
) -> None:
    first = ledger(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    process = ctx.Process(target=_hold_worker, args=(str(first.path), ready))
    process.start()
    try:
        assert ready.wait(10)
        with pytest.raises(BudgetCapacityError):
            first.acquire_worker()
    finally:
        process.terminate()
        process.join(10)
    first.acquire_worker()()
    assert first.used == 1


def test_failed_storage_write_cannot_authorize_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from assistant.runtime import ExecutionContext

    first = ledger(tmp_path)

    def read_only(path: Path, *, lock: bool = False) -> sqlite3.Connection:
        return sqlite3.connect(
            path.as_uri() + "?mode=ro", uri=True, isolation_level=None
        )

    with monkeypatch.context() as patch:
        patch.setattr(first, "_connect", read_only)
        context = ExecutionContext(time.monotonic() + 5, first)
        with pytest.raises(BudgetUnavailable):
            context.begin_provider("primary", "gemini-3.5-flash-lite")
        assert not context.status().provider_attempted
    assert first.used == 0


def test_cancelled_caller_does_not_release_another_process_worker(
    tmp_path: Path,
) -> None:
    import threading
    import time

    from assistant.runtime import BoundedAnswerExecutor, ExecutionContext

    first = ledger(tmp_path)
    second = PersistentBudget(first.path, IDENTITY, first.limits)
    entered, finish = threading.Event(), threading.Event()

    def blocked(context: ExecutionContext) -> str:
        context.begin_provider("primary", "gemini-3.5-flash-lite")
        entered.set()
        assert finish.wait(5)
        context.finish_provider("uncertain")
        return "done"

    one: BoundedAnswerExecutor[str] = BoundedAnswerExecutor(1, first.acquire_worker)
    two: BoundedAnswerExecutor[str] = BoundedAnswerExecutor(1, second.acquire_worker)
    context = ExecutionContext(time.monotonic() + 10, first)
    try:
        future = one.submit(blocked, context)
        assert entered.wait(5)
        context.cancel()
        assert not future.cancel()
        with pytest.raises(BudgetCapacityError):
            two.submit(
                lambda ctx: "unexpected", ExecutionContext(time.monotonic() + 5, second)
            )
        finish.set()
        assert future.result(timeout=5) == "done"
        # Callback cleanup completes as part of orderly shutdown.
        one.shutdown()
        assert (
            two.submit(
                lambda ctx: "admitted", ExecutionContext(time.monotonic() + 5, second)
            ).result(timeout=5)
            == "admitted"
        )
        assert second.used == 1
    finally:
        finish.set()
        one.shutdown()
        two.shutdown()


@pytest.mark.parametrize("machine", ["", "replica"])
def test_fly_replica_cannot_open_an_independent_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: str
) -> None:
    from assistant.persistent_budget import service_budget
    from assistant.settings import Settings

    first = ledger(tmp_path)
    monkeypatch.setenv("FLY_APP_NAME", "synthetic-app")
    monkeypatch.setenv("FLY_MACHINE_ID", machine)
    settings = Settings(
        budget_path=first.path,
        budget_ledger_id=IDENTITY,
        budget_machine_id="approved-machine",
    )
    with pytest.raises(BudgetUnavailable, match="deployment_or_mount"):
        service_budget(settings)
    assert first.used == 0


def test_bootstrap_requires_conservative_prior_spend_and_never_refunds(
    tmp_path: Path,
) -> None:
    path = tmp_path / "budget.db"
    initialize_ledger(path, IDENTITY, BudgetLimits(), carried_attempts=7)
    budget = PersistentBudget(path, IDENTITY, BudgetLimits())
    assert budget.used == 7
    assert budget.remaining == 3
    budget.spend()
    assert PersistentBudget(path, IDENTITY, BudgetLimits()).used == 8


@pytest.mark.parametrize(
    "day, month",
    [("2026-02-31", "2026-02"), ("garbage", "2026-09"), ("2026-09-01", "")],
)
def test_malformed_dates_cannot_hide_a_reservation(
    tmp_path: Path, day: str, month: str
) -> None:
    first = ledger(tmp_path)
    first.spend()
    with sqlite3.connect(first.path) as connection:
        connection.execute("UPDATE reservations SET day=?, month=?", (day, month))
    with pytest.raises(BudgetUnavailable, match="accounting"):
        first.spend()


def test_bootstrap_defaults_keep_the_runbook_command_meaning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented command must still stamp the live service envelope."""
    from assistant.persistent_budget import main

    path = tmp_path / "budget.sqlite3"
    monkeypatch.setattr(
        "sys.argv",
        [
            "persistent_budget",
            "--path",
            str(path),
            "--ledger-id",
            IDENTITY,
            "--carry-forward-attempts",
            "0",
        ],
    )

    main()

    PersistentBudget(path, IDENTITY, BudgetLimits())


def test_bootstrap_flags_stamp_a_capture_scoped_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without these flags the capture's service ledger cannot be created at all.

    `PersistentBudget` refuses on any mismatch between the stored limits and the
    ones it is constructed with, and the ledger may not be recreated to correct
    them, so the limits must be settable at bootstrap.
    """
    from assistant.persistent_budget import main

    path = tmp_path / "capture.sqlite3"
    capture = BudgetLimits(150, 150, 6_000_000, 6_000_000)
    monkeypatch.setattr(
        "sys.argv",
        [
            "persistent_budget",
            "--path",
            str(path),
            "--ledger-id",
            IDENTITY,
            "--carry-forward-attempts",
            "0",
            "--daily-attempts",
            "150",
            "--monthly-attempts",
            "150",
            "--daily-micro-usd",
            "6000000",
            "--monthly-micro-usd",
            "6000000",
        ],
    )

    main()

    assert PersistentBudget(path, IDENTITY, capture).remaining == 150
    with pytest.raises(BudgetUnavailable, match="budget_identity_or_limits"):
        PersistentBudget(path, IDENTITY, BudgetLimits())
