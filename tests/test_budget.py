"""Tests for the spend ceiling.

This is the control that makes a public, unauthenticated, paid endpoint
survivable. Its failure mode is money, so the tests are about the ways a naive
implementation leaks it: reserving too late, refunding too much, and forgetting
that concurrent requests exist.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest

from assistant.budget import BudgetExhausted, DailyCallBudget


def test_spending_within_the_limit_is_allowed() -> None:
    budget = DailyCallBudget(limit=3)
    for _ in range(3):
        budget.spend()
    assert budget.used == 3
    assert budget.remaining == 0


def test_exceeding_the_limit_raises_rather_than_spending() -> None:
    budget = DailyCallBudget(limit=2)
    budget.spend()
    budget.spend()

    with pytest.raises(BudgetExhausted, match="daily limit of 2"):
        budget.spend()

    # The rejected call must not have been counted, or the number reported to
    # an operator drifts upward every time someone retries.
    assert budget.used == 2


def test_a_refund_returns_the_reservation() -> None:
    """A failed provider call must not consume budget.

    Without this, an outage burns the day's allowance without answering
    anything — the worst of both outcomes.
    """
    budget = DailyCallBudget(limit=1)
    budget.spend()
    budget.refund()

    budget.spend()  # must not raise
    assert budget.used == 1


def test_refunding_more_than_was_spent_cannot_create_credit() -> None:
    budget = DailyCallBudget(limit=5)
    budget.spend()
    budget.refund()
    budget.refund()
    budget.refund()

    assert budget.used == 0, "refunds must not push the counter below zero"
    assert budget.remaining == 5


def test_the_counter_resets_on_a_new_utc_day(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = DailyCallBudget(limit=1)
    budget.spend()

    with pytest.raises(BudgetExhausted):
        budget.spend()

    # Advance the clock a day. UTC deliberately: a local-time reset moves twice
    # a year, producing one 23-hour and one 25-hour "day".
    tomorrow = datetime.now(UTC) + timedelta(days=1)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:  # type: ignore[override]
            return tomorrow

    monkeypatch.setattr("assistant.budget.datetime", FrozenDatetime)

    budget.spend()  # must not raise
    assert budget.used == 1


def test_concurrent_spending_never_exceeds_the_limit() -> None:
    """The failure this exists to prevent.

    Check-then-increment without a lock lets concurrent callers all pass the
    check before any of them increments, so the ceiling is breached. Measured
    against a deliberately unlocked implementation with a thread switch forced
    between the two steps: **76 calls granted against a limit of 50**, 52% over.

    A caveat worth recording, because it would otherwise be rediscovered as a
    surprise: simply removing the lock from `DailyCallBudget` does *not* make
    this test fail on CPython. The GIL keeps the window between the check and
    the increment narrow enough that 20 consecutive runs never lost the race.
    That is a fact about bytecode scheduling on one interpreter, not evidence
    the lock is unnecessary — the ordering is unsafe regardless, and a free-
    threaded build or any I/O between the steps widens the window immediately.

    So this test guards the invariant; it does not prove the mechanism. Do not
    read it passing without a lock as permission to remove one.
    """
    limit = 50
    budget = DailyCallBudget(limit=limit)
    granted = 0
    granted_lock = threading.Lock()
    start = threading.Event()

    def attempt() -> None:
        nonlocal granted
        start.wait()  # release all threads together to maximise contention
        try:
            budget.spend()
        except BudgetExhausted:
            return
        with granted_lock:
            granted += 1

    threads = [threading.Thread(target=attempt) for _ in range(200)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join()

    assert granted == limit, f"granted {granted} calls against a limit of {limit}"
    assert budget.used == limit


def test_a_zero_limit_refuses_everything() -> None:
    """The operator's off switch: answering can be disabled without redeploying."""
    budget = DailyCallBudget(limit=0)
    with pytest.raises(BudgetExhausted):
        budget.spend()
