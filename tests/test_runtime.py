"""Deterministic tests for bounded synchronous answer execution."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future

import pytest

from assistant.budget import BudgetExhausted, DailyCallBudget
from assistant.runtime import (
    BoundedAnswerExecutor,
    ExecutionCancelled,
    ExecutionContext,
    ExecutionDeadlineExceeded,
    ExecutorCapacityError,
    ExecutorClosedError,
    ProviderAttemptAlreadyBegun,
)


def context(*, budget: DailyCallBudget | None = None) -> ExecutionContext:
    return ExecutionContext(
        deadline=time.monotonic() + 30.0,
        budget=budget or DailyCallBudget(limit=100),
    )


def test_provider_reservation_is_single_and_conservative() -> None:
    budget = DailyCallBudget(limit=1)
    execution = context(budget=budget)

    execution.begin_provider()
    assert budget.used == 1
    assert execution.status().provider_outcome == "in_progress"

    with pytest.raises(ProviderAttemptAlreadyBegun):
        execution.begin_provider()
    execution.finish_provider("uncertain")

    status = execution.status()
    assert status.provider_attempted is True
    assert status.provider_outcome == "uncertain"
    assert status.input_tokens is None
    assert status.output_tokens is None
    assert budget.used == 1
    with pytest.raises(BudgetExhausted):
        budget.spend()


def test_cancel_and_deadline_stop_reservation_without_spending() -> None:
    cancelled_budget = DailyCallBudget(limit=1)
    cancelled = context(budget=cancelled_budget)
    cancelled.cancel()
    with pytest.raises(ExecutionCancelled):
        cancelled.begin_provider()
    assert cancelled_budget.used == 0

    expired_budget = DailyCallBudget(limit=1)
    expired = ExecutionContext(
        deadline=time.monotonic() - 0.001,
        budget=expired_budget,
    )
    with pytest.raises(ExecutionDeadlineExceeded):
        expired.begin_provider()
    assert expired_budget.used == 0


def test_provider_timeout_is_capped_by_config_and_remaining_deadline() -> None:
    configured = context()
    assert configured.available_provider_timeout(6.0, 0.5) == pytest.approx(
        6.0, abs=0.01
    )

    near_deadline = ExecutionContext(
        deadline=time.monotonic() + 1.0,
        budget=DailyCallBudget(limit=1),
    )
    available = near_deadline.available_provider_timeout(6.0, 0.25)
    assert 0.70 <= available <= 0.75

    no_margin = ExecutionContext(
        deadline=time.monotonic() + 0.1,
        budget=DailyCallBudget(limit=1),
    )
    with pytest.raises(ExecutionDeadlineExceeded):
        no_margin.available_provider_timeout(6.0, 0.2)


def test_status_contains_only_bounded_operational_values_and_is_a_copy() -> None:
    execution = context()
    execution.record_stage("retrieval", 12.5)
    execution.begin_provider()
    execution.finish_provider("completed", input_tokens=120, output_tokens=30)

    status = execution.status()
    assert status.provider_outcome == "completed"
    assert status.input_tokens == 120
    assert status.output_tokens == 30
    assert status.stages_ms == {"retrieval": 12.5}

    status.stages_ms["provider"] = 7.0
    assert execution.status().stages_ms == {"retrieval": 12.5}
    execution.record_stage("retrieval", 13.0)
    assert execution.status().stages_ms["retrieval"] == 25.5
    with pytest.raises(ValueError):
        execution.record_stage("provider", float("inf"))

    slow = context()
    slow.record_stage("provider", 90_000.0)
    assert slow.status().stages_ms["provider"] == 60_000.0


@pytest.mark.parametrize("max_workers", [1, 3, 5])
def test_exact_capacity_is_admitted_and_excess_is_rejected(max_workers: int) -> None:
    executor: BoundedAnswerExecutor[int] = BoundedAnswerExecutor(max_workers)
    release = threading.Event()
    all_started = threading.Event()
    started = 0
    started_lock = threading.Lock()

    def blocking(_execution: ExecutionContext) -> int:
        nonlocal started
        with started_lock:
            started += 1
            if started == max_workers:
                all_started.set()
        assert release.wait(timeout=5.0)
        return 7

    futures: list[Future[int]] = []
    try:
        futures = [executor.submit(blocking, context()) for _ in range(max_workers)]
        assert all_started.wait(timeout=5.0)
        with pytest.raises(ExecutorCapacityError):
            executor.submit(blocking, context())
    finally:
        release.set()
        for future in futures:
            assert future.result(timeout=5.0) == 7
        executor.shutdown()


def test_caller_cancellation_does_not_release_an_active_jobs_slot() -> None:
    executor: BoundedAnswerExecutor[str] = BoundedAnswerExecutor(1)
    started = threading.Event()
    release = threading.Event()
    active_context = context()

    def ignores_caller_cancellation(_execution: ExecutionContext) -> str:
        started.set()
        assert release.wait(timeout=5.0)
        return "late-result"

    future = executor.submit(ignores_caller_cancellation, active_context)
    assert started.wait(timeout=5.0)
    active_context.cancel()

    with pytest.raises(ExecutorCapacityError):
        executor.submit(ignores_caller_cancellation, context())

    release.set()
    assert future.result(timeout=5.0) == "late-result"
    recovered = executor.submit(lambda _execution: "recovered", context())
    assert recovered.result(timeout=5.0) == "recovered"
    executor.shutdown()


def test_exception_and_expired_context_release_capacity_after_completion() -> None:
    executor: BoundedAnswerExecutor[str] = BoundedAnswerExecutor(1)

    failed = executor.submit(
        lambda _execution: (_ for _ in ()).throw(RuntimeError("synthetic")),
        context(),
    )
    with pytest.raises(RuntimeError, match="synthetic"):
        failed.result(timeout=5.0)

    expired = ExecutionContext(
        deadline=time.monotonic() - 0.001,
        budget=DailyCallBudget(limit=1),
    )
    timed_out = executor.submit(lambda _execution: "never", expired)
    with pytest.raises(ExecutionDeadlineExceeded):
        timed_out.result(timeout=5.0)

    recovered = executor.submit(lambda _execution: "ok", context())
    assert recovered.result(timeout=5.0) == "ok"
    executor.shutdown()


def test_submit_failure_releases_the_reserved_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor: BoundedAnswerExecutor[str] = BoundedAnswerExecutor(1)
    original_submit = executor._executor.submit

    def fail_submit(*_args: object, **_kwargs: object) -> Future[str]:
        raise RuntimeError("synthetic submit failure")

    try:
        monkeypatch.setattr(executor._executor, "submit", fail_submit)
        with pytest.raises(RuntimeError, match="synthetic submit failure"):
            executor.submit(lambda _execution: "never", context())

        monkeypatch.setattr(executor._executor, "submit", original_submit)
        recovered = executor.submit(lambda _execution: "recovered", context())
        assert recovered.result(timeout=5.0) == "recovered"
    finally:
        executor.shutdown()


def test_shutdown_rejects_new_work_without_cancelling_active_work() -> None:
    executor: BoundedAnswerExecutor[str] = BoundedAnswerExecutor(1)
    started = threading.Event()
    release = threading.Event()

    def blocking(_execution: ExecutionContext) -> str:
        started.set()
        assert release.wait(timeout=5.0)
        return "finished"

    future = executor.submit(blocking, context())
    assert started.wait(timeout=5.0)
    executor.shutdown(wait=False)

    with pytest.raises(ExecutorClosedError):
        executor.submit(blocking, context())
    assert not future.done()

    release.set()
    assert future.result(timeout=5.0) == "finished"
    executor.shutdown(wait=True)
