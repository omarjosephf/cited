"""Offline transport, binding, replay and actual-worker lifetime checks."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import httpx
import pytest

from assistant.budget import BudgetExhausted
from assistant.persistent_budget import BudgetCapacityError, BudgetUnavailable
from assistant.runtime import (
    BoundedAnswerExecutor,
    ExecutionCancelled,
    ExecutionContext,
)
from assistant.shared_budget import (
    BudgetBinding,
    SharedJobBudget,
    SupabaseBudgetRpc,
    submit_shared_job,
    validate_policy,
)


def policy() -> dict[str, Any]:
    service = {
        "daily_attempts": 40,
        "monthly_attempts": 200,
        "daily_micro_usd": 400000,
        "monthly_micro_usd": 2000000,
        "lifetime_micro_usd": 3000000,
        "carried_lifetime_micro_usd": 0,
        "carried_day_micro_usd": 0,
        "carried_month_micro_usd": 0,
    }
    return {
        "version": 1,
        "reservation_micro_usd": 40000,
        "aggregate_cap_micro_usd": 3000000,
        "aggregate_carried_micro_usd": 0,
        "max_active": 1,
        "services": {
            "ev": {**service, "ledger_id": "e" * 64},
            "cited": {**service, "ledger_id": "c" * 64},
        },
    }


def binding() -> BudgetBinding:
    return BudgetBinding("a" * 64, "b" * 64, json.dumps(policy()))


class Rpc:
    """Controlled acknowledgments, not a substitute for database race tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.active: str | None = None
        self.used = 0
        self.fail: str | None = None
        self.alter: dict[str, Any] = {}
        self.released = threading.Event()

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        action = payload["p_action"]
        status = "ok"
        if action == "acquire":
            status = "acquired" if self.active is None else "busy"
            if status == "acquired":
                self.active = payload["p_job_id"]
        elif action == "reserve":
            self.used += 1
            status = "reserved"
        elif action == "release":
            self.active = None
            status = "released"
            self.released.set()
        if self.fail == action:
            raise TimeoutError("synthetic committed response lost")
        return {
            **{
                k: payload["p_" + k]
                for k in ("ledger_id", "config_id", "job_id", "owner_id", "ordinal")
            },
            "status": status,
            "used": self.used,
            "remaining": 10 - self.used,
            **self.alter,
        }


@pytest.mark.parametrize("service", ["ev", "cited"])
def test_each_workload_reserves_primary_and_fallback(service: str) -> None:
    rpc = Rpc()
    b = SharedJobBudget(rpc, binding(), service, 2)
    b.acquire()
    ctx = ExecutionContext(time.monotonic() + 5, b)
    ctx.begin_provider()
    ctx.finish_provider("uncertain")
    ctx.begin_provider("fallback")
    ctx.finish_provider("completed", input_tokens=4, output_tokens=2)
    assert b.used == 2
    assert [p["p_ordinal"] for p in rpc.calls if p["p_action"] == "reserve"] == [1, 2]
    assert {p["p_service"] for p in rpc.calls} == {service}
    with pytest.raises(BudgetExhausted):
        b.spend()
    b._release_after_exit()
    assert rpc.used == 2


def test_binding_is_immutable_and_config_digest_tracks_every_field() -> None:
    b = binding()
    changed = b.policy
    changed["max_active"] = 2
    assert b.policy["max_active"] == 1
    assert (
        BudgetBinding(
            b.ledger_id, b.carry_forward_sha256, json.dumps(changed)
        ).config_id
        != b.config_id
    )


@pytest.mark.parametrize("key", list(policy()))
def test_missing_top_policy_fields(key: str) -> None:
    p = policy()
    del p[key]
    with pytest.raises(ValueError):
        validate_policy(p)


@pytest.mark.parametrize("key", list(policy()["services"]["ev"]))
def test_missing_service_policy_fields(key: str) -> None:
    p = policy()
    del p["services"]["ev"][key]
    with pytest.raises(ValueError):
        validate_policy(p)


@pytest.mark.parametrize("value", [True, 1.5, "40000", -1, 39999, 40001])
def test_reservation_cannot_be_changed(value: Any) -> None:
    p = policy()
    p["reservation_micro_usd"] = value
    with pytest.raises(ValueError):
        validate_policy(p)


def test_no_reservation_before_acquisition() -> None:
    rpc = Rpc()
    with pytest.raises(BudgetUnavailable):
        SharedJobBudget(rpc, binding(), "ev", 2).spend()
    assert rpc.calls == []


@pytest.mark.parametrize("stage", ["acquire", "reserve"])
def test_lost_commit_response_stops_dispatch_and_never_retries(stage: str) -> None:
    rpc = Rpc()
    b = SharedJobBudget(rpc, binding(), "ev", 2)
    if stage == "reserve":
        b.acquire()
    rpc.fail = stage
    with pytest.raises(BudgetUnavailable):
        b.acquire() if stage == "acquire" else b.spend()
    calls = len(rpc.calls)
    with pytest.raises(BudgetUnavailable):
        b.spend()
    with pytest.raises(BudgetUnavailable):
        _ = b.remaining
    assert len(rpc.calls) == calls
    assert rpc.active == b.job_id
    assert rpc.used == (1 if stage == "reserve" else 0)


@pytest.mark.parametrize(
    "alter",
    [
        {"ledger_id": "f" * 64},
        {"config_id": "f" * 64},
        {"owner_id": "wrong"},
        {"job_id": "wrong"},
        {"ordinal": True},
        {"remaining": True},
        {"used": -1},
        {"status": "duplicate"},
        {"status": "reserved"},
        {"extra": 1},
    ],
)
def test_bad_acquisition_acknowledgment_poisoned(alter: dict[str, Any]) -> None:
    rpc = Rpc()
    rpc.alter = alter
    b = SharedJobBudget(rpc, binding(), "cited", 2)
    with pytest.raises(BudgetUnavailable):
        b.acquire()
    with pytest.raises(BudgetUnavailable):
        b.spend()
    assert len(rpc.calls) == 1


def test_timeout_and_caller_cancel_keep_slot_until_worker_really_exits() -> None:
    rpc = Rpc()
    b = SharedJobBudget(rpc, binding(), "ev", 2)
    started, finish = threading.Event(), threading.Event()
    executor: BoundedAnswerExecutor[int] = BoundedAnswerExecutor(1)
    ctx = ExecutionContext(time.monotonic() + 10, b)

    def work(context: ExecutionContext) -> int:
        context.begin_provider()
        started.set()
        assert finish.wait(4)
        context.finish_provider("uncertain")
        return 1

    try:
        future = submit_shared_job(executor, work, ctx)
        assert started.wait(2)
        with pytest.raises(TimeoutError):
            future.result(timeout=0.01)
        ctx.cancel()
        assert not future.cancel()
        assert rpc.active == b.job_id
        with pytest.raises(BudgetCapacityError):
            SharedJobBudget(rpc, binding(), "cited", 2).acquire()
        assert not rpc.released.is_set()
        finish.set()
        assert future.result(timeout=2) == 1
        assert rpc.released.wait(2)
        assert rpc.active is None and rpc.used == 1
    finally:
        finish.set()
        executor.shutdown()


def test_cancelled_before_submit_makes_no_rpc() -> None:
    rpc = Rpc()
    b = SharedJobBudget(rpc, binding(), "ev", 2)
    ctx = ExecutionContext(time.monotonic() + 5, b)
    ctx.cancel()
    executor: BoundedAnswerExecutor[int] = BoundedAnswerExecutor(1)
    try:
        with pytest.raises(ExecutionCancelled):
            submit_shared_job(executor, lambda _: 1, ctx)
        assert rpc.calls == []
    finally:
        executor.shutdown()


def test_submission_failure_releases_but_release_loss_is_visible() -> None:
    rpc = Rpc()
    rpc.fail = "release"
    b = SharedJobBudget(rpc, binding(), "ev", 2)
    executor: BoundedAnswerExecutor[int] = BoundedAnswerExecutor(1)
    executor.shutdown()
    with pytest.raises(RuntimeError):
        submit_shared_job(
            executor, lambda _: 1, ExecutionContext(time.monotonic() + 5, b)
        )
    assert b.release_failed
    assert [p["p_action"] for p in rpc.calls] == ["acquire", "release"]


@pytest.mark.parametrize(
    "status,body",
    [
        (302, b"{}"),
        (500, b"private error"),
        (200, b"x" * 8193),
        (200, b"[]"),
        (200, b"bad json"),
        (200, b"\xff"),
    ],
)
def test_transport_failures_are_sanitized_and_never_retried(
    status: int, body: bytes
) -> None:
    rpc = SupabaseBudgetRpc(
        "https://" + "a" * 20 + ".supabase.co", "sb_secret_synthetic"
    )
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, stream=httpx.ByteStream(body))

    rpc._client.close()
    rpc._client = httpx.Client(
        transport=httpx.MockTransport(handler), follow_redirects=False
    )
    try:
        with pytest.raises(BudgetUnavailable) as error:
            rpc.call({})
        assert "private" not in str(error.value)
        assert len(calls) == 1
    finally:
        rpc.close()


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost",
        "https://example.com",
        "https://" + "a" * 20 + ".supabase.co/evil",
    ],
)
def test_transport_requires_exact_bound_origin(origin: str) -> None:
    with pytest.raises(ValueError):
        SupabaseBudgetRpc(origin, "sb_secret_synthetic")
