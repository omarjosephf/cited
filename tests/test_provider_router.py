"""Offline routing tests for the two serial provider attempts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from assistant.budget import BudgetExhausted, DailyCallBudget
from assistant.provider_adapters import (
    GeneratedAnswer,
    ProviderResponseRejected,
    ProviderUnavailable,
)
from assistant.provider_router import FALLBACK_MODEL, PRIMARY_MODEL, ProviderRouter
from assistant.runtime import (
    ExecutionCancelled,
    ExecutionContext,
    ExecutionDeadlineExceeded,
)

REQUEST: dict[str, Any] = {
    "system": "system",
    "question": "question",
    "evidence": ({"id": "E01", "title": "Guide", "text": "evidence"},),
    "history": "bounded history",
}


@dataclass
class FakePost:
    closed: bool = False

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeAdapter:
    result: GeneratedAnswer | BaseException
    prepared_requests: list[dict[str, Any]] = field(default_factory=list)
    deadlines: list[float] = field(default_factory=list)
    posts: list[FakePost] = field(default_factory=list)
    dispatched: int = 0

    def prepare(self, **request: Any) -> FakePost:
        self.prepared_requests.append(request)
        post = FakePost()
        self.posts.append(post)
        return post

    def generate(
        self,
        *,
        prepared: FakePost,
        evidence: tuple[dict[str, str], ...],
        deadline: float,
        before_dispatch: Any = None,
    ) -> GeneratedAnswer:
        assert evidence is REQUEST["evidence"]
        self.deadlines.append(deadline)
        if before_dispatch is not None:
            before_dispatch()
        self.dispatched += 1
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def generated(route: str) -> GeneratedAnswer:
    return GeneratedAnswer(
        "not_covered",
        (),
        None,
        None,
        cast(Any, route),
        "end_turn",
    )


def router(
    primary: FakeAdapter,
    fallback: FakeAdapter | None,
    budget: DailyCallBudget,
) -> ProviderRouter:
    return ProviderRouter(
        cast(Any, primary),
        cast(Any, fallback),
        provider_timeout_seconds=6.0,
        primary_timeout_seconds=3.0,
        validation_margin_seconds=0.5,
        budget=budget,
    )


def context(budget: DailyCallBudget, seconds: float = 8.0) -> ExecutionContext:
    return ExecutionContext(deadline=time.monotonic() + seconds, budget=budget)


def test_primary_success_never_prepares_or_dispatches_fallback() -> None:
    budget = DailyCallBudget(5)
    primary = FakeAdapter(generated("primary"))
    fallback = FakeAdapter(generated("fallback"))
    result = router(primary, fallback, budget).create(
        **REQUEST, _execution_context=context(budget)
    )

    assert result.route == "primary"
    assert budget.used == 1
    assert primary.dispatched == 1
    assert fallback.prepared_requests == []
    assert primary.posts[0].closed


def test_availability_failure_uses_identical_evidence_and_bounded_input() -> None:
    budget = DailyCallBudget(5)
    primary = FakeAdapter(ProviderUnavailable("provider_unavailable"))
    fallback = FakeAdapter(generated("fallback"))
    execution = context(budget)

    result = router(primary, fallback, budget).create(
        **REQUEST, _execution_context=execution
    )

    assert result.route == "fallback"
    assert primary.prepared_requests == [REQUEST]
    assert fallback.prepared_requests == [REQUEST]
    assert (
        primary.prepared_requests[0]["evidence"]
        is fallback.prepared_requests[0]["evidence"]
    )
    assert budget.used == 2
    attempts = execution.status().attempts
    assert [(item.route, item.model, item.outcome) for item in attempts] == [
        ("primary", PRIMARY_MODEL, "uncertain"),
        ("fallback", FALLBACK_MODEL, "completed"),
    ]
    assert all(item.input_tokens is None for item in attempts)
    assert all(post.closed for post in primary.posts + fallback.posts)


@pytest.mark.parametrize(
    "failure",
    [
        ProviderResponseRejected("provider_json"),
        ExecutionDeadlineExceeded("deadline"),
        ExecutionCancelled("cancelled"),
    ],
)
def test_nonavailability_failures_never_fallback(failure: BaseException) -> None:
    budget = DailyCallBudget(5)
    primary = FakeAdapter(failure)
    fallback = FakeAdapter(generated("fallback"))

    with pytest.raises(type(failure)):
        router(primary, fallback, budget).create(
            **REQUEST, _execution_context=context(budget)
        )

    assert primary.dispatched == 1
    assert fallback.prepared_requests == []
    assert budget.used == 1


def test_cancelled_before_dispatch_is_uncertain_but_never_falls_back() -> None:
    budget = DailyCallBudget(5)
    execution = context(budget)

    class CancellingAdapter(FakeAdapter):
        def generate(self, **kwargs: Any) -> GeneratedAnswer:
            execution.cancel()
            return super().generate(**kwargs)

    primary = CancellingAdapter(generated("primary"))
    fallback = FakeAdapter(generated("fallback"))
    with pytest.raises(ExecutionCancelled):
        router(primary, fallback, budget).create(
            **REQUEST, _execution_context=execution
        )

    assert primary.dispatched == 0
    assert fallback.prepared_requests == []
    assert execution.status().attempts[0].outcome == "uncertain"
    assert budget.used == 1


def test_fallback_cannot_exceed_the_shared_attempt_allowance() -> None:
    budget = DailyCallBudget(1)
    primary = FakeAdapter(ProviderUnavailable("provider_unavailable"))
    fallback = FakeAdapter(generated("fallback"))

    with pytest.raises(BudgetExhausted):
        router(primary, fallback, budget).create(
            **REQUEST, _execution_context=context(budget)
        )

    assert primary.dispatched == 1
    assert fallback.dispatched == 0
    assert budget.used == 1
    assert fallback.posts[0].closed


def test_primary_and_shared_deadlines_are_nested() -> None:
    budget = DailyCallBudget(5)
    primary = FakeAdapter(ProviderUnavailable("provider_unavailable"))
    fallback = FakeAdapter(generated("fallback"))
    started = time.monotonic()

    router(primary, fallback, budget).create(
        **REQUEST, _execution_context=context(budget)
    )

    assert 2.5 <= primary.deadlines[0] - started <= 3.5
    assert 5.5 <= fallback.deadlines[0] - started <= 6.5
    assert primary.deadlines[0] < fallback.deadlines[0]


def test_expired_overall_deadline_spends_nothing_and_never_falls_back() -> None:
    budget = DailyCallBudget(5)
    primary = FakeAdapter(generated("primary"))
    fallback = FakeAdapter(generated("fallback"))
    expired = ExecutionContext(deadline=time.monotonic() - 1, budget=budget)

    with pytest.raises(ExecutionDeadlineExceeded):
        router(primary, fallback, budget).create(**REQUEST, _execution_context=expired)

    assert primary.dispatched == 0
    assert fallback.prepared_requests == []
    assert budget.used == 0
