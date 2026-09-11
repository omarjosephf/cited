"""Bounded execution and conservative provider-attempt accounting.

The HTTP layer may stop waiting for synchronous work, but that does not stop
the underlying thread or prove that a provider request was cancelled.  This
module keeps admission capacity attached to the actual job and keeps the daily
budget attached to the provider boundary.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

from assistant.budget import AttemptBudget

RuntimeStage = Literal["queue", "retrieval", "provider", "validation"]
ProviderCompletion = Literal["completed", "uncertain"]
ProviderOutcome = Literal["not_started", "in_progress", "completed", "uncertain"]
ProviderRoute = Literal["primary", "fallback"]

_RUNTIME_STAGES: tuple[RuntimeStage, ...] = (
    "queue",
    "retrieval",
    "provider",
    "validation",
)
_PROVIDER_COMPLETIONS: tuple[ProviderCompletion, ...] = ("completed", "uncertain")
MAX_WORKERS = 5
MAX_DURATION_MS = 60_000.0
MAX_PROVIDER_TIMEOUT_SECONDS = 60.0
MAX_USAGE_TOKENS = 1_000_000


class ExecutionCancelled(RuntimeError):
    """The caller cancelled before another unit of work could begin."""


class ExecutionDeadlineExceeded(TimeoutError):
    """The monotonic end-to-end deadline leaves no time for more work."""


class ProviderAttemptAlreadyBegun(RuntimeError):
    """A context may reserve at most two serial provider attempts."""


class ProviderAttemptNotBegun(RuntimeError):
    """Provider completion cannot be recorded without a reservation."""


class ExecutorCapacityError(RuntimeError):
    """All bounded execution slots are occupied."""


class ExecutorClosedError(RuntimeError):
    """The executor is closing and accepts no new work."""


def _bounded_duration(duration_ms: float) -> float:
    if not isinstance(duration_ms, int | float) or isinstance(duration_ms, bool):
        raise TypeError("duration_ms must be a number")
    value = float(duration_ms)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("duration_ms must be finite and nonnegative")
    return min(value, MAX_DURATION_MS)


def _usage_tokens(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer or None")
    if not 0 <= value <= MAX_USAGE_TOKENS:
        raise ValueError(f"{name} must be between 0 and {MAX_USAGE_TOKENS}")
    return value


@dataclass(frozen=True)
class ProviderAttemptStatus:
    route: ProviderRoute
    model: str
    outcome: ProviderCompletion
    input_tokens: int | None
    output_tokens: int | None
    duration_ms: float


@dataclass(frozen=True)
class ExecutionStatus:
    """A content-free snapshot safe to aggregate after a job exits."""

    provider_attempted: bool
    provider_outcome: ProviderOutcome
    input_tokens: int | None
    output_tokens: int | None
    stages_ms: dict[RuntimeStage, float]
    attempts: tuple[ProviderAttemptStatus, ...]


@dataclass
class ExecutionContext:
    """One request's deadline, cancellation and provider-attempt state."""

    deadline: float
    budget: AttemptBudget
    cancellation: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _provider_outcome: ProviderOutcome = field(default="not_started", init=False)
    _input_tokens: int | None = field(default=None, init=False)
    _output_tokens: int | None = field(default=None, init=False)
    _stages_ms: dict[RuntimeStage, float] = field(default_factory=dict, init=False)
    _active_provider: tuple[ProviderRoute, str, float] | None = field(
        default=None, init=False
    )
    _attempts: list[ProviderAttemptStatus] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.deadline, int | float) or isinstance(
            self.deadline, bool
        ):
            raise TypeError("deadline must be a monotonic number")
        self.deadline = float(self.deadline)
        if not math.isfinite(self.deadline) or self.deadline < 0.0:
            raise ValueError("deadline must be a finite nonnegative monotonic value")

    def cancel(self) -> None:
        """Prevent a new stage or provider attempt from beginning."""
        with self._lock:
            self.cancellation.set()

    def _raise_if_stopped_unlocked(self) -> None:
        if self.cancellation.is_set():
            raise ExecutionCancelled("execution was cancelled")
        if time.monotonic() >= self.deadline:
            raise ExecutionDeadlineExceeded("execution deadline was reached")

    def raise_if_stopped(self) -> None:
        """Fail before starting more work after cancellation or expiry."""
        with self._lock:
            self._raise_if_stopped_unlocked()

    def available_provider_timeout(
        self,
        configured_timeout_seconds: float,
        validation_margin_seconds: float,
    ) -> float:
        """Return the provider budget left before the validation margin."""
        configured = self._timeout_value(
            configured_timeout_seconds, "configured_timeout_seconds", positive=True
        )
        margin = self._timeout_value(
            validation_margin_seconds, "validation_margin_seconds", positive=False
        )
        with self._lock:
            self._raise_if_stopped_unlocked()
            available = self.deadline - time.monotonic() - margin
            if available <= 0.0:
                raise ExecutionDeadlineExceeded(
                    "execution deadline leaves no provider validation margin"
                )
            return min(configured, available)

    @staticmethod
    def _timeout_value(value: float, name: str, *, positive: bool) -> float:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise TypeError(f"{name} must be a number")
        result = float(value)
        minimum_ok = result > 0.0 if positive else result >= 0.0
        if (
            not math.isfinite(result)
            or not minimum_ok
            or result > MAX_PROVIDER_TIMEOUT_SECONDS
        ):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(
                f"{name} must be finite, {qualifier}, and at most "
                f"{MAX_PROVIDER_TIMEOUT_SECONDS}"
            )
        return result

    def begin_provider(
        self, route: ProviderRoute = "primary", model: str = "unknown"
    ) -> None:
        """Atomically reserve one of at most two serial provider attempts."""
        if route not in {"primary", "fallback"}:
            raise ValueError("unsupported provider route")
        if not isinstance(model, str) or not model or len(model) > 128:
            raise ValueError("model must be a non-empty bounded string")
        with self._lock:
            self._raise_if_stopped_unlocked()
            if self._active_provider is not None or len(self._attempts) >= 2:
                raise ProviderAttemptAlreadyBegun(
                    "this execution context cannot reserve another provider attempt"
                )
            if not self._attempts and route != "primary":
                raise ProviderAttemptAlreadyBegun("the first attempt must be primary")
            if self._attempts and route != "fallback":
                raise ProviderAttemptAlreadyBegun("the second attempt must be fallback")
            self.budget.spend()
            self._provider_outcome = "in_progress"
            self._active_provider = (route, model, time.monotonic())

    def finish_provider(
        self,
        outcome: ProviderCompletion,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Record the known result of the reserved provider attempt.

        Missing usage remains ``None``.  A reservation is never refunded here:
        any failure after this boundary may already have incurred provider cost.
        """
        if outcome not in _PROVIDER_COMPLETIONS:
            raise ValueError(f"unsupported provider outcome: {outcome!r}")
        validated_input = _usage_tokens(input_tokens, "input_tokens")
        validated_output = _usage_tokens(output_tokens, "output_tokens")
        if (validated_input is None) != (validated_output is None):
            raise ValueError(
                "input_tokens and output_tokens must both be known or unknown"
            )

        with self._lock:
            if self._active_provider is None:
                raise ProviderAttemptNotBegun("no provider attempt was reserved")
            route, model, started = self._active_provider
            duration = _bounded_duration((time.monotonic() - started) * 1000)
            self._attempts.append(
                ProviderAttemptStatus(
                    route, model, outcome, validated_input, validated_output, duration
                )
            )
            self._active_provider = None
            self._provider_outcome = outcome
            self._input_tokens = validated_input
            self._output_tokens = validated_output

    def record_stage(self, stage: RuntimeStage, duration_ms: float) -> None:
        """Accumulate bounded duration segments without accepting content."""
        if stage not in _RUNTIME_STAGES:
            raise ValueError(f"unsupported runtime stage: {stage!r}")
        duration = _bounded_duration(duration_ms)
        with self._lock:
            self._stages_ms[stage] = min(
                self._stages_ms.get(stage, 0.0) + duration,
                MAX_DURATION_MS,
            )

    def status(self) -> ExecutionStatus:
        """Return an immutable status whose stage mapping is a fresh copy."""
        with self._lock:
            return ExecutionStatus(
                provider_attempted=self._provider_outcome != "not_started",
                provider_outcome=self._provider_outcome,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                stages_ms=dict(self._stages_ms),
                attempts=tuple(self._attempts),
            )


@dataclass
class BoundedAnswerExecutor[T]:
    """A thread executor with no admission queue beyond its worker bound."""

    max_workers: int
    acquire_shared_worker: Callable[[], Callable[[], None]] | None = None
    _executor: ThreadPoolExecutor = field(init=False)
    _slots: threading.BoundedSemaphore = field(init=False)
    _state_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _closing: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_workers, int)
            or isinstance(self.max_workers, bool)
            or not 1 <= self.max_workers <= MAX_WORKERS
        ):
            raise ValueError(f"max_workers must be an integer from 1 to {MAX_WORKERS}")
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="assistant-answer",
        )
        self._slots = threading.BoundedSemaphore(self.max_workers)

    def submit(
        self,
        function: Callable[[ExecutionContext], T],
        context: ExecutionContext,
    ) -> Future[T]:
        """Admit work immediately or reject it without creating a queue."""
        with self._state_lock:
            if self._closing:
                raise ExecutorClosedError("answer executor is closing")
            if not self._slots.acquire(blocking=False):
                raise ExecutorCapacityError("answer executor is at capacity")
            release_shared: Callable[[], None] | None = None
            try:
                if self.acquire_shared_worker is not None:
                    release_shared = self.acquire_shared_worker()
                future = self._executor.submit(self._run, function, context)
            except BaseException:
                if release_shared is not None:
                    release_shared()
                self._slots.release()
                raise

            def release(_future: Future[T]) -> None:
                try:
                    if release_shared is not None:
                        release_shared()
                finally:
                    self._slots.release()

            future.add_done_callback(release)
            return future

    @staticmethod
    def _run(function: Callable[[ExecutionContext], T], context: ExecutionContext) -> T:
        context.raise_if_stopped()
        return function(context)

    def shutdown(self, wait: bool = True) -> None:
        """Reject new work and optionally wait for admitted jobs to exit."""
        with self._state_lock:
            self._closing = True
        self._executor.shutdown(wait=wait, cancel_futures=False)
