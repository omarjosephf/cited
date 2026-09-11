"""Aggregate operator metrics, with no visitor content in them.

Two requirements point in opposite directions and both are real. An assistant
that cannot be observed cannot be operated: without counts and latencies, "is it
working?" is answered by asking it a question and hoping. But the obvious way to
observe a question-answering service — log the questions — is exactly what the
privacy policy governing this deployment prohibits, and it is prohibited for a
good reason: a log is a place data goes to be retained and read by people it was
not sent to.

They are reconcilable, and the reconciliation is the whole design here: **count
outcomes, never content.** Knowing that 12% of questions were not covered by the
corpus is the useful signal — it says the corpus has a gap. Knowing *which*
questions were not covered would be more useful still, and is not worth what it
costs.

WHAT IS DELIBERATELY ABSENT, and must stay absent:
question text, transcripts, per-question records, IP addresses, user agents,
identifiers of any kind, and any structure that could be joined back to a person.

WHAT THESE NUMBERS ARE NOT:
lifetime totals. They live in process memory, so they reset whenever the machine
starts — which, under a scale-to-zero deployment, is routine rather than
exceptional. They are reported as "since last start" and labelled as such.
Persisting them is a separate decision with its own privacy question.
"""

from __future__ import annotations

import threading
from bisect import insort
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import ceil, isfinite
from typing import Any, Literal

Outcome = Literal["answered", "not_covered", "unavailable"]
"""The three states a visitor can observe. Metrics use the same vocabulary the
interface does, so an operator reading a count knows exactly what a visitor saw.
"""

AdmissionResult = Literal["admitted", "rejected_full", "rejected_closed"]
Stage = Literal["queue", "retrieval", "provider", "validation"]
AttemptOutcome = Literal["completed", "uncertain"]

_OUTCOMES: tuple[Outcome, ...] = ("answered", "not_covered", "unavailable")
_ADMISSION_RESULTS: tuple[AdmissionResult, ...] = (
    "admitted",
    "rejected_full",
    "rejected_closed",
)
_STAGES: tuple[Stage, ...] = ("queue", "retrieval", "provider", "validation")
_ATTEMPT_OUTCOMES: tuple[AttemptOutcome, ...] = ("completed", "uncertain")

MAX_DURATION_MS = 60_000.0
MAX_RECORDED_COUNT = 1_000_000
MAX_IDENTITY_LENGTH = 128

MAX_LATENCY_SAMPLES = 2048
"""Cap on retained latency samples, so memory cannot grow without bound.

Percentiles are computed from a sorted window rather than a streaming estimator:
at this volume the exact answer is cheap, and an approximate one would be harder
to explain than it is worth. When the window is full the oldest sample is
dropped, which biases percentiles toward recent behaviour — the right bias for
"is it healthy now?".
"""


def _duration(value: float) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError("duration must be a number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError("duration must be finite and nonnegative")
    # Observability must not mask a slow-job exception during cleanup.  The
    # final bucket is an explicit saturation point rather than an exact value.
    return min(result, MAX_DURATION_MS)


def _count(value: int | None, name: str, *, optional: bool) -> int | None:
    if value is None and optional:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        expected = "an integer or None" if optional else "an integer"
        raise TypeError(f"{name} must be {expected}")
    if not 0 <= value <= MAX_RECORDED_COUNT:
        raise ValueError(f"{name} must be between 0 and {MAX_RECORDED_COUNT}")
    return value


@dataclass(frozen=True)
class MetricsIdentity:
    """Trusted deployment identity, supplied once rather than per request."""

    model: str | None = None
    corpus: str | None = None
    prompt: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("model", self.model),
            ("corpus", self.corpus),
            ("prompt", self.prompt),
        ):
            if value is None:
                continue
            if not isinstance(value, str):
                raise TypeError(f"{name} identity must be a string or None")
            if not value or len(value) > MAX_IDENTITY_LENGTH or not value.isprintable():
                raise ValueError(
                    f"{name} identity must be 1-{MAX_IDENTITY_LENGTH} "
                    "printable characters"
                )


@dataclass
class AssistantMetrics:
    """Counters and latency percentiles for one process lifetime.

    Every method takes the lock. Contention is irrelevant at this scale, and the
    alternative — reasoning about which counter updates are atomic under which
    interpreter — is a false economy in code whose whole job is to be trusted.
    """

    identity: MetricsIdentity | None = None
    _outcomes: Counter[str] = field(default_factory=Counter, init=False)
    _latencies_ms: list[float] = field(default_factory=list, init=False)
    _order: list[float] = field(default_factory=list, init=False)
    _admissions: Counter[str] = field(default_factory=Counter, init=False)
    _stage_latencies_ms: dict[str, list[float]] = field(
        default_factory=lambda: {stage: [] for stage in _STAGES}, init=False
    )
    _stage_order: dict[str, list[float]] = field(
        default_factory=lambda: {stage: [] for stage in _STAGES}, init=False
    )
    _attempt_outcomes: Counter[str] = field(default_factory=Counter, init=False)
    _known_usage_attempts: int = field(default=0, init=False)
    _unknown_usage_attempts: int = field(default=0, init=False)
    _input_tokens: int = field(default=0, init=False)
    _output_tokens: int = field(default=0, init=False)
    _attempts_by_model: Counter[str] = field(default_factory=Counter, init=False)
    _attempts_by_route: Counter[str] = field(default_factory=Counter, init=False)
    _rejected_citations: int = field(default=0, init=False)
    _started: datetime = field(default_factory=lambda: datetime.now(UTC), init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def record(
        self,
        outcome: Outcome,
        latency_ms: float,
        rejected_citations: int = 0,
    ) -> None:
        """Record one completed request. Never called with the question."""
        if outcome not in _OUTCOMES:
            raise ValueError(f"unsupported request outcome: {outcome!r}")
        duration = _duration(latency_ms)
        rejected = _count(rejected_citations, "rejected_citations", optional=False)
        assert rejected is not None
        with self._lock:
            self._outcomes[outcome] += 1
            self._rejected_citations += rejected

            self._record_sample(self._latencies_ms, self._order, duration)

    def record_admission(self, result: AdmissionResult) -> None:
        """Count one fixed admission result."""
        if result not in _ADMISSION_RESULTS:
            raise ValueError(f"unsupported admission result: {result!r}")
        with self._lock:
            self._admissions[result] += 1

    def record_stage(self, stage: Stage, duration_ms: float) -> None:
        """Record a bounded aggregate stage duration without request content."""
        if stage not in _STAGES:
            raise ValueError(f"unsupported metrics stage: {stage!r}")
        duration = _duration(duration_ms)
        with self._lock:
            self._record_sample(
                self._stage_latencies_ms[stage], self._stage_order[stage], duration
            )

    def record_attempt(
        self,
        outcome: AttemptOutcome,
        *,
        route: Literal["primary", "fallback"] = "primary",
        model: str = "unknown",
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Count one provider attempt and preserve unknown usage as unknown."""
        if outcome not in _ATTEMPT_OUTCOMES:
            raise ValueError(f"unsupported provider attempt outcome: {outcome!r}")
        if route not in {"primary", "fallback"}:
            raise ValueError("unsupported provider route")
        if not isinstance(model, str) or not model or len(model) > MAX_IDENTITY_LENGTH:
            raise ValueError("model identity must be a non-empty bounded string")
        input_count = _count(input_tokens, "input_tokens", optional=True)
        output_count = _count(output_tokens, "output_tokens", optional=True)
        if (input_count is None) != (output_count is None):
            raise ValueError(
                "input_tokens and output_tokens must both be known or unknown"
            )

        with self._lock:
            self._attempt_outcomes[outcome] += 1
            self._attempts_by_model[model] += 1
            self._attempts_by_route[route] += 1
            if input_count is None:
                self._unknown_usage_attempts += 1
            else:
                assert output_count is not None
                self._known_usage_attempts += 1
                self._input_tokens += input_count
                self._output_tokens += output_count

    @staticmethod
    def _record_sample(
        sorted_samples: list[float], arrival_order: list[float], value: float
    ) -> None:
        if len(arrival_order) >= MAX_LATENCY_SAMPLES:
            oldest = arrival_order.pop(0)
            # Removing the first equal value is equivalent for a duration window.
            sorted_samples.remove(oldest)
        arrival_order.append(value)
        insort(sorted_samples, value)

    def _percentile(self, fraction: float) -> float | None:
        return self._sample_percentile(self._latencies_ms, fraction)

    @staticmethod
    def _sample_percentile(samples: list[float], fraction: float) -> float | None:
        if not samples:
            return None
        # Nearest-rank, `ceil(p x N)`. With a handful of samples an interpolating
        # percentile invents a latency that was never observed, and an operator
        # comparing it against a real request would rightly not trust it.
        #
        # `ceil` rather than `round`: Python's `round` is banker's rounding, so
        # `round(2.5)` is 2. That silently returned the *40th* percentile of a
        # five-sample window when asked for the 50th — a wrong number that looked
        # entirely plausible, which is the worst kind.
        count = len(samples)
        rank = max(1, min(count, ceil(fraction * count)))
        return round(samples[rank - 1], 1)

    def _duration_snapshot(self, samples: list[float]) -> dict[str, float | int | None]:
        return {
            "p50": self._sample_percentile(samples, 0.50),
            "p95": self._sample_percentile(samples, 0.95),
            "samples": len(samples),
        }

    def snapshot(self, answers_remaining_today: int | None = None) -> dict[str, Any]:
        """A privacy-safe report. Contains no visitor content by construction."""
        with self._lock:
            answered = self._outcomes["answered"]
            not_covered = self._outcomes["not_covered"]
            unavailable = self._outcomes["unavailable"]
            total = answered + not_covered + unavailable
            resolved = answered + not_covered

            return {
                # Named so it cannot be misread as a lifetime total.
                "since": self._started.isoformat(),
                "note": (
                    "Aggregate counters for this process only. They reset when "
                    "the machine starts. No question text is recorded."
                ),
                "requests": total,
                "outcomes": {
                    "answered": answered,
                    "not_covered": not_covered,
                    "unavailable": unavailable,
                },
                # Of the requests that actually reached the corpus, how many did
                # it cover? Unavailable requests are excluded: they say nothing
                # about the corpus, and including them would make an outage look
                # like a content gap.
                "refusal_rate": (
                    round(not_covered / resolved, 3) if resolved else None
                ),
                "latency_ms": {
                    "p50": self._percentile(0.50),
                    "p95": self._percentile(0.95),
                    "samples": len(self._latencies_ms),
                },
                "admission": {
                    result: self._admissions[result] for result in _ADMISSION_RESULTS
                },
                "stages_ms": {
                    stage: self._duration_snapshot(self._stage_latencies_ms[stage])
                    for stage in _STAGES
                },
                "provider_attempts": {
                    "retries": 0,  # The only provider adapter disables SDK retries.
                    "total": sum(self._attempt_outcomes.values()),
                    "outcomes": {
                        outcome: self._attempt_outcomes[outcome]
                        for outcome in _ATTEMPT_OUTCOMES
                    },
                    "uncertain": self._attempt_outcomes["uncertain"],
                    "by_model": dict(self._attempts_by_model),
                    "by_route": {
                        route: self._attempts_by_route[route]
                        for route in ("primary", "fallback")
                    },
                    "usage": {
                        "input_tokens": (
                            self._input_tokens if self._known_usage_attempts else None
                        ),
                        "output_tokens": (
                            self._output_tokens if self._known_usage_attempts else None
                        ),
                        "known_attempts": self._known_usage_attempts,
                        "unknown_attempts": self._unknown_usage_attempts,
                    },
                },
                "identity": {
                    "model": self.identity.model if self.identity else None,
                    "corpus": self.identity.corpus if self.identity else None,
                    "prompt": self.identity.prompt if self.identity else None,
                },
                # Expected to be zero. It is reported rather than assumed
                # precisely because a number that stops being zero is the signal
                # that the citation guarantee has broken.
                "rejected_citations": self._rejected_citations,
                "answers_remaining_today": answers_remaining_today,
            }
