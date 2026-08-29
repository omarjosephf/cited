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
from math import ceil
from typing import Any, Literal

Outcome = Literal["answered", "not_covered", "unavailable"]
"""The three states a visitor can observe. Metrics use the same vocabulary the
interface does, so an operator reading a count knows exactly what a visitor saw.
"""

MAX_LATENCY_SAMPLES = 2048
"""Cap on retained latency samples, so memory cannot grow without bound.

Percentiles are computed from a sorted window rather than a streaming estimator:
at this volume the exact answer is cheap, and an approximate one would be harder
to explain than it is worth. When the window is full the oldest sample is
dropped, which biases percentiles toward recent behaviour — the right bias for
"is it healthy now?".
"""


@dataclass
class AssistantMetrics:
    """Counters and latency percentiles for one process lifetime.

    Every method takes the lock. Contention is irrelevant at this scale, and the
    alternative — reasoning about which counter updates are atomic under which
    interpreter — is a false economy in code whose whole job is to be trusted.
    """

    _outcomes: Counter[str] = field(default_factory=Counter, init=False)
    _latencies_ms: list[float] = field(default_factory=list, init=False)
    _order: list[float] = field(default_factory=list, init=False)
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
        with self._lock:
            self._outcomes[outcome] += 1
            self._rejected_citations += rejected_citations

            if len(self._order) >= MAX_LATENCY_SAMPLES:
                oldest = self._order.pop(0)
                # `list.remove` deletes the first equal value, which for a sorted
                # window of durations is indistinguishable from the intended one.
                self._latencies_ms.remove(oldest)
            self._order.append(latency_ms)
            insort(self._latencies_ms, latency_ms)

    def _percentile(self, fraction: float) -> float | None:
        if not self._latencies_ms:
            return None
        # Nearest-rank, `ceil(p x N)`. With a handful of samples an interpolating
        # percentile invents a latency that was never observed, and an operator
        # comparing it against a real request would rightly not trust it.
        #
        # `ceil` rather than `round`: Python's `round` is banker's rounding, so
        # `round(2.5)` is 2. That silently returned the *40th* percentile of a
        # five-sample window when asked for the 50th — a wrong number that looked
        # entirely plausible, which is the worst kind.
        count = len(self._latencies_ms)
        rank = max(1, min(count, ceil(fraction * count)))
        return round(self._latencies_ms[rank - 1], 1)

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
                # Expected to be zero. It is reported rather than assumed
                # precisely because a number that stops being zero is the signal
                # that the citation guarantee has broken.
                "rejected_citations": self._rejected_citations,
                "answers_remaining_today": answers_remaining_today,
            }
