"""An in-memory limit on provider attempts per process and UTC day.

Rate limiting bounds how fast money leaves; it does not bound how much. A limit
of one question per five seconds still permits roughly seventeen thousand paid
calls a day, and an unauthenticated endpoint making paid calls is a financial
denial-of-service waiting to happen.

This counts dispatch attempts, not money. It resets on restart and does not
coordinate replicas; it cannot enforce a durable daily or monthly spend cap.
Uncertain attempts keep their reservations. Production also requires verified
provider spending controls and durable shared admission/accounting where the
deployment can restart or scale. Provider caps may themselves have enforcement
delay, so retain headroom rather than promising zero possible overage.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Protocol


class AttemptBudget(Protocol):
    """The reservation boundary shared by runtime and offline test budgets."""

    def spend(self) -> None: ...

    @property
    def used(self) -> int: ...

    @property
    def remaining(self) -> int: ...


class BudgetExhausted(RuntimeError):
    """The daily ceiling has been reached. Raised instead of making the call."""


@dataclass
class DailyCallBudget:
    """Counts paid calls per UTC day, and refuses once the ceiling is reached.

    In-process and in-memory, which has two consequences worth stating rather
    than discovering:

    * **It resets on restart.** A crash loop could spend several days' budget in
      an afternoon. The provider-side cap is what bounds that.
    * **It does not span replicas.** Two instances have two budgets. For a
      single-container demo that is the whole system; past that it needs shared
      state, and pretending otherwise would be worse than saying so.

    UTC rather than local time so the reset point does not move twice a year,
    which would otherwise produce one 23-hour and one 25-hour day.
    """

    limit: int
    _used: int = field(default=0, init=False)
    _day: date = field(default_factory=lambda: datetime.now(UTC).date(), init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def _roll(self) -> None:
        today = datetime.now(UTC).date()
        if today != self._day:
            self._day = today
            self._used = 0

    def spend(self) -> None:
        """Record one paid call, or raise if that would exceed the ceiling.

        Reserves *before* the call rather than recording after it. Recording
        afterwards would let an unbounded number of concurrent requests all pass
        the check and then all spend, which is precisely the burst this exists
        to prevent.
        """
        with self._lock:
            self._roll()
            if self._used >= self.limit:
                raise BudgetExhausted(
                    f"daily limit of {self.limit} answered questions reached"
                )
            self._used += 1

    @property
    def used(self) -> int:
        with self._lock:
            self._roll()
            return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)
