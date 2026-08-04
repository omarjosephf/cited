"""A hard ceiling on what the service may spend.

Rate limiting bounds how fast money leaves; it does not bound how much. A limit
of one question per five seconds still permits roughly seventeen thousand paid
calls a day, and an unauthenticated endpoint making paid calls is a financial
denial-of-service waiting to happen.

This is the control that makes the loss finite. It is deliberately crude: it
counts calls rather than tokens, because a count cannot be wrong in the
direction that costs money. Token accounting would be more accurate and would
fail open if the accounting itself had a bug.

**This is not the only defence, and should not be the last one.** The provider's
own spend cap is enforced by someone with no bug in this file. This exists so
the service stops before that limit is reached, with an explanation, rather than
failing with a billing error.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, date, datetime


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

    def refund(self) -> None:
        """Return a reservation when the call did not happen.

        Without this, a failed request would still consume budget, and a
        provider outage would silently burn the day's allowance without
        answering anything.
        """
        with self._lock:
            self._roll()
            self._used = max(0, self._used - 1)

    @property
    def used(self) -> int:
        with self._lock:
            self._roll()
            return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)
