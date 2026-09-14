"""Durable, content-free reservations for one explicitly pinned deployment.

Runtime never creates or repairs a ledger. Every attempted call retains its
entire conservative reservation, including successful calls; usage reports are
observations, not authority to replenish this budget. SQLite transactions share
the allowance across processes. A separate SQLite write lock admits one actual
worker across those processes without holding the accounting transaction open
over a network call. Both files must live on the same persistent local volume.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from assistant.budget import AttemptBudget
    from assistant.settings import Settings

from assistant.budget import BudgetExhausted

# Fixed reservation allowance, not a provider invoice guarantee. Activation
# requires verification of input/framing and all billed thinking/output bounds
# against current rates. Never reduce reservations using usage/cache discounts.
ATTEMPT_RESERVATION_MICRO_USD = 40_000


class BudgetUnavailable(RuntimeError):
    """Accounting cannot be trusted; no new paid work may begin."""


class BudgetCapacityError(RuntimeError):
    """The one shared worker is still occupied, even if its caller left."""


@dataclass(frozen=True)
class BudgetLimits:
    daily_attempts: int = 40
    monthly_attempts: int = 200
    daily_micro_usd: int = 400_000
    monthly_micro_usd: int = 2_000_000

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("budget limits must be positive integers")
        if self.daily_attempts > self.monthly_attempts:
            raise ValueError("daily attempts must fit within monthly attempts")
        if self.daily_micro_usd > self.monthly_micro_usd:
            raise ValueError("daily money must fit within monthly money")

    def encoded(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def _identity(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("ledger identity must be 64 lowercase hexadecimal characters")
    return value


def initialize_ledger(
    path: Path, ledger_id: str, limits: BudgetLimits, *, carried_attempts: int = 0
) -> None:
    """Explicit operator bootstrap only, never called by service startup.

    Refuse existing paths. Partial initialization is left visibly unusable;
    runtime must never interpret it as a fresh allowance. Restore/replacement
    requires conservative carry-forward through the operational runbook.
    """
    _identity(ledger_id)
    if type(carried_attempts) is not int or not 0 <= carried_attempts <= 100000:
        raise ValueError("invalid carry-forward attempt count")
    lock_path = path.with_name(path.name + ".admission")
    if path.exists() or lock_path.exists():
        raise ValueError("budget files already exist")
    for target in (path, lock_path):
        with target.open("xb"):
            pass
        with closing(sqlite3.connect(target)) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                "CREATE TABLE identity (ledger_id TEXT NOT NULL, limits TEXT NOT NULL, "
                "last_seen INTEGER NOT NULL CHECK(last_seen >= 0)) STRICT"
            )
            connection.execute(
                "INSERT INTO identity VALUES (?, ?, 0)", (ledger_id, limits.encoded())
            )
            if target == path:
                connection.execute(
                    "CREATE TABLE reservations (id INTEGER PRIMARY KEY, "
                    "day TEXT NOT NULL, month TEXT NOT NULL, "
                    "micro_usd INTEGER NOT NULL CHECK(micro_usd = 40000)) STRICT"
                )
                connection.execute("CREATE INDEX month_index ON reservations(month)")
                if carried_attempts:
                    now = datetime.now(UTC)
                    connection.executemany(
                        "INSERT INTO reservations(day, month, micro_usd) "
                        "VALUES (?, ?, ?)",
                        [
                            (
                                now.strftime("%Y-%m-%d"),
                                now.strftime("%Y-%m"),
                                ATTEMPT_RESERVATION_MICRO_USD,
                            )
                        ]
                        * carried_attempts,
                    )
                    connection.execute(
                        "UPDATE identity SET last_seen = ?", (int(now.timestamp()),)
                    )
            connection.execute("PRAGMA user_version=1")
            connection.commit()
        with target.open("r+b") as file:
            os.fsync(file.fileno())
    if os.name == "posix":
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class PersistentBudget:
    def __init__(
        self,
        path: Path,
        ledger_id: str,
        limits: BudgetLimits,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        try:
            self.path = path.resolve(strict=True)
        except OSError:
            raise BudgetUnavailable("budget_storage") from None
        self.lock_path = self.path.with_name(self.path.name + ".admission")
        self.ledger_id = _identity(ledger_id)
        self.limits = limits
        self._now = now
        # Pin file identities for this process. Unlink/recreate must not let two
        # processes lock different files or resume against an empty allowance.
        try:
            self._files = {
                target: (target.stat().st_dev, target.stat().st_ino)
                for target in (self.path, self.lock_path)
            }
        except OSError:
            raise BudgetUnavailable("budget_storage") from None
        for target in self._files:
            with closing(self._connect(target)) as connection:
                if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                    raise BudgetUnavailable("budget_integrity")
                self._check_identity(connection)

    def _connect(self, path: Path, *, lock: bool = False) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            for target, identity in self._files.items():
                stat = target.stat()
                if (stat.st_dev, stat.st_ino) != identity:
                    raise BudgetUnavailable("budget_replaced")
            connection = sqlite3.connect(
                path.as_uri() + "?mode=rw",
                uri=True,
                timeout=0.0 if lock else 0.1,
                isolation_level=None,
                check_same_thread=False,
            )
            connection.execute("PRAGMA synchronous=FULL")
            return connection
        except (OSError, sqlite3.Error):
            if connection is not None:
                connection.close()
            raise BudgetUnavailable("budget_storage") from None

    def _check_identity(self, connection: sqlite3.Connection) -> int:
        if connection.execute("PRAGMA user_version").fetchone() != (1,):
            raise BudgetUnavailable("budget_schema")
        rows = connection.execute(
            "SELECT ledger_id, limits, last_seen FROM identity"
        ).fetchall()
        if (
            len(rows) != 1
            or rows[0][:2] != (self.ledger_id, self.limits.encoded())
            or type(rows[0][2]) is not int
            or rows[0][2] < 0
        ):
            raise BudgetUnavailable("budget_identity_or_limits")
        return int(rows[0][2])

    def _totals(self, *, reserve: bool) -> tuple[int, int, int, int]:
        try:
            with closing(self._connect(self.path)) as connection:
                connection.execute("BEGIN IMMEDIATE")
                last_seen = self._check_identity(connection)
                now = self._now()
                if now.tzinfo is None or now.utcoffset() is None:
                    raise BudgetUnavailable("budget_clock")
                now = now.astimezone(UTC)
                stamp = int(now.timestamp())
                if stamp < last_seen:
                    raise BudgetUnavailable("budget_clock_regressed")
                day, month = now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")
                malformed = connection.execute(
                    "SELECT 1 FROM reservations WHERE micro_usd != ? "
                    "OR month != substr(day, 1, 7) "
                    "OR length(day) != 10 OR date(day, '+0 days') IS NOT day "
                    "OR day > ? LIMIT 1",
                    (ATTEMPT_RESERVATION_MICRO_USD, day),
                ).fetchone()
                if malformed is not None:
                    raise BudgetUnavailable("budget_accounting")
                counts = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(micro_usd), 0), "
                    "COALESCE(SUM(day = ?), 0), "
                    "COALESCE(SUM(CASE WHEN day = ? THEN micro_usd ELSE 0 END), 0) "
                    "FROM reservations WHERE month = ?",
                    (day, day, month),
                ).fetchone()
                if counts is None or any(
                    type(value) is not int or value < 0 for value in counts
                ):
                    raise BudgetUnavailable("budget_accounting")
                month_calls, month_money, day_calls, day_money = counts
                if (
                    month_money != month_calls * ATTEMPT_RESERVATION_MICRO_USD
                    or day_money != day_calls * ATTEMPT_RESERVATION_MICRO_USD
                ):
                    raise BudgetUnavailable("budget_accounting")
                remaining = min(
                    self.limits.daily_attempts - day_calls,
                    self.limits.monthly_attempts - month_calls,
                    (self.limits.daily_micro_usd - day_money)
                    // ATTEMPT_RESERVATION_MICRO_USD,
                    (self.limits.monthly_micro_usd - month_money)
                    // ATTEMPT_RESERVATION_MICRO_USD,
                )
                if reserve and remaining < 1:
                    raise BudgetExhausted("combined_budget_exhausted")
                if reserve:
                    connection.execute(
                        "INSERT INTO reservations(day, month, micro_usd) "
                        "VALUES (?, ?, ?)",
                        (day, month, ATTEMPT_RESERVATION_MICRO_USD),
                    )
                connection.execute("UPDATE identity SET last_seen = ?", (stamp,))
                connection.commit()  # The provider may only run after this succeeds.
                return day_calls, month_calls, month_money, max(0, remaining)
        except (OSError, sqlite3.Error):
            raise BudgetUnavailable("budget_storage_or_accounting") from None

    def spend(self) -> None:
        self._totals(reserve=True)

    @property
    def used(self) -> int:
        return self._totals(reserve=False)[0]

    @property
    def remaining(self) -> int:
        return self._totals(reserve=False)[3]

    def acquire_worker(self) -> Callable[[], None]:
        """Take a process-shared lock; release only when the real job exits.

        OS/SQLite releases it on process death. There is no lease timeout that
        could admit a replacement while a slow provider thread is still alive.
        """
        connection = self._connect(self.lock_path, lock=True)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._check_identity(connection)
        except sqlite3.OperationalError as error:
            connection.close()
            if error.sqlite_errorcode in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                raise BudgetCapacityError("shared_worker_busy") from None
            raise BudgetUnavailable("budget_admission_storage") from None
        except BaseException:
            connection.close()
            raise

        def release() -> None:
            try:
                connection.rollback()
            finally:
                connection.close()

        return release


def service_budget(settings: Settings) -> AttemptBudget:
    """Production cannot opt back into an ephemeral or independently replicated cap."""
    from assistant.budget import DailyCallBudget

    fly_app = os.environ.get("FLY_APP_NAME", "")
    if settings.budget_path is None:
        if fly_app or settings.answering_enabled:
            raise BudgetUnavailable("persistent_budget_required")
        return DailyCallBudget(settings.daily_answer_limit)
    path = settings.budget_path
    if (fly_app or settings.budget_machine_id) and (
        not settings.budget_machine_id
        or os.environ.get("FLY_MACHINE_ID") != settings.budget_machine_id
        or not path.is_absolute()
        or path.is_symlink()
        or path.with_name(path.name + ".admission").is_symlink()
        or path.parent != Path("/data")
        or not os.path.ismount(path.parent)
    ):
        raise BudgetUnavailable("budget_deployment_or_mount")
    return PersistentBudget(
        path,
        settings.budget_ledger_id,
        BudgetLimits(
            settings.daily_answer_limit,
            settings.monthly_answer_limit,
            settings.daily_budget_micro_usd,
            settings.monthly_budget_micro_usd,
        ),
    )


def main() -> None:
    """Bootstrap only through a deliberate operator command; never starts inference."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Initialize a new content-free budget ledger"
    )
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--ledger-id", required=True)
    parser.add_argument(
        "--carry-forward-attempts",
        type=int,
        required=True,
        help="reviewed prior reservations rounded up; all charged to today",
    )
    # The limits are written into the ledger at creation and `PersistentBudget`
    # refuses at runtime on any mismatch with the ones it is constructed with.
    # They default to the live service envelope, so the bootstrap command in the
    # runbook keeps its current meaning; a capture-scoped ledger must state its
    # own, because the ledger cannot be recreated to correct them afterwards.
    defaults = BudgetLimits()
    for flag, value in (
        ("--daily-attempts", defaults.daily_attempts),
        ("--monthly-attempts", defaults.monthly_attempts),
        ("--daily-micro-usd", defaults.daily_micro_usd),
        ("--monthly-micro-usd", defaults.monthly_micro_usd),
    ):
        parser.add_argument(
            flag,
            type=int,
            default=value,
            help=f"permanently stamped into the ledger; live default {value}",
        )
    args = parser.parse_args()
    limits = BudgetLimits(
        args.daily_attempts,
        args.monthly_attempts,
        args.daily_micro_usd,
        args.monthly_micro_usd,
    )
    initialize_ledger(
        args.path,
        args.ledger_id,
        limits,
        carried_attempts=args.carry_forward_attempts,
    )
    print("Budget ledger initialized. No provider request was made.")
    print(f"Permanent limits: {limits.encoded()}")


if __name__ == "__main__":
    main()
