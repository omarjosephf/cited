"""Opt-in shared qualification admission. Startup never provisions an allowance.

Not connected to service_budget or capture CLI. A binding must be reviewed and
provisioned separately; the existing real SQLite path remains authoritative.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

import httpx

from assistant.budget import BudgetExhausted
from assistant.persistent_budget import BudgetCapacityError, BudgetUnavailable
from assistant.runtime import BoundedAnswerExecutor, ExecutionContext

_TOP = {
    "version",
    "reservation_micro_usd",
    "aggregate_cap_micro_usd",
    "aggregate_carried_micro_usd",
    "max_active",
    "services",
}
_SERVICE = {
    "ledger_id",
    "daily_attempts",
    "monthly_attempts",
    "daily_micro_usd",
    "monthly_micro_usd",
    "lifetime_micro_usd",
    "carried_lifetime_micro_usd",
    "carried_day_micro_usd",
    "carried_month_micro_usd",
}


def _identity(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("invalid budget identity")
    return value


def _integer(value: Any, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def validate_policy(p: Any) -> None:
    """Validate the entire two-service policy, including conservative carry."""
    if not isinstance(p, dict) or set(p) != _TOP:
        raise ValueError("invalid shared policy")
    if (
        not _integer(p["version"], 1, 1)
        or not _integer(p["reservation_micro_usd"], 40000, 40000)
        or not _integer(p["aggregate_cap_micro_usd"], 40000, 3000000)
        or not _integer(p["aggregate_carried_micro_usd"], 0, 1000000000)
        or not _integer(p["max_active"], 1, 4)
        or not isinstance(p["services"], dict)
        or set(p["services"]) != {"ev", "cited"}
    ):
        raise ValueError("invalid shared policy")
    for s in p["services"].values():
        if not isinstance(s, dict) or set(s) != _SERVICE:
            raise ValueError("invalid service policy")
        _identity(s["ledger_id"])
        for key in _SERVICE - {"ledger_id"}:
            if not _integer(s[key], 0, 1000000000):
                raise ValueError("invalid service amount")
        if (
            not 1 <= s["daily_attempts"] <= s["monthly_attempts"] <= 200
            or s["daily_attempts"] > 40
            or not 40000 <= s["daily_micro_usd"] <= 400000
            or not s["daily_micro_usd"] <= s["monthly_micro_usd"] <= 2000000
            or not 40000 <= s["lifetime_micro_usd"] <= 3000000
            or not s["carried_day_micro_usd"]
            <= s["carried_month_micro_usd"]
            <= s["carried_lifetime_micro_usd"]
        ):
            raise ValueError("invalid service limits")
    if p["services"]["ev"]["ledger_id"] == p["services"]["cited"]["ledger_id"]:
        raise ValueError("service identities must differ")


@dataclass(frozen=True)
class BudgetBinding:
    ledger_id: str
    carry_forward_sha256: str
    policy_json: str

    def __post_init__(self) -> None:
        _identity(self.ledger_id)
        _identity(self.carry_forward_sha256)
        policy = json.loads(self.policy_json)
        validate_policy(policy)
        object.__setattr__(
            self,
            "policy_json",
            json.dumps(policy, sort_keys=True, separators=(",", ":")),
        )

    @property
    def config_id(self) -> str:
        return hashlib.sha256(self.policy_json.encode()).hexdigest()

    @property
    def policy(self) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(self.policy_json)
        return result


class BudgetRpc(Protocol):
    def call(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class SupabaseBudgetRpc:
    """Bounded HTTPS POST, no retries, redirects, env proxy or response logging."""

    def __init__(self, origin: str, secret: str) -> None:
        if not re.fullmatch(r"https://[a-z0-9]{20}\.supabase\.co", origin):
            raise ValueError("explicit managed Supabase origin required")
        if not secret.startswith("sb_secret_") or len(secret) > 1024:
            raise ValueError("server secret required")
        self._url = origin + "/rest/v1/rpc/ev_budget_admission"
        self._client = httpx.Client(
            headers={
                "apikey": secret,
                "Content-Type": "application/json",
                "Accept-Encoding": "identity",
            },
            follow_redirects=False,
            trust_env=False,
            transport=httpx.HTTPTransport(retries=0),
            timeout=httpx.Timeout(2.8, read=0.5),
        )

    def close(self) -> None:
        self._client.close()

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        deadline = time.monotonic() + 2.8
        try:
            with self._client.stream("POST", self._url, json=payload) as response:
                if response.status_code != 200:
                    raise BudgetUnavailable("shared_rpc_denied")
                if response.headers.get("content-encoding", "identity") != "identity":
                    raise BudgetUnavailable("shared_rpc_encoding")
                body = bytearray()
                for chunk in response.iter_raw():
                    if time.monotonic() >= deadline or len(body) + len(chunk) > 8192:
                        raise BudgetUnavailable("shared_rpc_bounds")
                    body.extend(chunk)
            if time.monotonic() >= deadline:
                raise BudgetUnavailable("shared_rpc_deadline")
            result = json.loads(body)
            if not isinstance(result, dict):
                raise BudgetUnavailable("shared_rpc_shape")
            return result
        except (httpx.HTTPError, ValueError, UnicodeError):
            raise BudgetUnavailable("shared_rpc_uncertain") from None


class SharedJobBudget:
    """One unreplayable execution identity, for either E.V or Cited.

    All exceptions/unknown acknowledgments poison this object. Reconstructing the
    same job never grants replay permission. Only the worker completion callback
    may release occupancy. No release ever refunds a reservation.
    """

    def __init__(
        self, rpc: BudgetRpc, binding: BudgetBinding, service: str, maximum: int
    ) -> None:
        if service not in {"ev", "cited"} or not _integer(maximum, 1, 2):
            raise ValueError("invalid shared job")
        self._rpc, self.binding, self.service = rpc, binding, service
        self.maximum = maximum
        self.job_id, self._owner_id = str(uuid4()), str(uuid4())
        self._lock = threading.Lock()
        self._used = 0
        self._acquired = False
        self._closed = False
        self._uncertain = False
        self.release_failed = False

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    def _call(self, action: str, ordinal: int = 0) -> dict[str, Any]:
        expected = {
            "ledger_id": self.binding.ledger_id,
            "config_id": self.binding.config_id,
            "job_id": self.job_id,
            "owner_id": self._owner_id,
            "ordinal": ordinal,
        }
        payload = {
            **{"p_" + key: value for key, value in expected.items()},
            "p_carry_forward_sha256": self.binding.carry_forward_sha256,
            "p_policy": self.binding.policy,
            "p_service": self.service,
            "p_maximum": self.maximum,
            "p_action": action,
        }
        try:
            result = self._rpc.call(payload)
            if (
                set(result) != set(expected) | {"status", "used", "remaining"}
                or any(result[key] != value for key, value in expected.items())
                or type(result["ordinal"]) is not int
                or not _integer(result["used"], 0, 1000000000)
                or not _integer(result["remaining"], 0, 75)
                or result["status"]
                not in {
                    "ok",
                    "acquired",
                    "reserved",
                    "released",
                    "busy",
                    "duplicate",
                    "closed",
                    "exhausted",
                }
            ):
                raise BudgetUnavailable("shared_acknowledgment")
            return result
        except Exception:
            self._uncertain = True
            raise BudgetUnavailable("shared_admission_uncertain") from None

    def _check(self) -> None:
        if self._uncertain or self._closed:
            raise BudgetUnavailable("shared_job_unavailable")

    @property
    def remaining(self) -> int:
        with self._lock:
            self._check()
            result = self._call("inspect")
            if result["status"] != "ok":
                self._uncertain = True
                raise BudgetUnavailable("shared_inspection")
            return min(self.maximum - self._used, int(result["remaining"]))

    def acquire(self) -> None:
        with self._lock:
            self._check()
            if self._acquired:
                raise BudgetUnavailable("shared_job_already_acquired")
            result = self._call("acquire")
            if result["status"] == "busy":
                self._closed = True
                raise BudgetCapacityError("shared_worker_busy")
            if result["status"] == "exhausted":
                self._closed = True
                raise BudgetExhausted("shared_allowance_exhausted")
            if result["status"] != "acquired":
                self._uncertain = True
                raise BudgetUnavailable("shared_acquisition_not_fresh")
            self._acquired = True

    def spend(self) -> None:
        with self._lock:
            self._check()
            if not self._acquired:
                raise BudgetUnavailable("shared_job_not_acquired")
            if self._used >= self.maximum:
                raise BudgetExhausted("shared_job_attempt_limit")
            result = self._call("reserve", self._used + 1)
            if result["status"] == "exhausted":
                raise BudgetExhausted("shared_allowance_exhausted")
            if result["status"] != "reserved":
                self._uncertain = True
                raise BudgetUnavailable("shared_reservation_not_fresh")
            self._used += 1

    def _release_after_exit(self) -> None:
        with self._lock:
            if self._closed or not self._acquired:
                return
            try:
                if self._call("release")["status"] != "released":
                    raise BudgetUnavailable("shared_release")
            except BudgetUnavailable:
                self.release_failed = True
            finally:
                self._closed = True


def submit_shared_job[T](
    executor: BoundedAnswerExecutor[T],
    function: Callable[[ExecutionContext], T],
    context: ExecutionContext,
) -> Future[T]:
    """Release only on actual Future completion, never on an HTTP wait timeout."""
    budget = context.budget
    if not isinstance(budget, SharedJobBudget):
        raise TypeError("shared job context required")
    context.raise_if_stopped()
    budget.acquire()
    try:
        context.raise_if_stopped()
        future = executor.submit(function, context)
    except BaseException:
        budget._release_after_exit()  # No work was submitted.
        raise
    future.add_done_callback(lambda _future: budget._release_after_exit())
    return future
