"""Synthetic HTTP stalls exercise the real admission/cancellation boundary."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.types import Message
from test_api import StubAnswerer, grounded_answer
from test_api import client as client

from assistant import api
from assistant.budget import DailyCallBudget
from assistant.runtime import (
    BoundedAnswerExecutor,
    ExecutionContext,
    ExecutorCapacityError,
)
from assistant.transport import UnsupportedResponse


@pytest.mark.parametrize("workers", [1, 3, 5])
def test_stalled_jobs_keep_slots_after_timeout_and_health_recovers(
    client: TestClient, workers: int
) -> None:
    api.state.executor.shutdown()
    api.state.executor = BoundedAnswerExecutor(workers)
    release = threading.Event()
    all_started = threading.Event()
    lock = threading.Lock()
    calls = 0

    class Blocking:
        def answer(self, *args: Any, context: ExecutionContext, **kwargs: Any) -> Any:
            nonlocal calls
            context.begin_provider()
            with lock:
                calls += 1
                if calls == workers:
                    all_started.set()
            try:
                assert release.wait(4), "test did not release controlled provider"
                context.finish_provider("completed", input_tokens=4, output_tokens=2)
                return grounded_answer()
            except BaseException:
                if context.status().provider_outcome == "in_progress":
                    context.finish_provider("uncertain")
                raise

    api.state.answerer = Blocking()  # type: ignore[assignment]

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test"
        ) as http:
            pending = [
                asyncio.create_task(
                    http.post(
                        "/ask",
                        json={"question": "synthetic"},
                        headers={"X-Assistant-Deadline-Ms": "900"},
                    )
                )
                for _ in range(workers)
            ]
            assert await asyncio.to_thread(all_started.wait, 2)
            start = time.monotonic()
            assert (await http.get("/health")).status_code == 200
            assert time.monotonic() - start < 0.3
            assert (
                await http.post("/ask", json={"question": "over capacity"})
            ).status_code == 503
            timed_out = await asyncio.gather(*pending)
            assert all(response.status_code == 504 for response in timed_out)
            assert calls == workers
            assert api.state.budget.used == workers
            assert (
                await http.post("/ask", json={"question": "still occupied"})
            ).status_code == 503
            release.set()
            # Probe the same executor until the actual worker exits. No replacement
            # executor can disguise a premature release or leaked capacity.
            until = time.monotonic() + 2
            while True:
                try:
                    probe = api.state.executor.submit(
                        lambda job: (UnsupportedResponse(model_route=None), 0, None),
                        ExecutionContext(deadline=until, budget=api.state.budget),
                    )
                    assert await asyncio.wrap_future(probe)
                    break
                except ExecutorCapacityError:
                    assert time.monotonic() < until
                    await asyncio.sleep(0.01)
            api.state.answerer = StubAnswerer()  # type: ignore[assignment]
            recovered = await http.post("/ask", json={"question": "recovered"})
            assert recovered.status_code == 200
            # One freed slot is enough for the probe, while the other timed-out
            # jobs can still be completing their metrics finally blocks. Wait for
            # the behavior under test rather than assuming completion order.
            metrics_deadline = time.monotonic() + 2
            while True:
                snapshot = (await http.get("/metrics")).json()
                if snapshot["provider_attempts"]["total"] == workers + 1:
                    break
                assert time.monotonic() < metrics_deadline
                await asyncio.sleep(0.01)
            assert snapshot["provider_attempts"]["total"] == workers + 1
            assert snapshot["admission"]["rejected_full"] == 2

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_authentication_precedes_body_consumption(client: TestClient) -> None:
    api.state.settings.require_shared_secret = True
    api.state.settings.shared_secret = SecretStr("correct")

    async def scenario() -> None:
        async def never_read() -> Any:
            pytest.fail("unauthenticated body was consumed")
            yield b"unreachable"

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test"
        ) as http:
            response = await http.post("/ask", content=never_read())
            assert response.status_code == 401
            assert response.headers["cache-control"] == "no-store"

    asyncio.run(scenario())


def test_stalled_and_oversized_request_bodies_do_not_dispatch(
    client: TestClient,
) -> None:
    async def scenario() -> None:
        async def stalled() -> Any:
            yield b'{"question":"'
            await asyncio.sleep(2)
            yield b'synthetic"}'

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test"
        ) as http:
            response = await http.post(
                "/ask", content=stalled(), headers={"X-Assistant-Deadline-Ms": "600"}
            )
            assert response.status_code == 504
            response = await http.post("/ask", content=b"x" * 8193)
            assert response.status_code == 413
            assert api.state.budget.used == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("header", ["NaN", "inf", "0", "-1", "9001", "invalid"])
def test_deadline_header_cannot_expand_budget(client: TestClient, header: str) -> None:
    response = client.post(
        "/ask", json={"question": "q"}, headers={"X-Assistant-Deadline-Ms": header}
    )
    assert response.status_code == 422
    assert api.state.budget.used == 0


def test_no_call_policy_does_not_spend_and_errors_never_refund_or_log_content(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    class NoCall:
        def answer(self, *args: Any, **kwargs: Any) -> Any:
            from assistant.answering import Answer

            return Answer("fixed", (), False, ())

    api.state.answerer = NoCall()  # type: ignore[assignment]
    assert client.post("/ask", json={"question": "q"}).status_code == 200
    assert api.state.budget.used == 0
    api.state.budget = DailyCallBudget(1)
    secret = "SYNTHETIC-SENSITIVE-PROVIDER-PAYLOAD"
    api.state.answerer = StubAnswerer(error=RuntimeError(secret))  # type: ignore[assignment]
    with caplog.at_level(logging.INFO, logger="assistant"):
        assert client.post("/ask", json={"question": secret}).status_code == 502
        assert client.post("/ask", json={"question": secret}).status_code == 503
    assert secret not in caplog.text
    assert all(
        record.exc_info is None
        for record in caplog.records
        if record.name.startswith("assistant")
    )
    snapshot = client.get("/metrics").json()
    assert snapshot["provider_attempts"]["uncertain"] == 1
    assert snapshot["provider_attempts"]["usage"]["unknown_attempts"] == 1
    assert api.state.budget.used == 1


def test_cancelled_http_waiter_keeps_slot_and_cannot_start_a_late_provider(
    client: TestClient,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    contexts: list[ExecutionContext] = []

    class BeforeProvider:
        def answer(self, *args: Any, context: ExecutionContext, **kwargs: Any) -> Any:
            contexts.append(context)
            entered.set()
            assert release.wait(3)
            context.begin_provider()  # Must refuse a cancelled request.
            pytest.fail("provider began after HTTP cancellation")

    api.state.answerer = BeforeProvider()  # type: ignore[assignment]

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test"
        ) as http:
            pending = asyncio.create_task(
                http.post("/ask", json={"question": "cancel"})
            )
            assert await asyncio.to_thread(entered.wait, 1)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert contexts[0].cancellation.is_set()
            assert (await http.get("/health")).status_code == 200
            assert (
                await http.post("/ask", json={"question": "occupied"})
            ).status_code == 503
            assert api.state.budget.used == 0
            release.set()
            await asyncio.to_thread(api.state.executor.shutdown)
            assert api.state.budget.used == 0

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_explicit_http_disconnect_suppresses_output_and_retains_capacity(
    client: TestClient,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    contexts: list[ExecutionContext] = []

    class Blocking:
        def answer(self, *args: Any, context: ExecutionContext, **kwargs: Any) -> Any:
            contexts.append(context)
            context.begin_provider()
            entered.set()
            assert release.wait(3)
            context.finish_provider("completed", input_tokens=1, output_tokens=1)
            return grounded_answer()

    api.state.answerer = Blocking()  # type: ignore[assignment]

    async def scenario() -> None:
        delivered = False
        sent: list[Message] = []

        async def receive() -> dict[str, Any]:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {
                    "type": "http.request",
                    "body": b'{"question":"synthetic"}',
                    "more_body": False,
                }
            assert await asyncio.to_thread(entered.wait, 1)
            return {"type": "http.disconnect"}

        async def send(message: Message) -> None:
            sent.append(message)

        scope: dict[str, Any] = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/ask",
            "raw_path": b"/ask",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234),
            "server": ("test", 80),
        }
        await asyncio.wait_for(api.app(scope, receive, send), 2)
        assert contexts[0].cancellation.is_set()
        assert not any(b'"answered"' in message.get("body", b"") for message in sent)
        assert api.state.budget.used == 1
        with pytest.raises(ExecutorCapacityError):
            api.state.executor.submit(
                lambda job: (UnsupportedResponse(model_route=None), 0, None),
                ExecutionContext(
                    deadline=time.monotonic() + 1, budget=api.state.budget
                ),
            )
        release.set()
        await asyncio.to_thread(api.state.executor.shutdown)
        assert api.state.budget.used == 1

    try:
        asyncio.run(scenario())
    finally:
        release.set()
