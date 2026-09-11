"""Authenticate /ask before reading or parsing its bounded request body."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable

from fastapi import HTTPException, Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from assistant.settings import Settings
from assistant.web_security import apply_security_headers

MAX_BODY_BYTES = 8 * 1024


class AskBoundaryMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        settings: Callable[[], Settings],
        authenticate: Callable[[Request], None],
    ) -> None:
        self.app = app
        self.settings = settings
        self.authenticate = authenticate

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("path") != "/ask"
            or scope.get("method") != "POST"
        ):
            await self.app(scope, receive, send)
            return

        async def reject(detail: str, status: int) -> None:
            response = JSONResponse({"detail": detail}, status_code=status)
            apply_security_headers(response, no_store=True)
            await response(scope, receive, send)

        request = Request(scope)
        try:
            self.authenticate(request)
        except HTTPException as exc:
            await reject(str(exc.detail), exc.status_code)
            return
        started = time.monotonic()
        deadline = started + self.settings().backend_timeout_seconds
        remaining_header = request.headers.get("X-Assistant-Deadline-Ms")
        if remaining_header is not None:
            try:
                remaining_ms = float(remaining_header)
            except ValueError:
                remaining_ms = 0
            if not math.isfinite(remaining_ms) or not 0 < remaining_ms <= 9000:
                await reject("Invalid request.", 422)
                return
            deadline = min(deadline, started + remaining_ms / 1000 - 0.5)
        scope.setdefault("state", {})["answer_deadline"] = deadline
        chunks: list[bytes] = []
        size = 0
        while True:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                message = await asyncio.wait_for(receive(), timeout=remaining)
            except TimeoutError:
                await reject("Service is unavailable.", 504)
                return
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                await reject("Request is too large.", 413)
                return
            if chunk:
                chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def bounded_receive() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)
