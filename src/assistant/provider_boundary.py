"""Bound provider HTTP bodies before the SDK parses success or error JSON.

This is ordinary non-streaming inference. The HTTP byte iterator is only a
resource boundary; no tokens or partial answers are sent to the browser.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import httpx
from anthropic.types import Message
from pydantic import ValidationError

MAX_PROVIDER_BODY_BYTES = 48 * 1024


class ProviderResponseRejected(RuntimeError):
    """Fixed error category; contains no provider content."""


class BoundedProviderStream(httpx.SyncByteStream):
    def __init__(self, stream: httpx.SyncByteStream, deadline: float) -> None:
        self.stream = stream
        self.deadline = deadline

    def __iter__(self) -> Iterator[bytes]:
        size = 0
        for chunk in self.stream:
            size += len(chunk)
            if time.monotonic() >= self.deadline:
                raise ProviderResponseRejected("provider_body_deadline")
            if size > MAX_PROVIDER_BODY_BYTES:
                raise ProviderResponseRejected("provider_body_limit")
            if chunk:
                yield chunk

    def close(self) -> None:
        self.stream.close()


def start_deadline(request: httpx.Request) -> None:
    timeout = request.extensions.get("timeout", {}).get("read", 6.0)
    request.extensions["assistant_body_deadline"] = time.monotonic() + float(timeout)


def bound_response(response: httpx.Response) -> None:
    try:
        if response.headers.get("content-encoding", "identity").lower() != "identity":
            raise ProviderResponseRejected("provider_body_encoding")
        length = response.headers.get("content-length")
        if length is not None and int(length) > MAX_PROVIDER_BODY_BYTES:
            raise ProviderResponseRejected("provider_body_limit")
        deadline = float(response.request.extensions["assistant_body_deadline"])
        if time.monotonic() >= deadline:
            raise ProviderResponseRejected("provider_body_deadline")
        if response.is_stream_consumed:
            if len(response.content) > MAX_PROVIDER_BODY_BYTES:
                raise ProviderResponseRejected("provider_body_limit")
        else:
            assert isinstance(response.stream, httpx.SyncByteStream)
            response.stream = BoundedProviderStream(response.stream, deadline)
        response.read()  # Bound before either strict or permissive SDK parsing.
        if response.is_success:
            try:
                Message.model_validate_json(response.content, strict=True)
            except ValidationError:
                raise ProviderResponseRejected("provider_schema") from None
        if time.monotonic() >= deadline:
            raise ProviderResponseRejected("provider_body_deadline")
    except BaseException:
        response.close()
        raise


def bounded_http_client(timeout: float) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        headers={"Accept-Encoding": "identity"},
        event_hooks={"request": [start_deadline], "response": [bound_response]},
        follow_redirects=False,
    )
