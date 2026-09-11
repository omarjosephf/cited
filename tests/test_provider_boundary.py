"""No network: the real SDK consumes a controlled HTTP byte stream."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

import httpx
import pytest
from anthropic import Anthropic, APIConnectionError

from assistant.provider_boundary import (
    MAX_PROVIDER_BODY_BYTES,
    BoundedProviderStream,
    ProviderResponseRejected,
    bound_response,
    start_deadline,
)


class Bytes(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield from self.chunks

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("status", [200, 400, 500])
def test_sdk_success_and_error_bodies_are_bounded_without_retry(status: int) -> None:
    body = Bytes([b"x" * 8192] * 7)
    attempts = 0

    def send(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(status, stream=body)

    with (
        httpx.Client(
            transport=httpx.MockTransport(send),
            event_hooks={"request": [start_deadline], "response": [bound_response]},
        ) as http,
        Anthropic(
            api_key="synthetic", http_client=http, max_retries=0, timeout=6
        ) as sdk,
        pytest.raises(APIConnectionError) as caught,
    ):
        sdk.messages.create(
            model="claude-haiku-4-5",
            max_tokens=10,
            messages=[{"role": "user", "content": "synthetic"}],
        )
    assert isinstance(caught.value.__cause__, ProviderResponseRejected)
    assert str(caught.value.__cause__) == "provider_body_limit"
    assert attempts == 1
    assert body.closed


def test_success_is_parsed_by_sdk_and_connection_closes() -> None:
    payload = {
        "id": "synthetic",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [{"type": "text", "text": "synthetic", "citations": []}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    body = Bytes([json.dumps(payload).encode()])
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, headers={"Content-Type": "application/json"}, stream=body
                )
            ),
            event_hooks={"request": [start_deadline], "response": [bound_response]},
        ) as http,
        Anthropic(
            api_key="synthetic", http_client=http, max_retries=0, timeout=6
        ) as sdk,
    ):
        result = sdk.messages.create(
            model="claude-haiku-4-5",
            max_tokens=10,
            messages=[{"role": "user", "content": "synthetic"}],
        )
        assert result.stop_reason == "end_turn"
    assert body.closed


@pytest.mark.parametrize(
    "headers",
    [
        {"Content-Length": str(MAX_PROVIDER_BODY_BYTES + 1)},
        {"Content-Encoding": "gzip"},
    ],
)
def test_unbounded_or_compressed_body_is_rejected_before_read(
    headers: dict[str, str],
) -> None:
    body = Bytes([])
    request = httpx.Request("POST", "https://synthetic.invalid")
    start_deadline(request)
    response = httpx.Response(200, request=request, headers=headers, stream=body)
    with pytest.raises(ProviderResponseRejected):
        bound_response(response)
    assert body.closed


def test_body_deadline_is_checked_while_bytes_arrive() -> None:
    body = Bytes([b"x"])
    request = httpx.Request("POST", "https://synthetic.invalid")
    start_deadline(request)
    request.extensions["assistant_body_deadline"] = time.monotonic() - 1
    response = httpx.Response(200, request=request, stream=body)
    with pytest.raises(ProviderResponseRejected, match="provider_body_deadline"):
        bound_response(response)
    assert body.closed


def test_late_body_chunk_is_never_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([1.0, 7.0])
    monkeypatch.setattr(
        "assistant.provider_boundary.time.monotonic", lambda: next(clock)
    )
    stream = BoundedProviderStream(Bytes([b"first", b"late"]), deadline=6.0)
    chunks = iter(stream)
    assert next(chunks) == b"first"
    with pytest.raises(ProviderResponseRejected, match="provider_body_deadline"):
        next(chunks)
