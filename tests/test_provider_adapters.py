"""Offline contract tests for the Luna and Gemini HTTP adapters."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from assistant.provider_adapters import (
    GEMINI_MODEL,
    OPENAI_MODEL,
    GeminiAdapter,
    OpenAIAdapter,
    PreparedPost,
    ProviderResponseRejected,
    ProviderUnavailable,
    _bounded_post,
)
from assistant.provider_boundary import MAX_PROVIDER_BODY_BYTES

EVIDENCE = ({"id": "E01", "title": "Guide", "text": "OJ builds websites."},)
ANSWER = {
    "status": "answered",
    "blocks": [
        {
            "text": "OJ builds websites.",
            "citations": [{"source_id": "E01", "quote": "OJ builds websites."}],
        }
    ],
}


def prepared(handler: Callable[[httpx.Request], httpx.Response]) -> PreparedPost:
    return PreparedPost(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        url="https://provider.invalid/generate",
        body=b"{}",
    )


def openai_response(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "model": OPENAI_MODEL,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(ANSWER)}],
            }
        ],
        "usage": {"input_tokens": 17, "output_tokens": 9},
    }
    value.update(overrides)
    return value


def gemini_response(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "modelVersion": GEMINI_MODEL,
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {"parts": [{"text": json.dumps(ANSWER)}]},
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 8,
            "thoughtsTokenCount": 3,
        },
    }
    value.update(overrides)
    return value


def json_handler(
    payload: dict[str, Any], status: int = 200
) -> Callable[[httpx.Request], httpx.Response]:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.dumps(payload).encode("utf-8")
        return httpx.Response(
            status,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(body),
            request=request,
        )

    return handle


def test_openai_request_is_storeless_toolless_and_strict() -> None:
    post = OpenAIAdapter("secret", "project", "none").prepare(
        system="system", question="question", evidence=EVIDENCE, history="history"
    )
    try:
        payload = json.loads(post.body)
        assert payload["model"] == OPENAI_MODEL
        assert payload["store"] is False
        assert "tools" not in payload
        assert payload["reasoning"] == {"effort": "none"}
        assert payload["text"]["format"]["strict"] is True
        assert post.client.headers["authorization"] == "Bearer secret"
        assert post.client.headers["openai-project"] == "project"
    finally:
        post.close()


def test_both_adapters_normalize_valid_answers_and_usage() -> None:
    cases = (
        (OpenAIAdapter("unused", "project"), openai_response(), "primary", (17, 9)),
        (GeminiAdapter("unused"), gemini_response(), "fallback", (20, 11)),
    )
    for adapter, payload, _role, usage in cases:
        post = prepared(json_handler(payload))
        try:
            answer = adapter.generate(
                prepared=post,
                evidence=EVIDENCE,
                deadline=time.monotonic() + 5,
            )
        finally:
            post.close()
        assert answer.route is None  # The router assigns role after validation.
        assert (answer.input_tokens, answer.output_tokens) == usage
        assert answer.blocks[0].citations[0].quote == "OJ builds websites."


@pytest.mark.parametrize("status", [408, 500, 502, 503, 504])
def test_only_documented_http_availability_statuses_are_retryable(status: int) -> None:
    post = prepared(json_handler({"error": {}}, status))
    try:
        with pytest.raises(ProviderUnavailable, match=r"^provider_unavailable$"):
            _bounded_post(post, deadline=time.monotonic() + 5, before_dispatch=None)
    finally:
        post.close()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409])
def test_other_http_errors_are_not_availability_failures(status: int) -> None:
    post = prepared(json_handler({"error": {"message": "sensitive"}}, status))
    try:
        with pytest.raises(ProviderResponseRejected, match=r"^provider_http_rejected$"):
            _bounded_post(post, deadline=time.monotonic() + 5, before_dispatch=None)
    finally:
        post.close()


@pytest.mark.parametrize(
    ("payload", "retryable"),
    [
        ({"error": {"code": "rate_limit_exceeded"}}, True),
        ({"error": {"status": "temporary_capacity_exhausted"}}, True),
        ({"error": {"code": "quota_exceeded"}}, False),
        ({"error": {"code": "billing_hard_limit_reached"}}, False),
        *[
            ({"error": {"code": code, "type": "rate_limit_exceeded"}}, False)
            for code in (
                "project_spend_limit_exceeded",
                "organization_spend_limit_exceeded",
                "organization_usage_limit_exceeded",
                "credit_balance_exhausted",
            )
        ],
        ({"error": {"code": "invalid_api_key"}}, False),
        ({"error": {"message": "rate limit"}}, False),
        (
            {
                "error": {
                    "code": "rate_limit_exceeded",
                    "details": [{"reason": "quota_exceeded"}],
                }
            },
            False,
        ),
    ],
)
def test_429_requires_a_bounded_temporary_code(
    payload: dict[str, Any], retryable: bool
) -> None:
    post = prepared(json_handler(payload, 429))
    expected = ProviderUnavailable if retryable else ProviderResponseRejected
    try:
        with pytest.raises(expected):
            _bounded_post(post, deadline=time.monotonic() + 5, before_dispatch=None)
    finally:
        post.close()


@pytest.mark.parametrize(
    ("content", "headers", "category"),
    [
        (b'{"model":"x","model":"y"}', {}, "provider_json"),
        (b"\xff", {}, "provider_utf8"),
        (b"{}", {"content-encoding": "gzip"}, "provider_body_encoding"),
        (b"{}", {"content-length": "invalid"}, "provider_content_length"),
    ],
)
def test_provider_bytes_and_json_fail_closed(
    content: bytes, headers: dict[str, str], category: str
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=httpx.ByteStream(content),
            headers=headers,
            request=request,
        )

    post = prepared(handle)
    try:
        with pytest.raises(ProviderResponseRejected, match=f"^{category}$"):
            _bounded_post(post, deadline=time.monotonic() + 5, before_dispatch=None)
    finally:
        post.close()


@pytest.mark.parametrize("announce_length", [False, True])
def test_provider_body_limit_applies_to_announced_and_streamed_bytes(
    announce_length: bool,
) -> None:
    body = b"x" * (MAX_PROVIDER_BODY_BYTES + 1)

    def handle(request: httpx.Request) -> httpx.Response:
        headers = {"content-length": str(len(body))} if announce_length else {}
        return httpx.Response(
            200,
            stream=httpx.ByteStream(body),
            headers=headers,
            request=request,
        )

    post = prepared(handle)
    try:
        with pytest.raises(ProviderResponseRejected, match="provider_body_limit"):
            _bounded_post(post, deadline=time.monotonic() + 5, before_dispatch=None)
    finally:
        post.close()


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({}, (None, None)),
        ({"totalTokenCount": 28}, (20, 8)),
        ({"totalTokenCount": 31}, (20, 11)),
        ({"totalTokenCount": 27}, (None, None)),
        ({"totalTokenCount": True}, (None, None)),
        ({"thoughtsTokenCount": 3, "totalTokenCount": 31}, (20, 11)),
        ({"thoughtsTokenCount": 3, "totalTokenCount": 30}, (None, None)),
        ({"thoughtsTokenCount": None, "totalTokenCount": 28}, (None, None)),
        ({"thoughtsTokenCount": -1}, (None, None)),
        ({"totalTokenCount": 1_000_021}, (None, None)),
    ],
)
def test_gemini_usage_requires_complete_consistent_accounting(
    extra: dict[str, Any], expected: tuple[int | None, int | None]
) -> None:
    payload = gemini_response(
        usageMetadata={"promptTokenCount": 20, "candidatesTokenCount": 8, **extra}
    )
    post = prepared(json_handler(payload))
    try:
        answer = GeminiAdapter("unused").generate(
            prepared=post, evidence=EVIDENCE, deadline=time.monotonic() + 5
        )
    finally:
        post.close()
    assert (answer.input_tokens, answer.output_tokens) == expected


def test_rejected_provider_payload_never_appears_in_error_or_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "SYNTHETIC-SENSITIVE-PROVIDER-PAYLOAD"
    post = prepared(json_handler({"error": {"message": sentinel}}, 401))
    try:
        with pytest.raises(ProviderResponseRejected) as caught:
            _bounded_post(post, deadline=time.monotonic() + 5, before_dispatch=None)
    finally:
        post.close()
    assert sentinel not in str(caught.value)
    assert sentinel not in caplog.text


def test_request_bounds_reject_before_client_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from assistant.provider_adapters import (
        GeminiAdapter,
        OpenAIAdapter,
        ProviderResponseRejected,
    )

    def reject_client(*args: object, **kwargs: object) -> None:
        raise AssertionError("HTTP client created before request size validation")

    monkeypatch.setattr("assistant.provider_adapters.httpx.Client", reject_client)
    for adapter in (
        GeminiAdapter("synthetic"),
        OpenAIAdapter("synthetic", "project", "none"),
    ):
        with pytest.raises(ProviderResponseRejected, match="provider_request_limit"):
            adapter.prepare(system="x" * 32001, question="Q", evidence=(), history="")
