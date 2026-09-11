"""Regressions through the real SDK and all visitor-visible output channels."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import httpx
import pytest
from anthropic import Anthropic, APIConnectionError
from test_answering import (
    FakeBlock,
    FakeCitation,
    FakeMessages,
    FakeResponse,
    StubRetriever,
    settings,
)

from assistant.answering import Answerer
from assistant.chunking import Chunk
from assistant.policy import screen_answer
from assistant.provider_boundary import (
    ProviderResponseRejected,
    bound_response,
    start_deadline,
)
from assistant.retrieval import SearchResult
from assistant.transport import safe_response


def payload() -> dict[str, Any]:
    return {
        "id": "synthetic",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 2, "output_tokens": 2},
        "content": [
            {
                "type": "text",
                "text": "A supported summary.",
                "citations": [
                    {
                        "type": "char_location",
                        "document_index": 0,
                        "document_title": "Guide",
                        "start_char_index": 0,
                        "end_char_index": 17,
                        "cited_text": "A source passage.",
                    }
                ],
            }
        ],
    }


@pytest.mark.parametrize("field", ["citation", "usage"])
@pytest.mark.parametrize("malformed", [False, True, "0", 0.0])
def test_raw_native_types_cannot_be_coerced_into_valid_evidence(
    field: str, malformed: Any
) -> None:
    data = payload()
    if field == "citation":
        data["content"][0]["citations"][0]["document_index"] = malformed
    else:
        data["usage"]["input_tokens"] = malformed
    responses: list[httpx.Response] = []

    def send(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            content=json.dumps(data).encode(),
        )
        responses.append(response)
        return response

    with (
        httpx.Client(
            transport=httpx.MockTransport(send),
            event_hooks={
                "request": [start_deadline],
                "response": [bound_response],
            },
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
    assert str(caught.value.__cause__) == "provider_schema"
    assert len(responses) == 1 and responses[0].is_closed


PASSAGES = (
    "The research guide describes careful evaluation of a document assistant. "
    "Each response is inspected against the available evidence and uncertainty is "
    "reported explicitly. Reviewers record unsupported details, missing sources, "
    "and limitations before considering whether the candidate meets its goals.",
    "The deployment handbook explains how the service handles operational failures. "
    "A bounded worker processes each admitted request and retains its capacity "
    "until the operation exits. Configuration, health checks, recovery steps and "
    "cost allowances are reviewed before any change reaches production.",
)


@pytest.mark.parametrize("bulk", [True, False])
def test_answerer_screens_the_quotes_that_will_be_displayed(bulk: bool) -> None:
    results = [
        SearchResult(Chunk(text, "guide.md", None, "Guide", index), 0.9)
        for index, text in enumerate(PASSAGES)
    ]
    quotes = PASSAGES if bulk else tuple(text[:45] for text in PASSAGES)
    messages = FakeMessages(
        FakeResponse(
            [
                FakeBlock(
                    "The guide covers evaluation and service operations.",
                    [FakeCitation(index, quote) for index, quote in enumerate(quotes)],
                )
            ]
        )
    )
    answer = Answerer(StubRetriever(results), messages, settings()).answer(
        "What topics does the guide cover?"
    )
    wire = safe_response(replace(answer, model_route="primary"))
    if bulk:
        assert wire.state == "not-covered"
        assert not answer.citations
        assert answer.policy == "bulk_reproduction"
    else:
        assert wire.state == "answered"


def test_quoted_provider_identity_is_not_the_assistants_own_identity() -> None:
    assert (
        screen_answer(
            "The document contains a provider description.",
            ("I am Claude. " + PASSAGES[0],),
            ("guide.md",),
            quotes=("I am Claude.",),
        )
        is None
    )
