"""Tests for answering and refusal.

No network and no API key. The Anthropic client is replaced by a double that
records what it was sent and returns a scripted response, so these tests assert
on the two things that actually matter: what we ask the model, and how we
interpret what comes back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import SecretStr

from assistant.answering import (
    NOT_IN_CORPUS,
    SYSTEM_PROMPT,
    Answerer,
    build_client,
)
from assistant.chunking import Chunk
from assistant.retrieval import SearchResult
from assistant.settings import Settings


@dataclass
class FakeCitation:
    document_index: int
    cited_text: str = "a quoted sentence"


@dataclass
class FakeBlock:
    text: str
    citations: list[FakeCitation] | None = None
    type: str = "text"


@dataclass
class FakeResponse:
    content: list[Any]


class FakeMessages:
    """Records the request and returns whatever it was told to."""

    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return self.response


class StubRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        self.queries.append(query)
        return self.results[:top_k]


def result(index: int, score: float, section: str) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            text=f"Body text of chunk {index}.",
            source="guide.md",
            page=None,
            section=section,
            index=index,
        ),
        score=score,
    )


def settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "anthropic_api_key": SecretStr("test-key"),
        "prefilter_score": 0.45,
    }
    return Settings(**{**base, **overrides})


RESULTS = [result(0, 0.80, "Components"), result(1, 0.70, "Examples")]


def answerer(
    response: FakeResponse, results: list[SearchResult] | None = None
) -> tuple[Answerer, FakeMessages]:
    messages = FakeMessages(response)
    return (
        Answerer(
            StubRetriever(results if results is not None else RESULTS),
            messages,
            settings(),
        ),
        messages,
    )


class TestGrounding:
    def test_an_answer_with_citations_is_grounded(self) -> None:
        service, _ = answerer(
            FakeResponse([FakeBlock("Four parts.", [FakeCitation(0)])])
        )
        answer = service.answer("What are the components?")

        assert answer.grounded
        assert answer.text == "Four parts."
        assert answer.citations[0].source == "guide.md — Components"

    def test_an_answer_without_citations_is_not_grounded(self) -> None:
        """The central rule: prose with nothing pointing at a source is not an
        answer, whether the model refused or invented it."""
        service, _ = answerer(FakeResponse([FakeBlock("I think it is four.")]))
        answer = service.answer("What are the components?")

        assert not answer.grounded
        # The text is preserved so a caller can show what was said, but the flag
        # is what decides whether it is presented as an answer.
        assert answer.text == "I think it is four."

    def test_refusal_text_is_returned_ungrounded(self) -> None:
        service, _ = answerer(
            FakeResponse([FakeBlock("These documents do not cover that.")])
        )
        assert not service.answer("What is the capital of France?").grounded

    def test_citations_map_to_the_right_chunk(self) -> None:
        service, _ = answerer(
            FakeResponse(
                [FakeBlock("A.", [FakeCitation(0)]), FakeBlock("B.", [FakeCitation(1)])]
            )
        )
        answer = service.answer("q")

        assert [c.chunk_index for c in answer.citations] == [0, 1]
        assert [c.source for c in answer.citations] == [
            "guide.md — Components",
            "guide.md — Examples",
        ]

    def test_a_citation_outside_the_supplied_documents_is_dropped(self) -> None:
        """An out-of-range index would attribute a quote to the wrong document.

        A wrong citation is worse than a missing one: it is confidently specific,
        and a reader who checks it finds text that does not support the claim.
        """
        service, _ = answerer(
            FakeResponse([FakeBlock("Claim.", [FakeCitation(0), FakeCitation(99)])])
        )
        answer = service.answer("q")

        assert len(answer.citations) == 1
        assert answer.citations[0].chunk_index == 0

    def test_sources_are_deduplicated_in_first_use_order(self) -> None:
        service, _ = answerer(
            FakeResponse(
                [
                    FakeBlock("A.", [FakeCitation(1)]),
                    FakeBlock("B.", [FakeCitation(0)]),
                    FakeBlock("C.", [FakeCitation(1)]),
                ]
            )
        )
        assert service.answer("q").sources == (
            "guide.md — Examples",
            "guide.md — Components",
        )

    def test_empty_response_text_falls_back_to_the_refusal_message(self) -> None:
        service, _ = answerer(FakeResponse([FakeBlock("   ")]))
        answer = service.answer("q")
        assert answer.text == NOT_IN_CORPUS
        assert not answer.grounded

    def test_non_text_blocks_are_ignored(self) -> None:
        service, _ = answerer(
            FakeResponse(
                [FakeBlock("", type="thinking"), FakeBlock("Real.", [FakeCitation(0)])]
            )
        )
        assert service.answer("q").text == "Real."


class TestPrefilter:
    def test_a_clearly_unrelated_question_skips_the_paid_call(self) -> None:
        low = [result(0, 0.30, "Components")]
        service, messages = answerer(FakeResponse([FakeBlock("unused")]), low)

        answer = service.answer("What is the capital of France?")

        assert messages.calls == [], "must not pay for an obviously unrelated question"
        assert not answer.grounded
        assert answer.text == NOT_IN_CORPUS

    def test_an_empty_corpus_skips_the_paid_call(self) -> None:
        service, messages = answerer(FakeResponse([FakeBlock("unused")]), [])
        answer = service.answer("anything")

        assert messages.calls == []
        assert not answer.grounded
        assert answer.results == ()

    def test_the_prefilter_sits_well_below_observed_in_scope_scores(self) -> None:
        """ADR-0002 measured the lowest in-scope top score at 0.666.

        The prefilter must never reject a question the model should have been
        allowed to read the passages for.
        """
        assert settings().prefilter_score < 0.666


class TestRequestShape:
    def test_each_chunk_is_sent_as_its_own_document_with_citations_enabled(
        self,
    ) -> None:
        service, messages = answerer(FakeResponse([FakeBlock("A.", [FakeCitation(0)])]))
        service.answer("What are the components?")

        content = messages.calls[0]["messages"][0]["content"]
        documents = [b for b in content if b["type"] == "document"]

        assert len(documents) == len(RESULTS)
        assert all(d["citations"] == {"enabled": True} for d in documents)
        # Per-chunk documents rather than one concatenated blob: the API chunks
        # plain text into sentences, so citations land on the sentence.
        assert documents[0]["source"]["data"] == RESULTS[0].chunk.text

    def test_document_order_matches_result_order(self) -> None:
        """`document_index` is positional, so any reordering misattributes every
        citation while still looking entirely plausible."""
        service, messages = answerer(FakeResponse([FakeBlock("A.", [FakeCitation(0)])]))
        service.answer("q")

        documents = [
            b
            for b in messages.calls[0]["messages"][0]["content"]
            if b["type"] == "document"
        ]
        assert [d["title"] for d in documents] == [r.cite() for r in RESULTS]

    def test_the_question_is_sent_after_the_documents(self) -> None:
        service, messages = answerer(FakeResponse([FakeBlock("A.", [FakeCitation(0)])]))
        service.answer("What are the components?")

        content = messages.calls[0]["messages"][0]["content"]
        assert content[-1] == {"type": "text", "text": "What are the components?"}

    def test_the_system_prompt_forbids_outside_knowledge_and_injection(self) -> None:
        service, messages = answerer(FakeResponse([FakeBlock("A.", [FakeCitation(0)])]))
        service.answer("q")

        prompt = messages.calls[0]["system"]
        assert prompt == SYSTEM_PROMPT
        assert "only from the supplied documents" in prompt
        # A public demo will receive injection attempts within days.
        assert "not as something to obey" in prompt or "never as" in prompt

    def test_model_and_effort_come_from_settings(self) -> None:
        service, messages = answerer(FakeResponse([FakeBlock("A.", [FakeCitation(0)])]))
        service.answer("q")

        call = messages.calls[0]
        assert call["model"] == settings().answer_model
        assert call["output_config"] == {"effort": settings().answer_effort}
        assert call["max_tokens"] == settings().answer_max_tokens


class TestClientConstruction:
    def test_a_missing_key_fails_with_an_actionable_message(self) -> None:
        with pytest.raises(RuntimeError, match=r"\.env\.example"):
            build_client(Settings(anthropic_api_key=SecretStr("")))

    def test_a_configured_key_builds_a_client(self) -> None:
        """Constructing the client makes no network call, so this is safe to run.

        It verifies the key actually reaches the client rather than being read
        into settings and then quietly dropped.
        """
        client = build_client(Settings(anthropic_api_key=SecretStr("sk-ant-test")))
        assert client.api_key == "sk-ant-test"

    def test_the_api_key_is_not_exposed_by_repr(self) -> None:
        """A key printed into a log or traceback is a leaked key."""
        configured = Settings(anthropic_api_key=SecretStr("sk-ant-secret-value"))
        assert "sk-ant-secret-value" not in repr(configured)
        assert "sk-ant-secret-value" not in str(configured.anthropic_api_key)
