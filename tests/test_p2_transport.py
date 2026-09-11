"""Synthetic counterexamples at the provider and v3 transport boundaries."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from assistant.answering import NOT_IN_CORPUS, Answerer
from assistant.chunking import Chunk
from assistant.evaluation import AnswerOutcome, AnswerReport, Question
from assistant.policy import POLICY_RESPONSES
from assistant.retrieval import SearchResult
from assistant.transport import safe_response

RESULT = SearchResult(Chunk("OJ builds websites.", "work.md", None, "Work", 0), 0.9)


def block(
    text: object = "OJ builds websites.", citations: object = None
) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text, citations=citations)


def quote(text: object = "OJ builds websites.", index: object = 0) -> SimpleNamespace:
    return SimpleNamespace(document_index=index, cited_text=text)


def response(blocks: object, **kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(content=blocks, stop_reason="end_turn", **kwargs)


@pytest.mark.parametrize(
    "bad",
    [
        block("Uncited invented award."),
        block("Invented award.", [quote("not in the passage")]),
        block("Invented award.", [quote(), quote(index=99)]),
        block("NOT_IN_DOCUMENTS\nInvented refusal explanation.", [quote()]),
        block("x" * 4001, [quote()]),
        block(123, [quote()]),
        block("Invented award.", [quote(index=True)]),
        SimpleNamespace(type="tool_use", name="exfiltrate"),
    ],
)
def test_any_invalid_block_suppresses_the_whole_generated_answer(bad: object) -> None:
    answer = Answerer._parse(response([block(citations=[quote()]), bad]), [RESULT])
    assert not answer.grounded
    assert answer.text == NOT_IN_CORPUS
    assert answer.citations == ()
    assert safe_response(answer).model_dump() == {
        "version": 3,
        "state": "not-covered",
        "policy": "unsupported",
        "model_route": None,
    }


@pytest.mark.parametrize("stop", [None, "max_tokens", "tool_use", "refusal", [], 123])
def test_truncated_or_unknown_stop_never_returns_generated_prose(stop: object) -> None:
    raw = response([block("Unfinished claim", [quote()])])
    raw.stop_reason = stop
    answer = Answerer._parse(raw, [RESULT])
    assert answer.text == NOT_IN_CORPUS
    assert not answer.grounded


def test_provider_cannot_construct_a_policy_response() -> None:
    raw = response(
        [block("Arbitrary malicious policy prose", [quote()])], policy="identity"
    )
    assert Answerer._parse(raw, [RESULT]).text == NOT_IN_CORPUS
    original = Answerer._parse(response([block(citations=[quote()])]), [RESULT])
    forged = replace(
        original,
        text="Arbitrary policy prose",
        grounded=False,
        citations=(),
        policy="identity",
        model_route="primary",
    )
    assert safe_response(forged).model_dump()["policy"] == "unsupported"


def test_only_exact_application_policy_copy_is_authorized() -> None:
    original = Answerer._parse(response([block(citations=[quote()])]), [RESULT])
    fixed = replace(
        original,
        text=POLICY_RESPONSES["identity"],
        grounded=False,
        citations=(),
        policy="identity",
        model_route="primary",
    )
    assert safe_response(fixed).model_dump() == {
        "version": 3,
        "state": "not-covered",
        "policy": "identity",
        "model_route": "primary",
    }


def test_evidence_identity_is_stable_and_not_a_display_label() -> None:
    one = Answerer._parse(response([block(citations=[quote()])]), [RESULT])
    two = Answerer._parse(
        response([block("Supported paraphrase", [quote()])]), [RESULT]
    )
    wire = safe_response(replace(one, model_route="primary")).model_dump()
    citation = wire["citations"][0]
    assert citation["source_id"] == "work.md"
    assert len(citation["evidence_id"]) == 64
    assert one.citations[0].evidence_id == two.citations[0].evidence_id
    altered = replace(one, citations=(replace(one.citations[0], source_id="other.md"),))
    assert safe_response(replace(altered, model_route="primary")).state == "not-covered"


def test_same_block_invention_is_not_awarded_semantic_success() -> None:
    parsed = Answerer._parse(
        response(
            [
                block(
                    "OJ builds websites. He won an invented global award.", [quote()]
                ),
            ]
        ),
        [RESULT],
    )
    assert parsed.grounded  # Structural coverage only; this is the known limit.
    outcome = AnswerOutcome(
        Question("q", True, "Work", outcome_class="supported_fact"),
        parsed.grounded,
        True,
        0,
        parsed.text,
    )
    assert outcome.materially_unsupported is None
    assert not outcome.task_success
    reviewed = replace(outcome, claims_supported=False, reviewed_task_success=False)
    assert reviewed.materially_unsupported is True
    assert not reviewed.task_success


def test_missing_provider_usage_remains_unknown_through_reporting() -> None:
    answer = Answerer._parse(response([block(citations=[quote()])]), [RESULT])
    assert answer.input_tokens is None and answer.output_tokens is None
    outcome = AnswerOutcome(
        Question("q", True, "Work"),
        True,
        True,
        0,
        answer.text,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
    )
    report = AnswerReport((outcome,), paid_calls=1)
    assert report.input_tokens is None and report.output_tokens is None
    assert report.cost_usd is None
