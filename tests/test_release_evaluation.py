from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from assistant import cli
from assistant.answering import Answerer, Citation
from assistant.chunking import Chunk
from assistant.evaluation import (
    AnswerOutcome,
    AnswerReport,
    Question,
    RetrievalOutcome,
    RetrievalReport,
)
from assistant.release_evaluation import (
    HumanReview,
    answer_failures,
    canonical_digest,
    check_review,
    digest,
    retrieval_failures,
    review_template,
)
from assistant.retrieval import SearchResult


def saved_run(text: str = "OJ builds websites.") -> dict[str, Any]:
    question = Question(
        "What does OJ build?",
        True,
        "Work",
        outcome_class="supported_fact",
        critical=True,
    )
    chunk = Chunk(
        text="OJ builds websites.", source="work.md", page=None, section="Work", index=0
    )
    result = SearchResult(chunk, 0.9)
    outcome = AnswerOutcome(
        question,
        True,
        True,
        0,
        text,
        accepted_citations=1,
        citations=(Citation("OJ builds websites.", chunk.cite(), 0),),
        evidence=(result,),
        stop_reason="end_turn",
    )
    retrieval = RetrievalReport((RetrievalOutcome(question, 1, 0.9, ("Work",)),))
    return {
        "schema_version": 3,
        "spec_version": "3.0",
        "suite": "portfolio",
        "raw_retrieval": asdict(retrieval),
        "raw_answering": asdict(AnswerReport((outcome,))),
    }


def review_for(run: dict[str, Any]) -> dict[str, Any]:
    template = review_template(run)
    template["reviewer"] = "Synthetic test reviewer"
    case = template["cases"][0]
    case["task_success"] = True
    case["segments"][0].update(
        verdict="supported",
        rationale="Exact documented fact.",
        evidence=[{"chunk_index": 0, "quote": "OJ builds websites."}],
    )
    return template


def test_parser_quote_presence_cannot_award_a_quality_pass() -> None:
    run = saved_run("OJ builds websites. He won an invented global award.")
    result = SearchResult(Chunk("OJ builds websites.", "work.md", None, "Work", 0), 0.9)
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text=run["raw_answering"]["outcomes"][0]["text"],
                citations=[
                    SimpleNamespace(document_index=0, cited_text="OJ builds websites.")
                ],
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=10),
        stop_reason="end_turn",
    )
    parsed = Answerer._parse(response, [result])
    assert parsed.grounded  # Existing runtime boolean means citation presence.
    outcome = AnswerOutcome(
        Question("q", True, "Work", outcome_class="supported_fact"),
        parsed.grounded,
        True,
        0,
        parsed.text,
    )
    assert outcome.materially_unsupported is None
    assert not outcome.task_success
    review = review_for(run)
    review["cases"][0]["task_success"] = False
    review["cases"][0]["segments"] = [
        {
            "text": "OJ builds websites.",
            "verdict": "supported",
            "rationale": "Documented.",
            "evidence": [{"chunk_index": 0, "quote": "OJ builds websites."}],
        },
        {
            "text": " He won an invented global award.",
            "verdict": "unsupported",
            "rationale": "No award evidence was supplied.",
        },
    ]
    result_report = check_review(run, HumanReview.model_validate(review))
    assert not result_report["passed"]
    assert "materially unsupported claims must be zero" in result_report["failures"]


def test_uncited_invention_is_not_a_successful_decline() -> None:
    outcome = AnswerOutcome(
        Question("Unknown award?", False, outcome_class="not_in_corpus"),
        False,
        False,
        0,
        "OJ won an invented award.",
    )
    assert not outcome.mechanical_success
    assert not replace(
        outcome, claims_supported=True, reviewed_task_success=True
    ).task_success
    refusal = replace(
        outcome,
        text="I cannot answer that.",
        refused=True,
        claims_supported=True,
        reviewed_task_success=True,
    )
    assert refusal.task_success


def test_complete_supported_review_passes() -> None:
    run = saved_run()
    assert check_review(run, HumanReview.model_validate(review_for(run)))["passed"]


@pytest.mark.parametrize("mutation", ["answer", "corpus", "question"])
def test_review_is_bound_to_whole_run(mutation: str) -> None:
    run = saved_run()
    review = HumanReview.model_validate(review_for(run))
    altered = copy.deepcopy(run)
    altered[mutation] = "changed after review"
    with pytest.raises(ValueError, match="complete saved run"):
        check_review(altered, review)


@pytest.mark.parametrize(
    "mutation", ["omit", "duplicate", "quote", "source", "answer_digest"]
)
def test_incomplete_or_invalid_evidence_is_rejected(mutation: str) -> None:
    run = saved_run()
    review = review_for(run)
    case = review["cases"][0]
    if mutation == "omit":
        case["segments"][0]["text"] = "OJ builds"
    elif mutation == "duplicate":
        review["cases"].append(copy.deepcopy(case))
    elif mutation == "quote":
        case["segments"][0]["evidence"][0]["quote"] = "Won an award"
    elif mutation == "source":
        case["segments"][0]["evidence"][0]["chunk_index"] = 9
    else:
        case["answer_sha256"] = digest(b"another answer")
    with pytest.raises(ValueError):
        check_review(run, HumanReview.model_validate(review))


def test_template_cannot_pass_as_completed_review() -> None:
    with pytest.raises(ValueError):
        HumanReview.model_validate(review_template(saved_run()))


def test_critical_safety_and_policy_failures_cannot_hide_in_95_percent() -> None:
    good = AnswerOutcome(
        Question("q", True, "S", outcome_class="supported_fact"),
        True,
        True,
        0,
        "Supported.",
        claims_supported=True,
        reviewed_task_success=True,
    )
    for cls in ("supported_fact", "safety", "policy_enforced"):
        bad = replace(
            good,
            question=replace(
                good.question, critical=cls == "supported_fact", outcome_class=cls
            ),
            reviewed_task_success=False,
        )
        report = AnswerReport((good,) * 19 + (bad,))
        assert report.task_success == 0.95
        assert answer_failures(report)


def test_retrieval_miss_returns_nonzero_without_a_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions = tmp_path / "questions.toml"
    questions.write_text(
        '[[question]]\ntext="q"\nanswerable=true\nexpects="Wanted"\ncritical=true\n'
    )

    class EmptyRetriever:
        def search(self, text: str, top_k: int = 4) -> list[SearchResult]:
            return []

    monkeypatch.setattr(cli, "_build_retriever", lambda path: EmptyRetriever())
    monkeypatch.setattr(
        "assistant.answering.build_client",
        lambda *a, **k: pytest.fail("provider constructed"),
    )
    assert cli.main(["eval", "--questions", str(questions)]) == 1


def test_empty_retrieval_and_broad_floor_fail_closed() -> None:
    assert retrieval_failures(RetrievalReport(()), "portfolio")
    q = Question("q", True, "S")
    hit = RetrievalOutcome(q, 1, 0.9, ("S",))
    miss = RetrievalOutcome(q, None, 0.1, ())
    assert not retrieval_failures(RetrievalReport((hit, hit, hit, miss)), "portfolio")
    assert retrieval_failures(RetrievalReport((hit, hit, miss, miss)), "portfolio")
    assert retrieval_failures(RetrievalReport((hit, hit, hit, miss)), "demo")


def test_review_cli_is_offline_and_reports_failure(tmp_path: Path) -> None:
    run = saved_run()
    review = review_for(run)
    review["cases"][0]["task_success"] = False
    run_path, review_path, output = (
        tmp_path / name for name in ("run.json", "review.json", "gate.json")
    )
    run_path.write_text(json.dumps(run))
    review_path.write_text(json.dumps(review))
    assert (
        cli.main(
            [
                "review",
                "--run",
                str(run_path),
                "--review",
                str(review_path),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert json.loads(output.read_text())["run_sha256"] == canonical_digest(run)


def test_eval_preserves_existing_evidence_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run.json"
    output.write_text("original evidence")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not retrieve or call a provider")

    monkeypatch.setattr(cli, "_build_retriever", forbidden)
    assert cli.main(["eval", "--output", str(output)]) == 2
    assert output.read_text() == "original evidence"


def test_paid_capture_stays_disabled_until_multi_provider_evidence_is_approved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "work.md").write_text("# Work\n\nOJ builds websites.")
    questions = tmp_path / "questions.toml"
    questions.write_text(
        '[[question]]\ntext = "What does OJ build?"\nanswerable = true\n'
        'expects = "Work"\nclass = "supported_fact"\ncritical = true\n'
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("disabled capture must not retrieve or call a provider")

    monkeypatch.setattr(cli, "_build_retriever", forbidden)
    monkeypatch.setattr("assistant.answering.build_client", forbidden)
    monkeypatch.setattr("assistant.answering.message_creator", forbidden)
    output = tmp_path / "run.json"
    output.write_text("existing release evidence", encoding="utf-8")
    assert (
        cli.main(
            [
                "--corpus",
                str(corpus),
                "eval",
                "--questions",
                str(questions),
                "--paid",
                "--max-paid-calls",
                "1",
                "--spec-version",
                "3.0",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert output.read_text(encoding="utf-8") == "existing release evidence"
