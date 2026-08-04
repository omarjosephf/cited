"""Tests for the evaluation harness.

A harness that scores wrongly is worse than none, because it produces a number
that gets believed. These tests care most about the ways it could report a good
score while measuring the wrong thing.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from assistant.answering import Answer, Citation
from assistant.chunking import Chunk
from assistant.evaluation import (
    AnswerOutcome,
    InvalidQuestionSet,
    Question,
    RetrievalOutcome,
    RetrievalReport,
    evaluate_answering,
    evaluate_retrieval,
    load_questions,
)
from assistant.retrieval import SearchResult


def chunk(section: str, index: int = 0) -> Chunk:
    return Chunk(
        text=f"Text of {section}.",
        source="guide.md",
        page=None,
        section=section,
        index=index,
    )


class StubRetriever:
    """Returns a fixed ranking, so scoring is tested rather than retrieval."""

    def __init__(self, ranking: list[tuple[str, float]]) -> None:
        self.ranking = ranking

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        return [
            SearchResult(chunk=chunk(section, i), score=score)
            for i, (section, score) in enumerate(self.ranking[:top_k])
        ]


def write_questions(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "questions.toml"
    path.write_text(body, encoding="utf-8")
    return path


class TestQuestionSetValidation:
    """Strict, because every failure here is silent rather than loud."""

    def test_a_valid_set_loads(self, tmp_path: Path) -> None:
        path = write_questions(
            tmp_path,
            '[[question]]\ntext = "Q1"\nexpects = "S"\nanswerable = true\n'
            '[[question]]\ntext = "Q2"\nanswerable = false\n',
        )
        questions = load_questions(path)

        assert [q.text for q in questions] == ["Q1", "Q2"]
        assert questions[0].expects == "S"
        assert questions[1].expects is None

    def test_an_answerable_question_must_name_an_expected_section(
        self, tmp_path: Path
    ) -> None:
        """Otherwise it is scored against nothing and silently drags the hit
        rate down, which looks like a retrieval regression."""
        path = write_questions(
            tmp_path, '[[question]]\ntext = "Q"\nanswerable = true\n'
        )
        with pytest.raises(InvalidQuestionSet, match="names no expected section"):
            load_questions(path)

    def test_an_unanswerable_question_must_not_name_one(self, tmp_path: Path) -> None:
        """Suggests the author confused the two categories."""
        path = write_questions(
            tmp_path,
            '[[question]]\ntext = "Q"\nexpects = "S"\nanswerable = false\n',
        )
        with pytest.raises(InvalidQuestionSet, match="unanswerable"):
            load_questions(path)

    def test_an_empty_question_is_rejected(self, tmp_path: Path) -> None:
        path = write_questions(
            tmp_path, '[[question]]\ntext = "   "\nanswerable = false\n'
        )
        with pytest.raises(InvalidQuestionSet, match="no text"):
            load_questions(path)

    def test_an_empty_set_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidQuestionSet, match="no questions"):
            load_questions(write_questions(tmp_path, "# nothing here\n"))

    def test_a_missing_file_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidQuestionSet, match="not found"):
            load_questions(tmp_path / "absent.toml")

    def test_the_committed_question_set_is_valid(self) -> None:
        """The one that actually produces the published numbers."""
        questions = load_questions()

        assert len(questions) >= 10
        assert any(q.answerable for q in questions)
        assert any(not q.answerable for q in questions), (
            "a set with no unanswerable questions cannot measure refusal"
        )


class TestRetrievalScoring:
    def test_a_hit_at_rank_one_scores_both_metrics(self) -> None:
        retriever = StubRetriever([("Wanted", 0.8), ("Other", 0.5)])
        report = evaluate_retrieval(
            retriever, [Question("Q", answerable=True, expects="Wanted")]
        )

        assert report.hit_rate == 1.0
        assert report.top_1_rate == 1.0
        assert report.outcomes[0].rank == 1

    def test_a_hit_below_rank_one_counts_for_hit_rate_only(self) -> None:
        retriever = StubRetriever([("Other", 0.8), ("Wanted", 0.7)])
        report = evaluate_retrieval(
            retriever, [Question("Q", answerable=True, expects="Wanted")]
        )

        assert report.hit_rate == 1.0
        assert report.top_1_rate == 0.0
        assert report.outcomes[0].rank == 2

    def test_a_section_outside_top_k_is_a_miss(self) -> None:
        """The unrecoverable failure: the model never sees the passage."""
        retriever = StubRetriever([("A", 0.8), ("B", 0.7), ("Wanted", 0.6)])
        report = evaluate_retrieval(
            retriever, [Question("Q", answerable=True, expects="Wanted")], top_k=2
        )

        assert report.hit_rate == 0.0
        assert report.outcomes[0].rank is None

    def test_unanswerable_questions_are_excluded_from_hit_rate(self) -> None:
        """Including them would let a system score well by retrieving nothing."""
        retriever = StubRetriever([("Wanted", 0.8)])
        report = evaluate_retrieval(
            retriever,
            [
                Question("Q1", answerable=True, expects="Wanted"),
                Question("Q2", answerable=False),
            ],
        )

        assert report.hit_rate == 1.0
        assert len(report.answerable) == 1
        assert len(report.unanswerable) == 1

    def test_score_separation_is_negative_when_ranges_overlap(self) -> None:
        """The ADR-0002 measurement, recomputed rather than trusted."""
        report = RetrievalReport(
            outcomes=(
                RetrievalOutcome(
                    Question("in", answerable=True, expects="S"), 1, 0.66, ("S",)
                ),
                RetrievalOutcome(Question("out", answerable=False), None, 0.75, ("T",)),
            )
        )
        assert report.score_separation == pytest.approx(-0.09)

    def test_score_separation_is_positive_when_a_threshold_would_work(self) -> None:
        report = RetrievalReport(
            outcomes=(
                RetrievalOutcome(
                    Question("in", answerable=True, expects="S"), 1, 0.80, ("S",)
                ),
                RetrievalOutcome(Question("out", answerable=False), None, 0.40, ("T",)),
            )
        )
        assert report.score_separation == pytest.approx(0.40)

    def test_separation_is_zero_when_a_group_is_empty(self) -> None:
        """Neither group alone says anything about separability."""
        report = RetrievalReport(
            outcomes=(
                RetrievalOutcome(
                    Question("in", answerable=True, expects="S"), 1, 0.8, ("S",)
                ),
            )
        )
        assert report.score_separation == 0.0

    def test_an_empty_retriever_does_not_crash_scoring(self) -> None:
        report = evaluate_retrieval(
            StubRetriever([]), [Question("Q", answerable=True, expects="S")]
        )
        assert report.hit_rate == 0.0
        assert report.outcomes[0].top_score == 0.0


class StubAnswerer:
    """Returns a scripted answer per question, so reporting is what is tested."""

    def __init__(self, answers: dict[str, Answer]) -> None:
        self.answers = answers

    def answer(self, question: str) -> Answer:
        return self.answers[question]


def answer(
    *, citations: tuple[Citation, ...] = (), rejected: int = 0, text: str = "text"
) -> Answer:
    return Answer(
        text=text,
        citations=citations,
        grounded=bool(citations),
        results=(),
        rejected_citations=rejected,
    )


def citation(source: str) -> Citation:
    return Citation(quoted_text="quote", source=source, chunk_index=0)


class TestAnswerReport:
    QUESTIONS: ClassVar[list[Question]] = [
        Question("in-1", answerable=True, expects="Wanted"),
        Question("in-2", answerable=True, expects="Wanted"),
        Question("out-1", answerable=False),
        Question("out-2", answerable=False),
    ]

    def test_a_perfect_run_scores_100_percent(self) -> None:
        report = evaluate_answering(
            StubAnswerer(
                {
                    "in-1": answer(citations=(citation("guide.md — Wanted"),)),
                    "in-2": answer(citations=(citation("guide.md — Wanted"),)),
                    "out-1": answer(),
                    "out-2": answer(),
                }
            ),
            self.QUESTIONS,
        )

        assert report.accuracy == 1.0
        assert report.refusal_accuracy == 1.0
        assert report.false_refusal_rate == 0.0

    def test_false_refusal_is_tracked_separately_from_refusal_accuracy(self) -> None:
        """A system that refuses everything scores 100% on refusal accuracy.

        Only the false-refusal rate reveals that it is useless — which is why
        the two are reported as separate numbers rather than averaged.
        """
        report = evaluate_answering(
            StubAnswerer(dict.fromkeys(["in-1", "in-2", "out-1", "out-2"], answer())),
            self.QUESTIONS,
        )

        assert report.refusal_accuracy == 1.0, "refuses everything, so 'correct'"
        assert report.false_refusal_rate == 1.0, "and useless, which this shows"
        assert report.accuracy == 0.5

    def test_citing_the_wrong_section_is_not_counted_as_correct(self) -> None:
        report = evaluate_answering(
            StubAnswerer(
                {
                    "in-1": answer(citations=(citation("guide.md — Elsewhere"),)),
                    "in-2": answer(citations=(citation("guide.md — Wanted"),)),
                    "out-1": answer(),
                    "out-2": answer(),
                }
            ),
            self.QUESTIONS,
        )

        assert report.accuracy == 0.75
        # It was grounded, so it is not a false refusal — a different failure.
        assert report.false_refusal_rate == 0.0

    def test_rejected_citations_are_totalled_across_the_run(self) -> None:
        """Should be zero forever. Surfaced so that it stopping being zero shows."""
        report = evaluate_answering(
            StubAnswerer(
                {
                    "in-1": answer(
                        citations=(citation("guide.md — Wanted"),), rejected=2
                    ),
                    "in-2": answer(citations=(citation("guide.md — Wanted"),)),
                    "out-1": answer(rejected=1),
                    "out-2": answer(),
                }
            ),
            self.QUESTIONS,
        )

        assert report.unverifiable_citations == 3


class TestAnswerCorrectness:
    def test_grounded_and_citing_the_right_section_is_correct(self) -> None:
        outcome = AnswerOutcome(
            Question("Q", answerable=True, expects="S"),
            grounded=True,
            cited_expected=True,
            rejected_citations=0,
            text="answer",
        )
        assert outcome.correct

    def test_grounded_but_citing_the_wrong_section_is_wrong(self) -> None:
        """Not a partial success.

        A confident citation to the wrong place is the exact failure this
        project exists to prevent — a reader who follows it finds text that does
        not support the claim, and concludes the whole tool is untrustworthy.
        """
        outcome = AnswerOutcome(
            Question("Q", answerable=True, expects="S"),
            grounded=True,
            cited_expected=False,
            rejected_citations=0,
            text="answer",
        )
        assert not outcome.correct

    def test_refusing_an_answerable_question_is_wrong(self) -> None:
        outcome = AnswerOutcome(
            Question("Q", answerable=True, expects="S"),
            grounded=False,
            cited_expected=False,
            rejected_citations=0,
            text="cannot answer",
        )
        assert not outcome.correct

    def test_refusing_an_unanswerable_question_is_correct(self) -> None:
        outcome = AnswerOutcome(
            Question("Q", answerable=False),
            grounded=False,
            cited_expected=False,
            rejected_citations=0,
            text="not covered",
        )
        assert outcome.correct

    def test_answering_an_unanswerable_question_is_wrong(self) -> None:
        """Grounded here means it cited something — for a question the corpus
        cannot answer, that is a confident answer built from adjacent text."""
        outcome = AnswerOutcome(
            Question("Q", answerable=False),
            grounded=True,
            cited_expected=False,
            rejected_citations=0,
            text="a plausible answer",
        )
        assert not outcome.correct
