"""Scoring the assistant against a committed question set.

Every quality claim in this project should be reproducible by running one
command. That is the point of this module: not to produce a flattering number,
but to make the number checkable by someone who does not trust the claim.

Two halves, deliberately separable:

* **Retrieval** — does the right passage come back? Runs locally, costs nothing,
  needs no API key.
* **Answering** — is the answer grounded, and is an unanswerable question
  refused? Needs a paid call per question.

They are separate because retrieval is the layer that fails first and silently.
If the right passage never reaches the model, no amount of prompting recovers
it, and being able to measure that without spending anything means it can be
measured often.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from assistant.answering import Answer
from assistant.retrieval import Retriever

DEFAULT_QUESTIONS = Path(__file__).resolve().parents[2] / "eval" / "questions.toml"


class InvalidQuestionSet(ValueError):
    """The question set is malformed in a way that would corrupt the scores."""


@dataclass(frozen=True)
class Question:
    text: str
    answerable: bool
    expects: str | None = None
    note: str | None = None


def load_questions(path: Path | None = None) -> list[Question]:
    """Load and validate the question set.

    Validation is strict because the failure mode is silent. An answerable
    question with no `expects` would be scored against nothing and quietly
    depress the hit rate; an unanswerable one *with* `expects` suggests the
    author confused the two categories. Both are bugs in the test, and a broken
    test that still produces a number is worse than one that refuses to run.
    """
    source = path or DEFAULT_QUESTIONS
    if not source.exists():
        raise InvalidQuestionSet(f"question set not found: {source}")

    raw = tomllib.loads(source.read_text(encoding="utf-8"))
    entries = raw.get("question", [])
    if not entries:
        raise InvalidQuestionSet(f"{source.name} contains no questions")

    questions: list[Question] = []
    for position, entry in enumerate(entries, start=1):
        text = entry.get("text", "").strip()
        if not text:
            raise InvalidQuestionSet(f"question {position} has no text")

        answerable = bool(entry.get("answerable", False))
        expects = entry.get("expects")

        if answerable and not expects:
            raise InvalidQuestionSet(
                f"{text!r} is marked answerable but names no expected section"
            )
        if not answerable and expects:
            raise InvalidQuestionSet(
                f"{text!r} is marked unanswerable but names an expected section"
            )

        questions.append(
            Question(
                text=text,
                answerable=answerable,
                expects=expects,
                note=entry.get("note"),
            )
        )
    return questions


@dataclass(frozen=True)
class RetrievalOutcome:
    question: Question
    rank: int | None
    """1-indexed position of the expected section, or `None` if absent."""
    top_score: float
    retrieved: tuple[str, ...]

    @property
    def hit(self) -> bool:
        return self.rank is not None

    @property
    def top_1(self) -> bool:
        return self.rank == 1


@dataclass(frozen=True)
class RetrievalReport:
    outcomes: tuple[RetrievalOutcome, ...]

    @property
    def answerable(self) -> tuple[RetrievalOutcome, ...]:
        return tuple(o for o in self.outcomes if o.question.answerable)

    @property
    def unanswerable(self) -> tuple[RetrievalOutcome, ...]:
        return tuple(o for o in self.outcomes if not o.question.answerable)

    @property
    def hit_rate(self) -> float:
        """Fraction of answerable questions whose section appears at all.

        The headline retrieval number. A miss here is unrecoverable: the model
        never sees the passage, so it can only refuse or invent.
        """
        return _fraction(o.hit for o in self.answerable)

    @property
    def top_1_rate(self) -> float:
        return _fraction(o.top_1 for o in self.answerable)

    @property
    def score_separation(self) -> float:
        """Lowest answerable top score minus highest unanswerable top score.

        Negative means the distributions overlap, and therefore that no
        similarity threshold can separate answerable from unanswerable
        questions. ADR-0002 records this measurement; the harness recomputes it
        so the claim is checked on every run rather than trusted from a document
        written once.
        """
        if not self.answerable or not self.unanswerable:
            return 0.0
        return min(o.top_score for o in self.answerable) - max(
            o.top_score for o in self.unanswerable
        )


def _fraction(values: object) -> float:
    items = list(values)  # type: ignore[call-overload]
    return sum(1 for v in items if v) / len(items) if items else 0.0


def evaluate_retrieval(
    retriever: Retriever,
    questions: list[Question],
    top_k: int = 4,
) -> RetrievalReport:
    """Score retrieval alone. No API key, no cost."""
    outcomes: list[RetrievalOutcome] = []

    for question in questions:
        results = retriever.search(question.text, top_k=top_k)
        sections = tuple(r.chunk.section or r.chunk.source for r in results)

        rank: int | None = None
        if question.expects is not None:
            for position, section in enumerate(sections, start=1):
                if section == question.expects:
                    rank = position
                    break

        outcomes.append(
            RetrievalOutcome(
                question=question,
                rank=rank,
                top_score=results[0].score if results else 0.0,
                retrieved=sections,
            )
        )

    return RetrievalReport(outcomes=tuple(outcomes))


@dataclass(frozen=True)
class AnswerOutcome:
    question: Question
    grounded: bool
    cited_expected: bool
    rejected_citations: int
    text: str

    @property
    def correct(self) -> bool:
        """Whether the system did the right thing.

        For an answerable question: grounded, and citing the section that
        actually contains the answer. Grounded-but-citing-the-wrong-section is
        counted as wrong, because a confident citation to the wrong place is the
        failure this project exists to prevent — not a partial success.

        For an unanswerable question: not grounded. Refusing is correct.
        """
        if self.question.answerable:
            return self.grounded and self.cited_expected
        return not self.grounded


@dataclass(frozen=True)
class AnswerReport:
    outcomes: tuple[AnswerOutcome, ...]

    @property
    def accuracy(self) -> float:
        return _fraction(o.correct for o in self.outcomes)

    @property
    def refusal_accuracy(self) -> float:
        """Fraction of unanswerable questions correctly refused."""
        return _fraction(
            not o.grounded for o in self.outcomes if not o.question.answerable
        )

    @property
    def false_refusal_rate(self) -> float:
        """Answerable questions wrongly refused.

        Tracked separately because it is the failure users actually notice and
        resent. A system tuned to refuse aggressively scores well on refusal
        accuracy while being useless, and only this number reveals that.
        """
        return _fraction(not o.grounded for o in self.outcomes if o.question.answerable)

    @property
    def unverifiable_citations(self) -> int:
        """Total citations rejected because the quote was not in the passage.

        Expected to be zero. A non-zero value means the API returned a quote we
        never sent, and the verifier in `answering.py` caught it.
        """
        return sum(o.rejected_citations for o in self.outcomes)


class AnswerProvider(Protocol):
    """Anything that can answer a question.

    A protocol rather than the concrete `Answerer` so the harness can be scored
    against a scripted double. Depending on the class would mean the only way to
    test this scoring is to make real API calls — slow, paid, and requiring a
    key — which in practice means the tests do not get written at all.
    """

    def answer(self, question: str) -> Answer: ...


def evaluate_answering(
    answerer: AnswerProvider, questions: list[Question]
) -> AnswerReport:
    """Score end-to-end answering. Costs one API call per question."""
    outcomes: list[AnswerOutcome] = []

    for question in questions:
        answer = answerer.answer(question.text)
        cited_expected = question.expects is not None and any(
            question.expects in citation.source for citation in answer.citations
        )
        outcomes.append(
            AnswerOutcome(
                question=question,
                grounded=answer.grounded,
                cited_expected=cited_expected,
                rejected_citations=answer.rejected_citations,
                text=answer.text,
            )
        )

    return AnswerReport(outcomes=tuple(outcomes))
