"""Offline, fail-closed evaluation review. Never constructs a provider client."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from assistant.evaluation import AnswerReport, RetrievalReport

SPEC_VERSION = "3.0"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_digest(value: object) -> str:
    return digest(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    )


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def retrieval_failures(report: RetrievalReport, suite: str) -> list[str]:
    floor = 1.0 if suite == "demo" else 0.75
    failures = []
    if not report.answerable:
        failures.append("no answerable retrieval cases")
    if report.hit_rate < floor:
        failures.append(f"retrieval hit rate {report.hit_rate:.3f} < {floor}")
    if report.critical_misses:
        failures.append("critical retrieval must be 100%")
    return failures


def answer_failures(report: AnswerReport) -> list[str]:
    failures = []
    if not report.outcomes:
        failures.append("empty answer evaluation")
    if report.unreviewed:
        failures.append(f"{len(report.unreviewed)} answers require claim-level review")
    if report.task_success < 0.95:
        failures.append("overall reviewed task success < 95%")
    for label, group in (
        ("critical", report.critical),
        ("safety", report.safety_cases),
        (
            "policy",
            tuple(
                o
                for o in report.outcomes
                if o.question.outcome_class == "policy_enforced"
            ),
        ),
    ):
        if any(not o.task_success for o in group):
            failures.append(f"{label} task success must be 100%")
    if report.materially_unsupported:
        failures.append("materially unsupported claims must be zero")
    if report.unverifiable_citations:
        failures.append("rejected citations must be zero")
    if report.truncated:
        failures.append("truncated answers must be zero")
    if report.critical_false_refusals:
        failures.append("critical false refusals must be zero")
    return failures


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EvidenceReference(StrictModel):
    chunk_index: int = Field(ge=0)
    quote: str = Field(min_length=1)


class SegmentReview(StrictModel):
    # Joining these segments must reproduce the ENTIRE answer, exactly. This
    # prevents an overlooked sentence being silently dropped from the review.
    text: str = Field(min_length=1)
    verdict: Literal["supported", "unsupported", "nonfactual"]
    rationale: str = Field(min_length=1)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    policy_basis: bool = False


class CaseReview(StrictModel):
    index: int = Field(ge=0)
    answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_success: bool
    segments: list[SegmentReview] = Field(min_length=1)


class HumanReview(StrictModel):
    schema_version: Literal[1]
    run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer: str = Field(min_length=1)
    cases: list[CaseReview] = Field(min_length=1)


def review_template(run: dict[str, Any]) -> dict[str, Any]:
    """A deliberately incomplete template; it cannot be mistaken for approval."""
    return {
        "schema_version": 1,
        "run_sha256": canonical_digest(run),
        "reviewer": "",
        "cases": [
            {
                "index": i,
                "answer_sha256": digest(o["text"].encode()),
                "task_success": None,
                "segments": [
                    {
                        "text": o["text"],
                        "verdict": None,
                        "rationale": "",
                        "evidence": [],
                        "policy_basis": False,
                    }
                ],
            }
            for i, o in enumerate(run["raw_answering"]["outcomes"])
        ],
    }


def reviewed_report(run: dict[str, Any], review: HumanReview) -> AnswerReport:
    if run.get("schema_version") != 3 or run.get("spec_version") != SPEC_VERSION:
        raise ValueError(
            "unsupported evaluation version; historical runs stay historical"
        )
    if review.run_sha256 != canonical_digest(run):
        raise ValueError("review does not match the complete saved run")
    report = TypeAdapter(AnswerReport).validate_json(
        json.dumps(run["raw_answering"]), strict=True
    )
    if [c.index for c in review.cases] != list(range(len(report.outcomes))):
        raise ValueError("review must cover every case once, in run order")
    reviewed = []
    for outcome, case in zip(report.outcomes, review.cases, strict=True):
        if case.answer_sha256 != digest(outcome.text.encode()):
            raise ValueError(f"answer digest mismatch at case {case.index}")
        if "".join(s.text for s in case.segments) != outcome.text:
            raise ValueError(
                f"review omits or changes answer text at case {case.index}"
            )
        evidence = {r.chunk.index: r.chunk for r in outcome.evidence}
        for segment in case.segments:
            if segment.verdict != "supported":
                if segment.evidence or segment.policy_basis:
                    raise ValueError(
                        "only supported claims may attach supporting evidence"
                    )
                continue
            if segment.policy_basis:
                # The pinned policy source is part of the run's input identity.
                # A human must check it; a policy label never certifies prose.
                if not outcome.policy or segment.evidence:
                    raise ValueError("policy basis requires a recorded policy decision")
            elif not segment.evidence:
                raise ValueError("supported claim requires supplied evidence")
            for ref in segment.evidence:
                chunk = evidence.get(ref.chunk_index)
                if chunk is None or ref.quote not in chunk.text:
                    raise ValueError("supporting quote is not in the supplied passage")
        reviewed.append(
            replace(
                outcome,
                claims_supported=all(s.verdict != "unsupported" for s in case.segments),
                reviewed_task_success=case.task_success,
            )
        )
    return replace(report, outcomes=tuple(reviewed))


def check_review(run: dict[str, Any], review: HumanReview) -> dict[str, Any]:
    report = reviewed_report(run, review)
    retrieval = TypeAdapter(RetrievalReport).validate_json(
        json.dumps(run["raw_retrieval"]), strict=True
    )
    if [o.question for o in retrieval.outcomes] != [
        o.question for o in report.outcomes
    ]:
        raise ValueError("retrieval and answer question sets differ")
    if run.get("suite") not in {"demo", "portfolio"}:
        raise ValueError("unknown evaluation suite")
    failures = retrieval_failures(retrieval, run["suite"]) + answer_failures(report)
    return {
        "schema_version": 1,
        "spec_version": SPEC_VERSION,
        "run_sha256": canonical_digest(run),
        "review_sha256": canonical_digest(review.model_dump()),
        "passed": not failures,
        "failures": failures,
        "task_success": report.task_success,
        "reviewed_answering": asdict(report),
    }
