"""Validate a release's version tuple and evidence, without deploying anything."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from assistant.corpus_checksum import corpus_checksum
from assistant.evaluation import RetrievalReport, load_questions
from assistant.release_evaluation import (
    SPEC_VERSION,
    HumanReview,
    StrictModel,
    check_review,
    file_digest,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Commit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
ANSWER_CONTRACT_VERSION = 3
ANSWER_RUNTIME_SOURCES = (
    "capture.py",
    "cli.py",
    "api.py",
    "answer_event.py",
    "answering.py",
    "provider_adapters.py",
    "provider_boundary.py",
    "provider_router.py",
    "runtime.py",
    "budget.py",
    "persistent_budget.py",
    "settings.py",
    "transport.py",
)


class Artifact(StrictModel):
    path: str = Field(min_length=1)
    sha256: Sha256

    def verify(self, root: Path) -> Path:
        target = contained_path(root, self.path)
        if not target.is_file() or file_digest(target) != self.sha256:
            raise ValueError(f"artifact missing or digest mismatch: {self.path}")
        return target


def contained_path(root: Path, value: str) -> Path:
    # Reject Windows drive syntax even when verification runs on Linux.
    if Path(value).is_absolute() or re.match(r"^[A-Za-z]:", value) or "\\" in value:
        raise ValueError("artifact paths must be relative POSIX paths")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes the release directory")
    return resolved


class EvaluationEvidence(StrictModel):
    suite: Literal["demo", "portfolio"]
    corpus_path: str
    corpus_sha256: Sha256
    questions: Artifact
    prompt: Artifact
    run: Artifact
    review: Artifact


class EmbeddingIdentity(StrictModel):
    model: Literal["BAAI/bge-small-en-v1.5"]
    dimensions: Literal[384]
    # The model lock records exact model/tokenizer files. A model name alone
    # does not identify the bytes used to produce or search an index.
    model_lock: Artifact
    vectors: Artifact


class ModelSnapshot(StrictModel):
    schema_version: Literal[1]
    model: Literal["BAAI/bge-small-en-v1.5"]
    repository: Literal["qdrant/bge-small-en-v1.5-onnx-q"]
    revision: Commit
    files: dict[str, Sha256] = Field(min_length=1)


class AnswerRuntimeArtifacts(StrictModel):
    capture: Artifact = Field(alias="capture.py")
    cli: Artifact = Field(alias="cli.py")
    api: Artifact = Field(alias="api.py")
    answer_event: Artifact = Field(alias="answer_event.py")
    answering: Artifact = Field(alias="answering.py")
    provider_adapters: Artifact = Field(alias="provider_adapters.py")
    provider_boundary: Artifact = Field(alias="provider_boundary.py")
    provider_router: Artifact = Field(alias="provider_router.py")
    runtime: Artifact = Field(alias="runtime.py")
    budget: Artifact = Field(alias="budget.py")
    persistent_budget: Artifact = Field(alias="persistent_budget.py")
    settings: Artifact = Field(alias="settings.py")
    transport: Artifact = Field(alias="transport.py")

    def by_source(self) -> dict[str, Artifact]:
        return {
            "capture.py": self.capture,
            "cli.py": self.cli,
            "api.py": self.api,
            "answer_event.py": self.answer_event,
            "answering.py": self.answering,
            "provider_adapters.py": self.provider_adapters,
            "provider_boundary.py": self.provider_boundary,
            "provider_router.py": self.provider_router,
            "runtime.py": self.runtime,
            "budget.py": self.budget,
            "persistent_budget.py": self.persistent_budget,
            "settings.py": self.settings,
            "transport.py": self.transport,
        }


class AnswerConfiguration(StrictModel):
    """The non-secret behavior identity an answer was produced under.

    Every field here changes what comes out of the service. Daily and monthly
    spend ceilings do not -- they govern how many answers may be asked for,
    never what any one of them says -- and they are no longer recorded here.

    They were, until 14 September 2026, when the capture-scoped ledger
    ADR-0015 authorises made the conflation load-bearing. `PersistentBudget`
    refuses any settings that disagree with its ledger's stamped limits, so a
    capture run against the 150-attempt ledger can only ever record 150 and
    US$6.00, while the deployment that evidence qualifies runs at 40 and
    US$0.40. `verify_manifest` requires a single `answer_configuration` to
    equal the config of every capture AND to describe the release. It cannot
    do both while these live here.

    The deployed envelope is still guarded, by a stronger check than this one
    was: `test_operating_caps_and_worker_settings_validate_against_runtime`
    reads the limits out of `fly.oj-assistant.toml`, the file that is actually
    deployed, rather than out of a schema. ADR-0015 accepted that substitution
    when it widened the settings bounds; this extends it one level up.

    `attempt_reservation_micro_usd` and `shared_worker_limit` stay. Both do
    shape answers: the reservation price is pinned by ADR-0015 and written into
    every ledger's `CHECK` constraint, and the worker limit is the concurrency
    the runtime actually admits.
    """

    primary_model: Literal["gemini-3.5-flash-lite"]
    fallback_model: Literal["gpt-5.6-luna"]
    answer_effort: Literal["none"]
    answer_max_tokens: Literal[1024]
    top_k: Literal[4]
    prefilter_score: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    backend_timeout_seconds: float = Field(ge=8.0, le=8.0, allow_inf_nan=False)
    provider_timeout_seconds: float = Field(ge=6.0, le=6.0, allow_inf_nan=False)
    primary_timeout_seconds: float = Field(ge=3.0, le=3.0, allow_inf_nan=False)
    validation_margin_seconds: float = Field(ge=0.5, le=0.5, allow_inf_nan=False)
    wire_version: Literal[3]
    max_attempts: Literal[2]
    retries: Literal[0]
    fallback_mode: Literal["availability_only"]
    complete_pair_required: Literal[True]
    max_provider_request_bytes: Literal[32000]
    attempt_reservation_micro_usd: Literal[40000]
    shared_worker_limit: Literal[1]
    budget_storage: Literal["persistent_local_sqlite"]


class ReleaseManifest(StrictModel):
    # v3 drops the four spend ceilings from `answer_configuration`; see that
    # model. Removing required fields is breaking in both directions, so it
    # takes a version rather than an edit to the published v2 document: a v2
    # and a v3 manifest must never be indistinguishable by their own stamp.
    # v1 and v2 stay on disk as history, referenced by nothing, and neither
    # can qualify a release.
    schema_version: Literal[3]
    deployment: Literal["oj-assistant", "cited-demo"]
    frontend_commit: Commit
    backend_commit: Commit
    backend_image: str = Field(pattern=r"^[^\s]+@sha256:[0-9a-f]{64}$")
    python_base: str = Field(pattern=r"^python:3\.12[^\s]*@sha256:[0-9a-f]{64}$")
    runtime_lock: Artifact
    evaluator: Artifact
    review_code: Artifact
    policy: Artifact
    corpus_path: str
    corpus_sha256: Sha256
    prompt: Artifact
    embedding: EmbeddingIdentity
    spec_version: Literal["3.0"]
    answer_contract_version: Literal[3]
    answer_runtime: AnswerRuntimeArtifacts
    answer_configuration: AnswerConfiguration
    evaluations: list[EvaluationEvidence] = Field(min_length=2, max_length=2)


def verify_manifest(manifest: ReleaseManifest, root: Path) -> None:
    from assistant.chunking import chunk_passages
    from assistant.documents import read_corpus
    from assistant.vectors import load

    for artifact in (
        manifest.runtime_lock,
        manifest.evaluator,
        manifest.review_code,
        manifest.policy,
        manifest.prompt,
        manifest.embedding.model_lock,
    ):
        artifact.verify(root)
    runtime_digests: dict[str, str] = {}
    for source, artifact in manifest.answer_runtime.by_source().items():
        artifact.verify(root)
        runtime_digests[source] = artifact.sha256
    corpus = contained_path(root, manifest.corpus_path)
    ModelSnapshot.model_validate_json(
        manifest.embedding.model_lock.verify(root).read_text(encoding="utf-8")
    )
    if not corpus.is_dir() or corpus_checksum(corpus) != manifest.corpus_sha256:
        raise ValueError("release corpus mismatch")
    vectors = manifest.embedding.vectors.verify(root)
    load(
        vectors,
        chunk_passages(read_corpus(corpus)),
        model=manifest.embedding.model,
        dimensions=manifest.embedding.dimensions,
    )
    if {e.suite for e in manifest.evaluations} != {"demo", "portfolio"}:
        raise ValueError("both distinct evaluation suites are required")
    selected = "portfolio" if manifest.deployment == "oj-assistant" else "demo"
    for evaluation in manifest.evaluations:
        if (
            corpus_checksum(contained_path(root, evaluation.corpus_path))
            != evaluation.corpus_sha256
        ):
            raise ValueError("evaluation corpus mismatch")
        evaluation.questions.verify(root)
        evaluation.prompt.verify(root)
        run = json.loads(evaluation.run.verify(root).read_text(encoding="utf-8"))
        if run.get("capture_state") != "complete":
            raise ValueError("release requires a complete routed capture")
        review = HumanReview.model_validate_json(
            evaluation.review.verify(root).read_text(encoding="utf-8")
        )
        retrieval = TypeAdapter(RetrievalReport).validate_json(
            json.dumps(run["raw_retrieval"]), strict=True
        )
        if [o.question for o in retrieval.outcomes] != load_questions(
            evaluation.questions.verify(root)
        ):
            raise ValueError("saved cases differ from the versioned question set")
        expected = {
            "spec_version": SPEC_VERSION,
            "suite": evaluation.suite,
            "corpus_sha256": evaluation.corpus_sha256,
            "questions_sha256": evaluation.questions.sha256,
            "prompt_sha256": evaluation.prompt.sha256,
            "runtime_lock_sha256": manifest.runtime_lock.sha256,
            "model_lock_sha256": manifest.embedding.model_lock.sha256,
            "evaluator_sha256": manifest.evaluator.sha256,
            "reviewer_code_sha256": manifest.review_code.sha256,
            "policy_sha256": manifest.policy.sha256,
            "top_k": manifest.answer_configuration.top_k,
            "answer_contract_version": manifest.answer_contract_version,
        }
        if any(run.get(key) != value for key, value in expected.items()):
            raise ValueError(f"stale or mismatched {evaluation.suite} evaluation")
        if run.get("answer_runtime_sha256") != runtime_digests:
            raise ValueError("answer runtime identity mismatch")
        try:
            config = AnswerConfiguration.model_validate(run.get("config"))
        except (TypeError, ValueError) as error:
            raise ValueError("answer configuration is missing or invalid") from error
        if config != manifest.answer_configuration:
            raise ValueError("answer configuration mismatch")
        if not check_review(run, review)["passed"]:
            raise ValueError(f"{evaluation.suite} reviewed quality gate failed")
        if evaluation.suite == selected and (
            evaluation.corpus_sha256 != manifest.corpus_sha256
            or evaluation.prompt.sha256 != manifest.prompt.sha256
        ):
            raise ValueError("deployed corpus/prompt differs from evaluated inputs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--observed",
        type=Path,
        help="separately collected deployed manifest for exact comparison",
    )
    args = parser.parse_args(argv)
    try:
        manifest = ReleaseManifest.model_validate_json(
            args.manifest.read_text(encoding="utf-8")
        )
        verify_manifest(manifest, args.artifacts)
        if args.observed:
            observed = ReleaseManifest.model_validate_json(
                args.observed.read_text(encoding="utf-8")
            )
            if observed != manifest:
                raise ValueError(
                    "observed deployment differs from the intended release"
                )
        print("Artifact checks passed; publication still requires owner approval.")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Release check failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
