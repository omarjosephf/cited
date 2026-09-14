"""Command line entry point.

Five commands, split by what they cost:

* `index`  — read the corpus and report what retrieval will see. Free.
* `embed`  — build the corpus vectors a deployment serves from. Free.
* `eval`   — score retrieval against the committed question set. Free.
* `inspect` — open a local, read-only corpus management panel. Free.
* `ask`    — answer one question. Costs an API call.

The free commands come first deliberately. Most of what goes wrong in a
retrieval system goes wrong before the model is involved, and being able to
inspect and score that without spending anything is what makes it cheap to check
often.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from assistant.answering import Answer, load_system_prompt
from assistant.chunking import chunk_passages
from assistant.documents import read_corpus
from assistant.embedding import MODEL_NAME, FastEmbedEmbedder
from assistant.evaluation import (
    DEFAULT_QUESTIONS,
    AnswerReport,
    InvalidQuestionSet,
    RetrievalReport,
    evaluate_retrieval,
    load_questions,
)
from assistant.inspection import CorpusProfile
from assistant.release_evaluation import (
    SPEC_VERSION,
    HumanReview,
    answer_failures,
    check_review,
    digest,
    file_digest,
    retrieval_failures,
    review_template,
)
from assistant.release_manifest import (
    ANSWER_CONTRACT_VERSION,
    ANSWER_RUNTIME_SOURCES,
    AnswerConfiguration,
)
from assistant.retrieval import InMemoryRetriever
from assistant.settings import Settings

DEFAULT_CORPUS = Path("content")
DEFAULT_INSPECTOR_PORT = 8765

_EST_INPUT_TOKENS = 2500
"""Rough input size per call: top-k passages, the system prompt and a question.

Used only for the worst-case figure printed before a run. Actual usage is read
from the provider afterwards — an estimate is for deciding whether to start, and
a measurement is for reporting what happened.
"""


def _build_retriever(corpus: Path) -> InMemoryRetriever:
    passages = read_corpus(corpus)
    if not passages:
        raise SystemExit(
            f"No documents found in {corpus}/. Add a .md, .txt, .docx or .pdf file."
        )
    chunks = chunk_passages(passages)
    return InMemoryRetriever(chunks, FastEmbedEmbedder())


def cmd_index(args: argparse.Namespace) -> int:
    passages = read_corpus(args.corpus)
    chunks = chunk_passages(passages)
    if not chunks:
        print(f"No documents found in {args.corpus}/.")
        return 1

    sizes = [len(c.text.split()) for c in chunks]
    sources = sorted({c.source for c in chunks})

    print(f"documents  {len(sources)}")
    for name in sources:
        print(f"           {name}")
    median = sorted(sizes)[len(sizes) // 2]
    print(f"passages   {len(passages)}")
    print(f"chunks     {len(chunks)}")
    print(f"words      min {min(sizes)}, median {median}, max {max(sizes)}")

    if args.verbose:
        print()
        for chunk in chunks:
            preview = chunk.text[:70].replace("\n", " ")
            print(f"  [{chunk.index:>3}] {chunk.cite()}")
            print(f"        {preview}...")
    return 0


def _print_retrieval(report: RetrievalReport) -> None:
    print("Retrieval")
    print(f"  hit rate        {report.hit_rate:.0%}  (expected section in top-k)")
    print(f"  top-1           {report.top_1_rate:.0%}  (expected section ranked first)")
    if report.critical:
        # Printed second, immediately under the aggregate, because the two are
        # read together: a healthy aggregate with a critical miss is a failure,
        # and separating them by a screen of output would let that be missed.
        status = "PASS" if not report.critical_misses else "FAIL"
        print(
            f"  critical core   {report.critical_hit_rate:.0%}  "
            f"({len(report.critical)} questions, must be 100%) [{status}]"
        )
    print(f"  separation      {report.score_separation:+.3f}")
    if report.score_separation < 0:
        print("                  negative: answerable and unanswerable score ranges")
        print("                  overlap, so no similarity threshold separates them")
        print("                  (ADR-0002)")

    if report.critical_misses:
        # Listed before the general misses. These are release blockers rather
        # than a score to note, so they must not be buried in a longer list.
        print(f"\n  {len(report.critical_misses)} CRITICAL question(s) missed:")
        for outcome in report.critical_misses:
            print(f"    {outcome.question.text}")
            print(f"      wanted   {outcome.question.expects}")
            print(f"      got      {', '.join(outcome.retrieved) or '(nothing)'}")

    misses = [o for o in report.answerable if not o.hit]
    if misses:
        print(f"\n  {len(misses)} answerable question(s) missed:")
        for outcome in misses:
            print(f"    {outcome.question.text}")
            print(f"      wanted   {outcome.question.expects}")
            print(f"      got      {', '.join(outcome.retrieved) or '(nothing)'}")

    demoted = [o for o in report.answerable if o.hit and not o.top_1]
    if demoted:
        print(f"\n  {len(demoted)} found but not ranked first:")
        for outcome in demoted:
            print(f"    rank {outcome.rank}  {outcome.question.text}")


def cmd_embed(args: argparse.Namespace) -> int:
    """Build the matrix a deployment serves from, on a machine that is not throttled.

    This is the whole point of the command: the work is identical wherever it
    runs, and running it here means a container does not repeat it on every cold
    start at a fraction of the CPU.
    """
    from assistant.corpus_checksum import corpus_checksum
    from assistant.vectors import chunk_digest, save

    passages = read_corpus(args.corpus)
    if not passages:
        raise SystemExit(f"No documents found in {args.corpus}/. Nothing to embed.")

    chunks = chunk_passages(passages)
    matrix = FastEmbedEmbedder().embed_passages(
        [chunk.indexed_text() for chunk in chunks]
    )
    checksum = corpus_checksum(args.corpus)
    save(args.out, chunks, matrix, model=MODEL_NAME, corpus_checksum=checksum)

    # Printed because these are the values a deployment check compares against,
    # and reading them out of a binary file afterwards is nobody's idea of a
    # release step.
    print(f"wrote {args.out}")
    print(f"  chunks           {len(chunks)}")
    print(f"  dimensions       {matrix.shape[1]}")
    print(f"  model            {MODEL_NAME}")
    print(f"  corpus           {args.corpus}")
    print(f"  corpus checksum  {checksum}")
    print(f"  chunk digest     {chunk_digest(chunks)}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    if args.output and args.output.exists():
        print("--output already exists; choose a new evidence file", file=sys.stderr)
        return 2
    if args.top_k < 1:
        print("--top-k must be positive", file=sys.stderr)
        return 2
    try:
        questions = load_questions(args.questions)
    except InvalidQuestionSet as error:
        print(f"Question set is invalid: {error}", file=sys.stderr)
        return 2

    retriever = _build_retriever(args.corpus)
    report = evaluate_retrieval(retriever, questions, top_k=args.top_k)
    args.captured_identity = _run_identity(args, report)

    answerable = len(report.answerable)
    print(
        f"{len(questions)} questions ({answerable} answerable, "
        f"{len(report.unanswerable)} not)\n"
    )
    _print_retrieval(report)
    failures = retrieval_failures(report, args.suite)
    for failure in failures:
        print(f"GATE FAIL: {failure}", file=sys.stderr)

    # Paid answering is OFF unless explicitly requested. A configured API key is
    # NOT a request: that conflation is exactly what turned a command intended as
    # a dry run into a real one. Nothing below this branch is reachable without
    # --paid, including the imports that construct a provider client.
    if not args.paid:
        print("")
        print("Answering")
        print("  skipped: --paid was not given, so no provider call was made.")
        print("  Retrieval scores above are complete and cost nothing.")
        # Build the evidence identity here too. It needs no credential, makes
        # no request and costs nothing, and it is the only thing standing
        # between a configuration that cannot be recorded and the single moment
        # that fault would otherwise surface: the paid confirmation prompt,
        # with the spend already authorised and the allowance non-renewing.
        # That is not hypothetical. On 14 September 2026 three out-of-range
        # bounds reached exactly that point, because nothing free constructed
        # this record. Since the spend ceilings left `AnswerConfiguration`,
        # this is byte-identical to what the paid capture will record, so a
        # clean rehearsal is now a real rehearsal of the identity.
        config: dict[str, object] | None = None
        try:
            settings = Settings(retrieval_top_k=args.top_k)
        except Exception:
            # A half-configured credential environment must not break a free
            # retrieval run: "a configured credential alone never triggers
            # inference" cuts both ways, and this path has to work on any
            # machine. Say the identity was not checked and carry on with the
            # exit code retrieval earned. Settings validation errors quote the
            # values they rejected and some are credentials, so never print it.
            print("  identity: NOT checked -- settings did not construct here.")
        else:
            try:
                config = _answer_configuration(settings, args.top_k)
            except (TypeError, ValueError) as error:
                # Settings constructed and the identity still cannot be built.
                # That is the defect this path exists to catch, so it is fatal
                # here rather than at the paid prompt. AnswerConfiguration is
                # non-secret by construction, so unlike the branch above this
                # one can say what is actually wrong -- the message that was
                # missing when it mattered.
                print(
                    f"Answer configuration is not recordable evidence: {error}",
                    file=sys.stderr,
                )
                return 2
            print("  identity: built and validated; a capture records this.")
        if args.output:
            payload = args.captured_identity
            if config is not None:
                payload = {**payload, "config": config}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 1 if failures else 0

    if args.allowance_ledger is None or not args.allowance_id:
        print(
            "Paid evaluation capture is disabled without an existing carried-forward "
            "qualification allowance (--allowance-ledger and --allowance-id).",
            file=sys.stderr,
        )
        return 2
    if (
        args.max_paid_calls is None
        or args.max_paid_calls < 1
        or args.output is None
        or args.spec_version != SPEC_VERSION
    ):
        print(
            "Paid capture requires a positive call ceiling, output and current spec.",
            file=sys.stderr,
        )
        return 2
    if failures:
        print("Retrieval gate failed; no paid calls made.", file=sys.stderr)
        return 1

    from assistant.answering import Answerer, message_creator
    from assistant.capture import CaptureBudget, QualificationAllowance, capture_answers
    from assistant.persistent_budget import PersistentBudget, service_budget

    try:
        settings = Settings(retrieval_top_k=args.top_k)
        if not settings.answering_enabled:
            raise ValueError("complete verified provider pair required")
        service = service_budget(settings)
        if not isinstance(service, PersistentBudget):
            raise ValueError("capture requires durable service accounting")
        allowance = QualificationAllowance(args.allowance_ledger, args.allowance_id)
        budget = CaptureBudget(service, allowance, args.max_paid_calls)
        # A finite conservative plan: every case could require both attempts.
        # Policy/prefilter cases normally use zero; that never increases authority.
        maximum = 2 * len(questions)
        if maximum > budget.remaining:
            print(
                f"Capture needs room for at most {maximum} attempts; only "
                f"{budget.remaining} are available across all ceilings. No calls made.",
                file=sys.stderr,
            )
            return 2
        prompt = load_system_prompt(settings)
        identity = {
            **args.captured_identity,
            "prompt_sha256": (
                file_digest(settings.system_prompt_file)
                if settings.system_prompt_file
                else digest(prompt.encode())
            ),
            "config": _answer_configuration(settings, args.top_k),
            "qualification_max_paid_calls": args.max_paid_calls,
            "qualification_allowance_id": args.allowance_id,
            "pricing_review_date": "2026-09-08",
            "reason": args.reason or "",
        }
        # No API requests happen during construction. Exclusive output creation
        # inside capture_answers must succeed before any job is dispatched.
        answers = capture_answers(
            Answerer(
                retriever, message_creator(settings), settings, system_prompt=prompt
            ),
            questions,
            settings,
            budget,
            args.output,
            identity,
        )
    except Exception:
        # Never print provider/configuration exceptions or secrets. Partial output
        # and durable reservations are retained; do not silently retry this run.
        print(
            "Capture could not complete. Retain partial evidence and all reservations.",
            file=sys.stderr,
        )
        return 2
    print(
        f"Saved {len(answers.outcomes)} cases and {answers.paid_calls} attempted calls."
    )
    print("Human claim review is required; this capture does not approve release.")
    return 1 if answer_failures(answers) else 0


def _run_identity(
    args: argparse.Namespace, report: RetrievalReport
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "spec_version": SPEC_VERSION,
        "suite": args.suite,
        "corpus_sha256": _corpus_digest(args.corpus),
        "questions_sha256": file_digest(args.questions or DEFAULT_QUESTIONS),
        "evaluator_sha256": file_digest(Path(__file__).with_name("evaluation.py")),
        "reviewer_code_sha256": file_digest(
            Path(__file__).with_name("release_evaluation.py")
        ),
        "policy_sha256": file_digest(Path(__file__).with_name("policy.py")),
        "answer_contract_version": ANSWER_CONTRACT_VERSION,
        "answer_runtime_sha256": {
            name: file_digest(Path(__file__).with_name(name))
            for name in ANSWER_RUNTIME_SOURCES
        },
        "runtime_lock_sha256": file_digest(
            Path(__file__).parents[2] / "requirements-runtime.lock"
        ),
        "model_lock_sha256": file_digest(Path(__file__).parents[2] / "model.lock.json"),
        "top_k": args.top_k,
        "raw_retrieval": asdict(report),
    }


def cmd_review(args: argparse.Namespace) -> int:
    """Review an existing artifact; this path cannot spend provider credit."""
    from pydantic import ValidationError

    try:
        run = json.loads(args.run.read_text(encoding="utf-8"))
        if args.template:
            payload = review_template(run)
            code = 0
        else:
            if args.review is None:
                raise ValueError("--review or --template is required")
            review = HumanReview.model_validate_json(
                args.review.read_text(encoding="utf-8")
            )
            payload = check_review(run, review)
            code = 0 if payload["passed"] else 1
        if args.output.resolve() in {
            args.run.resolve(),
            args.review.resolve() if args.review else args.run.resolve(),
        }:
            raise ValueError("output must not overwrite run or review evidence")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            "Review template written"
            if args.template
            else f"Reviewed gate: {'PASS' if code == 0 else 'FAIL'}"
        )
        return code
    except (OSError, ValueError, KeyError, TypeError, ValidationError) as error:
        print(f"Cannot review evaluation: {error}", file=sys.stderr)
        return 2


def _corpus_digest(corpus: Path) -> str:
    """The corpus fingerprint, recorded with every saved run.

    A result that cannot name the corpus it scored is not evidence about
    anything in particular.
    """
    from assistant.corpus_checksum import corpus_checksum

    return corpus_checksum(corpus)


def _answer_configuration(settings: Settings, top_k: int) -> dict[str, object]:
    """Return the non-secret answer behavior identity used by release evidence."""
    from assistant.transport import WIRE_VERSION

    return AnswerConfiguration.model_validate(
        {
            "primary_model": settings.answer_model,
            "fallback_model": settings.fallback_answer_model,
            "answer_effort": settings.answer_effort,
            "answer_max_tokens": settings.answer_max_tokens,
            "top_k": top_k,
            "prefilter_score": settings.prefilter_score,
            "backend_timeout_seconds": settings.backend_timeout_seconds,
            "provider_timeout_seconds": settings.provider_timeout_seconds,
            "primary_timeout_seconds": settings.primary_timeout_seconds,
            "validation_margin_seconds": settings.validation_margin_seconds,
            "wire_version": WIRE_VERSION,
            "max_attempts": 2,
            "retries": 0,
            "fallback_mode": "availability_only",
            "complete_pair_required": True,
            "max_provider_request_bytes": 32000,
            "attempt_reservation_micro_usd": 40000,
            "shared_worker_limit": 1,
            "budget_storage": "persistent_local_sqlite",
        }
    ).model_dump()


def _write_run(
    path: Path,
    args: argparse.Namespace,
    report: RetrievalReport,
    answers: AnswerReport,
    settings: Settings,
) -> None:
    """Persist the complete run.

    A paid run that is not saved has to be paid for twice — which is not a
    hypothetical: an earlier run of this harness was piped through `tail` and
    the results were lost, leaving the spend with nothing to show for it.

    Every answer is written in full rather than truncated for display, because
    the reason to keep a run is to be able to re-read the cases the summary
    only counted.
    """
    payload = {
        **args.captured_identity,
        "raw_answering": asdict(answers),
        "prompt_sha256": args.captured_prompt_sha256,
        "config": _answer_configuration(settings, args.top_k),
        "capture": {
            "corpus": str(args.corpus),
            "questions": str(args.questions) if args.questions else "default",
            "reason": args.reason or "",
            "spec_version": args.spec_version or "",
            "corpus_checksum": _corpus_digest(args.corpus),
        },
        "retrieval": {
            "hit_rate": report.hit_rate,
            "top_1_rate": report.top_1_rate,
            "critical_hit_rate": report.critical_hit_rate,
            "critical_misses": [o.question.text for o in report.critical_misses],
        },
        "answering": {
            "task_success": answers.task_success,
            "critical_task_success": answers.critical_task_success,
            "safety_success": answers.safety_success,
            "critical_false_refusals": len(answers.critical_false_refusals),
            "unreviewed": len(answers.unreviewed),
            "materially_unsupported": (
                None if answers.unreviewed else len(answers.materially_unsupported)
            ),
            "safety_violations": [
                {"question": o.question.text, "violations": list(o.safety_violations)}
                for o in answers.safety_violations
            ],
            "accuracy": answers.accuracy,
            "refusal_accuracy": answers.refusal_accuracy,
            "false_refusal_rate": answers.false_refusal_rate,
            "unverifiable_citations": answers.unverifiable_citations,
            "accepted_citations": answers.accepted_citations,
            "truncated": len(answers.truncated),
            "unsupported_prose": len(answers.unsupported_prose),
        },
        "usage": {
            "paid_calls": answers.paid_calls,
            "input_tokens": answers.input_tokens,
            "output_tokens": answers.output_tokens,
            "cost_usd": answers.cost_usd,
        },
        "outcomes": [
            {
                "question": o.question.text,
                "answerable": o.question.answerable,
                "critical": o.question.critical,
                "expects": o.question.expects,
                "class": o.question.outcome_class,
                "task_success": o.task_success,
                "safety_violations": list(o.safety_violations),
                "materially_unsupported": o.materially_unsupported,
                "correct": o.correct,
                "grounded": o.grounded,
                "refused": o.refused,
                "cited_expected": o.cited_expected,
                "accepted_citations": o.accepted_citations,
                "rejected_citations": o.rejected_citations,
                "input_tokens": o.input_tokens,
                "output_tokens": o.output_tokens,
                "stop_reason": o.stop_reason,
                "truncated": o.truncated,
                "unsupported_prose": o.unsupported_prose,
                "text": o.text,
            }
            for o in answers.outcomes
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cmd_ask(args: argparse.Namespace) -> int:
    settings = Settings()
    if not settings.answering_enabled:
        print(
            "The complete verified primary and fallback configuration is unavailable.\n"
            "`index` and `eval` work without one.",
            file=sys.stderr,
        )
        return 2

    from assistant.answering import Answerer, message_creator

    retriever = _build_retriever(args.corpus)
    import time

    from assistant.persistent_budget import PersistentBudget, service_budget
    from assistant.runtime import BoundedAnswerExecutor, ExecutionContext

    budget = service_budget(settings)
    if not isinstance(budget, PersistentBudget):
        print("A durable budget is required for answering.", file=sys.stderr)
        return 2
    answerer = Answerer(retriever, message_creator(settings), settings)
    context = ExecutionContext(
        time.monotonic() + settings.backend_timeout_seconds, budget
    )
    executor: BoundedAnswerExecutor[Answer] = BoundedAnswerExecutor(
        1, acquire_shared_worker=budget.acquire_worker
    )
    try:
        answer = executor.submit(
            lambda ctx: answerer.answer(args.question, context=ctx), context
        ).result(timeout=settings.backend_timeout_seconds)
    except Exception:
        context.cancel()
        print("Answer unavailable. Any reservation is retained.", file=sys.stderr)
        return 2
    finally:
        executor.shutdown(wait=True)

    print(answer.text)
    if answer.citations:
        print("\nSources")
        for source in answer.sources:
            print(f"  {source}")
    else:
        # Stated plainly rather than left for the reader to infer from silence.
        print("\n(No citation — this is not an answer grounded in the documents.)")

    if answer.rejected_citations:
        print(
            f"\nWarning: {answer.rejected_citations} citation(s) rejected because "
            "the quoted text was not in the passage supplied."
        )
    return 0


def _named_path(value: str) -> tuple[str, Path]:
    """Parse a human-readable LABEL=PATH option without touching the path."""
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label.strip(), Path(raw_path.strip())


def _deployment_label(name: str) -> str:
    words = name.replace("_", "-").split("-")
    acronyms = {"oj": "OJ", "rag": "RAG"}
    return " ".join(acronyms.get(word.casefold(), word.title()) for word in words)


def _inspector_profiles(args: argparse.Namespace) -> list[CorpusProfile]:
    """Resolve fixed startup profiles; browser input can never select a path."""
    from assistant.inspection import InspectionError

    configured: list[tuple[str, Path]] = list(args.corpus_profile)
    if not configured:
        configured.append(("Cited", args.corpus))
        deployments = args.corpus.parent / "deploy"
        if deployments.is_dir():
            for content in sorted(deployments.glob("*/content")):
                if content.is_dir():
                    configured.append((_deployment_label(content.parent.name), content))

    vectors_by_id: dict[str, Path] = {}
    for label, path in args.vectors:
        profile_id = CorpusProfile.create(label, Path(".")).id
        if profile_id in vectors_by_id:
            raise InspectionError(f"Duplicate vectors label: {label}")
        vectors_by_id[profile_id] = path

    profiles = [
        CorpusProfile.create(
            label,
            path,
            vectors_by_id.get(CorpusProfile.create(label, path).id),
        )
        for label, path in configured
    ]
    profile_ids = {profile.id for profile in profiles}
    unknown_vectors = sorted(set(vectors_by_id) - profile_ids)
    if unknown_vectors:
        raise InspectionError(
            "Vectors were supplied for an unknown corpus label: "
            + ", ".join(unknown_vectors)
        )
    if len(profile_ids) != len(profiles):
        raise InspectionError("Corpus labels must produce unique identifiers.")
    return profiles


def cmd_inspect(args: argparse.Namespace) -> int:
    """Start the free, loopback-only read-only inspection interface."""
    import uvicorn

    from assistant.inspection import InspectionError, inspect_corpus
    from assistant.inspector import create_inspector_app

    try:
        profiles = _inspector_profiles(args)
        snapshots = [inspect_corpus(profile) for profile in profiles]
    except InspectionError as error:
        print(f"Cannot start inspector: {error}", file=sys.stderr)
        return 2

    url = f"http://127.0.0.1:{args.port}"
    labels = ", ".join(snapshot.label for snapshot in snapshots)
    print(f"Cited RAG Management Panel: {url}")
    print(f"Corpora: {labels}")
    print("Read-only and local to this computer. Press Ctrl+C to stop.")
    uvicorn.run(
        create_inspector_app(snapshots),
        host="127.0.0.1",
        port=args.port,
        log_level="info",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="doc-assistant",
        description="Answer questions from your documents, with verifiable citations.",
    )
    parser.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS, help="document directory"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    index = subcommands.add_parser("index", help="inspect what retrieval will see")
    index.add_argument("--verbose", action="store_true", help="list every chunk")
    index.set_defaults(func=cmd_index)

    embed = subcommands.add_parser(
        "embed", help="build the corpus vectors a deployment serves from"
    )
    embed.add_argument(
        "--out",
        type=Path,
        required=True,
        help="destination .npz. Required rather than defaulted: this file is "
        "read at startup, and writing one somewhere unintended is a failure "
        "that only shows up on a deploy.",
    )
    embed.set_defaults(func=cmd_embed)

    evaluate = subcommands.add_parser("eval", help="score against the question set")
    evaluate.add_argument("--questions", type=Path, default=None)
    evaluate.add_argument("--top-k", type=int, default=4)
    evaluate.add_argument("--suite", choices=("demo", "portfolio"), default="demo")
    evaluate.add_argument(
        "--max-paid-calls",
        type=int,
        default=None,
        help=(
            "hard stop on provider calls; mandatory with --paid. The run "
            "raises rather than truncating, so a partial evaluation is never "
            "reported as a complete one."
        ),
    )
    evaluate.add_argument(
        "--paid",
        action="store_true",
        help=(
            "REQUIRED to make any provider call. Without it the command scores "
            "retrieval and stops, whether or not an API key is configured. The "
            "default is free; spending is opt-in and must be stated."
        ),
    )
    evaluate.add_argument(
        "--spec-version",
        type=str,
        default=None,
        help=(
            "evaluation specification version, recorded in the preflight and in "
            "the saved results so a run can be tied to the rules it was scored by."
        ),
    )
    evaluate.add_argument(
        "--reason",
        type=str,
        default=None,
        help="recorded in the preflight summary, e.g. who authorised the spend.",
    )
    evaluate.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "write the complete run to a JSON file. A paid run that is not "
            "saved has to be paid for twice."
        ),
    )
    evaluate.add_argument(
        "--allowance-ledger",
        type=Path,
        help="existing non-renewing qualification ledger; never auto-created",
    )
    evaluate.add_argument(
        "--allowance-id", help="identity of the approved carried-forward allowance"
    )
    evaluate.set_defaults(func=cmd_eval)

    review = subcommands.add_parser(
        "review", help="review saved evaluation evidence; free"
    )
    review.add_argument("--run", type=Path, required=True)
    review.add_argument("--review", type=Path)
    review.add_argument("--template", action="store_true")
    review.add_argument("--output", type=Path, required=True)
    review.set_defaults(func=cmd_review)

    ask = subcommands.add_parser("ask", help="answer one question (costs an API call)")
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)

    inspect = subcommands.add_parser(
        "inspect", help="open the local, read-only RAG management panel"
    )
    inspect.add_argument(
        "--corpus-profile",
        action="append",
        type=_named_path,
        default=[],
        metavar="LABEL=PATH",
        help=(
            "corpus to show; repeat for multiple corpora. By default, uses "
            "--corpus as Cited and discovers deploy/*/content directories"
        ),
    )
    inspect.add_argument(
        "--vectors",
        action="append",
        type=_named_path,
        default=[],
        metavar="LABEL=PATH",
        help="optional vectors file to validate for a matching corpus label",
    )
    inspect.add_argument(
        "--port",
        type=int,
        choices=range(1, 65536),
        default=DEFAULT_INSPECTOR_PORT,
        metavar="PORT",
    )
    inspect.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
