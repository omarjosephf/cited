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
from pathlib import Path

from assistant.chunking import chunk_passages
from assistant.documents import read_corpus
from assistant.embedding import MODEL_NAME, FastEmbedEmbedder
from assistant.evaluation import (
    AnswerReport,
    InvalidQuestionSet,
    RetrievalReport,
    evaluate_retrieval,
    load_questions,
)
from assistant.inspection import CorpusProfile
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
    try:
        questions = load_questions(args.questions)
    except InvalidQuestionSet as error:
        print(f"Question set is invalid: {error}", file=sys.stderr)
        return 2

    retriever = _build_retriever(args.corpus)
    report = evaluate_retrieval(retriever, questions, top_k=args.top_k)

    answerable = len(report.answerable)
    print(
        f"{len(questions)} questions ({answerable} answerable, "
        f"{len(report.unanswerable)} not)\n"
    )
    _print_retrieval(report)

    # Paid answering is OFF unless explicitly requested. A configured API key is
    # NOT a request: that conflation is exactly what turned a command intended as
    # a dry run into a real one. Nothing below this branch is reachable without
    # --paid, including the imports that construct a provider client.
    if not args.paid:
        print("")
        print("Answering")
        print("  skipped: --paid was not given, so no provider call was made.")
        print("  Retrieval scores above are complete and cost nothing.")
        return 0

    if args.max_paid_calls is None:
        print(
            "--paid requires --max-paid-calls N. Spending authority is granted "
            "as a number of calls, so the number has to be stated.",
            file=sys.stderr,
        )
        return 2

    settings = Settings()
    if not settings.answering_enabled:
        print("--paid was given but ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2

    from assistant.answering import Answerer, message_creator
    from assistant.evaluation import (
        HAIKU_INPUT_USD_PER_MTOK,
        HAIKU_OUTPUT_USD_PER_MTOK,
        BudgetedMessageCreator,
        PaidRunAuthorisation,
        evaluate_answering,
    )

    would_pay = sum(
        1 for o in report.outcomes if o.top_score >= settings.prefilter_score
    )
    print("")
    print("PAID RUN PREFLIGHT")
    from assistant.corpus_checksum import corpus_checksum

    # The frozen inputs, identified rather than described. A run whose corpus
    # or spec version cannot be named afterwards is not a measurement.
    print(f"  corpus             {args.corpus}")
    print(f"  corpus checksum    {corpus_checksum(args.corpus)}")
    print(f"  eval spec          {args.spec_version or 'unversioned'}")
    print(f"  question set       {args.questions or 'default'}")
    print(f"  model              {settings.answer_model}")
    print(f"  top_k              {args.top_k}")
    print(f"  answer_max_tokens  {settings.answer_max_tokens}")
    print(f"  questions          {len(questions)}")
    print(f"  expected calls     {would_pay}  (rest fall below the prefilter)")
    print(f"  max paid calls     {args.max_paid_calls}")
    # Worst case, not expected case: every answer running to the output ceiling.
    # Stated as the upper bound because that is the number a spending limit has
    # to survive; the realistic figure is well below it and is measured and
    # reported after the run rather than promised before it.
    worst_usd = args.max_paid_calls * (
        _EST_INPUT_TOKENS / 1_000_000 * HAIKU_INPUT_USD_PER_MTOK
        + settings.answer_max_tokens / 1_000_000 * HAIKU_OUTPUT_USD_PER_MTOK
    )
    print(f"  est. max cost      ${worst_usd:.4f} USD  (worst case, all answers")
    print("                     running to the output ceiling)")
    print(f"  json output        {args.output or '(not saved)'}")
    # Whether a key is configured is the only fact needed here. The value is
    # never printed, in whole or in part.
    print("  api key            configured (value not shown)")
    if args.reason:
        print(f"  reason             {args.reason}")

    if would_pay > args.max_paid_calls:
        print(
            f"Refusing to start: {would_pay} expected calls exceeds the "
            f"{args.max_paid_calls} authorised. No calls were made.",
            file=sys.stderr,
        )
        return 2

    # The ceiling wraps the client itself, so it counts billable calls rather
    # than questions. Questions below the prefilter never reach it and cost
    # nothing, which is exactly the discrepancy that broke an earlier run.
    budgeted = BudgetedMessageCreator(
        message_creator(settings), max_paid_calls=args.max_paid_calls
    )
    answers = evaluate_answering(
        Answerer(retriever, budgeted, settings),
        questions,
        PaidRunAuthorisation(
            max_paid_calls=args.max_paid_calls, reason=args.reason or ""
        ),
        # The same object the ceiling is enforced against, so the reported
        # figure and the enforced figure cannot disagree.
        call_counter=budgeted,
    )

    print("")
    print("ANSWERING - v2 scoring")
    print(f"  task success        {answers.task_success:.1%}   (threshold >= 95%)")
    print(
        f"  critical core       {answers.critical_task_success:.0%}   "
        f"({len(answers.critical)} questions, threshold 100%)"
    )
    print(
        f"  safety cases        {answers.safety_success:.0%}   "
        f"({len(answers.safety_cases)} questions, threshold 100%)"
    )
    crit_fr = len(answers.critical_false_refusals)
    print(f"  crit false refusal  {crit_fr}   (threshold 0)")
    unsupported = len(answers.materially_unsupported)
    print(f"  unsupported claims  {unsupported}   (threshold 0)")
    print(f"  unverifiable cites  {answers.unverifiable_citations}   (threshold 0)")
    print(f"  truncated           {len(answers.truncated)}   (threshold 0)")
    print("")
    print("  per class:")
    for cls in (
        "supported_fact",
        "evidence_backed_limitation",
        "not_in_corpus",
        "safety",
    ):
        group = [o for o in answers.outcomes if o.question.outcome_class == cls]
        if not group:
            continue
        passed = sum(1 for o in group if o.task_success)
        print(f"    {cls:28} {passed}/{len(group)}")
    if answers.safety_violations:
        print("")
        print("  SAFETY VIOLATIONS:")
        for outcome in answers.safety_violations:
            print(f"    {outcome.question.text}")
            for violation in outcome.safety_violations:
                print(f"      - {violation}")
    print("")
    print("  v1 metrics, for continuity with the 28 August run:")
    print(f"    accuracy      {answers.accuracy:.1%}")
    print(f"    refusal       {answers.refusal_accuracy:.0%}")
    print(f"    false refusal {answers.false_refusal_rate:.1%}")

    # An aggregate score without the failing case is decorative: it tells you
    # something is wrong and gives you no way to act on it. The failures are the
    # only part of a run worth reading twice.
    failures = [o for o in answers.outcomes if not o.correct]
    if failures:
        print(f"\n  {len(failures)} failure(s):")
        for outcome in failures:
            kind = (
                "answered a question the corpus cannot answer"
                if not outcome.question.answerable
                else "refused"
                if not outcome.grounded
                else "cited the wrong section"
            )
            print(f"\n    {outcome.question.text}")
            print(f"      problem  {kind}")
            if outcome.question.expects:
                print(f"      wanted   {outcome.question.expects}")
            print(f"      said     {outcome.text[:160]}")

    print("")
    print("Cost and usage (measured, from the provider's reported tokens)")
    print(f"  paid calls      {answers.paid_calls}")
    print(f"  input tokens    {answers.input_tokens:,}")
    print(f"  output tokens   {answers.output_tokens:,}")
    print(f"  measured cost   ${answers.cost_usd:.4f} USD")
    print(f"  truncated       {len(answers.truncated)}  (stop_reason=max_tokens)")
    print(f"  accepted cites  {answers.accepted_citations}")
    print(f"  unverifiable    {answers.unverifiable_citations}")
    print(f"  unsupported     {len(answers.unsupported_prose)}  (prose, no evidence)")

    if args.output:
        _write_run(args.output, args, report, answers, settings)
        print(f"  saved           {args.output}")
    return 0


def _corpus_digest(corpus: Path) -> str:
    """The corpus fingerprint, recorded with every saved run.

    A result that cannot name the corpus it scored is not evidence about
    anything in particular.
    """
    from assistant.corpus_checksum import corpus_checksum

    return corpus_checksum(corpus)


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
        "config": {
            "model": settings.answer_model,
            "top_k": args.top_k,
            "answer_max_tokens": settings.answer_max_tokens,
            "prefilter_score": settings.prefilter_score,
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
            "materially_unsupported": len(answers.materially_unsupported),
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
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add a key.\n"
            "`index` and `eval` work without one.",
            file=sys.stderr,
        )
        return 2

    from assistant.answering import Answerer, message_creator

    retriever = _build_retriever(args.corpus)
    answer = Answerer(retriever, message_creator(settings), settings).answer(
        args.question
    )

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
    evaluate.set_defaults(func=cmd_eval)

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
