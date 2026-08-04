"""Command line entry point.

Three commands, split by what they cost:

* `index`  — read the corpus and report what retrieval will see. Free.
* `eval`   — score retrieval against the committed question set. Free.
* `ask`    — answer one question. Costs an API call.

The free commands come first deliberately. Most of what goes wrong in a
retrieval system goes wrong before the model is involved, and being able to
inspect and score that without spending anything is what makes it cheap to check
often.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from assistant.chunking import chunk_passages
from assistant.documents import read_corpus
from assistant.embedding import FastEmbedEmbedder
from assistant.evaluation import (
    InvalidQuestionSet,
    RetrievalReport,
    evaluate_retrieval,
    load_questions,
)
from assistant.retrieval import InMemoryRetriever
from assistant.settings import Settings

DEFAULT_CORPUS = Path("content")


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
    print(f"  separation      {report.score_separation:+.3f}")
    if report.score_separation < 0:
        print("                  negative: answerable and unanswerable score ranges")
        print("                  overlap, so no similarity threshold separates them")
        print("                  (ADR-0002)")

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

    settings = Settings()
    if not settings.answering_enabled:
        print("\nAnswering")
        print("  skipped: ANTHROPIC_API_KEY is not set.")
        print("  Retrieval scores above are complete and cost nothing; answering")
        print("  and refusal accuracy need a key.")
        return 0

    from assistant.answering import Answerer, message_creator
    from assistant.evaluation import evaluate_answering

    answers = evaluate_answering(
        Answerer(retriever, message_creator(settings), settings), questions
    )

    print("\nAnswering")
    print(f"  accuracy        {answers.accuracy:.0%}")
    print(f"  refusal         {answers.refusal_accuracy:.0%}")
    print("                  unanswerable questions correctly refused")
    print(f"  false refusal   {answers.false_refusal_rate:.0%}")
    print("                  answerable questions wrongly refused")
    print(f"  bad citations   {answers.unverifiable_citations}")
    print("                  quotes not present in the passage we sent")
    return 0


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

    evaluate = subcommands.add_parser("eval", help="score against the question set")
    evaluate.add_argument("--questions", type=Path, default=None)
    evaluate.add_argument("--top-k", type=int, default=4)
    evaluate.set_defaults(func=cmd_eval)

    ask = subcommands.add_parser("ask", help="answer one question (costs an API call)")
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
