"""HTTP interface.

A thin wrapper over the library. Everything that decides an answer lives in
`answering.py` and `retrieval.py`; this module owns transport, protection and
presentation, and nothing else. That boundary is what makes a different
deployment a wrapper rather than a rewrite.

Three protections, because a public endpoint that makes paid calls needs all
three and none of them substitutes for another:

* **Rate limiting** bounds how fast money leaves.
* **The daily budget** bounds how much leaves in total.
* **A question length cap** bounds the size of any single call.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from assistant.answering import Answerer, message_creator
from assistant.budget import BudgetExhausted, DailyCallBudget
from assistant.chunking import chunk_passages
from assistant.documents import read_corpus
from assistant.embedding import FastEmbedEmbedder
from assistant.retrieval import InMemoryRetriever
from assistant.settings import Settings

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 500
"""Longer than any real question, short enough to bound the cost of one call.

The input is what the caller controls, so it is what needs a limit. Without
one, a single request can be arbitrarily expensive.
"""

limiter = Limiter(key_func=get_remote_address)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class CitationOut(BaseModel):
    quote: str
    source: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    grounded: bool
    """False means this is not an answer from the documents — a refusal, or
    something unsupported. Exposed so a client cannot present ungrounded prose
    as though it were sourced."""
    refused: bool


class State:
    """Built once at startup, shared by every request."""

    settings: Settings
    answerer: Answerer
    budget: DailyCallBudget
    chunk_count: int


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Index the corpus before serving.

    Done at startup rather than per request because embedding the corpus takes
    seconds, and because a corpus that fails to load should stop the process
    rather than surface as an error on someone's first question.
    """
    settings = Settings()
    corpus = Path("content")

    passages = read_corpus(corpus)
    if not passages:
        raise RuntimeError(f"No documents found in {corpus}/. Nothing to serve.")

    chunks = chunk_passages(passages)
    retriever = InMemoryRetriever(chunks, FastEmbedEmbedder())

    state.settings = settings
    state.chunk_count = len(chunks)
    state.budget = DailyCallBudget(limit=settings.daily_answer_limit)
    state.answerer = Answerer(retriever, message_creator(settings), settings)

    logger.info(
        "ready: %d chunks, model %s, daily limit %d",
        len(chunks),
        settings.answer_model,
        settings.daily_answer_limit,
    )
    yield


app = FastAPI(
    title="Document Assistant",
    description="Answers from a fixed set of documents, with verifiable citations.",
    lifespan=lifespan,
)
app.state.limiter = limiter
# slowapi's handler is typed for its own exception rather than the base
# `Exception` Starlette declares, so the cast is a typing accommodation, not a
# silenced error: the handler is only ever invoked for RateLimitExceeded.
app.add_exception_handler(RateLimitExceeded, cast(Any, _rate_limit_exceeded_handler))


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness plus the two numbers an operator actually needs."""
    return {
        "status": "ok",
        "chunks": state.chunk_count,
        "answers_remaining_today": state.budget.remaining,
    }


@app.post("/ask", response_model=AskResponse)
@limiter.limit("10/minute")
async def ask(request: Request, body: AskRequest) -> AskResponse:
    """Answer one question from the corpus.

    `request` is unused here but required: slowapi resolves the client address
    from it, and omitting it makes the decorator fail at runtime rather than at
    import.
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    # Reserved before the call, not recorded after it. Recording afterwards
    # would let concurrent requests all pass the check and then all spend.
    try:
        state.budget.spend()
    except BudgetExhausted:
        raise HTTPException(
            status_code=503,
            detail=(
                "This demo has reached its daily limit of answered questions. "
                "It resets at midnight UTC."
            ),
        ) from None

    try:
        answer = state.answerer.answer(question)
    except Exception:
        state.budget.refund()
        # Logged without the question: it is user input, and a log is a place
        # data goes to be retained and read by people it was not sent to.
        logger.exception("answering failed")
        raise HTTPException(
            status_code=502, detail="The answering service is unavailable."
        ) from None

    if answer.rejected_citations:
        # Should be impossible: the API computes citations from the documents
        # supplied. A non-zero count means an assumption has broken, so it is
        # logged loudly rather than silently discarded.
        logger.warning(
            "rejected %d citation(s): quoted text absent from the supplied passage",
            answer.rejected_citations,
        )

    return AskResponse(
        answer=answer.text,
        citations=[
            CitationOut(quote=c.quoted_text, source=c.source) for c in answer.citations
        ],
        grounded=answer.grounded,
        refused=answer.refused,
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    page = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.exception_handler(500)
async def internal_error(request: Request, exc: Exception) -> JSONResponse:
    """Never leak internals to a caller.

    A stack trace tells an attacker about paths, versions and structure. It goes
    to the log, where the operator can see it; the caller gets a fixed string.
    """
    logger.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal error."})
