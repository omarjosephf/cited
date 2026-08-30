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

import asyncio
import logging
import secrets
import time
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

from assistant.answering import (
    MAX_HISTORY_SOURCES,
    MAX_HISTORY_TURNS,
    Answerer,
    Turn,
    load_system_prompt,
    message_creator,
)
from assistant.budget import BudgetExhausted, DailyCallBudget
from assistant.chunking import chunk_passages
from assistant.corpus_checksum import verify_corpus
from assistant.documents import read_corpus
from assistant.embedding import MODEL_NAME, Embedder, FastEmbedEmbedder
from assistant.metrics import AssistantMetrics
from assistant.retrieval import InMemoryRetriever
from assistant.settings import Settings
from assistant.vectors import load as load_vectors

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 500
"""Longer than any real question, short enough to bound the cost of one call.

The input is what the caller controls, so it is what needs a limit. Without
one, a single request can be arbitrarily expensive.
"""

limiter = Limiter(key_func=get_remote_address)


class HistoryTurnIn(BaseModel):
    """One earlier exchange, supplied by the caller (ADR-0007 E2).

    The question and the labels of the documents that answered it. There is
    deliberately no field for the earlier ANSWER: the shape of this model is
    what stops generated passage text being replayed across the boundary, so
    the constraint lives in the contract rather than in a caller's good manners.
    """

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    sources: list[str] = Field(default_factory=list, max_length=MAX_HISTORY_SOURCES)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[HistoryTurnIn] = Field(
        default_factory=list, max_length=MAX_HISTORY_TURNS
    )
    """Earlier turns, oldest first. Optional in both directions on purpose: an
    older client omits it and is answered as a first turn, and a client that
    sends it against an older service has the field ignored (ADR-0007 E4)."""


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
    metrics: AssistantMetrics
    chunk_count: int
    corpus_checksum: str


state = State()

SECRET_HEADER = "X-Assistant-Secret"
"""Header carrying the shared secret when one is required.

A header rather than a query parameter: query strings are logged by proxies and
end up in browser history and referrers, which is a poor place for a credential.
"""


def require_caller_secret(request: Request) -> None:
    """Reject callers that cannot present the shared secret, when one is required.

    The point is narrow and worth stating so it is not oversold: this stops
    *someone who finds the hostname* from spending the owner's API budget. It is
    not authentication, there are no identities, and it does nothing whatsoever
    if the secret leaks.

    Compared with `compare_digest` rather than `==`, so the comparison does not
    return early on the first differing byte. Timing attacks on a header over a
    public network are close to impractical, but the constant-time version is one
    function call and needs no argument about whether the attack is feasible.
    """
    settings = state.settings
    if not settings.require_shared_secret:
        return

    expected = settings.shared_secret.get_secret_value()
    if not expected:
        # Configured to require a secret, with no secret. Failing closed is the
        # only safe reading: the alternative is serving unauthenticated while
        # believing otherwise.
        logger.error("require_shared_secret is set but shared_secret is empty")
        raise HTTPException(status_code=503, detail="Service is not configured.")

    presented = request.headers.get(SECRET_HEADER, "")
    if not secrets.compare_digest(presented, expected):
        # Deliberately says nothing about which part was wrong.
        raise HTTPException(status_code=401, detail="Not authorised.")


async def warm_embedder(embedder: Embedder) -> None:
    """Load the model off the startup path, once the service is already serving.

    With precomputed vectors nothing loads the model until a question arrives,
    which would move a cold start onto a visitor rather than removing it. This
    pays that cost in the background: the platform sees a healthy machine within
    seconds, and the model is ready well before anyone has finished typing.

    `to_thread` because the load is blocking CPU work and this runs on the event
    loop; without it a "background" warm-up would block every request it was
    supposed to protect.

    Failure is logged and swallowed on purpose. A warm-up is an optimisation —
    the first question loads the model itself if this did not — and taking the
    process down over a slow optimisation would trade a slow answer for none.
    """
    try:
        await asyncio.to_thread(embedder.embed_query, "warm-up")
    except Exception:
        logger.exception("embedder warm-up failed; the first question will load it")
    else:
        logger.info("embedder warm")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Index the corpus before serving.

    Done at startup rather than per request because a corpus that fails to load
    should stop the process rather than surface as an error on someone's first
    question.
    """
    settings = Settings()
    corpus = settings.corpus_dir

    if settings.require_shared_secret and not settings.shared_secret.get_secret_value():
        # Checked here as well as per request, so a misconfigured deployment
        # fails at startup rather than on a visitor's first question.
        raise RuntimeError(
            "REQUIRE_SHARED_SECRET is set but SHARED_SECRET is empty. "
            "Set the secret, or turn the requirement off deliberately."
        )

    # Before reading anything: is this the corpus that was approved? A stale or
    # partial copy answers confidently from the wrong content, and refusing to
    # start is the only response that cannot be mistaken for working.
    checksum = verify_corpus(corpus, settings.expected_corpus_checksum())

    passages = read_corpus(corpus)
    if not passages:
        raise RuntimeError(f"No documents found in {corpus}/. Nothing to serve.")

    chunks = chunk_passages(passages)
    embedder = FastEmbedEmbedder()

    # Precomputed or computed here, never "precomputed if it works": a
    # configured vectors file that does not describe this corpus raises out of
    # `load_vectors` and stops the process. See `vectors.py` for why silence is
    # the wrong response to that.
    matrix = None
    if settings.corpus_vectors_file is not None:
        matrix = load_vectors(
            settings.corpus_vectors_file,
            chunks,
            model=MODEL_NAME,
            dimensions=embedder.dimensions,
        )

    retriever = InMemoryRetriever(chunks, embedder, matrix)

    state.settings = settings
    state.chunk_count = len(chunks)
    state.corpus_checksum = checksum
    state.budget = DailyCallBudget(limit=settings.daily_answer_limit)
    state.metrics = AssistantMetrics()
    state.answerer = Answerer(
        retriever,
        message_creator(settings),
        settings,
        load_system_prompt(settings),
    )

    logger.info(
        "ready: %d chunks from %s, corpus %s, model %s, daily limit %d, "
        "shared secret %s",
        len(chunks),
        corpus,
        checksum[:12],
        settings.answer_model,
        settings.daily_answer_limit,
        "required" if settings.require_shared_secret else "not required",
    )

    # Started only when the corpus was not embedded here. Embedding it already
    # loaded the model, so a warm-up would be a second call that proves nothing.
    warmup = (
        asyncio.create_task(warm_embedder(embedder)) if matrix is not None else None
    )
    try:
        yield
    finally:
        if warmup is not None:
            warmup.cancel()


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
    """Liveness plus the numbers an operator actually needs.

    Deliberately unauthenticated and deliberately dull: it carries no visitor
    content and no aggregate behaviour, only whether the process is up and which
    corpus it is serving. The checksum prefix is what makes "did the new corpus
    actually deploy?" answerable without shell access.
    """
    return {
        "status": "ok",
        "chunks": state.chunk_count,
        "corpus": state.corpus_checksum[:12],
        "answers_remaining_today": state.budget.remaining,
    }


@app.get("/metrics")
async def metrics(request: Request) -> dict[str, Any]:
    """Aggregate operator metrics. Never public, never per-question.

    Behind the same secret as `/ask`, because how often an assistant refuses is
    operational information about someone's business rather than something a
    passer-by is owed. When no secret is required — the open demo — this endpoint
    is open too, which is consistent rather than accidental: that deployment has
    no operator whose numbers need protecting.
    """
    require_caller_secret(request)
    return state.metrics.snapshot(answers_remaining_today=state.budget.remaining)


@app.post("/ask", response_model=AskResponse)
@limiter.limit("10/minute")
async def ask(request: Request, body: AskRequest) -> AskResponse:
    """Answer one question from the corpus.

    `request` is unused by the body of this function but required: slowapi
    resolves the client address from it, and the secret check reads its headers.
    """
    require_caller_secret(request)

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    # Reserved before the call, not recorded after it. Recording afterwards
    # would let concurrent requests all pass the check and then all spend.
    try:
        state.budget.spend()
    except BudgetExhausted:
        state.metrics.record("unavailable", 0.0)
        raise HTTPException(
            status_code=503,
            detail=(
                "This service has reached its daily limit of answered "
                "questions. It resets at midnight UTC."
            ),
        ) from None

    # Mapped out of the request model rather than passed through it, so the
    # answering layer takes its own type and never a validated-but-foreign
    # object from the transport layer.
    history = tuple(
        Turn(turn.question.strip(), tuple(turn.sources)) for turn in body.history
    )

    started = time.monotonic()
    try:
        answer = state.answerer.answer(question, history)
    except Exception:
        state.budget.refund()
        state.metrics.record("unavailable", (time.monotonic() - started) * 1000)
        # Logged without the question: it is user input, and a log is a place
        # data goes to be retained and read by people it was not sent to.
        logger.exception("answering failed")
        raise HTTPException(
            status_code=502, detail="The answering service is unavailable."
        ) from None

    # Recorded by the outcome a visitor actually saw, so an operator reading
    # "not_covered" knows what was on screen. Still no question text.
    state.metrics.record(
        "answered" if answer.grounded else "not_covered",
        (time.monotonic() - started) * 1000,
        answer.rejected_citations,
    )

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


SECURITY_HEADERS = {
    # Two years, because a shorter max-age leaves a window where a downgrade
    # attack still works. Fly terminates TLS and redirects, but a header is what
    # stops the *first* request going out over HTTP next time.
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    # Without this a response the browser thinks might be script is treated as
    # script. Cheap, and there is no case where sniffing helps us.
    "X-Content-Type-Options": "nosniff",
    # Legacy twin of frame-ancestors, for anything that predates CSP.
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    # Nothing here uses any of these, so none should be reachable.
    "Permissions-Policy": "geolocation=(), microphone=(), camera=(), payment=()",
}

NONCE_PLACEHOLDER = "__CSP_NONCE__"
"""Substituted per response. A marker in the markup, never a real value."""


def _csp(nonce: str | None) -> str:
    """The policy, tightened as far as this page allows.

    `default-src 'none'` rather than `'self'`: the deny-by-default version means
    a directive that is missing fails closed. Every source below is one this
    page provably needs.

    The inline <style> and <script> carry a per-request nonce instead of
    'unsafe-inline'. They are the page's own code, but 'unsafe-inline' would
    also authorise anything injected into the markup later, which is the exact
    attack CSP exists to stop.
    """
    script = f"'nonce-{nonce}'" if nonce else "'none'"
    style = f"'nonce-{nonce}'" if nonce else "'none'"
    return "; ".join(
        [
            "default-src 'none'",
            f"script-src {script}",
            f"style-src {style}",
            # The page posts to /ask on its own origin and nowhere else.
            "connect-src 'self'",
            "img-src 'self' data:",
            "base-uri 'none'",
            "form-action 'none'",
            "frame-ancestors 'none'",
            "object-src 'none'",
            "upgrade-insecure-requests",
        ]
    )


@app.middleware("http")
async def security_headers(request: Request, call_next: Any) -> Any:
    """Applied to every response, including errors and rate-limit rejections.

    Set here rather than per route because the responses most likely to be
    forgotten — a 429, a 422, a 500 — are the ones written by the framework
    rather than by us.
    """
    response = await call_next(request)
    for header, value in SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    # The HTML route sets its own policy carrying that response's nonce; a
    # nonce reused across responses is no better than no nonce at all.
    response.headers.setdefault("Content-Security-Policy", _csp(None))
    return response


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    page = Path(__file__).parent / "static" / "index.html"
    # A fresh nonce per response, from the CSPRNG. `token_urlsafe` is base64url,
    # which is already valid inside a CSP source expression.
    nonce = secrets.token_urlsafe(16)
    html = page.read_text(encoding="utf-8").replace(NONCE_PLACEHOLDER, nonce)
    return HTMLResponse(html, headers={"Content-Security-Policy": _csp(nonce)})


@app.exception_handler(500)
async def internal_error(request: Request, exc: Exception) -> JSONResponse:
    """Never leak internals to a caller.

    A stack trace tells an attacker about paths, versions and structure. It goes
    to the log, where the operator can see it; the caller gets a fixed string.
    """
    logger.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal error."})
