"""HTTP interface.

A thin wrapper over the library. Everything that decides an answer lives in
`answering.py` and `retrieval.py`; this module owns transport, protection and
presentation, and nothing else. That boundary is what makes a different
deployment a wrapper rather than a rewrite.

Three protections, because a public endpoint that makes paid calls needs all
three and none of them substitutes for another:

* **Rate limiting** bounds how fast money leaves.
* **The durable budget** reserves combined API money and attempts before dispatch.
* **A question length cap** bounds the size of any single call.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, StringConstraints
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from assistant.answer_event import answer_event_header
from assistant.answering import (
    MAX_HISTORY_SOURCES,
    MAX_HISTORY_TURNS,
    Answerer,
    Turn,
    build_client,
    load_system_prompt,
)
from assistant.budget import AttemptBudget, BudgetExhausted
from assistant.chunking import chunk_passages
from assistant.corpus_checksum import verify_corpus
from assistant.documents import read_corpus
from assistant.embedding import MODEL_NAME, Embedder, FastEmbedEmbedder
from assistant.metrics import AssistantMetrics, MetricsIdentity
from assistant.persistent_budget import (
    BudgetCapacityError,
    BudgetUnavailable,
    PersistentBudget,
    service_budget,
)
from assistant.request_boundary import AskBoundaryMiddleware
from assistant.retrieval import InMemoryRetriever
from assistant.runtime import (
    BoundedAnswerExecutor,
    ExecutionContext,
    ExecutorCapacityError,
    ExecutorClosedError,
)
from assistant.settings import Settings
from assistant.transport import (
    AnsweredResponse,
    AskResponse,
    UnsupportedResponse,
    safe_response,
)
from assistant.vectors import load as load_vectors
from assistant.web_security import NONCE_PLACEHOLDER as NONCE_PLACEHOLDER
from assistant.web_security import apply_security_headers, content_security_policy

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
    sources: list[
        Annotated[str, StringConstraints(strict=True, min_length=1, max_length=80)]
    ] = Field(default_factory=list, max_length=MAX_HISTORY_SOURCES)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[HistoryTurnIn] = Field(
        default_factory=list, max_length=MAX_HISTORY_TURNS
    )
    """Earlier turns, oldest first. Optional in both directions on purpose: an
    older client omits it and is answered as a first turn, and a client that
    sends it against an older service has the field ignored (ADR-0007 E4)."""


class State:
    """Built once at startup, shared by every request."""

    settings: Settings
    answerer: Answerer
    budget: AttemptBudget
    metrics: AssistantMetrics
    chunk_count: int
    corpus_checksum: str
    prompt_checksum: str = ""
    executor: BoundedAnswerExecutor[
        tuple[AnsweredResponse | UnsupportedResponse, int, str | None]
    ]
    warmup: asyncio.Task[None] | None = None


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
    if not secrets.compare_digest(presented.encode("utf-8"), expected.encode("utf-8")):
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
        logger.warning("assistant_warmup_failed")
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
    state.budget = service_budget(settings)
    prompt = load_system_prompt(settings)
    state.prompt_checksum = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    state.metrics = AssistantMetrics(
        MetricsIdentity(
            model=f"{settings.answer_model}+{settings.fallback_answer_model}",
            corpus=checksum,
            prompt=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        )
    )
    state.executor = BoundedAnswerExecutor(
        settings.answer_workers,
        acquire_shared_worker=state.budget.acquire_worker
        if isinstance(state.budget, PersistentBudget)
        else None,
    )
    provider_router = build_client(settings)
    state.answerer = Answerer(
        retriever,
        provider_router,
        settings,
        prompt,
    )

    logger.info("assistant_ready")

    # Started only when the corpus was not embedded here. Embedding it already
    # loaded the model, so a warm-up would be a second call that proves nothing.
    warmup = (
        asyncio.create_task(warm_embedder(embedder)) if matrix is not None else None
    )
    state.warmup = warmup
    try:
        yield
    finally:
        if warmup is not None:
            await asyncio.gather(warmup, return_exceptions=True)
        await asyncio.to_thread(state.executor.shutdown, wait=True)


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
async def ask(request: Request, body: AskRequest, response: Response) -> AskResponse:
    """Answer one question from the corpus.

    `request` is unused by the body of this function but required: slowapi
    resolves the client address from it, and the secret check reads its headers.
    """
    require_caller_secret(request)

    if state.warmup is not None and not state.warmup.done():
        raise HTTPException(
            status_code=503, detail="The answering service is warming up."
        )

    expected = state.settings.shared_secret.get_secret_value()
    include_event = (
        bool(expected)
        and request.headers.get("X-Assistant-Event") == "1"
        and secrets.compare_digest(
            request.headers.get(SECRET_HEADER, "").encode("utf-8"),
            expected.encode("utf-8"),
        )
    )
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    history = tuple(
        Turn(turn.question.strip(), tuple(turn.sources)) for turn in body.history
    )
    started = time.monotonic()
    deadline = getattr(
        request.state,
        "answer_deadline",
        started + state.settings.backend_timeout_seconds,
    )
    context = ExecutionContext(deadline=deadline, budget=state.budget)

    def execute(
        job: ExecutionContext,
    ) -> tuple[AnsweredResponse | UnsupportedResponse, int, str | None]:
        try:
            answer = state.answerer.answer(question, history, context=job)
            job.raise_if_stopped()
            validation_started = time.monotonic()
            try:
                result = safe_response(answer)
                event = (
                    answer_event_header(
                        answer,
                        result,
                        job.status(),
                        corpus_sha256=state.corpus_checksum,
                        prompt_sha256=state.prompt_checksum,
                        latency_ms=int((time.monotonic() - started) * 1000),
                    )
                    if include_event
                    else None
                )
                return result, answer.rejected_citations, event
            finally:
                job.record_stage(
                    "validation",
                    min(60_000.0, (time.monotonic() - validation_started) * 1000),
                )
        finally:
            status = job.status()
            for stage, duration in status.stages_ms.items():
                state.metrics.record_stage(stage, duration)
            for attempt in status.attempts:
                state.metrics.record_attempt(
                    attempt.outcome,
                    route=attempt.route,
                    model=attempt.model,
                    input_tokens=attempt.input_tokens,
                    output_tokens=attempt.output_tokens,
                )

    try:
        future = state.executor.submit(execute, context)
    except (
        ExecutorCapacityError,
        ExecutorClosedError,
        BudgetCapacityError,
        BudgetUnavailable,
    ) as exc:
        state.metrics.record_admission(
            "rejected_full"
            if isinstance(exc, (ExecutorCapacityError, BudgetCapacityError))
            else "rejected_closed"
        )
        state.metrics.record(
            "unavailable", min(60_000.0, (time.monotonic() - started) * 1000)
        )
        logger.info("assistant_capacity_unavailable")
        raise HTTPException(
            status_code=503, detail="The answering service is unavailable."
        ) from None
    state.metrics.record_admission("admitted")
    state.metrics.record_stage(
        "queue", min(60_000.0, (time.monotonic() - started) * 1000)
    )
    wrapped = asyncio.wrap_future(future)
    # Observe late exceptions even after the HTTP waiter times out/disconnects.
    wrapped.add_done_callback(
        lambda done: None if done.cancelled() else done.exception()
    )

    async def disconnected() -> None:
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return

    disconnect_task = asyncio.create_task(disconnected())
    waiters: set[asyncio.Future[Any]] = {wrapped, disconnect_task}
    try:
        ready, _ = await asyncio.wait(
            waiters,
            timeout=max(0, deadline - time.monotonic()),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if disconnect_task in ready or wrapped not in ready:
            context.cancel()
            logger.info("assistant_deadline_or_disconnect")
            raise HTTPException(
                status_code=504, detail="The answering service is unavailable."
            )
        result, rejected_citations, event = wrapped.result()
        context.raise_if_stopped()
    except asyncio.CancelledError:
        context.cancel()
        state.metrics.record(
            "unavailable", min(60_000.0, (time.monotonic() - started) * 1000)
        )
        raise
    except (BudgetExhausted, BudgetUnavailable):
        state.metrics.record(
            "unavailable", min(60_000.0, (time.monotonic() - started) * 1000)
        )
        logger.info("assistant_budget_exhausted")
        raise HTTPException(
            status_code=503, detail="The answering service is unavailable."
        ) from None
    except HTTPException:
        state.metrics.record(
            "unavailable", min(60_000.0, (time.monotonic() - started) * 1000)
        )
        raise
    except Exception:
        context.cancel()
        state.metrics.record(
            "unavailable", min(60_000.0, (time.monotonic() - started) * 1000)
        )
        logger.warning("assistant_answer_failed")
        raise HTTPException(
            status_code=502, detail="The answering service is unavailable."
        ) from None
    finally:
        disconnect_task.cancel()
        await asyncio.gather(disconnect_task, return_exceptions=True)
    state.metrics.record(
        "answered" if result.state == "answered" else "not_covered",
        min(60_000.0, (time.monotonic() - started) * 1000),
        rejected_citations=rejected_citations,
    )
    if event is not None:
        response.headers["X-Assistant-Event"] = event
    return result


def _csp(nonce: str | None) -> str:
    """Compatibility name retained for the existing API and its tests."""
    return content_security_policy(nonce)


@app.middleware("http")
async def security_headers(request: Request, call_next: Any) -> Any:
    """Applied to every response, including errors and rate-limit rejections.

    Set here rather than per route because the responses most likely to be
    forgotten — a 429, a 422, a 500 — are the ones written by the framework
    rather than by us.
    """
    response = await call_next(request)
    apply_security_headers(response, no_store=request.url.path in {"/ask", "/metrics"})
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

    Neither the caller nor application logs receive the exception payload.
    """
    logger.error("assistant_unhandled_error")
    return JSONResponse(status_code=500, content={"detail": "Internal error."})


@app.exception_handler(RequestValidationError)
async def invalid_request(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # FastAPI's default error includes input values; do not echo those values.
    return JSONResponse(status_code=422, content={"detail": "Invalid request."})


app.add_middleware(
    AskBoundaryMiddleware,
    settings=lambda: state.settings,
    authenticate=require_caller_secret,
)
