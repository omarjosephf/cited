"""Local, read-only browser interface for inspecting one or more RAG corpora."""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from assistant.inspection import CorpusInspection
from assistant.inspector_report import render_report
from assistant.web_security import (
    NONCE_PLACEHOLDER,
    apply_security_headers,
    content_security_policy,
)

LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "testserver"})
MAX_SEARCH_CHARS = 200
MAX_SOURCE_CHARS = 500
MAX_PAGE_SIZE = 100


def create_inspector_app(corpora: list[CorpusInspection]) -> FastAPI:
    """Create an app over immutable snapshots selected before the server starts."""
    if not corpora:
        raise ValueError("At least one corpus inspection is required.")
    registry = {corpus.id: corpus for corpus in corpora}
    if len(registry) != len(corpora):
        raise ValueError("Corpus labels must produce unique identifiers.")

    app = FastAPI(
        title="Cited RAG Management Panel",
        description="Local, read-only visibility into configured RAG corpora.",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def local_and_secure(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Reject unexpected hosts and protect every response, including errors."""
        host = request.url.hostname or ""
        if host.casefold() not in LOCAL_HOSTS:
            response: Response = JSONResponse(
                status_code=400, content={"detail": "Local access only."}
            )
        else:
            response = await call_next(request)
        apply_security_headers(response, no_store=True, no_index=True)
        return response

    def selected(corpus_id: str) -> CorpusInspection:
        corpus = registry.get(corpus_id)
        if corpus is None:
            raise HTTPException(status_code=404, detail="Corpus not found.")
        return corpus

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        page = Path(__file__).parent / "static" / "inspector.html"
        nonce = secrets.token_urlsafe(16)
        html = page.read_text(encoding="utf-8").replace(NONCE_PLACEHOLDER, nonce)
        return HTMLResponse(
            html,
            headers={"Content-Security-Policy": content_security_policy(nonce)},
        )

    @app.get("/api/corpora")
    async def list_corpora() -> dict[str, Any]:
        return {"corpora": [corpus.summary() for corpus in corpora]}

    @app.get("/api/corpora/{corpus_id}")
    async def corpus_detail(corpus_id: str) -> dict[str, Any]:
        return selected(corpus_id).detail()

    @app.get("/api/corpora/{corpus_id}/report", response_class=HTMLResponse)
    async def corpus_report(corpus_id: str) -> HTMLResponse:
        corpus = selected(corpus_id)
        nonce = secrets.token_urlsafe(16)
        return HTMLResponse(
            render_report(corpus, nonce),
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{corpus.id}-rag-management-report.html"'
                ),
                "Content-Security-Policy": content_security_policy(nonce),
            },
        )

    @app.get("/api/corpora/{corpus_id}/chunks")
    async def list_chunks(
        corpus_id: str,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=25, ge=1, le=MAX_PAGE_SIZE),
        source: str = Query(default="", max_length=MAX_SOURCE_CHARS),
        q: str = Query(default="", max_length=MAX_SEARCH_CHARS),
    ) -> dict[str, Any]:
        corpus = selected(corpus_id)
        wanted_source = source.strip()
        needle = q.strip().casefold()
        matching = [
            chunk
            for chunk in corpus.chunks
            if (not wanted_source or chunk.source == wanted_source)
            and (
                not needle
                or needle in chunk.text.casefold()
                or needle in chunk.citation.casefold()
            )
        ]
        start = (page - 1) * page_size
        stop = start + page_size
        return {
            "items": [chunk.as_dict() for chunk in matching[start:stop]],
            "page": page,
            "page_size": page_size,
            "total": len(matching),
        }

    @app.get("/api/corpora/{corpus_id}/chunks/{index}")
    async def chunk_detail(corpus_id: str, index: int) -> dict[str, Any]:
        chunk = selected(corpus_id).chunk(index)
        if chunk is None:
            raise HTTPException(status_code=404, detail="Chunk not found.")
        return chunk.as_dict(include_text=True)

    @app.exception_handler(500)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        del request, exc
        return JSONResponse(status_code=500, content={"detail": "Internal error."})

    return app
