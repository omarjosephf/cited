"""Bounded observations for the authenticated management server, without prose."""

from __future__ import annotations

import base64
import json
import re

from assistant.answering import Answer
from assistant.runtime import ExecutionStatus
from assistant.transport import AnsweredResponse, UnsupportedResponse


def answer_event_header(
    answer: Answer,
    result: AnsweredResponse | UnsupportedResponse,
    status: ExecutionStatus,
    *,
    corpus_sha256: str,
    prompt_sha256: str,
    latency_ms: int,
) -> str | None:
    """Never diagnose missing knowledge or a retrieval miss from a refusal."""
    if not all(
        re.fullmatch(r"[a-f0-9]{64}", s) for s in (corpus_sha256, prompt_sha256)
    ):
        return None
    route = result.model_route or "none"
    model = next((a.model for a in reversed(status.attempts) if a.route == route), None)
    if route != "none" and model is None:
        return None
    retrieved = list(dict.fromkeys(r.chunk.source for r in answer.results))
    cited = (
        list(dict.fromkeys(c.source_id for c in result.citations))
        if isinstance(result, AnsweredResponse)
        else []
    )
    if len(retrieved) > 20 or any(not s or len(s) > 200 for s in retrieved + cited):
        return None
    if model is not None and (not model or len(model) > 100):
        return None
    event = {
        "version": 1,
        "outcome": "answered"
        if result.state == "answered"
        else "policy_boundary"
        if result.policy != "unsupported"
        else "not_covered",
        "route": route,
        "model": model,
        "retrieved": retrieved,
        "cited": cited,
        "latencyMs": max(0, min(600000, latency_ms)),
        "corpusSha256": corpus_sha256,
        "promptSha256": prompt_sha256,
    }
    encoded = (
        base64.urlsafe_b64encode(json.dumps(event, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )
    return encoded if len(encoded) <= 8000 else None
