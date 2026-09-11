"""Bounded v3 application results; provider prose cannot set its serving route."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from assistant.answering import Answer, Answerer, evidence_id, valid_source_id
from assistant.policy import POLICY_RESPONSES

WIRE_VERSION = 3
MAX_RESPONSE_BYTES = 48 * 1024
PolicyId = Literal[
    "unsupported",
    "identity",
    "architecture",
    "bulk_extraction",
    "unpublished_work",
    "provider_self_identification",
    "bulk_reproduction",
    "privacy",
]


class StrictResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[3] = 3


class CitationOut(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_id: str = Field(min_length=1, max_length=200)
    evidence_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    quote: str = Field(min_length=1, max_length=1000)


class AnsweredResponse(StrictResult):
    state: Literal["answered"] = "answered"
    answer: str = Field(min_length=1, max_length=4000)
    citations: list[CitationOut] = Field(min_length=1, max_length=8)
    model_route: Literal["primary", "fallback"]


class UnsupportedResponse(StrictResult):
    state: Literal["not-covered"] = "not-covered"
    policy: PolicyId = "unsupported"
    model_route: Literal["primary", "fallback"] | None


class UnavailableResponse(StrictResult):
    state: Literal["unavailable"] = "unavailable"


AskResponse = Annotated[
    AnsweredResponse | UnsupportedResponse | UnavailableResponse,
    Field(discriminator="state"),
]


def safe_response(answer: Answer) -> AnsweredResponse | UnsupportedResponse:
    """Recheck domain output at transport without calling it semantic proof.

    Every citation must identify an actual supplied chunk and its verified quote.
    This still cannot determine whether a paraphrase is entailed by that quote.
    The independent release review remains required for captured evaluation runs.
    """
    if answer.suppression_reason is not None:
        return UnsupportedResponse(model_route=answer.model_route)
    if answer.policy is not None:
        if (
            answer.policy in POLICY_RESPONSES
            and answer.text == POLICY_RESPONSES[answer.policy]
            and not answer.grounded
            and not answer.citations
        ):
            try:
                return UnsupportedResponse.model_validate(
                    {
                        "version": 3,
                        "state": "not-covered",
                        "policy": answer.policy,
                        "model_route": answer.model_route,
                    }
                )
            except ValidationError:
                pass
        return UnsupportedResponse(model_route=answer.model_route)
    if (
        not answer.grounded
        or answer.refused
        or answer.rejected_citations
        or answer.stop_reason != "end_turn"
        or not answer.text.strip()
        or not 1 <= len(answer.citations) <= 8
    ):
        return UnsupportedResponse(model_route=answer.model_route)
    mapped: list[CitationOut] = []
    for citation in answer.citations:
        candidates = [
            result
            for result in answer.results
            if result.chunk.index == citation.chunk_index
            and result.chunk.source == citation.source_id
            and result.cite() == citation.source
        ]
        if (
            len(candidates) != 1
            or not valid_source_id(citation.source_id)
            or not 1 <= len(citation.quoted_text) <= 1000
        ):
            return UnsupportedResponse(model_route=answer.model_route)
        result = candidates[0]
        if (
            not Answerer._quote_is_present(citation.quoted_text, result.chunk.text)
            or evidence_id(result, citation.quoted_text) != citation.evidence_id
        ):
            return UnsupportedResponse(model_route=answer.model_route)
        mapped.append(
            CitationOut(
                source_id=citation.source_id,
                evidence_id=citation.evidence_id,
                quote=citation.quoted_text,
            )
        )
    try:
        if answer.model_route not in {"primary", "fallback"}:
            return UnsupportedResponse(model_route=None)
        response = AnsweredResponse(
            answer=answer.text, citations=mapped, model_route=answer.model_route
        )
    except ValidationError:
        return UnsupportedResponse(model_route=answer.model_route)
    if len(response.model_dump_json().encode("utf-8")) > MAX_RESPONSE_BYTES:
        return UnsupportedResponse(model_route=answer.model_route)
    return response
