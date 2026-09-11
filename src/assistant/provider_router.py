"""Ordered primary/fallback routing with one shared provider-time budget."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from assistant.budget import AttemptBudget
from assistant.provider_adapters import (
    GEMINI_MODEL,
    OPENAI_MODEL,
    GeminiAdapter,
    GeneratedAnswer,
    OpenAIAdapter,
    ProviderUnavailable,
)
from assistant.runtime import ExecutionContext, ExecutionDeadlineExceeded, ProviderRoute

PRIMARY_MODEL = GEMINI_MODEL
FALLBACK_MODEL = OPENAI_MODEL


class ProviderRouter:
    """Try Gemini, then Luna only for a classified availability fault."""

    manages_attempts = True

    def __init__(
        self,
        primary: GeminiAdapter,
        fallback: OpenAIAdapter | None,
        *,
        provider_timeout_seconds: float,
        primary_timeout_seconds: float,
        validation_margin_seconds: float,
        budget: AttemptBudget | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._provider_timeout = provider_timeout_seconds
        self._primary_timeout = primary_timeout_seconds
        self._validation_margin = validation_margin_seconds
        self._budget = budget

    @property
    def calls(self) -> int:
        return self._budget.used if self._budget is not None else 0

    def create(self, **request: Any) -> GeneratedAnswer:
        context = request.pop("_execution_context", None)
        if context is not None and not isinstance(context, ExecutionContext):
            raise TypeError("invalid execution context")
        required = {"system", "question", "evidence", "history"}
        if set(request) != required:
            raise TypeError("invalid normalized provider request")

        started = time.monotonic()
        shared_deadline = started + self._provider_timeout
        if context is not None:
            shared_deadline = min(
                shared_deadline, context.deadline - self._validation_margin
            )
        primary_deadline = min(started + self._primary_timeout, shared_deadline)
        try:
            return self._attempt(
                self._primary,
                "primary",
                PRIMARY_MODEL,
                primary_deadline,
                context,
                request,
            )
        except ProviderUnavailable:
            if self._fallback is None:
                raise

        return self._attempt(
            self._fallback,
            "fallback",
            FALLBACK_MODEL,
            shared_deadline,
            context,
            request,
        )

    def _attempt(
        self,
        adapter: OpenAIAdapter | GeminiAdapter,
        route: ProviderRoute,
        model: str,
        deadline: float,
        context: ExecutionContext | None,
        request: dict[str, Any],
    ) -> GeneratedAnswer:
        prepared = adapter.prepare(**request)
        try:
            if context is not None:
                context.raise_if_stopped()
            if time.monotonic() >= deadline:
                raise ExecutionDeadlineExceeded("provider budget exhausted")
            if context is not None:
                context.begin_provider(route, model)
            elif self._budget is not None:
                self._budget.spend()
            else:
                raise RuntimeError("provider calls require an execution budget")

            try:
                result = adapter.generate(
                    prepared=prepared,
                    evidence=request["evidence"],
                    deadline=deadline,
                    before_dispatch=context.raise_if_stopped
                    if context is not None
                    else None,
                )
            except BaseException:
                if context is not None:
                    context.finish_provider("uncertain")
                raise
            if context is not None:
                context.finish_provider(
                    "completed",
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                )
            if time.monotonic() >= deadline:
                raise ExecutionDeadlineExceeded(
                    "provider result arrived after deadline"
                )
            if context is not None:
                context.raise_if_stopped()
            # Routing role belongs to this application, never to provider JSON
            # or an adapter's identity. Refusals receive the same route label.
            return replace(result, route=route)
        finally:
            prepared.close()
