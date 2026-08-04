"""Configuration, loaded from the environment.

The API key is a `SecretStr` so that it cannot be printed by accident. Pydantic
renders it as `**********` in reprs, logs and tracebacks — which matters most
precisely when something has gone wrong and objects are being dumped.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    """Runtime configuration. Values come from the environment or a `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `answer_model` would otherwise collide with pydantic's own `model_`
        # namespace and emit a warning on every import.
        protected_namespaces=(),
    )

    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""),
        description=(
            "Anthropic API key. Empty disables answering; retrieval still works."
        ),
    )

    answer_model: str = "claude-haiku-4-5"
    """The model that reads retrieved passages and decides whether they answer.

    Haiku by default. The task is reading four short passages and judging
    whether they contain an answer — comprehension, not reasoning — and the
    cheapest capable model is the right default for something served publicly.

    Cost per 1,000 questions at roughly 1,200 input / 200 output tokens:
    Opus 5 $11.00, Sonnet 5 $6.60, Haiku 4.5 $2.20. Whether Haiku actually
    costs accuracy is a question for the evaluation harness, not for intuition.
    """

    answer_max_tokens: int = 1024
    """Enough for a grounded answer with citations; not enough to ramble."""

    answer_effort: Effort | None = None
    """Thinking depth, or `None` to omit the parameter entirely.

    **Not every model accepts this.** `effort` is supported on the Opus and
    Sonnet tiers but is rejected by Haiku 4.5, so sending it unconditionally
    would fail every request under the default model. It is therefore omitted
    unless explicitly configured — and `Answerer` only includes `output_config`
    in the request when it is set.

    Set it when running on a model that supports it and the evaluation set shows
    a reason to.
    """

    retrieval_top_k: int = 4
    """How many chunks are put in front of the model."""

    prefilter_score: float = 0.45
    """Below this, skip the paid call entirely.

    Explicitly **not** the refusal mechanism — ADR-0002 measured that no
    threshold separates answerable from unanswerable questions. This is set well
    below the lowest observed in-scope score (0.666) so it only catches the
    obviously unrelated, where paying for a call is pointless.
    """

    @property
    def answering_enabled(self) -> bool:
        """Whether a key is configured. Retrieval works without one."""
        return bool(self.anthropic_api_key.get_secret_value())
