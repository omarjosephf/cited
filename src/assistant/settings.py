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

    answer_model: str = "claude-opus-5"
    """The model that reads retrieved passages and decides whether they answer.

    Configurable because it is the only paid component, and the right choice
    depends on whether this is running as a personal demo or in front of real
    traffic. Cost per million tokens at time of writing: Opus 5 $5/$25,
    Sonnet 5 $3/$15, Haiku 4.5 $1/$5.
    """

    answer_max_tokens: int = 1024
    """Enough for a grounded answer with citations; not enough to ramble."""

    answer_effort: Effort = "low"
    """Reading passages and judging whether they answer a question is not a
    reasoning-heavy task, and low effort keeps latency and cost down. Thinking
    is left on: disabling it entirely on current models can leak internal tags
    into the visible response, which would be user-facing damage for no saving.
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
