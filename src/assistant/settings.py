"""Configuration, loaded from the environment.

The API key is a `SecretStr` so that it cannot be printed by accident. Pydantic
renders it as `**********` in reprs, logs and tracebacks — which matters most
precisely when something has gone wrong and objects are being dumped.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["none", "low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    """Runtime configuration. Values come from the environment or a `.env` file."""

    PROXY_TIMEOUT_SECONDS: ClassVar[float] = 9.0
    UI_TIMEOUT_SECONDS: ClassVar[float] = 10.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `answer_model` would otherwise collide with pydantic's own `model_`
        # namespace and emit a warning on every import.
        protected_namespaces=(),
    )

    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API key. Empty disables answering; retrieval still works.",
    )
    openai_project_id: str = ""
    """Explicit OpenAI project boundary; required with the key."""

    openai_account_verified: bool = False
    """Operator configuration assertion, not machine verification."""

    gemini_api_key: SecretStr = Field(default=SecretStr(""))
    gemini_project_id: str = ""
    gemini_account_verified: bool = False
    enable_fallback: bool = False
    """Fallback is opt-in and requires its complete account configuration."""

    corpus_dir: Path = Path("content")
    """Where the served documents live.

    Configurable rather than hardcoded so one image can serve a different corpus
    per deployment. That is the difference between this being *one* assistant and
    being an assistant anyone can point at their own documents — and it costs a
    setting.
    """

    corpus_checksum: str = ""
    """Expected fingerprint of `corpus_dir`, or empty to skip verification.

    Set it for any deployment whose corpus is authored somewhere else and copied
    in: a stale or partial copy is otherwise indistinguishable from a correct one
    until someone reads an answer carefully. Empty is right for local work and
    for a deployment that owns its own corpus in the same repository.

    See `corpus_checksum.py` — the algorithm is a cross-language contract.
    """

    corpus_checksum_file: Path | None = None
    """A file containing the expected checksum, used when `corpus_checksum` is unset.

    The export artifact ships its own digest, so a deployment needs no per-release
    configuration edit. **Be clear about what that does and does not prove.** A
    digest travelling with the corpus it describes catches a partial copy, a
    corrupted transfer, and a corpus updated without its checksum — the realistic
    failures. It cannot catch a wholesale substitution of both, because it is not
    an independent witness.

    For that, set `corpus_checksum` explicitly to the value recorded in the
    release notes, and compare it against what `/health` reports. Both are
    supported; the explicit value wins.
    """

    corpus_vectors_file: Path | None = None
    """Vectors for `corpus_dir`, built ahead of time, or `None` to embed at startup.

    Embedding the corpus is the overwhelming majority of a cold start — 7.4s of
    8.4s on the development machine, three to four minutes on a throttled
    deployment CPU. Building the matrix at image-build time and pointing at it
    here removes that work from the startup path.

    **Set, but wrong, is a failure and not a fallback.** A file that is missing,
    unreadable, built by a different model, or built from different text stops
    the service starting. Quietly embedding the corpus instead would turn a
    stale artifact into a slow start nobody investigates, when the thing it is
    warning about — rows that no longer correspond to chunks — misattributes
    every citation it touches.
    """

    system_prompt_file: Path | None = None
    """A file containing the system prompt, or `None` for the built-in default.

    The prompt is roughly what a reader experiences as the assistant's character:
    its role, its tone, what it refuses, and what it does when it cannot help. A
    generic document-assistant voice is the right default for a generic tool and
    the wrong voice for anyone's actual assistant, so it becomes configuration.

    **Per-deployment only.** There is no interface for editing it and no request
    field that reaches it. A system prompt a caller can influence is not a system
    prompt.
    """

    shared_secret: SecretStr = Field(
        default=SecretStr(""),
        description=(
            "Shared secret required in the X-Assistant-Secret header when "
            "require_shared_secret is set. Empty disables the check."
        ),
    )

    require_shared_secret: bool = False
    """Whether `/ask` and `/metrics` demand the shared secret.

    Off by default because the public demo is deliberately open. On for a
    deployment funded by someone's own API budget: without it, anyone who finds
    the hostname can spend that budget, and rate limiting only decides how long
    it takes them.

    Enabling this with no secret configured is a misconfiguration that fails
    closed at startup rather than silently serving unauthenticated.
    """

    answer_model: Literal["gemini-3.5-flash-lite"] = "gemini-3.5-flash-lite"
    """The model that reads retrieved passages and decides whether they answer.

    Fixed to the owner-approved primary model. Model selection is not inferred
    from whichever provider credential happens to be present.
    """

    fallback_answer_model: Literal["gpt-5.6-luna"] = "gpt-5.6-luna"

    answer_max_tokens: int = Field(default=1024, ge=1024, le=1024)
    """Fixed to the 1024-token cap implemented by both approved adapters."""

    answer_effort: Literal["none"] = "none"
    """The implemented low-latency Luna configuration; currently fixed to none."""

    retrieval_top_k: int = 4
    """How many chunks are put in front of the model."""

    daily_answer_limit: int = Field(default=40, ge=1, le=40)
    monthly_answer_limit: int = Field(default=200, ge=1, le=200)
    daily_budget_micro_usd: int = Field(default=400_000, ge=1, le=400_000)
    monthly_budget_micro_usd: int = Field(default=2_000_000, ge=1, le=2_000_000)
    """Combined API reservation limits; integer micro-USD, never invoice totals."""

    budget_path: Path | None = None
    budget_ledger_id: str = ""
    budget_machine_id: str = ""
    """Operator-pinned persistent ledger and sole permitted Fly Machine."""

    prefilter_score: float = 0.45
    """Below this, skip the paid call entirely.

    Explicitly **not** the refusal mechanism — ADR-0002 measured that no
    threshold separates answerable from unanswerable questions. This is set well
    below the lowest observed in-scope score (0.666) so it only catches the
    obviously unrelated, where paying for a call is pointless.
    """

    answer_workers: int = Field(default=1, ge=1, le=5)
    """Maximum admitted synchronous answer jobs. One is the safe default."""

    @field_validator(
        "answer_workers",
        "daily_answer_limit",
        "monthly_answer_limit",
        "daily_budget_micro_usd",
        "monthly_budget_micro_usd",
        mode="before",
    )
    @classmethod
    def reject_boolean_worker_count(cls, value: object) -> object:
        """Accept environment strings while refusing bool-as-int coercion."""
        if isinstance(value, bool):
            raise ValueError("worker and budget limits must be integers, not booleans")
        return value

    backend_timeout_seconds: float = Field(
        default=8.0,
        gt=0.0,
        lt=PROXY_TIMEOUT_SECONDS,
        allow_inf_nan=False,
    )
    """Monotonic backend budget, strictly inside the trusted proxy budget."""

    provider_timeout_seconds: float = Field(
        default=6.0,
        gt=0.0,
        lt=PROXY_TIMEOUT_SECONDS,
        allow_inf_nan=False,
    )
    """Shared provider-phase duration across at most two serial attempts."""

    primary_timeout_seconds: float = Field(
        default=3.0, gt=0.0, lt=PROXY_TIMEOUT_SECONDS, allow_inf_nan=False
    )
    """Primary slice of the shared provider budget."""

    validation_margin_seconds: float = Field(
        default=0.5,
        ge=0.0,
        lt=PROXY_TIMEOUT_SECONDS,
        allow_inf_nan=False,
    )
    """Budget preserved after the provider attempt for result validation."""

    @model_validator(mode="after")
    def validate_timeout_ordering(self) -> Self:
        """Keep provider, backend, proxy and UI expiry strictly ordered."""
        if (
            self.provider_timeout_seconds + self.validation_margin_seconds
            >= self.backend_timeout_seconds
        ):
            raise ValueError(
                "provider_timeout_seconds + validation_margin_seconds must be "
                "less than backend_timeout_seconds"
            )
        if not self.backend_timeout_seconds < self.PROXY_TIMEOUT_SECONDS:
            raise ValueError(
                "backend_timeout_seconds must be less than the 9 second proxy budget"
            )
        if not self.PROXY_TIMEOUT_SECONDS < self.UI_TIMEOUT_SECONDS:
            raise ValueError("proxy timeout must be less than the UI timeout")
        if self.primary_timeout_seconds > self.provider_timeout_seconds:
            raise ValueError(
                "primary_timeout_seconds must not exceed provider_timeout_seconds"
            )
        if bool(self.openai_api_key.get_secret_value()) != bool(
            self.openai_project_id.strip()
        ):
            raise ValueError(
                "OPENAI_API_KEY and OPENAI_PROJECT_ID must be set together"
            )
        if bool(self.gemini_api_key.get_secret_value()) != bool(
            self.gemini_project_id.strip()
        ):
            raise ValueError(
                "GEMINI_API_KEY and GEMINI_PROJECT_ID must be set together"
            )
        if self.enable_fallback and not all(
            (
                self.openai_api_key.get_secret_value(),
                self.openai_project_id.strip(),
                self.openai_account_verified,
            )
        ):
            raise ValueError(
                "Luna fallback requires its key, project ID and verified account"
            )
        if bool(self.budget_path) != bool(self.budget_ledger_id):
            raise ValueError(
                "budget path and ledger identity must be configured together"
            )
        if self.budget_ledger_id and not re.fullmatch(
            r"[0-9a-f]{64}", self.budget_ledger_id
        ):
            raise ValueError("invalid budget ledger identity")
        if self.budget_path and self.answer_workers != 1:
            raise ValueError("persistent budget requires one shared answer worker")
        if self.daily_answer_limit > self.monthly_answer_limit:
            raise ValueError("daily attempt limit exceeds monthly limit")
        if self.daily_budget_micro_usd > self.monthly_budget_micro_usd:
            raise ValueError("daily money limit exceeds monthly limit")
        return self

    @property
    def answering_enabled(self) -> bool:
        """Whether the complete owner-selected two-provider arrangement is set."""
        return (
            bool(self.gemini_api_key.get_secret_value())
            and bool(self.gemini_project_id.strip())
            and self.gemini_account_verified
            and self.fallback_enabled
        )

    @property
    def fallback_enabled(self) -> bool:
        """Whether operator-supplied Luna backup configuration is complete."""
        return (
            self.enable_fallback
            and bool(self.openai_api_key.get_secret_value())
            and bool(self.openai_project_id.strip())
            and self.openai_account_verified
        )

    def expected_corpus_checksum(self) -> str:
        """The digest to verify against: explicit value first, then the file.

        A configured file that does not exist raises rather than degrading to
        "no verification". Someone asked for this corpus to be checked; silently
        not checking it is the one response that must not be possible.
        """
        if self.corpus_checksum.strip():
            return self.corpus_checksum.strip()

        path = self.corpus_checksum_file
        if path is None:
            return ""

        if not path.is_file():
            raise RuntimeError(
                f"corpus_checksum_file is set to {path}, which does not exist. "
                "The corpus artifact is incomplete, or the path is wrong."
            )
        return path.read_text(encoding="utf-8").strip()
