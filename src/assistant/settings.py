"""Configuration, loaded from the environment.

The API key is a `SecretStr` so that it cannot be printed by accident. Pydantic
renders it as `**********` in reprs, logs and tracebacks — which matters most
precisely when something has gone wrong and objects are being dumped.
"""

from __future__ import annotations

from pathlib import Path
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

    daily_answer_limit: int = 200
    """Hard ceiling on paid calls per UTC day for the HTTP service.

    At the default model this bounds the demo to roughly $0.44 a day. Rate
    limiting alone would not: 10 requests a minute still permits over fourteen
    thousand paid calls a day, which is a bill rather than a demo.

    This is a stop *before* the provider's own cap, so the service can explain
    itself rather than failing with a billing error. It is not a substitute for
    setting that cap — this counter resets when the process restarts, and the
    provider's does not.
    """

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
