"""Runtime deadline and worker settings stay strictly ordered and finite."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from assistant.settings import Settings


def settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type, call-arg]


def test_runtime_defaults_preserve_the_deadline_order() -> None:
    configured = settings()

    assert configured.answer_workers == 1
    assert configured.provider_timeout_seconds == 6.0
    assert configured.validation_margin_seconds == 0.5
    assert configured.backend_timeout_seconds == 8.0
    assert (
        configured.provider_timeout_seconds + configured.validation_margin_seconds
        < configured.backend_timeout_seconds
        < configured.PROXY_TIMEOUT_SECONDS
        < configured.UI_TIMEOUT_SECONDS
    )


@pytest.mark.parametrize("workers", [0, 6, True])
def test_answer_workers_stay_between_one_and_five(workers: object) -> None:
    with pytest.raises(ValidationError):
        settings(answer_workers=workers)


def test_answer_workers_can_be_loaded_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANSWER_WORKERS", "3")

    assert settings().answer_workers == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend_timeout_seconds", float("nan")),
        ("backend_timeout_seconds", float("inf")),
        ("provider_timeout_seconds", float("nan")),
        ("provider_timeout_seconds", float("inf")),
        ("validation_margin_seconds", float("nan")),
        ("validation_margin_seconds", float("inf")),
        ("validation_margin_seconds", -0.1),
    ],
)
def test_timeout_values_must_be_finite_and_nonnegative(
    field: str, value: float
) -> None:
    with pytest.raises(ValidationError):
        settings(**{field: value})


def test_provider_plus_margin_must_fit_strictly_inside_backend() -> None:
    with pytest.raises(ValidationError, match="validation_margin_seconds"):
        settings(
            provider_timeout_seconds=7.5,
            validation_margin_seconds=0.5,
            backend_timeout_seconds=8.0,
        )


def test_backend_must_fit_strictly_inside_fixed_proxy_budget() -> None:
    with pytest.raises(ValidationError):
        settings(backend_timeout_seconds=9.0)


def test_answering_requires_the_complete_explicit_provider_pair() -> None:
    configured = settings(
        openai_api_key=SecretStr("openai-test"),
        openai_project_id="openai-project",
        openai_account_verified=True,
        gemini_api_key=SecretStr("gemini-test"),
        gemini_project_id="gemini-project",
        gemini_account_verified=True,
        enable_fallback=True,
    )

    assert configured.answering_enabled
    assert configured.fallback_enabled


def test_primary_credentials_alone_do_not_enable_answering() -> None:
    configured = settings(
        openai_api_key=SecretStr("openai-test"),
        openai_project_id="openai-project",
        openai_account_verified=True,
    )

    assert not configured.answering_enabled
    assert not configured.fallback_enabled


def test_enabled_fallback_rejects_incomplete_configuration() -> None:
    with pytest.raises(ValidationError, match="Luna fallback requires"):
        settings(enable_fallback=True)
