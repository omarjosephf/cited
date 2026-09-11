"""Tests for aggregate metrics.

Two things are being tested and only one of them is arithmetic. The percentiles
and counts need to be right, but the requirement that actually matters is
negative: **no visitor content may ever appear in a snapshot.** A metrics module
is the natural place for that rule to erode — one field at a time, each
individually reasonable — so it is asserted directly rather than left to review.
"""

from __future__ import annotations

import json

import pytest

from assistant.metrics import (
    MAX_DURATION_MS,
    MAX_LATENCY_SAMPLES,
    AssistantMetrics,
    MetricsIdentity,
)


class TestCounting:
    def test_a_fresh_snapshot_reports_nothing_rather_than_zeroes_that_mislead(
        self,
    ) -> None:
        snapshot = AssistantMetrics().snapshot()

        assert snapshot["requests"] == 0
        # None, not 0.0: no requests means the refusal rate is unknown, and a
        # dashboard showing a confident 0% would be reporting a fact nobody has.
        assert snapshot["refusal_rate"] is None
        assert snapshot["latency_ms"]["p50"] is None

    def test_outcomes_are_counted_by_what_the_visitor_saw(self) -> None:
        metrics = AssistantMetrics()
        metrics.record("answered", 100.0)
        metrics.record("answered", 200.0)
        metrics.record("not_covered", 150.0)
        metrics.record("unavailable", 10.0)

        snapshot = metrics.snapshot()

        assert snapshot["outcomes"] == {
            "answered": 2,
            "not_covered": 1,
            "unavailable": 1,
        }
        assert snapshot["requests"] == 4

    def test_unavailable_requests_are_excluded_from_the_refusal_rate(self) -> None:
        """An outage is not a content gap.

        Including unavailable requests would make a provider failure look like a
        corpus that had stopped covering its subject — and the operator would go
        looking for the wrong problem.
        """
        metrics = AssistantMetrics()
        metrics.record("answered", 100.0)
        metrics.record("not_covered", 100.0)
        for _ in range(10):
            metrics.record("unavailable", 5.0)

        assert metrics.snapshot()["refusal_rate"] == 0.5

    def test_rejected_citations_accumulate(self) -> None:
        """Expected to stay at zero. Counted because a number that stops being
        zero is the signal that the citation guarantee has broken."""
        metrics = AssistantMetrics()
        metrics.record("answered", 100.0, rejected_citations=0)
        metrics.record("answered", 100.0, rejected_citations=2)

        assert metrics.snapshot()["rejected_citations"] == 2


class TestLatency:
    def test_percentiles_are_observed_values_rather_than_interpolations(self) -> None:
        metrics = AssistantMetrics()
        for value in (100.0, 200.0, 300.0, 400.0, 500.0):
            metrics.record("answered", value)

        latency = metrics.snapshot()["latency_ms"]

        assert latency["p50"] == 300.0
        assert latency["p95"] == 500.0
        assert latency["samples"] == 5

    def test_a_single_sample_is_both_percentiles(self) -> None:
        metrics = AssistantMetrics()
        metrics.record("answered", 42.5)

        latency = metrics.snapshot()["latency_ms"]

        assert latency["p50"] == 42.5
        assert latency["p95"] == 42.5

    def test_percentiles_do_not_depend_on_arrival_order(self) -> None:
        ascending = AssistantMetrics()
        descending = AssistantMetrics()
        for value in (10.0, 20.0, 30.0, 40.0):
            ascending.record("answered", value)
        for value in (40.0, 30.0, 20.0, 10.0):
            descending.record("answered", value)

        assert ascending.snapshot()["latency_ms"] == descending.snapshot()["latency_ms"]

    def test_retained_samples_are_bounded(self) -> None:
        """Memory must not grow with traffic. A long-lived process serving a
        popular assistant would otherwise accumulate a sample per request."""
        metrics = AssistantMetrics()
        for value in range(MAX_LATENCY_SAMPLES + 500):
            metrics.record("answered", float(value))

        assert metrics.snapshot()["latency_ms"]["samples"] == MAX_LATENCY_SAMPLES

    def test_the_window_keeps_recent_samples_and_drops_old_ones(self) -> None:
        metrics = AssistantMetrics()
        for _ in range(MAX_LATENCY_SAMPLES):
            metrics.record("answered", 1000.0)
        for _ in range(MAX_LATENCY_SAMPLES):
            metrics.record("answered", 5.0)

        # Every original sample has been evicted, so the window is all recent.
        assert metrics.snapshot()["latency_ms"]["p95"] == 5.0


class TestPrivacy:
    def test_a_snapshot_contains_no_visitor_content(self) -> None:
        """The rule the whole module exists to keep.

        Serialised and searched rather than key-by-key, so a field added later
        that smuggles content through a nested structure still fails this.

        `note` is excluded from the scan and pinned separately below: it is a
        fixed sentence that necessarily contains the phrase "question text",
        because its job is to promise there is none. Scanning it would make this
        test fail on the very disclosure it is checking for.
        """
        metrics = AssistantMetrics(
            identity=MetricsIdentity(
                model="model-version",
                corpus="corpus-version",
                prompt="prompt-version",
            )
        )
        metrics.record("answered", 120.0)
        metrics.record("not_covered", 90.0)

        snapshot = metrics.snapshot()
        assert isinstance(snapshot.pop("note"), str)
        serialised = json.dumps(snapshot).lower()

        for forbidden in ("private question sentinel", "visitor@example.test"):
            assert forbidden not in serialised

    def test_record_accepts_no_question_argument(self) -> None:
        """Structural, not a matter of discipline: there is no parameter through
        which question text could be passed, so it cannot be added by accident."""
        import inspect

        parameters = set(inspect.signature(AssistantMetrics.record).parameters)

        assert parameters == {"self", "outcome", "latency_ms", "rejected_citations"}


class TestOperationalMetrics:
    def test_admission_and_stage_categories_are_fixed_and_aggregated(self) -> None:
        metrics = AssistantMetrics()
        metrics.record_admission("admitted")
        metrics.record_admission("rejected_full")
        metrics.record_stage("retrieval", 10.0)
        metrics.record_stage("retrieval", 20.0)
        metrics.record_stage("provider", 90_000.0)

        snapshot = metrics.snapshot()
        assert snapshot["admission"] == {
            "admitted": 1,
            "rejected_full": 1,
            "rejected_closed": 0,
        }
        assert snapshot["stages_ms"]["retrieval"] == {
            "p50": 10.0,
            "p95": 20.0,
            "samples": 2,
        }
        assert snapshot["stages_ms"]["provider"]["p50"] == MAX_DURATION_MS

    def test_attempts_distinguish_uncertainty_and_unknown_usage(self) -> None:
        metrics = AssistantMetrics()
        metrics.record_attempt("completed", input_tokens=100, output_tokens=25)
        metrics.record_attempt("completed")
        metrics.record_attempt("uncertain")

        attempts = metrics.snapshot()["provider_attempts"]
        assert attempts["total"] == 3
        assert attempts["outcomes"] == {"completed": 2, "uncertain": 1}
        assert attempts["uncertain"] == 1
        assert attempts["usage"] == {
            "input_tokens": 100,
            "output_tokens": 25,
            "known_attempts": 1,
            "unknown_attempts": 2,
        }

    def test_no_known_usage_is_reported_as_unknown_not_zero(self) -> None:
        metrics = AssistantMetrics()
        metrics.record_attempt("uncertain")

        usage = metrics.snapshot()["provider_attempts"]["usage"]
        assert usage["input_tokens"] is None
        assert usage["output_tokens"] is None
        assert usage["unknown_attempts"] == 1

    def test_identity_is_constructor_only_trusted_version_data(self) -> None:
        metrics = AssistantMetrics(
            identity=MetricsIdentity(
                model="claude-haiku-4-5",
                corpus="sha256:abc",
                prompt="prompt-v2",
            )
        )

        assert metrics.snapshot()["identity"] == {
            "model": "claude-haiku-4-5",
            "corpus": "sha256:abc",
            "prompt": "prompt-v2",
        }

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("record", ("answered", float("nan"))),
            ("record", ("answered", -1.0)),
            ("record_stage", ("provider", float("inf"))),
            ("record_stage", ("unknown", 1.0)),
            ("record_admission", ("unknown",)),
            ("record_attempt", ("unknown",)),
        ],
    )
    def test_unbounded_or_unknown_categories_are_rejected(
        self, method: str, args: tuple[object, ...]
    ) -> None:
        with pytest.raises((TypeError, ValueError)):
            getattr(AssistantMetrics(), method)(*args)

    def test_the_snapshot_says_it_is_not_a_lifetime_total(self) -> None:
        """Under scale-to-zero these reset routinely. A reader who assumes
        otherwise will under-report traffic and misread a quiet day."""
        snapshot = AssistantMetrics().snapshot()

        assert "since" in snapshot
        note = snapshot["note"].lower()
        assert "reset" in note
        # The one place the phrase is allowed, and it must be a promise rather
        # than a field: this is what the scan above excludes, so pin its content.
        assert "no question text is recorded" in note

    def test_the_remaining_allowance_is_reported_when_supplied(self) -> None:
        snapshot = AssistantMetrics().snapshot(answers_remaining_today=37)

        assert snapshot["answers_remaining_today"] == 37
