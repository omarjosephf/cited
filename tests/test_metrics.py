"""Tests for aggregate metrics.

Two things are being tested and only one of them is arithmetic. The percentiles
and counts need to be right, but the requirement that actually matters is
negative: **no visitor content may ever appear in a snapshot.** A metrics module
is the natural place for that rule to erode — one field at a time, each
individually reasonable — so it is asserted directly rather than left to review.
"""

from __future__ import annotations

import json

from assistant.metrics import MAX_LATENCY_SAMPLES, AssistantMetrics


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
        metrics = AssistantMetrics()
        metrics.record("answered", 120.0)
        metrics.record("not_covered", 90.0)

        snapshot = metrics.snapshot()
        assert isinstance(snapshot.pop("note"), str)
        serialised = json.dumps(snapshot).lower()

        for forbidden in (
            "question",
            "query",
            "text",
            "prompt",
            "transcript",
            "ip",
            "user",
            "agent",
            "session",
        ):
            assert forbidden not in serialised, f"{forbidden!r} leaked into metrics"

    def test_record_accepts_no_question_argument(self) -> None:
        """Structural, not a matter of discipline: there is no parameter through
        which question text could be passed, so it cannot be added by accident."""
        import inspect

        parameters = set(inspect.signature(AssistantMetrics.record).parameters)

        assert parameters == {"self", "outcome", "latency_ms", "rejected_citations"}

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
