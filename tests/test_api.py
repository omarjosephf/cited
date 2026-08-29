"""Tests for the HTTP layer.

No network and no API key: the answerer is replaced by a stub, because what
needs testing here is transport and protection, not answering. Answering has its
own tests.

The protections are the reason this file exists. A public endpoint that makes
paid calls fails in ways that cost money rather than correctness, and those
paths are the ones least likely to be exercised by hand.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from assistant import api
from assistant.answering import Answer, Citation
from assistant.budget import DailyCallBudget
from assistant.chunking import Chunk
from assistant.metrics import AssistantMetrics
from assistant.retrieval import SearchResult
from assistant.settings import Settings


def chunk(index: int = 0, section: str = "Components") -> Chunk:
    return Chunk(
        text=f"Body of chunk {index}.",
        source="guide.md",
        page=None,
        section=section,
        index=index,
    )


def grounded_answer() -> Answer:
    return Answer(
        text="Four parts: role, task, context and format.",
        citations=(
            Citation(
                quoted_text="role, task, context and format",
                source="guide.md — Components",
                chunk_index=0,
            ),
        ),
        grounded=True,
        results=(SearchResult(chunk=chunk(), score=0.8),),
    )


class StubAnswerer:
    def __init__(self, answer: Answer | None = None, error: Exception | None = None):
        self.answer_value = answer or grounded_answer()
        self.error = error
        self.questions: list[str] = []

    def answer(self, question: str) -> Answer:
        self.questions.append(question)
        if self.error:
            raise self.error
        return self.answer_value


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A client with the corpus and answerer stubbed, and limits reset.

    Constructed **without** the context manager, deliberately. Entering
    `TestClient` as a context manager runs the app's lifespan, which reads the
    real corpus, loads the embedding model, and — the part that actually bit —
    overwrites every attribute set here with the production defaults. The stubs
    were being installed and then silently replaced, so the tests were
    exercising real configuration while appearing to control it.

    Skipping lifespan keeps these tests fast, independent of `content/`, and
    actually in charge of the state they assert on.
    """
    api.state.settings = Settings(anthropic_api_key=SecretStr("test-key"))
    api.state.answerer = StubAnswerer()  # type: ignore[assignment]
    api.state.budget = DailyCallBudget(limit=100)
    api.state.metrics = AssistantMetrics()
    api.state.chunk_count = 10
    # A fixed stand-in for the value `verify_corpus` would return. These tests
    # never read a corpus, so anything real here would be a fiction; what matters
    # is that the attribute the routes read is present and stable.
    api.state.corpus_checksum = "0" * 64
    # slowapi keeps counters between tests otherwise, so the first test to run
    # would consume the allowance for the rest.
    api.limiter.reset()
    yield TestClient(api.app)


class TestAsk:
    def test_a_grounded_answer_returns_its_citations(self, client: TestClient) -> None:
        response = client.post("/ask", json={"question": "What are the components?"})

        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert body["citations"][0]["source"] == "guide.md — Components"
        assert body["citations"][0]["quote"]

    def test_an_ungrounded_answer_is_flagged_not_hidden(
        self, client: TestClient
    ) -> None:
        """The client must be able to tell prose from a sourced answer.

        Returning it without the flag would let a caller present unsupported
        text as though it were cited, which is the failure the whole project
        exists to prevent.
        """
        api.state.answerer = StubAnswerer(  # type: ignore[assignment]
            Answer("Not covered.", (), grounded=False, results=(), refused=True)
        )
        body = client.post("/ask", json={"question": "anything"}).json()

        assert body["grounded"] is False
        assert body["refused"] is True
        assert body["answer"] == "Not covered."

    def test_the_question_is_passed_through_stripped(self, client: TestClient) -> None:
        stub = StubAnswerer()
        api.state.answerer = stub  # type: ignore[assignment]
        client.post("/ask", json={"question": "  spaced out  "})

        assert stub.questions == ["spaced out"]

    @pytest.mark.parametrize("question", ["", "   ", "\n\t"])
    def test_an_empty_question_is_rejected_without_paying(
        self, client: TestClient, question: str
    ) -> None:
        before = api.state.budget.used
        response = client.post("/ask", json={"question": question})

        assert response.status_code == 422
        assert api.state.budget.used == before, "an invalid request must cost nothing"

    def test_an_overlong_question_is_rejected(self, client: TestClient) -> None:
        """The caller controls the input, so the input needs a ceiling.

        Without one, a single request can be made arbitrarily expensive.
        """
        response = client.post(
            "/ask", json={"question": "x" * (api.MAX_QUESTION_CHARS + 1)}
        )
        assert response.status_code == 422


class TestBudget:
    def test_the_daily_ceiling_returns_503_rather_than_spending(
        self, client: TestClient
    ) -> None:
        api.state.budget = DailyCallBudget(limit=1)

        assert client.post("/ask", json={"question": "one"}).status_code == 200
        response = client.post("/ask", json={"question": "two"})

        assert response.status_code == 503
        assert "daily limit" in response.json()["detail"]

    def test_a_failed_call_refunds_its_reservation(self, client: TestClient) -> None:
        """A provider outage must not burn the day's allowance.

        Otherwise an hour of upstream failure leaves the demo unable to answer
        anything for the rest of the day, having answered nothing.
        """
        api.state.answerer = StubAnswerer(error=RuntimeError("upstream down"))  # type: ignore[assignment]
        before = api.state.budget.used

        response = client.post("/ask", json={"question": "anything"})

        assert response.status_code == 502
        assert api.state.budget.used == before

    def test_an_upstream_failure_does_not_leak_internals(
        self, client: TestClient
    ) -> None:
        """A stack trace tells an attacker about paths, versions and structure."""
        api.state.answerer = StubAnswerer(  # type: ignore[assignment]
            error=RuntimeError("connection to 10.0.0.5:443 failed: bad token sk-xyz")
        )
        detail = client.post("/ask", json={"question": "q"}).json()["detail"]

        assert detail == "The answering service is unavailable."
        assert "10.0.0.5" not in detail
        assert "sk-xyz" not in detail


class TestRateLimit:
    def test_a_burst_is_throttled(self, client: TestClient) -> None:
        api.state.budget = DailyCallBudget(limit=1000)
        codes = [
            client.post("/ask", json={"question": f"q{i}"}).status_code
            for i in range(15)
        ]

        assert 429 in codes, "expected the burst to be rate limited"
        assert codes.count(200) <= 10, "more requests served than the limit allows"


class TestHealth:
    def test_health_reports_what_an_operator_needs(self, client: TestClient) -> None:
        body = client.get("/health").json()

        assert body["status"] == "ok"
        assert body["chunks"] == 10
        assert body["answers_remaining_today"] == 100

    def test_health_is_not_rate_limited(self, client: TestClient) -> None:
        """A platform health check must not be throttled into reporting failure."""
        codes = {client.get("/health").status_code for _ in range(30)}
        assert codes == {200}

    def test_health_does_not_expose_configuration(self, client: TestClient) -> None:
        body: dict[str, Any] = client.get("/health").json()
        serialised = str(body).lower()

        for leak in ("key", "secret", "token", "sk-ant"):
            assert leak not in serialised


class TestPage:
    def test_the_page_renders(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert "Document Assistant" in response.text

    def test_the_page_never_assigns_untrusted_text_as_markup(
        self, client: TestClient
    ) -> None:
        """Model output is derived from documents; rendering it as HTML would be
        an injection route straight through the corpus.

        Matches an *assignment* rather than the bare word: the page's own
        comment explains why `innerHTML` is avoided, and a substring check on
        the word alone fails on the documentation of the very rule it enforces.
        """
        page = client.get("/").text

        for dangerous in ("innerHTML =", "innerHTML=", "outerHTML =", "document.write"):
            assert dangerous not in page, f"page uses {dangerous}"
        assert "textContent" in page, "expected text to be inserted safely"


class TestSecurityHeaders:
    """Headers the first deployment shipped without.

    Every functional check passed against the live service and the response
    carried no HSTS, no CSP, no nosniff and no frame protection — none of which
    breaks anything, which is exactly why nothing noticed. A header that is
    absent looks identical to a header that is working.
    """

    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ("Strict-Transport-Security", "max-age=63072000; includeSubDomains"),
            ("X-Content-Type-Options", "nosniff"),
            ("X-Frame-Options", "DENY"),
            ("Referrer-Policy", "strict-origin-when-cross-origin"),
        ],
    )
    def test_the_page_carries_them(
        self, client: TestClient, header: str, expected: str
    ) -> None:
        assert client.get("/").headers[header] == expected

    @pytest.mark.parametrize("path", ["/", "/health"])
    def test_json_routes_carry_them_too(self, client: TestClient, path: str) -> None:
        """Not just the HTML. An API response can be navigated to directly."""
        assert client.get(path).headers["X-Content-Type-Options"] == "nosniff"

    def test_even_a_rejected_request_carries_them(self, client: TestClient) -> None:
        """422s and 429s are written by the framework, not by us.

        Setting headers per route would silently miss every response we did not
        write ourselves, which is most of the error cases.
        """
        response = client.post("/ask", json={"question": "a" * 600})

        assert response.status_code == 422
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]

    def test_the_policy_denies_by_default(self, client: TestClient) -> None:
        policy = client.get("/").headers["Content-Security-Policy"]

        assert "default-src 'none'" in policy
        for directive in (
            "frame-ancestors 'none'",
            "object-src 'none'",
            "base-uri 'none'",
            "form-action 'none'",
            "upgrade-insecure-requests",
        ):
            assert directive in policy, f"missing {directive}"

    def test_the_policy_never_allows_unsafe_inline(self, client: TestClient) -> None:
        """The reason the nonce exists.

        'unsafe-inline' would authorise the page's own script *and* anything
        later injected into the markup — the precise attack CSP is for.
        """
        policy = client.get("/").headers["Content-Security-Policy"]

        assert "unsafe-inline" not in policy
        assert "unsafe-eval" not in policy

    def test_the_nonce_in_the_header_matches_the_one_in_the_markup(
        self, client: TestClient
    ) -> None:
        """A mismatch does not error, it silently blocks the page's own script.

        The UI would render and then do nothing when you pressed the button.
        """
        response = client.get("/")
        nonce = re.search(
            r"'nonce-([^']+)'", response.headers["Content-Security-Policy"]
        )

        assert nonce, "no nonce in the policy"
        assert response.text.count(f'nonce="{nonce.group(1)}"') == 2, (
            "both the inline <style> and the inline <script> must carry it"
        )

    def test_a_fresh_nonce_is_issued_per_response(self, client: TestClient) -> None:
        """A reused nonce is worth no more than 'unsafe-inline'.

        If it were constant, an attacker who read the page once could put the
        value on their own injected script and be authorised by it.
        """
        nonces = {
            re.search(  # type: ignore[union-attr]
                r"'nonce-([^']+)'", client.get("/").headers["Content-Security-Policy"]
            ).group(1)
            for _ in range(5)
        }

        assert len(nonces) == 5, f"nonce repeated across responses: {nonces}"

    def test_no_placeholder_survives_into_the_response(
        self, client: TestClient
    ) -> None:
        """A missed substitution ships the literal marker as the nonce."""
        assert api.NONCE_PLACEHOLDER not in client.get("/").text


class TestSharedSecret:
    """The control that stops a stranger spending the owner's API budget.

    Narrow by design and worth stating so it is not oversold: it is not
    authentication, there are no identities, and it protects nothing once the
    secret leaks. What it does is make the hostname insufficient — and for a
    service whose only cost is paid inference, that is the difference that
    matters.
    """

    def secured(self, secret: str = "correct-horse") -> None:
        api.state.settings = Settings(
            anthropic_api_key=SecretStr("test-key"),
            shared_secret=SecretStr(secret),
            require_shared_secret=True,
        )

    def test_ask_is_open_when_no_secret_is_required(self, client: TestClient) -> None:
        """The public demo stays public. The requirement is opt-in, so the
        default deployment is not silently broken by adding this."""
        response = client.post("/ask", json={"question": "What are the components?"})

        assert response.status_code == 200

    def test_ask_rejects_a_caller_with_no_secret(self, client: TestClient) -> None:
        self.secured()

        response = client.post("/ask", json={"question": "What are the components?"})

        assert response.status_code == 401

    def test_ask_rejects_a_caller_with_the_wrong_secret(
        self, client: TestClient
    ) -> None:
        self.secured()

        response = client.post(
            "/ask",
            json={"question": "What are the components?"},
            headers={api.SECRET_HEADER: "wrong"},
        )

        assert response.status_code == 401

    def test_ask_accepts_the_configured_secret(self, client: TestClient) -> None:
        self.secured()

        response = client.post(
            "/ask",
            json={"question": "What are the components?"},
            headers={api.SECRET_HEADER: "correct-horse"},
        )

        assert response.status_code == 200

    def test_a_rejected_caller_does_not_spend_budget(self, client: TestClient) -> None:
        """The whole point. If the check ran after the reservation, an
        unauthorised burst would still drain the day's allowance."""
        self.secured()
        before = api.state.budget.used

        client.post("/ask", json={"question": "expensive"})

        assert api.state.budget.used == before

    def test_a_rejection_says_nothing_about_which_part_was_wrong(
        self, client: TestClient
    ) -> None:
        self.secured()

        body = client.post("/ask", json={"question": "q"}).json()

        assert "correct-horse" not in str(body)
        assert body["detail"] == "Not authorised."

    def test_requiring_a_secret_without_setting_one_fails_closed(
        self, client: TestClient
    ) -> None:
        """Misconfiguration must not resolve to 'serve unauthenticated'.

        503 rather than 401: nothing is wrong with the caller, the service is
        not correctly configured, and saying so is what gets it fixed.
        """
        self.secured(secret="")

        response = client.post("/ask", json={"question": "q"})

        assert response.status_code == 503

    def test_health_stays_open(self, client: TestClient) -> None:
        """A platform health check cannot present a secret. Locking it would
        make the machine look dead to the thing that restarts it."""
        self.secured()

        assert client.get("/health").status_code == 200


class TestMetrics:
    def test_metrics_report_outcomes_without_question_text(
        self, client: TestClient
    ) -> None:
        client.post("/ask", json={"question": "What are the components?"})

        body = client.get("/metrics").json()

        assert body["outcomes"]["answered"] == 1
        assert "What are the components?" not in str(body)

    def test_an_ungrounded_answer_counts_as_not_covered(
        self, client: TestClient
    ) -> None:
        """Metrics use the vocabulary the visitor's screen uses, so a count and
        a screenshot describe the same thing."""
        api.state.answerer = StubAnswerer(  # type: ignore[assignment]
            Answer(
                text="Not in the documents.", citations=(), grounded=False, results=()
            )
        )

        client.post("/ask", json={"question": "Something else entirely"})

        assert client.get("/metrics").json()["outcomes"]["not_covered"] == 1

    def test_an_upstream_failure_counts_as_unavailable(
        self, client: TestClient
    ) -> None:
        api.state.answerer = StubAnswerer(error=RuntimeError("upstream down"))  # type: ignore[assignment]

        client.post("/ask", json={"question": "q"})

        assert client.get("/metrics").json()["outcomes"]["unavailable"] == 1

    def test_budget_exhaustion_counts_as_unavailable(self, client: TestClient) -> None:
        api.state.budget = DailyCallBudget(limit=0)

        client.post("/ask", json={"question": "q"})

        assert client.get("/metrics").json()["outcomes"]["unavailable"] == 1

    def test_metrics_require_the_secret_when_one_is_required(
        self, client: TestClient
    ) -> None:
        """How often an assistant refuses is operational information about
        someone's business, not something a passer-by is owed."""
        api.state.settings = Settings(
            anthropic_api_key=SecretStr("test-key"),
            shared_secret=SecretStr("correct-horse"),
            require_shared_secret=True,
        )

        assert client.get("/metrics").status_code == 401
        assert (
            client.get(
                "/metrics", headers={api.SECRET_HEADER: "correct-horse"}
            ).status_code
            == 200
        )

    def test_metrics_report_the_remaining_allowance(self, client: TestClient) -> None:
        body = client.get("/metrics").json()

        assert body["answers_remaining_today"] == api.state.budget.remaining
