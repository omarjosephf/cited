"""Local management-panel transport, privacy and browser protections."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from assistant.inspection import CorpusProfile, inspect_corpus
from assistant.inspector import create_inspector_app
from assistant.web_security import NONCE_PLACEHOLDER


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    snapshots = []
    for label, folder, subject in (
        ("Cited", "cited", "prompt design"),
        ("OJ Assistant", "oj", "portfolio evidence"),
    ):
        directory = tmp_path / folder
        directory.mkdir()
        (directory / "guide.md").write_text(
            "# Overview\n\n" + (f"This document covers {subject}. " * 30),
            encoding="utf-8",
        )
        snapshots.append(inspect_corpus(CorpusProfile.create(label, directory)))
    return TestClient(create_inspector_app(snapshots))


def test_corpus_selector_api_contains_both_independent_corpora(
    client: TestClient,
) -> None:
    response = client.get("/api/corpora")

    assert response.status_code == 200
    corpora = response.json()["corpora"]
    assert [(item["id"], item["label"]) for item in corpora] == [
        ("cited", "Cited"),
        ("oj-assistant", "OJ Assistant"),
    ]
    assert all(item["document_count"] == 1 for item in corpora)


def test_detail_and_chunks_are_read_only_relative_views(client: TestClient) -> None:
    detail = client.get("/api/corpora/oj-assistant")
    chunks = client.get("/api/corpora/oj-assistant/chunks?page=1&page_size=1")
    filtered = client.get("/api/corpora/oj-assistant/chunks?q=portfolio")

    assert detail.status_code == 200
    assert detail.json()["documents"][0]["source"] == "guide.md"
    assert len(detail.json()["documents"][0]["sha256"]) == 64
    assert detail.json()["documents"][0]["estimated_embedding_tokens"] > 0
    assert chunks.json()["page_size"] == 1
    assert filtered.json()["total"] >= 1
    first = filtered.json()["items"][0]
    assert "text" not in first
    assert client.get(f"/api/corpora/oj-assistant/chunks/{first['index']}").json()[
        "text"
    ]


def test_unknown_items_are_not_found_and_writes_are_not_exposed(
    client: TestClient,
) -> None:
    assert client.get("/api/corpora/unknown").status_code == 404
    assert client.get("/api/corpora/cited/chunks/9999").status_code == 404
    assert client.post("/api/corpora", json={}).status_code == 405
    assert client.put("/api/corpora/cited", json={}).status_code == 405
    assert client.delete("/api/corpora/cited").status_code == 405


def test_every_response_is_private_hardened_and_not_cacheable(
    client: TestClient,
) -> None:
    for response in (client.get("/"), client.get("/api/corpora"), client.get("/nope")):
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-robots-tag"] == "noindex, nofollow"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["content-security-policy"].startswith(
            "default-src 'none'"
        )


def test_html_uses_a_fresh_csp_nonce_and_safe_dom_updates(client: TestClient) -> None:
    first = client.get("/")
    second = client.get("/")
    nonce = re.search(
        r"script-src 'nonce-([^']+)'",
        first.headers["content-security-policy"],
    )

    assert nonce is not None
    assert f'nonce="{nonce.group(1)}"' in first.text
    assert NONCE_PLACEHOLDER not in first.text
    assert (
        first.headers["content-security-policy"]
        != second.headers["content-security-policy"]
    )
    assert "textContent" in first.text
    assert "innerHTML" not in first.text
    assert "outerHTML" not in first.text
    assert "insertAdjacentHTML" not in first.text


def test_unexpected_host_is_rejected_before_routing(client: TestClient) -> None:
    response = client.get("/api/corpora", headers={"host": "panel.example.com"})

    assert response.status_code == 400
    assert response.json() == {"detail": "Local access only."}
    assert response.headers["cache-control"] == "no-store"
