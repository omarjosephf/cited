from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_release_evaluation import review_for, saved_run

from assistant.chunking import chunk_passages
from assistant.corpus_checksum import corpus_checksum
from assistant.documents import read_corpus
from assistant.release_evaluation import file_digest
from assistant.release_manifest import (
    ANSWER_CONTRACT_VERSION,
    ANSWER_RUNTIME_SOURCES,
    ReleaseManifest,
    contained_path,
    main,
    verify_manifest,
)
from assistant.vectors import save


@pytest.fixture
def release(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    def artifact(name: str, text: str) -> dict[str, str]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return {"path": name, "sha256": file_digest(path)}

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "work.md").write_text("# Work\n\nOJ builds websites.\n", encoding="utf-8")
    checksum = corpus_checksum(corpus)
    chunks = chunk_passages(read_corpus(corpus))
    matrix = np.zeros((len(chunks), 384), dtype=np.float32)
    matrix[:, 0] = 1
    save(
        tmp_path / "vectors.npz",
        chunks,
        matrix,
        model="BAAI/bge-small-en-v1.5",
        corpus_checksum=checksum,
    )
    answer_runtime = {
        name: artifact(f"answer-runtime/{name}", f"source for {name}")
        for name in ANSWER_RUNTIME_SOURCES
    }
    answer_configuration = {
        "primary_model": "gemini-3.5-flash-lite",
        "fallback_model": "gpt-5.6-luna",
        "answer_effort": "none",
        "answer_max_tokens": 1024,
        "top_k": 4,
        "prefilter_score": 0.45,
        "backend_timeout_seconds": 8.0,
        "provider_timeout_seconds": 6.0,
        "primary_timeout_seconds": 3.0,
        "validation_margin_seconds": 0.5,
        "wire_version": 3,
        "max_attempts": 2,
        "retries": 0,
        "fallback_mode": "availability_only",
        "complete_pair_required": True,
        "max_provider_request_bytes": 32000,
        "attempt_reservation_micro_usd": 40000,
        "daily_attempt_limit": 40,
        "monthly_attempt_limit": 200,
        "daily_budget_micro_usd": 400000,
        "monthly_budget_micro_usd": 2000000,
        "shared_worker_limit": 1,
        "budget_storage": "persistent_local_sqlite",
    }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "deployment": "oj-assistant",
        "frontend_commit": "a" * 40,
        "backend_commit": "b" * 40,
        "backend_image": "example/cited@sha256:" + "c" * 64,
        "python_base": "python:3.12-slim@sha256:" + "d" * 64,
        "runtime_lock": artifact("runtime.lock", "runtime"),
        "evaluator": artifact("evaluation.py", "evaluation"),
        "review_code": artifact("review.py", "review"),
        "policy": artifact("policy.py", "policy"),
        "corpus_path": "corpus",
        "corpus_sha256": checksum,
        "prompt": artifact("prompt.md", "Answer from evidence."),
        "embedding": {
            "model": "BAAI/bge-small-en-v1.5",
            "dimensions": 384,
            "model_lock": artifact(
                "model.lock.json", Path("model.lock.json").read_text(encoding="utf-8")
            ),
            "vectors": {
                "path": "vectors.npz",
                "sha256": file_digest(tmp_path / "vectors.npz"),
            },
        },
        "spec_version": "3.0",
        "answer_contract_version": ANSWER_CONTRACT_VERSION,
        "answer_runtime": answer_runtime,
        "answer_configuration": answer_configuration,
        "evaluations": [],
    }
    questions = artifact(
        "questions.toml",
        '[[question]]\ntext = "What does OJ build?"\nanswerable = true\n'
        'expects = "Work"\nclass = "supported_fact"\ncritical = true\n',
    )
    for suite in ("demo", "portfolio"):
        run = saved_run()
        run.update(
            capture_state="complete",
            suite=suite,
            corpus_sha256=checksum,
            questions_sha256=questions["sha256"],
            prompt_sha256=manifest["prompt"]["sha256"],
            runtime_lock_sha256=manifest["runtime_lock"]["sha256"],
            model_lock_sha256=manifest["embedding"]["model_lock"]["sha256"],
            evaluator_sha256=manifest["evaluator"]["sha256"],
            reviewer_code_sha256=manifest["review_code"]["sha256"],
            policy_sha256=manifest["policy"]["sha256"],
            answer_contract_version=ANSWER_CONTRACT_VERSION,
            answer_runtime_sha256={
                name: artifact["sha256"] for name, artifact in answer_runtime.items()
            },
            top_k=4,
            config=copy.deepcopy(answer_configuration),
        )
        review = review_for(run)
        manifest["evaluations"].append(
            {
                "suite": suite,
                "corpus_path": "corpus",
                "corpus_sha256": checksum,
                "questions": questions,
                "prompt": manifest["prompt"],
                "run": artifact(f"{suite}-run.json", json.dumps(run)),
                "review": artifact(f"{suite}-review.json", json.dumps(review)),
            }
        )
    return manifest, tmp_path


def test_complete_synthetic_release_verifies(
    release: tuple[dict[str, Any], Path],
) -> None:
    manifest, root = release
    verify_manifest(ReleaseManifest.model_validate(manifest), root)


@pytest.mark.parametrize(
    "file",
    [
        "corpus/work.md",
        "prompt.md",
        "runtime.lock",
        "answer-runtime/provider_router.py",
        "vectors.npz",
        "portfolio-review.json",
    ],
)
def test_tampered_artifacts_fail(
    release: tuple[dict[str, Any], Path], file: str
) -> None:
    manifest, root = release
    (root / file).write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify_manifest(ReleaseManifest.model_validate(manifest), root)


def test_both_distinct_suites_are_required(
    release: tuple[dict[str, Any], Path],
) -> None:
    manifest, root = release
    manifest["evaluations"][1] = copy.deepcopy(manifest["evaluations"][0])
    with pytest.raises(ValueError, match="distinct"):
        verify_manifest(ReleaseManifest.model_validate(manifest), root)


def test_valid_artifact_with_stale_identity_fails(
    release: tuple[dict[str, Any], Path],
) -> None:
    manifest, root = release
    manifest["evaluations"][0]["suite"] = "portfolio"
    manifest["evaluations"][1]["suite"] = "demo"
    with pytest.raises(ValueError, match="mismatched"):
        verify_manifest(ReleaseManifest.model_validate(manifest), root)


def _rewrite_run(manifest: dict[str, Any], root: Path, mutate: Any) -> None:
    evidence = manifest["evaluations"][0]
    path = root / evidence["run"]["path"]
    run = json.loads(path.read_text(encoding="utf-8"))
    mutate(run)
    path.write_text(json.dumps(run), encoding="utf-8")
    evidence["run"]["sha256"] = file_digest(path)


def test_captured_runtime_must_exactly_match_candidate(
    release: tuple[dict[str, Any], Path],
) -> None:
    manifest, root = release
    _rewrite_run(
        manifest,
        root,
        lambda run: run["answer_runtime_sha256"].__setitem__(
            "provider_router.py", "0" * 64
        ),
    )
    with pytest.raises(ValueError, match="runtime identity"):
        verify_manifest(ReleaseManifest.model_validate(manifest), root)


def test_captured_runtime_rejects_unexpected_sources(
    release: tuple[dict[str, Any], Path],
) -> None:
    manifest, root = release
    _rewrite_run(
        manifest,
        root,
        lambda run: run["answer_runtime_sha256"].__setitem__("unreviewed.py", "0" * 64),
    )
    with pytest.raises(ValueError, match="runtime identity"):
        verify_manifest(ReleaseManifest.model_validate(manifest), root)


def test_captured_runtime_rejects_missing_sources(
    release: tuple[dict[str, Any], Path],
) -> None:
    manifest, root = release
    _rewrite_run(
        manifest,
        root,
        lambda run: run["answer_runtime_sha256"].pop("budget.py"),
    )
    with pytest.raises(ValueError, match="runtime identity"):
        verify_manifest(ReleaseManifest.model_validate(manifest), root)


def test_captured_configuration_must_exactly_match_candidate(
    release: tuple[dict[str, Any], Path],
) -> None:
    manifest, root = release
    _rewrite_run(
        manifest,
        root,
        lambda run: run["config"].__setitem__("prefilter_score", 0.4),
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        verify_manifest(ReleaseManifest.model_validate(manifest), root)


def test_captured_configuration_rejects_missing_fields(
    release: tuple[dict[str, Any], Path],
) -> None:
    manifest, root = release
    _rewrite_run(
        manifest,
        root,
        lambda run: run["config"].pop("fallback_model"),
    )
    with pytest.raises(ValueError, match="missing or invalid"):
        verify_manifest(ReleaseManifest.model_validate(manifest), root)


def test_captured_answer_contract_must_match_candidate(
    release: tuple[dict[str, Any], Path],
) -> None:
    manifest, root = release
    _rewrite_run(
        manifest,
        root,
        lambda run: run.__setitem__("answer_contract_version", 2),
    )
    with pytest.raises(ValueError, match="stale or mismatched"):
        verify_manifest(ReleaseManifest.model_validate(manifest), root)


def test_v1_manifest_is_historical_and_cannot_qualify(
    release: tuple[dict[str, Any], Path],
) -> None:
    manifest, _ = release
    manifest["schema_version"] = 1
    with pytest.raises(ValueError):
        ReleaseManifest.model_validate(manifest)


def test_checked_in_v2_schema_matches_the_verifier_model() -> None:
    schema = Path(__file__).parents[1] / "docs/schemas/release-manifest-v2.schema.json"
    assert json.loads(schema.read_text(encoding="utf-8")) == (
        ReleaseManifest.model_json_schema()
    )


@pytest.mark.parametrize(
    "path", ["../outside", "C:/outside", "/outside", "..\\outside"]
)
def test_artifact_paths_cannot_escape(tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError):
        contained_path(tmp_path, path)


def test_observed_release_mismatch_exits_nonzero(
    release: tuple[dict[str, Any], Path],
) -> None:
    manifest, root = release
    expected, observed = root / "manifest.json", root / "observed.json"
    expected.write_text(json.dumps(manifest))
    assert main([str(expected), "--artifacts", str(root)]) == 0
    manifest["backend_commit"] = "e" * 40
    observed.write_text(json.dumps(manifest))
    assert (
        main([str(expected), "--artifacts", str(root), "--observed", str(observed)])
        == 1
    )


def test_incomplete_capture_cannot_qualify_release(
    release: tuple[dict[str, Any], Path],
) -> None:
    manifest, root = release
    _rewrite_run(
        manifest, root, lambda run: run.__setitem__("capture_state", "incomplete")
    )
    with pytest.raises(ValueError, match="complete routed capture"):
        verify_manifest(ReleaseManifest.model_validate(manifest), root)
