"""Checks on the deployment configuration itself.

Docker is not installed on the development machine, so the image is built for
the first time by the deploy platform. That makes a broken Dockerfile a
*production* discovery: it fails after a push, in a build log, rather than in
front of whoever wrote it.

These tests do not build an image. They check the things that can be checked
without one — that the file list is complete, that the two halves of a
configuration agree, and that the guarantees written in comments are the
guarantees the file actually provides.

One of these is a regression test. The build stage did not copy `LICENSE`, which
`pyproject.toml` names, and hatchling refuses to generate metadata without it.
The image could not be built at all, and nothing in the repository said so.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO / "Dockerfile"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def pyproject() -> dict[str, object]:
    with (REPO / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def build_stage_sources(dockerfile: str) -> list[str]:
    """The files available to the build stage when `pip install` runs.

    A Dockerfile COPY list is just a file list, so this can be read exactly
    without Docker present.
    """
    sources: list[str] = []
    for line in dockerfile.splitlines():
        stripped = line.strip()
        if stripped.startswith("RUN pip install"):
            break  # everything after this is too late to matter
        if stripped.startswith("COPY ") and "--from" not in stripped:
            tokens = stripped.split()[1:]
            sources.extend(t for t in tokens[:-1] if not t.startswith("--"))
    return sources


class TestBuildStageHasEverythingItNeeds:
    def test_every_file_pyproject_names_is_copied_before_the_install(
        self, dockerfile: str, pyproject: dict[str, object]
    ) -> None:
        """Metadata generation reads these off disk; missing means build failure.

        This is the regression test. `license = { file = "LICENSE" }` was
        declared while the Dockerfile copied only pyproject.toml and README.md,
        so hatchling raised "License file does not exist: LICENSE" and the image
        could never be built.
        """
        project = pyproject["project"]
        assert isinstance(project, dict)

        required = {project["readme"]}
        licence = project["license"]
        if isinstance(licence, dict) and "file" in licence:
            required.add(licence["file"])

        copied = set(build_stage_sources(dockerfile))
        missing = required - copied
        assert not missing, (
            f"pyproject.toml references {sorted(missing)}, which the build stage "
            f"never copies. Metadata generation reads these files, so the image "
            f"build fails outright — on the deploy platform, after a push."
        )

    def test_the_package_source_is_copied(self, dockerfile: str) -> None:
        assert "src/" in build_stage_sources(dockerfile)

    def test_the_declared_python_version_matches_the_base_image(
        self, dockerfile: str, pyproject: dict[str, object]
    ) -> None:
        """A drifted base image fails on syntax, at import, in production.

        The site-packages path in the model-download step is also version-
        specific, so a bump has to happen in more than one place here.
        """
        project = pyproject["project"]
        assert isinstance(project, dict)
        requires = project["requires-python"]
        assert isinstance(requires, str)
        minimum = requires.removeprefix(">=").strip()

        images = re.findall(r"^FROM python:(\S+)", dockerfile, re.MULTILINE)
        assert images, "no python base image found"
        assert set(images) == {f"{minimum}-slim"}, (
            f"pyproject requires Python {minimum} but the image(s) are {images}"
        )
        assert f"/install/lib/python{minimum}/site-packages" in dockerfile, (
            "the site-packages path in the model-download step is hardcoded and "
            "must be bumped alongside the base image"
        )


@pytest.fixture(scope="module")
def fly() -> dict[str, Any]:
    with (REPO / "fly.toml").open("rb") as handle:
        return tomllib.load(handle)


class TestPlatformConfigAgreesWithTheImage:
    """Three files have to name the same port, in three different syntaxes.

    A mismatch does not raise anything. The container starts, serves nobody,
    fails its health check, and the platform reports the deployment as unhealthy
    without saying why.
    """

    def test_the_port_is_the_same_everywhere(
        self, fly: dict[str, Any], dockerfile: str
    ) -> None:
        declared = int(fly["env"]["PORT"])
        assert fly["http_service"]["internal_port"] == declared, (
            "fly routes to internal_port; the app listens on $PORT"
        )
        assert f"PORT={declared}" in dockerfile, (
            "the Dockerfile default must match, or a local `docker run` without "
            "an explicit PORT listens somewhere else"
        )
        assert f"EXPOSE {declared}" in dockerfile

    def test_https_is_forced(self, fly: dict[str, Any]) -> None:
        assert fly["http_service"]["force_https"] is True

    def test_the_health_check_avoids_the_paid_path(self, fly: dict[str, Any]) -> None:
        """Checked every 30s. Pointing it at /ask would bill for every check."""
        checks = fly["http_service"]["checks"]
        assert [c["path"] for c in checks] == ["/health"]

    def test_the_budget_survives_between_requests(self, fly: dict[str, Any]) -> None:
        """The daily budget is a counter in process memory.

        A machine that stops and restarts resets it, so the limit it appears to
        enforce is not the limit that applies. Scaling to zero is a legitimate
        choice, but it must be a deliberate one — this test makes changing it
        require saying so.
        """
        assert fly["http_service"]["min_machines_running"] == 1, (
            "setting this to 0 leaves the provider-side spend cap as the only "
            "real ceiling on spend; update this test deliberately if that is "
            "the intent"
        )

    def test_the_ha_trap_is_documented(self) -> None:
        """Fly adds a second machine unless a deploy flag says otherwise.

        There is no fly.toml setting for it, so the only thing that can carry
        this warning is the file a person reads before deploying. It happened on
        the first deploy of this app: two machines, two budget counters, a 200/day
        limit that permitted 400, and double the bill.
        """
        config = (REPO / "fly.toml").read_text(encoding="utf-8")

        assert "--ha=false" in config, (
            "the deploy command must be recorded here; nothing else prevents a "
            "second machine from silently doubling the spend ceiling"
        )

    def test_the_memory_tier_has_headroom_over_what_was_measured(
        self, fly: dict[str, Any]
    ) -> None:
        """Measured peak is 263 MB; ~300 MB with the web layer on top.

        Written down because the previous value was reasoned rather than
        measured, was wrong, and cost roughly three times the correct tier.
        """
        memory = str(fly["vm"][0]["memory"]).lower()
        megabytes = (
            int(memory.removesuffix("gb")) * 1024
            if memory.endswith("gb")
            else int(memory.removesuffix("mb"))
        )
        assert megabytes >= 512, f"{memory} is below the measured ~300 MB working set"


class TestSecretsStayOutOfTheImage:
    def test_dotenv_is_excluded(self) -> None:
        """A .env in an image is a credential published with it.

        Layers are readable by anyone who can pull the image, and deleting the
        file in a later layer does not remove it from the earlier one.
        """
        ignored = (REPO / ".dockerignore").read_text(encoding="utf-8").splitlines()
        entries = {line.strip() for line in ignored}
        assert ".env" in entries
        assert ".env.*" in entries
        assert "!.env.example" in entries, (
            "the example file carries no values and documents the variables; "
            "excluding it makes the image harder to understand for no gain"
        )

    def test_the_api_key_is_not_baked_in(self, dockerfile: str) -> None:
        """It belongs in the platform's secret store, set outside the build."""
        assert "ANTHROPIC_API_KEY" not in dockerfile


class TestTheCorpusReachesTheImage:
    """The documents are Markdown, and .dockerignore excludes `*.md`.

    Those two facts sit in different files and are fine only because Docker's
    patterns do not cross a `/` — `*.md` matches README.md at the root, not
    content/prompt-engineering-fundamentals.md. That is a documented behaviour
    the whole demo rests on, and a pattern changed to `**/*.md` in a tidy-up
    would empty the corpus.

    The failure is at least loud: the lifespan raises when no documents load, so
    the container dies at startup instead of serving a system that refuses every
    question. Loud and in production is still worse than loud and here.
    """

    def test_no_ignore_pattern_reaches_into_the_corpus(self) -> None:
        patterns = [
            line.strip()
            for line in (REPO / ".dockerignore")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.startswith("#")
        ]
        reaching = [
            p for p in patterns if p.startswith("**/") and not p.startswith("!")
        ]
        assert not reaching, (
            f"{reaching} cross directory boundaries and may exclude the corpus"
        )
        assert "content" not in patterns

    def test_the_corpus_is_copied_into_the_runtime_stage(self, dockerfile: str) -> None:
        assert "COPY --chown=app:app content/ ./content/" in dockerfile

    def test_the_working_directory_matches_the_corpus_path(
        self, dockerfile: str
    ) -> None:
        """The app opens `content/` relative to the working directory."""
        assert "WORKDIR /app" in dockerfile
        assert "./content/" in dockerfile


class TestRuntimeMatchesItsAssumptions:
    def test_it_runs_as_a_non_root_user(self, dockerfile: str) -> None:
        assert re.search(r"^USER app", dockerfile, re.MULTILINE)

    def test_it_binds_to_all_interfaces(self, dockerfile: str) -> None:
        """127.0.0.1 would start cleanly and be unreachable through the proxy."""
        assert "--host 0.0.0.0" in dockerfile

    def test_it_runs_a_single_worker(self, dockerfile: str) -> None:
        """Two workers would quietly double the daily budget.

        The budget is a counter in process memory (see assistant/budget.py), so
        each worker enforces its own limit. Two workers with a 200-call limit
        permit 400 calls, and the spend is real.
        """
        assert "--workers 1" in dockerfile

    def test_the_port_is_not_hardcoded(self, dockerfile: str) -> None:
        """The platform assigns it; a fixed port fails health checks silently."""
        assert "--port ${PORT}" in dockerfile
