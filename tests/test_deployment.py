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
