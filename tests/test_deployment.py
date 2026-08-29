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


@pytest.fixture(scope="module")
def oj_fly() -> dict[str, Any]:
    with (REPO / "fly.oj-assistant.toml").open("rb") as handle:
        return tomllib.load(handle)


class TestSecondDeploymentIsCoherent:
    """The OJ Assistant app: same image, different corpus.

    A second config sharing a repository with the first has one obvious sharp
    edge — running the wrong deploy command redeploys the wrong app — and one
    subtle one, which is that the demo's corpus is still inside the image. If
    CORPUS_DIR were wrong or absent, this deployment would answer questions about
    OJ using a prompt-engineering guide, sincerely and with citations.
    """

    def test_it_is_a_different_app_from_the_demo(
        self, oj_fly: dict[str, Any], fly: dict[str, Any]
    ) -> None:
        """Separate apps mean separate budgets: a burst against one cannot drain
        the other's allowance."""
        assert oj_fly["app"] != fly["app"]

    def test_the_port_is_the_same_everywhere(
        self, oj_fly: dict[str, Any], dockerfile: str
    ) -> None:
        declared = int(oj_fly["env"]["PORT"])
        assert oj_fly["http_service"]["internal_port"] == declared
        assert f"PORT={declared}" in dockerfile

    def test_https_is_forced(self, oj_fly: dict[str, Any]) -> None:
        assert oj_fly["http_service"]["force_https"] is True

    def test_it_points_away_from_the_demo_corpus(
        self, oj_fly: dict[str, Any], dockerfile: str
    ) -> None:
        """The failure this prevents is silent and confident: right service,
        wrong documents, every answer cited and wrong."""
        corpus = oj_fly["env"]["CORPUS_DIR"]

        assert corpus != "/app/content"
        assert corpus.startswith("/app/deploy/")
        assert "COPY --chown=app:app deploy/ ./deploy/" in dockerfile, (
            "the artifact directory must reach the runtime stage"
        )

    def test_the_corpus_is_verified_against_a_checksum(
        self, oj_fly: dict[str, Any]
    ) -> None:
        """A corpus copied from another repository is the one thing here that
        can go stale without anything failing."""
        env = oj_fly["env"]
        assert "CORPUS_CHECKSUM_FILE" in env or "CORPUS_CHECKSUM" in env

    def test_the_checksum_file_lives_inside_the_corpus_artifact(
        self, oj_fly: dict[str, Any]
    ) -> None:
        artifact_root = oj_fly["env"]["CORPUS_DIR"].rsplit("/", 1)[0]

        assert oj_fly["env"]["CORPUS_CHECKSUM_FILE"].startswith(artifact_root)

    def test_it_uses_its_own_system_prompt(self, oj_fly: dict[str, Any]) -> None:
        """Otherwise it answers as a generic document assistant — accurate, and
        not anybody's assistant."""
        assert oj_fly["env"]["SYSTEM_PROMPT_FILE"].endswith(".md")
        assert oj_fly["env"]["SYSTEM_PROMPT_FILE"].startswith("/app/deploy/")

    def test_only_an_authorised_caller_may_spend_the_budget(
        self, oj_fly: dict[str, Any]
    ) -> None:
        """This instance is funded by the owner's own key rather than being a
        public demo, so the hostname must not be enough to spend it."""
        assert oj_fly["env"]["REQUIRE_SHARED_SECRET"] == "true"

    def test_no_secret_is_present_in_the_committed_config(
        self, oj_fly: dict[str, Any]
    ) -> None:
        """Checked by key name and by value, not by substring.

        A substring scan flags `REQUIRE_SHARED_SECRET`, which is a boolean switch
        and carries nothing. A test that cries wolf on a correct config gets
        weakened or deleted, so it has to be precise about what it forbids: a
        *value-bearing* secret variable, and anything shaped like a real key.
        """
        env: dict[str, str] = oj_fly["env"]

        forbidden_keys = {"ANTHROPIC_API_KEY", "SHARED_SECRET"}
        assert not forbidden_keys & set(env), (
            "this file is committed; secrets belong in `fly secrets set`"
        )

        for name, value in env.items():
            assert not str(value).startswith("sk-"), f"{name} looks like a live key"

    def test_the_retrieval_baseline_is_explicit(self, oj_fly: dict[str, Any]) -> None:
        """Pinned rather than defaulted, so a change to the library default
        cannot quietly change what was evaluated."""
        assert oj_fly["env"]["RETRIEVAL_TOP_K"] == "4"
        assert oj_fly["env"]["ANSWER_MAX_TOKENS"] == "1024"

    def test_scale_to_zero_is_paired_with_its_consequence_in_writing(self) -> None:
        """The trade is only sound if the reader knows the counter resets.

        A future maintainer reading `min_machines_running = 0` and
        `DAILY_ANSWER_LIMIT = 40` would reasonably conclude the spend is capped
        at 40 answers a day. It is not: the machine stops when idle and the
        counter starts again at zero. The comment is the only thing that carries
        that, so its absence is a defect.
        """
        config = (REPO / "fly.oj-assistant.toml").read_text(encoding="utf-8").lower()

        assert "min_machines_running = 0" in config
        assert "reset" in config
        assert "spend cap" in config or "provider cap" in config

    def test_the_deploy_command_names_the_config_and_disables_ha(self) -> None:
        """Two traps, both of which produce a working deployment of the wrong
        thing: omitting --config redeploys the demo app, and omitting --ha=false
        doubles the budget by doubling the process."""
        config = (REPO / "fly.oj-assistant.toml").read_text(encoding="utf-8")

        assert "--config fly.oj-assistant.toml" in config
        assert "--ha=false" in config

    def test_the_memory_tier_has_headroom_over_what_was_measured(
        self, oj_fly: dict[str, Any]
    ) -> None:
        """263 MB measured peak. The failure mode below it is an OOM kill
        mid-request, which explains itself to nobody."""
        assert str(oj_fly["vm"][0]["memory"]).lower() == "512mb"

    def test_the_health_check_avoids_the_paid_path(
        self, oj_fly: dict[str, Any]
    ) -> None:
        checks = oj_fly["http_service"]["checks"]
        assert all(check["path"] == "/health" for check in checks)


class TestDeploymentStagingArea:
    def test_the_directory_survives_a_clean_checkout(self) -> None:
        """`COPY deploy/` fails on a missing source, so an empty staging area
        must still exist. Without the keepfile, cleaning generated artifacts
        would break every build of the default image."""
        assert (REPO / "deploy" / ".gitkeep").exists()

    def test_generated_artifacts_are_not_committed(self) -> None:
        """A corpus committed here would be a second editable copy of public
        claims — the exact drift the checksum exists to detect."""
        ignore = (REPO / ".gitignore").read_text(encoding="utf-8")

        assert "deploy/*" in ignore
        assert "!deploy/.gitkeep" in ignore
