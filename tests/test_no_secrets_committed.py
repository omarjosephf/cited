"""Guard against a secret reaching a tracked file.

This exists because it nearly happened. An API key was pasted into
`.env.example` — a *tracked* template — instead of `.env`, the ignored file
beside it. Nothing caught it; it was noticed by chance, one `git add -A` away
from entering history permanently.

Removing a secret from git history means rewriting history, not deleting a line,
and on a public repository the key must be treated as compromised regardless.
The cost of prevention is this file. The cost of the failure is disproportionate
to it, which is the whole argument.

The check runs against **tracked files only**, deliberately. `.env` is supposed
to contain a real key, and scanning it would make this test fail on every
correctly-configured machine — a test that cries wolf is a test that gets
disabled.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Prefixes for credentials that plausibly appear in this project or get pasted
# into it by accident. Matching the *prefix* rather than trying to detect
# "something secret-looking" keeps false positives at zero, which is what keeps
# the test trusted.
SECRET_PATTERNS = {
    "Anthropic API key": re.compile(r"sk-ant-[a-z0-9]+-[A-Za-z0-9_\-]{20,}"),
    "OpenAI API key": re.compile(r"sk-proj-[A-Za-z0-9_\-]{20,}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "Private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def tracked_files() -> list[Path]:
    """Every file git knows about — the only ones that can leak."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO / line for line in result.stdout.splitlines() if line]


def test_git_is_available() -> None:
    """If git cannot be queried, this whole file silently checks nothing."""
    assert tracked_files(), "expected at least one tracked file"


@pytest.mark.parametrize("label,pattern", list(SECRET_PATTERNS.items()))
def test_no_tracked_file_contains_a_secret(
    label: str, pattern: re.Pattern[str]
) -> None:
    offenders: list[str] = []

    for path in tracked_files():
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: cannot contain a pasted key
        if pattern.search(text):
            offenders.append(str(path.relative_to(REPO)))

    assert not offenders, (
        f"{label} found in tracked file(s): {', '.join(offenders)}. "
        "Move it to .env, which is git-ignored, and revoke the exposed "
        "credential — a secret that has been written into a tracked file must "
        "be treated as compromised even if it was never committed."
    )


def test_the_env_template_carries_no_value() -> None:
    """`.env.example` must document names, never values.

    Checked separately from the pattern scan because a key of some future
    format would not match any pattern above, while an empty assignment is
    unambiguous whatever the credential looks like.
    """
    template = REPO / ".env.example"
    assert template.exists(), ".env.example is the documented setup path"

    for number, line in enumerate(template.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        _, _, value = stripped.partition("=")
        assert not value.strip(), (
            f".env.example line {number} has a value: {stripped.split('=')[0]}=... "
            "The template must carry names and comments only."
        )


def test_dotenv_is_ignored_by_git() -> None:
    """The protection this all rests on."""
    result = subprocess.run(
        ["git", "check-ignore", ".env"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, ".env is NOT git-ignored — fix .gitignore first"


def test_dotenv_is_not_tracked() -> None:
    """Ignoring a file does nothing if it was already added."""
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".env"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        ".env is tracked by git. `git rm --cached .env` and revoke every "
        "credential it has ever contained."
    )
