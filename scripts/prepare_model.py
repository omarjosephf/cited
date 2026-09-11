"""Fetch the reviewed model revision and verify all model/tokenizer bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def verify(snapshot: Path, files: dict[str, str]) -> None:
    for name, expected in files.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe model lock path")
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"model file digest mismatch: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads((Path(__file__).parents[1] / "model.lock.json").read_text())
    snapshot = Path(
        snapshot_download(
            repo_id=lock["repository"],
            revision=lock["revision"],
            allow_patterns=list(lock["files"]),
            cache_dir=args.cache_dir,
        )
    )
    # Hugging Face snapshots use symlinks to cache blobs on Linux. Verify the
    # named snapshot entries, allowing those normal cache symlinks.
    verify(snapshot, lock["files"])
    # FastEmbed asks for the local main ref. Bind it to the verified immutable
    # snapshot, then run builds/containers with HF_HUB_OFFLINE=1.
    ref = (
        args.cache_dir
        / ("models--" + lock["repository"].replace("/", "--"))
        / "refs"
        / "main"
    )
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(lock["revision"], encoding="utf-8")
    print(f"Verified model snapshot {lock['revision']}")


if __name__ == "__main__":
    main()
