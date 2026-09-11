from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.prepare_model import verify


def test_model_verifier_rejects_changed_bytes(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"reviewed model")
    files = {model.name: hashlib.sha256(model.read_bytes()).hexdigest()}
    verify(tmp_path, files)
    model.write_bytes(b"different model")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify(tmp_path, files)


def test_model_verifier_rejects_parent_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        verify(tmp_path, {"../outside": "a" * 64})
