from __future__ import annotations

import hashlib
import subprocess

import pytest

from tools.run_outcome_mixture_screen import _validate_git_blob_binding


def _git(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _binding(*, path: str = "README.md", sha256: str | None = None):
    commit = _git("rev-parse", "HEAD").decode("ascii").strip()
    digest = hashlib.sha256(_git("show", f"{commit}:{path}")).hexdigest()
    return {
        "hash_mode": "git_blob_sha256_v1",
        "commit": commit,
        "files": [{"path": path, "sha256": sha256 or digest}],
    }


def test_git_blob_binding_is_cross_platform_and_current() -> None:
    _validate_git_blob_binding(_binding())


def test_git_blob_binding_rejects_hash_and_path_tampering() -> None:
    with pytest.raises(ValueError, match="implementation changed"):
        _validate_git_blob_binding(_binding(sha256="0" * 64))
    unsafe = _binding()
    unsafe["files"][0]["path"] = "../README.md"
    with pytest.raises(ValueError, match="path is unsafe"):
        _validate_git_blob_binding(unsafe)


def test_git_blob_binding_rejects_incomplete_contract() -> None:
    binding = _binding()
    binding["hash_mode"] = "workspace_bytes"
    with pytest.raises(ValueError, match="binding is incomplete"):
        _validate_git_blob_binding(binding)
