from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import tools.run_outcome_mixture_screen as screen
from tools.run_outcome_mixture_screen import (
    _validate_git_blob_binding,
    load_outcome_mixture_design,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-017-outcome-mixture-design.json"
)


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


def test_round17_design_is_hash_bound_current_and_terminal_sealed() -> None:
    design, design_sha256 = load_outcome_mixture_design(DESIGN)

    assert (
        design_sha256
        == "963ecc6d9fa384969992bed36addff0cfceb3e057fbe43a91725e15d037db1ee"
    )
    assert design["implementation"]["hash_mode"] == "git_blob_sha256_v1"
    assert design["implementation"]["commit"] == (
        "016b6890db4cf34610848aac3bf5effd428c0ef2"
    )
    assert design["design_revision"] == 2
    assert design["model"]["ranking_loss_weight"] == 0.0
    assert design["reserved_terminal"] == {
        "date": "2023-07-07",
        "included_in_dataset": False,
        "access_permitted": False,
    }
    assert design["leverage_applied"] is False


def test_round17_design_rejects_hash_tampering(tmp_path: Path) -> None:
    payload = json.loads(DESIGN.read_text(encoding="utf-8"))
    payload["model"]["expected_value_loss_weight"] = 0.5
    source = tmp_path / "tampered.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="design hash is invalid"):
        load_outcome_mixture_design(source, require_current=False)


def test_profile_evaluation_calls_the_sealed_threshold_api(monkeypatch) -> None:
    design, _design_sha256 = load_outcome_mixture_design(DESIGN, require_current=False)
    score = SimpleNamespace(eligible=np.zeros(4, dtype=bool))
    selection = SimpleNamespace(
        accepted=False,
        threshold_bps=None,
        asdict=lambda: {"accepted": False, "threshold_bps": None},
    )
    trace = SimpleNamespace(
        metrics=SimpleNamespace(trades=0, total_net_bps=0.0),
        asdict=lambda: {"metrics": {"trades": 0, "total_net_bps": 0.0}},
    )
    calls: list[tuple[tuple[float, ...], tuple[int, ...], object, float]] = []

    monkeypatch.setattr(screen, "derive_action_scores", lambda *_args: score)

    def select(
        _dataset,
        _targets,
        _score,
        *,
        quantiles,
        expected_days,
        gates,
        drawdown_penalty,
    ):
        calls.append((quantiles, expected_days, gates, drawdown_penalty))
        return selection

    monkeypatch.setattr(screen, "select_barrier_threshold", select)
    monkeypatch.setattr(screen, "_empty_profile_trace", lambda *_args, **_kwargs: trace)

    results, survivors = screen._evaluate_profiles(
        design=design,
        dataset=object(),
        targets=object(),
        calibration_prediction=object(),
        policy_prediction=object(),
        progress=lambda *_args, **_kwargs: None,
    )

    assert len(results) == 3
    assert survivors == []
    assert len(calls) == 3
    assert all(call[0] == (0.5, 0.7, 0.85, 0.95) for call in calls)
    assert all(call[3] == 0.5 for call in calls)
