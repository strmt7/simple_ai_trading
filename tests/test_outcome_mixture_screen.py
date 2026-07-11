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
DESIGN18 = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-018-ranked-outcome-mixture-design.json"
)
DESIGN19 = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-019-depth-normalized-order-flow-design.json"
)
DESIGN20 = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-020-parameter-matched-side-tower-design.json"
)
DESIGN21 = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-021-pairwise-net-return-ranking-design.json"
)
DESIGN21_V2 = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-021-pairwise-net-return-ranking-design-v2.json"
)
DESIGN21_V3 = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-021-pairwise-net-return-ranking-design-v3.json"
)


def _git(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _binding(*, path: str = "pyproject.toml", sha256: str | None = None):
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


def test_round17_design_is_hash_bound_to_historical_implementation() -> None:
    design, design_sha256 = load_outcome_mixture_design(DESIGN, require_current=False)

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


def test_round18_design_is_historical_and_changes_only_the_ranking_ablation() -> None:
    design, design_sha256 = load_outcome_mixture_design(DESIGN18, require_current=False)
    predecessor, _predecessor_sha256 = load_outcome_mixture_design(
        DESIGN, require_current=False
    )

    assert (
        design_sha256
        == "024b1146d3330e9306470dd29a3ec7c49c686e0fb66ad9c20c7be2d02afb5c40"
    )
    assert design["round"] == 18
    assert design["model"]["ranking_loss_weight"] == 0.1
    for section in (
        "data",
        "execution",
        "barrier_targets",
        "runtime_resources",
        "event_sampler",
        "training",
        "threshold_policy",
        "risk_profiles",
        "evaluation",
        "reserved_terminal",
    ):
        assert design[section] == predecessor[section]


def test_round19_contract_changes_only_the_causal_feature_family() -> None:
    contract = screen._ROUND_CONTRACTS[19]

    assert contract["design_revisions"] == {1, 2}
    assert contract["purposes"][2] == (
        "consumed_data_depth_normalized_order_flow_outcome_mixture_screen"
    )
    assert contract["ranking_loss_weight"] == 0.1
    assert contract["feature_version"] == "l1-tape-causal-v8"
    assert contract["predecessors"][2]["round"] == 18
    assert contract["predecessors"][2]["publication_sha256"] == (
        "1086ae098eb77679023c36dd3b42355aef52f6daa8de720b41c718ecaa00d378"
    )


def test_round20_contract_is_parameter_matched_and_changes_only_architecture() -> None:
    contract = screen._ROUND_CONTRACTS[20]

    assert contract["side_tower_mode"] == "independent"
    assert contract["hidden_dim"] == 88
    assert contract["residual_blocks"] == 2
    assert contract["trainable_parameter_count"] == 145_914
    assert abs(contract["trainable_parameter_count"] / 147_722 - 1.0) < 0.02
    assert contract["ranking_loss_weight"] == 0.1
    assert contract["feature_version"] == "l1-tape-causal-v8"
    assert contract["predecessor"]["round"] == 19
    assert contract["predecessor"]["publication_sha256"] == (
        "2b72894744be750357c5913ffe2b71787c3f70e595e41e08753f0a93bfc61c86"
    )


def test_round21_contract_changes_only_the_net_return_ranking_surrogate() -> None:
    contract = screen._ROUND_CONTRACTS[21]

    assert contract["design_revisions"] == {1, 2, 3}
    assert contract["purposes"][3] == (
        "consumed_data_pairwise_net_return_ranking_reproducible_artifact_screen"
    )
    assert contract["ranking_loss_mode"] == "pairwise_net_return"
    assert contract["ranking_loss_weight"] == 0.1
    assert contract["side_tower_mode"] == "independent"
    assert contract["hidden_dim"] == 88
    assert contract["residual_blocks"] == 2
    assert contract["trainable_parameter_count"] == 145_914
    assert contract["feature_version"] == "l1-tape-causal-v8"
    assert contract["predecessors"][3]["round"] == 20
    assert contract["predecessors"][3]["publication_sha256"] == (
        "3e8a22398871f80020743ee9987a670cfbf50292e351fa018f513c3c535c2033"
    )
    assert "CPU fallback" in contract["predecessors"][2]["finding"]
    assert "nonstable Safetensors" in contract["predecessors"][3]["finding"]


def test_round19_design_is_historical_and_preserves_sealed_controls() -> None:
    design, design_sha256 = load_outcome_mixture_design(DESIGN19, require_current=False)
    predecessor, _predecessor_sha256 = load_outcome_mixture_design(
        DESIGN18, require_current=False
    )

    assert design_sha256 == (
        "2a2c2e1c52d7dd0a6c8ac1a34e26defe3ec436a5051fcc49ed2172ef9f87ca77"
    )
    assert design["round"] == 19
    assert design["implementation"]["commit"] == (
        "123d6edf69411a3c942b6c7ce3a511706c5f0ccb"
    )
    assert screen.MICROSTRUCTURE_FEATURE_VERSION == "l1-tape-causal-v8"
    assert design["model"]["ranking_loss_weight"] == 0.1
    assert design["model"]["candidate_id"] == (
        "depth-normalized-order-flow-conditional-outcome-mixture"
    )
    for section in (
        "data",
        "execution",
        "barrier_targets",
        "runtime_resources",
        "event_sampler",
        "training",
        "threshold_policy",
        "risk_profiles",
        "evaluation",
        "reserved_terminal",
    ):
        assert design[section] == predecessor[section]


def test_round20_design_is_historical_and_changes_only_side_architecture() -> None:
    design, design_sha256 = load_outcome_mixture_design(DESIGN20, require_current=False)
    predecessor, _predecessor_sha256 = load_outcome_mixture_design(
        DESIGN19, require_current=False
    )

    assert design_sha256 == (
        "a6f4e82d82474d673c8495f9775f9d974b95a9cc2a8d497f7f45bce29ad965bb"
    )
    assert design["round"] == 20
    assert design["implementation"]["commit"] == (
        "99279b0c4127a04cd2d9c530b67cecdbd32815a8"
    )
    assert design["model"]["candidate_id"] == (
        "parameter-matched-independent-side-outcome-mixture"
    )
    assert design["model"]["side_tower_mode"] == "independent"
    assert design["model"]["hidden_dim"] == 88
    assert design["model"]["residual_blocks"] == 2
    for section in (
        "data",
        "execution",
        "barrier_targets",
        "runtime_resources",
        "event_sampler",
        "training",
        "threshold_policy",
        "risk_profiles",
        "evaluation",
        "reserved_terminal",
    ):
        assert design[section] == predecessor[section]


def test_round21_revision1_is_historical_and_changes_only_ranking_mode() -> None:
    design, design_sha256 = load_outcome_mixture_design(DESIGN21, require_current=False)
    predecessor, _predecessor_sha256 = load_outcome_mixture_design(
        DESIGN20, require_current=False
    )

    assert design_sha256 == (
        "e097162b1fda42439e3528526b37405a3f0d843f3e9490fc258ecdca90da200a"
    )
    assert design["round"] == 21
    assert design["implementation"]["commit"] == (
        "439487491b70fb2e932e19088b38564b6e26ffee"
    )
    assert design["model"]["candidate_id"] == (
        "pairwise-ranked-independent-side-outcome-mixture"
    )
    assert design["model"]["ranking_loss_mode"] == "pairwise_net_return"
    assert set(design["model"]) == set(predecessor["model"]) | {"ranking_loss_mode"}
    for name, value in predecessor["model"].items():
        if name != "candidate_id":
            assert design["model"][name] == value
    for section in (
        "data",
        "execution",
        "barrier_targets",
        "runtime_resources",
        "event_sampler",
        "training",
        "threshold_policy",
        "risk_profiles",
        "evaluation",
        "reserved_terminal",
    ):
        assert design[section] == predecessor[section]


def test_round21_revision2_is_historical_and_preserves_economic_contract() -> None:
    design, design_sha256 = load_outcome_mixture_design(
        DESIGN21_V2, require_current=False
    )
    revision1, _revision1_sha256 = load_outcome_mixture_design(
        DESIGN21, require_current=False
    )

    assert design_sha256 == (
        "b0697ec5d4c7df8ec14ed1bc46add327c6f1b0162a372802f1affd5ebcb8cb4a"
    )
    assert design["round"] == 21
    assert design["design_revision"] == 2
    assert design["implementation"]["commit"] == (
        "29daa3ac5f88becd4e1f8e8ded6ef1cd2d4fad2d"
    )
    assert "CPU fallback" in design["predecessor_evidence"]["finding"]
    for section in (
        "data",
        "execution",
        "barrier_targets",
        "runtime_resources",
        "event_sampler",
        "model",
        "training",
        "threshold_policy",
        "risk_profiles",
        "evaluation",
        "reserved_terminal",
    ):
        assert design[section] == revision1[section]


def test_round21_revision3_is_current_and_preserves_economic_contract() -> None:
    design, design_sha256 = load_outcome_mixture_design(DESIGN21_V3)
    revision2, _revision2_sha256 = load_outcome_mixture_design(
        DESIGN21_V2, require_current=False
    )

    assert design_sha256 == (
        "afcebb4d1d079bb91755bb14da4ed8684af141bf829941443509b803bbe4b9eb"
    )
    assert design["round"] == 21
    assert design["design_revision"] == 3
    assert design["implementation"]["commit"] == (
        "53a3c8cce6f07998e89eae3622b1af8d71dc1073"
    )
    assert "nonstable Safetensors" in design["predecessor_evidence"]["finding"]
    for section in (
        "data",
        "execution",
        "barrier_targets",
        "runtime_resources",
        "event_sampler",
        "model",
        "training",
        "threshold_policy",
        "risk_profiles",
        "evaluation",
        "reserved_terminal",
    ):
        assert design[section] == revision2[section]


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
