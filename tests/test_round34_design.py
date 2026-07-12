from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "action-value"
DESIGN = RESEARCH / "round-034-three-action-utility-design.json"
PREDECESSOR = RESEARCH / "round-033-selective-action-design.json"
FAILURE = RESEARCH / "round-033-failure-analysis.json"
REGISTRY = RESEARCH / "consumed-periods-through-round-033.json"
HISTORICAL_REGISTRY = RESEARCH / "consumed-periods-through-round-032.json"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_round34_design_is_hash_bound_to_rejection_and_consumed_registry() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    canonical = dict(design)
    claimed = canonical.pop("design_sha256")
    failure = json.loads(FAILURE.read_text(encoding="utf-8"))
    failure_canonical = dict(failure)
    failure_claimed = failure_canonical.pop("analysis_sha256")
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    registry_canonical = dict(registry)
    registry_claimed = registry_canonical.pop("registry_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert failure_claimed == _canonical_sha256(failure_canonical)
    assert registry_claimed == _canonical_sha256(registry_canonical)
    assert design["round"] == 34
    assert design["design_revision"] == 2
    assert design["supersedes"] == {
        "design_revision": 1,
        "design_sha256": (
            "0f8f3b074c5fd93727b0d4f60d82fa885420312af5c7d3debe3d10d12fa4846f"
        ),
        "file_sha256": (
            "a1ad2228bb1724307e09599d0cc4fb3d921cb4167913a35251e08c61ee9b6c52"
        ),
        "reason": (
            "Replace undeclared SciPy L-BFGS-B dependency with a deterministic "
            "projected Newton solver before implementation; all economic, data, "
            "model, and acceptance contracts remain unchanged."
        ),
    }
    assert design["predecessor"]["failure_analysis_file_sha256"] == _file_sha256(
        FAILURE
    )
    assert design["predecessor"]["failure_analysis_canonical_sha256"] == (
        failure_claimed
    )
    assert design["governance"]["consumed_period_registry_file_sha256"] == (
        _file_sha256(REGISTRY)
    )
    assert design["governance"][
        "consumed_period_registry_canonical_sha256"
    ] == registry_claimed
    assert registry["records"][-1]["round"] == 33
    assert registry["records"][-1]["outcome"] == "rejected"


def test_round34_registry_extends_without_mutating_round32_history() -> None:
    historical = json.loads(HISTORICAL_REGISTRY.read_text(encoding="utf-8"))
    current = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert _file_sha256(HISTORICAL_REGISTRY) == (
        "499a37db5b4c51de08c13baf2fd1e8090c6cd9889bc224c695c55788df02c1fb"
    )
    assert current["records"][:-1] == historical["records"]
    assert current["records"][-1]["round"] == 33


def test_round34_changes_only_the_frozen_joint_decision_architecture() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))

    for section in (
        "data",
        "execution",
        "barrier_targets",
        "event_sampler",
        "selection",
        "stage_evaluation_order",
        "runtime_resources",
    ):
        assert design[section] == predecessor[section]
    assert design["conditional_direction_confidence"] == predecessor[
        "conditional_direction_confidence"
    ]
    assert design["model"]["family"] == (
        "utility_weighted_symmetric_three_action_lightgbm_hurdle"
    )
    assert design["model"]["action_class_order"] == [
        "long",
        "abstain",
        "short",
    ]
    assert design["model"]["heads"] == [
        "three_action_probability",
        "shared_conditional_positive_magnitude",
        "shared_conditional_nonpositive_loss_magnitude",
        "shared_lower_quantile_0_10",
        "shared_upper_quantile_0_90",
    ]
    assert design["model"]["lightgbm"] == predecessor["model"]["lightgbm"]
    assert design["model"]["seeds"] == predecessor["model"]["seeds"]
    assert design["governance"]["variant_budget"] == 1
    assert design["governance"]["hyperparameter_search_permitted"] is False
    assert design["governance"]["risk_gate_relaxation_permitted"] is False


def test_round34_regret_weighting_and_calibration_are_bounded_and_symmetric() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    weighting = design["sample_weighting"]
    calibration = design["model"]["multiclass_calibration"]
    symmetry = design["model"]["symmetry"]

    assert weighting["regret_multiplier"].endswith("clipped_to_0_5_and_3_0")
    assert weighting["mirrored_multiclass_rows_share_half_event_weight"] is True
    assert weighting[
        "conditional_magnitude_and_quantile_heads_retain_unmodified_average_label_uniqueness_weight"
    ] is True
    assert calibration["log_temperature_bounds"] == [-4.0, 4.0]
    assert calibration["abstain_logit_bias_bounds"] == [-5.0, 5.0]
    assert calibration["optimizer"] == (
        "deterministic_projected_newton_in_inverse_temperature_and_abstain_bias"
    )
    assert calibration["maximum_iterations"] == 100
    assert calibration["gradient_tolerance"] == 1e-10
    assert calibration["hessian_diagonal_ridge"] == 1e-12
    assert calibration["long_short_bias_permitted"] is False
    assert "long_short_class_swap" in symmetry["prediction"]
    assert "preserves_long_short_mirror_equivariance" in symmetry["calibration"]


def test_round34_retains_fail_closed_financial_claims_and_primary_sources() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))

    assert design["claims"] == {
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "trading_authority": False,
        "leverage_applied": False,
    }
    assert design["acceptance_gates"]["economic"]["leverage_permitted"] is False
    assert design["acceptance_gates"]["calibration_architecture"] == {
        "minimum_opportunity_auc": 0.65,
        "minimum_conditional_direction_auc": 0.55,
        "maximum_multiclass_log_loss_to_class_prior_ratio": 1.0,
        "minimum_selected_top_100_mean_stress_net_bps": 0.0,
        "minimum_selected_top_500_mean_stress_net_bps": 0.0,
    }
    assert len(design["research_basis"]) >= 6
    assert all(
        item["url"].startswith("https://") for item in design["research_basis"]
    )
