from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs/model-research/action-value"
FAILURE = RESEARCH / "round-038-failure-analysis.json"
REGISTRY37 = RESEARCH / "consumed-periods-through-round-037.json"
REGISTRY38 = RESEARCH / "consumed-periods-through-round-038.json"
DESIGN = RESEARCH / "round-039-causal-refit-utility-ai-ablation-design.json"


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical_sha256(value: dict[str, object], field: str) -> str:
    canonical = dict(value)
    claimed = str(canonical.pop(field))
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    assert hashlib.sha256(encoded).hexdigest() == claimed
    return claimed


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_round38_failure_and_consumed_registry_are_hash_bound() -> None:
    failure = _read(FAILURE)
    registry37 = _read(REGISTRY37)
    registry38 = _read(REGISTRY38)

    assert _canonical_sha256(failure, "analysis_sha256") == (
        "f5b693fd00891ae8af9f93eec6925837c16c019aa246864ec9af29100496917c"
    )
    assert _canonical_sha256(registry38, "registry_sha256") == (
        "7596a73f6ff63699debe662ddd1dac4e7adf2c65e0c55c0232d202c2381fb84c"
    )
    assert registry38["records"][:-1] == registry37["records"]
    latest = registry38["records"][-1]
    assert latest["round"] == 38
    assert latest["windows"] == [
        {"start_date": "2022-01-01", "end_date": "2025-06-30"}
    ]
    assert latest["selection_confirmation_accessed"] is False
    assert latest["terminal_2026_accessed"] is False

    assert failure["activity_evidence"]["viability_trade_count_range"] == [313, 8515]
    assert failure["best_viability_candidate"]["trades"] == 789
    assert failure["best_viability_candidate"]["mean_net_bps"] < 0
    assert failure["calibration_decay_case"]["calibration"][
        "day_block_lower_95_mean_net_bps"
    ] > 0
    assert failure["calibration_decay_case"]["viability"]["mean_net_bps"] < 0
    assert failure["derivatives_feature_ablation"][
        "augmented_log_loss_improvements"
    ] == 0
    assert failure["trading_authority"] is False
    assert failure["profitability_claim"] is False


def test_round39_design_freezes_causal_refit_and_utility_ablation() -> None:
    design = _read(DESIGN)
    failure = _read(FAILURE)
    registry = _read(REGISTRY38)

    assert _canonical_sha256(design, "design_sha256") == (
        "cdfca0f55652737c10ed441ce88b5ab9a8e45499a24bd49a362609ee2e800a64"
    )
    assert design["schema_version"] == "causal-refit-utility-ai-ablation-design-v1"
    assert design["round"] == 39
    assert design["predecessor"]["failure_analysis_canonical_sha256"] == failure[
        "analysis_sha256"
    ]
    assert design["predecessor"]["failure_analysis_file_sha256"] == _file_sha256(
        FAILURE
    )
    assert design["governance"][
        "consumed_period_registry_canonical_sha256"
    ] == registry["registry_sha256"]
    assert design["governance"]["consumed_period_registry_file_sha256"] == (
        _file_sha256(REGISTRY38)
    )

    governance = design["governance"]
    for field in (
        "unregistered_hyperparameter_search_permitted",
        "selection_confirmation_access_permitted",
        "terminal_2026_access_permitted",
        "promotion_permitted",
        "trading_authority_permitted",
        "risk_gate_relaxation_permitted",
        "leverage_permitted",
        "maker_execution_assumption_permitted",
        "oracle_feature_or_runtime_label_use_permitted",
        "ai_can_create_reverse_or_increase_trades",
        "ai_model_selection_on_round39_evaluation_permitted",
        "profitability_target_override_permitted",
    ):
        assert governance[field] is False

    source = design["source_and_accounting_contract"]
    assert source["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert source["feature_set"] == "71 causal price_and_flow_features_only"
    assert source["derivatives_directional_features_permitted"] is False
    assert source["past_settled_funding_used_for_exact_cash_flow_accounting"] is True
    assert source["round_trip_execution_charge_bps"] == 12.0
    assert source["portfolio_capital_or_leverage_simulation_permitted"] is False

    walk = design["causal_walk_forward_contract"]
    assert walk["evaluation_months"] == [
        "2025-01",
        "2025-02",
        "2025-03",
        "2025-04",
        "2025-05",
        "2025-06",
    ]
    assert walk["explicit_first_schedule"]["training"] == [
        "2022-10-01",
        "2024-09-30",
    ]
    assert walk["explicit_last_schedule"]["training"] == [
        "2023-03-01",
        "2025-02-28",
    ]
    assert walk["full_refit_each_month"] is True
    assert walk["continued_training_or_booster_refit_permitted"] is False
    assert walk[
        "evaluation_month_outcomes_may_not_change_that_month_model_temperature_or_threshold"
    ] is True

    model = design["model_contract"]
    assert model["candidate_count"] == 4
    assert model["expected_model_artifact_count"] == 60
    assert {candidate["horizon_minutes"] for candidate in model["fixed_candidates"]} == {
        30,
        120,
    }
    assert {candidate["weighting"] for candidate in model["fixed_candidates"]} == {
        "equal",
        "bounded_economic_utility",
    }
    weighting = model["bounded_economic_utility_weighting"]
    assert weighting["weight_range"] == [1.0, 3.0]
    assert weighting["future_evaluation_utility_may_not_enter_weights"] is True

    thresholds = design["monthly_action_threshold_contract"]
    assert len(thresholds["maximum_action_probability_grid"]) == 5
    assert len(thresholds["direction_probability_margin_grid"]) == 4
    assert thresholds["minimum_trades_per_symbol"] == 5
    assert design["aggregate_ml_gate"]["minimum_nonoverlapping_trades_per_symbol"] == 45
    assert design["aggregate_ml_gate"][
        "maximum_single_month_fraction_of_positive_net_bps"
    ] == 0.50

    ai = design["ai_ablation_contract"]
    assert ai["models"] == ["qwen3:8b", "fino1:8b"]
    assert ai["batch_size"] == 12
    assert ai["maximum_cases"] == 543
    assert ai["ai_may_only_veto_or_reduce_risk"] is True
    assert ai["best_ai_model_may_not_be_selected_on_round39_evaluation"] is True
    assert design["claims"] == {
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }
