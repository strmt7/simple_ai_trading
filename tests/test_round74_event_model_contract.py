from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404

from simple_ai_trading.impact_absorption_ai_protocol import (
    ROUND74_AI_MODEL_MANIFEST_SCHEMA_VERSION,
    ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_ai_bridge import ROUND74_AI_RECENT_BLOCK_EVENTS
from simple_ai_trading.impact_absorption_ai_uplift import (
    ROUND74_AI_EXECUTION_REPLAY_EVIDENCE_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_ai_execution_replay import (
    ROUND74_AI_EXECUTION_REPLAY_PLAN_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_ai_worker import (
    ROUND74_AI_WORKER_ENVELOPE_SCHEMA_VERSION,
    ROUND74_AI_WORKER_RESULT_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_cohort import (
    ROUND74_EVENT_COHORT_BINDING_SCHEMA_VERSION,
    ROUND74_EVENT_COHORT_PLAN_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_calibration import (
    ROUND74_TUNING_SUBPARTITION_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_action_policy import (
    ROUND74_ACTION_DEFAULT_PROFILE,
    ROUND74_ACTION_HORIZONS_SECONDS,
    ROUND74_ACTION_PROFILES,
)
from simple_ai_trading.impact_absorption_event_scaling import (
    ROUND74_EVENT_SCALER_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_sealed_evaluation import (
    ROUND74_SEALED_AI_MODEL_COUNT,
    ROUND74_SEALED_BOOTSTRAP_DRAWS,
    ROUND74_SEALED_FAMILYWISE_ALPHA,
    ROUND74_SEALED_QUALIFICATION_CONFIGURATION_COUNT,
    ROUND74_TARGET_FREE_INFERENCE_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_sealed_ledger import (
    ROUND74_SEALED_CLAIM_SCHEMA_VERSION,
    ROUND74_SEALED_DATASET_IDENTITY_SCHEMA_VERSION,
    ROUND74_SEALED_LEDGER_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_DEFAULT_MAX_WINDOW_GAP_NS,
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_STATE_HALF_LIVES_SECONDS,
    ROUND74_EVENT_SYMBOLS,
)
from simple_ai_trading.impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_EVIDENCE_SCHEMA_VERSION,
    ROUND74_EVENT_TARGET_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_training import (
    ROUND74_COMPLEXITY_PROMOTION_COMPARISON_COUNT,
    ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION,
    ROUND74_EVENT_TRAINING_DEFAULT_SEEDS,
)
from simple_ai_trading.impact_absorption_event_model import (
    ROUND74_EVENT_ATTENTION_HEADS,
    ROUND74_EVENT_ATTENTION_HIDDEN_CHANNELS,
    ROUND74_EVENT_ATTENTION_LAYERS,
    ROUND74_EVENT_MODEL_CANDIDATES,
    ROUND74_EVENT_MODEL_SCHEMA_VERSION,
    ROUND74_EVENT_TCN_DILATIONS,
    ROUND74_EVENT_TCN_RECEPTIVE_FIELD,
)
from simple_ai_trading.impact_absorption_event_evidence import (
    ROUND74_BINANCE_CLOCK_PROBE_SCHEMA_VERSION,
    ROUND74_BINANCE_EVIDENCE_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_execution_evidence import (
    ROUND74_EXECUTION_CALIBRATION_MAXIMUM_BOOK_AGE_NS,
    ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL,
    ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL_SIDE,
    ROUND74_EXECUTION_CALIBRATION_QUANTILE,
    ROUND74_EXECUTION_CALIBRATION_QUANTILE_CONFIDENCE,
)
from simple_ai_trading.impact_absorption_exchange_info_evidence import (
    ROUND74_EXCHANGE_INFO_EVIDENCE_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_target_assembly import (
    ROUND74_SOURCE_TARGET_ASSEMBLY_SCHEMA_VERSION,
)

REPOSITORY = Path(__file__).resolve().parents[1]
RESEARCH = REPOSITORY / "docs" / "model-research" / "action-value"
DESIGN_PATH = RESEARCH / "round-074-event-sequence-model-design-v61.json"
TUNING_SEPARATION_DESIGN_PATH = (
    RESEARCH / "round-074-event-sequence-model-design-v62.json"
)
MODEL_INTEGRATION_DESIGN_PATH = (
    RESEARCH / "round-074-event-sequence-model-design-v63.json"
)
DURATION_NORMALIZED_DESIGN_PATH = (
    RESEARCH / "round-074-event-sequence-model-design-v76.json"
)
SEGMENTED_DIRECTML_PREFLIGHT_PATH = (
    RESEARCH / "round-074-segmented-training-directml-preflight-v1-2026-07-28.json"
)
SEGMENTED_DIRECTML_DESIGN_PATH = (
    RESEARCH / "round-074-event-sequence-model-design-v77.json"
)
SEGMENTED_POLICY_POPULATION_DESIGN_PATH = (
    RESEARCH / "round-074-event-sequence-model-design-v78.json"
)
SHARED_CAPITAL_DESIGN_PATH = RESEARCH / "round-074-event-sequence-model-design-v79.json"
OPEN_RISK_DRAWDOWN_DESIGN_PATH = (
    RESEARCH / "round-074-event-sequence-model-design-v80.json"
)
RISK_TAIL_CALIBRATION_DESIGN_PATH = (
    RESEARCH / "round-074-event-sequence-model-design-v81.json"
)
WORST_SYMBOL_TAIL_DESIGN_PATH = (
    RESEARCH / "round-074-event-sequence-model-design-v82.json"
)
SUBGROUP_PROMOTION_DESIGN_PATH = (
    RESEARCH / "round-074-event-sequence-model-design-v83.json"
)
DIRECTML_PATH = RESEARCH / "round-074-event-model-directml-preflight-2026-07-26.json"
REPLAY_PATH = RESEARCH / "round-074-event-sequence-host-replay-2026-07-26.json"
AI_RUNTIME_PREFLIGHT_PATH = (
    RESEARCH / "round-074-local-ai-runtime-preflight-v1-2026-07-27.json"
)
TRAINING_PATH = (
    RESEARCH
    / "round-074-event-training-directml-preflight-gap-reset-v11-2026-07-27.json"
)
CALIBRATION_PATH = (
    RESEARCH
    / "round-074-run-balanced-temperature-calibration-directml-preflight-2026-07-27.json"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _load_hash_bound(path: Path, field: str) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            assert key not in value
            value[key] = item
        return value

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )
    claimed = value.pop(field)
    assert claimed == _canonical_sha256(value)
    value[field] = claimed
    return value


def _file_sha256(relative_path: str) -> str:
    design = json.loads(DESIGN_PATH.read_text(encoding="ascii"))
    return _file_sha256_at(design["implementation_git_commit"], relative_path)


def _file_sha256_at(commit: str, relative_path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    payload = completed.stdout
    canonical = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def test_round74_duration_normalized_design_is_hash_bound_and_nonfinancial() -> None:
    design = _load_hash_bound(DURATION_NORMALIZED_DESIGN_PATH, "design_sha256")
    source = design["source_binding"]
    implementation = design["implemented_contract"]
    authority = design["authority"]

    assert design["schema_version"] == "round-074-event-sequence-model-design-v76"
    assert design["base_design"]["design_sha256"] == (
        "acf0c4be3a97a9d2fc08bc04be71ca5ebcc8ae49f51d4390467a67d6585c0dc2"
    )
    for path, claimed in source["files"].items():
        assert claimed == _file_sha256_at(design["implementation_git_commit"], path)
    assert source["partition_schema_version"] == "round-074-run-partition-v5"
    assert source["training_schema_version"] == "round-074-event-training-v18"
    assert implementation["short_epoch_cycling_permitted"] is False
    assert (
        implementation["target_feature_model_activity_used_for_quota_or_rank"] is False
    )
    assert implementation["cross_epoch_state_feature_or_target_permitted"] is False
    assert design["verification"]["focused_tests_passed"] == 32
    assert design["verification"]["real_cohort_training_performed"] is False
    assert design["verification"]["financial_backtest_performed"] is False
    assert authority["financial_edge_tested"] is False
    assert authority["profitability_claim"] is False
    assert authority["ai_uplift_claim"] is False
    assert authority["live_trading_authority"] is False


def test_round74_segmented_directml_preflight_is_bounded_and_hash_bound() -> None:
    preflight = _load_hash_bound(
        SEGMENTED_DIRECTML_PREFLIGHT_PATH,
        "artifact_sha256",
    )
    design = _load_hash_bound(
        SEGMENTED_DIRECTML_DESIGN_PATH,
        "design_sha256",
    )

    assert design["base_design"]["design_sha256"] == (
        "72e4fe99a52545e3f85756ccebd585f1ddb5cfebdc1ca7b7a11d9b9bb7dfa7a6"
    )
    assert (
        design["directml_preflight"]["artifact_sha256"]
        == (preflight["artifact_sha256"])
    )
    assert preflight["environment"]["directml_device_name"] == ("AMD Radeon RX 9070 XT")
    assert (
        preflight["segmented_training_smoke"]["policy_train_seal_reload_passed"] is True
    )
    assert preflight["segmented_training_smoke"]["cpu_fallback_warning_count"] == 0
    assert preflight["authority"]["representative_market_data_used"] is False
    assert preflight["authority"]["financial_edge_tested"] is False
    assert design["evidence_boundary"]["synthetic_data_only"] is True
    assert design["authority"]["profitability_claim"] is False
    assert design["authority"]["live_trading_authority"] is False


def test_round74_segmented_policy_population_is_consistent_and_source_bound() -> None:
    previous = _load_hash_bound(
        SEGMENTED_DIRECTML_DESIGN_PATH,
        "design_sha256",
    )
    design = _load_hash_bound(
        SEGMENTED_POLICY_POPULATION_DESIGN_PATH,
        "design_sha256",
    )
    source = design["source_binding"]
    population = design["segmented_optimization_population"]
    compatibility = design["legacy_compatibility"]
    authority = design["authority"]
    commit = str(design["implementation_git_commit"])

    assert design["schema_version"] == "round-074-event-sequence-model-design-v78"
    assert design["base_design"]["design_sha256"] == previous["design_sha256"]
    assert design["base_design"]["unchanged_except_declared_delta"] is True
    for prefix in ("calibration", "action_policy", "development_operator"):
        assert source[f"{prefix}_sha256"] == _file_sha256_at(
            commit,
            source[f"{prefix}_path"],
        )
    assert source["calibration_schema_version"] == (
        "round-074-temperature-calibration-v3"
    )
    assert source["action_policy_schema_version"] == ("round-074-action-policy-v9")
    assert source["development_operator_schema_version"] == (
        "round-074-development-policy-operator-v3"
    )
    assert source["development_bundle_schema_version"] == (
        "round-074-development-policy-bundle-v4"
    )
    assert population["unit"] == "eligible_target"
    assert population["precondition"] == (
        "duration_normalized_target_blind_window_quota"
    )
    assert population["short_epoch_equal_vote_permitted"] is False
    assert population["target_value_or_model_output_used_for_window_quota"] is False
    assert population["test_role_used"] is False
    assert (
        compatibility[
            "capture_run_population_remains_default_outside_segmented_execution"
        ]
        is True
    )
    assert (
        compatibility["legacy_equal_run_temperature_and_threshold_behavior_changed"]
        is False
    )
    assert design["evidence_boundary"]["synthetic_data_only"] is True
    assert design["evidence_boundary"]["real_cohort_data_used"] is False
    assert design["evidence_boundary"]["financial_backtest_performed"] is False
    assert authority["segmented_development_population_consistent"] is True
    assert authority["financial_edge_tested"] is False
    assert authority["profitability_claim"] is False
    assert authority["ai_uplift_claim"] is False
    assert authority["paper_trading_authority"] is False
    assert authority["testnet_trading_authority"] is False
    assert authority["live_trading_authority"] is False


def test_round74_shared_capital_replay_is_bounded_and_source_bound() -> None:
    previous = _load_hash_bound(
        SEGMENTED_POLICY_POPULATION_DESIGN_PATH,
        "design_sha256",
    )
    design = _load_hash_bound(SHARED_CAPITAL_DESIGN_PATH, "design_sha256")
    source = design["source_binding"]
    capital = design["shared_capital_contract"]
    authority = design["authority"]
    commit = str(design["implementation_git_commit"])

    assert design["schema_version"] == "round-074-event-sequence-model-design-v79"
    assert design["base_design"]["design_sha256"] == previous["design_sha256"]
    for prefix in ("action_policy", "ai_uplift", "sealed_evaluation"):
        assert source[f"{prefix}_sha256"] == _file_sha256_at(
            commit,
            source[f"{prefix}_path"],
        )
    assert source["action_policy_schema_version"] == ("round-074-action-policy-v10")
    assert capital["execution_mode"] == "unlevered_research_only"
    assert capital["symbol_sleeves"] == len(ROUND74_EVENT_SYMBOLS)
    assert capital["position_capital_fraction"] == (1.0 / len(ROUND74_EVENT_SYMBOLS))
    assert capital["maximum_concurrent_positions"] == len(ROUND74_EVENT_SYMBOLS)
    assert capital["maximum_concurrent_gross_capital_fraction"] == 1.0
    assert capital["same_symbol_overlap_permitted"] is False
    assert capital["cross_symbol_capital_reuse_permitted"] is False
    assert (
        capital["baseline_exact_replay_and_ai_overlay_use_same_capital_scale"] is True
    )
    assert capital["trace_metrics_recomputed_from_realized_trade_vectors"] is True
    assert capital["independently_tampered_summary_metrics_permitted"] is False
    assert capital["leverage_applied"] is False
    assert design["evidence_boundary"]["real_cohort_data_used"] is False
    assert design["evidence_boundary"]["financial_backtest_performed"] is False
    assert authority["shared_capital_replay_implementation"] is True
    assert authority["financial_edge_tested"] is False
    assert authority["profitability_claim"] is False
    assert authority["ai_uplift_claim"] is False
    assert authority["paper_trading_authority"] is False
    assert authority["testnet_trading_authority"] is False
    assert authority["live_trading_authority"] is False


def test_round74_open_risk_drawdown_is_conservative_and_source_bound() -> None:
    previous = _load_hash_bound(SHARED_CAPITAL_DESIGN_PATH, "design_sha256")
    design = _load_hash_bound(OPEN_RISK_DRAWDOWN_DESIGN_PATH, "design_sha256")
    source = design["source_binding"]
    contract = design["conservative_drawdown_contract"]
    authority = design["authority"]
    commit = str(design["implementation_git_commit"])

    assert design["schema_version"] == "round-074-event-sequence-model-design-v80"
    assert design["base_design"]["design_sha256"] == previous["design_sha256"]
    assert design["base_design"]["unchanged_except_declared_delta"] is True
    for prefix in (
        "financial_metrics",
        "action_policy",
        "ai_uplift",
        "sealed_evaluation",
    ):
        assert source[f"{prefix}_sha256"] == _file_sha256_at(
            commit,
            source[f"{prefix}_path"],
        )
    assert source["financial_metrics_schema_version"] == (
        "round-074-realized-payoff-metrics-v2"
    )
    assert source["action_policy_schema_version"] == "round-074-action-policy-v11"
    assert source["ai_uplift_schema_version"] == ("round-074-ai-uplift-development-v8")
    assert source["sealed_evaluation_schema_version"] == (
        "round-074-sealed-evaluation-v11"
    )
    assert contract["capital_basis"] == (
        "shared_unlevered_three_symbol_sleeves_from_v79"
    )
    assert contract["position_capital_fraction"] == (1.0 / len(ROUND74_EVENT_SYMBOLS))
    assert contract["reported_maximum_drawdown_is_at_least_realized_drawdown"]
    assert contract[
        "reported_maximum_drawdown_is_at_least_maximum_concurrent_adverse_excursion"
    ]
    assert contract["open_position_interval"] == (
        "half_open_entry_inclusive_exit_exclusive"
    )
    assert contract["exits_processed_before_entries_at_identical_timestamp"]
    assert contract["realized_equity_carried_across_capture_runs"]
    assert not contract["open_position_state_carried_across_capture_runs"]
    assert contract["ai_veto_or_reduction_uses_same_capital_scale_as_baseline"]
    assert not contract["leverage_applied"]
    assert not contract["mark_to_market_equity_curve_reconstructed"]
    assert not contract["individual_mae_minima_assumed_simultaneous"]
    assert design["evidence_boundary"]["synthetic_data_only"]
    assert not design["evidence_boundary"]["real_cohort_data_used"]
    assert not design["evidence_boundary"]["financial_backtest_performed"]
    assert authority["conservative_open_risk_drawdown_implementation"]
    assert not authority["financial_edge_tested"]
    assert not authority["profitability_claim"]
    assert not authority["drawdown_performance_claim"]
    assert not authority["ai_uplift_claim"]
    assert not authority["paper_trading_authority"]
    assert not authority["testnet_trading_authority"]
    assert not authority["live_trading_authority"]


def test_round74_risk_tail_calibration_is_frozen_conservative_and_source_bound() -> (
    None
):
    previous = _load_hash_bound(OPEN_RISK_DRAWDOWN_DESIGN_PATH, "design_sha256")
    design = _load_hash_bound(RISK_TAIL_CALIBRATION_DESIGN_PATH, "design_sha256")
    source = design["source_binding"]
    contract = design["held_out_risk_tail_calibration_contract"]
    parity = design["ml_ai_parity_contract"]
    authority = design["authority"]
    commit = str(design["implementation_git_commit"])

    assert design["schema_version"] == "round-074-event-sequence-model-design-v81"
    assert design["base_design"]["design_sha256"] == previous["design_sha256"]
    assert design["base_design"]["unchanged_except_declared_delta"]
    for prefix in (
        "calibration",
        "action_policy",
        "ai_bridge",
        "sealed_evaluation",
        "development_operator",
    ):
        assert source[f"{prefix}_sha256"] == _file_sha256_at(
            commit,
            source[f"{prefix}_path"],
        )
    assert source["probability_calibration_schema_version"] == (
        "round-074-temperature-calibration-v4"
    )
    assert source["risk_quantile_calibration_schema_version"] == (
        "round-074-risk-quantile-calibration-v1"
    )
    assert source["action_policy_schema_version"] == "round-074-action-policy-v12"
    assert source["ai_bridge_schema_version"] == "round-074-ai-bridge-v5"
    assert source["sealed_evaluation_schema_version"] == (
        "round-074-sealed-evaluation-v12"
    )
    assert source["development_operator_schema_version"] == (
        "round-074-development-policy-operator-v4"
    )
    assert source["development_bundle_schema_version"] == (
        "round-074-development-policy-bundle-v5"
    )
    assert contract["calibration_role"] == (
        "six_chronological_calibration_capture_runs_only"
    )
    assert contract["payoff_lower_nominal_quantiles"] == [0.1, 0.25]
    assert contract["maximum_adverse_excursion_upper_nominal_quantile"] == 0.9
    assert contract["payoff_lower_offsets_are_nonnegative"]
    assert contract["maximum_adverse_excursion_upper_offsets_are_nonnegative"]
    assert contract["quantile_monotonicity_revalidated_after_application"]
    assert contract["calibration_sample_nominal_coverage_fail_closed"]
    assert not contract["model_selection_rows_reused"]
    assert not contract["policy_selection_rows_reused"]
    assert not contract["sealed_test_rows_accessed"]
    assert not contract["symbol_conditional_coverage_established"]
    assert not contract["exchangeability_assumed_or_established"]
    assert not contract["future_coverage_claim"]
    assert not contract["online_adaptive_calibration_enabled"]
    assert contract["frozen_pretest_policy_preserved"]
    assert parity["action_policy_uses_frozen_risk_tail_offsets"]
    assert parity["sealed_predictive_metrics_use_frozen_risk_tail_offsets"]
    assert parity["ai_review_uses_shared_probability_calibration_function"]
    assert parity["ai_review_uses_shared_risk_tail_calibration_function"]
    assert not parity["ai_review_receives_raw_uncalibrated_model_tails"]
    assert parity["ai_review_calibration_sha256_binds_nested_risk_tail_calibration"]
    assert parity["ai_may_only_veto_or_reduce"]
    assert not parity["ai_may_create_or_expand_a_position"]
    assert design["evidence_boundary"]["synthetic_data_only"]
    assert design["evidence_boundary"][
        "real_prospective_cohort_capture_active_but_not_used"
    ]
    assert not design["evidence_boundary"]["financial_backtest_performed"]
    assert not design["evidence_boundary"]["profitability_or_edge_inference_permitted"]
    assert not design["evidence_boundary"]["future_coverage_inference_permitted"]
    assert not design["evidence_boundary"]["ai_model_evaluated"]
    assert authority["held_out_nonnegative_risk_tail_calibration_implementation"]
    assert authority["calibration_action_ai_sealed_parity_implementation"]
    assert not authority["representative_market_evidence"]
    assert not authority["symbol_conditional_risk_coverage"]
    assert not authority["financial_edge_tested"]
    assert not authority["profitability_claim"]
    assert not authority["drawdown_performance_claim"]
    assert not authority["ai_uplift_claim"]
    assert not authority["paper_trading_authority"]
    assert not authority["testnet_trading_authority"]
    assert not authority["live_trading_authority"]


def test_round74_worst_symbol_tail_calibration_is_complete_and_source_bound() -> None:
    previous = _load_hash_bound(RISK_TAIL_CALIBRATION_DESIGN_PATH, "design_sha256")
    design = _load_hash_bound(WORST_SYMBOL_TAIL_DESIGN_PATH, "design_sha256")
    source = design["source_binding"]
    contract = design["worst_run_symbol_tail_contract"]
    authority = design["authority"]
    commit = str(design["implementation_git_commit"])

    assert design["schema_version"] == "round-074-event-sequence-model-design-v82"
    assert design["base_design"]["design_sha256"] == previous["design_sha256"]
    assert design["base_design"]["unchanged_except_declared_delta"]
    for prefix in ("calibration", "development_operator"):
        assert source[f"{prefix}_sha256"] == _file_sha256_at(
            commit,
            source[f"{prefix}_path"],
        )
    assert source["probability_calibration_schema_version"] == (
        "round-074-temperature-calibration-v4"
    )
    assert source["risk_quantile_calibration_schema_version"] == (
        "round-074-risk-quantile-calibration-v2"
    )
    assert source["risk_quantile_prior_schema_version"] == (
        "round-074-risk-quantile-calibration-v1"
    )
    assert source["development_operator_schema_version"] == (
        "round-074-development-policy-operator-v4"
    )
    assert contract["symbols"] == list(ROUND74_EVENT_SYMBOLS)
    assert contract["capture_run_count"] == 6
    assert contract["required_group_count_per_horizon_side"] == (
        6 * len(ROUND74_EVENT_SYMBOLS)
    )
    assert contract["capture_run_grouping"] == "capture_run_x_symbol"
    assert contract["missing_eligible_group_fails_closed"]
    assert contract["capture_run_coverage_reporting"] == (
        "minimum_over_run_symbol_groups"
    )
    assert contract["eligible_target_complete_run_symbol_support_required"]
    assert not contract["runtime_symbol_router_required"]
    assert not contract["one_symbol_can_be_averaged_away_in_capture_run_mode"]
    assert contract["all_offsets_nonnegative"]
    assert contract["calibration_sample_nominal_coverage_fails_closed"]
    assert contract["prior_v1_payload_reload_supported"]
    assert not contract["sealed_test_rows_accessed"]
    assert not contract["exchangeability_assumed_or_established"]
    assert not contract["symbol_conditional_future_coverage_claim"]
    assert not contract["profitability_or_edge_claim"]
    assert design["evidence_boundary"]["synthetic_data_only"]
    assert design["evidence_boundary"][
        "real_prospective_cohort_capture_active_but_not_used"
    ]
    assert not design["evidence_boundary"]["financial_backtest_performed"]
    assert not design["evidence_boundary"]["profitability_or_edge_inference_permitted"]
    assert not design["evidence_boundary"]["future_coverage_inference_permitted"]
    assert not design["evidence_boundary"]["ai_model_evaluated"]
    assert authority["worst_run_symbol_tail_calibration_implementation"]
    assert authority["backward_compatible_v1_payload_loader"]
    assert not authority["representative_market_evidence"]
    assert not authority["symbol_conditional_future_coverage"]
    assert not authority["financial_edge_tested"]
    assert not authority["profitability_claim"]
    assert not authority["drawdown_performance_claim"]
    assert not authority["ai_uplift_claim"]
    assert not authority["paper_trading_authority"]
    assert not authority["testnet_trading_authority"]
    assert not authority["live_trading_authority"]


def test_round74_model_promotion_rejects_hidden_subgroup_degradation() -> None:
    previous = _load_hash_bound(WORST_SYMBOL_TAIL_DESIGN_PATH, "design_sha256")
    design = _load_hash_bound(SUBGROUP_PROMOTION_DESIGN_PATH, "design_sha256")
    source = design["source_binding"]
    contract = design["run_symbol_horizon_promotion_contract"]
    verification = design["verification"]
    authority = design["authority"]
    commit = str(design["implementation_git_commit"])

    assert design["schema_version"] == "round-074-event-sequence-model-design-v83"
    assert design["base_design"]["design_sha256"] == previous["design_sha256"]
    assert design["base_design"]["unchanged_except_declared_delta"]
    for prefix in ("event_training", "test"):
        assert source[f"{prefix}_sha256"] == _file_sha256_at(
            commit,
            source[f"{prefix}_path"],
        )
    assert source["event_training_schema_version"] == ("round-074-event-training-v19")
    assert source["pretest_policy_schema_version"] == (
        "round-074-event-pretest-policy-v18"
    )
    assert contract["candidate_panel"] == list(ROUND74_EVENT_MODEL_CANDIDATES)
    assert contract["model_selection_capture_runs"] == 12
    assert contract["symbols"] == list(ROUND74_EVENT_SYMBOLS)
    assert contract["horizons_seconds"] == list(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
    assert contract["required_cohort_group_count"] == (
        12 * len(ROUND74_EVENT_SYMBOLS) * len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
    )
    assert contract["long_and_short_sides_pooled_within_each_group"]
    assert contract["proper_loss_components_and_weights_match_training_objective"]
    assert contract["one_additional_bounded_inference_pass_per_candidate"]
    assert contract["minibatch_output_reused_across_symbol_horizon_groups"]
    assert contract["complete_symbol_panel_required_outside_preflight"]
    assert contract["missing_or_ineligible_required_group_fails_closed"]
    assert contract["group_keys_must_match_across_candidates"]
    assert contract["all_group_losses_must_be_finite"]
    assert contract["promotion_requires_complete_12_run_panel"]
    assert contract["promotion_requires_mean_run_loss_improvement_above_floor"]
    assert contract["promotion_requires_every_capture_run_noninferior"]
    assert contract["promotion_requires_every_run_symbol_horizon_group_noninferior"]
    assert contract["worst_degraded_group_recorded"]
    assert not contract["fixed_architecture_mode_changed_by_this_delta"]
    assert not contract["backtest_pnl_used_for_architecture_selection"]
    assert not contract["sealed_test_rows_accessed"]
    assert not contract["ai_output_used_for_architecture_selection"]
    assert verification["focused_training_tests_passed"] == 18
    assert (
        verification["downstream_policy_runtime_representation_contract_tests_passed"]
        == 44
    )
    assert verification["adversarial_hidden_sol_horizon_degradation_test_passed"]
    assert verification["train_seal_reload_test_passed"]
    assert verification["directml_host_probe"]["backend_kind"] == "directml"
    assert verification["directml_host_probe"]["group_count"] == (
        len(ROUND74_EVENT_SYMBOLS) * len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
    )
    assert verification["directml_host_probe"]["all_group_losses_finite"]
    assert not verification["full_repository_suite_run"]
    assert design["evidence_boundary"]["synthetic_data_only"]
    assert not design["evidence_boundary"]["financial_backtest_performed"]
    assert not design["evidence_boundary"]["profitability_or_edge_inference_permitted"]
    assert authority["run_symbol_horizon_proper_loss_gate_implementation"]
    assert authority["directml_group_evaluation_preflight"]
    assert not authority["representative_market_evidence"]
    assert not authority["financial_edge_tested"]
    assert not authority["profitability_claim"]
    assert not authority["drawdown_performance_claim"]
    assert not authority["ai_uplift_claim"]
    assert not authority["paper_trading_authority"]
    assert not authority["testnet_trading_authority"]
    assert not authority["live_trading_authority"]


def test_round74_tuning_role_correction_is_disjoint_and_source_bound() -> None:
    previous = _load_hash_bound(DESIGN_PATH, "design_sha256")
    design = _load_hash_bound(TUNING_SEPARATION_DESIGN_PATH, "design_sha256")
    source = design["source_binding"]
    dataset = design["dataset_assembly_contract"]
    training = design["development_training_contract"]
    correction = design["tuning_role_correction"]
    authority = design["authority"]
    commit = str(design["implementation_git_commit"])

    assert design["schema_version"] == "round-074-event-sequence-model-design-v62"
    assert design["supersedes_design_sha256"] == previous["design_sha256"]
    assert source["event_model_operator_schema_version"] == (
        "round-074-event-model-operator-v2"
    )
    assert source["event_model_operator_sha256"] == _file_sha256_at(
        commit,
        source["event_model_operator_path"],
    )
    assert source["tuning_role_contract_generator_sha256"] == _file_sha256_at(
        commit,
        source["tuning_role_contract_generator_path"],
    )
    assert dataset["disjoint_tuning_role_adapter_implemented_now"] is True
    assert dataset["model_selection_receives_only_first_12_tuning_runs"] is True
    assert dataset["probability_calibration_receives_only_next_6_tuning_runs"] is True
    assert dataset["action_policy_selection_receives_only_final_6_tuning_runs"] is True
    assert dataset["test_role_accessed_by_tuning_assignment"] is False
    assert training["complexity_promotion_required_paired_capture_runs"] == 12
    assert training["candidate_training_may_receive_calibration_runs"] is False
    assert training["candidate_training_may_receive_policy_selection_runs"] is False
    assert (
        training[
            "model_selection_probability_calibration_or_policy_run_reuse_permitted"
        ]
        is False
    )
    assert correction["raw_capture_contract_changed"] is False
    assert correction["cohort_schedule_changed"] is False
    assert correction[
        "selected_from_market_targets_model_results_or_profitability"
    ] is (False)
    assert correction["representative_market_training_completed"] is False
    assert correction["financial_edge_established"] is False
    assert correction["profitability_claim"] is False
    assert authority["disjoint_tuning_role_adapter_implementation"] is True
    assert authority["model_selection"] is False
    assert authority["profitability_claim"] is False


def test_round74_model_integration_is_bounded_target_blind_and_source_bound() -> None:
    previous = _load_hash_bound(
        TUNING_SEPARATION_DESIGN_PATH,
        "design_sha256",
    )
    design = _load_hash_bound(MODEL_INTEGRATION_DESIGN_PATH, "design_sha256")
    commit = str(design["implementation_git_commit"])
    source = design["source_binding"]
    sampling = design["representative_window_selection_contract"]
    device = design["device_execution_contract"]
    memory = design["development_memory_contract"]
    correction = design["model_integration_correction"]

    assert design["schema_version"] == "round-074-event-sequence-model-design-v63"
    assert design["supersedes_design_sha256"] == previous["design_sha256"]
    assert source["event_model_operator_schema_version"] == (
        "round-074-event-model-operator-v3"
    )
    assert source["event_training_schema_version"] == "round-074-event-training-v13"
    assert source["pretest_policy_schema_version"] == (
        "round-074-event-pretest-policy-v12"
    )
    for label in (
        "event_model_operator",
        "event_model",
        "event_training",
        "model_integration_contract_generator",
    ):
        assert source[f"{label}_sha256"] == _file_sha256_at(
            commit,
            source[f"{label}_path"],
        )
    assert sampling["schema_version"] == "round-074-target-blind-window-selection-v1"
    assert sampling["windows_per_capture_run"] == 768
    assert sampling["windows_per_symbol_per_capture_run"] == 256
    assert sampling["realized_targets_model_outputs_or_profitability_used"] is False
    assert sampling["underfilled_symbol_or_stratum_policy"] == "reject"
    assert device["training_capture_runs"] == 120
    assert device["model_selection_capture_runs"] == 12
    assert device["probability_calibration_capture_runs"] == 6
    assert device["action_policy_selection_capture_runs"] == 6
    assert device["default_device_run_group_size"] == 8
    assert device["optimizer_forwards_per_step_at_default_group_size"] == 15
    assert device["optimizer_forward_reduction_fraction"] == 0.875
    assert device["equal_capture_run_loss_normalization_preserved"] is True
    assert memory["representative_windows"] == 110592
    assert memory["feature_tensor_upper_bound_bytes"] == 3737124864
    assert memory["feature_tensor_upper_bound_gib"] == 3.48046875
    assert correction["representative_market_training_completed"] is False
    assert correction["financial_edge_established"] is False
    assert correction["profitability_claim"] is False


def test_round74_event_model_design_is_source_bound_and_causal() -> None:
    design = _load_hash_bound(DESIGN_PATH, "design_sha256")
    source = design["source_binding"]

    assert design["file_sha256_normalization"] == (
        "text_bytes_crlf_and_cr_normalized_to_lf_before_sha256"
    )
    assert source["file_sha256_normalization"] == (design["file_sha256_normalization"])
    assert source["event_sequence_sha256"] == _file_sha256(
        source["event_sequence_path"]
    )
    assert source["event_model_sha256"] == _file_sha256(source["event_model_path"])
    assert source["event_scaler_sha256"] == _file_sha256(source["event_scaler_path"])
    assert source["event_target_sha256"] == _file_sha256(source["event_target_path"])
    assert source["event_target_source_evidence_sha256"] == _file_sha256(
        source["event_target_source_evidence_path"]
    )
    assert (
        source["event_target_source_evidence_schema_version"]
        == ROUND74_BINANCE_EVIDENCE_SCHEMA_VERSION
    )
    assert (
        source["event_target_clock_probe_schema_version"]
        == ROUND74_BINANCE_CLOCK_PROBE_SCHEMA_VERSION
    )
    assert source["event_exchange_info_evidence_sha256"] == _file_sha256(
        source["event_exchange_info_evidence_path"]
    )
    assert (
        source["event_exchange_info_evidence_schema_version"]
        == ROUND74_EXCHANGE_INFO_EVIDENCE_SCHEMA_VERSION
    )
    assert source["event_target_assembly_sha256"] == _file_sha256(
        source["event_target_assembly_path"]
    )
    assert (
        source["event_target_assembly_schema_version"]
        == ROUND74_SOURCE_TARGET_ASSEMBLY_SCHEMA_VERSION
    )
    assert source["event_model_operator_sha256"] == _file_sha256(
        source["event_model_operator_path"]
    )
    assert source["event_model_operator_schema_version"] == (
        "round-074-event-model-operator-v1"
    )
    assert source["event_execution_calibration_sha256"] == _file_sha256(
        source["event_execution_calibration_path"]
    )
    assert (
        source["event_execution_calibration_schema_version"]
        == "round-074-execution-calibration-source-v1"
    )
    assert source["event_dataset_sha256"] == _file_sha256(source["event_dataset_path"])
    assert source["event_action_policy_sha256"] == _file_sha256(
        source["event_action_policy_path"]
    )
    assert source["ai_uplift_evaluator_sha256"] == _file_sha256(
        source["ai_uplift_evaluator_path"]
    )
    assert source["ai_execution_replay_sha256"] == _file_sha256(
        source["ai_execution_replay_path"]
    )
    assert source["sealed_ledger_sha256"] == _file_sha256(source["sealed_ledger_path"])
    assert source["sealed_evaluator_sha256"] == _file_sha256(
        source["sealed_evaluator_path"]
    )
    assert source["financial_metrics_sha256"] == _file_sha256(
        source["financial_metrics_path"]
    )
    assert source["ai_review_preparation_sha256"] == _file_sha256(
        source["ai_review_preparation_path"]
    )
    assert source["event_cohort_sha256"] == _file_sha256(source["event_cohort_path"])
    assert source["ai_protocol_sha256"] == _file_sha256(source["ai_protocol_path"])
    assert source["ai_bridge_sha256"] == _file_sha256(source["ai_bridge_path"])
    assert source["event_calibration_sha256"] == _file_sha256(
        source["event_calibration_path"]
    )
    assert source["ai_worker_sha256"] == _file_sha256(source["ai_worker_path"])
    assert source["ai_runtime_sha256"] == _file_sha256(source["ai_runtime_path"])
    assert source["event_training_sha256"] == _file_sha256(
        source["event_training_path"]
    )
    assert source["contract_generator_sha256"] == _file_sha256(
        source["contract_generator_path"]
    )
    assert source["storage_sha256"] == _file_sha256(source["storage_path"])
    assert source["event_sequence_schema_version"] == (
        "round-074-causal-event-sequence-v4"
    )
    assert source["event_model_schema_version"] == ROUND74_EVENT_MODEL_SCHEMA_VERSION
    assert source["event_scaler_schema_version"] == ROUND74_EVENT_SCALER_SCHEMA_VERSION
    assert source["event_target_schema_version"] == ROUND74_EVENT_TARGET_SCHEMA_VERSION
    assert source["event_target_evidence_schema_version"] == (
        ROUND74_EVENT_TARGET_EVIDENCE_SCHEMA_VERSION
    )
    assert source["event_dataset_schema_version"] == "round-074-event-dataset-v9"
    assert source["event_partition_schema_version"] == "round-074-run-partition-v4"
    assert (
        source["event_action_context_schema_version"] == "round-074-action-context-v4"
    )
    assert source["event_action_policy_schema_version"] == "round-074-action-policy-v7"
    assert source["ai_uplift_evaluator_schema_version"] == (
        "round-074-ai-uplift-development-v7"
    )
    assert source["ai_execution_replay_plan_schema_version"] == (
        ROUND74_AI_EXECUTION_REPLAY_PLAN_SCHEMA_VERSION
    )
    assert source["ai_execution_replay_evidence_schema_version"] == (
        ROUND74_AI_EXECUTION_REPLAY_EVIDENCE_SCHEMA_VERSION
    )
    assert (
        source["sealed_ledger_schema_version"] == ROUND74_SEALED_LEDGER_SCHEMA_VERSION
    )
    assert source["sealed_dataset_identity_schema_version"] == (
        ROUND74_SEALED_DATASET_IDENTITY_SCHEMA_VERSION
    )
    assert source["sealed_claim_schema_version"] == ROUND74_SEALED_CLAIM_SCHEMA_VERSION
    assert source["sealed_evaluator_schema_version"] == "round-074-sealed-evaluation-v9"
    assert source["financial_metrics_schema_version"] == (
        "round-074-realized-payoff-metrics-v1"
    )
    assert (
        source["target_free_inference_schema_version"]
        == ROUND74_TARGET_FREE_INFERENCE_SCHEMA_VERSION
    )
    assert source["ai_review_panel_schema_version"] == "round-074-ai-review-panel-v3"
    assert (
        source["event_cohort_plan_schema_version"]
        == ROUND74_EVENT_COHORT_PLAN_SCHEMA_VERSION
    )
    assert (
        source["event_cohort_binding_schema_version"]
        == ROUND74_EVENT_COHORT_BINDING_SCHEMA_VERSION
    )
    assert (
        source["ai_model_manifest_schema_version"]
        == ROUND74_AI_MODEL_MANIFEST_SCHEMA_VERSION
    )
    assert (
        source["ai_review_request_schema_version"] == "round-074-ai-review-request-v3"
    )
    assert (
        source["ai_review_decision_schema_version"]
        == ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION
    )
    assert (
        source["ai_worker_envelope_schema_version"]
        == ROUND74_AI_WORKER_ENVELOPE_SCHEMA_VERSION
    )
    assert (
        source["ai_worker_result_schema_version"]
        == ROUND74_AI_WORKER_RESULT_SCHEMA_VERSION
    )
    assert source["ai_runtime_outcome_schema_version"] == (
        "round-074-ai-runtime-outcome-v1"
    )
    assert source["ai_bridge_schema_version"] == "round-074-ai-bridge-v2"
    assert (
        source["tuning_subpartition_schema_version"]
        == ROUND74_TUNING_SUBPARTITION_SCHEMA_VERSION
    )
    assert (
        source["temperature_calibration_schema_version"]
        == "round-074-temperature-calibration-v2"
    )
    assert source["event_training_schema_version"] == "round-074-event-training-v12"
    assert source["pretest_policy_schema_version"] == (
        "round-074-event-pretest-policy-v11"
    )
    assert source["target_context_panel_schema_version"] == (
        ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION
    )
    assert source["feature_count"] == 66
    assert source["feature_names_sha256"] == (
        "a753db09a2d6a089af16ad40eb2d0d4781921d635a0822b9fd0523bcddb28f4a"
    )
    data_scope = design["data_scope"]
    assert data_scope["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert data_scope["exact_local_receipt_order_required"] is True
    assert data_scope["one_second_or_minute_collapse_permitted"] is False
    assert data_scope["listed_venue_calendar_may_create_crypto_close"] is False
    cohort = design["cohort_admission_contract"]
    assert cohort["implemented_now"] is True
    assert cohort["plan_sha256"] == (
        "acf3e4feb8a918b03ab8d85c9ce730022aed1581181301ed513bd4ab4399dfcb"
    )
    assert cohort["post_capture_dataset_schema_version"] == (
        "round-074-event-dataset-v9"
    )
    assert cohort["raw_capture_schedule_changed_for_multiscale_features"] is False
    assert cohort["role_counts"] == {
        "training": 120,
        "tuning": 24,
        "test": 24,
    }
    assert cohort["active_prerequisite_passed_now"] is False
    assert cohort["failed_or_missed_slot_replacement_permitted"] is False
    assert cohort["partition_hash_must_bind_plan_sha256"] is True
    assert cohort["maximum_target_span_seconds"] == 310.5
    assert cohort["minimum_purge_seconds"] == 310.5
    assert cohort["minimum_embargo_seconds"] == 310.5
    assert cohort["representative_market_data_collected_now"] is False
    features = design["causal_feature_contract"]
    assert features["per_event_asset_identity_retained"] is True
    assert features["window_may_cross_long_gap"] is False
    assert features["absolute_top_20_bid_and_ask_quote_depth_retained"] is True
    assert features["continuous_time_state_half_lives_seconds"] == list(
        ROUND74_EVENT_STATE_HALF_LIVES_SECONDS
    )
    assert features["continuous_time_decay_uses_receipt_nanoseconds"] is True
    assert features["fixed_event_rate_assumption_permitted"] is False
    assert features["state_future_receipt_or_target_access"] is False
    assert features["state_reset_after_receipt_gap_nanoseconds"] == (
        ROUND74_EVENT_DEFAULT_MAX_WINDOW_GAP_NS
    )
    assert features["unobserved_gap_interpolation_permitted"] is False
    assert features["gap_crossing_return_used_in_slow_state"] is False
    scaler = features["feature_scaler"]
    assert scaler["implemented_now"] is True
    assert scaler["fit_on_training_partition_only"] is True
    assert scaler["validation_or_test_statistics_permitted"] is False
    assert scaler["sample_indices_hash_bound"] is True
    assert scaler["sampling_algorithm"] == ("splitmix64-smallest-priority-v1")
    assert scaler["serialized_scaler_digest_verified_on_load"] is True
    assert scaler["model_bundle_must_bind_scaler_hash"] is True
    host = design["host_evidence_binding"]
    assert host["file_sha256_normalization"] == (design["file_sha256_normalization"])
    exchange_path = REPOSITORY / host["exchange_info_path"]
    assert host["exchange_info_file_sha256"] == _file_sha256(host["exchange_info_path"])
    exchange = _load_hash_bound(exchange_path, "artifact_sha256")
    assert exchange["artifact_sha256"] == host["exchange_info_artifact_sha256"]
    assert (
        exchange["execution_git_commit"] == (host["exchange_info_execution_git_commit"])
    )
    assert (
        exchange["target_evidence"]["evidence_sha256"]
        == (host["exchange_info_target_evidence_sha256"])
    )
    assert exchange["response"]["raw_payload_persisted"] is False
    funding_path = REPOSITORY / host["funding_path"]
    assert host["funding_file_sha256"] == _file_sha256(host["funding_path"])
    funding = _load_hash_bound(funding_path, "artifact_sha256")
    assert funding["artifact_sha256"] == host["funding_artifact_sha256"]
    assert funding["execution_git_commit"] == (host["funding_execution_git_commit"])
    assert (
        funding["target_evidence"]["evidence_sha256"]
        == (host["funding_target_evidence_sha256"])
    )
    assert funding["capture_binding"]["run_id"] == (host["funding_capture_run_id"])
    assert (
        funding["clock_binding"]["probe_count"] == (host["funding_clock_probe_count"])
    )
    assert funding["scope"]["capture_run_may_be_used_for_financial_evaluation"] is False


def test_round74_event_target_and_evaluation_contracts_fail_closed() -> None:
    design = _load_hash_bound(DESIGN_PATH, "design_sha256")
    targets = design["prospective_target_contract"]
    assert targets["implemented_now"] is True
    assert targets["real_market_targets_generated_now"] is False
    assert targets["source_derived_target_assembly_implemented"] is True
    assert (
        targets[
            "source_derived_target_assembly_accepts_caller_configured_fees_latency_slippage_or_quantity_rules"
        ]
        is False
    )
    assert (
        targets["source_derived_target_assembly_hash_binds_spec_and_quantity_claims"]
        is True
    )
    assert targets["source_derived_target_engine_revalidates_quantity_evidence"] is True
    assert targets["decision_order_key"] == [
        "received_monotonic_ns",
        "frame_index",
        "message_index",
    ]
    assert (
        targets[
            "later_messages_with_the_same_receipt_timestamp_permitted_in_decision_state"
        ]
        is False
    )
    assert (
        targets[
            "target_context_hash_binds_spec_quantity_filters_funding_schedule_and_execution_evidence"
        ]
        is True
    )
    assert (
        targets["reference_capital_normalization_uses_actual_walked_entry_quote"]
        is True
    )
    assert (
        targets["requested_quote_fraction_substitution_for_executed_notional_permitted"]
        is False
    )
    assert (
        targets["capital_scaled_payoff_and_mae_reconcile_quantity_price_and_reference"]
        is True
    )
    latency = targets["decision_to_entry_and_exit_execution_latency"]
    assert latency["must_be_measured_on_the_execution_host"] is True
    assert latency["entry_and_exit_must_be_measured_separately"] is True
    assert latency["entry_and_exit_must_be_bound_separately_for_each_symbol"] is True
    assert latency["single_global_latency_assumption_permitted"] is False
    assert latency["maximum_entry_latency_nanoseconds"] == 5_000_000_000
    assert latency["maximum_exit_latency_nanoseconds"] == 5_000_000_000
    assert latency["maximum_entry_state_lateness_nanoseconds"] == 250_000_000
    assert latency["maximum_exit_state_lateness_nanoseconds"] == 250_000_000
    assert latency["fixed_unverified_latency_assumption_permitted"] is False
    assert (
        latency[
            "structured_latency_evidence_record_must_bind_all_symbols_and_both_paths"
        ]
        is True
    )
    assert (
        latency["evidence_record_binds_environment_source_time_count_query_and_payload"]
        is True
    )
    entry = targets["entry_and_exit"]
    assert entry["initial_supported_execution"] == "marketable orders only"
    assert entry["quantity_rules_source_response_parser_implemented"] is True
    assert entry["quantity_rules_evidence_digest_required"] is True
    assert "MARKET_LOT_SIZE" in entry["quantity_rules_source"]
    assert "MIN_NOTIONAL" in entry["quantity_rules_source"]
    assert entry["quantity_precision_may_substitute_for_market_lot_size"] is False
    assert entry["exchange_info_requires_trading_perpetual_usdt_contracts"] is True
    assert entry["runtime_quantity_rules_must_match_evidence_claims"] is True
    assert entry["real_public_exchange_info_evidence_captured_now"] is True
    assert entry["passive_maker_fill_target_permitted_from_l2"] is False
    assert entry["insufficient_visible_depth_policy"] == (
        "censor target and forbid action"
    )
    costs = targets["costs"]
    assert costs["commission_evidence_digest_required"] is True
    assert costs["commission_must_be_bound_per_symbol"] is True
    assert costs["additional_residual_slippage_evidence_digest_required"] is True
    assert (
        costs[
            "additional_residual_slippage_must_be_bound_per_symbol_and_reference_notional"
        ]
        is True
    )
    assert costs["single_global_residual_slippage_assumption_permitted"] is False
    assert costs["funding_boundary_panel_required_for_all_three_symbols"] is True
    assert costs["funding_boundary_checked_against_requested_exit_before_entry"] is True
    assert costs["funding_boundary_rechecked_against_actual_exit_at_completion"] is True
    assert (
        costs["requested_exit_before_funding_may_override_actual_exit_after_funding"]
        is False
    )
    assert costs["funding_schedule_coverage_required_per_symbol"] is True
    assert costs["funding_boundaries_outside_declared_coverage_permitted"] is False
    assert (
        costs["entry_or_requested_or_actual_exit_outside_funding_coverage_policy"]
        == "censor target"
    )
    assert costs["funding_schedule_source_evidence_digest_required"] is True
    assert costs["silently_empty_or_partial_funding_schedule_permitted"] is False
    assert (
        costs["funding_timestamp_representation"]
        == "per-symbol monotonic lower and upper uncertainty interval"
    )
    assert costs["funding_interval_overlap_policy"] == "censor target"
    assert costs["exact_point_timestamp_required_or_assumed"] is False
    assert costs["touching_or_overlapping_funding_intervals_permitted"] is False
    assert costs["commission_source_response_parser_implemented"] is True
    assert costs["funding_history_source_response_parser_implemented"] is True
    assert (
        costs["empty_explicitly_bounded_funding_response_is_accepted_as_complete"]
        is True
    )
    assert costs["missing_symbol_funding_response_is_accepted_as_empty"] is False
    assert (
        costs["funding_evidence_record_count_includes_symbol_responses_and_rows"]
        is True
    )
    assert costs["credential_material_may_enter_target_evidence_or_artifacts"] is False
    assert costs["full_funding_limit_page_accepted_as_complete"] is False
    assert (
        "completed audited Round 74 v10 capture"
        in (costs["clock_probe_capture_requirement"])
    )
    assert "without interpolation" in costs["funding_clock_mapping"]
    assert costs["post_audit_clock_extraction_decompresses_only_probe_frames"] is True
    assert costs["real_authenticated_commission_evidence_captured_now"] is False
    assert costs["real_public_funding_evidence_captured_now"] is True
    assert "not target generation" in costs["real_public_funding_evidence_scope"]
    assert (
        costs[
            "execution_calibration_requires_flat_before_entry_and_after_reduce_only_exit"
        ]
        is True
    )
    assert costs["execution_calibration_client_order_id_prefix"] == "sat-r74-cal-"
    assert (
        costs["execution_calibration_reconciles_order_update_and_account_trade_fills"]
        is True
    )
    assert "fresh captured L2" in costs["execution_calibration_expected_price_source"]
    assert costs["execution_calibration_maximum_book_age_nanoseconds"] == (
        ROUND74_EXECUTION_CALIBRATION_MAXIMUM_BOOK_AGE_NS
    )
    assert costs["execution_calibration_minimum_completed_pairs_per_symbol"] == (
        ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL
    )
    assert (
        costs["execution_calibration_minimum_completed_pairs_per_symbol_entry_side"]
        == ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL_SIDE
    )
    assert costs["execution_tail_quantile"] == ROUND74_EXECUTION_CALIBRATION_QUANTILE
    assert costs["execution_tail_confidence"] == (
        ROUND74_EXECUTION_CALIBRATION_QUANTILE_CONFIDENCE
    )
    assert "distribution-free" in costs["execution_tail_estimator"]
    assert costs["execution_calibration_parser_places_orders"] is False
    assert costs["real_testnet_execution_calibration_completed_now"] is False
    assert (
        costs["structured_evidence_claims_must_match_exact_configured_values"] is True
    )
    assert (
        costs["mixed_mainnet_and_testnet_evidence_per_target_spec_permitted"] is False
    )
    assert costs["midpoint_payoff_reported_before_execution_friction"] is True
    assert costs["book_walk_implementation_shortfall_reported_separately"] is True
    assert costs["commission_and_residual_slippage_reported_separately"] is True
    assert costs["explicit_cost_equals_commission_plus_residual_slippage"] is True
    assert (
        costs["total_implementation_shortfall_equals_book_walk_plus_explicit_cost"]
        is True
    )
    assert (
        costs["net_payoff_equals_midpoint_payoff_minus_total_implementation_shortfall"]
        is True
    )
    assert costs["quote_and_basis_point_reconciliation_required"] is True
    assert costs["fees_plus_residual_slippage_may_be_labeled_total_cost"] is False
    assert costs["adverse_selection_definition"] == (
        "signed midpoint move against the selected side before execution friction"
    )
    assert costs["missing_account_fee_policy"] == "fail closed"
    assert costs["runtime_fee_mismatch_policy"] == (
        "model bundle is incompatible and cannot trade"
    )
    assert (
        targets["leverage"]["leverage_is_applied_only_by_independent_risk_controller"]
        is True
    )
    assert targets["path_risk"]["maximum_path_state_gap_nanoseconds"] == (250_000_000)
    assert targets["path_risk"]["excessive_path_state_gap_policy"] == ("censor target")
    dataset = design["dataset_assembly_contract"]
    assert dataset["implemented_now"] is True
    assert dataset["representative_market_dataset_built_now"] is False
    assert dataset["database_access"] == "read only"
    assert dataset["cohort_plan_digest_must_match_partition"] is True
    tuning = dataset["tuning_subpartition"]
    assert tuning["implemented_now"] is True
    assert tuning["expected_tuning_runs"] == 24
    assert tuning["model_selection_runs"] == 12
    assert tuning["probability_calibration_runs"] == 6
    assert tuning["action_policy_selection_runs"] == 6
    assert tuning["run_reuse_permitted"] is False
    assert tuning["sealed_test_accessed"] is False
    assert dataset["split_unit"] == "whole capture run"
    assert dataset["random_row_split_permitted"] is False
    assert dataset["minimum_purge_seconds"] == 310.5
    assert dataset["minimum_embargo_seconds"] == 310.5
    assert dataset[
        "maximum_target_span_seconds_including_latency_and_state_lateness_ceilings"
    ] == (310.5)
    assert dataset["batch_hash_binds_run_symbol_and_exact_decision_order"] is True
    assert (
        dataset["batch_hash_binds_exact_target_entry_and_exit_monotonic_times"] is True
    )
    assert dataset["assembled_test_batch_retains_single_one_use_access_digest"] is True
    assert dataset["development_batch_may_carry_test_access_digest"] is False
    assert dataset["per_capture_run_target_context_uniformity_required"] is True
    assert dataset["cross_run_target_context_change_permitted"] is True
    assert dataset["pretest_policy_binds_exact_sorted_target_context_panel"] is True
    assert dataset["model_payoff_label_uses_reference_capital_normalization"] is True
    assert dataset["model_mae_label_uses_reference_capital_normalization"] is True
    assert dataset["raw_position_bps_substituted_for_reference_capital_bps"] is False
    assert dataset["bounded_post_cohort_model_operator_implemented_now"] is True
    assert dataset["source_target_assembly_roundtrip_serialization_implemented"] is True
    assert dataset["per_capture_run_source_target_assembly_required"] is True
    assert dataset["scaler_fit_source"] == (
        "each unique raw training event exactly once"
    )
    assert dataset["scaler_fit_may_read_tuning_or_test_events"] is False
    assert dataset["training_source_replay_passes"] == 2
    assert dataset["tuning_source_replay_passes"] == 1
    assert dataset["test_source_replay_passes_during_development"] == 0
    assert dataset["one_in_memory_batch_per_capture_run"] is True
    assert dataset["intermediate_feature_or_target_cache_written"] is False
    assert dataset["source_database_write_access_required"] is False
    action = design["action_policy_contract"]
    assert action["implemented_now"] is True
    assert action["representative_market_policy_selected_now"] is False
    assert action["profiles"] == list(ROUND74_ACTION_PROFILES)
    assert action["default_profile"] == ROUND74_ACTION_DEFAULT_PROFILE
    assert action["candidate_horizons_seconds"] == list(ROUND74_ACTION_HORIZONS_SECONDS)
    assert action["candidate_derivation_receives_realized_targets"] is False
    assert action["candidate_input_context_contains_realized_target_fields"] is False
    assert (
        action["future_target_eligibility_may_delete_a_model_selected_action"] is False
    )
    assert (
        action["complete_executable_target_coverage_required_for_every_selected_action"]
        is True
    )
    assert action["selected_action_with_censored_target_policy"] == (
        "reject the threshold as unscorable"
    )
    assert (
        action[
            "fixed_zero_loss_or_profit_imputation_for_censored_selected_action_permitted"
        ]
        is False
    )
    assert action["calibrated_probabilities_required"] is True
    assert action["maximum_candidates_per_row"] == 1
    assert (
        action["threshold_selection_data"]
        == "six whole chronological policy-selection tuning runs only"
    )
    assert action["threshold_quantile_estimator"] == (
        "median of each active capture run's within-run linear quantile"
    )
    assert action["high_activity_run_receives_extra_threshold_weight"] is False
    assert action["busy_run_row_duplication_regression_tested"] is True
    assert action["sealed_test_accessed"] is False
    assert action["replay_uses_exact_captured_entry_and_exit_monotonic_times"] is True
    assert action["horizon_duration_used_as_exit_time_proxy"] is False
    assert action["drawdown_uses_actual_payoff_realization_order"] is True
    assert action["drawdown_order"] == (
        "cohort run then actual exit monotonic time then signal-order tie break"
    )
    assert (
        action[
            "shared_drawdown_implementation_required_for_ml_ai_and_sealed_evaluation"
        ]
        is True
    )
    assert action["diversification_requires_all_btc_eth_sol_symbols"] is True
    assert action["position_sizing_or_leverage_applied_here"] is False
    assert action["candidate_input_context_duplicates_full_feature_tensor"] is False
    assert action["selection_concatenates_full_feature_tensors"] is False
    assert action["selection_binds_every_target_batch_and_candidate_digest"] is True
    assert action["selection_objective"] == (
        "mean cumulative capture-run net bps minus profile-weighted worst realized "
        "drawdown and mean cumulative capture-run maximum-adverse-excursion penalties"
    )
    assert action["pooled_total_net_bps_used_for_selection"] is False
    assert action["total_net_bps_retained_as_diagnostic"] is True
    training = design["development_training_contract"]
    assert training["implemented_now"] is True
    assert training["representative_market_training_run_completed_now"] is False
    assert training["accepted_roles"] == ["training", "tuning"]
    assert training["test_role_rejected_before_backend_initialization"] is True
    assert training["training_and_tuning_sample_overlap_permitted"] is False
    assert training["minimum_role_transition_purge_seconds"] == 310.5
    assert training["mixed_partition_scaler_or_target_context_permitted"] is False
    assert training["one_capture_run_per_batch_required"] is True
    assert training["mixed_capture_run_batch_permitted"] is False
    assert training["repeated_capture_run_batch_permitted"] is False
    assert training["candidate_panel"] == list(ROUND74_EVENT_MODEL_CANDIDATES)
    assert training["complexity_promotion_planned_comparison_count"] == (
        ROUND74_COMPLEXITY_PROMOTION_COMPARISON_COUNT
    )
    assert training["complexity_promotion_required_paired_capture_runs"] == 24
    assert training["complexity_promotion_maximum_paired_run_loss_degradation"] == (
        1e-5
    )
    assert (
        training["complexity_promotion_statistical_independence_or_significance_claim"]
        is False
    )
    assert training["complexity_promotion_uses_sealed_test"] is False
    assert training["complexity_promotion_uses_backtest_pnl"] is False
    assert training["complexity_promotion_ledger_bound_into_pretest_policy"] is True
    assert training["default_seed_panel"] == list(ROUND74_EVENT_TRAINING_DEFAULT_SEEDS)
    assert training["backtest_roi_used_for_gradient_or_model_selection"] is False
    assert training["seed_ensemble_method"] == (
        "equal peer weights; arithmetic mean of peer quantiles for continuous "
        "heads and arithmetic mean of peer probabilities then logit conversion "
        "for classification heads"
    )
    assert training["classification_mean_logit_pooling_permitted"] is False
    assert training["run_balanced_loss_primary"] is True
    assert training["gradient_optimization_population_unit"] == "capture_run"
    assert (
        training["one_eligible_minibatch_per_training_run_per_optimizer_step"] is True
    )
    assert training["gradient_divisor"] == "training_capture_run_count"
    assert training["row_pooled_optimizer_steps_permitted"] is False
    assert training["shorter_run_minibatch_policy"] == (
        "deterministic epoch-rotated cycling"
    )
    assert training["high_activity_run_receives_extra_gradient_weight"] is False
    assert training["fully_censored_minibatches_contribute_gradients"] is False
    assert training["unequal_training_run_size_regression_tested"] is True
    assert training["two_run_directml_gradient_schedule_preflight_completed"] is True
    assert training["checkpoint_selection_metric"] == "run_balanced_loss"
    assert training["checkpoint_reload_verification_metric"] == ("run_balanced_loss")
    assert training["pooled_loss_used_for_checkpoint_reload_verification"] is False
    assert training["unequal_tuning_run_size_regression_tested"] is True
    assert training["per_capture_run_target_context_uniformity_required"] is True
    assert training["cross_run_target_context_change_permitted"] is True
    assert training["exact_sorted_target_context_panel_bound_into_policy"] is True
    assert training["fully_censored_minibatch_policy"] == (
        "skip before device transfer and record minibatch and row counts"
    )
    assert training["fully_censored_capture_run_policy"] == "reject"
    assert training["censored_targets_used_as_negative_labels"] is False
    assert training["fully_censored_minibatch_regression_tested"] is True
    assert training["high_activity_run_receives_extra_selection_weight"] is False
    assert training["probability_calibration_run_count"] == 6
    assert training["probability_calibration_exact_run_ids_bound"] is True
    assert (
        training["probability_calibration_tuning_subpartition_object_required"] is True
    )
    assert (
        training["probability_calibration_pooled_metrics_selection_permitted"] is False
    )
    assert training["probability_calibration_pooled_metrics_role"] == (
        "diagnostic_only"
    )
    assert training["unequal_calibration_run_size_regression_tested"] is True
    assert training["worst_run_loss_is_reported"] is True
    assert training["worst_run_loss_is_primary_optimization_objective"] is False
    assert training["probability_calibration_order"] == (
        "after seed-ensemble probability aggregation"
    )
    assert training["pickle_permitted"] is False
    assert training["policy_binds_entire_causal_source_chain"] is True
    assert training["cross_platform_bitwise_reproducibility_claim"] is False
    linear = design["candidate_panel"]["event_pooling_linear"]
    assert linear["parameter_count"] == 19_900
    assert linear["nonlinear_encoder"] is False
    assert linear["monotone_distributional_heads_shared_with_other_candidates"] is True
    assert design["candidate_panel"]["complexity_has_promotion_privilege"] is False
    temporal = design["candidate_panel"]["causal_event_tcn"]
    assert temporal["dilations"] == list(ROUND74_EVENT_TCN_DILATIONS)
    assert temporal["causal_receptive_field_events"] == (
        ROUND74_EVENT_TCN_RECEPTIVE_FIELD
    )
    assert temporal["frozen_sequence_fully_covered"] is True
    assert (
        temporal["frozen_sequence_length_events"]
        <= (temporal["causal_receptive_field_events"])
    )
    attention = design["candidate_panel"]["causal_event_attention"]
    assert attention["parameter_count"] == 153_532
    assert attention["hidden_channels"] == ROUND74_EVENT_ATTENTION_HIDDEN_CHANNELS
    assert attention["attention_heads"] == ROUND74_EVENT_ATTENTION_HEADS
    assert attention["layers"] == ROUND74_EVENT_ATTENTION_LAYERS
    assert attention["strict_causal_mask"] is True
    assert attention["frozen_sequence_length_events"] == 128
    evaluation = design["prospective_evaluation_contract"]
    assert "design-consumed" in evaluation["quiet_qualification_run_status"]
    assert evaluation["random_row_split_permitted"] is False
    assert evaluation["minimum_purge_seconds"] == 310.5
    assert evaluation["sealed_test_may_be_used_once"] is True
    assert evaluation["durable_one_use_ledger_implemented_now"] is True
    assert evaluation["sealed_evaluator_implemented_now"] is True
    assert (
        evaluation["reusable_target_free_candidate_inference_implemented_now"] is True
    )
    assert evaluation["reservation_reset_api_available"] is False
    assert (
        evaluation[
            "public_sealed_entrypoint_accepts_metadata_identity_not_target_batches"
        ]
        is True
    )
    assert evaluation["metadata_identity_reserved_before_test_batch_loader"] is True
    assert evaluation["test_batch_loader_invoked_only_after_live_reservation"] is True
    assert (
        evaluation["test_batch_identity_reconciled_to_reservation_before_scoring"]
        is True
    )
    assert (
        evaluation[
            "provider_failure_after_reservation_permanently_consumes_test_access"
        ]
        is True
    )
    assert evaluation["cryptographic_or_os_enforced_operator_blinding_claim"] is False
    assert evaluation["sealed_test_runs"] == 24
    assert evaluation["sealed_bootstrap_draws"] == (ROUND74_SEALED_BOOTSTRAP_DRAWS)
    assert evaluation["sealed_familywise_alpha"] == ROUND74_SEALED_FAMILYWISE_ALPHA
    assert evaluation["sealed_qualification_configuration_count"] == (
        ROUND74_SEALED_QUALIFICATION_CONFIGURATION_COUNT
    )
    assert evaluation["sealed_paired_ai_model_count"] == ROUND74_SEALED_AI_MODEL_COUNT
    assert evaluation["sealed_profitability_bound_alpha"] == (
        ROUND74_SEALED_FAMILYWISE_ALPHA
        / ROUND74_SEALED_QUALIFICATION_CONFIGURATION_COUNT
    )
    assert evaluation["sealed_paired_ai_uplift_bound_alpha"] == (
        ROUND74_SEALED_FAMILYWISE_ALPHA / ROUND74_SEALED_AI_MODEL_COUNT
    )
    assert evaluation["unadjusted_95_percent_lower_bounds_are_diagnostic_only"] is True
    assert (
        evaluation[
            "annualized_roi_sharpe_sortino_or_calmar_reported_without_a_capital_allocation_path"
        ]
        is False
    )
    assert evaluation["profitability_required_by_assertion_or_data_filter"] is False
    assert (
        evaluation["selected_action_with_censored_target_may_pass_financial_gates"]
        is False
    )
    assert evaluation["selected_action_target_coverage_failure_reason"] == (
        "selected_action_target_coverage_incomplete"
    )
    assert (
        design["ai_comparison_contract"]["ai_may_bypass_data_risk_or_execution_gate"]
        is False
    )
    ai = design["ai_comparison_contract"]
    assert ai["protocol_implemented_now"] is True
    assert ai["isolated_worker_implemented_now"] is True
    assert ai["fail_closed_parent_runtime_implemented_now"] is True
    assert ai["host_resource_preflight_completed_now"] is True
    assert ai["host_resource_preflight_passed_now"] is True
    assert ai["host_resource_preflight_failure"] is None
    assert ai["actual_model_inference_attempted_now"] is True
    assert ai["host_latency_preflight_completed_now"] is True
    assert ai["representative_market_ai_evaluation_completed_now"] is False
    assert ai["ai_uplift_established_now"] is False
    assert ai["post_inference_full_gpu_residency_required"] is True
    assert ai["causal_calibrated_model_bridge_implemented_now"] is True
    assert ai["causal_recent_direction_summary_implemented_now"] is True
    assert ai["causal_recent_direction_block_events"] == (
        ROUND74_AI_RECENT_BLOCK_EVENTS
    )
    assert ai["causal_recent_direction_realized_target_or_future_access"] is False
    assert ai["bridge_may_access_realized_targets"] is False
    assert ai["raw_uncalibrated_probability_permitted"] is False
    assert ai["request_binds_probability_calibration_sha256"] is True
    assert ai["actual_multibillion_parameter_inference_completed_now"] is True
    assert ai["supported_review_horizons_seconds"] == [30, 300]
    assert ai["one_and_five_second_ml_paths_wait_for_ai"] is False
    assert ai["ai_receives_only_a_preexisting_target_free_ml_candidate"] is True
    assert ai["ai_may_revive_an_ml_abstention"] is False
    assert ai["paired_development_evaluator_implemented_now"] is True
    assert ai["paired_sealed_evaluator_implemented_now"] is True
    assert ai["durable_one_use_sealed_ledger_implemented_now"] is True
    assert ai["two_model_target_free_review_preparation_implemented_now"] is True
    assert ai["review_progress_emitted_after_every_candidate_model_pair"] is True
    assert ai["paired_development_evaluator_may_select_ai_model_or_promote"] is False
    assert ai["missing_review_policy"] == "invalidate entire evaluation"
    assert ai["runtime_elapsed_nanoseconds_bound_into_each_review_hash"] is True
    assert ai["same_entry_latency_budget_bound_into_review_and_panel_hashes"] is True
    assert (
        ai[
            "same_entry_latency_budget_may_be_derived_from_forecast_horizon_or_request_validity"
        ]
        is False
    )
    assert ai["late_accepted_review_policy"] == (
        "retain the validated decision and rewalk the first eligible delayed L2 book "
        "within the frozen 30-second historical replay ceiling"
    )
    assert ai["same_entry_latency_eligibility_rate_gate"] is None
    assert ai["same_entry_latency_eligibility_is_diagnostic_only"] is True
    assert ai["same_entry_latency_eligibility_is_distinct_from_runtime_success"] is True
    assert ai["latency_adjusted_delayed_entry_replay_implemented_now"] is True
    assert ai["raw_feature_window_identity_preserved_into_ai_replay_plan"] is True
    assert (
        ai["ai_size_reduction_applied_before_quantity_quantization_and_book_walk"]
        is True
    )
    assert ai["delayed_entry_uses_first_observed_post_latency_l2_state"] is True
    assert ai["delayed_exit_and_path_risk_use_the_same_exact_replay"] is True
    assert ai["baseline_payoff_scaling_without_book_rewalk_permitted"] is False
    assert ai["nonexecuted_replay_statuses_apply_zero_exposure"] is True
    assert ai["sealed_ai_metrics_bind_each_execution_replay_sha256"] is True
    assert ai["exact_replay_binds_reference_quote_and_actual_entry_quote"] is True
    assert ai["actual_deployed_capital_bps_reconciled_per_execution"] is True
    assert ai["requested_size_multiplier_used_as_realized_notional_proxy"] is False
    assert ai["baseline_and_ai_use_same_reference_capital_denominator"] is True
    assert ai["capital_scaled_payoff_and_mae_recomputed_from_position_bps"] is True
    assert (
        ai["sealed_review_provider_receives_target_free_contexts_and_candidates_only"]
        is True
    )
    assert (
        ai["sealed_review_provider_receives_hash_bound_target_free_inference_only"]
        is True
    )
    assert ai["sealed_review_provider_concrete_adapter_implemented"] is True
    assert ai["sealed_replay_provider_invoked_only_after_live_reservation"] is True
    assert ai["sealed_replay_provider_read_only_store_adapter_implemented"] is True
    assert ai["sealed_replay_provider_requires_exact_test_run_assembly_panel"] is True
    assert ai["sealed_replay_provider_restores_global_instruction_order"] is True
    assert (
        ai["precomputed_execution_replay_evidence_accepted_by_public_sealed_entrypoint"]
        is False
    )
    assert (
        ai["replay_evidence_reconciled_to_post_reservation_instruction_panel"] is True
    )
    assert ai["historical_ai_queue_model_implemented_now"] is True
    assert ai["ai_queue_model"] == (
        "one FIFO single-server queue per alternative candidate model in historical "
        "decision order"
    )
    assert ai["review_row_binds_decision_wall_nanoseconds"] is True
    assert (
        ai["review_evidence_binds_service_queue_and_effective_latency_nanoseconds"]
        is True
    )
    assert ai["same_entry_latency_is_queue_delay_plus_measured_runtime_elapsed"] is True
    assert (
        ai["inference_service_time_alone_may_establish_same_entry_eligibility"] is False
    )
    assert (
        ai["future_target_eligibility_may_delete_a_model_selected_observation"] is False
    )
    assert ai["censored_selected_action_invalidates_ml_and_ai_financial_gates"] is True
    assert ai["absolute_date_or_real_symbol_exposed_to_ai"] is False
    assert (
        ai["target_eligibility_or_realized_exit_timing_may_select_ai_review_coverage"]
        is False
    )
    assert ai["ai_veto_may_admit_a_later_overlapping_candidate"] is False
    assert ai["ai_may_create_side_increase_size_set_leverage_or_touch_orders"] is False
    assert ai["deferred_finance_challenger_count"] == 2
    assert ai["deferred_challenger_downloaded_now"] is False
    assert ai["financial_qa_benchmark_implies_trading_edge"] is False
    assert (
        ai[
            "challenger_panel_expansion_requires_predeclared_resource_safe_packet_screen"
        ]
        is True
    )
    authority = design["authority"]
    assert authority["training_only_scaler_implementation"] is True
    assert authority["prospective_target_engine_implementation"] is True
    assert authority["leak_resistant_dataset_implementation"] is True
    assert authority["development_trainer_implementation"] is True
    assert (
        authority["paired_capture_run_complexity_promotion_gate_implementation"] is True
    )
    assert authority["immutable_pretest_policy_implementation"] is True
    assert authority["predeclared_cohort_admission_implementation"] is True
    assert authority["predeclared_cohort_plan"] is True
    assert authority["local_ai_review_protocol_implementation"] is True
    assert authority["tuning_subpartition_implementation"] is True
    assert authority["probability_calibration_implementation"] is True
    assert authority["selective_action_policy_implementation"] is True
    assert authority["paired_ai_uplift_development_evaluator_implementation"] is True
    assert authority["causal_calibrated_ai_bridge_implementation"] is True
    assert authority["durable_one_use_sealed_ledger_implementation"] is True
    assert authority["paired_ml_ai_sealed_evaluator_implementation"] is True
    assert authority["target_free_candidate_inference_implementation"] is True
    assert authority["two_model_ai_review_preparation_implementation"] is True
    assert authority["future_censored_action_rejection_implementation"] is True
    assert authority["equal_run_action_threshold_and_objective_implementation"] is True
    assert authority["historical_ai_queue_latency_implementation"] is True
    assert authority["exact_delayed_ai_execution_replay_implementation"] is True
    assert authority["sealed_exact_ai_execution_evidence_implementation"] is True
    assert authority["exact_reference_capital_accounting_implementation"] is True
    assert authority["metadata_only_sealed_reservation_implementation"] is True
    assert authority["post_reservation_test_batch_loader_implementation"] is True
    assert (
        authority["post_reservation_target_free_ai_review_provider_implementation"]
        is True
    )
    assert authority["post_reservation_exact_replay_provider_implementation"] is True
    assert authority["concrete_target_free_ai_review_adapter_implementation"] is True
    assert authority["read_only_store_exact_replay_adapter_implementation"] is True
    assert authority["operator_cannot_preinspect_test_data_claim"] is False
    assert authority["sealed_multiple_comparison_control_implementation"] is True
    assert authority["mandatory_funding_schedule_binding_implementation"] is True
    assert authority["symbol_specific_execution_evidence_implementation"] is True
    assert authority["implementation_shortfall_reconciliation_implementation"] is True
    assert authority["actual_exit_funding_recheck_implementation"] is True
    assert authority["funding_schedule_coverage_gate_implementation"] is True
    assert authority["funding_clock_uncertainty_interval_implementation"] is True
    assert authority["binance_source_evidence_parser_implementation"] is True
    assert authority["audited_clock_probe_loader_implementation"] is True
    assert authority["execution_calibration_evidence_parser_implementation"] is True
    assert authority["exchange_info_quantity_rules_parser_implementation"] is True
    assert (
        authority["quantity_rules_runtime_evidence_match_gate_implementation"] is True
    )
    assert authority["source_derived_target_assembly_implementation"] is True
    assert authority["bounded_post_cohort_model_data_operator_implementation"] is True
    assert (
        authority["source_target_assembly_roundtrip_serialization_implementation"]
        is True
    )
    assert authority["complete_empty_bounded_funding_response_implementation"] is True
    assert authority["real_public_exchange_info_evidence_captured"] is True
    assert authority["real_authenticated_commission_evidence_captured"] is False
    assert authority["real_public_funding_evidence_captured"] is True
    assert authority["real_testnet_execution_calibration_completed"] is False
    assert authority["probability_calibration_directml_compute_preflight"] is True
    assert authority["local_ai_isolated_worker_implementation"] is True
    assert authority["local_ai_fail_closed_parent_runtime_implementation"] is True
    assert authority["local_ai_host_resource_preflight"] is True
    assert authority["actual_multibillion_parameter_ai_inference"] is True
    for key in (
        "target_generation",
        "model_training",
        "model_selection",
        "financial_edge_tested",
        "profitability_claim",
        "ai_uplift_claim",
        "paper_trading_authority",
        "testnet_trading_authority",
        "live_trading_authority",
    ):
        assert authority[key] is False


def test_round74_event_replay_evidence_is_exact_read_only_and_pre_target() -> None:
    design = _load_hash_bound(DESIGN_PATH, "design_sha256")
    replay = _load_hash_bound(REPLAY_PATH, "artifact_sha256")

    binding = design["host_evidence_binding"]
    assert (
        replay["execution_git_commit"]
        == binding["event_sequence_replay_execution_git_commit"]
    )
    assert replay["artifact_sha256"] == binding["event_sequence_replay_artifact_sha256"]
    assert binding["event_sequence_replay_current_source_bound"] is False
    assert (
        replay["event_sequence_source_sha256"]
        != design["source_binding"]["event_sequence_sha256"]
    )
    assert "sequence-v2 only" in binding["event_sequence_replay_reuse_scope"]
    evidence = replay["replay"]
    assert evidence["observation_count"] == 50_444
    assert evidence["token_count"] == evidence["token_limit"] == 50_000
    assert evidence["synchronized_depth_state_count"] == 4_984
    assert sum(evidence["event_counts"].values()) == 50_000
    assert sum(evidence["symbol_counts"].values()) == 50_000
    assert set(evidence["symbol_counts"]) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert evidence["canonical_length_prefixed_token_sha256"] == (
        "03f52996078dbd514570b83434ae1bb15d713103448a624ddd82edc4be677f02"
    )
    storage = replay["storage_safety"]
    assert storage["database_open_mode"] == "read_only"
    assert (
        storage["database_size_before_bytes"] == (storage["database_size_after_bytes"])
    )
    assert storage["wal_size_before_bytes"] == storage["wal_size_after_bytes"] == 0
    assert storage["database_write_detected"] is False
    limitations = replay["observed_sample_limitations"]
    assert limitations["liquidation_event_count"] == 0
    assert limitations["all_observation_stale_depth_event_count"] == 90
    assert limitations["feature_ready_stale_depth_event_count"] == 0
    assert limitations["stale_depth_used_as_fresh_target_state"] is False
    assert limitations["sample_validates_liquidation_path"] is False
    assert limitations["sample_is_financially_representative"] is False
    scaler = replay["diagnostic_scaler"]
    assert scaler["input_event_rows"] == 50_000
    assert scaler["bounded_sample_rows"] == 10_000
    assert scaler["chunk_invariance_verified"] is True
    assert scaler["eligible_for_model_bundle"] is False
    interpretation = replay["interpretation"]
    assert interpretation["targets_constructed"] is False
    assert interpretation["models_evaluated"] is False
    assert interpretation["financial_edge_tested"] is False


def test_round74_local_ai_runtime_preflight_is_hash_bound_and_nonpromotional() -> None:
    design = _load_hash_bound(DESIGN_PATH, "design_sha256")
    evidence = _load_hash_bound(AI_RUNTIME_PREFLIGHT_PATH, "artifact_sha256")
    binding = design["host_evidence_binding"]["local_ai_runtime_preflight"]

    assert (
        binding["path"] == AI_RUNTIME_PREFLIGHT_PATH.relative_to(REPOSITORY).as_posix()
    )
    assert binding["file_sha256"] == _file_sha256(binding["path"])
    assert binding["artifact_sha256"] == evidence["artifact_sha256"]
    assert binding["schema_version"] == evidence["schema_version"]
    assert binding["execution_git_commit"] == evidence["execution_git_commit"]
    assert binding["publisher_path"] == evidence["source_binding"]["publisher_path"]
    assert binding["publisher_sha256"] == evidence["source_binding"]["publisher_sha256"]
    assert binding["protocol_sha256"] == design["source_binding"]["ai_protocol_sha256"]
    assert binding["model_count"] == 2
    assert binding["model_names"] == ["fino1:8b", "qwen3:8b"]
    assert binding["decision_schema_versions"] == [
        ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION,
        ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION,
    ]
    assert binding["all_models_accepted_by_protocol"] is True
    assert binding["all_models_fully_gpu_resident"] is True
    for key in (
        "all_models_remote_inference_used",
        "all_models_execution_authority",
        "all_models_may_increase_risk",
        "all_models_may_select_side",
        "all_models_may_set_leverage",
        "all_models_may_submit_or_cancel_orders",
        "representative_market_ai_evaluation_completed",
        "ai_uplift_established",
        "financial_edge_established",
        "profitability_claim",
    ):
        assert binding[key] is False
    assert binding["resident_models_before"] == []
    assert binding["resident_models_after"] == []

    assert evidence["verification"]["all_models_accepted_by_protocol"] is True
    assert evidence["verification"]["all_models_fully_gpu_resident"] is True
    assert evidence["runtime_isolation"]["model_process_terminated"] is False
    assert evidence["runtime_isolation"]["official_keep_alive_zero_api_used"] is True
    assert evidence["runtime_isolation"]["resident_models_after"] == []
    assert [value["model_name"] for value in evidence["model_outcomes"]] == [
        "fino1:8b",
        "qwen3:8b",
    ]
    for outcome in evidence["model_outcomes"]:
        runtime = outcome["outcome"]
        worker = runtime["worker_result"]
        decision = worker["decision"]
        assert runtime["status"] == "accepted"
        assert runtime["capability"]["model_parameters_b"] >= 8.0
        assert worker["full_gpu_residency_verified"] is True
        assert worker["residency"]["vram_to_model_ratio"] == 1.0
        assert worker["remote_inference_used"] is False
        assert worker["execution_authority"] is False
        assert decision["schema_version"] == ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION
        assert decision["may_increase_risk"] is False
        assert decision["may_select_side"] is False
        assert decision["may_set_leverage"] is False
        assert decision["may_submit_or_cancel_orders"] is False
    for key in (
        "representative_market_ai_evaluation_completed",
        "ai_uplift_established",
        "financial_edge_established",
        "profitability_claim",
        "paper_trading_authority",
        "testnet_trading_authority",
        "live_trading_authority",
    ):
        assert evidence["interpretation"][key] is False


def test_round74_directml_evidence_is_amd_accelerated_and_nonfinancial() -> None:
    design = _load_hash_bound(DESIGN_PATH, "design_sha256")
    evidence = _load_hash_bound(DIRECTML_PATH, "artifact_sha256")

    binding = design["host_evidence_binding"]
    assert (
        evidence["execution_git_commit"]
        == binding["event_model_directml_execution_git_commit"]
    )
    assert (
        evidence["artifact_sha256"] == binding["event_model_directml_artifact_sha256"]
    )
    assert (
        evidence["event_model_source_sha256"]
        == binding["event_model_directml_historical_source_sha256"]
    )
    assert binding["event_model_directml_current_source_bound"] is False
    assert (
        evidence["event_model_source_sha256"]
        != (design["source_binding"]["event_model_sha256"])
    )
    assert "Historical two-candidate" in binding["event_model_directml_reuse_scope"]
    backend = evidence["backend"]
    assert backend["requested"] == backend["kind"] == "directml"
    assert backend["vendor"] == "AMD Radeon RX 9070 XT"
    assert backend["accelerated"] is True
    verification = evidence["verification"]
    assert verification["forward_completed"] is True
    assert verification["backward_completed"] is True
    assert verification["parameter_update_completed"] is True
    assert verification["warning_count"] == 0
    assert verification["cpu_fallback_warning_count"] == 0
    assert verification["tcn_receptive_field_events"] == (
        ROUND74_EVENT_TCN_RECEPTIVE_FIELD
    )
    assert verification["frozen_sequence_fully_covered"] is True
    assert evidence["input_contract"]["masked_action_targets"] == 1
    for candidate in evidence["candidates"].values():
        assert candidate["parameter_count"] > 0
        assert candidate["parameter_max_abs_change"] > 0.0
    assert binding["event_model_directml_current_source_bound"] is False
    interpretation = evidence["interpretation"]
    assert interpretation["real_market_targets_used"] is False
    assert interpretation["model_fit_performed"] is False
    assert interpretation["financial_edge_tested"] is False
    assert interpretation["profitability_claim"] is False


def test_round74_training_preflight_is_repeated_amd_compute_only() -> None:
    design = _load_hash_bound(DESIGN_PATH, "design_sha256")
    evidence = _load_hash_bound(TRAINING_PATH, "artifact_sha256")
    binding = design["host_evidence_binding"]

    assert (
        evidence["execution_git_commit"]
        == binding["event_training_directml_execution_git_commit"]
    )
    assert (
        evidence["artifact_sha256"]
        == binding["event_training_directml_artifact_sha256"]
    )
    assert (
        _file_sha256(binding["event_training_directml_path"])
        == (binding["event_training_directml_file_sha256"])
    )
    source = evidence["source_binding"]
    assert (
        source["event_targets_sha256"]
        == (binding["event_training_directml_target_sha256"])
    )
    assert (
        source["event_targets_sha256"]
        == (binding["event_training_directml_target_legacy_checkout_sha256"])
    )
    assert (
        binding["event_training_directml_target_canonical_sha256"]
        == (design["source_binding"]["event_target_sha256"])
    )
    assert (
        binding["event_training_directml_legacy_checkout_hashes_canonicalized"] is True
    )
    assert binding["event_training_directml_current_target_source_bound"] is True
    assert binding["event_training_directml_model_source_bound"] is True
    assert binding["event_training_directml_training_source_bound"] is True
    assert (
        source["event_training_sha256"]
        == design["source_binding"]["event_training_sha256"]
    )
    assert (
        source["event_dataset_sha256"]
        == (binding["event_training_directml_dataset_legacy_checkout_sha256"])
    )
    assert (
        binding["event_training_directml_dataset_canonical_sha256"]
        == (design["source_binding"]["event_dataset_sha256"])
    )
    assert (
        source["event_model_sha256"] == design["source_binding"]["event_model_sha256"]
    )
    assert (
        source["event_cohort_sha256"]
        == binding["event_training_directml_event_cohort_sha256"]
    )
    assert (
        source["event_cohort_sha256"] == design["source_binding"]["event_cohort_sha256"]
    )
    assert (
        "constructed unequal training runs"
        in (binding["event_training_directml_reuse_scope"])
    )
    assert (
        "equal capture-run gradient weight"
        in (binding["event_training_directml_reuse_scope"])
    )
    assert "not market fit" in (binding["event_training_directml_reuse_scope"])
    assert source["preflight_runner_sha256"] == _file_sha256(
        source["preflight_runner_path"]
    )
    assert source["publisher_sha256"] == _file_sha256(source["publisher_path"])
    assert (
        source["publisher_sha256"]
        == (binding["event_training_directml_publisher_sha256"])
    )
    backend = evidence["backend"]
    assert backend["requested"] == backend["kind"] == "directml"
    assert backend["vendor"] == "AMD Radeon RX 9070 XT"
    assert backend["accelerated"] is True
    assert backend["safetensors_version"] == "0.8.0"
    assert backend["warning_count_per_execution"] == 0
    assert backend["cpu_fallback_warning_count_per_execution"] == 0
    inputs = evidence["input_contract"]
    assert inputs["real_market_events_used"] is False
    assert inputs["real_market_targets_used"] is False
    assert inputs["test_batches_consumed"] == 0
    assert inputs["candidate_ids"] == list(ROUND74_EVENT_MODEL_CANDIDATES)
    assert inputs["candidate_parameter_counts"] == {
        candidate_id: design["candidate_panel"][candidate_id]["parameter_count"]
        for candidate_id in ROUND74_EVENT_MODEL_CANDIDATES
    }
    assert inputs["feature_count"] == 66
    assert inputs["feature_names_sha256"] == (
        "a753db09a2d6a089af16ad40eb2d0d4781921d635a0822b9fd0523bcddb28f4a"
    )
    assert inputs["state_half_lives_seconds"] == list(
        ROUND74_EVENT_STATE_HALF_LIVES_SECONDS
    )
    assert inputs["seeds"] == list(ROUND74_EVENT_TRAINING_DEFAULT_SEEDS)
    assert inputs["training_capture_runs"] == 2
    assert inputs["optimization_population"] == {
        "fully_censored_capture_run_policy": "reject",
        "fully_censored_minibatches_contribute_gradients": False,
        "gradient_divisor": "training_capture_run_count",
        "optimizer_step": (
            "one eligible minibatch per training capture run with gradient accumulation"
        ),
        "row_pooled_optimizer_steps_permitted": False,
        "shorter_run_policy": (
            "deterministic epoch-rotated cycling of eligible minibatches"
        ),
        "unit": "capture_run",
    }
    verification = evidence["verification"]
    assert verification["fresh_process_execution_count"] == 2
    assert verification["candidate_count"] == len(ROUND74_EVENT_MODEL_CANDIDATES)
    assert verification["all_candidates_trained"] is True
    assert verification["seed_count_per_candidate"] == len(
        ROUND74_EVENT_TRAINING_DEFAULT_SEEDS
    )
    assert verification["peer_update_count"] == 12
    assert verification["cross_execution_complete_result_equal"] is True
    assert verification["cross_execution_policy_sha256_equal"] is True
    assert verification["cross_execution_model_sha256_equal"] is True
    assert verification["cross_execution_prediction_sha256_equal"] is True
    assert verification["cross_execution_candidate_metrics_equal"] is True
    assert verification["temporary_artifacts_removed_after_each_execution"] is True
    repeated = evidence["repeated_result"]
    schedules = repeated["peer_run_balanced_optimization_schedule"]
    assert set(schedules) == set(ROUND74_EVENT_MODEL_CANDIDATES)
    assert all(
        schedule
        == {
            "maximum_eligible_minibatches_per_run": 3.0,
            "maximum_run_minibatch_contributions": 3.0,
            "minimum_eligible_minibatches_per_run": 1.0,
            "minimum_run_minibatch_contributions": 3.0,
            "optimizer_steps": 3.0,
            "run_contributions_per_optimizer_step": 2.0,
            "run_count": 2.0,
        }
        for candidate_schedules in schedules.values()
        for schedule in candidate_schedules
    )
    assert (
        repeated["candidate_run_balanced_tuning_proper_loss"]
        == (repeated["candidate_worst_run_tuning_proper_loss"])
    )
    assert (
        repeated["candidate_run_balanced_tuning_proper_loss"]
        == (repeated["candidate_pooled_tuning_proper_loss"])
    )
    selection = repeated["selection"]
    assert selection["selected_candidate_id"] == "event_pooling_linear"
    assert (
        repeated["selected_candidate_id"]
        == (binding["event_training_directml_selected_candidate_id"])
    )
    assert selection["ordered_candidate_ids"] == list(ROUND74_EVENT_MODEL_CANDIDATES)
    assert selection["planned_comparison_count"] == (
        ROUND74_COMPLEXITY_PROMOTION_COMPARISON_COUNT
    )
    assert selection["required_paired_capture_run_count"] == 24
    assert selection["statistical_independence_or_significance_claim"] is False
    assert all(
        report["paired_capture_run_count"] == 1
        and report["required_paired_capture_run_count"] == 24
        and report["complete_tuning_panel"] is False
        and report["all_paired_runs_noninferior"] is True
        and report["promoted"] is False
        for report in selection["promotion_reports"]
    )
    assert binding["event_training_directml_observed_paired_run_count"] == 1
    assert binding["event_training_directml_candidate_count"] == len(
        ROUND74_EVENT_MODEL_CANDIDATES
    )
    assert binding["event_training_directml_required_paired_run_count"] == (24)
    assert binding["event_training_directml_complete_tuning_panel"] is False
    assert binding["event_training_directml_training_capture_run_count"] == 2
    assert binding["event_training_directml_optimizer_steps_per_peer"] == 3
    assert binding["event_training_directml_run_contributions_per_optimizer_step"] == 2
    assert binding["event_training_directml_minimum_eligible_minibatches_per_run"] == 1
    assert binding["event_training_directml_maximum_eligible_minibatches_per_run"] == 3
    assert binding["event_training_directml_minimum_run_minibatch_contributions"] == 3
    assert binding["event_training_directml_maximum_run_minibatch_contributions"] == 3
    assert binding["event_training_directml_optimization_population_unit"] == (
        "capture_run"
    )
    assert binding["event_training_directml_statistical_significance_claim"] is False
    assert (
        binding["event_training_directml_candidate_losses_have_financial_meaning"]
        is False
    )
    interpretation = evidence["interpretation"]
    assert interpretation["candidate_loss_has_financial_meaning"] is False
    assert interpretation["representative_market_training_performed"] is False
    assert interpretation["sealed_test_evaluated"] is False
    assert interpretation["financial_edge_tested"] is False
    assert interpretation["profitability_claim"] is False


def test_round74_calibration_preflight_is_amd_compute_only() -> None:
    design = _load_hash_bound(DESIGN_PATH, "design_sha256")
    evidence = _load_hash_bound(CALIBRATION_PATH, "artifact_sha256")
    binding = design["host_evidence_binding"]

    assert (
        evidence["execution_git_commit"]
        == binding["temperature_calibration_directml_execution_git_commit"]
    )
    assert (
        evidence["artifact_sha256"]
        == binding["temperature_calibration_directml_artifact_sha256"]
    )
    assert (
        evidence["source_binding"]["sha256"]
        == design["source_binding"]["event_calibration_sha256"]
    )
    backend = evidence["backend"]
    assert backend["requested"] == backend["kind"] == "directml"
    assert backend["vendor"] == "AMD Radeon RX 9070 XT"
    assert backend["accelerated"] is True
    assert backend["warning_count"] == 0
    assert backend["cpu_fallback_warning_count"] == 0
    verification = evidence["verification"]
    assert verification["candidate_temperature_count"] == 257
    assert verification["all_three_temperature_searches_completed"] is True
    assert (
        verification["all_head_temperatures_invariant_to_busy_run_duplication"] is True
    )
    assert (
        verification["baseline_run_balanced_nll_after"]
        < (verification["baseline_run_balanced_nll_before"])
    )
    assert (
        verification["duplicated_run_balanced_nll_after"]
        == (verification["baseline_run_balanced_nll_after"])
    )
    assert verification["duplicated_maximum_run_observations"] == 200
    assert verification["duplicated_minimum_run_observations"] == 2
    assert verification["hidden_cpu_fallback_detected"] is False
    interpretation = evidence["interpretation"]
    assert interpretation["equal_capture_run_weighting_verified"] is True
    assert interpretation["pooled_observation_metrics_used_for_selection"] is False
    assert interpretation["representative_market_calibration_performed"] is False
    assert interpretation["financial_edge_tested"] is False
    assert interpretation["profitability_claim"] is False
