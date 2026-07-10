from __future__ import annotations

import json

import numpy as np
import pytest

from simple_ai_trading.microstructure_features import (
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_VERSION,
    MicrostructureDataset,
)
from simple_ai_trading.microstructure_model import (
    MICROSTRUCTURE_MODEL_SCHEMA_VERSION,
    _apply_platt_scaling,
    _fit_platt_scaling,
    _performance_confidence,
    _purged_split,
    _select_threshold,
    _simulate_non_overlapping,
    _simulate_non_overlapping_trace,
    load_microstructure_action_scorer,
)


def _five_day_dataset() -> MicrostructureDataset:
    timestamps = np.concatenate(
        [day * 86_400_000 + np.arange(1_000, dtype=np.int64) * 1_000 for day in range(5)]
    )
    rows = len(timestamps)
    exits = timestamps + 60_100
    ones = np.ones(rows, dtype=np.float64)
    return MicrostructureDataset(
        symbol="BTCUSDT",
        feature_version="test",
        feature_names=("x", "y"),
        horizon_seconds=60,
        total_latency_ms=100,
        taker_fee_bps=5.0,
        reference_order_notional_quote=1.0,
        max_l1_participation=0.10,
        decision_cadence_seconds=1,
        target_mode="fixed_horizon",
        stop_loss_bps=None,
        take_profit_bps=None,
        trigger_execution_slippage_bps=None,
        path_resolution_ms=None,
        decision_time_ms=timestamps,
        long_exit_time_ms=exits,
        short_exit_time_ms=exits,
        features=np.zeros((rows, 2), dtype=np.float32),
        long_net_bps=-ones,
        short_net_bps=-ones,
        entry_spread_bps=ones,
        exit_spread_bps=ones,
        entry_quote_age_ms=np.zeros(rows, dtype=np.int64),
        exit_quote_age_ms=np.zeros(rows, dtype=np.int64),
        entry_bid_price=100.0 * ones,
        entry_ask_price=101.0 * ones,
        fixed_exit_bid_price=100.0 * ones,
        fixed_exit_ask_price=101.0 * ones,
        entry_bid_qty=ones,
        entry_ask_qty=ones,
        fixed_exit_bid_qty=ones,
        fixed_exit_ask_qty=ones,
        long_l1_participation=0.01 * ones,
        short_l1_participation=0.01 * ones,
        long_liquidity_eligible=np.ones(rows, dtype=bool),
        short_liquidity_eligible=np.ones(rows, dtype=bool),
    )


def test_purged_split_separates_policy_selection_and_terminal_days() -> None:
    dataset = _five_day_dataset()
    splits, evidence = _purged_split(dataset)

    assert set(splits) == {"train", "tuning", "policy", "selection", "terminal"}
    assert all(len(indexes) >= 256 for indexes in splits.values())
    assert evidence.policy_rows == len(splits["policy"])
    assert evidence.selection_rows == len(splits["selection"])
    for left, right in (("train", "tuning"), ("tuning", "policy"), ("policy", "selection")):
        assert (
            dataset.decision_time_ms[splits[left][-1]] + evidence.purge_ms
            < dataset.decision_time_ms[splits[right][0]]
        )
    assert (
        dataset.decision_time_ms[splits["selection"][-1]] + evidence.purge_ms
        < dataset.decision_time_ms[splits["terminal"][0]]
    )


def test_simulator_uses_eligible_side_and_realized_exit_time() -> None:
    metrics = _simulate_non_overlapping(
        timestamps=np.asarray([0, 1_000, 2_000], dtype=np.int64),
        long_exit_times=np.asarray([10_000, 5_000, 3_000], dtype=np.int64),
        short_exit_times=np.asarray([1_000, 5_000, 3_000], dtype=np.int64),
        long_targets=np.asarray([-9.0, 3.0, 100.0]),
        short_targets=np.asarray([2.0, -4.0, 100.0]),
        long_edge=np.asarray([10.0, 6.0, 8.0]),
        short_edge=np.asarray([5.0, 4.0, 7.0]),
        long_probability=np.asarray([0.1, 0.9, 0.9]),
        short_probability=np.asarray([0.9, 0.9, 0.9]),
        edge_threshold=1.0,
        probability_threshold=0.5,
    )

    assert metrics.trades == 2
    assert metrics.long_trades == 1
    assert metrics.short_trades == 1
    assert metrics.total_net_bps == 5.0


def test_simulator_never_executes_a_side_that_exceeds_the_l1_participation_gate() -> None:
    metrics = _simulate_non_overlapping(
        timestamps=np.asarray([0], dtype=np.int64),
        long_exit_times=np.asarray([1_000], dtype=np.int64),
        short_exit_times=np.asarray([1_000], dtype=np.int64),
        long_targets=np.asarray([50.0]),
        short_targets=np.asarray([2.0]),
        long_edge=np.asarray([100.0]),
        short_edge=np.asarray([3.0]),
        long_probability=np.asarray([0.99]),
        short_probability=np.asarray([0.90]),
        edge_threshold=1.0,
        probability_threshold=0.5,
        long_eligible=np.asarray([False]),
        short_eligible=np.asarray([True]),
    )

    assert metrics.trades == 1
    assert metrics.long_trades == 0
    assert metrics.short_trades == 1
    assert metrics.total_net_bps == 2.0


def test_day_block_confidence_includes_the_full_calendar_and_is_deterministic() -> None:
    timestamps = np.arange(60, dtype=np.int64) * 86_400_000
    ones = np.ones(60, dtype=np.float64)
    zeros = np.zeros(60, dtype=np.float64)
    trace = _simulate_non_overlapping_trace(
        timestamps=timestamps,
        long_exit_times=timestamps + 1_000,
        short_exit_times=timestamps + 1_000,
        long_targets=ones,
        short_targets=-ones,
        long_edge=ones,
        short_edge=zeros,
        long_probability=ones,
        short_probability=zeros,
        edge_threshold=0.5,
        probability_threshold=0.5,
    )

    first = _performance_confidence(trace, timestamps)
    second = _performance_confidence(trace, timestamps)

    assert first == second
    assert first.calendar_days == 60
    assert first.active_days == 60
    assert first.bootstrap_samples == 2_000
    assert first.mean_daily_net_bps_ci_lower == pytest.approx(1.0)
    assert first.bootstrap_probability_mean_positive == 1.0


def test_platt_scaling_corrects_probability_base_rate() -> None:
    labels = np.concatenate((np.zeros(800), np.ones(200)))
    raw = np.full(1_000, 0.40)

    calibration = _fit_platt_scaling(raw, labels)
    calibrated = _apply_platt_scaling(raw, calibration)

    assert calibration[0] > 0.0
    assert np.isclose(np.mean(calibrated), 0.20, atol=1e-4)
    assert np.mean(np.square(calibrated - labels)) < np.mean(np.square(raw - labels))


def test_threshold_search_uses_base_rate_relative_probability_tails() -> None:
    rows = 500
    timestamps = np.arange(rows, dtype=np.int64) * 1_000
    long_edge = np.full(rows, -1.0)
    long_edge[-100:] = np.linspace(0.1, 1.0, 100)
    long_probability = np.full(rows, 0.20)
    long_probability[-100:] = np.linspace(0.25, 0.35, 100)
    long_targets = np.full(rows, -2.0)
    long_targets[-100:] = 3.0

    policy, metrics, evidence = _select_threshold(
        risk_level="conservative",
        timestamps=timestamps,
        long_exit_times=timestamps + 1_000,
        short_exit_times=timestamps + 1_000,
        long_targets=long_targets,
        short_targets=np.full(rows, -2.0),
        long_edge=long_edge,
        short_edge=np.full(rows, -1.0),
        long_probability=long_probability,
        short_probability=np.full(rows, 0.20),
    )

    assert policy.minimum_profitable_probability < 0.50
    assert metrics.trades >= 20
    assert metrics.total_net_bps > 0.0
    assert evidence.minimum_required_trades == 20
    assert evidence.policies_meeting_trade_minimum > 0


def _runtime_artifact_payload(*, status: str = "accepted") -> dict[str, object]:
    return {
        "schema_version": MICROSTRUCTURE_MODEL_SCHEMA_VERSION,
        "status": status,
        "rejection_reasons": [] if status == "accepted" else ["selection_failed"],
        "symbol": "BTCUSDT",
        "risk_level": "conservative",
        "feature_version": MICROSTRUCTURE_FEATURE_VERSION,
        "feature_names": list(MICROSTRUCTURE_FEATURE_NAMES),
        "horizon_seconds": 60,
        "total_latency_ms": 250,
        "taker_fee_bps": 5.0,
        "reference_order_notional_quote": 1_000.0,
        "max_l1_participation": 0.10,
        "decision_cadence_seconds": 5,
        "target_mode": "exchange_trigger_market_exit_1s_adverse_first",
        "stop_loss_bps": 25.0,
        "take_profit_bps": 40.0,
        "trigger_execution_slippage_bps": 1.0,
        "path_resolution_ms": 1_000,
        "unique_utc_days": 365,
        "minimum_promotion_days": 365,
        "calendar_day_coverage_ratio": 1.0,
        "terminal_evaluated_at": "2026-01-01T00:00:00+00:00",
        "terminal_metrics": {
            "trades": 40,
            "total_net_bps": 100.0,
            "max_drawdown_bps": 20.0,
            "profit_factor": 1.5,
        },
        "selection_confidence": {
            "calendar_days": 55,
            "active_days": 40,
            "block_length_days": 4,
            "bootstrap_samples": 2_000,
            "mean_daily_net_bps": 2.0,
            "median_daily_net_bps": 1.0,
            "worst_daily_net_bps": -3.0,
            "best_daily_net_bps": 8.0,
            "annualized_daily_sharpe": 2.2,
            "mean_daily_net_bps_ci_lower": 0.4,
            "mean_daily_net_bps_ci_upper": 3.5,
            "bootstrap_probability_mean_positive": 0.99,
        },
        "terminal_confidence": {
            "calendar_days": 55,
            "active_days": 40,
            "block_length_days": 4,
            "bootstrap_samples": 2_000,
            "mean_daily_net_bps": 1.8,
            "median_daily_net_bps": 0.9,
            "worst_daily_net_bps": -4.0,
            "best_daily_net_bps": 7.0,
            "annualized_daily_sharpe": 1.9,
            "mean_daily_net_bps_ci_lower": 0.2,
            "mean_daily_net_bps_ci_upper": 3.2,
            "bootstrap_probability_mean_positive": 0.98,
        },
        "terminal_auc": {"long": 0.60, "short": 0.58},
        "terminal_brier": {"long": 0.20, "short": 0.21},
        "dataset_summary": {
            "trade_feature_embargo_ms": 1_000,
            "reference_order_notional_quote": 1_000.0,
            "max_l1_participation": 0.10,
            "decision_cadence_seconds": 5,
            "source_evidence": {
                "build_id": "c" * 64,
                "schema_version": "binance-usdm-tick-v4",
                "feature_build_version": "book-ticker-event-time-v1",
                "availability_clock": "event_time_ms",
                "symbol": "BTCUSDT",
                "status": "complete",
                "is_current": True,
                "manifest_fingerprint": "d" * 64,
                "source_archive_count": 365,
                "source_manifest_rows": 1_000_000,
                "source_raw_rows": 1_000_000,
                "feature_rows": 100_000,
                "first_feature_second_ms": 1_700_000_000_000,
                "last_feature_second_ms": 1_731_535_999_000,
                "duplicate_seconds": 0,
                "invalid_feature_rows": 0,
                "quote_update_sum": 1_000_000,
                "verified": True,
                "manifest_current": True,
            }
        },
        "threshold_policy": {
            "minimum_predicted_edge_bps": 1.0,
            "minimum_profitable_probability": 0.60,
            "selection_utility_bps": 1.0,
        },
        "probability_calibration": {"long": [1.0, 0.0], "short": [1.0, 0.0]},
        "model_strings": {
            "long_probability": "0.80",
            "long_win_magnitude": "4.0",
            "long_loss_magnitude": "1.0",
            "short_probability": "0.70",
            "short_win_magnitude": "3.0",
            "short_loss_magnitude": "2.0",
        },
    }


def test_promoted_runtime_scorer_loads_exact_contract_and_selects_best_side(
    tmp_path, monkeypatch
) -> None:
    class _Booster:
        def __init__(self, *, model_str: str) -> None:
            self.value = float(model_str)

        def predict(self, rows):
            return np.full(len(rows), self.value, dtype=np.float64)

    monkeypatch.setattr("simple_ai_trading.microstructure_model.lgb.Booster", _Booster)
    path = tmp_path / "microstructure-model.json"
    path.write_text(json.dumps(_runtime_artifact_payload()), encoding="utf-8")

    scorer = load_microstructure_action_scorer(path)
    features = np.zeros(len(MICROSTRUCTURE_FEATURE_NAMES), dtype=np.float32)
    result = scorer.score(
        features,
        decision_time_ms=10_000,
        order_notional_quote=1_000.0,
        close_bid=50_000.0,
        close_ask=50_001.0,
        close_bid_qty=1.0,
        close_ask_qty=1.0,
    )
    oversized = scorer.score(
        features,
        decision_time_ms=10_000,
        order_notional_quote=1_001.0,
        close_bid=50_000.0,
        close_ask=50_001.0,
        close_bid_qty=1.0,
        close_ask_qty=1.0,
    )

    assert scorer.symbol == "BTCUSDT"
    assert len(scorer.artifact_sha256) == 64
    assert result.side == "LONG"
    assert result.long_expected_net_bps == pytest.approx(3.0)
    assert result.short_expected_net_bps == pytest.approx(1.5)
    assert result.long_profitable_probability == pytest.approx(0.8)
    assert result.long_l1_participation < 0.10
    assert oversized.side == "FLAT"
    assert oversized.reason == "order_notional_exceeds_validated_reference"
    with pytest.raises(ValueError, match="violates decision cadence"):
        scorer.score(
            features,
            decision_time_ms=11_000,
            order_notional_quote=1_000.0,
            close_bid=50_000.0,
            close_ask=50_001.0,
            close_bid_qty=1.0,
            close_ask_qty=1.0,
        )


def test_runtime_scorer_rejects_unpromoted_or_drifted_artifacts(tmp_path) -> None:
    rejected = tmp_path / "rejected.json"
    rejected.write_text(json.dumps(_runtime_artifact_payload(status="rejected")), encoding="utf-8")
    with pytest.raises(ValueError, match="accepted artifact"):
        load_microstructure_action_scorer(rejected)

    drifted_payload = _runtime_artifact_payload()
    drifted_payload["feature_names"] = [*MICROSTRUCTURE_FEATURE_NAMES[:-1], "drifted"]
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(drifted_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="feature names"):
        load_microstructure_action_scorer(drifted)

    unproven_payload = _runtime_artifact_payload()
    unproven_payload["dataset_summary"]["source_evidence"] = None  # type: ignore[index]
    unproven = tmp_path / "unproven.json"
    unproven.write_text(json.dumps(unproven_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source provenance"):
        load_microstructure_action_scorer(unproven)

    unprotected_payload = _runtime_artifact_payload()
    unprotected_payload["target_mode"] = "fixed_horizon"
    unprotected = tmp_path / "unprotected.json"
    unprotected.write_text(json.dumps(unprotected_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="protective-exit contract"):
        load_microstructure_action_scorer(unprotected)
