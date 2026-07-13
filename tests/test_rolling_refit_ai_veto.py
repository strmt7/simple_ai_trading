from __future__ import annotations

from datetime import UTC, datetime
import json

import numpy as np

from simple_ai_trading.derivatives_hurdle_data import DerivativesHurdleDataset
from simple_ai_trading.derivatives_hurdle_model import (
    ActionReplayMetrics,
    ReplayOutcome,
)
from simple_ai_trading.rolling_refit_ai_veto import (
    _decision_schema,
    _parse_batch_response,
    build_rolling_ai_cases,
    rolling_case_set_sha256,
)
from simple_ai_trading.rolling_refit_model import RollingSupportCandidate


FEATURE_NAMES = (
    "target_return_5m_bps",
    "target_return_15m_bps",
    "target_return_60m_bps",
    "target_realized_volatility_60m_bps",
    "target_realized_volatility_240m_bps",
    "target_intrabar_range_bps",
    "target_path_efficiency_60m",
    "target_quote_volume_vs_60m_mean",
    "target_trade_count_vs_60m_mean",
    "target_signed_taker_flow_15m",
    "target_signed_taker_flow_60m",
    "target_return_zscore_240m",
    "target_beta_residual_return_60m_bps",
    "cross_asset_return_dispersion_15m_bps",
    "cross_asset_taker_flow_mean",
    "cross_asset_taker_flow_agreement",
    "target_to_btc_volatility_ratio_60m",
    "target_same_minute_of_week_liquidity_ratio",
    "target_premium_close_bps",
    "target_premium_zscore_240m",
    "target_premium_age_minutes",
    "target_premium_observed_fraction_240m",
    "cross_asset_premium_dispersion_bps",
    "target_last_settled_funding_rate_bps",
    "target_funding_interval_hours",
    "target_minutes_since_funding",
    "target_settled_funding_sum_24h_bps",
    "target_settled_funding_sum_168h_bps",
    "target_funding_event_zscore_30",
    "cross_asset_funding_dispersion_bps",
    "weekend_flag",
)


def _ms(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1000)


def _metrics(trades: int) -> ActionReplayMetrics:
    return ActionReplayMetrics(
        maximum_action_probability=None,
        direction_probability_margin=None,
        total_trades=trades,
        trades_by_symbol={"BTCUSDT": trades, "ETHUSDT": 0, "SOLUSDT": 0},
        maximum_single_symbol_fraction=1.0,
        active_utc_days=trades,
        total_net_bps=0.0,
        mean_net_bps=0.0,
        median_net_bps=0.0,
        positive_rate=0.0,
        profit_factor=0.0,
        median_monthly_net_bps=0.0,
        negative_month_fraction=1.0,
        maximum_peak_to_trough_drawdown_bps=0.0,
        longest_loss_streak=0,
        total_funding_cash_flow_bps=0.0,
        mean_funding_cash_flow_bps=0.0,
        day_block_bootstrap_mean_net_bps_lower_95=None,
        day_block_bootstrap_mean_net_bps_median=None,
        day_block_bootstrap_mean_net_bps_upper_95=None,
        candidate_rows=trades,
        overlap_rejections=0,
        nonfinite_outcomes=0,
    )


def _candidate(
    rows: np.ndarray,
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    candidate_id: str,
) -> RollingSupportCandidate:
    months = {f"2025-{month:02d}": (0.5, 0.1) for month in range(1, 7)}
    return RollingSupportCandidate(
        candidate_id=candidate_id,
        architecture="per_symbol_direct_multiclass_lightgbm",
        weighting="equal",
        horizon_minutes=30,
        probabilities=probabilities,
        monthly_calibration_masks={
            month: np.zeros(probabilities.shape[0], dtype=bool) for month in months
        },
        monthly_thresholds=months,
        evaluation=ReplayOutcome(
            metrics=_metrics(rows.size),
            selected_indices=rows,
            selected_direction=np.ones(rows.size, dtype=np.int8),
            net_return_bps=outcomes,
        ),
    )


def test_case_builder_keeps_one_highest_confidence_action_per_symbol_day() -> None:
    times = np.asarray(
        [
            _ms("2025-01-01T00:00:00"),
            _ms("2025-01-01T05:00:00"),
            _ms("2025-01-02T00:00:00"),
        ],
        dtype=np.int64,
    )
    dataset = DerivativesHurdleDataset(
        feature_names=FEATURE_NAMES,
        price_flow_feature_count=len(FEATURE_NAMES),
        features=np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32),
        decision_time_ms=times,
        symbol_index=np.asarray([0, 0, 1], dtype=np.int8),
        target_class={30: np.asarray([2, 2, 2], dtype=np.int8)},
        long_net_utility_bps={30: np.asarray([10, -5, 20], dtype=np.float32)},
        short_net_utility_bps={30: np.asarray([-5, 10, -5], dtype=np.float32)},
        funding_cash_flow_bps={30: np.zeros(3, dtype=np.float32)},
        role_masks={},
        source_evidence=None,  # type: ignore[arg-type]
        source_exclusions={},
    )
    probabilities = np.asarray(
        [[0.05, 0.15, 0.80], [0.02, 0.08, 0.90], [0.05, 0.10, 0.85]],
        dtype=np.float32,
    )
    candidate = _candidate(
        np.asarray([0, 1, 2]),
        probabilities,
        np.asarray([10.0, -5.0, 20.0]),
        "candidate",
    )
    cases = build_rolling_ai_cases(dataset, (candidate,))

    assert len(cases) == 2
    assert [case.dataset_row for case in cases] == [1, 2]
    assert cases[0].relative_day_index == 0
    assert cases[1].relative_day_index == 1
    assert "outcome_net_bps" not in json.dumps(cases[0].identity_payload())
    assert len(rolling_case_set_sha256(cases)) == 64


def test_batch_response_requires_exact_case_identity_and_valid_decisions() -> None:
    response = {
        "message": {
            "content": json.dumps(
                {
                    "decisions": [
                        {
                            "case_id": "a",
                            "action": "approve",
                            "risk_percent": 50,
                            "confidence_percent": 80,
                            "reason_codes": ["cost_ok"],
                        },
                        {
                            "case_id": "b",
                            "action": "veto",
                            "risk_percent": 100,
                            "confidence_percent": 70,
                            "reason_codes": ["weak_edge"],
                        },
                    ]
                }
            )
        }
    }
    decisions = _parse_batch_response(response, ("a", "b"))

    assert set(decisions) == {"a", "b"}
    assert decisions["a"].action == "approve"
    assert decisions["a"].risk_multiplier == 0.5
    assert decisions["b"].action == "veto"
    assert decisions["b"].risk_multiplier == 0.0
    assert _decision_schema()["properties"]["decisions"]["type"] == "array"


def test_case_builder_retains_ten_highest_confidence_cases_per_symbol_month() -> None:
    rows = 12
    times = np.asarray(
        [_ms(f"2025-01-{day:02d}T00:00:00") for day in range(1, rows + 1)],
        dtype=np.int64,
    )
    probabilities = np.column_stack(
        (
            np.full(rows, 0.05),
            np.linspace(0.44, 0.33, rows),
            np.linspace(0.51, 0.62, rows),
        )
    ).astype(np.float32)
    dataset = DerivativesHurdleDataset(
        feature_names=FEATURE_NAMES,
        price_flow_feature_count=len(FEATURE_NAMES),
        features=np.zeros((rows, len(FEATURE_NAMES)), dtype=np.float32),
        decision_time_ms=times,
        symbol_index=np.zeros(rows, dtype=np.int8),
        target_class={30: np.full(rows, 2, dtype=np.int8)},
        long_net_utility_bps={30: np.arange(rows, dtype=np.float32)},
        short_net_utility_bps={30: np.full(rows, -5.0, dtype=np.float32)},
        funding_cash_flow_bps={30: np.zeros(rows, dtype=np.float32)},
        role_masks={},
        source_evidence=None,  # type: ignore[arg-type]
        source_exclusions={},
    )
    candidate = _candidate(
        np.arange(rows, dtype=np.int64),
        probabilities,
        np.arange(rows, dtype=np.float64),
        "candidate",
    )

    cases = build_rolling_ai_cases(dataset, (candidate,))

    assert len(cases) == 10
    assert [case.dataset_row for case in cases] == list(range(2, 12))
