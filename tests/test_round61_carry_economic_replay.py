from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tools.run_round61_carry_economic_replay import (
    _archive_checksum_verified,
    _capacity_check,
    _risk_metrics,
    _score_episode,
    _source_gaps,
    _symbol_gate,
)


MINUTE_MS = 60_000


def _execution_row(
    *,
    high: float,
    low: float,
    close: float,
    quote: float = 4_000_000.0,
    taker_buy_quote: float = 2_000_000.0,
) -> dict[str, object]:
    return {
        "high": high,
        "low": low,
        "close": close,
        "quote_volume": quote,
        "taker_buy_quote_volume": taker_buy_quote,
    }


def _contracts() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    return (
        {
            "target_spot_entry_notional_usdt": 10_000.0,
            "committed_capital_usdt": 20_000.0,
        },
        {"maximum_same_side_one_minute_taker_participation": 0.01},
        {
            "spot_taker_fee_bps_per_fill": 10.0,
            "futures_taker_fee_bps_per_fill": 4.0,
            "additional_operational_slippage_bps_per_fill": 1.0,
        },
    )


def test_archive_checksum_validation_uses_certificate_field_names() -> None:
    row = {
        "archive_sha256": "a" * 64,
        "checksum_sha256": "a" * 64,
        "checksum_status": "verified",
    }

    assert _archive_checksum_verified(row) is True
    row["checksum_sha256"] = "b" * 64
    assert _archive_checksum_verified(row) is False


def test_capacity_check_handles_zero_flow_and_exact_boundary() -> None:
    unavailable = _capacity_check(
        name="entry",
        fill_notional_usdt=10_000.0,
        available_quote_usdt=0.0,
        maximum_participation=0.01,
    )
    boundary = _capacity_check(
        name="entry",
        fill_notional_usdt=10_000.0,
        available_quote_usdt=1_000_000.0,
        maximum_participation=0.01,
    )

    assert unavailable["participation_fraction"] is None
    assert unavailable["passed"] is False
    assert boundary["participation_fraction"] == pytest.approx(0.01)
    assert boundary["passed"] is True


def test_capacity_check_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="capacity inputs"):
        _capacity_check(
            name="entry",
            fill_notional_usdt=-1.0,
            available_quote_usdt=1.0,
            maximum_participation=0.01,
        )


def test_episode_accounting_uses_matched_base_actual_notionals_and_adverse_marks() -> (
    None
):
    position, capacity, costs = _contracts()
    episode = {
        "decision_time_ms": 0,
        "end_time_ms": MINUTE_MS,
        "future_funding_calc_times_ms": [30_001],
    }
    result = _score_episode(
        episode,
        spot={
            0: _execution_row(high=100.0, low=98.0, close=99.0),
            MINUTE_MS: _execution_row(high=107.0, low=105.0, close=106.0),
        },
        futures={
            0: _execution_row(high=103.0, low=101.0, close=102.0),
            MINUTE_MS: _execution_row(high=103.0, low=102.0, close=102.5),
        },
        marks={0: {"high": 103.0, "low": 102.0, "close": 102.5}},
        funding={30_001: {"funding_rate": 0.001}},
        position=position,
        capacity=capacity,
        costs=costs,
    )

    assert result["capacity_eligible"] is True
    assert result["economically_scored"] is True
    assert result["base_quantity"] == pytest.approx(100.0)
    assert result["spot_pnl_usdt"] == pytest.approx(500.0)
    assert result["perpetual_pnl_usdt"] == pytest.approx(-200.0)
    assert result["short_funding_pnl_usdt"] == pytest.approx(10.2)
    assert result["gross_pnl_usdt"] == pytest.approx(310.2)
    assert result["exchange_taker_fees_usdt"] == pytest.approx(28.66)
    assert result["additional_operational_slippage_usdt"] == pytest.approx(4.09)
    assert result["stress_net_pnl_usdt"] == pytest.approx(277.45)
    assert result["stress_net_committed_capital_bps"] == pytest.approx(138.725)


def test_negative_funding_uses_adverse_mark_high() -> None:
    position, capacity, costs = _contracts()
    episode = {
        "decision_time_ms": 0,
        "end_time_ms": MINUTE_MS,
        "future_funding_calc_times_ms": [30_001],
    }
    rows = {
        0: _execution_row(high=100.0, low=99.0, close=99.5),
        MINUTE_MS: _execution_row(high=101.0, low=100.0, close=100.5),
    }
    result = _score_episode(
        episode,
        spot=rows,
        futures=rows,
        marks={0: {"high": 103.0, "low": 102.0, "close": 102.5}},
        funding={30_001: {"funding_rate": -0.001}},
        position=position,
        capacity=capacity,
        costs=costs,
    )

    settlement = result["funding_settlements"][0]
    assert result["capacity_eligible"] is True
    assert settlement["adverse_mark_price"] == 103.0
    assert settlement["short_funding_pnl_usdt"] == pytest.approx(-10.3)


def test_capacity_ineligible_episode_is_not_economically_scored() -> None:
    position, capacity, costs = _contracts()
    episode = {
        "decision_time_ms": 0,
        "end_time_ms": MINUTE_MS,
        "future_funding_calc_times_ms": [30_001],
    }
    constrained = _execution_row(
        high=100.0,
        low=99.0,
        close=99.5,
        quote=100.0,
        taker_buy_quote=50.0,
    )
    result = _score_episode(
        episode,
        spot={0: constrained, MINUTE_MS: constrained},
        futures={0: constrained, MINUTE_MS: constrained},
        marks={0: {"high": 103.0, "low": 102.0, "close": 102.5}},
        funding={30_001: {"funding_rate": 0.001}},
        position=position,
        capacity=capacity,
        costs=costs,
    )

    assert result["capacity_eligible"] is False
    assert result["economically_scored"] is False
    assert "stress_net_pnl_usdt" not in result


def test_source_gap_excludes_episode_without_filling_missing_minute() -> None:
    episode = {
        "decision_time_ms": 1,
        "end_time_ms": MINUTE_MS + 1,
        "future_funding_calc_times_ms": [30_001],
    }
    gaps = _source_gaps(
        episode,
        spot={0: {}},
        futures={0: {}, MINUTE_MS: {}},
        marks={0: {}},
        funding={30_001: {}},
    )

    assert gaps == [f"spot_exit:{MINUTE_MS}"]


def test_risk_metrics_use_frozen_tail_drawdown_and_utc_year_formulas() -> None:
    start = int(datetime(2023, 1, 1, tzinfo=UTC).timestamp() * 1000)
    returns = [-10.0, -5.0, *([2.0] * 9)]
    episodes = [
        {
            "decision_time_ms": start + index * 24 * 60 * 60 * 1000,
            "stress_net_committed_capital_bps": value,
            "stress_net_pnl_usdt": value * 2.0,
            "basis_pnl_usdt": value,
            "short_funding_pnl_usdt": value,
            "exchange_taker_fees_usdt": 1.0,
            "additional_operational_slippage_usdt": 1.0,
        }
        for index, value in enumerate(returns)
    ]
    uncertainty = {
        "bootstrap_samples": 100,
        "mean_block_length_episodes": 2.0,
        "confidence_lower_quantile": 0.025,
        "confidence_upper_quantile": 0.975,
    }

    first = _risk_metrics(episodes, uncertainty=uncertainty, seed=6101)
    second = _risk_metrics(episodes, uncertainty=uncertainty, seed=6101)

    assert first == second
    assert first["expected_shortfall_10pct_committed_capital_bps"] == pytest.approx(
        -7.5
    )
    assert first["maximum_sequential_drawdown_committed_capital_bps"] == pytest.approx(
        15.0
    )
    assert first["distinct_calendar_years"] == 1
    assert first["yearly_results"][0]["year"] == 2023


def test_empty_risk_population_and_undefined_positive_share_fail_gate() -> None:
    uncertainty = {
        "bootstrap_samples": 10,
        "mean_block_length_episodes": 2.0,
        "confidence_lower_quantile": 0.025,
        "confidence_upper_quantile": 0.975,
    }
    metrics = _risk_metrics([], uncertainty=uncertainty, seed=6101)
    summary = {
        "source_rows_reconciled": True,
        "source_eligible_episodes": 0,
        "source_eligible_fraction": 0.0,
        "capacity_eligible_episodes": 0,
        "capacity_eligible_fraction": 0.0,
    }
    gate = {
        "minimum_source_eligible_episodes_per_symbol": 40,
        "minimum_source_eligible_fraction_per_symbol": 0.9,
        "minimum_capacity_eligible_episodes_per_symbol": 40,
        "minimum_capacity_eligible_fraction": 0.9,
        "minimum_stress_net_positive_fraction": 0.55,
        "median_stress_net_committed_capital_bps_strictly_above": 0.0,
        "bootstrap_lower_95_mean_stress_net_committed_capital_bps_strictly_above": 0.0,
        "maximum_sequential_drawdown_committed_capital_bps": 200.0,
        "minimum_worst_episode_committed_capital_bps": -200.0,
        "minimum_expected_shortfall_10pct_committed_capital_bps": -100.0,
        "minimum_distinct_calendar_years": 3,
        "minimum_positive_calendar_year_fraction": 0.6,
        "maximum_single_year_episode_fraction": 0.5,
        "maximum_single_episode_share_of_positive_pnl": 0.35,
    }

    result = _symbol_gate(summary, metrics, gate)

    assert result["passed"] is False
    assert all(
        row["passed"] is False
        for row in result["checks"]
        if row["check_id"] != "source_rows_reconciled"
    )


def test_symbol_gate_accepts_inclusive_limits_but_requires_strict_positive_edge() -> (
    None
):
    summary = {
        "source_rows_reconciled": True,
        "source_eligible_episodes": 40,
        "source_eligible_fraction": 0.9,
        "capacity_eligible_episodes": 40,
        "capacity_eligible_fraction": 0.9,
    }
    metrics = {
        "positive_stress_net_fraction": 0.55,
        "median_stress_net_committed_capital_bps": 0.01,
        "bootstrap_lower_95_mean_stress_net_committed_capital_bps": 0.01,
        "maximum_sequential_drawdown_committed_capital_bps": 200.0,
        "worst_episode_committed_capital_bps": -200.0,
        "expected_shortfall_10pct_committed_capital_bps": -100.0,
        "distinct_calendar_years": 3,
        "positive_calendar_year_fraction": 0.6,
        "maximum_single_year_episode_fraction": 0.5,
        "maximum_single_episode_share_of_positive_pnl": 0.35,
    }
    gate = {
        "minimum_source_eligible_episodes_per_symbol": 40,
        "minimum_source_eligible_fraction_per_symbol": 0.9,
        "minimum_capacity_eligible_episodes_per_symbol": 40,
        "minimum_capacity_eligible_fraction": 0.9,
        "minimum_stress_net_positive_fraction": 0.55,
        "median_stress_net_committed_capital_bps_strictly_above": 0.0,
        "bootstrap_lower_95_mean_stress_net_committed_capital_bps_strictly_above": 0.0,
        "maximum_sequential_drawdown_committed_capital_bps": 200.0,
        "minimum_worst_episode_committed_capital_bps": -200.0,
        "minimum_expected_shortfall_10pct_committed_capital_bps": -100.0,
        "minimum_distinct_calendar_years": 3,
        "minimum_positive_calendar_year_fraction": 0.6,
        "maximum_single_year_episode_fraction": 0.5,
        "maximum_single_episode_share_of_positive_pnl": 0.35,
    }

    assert _symbol_gate(summary, metrics, gate)["passed"] is True
    metrics["median_stress_net_committed_capital_bps"] = 0.0
    assert _symbol_gate(summary, metrics, gate)["passed"] is False
