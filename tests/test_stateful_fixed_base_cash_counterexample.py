from __future__ import annotations

from datetime import UTC, datetime
from fractions import Fraction

import numpy as np
import pytest

from simple_ai_trading.stateful_turnover_model import (
    StatefulHourlyDataset,
    replay_always_long,
)


def test_hourly_return_sum_does_not_prove_fixed_base_holding_cash() -> None:
    prices = (Fraction(100), Fraction(200), Fraction(100))
    hourly_bps = np.array(
        [
            float((exit_price / entry_price - 1) * 10_000)
            for entry_price, exit_price in zip(prices[:-1], prices[1:], strict=True)
        ]
    )
    start = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1000)
    features = np.zeros((6, 1), dtype=np.float32)
    dataset = StatefulHourlyDataset(
        feature_names=("synthetic",),
        baseline_features=features,
        augmented_features=features,
        decision_time_ms=np.repeat(start + np.arange(2) * 3_600_000, 3),
        symbol_index=np.tile(np.arange(3), 2),
        signed_pre_transition_utility_bps=np.repeat(hourly_bps, 3),
        funding_cash_flow_bps=np.zeros(6),
        source_evidence=None,
        dataset_sha256="synthetic_no_market_evidence",
    )
    replay = replay_always_long(dataset, cost_scenario="synthetic", cost_bps=6, seed=1)
    assert np.sum(replay.portfolio_return_bps) == pytest.approx(4_988)
    fixed_quantity = Fraction(1)
    fixed_base_gross = fixed_quantity * (prices[-1] - prices[0])
    assert fixed_base_gross == 0
    assert sum(Fraction(str(value)) for value in hourly_bps) == 5_000
    # Keeping the original 100-quote notional requires an uncharged interior trade.
    rebalance_quantity = prices[0] / prices[1] - fixed_quantity
    assert rebalance_quantity == Fraction(-1, 2)
    assert replay.transition_units[0, 0] == replay.transition_units[-1, 0] == 1
    assert replay.metrics["transition_units"] == 6
