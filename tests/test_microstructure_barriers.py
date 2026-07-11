from __future__ import annotations

import duckdb
import numpy as np
import pytest

from simple_ai_trading.microstructure_barriers import (
    AdaptiveBarrierSpec,
    build_adaptive_barrier_targets,
    _evaluate_path_scenario,
    _extreme_trees,
    _first_long_cross,
    _first_short_cross,
    _scenario_ranges,
    volatility_scaled_barriers,
)
from simple_ai_trading.microstructure_features import MicrostructureDataset


def _spec(**overrides: object) -> AdaptiveBarrierSpec:
    values: dict[str, object] = {
        "horizon_seconds": 900,
        "volatility_feature_name": "realized_volatility_300s_bps",
        "stop_volatility_multiple": 1.0,
        "take_volatility_multiple": 1.5,
        "minimum_stop_bps": 18.0,
        "maximum_stop_bps": 60.0,
        "minimum_take_bps": 27.0,
        "maximum_take_bps": 90.0,
        "base_protection_delay_ms": 250,
        "stress_protection_delay_ms": 750,
        "trigger_execution_slippage_bps": 1.0,
    }
    values.update(overrides)
    return AdaptiveBarrierSpec(**values)  # type: ignore[arg-type]


def _dataset() -> MicrostructureDataset:
    rows = 3
    ones = np.ones(rows, dtype=np.float64)
    return MicrostructureDataset(
        symbol="BTCUSDT",
        feature_version="test",
        feature_names=("realized_volatility_300s_bps",),
        horizon_seconds=900,
        total_latency_ms=750,
        taker_fee_bps=5.0,
        additional_slippage_bps_per_side=1.0,
        reference_order_notional_quote=1_000.0,
        max_l1_participation=1.0,
        max_quote_age_ms=1_000,
        decision_cadence_seconds=5,
        target_mode="fixed_horizon",
        stop_loss_bps=None,
        take_profit_bps=None,
        trigger_execution_slippage_bps=None,
        path_resolution_ms=None,
        decision_time_ms=np.asarray([0, 5_000, 10_000], dtype=np.int64),
        long_exit_time_ms=np.asarray([900_750, 905_750, 910_750], dtype=np.int64),
        short_exit_time_ms=np.asarray([900_750, 905_750, 910_750], dtype=np.int64),
        features=np.asarray([[0.5], [1.0], [3.0]], dtype=np.float32),
        long_net_bps=-10.0 * ones,
        short_net_bps=-10.0 * ones,
        entry_spread_bps=ones,
        exit_spread_bps=ones,
        entry_quote_age_ms=np.zeros(rows, dtype=np.int64),
        exit_quote_age_ms=np.zeros(rows, dtype=np.int64),
        entry_bid_price=100.0 * ones,
        entry_ask_price=100.1 * ones,
        fixed_exit_bid_price=100.0 * ones,
        fixed_exit_ask_price=100.1 * ones,
        entry_bid_qty=10.0 * ones,
        entry_ask_qty=10.0 * ones,
        fixed_exit_bid_qty=10.0 * ones,
        fixed_exit_ask_qty=10.0 * ones,
        long_l1_participation=0.01 * ones,
        short_l1_participation=0.01 * ones,
        long_liquidity_eligible=np.ones(rows, dtype=bool),
        short_liquidity_eligible=np.ones(rows, dtype=bool),
    )


def test_adaptive_barriers_are_causal_bounded_and_take_exceeds_stop() -> None:
    stop, take = volatility_scaled_barriers(
        _dataset(), np.asarray([0, 1, 2], dtype=np.int64), _spec()
    )

    np.testing.assert_allclose(stop, [18.0, 30.0, 60.0])
    np.testing.assert_allclose(take, [27.0, 45.0, 90.0])
    assert np.all(take > stop)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"take_volatility_multiple": 0.5}, "price bounds"),
        ({"base_protection_delay_ms": -1}, "protection delays"),
        ({"path_resolution_ms": 1_000}, "path resolution"),
        ({"same_utc_day_exit": False}, "UTC day"),
    ],
)
def test_adaptive_barrier_spec_rejects_incoherent_contracts(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _spec(**overrides)


def test_segment_tree_returns_first_cross_for_each_side() -> None:
    min_bid = np.asarray([100.0, 99.9, 99.0, 98.0])
    max_bid = np.asarray([100.1, 100.3, 101.0, 102.0])
    min_ask = min_bid + 0.1
    max_ask = max_bid + 0.1
    size, tree_min_bid, tree_max_bid, tree_min_ask, tree_max_ask = _extreme_trees(
        min_bid, max_bid, min_ask, max_ask
    )

    assert _first_long_cross(tree_min_bid, tree_max_bid, size, 0, 4, 99.5, 100.8) == 2
    assert _first_short_cross(tree_min_ask, tree_max_ask, size, 0, 4, 100.7, 99.4) == 2
    assert _first_long_cross(tree_min_bid, tree_max_bid, size, 0, 2, 90.0, 110.0) == -1


def test_scenario_range_excludes_bucket_ending_after_horizon() -> None:
    protected, end, gap, valid = _scenario_ranges(
        np.asarray([700, 800, 60_600, 60_700], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        total_latency_ms=750,
        protection_delay_ms=0,
        horizon_seconds=60,
        max_quote_age_ms=1_000,
        check_gap=True,
    )

    assert protected.tolist() == [1]
    assert gap.tolist() == [1]
    assert end.tolist() == [3]
    assert valid.tolist() == [True]


def _scenario(
    *,
    min_bid: np.ndarray,
    max_bid: np.ndarray,
    close_bid: np.ndarray,
    min_ask: np.ndarray,
    max_ask: np.ndarray,
    close_ask: np.ndarray,
    protected_start: int,
    gap_start: int,
    check_gap: bool,
    adverse: bool,
) -> tuple[np.ndarray, ...]:
    times = np.arange(len(min_bid), dtype=np.int64) * 100 + 1_000
    tree = _extreme_trees(min_bid, max_bid, min_ask, max_ask)
    return _evaluate_path_scenario(
        path_times_ms=times,
        min_bid=min_bid,
        max_bid=max_bid,
        close_bid=close_bid,
        min_ask=min_ask,
        max_ask=max_ask,
        close_ask=close_ask,
        tree_size=tree[0],
        tree_min_bid=tree[1],
        tree_max_bid=tree[2],
        tree_min_ask=tree[3],
        tree_max_ask=tree[4],
        protected_start_indexes=np.asarray([protected_start]),
        end_indexes=np.asarray([len(times)]),
        gap_start_indexes=np.asarray([gap_start]),
        check_protection_gap=check_gap,
        entry_bid=np.asarray([100.0]),
        entry_ask=np.asarray([100.1]),
        fixed_exit_bid=np.asarray([100.4]),
        fixed_exit_ask=np.asarray([100.5]),
        fixed_long_exit_time_ms=np.asarray([2_000]),
        fixed_short_exit_time_ms=np.asarray([2_000]),
        stop_bps=np.asarray([50.0]),
        take_bps=np.asarray([80.0]),
        cost_bps_per_side=6.0,
        trigger_slippage_fraction=0.0001,
        adverse_fill=adverse,
    )


def test_same_bucket_stop_and_take_is_adverse_first() -> None:
    result = _scenario(
        min_bid=np.asarray([100.0, 99.0, 100.0]),
        max_bid=np.asarray([100.2, 101.5, 100.2]),
        close_bid=np.asarray([100.1, 101.0, 100.1]),
        min_ask=np.asarray([100.1, 99.1, 100.1]),
        max_ask=np.asarray([100.3, 101.6, 100.3]),
        close_ask=np.asarray([100.2, 101.1, 100.2]),
        protected_start=0,
        gap_start=0,
        check_gap=False,
        adverse=False,
    )

    assert result[4].tolist() == [3]
    assert result[5].tolist() == [3]
    assert result[0][0] < 0.0
    assert result[1][0] < 0.0


def test_stress_scenario_counts_unprotected_stop_breach() -> None:
    result = _scenario(
        min_bid=np.asarray([100.0, 99.0, 100.2, 100.3]),
        max_bid=np.asarray([100.2, 100.3, 100.4, 100.5]),
        close_bid=np.asarray([100.1, 100.1, 100.3, 100.4]),
        min_ask=np.asarray([100.1, 100.0, 100.3, 100.4]),
        max_ask=np.asarray([100.3, 101.0, 100.5, 100.6]),
        close_ask=np.asarray([100.2, 100.2, 100.4, 100.5]),
        protected_start=2,
        gap_start=0,
        check_gap=True,
        adverse=True,
    )

    assert result[4].tolist() == [4]
    assert result[5].tolist() == [4]
    assert result[2].tolist() == [1_200]
    assert result[3].tolist() == [1_200]


def test_no_barrier_hit_uses_fixed_horizon_exit() -> None:
    result = _scenario(
        min_bid=np.asarray([100.0, 100.0, 100.0]),
        max_bid=np.asarray([100.2, 100.2, 100.2]),
        close_bid=np.asarray([100.1, 100.1, 100.1]),
        min_ask=np.asarray([100.1, 100.1, 100.1]),
        max_ask=np.asarray([100.3, 100.3, 100.3]),
        close_ask=np.asarray([100.2, 100.2, 100.2]),
        protected_start=0,
        gap_start=0,
        check_gap=False,
        adverse=False,
    )

    assert result[4].tolist() == [0]
    assert result[5].tolist() == [0]
    assert result[2].tolist() == [2_000]
    assert result[3].tolist() == [2_000]


def test_database_wrapper_separates_base_fill_from_protection_gap_stress() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE current_book_ticker_100ms (
            symbol VARCHAR, bucket_ms BIGINT, min_bid DOUBLE, max_bid DOUBLE,
            close_bid DOUBLE, min_ask DOUBLE, max_ask DOUBLE, close_ask DOUBLE
        )
        """
    )
    connection.executemany(
        "INSERT INTO current_book_ticker_100ms VALUES ('BTCUSDT', ?, ?, ?, ?, ?, ?, ?)",
        [
            (700, 100.0, 100.2, 100.1, 100.1, 100.3, 100.2),
            (800, 99.0, 100.2, 100.1, 100.1, 101.0, 100.2),
            (1_000, 100.0, 100.2, 100.1, 100.1, 100.3, 100.2),
            (1_500, 100.0, 100.2, 100.1, 100.1, 100.3, 100.2),
            (2_000, 100.0, 101.0, 100.8, 99.0, 100.3, 99.4),
            (60_700, 100.3, 100.5, 100.4, 100.4, 100.6, 100.5),
        ],
    )

    class _Warehouse:
        def connect(self):
            return connection

    ones = np.ones(1, dtype=np.float64)
    dataset = MicrostructureDataset(
        symbol="BTCUSDT",
        feature_version="test",
        feature_names=("realized_volatility_300s_bps",),
        horizon_seconds=60,
        total_latency_ms=750,
        taker_fee_bps=5.0,
        additional_slippage_bps_per_side=1.0,
        reference_order_notional_quote=1_000.0,
        max_l1_participation=1.0,
        max_quote_age_ms=1_000,
        decision_cadence_seconds=1,
        target_mode="fixed_horizon",
        stop_loss_bps=None,
        take_profit_bps=None,
        trigger_execution_slippage_bps=None,
        path_resolution_ms=None,
        decision_time_ms=np.asarray([0], dtype=np.int64),
        long_exit_time_ms=np.asarray([60_750], dtype=np.int64),
        short_exit_time_ms=np.asarray([60_750], dtype=np.int64),
        features=np.asarray([[1.0]], dtype=np.float32),
        long_net_bps=-10.0 * ones,
        short_net_bps=-10.0 * ones,
        entry_spread_bps=ones,
        exit_spread_bps=ones,
        entry_quote_age_ms=np.zeros(1, dtype=np.int64),
        exit_quote_age_ms=np.zeros(1, dtype=np.int64),
        entry_bid_price=100.0 * ones,
        entry_ask_price=100.1 * ones,
        fixed_exit_bid_price=100.4 * ones,
        fixed_exit_ask_price=100.5 * ones,
        entry_bid_qty=10.0 * ones,
        entry_ask_qty=10.0 * ones,
        fixed_exit_bid_qty=10.0 * ones,
        fixed_exit_ask_qty=10.0 * ones,
        long_l1_participation=0.01 * ones,
        short_l1_participation=0.01 * ones,
        long_liquidity_eligible=np.ones(1, dtype=bool),
        short_liquidity_eligible=np.ones(1, dtype=bool),
    )
    spec = _spec(
        horizon_seconds=60,
        minimum_stop_bps=20.0,
        maximum_stop_bps=60.0,
        minimum_take_bps=30.0,
        maximum_take_bps=90.0,
    )
    try:
        targets = build_adaptive_barrier_targets(
            _Warehouse(),  # type: ignore[arg-type]
            dataset,
            np.asarray([0], dtype=np.int64),
            spec,
        )
    finally:
        connection.close()

    assert targets.valid.tolist() == [True]
    assert targets.base_long_outcome.tolist() == [2]
    assert targets.stress_long_outcome.tolist() == [4]
    assert targets.base_long_net_bps[0] > 0.0
    assert targets.stress_long_net_bps[0] < 0.0
    assert targets.trading_authority is False
    assert targets.summary()["stress"]["long_outcomes"]["protection_gap_stop"] == 1
