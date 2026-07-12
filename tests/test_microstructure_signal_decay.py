from __future__ import annotations

import duckdb
import numpy as np
import pytest

from simple_ai_trading.microstructure_features import (
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_VERSION,
    MicrostructureDataset,
)
from simple_ai_trading.microstructure_signal_decay import (
    AsOfBboQuotes,
    build_horizon_path,
    chronological_nonoverlapping_mask,
    daily_direction_metrics,
    direction_metrics,
    exact_horizon_rows,
    linear_cross_spread_cash_returns_bps,
    load_bbo_quotes_asof,
    placebo_summary,
    placebo_weighted_auc_distribution,
    ranked_event_outcomes,
    routed_cost_metrics,
    weighted_roc_auc,
)


_DAY_MS = 86_400_000


def _dataset(rows: int = 24) -> MicrostructureDataset:
    decisions = np.arange(rows, dtype=np.int64) * 5_000 + 10_000
    steps = np.where(np.arange(rows) % 2 == 0, 0.20, -0.10)
    mid = 100.0 + np.cumsum(steps)
    entry_bid = mid - 0.05
    entry_ask = mid + 0.05
    features = np.zeros(
        (rows, len(MICROSTRUCTURE_FEATURE_NAMES)),
        dtype=np.float32,
    )
    return MicrostructureDataset(
        symbol="BTCUSDT",
        feature_version=MICROSTRUCTURE_FEATURE_VERSION,
        feature_names=MICROSTRUCTURE_FEATURE_NAMES,
        horizon_seconds=300,
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
        decision_time_ms=decisions,
        long_exit_time_ms=decisions + 300_750,
        short_exit_time_ms=decisions + 300_750,
        features=features,
        long_net_bps=np.zeros(rows),
        short_net_bps=np.zeros(rows),
        entry_spread_bps=np.full(rows, 10.0),
        exit_spread_bps=np.full(rows, 10.0),
        entry_quote_age_ms=np.full(rows, 25, dtype=np.int64),
        exit_quote_age_ms=np.full(rows, 30, dtype=np.int64),
        entry_bid_price=entry_bid,
        entry_ask_price=entry_ask,
        fixed_exit_bid_price=entry_bid,
        fixed_exit_ask_price=entry_ask,
        entry_bid_qty=np.full(rows, 1_000.0),
        entry_ask_qty=np.full(rows, 1_000.0),
        fixed_exit_bid_qty=np.full(rows, 1_000.0),
        fixed_exit_ask_qty=np.full(rows, 1_000.0),
        long_l1_participation=np.full(rows, 0.01),
        short_l1_participation=np.full(rows, 0.01),
        long_liquidity_eligible=np.ones(rows, dtype=bool),
        short_liquidity_eligible=np.ones(rows, dtype=bool),
    )


def _zero_latency_quotes(dataset: MicrostructureDataset) -> AsOfBboQuotes:
    rows = dataset.rows
    return AsOfBboQuotes(
        arrival_time_ms=dataset.decision_time_ms.copy(),
        available_time_ms=dataset.decision_time_ms.copy(),
        last_transaction_time_ms=dataset.decision_time_ms.copy(),
        bid_price=dataset.entry_bid_price.copy(),
        bid_qty=dataset.entry_bid_qty.copy(),
        ask_price=dataset.entry_ask_price.copy(),
        ask_qty=dataset.entry_ask_qty.copy(),
        quote_age_ms=np.zeros(rows, dtype=np.int64),
        valid=np.ones(rows, dtype=bool),
    )


def test_exact_horizon_rows_require_exact_same_day_support() -> None:
    times = np.asarray(
        [_DAY_MS - 5_000, _DAY_MS, _DAY_MS + 5_000],
        dtype=np.int64,
    )

    source, future, exclusions = exact_horizon_rows(
        times,
        np.asarray([0, 1], dtype=np.int64),
        horizon_seconds=5,
    )

    assert source.tolist() == [1]
    assert future.tolist() == [2]
    assert exclusions == {
        "requested_event_rows": 2,
        "missing_exact_future_row": 0,
        "cross_utc_day": 1,
        "retained_rows": 1,
    }


def test_cross_spread_returns_charge_actual_entry_and_exit_notionals() -> None:
    long_gross, short_gross, long_net, short_net = linear_cross_spread_cash_returns_bps(
        np.asarray([100.0]),
        np.asarray([101.0]),
        np.asarray([102.0]),
        np.asarray([98.0]),
        execution_cost_bps_per_side=6.0,
    )
    long_ratio = 102.0 / 101.0
    short_ratio = 98.0 / 100.0

    assert long_gross[0] == pytest.approx((long_ratio - 1.0) * 10_000.0)
    assert short_gross[0] == pytest.approx((1.0 - short_ratio) * 10_000.0)
    assert long_net[0] == pytest.approx(long_gross[0] - 6.0 * (1.0 + long_ratio))
    assert short_net[0] == pytest.approx(short_gross[0] - 6.0 * (1.0 + short_ratio))


def test_weighted_auc_handles_ties_and_class_weights_exactly() -> None:
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    weights = np.asarray([1.0, 2.0, 3.0, 4.0])

    assert weighted_roc_auc(labels, labels.astype(float), weights) == 1.0
    assert weighted_roc_auc(labels, np.zeros(4), weights) == 0.5
    assert (
        weighted_roc_auc(
            np.zeros(4, dtype=np.int8),
            np.arange(4, dtype=float),
            weights,
        )
        is None
    )


def test_read_only_asof_bbo_query_rejects_stale_quotes(tmp_path) -> None:
    del tmp_path
    connection = duckdb.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE current_book_ticker_100ms (
            symbol VARCHAR, bucket_ms BIGINT, close_bid DOUBLE,
            close_bid_qty DOUBLE, close_ask DOUBLE, close_ask_qty DOUBLE,
            last_transaction_time_ms BIGINT, available_time_ms BIGINT
        )
        """
    )
    connection.executemany(
        "INSERT INTO current_book_ticker_100ms VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("BTCUSDT", 1_000, 99.0, 10.0, 100.0, 10.0, 1_400, 1_400),
            ("BTCUSDT", 2_000, 100.0, 11.0, 101.0, 12.0, 2_300, 2_300),
        ],
    )

    class _Warehouse:
        def connect(self):
            return connection

    try:
        quotes = load_bbo_quotes_asof(
            _Warehouse(),  # type: ignore[arg-type]
            symbol="BTCUSDT",
            arrival_time_ms=np.asarray([1_500, 2_500, 4_000], dtype=np.int64),
            maximum_quote_age_ms=1_000,
        )
    finally:
        connection.close()

    assert quotes.bid_price.tolist() == [99.0, 100.0, 100.0]
    assert quotes.ask_price.tolist() == [100.0, 101.0, 101.0]
    assert quotes.quote_age_ms.tolist() == [100, 200, 1_700]
    assert quotes.valid.tolist() == [True, True, False]


def test_horizon_path_metrics_are_complete_and_placebo_is_reproducible() -> None:
    dataset = _dataset()
    endpoints = np.arange(dataset.rows - 1, dtype=np.int64)
    path = build_horizon_path(
        dataset,
        endpoints,
        _zero_latency_quotes(dataset),
        horizon_seconds=5,
    )
    signal = np.sign(path.delayed_midquote_return_bps)

    direction = direction_metrics(path, signal)
    daily = daily_direction_metrics(path, signal)
    costs = routed_cost_metrics(path, signal)
    ranked = ranked_event_outcomes(path, signal, requested_counts=[5, 10, 100])
    first = placebo_weighted_auc_distribution(
        path,
        signal,
        replicates=20,
        seed=3601,
    )
    second = placebo_weighted_auc_distribution(
        path,
        signal,
        replicates=20,
        seed=3601,
    )
    summary = placebo_summary(direction["weighted_roc_auc"], first)

    assert path.rows == dataset.rows - 1
    assert direction["weighted_roc_auc"] == 1.0
    assert direction["weighted_direction_accuracy"] == 1.0
    assert len(daily) == 1
    assert np.all(chronological_nonoverlapping_mask(path))
    assert costs["routed_rows"] == path.rows
    assert costs["delayed_l1_eligible_rows"] == path.rows
    assert costs["zero_latency_comparable_rows"] == path.rows
    assert ranked[-1]["actual_rows"] == path.rows
    assert all(item["event_outcomes_not_executable_trades"] for item in ranked)
    assert np.array_equal(first, second)
    assert summary["replicates"] == 20
    assert summary["formal_multiple_testing_significance_claim"] is False
