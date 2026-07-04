from __future__ import annotations

from simple_ai_trading.execution_profiles import (
    execution_profile_from_top_of_book,
    load_top_of_book_execution_profile,
)
from simple_ai_trading.market_store import MarketDataStore, TopOfBookSnapshot
from simple_ai_trading.types import StrategyConfig


def _snapshot(*, spread_bps: float = 2.0, depth: float = 100_000.0) -> TopOfBookSnapshot:
    return TopOfBookSnapshot(
        symbol="BTCUSDC",
        market_type="spot",
        provider="binance",
        ts_ms=1_700_000_000_000,
        bid_price=99.99,
        bid_qty=depth / 200.0,
        ask_price=100.01,
        ask_qty=depth / 200.0,
        mid_price=100.0,
        spread=0.02,
        spread_bps=spread_bps,
        depth_notional=depth,
        ingested_at_ms=1_700_000_000_010,
    )


def test_execution_profile_from_top_of_book_scores_spread_and_depth() -> None:
    strategy = StrategyConfig(
        max_spread_bps=5.0,
        min_quote_volume_usdc=50_000_000.0,
        latency_buffer_ms=900,
        testnet_liquidity_haircut=0.4,
    )

    liquid = execution_profile_from_top_of_book(_snapshot(spread_bps=1.0, depth=200_000.0), strategy)
    thin = execution_profile_from_top_of_book(_snapshot(spread_bps=12.0, depth=1_000.0), strategy)

    assert liquid.symbol == "BTCUSDC"
    assert liquid.latency_ms == 900
    assert liquid.liquidity_haircut == 0.4
    assert liquid.liquidity_score > thin.liquidity_score
    assert thin.spread_bps == 12.0


def test_load_top_of_book_execution_profile_uses_latest_snapshot(tmp_path) -> None:
    db = tmp_path / "market.sqlite"
    with MarketDataStore(db) as store:
        store.insert_top_of_book_snapshot(
            "binance",
            "BTCUSDC",
            "spot",
            {"bidPrice": "100.0", "bidQty": "10", "askPrice": "100.05", "askQty": "8"},
            ts_ms=1_700_000_000_000,
            ingested_at_ms=1_700_000_000_010,
        )
        store.insert_top_of_book_snapshot(
            "binance",
            "BTCUSDC",
            "spot",
            {"bidPrice": "100.0", "bidQty": "50", "askPrice": "100.02", "askQty": "50"},
            ts_ms=1_700_000_060_000,
            ingested_at_ms=1_700_000_060_010,
        )

    evidence = load_top_of_book_execution_profile(
        db,
        symbol="BTCUSDC",
        market_type="spot",
        strategy=StrategyConfig(),
        now_ms=1_700_000_060_500,
    )

    assert evidence.profile is not None
    assert evidence.source == "top_of_book:binance"
    assert evidence.snapshot_ts_ms == 1_700_000_060_000
    assert evidence.snapshot_age_ms == 500
    assert evidence.depth_notional == evidence.profile.quote_volume
    assert evidence.warning is None


def test_load_top_of_book_execution_profile_reports_missing_and_stale(tmp_path) -> None:
    missing = load_top_of_book_execution_profile(
        tmp_path / "missing.sqlite",
        symbol="BTCUSDC",
        market_type="spot",
        strategy=StrategyConfig(),
    )
    assert missing.profile is None
    assert "not found" in str(missing.warning)

    db = tmp_path / "market.sqlite"
    with MarketDataStore(db) as store:
        store.insert_top_of_book_snapshot(
            "binance",
            "BTCUSDC",
            "spot",
            {"bidPrice": "100.0", "bidQty": "1", "askPrice": "100.10", "askQty": "1"},
            ts_ms=1_700_000_000_000,
            ingested_at_ms=1_700_000_000_010,
        )

    stale = load_top_of_book_execution_profile(
        db,
        symbol="BTCUSDC",
        market_type="spot",
        strategy=StrategyConfig(),
        now_ms=1_700_000_000_000 + 25 * 60 * 60 * 1000,
    )
    assert stale.profile is not None
    assert "stale" in str(stale.warning)
