from __future__ import annotations

from decimal import Decimal

import duckdb
import pytest

from simple_ai_trading.impact_absorption import L2BookState
from simple_ai_trading.impact_absorption_store import IMPACT_CAPTURE_SYMBOLS
from simple_ai_trading.impact_absorption_target_store import (
    ROUND73_TARGET_OPTION_TABLE,
    _Anchor,
    _TargetReplay,
    _create_target_tables,
    _insert_option_batch,
    _option_from_row,
    _option_invariant_errors,
    _quantity_invariant_errors,
)
from simple_ai_trading.impact_absorption_targets import Round73MarketQuantityRules


BASE = 1_000_000_000_000
WALL = 1_780_000_000_000_000_000


def _state(
    symbol: str,
    *,
    update_id: int,
    bid: float = 99.9,
    ask: float = 100.1,
) -> L2BookState:
    bids = tuple((bid - index * 0.1, 10.0) for index in range(20))
    asks = tuple((ask + index * 0.1, 10.0) for index in range(20))

    def depth(levels: tuple[tuple[float, float], ...], count: int) -> float:
        return sum(price * quantity for price, quantity in levels[:count])

    bid5, ask5 = depth(bids, 5), depth(asks, 5)
    bid10, ask10 = depth(bids, 10), depth(asks, 10)
    bid20, ask20 = depth(bids, 20), depth(asks, 20)
    return L2BookState(
        symbol=symbol,
        update_id=update_id,
        best_bid=bid,
        best_ask=ask,
        spread_bps=(ask - bid) / ((ask + bid) / 2.0) * 10_000.0,
        mid=(ask + bid) / 2.0,
        bid_levels=bids,
        ask_levels=asks,
        bid_depth_quote_5=bid5,
        ask_depth_quote_5=ask5,
        bid_depth_quote_10=bid10,
        ask_depth_quote_10=ask10,
        bid_depth_quote_20=bid20,
        ask_depth_quote_20=ask20,
        imbalance_5=(bid5 - ask5) / (bid5 + ask5),
        imbalance_10=(bid10 - ask10) / (bid10 + ask10),
        imbalance_20=(bid20 - ask20) / (bid20 + ask20),
    )


def _rules() -> dict[str, Round73MarketQuantityRules]:
    return {
        symbol: Round73MarketQuantityRules.create(
            symbol=symbol,
            step_size="0.001",
            minimum_quantity="0.001",
            maximum_quantity="120",
            minimum_notional="5",
        )
        for symbol in IMPACT_CAPTURE_SYMBOLS
    }


def _anchors(*, include_all: bool = True) -> dict[str, list[_Anchor]]:
    return {
        symbol: (
            [
                _Anchor(
                    symbol=symbol,
                    anchor_index=0,
                    decision_monotonic_ns=BASE + 1_000_000_000,
                    decision_wall_ns=WALL + 1_000_000_000,
                    source_max_received_monotonic_ns=BASE + 900_000_000,
                )
            ]
            if include_all or symbol == "BTCUSDT"
            else []
        )
        for symbol in IMPACT_CAPTURE_SYMBOLS
    }


def _replay(*, include_all: bool = True) -> _TargetReplay:
    return _TargetReplay(
        run_id="a" * 32,
        anchors=_anchors(include_all=include_all),
        quantity_rules=_rules(),
        run_started_wall_ns=WALL,
        run_started_monotonic_ns=BASE,
        coverage_end_monotonic_ns=BASE + 30_000_000_000,
    )


def _seed(replay: _TargetReplay, *, include_all: bool = True) -> None:
    symbols = IMPACT_CAPTURE_SYMBOLS if include_all else ("BTCUSDT",)
    for symbol in symbols:
        replay.observe_mark(symbol=symbol, next_funding_time_ms=2_000_000_000_000)
        replay.observe_depth(
            symbol=symbol,
            received_monotonic_ns=BASE + 100_000_000,
            state=_state(symbol, update_id=1),
        )


def test_target_replay_materializes_every_dimension_without_shock_filter() -> None:
    replay = _replay()
    _seed(replay)
    replay.before_record(BASE + 1_000_000_000)
    event_offsets = (
        1_500_000_000,
        2_000_000_000,
        2_500_000_000,
        3_000_000_000,
        6_500_000_000,
        7_000_000_000,
        16_500_000_000,
        17_000_000_000,
    )
    update_id = 2
    for offset in event_offsets:
        replay.before_record(BASE + offset)
        for symbol in IMPACT_CAPTURE_SYMBOLS:
            replay.observe_depth(
                symbol=symbol,
                received_monotonic_ns=BASE + offset,
                state=_state(symbol, update_id=update_id),
            )
        update_id += 1

    rows = replay.finish()

    assert len(rows) == 3 * 36
    assert all(row.eligible for row in rows)
    assert len({row.key for row in rows}) == len(rows)
    assert all(_option_invariant_errors(row) == () for row in rows)
    assert {row.entry_delay_ms for row in rows} == {500, 1_000}
    assert {row.horizon_ms for row in rows} == {1_000, 5_000, 15_000}
    assert {row.reference_quote_notional for row in rows} == {
        100.0,
        1_000.0,
        5_000.0,
    }


def test_target_replay_retains_late_and_coverage_end_options() -> None:
    replay = _replay(include_all=False)
    _seed(replay, include_all=False)
    replay.before_record(BASE + 1_000_000_000)
    replay.before_record(BASE + 1_751_000_000)
    replay.observe_depth(
        symbol="BTCUSDT",
        received_monotonic_ns=BASE + 1_751_000_000,
        state=_state("BTCUSDT", update_id=2),
    )

    rows = replay.finish()

    assert len(rows) == 36
    reasons = [row.ineligible_reasons_json for row in rows]
    assert reasons.count('["entry_state_late"]') == 18
    assert reasons.count('["coverage_end"]') == 18
    assert all(not row.eligible for row in rows)
    assert all(row.net_payoff_bps is None for row in rows)


def test_target_replay_excludes_funding_boundary_without_fabricated_cashflow() -> None:
    replay = _replay(include_all=False)
    _seed(replay, include_all=False)
    replay.observe_mark(
        symbol="BTCUSDT",
        next_funding_time_ms=(WALL + 4_000_000_000) // 1_000_000,
    )
    replay.before_record(BASE + 1_000_000_000)
    for offset in (1_500_000_000, 2_000_000_000, 2_500_000_000, 3_000_000_000):
        replay.before_record(BASE + offset)
        replay.observe_depth(
            symbol="BTCUSDT",
            received_monotonic_ns=BASE + offset,
            state=_state("BTCUSDT", update_id=offset),
        )

    rows = replay.finish()

    funding_rows = [
        row for row in rows if row.ineligible_reasons_json == '["funding_boundary"]'
    ]
    assert len(funding_rows) == 24
    assert {row.horizon_ms for row in funding_rows} == {5_000, 15_000}
    assert all(row.net_payoff_quote is None for row in funding_rows)


def test_columnar_target_insert_preserves_nullable_booleans_and_64_bit_clocks() -> None:
    replay = _replay(include_all=False)
    _seed(replay, include_all=False)
    replay.before_record(BASE + 1_000_000_000)
    rows = replay.finish()
    connection = duckdb.connect(":memory:")
    try:
        _create_target_tables(connection)
        _insert_option_batch(connection, rows, batch_index=0)
        stored = connection.execute(
            f"SELECT * FROM {ROUND73_TARGET_OPTION_TABLE} "
            "ORDER BY symbol, anchor_index, entry_delay_ms, horizon_ms, "
            "reference_quote_notional, side"
        ).fetchall()
    finally:
        connection.close()

    restored = [_option_from_row(row) for row in stored]
    assert [row.option_sha256 for row in restored] == [row.option_sha256 for row in rows]
    assert all(row.positive_net_payoff is None for row in restored)
    assert all(row.decision_monotonic_ns == BASE + 1_000_000_000 for row in restored)


def test_target_option_hash_tampering_is_rejected() -> None:
    replay = _replay(include_all=False)
    _seed(replay, include_all=False)
    replay.before_record(BASE + 1_000_000_000)
    option = replay.finish()[0]
    values = option.__dict__.copy()
    values["decision_mid"] = option.decision_mid + 1.0
    tampered = type(option)(**values)

    assert "option_hash" in _option_invariant_errors(tampered)


def test_source_bound_quantity_tampering_is_rejected_even_when_rehashed() -> None:
    replay = _replay(include_all=False)
    _seed(replay, include_all=False)
    replay.before_record(BASE + 1_000_000_000)
    option = replay.finish()[0]
    assert option.base_quantity is not None
    values = option.__dict__.copy()
    values["base_quantity"] = option.base_quantity + 0.001
    values["option_sha256"] = option.option_sha256
    tampered = type(option)(**values)

    assert "quantity_identity" in _quantity_invariant_errors(
        tampered,
        _rules()["BTCUSDT"],
    )


def test_quantity_rule_uses_exact_decimal_step_alignment() -> None:
    rules = Round73MarketQuantityRules.create(
        symbol="SOLUSDT",
        step_size="0.01",
        minimum_quantity="0.01",
        maximum_quantity="80000",
        minimum_notional="5",
    )

    quantity = rules.quantize_reference_quantity(
        reference_quote_notional=100.0,
        decision_mid=187.1234,
    )

    assert quantity == 0.53
    assert Decimal(str(quantity)) % Decimal("0.01") == 0


def test_replay_rejects_future_decision_book_state() -> None:
    replay = _replay(include_all=False)
    replay.observe_mark(symbol="BTCUSDT", next_funding_time_ms=2_000_000_000_000)
    replay.observe_depth(
        symbol="BTCUSDT",
        received_monotonic_ns=BASE + 1_000_000_000,
        state=_state("BTCUSDT", update_id=1),
    )

    with pytest.raises(ValueError, match="future receipt"):
        replay.before_record(BASE + 1_000_000_000)
