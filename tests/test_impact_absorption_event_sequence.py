from __future__ import annotations

import json
import math

import pytest

from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_FEATURE_NAMES_SHA256,
    ROUND74_EVENT_SEQUENCE_SCHEMA_VERSION,
    ROUND74_EVENT_STATE_HALF_LIVES_SECONDS,
    Round74EventSequenceEncoder,
    Round74MultiSymbolEventReplay,
    iter_round74_event_windows,
    iter_round74_v10_event_tokens,
)
from simple_ai_trading.impact_absorption_store import ImpactAbsorptionStore
from simple_ai_trading.impact_capture_frame import ImpactCaptureFrameRecord


WALL_BASE = 1_784_058_600_000_000_000


def _snapshot() -> dict[str, object]:
    return {
        "lastUpdateId": 100,
        "bids": [[f"{100.0 - index * 0.1:.1f}", "2"] for index in range(25)],
        "asks": [[f"{100.1 + index * 0.1:.1f}", "3"] for index in range(25)],
    }


def _record(
    *,
    lane: str,
    sequence: int,
    monotonic_ns: int,
    stream_name: str,
    payload: dict[str, object],
) -> ImpactCaptureFrameRecord:
    return ImpactCaptureFrameRecord(
        stream=lane,
        connection_id=f"{lane}:test",
        sequence_number=sequence,
        received_wall_ns=WALL_BASE + monotonic_ns,
        received_monotonic_ns=monotonic_ns,
        raw_text=json.dumps(
            {"stream": stream_name, "data": payload},
            ensure_ascii=True,
            separators=(",", ":"),
        ),
    )


def _depth(
    *,
    first: int,
    final: int,
    previous: int,
    event_time_ms: int,
    bid_qty: str,
    ask_qty: str,
) -> dict[str, object]:
    return {
        "e": "depthUpdate",
        "E": event_time_ms,
        "T": event_time_ms - 1,
        "s": "BTCUSDT",
        "U": first,
        "u": final,
        "pu": previous,
        "b": [["100.0", bid_qty]],
        "a": [["100.1", ask_qty]],
        "st": 1,
        "ps": "BTCUSDT",
    }


def _ticker(event_time_ms: int) -> dict[str, object]:
    return {
        "e": "bookTicker",
        "E": event_time_ms,
        "T": event_time_ms - 1,
        "s": "BTCUSDT",
        "u": 103,
        "b": "100.0",
        "B": "5",
        "a": "100.1",
        "A": "2",
        "st": 1,
        "ps": "BTCUSDT",
    }


def _trade(event_time_ms: int, *, buyer_is_maker: bool = False) -> dict[str, object]:
    return {
        "e": "aggTrade",
        "E": event_time_ms,
        "T": event_time_ms - 1,
        "s": "BTCUSDT",
        "a": 500,
        "p": "100.1",
        "q": "2",
        "nq": "2",
        "f": 700,
        "l": 702,
        "m": buyer_is_maker,
        "st": 1,
    }


def _mark(event_time_ms: int) -> dict[str, object]:
    return {
        "e": "markPriceUpdate",
        "E": event_time_ms,
        "T": 2_000,
        "s": "BTCUSDT",
        "p": "100.0",
        "i": "99.9",
        "P": "0",
        "r": "-0.0001",
        "st": 1,
    }


def _liquidation(event_time_ms: int) -> dict[str, object]:
    return {
        "e": "forceOrder",
        "E": event_time_ms,
        "o": {
            "s": "BTCUSDT",
            "ps": "BTCUSDT",
            "st": 1,
            "S": "SELL",
            "o": "LIMIT",
            "f": "IOC",
            "q": "4",
            "p": "99",
            "ap": "98.5",
            "X": "FILLED",
            "l": "1",
            "z": "4",
            "T": event_time_ms - 1,
        },
    }


def _encoder(*, ready_offset_ns: int = 150) -> Round74EventSequenceEncoder:
    return Round74EventSequenceEncoder(
        symbol="BTCUSDT",
        tick_size="0.1",
        depth_snapshot=_snapshot(),
        feature_ready_wall_ns=WALL_BASE + ready_offset_ns,
    )


def _feature(token, name: str) -> float:
    return token.feature_values[ROUND74_EVENT_FEATURE_NAMES.index(name)]


def test_event_observation_retains_preready_and_stale_depth_state() -> None:
    replay = Round74MultiSymbolEventReplay(
        tick_sizes={
            "BTCUSDT": "0.1",
            "ETHUSDT": "0.1",
            "SOLUSDT": "0.1",
        },
        depth_snapshots={
            "BTCUSDT": _snapshot(),
            "ETHUSDT": _snapshot(),
            "SOLUSDT": _snapshot(),
        },
        feature_ready_wall_ns=WALL_BASE + 150,
    )
    preready = replay.consume_observation(
        frame_index=0,
        message_index=0,
        record=_record(
            lane="binance_futures_public",
            sequence=0,
            monotonic_ns=100,
            stream_name="btcusdt@depth@100ms",
            payload=_depth(
                first=101,
                final=102,
                previous=100,
                event_time_ms=1_001,
                bid_qty="4",
                ask_qty="1",
            ),
        ),
    )
    stale = replay.consume_observation(
        frame_index=0,
        message_index=1,
        record=_record(
            lane="binance_futures_public",
            sequence=1,
            monotonic_ns=200,
            stream_name="btcusdt@depth@100ms",
            payload=_depth(
                first=102,
                final=102,
                previous=101,
                event_time_ms=1_002,
                bid_qty="9",
                ask_qty="9",
            ),
        ),
    )

    assert preready is not None and stale is not None
    assert preready.token is None
    assert preready.depth_state is not None
    assert preready.depth_state.update_id == 102
    assert preready.depth_update_is_stale is False
    assert stale.token is not None
    assert stale.depth_state is not None
    assert stale.depth_state.update_id == 102
    assert stale.depth_update_is_stale is True
    assert _feature(stale.token, "depth_update_is_stale") == 1.0


def test_event_sequence_preserves_subsecond_order_and_financial_signs() -> None:
    encoder = _encoder()
    warmup = encoder.consume(
        frame_index=0,
        message_index=0,
        record=_record(
            lane="binance_futures_public",
            sequence=0,
            monotonic_ns=100,
            stream_name="btcusdt@depth@100ms",
            payload=_depth(
                first=101,
                final=102,
                previous=100,
                event_time_ms=1_001,
                bid_qty="4",
                ask_qty="1",
            ),
        ),
    )
    ticker = encoder.consume(
        frame_index=0,
        message_index=1,
        record=_record(
            lane="binance_futures_public",
            sequence=1,
            monotonic_ns=200,
            stream_name="btcusdt@bookTicker",
            payload=_ticker(1_010),
        ),
    )
    trade = encoder.consume(
        frame_index=0,
        message_index=2,
        record=_record(
            lane="binance_futures_market",
            sequence=0,
            monotonic_ns=250,
            stream_name="btcusdt@aggTrade",
            payload=_trade(1_020),
        ),
    )
    depth = encoder.consume(
        frame_index=0,
        message_index=3,
        record=_record(
            lane="binance_futures_public",
            sequence=2,
            monotonic_ns=300,
            stream_name="btcusdt@depth@100ms",
            payload=_depth(
                first=103,
                final=103,
                previous=102,
                event_time_ms=1_030,
                bid_qty="6",
                ask_qty="1",
            ),
        ),
    )

    assert warmup is None
    assert ticker is not None
    assert trade is not None
    assert depth is not None
    assert [item.received_monotonic_ns for item in (ticker, trade, depth)] == [
        200,
        250,
        300,
    ]
    assert (
        len({item.received_wall_ns // 1_000_000_000 for item in (ticker, trade, depth)})
        == 1
    )
    assert _feature(trade, "event_is_aggregate_trade") == 1.0
    assert _feature(trade, "symbol_is_btcusdt") == 1.0
    assert _feature(trade, "symbol_is_ethusdt") == 0.0
    assert _feature(trade, "symbol_is_solusdt") == 0.0
    assert _feature(trade, "trade_signed_quote_scaled") > 0.0
    assert _feature(trade, "trade_absolute_quote_scaled") > 0.0
    assert _feature(depth, "depth_signed_pressure_levels_1_5_scaled") > 0.0
    assert _feature(depth, "depth_absolute_flow_levels_1_5_scaled") > 0.0
    assert all(math.isfinite(value) for value in depth.feature_values)
    assert depth.as_dict()["schema_version"] == ROUND74_EVENT_SEQUENCE_SCHEMA_VERSION
    assert depth.as_dict()["feature_names_sha256"] == (
        ROUND74_EVENT_FEATURE_NAMES_SHA256
    )
    assert depth.as_dict()["target_constructed"] is False
    assert depth.as_dict()["model_evaluated"] is False


def test_event_sequence_retains_mark_and_liquidation_context() -> None:
    encoder = _encoder(ready_offset_ns=1)
    mark = encoder.consume(
        frame_index=0,
        message_index=0,
        record=_record(
            lane="binance_futures_market",
            sequence=0,
            monotonic_ns=100,
            stream_name="btcusdt@markPrice@1s",
            payload=_mark(1_030),
        ),
    )
    liquidation = encoder.consume(
        frame_index=0,
        message_index=1,
        record=_record(
            lane="binance_futures_market",
            sequence=1,
            monotonic_ns=200,
            stream_name="btcusdt@forceOrder",
            payload=_liquidation(2_000),
        ),
    )

    assert mark is not None
    assert liquidation is not None
    assert _feature(mark, "event_is_mark_price") == 1.0
    assert _feature(mark, "mark_to_mid_bps") < 0.0
    assert _feature(mark, "index_to_mid_bps") < _feature(mark, "mark_to_mid_bps")
    assert _feature(mark, "funding_rate_bps") < 0.0
    assert _feature(liquidation, "event_is_liquidation") == 1.0
    assert _feature(liquidation, "liquidation_signed_quote_scaled") < 0.0
    assert _feature(liquidation, "mark_to_mid_bps") == pytest.approx(
        _feature(mark, "mark_to_mid_bps")
    )


def test_event_sequence_adds_causal_multi_timescale_liquidity_state() -> None:
    encoder = _encoder(ready_offset_ns=1)
    first = encoder.consume(
        frame_index=0,
        message_index=0,
        record=_record(
            lane="binance_futures_market",
            sequence=0,
            monotonic_ns=1_000_000_000,
            stream_name="btcusdt@aggTrade",
            payload=_trade(1_020),
        ),
    )
    moved_ticker = _ticker(6_020)
    moved_ticker["b"] = "100.1"
    moved_ticker["a"] = "100.2"
    second = encoder.consume(
        frame_index=0,
        message_index=1,
        record=_record(
            lane="binance_futures_public",
            sequence=0,
            monotonic_ns=6_000_000_000,
            stream_name="btcusdt@bookTicker",
            payload=moved_ticker,
        ),
    )
    third = encoder.consume(
        frame_index=0,
        message_index=2,
        record=_record(
            lane="binance_futures_market",
            sequence=1,
            monotonic_ns=11_000_000_000,
            stream_name="btcusdt@aggTrade",
            payload=_trade(11_020, buyer_is_maker=True),
        ),
    )

    assert first is not None and second is not None and third is not None
    assert ROUND74_EVENT_STATE_HALF_LIVES_SECONDS == (5, 30, 300)
    assert _feature(first, "ewm_return_projection_5s_bps") == 0.0
    assert _feature(second, "ewm_return_projection_5s_bps") > 0.0
    assert _feature(second, "ewm_realized_volatility_300s_bps") > 0.0
    assert _feature(second, "ewm_signed_trade_pressure_per_second_30s") > 0.0
    assert _feature(third, "ewm_signed_trade_pressure_per_second_5s") < _feature(
        second, "ewm_signed_trade_pressure_per_second_5s"
    )
    assert _feature(third, "ewm_spread_300s_bps") > 0.0
    assert _feature(third, "log1p_bid_depth_quote_20") > 0.0
    assert _feature(third, "log1p_ask_depth_quote_20") > 0.0
    assert all(math.isfinite(value) for value in third.feature_values)


def test_event_windows_are_per_symbol_causal_and_break_on_long_gap() -> None:
    encoder = _encoder(ready_offset_ns=1)
    tokens = []
    for index, monotonic_ns in enumerate((100, 200, 300)):
        token = encoder.consume(
            frame_index=0,
            message_index=index,
            record=_record(
                lane="binance_futures_market",
                sequence=index,
                monotonic_ns=monotonic_ns,
                stream_name="btcusdt@aggTrade",
                payload=_trade(1_020 + index),
            ),
        )
        assert token is not None
        tokens.append(token)

    windows = list(
        iter_round74_event_windows(
            tokens,
            sequence_length=2,
            stride=1,
            maximum_gap_ns=1_000,
        )
    )

    assert len(windows) == 2
    assert windows[0].endpoint_message_index == 1
    assert windows[1].endpoint_message_index == 2
    assert windows[0].feature_values == (
        tokens[0].feature_values,
        tokens[1].feature_values,
    )

    late = encoder.consume(
        frame_index=1,
        message_index=0,
        record=_record(
            lane="binance_futures_market",
            sequence=3,
            monotonic_ns=10_000,
            stream_name="btcusdt@aggTrade",
            payload=_trade(1_100),
        ),
    )
    assert late is not None
    assert (
        list(
            iter_round74_event_windows(
                [tokens[-1], late],
                sequence_length=2,
                stride=1,
                maximum_gap_ns=1_000,
            )
        )
        == []
    )


def test_event_sequence_fails_closed_on_regression_lane_and_duplicate_json() -> None:
    encoder = _encoder(ready_offset_ns=1)
    valid = _record(
        lane="binance_futures_market",
        sequence=0,
        monotonic_ns=200,
        stream_name="btcusdt@aggTrade",
        payload=_trade(1_020),
    )
    assert encoder.consume(frame_index=0, message_index=0, record=valid) is not None

    with pytest.raises(ValueError, match="not monotone"):
        encoder.consume(
            frame_index=0,
            message_index=1,
            record=_record(
                lane="binance_futures_market",
                sequence=1,
                monotonic_ns=100,
                stream_name="btcusdt@aggTrade",
                payload=_trade(1_021),
            ),
        )

    wrong_lane = _record(
        lane="binance_futures_public",
        sequence=0,
        monotonic_ns=300,
        stream_name="btcusdt@aggTrade",
        payload=_trade(1_022),
    )
    with pytest.raises(ValueError, match="wrong lane"):
        _encoder(ready_offset_ns=1).consume(
            frame_index=0,
            message_index=0,
            record=wrong_lane,
        )

    duplicate = ImpactCaptureFrameRecord(
        stream="binance_futures_market",
        connection_id="market:test",
        sequence_number=0,
        received_wall_ns=WALL_BASE + 300,
        received_monotonic_ns=300,
        raw_text='{"stream":"btcusdt@aggTrade","stream":"x","data":{}}',
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _encoder(ready_offset_ns=1).consume(
            frame_index=0,
            message_index=0,
            record=duplicate,
        )


def test_event_window_rejects_global_receipt_regression() -> None:
    encoder = _encoder(ready_offset_ns=1)
    first = encoder.consume(
        frame_index=0,
        message_index=0,
        record=_record(
            lane="binance_futures_market",
            sequence=0,
            monotonic_ns=100,
            stream_name="btcusdt@aggTrade",
            payload=_trade(1_020),
        ),
    )
    second = encoder.consume(
        frame_index=0,
        message_index=1,
        record=_record(
            lane="binance_futures_market",
            sequence=1,
            monotonic_ns=200,
            stream_name="btcusdt@aggTrade",
            payload=_trade(1_021),
        ),
    )
    assert first is not None and second is not None

    with pytest.raises(ValueError, match="global event token order regressed"):
        list(
            iter_round74_event_windows(
                [second, first],
                sequence_length=2,
                stride=1,
            )
        )


def test_round74_v10_replay_requires_read_only_store() -> None:
    store = ImpactAbsorptionStore(":memory:")
    with pytest.raises(ValueError, match="read-only store"):
        list(iter_round74_v10_event_tokens(store, run_id="a" * 32))
