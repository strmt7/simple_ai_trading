from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import pytest

import simple_ai_trading.polymarket_round21_core_features as core_features
from simple_ai_trading.polymarket_capture_frame import CaptureFrameRecord
from simple_ai_trading.polymarket_redundant_union import (
    PolymarketClobLaneReceipt,
    PolymarketRedundantUnionBuilder,
    PolymarketUnionEvent,
)
from simple_ai_trading.polymarket_round21_binance_features import (
    Round21IndependentBinanceFeatureEngine,
)
from simple_ai_trading.polymarket_round21_core_features import (
    POLYMARKET_ROUND21_CORE_FEATURE_NAMES,
    POLYMARKET_ROUND21_FEATURE_POLICY_SHA256,
    POLYMARKET_ROUND21_FEATURE_SCHEMA,
    Round21CoreFeatureEngine,
    join_round21_causal_features,
    load_round21_feature_policy,
    parse_round21_chainlink_wire_text,
    validate_round21_feature_policy,
)


EVENT_START_MS = 1_800_000_000_000
CONDITION_ID = "0x" + "1" * 64
UP_TOKEN = "1" * 40
DOWN_TOKEN = "2" * 40
FEATURE_POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-021-causal-feature-policy-v3.json"
)


def _chainlink_record(sequence: int, offset_ms: int) -> CaptureFrameRecord:
    price = 60_000.0 + sequence * 0.5
    payload = {
        "topic": "crypto_prices_chainlink",
        "type": "update",
        "timestamp": EVENT_START_MS + offset_ms,
        "payload": {
            "symbol": "btc/usd",
            "timestamp": EVENT_START_MS + offset_ms,
            "value": format(price, ".8f"),
        },
    }
    return CaptureFrameRecord(
        stream="polymarket_rtds",
        connection_id="rtds:chainlink:btc:" + "c" * 32,
        sequence_number=sequence,
        received_wall_ms=EVENT_START_MS + offset_ms + 10,
        received_monotonic_ns=(EVENT_START_MS + offset_ms + 10) * 1_000_000,
        raw_text=json.dumps(payload, separators=(",", ":"), sort_keys=True),
    )


def _book_event(
    token: str,
    *,
    offset_ms: int,
    bid: str,
    ask: str,
) -> dict[str, object]:
    return {
        "event_type": "book",
        "market": CONDITION_ID,
        "asset_id": token,
        "timestamp": str(EVENT_START_MS + offset_ms),
        "hash": hashlib.sha256(f"{token}:{offset_ms}".encode("ascii")).hexdigest(),
        "bids": [
            {"price": bid, "size": "8"},
            {"price": format(float(bid) - 0.01, ".2f"), "size": "7"},
        ],
        "asks": [
            {"price": ask, "size": "9"},
            {"price": format(float(ask) + 0.01, ".2f"), "size": "6"},
        ],
    }


def _change_event(
    token: str,
    *,
    offset_ms: int,
    price: str,
    size: str,
    side: str,
    best_bid: str,
    best_ask: str,
) -> dict[str, object]:
    return {
        "event_type": "price_change",
        "market": CONDITION_ID,
        "timestamp": str(EVENT_START_MS + offset_ms),
        "price_changes": [
            {
                "asset_id": token,
                "price": price,
                "size": size,
                "side": side,
                "hash": hashlib.sha256(
                    f"{token}:{offset_ms}:{price}".encode("ascii")
                ).hexdigest(),
                "best_bid": best_bid,
                "best_ask": best_ask,
            }
        ],
    }


def _union_events(events: list[tuple[int, dict[str, object]]]) -> tuple:
    builder = PolymarketRedundantUnionBuilder(pairing_window_ms=2_000)
    sequence = {"clob-a": 0, "clob-b": 0}
    for offset_ms, payload in events:
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        for lane, delay_ns in (("clob-a", 0), ("clob-b", 10_000_000)):
            sequence[lane] += 1
            builder.add(
                PolymarketClobLaneReceipt(
                    lane_id=lane,
                    connection_id=f"{lane}:" + lane[-1] * 32,
                    sequence_number=sequence[lane],
                    received_wall_ms=EVENT_START_MS + offset_ms + 20,
                    received_monotonic_ns=(
                        (EVENT_START_MS + offset_ms + 20) * 1_000_000 + delay_ns
                    ),
                    raw_text=raw,
                )
            )
    union, _audit = builder.finish()
    return union


def _ready_engine() -> Round21CoreFeatureEngine:
    engine = Round21CoreFeatureEngine(
        condition_id=CONDITION_ID,
        up_token_id=UP_TOKEN,
        down_token_id=DOWN_TOKEN,
        event_start_ms=EVENT_START_MS,
    )
    for sequence in range(1, 22):
        engine.ingest_chainlink_record(
            _chainlink_record(sequence, (sequence - 1) * 1_500)
        )
    events = _union_events(
        [
            (29_000, _book_event(UP_TOKEN, offset_ms=29_000, bid="0.49", ask="0.51")),
            (
                29_010,
                _book_event(
                    DOWN_TOKEN,
                    offset_ms=29_010,
                    bid="0.48",
                    ask="0.52",
                ),
            ),
            (
                30_000,
                _change_event(
                    UP_TOKEN,
                    offset_ms=30_000,
                    price="0.49",
                    size="9",
                    side="BUY",
                    best_bid="0.49",
                    best_ask="0.51",
                ),
            ),
            (
                30_010,
                _change_event(
                    DOWN_TOKEN,
                    offset_ms=30_010,
                    price="0.52",
                    size="10",
                    side="SELL",
                    best_bid="0.48",
                    best_ask="0.52",
                ),
            ),
        ]
    )
    for event in events:
        engine.ingest_union_event(event)
    return engine


def _rehash_policy(value: dict[str, object]) -> dict[str, object]:
    body = dict(value)
    body.pop("design_sha256", None)
    body["design_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return body


def test_round21_feature_policy_binds_causality_and_independence() -> None:
    policy = load_round21_feature_policy(FEATURE_POLICY_PATH)

    assert policy["design_sha256"] == POLYMARKET_ROUND21_FEATURE_POLICY_SHA256
    assert (
        POLYMARKET_ROUND21_FEATURE_SCHEMA.feature_policy_sha256
        == POLYMARKET_ROUND21_FEATURE_POLICY_SHA256
    )
    assert policy["optional_binance"]["execution"] is False
    assert policy["optional_binance"]["risk_or_stop_dependency"] is False

    changed = json.loads(FEATURE_POLICY_PATH.read_text(encoding="utf-8"))
    changed["optional_binance"]["account_access"] = True
    with pytest.raises(ValueError, match="causal feature policy differs"):
        validate_round21_feature_policy(_rehash_policy(changed))


def test_round21_core_builds_receipt_time_structural_and_book_features() -> None:
    engine = _ready_engine()

    snapshot = engine.build(EVENT_START_MS + 31_000)

    assert snapshot.available is True
    assert snapshot.reasons == ()
    assert 0.5 < snapshot.structural_probability < 1.0
    assert 0.0 < snapshot.market_prior_probability < 1.0
    assert len(snapshot.values) == len(POLYMARKET_ROUND21_CORE_FEATURE_NAMES)
    assert (
        snapshot.values[
            POLYMARKET_ROUND21_CORE_FEATURE_NAMES.index("core.chainlink_return_count")
        ]
        == 20.0
    )
    assert snapshot.values[
        POLYMARKET_ROUND21_CORE_FEATURE_NAMES.index("core.complement_buy_overround")
    ] == pytest.approx(0.03)
    assert snapshot.values[
        POLYMARKET_ROUND21_CORE_FEATURE_NAMES.index("core.complement_sell_underround")
    ] == pytest.approx(0.03)
    assert (
        snapshot.values[
            POLYMARKET_ROUND21_CORE_FEATURE_NAMES.index("core.up_book_receipt_age_ms")
        ]
        == 980.0
    )
    assert (
        snapshot.values[
            POLYMARKET_ROUND21_CORE_FEATURE_NAMES.index("core.down_book_receipt_age_ms")
        ]
        == 970.0
    )
    assert (
        snapshot.values[
            POLYMARKET_ROUND21_CORE_FEATURE_NAMES.index("core.book_receipt_skew_ms")
        ]
        == 10.0
    )
    assert snapshot.source_chain_sha256 != hashlib.sha256(b"").hexdigest()
    assert snapshot.trading_authority is False
    assert engine.credentials_used is False
    assert engine.execution_connected is False


def test_round21_chainlink_bootstrap_is_sequence_accounted_control_only() -> None:
    snapshot = {
        "topic": "crypto_prices",
        "type": "subscribe",
        "timestamp": EVENT_START_MS,
        "payload": {
            "symbol": "btc/usd",
            "data": [
                {"timestamp": EVENT_START_MS - 2_000, "value": 59_999.0},
                {"timestamp": EVENT_START_MS - 1_000, "value": 60_000.0},
            ],
        },
    }
    raw = json.dumps(snapshot, separators=(",", ":"), sort_keys=True)
    assert (
        parse_round21_chainlink_wire_text(
            "",
            received_at_ms=EVENT_START_MS,
        )
        is None
    )
    assert (
        parse_round21_chainlink_wire_text(
            raw,
            received_at_ms=EVENT_START_MS,
        )
        is None
    )
    tick = parse_round21_chainlink_wire_text(
        _chainlink_record(1, 0).raw_text,
        received_at_ms=EVENT_START_MS + 10,
    )
    assert tick is not None
    assert float(tick.price) == pytest.approx(60_000.5)

    engine = Round21CoreFeatureEngine(
        condition_id=CONDITION_ID,
        up_token_id=UP_TOKEN,
        down_token_id=DOWN_TOKEN,
        event_start_ms=EVENT_START_MS,
    )
    for sequence, control in enumerate(("", raw, "PING"), start=1):
        engine.ingest_chainlink_record(
            CaptureFrameRecord(
                stream="polymarket_rtds",
                connection_id="rtds:chainlink:btc:" + "e" * 32,
                sequence_number=sequence,
                received_wall_ms=EVENT_START_MS + sequence,
                received_monotonic_ns=(EVENT_START_MS + sequence) * 1_000_000,
                raw_text=control,
            )
        )
    assert engine._chainlink_sequence == 3
    assert engine._chainlink.latest_received_ms is None

    snapshot["payload"]["data"][1]["timestamp"] = EVENT_START_MS - 2_000
    with pytest.raises(ValueError, match="subscription snapshot differs"):
        parse_round21_chainlink_wire_text(
            json.dumps(snapshot),
            received_at_ms=EVENT_START_MS,
        )


def test_round21_core_is_unavailable_before_causal_warmup() -> None:
    engine = Round21CoreFeatureEngine(
        condition_id=CONDITION_ID,
        up_token_id=UP_TOKEN,
        down_token_id=DOWN_TOKEN,
        event_start_ms=EVENT_START_MS,
    )
    engine.ingest_chainlink_record(_chainlink_record(1, 0))

    snapshot = engine.build(EVENT_START_MS + 1_000)

    assert snapshot.available is False
    assert "chainlink_return_count_below_minimum" in snapshot.reasons
    assert "up_book_invalid" in snapshot.reasons
    assert not any(snapshot.values)
    assert snapshot.maximum_receipt_ms == 0


def test_round21_book_flow_allows_only_roundoff_sized_correction() -> None:
    series = core_features._RollingBookSeries()
    series.received_ms = [1_000]
    series.log_microprice = [0.0]
    series.prefixes = [
        [0.0, 1.0 + 5e-13],
        [0.0, 1.0],
        [0.0, 1.0 + 5e-13],
        [0.0, -1.0 - 5e-13],
        [0.0, 1.0],
    ]

    corrected = series.window(1_000, 250)

    assert corrected.top_ofi == 1.0
    assert corrected.mean_imbalance == 1.0
    assert corrected.level_pressure == -1.0

    series.prefixes[0][1] = 1.0 + 2e-12
    with pytest.raises(RuntimeError, match="flow accounting"):
        series.window(1_000, 250)


def test_round21_short_windows_use_only_the_previous_tick_as_causal_anchor() -> None:
    prices = core_features._RollingPriceSeries()
    prices.append(700, 100.0)
    prices.append(800, 101.0)

    price_window = prices.window(1_000, 250)
    expected_return = math.log(101.0 / 100.0)

    assert price_window.count == 1
    assert price_window.log_return == pytest.approx(expected_return)
    assert price_window.realized_variance == pytest.approx(expected_return**2)

    books = core_features._RollingBookSeries()
    books.append(
        core_features._BookSnapshot(
            received_wall_ms=700,
            received_monotonic_ns=700_000_000,
            bids=((0.49, 8.0),),
            asks=((0.51, 8.0),),
        )
    )
    books.append(
        core_features._BookSnapshot(
            received_wall_ms=800,
            received_monotonic_ns=800_000_000,
            bids=((0.54, 8.0),),
            asks=((0.56, 8.0),),
        )
    )

    book_window = books.window(1_000, 250)

    assert book_window.count == 1
    assert book_window.microprice_return == pytest.approx(math.log(0.55 / 0.50))


def test_round21_core_fails_closed_on_contradictory_delta() -> None:
    engine = _ready_engine()
    bad = _union_events(
        [
            (
                30_500,
                _change_event(
                    UP_TOKEN,
                    offset_ms=30_500,
                    price="0.49",
                    size="10",
                    side="BUY",
                    best_bid="0.47",
                    best_ask="0.51",
                ),
            )
        ]
    )[0]
    engine.ingest_union_event(bad)

    snapshot = engine.build(EVENT_START_MS + 31_000)

    assert snapshot.available is False
    assert "up_book_invalid" in snapshot.reasons


def test_round21_core_rejects_tampered_union_and_future_receipt() -> None:
    engine = _ready_engine()
    event = _union_events(
        [(30_500, _book_event(UP_TOKEN, offset_ms=30_500, bid="0.49", ask="0.51"))]
    )[0]
    assert isinstance(event, PolymarketUnionEvent)
    with pytest.raises(ValueError, match="integrity differs"):
        engine.ingest_union_event(replace(event, event_sha256="f" * 64))

    engine.ingest_union_event(event)
    with pytest.raises(ValueError, match="future receipts"):
        engine.build(EVENT_START_MS + 30_000)


def test_round21_chainlink_reconnect_requires_explicit_epoch_reset() -> None:
    engine = Round21CoreFeatureEngine(
        condition_id=CONDITION_ID,
        up_token_id=UP_TOKEN,
        down_token_id=DOWN_TOKEN,
        event_start_ms=EVENT_START_MS,
    )
    engine.ingest_chainlink_record(_chainlink_record(1, 0))
    replacement = replace(
        _chainlink_record(1, 1_000),
        connection_id="rtds:chainlink:btc:" + "d" * 32,
    )

    with pytest.raises(ValueError, match="reconnect or chronology"):
        engine.ingest_chainlink_record(replacement)
    engine.start_chainlink_epoch(replacement.connection_id)
    engine.ingest_chainlink_record(replacement)
    for sequence in range(2, 23):
        engine.ingest_chainlink_record(
            replace(
                _chainlink_record(
                    sequence,
                    1_000 + (sequence - 1) * 1_500,
                ),
                connection_id=replacement.connection_id,
            )
        )
    snapshot = engine.build(EVENT_START_MS + 33_000)

    assert snapshot.available is False
    assert "chainlink_connection_gap" in snapshot.reasons
    assert "chainlink_return_count_below_minimum" not in snapshot.reasons
    assert "chainlink_coverage_below_minimum" not in snapshot.reasons


def test_round21_chainlink_condition_slice_binds_mid_epoch_sequence() -> None:
    engine = Round21CoreFeatureEngine(
        condition_id=CONDITION_ID,
        up_token_id=UP_TOKEN,
        down_token_id=DOWN_TOKEN,
        event_start_ms=EVENT_START_MS,
    )
    first = replace(_chainlink_record(417, 0), sequence_number=417)

    engine.start_chainlink_epoch(
        first.connection_id,
        first_sequence_number=first.sequence_number,
    )
    engine.ingest_chainlink_record(first)
    engine.ingest_chainlink_record(
        replace(_chainlink_record(418, 1_500), sequence_number=418)
    )
    with pytest.raises(ValueError, match="reconnect or chronology"):
        engine.ingest_chainlink_record(
            replace(_chainlink_record(420, 3_000), sequence_number=420)
        )
    with pytest.raises(ValueError, match="epoch identity"):
        engine.start_chainlink_epoch(first.connection_id, first_sequence_number=0)


def test_round21_exact_join_keeps_missing_binance_optional() -> None:
    core = _ready_engine().build(EVENT_START_MS + 31_000)
    optional = Round21IndependentBinanceFeatureEngine().build(EVENT_START_MS + 31_000)

    row = join_round21_causal_features(core, optional)

    assert row.feature_schema_sha256 == POLYMARKET_ROUND21_FEATURE_SCHEMA.schema_sha256
    assert row.spot_available is False
    assert row.usdm_available is False
    assert not any(row.spot_values)
    assert not any(row.usdm_values)
    assert row.trading_authority is False
