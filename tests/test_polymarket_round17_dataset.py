from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_recorder import DecodedPublicEvent
from simple_ai_trading.polymarket_replay import PolymarketRecordedBook
from simple_ai_trading.polymarket_round14_dataset import (
    PolymarketRound14ConditionAdmission,
)
from simple_ai_trading.polymarket_round14_features import (
    POLYMARKET_ROUND14_FEATURE_NAMES,
    POLYMARKET_ROUND14_FEATURE_NAMES_SHA256,
    PolymarketRound14FeatureRow,
)
from simple_ai_trading.polymarket_round17_dataset import (
    materialize_round17_condition_rows,
    parse_round17_binance_trade_event,
)
from simple_ai_trading.polymarket_round17_features import (
    POLYMARKET_ROUND17_FEATURE_NAMES,
)


EVENT_START_MS = 1_800_000_000_000
EVENT_END_MS = EVENT_START_MS + 300_000
DECISION_MS = EVENT_START_MS + 30_000
CONDITION_ID = "0x" + "1" * 64
UP_TOKEN_ID = "1" * 40
DOWN_TOKEN_ID = "2" * 40
RUN_ID = "round17-run"
ROOT = Path(__file__).resolve().parents[1]


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _admission() -> PolymarketRound14ConditionAdmission:
    provisional = PolymarketRound14ConditionAdmission(
        run_id=RUN_ID,
        condition_id=CONDITION_ID,
        event_start_ms=EVENT_START_MS,
        event_end_ms=EVENT_END_MS,
        candidate_row_count=1_120,
        materialized_row_count=1_120,
        external_row_count=1_120,
        row_coverage_fraction=1.0,
        external_coverage_fraction=1.0,
        maximum_consecutive_missing_ms=0,
        external_maximum_consecutive_missing_ms=0,
        chainlink_tick_count=300,
        spot_bbo_count=1_000,
        futures_bbo_count=1_000,
        spot_trade_count=1_000,
        futures_trade_count=1_000,
        ignored_futures_zero_trade_count=0,
        exact_chainlink_open_event_sha256="a" * 64,
        row_manifest_sha256="b" * 64,
        core_eligible=True,
        binance_layer_eligible=True,
        reasons=(),
        binance_reasons=(),
        admission_sha256="0" * 64,
    )
    return replace(
        provisional,
        admission_sha256=_sha256(provisional.identity_payload()),
    ).validated()


def _market() -> PolymarketFiveMinuteMarket:
    return PolymarketFiveMinuteMarket(
        asset="BTC",
        market_id="12345",
        condition_id=CONDITION_ID,
        slug=f"btc-updown-5m-{EVENT_START_MS // 1_000}",
        question="Bitcoin Up or Down",
        event_start_ms=EVENT_START_MS,
        end_ms=EVENT_END_MS,
        up_token_id=UP_TOKEN_ID,
        down_token_id=DOWN_TOKEN_ID,
        tick_size=Decimal("0.01"),
        minimum_order_size=Decimal("5"),
        fee_schedule=PolymarketFeeSchedule(
            enabled=True,
            rate=Decimal("0.07"),
            exponent=2,
            taker_only=True,
            rebate_rate=Decimal("0"),
        ),
        liquidity_quote=Decimal("100000"),
        volume_quote=Decimal("200000"),
        resolution_source="https://data.chain.link/streams/btc-usd",
        gamma_payload_sha256="c" * 64,
        gamma_payload_json="{}",
    )


def _base_rows() -> tuple[PolymarketRound14FeatureRow, ...]:
    return tuple(
        PolymarketRound14FeatureRow(
            condition_id=CONDITION_ID,
            decision_time_ms=decision,
            feature_names_sha256=POLYMARKET_ROUND14_FEATURE_NAMES_SHA256,
            input_sha256=_sha256(["base", decision]),
            values=(0.0,) * len(POLYMARKET_ROUND14_FEATURE_NAMES),
        )
        for decision in (DECISION_MS, DECISION_MS + 250)
    )


def _event(
    *,
    stream: str,
    event_type: str,
    connection_id: str,
    received_at_ms: int,
    monotonic_ns: int,
    body: dict[str, object],
) -> DecodedPublicEvent:
    event_sha256 = _sha256(body)
    source_time = body.get("timestamp")
    publisher_time = None
    if "data" in body:
        data = body["data"]
        assert isinstance(data, dict)
        source_time = data["T"]
        publisher_time = data["E"]
    else:
        payload = body["payload"]
        assert isinstance(payload, dict)
        source_time = payload["timestamp"]
        publisher_time = body["timestamp"]
    return DecodedPublicEvent(
        event_id=_sha256(["event", monotonic_ns]),
        run_id=RUN_ID,
        message_id=_sha256(["message", monotonic_ns]),
        sub_index=0,
        stream=stream,
        event_type=event_type,
        symbol="BTC",
        condition_id="",
        asset_id="",
        source_time_ms=int(source_time),
        publisher_time_ms=int(publisher_time),
        event_sha256=event_sha256,
        event=body,
        connection_id=connection_id,
        sequence_number=monotonic_ns,
        received_wall_ms=received_at_ms,
        received_monotonic_ns=monotonic_ns,
    )


def _chainlink(
    *,
    received_at_ms: int,
    monotonic_ns: int,
    connection_id: str = "rtds:chainlink:btc:" + "a" * 32,
) -> DecodedPublicEvent:
    source_time = received_at_ms - 2
    return _event(
        stream="polymarket_rtds",
        event_type="crypto_prices_chainlink:update",
        connection_id=connection_id,
        received_at_ms=received_at_ms,
        monotonic_ns=monotonic_ns,
        body={
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "timestamp": received_at_ms - 1,
            "payload": {
                "symbol": "btc/usd",
                "timestamp": source_time,
                "value": "100",
                "full_accuracy_value": "100000000000000000000",
            },
        },
    )


def _trade(
    market: str,
    *,
    received_at_ms: int,
    monotonic_ns: int,
    trade_id: int,
    connection_id: str | None = None,
    price: str = "100",
    quantity: str = "1",
) -> DecodedPublicEvent:
    stream = "binance_spot" if market == "spot" else "binance_futures"
    connection = connection_id or f"binance:{market}:" + "b" * 32
    return _event(
        stream=stream,
        event_type="trade",
        connection_id=connection,
        received_at_ms=received_at_ms,
        monotonic_ns=monotonic_ns,
        body={
            "stream": "btcusdt@trade",
            "data": {
                "e": "trade",
                "E": received_at_ms - 1,
                "s": "BTCUSDT",
                "t": trade_id,
                "p": price,
                "q": quantity,
                "T": received_at_ms - 2,
                "m": False,
                "M": True,
            },
        },
    )


def _book(
    market: PolymarketFiveMinuteMarket,
    token_id: str,
    *,
    received_at_ms: int,
    monotonic_ns: int,
    event_number: int,
) -> PolymarketRecordedBook:
    outcome = "Up" if token_id == UP_TOKEN_ID else "Down"
    snapshot = PaperBookSnapshot(
        venue="polymarket",
        market_id=CONDITION_ID,
        asset_id=token_id,
        bids=(BookLevel(price=Decimal("0.49"), quantity=Decimal("10")),),
        asks=(BookLevel(price=Decimal("0.51"), quantity=Decimal("10")),),
        source_time_ms=received_at_ms - 2,
        received_wall_ms=received_at_ms,
        received_monotonic_ns=monotonic_ns,
        source_payload_sha256=_sha256(["book", event_number]),
    )
    return PolymarketRecordedBook(
        run_id=RUN_ID,
        event_id=_sha256(["book-event", event_number]),
        event_type="book",
        connection_id="clob:" + "c" * 32,
        segment_id="segment-" + "d" * 32,
        sequence_number=event_number,
        sub_index=0,
        market=market,
        outcome=outcome,
        tick_size=Decimal("0.01"),
        snapshot=snapshot,
    )


def _sources() -> tuple[
    tuple[DecodedPublicEvent, ...],
    tuple[PolymarketRecordedBook, ...],
]:
    market = _market()
    events = (
        _chainlink(
            received_at_ms=DECISION_MS - 1_000,
            monotonic_ns=1,
        ),
        _trade(
            "spot",
            received_at_ms=DECISION_MS - 900,
            monotonic_ns=2,
            trade_id=10,
        ),
        _trade(
            "perpetual",
            received_at_ms=DECISION_MS - 800,
            monotonic_ns=3,
            trade_id=20,
        ),
        _chainlink(
            received_at_ms=DECISION_MS - 100,
            monotonic_ns=4,
        ),
    )
    books = (
        _book(
            market,
            UP_TOKEN_ID,
            received_at_ms=DECISION_MS - 700,
            monotonic_ns=5,
            event_number=1,
        ),
        _book(
            market,
            DOWN_TOKEN_ID,
            received_at_ms=DECISION_MS - 600,
            monotonic_ns=6,
            event_number=2,
        ),
    )
    return events, books


def test_round17_condition_materialization_is_deterministic_and_target_free() -> None:
    events, books = _sources()
    kwargs = {
        "market": _market(),
        "admission": _admission(),
        "base_rows": _base_rows(),
        "events": events,
        "books": books,
    }

    first = materialize_round17_condition_rows(**kwargs)
    second = materialize_round17_condition_rows(**kwargs)

    assert first.dataset_sha256 == second.dataset_sha256
    assert first.rows == second.rows
    assert first.base_row_count == len(first.rows) == 2
    assert len(first.rows[0].values) == len(POLYMARKET_ROUND17_FEATURE_NAMES) == 188
    assert first.chainlink_event_count == 2
    assert first.spot_trade_count == first.perpetual_trade_count == 1
    assert first.up_book_count == first.down_book_count == 1
    assert first.training_authority is False
    assert first.trading_authority is False
    serialized = first.asdict()
    assert serialized["labels_consulted"] is False
    assert serialized["outcomes_consulted"] is False
    assert serialized["resolutions_consulted"] is False
    assert serialized["model_scores_consulted"] is False


def test_round17_trade_parser_rejects_schema_drift_and_event_tampering() -> None:
    raw = _trade(
        "spot",
        received_at_ms=DECISION_MS - 10,
        monotonic_ns=1,
        trade_id=1,
    )
    parsed = parse_round17_binance_trade_event(raw)
    assert parsed is not None
    assert parsed.trade_id == 1
    assert parsed.trading_authority is False

    aggregate = replace(
        raw,
        event={
            **raw.event,
            "stream": "btcusdt@aggTrade",
        },
    )
    aggregate = replace(aggregate, event_sha256=_sha256(aggregate.event))
    with pytest.raises(ValueError, match="raw-trade stream"):
        parse_round17_binance_trade_event(aggregate)

    with pytest.raises(ValueError, match="integrity differs"):
        parse_round17_binance_trade_event(
            replace(raw, event={**raw.event, "unexpected": True})
        )


def test_round17_trade_parser_ignores_only_exact_futures_zero_sentinel() -> None:
    sentinel = _trade(
        "perpetual",
        received_at_ms=DECISION_MS - 10,
        monotonic_ns=1,
        trade_id=1,
        price="0",
        quantity="0",
    )
    body = dict(sentinel.event)
    data = dict(body["data"])  # type: ignore[arg-type]
    data.update({"X": "NA", "st": 1})
    body["data"] = data
    sentinel = replace(sentinel, event=body, event_sha256=_sha256(body))

    assert parse_round17_binance_trade_event(sentinel) is None
    malformed_body = dict(body)
    malformed_data = dict(data)
    malformed_data["q"] = "1"
    malformed_body["data"] = malformed_data
    malformed = replace(
        sentinel,
        event=malformed_body,
        event_sha256=_sha256(malformed_body),
    )
    with pytest.raises(ValueError, match="finite positive"):
        parse_round17_binance_trade_event(malformed)


def test_round17_condition_rejects_cross_connection_feature_windows() -> None:
    events, books = _sources()
    crossing = _chainlink(
        received_at_ms=DECISION_MS - 50,
        monotonic_ns=7,
        connection_id="rtds:chainlink:btc:" + "e" * 32,
    )
    ordered = tuple(
        sorted((*events, crossing), key=lambda event: event.received_monotonic_ns)
    )
    with pytest.raises(ValueError, match="crossed a connection epoch"):
        materialize_round17_condition_rows(
            market=_market(),
            admission=_admission(),
            base_rows=_base_rows(),
            events=ordered,
            books=books,
        )


def test_round17_dataset_source_never_reads_resolution_or_target_fields() -> None:
    source = (
        ROOT / "src" / "simple_ai_trading" / "polymarket_round17_dataset.py"
    ).read_text(encoding="utf-8")

    assert "polymarket_resolution_evidence" not in source
    assert "winning_outcome" not in source
    assert "close_price" not in source
    assert "official_up" not in source
