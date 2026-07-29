from __future__ import annotations

from decimal import Decimal
import json

import pytest

from simple_ai_trading.polymarket_btc_reference import (
    OfficialPolymarketBtcReferenceClient,
    PolymarketBtcEndpointEstimator,
    PolymarketBtcReferenceWindow,
    PolymarketChainlinkBtcTick,
    parse_polymarket_btc_reference_window,
    parse_polymarket_chainlink_btc_tick,
)


START_MS = 1_800_000_000_000
END_MS = START_MS + 300_000


def _reference(*, completed: bool = False) -> PolymarketBtcReferenceWindow:
    payload = {
        "openPrice": 100_000.0,
        "closePrice": 100_100.0 if completed else None,
        "timestamp": END_MS + 1_000 if completed else START_MS + 60_000,
        "completed": completed,
        "incomplete": not completed,
        "cached": True,
    }
    return parse_polymarket_btc_reference_window(
        payload,
        event_start_ms=START_MS,
        end_ms=END_MS,
    )


def _tick(
    index: int,
    *,
    price: str | None = None,
    received_offset_ms: int = 10,
) -> PolymarketChainlinkBtcTick:
    source_time = START_MS + index * 1_000
    return PolymarketChainlinkBtcTick(
        source_time_ms=source_time,
        publisher_time_ms=source_time + 5,
        received_at_ms=source_time + received_offset_ms,
        price=Decimal(price or str(100_000 + index)),
        source_payload_sha256=f"{index + 1:064x}",
    )


def test_reference_parser_binds_exact_window_and_completion() -> None:
    active = _reference()
    completed = _reference(completed=True)

    assert active.winning_outcome is None
    assert completed.winning_outcome == "Up"
    assert completed.asdict()["close_price"] == "100100.0"


def test_reference_parser_rejects_schema_and_state_drift() -> None:
    payload = {
        "openPrice": 100_000,
        "closePrice": None,
        "timestamp": START_MS,
        "completed": False,
        "incomplete": True,
        "cached": True,
    }
    with pytest.raises(ValueError, match="schema drifted"):
        parse_polymarket_btc_reference_window(
            {**payload, "extra": 1},
            event_start_ms=START_MS,
            end_ms=END_MS,
        )
    with pytest.raises(ValueError, match="close state contradicts"):
        parse_polymarket_btc_reference_window(
            {**payload, "completed": True, "incomplete": False},
            event_start_ms=START_MS,
            end_ms=END_MS,
        )
    with pytest.raises(ValueError, match="exact five minutes"):
        parse_polymarket_btc_reference_window(
            payload,
            event_start_ms=START_MS,
            end_ms=END_MS + 1_000,
        )


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.content = payload

    def raise_for_status(self) -> None:
        return None


class _Session:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[tuple[str, object, float]] = []

    def get(self, url: str, *, params: object, timeout: float) -> _Response:
        self.calls.append((url, params, timeout))
        return _Response(self.payload)


def test_reference_client_uses_exact_first_party_query() -> None:
    session = _Session(
        json.dumps(
            {
                "openPrice": 100_000,
                "closePrice": None,
                "timestamp": START_MS + 10_000,
                "completed": False,
                "incomplete": True,
                "cached": True,
            }
        ).encode("ascii")
    )
    client = OfficialPolymarketBtcReferenceClient(session=session)

    result = client.window(event_start_ms=START_MS, end_ms=END_MS)

    assert result.open_price == Decimal("100000")
    _, params, timeout = session.calls[0]
    assert params == {
        "symbol": "BTC",
        "eventStartTime": "2027-01-15T08:00:00Z",
        "variant": "fiveminute",
        "endDate": "2027-01-15T08:05:00Z",
    }
    assert timeout == 10.0


def test_reference_client_rejects_duplicate_json_keys() -> None:
    session = _Session(
        b'{"openPrice":1,"openPrice":2,"closePrice":null,'
        b'"timestamp":1800000000000,"completed":false,'
        b'"incomplete":true,"cached":true}'
    )
    client = OfficialPolymarketBtcReferenceClient(session=session)

    with pytest.raises(ValueError, match="duplicate keys"):
        client.window(event_start_ms=START_MS, end_ms=END_MS)


def test_chainlink_rtds_parser_requires_exact_btc_identity() -> None:
    payload = {
        "topic": "crypto_prices_chainlink",
        "type": "update",
        "timestamp": START_MS + 1_005,
        "payload": {
            "symbol": "btc/usd",
            "timestamp": START_MS + 1_000,
            "value": 100_001.25,
        },
    }

    tick = parse_polymarket_chainlink_btc_tick(
        payload,
        received_at_ms=START_MS + 1_010,
    )

    assert tick.price == Decimal("100001.25")
    assert tick.source_payload_sha256
    with pytest.raises(ValueError, match="not BTC/USD"):
        parse_polymarket_chainlink_btc_tick(
            {
                **payload,
                "payload": {**payload["payload"], "symbol": "eth/usd"},
            },
            received_at_ms=START_MS + 1_010,
        )


def test_chainlink_rtds_parser_cross_checks_live_exact_value_extension() -> None:
    payload = {
        "connection_id": "connection-123",
        "topic": "crypto_prices_chainlink",
        "type": "update",
        "timestamp": START_MS + 1_005,
        "payload": {
            "full_accuracy_value": "100001250000000000000000",
            "symbol": "btc/usd",
            "timestamp": START_MS + 1_000,
            "value": 100_001.25,
        },
    }

    tick = parse_polymarket_chainlink_btc_tick(
        payload,
        received_at_ms=START_MS + 1_010,
    )

    assert tick.price == Decimal("100001.25")
    with pytest.raises(ValueError, match="exact and rounded"):
        parse_polymarket_chainlink_btc_tick(
            {
                **payload,
                "payload": {
                    **payload["payload"],
                    "full_accuracy_value": "100002250000000000000000",
                },
            },
            received_at_ms=START_MS + 1_010,
        )


def test_endpoint_estimator_is_causal_authority_free_and_directional() -> None:
    estimator = PolymarketBtcEndpointEstimator(
        minimum_coverage_seconds=20,
        minimum_return_count=20,
    )
    for index in range(31):
        estimator.observe(_tick(index))

    estimate = estimator.estimate(
        _reference(),
        observed_at_ms=START_MS + 30_050,
    )

    assert estimate.available is True
    assert estimate.probability_up is not None
    assert estimate.probability_up > 0.5
    assert estimate.tick_count == 31
    assert estimate.grants_execution_authority is False


def test_endpoint_estimator_fails_closed_on_sparse_stale_or_terminal_data() -> None:
    estimator = PolymarketBtcEndpointEstimator(
        minimum_coverage_seconds=20,
        minimum_return_count=20,
        maximum_staleness_ms=1_000,
    )
    for index in range(10):
        estimator.observe(_tick(index))
    sparse = estimator.estimate(
        _reference(),
        observed_at_ms=START_MS + 9_050,
    )
    stale = estimator.estimate(
        _reference(),
        observed_at_ms=START_MS + 20_000,
    )
    terminal = estimator.estimate(
        _reference(completed=True),
        observed_at_ms=END_MS + 1_000,
    )

    assert sparse.available is False
    assert "insufficient_return_count" in sparse.reasons
    assert "chainlink_tick_stale_or_future" in stale.reasons
    assert "market_window_not_live" in terminal.reasons


def test_endpoint_estimator_rejects_contradictory_or_regressed_ticks() -> None:
    estimator = PolymarketBtcEndpointEstimator()
    assert estimator.observe(_tick(1)) is True
    assert estimator.observe(_tick(1)) is False
    with pytest.raises(ValueError, match="duplicate time contradicts"):
        estimator.observe(_tick(1, price="100123"))
    with pytest.raises(ValueError, match="source time regressed"):
        estimator.observe(_tick(0))
