from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json

import pytest

from simple_ai_trading.polymarket import (
    GAMMA_MARKETS_URL,
    PolymarketPublicClient,
    parse_polymarket_five_minute_market,
    validate_clob_order_book,
    validate_clob_market_info,
)


EPOCH = 1_784_058_600


def _market(asset: str = "BTC", *, epoch: int = EPOCH) -> dict[str, object]:
    lower = asset.lower()
    token_base = {"BTC": "7", "ETH": "8", "SOL": "9"}[asset]
    return {
        "id": f"market-{asset}-{epoch}",
        "question": f"{asset} Up or Down",
        "conditionId": "0x" + token_base * 64,
        "slug": f"{lower}-updown-5m-{epoch}",
        "eventStartTime": "2026-07-14T19:50:00Z",
        "endDate": "2026-07-14T19:55:00Z",
        "active": True,
        "closed": False,
        "enableOrderBook": True,
        "acceptingOrders": True,
        "clobTokenIds": json.dumps([token_base * 40, token_base * 39 + "1"]),
        "outcomes": '["Up", "Down"]',
        "orderPriceMinTickSize": 0.01,
        "orderMinSize": 5,
        "feesEnabled": True,
        "feeSchedule": {
            "exponent": 1,
            "rate": 0.07,
            "takerOnly": True,
            "rebateRate": 0.2,
        },
        "liquidityNum": 20_000.5,
        "volumeNum": 50_000.25,
        "resolutionSource": f"https://data.chain.link/streams/{lower}-usd",
    }


def test_market_parser_requires_exact_five_minute_chainlink_contract() -> None:
    market = parse_polymarket_five_minute_market(_market())

    assert market.asset == "BTC"
    assert market.event_start_ms == EPOCH * 1_000
    assert market.end_ms - market.event_start_ms == 300_000
    assert market.tick_size == Decimal("0.01")
    assert market.minimum_order_size == Decimal("5")
    assert market.fee_schedule.rate == Decimal("0.07")
    assert market.fee_schedule.fee_model()(
        Decimal("0.5"), Decimal("100"), "taker"
    ) == Decimal("1.75000")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("eventStartTime", "2026-07-14T19:50:01Z", "event window"),
        ("endDate", "2026-07-14T19:56:00Z", "event window"),
        ("resolutionSource", "https://example.com/btc", "Chainlink"),
        ("acceptingOrders", False, "accepting orders"),
        ("outcomes", '["Yes", "No"]', "Up/Down"),
    ],
)
def test_market_parser_fails_closed_on_metadata_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _market()
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        parse_polymarket_five_minute_market(payload)


def test_clob_market_info_must_match_gamma_tokens_tick_and_fee() -> None:
    market = parse_polymarket_five_minute_market(_market())
    payload = {
        "c": market.condition_id,
        "t": [
            {"t": market.up_token_id, "o": "Up"},
            {"t": market.down_token_id, "o": "Down"},
        ],
        "mos": 5,
        "mts": 0.01,
        "mbf": 1000,
        "tbf": 1000,
        "itode": True,
        "fd": {"r": 0.07, "e": 1, "to": True},
    }

    evidence = validate_clob_market_info(market, payload)
    assert evidence["taker_base_fee"] == 1000
    assert evidence["taker_order_delay_enabled"] is True

    drifted = deepcopy(payload)
    drifted["mts"] = 0.001
    with pytest.raises(ValueError, match="order parameters drifted"):
        validate_clob_market_info(market, drifted)


def test_clob_book_normalizes_raw_order_without_inventing_depth() -> None:
    market = parse_polymarket_five_minute_market(_market())
    book = validate_clob_order_book(
        market,
        market.up_token_id,
        {
            "market": market.condition_id,
            "asset_id": market.up_token_id,
            "timestamp": str(EPOCH * 1_000 + 1_000),
            "hash": "0xbook",
            "bids": [
                {"price": "0.48", "size": "10"},
                {"price": "0.50", "size": "5"},
            ],
            "asks": [
                {"price": "0.54", "size": "20"},
                {"price": "0.52", "size": "4"},
            ],
        },
        received_wall_ms=EPOCH * 1_000 + 1_100,
        received_monotonic_ns=123,
    )

    assert [level.price for level in book.bids] == [Decimal("0.50"), Decimal("0.48")]
    assert [level.quantity for level in book.asks] == [Decimal("4"), Decimal("20")]
    assert book.source_payload_sha256

    with pytest.raises(ValueError, match="crossed or locked"):
        validate_clob_order_book(
            market,
            market.up_token_id,
            {
                "market": market.condition_id,
                "asset_id": market.up_token_id,
                "timestamp": str(EPOCH * 1_000 + 1_000),
                "bids": [{"price": "0.55", "size": "1"}],
                "asks": [{"price": "0.54", "size": "1"}],
            },
            received_wall_ms=EPOCH * 1_000 + 1_100,
            received_monotonic_ns=123,
        )


class _Response:
    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.content = json.dumps(payload).encode("utf-8")

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class _Session:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls: list[tuple[str, object, float]] = []

    def get(self, url: str, *, params: object, timeout: float) -> _Response:
        self.calls.append((url, params, timeout))
        return _Response(self.payload)


def test_discovery_batches_all_assets_and_never_uses_precompiled_market_ids() -> None:
    session = _Session([_market(asset) for asset in ("BTC", "ETH", "SOL")])
    client = PolymarketPublicClient(session=session, timeout_seconds=3)

    markets = client.discover_five_minute_markets(
        now_ms=EPOCH * 1_000 + 30_000,
        include_next=True,
    )

    assert [market.asset for market in markets] == ["BTC", "ETH", "SOL"]
    assert len(session.calls) == 1
    url, params, timeout = session.calls[0]
    assert url == GAMMA_MARKETS_URL
    assert timeout == 3
    assert sum(1 for key, _ in params if key == "slug") == 6
    assert ("closed", "false") in params
