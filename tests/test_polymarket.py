from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json

import pytest

from simple_ai_trading.polymarket import (
    CLOB_BASE_URL,
    GAMMA_MARKETS_URL,
    POLYMARKET_FIVE_MINUTE_RESOLUTION_SOURCES,
    POLYMARKET_REQUIRED_CLOB_PROTOCOL_VERSION,
    PolymarketPublicClient,
    parse_polymarket_fifteen_minute_market,
    parse_polymarket_five_minute_market,
    validate_clob_order_book,
    validate_clob_market_info,
)


EPOCH = 1_784_058_600
EPOCH_15 = EPOCH - 300


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


def _fifteen_minute_market(*, epoch: int = EPOCH_15) -> dict[str, object]:
    payload = _market("BTC", epoch=epoch)
    payload["slug"] = f"btc-updown-15m-{epoch}"
    payload["eventStartTime"] = "2026-07-14T19:45:00Z"
    payload["endDate"] = "2026-07-14T20:00:00Z"
    return payload


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


def test_five_minute_parser_accepts_the_exact_twap_30s_source_contract() -> None:
    payload = _market()
    payload["resolutionSource"] = (
        "https://data.chain.link/streams/btc-usd-twap-30s-streams"
    )
    payload["cryptoMarketConfigId"] = "btc-5m-twap-30"
    payload["cryptoMarketConfig"] = {
        "asset": "btc",
        "duration": "5m",
        "id": "btc-5m-twap-30",
        "twapEnabled": True,
        "twapLookbackSeconds": 30,
    }

    market = parse_polymarket_five_minute_market(payload)

    assert market.resolution_source == payload["resolutionSource"]
    assert tuple(POLYMARKET_FIVE_MINUTE_RESOLUTION_SOURCES["BTC"]) == (
        "https://data.chain.link/streams/btc-usd",
        "https://data.chain.link/streams/btc-usd-twap-30s-streams",
        "https://data.chain.link/streams/btc-usd-twap-60s-streams",
    )

    payload["cryptoMarketConfig"]["twapLookbackSeconds"] = 60
    with pytest.raises(ValueError, match="TWAP resolution contract"):
        parse_polymarket_five_minute_market(payload)


def test_five_minute_parser_accepts_the_exact_twap_60s_source_contract() -> None:
    payload = _market()
    payload["resolutionSource"] = (
        "https://data.chain.link/streams/btc-usd-twap-60s-streams"
    )
    payload["cryptoMarketConfigId"] = "btc-5m-twap-60"
    payload["cryptoMarketConfig"] = {
        "asset": "btc",
        "duration": "5m",
        "id": "btc-5m-twap-60",
        "twapEnabled": True,
        "twapLookbackSeconds": 60,
    }

    market = parse_polymarket_five_minute_market(payload)

    assert market.resolution_source == payload["resolutionSource"]
    payload["cryptoMarketConfig"]["id"] = "btc-5m-twap-30"
    with pytest.raises(ValueError, match="TWAP resolution contract"):
        parse_polymarket_five_minute_market(payload)


def test_fifteen_minute_parser_is_separately_typed_and_exact() -> None:
    market = parse_polymarket_fifteen_minute_market(_fifteen_minute_market())

    assert market.asset == "BTC"
    assert market.event_start_ms == EPOCH_15 * 1_000
    assert market.end_ms - market.event_start_ms == 900_000
    assert market.horizon_minutes == 15
    assert market.asdict()["schema_version"] == "polymarket-crypto-15m-market-v1"
    assert market.asdict()["horizon_minutes"] == 15


def test_fifteen_minute_parser_rejects_wrong_horizon_and_asset() -> None:
    wrong_duration = _fifteen_minute_market()
    wrong_duration["endDate"] = "2026-07-14T19:50:00Z"
    with pytest.raises(ValueError, match="fifteen-minute event window"):
        parse_polymarket_fifteen_minute_market(wrong_duration)

    wrong_asset = _fifteen_minute_market()
    wrong_asset["slug"] = f"eth-updown-15m-{EPOCH_15}"
    with pytest.raises(ValueError, match="supported BTC fifteen-minute"):
        parse_polymarket_fifteen_minute_market(wrong_asset)

    wrong_source = _fifteen_minute_market()
    wrong_source["resolutionSource"] = (
        "https://data.chain.link/streams/btc-usd-twap-30s-streams"
    )
    with pytest.raises(ValueError, match="approved Chainlink"):
        parse_polymarket_fifteen_minute_market(wrong_source)


def test_market_parser_applies_recorded_v2_fee_exponent_exactly() -> None:
    payload = _market()
    payload["feeSchedule"] = {
        "exponent": 2,
        "rate": 0.25,
        "takerOnly": True,
        "rebateRate": 0.2,
    }
    market = parse_polymarket_five_minute_market(payload)
    fee = market.fee_schedule.fee_model()

    assert fee(Decimal("0.5"), Decimal("100"), "taker") == Decimal("1.56250")
    assert fee(Decimal("0.3"), Decimal("100"), "taker") == Decimal("1.10250")


def test_market_parser_rejects_fractional_fee_exponents() -> None:
    payload = _market()
    assert isinstance(payload["feeSchedule"], dict)
    payload["feeSchedule"]["exponent"] = 1.5

    with pytest.raises(ValueError, match="positive integer"):
        parse_polymarket_five_minute_market(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("eventStartTime", "2026-07-14T19:50:01Z", "event window"),
        ("endDate", "2026-07-14T19:56:00Z", "event window"),
        ("resolutionSource", "https://example.com/btc", "approved Chainlink"),
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
    assert evidence["general_order_delay_seconds"] == 0

    drifted = deepcopy(payload)
    drifted["mts"] = 0.001
    with pytest.raises(ValueError, match="order parameters drifted"):
        validate_clob_market_info(market, drifted)

    for field in ("mbf", "tbf", "oas", "sd"):
        invalid = deepcopy(payload)
        invalid[field] = 1.5
        with pytest.raises(ValueError, match="nonnegative integer"):
            validate_clob_market_info(market, invalid)


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


class _RoutingSession:
    def __init__(self, payloads: dict[str, object]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, object, float]] = []

    def get(self, url: str, *, params: object, timeout: float) -> _Response:
        self.calls.append((url, params, timeout))
        return _Response(self.payloads[url])


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


def test_discovery_can_request_only_btc_without_precompiled_market_ids() -> None:
    session = _Session([_market("BTC")])
    client = PolymarketPublicClient(session=session, timeout_seconds=3)

    markets = client.discover_five_minute_markets(
        now_ms=EPOCH * 1_000 + 30_000,
        include_next=True,
        assets=("BTC",),
    )

    assert [market.asset for market in markets] == ["BTC"]
    _url, params, _timeout = session.calls[0]
    slugs = [value for key, value in params if key == "slug"]
    assert slugs == [
        f"btc-updown-5m-{EPOCH}",
        f"btc-updown-5m-{EPOCH + 300}",
    ]


def test_discovery_can_pin_the_exact_twap_source_regime() -> None:
    payload = _market("BTC")
    payload["resolutionSource"] = (
        "https://data.chain.link/streams/btc-usd-twap-30s-streams"
    )
    payload["cryptoMarketConfigId"] = "btc-5m-twap-30"
    payload["cryptoMarketConfig"] = {
        "asset": "btc",
        "duration": "5m",
        "id": "btc-5m-twap-30",
        "twapEnabled": True,
        "twapLookbackSeconds": 30,
    }
    client = PolymarketPublicClient(
        session=_Session([payload]),
        required_five_minute_resolution_sources={
            "BTC": "https://data.chain.link/streams/btc-usd-twap-30s-streams"
        },
    )

    markets = client.discover_five_minute_markets(
        now_ms=EPOCH * 1_000 + 30_000,
        assets=("BTC",),
    )

    assert len(markets) == 1
    assert markets[0].resolution_source.endswith("btc-usd-twap-30s-streams")


def test_discovery_fails_closed_on_configured_source_regime_drift() -> None:
    client = PolymarketPublicClient(
        session=_Session([_market("BTC")]),
        required_five_minute_resolution_sources={
            "BTC": "https://data.chain.link/streams/btc-usd-twap-30s-streams"
        },
    )

    with pytest.raises(ValueError, match="configured exact source"):
        client.discover_five_minute_markets(
            now_ms=EPOCH * 1_000 + 30_000,
            assets=("BTC",),
        )

    with pytest.raises(ValueError, match="lacks a configured"):
        client.discover_five_minute_markets(
            now_ms=EPOCH * 1_000 + 30_000,
            assets=("ETH",),
        )


def test_fifteen_minute_discovery_uses_utc_epochs_not_market_ids() -> None:
    session = _Session([_fifteen_minute_market()])
    client = PolymarketPublicClient(session=session, timeout_seconds=3)

    markets = client.discover_fifteen_minute_markets(
        now_ms=EPOCH_15 * 1_000 + 30_000,
        include_next=True,
    )

    assert [market.asset for market in markets] == ["BTC"]
    _url, params, _timeout = session.calls[0]
    slugs = [value for key, value in params if key == "slug"]
    assert slugs == [
        f"btc-updown-15m-{EPOCH_15}",
        f"btc-updown-15m-{EPOCH_15 + 900}",
    ]


def test_public_client_uses_official_full_market_resolution_endpoints() -> None:
    condition_id = "0x" + "7" * 64
    version_url = f"{CLOB_BASE_URL}/version"
    market_url = f"{CLOB_BASE_URL}/markets/{condition_id}"
    gamma_url = f"{GAMMA_MARKETS_URL}/1001"
    session = _RoutingSession(
        {
            version_url: {"version": POLYMARKET_REQUIRED_CLOB_PROTOCOL_VERSION},
            market_url: {"condition_id": condition_id},
            gamma_url: {"condition_id": condition_id},
        }
    )
    client = PolymarketPublicClient(session=session, timeout_seconds=3)

    assert client.clob_market(condition_id) == {"condition_id": condition_id}
    assert session.calls[-1][0] == market_url
    assert client.clob_market(condition_id) == {"condition_id": condition_id}
    assert sum(url == version_url for url, _params, _timeout in session.calls) == 1
    assert client.gamma_market("1001") == {"condition_id": condition_id}
    assert session.calls[-1][0] == gamma_url


def test_public_client_batches_gamma_markets_with_exact_fallback() -> None:
    batch_url = f"{GAMMA_MARKETS_URL}/keyset"
    exact_url = f"{GAMMA_MARKETS_URL}/1002"
    session = _RoutingSession(
        {
            batch_url: {"markets": [{"id": "1001"}], "next_cursor": ""},
            exact_url: {"id": "1002"},
        }
    )
    client = PolymarketPublicClient(session=session, timeout_seconds=3)

    markets = client.gamma_markets(("1001", "1002"))

    assert tuple(markets) == ("1001", "1002")
    assert markets["1002"] == {"id": "1002"}
    _url, params, _timeout = session.calls[0]
    assert ("id", "1001") in params
    assert ("id", "1002") in params
    assert ("closed", "true") in params
    assert ("limit", "2") in params


@pytest.mark.parametrize(
    "market_ids",
    [
        (),
        ("1001", "1001"),
        ("not-numeric",),
        tuple(str(index) for index in range(101)),
    ],
)
def test_public_client_rejects_invalid_gamma_batches(
    market_ids: tuple[str, ...],
) -> None:
    client = PolymarketPublicClient(session=_RoutingSession({}), timeout_seconds=3)

    with pytest.raises(ValueError, match="batch is invalid"):
        client.gamma_markets(market_ids)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "must be an object"),
        ({}, "is malformed"),
        ({"markets": [None]}, "entry is malformed"),
        ({"markets": [{"id": "9999"}]}, "identity differs"),
    ],
)
def test_public_client_rejects_malformed_gamma_batch_responses(
    payload: object,
    message: str,
) -> None:
    client = PolymarketPublicClient(session=_Session(payload), timeout_seconds=3)

    with pytest.raises(ValueError, match=message):
        client.gamma_markets(("1001",))


def test_public_client_rejects_mismatched_exact_gamma_fallback() -> None:
    batch_url = f"{GAMMA_MARKETS_URL}/keyset"
    exact_url = f"{GAMMA_MARKETS_URL}/1001"
    client = PolymarketPublicClient(
        session=_RoutingSession(
            {
                batch_url: {"markets": [], "next_cursor": ""},
                exact_url: {"id": "9999"},
            }
        ),
        timeout_seconds=3,
    )

    with pytest.raises(ValueError, match="exact Gamma market identity differs"):
        client.gamma_markets(("1001",))


@pytest.mark.parametrize("payload", [{"version": 1}, {"version": "2"}, {}, []])
def test_public_client_fails_closed_on_unknown_clob_protocol(payload: object) -> None:
    version_url = f"{CLOB_BASE_URL}/version"
    client = PolymarketPublicClient(
        session=_RoutingSession({version_url: payload}),
        timeout_seconds=3,
    )

    with pytest.raises(ValueError, match="protocol"):
        client.protocol_version()


def test_public_client_rejects_invalid_ids_before_network_io() -> None:
    session = _RoutingSession({})
    client = PolymarketPublicClient(session=session, timeout_seconds=3)

    with pytest.raises(ValueError, match="condition_id"):
        client.clob_market("not-a-condition")

    assert session.calls == []
