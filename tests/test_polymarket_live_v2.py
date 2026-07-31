from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import time
from typing import Mapping

import pytest

from simple_ai_trading.polymarket import CLOB_BASE_URL
from simple_ai_trading.polymarket_live import (
    PolymarketLiveBlocked,
    PolymarketLiveOrderIntent,
    PolymarketPreparedOrder,
    PolymarketVenueRejected,
)
from simple_ai_trading.polymarket_live_v2 import (
    OfficialPolymarketV2Venue,
    POLYGON_CHAIN_ID,
    POLYMARKET_DATA_POSITIONS_URL,
    POLYMARKET_GEOBLOCK_URL,
    PolymarketLiveCredentials,
)


MARKET_ID = "0x" + "1" * 64
TOKEN_ID = "1" * 40
PRIVATE_KEY = "0x" + "1" * 64


def _credentials() -> PolymarketLiveCredentials:
    eth_account = pytest.importorskip("eth_account")
    address = eth_account.Account.from_key(PRIVATE_KEY).address.lower()
    return PolymarketLiveCredentials(
        private_key=PRIVATE_KEY,
        api_key="offline-api-key",
        api_secret="offline-api-secret",
        api_passphrase="offline-passphrase",
        funder_address=address,
        signature_type=0,
    )


def _intent(*, side: str = "BUY") -> PolymarketLiveOrderIntent:
    now = int(time.time() * 1_000)
    return PolymarketLiveOrderIntent(
        intent_id="official-v2-test-intent",
        bot_id="official-v2-test-bot",
        market_id=MARKET_ID,
        token_id=TOKEN_ID,
        symbol="BTC",
        outcome="Up",
        side=side,
        order_type="FAK",
        limit_price=Decimal("0.50"),
        quantity=Decimal("5"),
        fee_reserve_quote=Decimal("0.10"),
        created_at_ms=now,
        expires_at_ms=now + 120_000,
        parent_intent_id="official-parent-intent" if side == "SELL" else "",
        closing_only=side == "SELL",
    )


class FakeResponse:
    def __init__(self, value: object, *, status_code: int = 200) -> None:
        self.value = value
        self.status_code = status_code
        self.content = json.dumps(value).encode("utf-8")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> object:
        return self.value


class FakeSession:
    def __init__(self, responses: Mapping[str, object]) -> None:
        self.responses = dict(responses)

    def get(
        self,
        url: str,
        *,
        params: object = None,
        timeout: float,
    ) -> FakeResponse:
        del params, timeout
        return FakeResponse(self.responses[url])


class FakeClient:
    def __init__(self) -> None:
        self.balance_response: object = {}
        self.trades: list[object] = []
        self.cancel_response: object = {"canceled": [], "not_canceled": {}}
        self.post_response: object = {}
        self.post_error: Exception | None = None
        self.post_calls = 0
        self.orders: dict[str, object] = {}
        self.get_order_calls: list[str] = []
        self.get_order_error: Exception | None = None
        self.order_book: object = {}
        self.market_info: object = {}
        self.tick_size = "0.01"
        self.neg_risk = False

    def get_balance_allowance(self, params: object) -> object:
        del params
        return self.balance_response

    def get_trades(self, params: object) -> list[object]:
        del params
        return self.trades

    def get_order(self, order_id: str) -> object:
        self.get_order_calls.append(order_id)
        if self.get_order_error is not None:
            raise self.get_order_error
        return self.orders[order_id]

    def get_order_book(self, token_id: str) -> object:
        del token_id
        return self.order_book

    def get_clob_market_info(self, condition_id: str) -> object:
        del condition_id
        return self.market_info

    def get_tick_size(self, token_id: str) -> str:
        del token_id
        return self.tick_size

    def get_neg_risk(self, token_id: str) -> bool:
        del token_id
        return self.neg_risk

    def cancel_orders(self, order_ids: list[str]) -> object:
        del order_ids
        return self.cancel_response

    def post_order(
        self,
        order: object,
        *,
        order_type: str,
        defer_exec: bool,
    ) -> object:
        del order, order_type, defer_exec
        self.post_calls += 1
        if self.post_error is not None:
            raise self.post_error
        return self.post_response


def test_official_sdk_builds_exact_v2_hash_and_preserves_economics() -> None:
    clob = pytest.importorskip("py_clob_client_v2")
    from py_clob_client_v2.order_utils import ExchangeOrderBuilderV2
    from py_clob_client_v2.config import get_contract_config

    credentials = _credentials()
    client = clob.ClobClient(
        host=CLOB_BASE_URL,
        chain_id=POLYGON_CHAIN_ID,
        key=credentials.private_key,
        creds=clob.ApiCreds(
            api_key=credentials.api_key,
            api_secret=credentials.api_secret,
            api_passphrase=credentials.api_passphrase,
        ),
        signature_type=credentials.signature_type,
        funder=credentials.funder_address,
        use_server_time=False,
        retry_on_error=False,
    )
    client.get_version = lambda: 2
    client.get_tick_size = lambda token_id: "0.01"
    venue = OfficialPolymarketV2Venue(credentials, client=client)

    prepared = venue.prepare_order(
        _intent(),
        tick_size=Decimal("0.01"),
        neg_risk=False,
    )

    config = get_contract_config(POLYGON_CHAIN_ID)
    builder = ExchangeOrderBuilderV2(
        config.exchange_v2,
        POLYGON_CHAIN_ID,
        client.builder.signer,
    )
    expected = builder.build_order_hash(
        builder.build_order_typed_data(prepared.opaque_signed_order)
    ).lower()
    assert prepared.expected_order_id == expected
    assert str(prepared.opaque_signed_order.makerAmount) == "2500000"
    assert str(prepared.opaque_signed_order.takerAmount) == "5000000"
    assert prepared.opaque_signed_order.metadata == prepared.intent.metadata
    assert prepared.opaque_signed_order.maker.lower() == credentials.funder_address
    assert prepared.opaque_signed_order.signer.lower() == credentials.funder_address
    assert int(prepared.opaque_signed_order.signatureType) == 0


def test_official_sdk_deposit_wallet_binds_maker_signer_and_signature_type() -> None:
    clob = pytest.importorskip("py_clob_client_v2")
    credentials = PolymarketLiveCredentials(
        private_key=PRIVATE_KEY,
        api_key="offline-api-key",
        api_secret="offline-api-secret",
        api_passphrase="offline-passphrase",
        funder_address="0x" + "2" * 40,
        signature_type=3,
    )
    client = clob.ClobClient(
        host=CLOB_BASE_URL,
        chain_id=POLYGON_CHAIN_ID,
        key=credentials.private_key,
        creds=clob.ApiCreds(
            api_key=credentials.api_key,
            api_secret=credentials.api_secret,
            api_passphrase=credentials.api_passphrase,
        ),
        signature_type=credentials.signature_type,
        funder=credentials.funder_address,
        use_server_time=False,
        retry_on_error=False,
    )
    client.get_version = lambda: 2
    client.get_tick_size = lambda token_id: "0.01"

    prepared = OfficialPolymarketV2Venue(
        credentials,
        client=client,
    ).prepare_order(
        _intent(),
        tick_size=Decimal("0.01"),
        neg_risk=False,
    )
    signed = prepared.opaque_signed_order

    assert signed.maker.lower() == credentials.funder_address
    assert signed.signer.lower() == credentials.funder_address
    assert int(signed.signatureType) == 3


@pytest.mark.parametrize(
    ("side", "neg_risk", "expected_asset", "expected_token"),
    [
        ("BUY", False, "COLLATERAL", ""),
        ("SELL", True, "CONDITIONAL", TOKEN_ID),
    ],
)
def test_funding_selects_allowance_for_exact_v2_exchange(
    side: str,
    neg_risk: bool,
    expected_asset: str,
    expected_token: str,
) -> None:
    pytest.importorskip("py_clob_client_v2")
    from py_clob_client_v2.config import get_contract_config

    credentials = _credentials()
    client = FakeClient()
    config = get_contract_config(POLYGON_CHAIN_ID)
    exchange = config.neg_risk_exchange_v2 if neg_risk else config.exchange_v2
    client.balance_response = {
        "balance": "1234567",
        "allowances": {
            str(exchange).upper(): "7654321",
            "0x" + "f" * 40: "0",
        },
    }
    venue = OfficialPolymarketV2Venue(credentials, client=client)

    result = venue.funding(_intent(side=side), neg_risk=neg_risk)

    assert result.asset_type == expected_asset
    assert result.token_id == expected_token
    assert result.available_balance == Decimal("1.234567")
    assert result.available_allowance == Decimal("7.654321")


def test_trade_parser_binds_maker_fill_to_exact_owned_hash_and_normalizes_status() -> (
    None
):
    pytest.importorskip("py_clob_client_v2")
    credentials = _credentials()
    client = FakeClient()
    owned = "0x" + "2" * 64
    client.trades = [
        {
            "id": "trade-official-0001",
            "taker_order_id": "0x" + "3" * 64,
            "market": MARKET_ID,
            "asset_id": TOKEN_ID,
            "side": "BUY",
            "size": "5",
            "price": "0.5",
            "status": "TRADE_STATUS_CONFIRMED",
            "last_update": str(int(time.time())),
            "maker_orders": [
                {
                    "order_id": owned,
                    "asset_id": TOKEN_ID,
                    "matched_amount": "5",
                    "price": "0.5",
                }
            ],
        }
    ]
    venue = OfficialPolymarketV2Venue(credentials, client=client)

    fills = venue.fills_for_orders((owned,), market_ids=(MARKET_ID,))

    assert len(fills) == 1
    assert fills[0].order_id == owned
    assert fills[0].side == "SELL"
    assert fills[0].status == "CONFIRMED"


def test_submission_transport_error_is_propagated_after_one_attempt() -> None:
    credentials = _credentials()
    client = FakeClient()
    client.post_error = TimeoutError("unknown outcome")
    venue = OfficialPolymarketV2Venue(credentials, client=client)
    prepared = PolymarketPreparedOrder(
        intent=_intent(),
        expected_order_id="0x" + "2" * 64,
        metadata=_intent().metadata,
        opaque_signed_order=object(),
    )

    with pytest.raises(TimeoutError, match="unknown outcome"):
        venue.submit_order(prepared)

    assert client.post_calls == 1


def test_duplicate_capable_http_400_requires_exact_hash_reconciliation() -> None:
    class ApiError(RuntimeError):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"HTTP {status_code}")
            self.status_code = status_code

    client = FakeClient()
    prepared = PolymarketPreparedOrder(
        intent=_intent(),
        expected_order_id="0x" + "2" * 64,
        metadata=_intent().metadata,
        opaque_signed_order=object(),
    )
    venue = OfficialPolymarketV2Venue(_credentials(), client=client)

    client.post_error = ApiError(400)
    with pytest.raises(ApiError, match="400"):
        venue.submit_order(prepared)

    client.post_error = ApiError(422)
    with pytest.raises(PolymarketVenueRejected, match="HTTP 422"):
        venue.submit_order(prepared)

    assert client.post_calls == 2


def test_exact_order_lookup_parses_only_requested_owned_hashes() -> None:
    credentials = _credentials()
    client = FakeClient()
    first = "0x" + "2" * 64
    second = "0x" + "3" * 64
    for order_id in (first, second):
        client.orders[order_id] = {
            "id": order_id,
            "market": MARKET_ID,
            "asset_id": TOKEN_ID,
            "maker_address": credentials.funder_address,
            "side": "BUY",
            "order_type": "FAK",
            "price": "0.50",
            "status": "ORDER_STATUS_CANCELED",
            "original_size": "5",
            "size_matched": "0",
        }
    venue = OfficialPolymarketV2Venue(credentials, client=client)

    orders = venue.orders_by_id((first, first, second))

    assert client.get_order_calls == [first, second]
    assert tuple(order.order_id for order in orders) == (first, second)
    assert all(order.status == "ORDER_STATUS_CANCELED" for order in orders)
    assert all(order.maker_address == credentials.funder_address for order in orders)
    assert all(order.price == Decimal("0.50") for order in orders)
    assert all(order.order_type == "FAK" for order in orders)


def test_exact_order_lookup_treats_only_authenticated_404_as_absent() -> None:
    class ApiError(RuntimeError):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"HTTP {status_code}")
            self.status_code = status_code

    credentials = _credentials()
    client = FakeClient()
    order_id = "0x" + "2" * 64
    client.get_order_error = ApiError(404)
    venue = OfficialPolymarketV2Venue(credentials, client=client)

    assert venue.orders_by_id((order_id,)) == ()

    client.get_order_error = ApiError(503)
    with pytest.raises(ApiError, match="503"):
        venue.orders_by_id((order_id,))


def test_exact_order_lookup_rejects_response_identity_mismatch() -> None:
    credentials = _credentials()
    client = FakeClient()
    requested = "0x" + "2" * 64
    client.orders[requested] = {
        "id": "0x" + "3" * 64,
        "market": MARKET_ID,
        "asset_id": TOKEN_ID,
        "maker_address": credentials.funder_address,
        "side": "BUY",
        "order_type": "FAK",
        "price": "0.50",
        "status": "ORDER_STATUS_LIVE",
        "original_size": "5",
        "size_matched": "0",
    }
    venue = OfficialPolymarketV2Venue(credentials, client=client)

    with pytest.raises(Exception, match="response ID differs"):
        venue.orders_by_id((requested,))


def test_exact_order_lookup_rejects_foreign_maker_address() -> None:
    credentials = _credentials()
    client = FakeClient()
    requested = "0x" + "2" * 64
    client.orders[requested] = {
        "id": requested,
        "market": MARKET_ID,
        "asset_id": TOKEN_ID,
        "maker_address": "0x" + "f" * 40,
        "side": "BUY",
        "order_type": "FAK",
        "price": "0.50",
        "status": "ORDER_STATUS_LIVE",
        "original_size": "5",
        "size_matched": "0",
    }
    venue = OfficialPolymarketV2Venue(credentials, client=client)

    with pytest.raises(PolymarketLiveBlocked, match="maker differs"):
        venue.orders_by_id((requested,))


def test_exact_order_lookup_rejects_invalid_requested_hash() -> None:
    venue = OfficialPolymarketV2Venue(_credentials(), client=FakeClient())

    with pytest.raises(ValueError, match="invalid ID"):
        venue.orders_by_id(("not-an-order",))


def test_open_quote_walks_exact_asks_and_reconciles_per_level_fees() -> None:
    credentials = _credentials()
    client = FakeClient()
    now = int(time.time() * 1_000)
    client.order_book = {
        "market": MARKET_ID,
        "asset_id": TOKEN_ID,
        "timestamp": str(now),
        "tick_size": "0.01",
        "min_order_size": "5",
        "neg_risk": False,
        "bids": [{"price": "0.48", "size": "10"}],
        "asks": [
            {"price": "0.50", "size": "3"},
            {"price": "0.51", "size": "4"},
        ],
    }
    client.market_info = {
        "c": MARKET_ID,
        "t": [{"t": TOKEN_ID, "o": "Up"}, {"t": "2" * 40, "o": "Down"}],
        "mos": 5,
        "mts": 0.01,
        "fd": {"r": 0.07, "e": 1, "to": True},
    }
    venue = OfficialPolymarketV2Venue(credentials, client=client)

    quote = venue.open_quote(
        market_id=MARKET_ID,
        token_id=TOKEN_ID,
        outcome="Up",
        quantity=Decimal("5"),
        maximum_book_age_ms=1_500,
    )

    assert quote.limit_price == Decimal("0.51")
    assert quote.average_price == Decimal("0.504")
    assert quote.fee_quote == Decimal("0.08749")
    assert quote.total_quote == Decimal("2.60749")
    assert quote.fee_per_share == Decimal("0.017498")
    assert quote.source_age_ms >= 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"asks": [{"price": "0.50", "size": "4"}]}, "cannot fill"),
        (
            {
                "bids": [{"price": "0.50", "size": "10"}],
                "asks": [{"price": "0.50", "size": "5"}],
            },
            "crossed or locked",
        ),
    ],
)
def test_open_quote_rejects_non_executable_books(
    mutation: Mapping[str, object],
    message: str,
) -> None:
    credentials = _credentials()
    client = FakeClient()
    now = int(time.time() * 1_000)
    client.order_book = {
        "market": MARKET_ID,
        "asset_id": TOKEN_ID,
        "timestamp": str(now),
        "tick_size": "0.01",
        "min_order_size": "5",
        "neg_risk": False,
        "bids": [{"price": "0.48", "size": "10"}],
        "asks": [{"price": "0.50", "size": "5"}],
        **mutation,
    }
    client.market_info = {
        "c": MARKET_ID,
        "t": [{"t": TOKEN_ID, "o": "Up"}],
        "mos": 5,
        "mts": 0.01,
        "fd": {"r": 0.07, "e": 1, "to": True},
    }
    venue = OfficialPolymarketV2Venue(credentials, client=client)

    with pytest.raises(PolymarketLiveBlocked, match=message):
        venue.open_quote(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            outcome="Up",
            quantity=Decimal("5"),
            maximum_book_age_ms=1_500,
        )


def test_open_quote_rejects_market_or_fee_parameter_drift() -> None:
    credentials = _credentials()
    client = FakeClient()
    now = int(time.time() * 1_000)
    client.order_book = {
        "market": MARKET_ID,
        "asset_id": TOKEN_ID,
        "timestamp": str(now),
        "tick_size": "0.01",
        "min_order_size": "5",
        "neg_risk": False,
        "bids": [{"price": "0.48", "size": "10"}],
        "asks": [{"price": "0.50", "size": "5"}],
    }
    client.market_info = {
        "c": MARKET_ID,
        "t": [{"t": TOKEN_ID, "o": "Up"}],
        "mos": 5,
        "mts": 0.01,
        "fd": {"r": 0.07, "e": 1, "to": False},
    }
    venue = OfficialPolymarketV2Venue(credentials, client=client)

    with pytest.raises(ValueError, match="fee parameters"):
        venue.open_quote(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            outcome="Up",
            quantity=Decimal("5"),
            maximum_book_age_ms=1_500,
        )
    client.market_info = {
        **client.market_info,
        "t": [{"t": "2" * 40, "o": "Down"}],
        "fd": {"r": 0.07, "e": 1, "to": True},
    }
    with pytest.raises(PolymarketLiveBlocked, match="market token differs"):
        venue.open_quote(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            outcome="Up",
            quantity=Decimal("5"),
            maximum_book_age_ms=1_500,
        )
    client.market_info = {
        **client.market_info,
        "t": [{"t": TOKEN_ID, "o": "Down"}],
    }
    with pytest.raises(PolymarketLiveBlocked, match="token outcome differs"):
        venue.open_quote(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            outcome="Up",
            quantity=Decimal("5"),
            maximum_book_age_ms=1_500,
        )


def test_close_quote_walks_exact_displayed_bids_and_cross_checks_parameters() -> None:
    credentials = _credentials()
    client = FakeClient()
    now = int(time.time() * 1_000)
    client.order_book = {
        "market": MARKET_ID,
        "asset_id": TOKEN_ID,
        "timestamp": str(now - 25),
        "hash": "0xbook",
        "bids": [
            {"price": "0.47", "size": "4"},
            {"price": "0.49", "size": "2"},
            {"price": "0.48", "size": "3"},
        ],
        "asks": [{"price": "0.51", "size": "10"}],
        "min_order_size": "5",
        "tick_size": "0.01",
        "neg_risk": False,
    }
    venue = OfficialPolymarketV2Venue(credentials, client=client)

    quote = venue.close_quote(
        market_id=MARKET_ID,
        token_id=TOKEN_ID,
        quantity=Decimal("5"),
        maximum_book_age_ms=500,
    )

    assert quote.limit_price == Decimal("0.48")
    assert quote.quantity == Decimal("5")
    assert quote.tick_size == Decimal("0.01")
    assert quote.neg_risk is False
    assert 0 <= quote.source_age_ms <= 500
    assert len(quote.book_payload_sha256) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"timestamp": "1"}, "stale"),
        ({"market": "0x" + "2" * 64}, "condition differs"),
        ({"asset_id": "2" * 40}, "token differs"),
        ({"bids": [{"price": "0.49", "size": "4"}]}, "cannot close"),
        (
            {
                "bids": [{"price": "0.51", "size": "5"}],
                "asks": [{"price": "0.51", "size": "5"}],
            },
            "crossed or locked",
        ),
    ],
)
def test_close_quote_fails_closed_on_book_evidence_drift(
    mutation: dict[str, object],
    message: str,
) -> None:
    credentials = _credentials()
    client = FakeClient()
    payload: dict[str, object] = {
        "market": MARKET_ID,
        "asset_id": TOKEN_ID,
        "timestamp": str(int(time.time() * 1_000)),
        "bids": [{"price": "0.49", "size": "5"}],
        "asks": [{"price": "0.51", "size": "5"}],
        "min_order_size": "5",
        "tick_size": "0.01",
        "neg_risk": False,
    }
    payload.update(mutation)
    client.order_book = payload
    venue = OfficialPolymarketV2Venue(credentials, client=client)

    with pytest.raises(Exception, match=message):
        venue.close_quote(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("5"),
            maximum_book_age_ms=500,
        )


def test_close_quote_rejects_subminimum_dust_and_sdk_parameter_drift() -> None:
    credentials = _credentials()
    client = FakeClient()
    client.order_book = {
        "market": MARKET_ID,
        "asset_id": TOKEN_ID,
        "timestamp": str(int(time.time() * 1_000)),
        "bids": [{"price": "0.49", "size": "10"}],
        "asks": [{"price": "0.51", "size": "10"}],
        "min_order_size": "5",
        "tick_size": "0.01",
        "neg_risk": False,
    }
    venue = OfficialPolymarketV2Venue(credentials, client=client)

    with pytest.raises(Exception, match="below the venue minimum"):
        venue.close_quote(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("4.9"),
            maximum_book_age_ms=500,
        )

    client.tick_size = "0.001"
    with pytest.raises(Exception, match="parameters differ"):
        venue.close_quote(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("5"),
            maximum_book_age_ms=500,
        )


def test_close_quote_rejects_invalid_inputs_and_malformed_book_fields() -> None:
    credentials = _credentials()
    client = FakeClient()
    base: dict[str, object] = {
        "market": MARKET_ID,
        "asset_id": TOKEN_ID,
        "timestamp": str(int(time.time() * 1_000)),
        "bids": [{"price": "0.49", "size": "5"}],
        "asks": [{"price": "0.51", "size": "5"}],
        "min_order_size": "5",
        "tick_size": "0.01",
        "neg_risk": False,
    }
    client.order_book = dict(base)
    venue = OfficialPolymarketV2Venue(credentials, client=client)

    for kwargs, message in (
        ({"market_id": "bad"}, "condition ID"),
        ({"token_id": "bad"}, "token or quantity"),
        ({"quantity": Decimal("0")}, "token or quantity"),
        ({"maximum_book_age_ms": 99}, "book-age"),
    ):
        arguments = {
            "market_id": MARKET_ID,
            "token_id": TOKEN_ID,
            "quantity": Decimal("5"),
            "maximum_book_age_ms": 500,
            **kwargs,
        }
        with pytest.raises(ValueError, match=message):
            venue.close_quote(**arguments)

    for mutation, message in (
        ({"timestamp": ""}, "timestamp"),
        ({"tick_size": "0"}, "parameters"),
        ({"neg_risk": None}, "neg-risk flag"),
        ({"bids": {}}, "bids are invalid"),
        ({"bids": [{"price": "0.495", "size": "5"}]}, "level is invalid"),
        (
            {
                "bids": [
                    {"price": "0.49", "size": "2"},
                    {"price": "0.49", "size": "3"},
                ]
            },
            "duplicate prices",
        ),
    ):
        client.order_book = {**base, **mutation}
        with pytest.raises((ValueError, PolymarketLiveBlocked), match=message):
            venue.close_quote(
                market_id=MARKET_ID,
                token_id=TOKEN_ID,
                quantity=Decimal("5"),
                maximum_book_age_ms=500,
            )

    client.order_book = dict(base)
    client.neg_risk = True
    with pytest.raises(PolymarketLiveBlocked, match="neg-risk parameters"):
        venue.close_quote(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("5"),
            maximum_book_age_ms=500,
        )

    client.neg_risk = False
    venue.maximum_response_bytes = 1
    with pytest.raises(ValueError, match="bounded size"):
        venue.close_quote(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("5"),
            maximum_book_age_ms=500,
        )


def test_missing_environment_and_foreign_cancel_response_fail_closed() -> None:
    with pytest.raises(ValueError, match="missing Polymarket"):
        PolymarketLiveCredentials.from_environment({})

    client = FakeClient()
    requested = "0x" + "2" * 64
    client.cancel_response = {
        "canceled": ["0x" + "3" * 64],
        "not_canceled": {},
    }
    venue = OfficialPolymarketV2Venue(_credentials(), client=client)
    with pytest.raises(PolymarketLiveBlocked, match="foreign order"):
        venue.cancel_orders((requested,))


def test_position_pagination_has_a_hard_bound() -> None:
    row = {
        "conditionId": MARKET_ID,
        "asset": TOKEN_ID,
        "size": "1",
        "redeemable": False,
    }
    session = FakeSession(
        {
            POLYMARKET_DATA_POSITIONS_URL: [row] * 500,
        }
    )
    venue = OfficialPolymarketV2Venue(
        _credentials(),
        client=FakeClient(),
        session=session,  # type: ignore[arg-type]
    )

    with pytest.raises(PolymarketLiveBlocked, match="pagination exceeded"):
        venue.positions()


def test_public_preflight_parser_requires_v2_and_retains_geoblock_result() -> None:
    credentials = _credentials()
    client = FakeClient()
    client.get_closed_only_mode = lambda: {"closed_only": False}
    client.get_open_orders = lambda: []
    session = FakeSession(
        {
            POLYMARKET_GEOBLOCK_URL: {
                "blocked": True,
                "country": "US",
                "region": "NY",
            },
            f"{CLOB_BASE_URL}/version": {"version": 2},
            f"{CLOB_BASE_URL}/time": int(time.time()),
        }
    )
    venue = OfficialPolymarketV2Venue(
        credentials,
        client=client,
        session=session,  # type: ignore[arg-type]
    )
    venue.positions = lambda: ()  # type: ignore[method-assign]

    result = venue.preflight()

    assert result.protocol_version == 2
    assert result.geoblocked is True
    assert result.country == "US"
    assert result.closed_only is False


def test_live_adapter_has_no_account_wide_cancellation_or_heartbeat() -> None:
    source_path = (
        Path(__file__).parents[1]
        / "src"
        / "simple_ai_trading"
        / "polymarket_live_v2.py"
    )
    source = source_path.read_text(encoding="utf-8").lower()

    assert "cancel_all(" not in source
    assert "cancel_market_orders(" not in source
    assert "heartbeat" not in source
    assert "retry_on_error=false" in source.replace(" ", "")
