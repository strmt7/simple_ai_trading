from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import time
from typing import Mapping

import pytest

from simple_ai_trading.polymarket import CLOB_BASE_URL
from simple_ai_trading.polymarket_live import (
    PolymarketLiveOrderIntent,
    PolymarketPreparedOrder,
)
from simple_ai_trading.polymarket_live_v2 import (
    OfficialPolymarketV2Venue,
    POLYGON_CHAIN_ID,
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

    def get_balance_allowance(self, params: object) -> object:
        del params
        return self.balance_response

    def get_trades(self, params: object) -> list[object]:
        del params
        return self.trades

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


def test_trade_parser_binds_maker_fill_to_exact_owned_hash_and_normalizes_status() -> None:
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
