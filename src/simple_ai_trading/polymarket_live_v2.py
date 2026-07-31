"""Official CLOB V2 adapter for the independent Polymarket live boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
import re
import time
from typing import Mapping, Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .polymarket import CLOB_BASE_URL, POLYMARKET_REQUIRED_CLOB_PROTOCOL_VERSION
from .polymarket_live import (
    PolymarketCancelResult,
    PolymarketCloseQuote,
    PolymarketFundingPreflight,
    PolymarketLiveBlocked,
    PolymarketLiveOrderIntent,
    PolymarketOpenQuote,
    PolymarketPreparedOrder,
    PolymarketRemoteFill,
    PolymarketRemoteOrder,
    PolymarketRemotePosition,
    PolymarketSubmission,
    PolymarketVenuePreflight,
    PolymarketVenueRejected,
)
from .paper_execution import PolymarketFeeModel


POLYMARKET_LIVE_SDK_VERSION = "1.1.0"
POLYGON_CHAIN_ID = 137
POLYMARKET_GEOBLOCK_URL = "https://polymarket.com/api/geoblock"
POLYMARKET_DATA_POSITIONS_URL = "https://data-api.polymarket.com/positions"

_PRIVATE_KEY = re.compile(r"^0x[0-9a-fA-F]{64}$")
_ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")
_ORDER_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")
_SAFE_REJECTION_STATUSES = frozenset({401, 403, 404, 422})
_TOKEN_SCALE = Decimal("1000000")


@dataclass(frozen=True, slots=True)
class PolymarketLiveCredentials:
    """Credential material loaded from process environment only."""

    private_key: str = field(repr=False)
    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)
    api_passphrase: str = field(repr=False)
    funder_address: str
    signature_type: int

    def __post_init__(self) -> None:
        if _PRIVATE_KEY.fullmatch(self.private_key) is None:
            raise ValueError("Polymarket private key format is invalid")
        for name, value in (
            ("API key", self.api_key),
            ("API secret", self.api_secret),
            ("API passphrase", self.api_passphrase),
        ):
            if not isinstance(value, str) or not 8 <= len(value) <= 512:
                raise ValueError(f"Polymarket {name} format is invalid")
        funder = str(self.funder_address or "").strip().lower()
        if _ADDRESS.fullmatch(funder) is None:
            raise ValueError("Polymarket funder address format is invalid")
        object.__setattr__(self, "funder_address", funder)
        signature_type = int(self.signature_type)
        if signature_type not in {0, 1, 2, 3}:
            raise ValueError("Polymarket signature type is invalid")
        object.__setattr__(self, "signature_type", signature_type)

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "PolymarketLiveCredentials":
        source = os.environ if environment is None else environment
        names = {
            "private_key": "SIMPLE_AI_TRADING_POLYMARKET_PRIVATE_KEY",
            "api_key": "SIMPLE_AI_TRADING_POLYMARKET_API_KEY",
            "api_secret": "SIMPLE_AI_TRADING_POLYMARKET_API_SECRET",
            "api_passphrase": "SIMPLE_AI_TRADING_POLYMARKET_API_PASSPHRASE",
            "funder_address": "SIMPLE_AI_TRADING_POLYMARKET_FUNDER_ADDRESS",
            "signature_type": "SIMPLE_AI_TRADING_POLYMARKET_SIGNATURE_TYPE",
        }
        missing = [
            name for name in names.values() if not str(source.get(name, "")).strip()
        ]
        if missing:
            raise ValueError(
                "missing Polymarket live environment variables: " + ",".join(missing)
            )
        try:
            signature_type = int(source[names["signature_type"]])
        except ValueError as exc:
            raise ValueError("Polymarket signature type is invalid") from exc
        return cls(
            private_key=source[names["private_key"]],
            api_key=source[names["api_key"]],
            api_secret=source[names["api_secret"]],
            api_passphrase=source[names["api_passphrase"]],
            funder_address=source[names["funder_address"]],
            signature_type=signature_type,
        )


def _public_session() -> requests.Session:
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.25,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "simple-ai-trading/0.1.0-beta.1 polymarket-live-preflight"}
    )
    session.mount(
        "https://",
        HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4),
    )
    return session


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _server_time_ms(value: object) -> int:
    payload = value
    if isinstance(value, Mapping):
        payload = value.get("time", value.get("timestamp"))
    parsed = int(payload)
    if parsed <= 0:
        raise ValueError("Polymarket server time is invalid")
    return parsed if parsed >= 10_000_000_000 else parsed * 1_000


def _observed_time_ms(value: object) -> int:
    if isinstance(value, datetime):
        return int(value.timestamp() * 1_000)
    text = str(value or "").strip()
    if not text:
        return int(time.time() * 1_000)
    try:
        numeric = int(text)
    except ValueError:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return int(parsed.timestamp() * 1_000)
    return numeric if numeric >= 10_000_000_000 else numeric * 1_000


class OfficialPolymarketV2Venue:
    """Authenticated stable-SDK adapter with no automatic order retry."""

    def __init__(
        self,
        credentials: PolymarketLiveCredentials,
        *,
        client: object | None = None,
        session: requests.Session | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.credentials = credentials
        self.session = session or _public_session()
        self.timeout_seconds = max(1.0, min(30.0, float(timeout_seconds)))
        self.maximum_response_bytes = 8 * 1024 * 1024
        self._client = client or self._build_client()

    def _build_client(self) -> object:
        try:
            from py_clob_client_v2 import ApiCreds, ClobClient
        except ImportError as exc:
            raise RuntimeError(
                "Polymarket live execution requires the 'polymarket-live' extra"
            ) from exc
        try:
            sdk_version = package_version("py-clob-client-v2")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "Polymarket live execution requires the 'polymarket-live' extra"
            ) from exc
        if sdk_version != POLYMARKET_LIVE_SDK_VERSION:
            raise RuntimeError(
                "Polymarket CLOB SDK version differs from the audited pin"
            )
        creds = ApiCreds(
            api_key=self.credentials.api_key,
            api_secret=self.credentials.api_secret,
            api_passphrase=self.credentials.api_passphrase,
        )
        client = ClobClient(
            host=CLOB_BASE_URL,
            chain_id=POLYGON_CHAIN_ID,
            key=self.credentials.private_key,
            creds=creds,
            signature_type=self.credentials.signature_type,
            funder=self.credentials.funder_address,
            use_server_time=True,
            retry_on_error=False,
        )
        funder = str(getattr(client.builder, "funder", "")).lower()
        if funder != self.credentials.funder_address:
            raise PolymarketLiveBlocked(
                "SDK funder differs from configured dedicated wallet"
            )
        return client

    def _get_json(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> object:
        response = self.session.get(
            url,
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        if len(response.content) > self.maximum_response_bytes:
            raise ValueError("Polymarket public response exceeded the bounded size")
        try:
            return response.json()
        except requests.JSONDecodeError as exc:
            raise ValueError("Polymarket public response was not JSON") from exc

    def _protocol_and_time(self) -> tuple[int, int, int]:
        started = int(time.time() * 1_000)
        version_payload = _mapping(
            self._get_json(f"{CLOB_BASE_URL}/version"),
            name="CLOB version response",
        )
        version = version_payload.get("version")
        if type(version) is not int:
            raise ValueError("CLOB protocol version is invalid")
        server_time = _server_time_ms(self._get_json(f"{CLOB_BASE_URL}/time"))
        finished = int(time.time() * 1_000)
        return int(version), server_time, (started + finished) // 2

    def _geoblock(self) -> tuple[bool, str, str]:
        payload = _mapping(
            self._get_json(POLYMARKET_GEOBLOCK_URL),
            name="geoblock response",
        )
        blocked = payload.get("blocked")
        country = str(payload.get("country") or "").strip().upper()
        region = str(payload.get("region") or "").strip().upper()
        if type(blocked) is not bool or not re.fullmatch(r"[A-Z]{2}", country):
            raise ValueError("Polymarket geoblock response is invalid")
        return blocked, country, region

    @staticmethod
    def _closed_only(value: object) -> bool:
        if type(value) is bool:
            return value
        payload = _mapping(value, name="closed-only response")
        for key in ("closed_only", "closedOnly"):
            if type(payload.get(key)) is bool:
                return bool(payload[key])
        raise ValueError("Polymarket closed-only response is invalid")

    @staticmethod
    def _remote_order(value: object) -> PolymarketRemoteOrder:
        payload = _mapping(value, name="open order")
        return PolymarketRemoteOrder(
            order_id=str(payload.get("id") or ""),
            market_id=str(payload.get("market") or ""),
            token_id=str(payload.get("asset_id") or ""),
            side=str(payload.get("side") or ""),
            status=str(payload.get("status") or ""),
            original_quantity=Decimal(str(payload.get("original_size"))),
            matched_quantity=Decimal(str(payload.get("size_matched") or "0")),
        )

    def open_orders(self) -> tuple[PolymarketRemoteOrder, ...]:
        rows = self._client.get_open_orders()
        if not isinstance(rows, list):
            raise ValueError("Polymarket open-order response is invalid")
        orders = tuple(self._remote_order(row) for row in rows)
        if len({order.order_id for order in orders}) != len(orders):
            raise ValueError("Polymarket open-order response contains duplicates")
        return orders

    def orders_by_id(
        self,
        order_ids: Sequence[str],
    ) -> tuple[PolymarketRemoteOrder, ...]:
        requested = tuple(dict.fromkeys(str(value).lower() for value in order_ids))
        if any(_ORDER_ID.fullmatch(order_id) is None for order_id in requested):
            raise ValueError("Polymarket exact-order request contains an invalid ID")
        output: list[PolymarketRemoteOrder] = []
        for order_id in requested:
            try:
                payload = self._client.get_order(order_id)
            except Exception as exc:
                if self._api_status(exc) == 404:
                    continue
                raise
            remote = self._remote_order(payload)
            if remote.order_id != order_id:
                raise PolymarketLiveBlocked(
                    "Polymarket exact-order response ID differs from request"
                )
            output.append(remote)
        return tuple(output)

    def open_quote(
        self,
        *,
        market_id: str,
        token_id: str,
        outcome: str,
        quantity: Decimal,
        maximum_book_age_ms: int,
    ) -> PolymarketOpenQuote:
        condition = str(market_id or "").strip().lower()
        token = str(token_id or "").strip()
        expected_outcome = str(outcome or "").strip().title()
        requested_quantity = Decimal(str(quantity))
        maximum_age = int(maximum_book_age_ms)
        if re.fullmatch(r"^0x[0-9a-f]{64}$", condition) is None:
            raise ValueError("Polymarket open condition ID is invalid")
        if _TOKEN_ID.fullmatch(token) is None or requested_quantity <= 0:
            raise ValueError("Polymarket open token or quantity is invalid")
        if expected_outcome not in {"Up", "Down"}:
            raise ValueError("Polymarket open outcome is invalid")
        if not 100 <= maximum_age <= 5_000:
            raise ValueError("Polymarket open book-age bound is invalid")
        payload = _mapping(
            self._client.get_order_book(token),
            name="open order book",
        )
        market_info = _mapping(
            self._client.get_clob_market_info(condition),
            name="open market info",
        )
        observed_at_ms = int(time.time() * 1_000)
        if str(payload.get("market") or "").strip().lower() != condition:
            raise PolymarketLiveBlocked("Polymarket open book condition differs")
        if str(payload.get("asset_id") or "").strip() != token:
            raise PolymarketLiveBlocked("Polymarket open book token differs")
        if str(market_info.get("c") or "").strip().lower() != condition:
            raise PolymarketLiveBlocked("Polymarket open market identity differs")
        market_tokens = market_info.get("t")
        if not isinstance(market_tokens, list):
            raise ValueError("Polymarket open market token mapping is invalid")
        token_outcomes: dict[str, str] = {}
        for value in market_tokens:
            market_token = _mapping(value, name="open market token")
            market_token_id = str(market_token.get("t") or "").strip()
            market_outcome = str(market_token.get("o") or "").strip().title()
            if (
                _TOKEN_ID.fullmatch(market_token_id) is None
                or market_outcome not in {"Up", "Down"}
                or market_token_id in token_outcomes
                or market_outcome in token_outcomes.values()
            ):
                raise ValueError("Polymarket open market token mapping is invalid")
            token_outcomes[market_token_id] = market_outcome
        if token not in token_outcomes:
            raise PolymarketLiveBlocked("Polymarket open market token differs")
        if token_outcomes[token] != expected_outcome:
            raise PolymarketLiveBlocked("Polymarket open token outcome differs")
        try:
            source_time_ms = int(str(payload.get("timestamp") or ""))
        except ValueError as exc:
            raise ValueError("Polymarket open book timestamp is invalid") from exc
        if source_time_ms < 10_000_000_000:
            source_time_ms *= 1_000
        source_age_ms = observed_at_ms - source_time_ms
        if source_age_ms < -5_000 or source_age_ms > maximum_age:
            raise PolymarketLiveBlocked("Polymarket open book is stale")
        tick = Decimal(str(payload.get("tick_size")))
        minimum = Decimal(str(payload.get("min_order_size")))
        neg_risk = payload.get("neg_risk")
        if tick <= 0 or tick > Decimal("0.1") or minimum <= 0:
            raise ValueError("Polymarket open book parameters are invalid")
        if type(neg_risk) is not bool:
            raise ValueError("Polymarket open book neg-risk flag is invalid")
        sdk_tick = Decimal(str(self._client.get_tick_size(token)))
        sdk_neg_risk = self._client.get_neg_risk(token)
        info_tick = Decimal(str(market_info.get("mts")))
        info_minimum = Decimal(str(market_info.get("mos")))
        if (
            tick != sdk_tick
            or tick != info_tick
            or minimum != info_minimum
            or type(sdk_neg_risk) is not bool
        ):
            raise PolymarketLiveBlocked("Polymarket open execution parameters differ")
        if neg_risk is not sdk_neg_risk:
            raise PolymarketLiveBlocked("Polymarket open neg-risk parameters differ")
        if requested_quantity < minimum:
            raise PolymarketLiveBlocked(
                "proposed Polymarket quantity is below the venue minimum"
            )
        fee_details = _mapping(
            market_info.get("fd"),
            name="open market fee details",
        )
        fee_rate = Decimal(str(fee_details.get("r")))
        fee_exponent_raw = fee_details.get("e")
        taker_only = fee_details.get("to")
        if (
            fee_rate < 0
            or fee_rate > 1
            or type(fee_exponent_raw) is not int
            or fee_exponent_raw <= 0
            or taker_only is not True
        ):
            raise ValueError("Polymarket open fee parameters are invalid")
        fee_model = PolymarketFeeModel(
            enabled=fee_rate > 0,
            rate=fee_rate,
            exponent=fee_exponent_raw,
            taker_only=True,
        )

        def levels(name: str, *, reverse: bool) -> tuple[tuple[Decimal, Decimal], ...]:
            raw_levels = payload.get(name)
            if not isinstance(raw_levels, list):
                raise ValueError(f"Polymarket open book {name} are invalid")
            parsed: list[tuple[Decimal, Decimal]] = []
            for raw in raw_levels:
                level = _mapping(raw, name=f"open book {name} level")
                price = Decimal(str(level.get("price")))
                size = Decimal(str(level.get("size")))
                if price <= 0 or price >= 1 or price % tick or size <= 0:
                    raise ValueError(f"Polymarket open book {name} level is invalid")
                parsed.append((price, size))
            if len({price for price, _ in parsed}) != len(parsed):
                raise ValueError(
                    f"Polymarket open book {name} contain duplicate prices"
                )
            return tuple(sorted(parsed, key=lambda item: item[0], reverse=reverse))

        bids = levels("bids", reverse=True)
        asks = levels("asks", reverse=False)
        if bids and asks and bids[0][0] >= asks[0][0]:
            raise PolymarketLiveBlocked("Polymarket open book is crossed or locked")
        remaining = requested_quantity
        limit_price: Decimal | None = None
        notional = Decimal("0")
        fee_quote = Decimal("0")
        for price, size in asks:
            consumed = min(size, remaining)
            if consumed:
                remaining -= consumed
                limit_price = price
                notional += price * consumed
                fee_quote += fee_model(price, consumed, "taker")
            if remaining <= 0:
                break
        if remaining > 0 or limit_price is None:
            raise PolymarketLiveBlocked(
                "displayed Polymarket asks cannot fill the proposed quantity"
            )
        identity_payload = {
            "book": dict(payload),
            "market_info": dict(market_info),
        }
        payload_json = json.dumps(
            identity_payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        if len(payload_json.encode("ascii")) > self.maximum_response_bytes:
            raise ValueError("Polymarket open evidence exceeded the bounded size")
        average_price = notional / requested_quantity
        return PolymarketOpenQuote(
            market_id=condition,
            token_id=token,
            outcome=expected_outcome,
            quantity=requested_quantity,
            limit_price=limit_price,
            average_price=average_price,
            fee_quote=fee_quote,
            total_quote=notional + fee_quote,
            fee_rate=fee_rate,
            fee_exponent=fee_exponent_raw,
            tick_size=tick,
            minimum_order_size=minimum,
            neg_risk=neg_risk,
            source_time_ms=source_time_ms,
            observed_at_ms=observed_at_ms,
            book_payload_sha256=hashlib.sha256(
                payload_json.encode("ascii")
            ).hexdigest(),
        )

    def close_quote(
        self,
        *,
        market_id: str,
        token_id: str,
        quantity: Decimal,
        maximum_book_age_ms: int,
    ) -> PolymarketCloseQuote:
        condition = str(market_id or "").strip().lower()
        token = str(token_id or "").strip()
        requested_quantity = Decimal(str(quantity))
        maximum_age = int(maximum_book_age_ms)
        if re.fullmatch(r"^0x[0-9a-f]{64}$", condition) is None:
            raise ValueError("Polymarket close condition ID is invalid")
        if _TOKEN_ID.fullmatch(token) is None or requested_quantity <= 0:
            raise ValueError("Polymarket close token or quantity is invalid")
        if not 100 <= maximum_age <= 5_000:
            raise ValueError("Polymarket close book-age bound is invalid")
        payload = _mapping(
            self._client.get_order_book(token),
            name="close order book",
        )
        observed_at_ms = int(time.time() * 1_000)
        if str(payload.get("market") or "").strip().lower() != condition:
            raise PolymarketLiveBlocked("Polymarket close book condition differs")
        if str(payload.get("asset_id") or "").strip() != token:
            raise PolymarketLiveBlocked("Polymarket close book token differs")
        try:
            source_time_ms = int(str(payload.get("timestamp") or ""))
        except ValueError as exc:
            raise ValueError("Polymarket close book timestamp is invalid") from exc
        if source_time_ms < 10_000_000_000:
            source_time_ms *= 1_000
        source_age_ms = observed_at_ms - source_time_ms
        if source_age_ms < -5_000 or source_age_ms > maximum_age:
            raise PolymarketLiveBlocked("Polymarket close book is stale")
        tick = Decimal(str(payload.get("tick_size")))
        minimum = Decimal(str(payload.get("min_order_size")))
        neg_risk = payload.get("neg_risk")
        if tick <= 0 or tick > Decimal("0.1") or minimum <= 0:
            raise ValueError("Polymarket close book parameters are invalid")
        if type(neg_risk) is not bool:
            raise ValueError("Polymarket close book neg-risk flag is invalid")
        sdk_tick = Decimal(str(self._client.get_tick_size(token)))
        sdk_neg_risk = self._client.get_neg_risk(token)
        if tick != sdk_tick or type(sdk_neg_risk) is not bool:
            raise PolymarketLiveBlocked("Polymarket close execution parameters differ")
        if neg_risk is not sdk_neg_risk:
            raise PolymarketLiveBlocked("Polymarket close neg-risk parameters differ")
        if requested_quantity < minimum:
            raise PolymarketLiveBlocked(
                "bot-owned close quantity is below the venue minimum"
            )

        def levels(name: str, *, reverse: bool) -> tuple[tuple[Decimal, Decimal], ...]:
            raw_levels = payload.get(name)
            if not isinstance(raw_levels, list):
                raise ValueError(f"Polymarket close book {name} are invalid")
            parsed: list[tuple[Decimal, Decimal]] = []
            for raw in raw_levels:
                level = _mapping(raw, name=f"close book {name} level")
                price = Decimal(str(level.get("price")))
                size = Decimal(str(level.get("size")))
                if price <= 0 or price >= 1 or price % tick or size <= 0:
                    raise ValueError(f"Polymarket close book {name} level is invalid")
                parsed.append((price, size))
            if len({price for price, _ in parsed}) != len(parsed):
                raise ValueError(
                    f"Polymarket close book {name} contain duplicate prices"
                )
            return tuple(sorted(parsed, key=lambda item: item[0], reverse=reverse))

        bids = levels("bids", reverse=True)
        asks = levels("asks", reverse=False)
        if bids and asks and bids[0][0] >= asks[0][0]:
            raise PolymarketLiveBlocked("Polymarket close book is crossed or locked")
        remaining = requested_quantity
        limit_price: Decimal | None = None
        for price, size in bids:
            consumed = min(size, remaining)
            if consumed:
                remaining -= consumed
                limit_price = price
            if remaining <= 0:
                break
        if remaining > 0 or limit_price is None:
            raise PolymarketLiveBlocked(
                "displayed Polymarket bids cannot close the bot-owned lot"
            )
        payload_json = json.dumps(
            dict(payload),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        if len(payload_json.encode("ascii")) > self.maximum_response_bytes:
            raise ValueError("Polymarket close book exceeded the bounded size")
        return PolymarketCloseQuote(
            market_id=condition,
            token_id=token,
            quantity=requested_quantity,
            limit_price=limit_price,
            tick_size=tick,
            minimum_order_size=minimum,
            neg_risk=neg_risk,
            source_time_ms=source_time_ms,
            observed_at_ms=observed_at_ms,
            book_payload_sha256=hashlib.sha256(
                payload_json.encode("ascii")
            ).hexdigest(),
        )

    def positions(self) -> tuple[PolymarketRemotePosition, ...]:
        output: list[PolymarketRemotePosition] = []
        offset = 0
        while True:
            payload = self._get_json(
                POLYMARKET_DATA_POSITIONS_URL,
                params={
                    "user": self.credentials.funder_address,
                    "sizeThreshold": 0,
                    "limit": 500,
                    "offset": offset,
                },
            )
            if not isinstance(payload, list):
                raise ValueError("Polymarket position response is invalid")
            for row in payload:
                item = _mapping(row, name="position")
                quantity = Decimal(str(item.get("size")))
                if quantity <= 0:
                    continue
                output.append(
                    PolymarketRemotePosition(
                        market_id=str(item.get("conditionId") or ""),
                        token_id=str(item.get("asset") or ""),
                        quantity=quantity,
                        redeemable=bool(item.get("redeemable", False)),
                    )
                )
            if len(payload) < 500:
                break
            offset += 500
            if offset > 10_000:
                raise PolymarketLiveBlocked(
                    "Polymarket position pagination exceeded its bound"
                )
        if len({item.token_id for item in output}) != len(output):
            raise ValueError("Polymarket position response contains duplicate tokens")
        return tuple(sorted(output, key=lambda item: (item.market_id, item.token_id)))

    @staticmethod
    def _asset_amount(value: object, *, name: str) -> Decimal:
        text = str(value or "").strip()
        if re.fullmatch(r"[0-9]+", text) is None:
            raise ValueError(f"Polymarket {name} is not an integer token amount")
        return Decimal(text) / _TOKEN_SCALE

    def funding(
        self,
        intent: PolymarketLiveOrderIntent,
        *,
        neg_risk: bool,
    ) -> PolymarketFundingPreflight:
        try:
            from py_clob_client_v2 import AssetType, BalanceAllowanceParams
            from py_clob_client_v2.config import get_contract_config
        except ImportError as exc:
            raise RuntimeError(
                "Polymarket live execution requires the 'polymarket-live' extra"
            ) from exc
        asset_type = (
            AssetType.COLLATERAL if intent.side == "BUY" else AssetType.CONDITIONAL
        )
        token_id = "" if intent.side == "BUY" else intent.token_id
        response = _mapping(
            self._client.get_balance_allowance(
                BalanceAllowanceParams(
                    asset_type=asset_type,
                    token_id=token_id or None,
                    signature_type=self.credentials.signature_type,
                )
            ),
            name="balance-allowance response",
        )
        config = get_contract_config(POLYGON_CHAIN_ID)
        exchange = str(
            config.neg_risk_exchange_v2 if neg_risk else config.exchange_v2
        ).lower()
        allowance_value = response.get("allowance")
        if allowance_value is None:
            allowances = _mapping(
                response.get("allowances"),
                name="balance-allowance allowances",
            )
            normalized_allowances = {
                str(key).lower(): value for key, value in allowances.items()
            }
            if exchange not in normalized_allowances:
                raise ValueError(
                    "balance-allowance response omitted the exact V2 exchange"
                )
            allowance_value = normalized_allowances[exchange]
        return PolymarketFundingPreflight(
            asset_type=asset_type,
            token_id=token_id,
            available_balance=self._asset_amount(
                response.get("balance"),
                name="balance",
            ),
            available_allowance=self._asset_amount(
                allowance_value,
                name="allowance",
            ),
        )

    def preflight(self) -> PolymarketVenuePreflight:
        blocked, country, region = self._geoblock()
        version, server_time_ms, observed_at_ms = self._protocol_and_time()
        if version != POLYMARKET_REQUIRED_CLOB_PROTOCOL_VERSION:
            raise PolymarketLiveBlocked("unsupported Polymarket CLOB protocol version")
        closed_only = self._closed_only(self._client.get_closed_only_mode())
        return PolymarketVenuePreflight(
            protocol_version=version,
            server_time_ms=server_time_ms,
            observed_at_ms=observed_at_ms,
            geoblocked=blocked,
            country=country,
            region=region,
            closed_only=closed_only,
            wallet_address=self.credentials.funder_address,
            open_orders=self.open_orders(),
            positions=self.positions(),
        )

    def prepare_order(
        self,
        intent: PolymarketLiveOrderIntent,
        *,
        tick_size: Decimal,
        neg_risk: bool,
    ) -> PolymarketPreparedOrder:
        try:
            from py_clob_client_v2 import (
                OrderArgsV2,
                PartialCreateOrderOptions,
                Side,
            )
            from py_clob_client_v2.config import get_contract_config
            from py_clob_client_v2.order_utils import ExchangeOrderBuilderV2
        except ImportError as exc:
            raise RuntimeError(
                "Polymarket live execution requires the 'polymarket-live' extra"
            ) from exc
        expiration = 0
        if intent.order_type == "GTD":
            expiration = intent.expires_at_ms // 1_000
            if expiration - int(time.time()) < 180:
                raise PolymarketLiveBlocked(
                    "GTD expiry is too close for the official venue minimum"
                )
        side = Side.BUY if intent.side == "BUY" else Side.SELL
        signed = self._client.create_order(
            OrderArgsV2(
                token_id=intent.token_id,
                price=float(intent.limit_price),
                size=float(intent.quantity),
                side=side,
                expiration=expiration,
                metadata=intent.metadata,
            ),
            PartialCreateOrderOptions(
                tick_size=format(tick_size, "f"),
                neg_risk=bool(neg_risk),
            ),
        )
        sdk_signer = str(self._client.builder.signer.address()).lower()
        expected_signer = (
            self.credentials.funder_address
            if self.credentials.signature_type == 3
            else sdk_signer
        )
        if (
            str(getattr(signed, "maker", "")).lower() != self.credentials.funder_address
            or str(getattr(signed, "signer", "")).lower() != expected_signer
            or int(getattr(signed, "signatureType", -1))
            != self.credentials.signature_type
            or str(getattr(signed, "tokenId", "")) != intent.token_id
            or getattr(signed, "side", None) != side
            or int(getattr(signed, "expiration", -1)) != expiration
        ):
            raise ValueError("official SDK changed signed order identity")
        maker_amount = Decimal(str(getattr(signed, "makerAmount", ""))) / _TOKEN_SCALE
        taker_amount = Decimal(str(getattr(signed, "takerAmount", ""))) / _TOKEN_SCALE
        if maker_amount <= 0 or taker_amount <= 0:
            raise ValueError("official SDK produced empty order economics")
        if intent.side == "BUY":
            effective_quantity = taker_amount
            effective_price = maker_amount / taker_amount
            worse_price = effective_price > intent.limit_price
        else:
            effective_quantity = maker_amount
            effective_price = taker_amount / maker_amount
            worse_price = effective_price < intent.limit_price
        if effective_quantity != intent.quantity:
            raise PolymarketLiveBlocked(
                "official SDK rounded the requested order quantity"
            )
        if worse_price or abs(effective_price - intent.limit_price) >= tick_size / 2:
            raise PolymarketLiveBlocked(
                "official SDK changed the requested limit economics"
            )
        config = get_contract_config(POLYGON_CHAIN_ID)
        exchange = config.neg_risk_exchange_v2 if neg_risk else config.exchange_v2
        hash_builder = ExchangeOrderBuilderV2(
            exchange,
            POLYGON_CHAIN_ID,
            self._client.builder.signer,
        )
        typed_data = hash_builder.build_order_typed_data(signed)
        expected_order_id = hash_builder.build_order_hash(typed_data).lower()
        if _ORDER_ID.fullmatch(expected_order_id) is None:
            raise ValueError("official SDK produced an invalid order hash")
        if str(getattr(signed, "metadata", "")).lower() != intent.metadata:
            raise ValueError("official SDK did not preserve order metadata")
        return PolymarketPreparedOrder(
            intent=intent,
            expected_order_id=expected_order_id,
            metadata=intent.metadata,
            opaque_signed_order=signed,
        )

    @staticmethod
    def _api_status(error: Exception) -> int | None:
        status = getattr(error, "status_code", None)
        return int(status) if isinstance(status, int) else None

    def submit_order(self, prepared: PolymarketPreparedOrder) -> PolymarketSubmission:
        try:
            response = self._client.post_order(
                prepared.opaque_signed_order,
                order_type=prepared.intent.order_type,
                defer_exec=True,
            )
        except Exception as exc:
            status = self._api_status(exc)
            if status in _SAFE_REJECTION_STATUSES:
                raise PolymarketVenueRejected(
                    f"Polymarket rejected the order with HTTP {status}"
                ) from exc
            raise
        payload = _mapping(response, name="order submission response")
        success = payload.get("success")
        if success is False:
            return PolymarketSubmission(
                accepted=False,
                order_id="",
                status=str(payload.get("status") or "rejected"),
                rejection_code="venue_rejected",
            )
        if success is not True:
            raise ValueError("Polymarket order response omitted acceptance state")
        return PolymarketSubmission(
            accepted=True,
            order_id=str(payload.get("orderID") or ""),
            status=str(payload.get("status") or ""),
        )

    @staticmethod
    def _fill(
        *,
        trade: Mapping[str, object],
        order_id: str,
        token_id: str,
        side: str,
        quantity: object,
        price: object,
    ) -> PolymarketRemoteFill:
        status = str(trade.get("status") or "").strip().upper()
        if status.startswith("TRADE_STATUS_"):
            status = status.removeprefix("TRADE_STATUS_")
        return PolymarketRemoteFill(
            trade_id=str(trade.get("id") or ""),
            order_id=order_id,
            market_id=str(trade.get("market") or ""),
            token_id=token_id,
            side=side,
            quantity=Decimal(str(quantity)),
            price=Decimal(str(price)),
            status=status,
            observed_at_ms=_observed_time_ms(
                trade.get("last_update") or trade.get("matchtime")
            ),
        )

    def fills_for_orders(
        self,
        order_ids: Sequence[str],
        *,
        market_ids: Sequence[str],
    ) -> tuple[PolymarketRemoteFill, ...]:
        owned = {str(value).lower() for value in order_ids}
        if not owned:
            return ()
        try:
            from py_clob_client_v2 import TradeParams
        except ImportError as exc:
            raise RuntimeError(
                "Polymarket live execution requires the 'polymarket-live' extra"
            ) from exc
        output: dict[tuple[str, str], PolymarketRemoteFill] = {}
        for market_id in sorted(set(market_ids)):
            rows = self._client.get_trades(TradeParams(market=market_id))
            if not isinstance(rows, list):
                raise ValueError("Polymarket trade response is invalid")
            for raw in rows:
                trade = _mapping(raw, name="trade")
                taker_order_id = str(trade.get("taker_order_id") or "").lower()
                if taker_order_id in owned:
                    fill = self._fill(
                        trade=trade,
                        order_id=taker_order_id,
                        token_id=str(trade.get("asset_id") or ""),
                        side=str(trade.get("side") or ""),
                        quantity=trade.get("size"),
                        price=trade.get("price"),
                    )
                    output[(fill.trade_id, fill.order_id)] = fill
                maker_orders = trade.get("maker_orders") or []
                if not isinstance(maker_orders, list):
                    raise ValueError("Polymarket maker-order response is invalid")
                for raw_maker in maker_orders:
                    maker = _mapping(raw_maker, name="maker order")
                    maker_order_id = str(maker.get("order_id") or "").lower()
                    if maker_order_id not in owned:
                        continue
                    maker_side = str(maker.get("side") or "").strip().upper()
                    if not maker_side:
                        taker_side = str(trade.get("side") or "").strip().upper()
                        maker_side = "SELL" if taker_side == "BUY" else "BUY"
                    fill = self._fill(
                        trade=trade,
                        order_id=maker_order_id,
                        token_id=str(maker.get("asset_id") or ""),
                        side=maker_side,
                        quantity=maker.get("matched_amount"),
                        price=maker.get("price"),
                    )
                    output[(fill.trade_id, fill.order_id)] = fill
        return tuple(
            output[key] for key in sorted(output, key=lambda item: (item[0], item[1]))
        )

    def cancel_orders(self, order_ids: Sequence[str]) -> PolymarketCancelResult:
        requested = tuple(dict.fromkeys(str(value).lower() for value in order_ids))
        if not requested:
            return PolymarketCancelResult((), ())
        try:
            response = self._client.cancel_orders(list(requested))
        except Exception:
            raise
        payload = _mapping(response, name="cancel response")
        cancelled_raw = payload.get("canceled") or []
        failed_raw = payload.get("not_canceled") or {}
        if not isinstance(cancelled_raw, list) or not isinstance(failed_raw, Mapping):
            raise ValueError("Polymarket cancel response is invalid")
        cancelled = tuple(str(value).lower() for value in cancelled_raw)
        failed = tuple(str(value).lower() for value in failed_raw)
        if (set(cancelled) | set(failed)) - set(requested):
            raise PolymarketLiveBlocked(
                "Polymarket cancel response included a foreign order"
            )
        return PolymarketCancelResult(cancelled, failed)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
        self.session.close()


__all__ = [
    "OfficialPolymarketV2Venue",
    "POLYGON_CHAIN_ID",
    "POLYMARKET_DATA_POSITIONS_URL",
    "POLYMARKET_GEOBLOCK_URL",
    "POLYMARKET_LIVE_SDK_VERSION",
    "PolymarketLiveCredentials",
]
