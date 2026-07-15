"""Strict public Polymarket market discovery for BTC/ETH/SOL five-minute paper trading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Mapping, Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .paper_execution import BookLevel, PaperBookSnapshot, PolymarketFeeModel


POLYMARKET_MARKET_SCHEMA_VERSION = "polymarket-crypto-5m-market-v1"
SUPPORTED_POLYMARKET_ASSETS = ("BTC", "ETH", "SOL")
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
CLOB_BASE_URL = "https://clob.polymarket.com"
_SLUG = re.compile(r"^(btc|eth|sol)-updown-5m-([0-9]{10})$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")
_RESOLUTION_PATH = {
    "BTC": "/streams/btc-usd",
    "ETH": "/streams/eth-usd",
    "SOL": "/streams/sol-usd",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _decimal(value: object, *, name: str, minimum: Decimal | None = None) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite() or (minimum is not None and parsed < minimum):
        raise ValueError(f"{name} is outside its supported range")
    return parsed


def _json_list(value: object, *, name: str) -> list[object]:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{name} must be a list")
    return parsed


def _utc_ms(value: object, *, name: str) -> int:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} is missing")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a UTC offset")
    return int(parsed.astimezone(timezone.utc).timestamp() * 1_000)


@dataclass(frozen=True)
class PolymarketFeeSchedule:
    enabled: bool
    rate: Decimal
    exponent: int
    taker_only: bool
    rebate_rate: Decimal

    def fee_model(self) -> PolymarketFeeModel:
        return PolymarketFeeModel(
            enabled=self.enabled,
            rate=self.rate,
            exponent=self.exponent,
            taker_only=self.taker_only,
        )


@dataclass(frozen=True)
class PolymarketFiveMinuteMarket:
    asset: str
    market_id: str
    condition_id: str
    slug: str
    question: str
    event_start_ms: int
    end_ms: int
    up_token_id: str
    down_token_id: str
    tick_size: Decimal
    minimum_order_size: Decimal
    fee_schedule: PolymarketFeeSchedule
    liquidity_quote: Decimal
    volume_quote: Decimal
    resolution_source: str
    gamma_payload_sha256: str
    gamma_payload_json: str

    @property
    def token_ids(self) -> tuple[str, str]:
        return self.up_token_id, self.down_token_id

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_MARKET_SCHEMA_VERSION,
            "asset": self.asset,
            "market_id": self.market_id,
            "condition_id": self.condition_id,
            "slug": self.slug,
            "question": self.question,
            "event_start_ms": self.event_start_ms,
            "end_ms": self.end_ms,
            "up_token_id": self.up_token_id,
            "down_token_id": self.down_token_id,
            "tick_size": format(self.tick_size, "f"),
            "minimum_order_size": format(self.minimum_order_size, "f"),
            "fees_enabled": self.fee_schedule.enabled,
            "fee_rate": format(self.fee_schedule.rate, "f"),
            "fee_exponent": self.fee_schedule.exponent,
            "fee_taker_only": self.fee_schedule.taker_only,
            "fee_rebate_rate": format(self.fee_schedule.rebate_rate, "f"),
            "liquidity_quote": format(self.liquidity_quote, "f"),
            "volume_quote": format(self.volume_quote, "f"),
            "resolution_source": self.resolution_source,
            "gamma_payload_sha256": self.gamma_payload_sha256,
        }


def parse_polymarket_five_minute_market(
    payload: Mapping[str, object],
) -> PolymarketFiveMinuteMarket:
    """Parse one Gamma object without accepting inferred market metadata."""

    raw = dict(payload)
    slug = str(raw.get("slug") or "").strip().lower()
    match = _SLUG.fullmatch(slug)
    if match is None:
        raise ValueError(
            "market slug is not a supported BTC/ETH/SOL five-minute market"
        )
    asset = match.group(1).upper()
    slug_epoch_ms = int(match.group(2)) * 1_000
    if raw.get("active") is not True or raw.get("closed") is not False:
        raise ValueError("market is not active and open")
    if raw.get("enableOrderBook") is not True or raw.get("acceptingOrders") is not True:
        raise ValueError("market order book is not accepting orders")
    event_start_ms = _utc_ms(raw.get("eventStartTime"), name="eventStartTime")
    end_ms = _utc_ms(raw.get("endDate"), name="endDate")
    if event_start_ms != slug_epoch_ms or end_ms - event_start_ms != 300_000:
        raise ValueError("slug epoch and exact five-minute event window disagree")
    condition_id = str(raw.get("conditionId") or "").strip().lower()
    if not _CONDITION_ID.fullmatch(condition_id):
        raise ValueError("conditionId is invalid")
    market_id = str(raw.get("id") or "").strip()
    if not market_id or len(market_id) > 80:
        raise ValueError("market id is invalid")
    tokens = [
        str(value).strip()
        for value in _json_list(raw.get("clobTokenIds"), name="clobTokenIds")
    ]
    outcomes = [
        str(value).strip() for value in _json_list(raw.get("outcomes"), name="outcomes")
    ]
    if outcomes != ["Up", "Down"] or len(tokens) != 2 or len(set(tokens)) != 2:
        raise ValueError("market must have an exact Up/Down token mapping")
    if not all(_TOKEN_ID.fullmatch(token) for token in tokens):
        raise ValueError("market token id is invalid")
    tick_size = _decimal(
        raw.get("orderPriceMinTickSize"),
        name="orderPriceMinTickSize",
        minimum=Decimal("0.0001"),
    )
    if tick_size > Decimal("0.1"):
        raise ValueError("market tick size is too coarse")
    minimum_order_size = _decimal(
        raw.get("orderMinSize"),
        name="orderMinSize",
        minimum=Decimal("0.000001"),
    )
    fees_enabled = raw.get("feesEnabled") is True
    fee_payload = raw.get("feeSchedule")
    if not isinstance(fee_payload, Mapping):
        raise ValueError("feeSchedule is missing")
    fee_rate = _decimal(
        fee_payload.get("rate"), name="feeSchedule.rate", minimum=Decimal("0")
    )
    rebate_rate = _decimal(
        fee_payload.get("rebateRate"),
        name="feeSchedule.rebateRate",
        minimum=Decimal("0"),
    )
    try:
        exponent = int(fee_payload.get("exponent"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("feeSchedule.exponent is invalid") from exc
    taker_only = fee_payload.get("takerOnly") is True
    if exponent <= 0 or fee_rate > 1 or rebate_rate > 1:
        raise ValueError("fee schedule is outside supported bounds")
    if fees_enabled and (fee_rate <= 0 or not taker_only):
        raise ValueError("enabled crypto fees require a positive taker-only schedule")
    if not fees_enabled and fee_rate != 0:
        raise ValueError("disabled fees cannot carry a positive fee rate")
    resolution_source = str(raw.get("resolutionSource") or "").strip().lower()
    expected_source = f"https://data.chain.link{_RESOLUTION_PATH[asset]}"
    if resolution_source.rstrip("/") != expected_source:
        raise ValueError(
            "market resolution source is not the expected Chainlink stream"
        )
    liquidity = _decimal(
        raw.get("liquidityNum", 0), name="liquidityNum", minimum=Decimal("0")
    )
    volume = _decimal(raw.get("volumeNum", 0), name="volumeNum", minimum=Decimal("0"))
    question = str(raw.get("question") or "").strip()
    if not question or len(question) > 500:
        raise ValueError("market question is invalid")
    canonical = _canonical_json(raw)
    return PolymarketFiveMinuteMarket(
        asset=asset,
        market_id=market_id,
        condition_id=condition_id,
        slug=slug,
        question=question,
        event_start_ms=event_start_ms,
        end_ms=end_ms,
        up_token_id=tokens[0],
        down_token_id=tokens[1],
        tick_size=tick_size,
        minimum_order_size=minimum_order_size,
        fee_schedule=PolymarketFeeSchedule(
            enabled=fees_enabled,
            rate=fee_rate,
            exponent=exponent,
            taker_only=taker_only,
            rebate_rate=rebate_rate,
        ),
        liquidity_quote=liquidity,
        volume_quote=volume,
        resolution_source=resolution_source,
        gamma_payload_sha256=hashlib.sha256(canonical.encode("ascii")).hexdigest(),
        gamma_payload_json=canonical,
    )


def validate_clob_market_info(
    market: PolymarketFiveMinuteMarket,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Require CLOB V2 identity and executable parameters to match Gamma."""

    raw = dict(payload)
    condition = str(raw.get("c") or "").strip().lower()
    token_rows = raw.get("t")
    if condition != market.condition_id or not isinstance(token_rows, list):
        raise ValueError("CLOB market identity does not match Gamma")
    parsed_tokens: list[tuple[str, str]] = []
    for row in token_rows:
        if not isinstance(row, Mapping):
            raise ValueError("CLOB token mapping is malformed")
        parsed_tokens.append((str(row.get("t") or ""), str(row.get("o") or "")))
    if parsed_tokens != [
        (market.up_token_id, "Up"),
        (market.down_token_id, "Down"),
    ]:
        raise ValueError("CLOB token mapping drifted from Gamma")
    minimum_order = _decimal(
        raw.get("mos"), name="CLOB minimum order", minimum=Decimal("0")
    )
    tick_size = _decimal(raw.get("mts"), name="CLOB tick size", minimum=Decimal("0"))
    if minimum_order != market.minimum_order_size or tick_size != market.tick_size:
        raise ValueError("CLOB order parameters drifted from Gamma")
    fee_details = raw.get("fd")
    if not isinstance(fee_details, Mapping):
        raise ValueError("CLOB fee details are missing")
    fee_rate = _decimal(
        fee_details.get("r"), name="CLOB fee rate", minimum=Decimal("0")
    )
    try:
        exponent = int(fee_details.get("e"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("CLOB fee exponent is invalid") from exc
    if (
        fee_rate != market.fee_schedule.rate
        or exponent != market.fee_schedule.exponent
        or (fee_details.get("to") is True) != market.fee_schedule.taker_only
    ):
        raise ValueError("CLOB fee schedule drifted from Gamma")
    return {
        "payload_json": _canonical_json(raw),
        "payload_sha256": _canonical_sha256(raw),
        "maker_base_fee": int(raw.get("mbf") or 0),
        "taker_base_fee": int(raw.get("tbf") or 0),
        "taker_order_delay_enabled": raw.get("itode") is True,
        "minimum_order_age_seconds": int(raw.get("oas") or 0),
    }


def validate_clob_order_book(
    market: PolymarketFiveMinuteMarket,
    token_id: str,
    payload: Mapping[str, object],
    *,
    received_wall_ms: int,
    received_monotonic_ns: int,
) -> PaperBookSnapshot:
    """Build a conservative shared book snapshot while retaining raw identity."""

    token = str(token_id or "").strip()
    if token not in market.token_ids:
        raise ValueError("order-book token is not part of the market")
    raw = dict(payload)
    if str(raw.get("market") or "").strip().lower() != market.condition_id:
        raise ValueError("order-book condition does not match the market")
    if str(raw.get("asset_id") or "").strip() != token:
        raise ValueError("order-book asset does not match the requested token")
    try:
        source_time_ms = int(raw.get("timestamp"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("order-book timestamp is invalid") from exc
    if source_time_ms < 0:
        raise ValueError("order-book timestamp is invalid")
    payload_json = _canonical_json(raw)
    payload_sha = hashlib.sha256(payload_json.encode("ascii")).hexdigest()

    def levels(name: str, *, reverse: bool) -> tuple[BookLevel, ...]:
        rows = raw.get(name)
        if not isinstance(rows, list):
            raise ValueError(f"order-book {name} must be a list")
        parsed: list[BookLevel] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"order-book {name} level is malformed")
            price = _decimal(row.get("price"), name=f"{name} price", minimum=Decimal("0.0001"))
            size = _decimal(row.get("size"), name=f"{name} size", minimum=Decimal("0.000001"))
            if price >= 1:
                raise ValueError(f"order-book {name} price must be below one")
            parsed.append(BookLevel(price=price, quantity=size))
        if len({item.price for item in parsed}) != len(parsed):
            raise ValueError(f"order-book {name} has duplicate prices")
        return tuple(sorted(parsed, key=lambda item: item.price, reverse=reverse))

    return PaperBookSnapshot(
        venue="polymarket",
        market_id=market.condition_id,
        asset_id=token,
        bids=levels("bids", reverse=True),
        asks=levels("asks", reverse=False),
        source_time_ms=source_time_ms,
        received_wall_ms=int(received_wall_ms),
        received_monotonic_ns=int(received_monotonic_ns),
        source_payload_sha256=payload_sha,
        connected=True,
        gap_free=True,
    ).validated()


def _public_http_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "simple-ai-trading/0.1.0-beta.1 public-paper-recorder"}
    )
    session.mount(
        "https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    )
    return session


class PolymarketPublicClient:
    """Bounded public-only HTTP client; no wallet or API credentials are accepted."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        timeout_seconds: float = 10.0,
        maximum_response_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self.session = session or _public_http_session()
        self.timeout_seconds = max(1.0, min(60.0, float(timeout_seconds)))
        self.maximum_response_bytes = max(
            1024, min(64 * 1024 * 1024, int(maximum_response_bytes))
        )

    def _get_json(
        self,
        url: str,
        *,
        params: Sequence[tuple[str, str]] | Mapping[str, str] | None = None,
    ) -> object:
        response = self.session.get(url, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        if len(response.content) > self.maximum_response_bytes:
            raise ValueError("public Polymarket response exceeded the bounded size")
        try:
            return response.json()
        except requests.JSONDecodeError as exc:
            raise ValueError("public Polymarket response was not JSON") from exc

    def discover_five_minute_markets(
        self,
        *,
        now_ms: int,
        include_next: bool = True,
        require_all_assets: bool = True,
    ) -> tuple[PolymarketFiveMinuteMarket, ...]:
        now = int(now_ms)
        if now < 0:
            raise ValueError("now_ms must be non-negative")
        base_epoch = now // 300_000 * 300
        epochs = (base_epoch, base_epoch + 300) if include_next else (base_epoch,)
        slugs = [
            f"{asset.lower()}-updown-5m-{epoch}"
            for asset in SUPPORTED_POLYMARKET_ASSETS
            for epoch in epochs
        ]
        params = [("slug", slug) for slug in slugs]
        params.append(("closed", "false"))
        response = self._get_json(GAMMA_MARKETS_URL, params=params)
        if not isinstance(response, list):
            raise ValueError("Gamma market response must be a list")
        markets = tuple(
            sorted(
                (
                    parse_polymarket_five_minute_market(row)
                    for row in response
                    if isinstance(row, Mapping)
                ),
                key=lambda item: (item.event_start_ms, item.asset),
            )
        )
        duplicates = len({market.condition_id for market in markets}) != len(markets)
        unexpected = [market.slug for market in markets if market.slug not in slugs]
        if duplicates or unexpected:
            raise ValueError("Gamma returned duplicate or unrequested markets")
        if require_all_assets:
            assets = {market.asset for market in markets if market.end_ms > now}
            missing = sorted(set(SUPPORTED_POLYMARKET_ASSETS) - assets)
            if missing:
                raise ValueError(
                    f"active five-minute markets missing for: {','.join(missing)}"
                )
        return markets

    def clob_market_info(self, condition_id: str) -> Mapping[str, object]:
        condition = str(condition_id or "").strip().lower()
        if not _CONDITION_ID.fullmatch(condition):
            raise ValueError("condition_id is invalid")
        response = self._get_json(f"{CLOB_BASE_URL}/clob-markets/{condition}")
        if not isinstance(response, Mapping):
            raise ValueError("CLOB market-info response must be an object")
        return response

    def clob_market(self, condition_id: str) -> Mapping[str, object]:
        """Fetch the official full CLOB market state by condition ID."""

        condition = str(condition_id or "").strip().lower()
        if not _CONDITION_ID.fullmatch(condition):
            raise ValueError("condition_id is invalid")
        response = self._get_json(f"{CLOB_BASE_URL}/markets/{condition}")
        if not isinstance(response, Mapping):
            raise ValueError("CLOB market response must be an object")
        return response

    def gamma_market(self, market_id: str) -> Mapping[str, object]:
        """Fetch one official Gamma market using its numeric market ID."""

        normalized = str(market_id or "").strip()
        if not normalized.isdigit() or len(normalized) > 20:
            raise ValueError("market_id is invalid")
        response = self._get_json(f"{GAMMA_MARKETS_URL}/{normalized}")
        if not isinstance(response, Mapping):
            raise ValueError("Gamma market response must be an object")
        return response

    def fee_rate(self, token_id: str) -> Mapping[str, object]:
        token = str(token_id or "").strip()
        if not _TOKEN_ID.fullmatch(token):
            raise ValueError("token_id is invalid")
        response = self._get_json(f"{CLOB_BASE_URL}/fee-rate/{token}")
        if not isinstance(response, Mapping) or not isinstance(
            response.get("base_fee"), int
        ):
            raise ValueError("CLOB fee-rate response is invalid")
        return response

    def order_book(self, token_id: str) -> Mapping[str, object]:
        token = str(token_id or "").strip()
        if not _TOKEN_ID.fullmatch(token):
            raise ValueError("token_id is invalid")
        response = self._get_json(f"{CLOB_BASE_URL}/book", params={"token_id": token})
        if not isinstance(response, Mapping):
            raise ValueError("CLOB order-book response must be an object")
        return response


__all__ = [
    "CLOB_BASE_URL",
    "GAMMA_MARKETS_URL",
    "POLYMARKET_MARKET_SCHEMA_VERSION",
    "SUPPORTED_POLYMARKET_ASSETS",
    "PolymarketFeeSchedule",
    "PolymarketFiveMinuteMarket",
    "PolymarketPublicClient",
    "parse_polymarket_five_minute_market",
    "validate_clob_order_book",
    "validate_clob_market_info",
]
