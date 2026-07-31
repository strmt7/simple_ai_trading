"""Independent receipt-time Binance features for Polymarket Round 21."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
from typing import Literal, Mapping

from .polymarket_capture_frame import CaptureFrameRecord
from .polymarket_round21_dataset import POLYMARKET_ROUND21_MAXIMUM_LOOKBACK_MS


POLYMARKET_ROUND21_BINANCE_FEATURE_SCHEMA_VERSION = (
    "polymarket-round21-independent-binance-features-v1"
)
POLYMARKET_ROUND21_BINANCE_WINDOWS_MS = (
    250,
    1_000,
    5_000,
    15_000,
    30_000,
    60_000,
    120_000,
)
POLYMARKET_ROUND21_BINANCE_MAXIMUM_BBO_AGE_MS = 1_000
_MARKETS = ("spot", "usdm")
_STREAM_MARKET = {"binance_spot": "spot", "binance_futures": "usdm"}
_TRADE_METRICS = (
    "log_return",
    "realized_variance",
    "signed_quote_imbalance",
    "log1p_quote_notional",
    "trade_count",
)
_BOOK_METRICS = (
    "mid_log_return",
    "microprice_log_return",
    "mean_relative_spread",
    "mean_top_quantity_imbalance",
    "mean_log1p_top_quote_depth",
    "book_update_count",
)
_MARKET_FEATURE_NAMES = tuple(
    (
        *(
            f"{{market}}.trade_{metric}_{window}ms"
            for window in POLYMARKET_ROUND21_BINANCE_WINDOWS_MS
            for metric in _TRADE_METRICS
        ),
        *(
            f"{{market}}.book_{metric}_{window}ms"
            for window in POLYMARKET_ROUND21_BINANCE_WINDOWS_MS
            for metric in _BOOK_METRICS
        ),
    )
)
POLYMARKET_ROUND21_SPOT_FEATURE_NAMES = tuple(
    value.format(market="spot") for value in _MARKET_FEATURE_NAMES
)
POLYMARKET_ROUND21_USDM_FEATURE_NAMES = (
    *(
        value.format(market="usdm")
        for value in _MARKET_FEATURE_NAMES
    ),
    "usdm.log_mid_basis",
    "usdm.log_microprice_basis",
    *(
        f"usdm.spot_minus_usdm_mid_log_return_{window}ms"
        for window in POLYMARKET_ROUND21_BINANCE_WINDOWS_MS
    ),
)
_CONNECTION_ID = re.compile(r"^[a-z0-9][a-z0-9:_-]{1,159}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SPOT_BOOK_KEYS = frozenset({"u", "s", "b", "B", "a", "A"})
_USDM_BOOK_KEYS = frozenset({"e", "E", "T", "u", "s", "b", "B", "a", "A"})
_TRADE_REQUIRED_KEYS = frozenset({"e", "E", "s", "t", "p", "q", "T", "m"})
_TRADE_OPTIONAL_KEYS = frozenset({"M", "a", "b", "X", "st"})
_COMPACT_THRESHOLD = 250_000
_COMPACT_MINIMUM_DROP = 50_000


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Round 21 Binance payload contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 21 Binance payload contains {value}")


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


def _positive_decimal(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Round 21 Binance {name} is invalid")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Round 21 Binance {name} is invalid") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"Round 21 Binance {name} is invalid")
    output = float(parsed)
    if not math.isfinite(output):
        raise ValueError(f"Round 21 Binance {name} is invalid")
    return output


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Round 21 Binance {name} is invalid")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Round 21 Binance {name} is invalid") from exc
    if parsed <= 0 or str(parsed) != str(value):
        raise ValueError(f"Round 21 Binance {name} is invalid")
    return parsed


def _payload(record: CaptureFrameRecord) -> tuple[str, Mapping[str, object], str]:
    if not isinstance(record, CaptureFrameRecord):
        raise TypeError("Round 21 Binance input is not a capture-frame record")
    market = _STREAM_MARKET.get(str(record.stream))
    connection = str(record.connection_id or "").strip().lower()
    if (
        market is None
        or _CONNECTION_ID.fullmatch(connection) is None
        or int(record.sequence_number) <= 0
        or int(record.received_wall_ms) <= 0
        or int(record.received_monotonic_ns) <= 0
    ):
        raise ValueError("Round 21 Binance capture metadata is invalid")
    raw = str(record.raw_text)
    if raw in {"PING", "PONG"}:
        return market, {}, ""
    try:
        envelope = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 21 Binance payload is not strict JSON") from exc
    if not isinstance(envelope, Mapping) or set(envelope) != {"stream", "data"}:
        raise ValueError("Round 21 Binance combined-stream envelope drifted")
    stream_name = str(envelope["stream"] or "").strip()
    body = envelope["data"]
    if (
        stream_name not in {"btcusdt@bookTicker", "btcusdt@trade"}
        or not isinstance(body, Mapping)
    ):
        raise ValueError("Round 21 Binance combined-stream identity differs")
    return market, body, stream_name


@dataclass(frozen=True, slots=True)
class Round21BinanceBookTicker:
    market: Literal["spot", "usdm"]
    connection_id: str
    sequence_number: int
    received_wall_ms: int
    received_monotonic_ns: int
    update_id: int
    event_time_ms: int | None
    transaction_time_ms: int | None
    bid: float
    bid_quantity: float
    ask: float
    ask_quantity: float
    source_payload_sha256: str
    trading_authority: bool = False

    def __post_init__(self) -> None:
        if (
            self.market not in _MARKETS
            or _CONNECTION_ID.fullmatch(self.connection_id) is None
            or min(
                self.sequence_number,
                self.received_wall_ms,
                self.received_monotonic_ns,
                self.update_id,
            )
            <= 0
            or not all(
                math.isfinite(value) and value > 0.0
                for value in (
                    self.bid,
                    self.bid_quantity,
                    self.ask,
                    self.ask_quantity,
                )
            )
            or self.bid >= self.ask
            or _SHA256.fullmatch(self.source_payload_sha256) is None
            or self.trading_authority
        ):
            raise ValueError("Round 21 Binance book ticker is invalid")
        if self.market == "spot":
            if self.event_time_ms is not None or self.transaction_time_ms is not None:
                raise ValueError("Round 21 Spot book ticker invented source time")
        elif (
            self.event_time_ms is None
            or self.transaction_time_ms is None
            or not 0 < self.transaction_time_ms <= self.event_time_ms
            or self.event_time_ms > self.received_wall_ms + 5_000
        ):
            raise ValueError("Round 21 USD-M book ticker chronology is invalid")

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def microprice(self) -> float:
        return (
            self.ask * self.bid_quantity + self.bid * self.ask_quantity
        ) / (self.bid_quantity + self.ask_quantity)


@dataclass(frozen=True, slots=True)
class Round21BinanceTrade:
    market: Literal["spot", "usdm"]
    connection_id: str
    sequence_number: int
    received_wall_ms: int
    received_monotonic_ns: int
    event_time_ms: int
    trade_time_ms: int
    trade_id: int
    price: float
    quantity: float
    buyer_is_maker: bool
    source_payload_sha256: str
    trading_authority: bool = False

    def __post_init__(self) -> None:
        if (
            self.market not in _MARKETS
            or _CONNECTION_ID.fullmatch(self.connection_id) is None
            or min(
                self.sequence_number,
                self.received_wall_ms,
                self.received_monotonic_ns,
                self.event_time_ms,
                self.trade_time_ms,
                self.trade_id,
            )
            <= 0
            or not 0 < self.trade_time_ms <= self.event_time_ms
            or self.event_time_ms > self.received_wall_ms + 5_000
            or not math.isfinite(self.price)
            or self.price <= 0.0
            or not math.isfinite(self.quantity)
            or self.quantity <= 0.0
            or type(self.buyer_is_maker) is not bool
            or _SHA256.fullmatch(self.source_payload_sha256) is None
            or self.trading_authority
        ):
            raise ValueError("Round 21 Binance trade is invalid")

    @property
    def quote_notional(self) -> float:
        return self.price * self.quantity


@dataclass(frozen=True, slots=True)
class _Round21BinanceControl:
    market: Literal["spot", "usdm"]
    connection_id: str
    sequence_number: int
    received_wall_ms: int
    received_monotonic_ns: int
    source_payload_sha256: str
    control_type: str


Round21BinanceObservation = (
    Round21BinanceBookTicker | Round21BinanceTrade | _Round21BinanceControl
)


def parse_round21_binance_record(
    record: CaptureFrameRecord,
) -> Round21BinanceObservation | None:
    """Parse one exact public sidecar record without credentials or execution."""

    market, body, stream_name = _payload(record)
    raw_sha = hashlib.sha256(str(record.raw_text).encode("utf-8")).hexdigest()
    connection = str(record.connection_id).strip().lower()
    if not stream_name:
        return _Round21BinanceControl(
            market=market,  # type: ignore[arg-type]
            connection_id=connection,
            sequence_number=int(record.sequence_number),
            received_wall_ms=int(record.received_wall_ms),
            received_monotonic_ns=int(record.received_monotonic_ns),
            source_payload_sha256=raw_sha,
            control_type=str(record.raw_text).lower(),
        )
    if stream_name == "btcusdt@bookTicker":
        expected = _SPOT_BOOK_KEYS if market == "spot" else _USDM_BOOK_KEYS
        if set(body) != expected or str(body.get("s") or "").upper() != "BTCUSDT":
            raise ValueError("Round 21 Binance book-ticker schema drifted")
        if market == "usdm" and body["e"] != "bookTicker":
            raise ValueError("Round 21 USD-M book-ticker event differs")
        bid = _positive_decimal(body["b"], name="best bid")
        ask = _positive_decimal(body["a"], name="best ask")
        return Round21BinanceBookTicker(
            market=market,  # type: ignore[arg-type]
            connection_id=connection,
            sequence_number=int(record.sequence_number),
            received_wall_ms=int(record.received_wall_ms),
            received_monotonic_ns=int(record.received_monotonic_ns),
            update_id=_positive_integer(body["u"], name="update ID"),
            event_time_ms=(
                None
                if market == "spot"
                else _positive_integer(body["E"], name="event time")
            ),
            transaction_time_ms=(
                None
                if market == "spot"
                else _positive_integer(body["T"], name="transaction time")
            ),
            bid=bid,
            bid_quantity=_positive_decimal(body["B"], name="best bid quantity"),
            ask=ask,
            ask_quantity=_positive_decimal(body["A"], name="best ask quantity"),
            source_payload_sha256=raw_sha,
        )
    keys = frozenset(body)
    if (
        not _TRADE_REQUIRED_KEYS.issubset(keys)
        or keys - _TRADE_REQUIRED_KEYS - _TRADE_OPTIONAL_KEYS
        or body["e"] != "trade"
        or str(body["s"] or "").upper() != "BTCUSDT"
    ):
        raise ValueError("Round 21 Binance trade schema drifted")
    price = Decimal(str(body["p"]))
    quantity = Decimal(str(body["q"]))
    if (
        market == "usdm"
        and price == 0
        and quantity == 0
        and body.get("X") == "NA"
        and body.get("st") == 1
    ):
        return _Round21BinanceControl(
            market="usdm",
            connection_id=connection,
            sequence_number=int(record.sequence_number),
            received_wall_ms=int(record.received_wall_ms),
            received_monotonic_ns=int(record.received_monotonic_ns),
            source_payload_sha256=raw_sha,
            control_type="documented_zero_trade_sentinel",
        )
    if type(body["m"]) is not bool:
        raise ValueError("Round 21 Binance maker flag is invalid")
    return Round21BinanceTrade(
        market=market,  # type: ignore[arg-type]
        connection_id=connection,
        sequence_number=int(record.sequence_number),
        received_wall_ms=int(record.received_wall_ms),
        received_monotonic_ns=int(record.received_monotonic_ns),
        event_time_ms=_positive_integer(body["E"], name="event time"),
        trade_time_ms=_positive_integer(body["T"], name="trade time"),
        trade_id=_positive_integer(body["t"], name="trade ID"),
        price=_positive_decimal(body["p"], name="trade price"),
        quantity=_positive_decimal(body["q"], name="trade quantity"),
        buyer_is_maker=body["m"],
        source_payload_sha256=raw_sha,
    )


@dataclass(frozen=True, slots=True)
class _TradeWindow:
    log_return: float
    realized_variance: float
    signed_quote_imbalance: float
    quote_notional: float
    count: int


class _RollingTradeSeries:
    def __init__(self) -> None:
        self.received_ms: list[int] = []
        self.log_price: list[float] = []
        self.squared_return: list[float] = []
        self.signed_quote: list[float] = []
        self.quote: list[float] = []
        self.squared_prefix: list[float] = [0.0]
        self.signed_prefix: list[float] = [0.0]
        self.quote_prefix: list[float] = [0.0]
        self.last_trade_id = 0

    def _rebuild_prefix(self) -> None:
        self.squared_prefix = [0.0]
        self.signed_prefix = [0.0]
        self.quote_prefix = [0.0]
        for squared, signed, quote in zip(
            self.squared_return,
            self.signed_quote,
            self.quote,
            strict=True,
        ):
            self.squared_prefix.append(self.squared_prefix[-1] + squared)
            self.signed_prefix.append(self.signed_prefix[-1] + signed)
            self.quote_prefix.append(self.quote_prefix[-1] + quote)

    def _compact(self, latest_ms: int) -> None:
        if len(self.received_ms) < _COMPACT_THRESHOLD:
            return
        cutoff = bisect_left(
            self.received_ms,
            latest_ms - POLYMARKET_ROUND21_MAXIMUM_LOOKBACK_MS - 1_000,
        )
        drop = max(0, cutoff - 1)
        if drop < _COMPACT_MINIMUM_DROP:
            return
        self.received_ms = self.received_ms[drop:]
        self.log_price = self.log_price[drop:]
        self.signed_quote = self.signed_quote[drop:]
        self.quote = self.quote[drop:]
        self.squared_return = [
            0.0,
            *(
                (self.log_price[index] - self.log_price[index - 1]) ** 2
                for index in range(1, len(self.log_price))
            ),
        ]
        self._rebuild_prefix()

    def append(self, observation: Round21BinanceTrade) -> None:
        if (
            self.received_ms
            and observation.received_wall_ms < self.received_ms[-1]
        ) or observation.trade_id <= self.last_trade_id:
            raise ValueError("Round 21 Binance trade chronology differs")
        price = math.log(observation.price)
        quote = observation.quote_notional
        signed = -quote if observation.buyer_is_maker else quote
        squared = (
            0.0 if not self.log_price else (price - self.log_price[-1]) ** 2
        )
        self.received_ms.append(observation.received_wall_ms)
        self.log_price.append(price)
        self.squared_return.append(squared)
        self.signed_quote.append(signed)
        self.quote.append(quote)
        self.squared_prefix.append(self.squared_prefix[-1] + squared)
        self.signed_prefix.append(self.signed_prefix[-1] + signed)
        self.quote_prefix.append(self.quote_prefix[-1] + quote)
        self.last_trade_id = observation.trade_id
        self._compact(observation.received_wall_ms)

    def window(self, decision_ms: int, window_ms: int) -> _TradeWindow:
        left = bisect_left(self.received_ms, decision_ms - window_ms)
        right = bisect_right(self.received_ms, decision_ms)
        count = right - left
        if count <= 0:
            return _TradeWindow(0.0, 0.0, 0.0, 0.0, 0)
        quote = self.quote_prefix[right] - self.quote_prefix[left]
        signed = self.signed_prefix[right] - self.signed_prefix[left]
        realized = (
            0.0
            if count < 2
            else self.squared_prefix[right] - self.squared_prefix[left + 1]
        )
        tolerance = 1e-12 * max(1.0, quote)
        if (
            quote <= 0.0
            or realized < -tolerance
            or abs(signed) > quote + tolerance
        ):
            raise RuntimeError("Round 21 Binance trade-window accounting differs")
        if realized < 0.0:
            realized = 0.0
        if abs(signed) > quote:
            signed = math.copysign(quote, signed)
        return _TradeWindow(
            log_return=(
                0.0
                if count < 2
                else self.log_price[right - 1] - self.log_price[left]
            ),
            realized_variance=realized,
            signed_quote_imbalance=signed / quote,
            quote_notional=quote,
            count=count,
        )


@dataclass(frozen=True, slots=True)
class _BookWindow:
    mid_log_return: float
    microprice_log_return: float
    mean_relative_spread: float
    mean_quantity_imbalance: float
    mean_log_depth: float
    count: int


class _RollingBookSeries:
    def __init__(self) -> None:
        self.received_ms: list[int] = []
        self.log_mid: list[float] = []
        self.log_microprice: list[float] = []
        self.relative_spread: list[float] = []
        self.quantity_imbalance: list[float] = []
        self.log_depth: list[float] = []
        self.spread_prefix: list[float] = [0.0]
        self.imbalance_prefix: list[float] = [0.0]
        self.depth_prefix: list[float] = [0.0]
        self.last_update_id = 0

    def _rebuild_prefix(self) -> None:
        self.spread_prefix = [0.0]
        self.imbalance_prefix = [0.0]
        self.depth_prefix = [0.0]
        for spread, imbalance, depth in zip(
            self.relative_spread,
            self.quantity_imbalance,
            self.log_depth,
            strict=True,
        ):
            self.spread_prefix.append(self.spread_prefix[-1] + spread)
            self.imbalance_prefix.append(self.imbalance_prefix[-1] + imbalance)
            self.depth_prefix.append(self.depth_prefix[-1] + depth)

    def _compact(self, latest_ms: int) -> None:
        if len(self.received_ms) < _COMPACT_THRESHOLD:
            return
        cutoff = bisect_left(
            self.received_ms,
            latest_ms - POLYMARKET_ROUND21_MAXIMUM_LOOKBACK_MS - 1_000,
        )
        drop = max(0, cutoff - 1)
        if drop < _COMPACT_MINIMUM_DROP:
            return
        self.received_ms = self.received_ms[drop:]
        self.log_mid = self.log_mid[drop:]
        self.log_microprice = self.log_microprice[drop:]
        self.relative_spread = self.relative_spread[drop:]
        self.quantity_imbalance = self.quantity_imbalance[drop:]
        self.log_depth = self.log_depth[drop:]
        self._rebuild_prefix()

    def append(self, observation: Round21BinanceBookTicker) -> None:
        if (
            self.received_ms
            and observation.received_wall_ms < self.received_ms[-1]
        ) or observation.update_id <= self.last_update_id:
            raise ValueError("Round 21 Binance book-ticker chronology differs")
        midpoint = observation.midpoint
        quantity_total = observation.bid_quantity + observation.ask_quantity
        relative_spread = (observation.ask - observation.bid) / midpoint
        imbalance = (
            observation.bid_quantity - observation.ask_quantity
        ) / quantity_total
        log_depth = math.log1p(midpoint * quantity_total)
        self.received_ms.append(observation.received_wall_ms)
        self.log_mid.append(math.log(midpoint))
        self.log_microprice.append(math.log(observation.microprice))
        self.relative_spread.append(relative_spread)
        self.quantity_imbalance.append(imbalance)
        self.log_depth.append(log_depth)
        self.spread_prefix.append(self.spread_prefix[-1] + relative_spread)
        self.imbalance_prefix.append(self.imbalance_prefix[-1] + imbalance)
        self.depth_prefix.append(self.depth_prefix[-1] + log_depth)
        self.last_update_id = observation.update_id
        self._compact(observation.received_wall_ms)

    def window(self, decision_ms: int, window_ms: int) -> _BookWindow:
        left = bisect_left(self.received_ms, decision_ms - window_ms)
        right = bisect_right(self.received_ms, decision_ms)
        count = right - left
        if count <= 0:
            return _BookWindow(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        divisor = float(count)
        return _BookWindow(
            mid_log_return=(
                0.0
                if count < 2
                else self.log_mid[right - 1] - self.log_mid[left]
            ),
            microprice_log_return=(
                0.0
                if count < 2
                else self.log_microprice[right - 1]
                - self.log_microprice[left]
            ),
            mean_relative_spread=(
                self.spread_prefix[right] - self.spread_prefix[left]
            )
            / divisor,
            mean_quantity_imbalance=(
                self.imbalance_prefix[right] - self.imbalance_prefix[left]
            )
            / divisor,
            mean_log_depth=(
                self.depth_prefix[right] - self.depth_prefix[left]
            )
            / divisor,
            count=count,
        )

    def current(self, decision_ms: int) -> tuple[float, float] | None:
        right = bisect_right(self.received_ms, decision_ms)
        if right <= 0:
            return None
        index = right - 1
        return self.log_mid[index], self.log_microprice[index]

    @property
    def latest_received_ms(self) -> int | None:
        return None if not self.received_ms else self.received_ms[-1]


@dataclass(frozen=True, slots=True)
class _MarketSnapshot:
    available: bool
    values: tuple[float, ...]
    mid_returns: tuple[float, ...]
    current_log_mid: float
    current_log_microprice: float
    source_chain_sha256: str
    maximum_receipt_ms: int


class _MarketAccumulator:
    def __init__(
        self,
        *,
        market: Literal["spot", "usdm"],
        connection_id: str,
    ) -> None:
        self.market = market
        self.connection_id = connection_id
        self.trades = _RollingTradeSeries()
        self.books = _RollingBookSeries()
        self.last_sequence_number = 0
        self.last_received_monotonic_ns = 0
        self.last_received_wall_ms = 0
        self.source_chain_sha256 = _EMPTY_SHA256

    def ingest(self, observation: Round21BinanceObservation) -> None:
        if (
            observation.market != self.market
            or observation.connection_id != self.connection_id
            or observation.sequence_number != self.last_sequence_number + 1
            or observation.received_monotonic_ns
            <= self.last_received_monotonic_ns
            or observation.received_wall_ms < self.last_received_wall_ms
        ):
            raise ValueError("Round 21 Binance connection epoch differs")
        identity = {
            "market": observation.market,
            "connection_id": observation.connection_id,
            "sequence_number": observation.sequence_number,
            "received_wall_ms": observation.received_wall_ms,
            "received_monotonic_ns": observation.received_monotonic_ns,
            "source_payload_sha256": observation.source_payload_sha256,
            "observation_type": (
                "book_ticker"
                if isinstance(observation, Round21BinanceBookTicker)
                else (
                    "trade"
                    if isinstance(observation, Round21BinanceTrade)
                    else observation.control_type
                )
            ),
        }
        self.source_chain_sha256 = hashlib.sha256(
            bytes.fromhex(self.source_chain_sha256)
            + _canonical_json(identity).encode("ascii")
        ).hexdigest()
        if isinstance(observation, Round21BinanceBookTicker):
            self.books.append(observation)
        elif isinstance(observation, Round21BinanceTrade):
            self.trades.append(observation)
        self.last_sequence_number = observation.sequence_number
        self.last_received_monotonic_ns = observation.received_monotonic_ns
        self.last_received_wall_ms = observation.received_wall_ms

    def snapshot(self, decision_ms: int) -> _MarketSnapshot:
        decision = int(decision_ms)
        if self.last_received_wall_ms > decision:
            raise ValueError("Round 21 Binance accumulator contains future receipts")
        latest_book = self.books.latest_received_ms
        available = (
            latest_book is not None
            and 0 <= decision - latest_book
            <= POLYMARKET_ROUND21_BINANCE_MAXIMUM_BBO_AGE_MS
        )
        zero_count = len(_MARKET_FEATURE_NAMES)
        if not available:
            return _MarketSnapshot(
                available=False,
                values=(0.0,) * zero_count,
                mid_returns=(0.0,) * len(POLYMARKET_ROUND21_BINANCE_WINDOWS_MS),
                current_log_mid=0.0,
                current_log_microprice=0.0,
                source_chain_sha256=_EMPTY_SHA256,
                maximum_receipt_ms=0,
            )
        values: list[float] = []
        mid_returns: list[float] = []
        for window_ms in POLYMARKET_ROUND21_BINANCE_WINDOWS_MS:
            item = self.trades.window(decision, window_ms)
            values.extend(
                (
                    item.log_return,
                    item.realized_variance,
                    item.signed_quote_imbalance,
                    math.log1p(item.quote_notional),
                    float(item.count),
                )
            )
        for window_ms in POLYMARKET_ROUND21_BINANCE_WINDOWS_MS:
            item = self.books.window(decision, window_ms)
            mid_returns.append(item.mid_log_return)
            values.extend(
                (
                    item.mid_log_return,
                    item.microprice_log_return,
                    item.mean_relative_spread,
                    item.mean_quantity_imbalance,
                    item.mean_log_depth,
                    float(item.count),
                )
            )
        current = self.books.current(decision)
        if current is None:
            raise RuntimeError("Round 21 available Binance book has no current state")
        return _MarketSnapshot(
            available=True,
            values=tuple(values),
            mid_returns=tuple(mid_returns),
            current_log_mid=current[0],
            current_log_microprice=current[1],
            source_chain_sha256=self.source_chain_sha256,
            maximum_receipt_ms=self.last_received_wall_ms,
        )


@dataclass(frozen=True, slots=True)
class Round21OptionalBinanceFeatures:
    decision_time_ms: int
    spot_values: tuple[float, ...]
    usdm_values: tuple[float, ...]
    spot_available: bool
    usdm_available: bool
    spot_source_chain_sha256: str
    usdm_source_chain_sha256: str
    spot_maximum_receipt_ms: int
    usdm_maximum_receipt_ms: int
    trading_authority: bool = False

    def __post_init__(self) -> None:
        if (
            self.decision_time_ms <= 0
            or len(self.spot_values) != len(POLYMARKET_ROUND21_SPOT_FEATURE_NAMES)
            or len(self.usdm_values) != len(POLYMARKET_ROUND21_USDM_FEATURE_NAMES)
            or any(not math.isfinite(value) for value in self.spot_values)
            or any(not math.isfinite(value) for value in self.usdm_values)
            or type(self.spot_available) is not bool
            or type(self.usdm_available) is not bool
            or self.usdm_available and not self.spot_available
            or _SHA256.fullmatch(self.spot_source_chain_sha256) is None
            or _SHA256.fullmatch(self.usdm_source_chain_sha256) is None
            or self.trading_authority
        ):
            raise ValueError("Round 21 optional Binance features are invalid")
        if (
            not self.spot_available
            and (
                any(self.spot_values)
                or self.spot_source_chain_sha256 != _EMPTY_SHA256
                or self.spot_maximum_receipt_ms != 0
            )
        ) or (
            not self.usdm_available
            and (
                any(self.usdm_values)
                or self.usdm_source_chain_sha256 != _EMPTY_SHA256
                or self.usdm_maximum_receipt_ms != 0
            )
        ):
            raise ValueError("Round 21 optional Binance missingness differs")


class Round21IndependentBinanceFeatureEngine:
    """Keep optional Binance epochs separate from Polymarket state."""

    credentials_used = False
    account_connected = False
    execution_connected = False
    trading_authority = False

    def __init__(self) -> None:
        self._markets: dict[str, _MarketAccumulator] = {}

    def reset_market(self, market: str, connection_id: str) -> None:
        selected = str(market or "").strip().lower()
        connection = str(connection_id or "").strip().lower()
        if selected not in _MARKETS or _CONNECTION_ID.fullmatch(connection) is None:
            raise ValueError("Round 21 Binance reset identity is invalid")
        self._markets[selected] = _MarketAccumulator(
            market=selected,  # type: ignore[arg-type]
            connection_id=connection,
        )

    def ingest_record(self, record: CaptureFrameRecord) -> None:
        observation = parse_round21_binance_record(record)
        if observation is None:
            raise RuntimeError("Round 21 Binance parser returned no observation")
        accumulator = self._markets.get(observation.market)
        if accumulator is None:
            self.reset_market(observation.market, observation.connection_id)
            accumulator = self._markets[observation.market]
        elif accumulator.connection_id != observation.connection_id:
            raise ValueError(
                "Round 21 Binance reconnect requires an explicit epoch reset"
            )
        accumulator.ingest(observation)

    def build(self, decision_time_ms: int) -> Round21OptionalBinanceFeatures:
        decision = int(decision_time_ms)
        empty = _MarketSnapshot(
            available=False,
            values=(0.0,) * len(_MARKET_FEATURE_NAMES),
            mid_returns=(0.0,) * len(POLYMARKET_ROUND21_BINANCE_WINDOWS_MS),
            current_log_mid=0.0,
            current_log_microprice=0.0,
            source_chain_sha256=_EMPTY_SHA256,
            maximum_receipt_ms=0,
        )
        spot = (
            empty
            if "spot" not in self._markets
            else self._markets["spot"].snapshot(decision)
        )
        futures = (
            empty
            if "usdm" not in self._markets
            else self._markets["usdm"].snapshot(decision)
        )
        usdm_available = spot.available and futures.available
        if usdm_available:
            cross = (
                futures.current_log_mid - spot.current_log_mid,
                futures.current_log_microprice - spot.current_log_microprice,
                *(
                    spot_return - futures_return
                    for spot_return, futures_return in zip(
                        spot.mid_returns,
                        futures.mid_returns,
                        strict=True,
                    )
                ),
            )
            usdm_values = (*futures.values, *cross)
            usdm_chain = _canonical_sha256(
                {
                    "spot_source_chain_sha256": spot.source_chain_sha256,
                    "usdm_source_chain_sha256": futures.source_chain_sha256,
                    "decision_time_ms": decision,
                }
            )
            usdm_receipt = max(
                spot.maximum_receipt_ms,
                futures.maximum_receipt_ms,
            )
        else:
            usdm_values = (0.0,) * len(POLYMARKET_ROUND21_USDM_FEATURE_NAMES)
            usdm_chain = _EMPTY_SHA256
            usdm_receipt = 0
        return Round21OptionalBinanceFeatures(
            decision_time_ms=decision,
            spot_values=spot.values,
            usdm_values=tuple(usdm_values),
            spot_available=spot.available,
            usdm_available=usdm_available,
            spot_source_chain_sha256=spot.source_chain_sha256,
            usdm_source_chain_sha256=usdm_chain,
            spot_maximum_receipt_ms=spot.maximum_receipt_ms,
            usdm_maximum_receipt_ms=usdm_receipt,
        )


__all__ = [
    "POLYMARKET_ROUND21_BINANCE_FEATURE_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_BINANCE_MAXIMUM_BBO_AGE_MS",
    "POLYMARKET_ROUND21_BINANCE_WINDOWS_MS",
    "POLYMARKET_ROUND21_SPOT_FEATURE_NAMES",
    "POLYMARKET_ROUND21_USDM_FEATURE_NAMES",
    "Round21BinanceBookTicker",
    "Round21BinanceObservation",
    "Round21BinanceTrade",
    "Round21IndependentBinanceFeatureEngine",
    "Round21OptionalBinanceFeatures",
    "parse_round21_binance_record",
]
