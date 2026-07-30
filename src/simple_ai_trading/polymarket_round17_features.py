"""Target-free multiscale flow features for the Round 17 BTC 5m model."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping, Sequence

from .paper_execution import PaperBookSnapshot
from .polymarket_btc_reference import PolymarketChainlinkBtcTick
from .polymarket_round14_dataset import PolymarketRound14ConditionAdmission
from .polymarket_round14_features import (
    POLYMARKET_ROUND14_FEATURE_NAMES,
    POLYMARKET_ROUND14_FEATURE_NAMES_SHA256,
    PolymarketRound14FeatureRow,
)


POLYMARKET_ROUND17_FEATURE_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-causal-flow-feature-v1"
)
POLYMARKET_ROUND17_CONTRACT_SHA256 = (
    "f9cbe114959fce14ea3292b3fc65de68b6a86b157c2402eeeefca2a95a808b44"
)
POLYMARKET_ROUND17_CHAINLINK_WINDOWS_MS = (
    250,
    1_000,
    5_000,
    15_000,
    30_000,
    60_000,
    120_000,
)
POLYMARKET_ROUND17_BINANCE_WINDOWS_MS = (
    250,
    1_000,
    5_000,
    15_000,
    30_000,
)
POLYMARKET_ROUND17_BOOK_WINDOWS_MS = (250, 1_000, 5_000, 15_000)
_CHAINLINK_METRICS = (
    "log_return",
    "realized_variance",
    "bipower_variation",
    "jump_fraction",
    "tick_count",
)
_BINANCE_METRICS = (
    "log_return",
    "realized_variance",
    "signed_quote_imbalance",
    "log1p_quote_notional",
    "trade_count",
)
_BOOK_METRICS = (
    "top_of_book_order_flow_imbalance",
    "mean_top_imbalance",
    "microprice_log_return",
    "level_quantity_flow_pressure",
    "log1p_gross_level_quantity_change",
    "book_update_count",
)
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONNECTION_ID = re.compile(r"^[A-Za-z0-9:._-]{1,160}$")
_EMPTY_CHAIN_SHA256 = hashlib.sha256(b"").hexdigest()


def _window_name(window_ms: int) -> str:
    return f"{window_ms}ms"


POLYMARKET_ROUND17_TEMPORAL_FEATURE_NAMES = (
    *(
        f"chainlink_{metric}_{_window_name(window)}"
        for window in POLYMARKET_ROUND17_CHAINLINK_WINDOWS_MS
        for metric in _CHAINLINK_METRICS
    ),
    *(
        f"binance_{market}_{metric}_{_window_name(window)}"
        for market in ("spot", "perpetual")
        for window in POLYMARKET_ROUND17_BINANCE_WINDOWS_MS
        for metric in _BINANCE_METRICS
    ),
    *(
        f"polymarket_{outcome}_{metric}_{_window_name(window)}"
        for outcome in ("up", "down")
        for window in POLYMARKET_ROUND17_BOOK_WINDOWS_MS
        for metric in _BOOK_METRICS
    ),
    *(
        f"binance_spot_minus_perpetual_log_return_{_window_name(window)}"
        for window in POLYMARKET_ROUND17_BINANCE_WINDOWS_MS
    ),
)
POLYMARKET_ROUND17_FEATURE_NAMES = (
    *POLYMARKET_ROUND14_FEATURE_NAMES,
    *POLYMARKET_ROUND17_TEMPORAL_FEATURE_NAMES,
)
POLYMARKET_ROUND17_FEATURE_NAMES_SHA256 = hashlib.sha256(
    json.dumps(
        POLYMARKET_ROUND17_FEATURE_NAMES,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
).hexdigest()


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


def _validated_float(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    parsed = float(value)
    if not math.isfinite(parsed) or (positive and parsed <= 0.0):
        raise ValueError(f"{name} must be finite")
    return parsed


def _chain_sha256(previous: str, payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        bytes.fromhex(previous) + _canonical_json(payload).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class PolymarketRound17BinanceTrade:
    """One credential-free Binance raw trade with no execution authority."""

    market: str
    source: str
    symbol: str
    connection_id: str
    event_time_ms: int
    received_at_ms: int
    trade_id: int
    price: float
    quantity: float
    buyer_is_maker: bool
    source_event_sha256: str
    trading_authority: bool = False

    def __post_init__(self) -> None:
        market = str(self.market or "").strip().lower()
        expected_source = {
            "spot": "BINANCE_SPOT",
            "perpetual": "BINANCE_USD_M_FUTURES",
        }.get(market)
        source = str(self.source or "").strip().upper()
        symbol = str(self.symbol or "").strip().upper()
        connection = str(self.connection_id or "").strip()
        event_time = int(self.event_time_ms)
        received = int(self.received_at_ms)
        trade_id = int(self.trade_id)
        price = _validated_float(self.price, name="Binance trade price", positive=True)
        quantity = _validated_float(
            self.quantity,
            name="Binance trade quantity",
            positive=True,
        )
        source_sha256 = str(self.source_event_sha256 or "").strip().lower()
        if (
            expected_source is None
            or source != expected_source
            or symbol != "BTCUSDT"
            or _CONNECTION_ID.fullmatch(connection) is None
            or event_time <= 0
            or received <= 0
            or received < event_time - 250
            or trade_id < 0
            or not math.isfinite(price * quantity)
            or type(self.buyer_is_maker) is not bool
            or _SHA256.fullmatch(source_sha256) is None
            or self.trading_authority
        ):
            raise ValueError("Round 17 Binance raw trade is invalid")
        object.__setattr__(self, "market", market)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "connection_id", connection)
        object.__setattr__(self, "event_time_ms", event_time)
        object.__setattr__(self, "received_at_ms", received)
        object.__setattr__(self, "trade_id", trade_id)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "source_event_sha256", source_sha256)

    @property
    def quote_notional(self) -> float:
        return self.price * self.quantity


@dataclass(frozen=True, slots=True)
class _PriceWindow:
    log_return: float
    realized_variance: float
    bipower_variation: float
    jump_fraction: float
    observation_count: int
    signed_quote_imbalance: float
    quote_notional: float


class _PriceSeries:
    def __init__(self, *, source: str, maximum_lookback_ms: int) -> None:
        self.source = source
        self.maximum_lookback_ms = int(maximum_lookback_ms)
        self.received_ms: list[int] = []
        self.event_ms: list[int] = []
        self.log_prices: list[float] = []
        self.cumulative_squared_return: list[float] = []
        self.cumulative_bipower_product: list[float] = []
        self.cumulative_signed_quote: list[float] = []
        self.cumulative_quote: list[float] = []
        self.chain_sha256 = _EMPTY_CHAIN_SHA256
        self._observation_sha256: dict[str, str] = {}

    def append(
        self,
        *,
        event_time_ms: int,
        received_at_ms: int,
        price: object,
        signed_quote: object = 0.0,
        quote_notional: object = 0.0,
        observation_id: str,
        identity: Mapping[str, object],
    ) -> bool:
        event_time = int(event_time_ms)
        received = int(received_at_ms)
        parsed_price = _validated_float(
            price, name=f"{self.source} price", positive=True
        )
        signed = _validated_float(
            signed_quote,
            name=f"{self.source} signed quote",
        )
        gross = _validated_float(
            quote_notional,
            name=f"{self.source} quote notional",
        )
        normalized_observation_id = str(observation_id or "").strip()
        payload = {
            "source": self.source,
            "event_time_ms": event_time,
            "received_at_ms": received,
            "price": format(parsed_price, ".17g"),
            "signed_quote": format(signed, ".17g"),
            "quote_notional": format(gross, ".17g"),
            "identity": dict(identity),
        }
        payload_sha256 = _canonical_sha256(payload)
        prior_sha256 = self._observation_sha256.get(normalized_observation_id)
        if not normalized_observation_id:
            raise ValueError(f"{self.source} observation identity is invalid")
        if prior_sha256 is not None:
            if prior_sha256 != payload_sha256:
                raise ValueError(f"{self.source} duplicate observation conflicts")
            return False
        if (
            event_time <= 0
            or received <= 0
            or received < event_time - 250
            or gross < 0.0
            or abs(signed) > gross + 1e-9
            or (self.received_ms and received < self.received_ms[-1])
        ):
            raise ValueError(f"{self.source} chronology or quote flow is invalid")
        log_price = math.log(parsed_price)
        if self.log_prices:
            price_return = log_price - self.log_prices[-1]
            squared = self.cumulative_squared_return[-1] + price_return**2
            prior_return = (
                0.0
                if len(self.log_prices) < 2
                else self.log_prices[-1] - self.log_prices[-2]
            )
            bipower = self.cumulative_bipower_product[-1] + abs(
                price_return * prior_return
            )
            signed_total = self.cumulative_signed_quote[-1] + signed
            gross_total = self.cumulative_quote[-1] + gross
        else:
            squared = 0.0
            bipower = 0.0
            signed_total = signed
            gross_total = gross
        self.received_ms.append(received)
        self.event_ms.append(event_time)
        self.log_prices.append(log_price)
        self.cumulative_squared_return.append(squared)
        self.cumulative_bipower_product.append(bipower)
        self.cumulative_signed_quote.append(signed_total)
        self.cumulative_quote.append(gross_total)
        self.chain_sha256 = _chain_sha256(self.chain_sha256, payload)
        self._observation_sha256[normalized_observation_id] = payload_sha256
        return True

    @staticmethod
    def _prefix_difference(values: Sequence[float], end: int, before: int) -> float:
        return values[end] - (0.0 if before < 0 else values[before])

    def window(self, *, decision_time_ms: int, window_ms: int) -> _PriceWindow:
        decision = int(decision_time_ms)
        start = decision - int(window_ms)
        end_index = bisect_right(self.received_ms, decision) - 1
        baseline_index = bisect_right(self.received_ms, start) - 1
        first_index = bisect_left(self.received_ms, start)
        if end_index < 0:
            return _PriceWindow(0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
        count = max(0, end_index - first_index + 1)
        before_index = first_index - 1
        signed_quote = self._prefix_difference(
            self.cumulative_signed_quote,
            end_index,
            before_index,
        )
        quote_notional = self._prefix_difference(
            self.cumulative_quote,
            end_index,
            before_index,
        )
        signed_imbalance = (
            0.0
            if quote_notional <= 0.0
            else max(-1.0, min(1.0, signed_quote / quote_notional))
        )
        if baseline_index < 0 or baseline_index >= end_index:
            return _PriceWindow(
                0.0,
                0.0,
                0.0,
                0.0,
                count,
                signed_imbalance,
                quote_notional,
            )
        realized_variance = self._prefix_difference(
            self.cumulative_squared_return,
            end_index,
            baseline_index,
        )
        first_return_index = baseline_index + 1
        bipower_product = (
            0.0
            if end_index <= first_return_index
            else self._prefix_difference(
                self.cumulative_bipower_product,
                end_index,
                first_return_index,
            )
        )
        bipower_variation = (math.pi / 2.0) * bipower_product
        jump_fraction = (
            0.0
            if realized_variance <= 0.0
            else max(
                0.0,
                min(
                    1.0,
                    (realized_variance - bipower_variation) / realized_variance,
                ),
            )
        )
        return _PriceWindow(
            log_return=self.log_prices[end_index] - self.log_prices[baseline_index],
            realized_variance=realized_variance,
            bipower_variation=bipower_variation,
            jump_fraction=jump_fraction,
            observation_count=count,
            signed_quote_imbalance=signed_imbalance,
            quote_notional=quote_notional,
        )

    @property
    def latest_received_ms(self) -> int | None:
        return None if not self.received_ms else self.received_ms[-1]


@dataclass(frozen=True, slots=True)
class _BookWindow:
    top_of_book_order_flow_imbalance: float
    mean_top_imbalance: float
    microprice_log_return: float
    level_quantity_flow_pressure: float
    gross_level_quantity_change: float
    update_count: int


def _top_levels(
    snapshot: PaperBookSnapshot,
) -> tuple[dict[float, float], dict[float, float]]:
    bids = {float(level.price): float(level.quantity) for level in snapshot.bids[:20]}
    asks = {float(level.price): float(level.quantity) for level in snapshot.asks[:20]}
    return bids, asks


def _top_of_book(
    snapshot: PaperBookSnapshot,
) -> tuple[float, float, float, float, float, float]:
    if not snapshot.bids or not snapshot.asks:
        raise ValueError("Round 17 book snapshot requires both sides")
    bid_price = float(snapshot.bids[0].price)
    bid_quantity = float(snapshot.bids[0].quantity)
    ask_price = float(snapshot.asks[0].price)
    ask_quantity = float(snapshot.asks[0].quantity)
    gross = bid_quantity + ask_quantity
    if gross <= 0.0 or bid_price >= ask_price:
        raise ValueError("Round 17 book snapshot is invalid")
    imbalance = (bid_quantity - ask_quantity) / gross
    microprice = (ask_price * bid_quantity + bid_price * ask_quantity) / gross
    return bid_price, bid_quantity, ask_price, ask_quantity, imbalance, microprice


class _BookSeries:
    def __init__(
        self,
        *,
        outcome: str,
        market_id: str,
        asset_id: str,
    ) -> None:
        self.outcome = outcome
        self.market_id = market_id
        self.asset_id = asset_id
        self.received_ms: list[int] = []
        self.log_microprices: list[float] = []
        self.cumulative_signed_ofi: list[float] = []
        self.cumulative_absolute_ofi: list[float] = []
        self.cumulative_imbalance: list[float] = []
        self.cumulative_level_pressure: list[float] = []
        self.cumulative_level_gross: list[float] = []
        self.chain_sha256 = _EMPTY_CHAIN_SHA256
        self._prior_top: tuple[float, float, float, float] | None = None
        self._prior_levels: tuple[dict[float, float], dict[float, float]] | None = None
        self._observation_sha256: dict[str, str] = {}

    def append(self, snapshot: PaperBookSnapshot) -> bool:
        selected = snapshot.validated()
        received = int(selected.received_wall_ms)
        if (
            selected.venue != "polymarket"
            or selected.market_id != self.market_id
            or selected.asset_id != self.asset_id
            or not selected.connected
            or not selected.gap_free
        ):
            raise ValueError("Round 17 book identity or gap state differs")
        (
            bid_price,
            bid_quantity,
            ask_price,
            ask_quantity,
            imbalance,
            microprice,
        ) = _top_of_book(selected)
        levels = _top_levels(selected)
        payload = {
            "outcome": self.outcome,
            "venue": selected.venue,
            "market_id": selected.market_id,
            "asset_id": selected.asset_id,
            "received_wall_ms": received,
            "source_time_ms": selected.source_time_ms,
            "source_payload_sha256": selected.source_payload_sha256,
            "bids": [
                [format(price, ".17g"), format(quantity, ".17g")]
                for price, quantity in levels[0].items()
            ],
            "asks": [
                [format(price, ".17g"), format(quantity, ".17g")]
                for price, quantity in levels[1].items()
            ],
        }
        payload_sha256 = _canonical_sha256(payload)
        prior_sha256 = self._observation_sha256.get(selected.source_payload_sha256)
        if prior_sha256 is not None:
            if prior_sha256 != payload_sha256:
                raise ValueError("Round 17 duplicate book observation conflicts")
            return False
        if self.received_ms and received < self.received_ms[-1]:
            raise ValueError("Round 17 book chronology differs")
        signed_ofi = 0.0
        absolute_ofi = 0.0
        level_pressure = 0.0
        level_gross = 0.0
        if self._prior_top is not None and self._prior_levels is not None:
            prior_bid_price, prior_bid_quantity, prior_ask_price, prior_ask_quantity = (
                self._prior_top
            )
            bid_component = (bid_quantity if bid_price >= prior_bid_price else 0.0) - (
                prior_bid_quantity if bid_price <= prior_bid_price else 0.0
            )
            ask_component = (ask_quantity if ask_price <= prior_ask_price else 0.0) - (
                prior_ask_quantity if ask_price >= prior_ask_price else 0.0
            )
            signed_ofi = bid_component - ask_component
            absolute_ofi = abs(bid_component) + abs(ask_component)
            prior_bids, prior_asks = self._prior_levels
            current_bids, current_asks = levels
            for price in prior_bids.keys() | current_bids.keys():
                change = current_bids.get(price, 0.0) - prior_bids.get(price, 0.0)
                level_pressure += change
                level_gross += abs(change)
            for price in prior_asks.keys() | current_asks.keys():
                change = current_asks.get(price, 0.0) - prior_asks.get(price, 0.0)
                level_pressure -= change
                level_gross += abs(change)
        self.received_ms.append(received)
        self.log_microprices.append(math.log(microprice))
        self.cumulative_signed_ofi.append(
            signed_ofi
            + (
                0.0
                if not self.cumulative_signed_ofi
                else self.cumulative_signed_ofi[-1]
            )
        )
        self.cumulative_absolute_ofi.append(
            absolute_ofi
            + (
                0.0
                if not self.cumulative_absolute_ofi
                else self.cumulative_absolute_ofi[-1]
            )
        )
        self.cumulative_imbalance.append(
            imbalance
            + (0.0 if not self.cumulative_imbalance else self.cumulative_imbalance[-1])
        )
        self.cumulative_level_pressure.append(
            level_pressure
            + (
                0.0
                if not self.cumulative_level_pressure
                else self.cumulative_level_pressure[-1]
            )
        )
        self.cumulative_level_gross.append(
            level_gross
            + (
                0.0
                if not self.cumulative_level_gross
                else self.cumulative_level_gross[-1]
            )
        )
        self.chain_sha256 = _chain_sha256(self.chain_sha256, payload)
        self._observation_sha256[selected.source_payload_sha256] = payload_sha256
        self._prior_top = bid_price, bid_quantity, ask_price, ask_quantity
        self._prior_levels = levels
        return True

    @staticmethod
    def _difference(values: Sequence[float], end: int, before: int) -> float:
        return values[end] - (0.0 if before < 0 else values[before])

    def window(self, *, decision_time_ms: int, window_ms: int) -> _BookWindow:
        decision = int(decision_time_ms)
        start = decision - int(window_ms)
        end_index = bisect_right(self.received_ms, decision) - 1
        baseline_index = bisect_right(self.received_ms, start) - 1
        first_index = bisect_left(self.received_ms, start)
        if end_index < 0 or first_index > end_index:
            return _BookWindow(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        before_index = first_index - 1
        updates = end_index - first_index + 1
        signed_ofi = self._difference(
            self.cumulative_signed_ofi,
            end_index,
            before_index,
        )
        absolute_ofi = self._difference(
            self.cumulative_absolute_ofi,
            end_index,
            before_index,
        )
        mean_imbalance = (
            self._difference(
                self.cumulative_imbalance,
                end_index,
                before_index,
            )
            / updates
        )
        level_pressure = self._difference(
            self.cumulative_level_pressure,
            end_index,
            before_index,
        )
        level_gross = self._difference(
            self.cumulative_level_gross,
            end_index,
            before_index,
        )
        microprice_return = (
            0.0
            if baseline_index < 0 or baseline_index >= end_index
            else self.log_microprices[end_index] - self.log_microprices[baseline_index]
        )
        return _BookWindow(
            top_of_book_order_flow_imbalance=(
                0.0
                if absolute_ofi <= 0.0
                else max(-1.0, min(1.0, signed_ofi / absolute_ofi))
            ),
            mean_top_imbalance=max(-1.0, min(1.0, mean_imbalance)),
            microprice_log_return=microprice_return,
            level_quantity_flow_pressure=(
                0.0
                if level_gross <= 0.0
                else max(-1.0, min(1.0, level_pressure / level_gross))
            ),
            gross_level_quantity_change=level_gross,
            update_count=updates,
        )

    @property
    def latest_received_ms(self) -> int | None:
        return None if not self.received_ms else self.received_ms[-1]


@dataclass(frozen=True, slots=True)
class PolymarketRound17FeatureRow:
    condition_id: str
    decision_time_ms: int
    admission_sha256: str
    causal_segment_sha256: str
    feature_names_sha256: str
    input_sha256: str
    values_sha256: str
    values: tuple[float, ...]
    trading_authority: bool = False

    def __post_init__(self) -> None:
        normalized_values = tuple(float(value) for value in self.values)
        if (
            _CONDITION_ID.fullmatch(str(self.condition_id)) is None
            or int(self.decision_time_ms) <= 0
            or _SHA256.fullmatch(str(self.admission_sha256)) is None
            or _SHA256.fullmatch(str(self.causal_segment_sha256)) is None
            or self.feature_names_sha256 != POLYMARKET_ROUND17_FEATURE_NAMES_SHA256
            or _SHA256.fullmatch(str(self.input_sha256)) is None
            or _SHA256.fullmatch(str(self.values_sha256)) is None
            or len(self.values) != len(POLYMARKET_ROUND17_FEATURE_NAMES)
            or any(
                isinstance(value, bool) or not math.isfinite(float(value))
                for value in self.values
            )
            or self.values_sha256 != _canonical_sha256(list(normalized_values))
            or self.trading_authority
        ):
            raise ValueError("Round 17 feature row is invalid")

    def value_map(self) -> dict[str, float]:
        return dict(zip(POLYMARKET_ROUND17_FEATURE_NAMES, self.values, strict=True))

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND17_FEATURE_SCHEMA_VERSION,
            "condition_id": self.condition_id,
            "decision_time_ms": self.decision_time_ms,
            "admission_sha256": self.admission_sha256,
            "causal_segment_sha256": self.causal_segment_sha256,
            "feature_names_sha256": self.feature_names_sha256,
            "input_sha256": self.input_sha256,
            "values_sha256": self.values_sha256,
            "values": list(self.values),
            "trading_authority": False,
        }


class PolymarketRound17FeatureAccumulator:
    """Incrementally build target-free features from receipt-time observations."""

    def __init__(
        self,
        *,
        condition_id: str,
        market_id: str,
        up_token_id: str,
        down_token_id: str,
        event_start_ms: int,
        event_end_ms: int,
        admission: PolymarketRound14ConditionAdmission,
        causal_segment_sha256: str,
    ) -> None:
        condition = str(condition_id or "").strip().lower()
        market = str(market_id or "").strip()
        up_token = str(up_token_id or "").strip()
        down_token = str(down_token_id or "").strip()
        start = int(event_start_ms)
        end = int(event_end_ms)
        if not isinstance(admission, PolymarketRound14ConditionAdmission):
            raise TypeError("Round 17 condition admission type differs")
        verified_admission = admission.validated()
        causal_segment = str(causal_segment_sha256 or "").strip().lower()
        if (
            _CONDITION_ID.fullmatch(condition) is None
            or market != condition
            or _TOKEN_ID.fullmatch(up_token) is None
            or _TOKEN_ID.fullmatch(down_token) is None
            or up_token == down_token
            or start <= 0
            or start % 300_000
            or end - start != 300_000
            or _SHA256.fullmatch(causal_segment) is None
            or not verified_admission.core_eligible
            or verified_admission.condition_id != condition
            or verified_admission.event_start_ms != start
            or verified_admission.event_end_ms != end
        ):
            raise ValueError("Round 17 accumulator identity is invalid")
        self.condition_id = condition
        self.market_id = market
        self.up_token_id = up_token
        self.down_token_id = down_token
        self.event_start_ms = start
        self.event_end_ms = end
        self.capture_run_id = verified_admission.run_id
        self.admission_sha256 = verified_admission.admission_sha256
        self.binance_layer_eligible = verified_admission.binance_layer_eligible
        self.causal_segment_sha256 = causal_segment
        self.maximum_lookback_ms = max(POLYMARKET_ROUND17_CHAINLINK_WINDOWS_MS)
        self._chainlink = _PriceSeries(
            source="chainlink",
            maximum_lookback_ms=self.maximum_lookback_ms,
        )
        self._binance = {
            market_name: _PriceSeries(
                source=f"binance_{market_name}",
                maximum_lookback_ms=self.maximum_lookback_ms,
            )
            for market_name in ("spot", "perpetual")
        }
        self._books = {
            "up": _BookSeries(
                outcome="up",
                market_id=market,
                asset_id=up_token,
            ),
            "down": _BookSeries(
                outcome="down",
                market_id=market,
                asset_id=down_token,
            ),
        }
        self._last_decision_ms = 0

    def _validate_causal_segment(self, causal_segment_sha256: str) -> None:
        selected = str(causal_segment_sha256 or "").strip().lower()
        if selected != self.causal_segment_sha256:
            raise ValueError("Round 17 input crossed a causal stream segment")

    def _validate_observation_time(self, received_at_ms: int) -> None:
        received = int(received_at_ms)
        if not (
            self.event_start_ms - self.maximum_lookback_ms
            <= received
            < self.event_end_ms
        ):
            raise ValueError("Round 17 observation lies outside the causal window")

    def ingest_chainlink(
        self,
        tick: PolymarketChainlinkBtcTick,
        *,
        causal_segment_sha256: str,
    ) -> None:
        if not isinstance(tick, PolymarketChainlinkBtcTick):
            raise TypeError("Round 17 Chainlink input type differs")
        self._validate_causal_segment(causal_segment_sha256)
        self._validate_observation_time(tick.received_at_ms)
        self._chainlink.append(
            event_time_ms=tick.source_time_ms,
            received_at_ms=tick.received_at_ms,
            price=tick.price,
            observation_id=tick.source_payload_sha256,
            identity={"source_payload_sha256": tick.source_payload_sha256},
        )

    def ingest_binance(
        self,
        trade: PolymarketRound17BinanceTrade,
        *,
        causal_segment_sha256: str,
    ) -> None:
        if not isinstance(trade, PolymarketRound17BinanceTrade):
            raise TypeError("Round 17 Binance input type differs")
        self._validate_causal_segment(causal_segment_sha256)
        if not self.binance_layer_eligible:
            raise ValueError("Round 17 Binance input lacks condition admission")
        self._validate_observation_time(trade.received_at_ms)
        quote = trade.quote_notional
        self._binance[trade.market].append(
            event_time_ms=trade.event_time_ms,
            received_at_ms=trade.received_at_ms,
            price=trade.price,
            signed_quote=(-quote if trade.buyer_is_maker else quote),
            quote_notional=quote,
            observation_id=f"{trade.connection_id}:{trade.trade_id}",
            identity={
                "source": trade.source,
                "symbol": trade.symbol,
                "connection_id": trade.connection_id,
                "trade_id": trade.trade_id,
                "source_event_sha256": trade.source_event_sha256,
            },
        )

    def ingest_book(
        self,
        outcome: str,
        snapshot: PaperBookSnapshot,
        *,
        causal_segment_sha256: str,
    ) -> None:
        selected = str(outcome or "").strip().lower()
        if selected not in self._books:
            raise ValueError("Round 17 book outcome is invalid")
        self._validate_causal_segment(causal_segment_sha256)
        self._validate_observation_time(snapshot.received_wall_ms)
        self._books[selected].append(snapshot)

    def _assert_no_future_input(self, decision_time_ms: int) -> None:
        latest = (
            self._chainlink.latest_received_ms,
            *(series.latest_received_ms for series in self._binance.values()),
            *(series.latest_received_ms for series in self._books.values()),
        )
        if any(value is not None and value > decision_time_ms for value in latest):
            raise ValueError("Round 17 accumulator contains future receipt evidence")

    def build(self, base: PolymarketRound14FeatureRow) -> PolymarketRound17FeatureRow:
        if not isinstance(base, PolymarketRound14FeatureRow):
            raise TypeError("Round 17 base feature type differs")
        decision = int(base.decision_time_ms)
        if (
            base.condition_id != self.condition_id
            or base.feature_names_sha256 != POLYMARKET_ROUND14_FEATURE_NAMES_SHA256
            or not self.event_start_ms <= decision < self.event_end_ms
            or (decision - self.event_start_ms) % 250
            or decision < self._last_decision_ms
        ):
            raise ValueError("Round 17 base feature identity or chronology differs")
        self._assert_no_future_input(decision)
        temporal: list[float] = []
        for window_ms in POLYMARKET_ROUND17_CHAINLINK_WINDOWS_MS:
            item = self._chainlink.window(
                decision_time_ms=decision,
                window_ms=window_ms,
            )
            temporal.extend(
                (
                    item.log_return,
                    item.realized_variance,
                    item.bipower_variation,
                    item.jump_fraction,
                    float(item.observation_count),
                )
            )
        binance_windows: dict[str, list[_PriceWindow]] = {}
        for market_name in ("spot", "perpetual"):
            market_windows: list[_PriceWindow] = []
            for window_ms in POLYMARKET_ROUND17_BINANCE_WINDOWS_MS:
                item = self._binance[market_name].window(
                    decision_time_ms=decision,
                    window_ms=window_ms,
                )
                market_windows.append(item)
                temporal.extend(
                    (
                        item.log_return,
                        item.realized_variance,
                        item.signed_quote_imbalance,
                        math.log1p(item.quote_notional),
                        float(item.observation_count),
                    )
                )
            binance_windows[market_name] = market_windows
        for outcome in ("up", "down"):
            for window_ms in POLYMARKET_ROUND17_BOOK_WINDOWS_MS:
                item = self._books[outcome].window(
                    decision_time_ms=decision,
                    window_ms=window_ms,
                )
                temporal.extend(
                    (
                        item.top_of_book_order_flow_imbalance,
                        item.mean_top_imbalance,
                        item.microprice_log_return,
                        item.level_quantity_flow_pressure,
                        math.log1p(item.gross_level_quantity_change),
                        float(item.update_count),
                    )
                )
        temporal.extend(
            spot.log_return - perpetual.log_return
            for spot, perpetual in zip(
                binance_windows["spot"],
                binance_windows["perpetual"],
                strict=True,
            )
        )
        values = tuple((*base.values, *temporal))
        values_sha256 = _canonical_sha256(list(values))
        input_payload = {
            "schema_version": POLYMARKET_ROUND17_FEATURE_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
            "condition_id": self.condition_id,
            "market_id": self.market_id,
            "up_token_id": self.up_token_id,
            "down_token_id": self.down_token_id,
            "event_start_ms": self.event_start_ms,
            "event_end_ms": self.event_end_ms,
            "decision_time_ms": decision,
            "capture_run_id": self.capture_run_id,
            "admission_sha256": self.admission_sha256,
            "binance_layer_eligible": self.binance_layer_eligible,
            "causal_segment_sha256": self.causal_segment_sha256,
            "base_input_sha256": base.input_sha256,
            "feature_names_sha256": POLYMARKET_ROUND17_FEATURE_NAMES_SHA256,
            "values_sha256": values_sha256,
            "source_chain_sha256": {
                "chainlink": self._chainlink.chain_sha256,
                "binance_spot": self._binance["spot"].chain_sha256,
                "binance_perpetual": self._binance["perpetual"].chain_sha256,
                "polymarket_up": self._books["up"].chain_sha256,
                "polymarket_down": self._books["down"].chain_sha256,
            },
            "source_observation_counts": {
                "chainlink": len(self._chainlink.received_ms),
                "binance_spot": len(self._binance["spot"].received_ms),
                "binance_perpetual": len(self._binance["perpetual"].received_ms),
                "polymarket_up": len(self._books["up"].received_ms),
                "polymarket_down": len(self._books["down"].received_ms),
            },
        }
        self._last_decision_ms = decision
        return PolymarketRound17FeatureRow(
            condition_id=self.condition_id,
            decision_time_ms=decision,
            admission_sha256=self.admission_sha256,
            causal_segment_sha256=self.causal_segment_sha256,
            feature_names_sha256=POLYMARKET_ROUND17_FEATURE_NAMES_SHA256,
            input_sha256=_canonical_sha256(input_payload),
            values_sha256=values_sha256,
            values=values,
        )


__all__ = [
    "POLYMARKET_ROUND17_BINANCE_WINDOWS_MS",
    "POLYMARKET_ROUND17_BOOK_WINDOWS_MS",
    "POLYMARKET_ROUND17_CHAINLINK_WINDOWS_MS",
    "POLYMARKET_ROUND17_CONTRACT_SHA256",
    "POLYMARKET_ROUND17_FEATURE_NAMES",
    "POLYMARKET_ROUND17_FEATURE_NAMES_SHA256",
    "POLYMARKET_ROUND17_FEATURE_SCHEMA_VERSION",
    "POLYMARKET_ROUND17_TEMPORAL_FEATURE_NAMES",
    "PolymarketRound17BinanceTrade",
    "PolymarketRound17FeatureAccumulator",
    "PolymarketRound17FeatureRow",
]
