"""Automatic market eligibility and diversification checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

from .api import BinanceAPIError, BinanceClient
from .assets import DEFAULT_QUOTE_ASSET, DEFAULT_SYMBOL, normalize_symbol, normalize_symbols
from .execution_simulation import SymbolExecutionProfile
from .types import StrategyConfig

_DANGEROUS_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


@dataclass(frozen=True)
class MarketEligibility:
    symbol: str
    eligible: bool
    status: str
    quote_volume: float
    trade_count: int
    spread_bps: float
    liquidity_score: float
    reasons: tuple[str, ...]

    def asdict(self) -> dict[str, object]:
        return asdict(self)

    def execution_profile(self, *, latency_ms: int, liquidity_haircut: float) -> SymbolExecutionProfile:
        return SymbolExecutionProfile(
            symbol=self.symbol,
            spread_bps=self.spread_bps,
            quote_volume=self.quote_volume,
            trade_count=self.trade_count,
            liquidity_score=self.liquidity_score,
            latency_ms=latency_ms,
            liquidity_haircut=liquidity_haircut,
        )


@dataclass(frozen=True)
class UniverseSelection:
    quote_asset: str
    requested: tuple[str, ...]
    eligible: tuple[MarketEligibility, ...]
    rejected: tuple[MarketEligibility, ...]
    min_required: int

    @property
    def allowed(self) -> bool:
        return len(self.eligible) >= self.min_required

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(item.symbol for item in self.eligible)

    def execution_profiles(self, strategy: StrategyConfig) -> tuple[SymbolExecutionProfile, ...]:
        return tuple(
            item.execution_profile(
                latency_ms=strategy.latency_buffer_ms,
                liquidity_haircut=strategy.testnet_liquidity_haircut,
            )
            for item in self.eligible
        )

    def asdict(self) -> dict[str, object]:
        return {
            "quote_asset": self.quote_asset,
            "requested": list(self.requested),
            "eligible": [item.asdict() for item in self.eligible],
            "rejected": [item.asdict() for item in self.rejected],
            "min_required": self.min_required,
            "allowed": self.allowed,
        }


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if parsed == parsed and parsed not in {float("inf"), float("-inf")} else default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _base_symbol(symbol: str, quote_asset: str) -> str:
    return symbol[: -len(quote_asset)] if quote_asset and symbol.endswith(quote_asset) else symbol


def _looks_structurally_dangerous(symbol: str, quote_asset: str) -> bool:
    base = _base_symbol(symbol, quote_asset)
    return any(base.endswith(suffix) for suffix in _DANGEROUS_SUFFIXES)


def _spread_bps(book: Mapping[str, object]) -> float:
    bid = _safe_float(book.get("bidPrice"))
    ask = _safe_float(book.get("askPrice"))
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
    if mid <= 0:
        return float("inf")
    return max(0.0, ((ask - bid) / mid) * 10_000.0)


def _score_liquidity(*, quote_volume: float, trade_count: int, spread_bps: float, strategy: StrategyConfig) -> float:
    volume_score = min(1.0, quote_volume / max(1.0, strategy.min_quote_volume_usdc))
    trade_score = min(1.0, trade_count / max(1, strategy.min_trade_count_24h))
    spread_score = 1.0 if spread_bps <= 0 else min(1.0, max(0.0, strategy.max_spread_bps / spread_bps))
    return max(0.0, min(1.0, 0.50 * volume_score + 0.30 * trade_score + 0.20 * spread_score))


def _exchange_symbol_map(client: BinanceClient) -> dict[str, Mapping[str, object]]:
    info = client.get_exchange_info()
    symbols = info.get("symbols") if isinstance(info, Mapping) else None
    if not isinstance(symbols, list):
        raise BinanceAPIError("Unexpected exchangeInfo symbols payload")
    return {
        str(item.get("symbol") or "").upper(): item
        for item in symbols
        if isinstance(item, Mapping) and item.get("symbol")
    }


def assess_symbol_liquidity(
    client: BinanceClient,
    symbol: str,
    strategy: StrategyConfig,
    *,
    quote_asset: str = DEFAULT_QUOTE_ASSET,
    exchange_symbols: Mapping[str, Mapping[str, object]] | None = None,
) -> MarketEligibility:
    """Measure one symbol and fail closed when liquidity cannot be proven."""

    normalized = normalize_symbol(symbol, default=DEFAULT_SYMBOL)
    quote_asset = str(quote_asset or DEFAULT_QUOTE_ASSET).upper()
    reasons: list[str] = []
    exchange_symbols = exchange_symbols or _exchange_symbol_map(client)
    symbol_info = exchange_symbols.get(normalized)
    status = str(symbol_info.get("status") if symbol_info else "MISSING")
    if not symbol_info:
        reasons.append("missing_exchange_info")
    elif status != "TRADING":
        reasons.append(f"status_{status.lower()}")
    if not normalized.endswith(quote_asset):
        reasons.append(f"quote_not_{quote_asset}")
    if _looks_structurally_dangerous(normalized, quote_asset):
        reasons.append("leveraged_or_inverse_token_pattern")

    quote_volume = 0.0
    trade_count = 0
    spread_bps = float("inf")
    try:
        ticker = client.get_ticker_24h(normalized)
        book = client.get_book_ticker(normalized)
        quote_volume = _safe_float(ticker.get("quoteVolume"))
        trade_count = _safe_int(ticker.get("count"))
        spread_bps = _spread_bps(book)
    except BinanceAPIError as exc:
        reasons.append(f"market_data_unavailable:{exc}")

    if quote_volume < strategy.min_quote_volume_usdc:
        reasons.append("quote_volume_below_threshold")
    if trade_count < strategy.min_trade_count_24h:
        reasons.append("trade_count_below_threshold")
    if spread_bps > strategy.max_spread_bps:
        reasons.append("spread_above_threshold")

    liquidity_score = _score_liquidity(
        quote_volume=quote_volume,
        trade_count=trade_count,
        spread_bps=spread_bps,
        strategy=strategy,
    )
    if liquidity_score < strategy.min_liquidity_score:
        reasons.append("liquidity_score_below_threshold")

    return MarketEligibility(
        symbol=normalized,
        eligible=not reasons,
        status=status,
        quote_volume=quote_volume,
        trade_count=trade_count,
        spread_bps=spread_bps,
        liquidity_score=liquidity_score,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def select_tradeable_universe(
    client: BinanceClient,
    requested_symbols: Iterable[str],
    strategy: StrategyConfig,
    *,
    quote_asset: str = DEFAULT_QUOTE_ASSET,
) -> UniverseSelection:
    """Return measured eligible symbols; no precompiled allowlist is used."""

    requested = normalize_symbols(requested_symbols)
    exchange_symbols = _exchange_symbol_map(client)
    assessed = [
        assess_symbol_liquidity(
            client,
            symbol,
            strategy,
            quote_asset=quote_asset,
            exchange_symbols=exchange_symbols,
        )
        for symbol in requested
    ]
    eligible = tuple(item for item in assessed if item.eligible)
    rejected = tuple(item for item in assessed if not item.eligible)
    return UniverseSelection(
        quote_asset=quote_asset,
        requested=requested,
        eligible=eligible,
        rejected=rejected,
        min_required=max(1, strategy.min_diversified_assets),
    )


def rank_high_liquidity_universe(
    client: BinanceClient,
    strategy: StrategyConfig,
    *,
    quote_asset: str = DEFAULT_QUOTE_ASSET,
    max_symbols: int = 12,
    max_scan: int = 250,
) -> UniverseSelection:
    """Automatically rank exchange symbols by live liquidity without an allowlist."""

    quote_asset = str(quote_asset or DEFAULT_QUOTE_ASSET).upper()
    exchange_symbols = _exchange_symbol_map(client)
    tickers = {
        str(item.get("symbol") or "").upper(): item
        for item in client.get_all_tickers_24h()
        if isinstance(item, Mapping) and item.get("symbol")
    }
    books = {
        str(item.get("symbol") or "").upper(): item
        for item in client.get_all_book_tickers()
        if isinstance(item, Mapping) and item.get("symbol")
    }
    candidates: list[tuple[float, str, MarketEligibility]] = []
    rejected: list[MarketEligibility] = []
    for symbol, symbol_info in exchange_symbols.items():
        if not symbol.endswith(quote_asset):
            continue
        ticker = tickers.get(symbol)
        if ticker is None:
            continue
        quote_volume = _safe_float(ticker.get("quoteVolume"))
        trade_count = _safe_int(ticker.get("count"))
        rank = quote_volume + (trade_count * 1000.0)
        status = str(symbol_info.get("status") if symbol_info else "MISSING")
        reasons: list[str] = []
        if status != "TRADING":
            reasons.append(f"status_{status.lower()}")
        if _looks_structurally_dangerous(symbol, quote_asset):
            reasons.append("leveraged_or_inverse_token_pattern")
        spread_bps = _spread_bps(books.get(symbol, {}))
        if quote_volume < strategy.min_quote_volume_usdc:
            reasons.append("quote_volume_below_threshold")
        if trade_count < strategy.min_trade_count_24h:
            reasons.append("trade_count_below_threshold")
        if spread_bps > strategy.max_spread_bps:
            reasons.append("spread_above_threshold")
        liquidity_score = _score_liquidity(
            quote_volume=quote_volume,
            trade_count=trade_count,
            spread_bps=spread_bps,
            strategy=strategy,
        )
        if liquidity_score < strategy.min_liquidity_score:
            reasons.append("liquidity_score_below_threshold")
        item = MarketEligibility(
            symbol=symbol,
            eligible=not reasons,
            status=status,
            quote_volume=quote_volume,
            trade_count=trade_count,
            spread_bps=spread_bps,
            liquidity_score=liquidity_score,
            reasons=tuple(dict.fromkeys(reasons)),
        )
        if item.eligible:
            candidates.append((rank, symbol, item))
        else:
            rejected.append(item)
    candidates.sort(key=lambda entry: (entry[0], entry[2].liquidity_score), reverse=True)
    rejected.sort(key=lambda item: (item.quote_volume, item.trade_count), reverse=True)
    selected = tuple(item for _rank, _symbol, item in candidates[: max(1, int(max_symbols))])
    requested = tuple(item.symbol for _rank, _symbol, item in candidates[: max(1, int(max_scan))])
    return UniverseSelection(
        quote_asset=quote_asset,
        requested=requested or tuple(item.symbol for item in selected),
        eligible=selected,
        rejected=tuple(rejected[: max(0, int(max_scan))]),
        min_required=max(1, min(strategy.min_diversified_assets, int(max_symbols))),
    )
