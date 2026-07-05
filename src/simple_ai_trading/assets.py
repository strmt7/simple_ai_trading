"""Asset-universe helpers shared by trading, risk, and UI surfaces."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

DEFAULT_QUOTE_ASSET = "USDC"
DEFAULT_SYMBOL = "BTCUSDC"
SUPPORTED_MAJOR_BASE_ASSETS = ("BTC", "ETH", "SOL")
SUPPORTED_MAJOR_QUOTE_ASSETS = ("USDC", "USDT")
DEFAULT_SYMBOLS = ("BTCUSDC", "ETHUSDC", "SOLUSDC")
DEFAULT_MIN_DIVERSIFIED_ASSETS = 3
MAX_AUTONOMOUS_LEVERAGE = 20.0
DEFAULT_CONSERVATIVE_LEVERAGE = 5.0
DEFAULT_REGULAR_LEVERAGE = 10.0
DEFAULT_AGGRESSIVE_LEVERAGE = 15.0
DEFAULT_LEVERAGE_BY_RISK_LEVEL = {
    "conservative": DEFAULT_CONSERVATIVE_LEVERAGE,
    "regular": DEFAULT_REGULAR_LEVERAGE,
    "aggressive": DEFAULT_AGGRESSIVE_LEVERAGE,
}

_SYMBOL_RE = re.compile(r"^[A-Z0-9]{3,30}$")


def normalize_symbol(value: object, *, default: str = DEFAULT_SYMBOL) -> str:
    """Return an uppercase exchange symbol or the default when malformed."""

    candidate = str(value or "").strip().upper().replace("-", "").replace("/", "")
    if not candidate or not _SYMBOL_RE.fullmatch(candidate):
        return default
    return candidate


def symbol_base_for_supported_quote(symbol: object, quote_asset: object | None = None) -> str:
    """Return the base asset when the symbol uses a supported quote asset."""

    normalized = normalize_symbol(symbol, default="")
    if not normalized:
        return ""
    if quote_asset is not None:
        quote = str(quote_asset or "").strip().upper()
        if quote not in SUPPORTED_MAJOR_QUOTE_ASSETS or not normalized.endswith(quote):
            return ""
        return normalized[: -len(quote)]
    for quote in sorted(SUPPORTED_MAJOR_QUOTE_ASSETS, key=len, reverse=True):
        if normalized.endswith(quote):
            return normalized[: -len(quote)]
    return ""


def is_supported_major_symbol(symbol: object, quote_asset: object | None = None) -> bool:
    """Return True only for BTC, ETH, or SOL quoted in USDC/USDT."""

    base = symbol_base_for_supported_quote(symbol, quote_asset=quote_asset)
    return base in SUPPORTED_MAJOR_BASE_ASSETS


def major_symbols_for_quote(quote_asset: object = DEFAULT_QUOTE_ASSET) -> tuple[str, ...]:
    """Return the supported BTC/ETH/SOL symbols for a quote asset."""

    quote = str(quote_asset or DEFAULT_QUOTE_ASSET).strip().upper()
    if quote not in SUPPORTED_MAJOR_QUOTE_ASSETS:
        quote = DEFAULT_QUOTE_ASSET
    return tuple(f"{base}{quote}" for base in SUPPORTED_MAJOR_BASE_ASSETS)


def normalize_symbols(values: object, *, default: Iterable[str] = DEFAULT_SYMBOLS) -> tuple[str, ...]:
    """Return a stable, de-duplicated symbol tuple."""

    if isinstance(values, str):
        raw_values: Iterable[object] = [part for part in values.split(",") if part.strip()]
    elif isinstance(values, Iterable):
        raw_values = values
    else:
        raw_values = ()

    seen: set[str] = set()
    normalized: list[str] = []
    for value in raw_values:
        symbol = normalize_symbol(value, default="")
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)

    if not normalized:
        for value in default:
            symbol = normalize_symbol(value)
            if symbol not in seen:
                seen.add(symbol)
                normalized.append(symbol)
    return tuple(normalized)


def default_leverage_for_risk_level(risk_level: object) -> float:
    """Return the shared futures leverage default for a configured risk level."""

    key = str(risk_level or "conservative").strip().lower()
    return float(DEFAULT_LEVERAGE_BY_RISK_LEVEL.get(key, DEFAULT_CONSERVATIVE_LEVERAGE))


def base_asset(symbol: str, quote_asset: str = DEFAULT_QUOTE_ASSET) -> str:
    normalized_symbol = normalize_symbol(symbol)
    quote = str(quote_asset or DEFAULT_QUOTE_ASSET).strip().upper()
    if quote and normalized_symbol.endswith(quote):
        return normalized_symbol[: -len(quote)]
    return normalized_symbol


def default_history_path(symbol: str = DEFAULT_SYMBOL) -> str:
    return f"data/historical_{normalize_symbol(symbol).lower()}.json"


def safe_symbol_stem(symbol: str) -> str:
    return normalize_symbol(symbol).lower()


@dataclass(frozen=True)
class AssetAllocation:
    symbol: str
    weight: float

    def asdict(self) -> dict[str, object]:
        return {"symbol": self.symbol, "weight": self.weight}


def equal_weight_allocations(symbols: Iterable[str]) -> tuple[AssetAllocation, ...]:
    normalized = normalize_symbols(tuple(symbols))
    weight = 1.0 / len(normalized) if normalized else 0.0
    return tuple(AssetAllocation(symbol=symbol, weight=weight) for symbol in normalized)
