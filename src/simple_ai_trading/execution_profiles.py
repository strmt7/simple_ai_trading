"""Load symbol-specific execution assumptions from persisted market evidence."""

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .execution_simulation import SymbolExecutionProfile
from .market_store import MarketDataStore, TopOfBookSnapshot
from .types import StrategyConfig


_STALE_TOP_OF_BOOK_MS = 24 * 60 * 60 * 1000


@dataclass(frozen=True)
class ExecutionProfileEvidence:
    """Audit payload for a profile loaded from market data."""

    profile: SymbolExecutionProfile | None
    source: str
    db_path: str | None = None
    snapshot_ts_ms: int | None = None
    snapshot_age_ms: int | None = None
    spread_bps: float | None = None
    depth_notional: float | None = None
    warning: str | None = None

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["profile"] = self.profile.asdict() if self.profile is not None else None
        return payload


def execution_profile_from_top_of_book(
    snapshot: TopOfBookSnapshot,
    strategy: StrategyConfig,
) -> SymbolExecutionProfile:
    """Convert a typed top-of-book row into pessimistic fill assumptions."""

    spread_bps = max(0.0, _finite(snapshot.spread_bps, 0.0))
    depth_notional = max(0.0, _finite(snapshot.depth_notional, 0.0))
    max_spread_bps = max(0.1, float(strategy.max_spread_bps))
    target_depth = _target_top_of_book_depth(strategy)
    spread_score = 1.0 - min(1.0, spread_bps / max_spread_bps)
    depth_score = min(1.0, depth_notional / target_depth) if target_depth > 0.0 else 0.0
    liquidity_score = _clamp(0.35 * spread_score + 0.65 * depth_score, 0.0, 1.0)
    return SymbolExecutionProfile(
        symbol=snapshot.symbol.upper(),
        spread_bps=spread_bps,
        quote_volume=depth_notional,
        trade_count=0,
        liquidity_score=liquidity_score,
        latency_ms=max(0, int(strategy.latency_buffer_ms)),
        liquidity_haircut=_clamp(float(strategy.testnet_liquidity_haircut), 0.0, 1.0),
    )


def load_top_of_book_execution_profile(
    db_path: str | Path | None,
    *,
    symbol: str,
    market_type: str,
    strategy: StrategyConfig,
    now_ms: int | None = None,
) -> ExecutionProfileEvidence:
    """Load the latest top-of-book profile for ``symbol`` from SQLite.

    Missing data is reported as evidence instead of silently creating a new
    empty database. Existing file-based backtests can therefore remain fast,
    while operators who pass an execution DB get explicit realism metadata.
    """

    if not db_path:
        return ExecutionProfileEvidence(profile=None, source="disabled")
    path = Path(db_path)
    if not path.exists():
        return ExecutionProfileEvidence(
            profile=None,
            source="top_of_book",
            db_path=str(path),
            warning=f"execution DB not found: {path}",
        )
    try:
        with MarketDataStore(path) as store:
            snapshot = store.latest_top_of_book(symbol, market_type)
    except (OSError, sqlite3.Error, ValueError) as exc:
        return ExecutionProfileEvidence(
            profile=None,
            source="top_of_book",
            db_path=str(path),
            warning=f"execution DB unreadable: {exc}",
        )
    if snapshot is None:
        return ExecutionProfileEvidence(
            profile=None,
            source="top_of_book",
            db_path=str(path),
            warning=f"no top-of-book snapshot for {symbol.upper()} {market_type}",
        )
    profile = execution_profile_from_top_of_book(snapshot, strategy)
    current_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    age_ms = max(0, current_ms - int(snapshot.ts_ms))
    warning = None
    if age_ms > _STALE_TOP_OF_BOOK_MS:
        warning = f"top-of-book snapshot is stale: age_ms={age_ms}"
    return ExecutionProfileEvidence(
        profile=profile,
        source=f"top_of_book:{snapshot.provider}",
        db_path=str(path),
        snapshot_ts_ms=int(snapshot.ts_ms),
        snapshot_age_ms=age_ms,
        spread_bps=float(snapshot.spread_bps),
        depth_notional=float(snapshot.depth_notional),
        warning=warning,
    )


def _target_top_of_book_depth(strategy: StrategyConfig) -> float:
    # 24h quote volume and L1 depth are different quantities. Using 0.1% of the
    # configured 24h floor, capped, gives liquid pairs credit without pretending
    # a single top level must contain the whole daily-volume threshold.
    return max(10_000.0, min(250_000.0, float(strategy.min_quote_volume_usdc) * 0.001))


def _finite(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


__all__ = [
    "ExecutionProfileEvidence",
    "execution_profile_from_top_of_book",
    "load_top_of_book_execution_profile",
]
