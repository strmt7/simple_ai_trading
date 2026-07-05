"""Exchange/account reconciliation against the local autonomous position ledger."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

from .positions import OpenPosition, PositionsStore, bot_ownership_rejection_reason
from .types import RuntimeConfig


@dataclass(frozen=True)
class ExchangeExposure:
    symbol: str
    market_type: str
    side: str
    qty: float
    entry_price: float
    notional: float
    source: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReconciliationMismatch:
    symbol: str
    side: str
    local_qty: float
    exchange_qty: float
    difference: float
    reason: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReconciliationReport:
    ok: bool
    market_type: str
    symbols_checked: list[str]
    local_open_count: int
    local_live_open_count: int
    local_paper_open_count: int
    exchange_exposure_count: int
    mismatches: list[ReconciliationMismatch] = field(default_factory=list)
    exchange_exposures: list[ExchangeExposure] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    external_exchange_exposure_count: int = 0
    stale_local_position_count: int = 0
    unverified_local_position_count: int = 0
    invalid_account_payload_count: int = 0
    ledger_integrity_error_count: int = 0

    def asdict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "market_type": self.market_type,
            "symbols_checked": list(self.symbols_checked),
            "local_open_count": self.local_open_count,
            "local_live_open_count": self.local_live_open_count,
            "local_paper_open_count": self.local_paper_open_count,
            "exchange_exposure_count": self.exchange_exposure_count,
            "mismatches": [item.asdict() for item in self.mismatches],
            "exchange_exposures": [item.asdict() for item in self.exchange_exposures],
            "warnings": list(self.warnings),
            "external_exchange_exposure_count": self.external_exchange_exposure_count,
            "stale_local_position_count": self.stale_local_position_count,
            "unverified_local_position_count": self.unverified_local_position_count,
            "invalid_account_payload_count": self.invalid_account_payload_count,
            "ledger_integrity_error_count": self.ledger_integrity_error_count,
        }


def _finite(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _symbol_set(runtime: RuntimeConfig, local_positions: Sequence[OpenPosition]) -> set[str]:
    symbols = {str(runtime.symbol).upper()}
    symbols.update(str(symbol).upper() for symbol in runtime.symbols)
    symbols.update(str(position.symbol).upper() for position in local_positions)
    return {symbol for symbol in symbols if symbol}


def _base_asset(symbol: str, quote_asset: str) -> str:
    return symbol[: -len(quote_asset)] if quote_asset and symbol.endswith(quote_asset) else symbol


def _exposures_from_futures_account(
    account: Mapping[str, object],
    symbols: set[str],
) -> list[ExchangeExposure]:
    positions = account.get("positions")
    exposures: list[ExchangeExposure] = []
    if not isinstance(positions, list):
        return exposures
    for item in positions:
        if not isinstance(item, Mapping):
            continue
        symbol = str(item.get("symbol") or "").upper()
        if symbol not in symbols:
            continue
        amount = _finite(item.get("positionAmt"))
        if abs(amount) <= 1e-10:
            continue
        side = "LONG" if amount > 0 else "SHORT"
        qty = abs(amount)
        entry_price = max(0.0, _finite(item.get("entryPrice")))
        notional = abs(_finite(item.get("notional"), qty * entry_price))
        exposures.append(ExchangeExposure(
            symbol=symbol,
            market_type="futures",
            side=side,
            qty=qty,
            entry_price=entry_price,
            notional=notional,
            source="futures_position",
        ))
    return exposures


def _exposures_from_spot_account(
    account: Mapping[str, object],
    symbols: set[str],
    quote_asset: str,
) -> list[ExchangeExposure]:
    balances = account.get("balances")
    exposures: list[ExchangeExposure] = []
    if not isinstance(balances, list):
        return exposures
    asset_to_symbol = {
        _base_asset(symbol, quote_asset): symbol
        for symbol in symbols
        if symbol.endswith(quote_asset)
    }
    for item in balances:
        if not isinstance(item, Mapping):
            continue
        asset = str(item.get("asset") or "").upper()
        if asset == quote_asset or asset not in asset_to_symbol:
            continue
        qty = max(0.0, _finite(item.get("free")) + _finite(item.get("locked")))
        if qty <= 1e-10:
            continue
        exposures.append(ExchangeExposure(
            symbol=asset_to_symbol[asset],
            market_type="spot",
            side="LONG",
            qty=qty,
            entry_price=0.0,
            notional=0.0,
            source="spot_balance",
        ))
    return exposures


def exchange_exposures_from_account(
    account: Mapping[str, object],
    runtime: RuntimeConfig,
    local_positions: Sequence[OpenPosition],
) -> list[ExchangeExposure]:
    symbols = _symbol_set(runtime, local_positions)
    if runtime.market_type == "futures":
        return _exposures_from_futures_account(account, symbols)
    return _exposures_from_spot_account(account, symbols, runtime.quote_asset)


def _aggregate_local(positions: Sequence[OpenPosition]) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], float] = {}
    for position in positions:
        key = (str(position.symbol).upper(), str(position.side).upper())
        totals[key] = totals.get(key, 0.0) + max(0.0, float(position.qty))
    return totals


def _aggregate_exchange(exposures: Sequence[ExchangeExposure]) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], float] = {}
    for exposure in exposures:
        key = (str(exposure.symbol).upper(), str(exposure.side).upper())
        totals[key] = totals.get(key, 0.0) + max(0.0, float(exposure.qty))
    return totals


def _account_payload_rejection(account: object, runtime: RuntimeConfig) -> str | None:
    if not isinstance(account, Mapping):
        return "account_payload_not_mapping"
    if runtime.market_type == "futures":
        return None if isinstance(account.get("positions"), list) else "futures_positions_missing_or_not_list"
    return None if isinstance(account.get("balances"), list) else "spot_balances_missing_or_not_list"


def reconcile_account_positions(
    account: object,
    runtime: RuntimeConfig,
    store: PositionsStore,
    *,
    quantity_tolerance: float = 1e-8,
) -> ReconciliationReport:
    ledger_integrity_errors = store.open_integrity_errors()
    open_positions = [] if ledger_integrity_errors else store.load_open()
    live_positions = [position for position in open_positions if not position.dry_run]
    paper_positions = [position for position in open_positions if position.dry_run]
    unverified_live_positions = [
        (position, reason)
        for position in live_positions
        if (reason := bot_ownership_rejection_reason(position)) is not None
    ]
    verified_live_positions = [
        position
        for position in live_positions
        if bot_ownership_rejection_reason(position) is None
    ]
    account_rejection = _account_payload_rejection(account, runtime)
    account_mapping: Mapping[str, object] = account if isinstance(account, Mapping) else {}
    exposures = exchange_exposures_from_account(account_mapping, runtime, open_positions)
    local = _aggregate_local(verified_live_positions)
    exchange = _aggregate_exchange(exposures)
    keys = sorted(set(local) | set(exchange))
    mismatches: list[ReconciliationMismatch] = []
    for error in ledger_integrity_errors:
        mismatches.append(ReconciliationMismatch(
            symbol=str(runtime.symbol).upper(),
            side="UNKNOWN",
            local_qty=0.0,
            exchange_qty=0.0,
            difference=0.0,
            reason=f"local_ledger_integrity_failed:{error}",
        ))
    if account_rejection is not None:
        mismatches.append(ReconciliationMismatch(
            symbol=str(runtime.symbol).upper(),
            side="UNKNOWN",
            local_qty=0.0,
            exchange_qty=0.0,
            difference=0.0,
            reason=f"account_payload_invalid:{account_rejection}",
        ))
    for symbol, side in keys:
        local_qty = local.get((symbol, side), 0.0)
        exchange_qty = exchange.get((symbol, side), 0.0)
        difference = exchange_qty - local_qty
        if abs(difference) <= max(0.0, float(quantity_tolerance)):
            continue
        if local_qty <= 0.0:
            reason = "exchange_exposure_without_local_position"
        elif exchange_qty <= 0.0:
            reason = "local_position_without_exchange_exposure"
        else:
            reason = "quantity_mismatch"
        mismatches.append(ReconciliationMismatch(
            symbol=symbol,
            side=side,
            local_qty=local_qty,
            exchange_qty=exchange_qty,
            difference=difference,
            reason=reason,
        ))
    for position, rejection in unverified_live_positions:
        symbol = str(position.symbol).upper()
        side = str(position.side).upper()
        local_qty = max(0.0, float(position.qty))
        exchange_qty = exchange.get((symbol, side), 0.0)
        mismatches.append(ReconciliationMismatch(
            symbol=symbol,
            side=side,
            local_qty=local_qty,
            exchange_qty=exchange_qty,
            difference=exchange_qty - local_qty,
            reason=f"local_position_without_bot_ownership:{rejection}",
        ))
    warnings: list[str] = []
    if paper_positions:
        warnings.append(f"paper_positions_ignored={len(paper_positions)}")
    warnings.extend(
        f"local_ledger_integrity_failed:{error}"
        for error in ledger_integrity_errors
    )
    if account_rejection is not None:
        warnings.append(account_rejection)
    external_exposure_count = sum(
        1
        for mismatch in mismatches
        if mismatch.reason == "exchange_exposure_without_local_position"
    )
    stale_local_count = sum(
        1
        for mismatch in mismatches
        if mismatch.reason == "local_position_without_exchange_exposure"
    ) + len(unverified_live_positions)
    symbols_checked = sorted(_symbol_set(runtime, open_positions))
    return ReconciliationReport(
        ok=not mismatches,
        market_type=runtime.market_type,
        symbols_checked=symbols_checked,
        local_open_count=len(open_positions),
        local_live_open_count=len(live_positions),
        local_paper_open_count=len(paper_positions),
        exchange_exposure_count=len(exposures),
        mismatches=mismatches,
        exchange_exposures=list(exposures),
        warnings=warnings,
        external_exchange_exposure_count=external_exposure_count,
        stale_local_position_count=stale_local_count,
        unverified_local_position_count=len(unverified_live_positions),
        invalid_account_payload_count=1 if account_rejection is not None else 0,
        ledger_integrity_error_count=len(ledger_integrity_errors),
    )
