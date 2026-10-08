"""Forward, supplied-price inventory replay; never a venue-profit admission."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re
from typing import Mapping

import numpy as np

from .cross_asset_cost_data import MINUTE_MS, SYMBOLS, MinuteSeries
from .derivatives_hurdle_data import FundingState
from .funding_cash_inventory import (
    InventoryCashPath,
    InventoryCashReplay,
    replay_fixed_base_inventory,
)
from .funding_cash_labels import FundingCashLabelSeries, bind_funding_cash_panel
from .stateful_position_policy import stateful_position_schedule
from .stateful_turnover_model import (
    COST_FILTER_LAMBDA,
    HORIZON_MINUTES,
    MAXIMUM_HOLDING_HOURS,
    StatefulHourlyDataset,
    _combined_evaluation_mask,
)


@dataclass(frozen=True)
class StatefulInventoryReplay:
    decision_time_ms: tuple[int, ...]
    positions: tuple[tuple[int, ...], ...]
    symbol_ledgers: tuple[InventoryCashReplay, ...]
    initial_quote_capital: Fraction
    portfolio_interval_cash_lower: tuple[Fraction, ...]
    portfolio_interval_cash_upper: tuple[Fraction, ...]
    quote_equity_lower: tuple[Fraction, ...]
    quote_equity_upper: tuple[Fraction, ...]
    total_return_lower: Fraction
    total_return_upper: Fraction
    modeled_capital_exhausted: bool
    replay_input_sha256: str
    financially_qualified: bool = False


def _boundary_prices(series: MinuteSeries, clocks: np.ndarray) -> tuple[Fraction, ...]:
    times, prices = np.asarray(series.open_time_ms), np.asarray(series.open)
    if (
        times.ndim != 1
        or not times.size
        or times.dtype.kind not in "iu"
        or prices.shape != times.shape
        or prices.dtype.kind not in "fiu"
        or np.any(times < 0)
        or np.any(times >= 2**63)
        or np.any(times[1:] <= times[:-1])
    ):
        raise ValueError("inventory price clock contract is invalid")
    indexes = np.searchsorted(times, clocks)
    if np.any(indexes >= times.size) or not np.array_equal(times[indexes], clocks):
        raise ValueError(
            "inventory price boundaries are missing; interpolation is forbidden"
        )
    selected = prices[indexes].astype(np.float64)
    if not np.isfinite(selected).all() or np.any(selected <= 0):
        raise ValueError("inventory boundary price is invalid")
    # Preserve supplied binary float values; do not invent native decimal bytes.
    return tuple(Fraction.from_float(float(price)) for price in selected)


def replay_stateful_fixed_base_cash(
    dataset: StatefulHourlyDataset,
    predictions: np.ndarray,
    *,
    price_panel: Mapping[str, MinuteSeries],
    funding: Mapping[str, FundingState],
    funding_cash: Mapping[str, FundingCashLabelSeries],
    initial_quote_capital: Fraction,
    one_way_cost_fraction: Fraction,
    mode: str,
    price_source_sha256: str,
) -> StatefulInventoryReplay:
    """Reuse forecast decisions with exact fixed-base USDT inventory cash.

    Entry is the supplied open one minute after decision; subsequent boundaries
    are hourly opens. This clock/price/cost model is conditional, not fills or
    guaranteed entitlement. Every direction change opens the original per-symbol
    quote budget, not compounded equity. Intrabar margin, liquidation, hedges,
    capital financing and source authenticity require independent qualification.
    Historical scalar targets are not used to decide or compute this cash path.
    """
    if (
        set(price_panel) != set(SYMBOLS)
        or set(funding) != set(SYMBOLS)
        or funding_cash is None
        or set(funding_cash) != set(SYMBOLS)
        or not isinstance(initial_quote_capital, Fraction)
        or initial_quote_capital <= 0
        or not isinstance(one_way_cost_fraction, Fraction)
        or one_way_cost_fraction <= 0
        or not isinstance(price_source_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", price_source_sha256) is None
        or not isinstance(dataset.dataset_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", dataset.dataset_sha256) is None
    ):
        raise ValueError("stateful inventory cash inputs are invalid")
    provenance = bind_funding_cash_panel(funding_cash, funding)
    decisions = np.asarray(dataset.decision_time_ms)
    symbols = np.asarray(dataset.symbol_index)
    forecasts = np.asarray(predictions)
    if (
        decisions.shape != (dataset.rows,)
        or decisions.dtype.kind not in "iu"
        or symbols.shape != decisions.shape
        or symbols.dtype.kind not in "iu"
        or forecasts.shape != decisions.shape
        or forecasts.dtype.kind not in "fiu"
        or np.any(decisions < 0)
        or np.any(decisions >= 2**63 - (HORIZON_MINUTES + 1) * MINUTE_MS)
    ):
        raise ValueError("stateful inventory decision grid is invalid")
    indexes = np.flatnonzero(_combined_evaluation_mask(dataset))
    if not indexes.size or indexes.size % len(SYMBOLS):
        raise ValueError("stateful inventory evaluation groups are incomplete")
    times = decisions[indexes].astype(np.int64).reshape(-1, len(SYMBOLS))
    symbol_grid = symbols[indexes].reshape(-1, len(SYMBOLS))
    if (
        not np.all(times == times[:, :1])
        or not np.all(symbol_grid == np.arange(len(SYMBOLS)))
        or np.any(np.diff(times[:, 0]) != HORIZON_MINUTES * MINUTE_MS)
    ):
        raise ValueError("stateful inventory grid is unordered, gapped or misbound")
    forecast_grid = forecasts[indexes].astype(np.float64).reshape(-1, len(SYMBOLS))
    try:
        policy_cost_bps = float(one_way_cost_fraction * 10_000)
    except OverflowError as exc:
        raise ValueError("inventory cost exceeds the decision-policy dtype") from exc
    schedule = stateful_position_schedule(
        forecast_grid,
        mode=mode,
        cost_bps=policy_cost_bps,
        maximum_holding_hours=MAXIMUM_HOLDING_HOURS,
        cost_filter_multiplier=COST_FILTER_LAMBDA,
    )
    boundaries = np.append(
        times[:, 0] + MINUTE_MS, times[-1, 0] + (HORIZON_MINUTES + 1) * MINUTE_MS
    )
    clocks = tuple(int(time) for time in boundaries)
    positions = tuple(tuple(int(value) for value in row) for row in schedule.positions)
    ledgers: list[InventoryCashReplay] = []
    bound_prices: dict[str, list[str]] = {}
    for column, symbol in enumerate(SYMBOLS):
        series = price_panel[symbol]
        if series.symbol != symbol:
            raise ValueError("inventory price panel symbol binding is invalid")
        prices = _boundary_prices(series, boundaries)
        source = funding_cash[symbol]
        path = InventoryCashPath(
            symbol=symbol,
            boundary_time_ms=clocks,
            boundary_price=prices,
            desired_position=tuple(row[column] for row in positions),
            opening_quote_notional=initial_quote_capital / len(SYMBOLS),
            one_way_cost_fraction=one_way_cost_fraction,
            price_source_sha256=price_source_sha256,
            settlements=source.settlements,
            expected_events=source.expected_events,
            coverage_start_ms=source.coverage_start_ms,
            coverage_end_exclusive_ms=source.coverage_end_exclusive_ms,
            population_certificate_sha256=source.population_certificate_sha256,
        )
        ledgers.append(replay_fixed_base_inventory(path))
        bound_prices[symbol] = [str(price) for price in prices]
    low = tuple(
        sum((ledger.intervals[index].net_cash_lower for ledger in ledgers), Fraction(0))
        for index in range(len(positions))
    )
    high = tuple(
        sum((ledger.intervals[index].net_cash_upper for ledger in ledgers), Fraction(0))
        for index in range(len(positions))
    )
    equity_low, equity_high = [initial_quote_capital], [initial_quote_capital]
    for cash_low, cash_high in zip(low, high, strict=True):
        equity_low.append(equity_low[-1] + cash_low)
        equity_high.append(equity_high[-1] + cash_high)
    binding = {
        "schema": "supplied-stateful-fixed-base-cash-v1",
        "dataset_reference_sha256": dataset.dataset_sha256,
        "price_source_sha256": price_source_sha256,
        "boundary_time_ms": clocks,
        "boundary_prices": bound_prices,
        "forecasts": forecast_grid.tolist(),
        "positions": positions,
        "initial_quote_capital": str(initial_quote_capital),
        "one_way_cost_fraction": str(one_way_cost_fraction),
        "mode": mode,
        "maximum_holding_hours": MAXIMUM_HOLDING_HOURS,
        "cost_filter_multiplier": COST_FILTER_LAMBDA,
        "funding": provenance,
        "qualification": "supplied_inputs_only_not_venue_profit",
    }
    digest = hashlib.sha256(
        json.dumps(
            binding, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    ).hexdigest()
    return StatefulInventoryReplay(
        decision_time_ms=tuple(int(time) for time in times[:, 0]),
        positions=positions,
        symbol_ledgers=tuple(ledgers),
        initial_quote_capital=initial_quote_capital,
        portfolio_interval_cash_lower=low,
        portfolio_interval_cash_upper=high,
        quote_equity_lower=tuple(equity_low),
        quote_equity_upper=tuple(equity_high),
        total_return_lower=(equity_low[-1] - initial_quote_capital)
        / initial_quote_capital,
        total_return_upper=(equity_high[-1] - initial_quote_capital)
        / initial_quote_capital,
        modeled_capital_exhausted=any(value <= 0 for value in equity_low),
        replay_input_sha256=digest,
    )
