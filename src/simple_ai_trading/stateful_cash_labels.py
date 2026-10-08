"""Two-sided supplied hourly cash labels, distinct from legacy signed targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import re
from typing import Mapping

import numpy as np

from .cross_asset_cost_data import MINUTE_MS, SYMBOLS, MinuteSeries
from .derivatives_hurdle_data import FundingState
from .funding_cash_labels import bind_funding_cash_panel, FundingCashLabelSeries
from .stateful_cash_replay import _boundary_prices
from .stateful_turnover_model import (
    HORIZON_MINUTES,
    StatefulHourlyDataset,
    _array_identity,
)


@dataclass(frozen=True)
class StatefulCashLabelBatch:
    decision_time_ms: np.ndarray
    symbol_index: np.ndarray
    entry_time_ms: np.ndarray
    exit_time_ms: np.ndarray
    long_net_lower_bps: np.ndarray
    long_net_upper_bps: np.ndarray
    short_net_lower_bps: np.ndarray
    short_net_upper_bps: np.ndarray
    uncertain_funding_events: np.ndarray
    reference_dataset_sha256: str
    feature_grid_sha256: str
    price_source_sha256: str
    funding_provenance_json: str
    one_way_cost_fraction: Fraction
    label_sha256: str
    schema_version: str = field(default="supplied-hourly-two-sided-cash-v1", init=False)
    financially_qualified: bool = field(default=False, init=False)


def _float_enclosure(value: Fraction) -> tuple[float, float]:
    try:
        nearest = float(value)
    except OverflowError as exc:
        raise ValueError("hourly price cash exceeds label dtype") from exc
    if not np.isfinite(nearest):
        raise ValueError("hourly price cash exceeds label dtype")
    represented = Fraction.from_float(nearest)
    return (
        float(np.nextafter(nearest, -np.inf)) if represented > value else nearest,
        float(np.nextafter(nearest, np.inf)) if represented < value else nearest,
    )


def build_stateful_cash_labels(
    dataset: StatefulHourlyDataset,
    *,
    price_panel: Mapping[str, MinuteSeries],
    funding: Mapping[str, FundingState],
    funding_cash: Mapping[str, FundingCashLabelSeries],
    one_way_cost_fraction: Fraction,
    price_source_sha256: str,
) -> StatefulCashLabelBatch:
    """Keep each side's adverse/upper cash, including modeled entry and exit costs.

    Each row is an independently opened, fixed-base hourly counterfactual, not a
    continuously held stateful policy, a complete neutral hedge or native fills.
    Prices are supplied opens at decision+1 minute and entry+60 minutes. Their
    binary float values are retained exactly; candles never replace funding
    marks. Legacy targets do not enter these labels or their feature binding.
    Origin, coverage, costs, financing and margin need independent admission.
    """
    if (
        not isinstance(dataset, StatefulHourlyDataset)
        or set(price_panel) != set(SYMBOLS)
        or set(funding) != set(SYMBOLS)
        or funding_cash is None
        or set(funding_cash) != set(SYMBOLS)
        or not isinstance(one_way_cost_fraction, Fraction)
        or one_way_cost_fraction <= 0
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in (price_source_sha256, dataset.dataset_sha256)
        )
    ):
        raise ValueError("hourly cash source or cost contract is invalid")
    provenance = bind_funding_cash_panel(funding_cash, funding)
    clocks = np.asarray(dataset.decision_time_ms)
    symbols = np.asarray(dataset.symbol_index)
    baseline = np.asarray(dataset.baseline_features)
    augmented = np.asarray(dataset.augmented_features)
    if baseline.ndim != 2:
        raise ValueError("hourly cash feature/decision grid is invalid")
    rows = baseline.shape[0]
    if (
        not rows
        or rows % len(SYMBOLS)
        or clocks.shape != (rows,)
        or clocks.dtype.kind not in "iu"
        or symbols.shape != clocks.shape
        or symbols.dtype.kind not in "iu"
        or np.any(clocks < 0)
        or np.any(clocks >= 2**63 - (HORIZON_MINUTES + 1) * MINUTE_MS)
        or np.any(clocks % (HORIZON_MINUTES * MINUTE_MS))
        or baseline.ndim != 2
        or baseline.shape[1] != len(dataset.feature_names)
        or augmented.ndim != 2
        or augmented.shape[0] != rows
        or any(
            x.dtype.kind not in "fiu" or not np.isfinite(x).all()
            for x in (baseline, augmented)
        )
    ):
        raise ValueError("hourly cash feature/decision grid is invalid")
    times = clocks.reshape(-1, len(SYMBOLS))
    if (
        not np.all(times == times[:, :1])
        or not np.all(symbols.reshape(-1, len(SYMBOLS)) == np.arange(len(SYMBOLS)))
        or np.any(np.diff(times[:, 0].astype(np.int64)) != HORIZON_MINUTES * MINUTE_MS)
    ):
        raise ValueError("hourly cash grid is unordered, gapped or incomplete")
    entry = clocks.astype(np.int64) + MINUTE_MS
    exit_ = entry + HORIZON_MINUTES * MINUTE_MS
    long_low, long_high, short_low, short_high = [np.empty(rows) for _ in range(4)]
    uncertain = np.empty(rows, dtype=np.int64)
    price_bindings: dict[str, list[list[str]]] = {}
    for column, symbol in enumerate(SYMBOLS):
        selected = np.arange(column, rows, len(SYMBOLS))
        series = price_panel[symbol]
        if not isinstance(series, MinuteSeries) or series.symbol != symbol:
            raise ValueError("hourly price panel symbol binding is invalid")
        entries = _boundary_prices(series, entry[selected])
        exits = _boundary_prices(series, exit_[selected])
        price_bindings[symbol] = [
            [str(a), str(b)] for a, b in zip(entries, exits, strict=True)
        ]
        source = funding_cash[symbol]
        for side, lower, upper in (
            (1, long_low, long_high),
            (-1, short_low, short_high),
        ):
            bounds = source.holding_bounds(
                entry[selected],
                exit_[selected],
                exit_[selected],
                np.asarray([float(price) for price in entries]),
                side=side,
            )
            for local, row in enumerate(selected):
                ratio = exits[local] / entries[local]
                price_and_cost = (
                    side * (ratio - 1) - one_way_cost_fraction * (1 + ratio)
                ) * 10_000
                low, high = _float_enclosure(price_and_cost)
                lower[row] = np.nextafter(low + bounds.lower_bps[local], -np.inf)
                upper[row] = np.nextafter(high + bounds.upper_bps[local], np.inf)
            uncertain[selected] = bounds.uncertain_events
    outputs = (long_low, long_high, short_low, short_high)
    if any(not np.isfinite(value).all() for value in outputs) or any(
        np.any(low > high)
        for low, high in ((long_low, long_high), (short_low, short_high))
    ):
        raise ValueError("hourly cash label enclosure is nonfinite or regressed")
    feature_digest = _array_identity(
        clocks, symbols, baseline, augmented, names=dataset.feature_names
    )
    provenance_json = json.dumps(
        provenance, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    binding = {
        "schema": "supplied-hourly-two-sided-cash-v1",
        "reference_dataset_sha256": dataset.dataset_sha256,
        "feature_grid_sha256": feature_digest,
        "price_source_sha256": price_source_sha256,
        "prices": price_bindings,
        "funding": provenance,
        "one_way_cost_fraction": str(one_way_cost_fraction),
        "numeric_labels_sha256": _array_identity(
            entry, exit_, *outputs, uncertain, names=()
        ),
        "admission": "supplied_inputs_only_not_complete_hedge_or_training_admission",
    }
    digest = hashlib.sha256(
        json.dumps(
            binding, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    ).hexdigest()
    arrays = (clocks.copy(), symbols.copy(), entry, exit_, *outputs, uncertain)
    for value in arrays:
        value.setflags(write=False)
    return StatefulCashLabelBatch(
        decision_time_ms=arrays[0],
        symbol_index=arrays[1],
        entry_time_ms=entry,
        exit_time_ms=exit_,
        long_net_lower_bps=long_low,
        long_net_upper_bps=long_high,
        short_net_lower_bps=short_low,
        short_net_upper_bps=short_high,
        uncertain_funding_events=uncertain,
        reference_dataset_sha256=dataset.dataset_sha256,
        feature_grid_sha256=feature_digest,
        price_source_sha256=price_source_sha256,
        funding_provenance_json=provenance_json,
        one_way_cost_fraction=one_way_cost_fraction,
        label_sha256=digest,
    )
