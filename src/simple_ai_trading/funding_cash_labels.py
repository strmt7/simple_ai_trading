"""Mark-aware funding label bounds; supplied evidence is not venue admission."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re
from typing import TYPE_CHECKING, Mapping

import numpy as np

from .funding_cash import (
    FundingEntitlement,
    LinearFundingSettlement,
    funding_cash_bounds_for_events,
    reconcile_funding_event_population,
)

if TYPE_CHECKING:
    from .derivatives_hurdle_data import FundingState


def _prefix_enclosure(values: tuple[Fraction, ...]) -> tuple[np.ndarray, np.ndarray]:
    prefix = [Fraction(0)]
    for value in values:
        prefix.append(prefix[-1] + value)
    try:
        nearest = np.asarray([float(value) for value in prefix], dtype=np.float64)
    except OverflowError as exc:
        raise ValueError("funding cash prefix exceeds the label dtype") from exc
    if not np.isfinite(nearest).all():
        raise ValueError("funding cash prefix exceeds the label dtype")
    low, high = nearest.copy(), nearest.copy()
    for index, value in enumerate(prefix):
        represented = Fraction.from_float(nearest[index])
        if represented > value:
            low[index] = np.nextafter(low[index], -np.inf)
        elif represented < value:
            high[index] = np.nextafter(high[index], np.inf)
    return low, high


def outward_float32(values: np.ndarray, *, lower: bool) -> np.ndarray:
    """Keep a label bound outward when storing it in the existing float32 format."""
    original = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        stored = original.astype(np.float32)
    favorable = stored > original if lower else stored < original
    stored[favorable] = np.nextafter(stored[favorable], -np.inf if lower else np.inf)
    if not np.isfinite(original).all() or not np.isfinite(stored).all():
        raise ValueError("funding cash bound exceeds the stored label dtype")
    return stored


@dataclass(frozen=True)
class FundingLabelBounds:
    lower_bps: np.ndarray
    upper_bps: np.ndarray
    uncertain_events: np.ndarray


@dataclass(frozen=True)
class FundingCashLabelSeries:
    """Bind marked events to a separately certified population and UTC scope.

    The certificate digest identifies the caller's evidence; it does not prove
    authenticity or completeness by itself. A capture-origin/coverage auditor
    must qualify that certificate before training or financial admission.
    """

    symbol: str
    settlements: tuple[LinearFundingSettlement, ...]
    expected_events: tuple[tuple[int, Fraction], ...]
    coverage_start_ms: int
    coverage_end_exclusive_ms: int
    population_certificate_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.coverage_start_ms) is not int
            or type(self.coverage_end_exclusive_ms) is not int
            or not 0 <= self.coverage_start_ms < self.coverage_end_exclusive_ms < 2**63
            or not isinstance(self.settlements, tuple)
            or not isinstance(self.expected_events, tuple)
            or not isinstance(self.population_certificate_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.population_certificate_sha256) is None
        ):
            raise ValueError("funding cash label scope is invalid")
        reconcile_funding_event_population(
            self.settlements, self.expected_events, expected_symbol=self.symbol
        )
        if any(
            not self.coverage_start_ms
            <= event.funding_time_ms
            < self.coverage_end_exclusive_ms
            for event in self.settlements
        ):
            raise ValueError("funding cash event is outside its certified scope")

    def provenance(self) -> dict[str, object]:
        """Bind exact supplied values, certificates and the declared label law."""
        value: dict[str, object] = {
            "schema": "supplied-linear-funding-label-bounds-v1",
            "symbol": self.symbol,
            "coverage_start_ms": self.coverage_start_ms,
            "coverage_end_exclusive_ms": self.coverage_end_exclusive_ms,
            "population_certificate_sha256": self.population_certificate_sha256,
            "payment_asset": "USDT",
            "quantity_semantics": "fixed_base_entry_notional_normalization",
            "boundary_semantics": "entry_and_exit_interval_unknown_entitlement",
            "numeric_semantics": "exact_prefixes_outward_float64_enclosure",
            "admission": "origin_coverage_and_execution_require_independent_qualification",
            "events": [
                [
                    event.funding_time_ms,
                    str(event.rate),
                    str(event.settlement_mark),
                    event.source_body_sha256,
                    event.rate_type,
                ]
                for event in self.settlements
            ],
        }
        value["funding_cash_label_sha256"] = hashlib.sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("ascii")
        ).hexdigest()
        return value

    def validate_rate_state(self, state: FundingState) -> None:
        """Reject a cash source that differs from the consumer's rate population."""
        times = np.asarray(state.event_time_ms)
        rates = np.asarray(state.event_rate)
        if (
            times.ndim != 1
            or rates.shape != times.shape
            or rates.dtype.kind not in "fiu"
            or times.dtype.kind not in "iu"
            or times.dtype.kind == "b"
            or np.any(times < 0)
            or np.any(times >= 2**63)
            or np.any(times[1:] <= times[:-1])
            or not np.isfinite(rates).all()
        ):
            raise ValueError("funding cash consumer rate state is invalid")
        scope = (times >= self.coverage_start_ms) & (
            times < self.coverage_end_exclusive_ms
        )
        if not np.array_equal(
            times[scope], [event[0] for event in self.expected_events]
        ) or not np.array_equal(
            rates[scope], [float(event[1]) for event in self.expected_events]
        ):
            raise ValueError(
                "funding cash consumer population conflicts with marked events"
            )

    def holding_bounds(
        self,
        entry_time_ms: np.ndarray,
        earliest_exit_time_ms: np.ndarray,
        latest_exit_time_ms: np.ndarray,
        entry_price: np.ndarray,
        *,
        side: int,
    ) -> FundingLabelBounds:
        """Enclose fixed-base payments for a supplied holding/exit clock model.

        Interior events before the earliest exit are modeled HELD. Entry and
        the inclusive exit interval are UNKNOWN. This is not a fill receipt or
        a bound on exchange settlement delays, fees, margin or total P&L.
        """
        clocks = tuple(
            np.asarray(value)
            for value in (entry_time_ms, earliest_exit_time_ms, latest_exit_time_ms)
        )
        price = np.asarray(entry_price, dtype=np.float64)
        entry, early, late = clocks
        if (
            type(side) is not int
            or side not in (-1, 1)
            or any(value.ndim != 1 or value.dtype.kind not in "iu" for value in clocks)
            or any(value.shape != entry.shape for value in (*clocks, price))
            or np.any(entry < self.coverage_start_ms)
            or np.any(late >= self.coverage_end_exclusive_ms)
            or np.any(early < entry)
            or np.any(late < early)
            or not np.isfinite(price).all()
            or np.any(price <= 0)
        ):
            raise ValueError(
                "funding cash holding window is invalid or outside coverage"
            )
        times = np.asarray(
            [event.funding_time_ms for event in self.settlements], dtype=np.int64
        )
        # Use the shared exact law once per event, not once per training row.
        payments = tuple(
            funding_cash_bounds_for_events(
                (event,),
                (FundingEntitlement.HELD,),
                expected_symbol=self.symbol,
                signed_base_quantity=Fraction(side),
                entry_price=Fraction(1),
            ).cash_lower
            for event in self.settlements
        )
        possible_start = np.searchsorted(times, entry, side="left")
        possible_end = np.searchsorted(times, late, side="right")
        held_start = np.searchsorted(times, entry, side="right")
        held_end = np.maximum(held_start, np.searchsorted(times, early, side="left"))

        def totals(
            prefix: tuple[np.ndarray, np.ndarray],
            start: np.ndarray,
            end: np.ndarray,
            *,
            lower: bool,
        ) -> np.ndarray:
            low, high = prefix
            value = low[end] - high[start] if lower else high[end] - low[start]
            return np.where(
                start == end, 0.0, np.nextafter(value, -np.inf if lower else np.inf)
            )

        held_prefix = _prefix_enclosure(payments)
        negative_prefix = _prefix_enclosure(
            tuple(min(Fraction(0), value) for value in payments)
        )
        positive_prefix = _prefix_enclosure(
            tuple(max(Fraction(0), value) for value in payments)
        )

        def enclosed_cash(*, lower: bool) -> np.ndarray:
            value = totals(held_prefix, held_start, held_end, lower=lower)
            prefix = negative_prefix if lower else positive_prefix
            for start, end in ((possible_start, held_start), (held_end, possible_end)):
                uncertain = totals(prefix, start, end, lower=lower)
                value = np.nextafter(value + uncertain, -np.inf if lower else np.inf)
            return np.nextafter(
                np.nextafter(value / price, -np.inf if lower else np.inf) * 10_000.0,
                -np.inf if lower else np.inf,
            )

        lower_bps, upper_bps = enclosed_cash(lower=True), enclosed_cash(lower=False)
        negative_count = np.concatenate(
            ([0], np.cumsum([value < 0 for value in payments]))
        )
        positive_count = np.concatenate(
            ([0], np.cumsum([value > 0 for value in payments]))
        )
        no_negative = negative_count[possible_end] == negative_count[possible_start]
        no_positive = positive_count[possible_end] == positive_count[possible_start]
        lower_bps[no_negative] = np.maximum(lower_bps[no_negative], 0.0)
        upper_bps[no_positive] = np.minimum(upper_bps[no_positive], 0.0)
        no_payment = possible_start == possible_end
        lower_bps[no_payment] = upper_bps[no_payment] = 0.0
        if not np.isfinite(lower_bps).all() or not np.isfinite(upper_bps).all():
            raise ValueError("funding cash normalized labels are nonfinite")
        return FundingLabelBounds(
            lower_bps=lower_bps,
            upper_bps=upper_bps,
            uncertain_events=(possible_end - possible_start) - (held_end - held_start),
        )


def bind_funding_cash_panel(
    funding_cash: Mapping[str, FundingCashLabelSeries] | None,
    funding: Mapping[str, FundingState],
) -> dict[str, object] | None:
    """Require complete matching cash/rate symbols without a proxy fallback."""
    if funding_cash is None:
        return None
    if set(funding_cash) != set(funding):
        raise ValueError("funding cash panel symbols are incomplete")
    provenance: dict[str, object] = {}
    for symbol in sorted(funding):
        source = funding_cash[symbol]
        if not isinstance(source, FundingCashLabelSeries) or source.symbol != symbol:
            raise ValueError("funding cash panel symbol binding is invalid")
        source.validate_rate_state(funding[symbol])
        provenance[symbol] = source.provenance()
    return provenance
