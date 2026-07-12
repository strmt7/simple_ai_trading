"""Consumed-data signal-decay diagnostics with exact BBO cost accounting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .microstructure_architecture import average_label_uniqueness
from .microstructure_features import (
    MicrostructureDataset,
    validate_microstructure_dataset,
)
from .microstructure_warehouse import MicrostructureWarehouse


_DAY_MS = 86_400_000


@dataclass(frozen=True)
class AsOfBboQuotes:
    """Best-bid-offer states aligned to requested arrival timestamps."""

    arrival_time_ms: np.ndarray
    available_time_ms: np.ndarray
    last_transaction_time_ms: np.ndarray
    bid_price: np.ndarray
    bid_qty: np.ndarray
    ask_price: np.ndarray
    ask_qty: np.ndarray
    quote_age_ms: np.ndarray
    valid: np.ndarray

    @property
    def rows(self) -> int:
        return int(len(self.arrival_time_ms))


@dataclass(frozen=True)
class HorizonPath:
    """Exact delayed and zero-latency quote outcomes for one frozen horizon."""

    horizon_seconds: int
    source_indexes: np.ndarray
    future_indexes: np.ndarray
    decision_time_ms: np.ndarray
    future_time_ms: np.ndarray
    uniqueness_weight: np.ndarray
    delayed_midquote_return_bps: np.ndarray
    delayed_long_cross_spread_gross_bps: np.ndarray
    delayed_short_cross_spread_gross_bps: np.ndarray
    delayed_long_net_bps: np.ndarray
    delayed_short_net_bps: np.ndarray
    delayed_long_liquidity_eligible: np.ndarray
    delayed_short_liquidity_eligible: np.ndarray
    zero_latency_long_net_bps: np.ndarray
    zero_latency_short_net_bps: np.ndarray
    zero_latency_long_liquidity_eligible: np.ndarray
    zero_latency_short_liquidity_eligible: np.ndarray
    zero_latency_quote_valid: np.ndarray
    exclusion_counts: Mapping[str, int]

    @property
    def rows(self) -> int:
        return int(len(self.source_indexes))


def _filled_float(values: object) -> np.ndarray:
    return np.asarray(np.ma.filled(values, np.nan), dtype=np.float64)


def _filled_int(values: object) -> np.ndarray:
    return np.asarray(np.ma.filled(values, -1), dtype=np.int64)


def load_bbo_quotes_asof(
    warehouse: MicrostructureWarehouse,
    *,
    symbol: str,
    arrival_time_ms: np.ndarray,
    maximum_quote_age_ms: int,
) -> AsOfBboQuotes:
    """Read one BBO state per timestamp without creating a persistent table."""

    timestamps = np.asarray(arrival_time_ms, dtype=np.int64)
    max_age = int(maximum_quote_age_ms)
    normalized_symbol = str(symbol).strip().upper()
    if (
        timestamps.ndim != 1
        or timestamps.size == 0
        or np.any(np.diff(timestamps) <= 0)
        or max_age <= 0
        or not normalized_symbol
    ):
        raise ValueError("BBO ASOF request contract is invalid")
    connection = warehouse.connect()
    relation_name = "_signal_decay_quote_requests"
    request = {
        "request_id": np.arange(len(timestamps), dtype=np.int64),
        "symbol": np.full(len(timestamps), normalized_symbol),
        "arrival_time_ms": timestamps,
    }
    connection.register(relation_name, request)
    try:
        values = connection.execute(
            f"""
            SELECT
                r.request_id,
                r.arrival_time_ms,
                q.available_time_ms,
                q.last_transaction_time_ms,
                q.close_bid AS bid_price,
                q.close_bid_qty AS bid_qty,
                q.close_ask AS ask_price,
                q.close_ask_qty AS ask_qty
            FROM {relation_name} r
            ASOF LEFT JOIN (
                SELECT
                    symbol,
                    available_time_ms,
                    last_transaction_time_ms,
                    close_bid,
                    close_bid_qty,
                    close_ask,
                    close_ask_qty
                FROM current_book_ticker_100ms
                WHERE symbol = ?
                  AND bucket_ms BETWEEN ? AND ?
            ) q
              ON r.symbol = q.symbol
             AND r.arrival_time_ms >= q.available_time_ms
            ORDER BY r.request_id
            """,
            [
                normalized_symbol,
                int(timestamps[0]) - max_age - 1_000,
                int(timestamps[-1]) + 1_000,
            ],
        ).fetchnumpy()
    finally:
        connection.unregister(relation_name)
    returned = _filled_int(values["arrival_time_ms"])
    available = _filled_int(values["available_time_ms"])
    transaction = _filled_int(values["last_transaction_time_ms"])
    bid = _filled_float(values["bid_price"])
    bid_qty = _filled_float(values["bid_qty"])
    ask = _filled_float(values["ask_price"])
    ask_qty = _filled_float(values["ask_qty"])
    if not np.array_equal(returned, timestamps):
        raise ValueError("BBO ASOF query changed request ordering")
    age = np.where(transaction >= 0, timestamps - transaction, -1)
    valid = (
        (available >= 0)
        & (available <= timestamps)
        & (age >= 0)
        & (age <= max_age)
        & np.isfinite(bid)
        & np.isfinite(ask)
        & np.isfinite(bid_qty)
        & np.isfinite(ask_qty)
        & (bid > 0.0)
        & (ask > bid)
        & (bid_qty > 0.0)
        & (ask_qty > 0.0)
    )
    return AsOfBboQuotes(
        arrival_time_ms=timestamps.copy(),
        available_time_ms=available,
        last_transaction_time_ms=transaction,
        bid_price=bid,
        bid_qty=bid_qty,
        ask_price=ask,
        ask_qty=ask_qty,
        quote_age_ms=age,
        valid=np.asarray(valid, dtype=bool),
    )


def exact_horizon_rows(
    decision_time_ms: np.ndarray,
    endpoints: np.ndarray,
    *,
    horizon_seconds: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Match each endpoint to an exact same-day future decision row."""

    times = np.asarray(decision_time_ms, dtype=np.int64)
    source = np.asarray(endpoints, dtype=np.int64)
    horizon = int(horizon_seconds)
    if (
        times.ndim != 1
        or source.ndim != 1
        or source.size == 0
        or np.any(np.diff(times) <= 0)
        or np.any(np.diff(source) <= 0)
        or source[0] < 0
        or source[-1] >= len(times)
        or horizon <= 0
    ):
        raise ValueError("exact horizon input contract is invalid")
    target_times = times[source] + horizon * 1_000
    positions = np.searchsorted(times, target_times)
    in_bounds = positions < len(times)
    exact = np.zeros(len(source), dtype=bool)
    exact[in_bounds] = times[positions[in_bounds]] == target_times[in_bounds]
    same_day = np.zeros(len(source), dtype=bool)
    same_day[exact] = (
        times[source[exact]] // _DAY_MS == times[positions[exact]] // _DAY_MS
    )
    keep = exact & same_day
    exclusions = {
        "requested_event_rows": int(len(source)),
        "missing_exact_future_row": int(np.sum(~exact)),
        "cross_utc_day": int(np.sum(exact & ~same_day)),
        "retained_rows": int(np.sum(keep)),
    }
    return source[keep], positions[keep].astype(np.int64), exclusions


def linear_cross_spread_cash_returns_bps(
    entry_bid: np.ndarray,
    entry_ask: np.ndarray,
    exit_bid: np.ndarray,
    exit_ask: np.ndarray,
    *,
    execution_cost_bps_per_side: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return gross and net linear-contract cash PnL for both taker sides."""

    cost = float(execution_cost_bps_per_side)
    arrays = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (entry_bid, entry_ask, exit_bid, exit_ask)
    )
    if (
        not math.isfinite(cost)
        or cost < 0.0
        or any(values.shape != arrays[0].shape for values in arrays[1:])
        or any(np.any(~np.isfinite(values)) for values in arrays)
        or any(np.any(values <= 0.0) for values in arrays)
    ):
        raise ValueError("cross-spread cash-return inputs are invalid")
    bid_in, ask_in, bid_out, ask_out = arrays
    long_ratio = bid_out / ask_in
    short_ratio = ask_out / bid_in
    long_gross = (long_ratio - 1.0) * 10_000.0
    short_gross = (1.0 - short_ratio) * 10_000.0
    long_net = long_gross - cost * (1.0 + long_ratio)
    short_net = short_gross - cost * (1.0 + short_ratio)
    return long_gross, short_gross, long_net, short_net


def _liquidity_eligibility(
    entry_bid: np.ndarray,
    entry_ask: np.ndarray,
    entry_bid_qty: np.ndarray,
    entry_ask_qty: np.ndarray,
    exit_bid_qty: np.ndarray,
    exit_ask_qty: np.ndarray,
    *,
    reference_order_notional_quote: float,
    maximum_l1_participation: float,
) -> tuple[np.ndarray, np.ndarray]:
    notional = float(reference_order_notional_quote)
    limit = float(maximum_l1_participation)
    if not math.isfinite(notional) or notional <= 0.0 or not 0.0 < limit <= 1.0:
        raise ValueError("L1 participation contract is invalid")
    long_qty = notional / entry_ask
    short_qty = notional / entry_bid
    long_participation = np.maximum(
        long_qty / entry_ask_qty,
        long_qty / exit_bid_qty,
    )
    short_participation = np.maximum(
        short_qty / entry_bid_qty,
        short_qty / exit_ask_qty,
    )
    return long_participation <= limit, short_participation <= limit


def _quote_positions(quotes: AsOfBboQuotes, timestamps: np.ndarray) -> np.ndarray:
    requested = np.asarray(timestamps, dtype=np.int64)
    positions = np.searchsorted(quotes.arrival_time_ms, requested)
    if np.any(positions >= quotes.rows) or not np.array_equal(
        quotes.arrival_time_ms[positions], requested
    ):
        raise ValueError("zero-latency quote support is incomplete")
    return positions.astype(np.int64)


def build_horizon_path(
    dataset: MicrostructureDataset,
    endpoints: np.ndarray,
    zero_latency_quotes: AsOfBboQuotes,
    *,
    horizon_seconds: int,
) -> HorizonPath:
    """Build exact delayed outcomes and a historical zero-latency comparison."""

    validate_microstructure_dataset(dataset)
    source, future, exclusions = exact_horizon_rows(
        dataset.decision_time_ms,
        endpoints,
        horizon_seconds=horizon_seconds,
    )
    if source.size == 0:
        raise ValueError("horizon path has no supported event rows")
    times = dataset.decision_time_ms[source]
    future_times = dataset.decision_time_ms[future]
    entry_bid = np.asarray(dataset.entry_bid_price[source], dtype=np.float64)
    entry_ask = np.asarray(dataset.entry_ask_price[source], dtype=np.float64)
    exit_bid = np.asarray(dataset.entry_bid_price[future], dtype=np.float64)
    exit_ask = np.asarray(dataset.entry_ask_price[future], dtype=np.float64)
    execution_cost = dataset.taker_fee_bps + dataset.additional_slippage_bps_per_side
    (
        delayed_long_gross,
        delayed_short_gross,
        delayed_long_net,
        delayed_short_net,
    ) = linear_cross_spread_cash_returns_bps(
        entry_bid,
        entry_ask,
        exit_bid,
        exit_ask,
        execution_cost_bps_per_side=execution_cost,
    )
    delayed_long_liquid, delayed_short_liquid = _liquidity_eligibility(
        entry_bid,
        entry_ask,
        np.asarray(dataset.entry_bid_qty[source], dtype=np.float64),
        np.asarray(dataset.entry_ask_qty[source], dtype=np.float64),
        np.asarray(dataset.entry_bid_qty[future], dtype=np.float64),
        np.asarray(dataset.entry_ask_qty[future], dtype=np.float64),
        reference_order_notional_quote=dataset.reference_order_notional_quote,
        maximum_l1_participation=dataset.max_l1_participation,
    )
    entry_mid = (entry_bid + entry_ask) / 2.0
    exit_mid = (exit_bid + exit_ask) / 2.0
    delayed_mid_return = (exit_mid / entry_mid - 1.0) * 10_000.0
    zero_entry_positions = _quote_positions(zero_latency_quotes, times)
    zero_exit_positions = _quote_positions(zero_latency_quotes, future_times)
    zero_valid = (
        zero_latency_quotes.valid[zero_entry_positions]
        & zero_latency_quotes.valid[zero_exit_positions]
    )
    zero_long_net = np.full(len(source), np.nan, dtype=np.float64)
    zero_short_net = np.full(len(source), np.nan, dtype=np.float64)
    zero_long_liquid = np.zeros(len(source), dtype=bool)
    zero_short_liquid = np.zeros(len(source), dtype=bool)
    if np.any(zero_valid):
        entry_positions = zero_entry_positions[zero_valid]
        exit_positions = zero_exit_positions[zero_valid]
        zero_entry_bid = zero_latency_quotes.bid_price[entry_positions]
        zero_entry_ask = zero_latency_quotes.ask_price[entry_positions]
        zero_exit_bid = zero_latency_quotes.bid_price[exit_positions]
        zero_exit_ask = zero_latency_quotes.ask_price[exit_positions]
        _long_gross, _short_gross, long_net, short_net = (
            linear_cross_spread_cash_returns_bps(
                zero_entry_bid,
                zero_entry_ask,
                zero_exit_bid,
                zero_exit_ask,
                execution_cost_bps_per_side=execution_cost,
            )
        )
        long_liquid, short_liquid = _liquidity_eligibility(
            zero_entry_bid,
            zero_entry_ask,
            zero_latency_quotes.bid_qty[entry_positions],
            zero_latency_quotes.ask_qty[entry_positions],
            zero_latency_quotes.bid_qty[exit_positions],
            zero_latency_quotes.ask_qty[exit_positions],
            reference_order_notional_quote=dataset.reference_order_notional_quote,
            maximum_l1_participation=dataset.max_l1_participation,
        )
        zero_long_net[zero_valid] = long_net
        zero_short_net[zero_valid] = short_net
        zero_long_liquid[zero_valid] = long_liquid
        zero_short_liquid[zero_valid] = short_liquid
    target_exit = (
        np.asarray(dataset.decision_time_ms, dtype=np.int64)
        + int(horizon_seconds) * 1_000
        + int(dataset.total_latency_ms)
    )
    weights = average_label_uniqueness(
        dataset.decision_time_ms,
        target_exit,
        source,
    ).astype(np.float64)
    exclusions = {
        **exclusions,
        "zero_latency_entry_or_exit_quote_invalid": int(np.sum(~zero_valid)),
        "delayed_long_l1_participation_exceeded": int(np.sum(~delayed_long_liquid)),
        "delayed_short_l1_participation_exceeded": int(np.sum(~delayed_short_liquid)),
        "zero_latency_long_l1_participation_exceeded": int(
            np.sum(zero_valid & ~zero_long_liquid)
        ),
        "zero_latency_short_l1_participation_exceeded": int(
            np.sum(zero_valid & ~zero_short_liquid)
        ),
    }
    return HorizonPath(
        horizon_seconds=int(horizon_seconds),
        source_indexes=source,
        future_indexes=future,
        decision_time_ms=times,
        future_time_ms=future_times,
        uniqueness_weight=weights,
        delayed_midquote_return_bps=delayed_mid_return,
        delayed_long_cross_spread_gross_bps=delayed_long_gross,
        delayed_short_cross_spread_gross_bps=delayed_short_gross,
        delayed_long_net_bps=delayed_long_net,
        delayed_short_net_bps=delayed_short_net,
        delayed_long_liquidity_eligible=delayed_long_liquid,
        delayed_short_liquidity_eligible=delayed_short_liquid,
        zero_latency_long_net_bps=zero_long_net,
        zero_latency_short_net_bps=zero_short_net,
        zero_latency_long_liquidity_eligible=zero_long_liquid,
        zero_latency_short_liquidity_eligible=zero_short_liquid,
        zero_latency_quote_valid=zero_valid,
        exclusion_counts=exclusions,
    )


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks with stable tie handling."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or np.any(~np.isfinite(source)):
        raise ValueError("rank inputs must be a finite vector")
    order = np.argsort(source, kind="stable")
    sorted_values = source[order]
    starts = np.flatnonzero(
        np.concatenate(([True], sorted_values[1:] != sorted_values[:-1]))
    )
    ends = np.concatenate((starts[1:], [len(source)]))
    ranks = np.empty(len(source), dtype=np.float64)
    for start, end in zip(starts, ends, strict=True):
        ranks[order[start:end]] = (start + end - 1) / 2.0
    return ranks


def weighted_roc_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    sample_weight: np.ndarray,
) -> float | None:
    """Return weighted binary ROC AUC with exact half credit for score ties."""

    y = np.asarray(labels, dtype=np.int8)
    score = np.asarray(scores, dtype=np.float64)
    weight = np.asarray(sample_weight, dtype=np.float64)
    if (
        y.ndim != 1
        or score.shape != y.shape
        or weight.shape != y.shape
        or np.any((y != 0) & (y != 1))
        or np.any(~np.isfinite(score))
        or np.any(~np.isfinite(weight))
        or np.any(weight <= 0.0)
    ):
        raise ValueError("weighted AUC inputs are invalid")
    positive_total = float(np.sum(weight[y == 1]))
    negative_total = float(np.sum(weight[y == 0]))
    if positive_total <= 0.0 or negative_total <= 0.0:
        return None
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    sorted_y = y[order]
    sorted_weight = weight[order]
    starts = np.flatnonzero(
        np.concatenate(([True], sorted_score[1:] != sorted_score[:-1]))
    )
    positive = np.add.reduceat(sorted_weight * sorted_y, starts)
    negative = np.add.reduceat(sorted_weight * (1 - sorted_y), starts)
    negative_before = np.cumsum(negative) - negative
    numerator = float(np.sum(positive * (negative_before + 0.5 * negative)))
    return numerator / (positive_total * negative_total)


def spearman_information_coefficient(
    signal: np.ndarray,
    outcome: np.ndarray,
) -> float | None:
    """Return unweighted Spearman rank correlation or None when undefined."""

    left = np.asarray(signal, dtype=np.float64)
    right = np.asarray(outcome, dtype=np.float64)
    if (
        left.ndim != 1
        or right.shape != left.shape
        or np.any(~np.isfinite(left))
        or np.any(~np.isfinite(right))
    ):
        raise ValueError("Spearman inputs are invalid")
    if len(left) < 2:
        return None
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if denominator <= 0.0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def direction_metrics(
    path: HorizonPath,
    signal: np.ndarray,
    *,
    row_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Measure prespecified signal orientation against delayed midquote direction."""

    values = np.asarray(signal, dtype=np.float64)
    if values.shape != (path.rows,):
        raise ValueError("signal length differs from the horizon path")
    mask = np.isfinite(values) & np.isfinite(path.delayed_midquote_return_bps)
    mask &= path.delayed_midquote_return_bps != 0.0
    if row_mask is not None:
        supplied = np.asarray(row_mask, dtype=bool)
        if supplied.shape != mask.shape:
            raise ValueError("direction metric row mask is invalid")
        mask &= supplied
    selected_signal = values[mask]
    outcomes = path.delayed_midquote_return_bps[mask]
    labels = np.asarray(outcomes > 0.0, dtype=np.int8)
    weights = path.uniqueness_weight[mask]
    weighted_auc = weighted_roc_auc(labels, selected_signal, weights)
    unweighted_auc = weighted_roc_auc(
        labels,
        selected_signal,
        np.ones(len(labels), dtype=np.float64),
    )
    predicted_up = selected_signal > 0.0
    predicted_down = selected_signal < 0.0
    routed = predicted_up | predicted_down
    correct = (predicted_up & (labels == 1)) | (predicted_down & (labels == 0))
    weighted_accuracy = (
        float(np.sum(weights[correct]) / np.sum(weights[routed]))
        if np.any(routed)
        else None
    )
    unweighted_accuracy = float(np.mean(correct[routed])) if np.any(routed) else None
    return {
        "rows": int(np.sum(mask)),
        "positive_direction_rows": int(np.sum(labels == 1)),
        "negative_direction_rows": int(np.sum(labels == 0)),
        "unrouted_zero_signal_rows": int(np.sum(mask & (values == 0.0))),
        "weighted_roc_auc": weighted_auc,
        "unweighted_roc_auc": unweighted_auc,
        "weighted_direction_accuracy": weighted_accuracy,
        "unweighted_direction_accuracy": unweighted_accuracy,
        "spearman_information_coefficient": spearman_information_coefficient(
            selected_signal,
            outcomes,
        ),
    }


def daily_direction_metrics(
    path: HorizonPath,
    signal: np.ndarray,
) -> list[dict[str, object]]:
    """Return one direction record per UTC day."""

    days = path.decision_time_ms // _DAY_MS
    records: list[dict[str, object]] = []
    for day in np.unique(days):
        metrics = direction_metrics(path, signal, row_mask=days == day)
        records.append({"utc_day": int(day), **metrics})
    return records


def chronological_nonoverlapping_mask(path: HorizonPath) -> np.ndarray:
    """Select the first chronological nonoverlapping event in each UTC day."""

    keep = np.zeros(path.rows, dtype=bool)
    days = path.decision_time_ms // _DAY_MS
    interval_ms = path.horizon_seconds * 1_000
    for day in np.unique(days):
        indexes = np.flatnonzero(days == day)
        next_available = -1
        for index in indexes:
            timestamp = int(path.decision_time_ms[index])
            if timestamp < next_available:
                continue
            keep[index] = True
            next_available = timestamp + interval_ms
    return keep


def routed_cost_metrics(
    path: HorizonPath,
    signal: np.ndarray,
    *,
    row_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Decompose delayed taker event outcomes for the signal's fixed orientation."""

    values = np.asarray(signal, dtype=np.float64)
    if values.shape != (path.rows,):
        raise ValueError("signal length differs from the horizon path")
    side = np.zeros(path.rows, dtype=np.int8)
    finite = np.isfinite(values)
    side[finite] = np.sign(values[finite]).astype(np.int8)
    mask = finite & (side != 0)
    if row_mask is not None:
        supplied = np.asarray(row_mask, dtype=bool)
        if supplied.shape != mask.shape:
            raise ValueError("cost metric row mask is invalid")
        mask &= supplied
    delayed_liquid = np.where(
        side > 0,
        path.delayed_long_liquidity_eligible,
        path.delayed_short_liquidity_eligible,
    )
    eligible = mask & delayed_liquid
    gross_mid = np.where(
        side > 0,
        path.delayed_midquote_return_bps,
        -path.delayed_midquote_return_bps,
    )
    cross_gross = np.where(
        side > 0,
        path.delayed_long_cross_spread_gross_bps,
        path.delayed_short_cross_spread_gross_bps,
    )
    delayed_net = np.where(
        side > 0,
        path.delayed_long_net_bps,
        path.delayed_short_net_bps,
    )
    zero_net = np.where(
        side > 0,
        path.zero_latency_long_net_bps,
        path.zero_latency_short_net_bps,
    )
    zero_liquid = np.where(
        side > 0,
        path.zero_latency_long_liquidity_eligible,
        path.zero_latency_short_liquidity_eligible,
    )
    latency = eligible & path.zero_latency_quote_valid & zero_liquid

    def mean(values_: np.ndarray, selected: np.ndarray) -> float | None:
        return float(np.mean(values_[selected])) if np.any(selected) else None

    return {
        "routed_rows": int(np.sum(mask)),
        "delayed_l1_eligible_rows": int(np.sum(eligible)),
        "delayed_l1_ineligible_rows": int(np.sum(mask & ~delayed_liquid)),
        "zero_latency_comparable_rows": int(np.sum(latency)),
        "mean_signal_aligned_gross_midquote_return_bps": mean(gross_mid, eligible),
        "mean_cross_spread_gross_return_bps": mean(cross_gross, eligible),
        "mean_spread_crossing_cost_bps": mean(gross_mid - cross_gross, eligible),
        "mean_fee_and_slippage_cost_bps": mean(cross_gross - delayed_net, eligible),
        "mean_historical_latency_drag_bps": mean(zero_net - delayed_net, latency),
        "mean_delayed_net_return_bps": mean(delayed_net, eligible),
        "delayed_net_positive_rate": (
            float(np.mean(delayed_net[eligible] > 0.0)) if np.any(eligible) else None
        ),
    }


def ranked_event_outcomes(
    path: HorizonPath,
    signal: np.ndarray,
    *,
    requested_counts: Sequence[int],
) -> list[dict[str, object]]:
    """Report fixed absolute-signal ranks as event outcomes, never as trades."""

    values = np.asarray(signal, dtype=np.float64)
    side = np.zeros(path.rows, dtype=np.int8)
    finite = np.isfinite(values)
    side[finite] = np.sign(values[finite]).astype(np.int8)
    delayed_liquid = np.where(
        side > 0,
        path.delayed_long_liquidity_eligible,
        path.delayed_short_liquidity_eligible,
    )
    eligible_indexes = np.flatnonzero(finite & (side != 0) & delayed_liquid)
    order = np.lexsort(
        (
            path.decision_time_ms[eligible_indexes],
            -np.abs(values[eligible_indexes]),
        )
    )
    ranked = eligible_indexes[order]
    net = np.where(
        side > 0,
        path.delayed_long_net_bps,
        path.delayed_short_net_bps,
    )
    records: list[dict[str, object]] = []
    for raw_count in requested_counts:
        requested = int(raw_count)
        if requested <= 0:
            raise ValueError("ranked event count must be positive")
        actual = min(requested, len(ranked))
        selected = ranked[:actual]
        records.append(
            {
                "requested_rows": requested,
                "actual_rows": actual,
                "mean_delayed_net_return_bps": (
                    float(np.mean(net[selected])) if actual else None
                ),
                "median_delayed_net_return_bps": (
                    float(np.median(net[selected])) if actual else None
                ),
                "delayed_net_positive_rate": (
                    float(np.mean(net[selected] > 0.0)) if actual else None
                ),
                "event_outcomes_not_executable_trades": True,
            }
        )
    return records


def placebo_weighted_auc_distribution(
    path: HorizonPath,
    signal: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Permute signal values within UTC day and return weighted AUC placebos."""

    values = np.asarray(signal, dtype=np.float64)
    count = int(replicates)
    if values.shape != (path.rows,) or count <= 0:
        raise ValueError("placebo contract is invalid")
    valid = (
        np.isfinite(values)
        & np.isfinite(path.delayed_midquote_return_bps)
        & (path.delayed_midquote_return_bps != 0.0)
    )
    scores = values[valid]
    labels = np.asarray(path.delayed_midquote_return_bps[valid] > 0.0, dtype=np.int8)
    weights = path.uniqueness_weight[valid]
    days = path.decision_time_ms[valid] // _DAY_MS
    if len(np.unique(labels)) != 2:
        raise ValueError("placebo direction target has one class")
    score_order = np.argsort(scores, kind="stable")
    ordered_scores = scores[score_order]
    group_starts = np.flatnonzero(
        np.concatenate(([True], ordered_scores[1:] != ordered_scores[:-1]))
    )
    day_indexes = [np.flatnonzero(days == day) for day in np.unique(days)]
    rng = np.random.default_rng(int(seed))
    output = np.empty(count, dtype=np.float64)
    target_for_source = np.arange(len(scores), dtype=np.int64)
    for replicate in range(count):
        for indexes in day_indexes:
            target_for_source[indexes] = rng.permutation(indexes)
        target_order = target_for_source[score_order]
        ordered_labels = labels[target_order]
        ordered_weights = weights[target_order]
        positive = np.add.reduceat(
            ordered_weights * ordered_labels,
            group_starts,
        )
        negative = np.add.reduceat(
            ordered_weights * (1 - ordered_labels),
            group_starts,
        )
        positive_total = float(np.sum(positive))
        negative_total = float(np.sum(negative))
        negative_before = np.cumsum(negative) - negative
        output[replicate] = float(
            np.sum(positive * (negative_before + 0.5 * negative))
            / (positive_total * negative_total)
        )
    return output


def placebo_summary(observed: float | None, values: np.ndarray) -> dict[str, object]:
    """Summarize a descriptive placebo distribution without a significance claim."""

    samples = np.asarray(values, dtype=np.float64)
    if observed is None or not math.isfinite(observed):
        raise ValueError("observed placebo statistic is undefined")
    if samples.ndim != 1 or samples.size == 0 or np.any(~np.isfinite(samples)):
        raise ValueError("placebo samples are invalid")
    return {
        "replicates": int(len(samples)),
        "observed_weighted_roc_auc": float(observed),
        "observed_rank_descending": int(1 + np.sum(samples > observed)),
        "one_sided_empirical_exceedance_fraction": float(
            (1 + np.sum(samples >= observed)) / (len(samples) + 1)
        ),
        "placebo_mean": float(np.mean(samples)),
        "placebo_standard_deviation": float(np.std(samples)),
        "placebo_95_percent_interval": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "formal_multiple_testing_significance_claim": False,
    }
