"""Causal queue-censored targets and flow features for make/take research."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Sequence

from numba import njit
import numpy as np


PASSIVE_FILL_SCHEMA_VERSION = "queue-censored-passive-fill-v1"
EXPONENTIAL_FLOW_SCHEMA_VERSION = "causal-exponential-trade-flow-v1"
PASSIVE_FILL_BUCKETS_MS = (5_000, 10_000, 15_000)
EXPONENTIAL_FLOW_HALF_LIVES_SECONDS = (1, 2, 5, 10, 30, 60)
EXPONENTIAL_FLOW_OBSERVATION_DELAY_MS = 1_000


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PassiveFillResult:
    """One side of conservative full-fill evidence at exact placement prices."""

    schema_version: str
    buyer_is_maker: bool
    expiry_ms: int
    order_notional_quote: float
    source_trade_sha256: str
    arrival_time_ms: np.ndarray
    placement_price: np.ndarray
    queue_ahead_quantity: np.ndarray
    own_quantity: np.ndarray
    required_printed_quantity: np.ndarray
    filled: np.ndarray
    fill_bucket: np.ndarray
    fill_time_ms: np.ndarray
    first_matching_trade_id: np.ndarray
    completion_trade_id: np.ndarray
    matching_trade_count: np.ndarray
    printed_quantity_through_fill: np.ndarray
    result_sha256: str

    @property
    def rows(self) -> int:
        return int(self.arrival_time_ms.size)

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "buyer_is_maker": self.buyer_is_maker,
            "expiry_ms": self.expiry_ms,
            "order_notional_quote": self.order_notional_quote,
            "source_trade_sha256": self.source_trade_sha256,
            "rows": self.rows,
            "filled_rows": int(np.count_nonzero(self.filled)),
            "fill_rate": float(np.mean(self.filled)),
            "fill_bucket_counts": {
                str(bucket): int(np.count_nonzero(self.fill_bucket == bucket))
                for bucket in range(4)
            },
            "result_sha256": self.result_sha256,
            "trading_authority": False,
            "profitability_claim": False,
        }


def _passive_fill_payload(result: PassiveFillResult) -> dict[str, object]:
    return {
        "schema_version": result.schema_version,
        "buyer_is_maker": result.buyer_is_maker,
        "expiry_ms": result.expiry_ms,
        "order_notional_quote": format(result.order_notional_quote, ".17g"),
        "source_trade_sha256": result.source_trade_sha256,
        "arrays": {
            name: _array_sha256(getattr(result, name))
            for name in (
                "arrival_time_ms",
                "placement_price",
                "queue_ahead_quantity",
                "own_quantity",
                "required_printed_quantity",
                "filled",
                "fill_bucket",
                "fill_time_ms",
                "first_matching_trade_id",
                "completion_trade_id",
                "matching_trade_count",
                "printed_quantity_through_fill",
            )
        },
    }


def _validated_float_array(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
    rows: int | None = None,
    nonnegative: bool = False,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} is invalid")
    if raw.size == 0:
        output = np.empty(0, dtype=np.float64)
    elif raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} is invalid")
    else:
        output = np.array(raw, dtype=np.float64, order="C", copy=True)
    if (
        (rows is not None and output.size != rows)
        or not np.isfinite(output).all()
        or (nonnegative and np.any(output < 0.0))
        or (not nonnegative and np.any(output <= 0.0))
    ):
        raise ValueError(f"{name} is invalid")
    return output


def _validated_int64_array(
    value: Sequence[int] | np.ndarray,
    *,
    name: str,
    allow_empty: bool,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or (raw.size == 0 and not allow_empty):
        raise ValueError(f"{name} is invalid")
    if raw.size == 0:
        return np.empty(0, dtype=np.int64)
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{name} is invalid")
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError(f"{name} is invalid")
    output = np.array(raw, dtype=np.int64, order="C", copy=True)
    if np.any(output < 0):
        raise ValueError(f"{name} is invalid")
    return output


def _validated_bool_array(
    value: Sequence[bool] | np.ndarray,
    *,
    name: str,
    rows: int,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size != rows or (raw.size and raw.dtype.kind != "b"):
        raise ValueError(f"{name} is invalid")
    return np.array(raw, dtype=np.bool_, order="C", copy=True)


def _read_only(*arrays: np.ndarray) -> None:
    for array in arrays:
        array.setflags(write=False)


def _source_trade_sha256(
    *,
    trade_id: np.ndarray | None,
    trade_time_ms: np.ndarray,
    trade_price: np.ndarray,
    trade_quantity: np.ndarray,
    trade_buyer_is_maker: np.ndarray,
) -> str:
    arrays = {
        "trade_time_ms": _array_sha256(trade_time_ms),
        "trade_price": _array_sha256(trade_price),
        "trade_quantity": _array_sha256(trade_quantity),
        "trade_buyer_is_maker": _array_sha256(trade_buyer_is_maker),
    }
    if trade_id is not None:
        arrays["trade_id"] = _array_sha256(trade_id)
    return _sha256({"arrays": arrays, "rows": int(trade_time_ms.size)})


@njit(cache=True)
def _search_right_range(
    values: np.ndarray,
    target: int,
    left: int,
    right: int,
) -> int:
    while left < right:
        middle = (left + right) // 2
        if values[middle] <= target:
            left = middle + 1
        else:
            right = middle
    return left


@njit(cache=True)
def _search_left_range(
    values: np.ndarray,
    target: float,
    left: int,
    right: int,
) -> int:
    while left < right:
        middle = (left + right) // 2
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle
    return left


@njit(cache=True)
def _passive_fill_kernel(
    arrivals: np.ndarray,
    required: np.ndarray,
    candidate_bits: np.ndarray,
    candidate_order: np.ndarray,
    trade_ids: np.ndarray,
    trade_times: np.ndarray,
    trade_quantities: np.ndarray,
    trade_price_bits: np.ndarray,
    expiry_ms: int,
    bucket_edges_ms: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rows = arrivals.size
    filled = np.zeros(rows, dtype=np.bool_)
    fill_bucket = np.zeros(rows, dtype=np.uint8)
    fill_time = np.full(rows, -1, dtype=np.int64)
    first_id = np.full(rows, -1, dtype=np.int64)
    completion_id = np.full(rows, -1, dtype=np.int64)
    print_count = np.zeros(rows, dtype=np.uint32)
    printed = np.zeros(rows, dtype=np.float64)
    if trade_times.size == 0:
        return (
            filled,
            fill_bucket,
            fill_time,
            first_id,
            completion_id,
            print_count,
            printed,
        )

    within_price_cumulative = np.empty(trade_times.size, dtype=np.float64)
    running = 0.0
    for trade_index in range(trade_times.size):
        if (
            trade_index == 0
            or trade_price_bits[trade_index] != trade_price_bits[trade_index - 1]
        ):
            running = 0.0
        running += trade_quantities[trade_index]
        within_price_cumulative[trade_index] = running

    candidate_position = 0
    trade_position = 0
    while candidate_position < rows:
        price_key = candidate_bits[candidate_order[candidate_position]]
        candidate_end = candidate_position + 1
        while (
            candidate_end < rows
            and candidate_bits[candidate_order[candidate_end]] == price_key
        ):
            candidate_end += 1

        while (
            trade_position < trade_times.size
            and trade_price_bits[trade_position] < price_key
        ):
            skipped_key = trade_price_bits[trade_position]
            while (
                trade_position < trade_times.size
                and trade_price_bits[trade_position] == skipped_key
            ):
                trade_position += 1
        if (
            trade_position >= trade_times.size
            or trade_price_bits[trade_position] != price_key
        ):
            candidate_position = candidate_end
            continue

        trade_start = trade_position
        while (
            trade_position < trade_times.size
            and trade_price_bits[trade_position] == price_key
        ):
            trade_position += 1
        trade_end = trade_position
        for position in range(candidate_position, candidate_end):
            row = candidate_order[position]
            begin = _search_right_range(
                trade_times,
                arrivals[row],
                trade_start,
                trade_end,
            )
            finish = _search_right_range(
                trade_times,
                arrivals[row] + expiry_ms,
                begin,
                trade_end,
            )
            if begin >= finish:
                continue
            base_quantity = (
                within_price_cumulative[begin - 1] if begin > trade_start else 0.0
            )
            completion = _search_left_range(
                within_price_cumulative,
                base_quantity + required[row],
                begin,
                finish,
            )
            if completion >= finish:
                continue
            delay_ms = trade_times[completion] - arrivals[row]
            filled[row] = True
            for bucket_index in range(bucket_edges_ms.size):
                if delay_ms <= bucket_edges_ms[bucket_index]:
                    fill_bucket[row] = bucket_index + 1
                    break
            fill_time[row] = trade_times[completion]
            first_id[row] = trade_ids[begin]
            completion_id[row] = trade_ids[completion]
            print_count[row] = completion - begin + 1
            printed[row] = within_price_cumulative[completion] - base_quantity
        candidate_position = candidate_end

    return (
        filled,
        fill_bucket,
        fill_time,
        first_id,
        completion_id,
        print_count,
        printed,
    )


def build_passive_fill_result(
    *,
    arrival_time_ms: Sequence[int] | np.ndarray,
    placement_price: Sequence[float] | np.ndarray,
    queue_ahead_quantity: Sequence[float] | np.ndarray,
    buyer_is_maker: bool,
    order_notional_quote: float,
    trade_id: Sequence[int] | np.ndarray,
    trade_time_ms: Sequence[int] | np.ndarray,
    trade_price: Sequence[float] | np.ndarray,
    trade_quantity: Sequence[float] | np.ndarray,
    trade_buyer_is_maker: Sequence[bool] | np.ndarray,
    expiry_ms: int = PASSIVE_FILL_BUCKETS_MS[-1],
) -> PassiveFillResult:
    """Require exact-price post-arrival prints to consume queue plus own size."""

    arrivals = _validated_int64_array(
        arrival_time_ms,
        name="passive-fill arrival times",
        allow_empty=False,
    )
    if np.any(np.diff(arrivals) < 0):
        raise ValueError("passive-fill arrival times are invalid")
    prices = _validated_float_array(
        placement_price, name="passive-fill placement prices", rows=arrivals.size
    )
    queue = _validated_float_array(
        queue_ahead_quantity,
        name="passive-fill queue",
        rows=arrivals.size,
        nonnegative=True,
    )
    if (
        not isinstance(buyer_is_maker, (bool, np.bool_))
        or isinstance(order_notional_quote, (bool, np.bool_))
        or not isinstance(order_notional_quote, Real)
        or isinstance(expiry_ms, (bool, np.bool_))
        or not isinstance(expiry_ms, Integral)
    ):
        raise ValueError("passive-fill execution contract is invalid")
    notional = float(order_notional_quote)
    expiry = int(expiry_ms)
    if (
        not math.isfinite(notional)
        or notional <= 0.0
        or expiry != PASSIVE_FILL_BUCKETS_MS[-1]
    ):
        raise ValueError("passive-fill execution contract is invalid")

    ids_raw = _validated_int64_array(
        trade_id,
        name="passive-fill trade ids",
        allow_empty=True,
    )
    times_raw = _validated_int64_array(
        trade_time_ms,
        name="passive-fill trade times",
        allow_empty=True,
    )
    trade_prices_raw = _validated_float_array(
        trade_price, name="passive-fill trade prices"
    )
    quantities_raw = _validated_float_array(
        trade_quantity, name="passive-fill trade quantities"
    )
    trade_rows = times_raw.size
    sides_raw = _validated_bool_array(
        trade_buyer_is_maker,
        name="passive-fill trade sides",
        rows=trade_rows,
    )
    if (
        ids_raw.size != trade_rows
        or trade_prices_raw.size != trade_rows
        or quantities_raw.size != trade_rows
    ):
        raise ValueError("passive-fill trade arrays are invalid")
    source_trade_sha256 = _source_trade_sha256(
        trade_id=ids_raw,
        trade_time_ms=times_raw,
        trade_price=trade_prices_raw,
        trade_quantity=quantities_raw,
        trade_buyer_is_maker=sides_raw,
    )
    ids = ids_raw
    side_mask = sides_raw == buyer_is_maker
    ids = ids[side_mask]
    times = times_raw[side_mask]
    trade_prices = trade_prices_raw[side_mask]
    quantities = quantities_raw[side_mask]

    own = notional / prices
    required = queue + own
    if ids.size:
        price_bits = np.ascontiguousarray(trade_prices).view(np.uint64)
        order = np.lexsort((ids, times, price_bits))
        ids = ids[order]
        times = times[order]
        quantities = quantities[order]
        price_bits = price_bits[order]
    else:
        price_bits = np.empty(0, dtype=np.uint64)
    candidate_bits = np.ascontiguousarray(prices).view(np.uint64)
    candidate_order = np.argsort(candidate_bits, kind="stable")
    (
        filled,
        bucket,
        fill_time,
        first_id,
        completion_id,
        print_count,
        printed,
    ) = _passive_fill_kernel(
        arrivals,
        required,
        candidate_bits,
        candidate_order,
        ids,
        times,
        quantities,
        price_bits,
        expiry,
        np.asarray(PASSIVE_FILL_BUCKETS_MS, dtype=np.int64),
    )
    _read_only(
        arrivals,
        prices,
        queue,
        own,
        required,
        filled,
        bucket,
        fill_time,
        first_id,
        completion_id,
        print_count,
        printed,
    )

    provisional = PassiveFillResult(
        schema_version=PASSIVE_FILL_SCHEMA_VERSION,
        buyer_is_maker=buyer_is_maker,
        expiry_ms=expiry,
        order_notional_quote=notional,
        source_trade_sha256=source_trade_sha256,
        arrival_time_ms=arrivals,
        placement_price=prices,
        queue_ahead_quantity=queue,
        own_quantity=own,
        required_printed_quantity=required,
        filled=filled,
        fill_bucket=bucket,
        fill_time_ms=fill_time,
        first_matching_trade_id=first_id,
        completion_trade_id=completion_id,
        matching_trade_count=print_count,
        printed_quantity_through_fill=printed,
        result_sha256="",
    )
    result = replace(provisional, result_sha256=_sha256(_passive_fill_payload(provisional)))
    if (
        np.any(result.filled != (result.fill_bucket > 0))
        or np.any(result.filled != (result.fill_time_ms >= 0))
        or np.any(result.fill_bucket > len(PASSIVE_FILL_BUCKETS_MS))
        or np.any(result.printed_quantity_through_fill[result.filled] + 1e-12 < required[result.filled])
        or _sha256(_passive_fill_payload(result)) != result.result_sha256
    ):
        raise RuntimeError("passive-fill result invariant failed")
    return result


@dataclass(frozen=True)
class ExponentialFlowBatch:
    schema_version: str
    observation_delay_ms: int
    half_lives_seconds: tuple[int, ...]
    feature_names: tuple[str, ...]
    source_trade_sha256: str
    decision_time_ms: np.ndarray
    features: np.ndarray
    batch_sha256: str


@njit(cache=True)
def _exponential_flow_kernel(
    decisions: np.ndarray,
    trade_times: np.ndarray,
    trade_quote: np.ndarray,
    aggressive_buy: np.ndarray,
    half_lives_ms: np.ndarray,
    observation_delay_ms: int,
) -> np.ndarray:
    output = np.empty((decisions.size, half_lives_ms.size * 2), dtype=np.float64)
    buys = np.zeros(half_lives_ms.size, dtype=np.float64)
    sells = np.zeros(half_lives_ms.size, dtype=np.float64)
    event_index = 0
    state_time = trade_times[0] if trade_times.size else decisions[0] - observation_delay_ms
    log_two = math.log(2.0)
    for row in range(decisions.size):
        cutoff = decisions[row] - observation_delay_ms
        while event_index < trade_times.size and trade_times[event_index] <= cutoff:
            event_time = trade_times[event_index]
            elapsed = max(0, event_time - state_time)
            for horizon in range(half_lives_ms.size):
                decay = math.exp(-log_two * elapsed / half_lives_ms[horizon])
                buys[horizon] *= decay
                sells[horizon] *= decay
            if aggressive_buy[event_index]:
                buys += trade_quote[event_index]
            else:
                sells += trade_quote[event_index]
            state_time = event_time
            event_index += 1
        elapsed = max(0, cutoff - state_time)
        for horizon in range(half_lives_ms.size):
            decay = math.exp(-log_two * elapsed / half_lives_ms[horizon])
            buys[horizon] *= decay
            sells[horizon] *= decay
            total = buys[horizon] + sells[horizon]
            output[row, horizon * 2] = math.log1p(
                total / (half_lives_ms[horizon] / 1000.0)
            )
            output[row, horizon * 2 + 1] = (
                (buys[horizon] - sells[horizon]) / total if total > 0.0 else 0.0
            )
        state_time = cutoff
    return output


def build_exponential_flow_features(
    *,
    decision_time_ms: Sequence[int] | np.ndarray,
    trade_time_ms: Sequence[int] | np.ndarray,
    trade_price: Sequence[float] | np.ndarray,
    trade_quantity: Sequence[float] | np.ndarray,
    trade_buyer_is_maker: Sequence[bool] | np.ndarray,
    observation_delay_ms: int = EXPONENTIAL_FLOW_OBSERVATION_DELAY_MS,
    half_lives_seconds: Sequence[int] = EXPONENTIAL_FLOW_HALF_LIVES_SECONDS,
) -> ExponentialFlowBatch:
    """Build event-time flow state using only prints observable by each decision."""

    decisions = _validated_int64_array(
        decision_time_ms,
        name="flow decision times",
        allow_empty=False,
    )
    times = _validated_int64_array(
        trade_time_ms,
        name="flow trade times",
        allow_empty=True,
    )
    prices = _validated_float_array(trade_price, name="flow trade prices")
    quantities = _validated_float_array(trade_quantity, name="flow trade quantities")
    sides = _validated_bool_array(
        trade_buyer_is_maker,
        name="flow trade sides",
        rows=times.size,
    )
    raw_half_lives = tuple(half_lives_seconds)
    if (
        isinstance(observation_delay_ms, (bool, np.bool_))
        or not isinstance(observation_delay_ms, Integral)
        or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            for value in raw_half_lives
        )
    ):
        raise ValueError("causal exponential-flow contract is invalid")
    delay = int(observation_delay_ms)
    half_lives = tuple(int(value) for value in raw_half_lives)
    if (
        decisions.ndim != 1
        or decisions.size == 0
        or np.any(np.diff(decisions) <= 0)
        or prices.size != times.size
        or quantities.size != times.size
        or delay != EXPONENTIAL_FLOW_OBSERVATION_DELAY_MS
        or half_lives != EXPONENTIAL_FLOW_HALF_LIVES_SECONDS
    ):
        raise ValueError("causal exponential-flow contract is invalid")
    order = np.argsort(times, kind="stable")
    source_trade_sha256 = _source_trade_sha256(
        trade_id=None,
        trade_time_ms=times,
        trade_price=prices,
        trade_quantity=quantities,
        trade_buyer_is_maker=sides,
    )
    times = times[order]
    quote = prices[order] * quantities[order]
    aggressive_buy = ~sides[order]
    features = _exponential_flow_kernel(
        decisions,
        times,
        quote,
        aggressive_buy,
        np.asarray(half_lives, dtype=np.float64) * 1_000.0,
        delay,
    )
    names = tuple(
        name
        for half_life in half_lives
        for name in (
            f"flow_log_quote_intensity_h{half_life}s",
            f"flow_imbalance_h{half_life}s",
        )
    )
    stored_features = np.ascontiguousarray(features, dtype=np.float32)
    _read_only(decisions, stored_features)
    payload = {
        "schema_version": EXPONENTIAL_FLOW_SCHEMA_VERSION,
        "observation_delay_ms": delay,
        "half_lives_seconds": list(half_lives),
        "feature_names": list(names),
        "source_trade_sha256": source_trade_sha256,
        "decision_time_ms_sha256": _array_sha256(decisions),
        "features_sha256": _array_sha256(stored_features),
    }
    batch = ExponentialFlowBatch(
        schema_version=EXPONENTIAL_FLOW_SCHEMA_VERSION,
        observation_delay_ms=delay,
        half_lives_seconds=half_lives,
        feature_names=names,
        source_trade_sha256=source_trade_sha256,
        decision_time_ms=decisions,
        features=stored_features,
        batch_sha256=_sha256(payload),
    )
    if (
        batch.features.shape != (decisions.size, len(names))
        or not np.isfinite(batch.features).all()
        or _sha256(payload) != batch.batch_sha256
    ):
        raise RuntimeError("causal exponential-flow result invariant failed")
    return batch


__all__ = [
    "EXPONENTIAL_FLOW_HALF_LIVES_SECONDS",
    "EXPONENTIAL_FLOW_OBSERVATION_DELAY_MS",
    "EXPONENTIAL_FLOW_SCHEMA_VERSION",
    "PASSIVE_FILL_BUCKETS_MS",
    "PASSIVE_FILL_SCHEMA_VERSION",
    "ExponentialFlowBatch",
    "PassiveFillResult",
    "build_exponential_flow_features",
    "build_passive_fill_result",
]
