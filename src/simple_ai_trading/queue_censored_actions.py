"""Causal queue-censored targets and flow features for make/take research."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Callable, Mapping, Sequence

from numba import njit
import numpy as np


PASSIVE_FILL_SCHEMA_VERSION = "queue-censored-passive-fill-v1"
EXPONENTIAL_FLOW_SCHEMA_VERSION = "causal-exponential-trade-flow-v1"
PASSIVE_FILL_BUCKETS_MS = (5_000, 10_000, 15_000)
EXPONENTIAL_FLOW_HALF_LIVES_SECONDS = (1, 2, 5, 10, 30, 60)
EXPONENTIAL_FLOW_OBSERVATION_DELAY_MS = 1_000
CHUNKED_QUEUE_INPUT_SCHEMA_VERSION = "chunked-queue-censored-action-inputs-v1"
CHUNKED_TRADE_SOURCE_SCHEMA_VERSION = "chunked-exact-trade-source-v1"
_FILL_OUTPUT_NAMES = (
    "filled",
    "fill_bucket",
    "fill_time_ms",
    "first_matching_trade_id",
    "completion_trade_id",
    "matching_trade_count",
    "printed_quantity_through_fill",
)


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


@dataclass(frozen=True)
class PassiveFillRequest:
    """One exact passive-order stream evaluated against shared trade chunks."""

    name: str
    buyer_is_maker: bool
    arrival_time_ms: Sequence[int] | np.ndarray
    placement_price: Sequence[float] | np.ndarray
    queue_ahead_quantity: Sequence[float] | np.ndarray


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


def validate_passive_fill_result(result: PassiveFillResult) -> None:
    """Recompute the complete passive-fill identity and lifecycle invariants."""

    rows = result.rows
    vectors = (
        result.arrival_time_ms,
        result.placement_price,
        result.queue_ahead_quantity,
        result.own_quantity,
        result.required_printed_quantity,
        result.filled,
        result.fill_bucket,
        result.fill_time_ms,
        result.first_matching_trade_id,
        result.completion_trade_id,
        result.matching_trade_count,
        result.printed_quantity_through_fill,
    )
    source_hash = str(result.source_trade_sha256)
    result_hash = str(result.result_sha256)
    if (
        result.schema_version != PASSIVE_FILL_SCHEMA_VERSION
        or not isinstance(result.buyer_is_maker, (bool, np.bool_))
        or result.expiry_ms != PASSIVE_FILL_BUCKETS_MS[-1]
        or not math.isfinite(result.order_notional_quote)
        or result.order_notional_quote <= 0.0
        or rows <= 0
        or any(np.asarray(value).shape != (rows,) for value in vectors)
        or np.asarray(result.arrival_time_ms).dtype != np.dtype(np.int64)
        or np.asarray(result.placement_price).dtype != np.dtype(np.float64)
        or np.asarray(result.queue_ahead_quantity).dtype != np.dtype(np.float64)
        or np.asarray(result.own_quantity).dtype != np.dtype(np.float64)
        or np.asarray(result.required_printed_quantity).dtype != np.dtype(np.float64)
        or np.asarray(result.filled).dtype != np.dtype(np.bool_)
        or np.asarray(result.fill_bucket).dtype != np.dtype(np.uint8)
        or np.asarray(result.fill_time_ms).dtype != np.dtype(np.int64)
        or np.asarray(result.first_matching_trade_id).dtype != np.dtype(np.int64)
        or np.asarray(result.completion_trade_id).dtype != np.dtype(np.int64)
        or np.asarray(result.matching_trade_count).dtype != np.dtype(np.uint32)
        or np.asarray(result.printed_quantity_through_fill).dtype
        != np.dtype(np.float64)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdef" for character in source_hash)
        or len(result_hash) != 64
        or any(character not in "0123456789abcdef" for character in result_hash)
    ):
        raise ValueError("passive-fill result contract is invalid")
    arrivals = np.asarray(result.arrival_time_ms)
    prices = np.asarray(result.placement_price)
    queue = np.asarray(result.queue_ahead_quantity)
    filled = np.asarray(result.filled)
    unfilled = ~filled
    delay = np.asarray(result.fill_time_ms)[filled] - arrivals[filled]
    expected_bucket = (
        np.searchsorted(PASSIVE_FILL_BUCKETS_MS, delay, side="left") + 1
    ).astype(np.uint8)
    if (
        np.any(np.diff(arrivals) < 0)
        or not np.isfinite(prices).all()
        or np.any(prices <= 0.0)
        or not np.isfinite(queue).all()
        or np.any(queue < 0.0)
        or not np.allclose(
            result.own_quantity,
            result.order_notional_quote / prices,
            rtol=0.0,
            atol=0.0,
        )
        or not np.allclose(
            result.required_printed_quantity,
            queue + result.own_quantity,
            rtol=0.0,
            atol=0.0,
        )
        or np.any(result.filled != (result.fill_bucket > 0))
        or np.any(result.filled != (result.fill_time_ms >= 0))
        or np.any(result.fill_bucket > len(PASSIVE_FILL_BUCKETS_MS))
        or not np.array_equal(result.fill_bucket[filled], expected_bucket)
        or np.any(delay <= 0)
        or np.any(delay > result.expiry_ms)
        or np.any(result.first_matching_trade_id[filled] < 0)
        or np.any(result.completion_trade_id[filled] < 0)
        or np.any(result.matching_trade_count[filled] == 0)
        or np.any(
            result.printed_quantity_through_fill[filled] + 1e-12
            < result.required_printed_quantity[filled]
        )
        or np.any(result.fill_bucket[unfilled] != 0)
        or np.any(result.fill_time_ms[unfilled] != -1)
        or np.any(result.first_matching_trade_id[unfilled] != -1)
        or np.any(result.completion_trade_id[unfilled] != -1)
        or np.any(result.matching_trade_count[unfilled] != 0)
        or np.any(result.printed_quantity_through_fill[unfilled] != 0.0)
        or _sha256(_passive_fill_payload(result)) != result_hash
    ):
        raise ValueError("passive-fill result lifecycle is invalid")


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
        buyer_is_maker=bool(buyer_is_maker),
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
    validate_passive_fill_result(result)
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


@dataclass(frozen=True)
class TradeChunkEvidence:
    start_ms: int
    end_ms: int
    rows: int
    source_trade_sha256: str
    first_trade_id: int | None
    last_trade_id: int | None
    first_trade_time_ms: int | None
    last_trade_time_ms: int | None


@dataclass(frozen=True)
class ChunkedQueueCensoredInputBatch:
    """Bounded-memory causal flow and passive-fill evidence from one source pass."""

    schema_version: str
    source_start_ms: int
    source_end_ms: int
    source_trade_rows: int
    source_trade_sha256: str
    source_chunks: tuple[TradeChunkEvidence, ...]
    flow: ExponentialFlowBatch
    fills: tuple[tuple[str, PassiveFillResult], ...]
    batch_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False

    def fill(self, name: str) -> PassiveFillResult:
        matches = [value for key, value in self.fills if key == name]
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_start_ms": self.source_start_ms,
            "source_end_ms": self.source_end_ms,
            "source_trade_rows": self.source_trade_rows,
            "source_trade_sha256": self.source_trade_sha256,
            "chunk_count": len(self.source_chunks),
            "decision_rows": int(self.flow.decision_time_ms.size),
            "fills": {name: value.summary() for name, value in self.fills},
            "batch_sha256": self.batch_sha256,
            "trading_authority": False,
            "profitability_claim": False,
        }


TradeChunkLoader = Callable[[int, int], Mapping[str, Sequence[object] | np.ndarray]]


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


@njit(cache=True)
def _exponential_flow_stateful_kernel(
    decisions: np.ndarray,
    trade_times: np.ndarray,
    trade_quote: np.ndarray,
    aggressive_buy: np.ndarray,
    half_lives_ms: np.ndarray,
    observation_delay_ms: int,
    buys: np.ndarray,
    sells: np.ndarray,
    state_time_ms: int,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, int]:
    output = np.empty((decisions.size, half_lives_ms.size * 2), dtype=np.float64)
    event_index = 0
    state_time = state_time_ms
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
    return output, event_index, buys, sells, state_time


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


def _exponential_flow_payload(batch: ExponentialFlowBatch) -> dict[str, object]:
    return {
        "schema_version": batch.schema_version,
        "observation_delay_ms": batch.observation_delay_ms,
        "half_lives_seconds": list(batch.half_lives_seconds),
        "feature_names": list(batch.feature_names),
        "source_trade_sha256": batch.source_trade_sha256,
        "decision_time_ms_sha256": _array_sha256(batch.decision_time_ms),
        "features_sha256": _array_sha256(batch.features),
    }


def validate_exponential_flow_batch(batch: ExponentialFlowBatch) -> None:
    """Recompute the causal-flow artifact identity and numeric bounds."""

    decisions = np.asarray(batch.decision_time_ms)
    features = np.asarray(batch.features)
    expected_names = tuple(
        name
        for half_life in EXPONENTIAL_FLOW_HALF_LIVES_SECONDS
        for name in (
            f"flow_log_quote_intensity_h{half_life}s",
            f"flow_imbalance_h{half_life}s",
        )
    )
    source_hash = str(batch.source_trade_sha256)
    batch_hash = str(batch.batch_sha256)
    if (
        batch.schema_version != EXPONENTIAL_FLOW_SCHEMA_VERSION
        or batch.observation_delay_ms != EXPONENTIAL_FLOW_OBSERVATION_DELAY_MS
        or batch.half_lives_seconds != EXPONENTIAL_FLOW_HALF_LIVES_SECONDS
        or batch.feature_names != expected_names
        or decisions.ndim != 1
        or decisions.size == 0
        or decisions.dtype != np.dtype(np.int64)
        or np.any(np.diff(decisions) <= 0)
        or features.shape != (decisions.size, len(expected_names))
        or features.dtype != np.dtype(np.float32)
        or not np.isfinite(features).all()
        or np.any(features[:, 0::2] < 0.0)
        or np.any(np.abs(features[:, 1::2]) > 1.0 + 1e-7)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdef" for character in source_hash)
        or len(batch_hash) != 64
        or any(character not in "0123456789abcdef" for character in batch_hash)
        or _sha256(_exponential_flow_payload(batch)) != batch_hash
    ):
        raise ValueError("causal exponential-flow batch is invalid")


def _fill_output_arrays(rows: int) -> dict[str, np.ndarray]:
    return {
        "filled": np.zeros(rows, dtype=np.bool_),
        "fill_bucket": np.zeros(rows, dtype=np.uint8),
        "fill_time_ms": np.full(rows, -1, dtype=np.int64),
        "first_matching_trade_id": np.full(rows, -1, dtype=np.int64),
        "completion_trade_id": np.full(rows, -1, dtype=np.int64),
        "matching_trade_count": np.zeros(rows, dtype=np.uint32),
        "printed_quantity_through_fill": np.zeros(rows, dtype=np.float64),
    }


def _finalize_passive_fill_result(
    *,
    buyer_is_maker: bool,
    order_notional_quote: float,
    source_trade_sha256: str,
    arrival_time_ms: np.ndarray,
    placement_price: np.ndarray,
    queue_ahead_quantity: np.ndarray,
    output: Mapping[str, np.ndarray],
) -> PassiveFillResult:
    own = order_notional_quote / placement_price
    required = queue_ahead_quantity + own
    _read_only(
        arrival_time_ms,
        placement_price,
        queue_ahead_quantity,
        own,
        required,
        *(output[name] for name in _FILL_OUTPUT_NAMES),
    )
    provisional = PassiveFillResult(
        schema_version=PASSIVE_FILL_SCHEMA_VERSION,
        buyer_is_maker=buyer_is_maker,
        expiry_ms=PASSIVE_FILL_BUCKETS_MS[-1],
        order_notional_quote=order_notional_quote,
        source_trade_sha256=source_trade_sha256,
        arrival_time_ms=arrival_time_ms,
        placement_price=placement_price,
        queue_ahead_quantity=queue_ahead_quantity,
        own_quantity=own,
        required_printed_quantity=required,
        filled=output["filled"],
        fill_bucket=output["fill_bucket"],
        fill_time_ms=output["fill_time_ms"],
        first_matching_trade_id=output["first_matching_trade_id"],
        completion_trade_id=output["completion_trade_id"],
        matching_trade_count=output["matching_trade_count"],
        printed_quantity_through_fill=output["printed_quantity_through_fill"],
        result_sha256="",
    )
    result = replace(
        provisional,
        result_sha256=_sha256(_passive_fill_payload(provisional)),
    )
    validate_passive_fill_result(result)
    return result


def _chunked_batch_payload(
    batch: ChunkedQueueCensoredInputBatch,
) -> dict[str, object]:
    return {
        "schema_version": batch.schema_version,
        "source_start_ms": batch.source_start_ms,
        "source_end_ms": batch.source_end_ms,
        "source_trade_rows": batch.source_trade_rows,
        "source_trade_sha256": batch.source_trade_sha256,
        "source_chunks": [asdict(value) for value in batch.source_chunks],
        "flow_sha256": batch.flow.batch_sha256,
        "fills": [
            {"name": name, "result_sha256": value.result_sha256}
            for name, value in batch.fills
        ],
        "trading_authority": batch.trading_authority,
        "execution_claim": batch.execution_claim,
        "profitability_claim": batch.profitability_claim,
    }


def validate_chunked_queue_censored_inputs(
    batch: ChunkedQueueCensoredInputBatch,
) -> None:
    """Validate chunk coverage, shared source identity, and all child artifacts."""

    validate_exponential_flow_batch(batch.flow)
    chunks = batch.source_chunks
    fills = batch.fills
    names = tuple(name for name, _value in fills)
    for _name, value in fills:
        validate_passive_fill_result(value)
    if (
        batch.schema_version != CHUNKED_QUEUE_INPUT_SCHEMA_VERSION
        or batch.source_start_ms < 0
        or batch.source_end_ms <= batch.source_start_ms
        or batch.source_trade_rows < 0
        or not chunks
        or chunks[0].start_ms != batch.source_start_ms
        or chunks[-1].end_ms != batch.source_end_ms
        or any(
            left.end_ms != right.start_ms
            for left, right in zip(chunks, chunks[1:])
        )
        or any(chunk.start_ms >= chunk.end_ms or chunk.rows < 0 for chunk in chunks)
        or sum(chunk.rows for chunk in chunks) != batch.source_trade_rows
        or tuple(sorted(names)) != names
        or len(set(names)) != len(names)
        or not names
        or any(
            value.source_trade_sha256 != batch.source_trade_sha256
            for _name, value in fills
        )
        or batch.flow.source_trade_sha256 != batch.source_trade_sha256
        or batch.trading_authority is not False
        or batch.execution_claim is not False
        or batch.profitability_claim is not False
        or _sha256(
            {
                "schema_version": CHUNKED_TRADE_SOURCE_SCHEMA_VERSION,
                "chunks": [asdict(value) for value in chunks],
            }
        )
        != batch.source_trade_sha256
        or _sha256(_chunked_batch_payload(batch)) != batch.batch_sha256
    ):
        raise ValueError("chunked queue-censored input batch is invalid")


def build_chunked_queue_censored_inputs(
    *,
    decision_time_ms: Sequence[int] | np.ndarray,
    fill_requests: Sequence[PassiveFillRequest],
    source_chunks: Sequence[tuple[int, int]],
    load_trade_chunk: TradeChunkLoader,
    order_notional_quote: float = 1_000.0,
) -> ChunkedQueueCensoredInputBatch:
    """Build exact flow and fill evidence while bounding raw-trade memory by chunk."""

    decisions = _validated_int64_array(
        decision_time_ms,
        name="chunked flow decision times",
        allow_empty=False,
    )
    raw_bounds = tuple(source_chunks)
    if (
        np.any(np.diff(decisions) <= 0)
        or not raw_bounds
        or not callable(load_trade_chunk)
        or isinstance(order_notional_quote, (bool, np.bool_))
        or not isinstance(order_notional_quote, Real)
        or not math.isfinite(float(order_notional_quote))
        or float(order_notional_quote) <= 0.0
    ):
        raise ValueError("chunked queue-censored source contract is invalid")
    bounds: list[tuple[int, int]] = []
    for raw_start, raw_end in raw_bounds:
        if (
            isinstance(raw_start, (bool, np.bool_))
            or isinstance(raw_end, (bool, np.bool_))
            or not isinstance(raw_start, Integral)
            or not isinstance(raw_end, Integral)
        ):
            raise ValueError("chunked trade bounds are invalid")
        bounds.append((int(raw_start), int(raw_end)))
    if (
        bounds[0][0] < 0
        or any(start >= end for start, end in bounds)
        or any(left[1] != right[0] for left, right in zip(bounds, bounds[1:]))
    ):
        raise ValueError("chunked trade bounds are invalid")
    source_start = bounds[0][0]
    source_end = bounds[-1][1]
    cutoffs = decisions - EXPONENTIAL_FLOW_OBSERVATION_DELAY_MS
    if cutoffs[0] < source_start or cutoffs[-1] >= source_end:
        raise ValueError("chunked flow decisions escape the source interval")

    request_rows: list[dict[str, object]] = []
    ordered_requests = sorted(tuple(fill_requests), key=lambda value: value.name)
    for request in ordered_requests:
        name = str(request.name)
        arrivals = _validated_int64_array(
            request.arrival_time_ms,
            name=f"{name} arrivals",
            allow_empty=False,
        )
        prices = _validated_float_array(
            request.placement_price,
            name=f"{name} placement prices",
            rows=decisions.size,
        )
        queues = _validated_float_array(
            request.queue_ahead_quantity,
            name=f"{name} queue",
            rows=decisions.size,
            nonnegative=True,
        )
        if (
            not name
            or not name.isascii()
            or any(
                not (character.isalnum() or character == "_")
                for character in name
            )
            or not isinstance(request.buyer_is_maker, (bool, np.bool_))
            or arrivals.size != decisions.size
            or np.any(np.diff(arrivals) < 0)
            or arrivals[0] < source_start
            or arrivals[-1] + PASSIVE_FILL_BUCKETS_MS[-1] >= source_end
        ):
            raise ValueError("chunked passive-fill request is invalid")
        request_rows.append(
            {
                "name": name,
                "buyer_is_maker": bool(request.buyer_is_maker),
                "arrivals": arrivals,
                "prices": prices,
                "queues": queues,
                "assigned": np.zeros(decisions.size, dtype=np.bool_),
                "output": _fill_output_arrays(decisions.size),
            }
        )
    request_names = tuple(str(value["name"]) for value in request_rows)
    if not request_rows or len(set(request_names)) != len(request_names):
        raise ValueError("chunked passive-fill request names are invalid")

    flow_features = np.empty(
        (decisions.size, len(EXPONENTIAL_FLOW_HALF_LIVES_SECONDS) * 2),
        dtype=np.float64,
    )
    half_lives_ms = (
        np.asarray(EXPONENTIAL_FLOW_HALF_LIVES_SECONDS, dtype=np.float64) * 1_000.0
    )
    buys = np.zeros(len(half_lives_ms), dtype=np.float64)
    sells = np.zeros(len(half_lives_ms), dtype=np.float64)
    state_time = source_start
    flow_position = 0
    carry_times = np.empty(0, dtype=np.int64)
    carry_prices = np.empty(0, dtype=np.float64)
    carry_quantities = np.empty(0, dtype=np.float64)
    carry_sides = np.empty(0, dtype=np.bool_)
    chunk_evidence: list[TradeChunkEvidence] = []
    previous_trade_id: int | None = None

    for start, end in bounds:
        loaded = dict(load_trade_chunk(start, end))
        required_fields = {
            "trade_id",
            "trade_time_ms",
            "trade_price",
            "trade_quantity",
            "trade_buyer_is_maker",
        }
        if set(loaded) != required_fields:
            raise ValueError("chunked trade loader fields are invalid")
        trade_ids = _validated_int64_array(
            loaded["trade_id"], name="chunked trade ids", allow_empty=True
        )
        trade_times = _validated_int64_array(
            loaded["trade_time_ms"], name="chunked trade times", allow_empty=True
        )
        trade_prices = _validated_float_array(
            loaded["trade_price"], name="chunked trade prices"
        )
        trade_quantities = _validated_float_array(
            loaded["trade_quantity"], name="chunked trade quantities"
        )
        trade_sides = _validated_bool_array(
            loaded["trade_buyer_is_maker"],
            name="chunked trade sides",
            rows=trade_times.size,
        )
        rows = trade_times.size
        if (
            trade_ids.size != rows
            or trade_prices.size != rows
            or trade_quantities.size != rows
            or (rows and (trade_times[0] < start or trade_times[-1] >= end))
            or (rows and np.any(trade_times[1:] < trade_times[:-1]))
            or (
                rows > 1
                and np.any(
                    (trade_times[1:] == trade_times[:-1])
                    & (trade_ids[1:] <= trade_ids[:-1])
                )
            )
            or (
                rows
                and previous_trade_id is not None
                and trade_ids[0] <= previous_trade_id
            )
            or (rows > 1 and np.any(trade_ids[1:] <= trade_ids[:-1]))
        ):
            raise ValueError("chunked trade rows are invalid or unordered")
        if rows:
            previous_trade_id = int(trade_ids[-1])
        chunk_hash = _source_trade_sha256(
            trade_id=trade_ids,
            trade_time_ms=trade_times,
            trade_price=trade_prices,
            trade_quantity=trade_quantities,
            trade_buyer_is_maker=trade_sides,
        )
        chunk_evidence.append(
            TradeChunkEvidence(
                start_ms=start,
                end_ms=end,
                rows=rows,
                source_trade_sha256=chunk_hash,
                first_trade_id=int(trade_ids[0]) if rows else None,
                last_trade_id=int(trade_ids[-1]) if rows else None,
                first_trade_time_ms=int(trade_times[0]) if rows else None,
                last_trade_time_ms=int(trade_times[-1]) if rows else None,
            )
        )

        if carry_times.size:
            available_times = np.concatenate((carry_times, trade_times))
            available_prices = np.concatenate((carry_prices, trade_prices))
            available_quantities = np.concatenate(
                (carry_quantities, trade_quantities)
            )
            available_sides = np.concatenate((carry_sides, trade_sides))
        else:
            available_times = trade_times
            available_prices = trade_prices
            available_quantities = trade_quantities
            available_sides = trade_sides
        flow_end = int(np.searchsorted(cutoffs, end, side="left"))
        if flow_end > flow_position:
            selected_decisions = decisions[flow_position:flow_end]
            values, consumed, buys, sells, state_time = (
                _exponential_flow_stateful_kernel(
                    selected_decisions,
                    available_times,
                    available_prices * available_quantities,
                    ~available_sides,
                    half_lives_ms,
                    EXPONENTIAL_FLOW_OBSERVATION_DELAY_MS,
                    buys,
                    sells,
                    state_time,
                )
            )
            flow_features[flow_position:flow_end] = values
            flow_position = flow_end
        else:
            consumed = 0
        carry_times = np.ascontiguousarray(available_times[consumed:])
        carry_prices = np.ascontiguousarray(available_prices[consumed:])
        carry_quantities = np.ascontiguousarray(available_quantities[consumed:])
        carry_sides = np.ascontiguousarray(available_sides[consumed:])

        side_data: dict[bool, tuple[np.ndarray, ...]] = {}
        for side in (False, True):
            selected = np.flatnonzero(trade_sides == side)
            ids = trade_ids[selected]
            times = trade_times[selected]
            quantities = trade_quantities[selected]
            price_bits = np.ascontiguousarray(trade_prices[selected]).view(np.uint64)
            if selected.size:
                order = np.lexsort((ids, times, price_bits))
                ids = ids[order]
                times = times[order]
                quantities = quantities[order]
                price_bits = price_bits[order]
            side_data[side] = (ids, times, quantities, price_bits)
        for request in request_rows:
            arrivals = request["arrivals"]
            indexes = np.flatnonzero(
                (arrivals >= start)
                & (arrivals + PASSIVE_FILL_BUCKETS_MS[-1] < end)
            )
            if indexes.size == 0:
                continue
            assigned = request["assigned"]
            if np.any(assigned[indexes]):
                raise RuntimeError("chunked passive-fill rows were assigned twice")
            prices = request["prices"]
            queues = request["queues"]
            candidate_prices = prices[indexes]
            candidate_bits = np.ascontiguousarray(candidate_prices).view(np.uint64)
            candidate_order = np.argsort(candidate_bits, kind="stable")
            ids, times, quantities, price_bits = side_data[
                bool(request["buyer_is_maker"])
            ]
            result = _passive_fill_kernel(
                arrivals[indexes],
                queues[indexes] + float(order_notional_quote) / candidate_prices,
                candidate_bits,
                candidate_order,
                ids,
                times,
                quantities,
                price_bits,
                PASSIVE_FILL_BUCKETS_MS[-1],
                np.asarray(PASSIVE_FILL_BUCKETS_MS, dtype=np.int64),
            )
            output = request["output"]
            for name, values in zip(_FILL_OUTPUT_NAMES, result):
                output[name][indexes] = values
            assigned[indexes] = True

    if flow_position != decisions.size:
        raise RuntimeError("chunked flow did not consume every decision")
    if any(not np.all(request["assigned"]) for request in request_rows):
        raise ValueError("passive-fill windows cross a chunk or role boundary")
    chunks = tuple(chunk_evidence)
    source_hash = _sha256(
        {
            "schema_version": CHUNKED_TRADE_SOURCE_SCHEMA_VERSION,
            "chunks": [asdict(value) for value in chunks],
        }
    )
    flow_names = tuple(
        name
        for half_life in EXPONENTIAL_FLOW_HALF_LIVES_SECONDS
        for name in (
            f"flow_log_quote_intensity_h{half_life}s",
            f"flow_imbalance_h{half_life}s",
        )
    )
    stored_decisions = np.ascontiguousarray(decisions, dtype=np.int64)
    stored_features = np.ascontiguousarray(flow_features, dtype=np.float32)
    _read_only(stored_decisions, stored_features)
    flow = ExponentialFlowBatch(
        schema_version=EXPONENTIAL_FLOW_SCHEMA_VERSION,
        observation_delay_ms=EXPONENTIAL_FLOW_OBSERVATION_DELAY_MS,
        half_lives_seconds=EXPONENTIAL_FLOW_HALF_LIVES_SECONDS,
        feature_names=flow_names,
        source_trade_sha256=source_hash,
        decision_time_ms=stored_decisions,
        features=stored_features,
        batch_sha256="",
    )
    flow = replace(flow, batch_sha256=_sha256(_exponential_flow_payload(flow)))
    validate_exponential_flow_batch(flow)
    fills = tuple(
        (
            str(request["name"]),
            _finalize_passive_fill_result(
                buyer_is_maker=bool(request["buyer_is_maker"]),
                order_notional_quote=float(order_notional_quote),
                source_trade_sha256=source_hash,
                arrival_time_ms=request["arrivals"],
                placement_price=request["prices"],
                queue_ahead_quantity=request["queues"],
                output=request["output"],
            ),
        )
        for request in request_rows
    )
    provisional = ChunkedQueueCensoredInputBatch(
        schema_version=CHUNKED_QUEUE_INPUT_SCHEMA_VERSION,
        source_start_ms=source_start,
        source_end_ms=source_end,
        source_trade_rows=sum(value.rows for value in chunks),
        source_trade_sha256=source_hash,
        source_chunks=chunks,
        flow=flow,
        fills=fills,
        batch_sha256="",
    )
    batch = replace(
        provisional,
        batch_sha256=_sha256(_chunked_batch_payload(provisional)),
    )
    validate_chunked_queue_censored_inputs(batch)
    return batch


__all__ = [
    "CHUNKED_QUEUE_INPUT_SCHEMA_VERSION",
    "CHUNKED_TRADE_SOURCE_SCHEMA_VERSION",
    "ChunkedQueueCensoredInputBatch",
    "EXPONENTIAL_FLOW_HALF_LIVES_SECONDS",
    "EXPONENTIAL_FLOW_OBSERVATION_DELAY_MS",
    "EXPONENTIAL_FLOW_SCHEMA_VERSION",
    "PASSIVE_FILL_BUCKETS_MS",
    "PASSIVE_FILL_SCHEMA_VERSION",
    "ExponentialFlowBatch",
    "PassiveFillRequest",
    "PassiveFillResult",
    "TradeChunkEvidence",
    "TradeChunkLoader",
    "build_chunked_queue_censored_inputs",
    "build_exponential_flow_features",
    "build_passive_fill_result",
    "validate_chunked_queue_censored_inputs",
    "validate_exponential_flow_batch",
    "validate_passive_fill_result",
]
