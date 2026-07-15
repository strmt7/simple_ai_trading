"""Frozen base/stress order-entry states for Round 57 make/take research."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np

from .make_take_action_features import MAKE_TAKE_ACTION_NAMES
from .queue_censored_actions import PASSIVE_FILL_BUCKETS_MS, PassiveFillResult


MAKE_TAKE_SCENARIO_ENTRY_SCHEMA_VERSION = "queue-censored-scenario-entry-v1"
MAKE_TAKE_SCENARIOS = ("base", "stress")
MAKE_TAKE_ORDER_NOTIONAL_QUOTE = 1_000.0
MAKE_TAKE_MAX_L1_PARTICIPATION = 0.10
_ACTION_CODE_PATTERN = np.arange(4, dtype=np.uint8)
_ACTION_SIDE_PATTERN = np.asarray([1, -1, 1, -1], dtype=np.int8)


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


def _int_array(value: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.ndim != 1
        or raw.size == 0
        or raw.dtype.kind not in "iu"
        or (raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max))
    ):
        raise ValueError(f"{name} is invalid")
    output = np.array(raw, dtype=np.int64, order="C", copy=True)
    if np.any(output < 0):
        raise ValueError(f"{name} is invalid")
    return output


def _quote_array(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
    rows: int,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size != rows or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} is invalid")
    output = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(output).all() or np.any(output <= 0.0):
        raise ValueError(f"{name} is invalid")
    return output


def _scenario_contract(scenario: str) -> dict[str, object]:
    if scenario == "base":
        return {
            "placement_latency_ms": 750,
            "passive_entry_fee_bps": 2.0,
            "aggressive_entry_fee_bps": 5.0,
            "exit_fee_bps": 5.0,
            "additional_slippage_bps_per_side": 1.0,
        }
    if scenario == "stress":
        return {
            "placement_latency_ms": 1_500,
            "passive_entry_fee_bps": 5.0,
            "aggressive_entry_fee_bps": 5.0,
            "exit_fee_bps": 5.0,
            "additional_slippage_bps_per_side": 3.0,
        }
    raise ValueError("make/take entry scenario is unsupported")


@dataclass(frozen=True)
class MakeTakeScenarioEntryBatch:
    schema_version: str
    scenario: str
    placement_latency_ms: int
    passive_expiry_ms: int
    order_notional_quote: float
    max_l1_participation: float
    passive_entry_fee_bps: float
    aggressive_entry_fee_bps: float
    exit_fee_bps: float
    additional_slippage_bps_per_side: float
    long_fill_sha256: str
    short_fill_sha256: str
    event_rows: int
    action_code: np.ndarray
    action_side: np.ndarray
    passive: np.ndarray
    eligible: np.ndarray
    filled: np.ndarray
    fill_bucket: np.ndarray
    order_start_time_ms: np.ndarray
    entry_time_ms: np.ndarray
    unfilled_expiry_time_ms: np.ndarray
    entry_price: np.ndarray
    displayed_l1_participation: np.ndarray
    entry_cost_bps: np.ndarray
    exit_cost_bps: np.ndarray
    batch_sha256: str

    @property
    def action_rows(self) -> int:
        return int(self.action_code.size)

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scenario": self.scenario,
            "event_rows": self.event_rows,
            "action_rows": self.action_rows,
            "eligible_by_action": {
                name: int(np.count_nonzero(self.eligible[offset::4]))
                for offset, name in enumerate(MAKE_TAKE_ACTION_NAMES)
            },
            "filled_by_action": {
                name: int(np.count_nonzero(self.filled[offset::4]))
                for offset, name in enumerate(MAKE_TAKE_ACTION_NAMES)
            },
            "long_fill_sha256": self.long_fill_sha256,
            "short_fill_sha256": self.short_fill_sha256,
            "batch_sha256": self.batch_sha256,
            "trading_authority": False,
            "profitability_claim": False,
        }


def _validate_fill_contract(
    fill: PassiveFillResult,
    *,
    buyer_is_maker: bool,
    arrivals: np.ndarray,
    prices: np.ndarray,
    queue: np.ndarray,
) -> None:
    if (
        fill.buyer_is_maker is not buyer_is_maker
        or fill.expiry_ms != PASSIVE_FILL_BUCKETS_MS[-1]
        or fill.order_notional_quote != MAKE_TAKE_ORDER_NOTIONAL_QUOTE
        or fill.rows != arrivals.size
        or not np.array_equal(fill.arrival_time_ms, arrivals)
        or not np.array_equal(fill.placement_price, prices)
        or not np.array_equal(fill.queue_ahead_quantity, queue)
        or len(fill.result_sha256) != 64
    ):
        raise ValueError("make/take passive-fill source contract drifted")


def build_make_take_scenario_entries(
    *,
    scenario: str,
    decision_time_ms: Sequence[int] | np.ndarray,
    bid_price: Sequence[float] | np.ndarray,
    ask_price: Sequence[float] | np.ndarray,
    bid_quantity: Sequence[float] | np.ndarray,
    ask_quantity: Sequence[float] | np.ndarray,
    long_fill: PassiveFillResult,
    short_fill: PassiveFillResult,
) -> MakeTakeScenarioEntryBatch:
    """Bind exact passive fills and crossing entries to one frozen scenario."""

    selected_scenario = str(scenario)
    contract = _scenario_contract(selected_scenario)
    decisions = _int_array(decision_time_ms, name="make/take decision times")
    if np.any(np.diff(decisions) <= 0):
        raise ValueError("make/take decision times are invalid")
    rows = decisions.size
    bid = _quote_array(bid_price, name="make/take bid prices", rows=rows)
    ask = _quote_array(ask_price, name="make/take ask prices", rows=rows)
    bid_qty = _quote_array(bid_quantity, name="make/take bid quantities", rows=rows)
    ask_qty = _quote_array(ask_quantity, name="make/take ask quantities", rows=rows)
    if np.any(bid >= ask):
        raise ValueError("make/take entry quotes are crossed or locked")
    placement_latency = int(contract["placement_latency_ms"])
    if np.any(decisions > np.iinfo(np.int64).max - placement_latency - PASSIVE_FILL_BUCKETS_MS[-1]):
        raise ValueError("make/take decision times overflow the order lifecycle")
    arrivals = decisions + placement_latency
    _validate_fill_contract(
        long_fill,
        buyer_is_maker=True,
        arrivals=arrivals,
        prices=bid,
        queue=bid_qty,
    )
    _validate_fill_contract(
        short_fill,
        buyer_is_maker=False,
        arrivals=arrivals,
        prices=ask,
        queue=ask_qty,
    )

    action_code = np.tile(_ACTION_CODE_PATTERN, rows)
    action_side = np.tile(_ACTION_SIDE_PATTERN, rows)
    passive = action_code < 2
    filled = np.column_stack(
        (
            long_fill.filled,
            short_fill.filled,
            np.ones(rows, dtype=np.bool_),
            np.ones(rows, dtype=np.bool_),
        )
    ).ravel()
    fill_bucket = np.column_stack(
        (
            long_fill.fill_bucket,
            short_fill.fill_bucket,
            np.zeros(rows, dtype=np.uint8),
            np.zeros(rows, dtype=np.uint8),
        )
    ).ravel()
    order_start = np.repeat(arrivals, 4)
    entry_time = np.column_stack(
        (
            long_fill.fill_time_ms,
            short_fill.fill_time_ms,
            arrivals,
            arrivals,
        )
    ).ravel()
    unfilled_expiry = np.column_stack(
        (
            arrivals + PASSIVE_FILL_BUCKETS_MS[-1],
            arrivals + PASSIVE_FILL_BUCKETS_MS[-1],
            np.full(rows, -1, dtype=np.int64),
            np.full(rows, -1, dtype=np.int64),
        )
    ).ravel()
    prices = np.column_stack((bid, ask, ask, bid)).ravel()
    displayed = np.column_stack((bid_qty, ask_qty, ask_qty, bid_qty)).ravel()
    participation = (MAKE_TAKE_ORDER_NOTIONAL_QUOTE / prices) / displayed
    eligible = participation <= MAKE_TAKE_MAX_L1_PARTICIPATION
    entry_fee = np.where(
        passive,
        float(contract["passive_entry_fee_bps"]),
        float(contract["aggressive_entry_fee_bps"]),
    )
    slippage = float(contract["additional_slippage_bps_per_side"])
    entry_cost = entry_fee + slippage
    exit_cost = np.full(rows * 4, float(contract["exit_fee_bps"]) + slippage)
    payload = {
        "schema_version": MAKE_TAKE_SCENARIO_ENTRY_SCHEMA_VERSION,
        "scenario": selected_scenario,
        "contract": contract,
        "passive_expiry_ms": PASSIVE_FILL_BUCKETS_MS[-1],
        "order_notional_quote": MAKE_TAKE_ORDER_NOTIONAL_QUOTE,
        "max_l1_participation": MAKE_TAKE_MAX_L1_PARTICIPATION,
        "long_fill_sha256": long_fill.result_sha256,
        "short_fill_sha256": short_fill.result_sha256,
        "action_names": list(MAKE_TAKE_ACTION_NAMES),
        "arrays": {
            "action_code": _array_sha256(action_code),
            "action_side": _array_sha256(action_side),
            "passive": _array_sha256(passive),
            "eligible": _array_sha256(eligible),
            "filled": _array_sha256(filled),
            "fill_bucket": _array_sha256(fill_bucket),
            "order_start_time_ms": _array_sha256(order_start),
            "entry_time_ms": _array_sha256(entry_time),
            "unfilled_expiry_time_ms": _array_sha256(unfilled_expiry),
            "entry_price": _array_sha256(prices),
            "displayed_l1_participation": _array_sha256(participation),
            "entry_cost_bps": _array_sha256(entry_cost),
            "exit_cost_bps": _array_sha256(exit_cost),
        },
    }
    batch_sha256 = _sha256(payload)
    retained = (
        action_code,
        action_side,
        passive,
        eligible,
        filled,
        fill_bucket,
        order_start,
        entry_time,
        unfilled_expiry,
        prices,
        participation,
        entry_cost,
        exit_cost,
    )
    for array in retained:
        array.setflags(write=False)
    return MakeTakeScenarioEntryBatch(
        schema_version=MAKE_TAKE_SCENARIO_ENTRY_SCHEMA_VERSION,
        scenario=selected_scenario,
        placement_latency_ms=placement_latency,
        passive_expiry_ms=PASSIVE_FILL_BUCKETS_MS[-1],
        order_notional_quote=MAKE_TAKE_ORDER_NOTIONAL_QUOTE,
        max_l1_participation=MAKE_TAKE_MAX_L1_PARTICIPATION,
        passive_entry_fee_bps=float(contract["passive_entry_fee_bps"]),
        aggressive_entry_fee_bps=float(contract["aggressive_entry_fee_bps"]),
        exit_fee_bps=float(contract["exit_fee_bps"]),
        additional_slippage_bps_per_side=slippage,
        long_fill_sha256=long_fill.result_sha256,
        short_fill_sha256=short_fill.result_sha256,
        event_rows=rows,
        action_code=action_code,
        action_side=action_side,
        passive=passive,
        eligible=eligible,
        filled=filled,
        fill_bucket=fill_bucket,
        order_start_time_ms=order_start,
        entry_time_ms=entry_time,
        unfilled_expiry_time_ms=unfilled_expiry,
        entry_price=prices,
        displayed_l1_participation=participation,
        entry_cost_bps=entry_cost,
        exit_cost_bps=exit_cost,
        batch_sha256=batch_sha256,
    )


__all__ = [
    "MAKE_TAKE_MAX_L1_PARTICIPATION",
    "MAKE_TAKE_ORDER_NOTIONAL_QUOTE",
    "MAKE_TAKE_SCENARIOS",
    "MAKE_TAKE_SCENARIO_ENTRY_SCHEMA_VERSION",
    "MakeTakeScenarioEntryBatch",
    "build_make_take_scenario_entries",
]
