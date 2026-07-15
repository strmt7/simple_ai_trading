"""Fixed-ledger, non-overlapping base/stress replay for make/take actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Sequence

import numpy as np

from .make_take_action_features import MAKE_TAKE_ACTION_NAMES
from .make_take_action_values import (
    MakeTakeActionValueBatch,
    validate_make_take_action_value_batch,
)
from .make_take_payoff_panel import MAKE_TAKE_PAYOFF_SYMBOLS
from .make_take_targets import (
    MAKE_TAKE_UNFILLED_OUTCOME,
    MakeTakeTargetBatch,
    validate_make_take_target_batch,
)


MAKE_TAKE_FIXED_LEDGER_SCHEMA_VERSION = "make-take-fixed-ledger-v1"
MAKE_TAKE_REPLAY_SCENARIOS = ("base", "stress")
_DAY_MS = 86_400_000


@dataclass(frozen=True)
class MakeTakeLedgerOrder:
    order_id: str
    symbol: str
    source_event_index: int
    target_row: int
    decision_time_ms: int
    action_code: int
    action_name: str
    action_side: int
    fill_probability_15s: float
    conditional_mean_bps: float
    conditional_q20_bps: float
    expected_mean_bps: float
    base_filled: bool
    base_terminal_time_ms: int
    base_realized_net_bps: float
    base_outcome: int
    stress_filled: bool
    stress_terminal_time_ms: int
    stress_realized_net_bps: float
    stress_outcome: int


@dataclass(frozen=True)
class MakeTakeFixedLedger:
    schema_version: str
    expected_mean_threshold_bps: float
    conditional_q20_floor_bps: float
    source_action_value_sha256_by_symbol: tuple[tuple[str, str], ...]
    base_target_sha256_by_symbol: tuple[tuple[str, str], ...]
    stress_target_sha256_by_symbol: tuple[tuple[str, str], ...]
    fill_model_sha256: str
    payoff_model_sha256: str
    orders: tuple[MakeTakeLedgerOrder, ...]
    ledger_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    @property
    def selected_orders(self) -> int:
        return len(self.orders)

    def evidence(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MakeTakeReplayMetrics:
    scenario: str
    selected_orders: int
    closed_trades: int
    unfilled_orders: int
    total_net_bps: float
    mean_closed_trade_net_bps: float
    profit_factor: float | None
    maximum_drawdown_bps: float
    worst_closed_trade_bps: float
    positive_symbols: int
    maximum_single_symbol_positive_pnl_share: float
    daily_net_bps: tuple[tuple[int, float], ...]
    symbol_net_bps: tuple[tuple[str, float], ...]

    def evidence(self) -> dict[str, object]:
        return asdict(self)


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


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _ledger_payload(ledger: MakeTakeFixedLedger) -> dict[str, object]:
    payload = asdict(ledger)
    payload.pop("ledger_sha256")
    return payload


def _ordered_values(
    batches: Sequence[MakeTakeActionValueBatch],
) -> tuple[MakeTakeActionValueBatch, ...]:
    values = tuple(batches)
    for batch in values:
        validate_make_take_action_value_batch(batch)
    if (
        len(values) != len(MAKE_TAKE_PAYOFF_SYMBOLS)
        or {batch.symbol for batch in values} != set(MAKE_TAKE_PAYOFF_SYMBOLS)
    ):
        raise ValueError("make/take replay action-value set is invalid")
    return tuple(sorted(values, key=lambda batch: batch.symbol))


def _ordered_targets(
    batches: Sequence[MakeTakeTargetBatch],
    *,
    scenario: str,
) -> tuple[MakeTakeTargetBatch, ...]:
    values = tuple(batches)
    for batch in values:
        validate_make_take_target_batch(batch)
    if (
        len(values) != len(MAKE_TAKE_PAYOFF_SYMBOLS)
        or {batch.symbol for batch in values} != set(MAKE_TAKE_PAYOFF_SYMBOLS)
        or any(batch.scenario != scenario for batch in values)
    ):
        raise ValueError(f"make/take replay {scenario} target set is invalid")
    return tuple(sorted(values, key=lambda batch: batch.symbol))


def _order_id(
    *,
    symbol: str,
    source_event_index: int,
    decision_time_ms: int,
    action_code: int,
    base_target_sha256: str,
    stress_target_sha256: str,
) -> str:
    value = ":".join(
        (
            symbol,
            str(source_event_index),
            str(decision_time_ms),
            str(action_code),
            base_target_sha256,
            stress_target_sha256,
        )
    )
    return hashlib.sha256(value.encode("ascii")).hexdigest()[:24]


def validate_make_take_fixed_ledger(ledger: MakeTakeFixedLedger) -> None:
    maps = (
        ledger.source_action_value_sha256_by_symbol,
        ledger.base_target_sha256_by_symbol,
        ledger.stress_target_sha256_by_symbol,
    )
    order_ids: set[str] = set()
    free_at = {symbol: -1 for symbol in MAKE_TAKE_PAYOFF_SYMBOLS}
    previous_key: tuple[int, str, int] | None = None
    if (
        ledger.schema_version != MAKE_TAKE_FIXED_LEDGER_SCHEMA_VERSION
        or not math.isfinite(ledger.expected_mean_threshold_bps)
        or not math.isfinite(ledger.conditional_q20_floor_bps)
        or ledger.expected_mean_threshold_bps < 0.0
        or not -100.0 <= ledger.conditional_q20_floor_bps <= 50.0
        or any(
            tuple(symbol for symbol, _sha in values)
            != tuple(sorted(MAKE_TAKE_PAYOFF_SYMBOLS))
            or any(not _is_sha256(sha) for _symbol, sha in values)
            for values in maps
        )
        or not _is_sha256(ledger.fill_model_sha256)
        or not _is_sha256(ledger.payoff_model_sha256)
        or ledger.trading_authority is not False
        or ledger.execution_claim is not False
        or ledger.profitability_claim is not False
        or ledger.portfolio_claim is not False
        or ledger.leverage_applied is not False
        or not _is_sha256(ledger.ledger_sha256)
        or ledger.ledger_sha256 != _sha256(_ledger_payload(ledger))
    ):
        raise ValueError("make/take fixed ledger is invalid")
    for order in ledger.orders:
        numeric = (
            order.fill_probability_15s,
            order.conditional_mean_bps,
            order.conditional_q20_bps,
            order.expected_mean_bps,
            order.base_realized_net_bps,
            order.stress_realized_net_bps,
        )
        key = (order.decision_time_ms, order.symbol, order.action_code)
        if (
            order.order_id in order_ids
            or len(order.order_id) != 24
            or any(character not in "0123456789abcdef" for character in order.order_id)
            or order.symbol not in MAKE_TAKE_PAYOFF_SYMBOLS
            or order.source_event_index < 0
            or order.target_row != order.source_event_index * 4 + order.action_code
            or order.decision_time_ms < 0
            or order.action_code not in (0, 1, 2, 3)
            or order.action_name != MAKE_TAKE_ACTION_NAMES[order.action_code]
            or order.action_side != (1 if order.action_code in (0, 2) else -1)
            or not all(math.isfinite(value) for value in numeric)
            or not 0.0 <= order.fill_probability_15s <= 1.0
            or order.conditional_q20_bps > order.conditional_mean_bps + 1e-12
            or order.expected_mean_bps <= ledger.expected_mean_threshold_bps
            or order.conditional_q20_bps < ledger.conditional_q20_floor_bps
            or order.base_terminal_time_ms < order.decision_time_ms
            or order.stress_terminal_time_ms < order.decision_time_ms
            or order.base_outcome not in (-2, 0, 1, 2, 3, 4)
            or order.stress_outcome not in (-2, 0, 1, 2, 3, 4)
            or (not order.base_filled)
            != (order.base_outcome == MAKE_TAKE_UNFILLED_OUTCOME)
            or (not order.stress_filled)
            != (order.stress_outcome == MAKE_TAKE_UNFILLED_OUTCOME)
            or (not order.base_filled and order.base_realized_net_bps != 0.0)
            or (not order.stress_filled and order.stress_realized_net_bps != 0.0)
            or order.decision_time_ms < free_at[order.symbol]
            or (previous_key is not None and key < previous_key)
        ):
            raise ValueError("make/take fixed ledger order is invalid")
        order_ids.add(order.order_id)
        free_at[order.symbol] = max(
            order.base_terminal_time_ms,
            order.stress_terminal_time_ms,
        )
        previous_key = key


def build_make_take_fixed_ledger(
    *,
    action_values: Sequence[MakeTakeActionValueBatch],
    base_targets: Sequence[MakeTakeTargetBatch],
    stress_targets: Sequence[MakeTakeTargetBatch],
    expected_mean_threshold_bps: float,
    conditional_q20_floor_bps: float = 0.0,
) -> MakeTakeFixedLedger:
    """Choose one action per event and preserve feasibility in both scenarios."""

    threshold = float(expected_mean_threshold_bps)
    q20_floor = float(conditional_q20_floor_bps)
    if (
        not math.isfinite(threshold)
        or threshold < 0.0
        or not math.isfinite(q20_floor)
        or not -100.0 <= q20_floor <= 50.0
    ):
        raise ValueError("make/take fixed policy thresholds are invalid")
    values = _ordered_values(action_values)
    base = _ordered_targets(base_targets, scenario="base")
    stress = _ordered_targets(stress_targets, scenario="stress")
    fill_models = {batch.fill_model_sha256 for batch in values}
    payoff_models = {batch.payoff_model_sha256 for batch in values}
    if len(fill_models) != 1 or len(payoff_models) != 1:
        raise ValueError("make/take action-value model identities drifted")
    orders: list[MakeTakeLedgerOrder] = []
    for value, base_target, stress_target in zip(values, base, stress, strict=True):
        if (
            value.symbol != base_target.symbol
            or value.symbol != stress_target.symbol
            or value.source_dataset_sha256 != base_target.source_dataset_sha256
            or value.source_dataset_sha256 != stress_target.source_dataset_sha256
            or base_target.event_rows != stress_target.event_rows
            or not np.array_equal(base_target.action_code, stress_target.action_code)
            or not np.array_equal(base_target.action_side, stress_target.action_side)
            or not np.array_equal(base_target.eligible, stress_target.eligible)
            or value.event_index[-1] >= base_target.event_rows
        ):
            raise ValueError("make/take replay source targets drifted")
        free_at_ms = -1
        for event_position in range(value.event_rows):
            local_start = event_position * 4
            local = np.arange(local_start, local_start + 4, dtype=np.int64)
            source_event = int(value.event_index[local_start])
            source = source_event * 4 + np.arange(4, dtype=np.int64)
            decision_time = int(value.decision_time_ms[local_start])
            if (
                not np.all(value.event_index[local] == source_event)
                or not np.all(value.decision_time_ms[local] == decision_time)
            ):
                raise ValueError("make/take action-value event grouping drifted")
            candidates = (
                value.eligible[local]
                & base_target.realized_valid[source]
                & stress_target.realized_valid[source]
                & (value.expected_mean_bps[local] > threshold)
                & (value.conditional_q20_bps[local] >= q20_floor)
            )
            if decision_time < free_at_ms or not np.any(candidates):
                continue
            candidate_local = local[candidates]
            selected_local = int(
                candidate_local[
                    np.argmax(value.expected_mean_bps[candidate_local])
                ]
            )
            action = int(value.action_code[selected_local])
            target_row = source_event * 4 + action
            base_terminal = int(base_target.terminal_time_ms[target_row])
            stress_terminal = int(stress_target.terminal_time_ms[target_row])
            order = MakeTakeLedgerOrder(
                order_id=_order_id(
                    symbol=value.symbol,
                    source_event_index=source_event,
                    decision_time_ms=decision_time,
                    action_code=action,
                    base_target_sha256=base_target.target_sha256,
                    stress_target_sha256=stress_target.target_sha256,
                ),
                symbol=value.symbol,
                source_event_index=source_event,
                target_row=target_row,
                decision_time_ms=decision_time,
                action_code=action,
                action_name=MAKE_TAKE_ACTION_NAMES[action],
                action_side=int(value.action_side[selected_local]),
                fill_probability_15s=float(
                    value.fill_probability_15s[selected_local]
                ),
                conditional_mean_bps=float(
                    value.conditional_mean_bps[selected_local]
                ),
                conditional_q20_bps=float(value.conditional_q20_bps[selected_local]),
                expected_mean_bps=float(value.expected_mean_bps[selected_local]),
                base_filled=bool(base_target.filled[target_row]),
                base_terminal_time_ms=base_terminal,
                base_realized_net_bps=float(base_target.realized_net_bps[target_row]),
                base_outcome=int(base_target.outcome[target_row]),
                stress_filled=bool(stress_target.filled[target_row]),
                stress_terminal_time_ms=stress_terminal,
                stress_realized_net_bps=float(
                    stress_target.realized_net_bps[target_row]
                ),
                stress_outcome=int(stress_target.outcome[target_row]),
            )
            orders.append(order)
            free_at_ms = max(base_terminal, stress_terminal)
    ordered_orders = tuple(
        sorted(
            orders,
            key=lambda order: (
                order.decision_time_ms,
                order.symbol,
                order.action_code,
            ),
        )
    )
    provisional = MakeTakeFixedLedger(
        schema_version=MAKE_TAKE_FIXED_LEDGER_SCHEMA_VERSION,
        expected_mean_threshold_bps=threshold,
        conditional_q20_floor_bps=q20_floor,
        source_action_value_sha256_by_symbol=tuple(
            (batch.symbol, batch.batch_sha256) for batch in values
        ),
        base_target_sha256_by_symbol=tuple(
            (batch.symbol, batch.target_sha256) for batch in base
        ),
        stress_target_sha256_by_symbol=tuple(
            (batch.symbol, batch.target_sha256) for batch in stress
        ),
        fill_model_sha256=next(iter(fill_models)),
        payoff_model_sha256=next(iter(payoff_models)),
        orders=ordered_orders,
        ledger_sha256="",
    )
    ledger = MakeTakeFixedLedger(
        **{**provisional.__dict__, "ledger_sha256": _sha256(_ledger_payload(provisional))}
    )
    validate_make_take_fixed_ledger(ledger)
    return ledger


def replay_make_take_fixed_ledger(
    ledger: MakeTakeFixedLedger,
    *,
    scenario: str,
    expected_days: Sequence[int],
) -> MakeTakeReplayMetrics:
    """Replay the unchanged ledger and report unlevered trade-bps evidence."""

    validate_make_take_fixed_ledger(ledger)
    days = tuple(int(value) for value in expected_days)
    if (
        scenario not in MAKE_TAKE_REPLAY_SCENARIOS
        or not days
        or len(set(days)) != len(days)
        or tuple(sorted(days)) != days
    ):
        raise ValueError("make/take replay role days are invalid")
    daily = {day: 0.0 for day in days}
    symbol_net = {symbol: 0.0 for symbol in MAKE_TAKE_PAYOFF_SYMBOLS}
    booked: list[tuple[int, str, float, bool]] = []
    for order in ledger.orders:
        decision_day = order.decision_time_ms // _DAY_MS
        if decision_day not in daily:
            raise ValueError("make/take ledger order lies outside replay days")
        if scenario == "base":
            terminal = order.base_terminal_time_ms
            net = order.base_realized_net_bps
            filled = order.base_filled
        else:
            terminal = order.stress_terminal_time_ms
            net = order.stress_realized_net_bps
            filled = order.stress_filled
        daily[decision_day] += net
        symbol_net[order.symbol] += net
        booked.append((terminal, order.symbol, net, filled))
    booked.sort(key=lambda item: (item[0], item[1]))
    net_values = np.asarray([item[2] for item in booked], dtype=np.float64)
    filled_values = np.asarray([item[3] for item in booked], dtype=np.bool_)
    closed_values = net_values[filled_values]
    cumulative = np.concatenate(
        (np.asarray([0.0]), np.cumsum(net_values, dtype=np.float64))
    )
    peaks = np.maximum.accumulate(cumulative)
    maximum_drawdown = float(np.max(peaks - cumulative))
    positive = float(np.sum(closed_values[closed_values > 0.0]))
    negative = float(-np.sum(closed_values[closed_values < 0.0]))
    profit_factor = positive / negative if negative > 0.0 else None
    positive_symbol_values = [max(value, 0.0) for value in symbol_net.values()]
    positive_symbol_total = float(sum(positive_symbol_values))
    concentration = (
        max(positive_symbol_values) / positive_symbol_total
        if positive_symbol_total > 0.0
        else 1.0
    )
    return MakeTakeReplayMetrics(
        scenario=scenario,
        selected_orders=len(ledger.orders),
        closed_trades=int(np.count_nonzero(filled_values)),
        unfilled_orders=int(len(ledger.orders) - np.count_nonzero(filled_values)),
        total_net_bps=float(np.sum(net_values, dtype=np.float64)),
        mean_closed_trade_net_bps=(
            float(np.mean(closed_values, dtype=np.float64))
            if closed_values.size
            else 0.0
        ),
        profit_factor=profit_factor,
        maximum_drawdown_bps=maximum_drawdown,
        worst_closed_trade_bps=(float(np.min(closed_values)) if closed_values.size else 0.0),
        positive_symbols=int(sum(value > 0.0 for value in symbol_net.values())),
        maximum_single_symbol_positive_pnl_share=float(concentration),
        daily_net_bps=tuple((day, float(daily[day])) for day in days),
        symbol_net_bps=tuple(
            (symbol, float(symbol_net[symbol])) for symbol in sorted(symbol_net)
        ),
    )


__all__ = [
    "MAKE_TAKE_FIXED_LEDGER_SCHEMA_VERSION",
    "MAKE_TAKE_REPLAY_SCENARIOS",
    "MakeTakeFixedLedger",
    "MakeTakeLedgerOrder",
    "MakeTakeReplayMetrics",
    "build_make_take_fixed_ledger",
    "replay_make_take_fixed_ledger",
    "validate_make_take_fixed_ledger",
]
