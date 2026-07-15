"""Join scenario entries to day-partitioned causal path targets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from .make_take_action_features import MAKE_TAKE_ACTION_NAMES
from .make_take_path_payoffs import build_action_path_payoffs
from .make_take_scenario_entries import MakeTakeScenarioEntryBatch


MAKE_TAKE_TARGET_SCHEMA_VERSION = "queue-censored-make-take-targets-v1"
MAKE_TAKE_UNFILLED_OUTCOME = -2
_DAY_MS = 86_400_000
DayPathLoader = Callable[[int], Mapping[str, Sequence[int] | Sequence[float] | np.ndarray]]


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


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _barrier_array(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
    rows: int,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size != rows or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} is invalid")
    output = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(output).all():
        raise ValueError(f"{name} is invalid")
    return output


@dataclass(frozen=True)
class MakeTakeTargetBatch:
    schema_version: str
    scenario: str
    symbol: str
    source_dataset_sha256: str
    source_entry_sha256: str
    day_path_sha256: Mapping[str, str]
    event_rows: int
    action_code: np.ndarray
    action_side: np.ndarray
    eligible: np.ndarray
    filled: np.ndarray
    fill_bucket: np.ndarray
    conditional_payoff_valid: np.ndarray
    realized_valid: np.ndarray
    conditional_net_bps: np.ndarray
    realized_net_bps: np.ndarray
    terminal_time_ms: np.ndarray
    outcome: np.ndarray
    markout_5s_bps: np.ndarray
    markout_15s_bps: np.ndarray
    stop_bps: np.ndarray
    take_bps: np.ndarray
    target_sha256: str

    @property
    def action_rows(self) -> int:
        return int(self.action_code.size)

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scenario": self.scenario,
            "symbol": self.symbol,
            "event_rows": self.event_rows,
            "action_rows": self.action_rows,
            "conditional_payoff_rows_by_action": {
                name: int(np.count_nonzero(self.conditional_payoff_valid[offset::4]))
                for offset, name in enumerate(MAKE_TAKE_ACTION_NAMES)
            },
            "realized_valid_rows_by_action": {
                name: int(np.count_nonzero(self.realized_valid[offset::4]))
                for offset, name in enumerate(MAKE_TAKE_ACTION_NAMES)
            },
            "unfilled_passive_rows": int(
                np.count_nonzero(self.outcome == MAKE_TAKE_UNFILLED_OUTCOME)
            ),
            "day_path_sha256": dict(self.day_path_sha256),
            "target_sha256": self.target_sha256,
            "trading_authority": False,
            "profitability_claim": False,
        }


def build_make_take_targets(
    *,
    symbol: str,
    source_dataset_sha256: str,
    entries: MakeTakeScenarioEntryBatch,
    event_stop_bps: Sequence[float] | np.ndarray,
    event_take_bps: Sequence[float] | np.ndarray,
    load_day_path: DayPathLoader,
    progress: Callable[[int, int, int], None] | None = None,
) -> MakeTakeTargetBatch:
    """Build conditional and realized action targets without hiding non-fills."""

    normalized_symbol = str(symbol).strip().upper()
    if (
        normalized_symbol not in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
        or not _is_sha256(source_dataset_sha256)
        or not callable(load_day_path)
        or entries.event_rows <= 0
        or entries.action_rows != entries.event_rows * 4
    ):
        raise ValueError("make/take target source contract is invalid")
    stops_event = _barrier_array(
        event_stop_bps,
        name="make/take event stop distances",
        rows=entries.event_rows,
    )
    takes_event = _barrier_array(
        event_take_bps,
        name="make/take event take distances",
        rows=entries.event_rows,
    )
    if (
        np.any(stops_event < 18.0)
        or np.any(stops_event > 80.0)
        or np.any(takes_event < 30.0)
        or np.any(takes_event > 120.0)
        or np.any(takes_event <= stops_event)
    ):
        raise ValueError("make/take target barrier contract is invalid")
    stops = np.repeat(stops_event, 4)
    takes = np.repeat(takes_event, 4)
    action_rows = entries.action_rows
    conditional_valid = np.zeros(action_rows, dtype=np.bool_)
    realized_valid = np.zeros(action_rows, dtype=np.bool_)
    conditional_net = np.full(action_rows, np.nan, dtype=np.float64)
    realized_net = np.full(action_rows, np.nan, dtype=np.float64)
    terminal_time = np.full(action_rows, -1, dtype=np.int64)
    outcome = np.full(action_rows, -1, dtype=np.int8)
    markout_5s = np.full(action_rows, np.nan, dtype=np.float64)
    markout_15s = np.full(action_rows, np.nan, dtype=np.float64)

    unfilled = entries.eligible & entries.passive & ~entries.filled
    realized_valid[unfilled] = True
    realized_net[unfilled] = 0.0
    terminal_time[unfilled] = entries.unfilled_expiry_time_ms[unfilled]
    outcome[unfilled] = MAKE_TAKE_UNFILLED_OUTCOME

    executed = entries.eligible & entries.filled
    executed_rows = np.flatnonzero(executed)
    day_path_sha256: dict[str, str] = {}
    if executed_rows.size:
        day_ids = entries.entry_time_ms[executed_rows] // _DAY_MS
        unique_days = np.unique(day_ids)
        for day_offset, day_id in enumerate(unique_days, start=1):
            local_rows = executed_rows[day_ids == day_id]
            ordering = np.argsort(entries.entry_time_ms[local_rows], kind="stable")
            ordered_rows = local_rows[ordering]
            day_start_ms = int(day_id) * _DAY_MS
            path = dict(load_day_path(day_start_ms))
            required_path_fields = {
                "path_time_ms",
                "path_min_bid",
                "path_max_bid",
                "path_close_bid",
                "path_min_ask",
                "path_max_ask",
                "path_close_ask",
            }
            if set(path) != required_path_fields:
                raise ValueError("make/take day path contract is invalid")
            payoff = build_action_path_payoffs(
                scenario=entries.scenario,
                **path,
                entry_time_ms=entries.entry_time_ms[ordered_rows],
                action_side=entries.action_side[ordered_rows],
                entry_price=entries.entry_price[ordered_rows],
                entry_cost_bps=entries.entry_cost_bps[ordered_rows],
                exit_cost_bps=entries.exit_cost_bps[ordered_rows],
                stop_bps=stops[ordered_rows],
                take_bps=takes[ordered_rows],
            )
            day_key = str(day_start_ms)
            day_path_sha256[day_key] = payoff.source_path_sha256
            valid_rows = ordered_rows[payoff.valid]
            conditional_valid[valid_rows] = True
            realized_valid[valid_rows] = True
            conditional_net[valid_rows] = payoff.net_bps[payoff.valid]
            realized_net[valid_rows] = payoff.net_bps[payoff.valid]
            terminal_time[valid_rows] = payoff.exit_time_ms[payoff.valid]
            outcome[valid_rows] = payoff.outcome[payoff.valid]
            markout_5s[valid_rows] = payoff.markout_5s_bps[payoff.valid]
            markout_15s[valid_rows] = payoff.markout_15s_bps[payoff.valid]
            if progress is not None:
                progress(day_offset, len(unique_days), int(np.count_nonzero(realized_valid)))

    arrays = {
        "action_code": np.asarray(entries.action_code),
        "action_side": np.asarray(entries.action_side),
        "eligible": np.asarray(entries.eligible),
        "filled": np.asarray(entries.filled),
        "fill_bucket": np.asarray(entries.fill_bucket),
        "conditional_payoff_valid": conditional_valid,
        "realized_valid": realized_valid,
        "conditional_net_bps": conditional_net,
        "realized_net_bps": realized_net,
        "terminal_time_ms": terminal_time,
        "outcome": outcome,
        "markout_5s_bps": markout_5s,
        "markout_15s_bps": markout_15s,
        "stop_bps": stops,
        "take_bps": takes,
    }
    payload = {
        "schema_version": MAKE_TAKE_TARGET_SCHEMA_VERSION,
        "scenario": entries.scenario,
        "symbol": normalized_symbol,
        "source_dataset_sha256": str(source_dataset_sha256),
        "source_entry_sha256": entries.batch_sha256,
        "day_path_sha256": day_path_sha256,
        "action_names": list(MAKE_TAKE_ACTION_NAMES),
        "arrays": {name: _array_sha256(value) for name, value in arrays.items()},
    }
    target_sha256 = _sha256(payload)
    retained = (
        conditional_valid,
        realized_valid,
        conditional_net,
        realized_net,
        terminal_time,
        outcome,
        markout_5s,
        markout_15s,
        stops,
        takes,
    )
    for array in retained:
        array.setflags(write=False)
    return MakeTakeTargetBatch(
        schema_version=MAKE_TAKE_TARGET_SCHEMA_VERSION,
        scenario=entries.scenario,
        symbol=normalized_symbol,
        source_dataset_sha256=str(source_dataset_sha256),
        source_entry_sha256=entries.batch_sha256,
        day_path_sha256=MappingProxyType(dict(day_path_sha256)),
        event_rows=entries.event_rows,
        action_code=entries.action_code,
        action_side=entries.action_side,
        eligible=entries.eligible,
        filled=entries.filled,
        fill_bucket=entries.fill_bucket,
        conditional_payoff_valid=conditional_valid,
        realized_valid=realized_valid,
        conditional_net_bps=conditional_net,
        realized_net_bps=realized_net,
        terminal_time_ms=terminal_time,
        outcome=outcome,
        markout_5s_bps=markout_5s,
        markout_15s_bps=markout_15s,
        stop_bps=stops,
        take_bps=takes,
        target_sha256=target_sha256,
    )


__all__ = [
    "MAKE_TAKE_TARGET_SCHEMA_VERSION",
    "MAKE_TAKE_UNFILLED_OUTCOME",
    "DayPathLoader",
    "MakeTakeTargetBatch",
    "build_make_take_targets",
]
