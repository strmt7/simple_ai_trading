"""Shared four-action features for queue-censored make/take research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Sequence

import numpy as np

from .microstructure_action_features import build_action_conditional_features
from .microstructure_features import (
    AGGREGATE_DEPTH_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_NAMES,
)
from .queue_censored_actions import (
    EXPONENTIAL_FLOW_HALF_LIVES_SECONDS,
    ExponentialFlowBatch,
)


MAKE_TAKE_ACTION_FEATURE_SCHEMA_VERSION = "queue-censored-make-take-actions-v1"
MAKE_TAKE_ACTION_NAMES = (
    "passive_long",
    "passive_short",
    "aggressive_long",
    "aggressive_short",
)
_ACTION_SIDE_PATTERN = np.asarray([1, -1, 1, -1], dtype=np.int8)
_ACTION_CODE_PATTERN = np.arange(4, dtype=np.uint8)
_AGGREGATE_SIGNED_NAMES = frozenset(
    {
        "aggregate_depth_notional_imbalance_1pct",
        "aggregate_depth_notional_imbalance_5pct",
        "aggregate_depth_concentration_skew",
    }
)
_AGGREGATE_SWAP_NAMES = (
    ("log_bid_notional_within_1pct", "log_ask_notional_within_1pct"),
    ("log_bid_notional_within_5pct", "log_ask_notional_within_5pct"),
    (
        "bid_depth_concentration_1pct_to_5pct",
        "ask_depth_concentration_1pct_to_5pct",
    ),
)
_AGGREGATE_CANONICAL_NAMES = {
    "log_bid_notional_within_1pct": "log_supporting_notional_within_1pct",
    "log_ask_notional_within_1pct": "log_opposing_notional_within_1pct",
    "log_bid_notional_within_5pct": "log_supporting_notional_within_5pct",
    "log_ask_notional_within_5pct": "log_opposing_notional_within_5pct",
    "bid_depth_concentration_1pct_to_5pct": (
        "supporting_depth_concentration_1pct_to_5pct"
    ),
    "ask_depth_concentration_1pct_to_5pct": (
        "opposing_depth_concentration_1pct_to_5pct"
    ),
}
_ACTION_SPECIFIC_FEATURE_NAMES = (
    "action_is_passive",
    "log_queue_ahead_quote",
    "log_required_print_quote",
    "log1p_queue_ahead_to_own",
    "displayed_l1_participation",
    "known_round_trip_cost_bps",
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


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _numeric_array(
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


def _indexes(value: Sequence[int] | np.ndarray | None, *, rows: int) -> np.ndarray:
    if value is None:
        return np.arange(rows, dtype=np.int64)
    raw = np.asarray(value)
    if (
        raw.ndim != 1
        or raw.size == 0
        or raw.dtype.kind not in "iu"
        or (raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max))
    ):
        raise ValueError("make/take event indexes are invalid")
    output = np.array(raw, dtype=np.int64, order="C", copy=True)
    if output[0] < 0 or output[-1] >= rows or np.any(np.diff(output) <= 0):
        raise ValueError("make/take event indexes are invalid")
    return output


@dataclass(frozen=True)
class MakeTakeFeatureSpec:
    """Observable execution constants carried by one model feature contract."""

    placement_latency_ms: int = 750
    order_notional_quote: float = 1_000.0
    max_l1_participation: float = 0.10
    maker_entry_fee_bps: float = 2.0
    taker_entry_fee_bps: float = 5.0
    taker_exit_fee_bps: float = 5.0
    additional_slippage_bps_per_side: float = 1.0

    def validate(self) -> None:
        integer = self.placement_latency_ms
        numeric = (
            self.order_notional_quote,
            self.max_l1_participation,
            self.maker_entry_fee_bps,
            self.taker_entry_fee_bps,
            self.taker_exit_fee_bps,
            self.additional_slippage_bps_per_side,
        )
        if (
            isinstance(integer, (bool, np.bool_))
            or not isinstance(integer, Integral)
            or not 0 <= int(integer) <= 10_000
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in numeric
            )
            or not 0.0 < float(self.order_notional_quote) <= 1_000_000.0
            or not 0.0 < float(self.max_l1_participation) <= 1.0
            or not -10.0 <= float(self.maker_entry_fee_bps) <= 100.0
            or not 0.0 <= float(self.taker_entry_fee_bps) <= 100.0
            or not 0.0 <= float(self.taker_exit_fee_bps) <= 100.0
            or not 0.0 <= float(self.additional_slippage_bps_per_side) <= 100.0
        ):
            raise ValueError("make/take feature specification is invalid")

    @property
    def spec_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


@dataclass(frozen=True)
class MakeTakeActionFeatureBatch:
    schema_version: str
    spec: MakeTakeFeatureSpec
    spec_sha256: str
    source_dataset_sha256: str
    source_flow_sha256: str
    feature_names: tuple[str, ...]
    event_indexes: np.ndarray
    action_code: np.ndarray
    action_side: np.ndarray
    eligible: np.ndarray
    features: np.ndarray
    batch_sha256: str

    @property
    def event_rows(self) -> int:
        return int(self.event_indexes.size)

    @property
    def action_rows(self) -> int:
        return int(self.action_code.size)

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "event_rows": self.event_rows,
            "action_rows": self.action_rows,
            "feature_count": len(self.feature_names),
            "eligible_by_action": {
                action: int(np.count_nonzero(self.eligible[offset::4]))
                for offset, action in enumerate(MAKE_TAKE_ACTION_NAMES)
            },
            "source_dataset_sha256": self.source_dataset_sha256,
            "source_flow_sha256": self.source_flow_sha256,
            "batch_sha256": self.batch_sha256,
            "trading_authority": False,
            "profitability_claim": False,
        }


def _aggregate_action_features(source: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
    aggregate_names = AGGREGATE_DEPTH_FEATURE_NAMES[len(MICROSTRUCTURE_FEATURE_NAMES) :]
    if source.shape[1] != len(aggregate_names):
        raise ValueError("aggregate-depth action feature contract is invalid")
    rows = source.shape[0]
    paired = np.empty((rows * 2, source.shape[1]), dtype=np.float32)
    paired[0::2] = source
    paired[1::2] = source
    for name in _AGGREGATE_SIGNED_NAMES:
        paired[1::2, aggregate_names.index(name)] *= -1.0
    for bid_name, ask_name in _AGGREGATE_SWAP_NAMES:
        bid_index = aggregate_names.index(bid_name)
        ask_index = aggregate_names.index(ask_name)
        paired[1::2, bid_index] = source[:, ask_index]
        paired[1::2, ask_index] = source[:, bid_index]
    canonical_names = tuple(
        f"action_aligned_{name}"
        if name in _AGGREGATE_SIGNED_NAMES
        else _AGGREGATE_CANONICAL_NAMES.get(name, name)
        for name in aggregate_names
    )
    return canonical_names, paired


def _flow_action_features(flow: ExponentialFlowBatch, indexes: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
    expected_names = tuple(
        name
        for half_life in EXPONENTIAL_FLOW_HALF_LIVES_SECONDS
        for name in (
            f"flow_log_quote_intensity_h{half_life}s",
            f"flow_imbalance_h{half_life}s",
        )
    )
    if flow.feature_names != expected_names:
        raise ValueError("make/take exponential-flow feature contract drifted")
    selected = np.asarray(flow.features[indexes], dtype=np.float32)
    paired = np.repeat(selected, 2, axis=0)
    paired[1::2, 1::2] *= -1.0
    names = tuple(
        f"action_aligned_{name}" if "imbalance" in name else name
        for name in expected_names
    )
    return names, paired


def build_make_take_action_features(
    *,
    source_features: np.ndarray,
    source_feature_names: Sequence[str],
    decision_time_ms: Sequence[int] | np.ndarray,
    bid_price: Sequence[float] | np.ndarray,
    ask_price: Sequence[float] | np.ndarray,
    bid_quantity: Sequence[float] | np.ndarray,
    ask_quantity: Sequence[float] | np.ndarray,
    flow: ExponentialFlowBatch,
    source_dataset_sha256: str,
    event_indexes: Sequence[int] | np.ndarray | None = None,
    spec: MakeTakeFeatureSpec = MakeTakeFeatureSpec(),
) -> MakeTakeActionFeatureBatch:
    """Build event-major passive/active long/short rows on one common scale."""

    spec.validate()
    names = tuple(str(name) for name in source_feature_names)
    values = np.asarray(source_features)
    decisions_raw = np.asarray(decision_time_ms)
    if (
        names != AGGREGATE_DEPTH_FEATURE_NAMES
        or values.ndim != 2
        or values.shape[1] != len(names)
        or values.shape[0] == 0
        or values.dtype.kind not in "fiu"
        or not np.isfinite(values).all()
        or decisions_raw.ndim != 1
        or decisions_raw.size != values.shape[0]
        or decisions_raw.dtype.kind not in "iu"
        or (decisions_raw.dtype.kind == "u" and np.any(decisions_raw > np.iinfo(np.int64).max))
        or not _is_sha256(source_dataset_sha256)
    ):
        raise ValueError("make/take source feature contract is invalid")
    decisions = np.array(decisions_raw, dtype=np.int64, order="C", copy=True)
    if (
        decisions[0] < 0
        or np.any(np.diff(decisions) <= 0)
        or not np.array_equal(decisions, flow.decision_time_ms)
    ):
        raise ValueError("make/take source decision times do not match causal flow")
    rows = values.shape[0]
    selected = _indexes(event_indexes, rows=rows)
    bid = _numeric_array(bid_price, name="make/take bid prices", rows=rows)
    ask = _numeric_array(ask_price, name="make/take ask prices", rows=rows)
    bid_qty = _numeric_array(bid_quantity, name="make/take bid quantities", rows=rows)
    ask_qty = _numeric_array(ask_quantity, name="make/take ask quantities", rows=rows)
    if np.any(bid >= ask):
        raise ValueError("make/take source quotes are crossed or locked")

    source = np.array(values[selected], dtype=np.float32, order="C", copy=True)
    base = build_action_conditional_features(
        source[:, : len(MICROSTRUCTURE_FEATURE_NAMES)],
        MICROSTRUCTURE_FEATURE_NAMES,
    )
    aggregate_names, aggregate = _aggregate_action_features(
        source[:, len(MICROSTRUCTURE_FEATURE_NAMES) :]
    )
    flow_names, directional_flow = _flow_action_features(flow, selected)
    event_rows = selected.size
    directional_indexes = np.column_stack(
        (
            np.arange(0, event_rows * 2, 2),
            np.arange(1, event_rows * 2, 2),
            np.arange(0, event_rows * 2, 2),
            np.arange(1, event_rows * 2, 2),
        )
    ).ravel()

    selected_bid = bid[selected]
    selected_ask = ask[selected]
    selected_bid_qty = bid_qty[selected]
    selected_ask_qty = ask_qty[selected]
    prices = np.column_stack(
        (selected_bid, selected_ask, selected_ask, selected_bid)
    ).ravel()
    displayed = np.column_stack(
        (selected_bid_qty, selected_ask_qty, selected_ask_qty, selected_bid_qty)
    ).ravel()
    queue = np.column_stack(
        (
            selected_bid_qty,
            selected_ask_qty,
            np.zeros(event_rows),
            np.zeros(event_rows),
        )
    ).ravel()
    own = float(spec.order_notional_quote) / prices
    participation = own / displayed
    passive = np.tile(np.asarray([1.0, 1.0, 0.0, 0.0]), event_rows)
    aggressive_long_ratio = selected_bid / selected_ask
    aggressive_short_ratio = selected_ask / selected_bid
    crossing = np.column_stack(
        (
            np.zeros(event_rows),
            np.zeros(event_rows),
            (1.0 - aggressive_long_ratio) * 10_000.0,
            (aggressive_short_ratio - 1.0) * 10_000.0,
        )
    ).ravel()
    exit_ratio = np.column_stack(
        (
            np.ones(event_rows),
            np.ones(event_rows),
            aggressive_long_ratio,
            aggressive_short_ratio,
        )
    ).ravel()
    entry_fee = np.where(
        passive > 0.5,
        float(spec.maker_entry_fee_bps),
        float(spec.taker_entry_fee_bps),
    )
    known_cost = (
        crossing
        + entry_fee
        + float(spec.additional_slippage_bps_per_side)
        + (
            float(spec.taker_exit_fee_bps)
            + float(spec.additional_slippage_bps_per_side)
        )
        * exit_ratio
    )
    action_specific = np.column_stack(
        (
            passive,
            np.log1p(queue * prices),
            np.log1p((queue + own) * prices) * passive,
            np.log1p(queue / own),
            participation,
            known_cost,
        )
    ).astype(np.float32)
    features = np.column_stack(
        (
            base.features[directional_indexes],
            aggregate[directional_indexes],
            directional_flow[directional_indexes],
            action_specific,
        )
    ).astype(np.float32)
    feature_names = (
        tuple(base.feature_names)
        + aggregate_names
        + flow_names
        + _ACTION_SPECIFIC_FEATURE_NAMES
    )
    action_code = np.tile(_ACTION_CODE_PATTERN, event_rows)
    action_side = np.tile(_ACTION_SIDE_PATTERN, event_rows)
    eligible = participation <= float(spec.max_l1_participation)
    payload = {
        "schema_version": MAKE_TAKE_ACTION_FEATURE_SCHEMA_VERSION,
        "spec_sha256": spec.spec_sha256,
        "source_dataset_sha256": str(source_dataset_sha256),
        "source_flow_sha256": flow.batch_sha256,
        "action_names": list(MAKE_TAKE_ACTION_NAMES),
        "feature_names": list(feature_names),
        "arrays": {
            "event_indexes": _array_sha256(selected),
            "action_code": _array_sha256(action_code),
            "action_side": _array_sha256(action_side),
            "eligible": _array_sha256(eligible),
            "features": _array_sha256(features),
        },
    }
    batch_sha256 = _sha256(payload)
    for array in (selected, action_code, action_side, eligible, features):
        array.setflags(write=False)
    return MakeTakeActionFeatureBatch(
        schema_version=MAKE_TAKE_ACTION_FEATURE_SCHEMA_VERSION,
        spec=spec,
        spec_sha256=spec.spec_sha256,
        source_dataset_sha256=str(source_dataset_sha256),
        source_flow_sha256=flow.batch_sha256,
        feature_names=feature_names,
        event_indexes=selected,
        action_code=action_code,
        action_side=action_side,
        eligible=eligible,
        features=features,
        batch_sha256=batch_sha256,
    )


__all__ = [
    "MAKE_TAKE_ACTION_FEATURE_SCHEMA_VERSION",
    "MAKE_TAKE_ACTION_NAMES",
    "MakeTakeActionFeatureBatch",
    "MakeTakeFeatureSpec",
    "build_make_take_action_features",
]
