"""Hash-bound conditional payoff panels for shared make/take models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .make_take_action_features import (
    MAKE_TAKE_ACTION_NAMES,
    MAKE_TAKE_ACTION_FEATURE_SCHEMA_VERSION,
    MakeTakeActionFeatureBatch,
)
from .make_take_path_payoffs import ACTION_PATH_HORIZON_SECONDS
from .make_take_scenario_entries import (
    MAKE_TAKE_SCENARIO_ENTRY_SCHEMA_VERSION,
    MakeTakeScenarioEntryBatch,
)
from .make_take_targets import (
    MAKE_TAKE_TARGET_SCHEMA_VERSION,
    MakeTakeTargetBatch,
)


MAKE_TAKE_PAYOFF_PANEL_SCHEMA_VERSION = "queue-censored-make-take-payoff-panel-v1"
MAKE_TAKE_PAYOFF_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
_ACTION_CODE_PATTERN = np.arange(4, dtype=np.uint8)


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


@dataclass(frozen=True)
class MakeTakeConditionalPayoffPanel:
    schema_version: str
    scenario: str
    symbol: str
    feature_names: tuple[str, ...]
    source_feature_spec_sha256: str
    source_action_feature_sha256: str
    source_entry_sha256: str
    source_target_sha256: str
    source_dataset_sha256: str
    source_first_decision_time_ms: int
    source_last_decision_time_ms: int
    source_label_end_ms: int
    event_index: np.ndarray
    decision_time_ms: np.ndarray
    action_code: np.ndarray
    action_side: np.ndarray
    features: np.ndarray
    net_bps: np.ndarray
    markout_5s_bps: np.ndarray
    markout_15s_bps: np.ndarray
    terminal_time_ms: np.ndarray
    stop_bps: np.ndarray
    take_bps: np.ndarray
    panel_sha256: str

    @property
    def rows(self) -> int:
        return int(self.action_code.size)

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scenario": self.scenario,
            "symbol": self.symbol,
            "rows": self.rows,
            "feature_count": len(self.feature_names),
            "rows_by_action": {
                name: int(np.count_nonzero(self.action_code == offset))
                for offset, name in enumerate(MAKE_TAKE_ACTION_NAMES)
            },
            "target_mean_bps": float(np.mean(self.net_bps, dtype=np.float64)),
            "adverse_markout_5s_fraction": float(np.mean(self.markout_5s_bps < 0.0)),
            "adverse_markout_15s_fraction": float(np.mean(self.markout_15s_bps < 0.0)),
            "source_label_end_ms": self.source_label_end_ms,
            "panel_sha256": self.panel_sha256,
            "trading_authority": False,
            "profitability_claim": False,
        }


def _panel_payload(panel: MakeTakeConditionalPayoffPanel) -> dict[str, object]:
    return {
        "schema_version": panel.schema_version,
        "scenario": panel.scenario,
        "symbol": panel.symbol,
        "feature_names": list(panel.feature_names),
        "source_feature_spec_sha256": panel.source_feature_spec_sha256,
        "source_action_feature_sha256": panel.source_action_feature_sha256,
        "source_entry_sha256": panel.source_entry_sha256,
        "source_target_sha256": panel.source_target_sha256,
        "source_dataset_sha256": panel.source_dataset_sha256,
        "source_first_decision_time_ms": panel.source_first_decision_time_ms,
        "source_last_decision_time_ms": panel.source_last_decision_time_ms,
        "source_label_end_ms": panel.source_label_end_ms,
        "arrays": {
            "event_index": _array_sha256(panel.event_index),
            "decision_time_ms": _array_sha256(panel.decision_time_ms),
            "action_code": _array_sha256(panel.action_code),
            "action_side": _array_sha256(panel.action_side),
            "features": _array_sha256(panel.features),
            "net_bps": _array_sha256(panel.net_bps),
            "markout_5s_bps": _array_sha256(panel.markout_5s_bps),
            "markout_15s_bps": _array_sha256(panel.markout_15s_bps),
            "terminal_time_ms": _array_sha256(panel.terminal_time_ms),
            "stop_bps": _array_sha256(panel.stop_bps),
            "take_bps": _array_sha256(panel.take_bps),
        },
    }


def validate_make_take_conditional_payoff_panel(
    panel: MakeTakeConditionalPayoffPanel,
) -> None:
    """Validate all arrays and independently recompute the panel identity."""

    rows = panel.rows
    vectors = (
        panel.event_index,
        panel.decision_time_ms,
        panel.action_code,
        panel.action_side,
        panel.net_bps,
        panel.markout_5s_bps,
        panel.markout_15s_bps,
        panel.terminal_time_ms,
        panel.stop_bps,
        panel.take_bps,
    )
    if (
        panel.schema_version != MAKE_TAKE_PAYOFF_PANEL_SCHEMA_VERSION
        or panel.scenario != "base"
        or panel.symbol not in MAKE_TAKE_PAYOFF_SYMBOLS
        or rows <= 0
        or not panel.feature_names
        or len(set(panel.feature_names)) != len(panel.feature_names)
        or any(not isinstance(name, str) or not name.strip() for name in panel.feature_names)
        or any(
            not _is_sha256(value)
            for value in (
                panel.source_feature_spec_sha256,
                panel.source_action_feature_sha256,
                panel.source_entry_sha256,
                panel.source_target_sha256,
                panel.source_dataset_sha256,
                panel.panel_sha256,
            )
        )
        or any(np.asarray(value).shape != (rows,) for value in vectors)
        or np.asarray(panel.features).shape != (rows, len(panel.feature_names))
        or np.any(np.diff(panel.decision_time_ms) < 0)
        or np.any(panel.event_index < 0)
        or not np.all(np.isin(panel.action_code, _ACTION_CODE_PATTERN))
        or not np.all(np.isin(panel.action_side, (-1, 1)))
        or not np.array_equal(
            panel.action_side,
            np.where(np.isin(panel.action_code, (0, 2)), 1, -1),
        )
        or not np.isfinite(panel.features).all()
        or not np.isfinite(panel.net_bps).all()
        or not np.isfinite(panel.markout_5s_bps).all()
        or not np.isfinite(panel.markout_15s_bps).all()
        or not np.isfinite(panel.stop_bps).all()
        or not np.isfinite(panel.take_bps).all()
        or np.any(panel.stop_bps < 18.0)
        or np.any(panel.stop_bps > 80.0)
        or np.any(panel.take_bps < 30.0)
        or np.any(panel.take_bps > 120.0)
        or np.any(panel.take_bps <= panel.stop_bps)
        or np.any(panel.terminal_time_ms < panel.decision_time_ms)
        or panel.source_first_decision_time_ms > int(np.min(panel.decision_time_ms))
        or panel.source_last_decision_time_ms < int(np.max(panel.decision_time_ms))
        or panel.source_first_decision_time_ms > panel.source_last_decision_time_ms
        or panel.source_label_end_ms <= panel.source_last_decision_time_ms
        or int(np.max(panel.terminal_time_ms)) > panel.source_label_end_ms
        or panel.panel_sha256 != _sha256(_panel_payload(panel))
    ):
        raise ValueError("make/take conditional payoff panel is invalid")


def build_make_take_conditional_payoff_panel(
    *,
    symbol: str,
    action_features: MakeTakeActionFeatureBatch,
    entries: MakeTakeScenarioEntryBatch,
    targets: MakeTakeTargetBatch,
) -> MakeTakeConditionalPayoffPanel:
    """Join observable action features only to executed conditional payoffs."""

    normalized_symbol = str(symbol).strip().upper()
    event_indexes = np.asarray(action_features.event_indexes, dtype=np.int64)
    if (
        normalized_symbol not in MAKE_TAKE_PAYOFF_SYMBOLS
        or action_features.schema_version != MAKE_TAKE_ACTION_FEATURE_SCHEMA_VERSION
        or entries.schema_version != MAKE_TAKE_SCENARIO_ENTRY_SCHEMA_VERSION
        or targets.schema_version != MAKE_TAKE_TARGET_SCHEMA_VERSION
        or entries.scenario != "base"
        or targets.scenario != "base"
        or targets.symbol != normalized_symbol
        or action_features.source_dataset_sha256 != targets.source_dataset_sha256
        or targets.source_entry_sha256 != entries.batch_sha256
        or entries.action_rows != entries.event_rows * 4
        or targets.action_rows != entries.action_rows
        or action_features.action_rows != action_features.event_rows * 4
        or event_indexes.size == 0
        or event_indexes[0] < 0
        or event_indexes[-1] >= entries.event_rows
        or np.any(np.diff(event_indexes) <= 0)
    ):
        raise ValueError("make/take payoff panel source contract is invalid")
    if (
        action_features.spec.placement_latency_ms != entries.placement_latency_ms
        or action_features.spec.order_notional_quote != entries.order_notional_quote
        or action_features.spec.max_l1_participation != entries.max_l1_participation
        or action_features.spec.maker_entry_fee_bps != entries.passive_entry_fee_bps
        or action_features.spec.taker_entry_fee_bps != entries.aggressive_entry_fee_bps
        or action_features.spec.taker_exit_fee_bps != entries.exit_fee_bps
        or action_features.spec.additional_slippage_bps_per_side
        != entries.additional_slippage_bps_per_side
    ):
        raise ValueError("make/take payoff panel execution contract drifted")
    source_rows = (
        event_indexes[:, None] * 4 + _ACTION_CODE_PATTERN.astype(np.int64)[None, :]
    ).ravel()
    source_decisions = (
        entries.order_start_time_ms[source_rows] - entries.placement_latency_ms
    )
    local_decisions = np.repeat(action_features.decision_time_ms, 4)
    if (
        not np.array_equal(action_features.action_code, entries.action_code[source_rows])
        or not np.array_equal(action_features.action_side, entries.action_side[source_rows])
        or not np.array_equal(action_features.eligible, entries.eligible[source_rows])
        or not np.array_equal(entries.action_code, targets.action_code)
        or not np.array_equal(entries.action_side, targets.action_side)
        or not np.array_equal(entries.eligible, targets.eligible)
        or not np.array_equal(entries.filled, targets.filled)
        or not np.array_equal(entries.fill_bucket, targets.fill_bucket)
        or not np.array_equal(source_decisions, local_decisions)
    ):
        raise ValueError("make/take payoff panel action alignment drifted")
    valid = np.asarray(targets.conditional_payoff_valid[source_rows], dtype=np.bool_)
    if np.any(valid & (~entries.eligible[source_rows] | ~entries.filled[source_rows])):
        raise ValueError("make/take conditional payoff includes an unexecuted action")
    keep = valid & np.asarray(action_features.eligible, dtype=np.bool_)
    if not np.any(keep):
        raise ValueError("make/take payoff panel has no conditional payoff rows")
    selected_source = source_rows[keep]
    selected_local = np.flatnonzero(keep)
    source_last = int(action_features.decision_time_ms[-1])
    maximum_lifecycle_ms = (
        entries.placement_latency_ms
        + entries.passive_expiry_ms
        + ACTION_PATH_HORIZON_SECONDS * 1_000
    )
    if source_last > np.iinfo(np.int64).max - maximum_lifecycle_ms:
        raise ValueError("make/take payoff panel label horizon overflows")
    arrays = {
        "event_index": np.repeat(event_indexes, 4)[keep].astype(np.int64, copy=True),
        "decision_time_ms": local_decisions[keep].astype(np.int64, copy=True),
        "action_code": action_features.action_code[selected_local].astype(np.uint8, copy=True),
        "action_side": action_features.action_side[selected_local].astype(np.int8, copy=True),
        "features": np.array(
            action_features.features[selected_local], dtype=np.float32, order="C", copy=True
        ),
        "net_bps": targets.conditional_net_bps[selected_source].astype(np.float64, copy=True),
        "markout_5s_bps": targets.markout_5s_bps[selected_source].astype(
            np.float64, copy=True
        ),
        "markout_15s_bps": targets.markout_15s_bps[selected_source].astype(
            np.float64, copy=True
        ),
        "terminal_time_ms": targets.terminal_time_ms[selected_source].astype(
            np.int64, copy=True
        ),
        "stop_bps": targets.stop_bps[selected_source].astype(np.float64, copy=True),
        "take_bps": targets.take_bps[selected_source].astype(np.float64, copy=True),
    }
    provisional = MakeTakeConditionalPayoffPanel(
        schema_version=MAKE_TAKE_PAYOFF_PANEL_SCHEMA_VERSION,
        scenario="base",
        symbol=normalized_symbol,
        feature_names=action_features.feature_names,
        source_feature_spec_sha256=action_features.spec_sha256,
        source_action_feature_sha256=action_features.batch_sha256,
        source_entry_sha256=entries.batch_sha256,
        source_target_sha256=targets.target_sha256,
        source_dataset_sha256=action_features.source_dataset_sha256,
        source_first_decision_time_ms=int(action_features.decision_time_ms[0]),
        source_last_decision_time_ms=source_last,
        source_label_end_ms=source_last + maximum_lifecycle_ms,
        panel_sha256="",
        **arrays,
    )
    panel = MakeTakeConditionalPayoffPanel(
        **{**provisional.__dict__, "panel_sha256": _sha256(_panel_payload(provisional))}
    )
    for array in arrays.values():
        array.setflags(write=False)
    validate_make_take_conditional_payoff_panel(panel)
    return panel


__all__ = [
    "MAKE_TAKE_PAYOFF_PANEL_SCHEMA_VERSION",
    "MAKE_TAKE_PAYOFF_SYMBOLS",
    "MakeTakeConditionalPayoffPanel",
    "build_make_take_conditional_payoff_panel",
    "validate_make_take_conditional_payoff_panel",
]
