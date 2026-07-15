"""Exact fill-aware action values for passive and aggressive entries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .make_take_action_features import (
    MAKE_TAKE_ACTION_FEATURE_SCHEMA_VERSION,
    MAKE_TAKE_ACTION_NAMES,
    MakeTakeActionFeatureBatch,
)
from .make_take_payoff_lightgbm import MakeTakePayoffPredictionBatch
from .make_take_payoff_panel import MAKE_TAKE_PAYOFF_SYMBOLS
from .queue_fill_lightgbm import QueueFillPredictionBatch


MAKE_TAKE_ACTION_VALUE_SCHEMA_VERSION = "fill-aware-make-take-action-values-v1"
_ACTION_CODES = np.arange(4, dtype=np.uint8)
_ACTION_SIDES = np.asarray([1, -1, 1, -1], dtype=np.int8)


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
class MakeTakeActionValueBatch:
    schema_version: str
    symbol: str
    source_action_feature_sha256: str
    source_fill_panel_sha256: str
    fill_model_sha256: str
    payoff_model_sha256: str
    event_index: np.ndarray
    decision_time_ms: np.ndarray
    action_code: np.ndarray
    action_side: np.ndarray
    eligible: np.ndarray
    fill_probability_15s: np.ndarray
    conditional_mean_bps: np.ndarray
    conditional_q20_bps: np.ndarray
    expected_mean_bps: np.ndarray
    batch_sha256: str

    @property
    def rows(self) -> int:
        return int(self.action_code.size)

    @property
    def event_rows(self) -> int:
        return self.rows // 4

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "event_rows": self.event_rows,
            "action_rows": self.rows,
            "by_action": {
                name: {
                    "eligible_rows": int(np.count_nonzero(self.eligible[offset::4])),
                    "mean_fill_probability_15s": float(
                        np.mean(self.fill_probability_15s[offset::4])
                    ),
                    "mean_expected_value_bps": float(
                        np.mean(self.expected_mean_bps[offset::4])
                    ),
                }
                for offset, name in enumerate(MAKE_TAKE_ACTION_NAMES)
            },
            "batch_sha256": self.batch_sha256,
            "trading_authority": False,
            "profitability_claim": False,
        }


def _batch_payload(batch: MakeTakeActionValueBatch) -> dict[str, object]:
    return {
        "schema_version": batch.schema_version,
        "symbol": batch.symbol,
        "source_action_feature_sha256": batch.source_action_feature_sha256,
        "source_fill_panel_sha256": batch.source_fill_panel_sha256,
        "fill_model_sha256": batch.fill_model_sha256,
        "payoff_model_sha256": batch.payoff_model_sha256,
        "arrays": {
            "event_index": _array_sha256(batch.event_index),
            "decision_time_ms": _array_sha256(batch.decision_time_ms),
            "action_code": _array_sha256(batch.action_code),
            "action_side": _array_sha256(batch.action_side),
            "eligible": _array_sha256(batch.eligible),
            "fill_probability_15s": _array_sha256(batch.fill_probability_15s),
            "conditional_mean_bps": _array_sha256(batch.conditional_mean_bps),
            "conditional_q20_bps": _array_sha256(batch.conditional_q20_bps),
            "expected_mean_bps": _array_sha256(batch.expected_mean_bps),
        },
    }


def validate_make_take_action_value_batch(batch: MakeTakeActionValueBatch) -> None:
    rows = batch.rows
    if rows <= 0 or rows % 4 != 0:
        raise ValueError("make/take action value batch is invalid")
    event_index = np.asarray(batch.event_index).reshape(-1, 4)
    decision_time = np.asarray(batch.decision_time_ms).reshape(-1, 4)
    vectors = (
        batch.event_index,
        batch.decision_time_ms,
        batch.action_code,
        batch.action_side,
        batch.eligible,
        batch.fill_probability_15s,
        batch.conditional_mean_bps,
        batch.conditional_q20_bps,
        batch.expected_mean_bps,
    )
    if (
        batch.schema_version != MAKE_TAKE_ACTION_VALUE_SCHEMA_VERSION
        or batch.symbol not in MAKE_TAKE_PAYOFF_SYMBOLS
        or any(
            not _is_sha256(value)
            for value in (
                batch.source_action_feature_sha256,
                batch.source_fill_panel_sha256,
                batch.fill_model_sha256,
                batch.payoff_model_sha256,
                batch.batch_sha256,
            )
        )
        or any(np.asarray(value).shape != (rows,) for value in vectors)
        or np.asarray(batch.eligible).dtype != np.dtype(np.bool_)
        or not np.array_equal(batch.action_code, np.tile(_ACTION_CODES, batch.event_rows))
        or not np.array_equal(batch.action_side, np.tile(_ACTION_SIDES, batch.event_rows))
        or not np.all(event_index == event_index[:, :1])
        or not np.all(decision_time == decision_time[:, :1])
        or np.any(np.diff(event_index[:, 0]) <= 0)
        or np.any(np.diff(decision_time[:, 0]) <= 0)
        or np.any(batch.event_index < 0)
        or not np.isfinite(batch.fill_probability_15s).all()
        or np.any(batch.fill_probability_15s < 0.0)
        or np.any(batch.fill_probability_15s > 1.0)
        or not np.isfinite(batch.conditional_mean_bps).all()
        or not np.isfinite(batch.conditional_q20_bps).all()
        or not np.isfinite(batch.expected_mean_bps).all()
        or np.any(batch.conditional_q20_bps > batch.conditional_mean_bps + 1e-12)
        or not np.allclose(
            batch.expected_mean_bps,
            batch.fill_probability_15s * batch.conditional_mean_bps,
            atol=1e-12,
            rtol=1e-12,
        )
        or np.any(batch.fill_probability_15s[2::4] != 1.0)
        or np.any(batch.fill_probability_15s[3::4] != 1.0)
        or np.any(
            batch.fill_probability_15s[np.isin(batch.action_code, (0, 1)) & ~batch.eligible]
            != 0.0
        )
        or batch.batch_sha256 != _sha256(_batch_payload(batch))
    ):
        raise ValueError("make/take action value batch is invalid")


def build_make_take_action_values(
    *,
    symbol: str,
    action_features: MakeTakeActionFeatureBatch,
    fill_predictions: QueueFillPredictionBatch,
    payoff_predictions: MakeTakePayoffPredictionBatch,
) -> MakeTakeActionValueBatch:
    """Join exact passive fills with conditional after-cost action payoffs."""

    normalized_symbol = str(symbol).strip().upper()
    fill_predictions.__post_init__()
    payoff_predictions.__post_init__()
    event_rows = action_features.event_rows
    action_rows = action_features.action_rows
    expected_code = np.tile(_ACTION_CODES, event_rows)
    expected_side = np.tile(_ACTION_SIDES, event_rows)
    event_index = np.repeat(action_features.event_indexes, 4)
    decision_time = np.repeat(action_features.decision_time_ms, 4)
    if (
        normalized_symbol not in MAKE_TAKE_PAYOFF_SYMBOLS
        or action_features.schema_version != MAKE_TAKE_ACTION_FEATURE_SCHEMA_VERSION
        or event_rows <= 0
        or action_rows != event_rows * 4
        or not _is_sha256(action_features.batch_sha256)
        or action_features.event_indexes.shape != (event_rows,)
        or action_features.decision_time_ms.shape != (event_rows,)
        or action_features.eligible.shape != (action_rows,)
        or not np.array_equal(action_features.action_code, expected_code)
        or not np.array_equal(action_features.action_side, expected_side)
        or fill_predictions.symbol != normalized_symbol
        or payoff_predictions.symbol != normalized_symbol
        or fill_predictions.source_action_feature_sha256 != action_features.batch_sha256
        or payoff_predictions.source_action_feature_sha256 != action_features.batch_sha256
        or payoff_predictions.rows != action_rows
        or not np.array_equal(payoff_predictions.event_index, event_index)
        or not np.array_equal(payoff_predictions.decision_time_ms, decision_time)
        or not np.array_equal(payoff_predictions.action_code, expected_code)
        or not np.array_equal(payoff_predictions.action_side, expected_side)
    ):
        raise ValueError("make/take action value source contract is invalid")
    passive_local = np.column_stack(
        (
            np.arange(0, action_rows, 4, dtype=np.int64),
            np.arange(1, action_rows, 4, dtype=np.int64),
        )
    ).ravel()
    passive_eligible = np.asarray(action_features.eligible[passive_local], dtype=np.bool_)
    selected_passive = passive_local[passive_eligible]
    if (
        fill_predictions.rows != selected_passive.size
        or not np.array_equal(fill_predictions.event_index, event_index[selected_passive])
        or not np.array_equal(
            fill_predictions.decision_time_ms, decision_time[selected_passive]
        )
        or not np.array_equal(fill_predictions.action_side, expected_side[selected_passive])
    ):
        raise ValueError("make/take passive fill predictions are misaligned")
    fill_probability = np.ones(action_rows, dtype=np.float64)
    fill_probability[passive_local] = 0.0
    fill_probability[selected_passive] = fill_predictions.fill_probability_15s
    conditional_mean = np.array(
        payoff_predictions.conditional_mean_bps,
        dtype=np.float64,
        order="C",
        copy=True,
    )
    conditional_q20 = np.array(
        payoff_predictions.conditional_q20_bps,
        dtype=np.float64,
        order="C",
        copy=True,
    )
    expected_mean = fill_probability * conditional_mean
    arrays = {
        "event_index": np.array(event_index, dtype=np.int64, order="C", copy=True),
        "decision_time_ms": np.array(
            decision_time, dtype=np.int64, order="C", copy=True
        ),
        "action_code": np.array(expected_code, dtype=np.uint8, order="C", copy=True),
        "action_side": np.array(expected_side, dtype=np.int8, order="C", copy=True),
        "eligible": np.array(
            action_features.eligible, dtype=np.bool_, order="C", copy=True
        ),
        "fill_probability_15s": fill_probability,
        "conditional_mean_bps": conditional_mean,
        "conditional_q20_bps": conditional_q20,
        "expected_mean_bps": expected_mean,
    }
    provisional = MakeTakeActionValueBatch(
        schema_version=MAKE_TAKE_ACTION_VALUE_SCHEMA_VERSION,
        symbol=normalized_symbol,
        source_action_feature_sha256=action_features.batch_sha256,
        source_fill_panel_sha256=fill_predictions.source_panel_sha256,
        fill_model_sha256=fill_predictions.model_sha256,
        payoff_model_sha256=payoff_predictions.model_sha256,
        batch_sha256="",
        **arrays,
    )
    batch = MakeTakeActionValueBatch(
        **{**provisional.__dict__, "batch_sha256": _sha256(_batch_payload(provisional))}
    )
    for array in arrays.values():
        array.setflags(write=False)
    validate_make_take_action_value_batch(batch)
    return batch


__all__ = [
    "MAKE_TAKE_ACTION_VALUE_SCHEMA_VERSION",
    "MakeTakeActionValueBatch",
    "build_make_take_action_values",
    "validate_make_take_action_value_batch",
]
