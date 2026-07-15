"""Discrete passive-fill survival panels and proper scoring rules."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral

import numpy as np

from .make_take_action_features import MakeTakeActionFeatureBatch
from .make_take_scenario_entries import MakeTakeScenarioEntryBatch


PASSIVE_FILL_SURVIVAL_SCHEMA_VERSION = "passive-fill-discrete-survival-v1"
PASSIVE_FILL_PROBABILITY_NAMES = (
    "fill_0_5s",
    "fill_5_10s",
    "fill_10_15s",
    "no_fill_15s",
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
class PassiveFillSurvivalPanel:
    schema_version: str
    symbol: str
    feature_names: tuple[str, ...]
    source_action_feature_sha256: str
    source_entry_sha256: str
    source_dataset_sha256: str
    event_index: np.ndarray
    decision_time_ms: np.ndarray
    action_side: np.ndarray
    features: np.ndarray
    fill_bucket: np.ndarray
    panel_sha256: str

    @property
    def rows(self) -> int:
        return int(self.fill_bucket.size)

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "rows": self.rows,
            "feature_count": len(self.feature_names),
            "long_rows": int(np.count_nonzero(self.action_side == 1)),
            "short_rows": int(np.count_nonzero(self.action_side == -1)),
            "bucket_counts": {
                PASSIVE_FILL_PROBABILITY_NAMES[bucket - 1]
                if bucket > 0
                else PASSIVE_FILL_PROBABILITY_NAMES[-1]: int(
                    np.count_nonzero(self.fill_bucket == bucket)
                )
                for bucket in range(4)
            },
            "panel_sha256": self.panel_sha256,
            "trading_authority": False,
            "profitability_claim": False,
        }


@dataclass(frozen=True)
class HazardRiskSet:
    hazard_index: int
    source_rows: np.ndarray
    features: np.ndarray
    labels: np.ndarray


def build_passive_fill_survival_panel(
    action_features: MakeTakeActionFeatureBatch,
    entries: MakeTakeScenarioEntryBatch,
    *,
    symbol: str,
) -> PassiveFillSurvivalPanel:
    """Bind eligible passive action rows to exact right-censored fill buckets."""

    normalized_symbol = str(symbol).strip().upper()
    if (
        normalized_symbol not in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
        or action_features.decision_time_ms.shape != (action_features.event_rows,)
        or entries.scenario != "base"
        or action_features.spec.placement_latency_ms != entries.placement_latency_ms
        or action_features.spec.order_notional_quote != entries.order_notional_quote
        or action_features.spec.max_l1_participation != entries.max_l1_participation
        or action_features.spec.maker_entry_fee_bps != entries.passive_entry_fee_bps
        or action_features.spec.taker_entry_fee_bps
        != entries.aggressive_entry_fee_bps
        or action_features.spec.taker_exit_fee_bps != entries.exit_fee_bps
        or action_features.spec.additional_slippage_bps_per_side
        != entries.additional_slippage_bps_per_side
        or action_features.source_dataset_sha256 == ""
        or action_features.action_rows != action_features.event_rows * 4
        or entries.action_rows != entries.event_rows * 4
    ):
        raise ValueError("passive-fill survival source contract is invalid")
    event_indexes = np.asarray(action_features.event_indexes, dtype=np.int64)
    if event_indexes[-1] >= entries.event_rows:
        raise ValueError("passive-fill survival event indexes exceed entry evidence")
    local_passive = np.column_stack(
        (
            np.arange(0, action_features.action_rows, 4, dtype=np.int64),
            np.arange(1, action_features.action_rows, 4, dtype=np.int64),
        )
    ).ravel()
    source_passive = np.column_stack(
        (event_indexes * 4, event_indexes * 4 + 1)
    ).ravel()
    if (
        not np.array_equal(
            action_features.action_code[local_passive],
            entries.action_code[source_passive],
        )
        or not np.array_equal(
            action_features.action_side[local_passive],
            entries.action_side[source_passive],
        )
        or not np.array_equal(
            action_features.eligible[local_passive],
            entries.eligible[source_passive],
        )
    ):
        raise ValueError("passive-fill survival action rows drifted")
    keep = np.asarray(entries.eligible[source_passive], dtype=np.bool_)
    if not np.any(keep):
        raise ValueError("passive-fill survival has no eligible action rows")
    selected_local = local_passive[keep]
    selected_source = source_passive[keep]
    repeated_event_indexes = np.repeat(event_indexes, 2)[keep]
    repeated_decision_times = np.repeat(action_features.decision_time_ms, 2)[keep]
    action_side = np.array(
        entries.action_side[selected_source], dtype=np.int8, order="C", copy=True
    )
    features = np.array(
        action_features.features[selected_local], dtype=np.float32, order="C", copy=True
    )
    fill_bucket = np.array(
        entries.fill_bucket[selected_source], dtype=np.uint8, order="C", copy=True
    )
    if (
        np.any(fill_bucket > 3)
        or not np.isfinite(features).all()
        or not np.all(np.isin(action_side, (-1, 1)))
    ):
        raise ValueError("passive-fill survival labels or features are invalid")
    payload = {
        "schema_version": PASSIVE_FILL_SURVIVAL_SCHEMA_VERSION,
        "symbol": normalized_symbol,
        "feature_names": list(action_features.feature_names),
        "source_action_feature_sha256": action_features.batch_sha256,
        "source_entry_sha256": entries.batch_sha256,
        "source_dataset_sha256": action_features.source_dataset_sha256,
        "arrays": {
            "event_index": _array_sha256(repeated_event_indexes),
            "decision_time_ms": _array_sha256(repeated_decision_times),
            "action_side": _array_sha256(action_side),
            "features": _array_sha256(features),
            "fill_bucket": _array_sha256(fill_bucket),
        },
    }
    panel_sha256 = _sha256(payload)
    retained = (
        repeated_event_indexes,
        repeated_decision_times,
        action_side,
        features,
        fill_bucket,
    )
    for array in retained:
        array.setflags(write=False)
    return PassiveFillSurvivalPanel(
        schema_version=PASSIVE_FILL_SURVIVAL_SCHEMA_VERSION,
        symbol=normalized_symbol,
        feature_names=action_features.feature_names,
        source_action_feature_sha256=action_features.batch_sha256,
        source_entry_sha256=entries.batch_sha256,
        source_dataset_sha256=action_features.source_dataset_sha256,
        event_index=repeated_event_indexes,
        decision_time_ms=repeated_decision_times,
        action_side=action_side,
        features=features,
        fill_bucket=fill_bucket,
        panel_sha256=panel_sha256,
    )


def build_hazard_risk_set(
    panel: PassiveFillSurvivalPanel,
    hazard_index: int,
) -> HazardRiskSet:
    """Return rows still at risk and the conditional event label for one interval."""

    if isinstance(hazard_index, (bool, np.bool_)) or not isinstance(
        hazard_index, Integral
    ):
        raise ValueError("fill hazard index is invalid")
    head = int(hazard_index)
    if head not in (0, 1, 2):
        raise ValueError("fill hazard index is invalid")
    survived = np.ones(panel.rows, dtype=np.bool_)
    if head >= 1:
        survived &= panel.fill_bucket != 1
    if head >= 2:
        survived &= panel.fill_bucket != 2
    source_rows = np.flatnonzero(survived)
    labels = np.asarray(panel.fill_bucket[source_rows] == head + 1, dtype=np.float32)
    features = np.asarray(panel.features[source_rows], dtype=np.float32)
    if source_rows.size == 0 or not np.all(np.isin(labels, (0.0, 1.0))):
        raise ValueError("fill hazard risk set is empty or invalid")
    for array in (source_rows, labels, features):
        array.setflags(write=False)
    return HazardRiskSet(
        hazard_index=head,
        source_rows=source_rows,
        features=features,
        labels=labels,
    )


def hazards_to_bucket_probabilities(hazards: np.ndarray) -> np.ndarray:
    """Convert three conditional hazards to three fill buckets plus no-fill."""

    values = np.asarray(hazards, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] != 3
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("fill hazard probability matrix is invalid")
    first = values[:, 0]
    survive_first = 1.0 - first
    second = survive_first * values[:, 1]
    survive_second = survive_first * (1.0 - values[:, 1])
    third = survive_second * values[:, 2]
    no_fill = survive_second * (1.0 - values[:, 2])
    output = np.column_stack((first, second, third, no_fill))
    output /= np.sum(output, axis=1, keepdims=True)
    return output


def fill_bucket_prevalence(panel: PassiveFillSurvivalPanel) -> np.ndarray:
    """Return Laplace-smoothed chronological baseline probabilities."""

    counts = np.asarray(
        [
            np.count_nonzero(panel.fill_bucket == 1),
            np.count_nonzero(panel.fill_bucket == 2),
            np.count_nonzero(panel.fill_bucket == 3),
            np.count_nonzero(panel.fill_bucket == 0),
        ],
        dtype=np.float64,
    )
    return (counts + 1.0) / (panel.rows + 4.0)


def evaluate_fill_survival_probabilities(
    panel: PassiveFillSurvivalPanel,
    bucket_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
) -> dict[str, object]:
    """Evaluate censored fill distributions with log and integrated Brier scores."""

    probabilities = np.asarray(bucket_probabilities, dtype=np.float64)
    baseline = np.asarray(baseline_probabilities, dtype=np.float64)
    if (
        probabilities.shape != (panel.rows, 4)
        or baseline.shape != (4,)
        or not np.isfinite(probabilities).all()
        or not np.isfinite(baseline).all()
        or np.any(probabilities < 0.0)
        or np.any(baseline < 0.0)
        or not np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-10)
        or not np.isclose(np.sum(baseline), 1.0, atol=1e-10)
    ):
        raise ValueError("fill survival evaluation probabilities are invalid")
    class_index = np.where(panel.fill_bucket == 0, 3, panel.fill_bucket - 1).astype(
        np.int64
    )
    row_index = np.arange(panel.rows)
    epsilon = 1e-12
    log_loss = float(
        -np.mean(np.log(np.clip(probabilities[row_index, class_index], epsilon, 1.0)))
    )
    baseline_log_loss = float(
        -np.mean(np.log(np.clip(baseline[class_index], epsilon, 1.0)))
    )
    cumulative = np.cumsum(probabilities[:, :3], axis=1)
    baseline_cumulative = np.cumsum(baseline[:3])
    observations = np.column_stack(
        tuple(
            (panel.fill_bucket > 0) & (panel.fill_bucket <= bucket)
            for bucket in (1, 2, 3)
        )
    ).astype(np.float64)
    horizon_brier = np.mean((cumulative - observations) ** 2, axis=0)
    baseline_horizon_brier = np.mean(
        (baseline_cumulative[None, :] - observations) ** 2,
        axis=0,
    )
    integrated_brier = float(np.mean(horizon_brier))
    baseline_integrated_brier = float(np.mean(baseline_horizon_brier))
    return {
        "rows": panel.rows,
        "log_loss": log_loss,
        "baseline_log_loss": baseline_log_loss,
        "log_loss_skill": 1.0 - log_loss / baseline_log_loss,
        "integrated_brier": integrated_brier,
        "baseline_integrated_brier": baseline_integrated_brier,
        "integrated_brier_skill": 1.0
        - integrated_brier / baseline_integrated_brier,
        "horizon_brier": {
            f"{seconds}s": float(value)
            for seconds, value in zip((5, 10, 15), horizon_brier, strict=True)
        },
        "predicted_fill_probability": float(np.mean(cumulative[:, -1])),
        "observed_fill_ratio": float(np.mean(panel.fill_bucket > 0)),
        "profitability_claim": False,
    }


__all__ = [
    "PASSIVE_FILL_PROBABILITY_NAMES",
    "PASSIVE_FILL_SURVIVAL_SCHEMA_VERSION",
    "HazardRiskSet",
    "PassiveFillSurvivalPanel",
    "build_hazard_risk_set",
    "build_passive_fill_survival_panel",
    "evaluate_fill_survival_probabilities",
    "fill_bucket_prevalence",
    "hazards_to_bucket_probabilities",
]
