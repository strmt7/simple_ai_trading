"""One-use evaluation of the frozen BTC Polymarket feature-support gate."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .polymarket_historical_dataset import FEATURE_NAMES
from .polymarket_historical_model import (
    HistoricalModelPanel,
    condition_balanced_binary_metrics,
    load_historical_model_panel,
    predict_historical_candidate,
)
from .polymarket_historical_screen import HistoricalScreenStore
from .polymarket_historical_shadow import VerifiedHistoricalShadowPredictor
from .polymarket_historical_support import HistoricalFeatureSupportProfile


SUPPORT_EVALUATION_SCHEMA_VERSION = (
    "polymarket-historical-btc-feature-support-evaluation-v1"
)
_METRIC_NAMES = (
    "log_loss",
    "brier_score",
    "accuracy",
    "balanced_accuracy",
    "calibration_intercept",
    "calibration_slope",
    "expected_calibration_error",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_commit(value: str) -> str:
    commit = str(value or "").strip().lower()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("historical support evaluation source commit is invalid")
    return commit


def _finite_metrics(value: Mapping[str, float]) -> dict[str, float]:
    metrics = {name: float(value[name]) for name in _METRIC_NAMES}
    if any(not math.isfinite(item) for item in metrics.values()):
        raise ValueError("historical support evaluation metrics are nonfinite")
    return metrics


def _subset_panel(
    panel: HistoricalModelPanel,
    selected: np.ndarray,
) -> HistoricalModelPanel:
    return HistoricalModelPanel(
        condition_ids=panel.condition_ids[selected],
        roles=panel.roles[selected],
        event_start_ms=panel.event_start_ms[selected],
        decision_time_ms=panel.decision_time_ms[selected],
        features=panel.features[selected],
        labels=panel.labels[selected],
        dataset_sha256=panel.dataset_sha256,
    )


def evaluate_historical_support_role(
    panel: HistoricalModelPanel,
    probability: np.ndarray,
    profile: HistoricalFeatureSupportProfile,
) -> Mapping[str, object]:
    """Measure a frozen support rule without selecting or changing it."""

    role_values = tuple(str(item) for item in np.unique(panel.roles))
    if len(role_values) != 1 or role_values[0] not in {"tune", "test"}:
        raise ValueError("historical support evaluation role differs")
    panel.validate(expected_roles=role_values)
    prediction = np.asarray(probability, dtype=np.float64)
    if (
        prediction.shape != panel.labels.shape
        or np.any(~np.isfinite(prediction))
        or np.any((prediction <= 0.0) | (prediction >= 1.0))
    ):
        raise ValueError("historical support evaluation probability differs")

    outside = np.sum(
        (panel.features < profile.minimum)
        | (panel.features > profile.maximum),
        axis=1,
    )
    extreme = np.sum(
        (panel.features < profile.outer_lower)
        | (panel.features > profile.outer_upper),
        axis=1,
    )
    outside_failure = outside > profile.maximum_outside_training_range
    extreme_failure = extreme > profile.maximum_extreme_outliers
    admitted = ~(outside_failure | extreme_failure)
    if not np.any(admitted):
        raise ValueError("historical support evaluation admits no rows")

    admitted_panel = _subset_panel(panel, admitted)
    all_metrics = _finite_metrics(
        condition_balanced_binary_metrics(panel, prediction)
    )
    admitted_metrics = _finite_metrics(
        condition_balanced_binary_metrics(
            admitted_panel,
            prediction[admitted],
        )
    )
    unique_conditions, admitted_counts = np.unique(
        panel.condition_ids[admitted],
        return_counts=True,
    )
    total_condition_counts = Counter(
        str(condition) for condition in panel.condition_ids
    )
    conditions_all_rows = sum(
        int(admitted_count) == total_condition_counts[str(condition)]
        for condition, admitted_count in zip(
            unique_conditions,
            admitted_counts,
            strict=True,
        )
    )
    offsets = (
        (panel.decision_time_ms - panel.event_start_ms) // 1_000
    ).astype(np.int64)
    offset_rows = {}
    for offset in sorted(int(item) for item in np.unique(offsets)):
        selected = offsets == offset
        offset_rows[str(offset)] = {
            "total": int(np.count_nonzero(selected)),
            "admitted": int(np.count_nonzero(selected & admitted)),
            "abstained": int(np.count_nonzero(selected & ~admitted)),
        }

    outside_features = Counter[str]()
    extreme_features = Counter[str]()
    for index, name in enumerate(FEATURE_NAMES):
        outside_features[name] = int(
            np.count_nonzero(
                (panel.features[:, index] < profile.minimum[index])
                | (panel.features[:, index] > profile.maximum[index])
            )
        )
        extreme_features[name] = int(
            np.count_nonzero(
                (panel.features[:, index] < profile.outer_lower[index])
                | (panel.features[:, index] > profile.outer_upper[index])
            )
        )
    return {
        "role": role_values[0],
        "rows": int(len(panel.labels)),
        "admitted_rows": int(np.count_nonzero(admitted)),
        "abstained_rows": int(np.count_nonzero(~admitted)),
        "row_coverage": float(np.mean(admitted)),
        "conditions": int(len(np.unique(panel.condition_ids))),
        "conditions_with_any_admitted_row": int(len(unique_conditions)),
        "conditions_with_all_rows_admitted": int(conditions_all_rows),
        "abstention_causes": {
            "outside_training_range_limit": int(
                np.count_nonzero(outside_failure)
            ),
            "extreme_outer_fence": int(np.count_nonzero(extreme_failure)),
            "both": int(
                np.count_nonzero(outside_failure & extreme_failure)
            ),
        },
        "decision_offset_seconds": offset_rows,
        "outside_training_range_feature_counts": {
            name: count
            for name, count in outside_features.items()
            if count > 0
        },
        "extreme_outer_fence_feature_counts": {
            name: count
            for name, count in extreme_features.items()
            if count > 0
        },
        "challenger_metrics_all_rows": all_metrics,
        "challenger_metrics_admitted_rows": admitted_metrics,
        "admitted_minus_all_metric_delta": {
            name: admitted_metrics[name] - all_metrics[name]
            for name in _METRIC_NAMES
        },
    }


def evaluate_historical_feature_support_once(
    store: HistoricalScreenStore,
    predictor: VerifiedHistoricalShadowPredictor,
    *,
    historical_evaluation: Mapping[str, object],
    source_commit: str,
) -> tuple[Mapping[str, object], str]:
    """Evaluate the pre-frozen gate without granting execution authority."""

    if store.state != "evaluated":
        raise ValueError("historical support evaluation requires evaluated data")
    if predictor.trading_authority:
        raise ValueError("historical support evaluator cannot accept authority")
    candidate_metrics = historical_evaluation.get("candidate_metrics")
    if not isinstance(candidate_metrics, Mapping):
        raise ValueError("historical support baseline metrics are missing")
    frozen_test_metrics = candidate_metrics.get(predictor.candidate_id)
    if not isinstance(frozen_test_metrics, Mapping):
        raise ValueError("historical support challenger metrics are missing")

    roles: dict[str, Mapping[str, object]] = {}
    for role in ("tune", "test"):
        panel = load_historical_model_panel(store, roles=(role,))
        if panel.dataset_sha256 != predictor.dataset_sha256:
            raise ValueError("historical support evaluation dataset differs")
        probability = predict_historical_candidate(
            predictor.candidate,
            panel.features,
        )
        roles[role] = evaluate_historical_support_role(
            panel,
            probability,
            predictor.support_profile,
        )

    recomputed_test = roles["test"]["challenger_metrics_all_rows"]
    if not isinstance(recomputed_test, Mapping) or any(
        not math.isclose(
            float(recomputed_test[name]),
            float(frozen_test_metrics[name]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for name in _METRIC_NAMES
    ):
        raise ValueError("historical support baseline reproduction failed")

    source_root = Path(__file__).parent
    artifact: dict[str, object] = {
        "schema_version": SUPPORT_EVALUATION_SCHEMA_VERSION,
        "contract_sha256": store.contract.contract_sha256,
        "dataset_sha256": predictor.dataset_sha256,
        "pretest_artifact_sha256": predictor.pretest_artifact_sha256,
        "historical_evaluation_artifact_sha256": (
            predictor.evaluation_artifact_sha256
        ),
        "support_profile_sha256": (
            predictor.support_profile.artifact_sha256
        ),
        "source_commit": _source_commit(source_commit),
        "implementation_sha256": {
            "evaluation": _file_sha256(Path(__file__)),
            "support": _file_sha256(
                source_root / "polymarket_historical_support.py"
            ),
            "model": _file_sha256(
                source_root / "polymarket_historical_model.py"
            ),
            "dataset": _file_sha256(
                source_root / "polymarket_historical_dataset.py"
            ),
        },
        "policy": {
            "support_frozen_before_test_access": True,
            "gate_changed_after_evaluation": False,
            "test_driven_gate_changes_allowed": False,
            "safety_abstention_only": True,
        },
        "roles": roles,
        "conclusion": {
            "predictive_improvement_claim": False,
            "execution_or_profitability_claim": False,
            "trading_authority": False,
            "interpretation": (
                "The gate is retained only as a rare out-of-distribution "
                "abstention; admitted-row metrics do not establish improved "
                "predictive performance."
            ),
        },
        "trading_authority": False,
        "profitability_claim": False,
    }
    artifact_sha = _canonical_sha256(artifact)
    return {**artifact, "artifact_sha256": artifact_sha}, artifact_sha


__all__ = [
    "SUPPORT_EVALUATION_SCHEMA_VERSION",
    "evaluate_historical_feature_support_once",
    "evaluate_historical_support_role",
]
