"""One-use held-out predictive evaluation for Polymarket Round 16."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
import time
from typing import Mapping

import numpy as np

from .polymarket_historical_model import condition_balanced_binary_metrics
from .polymarket_historical_screen import HistoricalScreenStore
from .polymarket_round16 import (
    ROUND16_MARKETS_PER_DAY,
    ROUND16_TEST_DAYS,
    Round16HistoricalContract,
)
from .polymarket_round16_model import (
    ROUND16_PRETEST_SCHEMA_VERSION,
    Round16ModelPanel,
    load_round16_model_panel,
    predict_round16_candidate,
    round16_feature_support_admission,
    round16_settlement_admission_mask,
)


ROUND16_EVALUATION_SCHEMA_VERSION = "polymarket-round16-btc-15m-evaluation-v2"
ROUND16_BOOTSTRAP_SEED = 16_016


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


def _probability(value: np.ndarray) -> np.ndarray:
    output = np.clip(np.asarray(value, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if output.ndim != 1 or not np.all(np.isfinite(output)):
        raise ValueError("Round 16 probabilities are invalid")
    return output


def _ordered_conditions(condition_ids: np.ndarray) -> tuple[object, ...]:
    return tuple(dict.fromkeys(np.asarray(condition_ids, dtype=object).tolist()))


def _condition_loss_delta(
    panel: Round16ModelPanel,
    control: np.ndarray,
    challenger: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    truth = panel.labels
    control_probability = _probability(control)
    challenger_probability = _probability(challenger)
    control_loss = -(
        truth * np.log(control_probability)
        + (1.0 - truth) * np.log1p(-control_probability)
    )
    challenger_loss = -(
        truth * np.log(challenger_probability)
        + (1.0 - truth) * np.log1p(-challenger_probability)
    )
    conditions = _ordered_conditions(panel.condition_ids)
    delta = control_loss - challenger_loss
    values: list[float] = []
    utc_days: list[str] = []
    for condition in conditions:
        selected = np.flatnonzero(panel.condition_ids == condition)
        event_start_ms = np.unique(panel.event_start_ms[selected])
        if len(event_start_ms) != 1:
            raise ValueError("Round 16 condition spans multiple event starts")
        values.append(float(np.mean(delta[selected])))
        utc_days.append(
            datetime.fromtimestamp(
                int(event_start_ms[0]) / 1_000,
                tz=UTC,
            ).date().isoformat()
        )
    return np.asarray(values, dtype=np.float64), tuple(utc_days)


def _daily_loss_delta(
    condition_delta: np.ndarray,
    condition_utc_days: tuple[str, ...],
    expected_utc_days: tuple[str, ...],
) -> tuple[np.ndarray, Mapping[str, int]]:
    values = np.asarray(condition_delta, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) != len(condition_utc_days)
        or not np.all(np.isfinite(values))
        or len(set(expected_utc_days)) != len(expected_utc_days)
    ):
        raise ValueError("Round 16 daily loss inputs are invalid")
    grouped: dict[str, list[float]] = {}
    for day, value in zip(condition_utc_days, values, strict=True):
        grouped.setdefault(day, []).append(float(value))
    daily_delta = np.asarray(
        [
            float(np.mean(grouped[day]))
            for day in expected_utc_days
            if day in grouped
        ],
        dtype=np.float64,
    )
    return daily_delta, {
        day: len(grouped.get(day, ()))
        for day in (*expected_utc_days, *sorted(set(grouped) - set(expected_utc_days)))
    }


def _paired_utc_day_bootstrap(
    daily_delta: np.ndarray,
    *,
    repetitions: int,
) -> Mapping[str, float | int | str]:
    values = np.asarray(daily_delta, dtype=np.float64)
    selected_repetitions = int(repetitions)
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.all(np.isfinite(values))
        or not 100 <= selected_repetitions <= 100_000
    ):
        raise ValueError("Round 16 paired UTC-day bootstrap inputs are invalid")
    generator = np.random.default_rng(ROUND16_BOOTSTRAP_SEED)
    estimates = np.empty(selected_repetitions, dtype=np.float64)
    batch_size = 256
    for first in range(0, selected_repetitions, batch_size):
        count = min(batch_size, selected_repetitions - first)
        indexes = generator.integers(
            0,
            len(values),
            size=(count, len(values)),
        )
        sampled = values[indexes]
        estimates[first : first + count] = np.mean(sampled, axis=1)
    return {
        "repetitions": selected_repetitions,
        "resampling_unit": "whole_UTC_day",
        "day_count": len(values),
        "within_day_aggregation": "equal_weight_condition_mean",
        "lower_95": float(np.quantile(estimates, 0.025)),
        "median": float(np.quantile(estimates, 0.5)),
        "upper_95": float(np.quantile(estimates, 0.975)),
    }


def _verify_candidate(candidate: Mapping[str, object]) -> None:
    body = dict(candidate)
    claimed = str(body.pop("artifact_sha256", ""))
    if len(claimed) != 64 or _canonical_sha256(body) != claimed:
        raise ValueError("Round 16 candidate artifact integrity failed")


def _settlement_screen_report(
    panel: Round16ModelPanel,
    pretest: Mapping[str, object],
) -> Mapping[str, object]:
    controls = pretest.get("settlement_manipulation_controls")
    if not isinstance(controls, Mapping):
        raise ValueError("Round 16 settlement screen is missing")
    admitted = round16_settlement_admission_mask(panel.features, controls)
    remaining_ms = panel.event_start_ms + 900_000 - panel.decision_time_ms

    def coverage(selected: np.ndarray) -> Mapping[str, int]:
        rows = int(np.count_nonzero(selected))
        admitted_rows = int(np.count_nonzero(admitted & selected))
        return {
            "rows": rows,
            "admitted_rows": admitted_rows,
            "abstained_rows": rows - admitted_rows,
        }

    all_rows = np.ones(len(panel.labels), dtype=np.bool_)
    return {
        "action": "abstain",
        "threshold_partition": "tune",
        "labels_used_to_select_thresholds": False,
        "all_decisions": coverage(all_rows),
        "last_180_seconds": coverage(remaining_ms <= 180_000),
        "last_120_seconds": coverage(remaining_ms <= 120_000),
        "last_60_seconds": coverage(remaining_ms <= 60_000),
        "changes_predictive_metrics": False,
        "paper_authority": False,
        "live_authority": False,
    }


def _feature_support_report(
    panel: Round16ModelPanel,
    pretest: Mapping[str, object],
) -> Mapping[str, object]:
    support = pretest.get("feature_support")
    if not isinstance(support, Mapping):
        raise ValueError("Round 16 feature-support screen is missing")
    admitted, outside, extreme = round16_feature_support_admission(
        panel.features,
        support,
    )
    return {
        "action": "abstain",
        "bounds_partition": "train",
        "labels_used_to_select_bounds": False,
        "rows": len(panel.labels),
        "admitted_rows": int(np.count_nonzero(admitted)),
        "abstained_rows": int(np.count_nonzero(~admitted)),
        "maximum_outside_training_range_observed": int(
            np.max(outside, initial=0)
        ),
        "maximum_extreme_outliers_observed": int(
            np.max(extreme, initial=0)
        ),
        "changes_predictive_metrics": False,
        "paper_authority": False,
        "live_authority": False,
    }


def evaluate_round16_panel(
    panel: Round16ModelPanel,
    pretest: Mapping[str, object],
    contract: Round16HistoricalContract,
) -> Mapping[str, object]:
    """Score one already-authorized held-out panel without mutating storage."""

    panel.validate(expected_roles=("test",))
    pretest_body = dict(pretest)
    pretest_claimed = str(pretest_body.pop("artifact_sha256", ""))
    if (
        len(pretest_claimed) != 64
        or _canonical_sha256(pretest_body) != pretest_claimed
        or pretest.get("schema_version") != ROUND16_PRETEST_SCHEMA_VERSION
        or pretest.get("contract_sha256") != contract.contract_sha256
        or pretest.get("dataset_sha256") != panel.dataset_sha256
        or pretest.get("test_targets_accessed") is not False
    ):
        raise ValueError("Round 16 pretest binding differs")
    candidates_value = pretest.get("candidates")
    if not isinstance(candidates_value, list) or len(candidates_value) != 4:
        raise ValueError("Round 16 pretest candidates differ")
    candidates = tuple(
        dict(candidate)
        for candidate in candidates_value
        if isinstance(candidate, Mapping)
    )
    if len(candidates) != 4:
        raise ValueError("Round 16 candidate payload is malformed")
    metrics: dict[str, Mapping[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    for candidate in candidates:
        _verify_candidate(candidate)
        candidate_id = str(candidate["candidate_id"])
        prediction = predict_round16_candidate(candidate, panel.features)
        predictions[candidate_id] = prediction
        metrics[candidate_id] = condition_balanced_binary_metrics(
            panel,
            prediction,
        )
    control_id = str(pretest.get("selected_best_control"))
    challenger_id = str(pretest.get("selected_best_challenger"))
    if control_id not in metrics or challenger_id not in metrics:
        raise ValueError("Round 16 selected candidate identity differs")
    control = metrics[control_id]
    challenger = metrics[challenger_id]
    gates_contract = contract.historical.test_gates
    condition_delta, condition_utc_days = _condition_loss_delta(
        panel,
        predictions[control_id],
        predictions[challenger_id],
    )
    expected_test_days = tuple(
        day
        for day in contract.historical.eligible_days
        if contract.historical.roles[day] == "test"
    )
    daily_delta, conditions_by_day = _daily_loss_delta(
        condition_delta,
        condition_utc_days,
        expected_test_days,
    )
    complete_test_days = tuple(
        day
        for day in expected_test_days
        if conditions_by_day[day] == ROUND16_MARKETS_PER_DAY
    )
    unexpected_test_days = tuple(
        day for day in conditions_by_day if day not in expected_test_days
    )
    bootstrap = _paired_utc_day_bootstrap(
        daily_delta,
        repetitions=gates_contract.bootstrap_repetitions,
    )
    unique_conditions = _ordered_conditions(panel.condition_ids)
    condition_labels = np.asarray(
        [
            panel.labels[np.flatnonzero(panel.condition_ids == condition)[0]]
            for condition in unique_conditions
        ],
        dtype=np.float64,
    )
    gates = {
        "minimum_terminal_conditions": (
            len(unique_conditions) >= gates_contract.minimum_terminal_conditions
        ),
        "complete_utc_test_days": (
            len(complete_test_days) == ROUND16_TEST_DAYS
            and len(expected_test_days) == ROUND16_TEST_DAYS
            and not unexpected_test_days
        ),
        "minimum_outcomes_per_class": min(
            np.count_nonzero(condition_labels == 0.0),
            np.count_nonzero(condition_labels == 1.0),
        )
        >= gates_contract.minimum_outcomes_per_class,
        "minimum_decision_rows": (
            len(panel.labels) >= gates_contract.minimum_decision_rows
        ),
        "challenger_log_loss_skill_positive": (
            float(challenger["log_loss"]) < float(control["log_loss"])
        ),
        "challenger_brier_skill_positive": (
            float(challenger["brier_score"]) < float(control["brier_score"])
        ),
        "challenger_balanced_accuracy_not_lower": (
            float(challenger["balanced_accuracy"])
            >= float(control["balanced_accuracy"])
        ),
        "paired_log_loss_improvement_lower_positive": (
            float(bootstrap["lower_95"]) > 0.0
        ),
        "calibration_slope_in_range": (
            gates_contract.calibration_slope_minimum
            <= float(challenger["calibration_slope"])
            <= gates_contract.calibration_slope_maximum
        ),
        "expected_calibration_error_at_most_contract_maximum": (
            float(challenger["expected_calibration_error"])
            <= gates_contract.expected_calibration_error_maximum
        ),
    }
    accepted = all(gates.values())
    return {
        "schema_version": ROUND16_EVALUATION_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256,
        "dataset_sha256": panel.dataset_sha256,
        "scope": {
            "venue": "polymarket",
            "asset": "BTC",
            "market_variant": "fifteenminute",
            "predictive_screen_only": True,
            "execution_or_profitability_claim": False,
        },
        "test": {
            "conditions": len(unique_conditions),
            "decision_rows": len(panel.labels),
            "up_conditions": int(np.count_nonzero(condition_labels == 1.0)),
            "down_conditions": int(np.count_nonzero(condition_labels == 0.0)),
            "first_event_start_ms": int(np.min(panel.event_start_ms)),
            "last_event_start_ms": int(np.max(panel.event_start_ms)),
            "expected_utc_days": list(expected_test_days),
            "complete_utc_days": list(complete_test_days),
            "unexpected_utc_days": list(unexpected_test_days),
            "conditions_by_utc_day": conditions_by_day,
        },
        "best_control_id": control_id,
        "best_challenger_id": challenger_id,
        "candidate_metrics": metrics,
        "challenger_skill": {
            "log_loss": 1.0
            - float(challenger["log_loss"]) / float(control["log_loss"]),
            "brier": 1.0
            - float(challenger["brier_score"]) / float(control["brier_score"]),
        },
        "paired_utc_day_bootstrap": bootstrap,
        "settlement_manipulation_screen": _settlement_screen_report(
            panel,
            pretest,
        ),
        "feature_support_screen": _feature_support_report(panel, pretest),
        "gates": gates,
        "accepted_predictive_edge": accepted,
        "failure_action": (
            "continue_to_separate_prospective_after_cost_shadow"
            if accepted
            else "reject_candidate_without_trading_or_profitability_claim"
        ),
        "paper_authority": False,
        "live_authority": False,
        "profitability_claim": False,
    }


def evaluate_round16_test_once(
    store: HistoricalScreenStore,
    contract: Round16HistoricalContract,
) -> tuple[Mapping[str, object], str]:
    if store.contract != contract.historical or store.state != "targets_complete":
        raise ValueError("Round 16 evaluation requires one-use test labels")
    pretest, pretest_envelope_sha = store.pretest_artifact()
    panel = load_round16_model_panel(store, contract, roles=("test",))
    body = dict(evaluate_round16_panel(panel, pretest, contract))
    body["pretest_artifact_sha256"] = pretest_envelope_sha
    artifact = {**body, "artifact_sha256": _canonical_sha256(body)}
    canonical = _canonical_json(artifact)
    envelope_sha = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    now = time.time_ns() // 1_000_000
    connection = store.connect()
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(
            """
            INSERT INTO target.evaluation_manifest VALUES (
                true, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                ROUND16_EVALUATION_SCHEMA_VERSION,
                contract.contract_sha256,
                panel.dataset_sha256,
                pretest_envelope_sha,
                canonical,
                envelope_sha,
                now,
            ],
        )
        changed = connection.execute(
            """
            UPDATE feature.screen_manifest
            SET state = 'evaluated', updated_at_ms = ?
            WHERE singleton AND state = 'targets_complete'
            RETURNING state
            """,
            [now],
        ).fetchone()
        if changed is None:
            raise ValueError("Round 16 evaluation state changed concurrently")
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    return artifact, envelope_sha


__all__ = [
    "ROUND16_BOOTSTRAP_SEED",
    "ROUND16_EVALUATION_SCHEMA_VERSION",
    "evaluate_round16_panel",
    "evaluate_round16_test_once",
]
