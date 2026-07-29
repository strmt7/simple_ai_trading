"""One-use held-out predictive evaluation for Polymarket Round 16."""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Mapping

import numpy as np

from .polymarket_historical_model import condition_balanced_binary_metrics
from .polymarket_historical_screen import HistoricalScreenStore
from .polymarket_round16 import Round16HistoricalContract
from .polymarket_round16_model import (
    ROUND16_PRETEST_SCHEMA_VERSION,
    Round16ModelPanel,
    load_round16_model_panel,
    predict_round16_candidate,
)


ROUND16_EVALUATION_SCHEMA_VERSION = "polymarket-round16-btc-15m-evaluation-v1"
ROUND16_BOOTSTRAP_BLOCK_CONDITIONS = 12
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
) -> np.ndarray:
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
    return np.asarray(
        [
            float(
                np.mean(
                    (control_loss - challenger_loss)[
                        panel.condition_ids == condition
                    ]
                )
            )
            for condition in _ordered_conditions(panel.condition_ids)
        ],
        dtype=np.float64,
    )


def _paired_block_bootstrap(
    delta: np.ndarray,
    *,
    repetitions: int,
    minimum_conditions: int,
) -> Mapping[str, float | int]:
    values = np.asarray(delta, dtype=np.float64)
    selected_repetitions = int(repetitions)
    required_conditions = int(minimum_conditions)
    if (
        values.ndim != 1
        or len(values) < required_conditions
        or not np.all(np.isfinite(values))
        or not 100 <= selected_repetitions <= 100_000
        or required_conditions < 2
    ):
        raise ValueError("Round 16 paired bootstrap inputs are invalid")
    generator = np.random.default_rng(ROUND16_BOOTSTRAP_SEED)
    sample_count = math.ceil(len(values) / ROUND16_BOOTSTRAP_BLOCK_CONDITIONS)
    offsets = np.arange(ROUND16_BOOTSTRAP_BLOCK_CONDITIONS, dtype=np.int64)
    estimates = np.empty(selected_repetitions, dtype=np.float64)
    batch_size = 256
    for first in range(0, selected_repetitions, batch_size):
        count = min(batch_size, selected_repetitions - first)
        starts = generator.integers(
            0,
            len(values),
            size=(count, sample_count),
        )
        indexes = (
            starts[:, :, None] + offsets[None, None, :]
        ) % len(values)
        sampled = values[indexes.reshape(count, -1)[:, : len(values)]]
        estimates[first : first + count] = np.mean(sampled, axis=1)
    return {
        "repetitions": selected_repetitions,
        "block_conditions": ROUND16_BOOTSTRAP_BLOCK_CONDITIONS,
        "lower_95": float(np.quantile(estimates, 0.025)),
        "median": float(np.quantile(estimates, 0.5)),
        "upper_95": float(np.quantile(estimates, 0.975)),
    }


def _verify_candidate(candidate: Mapping[str, object]) -> None:
    body = dict(candidate)
    claimed = str(body.pop("artifact_sha256", ""))
    if len(claimed) != 64 or _canonical_sha256(body) != claimed:
        raise ValueError("Round 16 candidate artifact integrity failed")


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
    bootstrap = _paired_block_bootstrap(
        _condition_loss_delta(
            panel,
            predictions[control_id],
            predictions[challenger_id],
        ),
        repetitions=gates_contract.bootstrap_repetitions,
        minimum_conditions=gates_contract.minimum_terminal_conditions,
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
        "paired_condition_block_bootstrap": bootstrap,
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
    "ROUND16_BOOTSTRAP_BLOCK_CONDITIONS",
    "ROUND16_BOOTSTRAP_SEED",
    "ROUND16_EVALUATION_SCHEMA_VERSION",
    "evaluate_round16_panel",
    "evaluate_round16_test_once",
]
