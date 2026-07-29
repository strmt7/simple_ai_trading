from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from simple_ai_trading.polymarket_round16 import (
    load_round16_historical_contract,
)
from simple_ai_trading.polymarket_round16_dataset import ROUND16_FEATURE_NAMES
from simple_ai_trading.polymarket_round16_evaluation import evaluate_round16_panel
from simple_ai_trading.polymarket_historical_screen import HistoricalScreenStore
from simple_ai_trading.polymarket_round16_model import (
    ROUND16_SETTLEMENT_DISAGREEMENT_FEATURE,
    ROUND16_SETTLEMENT_QUOTE_FEATURE,
    Round16ModelPanel,
    build_round16_pretest_artifact,
    fit_round16_pretest_candidates,
    freeze_round16_feature_support,
    freeze_round16_settlement_controls,
    predict_round16_candidate,
    record_round16_pretest_artifact,
    round16_feature_support_admission,
    round16_settlement_admission_mask,
)


ROOT = Path(__file__).parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-016-btc-15m-horizon-comparison-v1.json"
)


def _panel(*, role: str, condition_count: int, seed: int) -> Round16ModelPanel:
    rng = np.random.default_rng(seed)
    condition_signal = rng.normal(size=condition_count)
    labels_by_condition = (condition_signal > 0).astype(np.float64)
    condition_ids = np.repeat(
        np.asarray(
            [f"0x{index:064x}" for index in range(1, condition_count + 1)],
            dtype=object,
        ),
        14,
    )
    labels = np.repeat(labels_by_condition, 14)
    features = rng.normal(
        scale=0.2,
        size=(condition_count * 14, len(ROUND16_FEATURE_NAMES)),
    ).astype(np.float32)
    features[:, 0] += np.repeat(condition_signal, 14).astype(np.float32)
    offsets = np.tile(np.arange(1, 15, dtype=np.int64), condition_count)
    event_start = np.repeat(
        np.arange(condition_count, dtype=np.int64) * 900_000 + 1_700_000_000_000,
        14,
    )
    return Round16ModelPanel(
        condition_ids=condition_ids,
        roles=np.full(condition_count * 14, role, dtype=object),
        event_start_ms=event_start,
        decision_time_ms=event_start + offsets * 60_000,
        features=features,
        labels=labels,
        dataset_sha256="a" * 64,
    )


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def test_round16_candidate_screen_is_bounded_and_reproducible() -> None:
    train = _panel(role="train", condition_count=80, seed=16015)
    tune = _panel(role="tune", condition_count=40, seed=16016)

    candidates = fit_round16_pretest_candidates(
        train,
        tune,
        compute_backend="cpu",
    )

    assert len(candidates) == 4
    assert {candidate["kind"] for candidate in candidates} == {
        "control",
        "challenger",
    }
    assert {candidate["family"] for candidate in candidates} == {
        "training_prevalence",
        "calendar_ridge_logistic",
        "binance_ridge_logistic",
        "binance_shallow_lightgbm",
    }
    for candidate in candidates:
        prediction = predict_round16_candidate(candidate, tune.features)
        assert prediction.shape == tune.labels.shape
        assert np.all((prediction > 0) & (prediction < 1))
        body = dict(candidate)
        claimed = body.pop("artifact_sha256")
        assert _canonical_sha256(body) == claimed


def test_round16_pretest_artifact_has_no_test_or_trading_authority(
    tmp_path: Path,
) -> None:
    contract = load_round16_historical_contract(CONTRACT_PATH)
    train = _panel(role="train", condition_count=80, seed=16015)
    tune = _panel(role="tune", condition_count=40, seed=16016)
    candidates = fit_round16_pretest_candidates(
        train,
        tune,
        compute_backend="cpu",
    )

    artifact = build_round16_pretest_artifact(
        train,
        tune,
        candidates,
        contract=contract,
        source_commit="b" * 40,
    )

    assert artifact["test_targets_accessed"] is False
    assert artifact["paper_authority"] is False
    assert artifact["live_authority"] is False
    assert artifact["profitability_claim"] is False
    controls = artifact["settlement_manipulation_controls"]
    assert controls["partition"] == "tune"
    assert controls["labels_used"] is False
    assert controls["abnormal_action"] == "abstain"
    assert controls["trading_authority"] is False
    support = artifact["feature_support"]
    assert support["partition"] == "train"
    assert support["labels_used"] is False
    assert support["gate"]["action"] == "abstain"
    assert support["trading_authority"] is False
    body = dict(artifact)
    claimed = body.pop("artifact_sha256")
    assert _canonical_sha256(body) == claimed

    tampered = [dict(candidate) for candidate in candidates]
    tampered[0]["raw_tune_log_loss"] = 0.0
    with pytest.raises(ValueError, match="artifact integrity"):
        build_round16_pretest_artifact(
            train,
            tune,
            tampered,
            contract=contract,
            source_commit="b" * 40,
        )

    with HistoricalScreenStore(
        tmp_path / "round16-model.duckdb",
        contract=contract.historical,
    ) as store:
        store.transition("initialized", "identities_complete")
        store.transition("identities_complete", "features_complete")
        store.transition("features_complete", "development_targets_complete")
        store.connect().execute(
            """
            CREATE TABLE feature.round16_dataset_manifest (
                singleton BOOLEAN PRIMARY KEY,
                manifest_json VARCHAR NOT NULL,
                dataset_sha256 VARCHAR NOT NULL,
                created_at_ms BIGINT NOT NULL
            )
            """
        )
        store.connect().execute(
            "INSERT INTO feature.round16_dataset_manifest VALUES (true, '{}', ?, 1)",
            [train.dataset_sha256],
        )
        tampered_artifact = dict(artifact)
        tampered_artifact["selected_best_control"] = "tampered"
        with pytest.raises(ValueError, match="integrity differs"):
            record_round16_pretest_artifact(
                store,
                contract,
                tampered_artifact,
            )

        envelope_sha = record_round16_pretest_artifact(
            store,
            contract,
            artifact,
        )

        assert len(envelope_sha) == 64
        assert store.state == "pretest_complete"
        stored, stored_sha = store.pretest_artifact()
    assert stored["artifact_sha256"] == artifact["artifact_sha256"]
    assert stored_sha == envelope_sha

    test = _panel(role="test", condition_count=1_200, seed=16017)
    evaluation = evaluate_round16_panel(test, artifact, contract)

    assert evaluation["test"]["conditions"] == 1_200
    assert evaluation["test"]["decision_rows"] == 16_800
    assert evaluation["scope"]["predictive_screen_only"] is True
    assert evaluation["scope"]["execution_or_profitability_claim"] is False
    assert evaluation["paper_authority"] is False
    assert evaluation["live_authority"] is False
    assert evaluation["profitability_claim"] is False
    settlement = evaluation["settlement_manipulation_screen"]
    assert settlement["threshold_partition"] == "tune"
    assert settlement["changes_predictive_metrics"] is False
    assert settlement["all_decisions"]["rows"] == 16_800
    assert settlement["last_180_seconds"]["rows"] == 3_600
    assert settlement["last_120_seconds"]["rows"] == 2_400
    assert settlement["last_60_seconds"]["rows"] == 1_200
    assert (
        settlement["all_decisions"]["admitted_rows"]
        + settlement["all_decisions"]["abstained_rows"]
        == 16_800
    )
    support_screen = evaluation["feature_support_screen"]
    assert support_screen["bounds_partition"] == "train"
    assert support_screen["changes_predictive_metrics"] is False
    assert (
        support_screen["admitted_rows"]
        + support_screen["abstained_rows"]
        == 16_800
    )
    assert set(evaluation["gates"]) == {
        "minimum_terminal_conditions",
        "minimum_outcomes_per_class",
        "minimum_decision_rows",
        "challenger_log_loss_skill_positive",
        "challenger_brier_skill_positive",
        "challenger_balanced_accuracy_not_lower",
        "paired_log_loss_improvement_lower_positive",
        "calibration_slope_in_range",
        "expected_calibration_error_at_most_contract_maximum",
    }


def test_round16_panel_requires_fourteen_decisions_per_condition() -> None:
    panel = _panel(role="train", condition_count=4, seed=16015)
    invalid = Round16ModelPanel(
        condition_ids=panel.condition_ids[:-1],
        roles=panel.roles[:-1],
        event_start_ms=panel.event_start_ms[:-1],
        decision_time_ms=panel.decision_time_ms[:-1],
        features=panel.features[:-1],
        labels=panel.labels[:-1],
        dataset_sha256=panel.dataset_sha256,
    )

    with pytest.raises(ValueError, match="decision coverage"):
        invalid.validate(expected_roles=("train",))


def test_round16_settlement_screen_is_label_blind_and_abstains_on_anomalies() -> None:
    tune = _panel(role="tune", condition_count=40, seed=16016)
    controls = freeze_round16_settlement_controls(tune)
    matrix = np.zeros((2, len(ROUND16_FEATURE_NAMES)), dtype=np.float32)
    quote_index = ROUND16_FEATURE_NAMES.index(ROUND16_SETTLEMENT_QUOTE_FEATURE)
    disagreement_index = ROUND16_FEATURE_NAMES.index(
        ROUND16_SETTLEMENT_DISAGREEMENT_FEATURE
    )
    matrix[1, quote_index] = (
        float(controls["quote_upper_threshold"]) + 1.0
    )
    matrix[1, disagreement_index] = (
        float(controls["disagreement_absolute_threshold"]) + 1.0
    )

    admitted = round16_settlement_admission_mask(matrix, controls)

    assert admitted.tolist() == [True, False]
    assert controls["labels_used"] is False
    with pytest.raises(ValueError, match="identity differs"):
        round16_settlement_admission_mask(
            matrix,
            {**controls, "partition": "test"},
        )


def test_round16_feature_support_is_train_only_and_fails_closed() -> None:
    train = _panel(role="train", condition_count=80, seed=16015)
    support = freeze_round16_feature_support(train)
    inside = np.asarray(train.features[:1], dtype=np.float32)
    outside = inside.copy()
    outside[0, :5] = 1e20

    admitted, outside_count, extreme_count = (
        round16_feature_support_admission(
            np.concatenate((inside, outside), axis=0),
            support,
        )
    )

    assert admitted.tolist() == [True, False]
    assert outside_count.tolist() == [0, 5]
    assert extreme_count.tolist() == [0, 5]
    assert support["labels_used"] is False
    with pytest.raises(ValueError, match="identity differs"):
        round16_feature_support_admission(
            inside,
            {**support, "trading_authority": True},
        )
