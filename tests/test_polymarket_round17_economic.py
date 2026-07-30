from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from simple_ai_trading.polymarket_round14_contract import load_round14_contract
from simple_ai_trading.polymarket_round17_economic import (
    POLYMARKET_ROUND17_ECONOMIC_CONTRACT_SHA256,
    POLYMARKET_ROUND17_ECONOMIC_PATHS,
    POLYMARKET_ROUND17_ECONOMIC_THRESHOLDS,
    build_round17_condition_economic_outcome,
    fit_round17_economic_pretest,
    validate_round17_economic_pretest,
)
from simple_ai_trading.polymarket_round17_features import (
    POLYMARKET_ROUND17_FEATURE_NAMES,
)
from simple_ai_trading.polymarket_round17_model import (
    Round17DevelopmentPanel,
    fit_round17_development_pretest,
)
from simple_ai_trading.polymarket_round17_uncertainty import (
    fit_round17_probability_calibration,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-014-btc-5m-prospective-contract-v1.json"
)
START_MS = 1_800_057_600_000
PARENT_START_MS = START_MS - 72_000_000
DATASET_SHA256 = "d" * 64
TARGET_MANIFEST_SHA256 = "e" * 64
ECONOMIC_CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-017-btc-5m-economic-pretest-v1.json"
)


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _panel(
    role: str,
    *,
    first_event_start_ms: int,
    condition_count: int,
) -> Round17DevelopmentPanel:
    condition_ids: list[str] = []
    event_starts: list[int] = []
    decisions: list[int] = []
    labels: list[float] = []
    features: list[np.ndarray] = []
    signal_index = POLYMARKET_ROUND17_FEATURE_NAMES.index("chainlink_log_return_1000ms")
    structural_index = POLYMARKET_ROUND17_FEATURE_NAMES.index(
        "structural_probability_up"
    )
    prior_index = POLYMARKET_ROUND17_FEATURE_NAMES.index("normalized_market_prior_up")
    for condition_index in range(condition_count):
        event_start = first_event_start_ms + condition_index * 300_000
        label = float(condition_index % 2)
        condition = "0x" + _sha256([role, condition_index])
        for row_index, offset in enumerate((60_000, 120_000, 180_000)):
            row = np.zeros(len(POLYMARKET_ROUND17_FEATURE_NAMES), dtype=np.float64)
            row[structural_index] = 0.5
            row[prior_index] = 0.85 if label else 0.15
            row[signal_index] = (2.0 if label else -2.0) + row_index * 0.01
            condition_ids.append(condition)
            event_starts.append(event_start)
            decisions.append(event_start + offset)
            labels.append(label)
            features.append(row)
    return Round17DevelopmentPanel(
        role=role,
        condition_ids=np.asarray(condition_ids, dtype=object),
        event_start_ms=np.asarray(event_starts, dtype=np.int64),
        decision_time_ms=np.asarray(decisions, dtype=np.int64),
        features=np.asarray(features, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.float64),
        dataset_sha256=DATASET_SHA256,
        target_manifest_sha256=TARGET_MANIFEST_SHA256,
    ).validate()


@pytest.fixture(scope="module")
def round17_parents() -> tuple[dict[str, object], dict[str, object]]:
    train = _panel(
        "train",
        first_event_start_ms=PARENT_START_MS,
        condition_count=24,
    )
    train_end = int(np.max(train.event_start_ms)) + 300_000
    calibration = _panel(
        "tune_calibration",
        first_event_start_ms=train_end + 3_600_000,
        condition_count=10,
    )
    calibration_end = int(np.max(calibration.event_start_ms)) + 300_000
    selection = _panel(
        "tune_selection",
        first_event_start_ms=calibration_end + 3_600_000,
        condition_count=10,
    )
    pretest = fit_round17_development_pretest(
        train,
        calibration,
        selection,
        compute_backend="cpu",
    )
    selection_end = int(np.max(selection.event_start_ms)) + 300_000
    uncertainty = _panel(
        "tune_uncertainty",
        first_event_start_ms=selection_end + 3_600_000,
        condition_count=100,
    )
    return pretest, fit_round17_probability_calibration(uncertainty, pretest)


def test_round17_economic_contract_is_frozen_before_campaign_access() -> None:
    contract = json.loads(ECONOMIC_CONTRACT_PATH.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")

    assert _sha256(contract) == claimed
    assert claimed == POLYMARKET_ROUND17_ECONOMIC_CONTRACT_SHA256
    assert contract["status"] == (
        "preregistered_before_active_campaign_outcome_or_model_score_access"
    )
    assert contract["parent"]["active_campaign_outcomes_consulted"] is False
    assert contract["parent"]["active_campaign_model_scores_consulted"] is False
    assert contract["parent"]["active_campaign_execution_scores_consulted"] is False
    assert contract["parent"]["round17_cohort_plan_sha256"] == (
        "37fede4da0d6c504bce7cb763b9bd49032e0252a8cede045f29f05acff67fc00"
    )
    assert contract["partition_protocol"] == {
        "chronology": [
            "train",
            "tune_calibration",
            "tune_selection",
            "tune_uncertainty",
            "tune_economic",
            "test",
        ],
        "embargo_ms_between_adjacent_roles": 3_600_000,
        "event_duration_ms": 300_000,
        "condition_ids_disjoint_across_roles": True,
        "model_selection_frozen_before_tune_uncertainty": True,
        "probability_calibration_frozen_before_tune_economic": True,
        "economic_policy_frozen_before_test": True,
        "test_may_be_accessed_once": True,
        "failed_test_may_not_return_to_development": True,
        "development_manifests_may_not_contain_test_features_or_targets": True,
    }
    assert contract["probability_uncertainty"] == {
        "source_partition": "tune_uncertainty",
        "method": (
            "condition-block bootstrap of condition-equal-weighted "
            "50-bin isotonic calibration"
        ),
        "bin_count": 50,
        "bootstrap_unit": "condition",
        "bootstrap_samples": 500,
        "bootstrap_seed": 117_017,
        "lower_quantile": "0.05",
        "upper_quantile": "0.95",
        "minimum_total_conditions": 100,
        "minimum_conditions_per_interpolated_bin": 30,
        "unsupported_probability_region_action": "abstain",
        "caller_supplied_probability_prohibited": True,
        "selected_model_artifact_required": True,
        "development_target_manifest_required": True,
        "feature_row_hash_binding_required": True,
    }
    assert [item["name"] for item in contract["action_paths"]] == list(
        POLYMARKET_ROUND17_ECONOMIC_PATHS
    )
    assert contract["minimum_edge_grid_quote_per_share"] == [
        format(value, "f") for value in POLYMARKET_ROUND17_ECONOMIC_THRESHOLDS
    ]
    assert contract["execution_lifecycle"]["future_book_selection_prohibited"] is True
    assert (
        contract["execution_lifecycle"][
            "midpoint_or_hidden_liquidity_credit_prohibited"
        ]
        is True
    )
    assert (
        contract["execution_lifecycle"]["retry_interval_after_known_no_fill_ms"] == 1000
    )
    assert contract["authority"] == {
        "test_features_accessed": False,
        "test_targets_accessed": False,
        "test_execution_accessed": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }


@lru_cache(maxsize=2)
def _outcomes(*, negative: bool = False):
    program = load_round14_contract(CONTRACT_PATH)
    values = []
    for condition_index in range(110):
        condition_id = "0x" + _sha256(["condition", condition_index])
        event_start = START_MS + condition_index * 2_700_000
        for path in POLYMARKET_ROUND17_ECONOMIC_PATHS:
            for profile in program.risk_profiles:
                maximum_loss = (
                    Decimal("1000") * profile.maximum_event_loss_capital_fraction
                )
                for threshold in POLYMARKET_ROUND17_ECONOMIC_THRESHOLDS:
                    for scenario in program.scenarios:
                        pnl = (
                            Decimal("-0.02")
                            if negative
                            else Decimal("0.01") + threshold / Decimal("100")
                        )
                        values.append(
                            build_round17_condition_economic_outcome(
                                condition_id=condition_id,
                                event_start_ms=event_start,
                                path=path,
                                risk_profile=profile.name,
                                scenario=scenario.name,
                                minimum_edge_quote_per_share=threshold,
                                risk_capital_quote=Decimal("1000"),
                                entry_executed=True,
                                realized_net_quote=pnl,
                                maximum_loss_quote=maximum_loss,
                                unknown_state=False,
                                lifecycle_violation=False,
                                ownership_violation=False,
                                decision_sha256=_sha256(
                                    [
                                        condition_id,
                                        path,
                                        profile.name,
                                        str(threshold),
                                        scenario.name,
                                    ]
                                ),
                                source_evidence_sha256=_sha256(
                                    ["execution", condition_id, scenario.name]
                                ),
                            )
                        )
    return program, values


def test_round17_economic_pretest_is_complete_test_blind_and_hash_bound(
    round17_parents: tuple[dict[str, object], dict[str, object]],
) -> None:
    program, outcomes = _outcomes()
    pretest, calibration = round17_parents

    artifact = fit_round17_economic_pretest(
        outcomes,
        program,
        model_pretest=pretest,
        probability_calibration=calibration,
        dataset_sha256=DATASET_SHA256,
    )
    verified = validate_round17_economic_pretest(
        artifact,
        model_pretest=pretest,
        probability_calibration=calibration,
    )

    assert len(verified["candidate_ledger"]) == 45
    assert set(verified["selected_by_profile"]) == {
        "conservative",
        "regular",
        "aggressive",
    }
    assert verified["development_accepted"] is True
    assert verified["test_features_accessed"] is False
    assert verified["test_targets_accessed"] is False
    assert verified["test_execution_accessed"] is False
    assert verified["profitability_claim"] is False
    assert verified["paper_trading_authority"] is False
    assert verified["live_trading_authority"] is False
    assert verified["binance_credentials_used"] is False
    assert verified["binance_execution_connected"] is False
    assert verified["target_manifest_sha256"] == TARGET_MANIFEST_SHA256


def test_round17_economic_pretest_rejects_unprofitable_development_paths(
    round17_parents: tuple[dict[str, object], dict[str, object]],
) -> None:
    program, outcomes = _outcomes(negative=True)
    pretest, calibration = round17_parents

    artifact = fit_round17_economic_pretest(
        outcomes,
        program,
        model_pretest=pretest,
        probability_calibration=calibration,
        dataset_sha256=DATASET_SHA256,
    )

    assert artifact["development_accepted"] is False
    assert all(
        item["development_accepted"] is False
        for item in artifact["selected_by_profile"].values()
    )


def test_round17_economic_pretest_rejects_incomplete_grid_and_rehashed_drift(
    round17_parents: tuple[dict[str, object], dict[str, object]],
) -> None:
    program, outcomes = _outcomes()
    pretest, calibration = round17_parents

    omitted_key = (
        POLYMARKET_ROUND17_ECONOMIC_PATHS[0],
        "conservative",
        POLYMARKET_ROUND17_ECONOMIC_THRESHOLDS[0],
        "primary",
    )
    incomplete = tuple(
        item
        for item in outcomes
        if (
            item.path,
            item.risk_profile,
            item.minimum_edge_quote_per_share,
            item.scenario,
        )
        != omitted_key
    )
    with pytest.raises(ValueError, match="grid is incomplete"):
        fit_round17_economic_pretest(
            incomplete,
            program,
            model_pretest=pretest,
            probability_calibration=calibration,
            dataset_sha256=DATASET_SHA256,
        )

    artifact = fit_round17_economic_pretest(
        outcomes,
        program,
        model_pretest=pretest,
        probability_calibration=calibration,
        dataset_sha256=DATASET_SHA256,
    )
    artifact["profitability_claim"] = True
    artifact.pop("economic_pretest_sha256")
    artifact["economic_pretest_sha256"] = _sha256(artifact)
    with pytest.raises(ValueError, match="integrity differs"):
        validate_round17_economic_pretest(artifact)


def test_round17_condition_outcome_cannot_exceed_its_declared_loss_bound() -> None:
    with pytest.raises(ValueError, match="outcome is invalid"):
        build_round17_condition_economic_outcome(
            condition_id="0x" + "1" * 64,
            event_start_ms=START_MS,
            path="settlement_directional",
            risk_profile="conservative",
            scenario="primary",
            minimum_edge_quote_per_share=Decimal("0.02"),
            risk_capital_quote=Decimal("1000"),
            entry_executed=True,
            realized_net_quote=Decimal("-1.01"),
            maximum_loss_quote=Decimal("1"),
            unknown_state=False,
            lifecycle_violation=False,
            ownership_violation=False,
            decision_sha256="b" * 64,
            source_evidence_sha256="c" * 64,
        )
