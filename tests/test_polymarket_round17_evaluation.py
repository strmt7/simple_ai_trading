from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from simple_ai_trading import polymarket_round17_economic as economic
from simple_ai_trading import polymarket_round17_evaluation as evaluation
from simple_ai_trading.polymarket_round14_contract import load_round14_contract
from simple_ai_trading.polymarket_round17_economic import (
    build_round17_condition_economic_outcome,
    evaluate_round17_economic_holdout,
)
from simple_ai_trading.polymarket_round17_evaluation import (
    POLYMARKET_ROUND17_ENDPOINT_CONTROL_IDS,
    Round17OneUseEvaluationConfig,
    build_round17_one_use_result,
    evaluate_round17_endpoint_holdout,
    run_round17_one_use_evaluation,
)
from simple_ai_trading.polymarket_round17_campaign_operator import (
    Round17CampaignOperatorConfig,
)
from simple_ai_trading.polymarket_round17_features import (
    POLYMARKET_ROUND17_FEATURE_NAMES,
)
from simple_ai_trading.polymarket_round17_model import Round17DevelopmentPanel
from simple_ai_trading.polymarket_round17_one_use import (
    POLYMARKET_ROUND17_TEST_START_MS,
    Round17OneUseClaimStore,
    Round17TestAccessClaim,
)


ROOT = Path(__file__).resolve().parents[1]
RISK_CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-014-btc-5m-prospective-contract-v1.json"
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


def _claim() -> Round17TestAccessClaim:
    provisional = Round17TestAccessClaim(
        development_result_sha256="1" * 64,
        campaign_readiness_sha256="2" * 64,
        campaign_development_index_sha256="3" * 64,
        cohort_manifest_sha256="4" * 64,
        target_manifest_sha256="5" * 64,
        model_pretest_sha256="6" * 64,
        probability_calibration_sha256="7" * 64,
        economic_pretest_sha256="8" * 64,
        implementation_manifest_sha256="9" * 64,
        repository_commit_sha="a" * 40,
        opened_at_ms=1_800_000_000_000,
    )
    return replace(
        provisional,
        claim_sha256=_sha256(provisional.identity_payload()),
    ).validated()


def _test_panel(condition_count: int = 1_800) -> Round17DevelopmentPanel:
    event_starts = (
        POLYMARKET_ROUND17_TEST_START_MS
        + np.arange(condition_count, dtype=np.int64) * 300_000
    )
    labels = np.arange(condition_count, dtype=np.float64) % 2
    features = np.zeros(
        (condition_count, len(POLYMARKET_ROUND17_FEATURE_NAMES)),
        dtype=np.float32,
    )
    features[
        :,
        POLYMARKET_ROUND17_FEATURE_NAMES.index("structural_probability_up"),
    ] = 0.5
    features[
        :,
        POLYMARKET_ROUND17_FEATURE_NAMES.index("normalized_market_prior_up"),
    ] = 0.5
    return Round17DevelopmentPanel(
        role="test",
        condition_ids=np.asarray(
            [f"0x{index:064x}" for index in range(condition_count)],
            dtype=object,
        ),
        event_start_ms=event_starts,
        decision_time_ms=event_starts + 60_000,
        features=features,
        labels=labels,
        dataset_sha256="b" * 64,
        target_manifest_sha256="c" * 64,
    ).validate()


def _development_result() -> dict[str, object]:
    controls = [
        {
            "candidate_id": control_id,
            "model": {"candidate_id": control_id},
        }
        for control_id in POLYMARKET_ROUND17_ENDPOINT_CONTROL_IDS
    ]
    return {
        "status": "development_accepted",
        "result_sha256": "1" * 64,
        "parents": {
            "model_pretest_sha256": "6" * 64,
            "probability_calibration_sha256": "7" * 64,
            "economic_pretest_sha256": "8" * 64,
        },
        "artifacts": {
            "model_pretest": {
                "pretest_sha256": "6" * 64,
                "selected_candidate_id": "selected",
                "selected_candidate": {"candidate_id": "selected"},
                "strongest_control_id": POLYMARKET_ROUND17_ENDPOINT_CONTROL_IDS[0],
                "controls": controls,
                "dataset_and_partition": {
                    "roles": {
                        "train": {"condition_ids": ["0x" + "f" * 64]},
                    }
                },
            }
        },
    }


def test_round17_endpoint_holdout_applies_every_frozen_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _test_panel()
    parent = _development_result()
    access = "d" * 64
    monkeypatch.setattr(
        evaluation,
        "validate_round17_development_result",
        lambda value: value,
    )

    def predict(candidate, selected_panel):
        labels = selected_panel.labels
        confidence = {
            "selected": 0.9,
            POLYMARKET_ROUND17_ENDPOINT_CONTROL_IDS[0]: 0.7,
            POLYMARKET_ROUND17_ENDPOINT_CONTROL_IDS[1]: 0.65,
            POLYMARKET_ROUND17_ENDPOINT_CONTROL_IDS[2]: 0.6,
        }[candidate["candidate_id"]]
        return np.where(labels == 1.0, confidence, 1.0 - confidence)

    monkeypatch.setattr(evaluation, "predict_round17_candidate", predict)
    result = evaluate_round17_endpoint_holdout(
        panel,
        development_result=parent,
        claim=_claim(),
        test_access_sha256=access,
    )

    assert result["endpoint_accepted"] is True
    assert result["condition_count"] == 1_800
    assert result["calendar_day_count"] >= 7
    assert result["gates"] == {name: True for name in result["gates"]}
    assert result["automatic_promotion"] is False
    assert result["profitability_claim"] is False


def test_round17_economic_holdout_replays_only_selected_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = {
        profile: {
            "candidate_id": f"selected-{profile}",
            "path": "settlement_directional",
            "minimum_edge_quote_per_share": "0.01",
        }
        for profile in ("conservative", "regular", "aggressive")
    }
    monkeypatch.setattr(
        economic,
        "validate_round17_economic_pretest",
        lambda *_args, **_kwargs: {
            "development_accepted": True,
            "economic_pretest_sha256": "8" * 64,
            "selected_by_profile": selected,
        },
    )
    outcomes = []
    for profile in selected:
        for scenario in (
            "primary",
            "latency_250ms",
            "latency_750ms",
            "latency_1000ms",
            "half_depth",
            "quarter_depth",
            "one_tick_adverse",
            "combined",
        ):
            for index in range(2):
                outcomes.append(
                    build_round17_condition_economic_outcome(
                        condition_id=f"0x{index + 1:064x}",
                        event_start_ms=1_800_057_600_000 + index * 300_000,
                        path="settlement_directional",
                        risk_profile=profile,
                        scenario=scenario,
                        minimum_edge_quote_per_share=Decimal("0.01"),
                        risk_capital_quote=Decimal("10000"),
                        entry_executed=True,
                        realized_net_quote=Decimal("1"),
                        maximum_loss_quote=Decimal("1"),
                        unknown_state=False,
                        lifecycle_violation=False,
                        ownership_violation=False,
                        decision_sha256="a" * 64,
                        source_evidence_sha256="b" * 64,
                        source_partition="test",
                    )
                )
    result = evaluate_round17_economic_holdout(
        outcomes,
        load_round14_contract(RISK_CONTRACT),
        economic_pretest={},
        model_pretest={},
        probability_calibration={},
        test_access_sha256="d" * 64,
        minimum_conditions=2,
        minimum_actions=1,
        minimum_calendar_days=1,
    )

    assert result["economic_accepted"] is True
    assert all(result["profile_gates"].values())
    assert result["policy_refit"] is False
    assert result["test_execution_accessed"] is True


def test_round17_final_result_never_promotes_or_returns_to_development() -> None:
    claim = _claim()
    endpoint: dict[str, object] = {
        "claim_sha256": claim.claim_sha256,
        "test_access_sha256": "d" * 64,
        "test_target_manifest_sha256": "e" * 64,
        "endpoint_accepted": True,
    }
    endpoint["endpoint_holdout_sha256"] = _sha256(endpoint)
    economic_holdout: dict[str, object] = {
        "test_access_sha256": "d" * 64,
        "economic_accepted": False,
    }
    economic_holdout["economic_holdout_sha256"] = _sha256(economic_holdout)

    result = build_round17_one_use_result(
        claim=claim,
        test_access_sha256="d" * 64,
        test_index_sha256="f" * 64,
        test_resolution_acquisition_sha256="1" * 64,
        test_target_manifest_sha256="e" * 64,
        endpoint_holdout=endpoint,
        economic_holdout=economic_holdout,
    )

    assert result["status"] == "heldout_rejected"
    assert result["return_to_development"] is False
    assert result["automatic_promotion"] is False
    assert result["live_trading_authority"] is False


def test_round17_completed_claim_is_republished_without_campaign_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _claim()
    store_path = tmp_path / "one-use.sqlite3"
    with Round17OneUseClaimStore(store_path) as store:
        store.open_claim(claim)
        access = store.consume_test_access(claim)
        persisted: dict[str, object] = {
            "schema_version": "completed-test-v1",
            "status": "heldout_rejected",
            "claim_sha256": claim.claim_sha256,
            "test_access_sha256": access,
            "test_access_consumed": True,
            "automatic_promotion": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
            "binance_credentials_used": False,
            "binance_execution_connected": False,
        }
        persisted["result_sha256"] = _sha256(persisted)
        store.complete(claim, persisted)
    files = [tmp_path / f"input-{index}.json" for index in range(6)]
    for path in files:
        path.write_text("{}", encoding="utf-8")
    campaign = Round17CampaignOperatorConfig(
        campaign_plan_path=files[0],
        cohort_plan_path=files[1],
        admission_spec_path=files[2],
        database_path=tmp_path / "never-open.duckdb",
        state_root=tmp_path / "never-open-state",
    )
    output = tmp_path / "republished.json"
    config = Round17OneUseEvaluationConfig(
        repository=tmp_path,
        repository_commit_sha="f" * 40,
        contract_path=files[3],
        development_result_path=files[4],
        risk_contract_path=files[5],
        claim_store_path=store_path,
        resolution_checkpoint_path=tmp_path / "resolution.json",
        output_path=output,
        campaign=campaign,
    )
    monkeypatch.setattr(
        evaluation,
        "stage_round17_one_use_claim",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed claim attempted to restage")
        ),
    )

    result = run_round17_one_use_evaluation(config)

    assert result == persisted
    assert json.loads(output.read_text(encoding="ascii")) == persisted
    assert not campaign.database_path.exists()
