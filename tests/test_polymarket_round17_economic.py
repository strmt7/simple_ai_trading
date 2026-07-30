from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
import hashlib
import json
from pathlib import Path

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


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-014-btc-5m-prospective-contract-v1.json"
)
START_MS = 1_800_057_600_000
MODEL_SHA256 = "a" * 64
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


def test_round17_economic_pretest_is_complete_test_blind_and_hash_bound() -> None:
    program, outcomes = _outcomes()

    artifact = fit_round17_economic_pretest(
        outcomes,
        program,
        model_pretest_sha256=MODEL_SHA256,
    )
    verified = validate_round17_economic_pretest(artifact)

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


def test_round17_economic_pretest_rejects_unprofitable_development_paths() -> None:
    program, outcomes = _outcomes(negative=True)

    artifact = fit_round17_economic_pretest(
        outcomes,
        program,
        model_pretest_sha256=MODEL_SHA256,
    )

    assert artifact["development_accepted"] is False
    assert all(
        item["development_accepted"] is False
        for item in artifact["selected_by_profile"].values()
    )


def test_round17_economic_pretest_rejects_incomplete_grid_and_rehashed_drift() -> None:
    program, outcomes = _outcomes()

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
            model_pretest_sha256=MODEL_SHA256,
        )

    artifact = fit_round17_economic_pretest(
        outcomes,
        program,
        model_pretest_sha256=MODEL_SHA256,
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
