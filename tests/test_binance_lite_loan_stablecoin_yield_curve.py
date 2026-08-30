from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-lite-loan-stablecoin-yield-curve-contract-v1.json"
)
RESULT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-lite-loan-stablecoin-yield-curve-v1-2026-08-27.json"
)
REGISTRY_PATH = (
    ROOT / "docs" / "model-research" / "structural-edge-priority-registry-v1.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "73f6a0362ca88db393b723119b21e603f202345f570b59a70204fa3779349d41"
)
EXPECTED_RESULT_SHA256 = (
    "65f223a245fa1bb65a8fd791275da0dbd71d3c52ee2d232ac1420feb198b129d"
)
EXPECTED_FEE_EVIDENCE_SHA256 = (
    "4842bebff1b6177b2053d0fdc40680a2224f01fb541d0efcc85b08f049f68184"
)
EXPECTED_REGISTRY_SHA256 = json.loads(
    (ROOT / "docs/model-research/structural-edge-priority-registry-v1.json").read_text(
        encoding="utf-8"
    )
)["result_sha256"]


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_bytes())


def _embedded_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    canonical = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def test_contract_and_result_reconstruct_from_the_frozen_evidence() -> None:
    contract = _load(CONTRACT_PATH)
    result = _load(RESULT_PATH)

    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == (
        EXPECTED_CONTRACT_SHA256
    )
    assert result["contract"] == {
        "path": (
            "docs/model-research/action-value/"
            "binance-lite-loan-stablecoin-yield-curve-contract-v1.json"
        ),
        "sha256": EXPECTED_CONTRACT_SHA256,
    }
    assert result["result_sha256"] == EXPECTED_RESULT_SHA256
    assert _embedded_hash(result) == EXPECTED_RESULT_SHA256
    assert contract["frozen_at_utc"] < result["raw_evidence"]["requested_before_utc"]
    assert (
        result["raw_evidence"]["requested_before_utc"]
        < (result["raw_evidence"]["received_after_utc"])
    )
    assert (
        result["raw_evidence"]["requested_before_utc"]
        < (contract["source_bound_inputs"]["lite_loan_promotion"]["end_utc"])
    )
    assert all(
        result["raw_evidence"]["requested_before_utc"] < row["promotion_end_utc"]
        for row in contract["yield_routes"]
    )
    assert contract["source_bound_inputs"]["current_zero_fee_row"] == {
        "normalized_evidence_sha256": EXPECTED_FEE_EVIDENCE_SHA256,
        "observed_at_utc": "2026-08-27T03:57:49.7829309Z",
        "row": "USD1/USDT|maker_buy_sell=0%|taker_buy_sell=0%|action=Trade",
        "url": "https://www.binance.com/en/fee/tradingPromote",
    }


def test_one_public_request_used_no_credentials_or_trading_authority() -> None:
    contract = _load(CONTRACT_PATH)
    result = _load(RESULT_PATH)

    assert contract["market_data_request"]["method"] == "GET"
    assert contract["market_data_request"]["maximum_requests"] == 1
    assert contract["market_data_request"]["no_retry"] is True
    assert contract["authority"]["credentials_allowed"] is False
    assert contract["authority"]["funded_actions_allowed"] is False
    assert result["authority"] == {
        "account_requests": 0,
        "credentials_used": False,
        "funded_actions": 0,
        "orders_conversions_subscriptions_borrows_or_repays": 0,
        "public_market_data_requests": 1,
    }
    assert result["adjudication"]["trading_authority"] is False
    assert result["adjudication"]["deployment_ready"] is False


def test_only_usd1_survives_the_fixed_bonus_historical_stress_gate() -> None:
    result = _load(RESULT_PATH)
    routes = {row["asset"]: row for row in result["routes"]}

    assert result["adjudication"][
        "public_positive_fixed_bonus_historical_stress_candidates"
    ] == ["USD1"]
    assert result["adjudication"]["accepted_edge"] is False
    assert result["adjudication"]["stable_edge"] is False
    assert result["adjudication"]["profitability_claim"] is False
    assert all(row["real_time_APR_credited"] is False for row in routes.values())
    assert routes["USD1"]["fixed_bonus_only_historical_stress_candidate"] is True
    assert routes["U"]["fixed_bonus_only_historical_stress_candidate"] is False
    assert routes["USDT"]["fixed_bonus_only_historical_stress_candidate"] is False

    expected_usd1_stressed_bips = {
        "100": Decimal("1.1078834807974437609637580520820740821917808220000"),
        "500": Decimal("1.2964161079537919049695990099734805917808219178000"),
        "1000": Decimal("1.3278382124798499289705725029553816767123287670000"),
    }
    for evaluation in routes["USD1"]["evaluations"]:
        loan = evaluation["loan_amount_USDT"]
        assert (
            Decimal(evaluation["stressed_net_bips_of_loan"])
            == (expected_usd1_stressed_bips[loan])
        )
        assert Decimal(evaluation["stressed_net_USDT"]) > 0
        assert evaluation["entry_capacity_valid"] is True
        assert evaluation["exit_capacity_valid"] is True

    for asset in ("U", "USDT"):
        assert all(
            Decimal(row["stressed_net_bips_of_loan"]) <= 0
            for row in routes[asset]["evaluations"]
        )


def test_registry_records_candidate_without_inflating_accepted_edges() -> None:
    registry = _load(REGISTRY_PATH)

    assert registry["result_sha256"] == EXPECTED_REGISTRY_SHA256
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_SHA256
    assert registry["accepted_edge_count"] == 21
    assert [row["priority_rank"] for row in registry["prioritized_hypotheses"]] == list(
        range(1, 45)
    )
    candidate = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "same_account_stable_value_yield_allocation"
    )
    artifacts = {
        row["path"]: row["result_sha256"] for row in candidate["canonical_artifacts"]
    }
    assert artifacts[CONTRACT_PATH.relative_to(ROOT).as_posix()] == (
        EXPECTED_CONTRACT_SHA256
    )
    assert artifacts[RESULT_PATH.relative_to(ROOT).as_posix()] == (
        EXPECTED_RESULT_SHA256
    )
    assert candidate["market_direction_forecast_required"] is False
    assert "none_is_deployment_ready" in candidate["current_status"]
    assert "separate_funded_authority" in candidate["next_action"]
