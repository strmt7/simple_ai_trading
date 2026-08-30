from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "crypto-maker-rebate-public-recurrence-v2-2026-08-26.json"
)
MAKER_FIRST_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "polymarket-maker-first-taker-hedge-complete-set-candidate-v1-2026-08-27.json"
)
MAKER_FIRST_DIAGNOSTIC_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "maker-first-taker-hedge-historical-diagnostic-v1-2026-08-27.json"
)
MANIPULATION_GATE_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "polymarket-maker-execution-manipulation-regime-gate-v1-2026-08-27.json"
)
TAKER_DELAY_REGIME_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "polymarket-crypto-taker-delay-regime-change-v1-2026-08-27.json"
)
REGISTRY_PATH = (
    ROOT / "docs" / "model-research" / "structural-edge-priority-registry-v1.json"
)
EXPECTED_RESULT_SHA256 = (
    "c992e0e1febc1a9789289cb129c166280ee0192cab203d3a6935a8c40e949612"
)
EXPECTED_REGISTRY_SHA256 = json.loads(
    (ROOT / "docs/model-research/structural-edge-priority-registry-v1.json").read_text(
        encoding="utf-8"
    )
)["result_sha256"]
EXPECTED_MAKER_FIRST_SHA256 = (
    "4fe308ddeb6fd080bbd8548347a095762d8fc67eb5820fb0c7b3c2d6b7430d69"
)
EXPECTED_MAKER_FIRST_DIAGNOSTIC_RESULT_SHA256 = (
    "d11d0f42a082fc19b1cd6ee3ee6cd226f95b28db2e1f9c5cf965913e35ce6b07"
)
EXPECTED_MANIPULATION_GATE_SHA256 = (
    "7d3387289a7e82b33fa52c03b2bc134864259a001c3d28524745026bb83db387"
)
EXPECTED_TWAP_CONTRACT_SHA256 = (
    "0f872c30360d8184905d7468f2ad78c1de439f3e97ad2e4ec1f082b71c60edda"
)
EXPECTED_TWAP_FAILURE_SHA256 = (
    "e486f2928a326e6829cbe3c07aad5a47bb25a63783a935273606df00cea98c66"
)
EXPECTED_TAKER_DELAY_REGIME_SHA256 = (
    "c7b785a1fbf4d6380033810338b2cf2845399f2a7464688c8ac36427b375a777"
)


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


def test_public_recurrence_hash_and_authority_reconstruct() -> None:
    artifact = _load(ARTIFACT_PATH)

    assert artifact["result_sha256"] == EXPECTED_RESULT_SHA256
    assert _embedded_hash(artifact) == EXPECTED_RESULT_SHA256
    assert artifact["authority"]["authenticated_requests"] == 0
    assert artifact["authority"]["orders_submitted"] == 0
    assert artifact["adjudication"]["public_program_payment_recurrence_proved"]
    assert not artifact["adjudication"]["accepted_edge"]
    assert (
        artifact["adjudication"][
            "publicly_proven_payout_for_a_fresh_hypothetical_order"
        ]
        == "0"
    )


def test_exact_wallet_day_is_joined_to_btc_eth_sol_markets() -> None:
    artifact = _load(ARTIFACT_PATH)
    day = artifact["exact_scope_day"]
    scope = day["scoped_btc_eth_sol"]

    assert day["gamma_market_join"]["missing_market_count"] == 0
    assert scope["row_count"] == 668
    assert Decimal(scope["total_rebate_usdc"]) == Decimal("7017.331032")
    assert Decimal(scope["share_of_wallet_total_rebate"]) > Decimal("0.94")
    assert {row["asset"] for row in scope["assets"]} == {"btc", "eth", "sol"}
    assert scope["all_fee_rates"] == ["0.07"]
    assert scope["all_rebate_rates"] == ["0.2"]


def test_cohort_recurrence_does_not_promote_diagnostic_ratio() -> None:
    artifact = _load(ARTIFACT_PATH)
    week = artifact["week_volume_cohort_7_day_diagnostic"]
    ledger = artifact["request_ledger"]

    assert week["wallet_count"] == 10
    assert week["wallets_with_receipts_on_all_7_utc_dates"] == 10
    assert Decimal(week["aggregate_receipt_to_crypto_volume_bips_diagnostic_only"]) > 0
    assert any(
        "Receipt-to-volume figures are diagnostics" in limitation
        for limitation in artifact["limitations"]
    )
    assert ledger["direct_public_http_response_count"] == (
        ledger["canonical_evidence_response_count"]
        + ledger["discarded_exploratory_or_repeated_response_count"]
    )
    assert "reuse retained evidence" in ledger["workflow_correction"]


def test_maker_first_diagnostic_rejects_stable_edge_claim() -> None:
    candidate = _load(MAKER_FIRST_PATH)
    diagnostic = _load(MAKER_FIRST_DIAGNOSTIC_PATH)

    assert candidate["result_sha256"] == EXPECTED_MAKER_FIRST_SHA256
    assert _embedded_hash(candidate) == EXPECTED_MAKER_FIRST_SHA256
    assert diagnostic["result_sha256"] == EXPECTED_MAKER_FIRST_DIAGNOSTIC_RESULT_SHA256
    assert _embedded_hash(diagnostic) == EXPECTED_MAKER_FIRST_DIAGNOSTIC_RESULT_SHA256
    assert not candidate["adjudication"]["accepted_edge"]
    assert candidate["adjudication"]["public_after_cost_profit_floor_pusd"] == "0"
    assert diagnostic["results"]["overall"]["sequence_count"] == 159
    assert (
        diagnostic["results"]["overall"]["current_fee_sensitive_positive_count"] == 75
    )
    assert diagnostic["results"]["overall"]["aggregate_current_fee_sensitive_pnl"] < 0
    assert diagnostic["results"]["zero_sequence_assets"] == ["SOL"]
    assert not diagnostic["verdict"]["market_direction_independent_edge_proved"]


def test_manipulation_regime_gate_rejects_five_minute_all_situation_edge() -> None:
    gate = _load(MANIPULATION_GATE_PATH)

    assert gate["result_sha256"] == EXPECTED_MANIPULATION_GATE_SHA256
    assert _embedded_hash(gate) == EXPECTED_MANIPULATION_GATE_SHA256
    assert not gate["adjudication"]["accepted_edge"]
    assert gate["adjudication"]["public_after_cost_profit_floor_pusd"] == "0"
    five_minute = gate["exact_five_minute_evidence"]
    assert (
        Decimal(
            five_minute["manipulated_cycles"]["market_maker_total_reported_PnL_usd"]
        )
        < 0
    )
    assert (
        Decimal(five_minute["normal_cycles"]["market_maker_total_reported_PnL_usd"]) > 0
    )
    assert gate["prospective_population_and_risk_gate"]["first_cohort"] == (
        "BTC_ETH_SOL_fifteen_minute_only"
    )


def test_current_crypto_taker_delay_rejects_stale_250ms_assumption() -> None:
    regime = _load(TAKER_DELAY_REGIME_PATH)

    assert regime["result_sha256"] == EXPECTED_TAKER_DELAY_REGIME_SHA256
    assert _embedded_hash(regime) == EXPECTED_TAKER_DELAY_REGIME_SHA256
    assert regime["authority"]["orders_or_cancellations"] == 0
    assert regime["execution_impact"]["old_crypto_taker_delay_ms"] == 250
    assert regime["execution_impact"]["current_crypto_taker_delay_ms"] == 50
    assert regime["execution_impact"]["reduction_fraction"] == "0.8"
    assert regime["research_consequence"]["public_after_cost_profit_floor_pusd"] == "0"
    assert regime["adjudication"]["accepted_edge"] is False


def test_registry_tracks_recurrence_without_increasing_accepted_edges() -> None:
    registry = _load(REGISTRY_PATH)

    assert registry["result_sha256"] == EXPECTED_REGISTRY_SHA256
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_SHA256
    candidate = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "paired_maker_rebates_and_liquidity_rewards"
    )
    assert candidate["priority_rank"] == 17
    artifacts_by_path = {
        artifact["path"]: artifact["result_sha256"]
        for artifact in candidate["canonical_artifacts"]
    }
    assert (
        artifacts_by_path[
            "docs/model-research/polymarket/"
            "crypto-maker-rebate-public-recurrence-v2-2026-08-26.json"
        ]
        == EXPECTED_RESULT_SHA256
    )
    assert (
        artifacts_by_path[
            "docs/model-research/action-value/"
            "polymarket-maker-first-taker-hedge-complete-set-candidate-v1-2026-08-27.json"
        ]
        == EXPECTED_MAKER_FIRST_SHA256
    )
    assert (
        artifacts_by_path[
            "docs/model-research/action-value/"
            "polymarket-maker-execution-manipulation-regime-gate-v1-2026-08-27.json"
        ]
        == EXPECTED_MANIPULATION_GATE_SHA256
    )
    assert (
        artifacts_by_path[
            "docs/model-research/polymarket/"
            "crypto-twap-liquidity-reward-screen-contract-v1.json"
        ]
        == EXPECTED_TWAP_CONTRACT_SHA256
    )
    assert (
        artifacts_by_path[
            "docs/model-research/polymarket/"
            "crypto-twap-liquidity-reward-screen-attempt1-failure-v1.json"
        ]
        == EXPECTED_TWAP_FAILURE_SHA256
    )
