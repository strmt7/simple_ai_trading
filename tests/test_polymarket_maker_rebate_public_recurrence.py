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
REGISTRY_PATH = (
    ROOT / "docs" / "model-research" / "structural-edge-priority-registry-v1.json"
)
EXPECTED_RESULT_SHA256 = (
    "c992e0e1febc1a9789289cb129c166280ee0192cab203d3a6935a8c40e949612"
)
EXPECTED_REGISTRY_SHA256 = (
    "5be46743ddcfa82d7526450272d35528d01e2d81e75b803924e7f947178d820d"
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
    assert artifact["adjudication"][
        "publicly_proven_payout_for_a_fresh_hypothetical_order"
    ] == "0"


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
    assert Decimal(
        week["aggregate_receipt_to_crypto_volume_bips_diagnostic_only"]
    ) > 0
    assert any(
        "Receipt-to-volume figures are diagnostics" in limitation
        for limitation in artifact["limitations"]
    )
    assert ledger["direct_public_http_response_count"] == (
        ledger["canonical_evidence_response_count"]
        + ledger["discarded_exploratory_or_repeated_response_count"]
    )
    assert "reuse retained evidence" in ledger["workflow_correction"]


def test_registry_tracks_recurrence_without_increasing_accepted_edges() -> None:
    registry = _load(REGISTRY_PATH)

    assert registry["result_sha256"] == EXPECTED_REGISTRY_SHA256
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_SHA256
    assert registry["accepted_edge_count"] == 3
    candidate = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "paired_crypto_maker_rebates"
    )
    assert candidate["priority_rank"] == 14
    assert candidate["canonical_artifacts"][-1] == {
        "path": (
            "docs/model-research/polymarket/"
            "crypto-maker-rebate-public-recurrence-v2-2026-08-26.json"
        ),
        "result_sha256": EXPECTED_RESULT_SHA256,
    }
