from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ARTIFACT_PATH = Path(
    "docs/model-research/action-value/"
    "binance-bstock-funding-full-universe-v1-2026-08-26.json"
)
CONTRACT_PATH = Path(
    "docs/model-research/action-value/"
    "binance-bstock-funding-full-universe-contract-v1.json"
)
TOOL_PATH = Path("tools/screen_binance_bstock_funding_full_universe.py")
EXPECTED_RESULT_HASH = (
    "ad3fbc7a09ff6b467955eeef8bf1e8df4ba7d20ca9e7659fcaf75069da622d3f"
)
EXPECTED_CONTRACT_HASH = (
    "5c86b4cf42e74a1feb2fa75895aeddfd0db120c93b63952609b66068bf5aa342"
)
EXPECTED_TOOL_HASH = "1b602841d8340e7adc137570ba50b3f467107505d6889445784f9b21fd9675d3"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def test_full_universe_artifact_contract_and_tool_are_source_bound() -> None:
    artifact = _load(ARTIFACT_PATH)
    body = dict(artifact)
    assert body.pop("result_sha256") == EXPECTED_RESULT_HASH
    assert hashlib.sha256(_canonical_json(body).encode("ascii")).hexdigest() == (
        EXPECTED_RESULT_HASH
    )
    assert artifact["implementation"] == {
        "path": TOOL_PATH.as_posix(),
        "sha256": EXPECTED_TOOL_HASH,
    }
    assert hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest() == EXPECTED_TOOL_HASH

    contract = _load(CONTRACT_PATH)
    contract_body = dict(contract)
    assert contract_body.pop("contract_sha256") == EXPECTED_CONTRACT_HASH
    assert (
        hashlib.sha256(_canonical_json(contract_body).encode("ascii")).hexdigest()
        == EXPECTED_CONTRACT_HASH
    )
    assert artifact["contract"] == {
        "path": CONTRACT_PATH.as_posix(),
        "sha256": EXPECTED_CONTRACT_HASH,
        "status": "frozen_before_unobserved_full_universe_funding_outcome_access",
    }


def test_full_exact_multiplier_universe_was_screened_without_exposed_confirmation() -> (
    None
):
    artifact = _load(ARTIFACT_PATH)
    assert artifact["scope"] == {
        "confirmation_eligible_count": 57,
        "exact_multiplier_count": 60,
        "matched_structural_count": 66,
        "non_exact_multiplier_count": 6,
        "previously_observed_exact_count": 3,
        "research_only_outside_current_btc_eth_sol_execution_scope": True,
    }
    rows = artifact["training_screen"]
    assert len(rows) == 60
    assert len(artifact["sources"]["funding_histories"]) == 60
    assert {row["ticker"] for row in rows if row["previously_observed"] is True} == {
        "DRAM",
        "MRVL",
        "SNDK",
    }
    assert all(
        row["confirmation_eligible"] is False
        for row in rows
        if row["ticker"] in {"DRAM", "MRVL", "SNDK"}
    )


def test_frozen_training_gate_rejects_all_symbols_without_opening_sealed_roles() -> (
    None
):
    artifact = _load(ARTIFACT_PATH)
    eligible = [
        row for row in artifact["training_screen"] if row["confirmation_eligible"]
    ]
    assert len(eligible) == 57
    assert (
        sum(
            Decimal(row["training"]["net_after_frozen_hurdles_bips"]) > 0
            for row in eligible
        )
        == 16
    )
    assert all(
        "family_adjusted_bootstrap_lower_bound_not_positive"
        in row["training"]["rejection_reasons"]
        for row in eligible
    )
    assert (
        sum(
            "one_or_more_required_slices_failed" in row["training"]["rejection_reasons"]
            for row in eligible
        )
        == 56
    )
    assert artifact["training_selected_count"] == 0
    assert artifact["causal_confirmation"] == []
    assert artifact["empirical_gate_pass_count"] == 0
    assert artifact["sources"]["selected_current_depth"] == []


def test_lite_is_the_best_stressed_training_candidate_but_still_fails() -> None:
    artifact = _load(ARTIFACT_PATH)
    eligible = [
        row for row in artifact["training_screen"] if row["confirmation_eligible"]
    ]
    best = max(
        eligible,
        key=lambda row: Decimal(row["training"]["net_after_frozen_hurdles_bips"]),
    )
    assert best["ticker"] == "LITE"
    assert Decimal(best["training"]["gross_funding_bips"]) == Decimal("497.67570000")
    assert Decimal(best["training"]["net_after_frozen_hurdles_bips"]) == Decimal(
        "297.0364305936073059360730594"
    )
    assert Decimal(
        best["training"]["family_adjusted_bootstrap_net_lower_bound_bips"]
    ) == Decimal("-5.9967694063926940639269406")
    assert best["selected_by_training_only"] is False


def test_failure_remains_research_only_and_cannot_expand_execution_scope() -> None:
    artifact = _load(ARTIFACT_PATH)
    assert artifact["verdict"] == {
        "accepted_edge": False,
        "deployment_ready": False,
        "empirical_gate_pass": False,
        "status": "full_universe_causal_cross_regime_gate_rejected",
        "trading_authority": False,
    }
    assert artifact["authority"] == {
        "conversions_requested": False,
        "credentials_used": False,
        "funds_used": False,
        "orders_placed": False,
        "trading_authority": False,
    }
    assert (
        "explicit scope expansion beyond BTC ETH and SOL before execution work"
        in artifact["blocking_evidence"]
    )
