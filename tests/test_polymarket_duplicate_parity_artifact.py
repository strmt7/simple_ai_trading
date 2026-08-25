from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simple_ai_trading.polymarket_duplicate_parity import (
    DuplicateContractTerms,
    discover_duplicate_contracts,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "duplicate-contract-parity-snapshot-v1-2026-08-25.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict[str, object]:
    return json.loads(ARTIFACT.read_text(encoding="ascii"))


def _contract(row: dict[str, object]) -> DuplicateContractTerms:
    return DuplicateContractTerms(
        event_id=str(row["event_id"]),
        market_id=str(row["market_id"]),
        condition_id=str(row["condition_id"]),
        question=str(row["question"]),
        description=str(row["description"]),
        end_date=str(row["end_date"]),
        resolution_source=str(row["resolution_source"]),
        group_item_title=str(row["group_item_title"]),
        outcomes=tuple(row["outcomes"]),  # type: ignore[arg-type]
        token_ids=tuple(row["token_ids"]),  # type: ignore[arg-type]
    )


def test_duplicate_artifact_reconstructs_result_and_implementation_hashes() -> None:
    report = _report()
    claimed = report.pop("result_sha256")
    canonical = json.dumps(
        report,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    assert hashlib.sha256(canonical).hexdigest() == claimed

    implementation = report["source_contract"]["implementation"]
    assert (
        _sha256(ROOT / "tools" / implementation["tool_path"])
        == implementation["tool_sha256"]
    )
    assert (
        _sha256(ROOT / "src" / "simple_ai_trading" / implementation["module_path"])
        == implementation["module_sha256"]
    )


def test_duplicate_artifact_reconstructs_strict_rule_mismatch() -> None:
    report = _report()
    rows = [
        contract
        for group in report["exact_question_groups"]
        for contract in group["contracts"]
    ]
    contracts = tuple(_contract(row) for row in rows)
    discovery = discover_duplicate_contracts(contracts)

    assert report["universe"]["canonical_candidate_contract_count"] == len(contracts)
    assert report["universe"]["exact_question_candidate_group_count"] == len(
        discovery.exact_question_groups
    )
    assert len(discovery.exact_payout_rule_groups) == 0
    assert report["exact_payout_rule_groups"] == []
    assert report["exact_question_groups"][0]["differing_payout_rule_fields"] == [
        "description",
        "group_item_title",
    ]
    for contract, row in zip(
        discovery.exact_question_groups[0].contracts,
        report["exact_question_groups"][0]["contracts"],
        strict=True,
    ):
        assert row["payout_rule_sha256"] == contract.payout_rule_sha256


def test_duplicate_artifact_fails_before_price_or_edge_claim() -> None:
    report = _report()
    assert report["verdict"] == {
        "accepted_edge": False,
        "depth_or_fee_screen_performed": False,
        "exact_payout_rule_group_count": 0,
        "exact_question_group_count": 1,
        "status": "rejected_no_exact_payout_rule_duplicate",
        "trading_authority": False,
    }
    assert report["safety"] == {
        "credentials_used": False,
        "non_equivalent_pairs_priced_as_guaranteed_bundles": False,
        "orders_placed": False,
        "public_market_data_only": True,
        "title_only_equivalence_allowed": False,
    }
