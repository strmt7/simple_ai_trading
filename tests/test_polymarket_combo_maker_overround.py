from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

from tools import analyze_polymarket_combo_maker_overround as analysis


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "docs/model-research/action-value/"
    "polymarket-combo-maker-overround-validation-v1-2026-08-27.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
RESULT_HASH = "416daf4d279e06a2353127e642d588a39ae85be0709c2d7498896c1d182847ee"
REGISTRY_HASH = "2baf1b76070e0ef9081f9eb5fba41f3977b5fd1aa74759ed85034947e9ad1c5a"


def test_economics_excludes_buyer_fees_from_opposite_side_proxy() -> None:
    row = analysis._economics(
        {
            "gross_entry_cost_usdc": "100.000000",
            "entry_fees_usdc": "4.000000",
            "realized_payout_usdc": "90.000000",
            "first_entry_at": "2026-08-27T00:00:00Z",
        }
    )

    assert row["buyer_net"] == Decimal("-10.000000")
    assert row["seller_proxy"] == Decimal("6.000000")
    assert row["entry_date"] == "2026-08-27"


def test_combined_status_filter_is_explicit_and_complete() -> None:
    contract = json.loads(
        (
            ROOT
            / "docs/model-research/action-value/"
            "polymarket-combo-maker-overround-validation-contract-v1.json"
        ).read_text(encoding="utf-8")
    )

    assert analysis.RESOLVED_STATUSES == (
        "RESOLVED_WIN,RESOLVED_PARTIAL,RESOLVED_LOSS"
    )
    assert (
        contract["unseen_validation_cohort"]["position_parameters"]["status"]
        == analysis.RESOLVED_STATUSES
    )
    assert "use_the_default_combo_position_status_listing" in contract[
        "prohibited_shortcuts"
    ]


def test_canonical_result_terminalizes_broad_combo_overround() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    claimed = result.pop("result_sha256")
    assert hashlib.sha256(analysis._canonical_json(result).encode("ascii")).hexdigest() == claimed
    assert claimed == RESULT_HASH
    assert result["collection"]["resolved_yes_position_count"] == 6264
    assert result["collection"]["http_status_counts"] == {"200": 1240}
    assert (
        result["economics"]["overall"]["opposite_side_gross_spread_proxy_pUSD"]
        == "-73368.711836"
    )
    assert result["gate"]["passes_unaccepted_candidate_gate"] is False
    assert result["verdict"]["accepted_edge"] is False

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    registry_claimed = registry.pop("result_sha256")
    assert (
        hashlib.sha256(analysis._canonical_json(registry).encode("ascii")).hexdigest()
        == registry_claimed
    )
    assert registry_claimed == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 19
    terminal = {row["family"]: row for row in registry["terminal_do_not_repeat"]}
    assert terminal[
        "polymarket_broad_sports_combo_requester_overround_as_market_maker_edge"
    ]["canonical_result_sha256"] == RESULT_HASH
