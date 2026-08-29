from __future__ import annotations

import hashlib
import json
import statistics
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs" / "model-research" / "action-value"
ARTIFACT_PATH = (
    ACTION_VALUE
    / "binance-polymarket-option-threshold-wedge-gate-v1-2026-08-26.json"
)
REGISTRY_PATH = (
    ROOT / "docs" / "model-research" / "structural-edge-priority-registry-v1.json"
)
EXPECTED_ARTIFACT_HASH = (
    "22a99f25de487774ac4d22f4666a242fe3cb961e31f7f610de7a079cd6d9d7e7"
)
EXPECTED_REGISTRY_HASH = (
    "a94c524d3c73dfdf52275b384b8c18d84314ceb53be57ae744df258cbe7cdef0"
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


def test_gate_is_hash_bound_public_only_and_grants_no_authority() -> None:
    artifact = _load(ARTIFACT_PATH)

    assert artifact["result_sha256"] == EXPECTED_ARTIFACT_HASH
    assert _embedded_hash(artifact) == EXPECTED_ARTIFACT_HASH
    assert artifact["authority"] == {
        "accepted_edge": False,
        "credentials_used": False,
        "funds_used": False,
        "live_trading_authority": False,
        "orders_placed": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
        "signed_requests_made": 0,
    }
    assert {row["status_code"] for row in artifact["current_public_model_snapshot"]["input_sources"]} == {200}


def test_catalog_proves_no_exact_current_payoff_pair() -> None:
    artifact = _load(ARTIFACT_PATH)
    catalog = artifact["current_public_catalog_gate"]

    assert catalog["active_btc_tagged_event_count"] == 505
    assert catalog["active_point_threshold_market_count"] == 83
    assert catalog["binance_btc_call_count"] == 337
    assert catalog["exact_same_strike_and_expiry_pair_count"] == 0
    assert catalog["same_calendar_date_and_strike_pair_count"] == 20
    assert catalog["same_calendar_date_pair_expiry_offset_hours"] == "8"
    assert catalog["settlement_identity"]["exact"] is False


def test_current_model_wedge_reconstructs_and_fails_escalation() -> None:
    artifact = _load(ARTIFACT_PATH)
    snapshot = artifact["current_public_model_snapshot"]
    rows = snapshot["pair_rows"]
    summary = snapshot["summary"]
    gaps = [Decimal(row["model_mid_gap"]) for row in rows]
    quantum = Decimal("0.000000000001")

    assert len(rows) == summary["pair_count"] == 20
    assert sum(row["option_two_sided_quote"] for row in rows) == 15
    assert sum(gap > 0 for gap in gaps) == summary["positive_mid_gap_count"] == 16
    assert sum(gap < 0 for gap in gaps) == summary["negative_mid_gap_count"] == 4
    assert (sum(gaps) / len(gaps)).quantize(quantum, rounding=ROUND_HALF_UP) == Decimal(
        summary["mean_mid_gap"]
    )
    assert statistics.median(gaps) == Decimal(summary["median_mid_gap"])
    assert min(gaps) == Decimal(summary["minimum_mid_gap"])
    assert max(gaps) == Decimal(summary["maximum_mid_gap"])
    assert max(abs(gap) for gap in gaps) == Decimal(
        summary["maximum_absolute_mid_gap"]
    )
    friction = Decimal(
        snapshot["paper_mean_friction_screen"]["historical_mean_friction_term"]
    )
    assert sum(abs(gap) > friction for gap in gaps) == 0
    assert (
        snapshot["paper_mean_friction_screen"][
            "pairs_whose_absolute_model_midpoint_gap_exceeded_historical_mean_friction"
        ]
        == 0
    )


def test_registry_separates_statistical_lead_from_terminal_exact_parity() -> None:
    registry = _load(REGISTRY_PATH)

    assert registry["result_sha256"] == EXPECTED_REGISTRY_HASH
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_HASH
    assert [row["priority_rank"] for row in registry["prioritized_hypotheses"]] == list(
        range(1, 45)
    )
    lead = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "cross_venue_option_implied_prediction_threshold_wedge"
    )
    assert lead["priority_rank"] == 10
    assert lead["market_direction_forecast_required"] is False
    assert lead["canonical_artifacts"] == [
        {
            "path": (
                "docs/model-research/action-value/"
                "binance-polymarket-option-threshold-wedge-gate-v1-2026-08-26.json"
            ),
            "result_sha256": EXPECTED_ARTIFACT_HASH,
        }
    ]
    terminal = {row["family"]: row for row in registry["terminal_do_not_repeat"]}
    assert (
        terminal["binance_polymarket_exact_point_threshold_payoff_parity_current_catalog"][
            "canonical_result_sha256"
        ]
        == EXPECTED_ARTIFACT_HASH
    )
