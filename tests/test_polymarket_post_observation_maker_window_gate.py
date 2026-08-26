from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "docs/model-research/action-value/polymarket-post-observation-maker-window-gate-v1-2026-08-26.json"
EXPECTED_HASH = "d6f790d7de78fc4f1c527b9a6528ebb3faa31fc11d810d9daccec3e4a7d9084e"
REGISTRY_PATH = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_REGISTRY_HASH = "35adedbfbb8b11776592602a9c64b692b90b12c7fe458f0e52b65ee218ca6d10"


def _load() -> dict[str, object]:
    return json.loads(PATH.read_bytes())


def _embedded_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    canonical = json.dumps(body, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def test_gate_is_hash_bound_and_grants_no_authority() -> None:
    artifact = _load()
    assert artifact["result_sha256"] == EXPECTED_HASH
    assert _embedded_hash(artifact) == EXPECTED_HASH
    assert artifact["authority"]["accepted_edge"] is False
    assert artifact["authority"]["orders_placed"] is False
    assert artifact["authority"]["profitability_claim"] is False


def test_public_causal_rows_reconstruct_direction_slices_and_gross() -> None:
    artifact = _load()
    rows = artifact["evidence_rows"]
    summary = artifact["economic_summary"]
    assert len(rows) == summary["complete_conditions"] == 10
    assert all(row["oracle_receipt_delay_ms"] < row["first_winner_bid_growth_delay_ms"] <= row["first_later_winner_sell_fill_delay_ms"] for row in rows)
    up = [row for row in rows if row["outcome"] == "Up"]
    down = [row for row in rows if row["outcome"] == "Down"]
    assert len(up) == summary["up_condition_count"] == 3
    assert len(down) == summary["down_condition_count"] == 7
    assert sum(Decimal(row["observed_gross_pusd"]) for row in up) == Decimal(summary["up_condition_observed_gross_pusd"])
    assert sum(Decimal(row["observed_gross_pusd"]) for row in down) == Decimal(summary["down_condition_observed_gross_pusd"])
    assert sum(Decimal(row["observed_gross_pusd"]) for row in rows) == Decimal(summary["observed_gross_pusd"])


def test_gate_preserves_the_decisive_execution_unknown() -> None:
    artifact = _load()
    assert artifact["candidate_status"] == "high_priority_conditional_execution_lead_not_an_accepted_edge"
    limitations = " ".join(artifact["unresolved_gates"])
    assert "does not" in limitations
    assert "authenticated order" in limitations
    assert "one degraded BTC hour" in limitations


def test_registry_promotes_only_a_conditional_lead() -> None:
    registry = json.loads(REGISTRY_PATH.read_bytes())
    assert registry["result_sha256"] == EXPECTED_REGISTRY_HASH
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_HASH
    hypotheses = registry["prioritized_hypotheses"]
    assert [row["priority_rank"] for row in hypotheses] == list(range(1, 14))
    lead = next(
        row
        for row in hypotheses
        if row["mechanism"] == "post_observation_oracle_to_clob_close_maker_window"
    )
    assert lead["priority_rank"] == 2
    assert lead["market_direction_forecast_required"] is False
    assert lead["canonical_artifacts"][0]["result_sha256"] == EXPECTED_HASH
    assert "authenticated_order_acceptance" in lead["current_status"]
