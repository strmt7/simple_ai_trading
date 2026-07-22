from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-073-impact-absorption-design.json"
)
CAPTURE_CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-073-capture-contract.json"
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def test_round73_design_is_sealed_and_fail_closed() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    claimed = design.pop("design_sha256")

    assert claimed == _canonical_sha256(design)
    assert design["round"] == 73
    assert design["schema_version"] == "round-073-impact-absorption-design-v2"
    assert design["revision"]["modeling_capture_observed_before_revision"] is False
    assert design["source_contract"]["symbols"] == [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    ]
    assert design["source_contract"]["market_calendar"].startswith("continuous")
    assert design["source_contract"]["listed_etf_or_equity_close_feature"] is False
    assert design["order_book_integrity_contract"]["sequence_gap_policy"].startswith(
        "invalidate"
    )
    assert design["order_book_integrity_contract"]["queue_overflow_policy"].startswith(
        "invalidate"
    )
    assert design["depth_change_semantics"]["exact_cancellation_observable"] is False
    assert (
        design["depth_change_semantics"]["unmatched_removal_is_cancellation_label"]
        is False
    )
    assert design["model_contract"]["temporal_neural_challenger_permitted"] is False
    assert design["model_contract"]["reinforcement_learning_permitted"] is False
    assert design["model_contract"]["ai_veto_permitted"] is False
    assert design["economic_gate_after_predictive_pass"]["unlevered_only"] is True
    assert design["evaluation_contract"]["minimum_symbols_for_portfolio_research"] == 2
    assert design["governance"]["profitability_claim_permitted"] is False
    assert design["governance"]["trading_authority_permitted"] is False


def test_round73_capture_contract_closes_storage_and_calendar_ambiguity() -> None:
    contract = json.loads(CAPTURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    claimed = contract.pop("capture_contract_sha256")

    assert claimed == _canonical_sha256(contract)
    assert contract["parent_design_sha256"] == (
        "84b5e6c942d03ebd97b7e120951ed576e3fd8161d65755734c359b7261d6b1fe"
    )
    assert contract["scope"]["market_calendar"].startswith("continuous")
    assert "never a crypto market close" in contract["scope"]["utc_day_semantics"]
    assert contract["wire_evidence"]["credentials_permitted"] is False
    assert contract["writer"]["writer_count"] == 1
    assert contract["writer"]["queue_capacity_messages"] == 65_536
    assert contract["writer"]["hard_compressed_payload_cap_required"] is True
    assert (
        contract["raw_to_typed_link"]["typed_event_without_raw_reference_permitted"]
        is False
    )
    assert contract["depth_state"]["stored_levels_per_side"] == 20
    assert contract["depth_state"]["state_across_integrity_segment_permitted"] is False
    assert contract["depth_state"]["exact_cancellation_label_permitted"] is False
    assert contract["qualification"]["failure_authorizes_modeling"] is False
