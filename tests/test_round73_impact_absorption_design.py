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
BASE_CAPTURE_CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-073-capture-contract.json"
)
CAPTURE_CONTRACT_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-capture-contract-v2.json"
)
CORRECTION_EVIDENCE_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-feed-contract-correction-evidence-2026-07-22.json"
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
    base_contract = json.loads(BASE_CAPTURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    base_claimed = base_contract.pop("capture_contract_sha256")
    assert base_claimed == _canonical_sha256(base_contract)

    contract = json.loads(CAPTURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    claimed = contract.pop("capture_contract_sha256")

    assert claimed == _canonical_sha256(contract)
    assert contract["schema_version"] == "round-073-prospective-capture-contract-v2"
    assert contract["parent_design_sha256"] == (
        "84b5e6c942d03ebd97b7e120951ed576e3fd8161d65755734c359b7261d6b1fe"
    )
    assert contract["inheritance"]["base_contract_sha256"] == base_claimed
    assert contract["scope"]["market_calendar"].startswith("continuous")
    assert "never a crypto market close" in contract["scope"]["utc_day_semantics"]
    assert base_contract["wire_evidence"]["credentials_permitted"] is False
    assert base_contract["writer"]["writer_count"] == 1
    assert base_contract["writer"]["queue_capacity_messages"] == 65_536
    assert base_contract["writer"]["hard_compressed_payload_cap_required"] is True
    assert (
        base_contract["raw_to_typed_link"][
            "typed_event_without_raw_reference_permitted"
        ]
        is False
    )
    assert base_contract["depth_state"]["stored_levels_per_side"] == 20
    assert (
        base_contract["depth_state"]["state_across_integrity_segment_permitted"]
        is False
    )
    assert base_contract["depth_state"]["exact_cancellation_label_permitted"] is False
    assert (
        contract["combined_stream_identity"]["wrapper_must_match_exact_subscription"]
        is True
    )
    assert (
        contract["stream_specific_fields"]["forceOrder"]["stream_type_path"]
        == "data.o.st"
    )
    assert (
        contract["rejected_wire_evidence"]["parse_before_persistence_permitted"]
        is False
    )
    assert contract["qualification"]["failure_authorizes_modeling"] is False


def test_round73_feed_contract_correction_evidence_is_hash_bound() -> None:
    evidence = json.loads(CORRECTION_EVIDENCE_PATH.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["credentials_used"] is False
    assert evidence["orders_submitted"] is False
    assert len(evidence["failed_qualification_attempts"]) == 2
    probe = evidence["live_force_order_probe"]
    assert probe["top_level_data_st_present"] is False
    assert probe["observed_paths"]["stream_type"] == "data.o.st"
    assert probe["raw_sha256"] == (
        "972aacab0e6cbd8d026505e9ab4d7721a9a6275346ddd845fb32fa2186b8d805"
    )
