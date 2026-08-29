from __future__ import annotations

import gzip
import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    ROOT / "docs/model-research/action-value/"
    "polymarket-negrisk-maker-input-gate-v1-2026-08-26.json"
)
TERMINAL_PATH = (
    ROOT / "docs/model-research/action-value/"
    "polymarket-negrisk-maker-input-prospective-terminal-v1-2026-08-29.json"
)
REGISTRY_PATH = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_ARTIFACT_HASH = (
    "d4e02d2d1cc6b0a598265af734b29f62aec6145bc5a1cc3b3d65771ba2031d2a"
)
EXPECTED_REGISTRY_HASH = (
    "23479942f0f50760ad35df84f91707716d36ef06026c4ff664f618944e528680"
)
EXPECTED_TERMINAL_HASH = (
    "613453649f84407d6216e72228bdb16005b0a5c290c6bd58fa522007de5317e5"
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


def test_gate_is_hash_bound_fail_closed_and_direction_independent() -> None:
    artifact = _load(ARTIFACT_PATH)

    assert artifact["result_sha256"] == EXPECTED_ARTIFACT_HASH
    assert _embedded_hash(artifact) == EXPECTED_ARTIFACT_HASH
    assert artifact["adjudication"] == {
        **artifact["adjudication"],
        "accepted_edge": False,
        "candidate_edge": True,
        "deployment_ready": False,
        "market_direction_forecast_required": False,
    }
    assert artifact["authority"]["credentials_used"] is False
    assert artifact["authority"]["orders_or_conversions_submitted"] == 0
    economics = {
        (row["path"], row["quantity_shares"]): row["net_quote"]
        for row in artifact["role_economics_at_live_state"]["paths"]
    }
    assert economics[("all_taker", "20")] == "-0.28328"
    assert (
        economics[("maker_input_Bitcoin_NO_then_taker_sell_both_outputs", "20")]
        == "0.02960"
    )
    route = artifact["current_fee_and_conversion_contract"][
        "v2_collateral_adapter_route"
    ]
    assert route["current_official_python_sdk_address"].lower() == (
        route["actual_conversion"]["event_stakeholder"].lower()
    )
    assert route["actual_conversion"]["index_set"] == 7
    assert route["readme_conflicting_address"].lower() != (
        route["current_official_python_sdk_address"].lower()
    )


def test_exact_conversion_and_current_cost_sensitivity_remain_fail_closed() -> None:
    artifact = _load(ARTIFACT_PATH)
    access = artifact["conversion_access_and_latency"]
    receipt = access["exact_successful_conversion_receipt"]
    sensitivity = access["current_whole_transaction_gas_sensitivity"]

    assert receipt["status"] == 1
    assert receipt["index_set"] == 7
    assert receipt["gas_used_whole_outer_transaction"] == 479446
    assert access["conversion_gas_or_relayer_cost_bound"] is False
    assert access["installed_sdk_method_audit"] == {
        **access["installed_sdk_method_audit"],
        "dedicated_convert_method_matches": 0,
        "selector_literal_convertPositions_matches": 0,
    }
    assert sensitivity["request_count"] == 2
    assert sensitivity["margin_5_pusd_minus_usdt_sensitivity"].startswith("-")
    assert sensitivity["margin_20_pusd_minus_usdt_sensitivity"] == (
        "0.0092570280188902383064000"
    )
    assert sensitivity["adjudication"] == (
        "sensitivity_only_not_cross_stablecoin_after_cost_proof"
    )


def test_current_whole_transaction_gas_sensitivity_recomputes_exactly() -> None:
    artifact = _load(ARTIFACT_PATH)
    sensitivity = artifact["conversion_access_and_latency"][
        "current_whole_transaction_gas_sensitivity"
    ]
    raw_root = (
        ROOT
        / "docs/model-research/action-value/raw/polymarket-negrisk-maker-input-v2"
    )
    gas = _load(raw_root / "current-polygon-gas-station.json")
    ticker = _load(raw_root / "current-binance-polusdt-book-ticker.json")

    gas_cost_pol = (
        Decimal(479446)
        * Decimal(str(gas["standard"]["maxFee"]))
        / Decimal(1_000_000_000)
    )
    gas_cost_usdt = gas_cost_pol * Decimal(ticker["askPrice"])

    assert str(gas_cost_pol) == sensitivity["reused_whole_transaction_gas_cost_pol"]
    assert str(gas_cost_usdt) == (
        sensitivity["reused_whole_transaction_gas_cost_usdt_sensitivity"]
    )
    assert Decimal("0.00740") - gas_cost_usdt == Decimal(
        sensitivity["margin_5_pusd_minus_usdt_sensitivity"]
    )
    assert Decimal("0.02960") - gas_cost_usdt == Decimal(
        sensitivity["margin_20_pusd_minus_usdt_sensitivity"]
    )


def test_every_tracked_source_receipt_reconstructs_exactly() -> None:
    artifact = _load(ARTIFACT_PATH)

    for receipt in artifact["tracked_evidence"]:
        path = ROOT / receipt["path"]
        payload = path.read_bytes()
        assert len(payload) == receipt["bytes"]
        assert hashlib.sha256(payload).hexdigest() == receipt["sha256"]
        if "decompressed_sha256" in receipt:
            decompressed = gzip.decompress(payload)
            assert (
                hashlib.sha256(decompressed).hexdigest()
                == receipt["decompressed_sha256"]
            )
            if "decompressed_bytes" in receipt:
                assert len(decompressed) == receipt["decompressed_bytes"]


def test_prospective_terminal_capture_fails_duration_continuity_and_queue() -> None:
    terminal = _load(TERMINAL_PATH)

    assert terminal["result_sha256"] == EXPECTED_TERMINAL_HASH
    assert _embedded_hash(terminal) == EXPECTED_TERMINAL_HASH
    assert terminal["adjudication"] == {
        **terminal["adjudication"],
        "accepted_edge": False,
        "causally_subsequent_output_unwind_evaluated": False,
        "same_contract_rerun_permitted": False,
    }
    capture = terminal["capture_terminal"]
    assert Decimal(capture["elapsed_seconds"]) < Decimal(capture["planned_seconds"])
    assert capture["source_continuity_admitted"] is False
    assert capture["terminal_error"].startswith("ConnectionClosedError:")
    queue = terminal["queue_censored_input_audit"]
    assert queue["five_share_fill_proved"] is False
    assert queue["twenty_share_fill_proved"] is False
    assert Decimal(queue["quantity_beyond_initial_visible_queue_shares"]) < 5

    for receipt in terminal["tracked_evidence"]:
        payload = (ROOT / receipt["path"]).read_bytes()
        assert len(payload) == receipt["stored_bytes"]
        assert hashlib.sha256(payload).hexdigest() == receipt["stored_sha256"]
        decompressed = gzip.decompress(payload)
        assert len(decompressed) == receipt["decompressed_bytes"]
        assert hashlib.sha256(decompressed).hexdigest() == (
            receipt["decompressed_sha256"]
        )


def test_registry_binds_the_exact_event_scope_without_promoting_the_edge() -> None:
    artifact = _load(ARTIFACT_PATH)
    registry = _load(REGISTRY_PATH)

    assert registry["result_sha256"] == EXPECTED_REGISTRY_HASH
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_HASH
    candidate = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "polymarket_negative_risk_NO_to_YES_converter_recurrence"
    )
    assert candidate["priority_rank"] == 2
    assert candidate["venue_scope"] == (
        "polymarket_event_106981_bitcoin_gold_sp500_fixed_non_augmented_negative_risk"
    )
    assert candidate["canonical_artifacts"][:2] == [
        {
            "path": TERMINAL_PATH.relative_to(ROOT).as_posix(),
            "result_sha256": EXPECTED_TERMINAL_HASH,
        },
        {
            "path": ARTIFACT_PATH.relative_to(ROOT).as_posix(),
            "result_sha256": artifact["result_sha256"],
        },
    ]
    assert candidate["current_status"].startswith(
        "terminally_rejected_under_the_consumed_one_use_prospective_contract"
    )
    assert candidate["retry_trigger"].startswith(
        "materially_new_primary_evidence_that_changes_queue_attribution"
    )
    assert registry["accepted_edge_count"] == 19
    assert artifact["research_decision"]["accepted_edge_count_change"] == 0
    assert (
        artifact["prospective_fill_and_unwind_capture"]["active_result_claim"] is False
    )
