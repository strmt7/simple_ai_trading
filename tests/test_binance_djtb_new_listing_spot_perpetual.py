from __future__ import annotations

from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION_VALUE / "binance-djtb-new-listing-spot-perpetual-contract-v1.json"
CORRECTION = (
    ACTION_VALUE / "binance-djtb-new-listing-contract-timestamp-correction-v1.json"
)
RESULT = ACTION_VALUE / "binance-djtb-new-listing-spot-perpetual-v1-2026-08-27.json"
DELTA_CONTRACT = (
    ACTION_VALUE / "binance-bstock-inventory-delta-contract-v1-2026-08-29.json"
)
DELTA_RESULT = ACTION_VALUE / "binance-bstock-inventory-delta-result-v1-2026-08-29.json"
DELTA_RAW = ROOT / "data/binance-bstock-inventory-delta-v1/raw/bstock_inventory.json"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
IMPLEMENTATION = ROOT / "tools/screen_binance_djtb_new_listing_spot_perpetual.py"
CONTRACT_HASH = "087b06191378fa949ac62340d9f3e5c625aa31feff71bba3c7fd20cae1155ee8"
CORRECTION_HASH = "692ee1f9a0374726b0adeccbe4fcd710c9c3a86d49b4b33210a8c34b2c79c4d3"
RESULT_HASH = "2b85a6eca339799a6eb07ba48069e3a2943d97116a9320ce20400d260227e1be"
DELTA_CONTRACT_HASH = "9755578775ee4082c394a4e6ae96b8ec0fa1b7946a2d4dc92383be6f562db0f8"
DELTA_RESULT_HASH = "c343614b061e19ba32813b911d984630d8260cb3a46a1216389a63609a75925c"
DELTA_RAW_HASH = "87aa11d459f9babcba9837743ab616fef4c066b20e209524b43ee383429cde3d"
REGISTRY_HASH = json.loads(
    (ROOT / "docs/model-research/structural-edge-priority-registry-v1.json").read_text(
        encoding="utf-8"
    )
)["result_sha256"]


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_contract_is_new_symbol_only_frozen_and_action_free() -> None:
    contract = _load(CONTRACT)

    assert contract["contract_sha256"] == CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == CONTRACT_HASH
    boundary = contract["material_new_listing_boundary"]
    assert boundary["spot_pair"] == "DJTBUSDT"
    assert boundary["perpetual_symbol"] == "DJTUSDT"
    identity = contract["discovery_identity_already_observed"]["required_row"]
    assert identity["ticker"] == "DJT"
    assert identity["multiplier"] == "1"
    assert contract["capture"]["maximum_http_requests"] == 7
    assert contract["capture"]["retry_permitted"] is False
    assert contract["economics"]["fixed_stress_bps_of_spot_cost"] == "50"


def test_clerical_timestamp_error_is_preserved_and_explicitly_corrected() -> None:
    correction = _load(CORRECTION)

    assert correction["result_sha256"] == CORRECTION_HASH
    assert _canonical_hash(correction, "result_sha256") == CORRECTION_HASH
    assert correction["original_contract"]["sha256"] == CONTRACT_HASH
    assert correction["correction"]["post_capture_economic_amendment"] is False
    effective = datetime.fromisoformat(
        correction["correction"]["effective_frozen_at_utc"].replace("Z", "+00:00")
    )
    started = datetime.fromisoformat(
        correction["capture"]["capture_started_at_utc"].replace("Z", "+00:00")
    )
    assert effective < started


def test_capture_is_complete_hash_bound_and_has_only_new_pair() -> None:
    result = _load(RESULT)

    assert result["result_sha256"] == RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == RESULT_HASH
    assert result["contract"]["sha256"] == CONTRACT_HASH
    assert result["authority"] == {
        "public_unauthenticated_GET_requests": 7,
        "authenticated_requests": 0,
        "account_state_accessed": False,
        "orders_transfers_conversions_wallet_or_margin_actions": 0,
        "paper_or_live_trading_authority": False,
    }
    assert result["capture"]["request_count"] == 7
    assert result["capture"]["all_status_codes_200"] is True
    assert result["identity"]["spot_symbol"] == "DJTBUSDT"
    assert result["identity"]["futures_symbol"] == "DJTUSDT"
    assert result["identity"]["exact_multiplier"] == "1"
    assert (
        result["implementation"]["sha256"]
        == hashlib.sha256(IMPLEMENTATION.read_bytes()).hexdigest()
    )


def test_all_capacity_valid_sizes_are_negative_before_and_after_stress() -> None:
    result = _load(RESULT)
    economics = result["economics"]

    assert result["funding"]["history_event_count"] == 6
    assert result["funding"]["eight_event_persistence_available"] is False
    assert Decimal(result["funding"]["adverse_negative_rate_sum"]) == Decimal(
        "0.00263045"
    )
    assert economics["row_count"] == 3
    assert economics["all_targets_capacity_valid"] is True
    assert economics["after_all_frozen_stress_positive_count"] == 0
    for row in economics["rows"]:
        quantity = Decimal(row["common_quantity"])
        spot_cost = Decimal(row["spot_ask_cost_usdt"])
        futures_proceeds = Decimal(row["futures_bid_proceeds_usdt"])
        gross = futures_proceeds - spot_cost
        assert quantity > 0
        assert row["top_level_depth_capacity_passes"] is True
        assert Decimal(row["gross_entry_headroom_usdt"]) == gross
        assert gross < 0
        fixed = spot_cost * Decimal("50") / Decimal("10000")
        funding = spot_cost * Decimal("0.00263045")
        assert Decimal(row["fixed_stress_usdt"]) == fixed
        assert Decimal(row["adverse_funding_stress_usdt"]) == funding
        assert Decimal(row["after_all_frozen_stress_usdt"]) == gross - fixed - funding
        assert row["after_all_frozen_stress_positive"] is False

    assert result["adjudication"]["accepted_edge"] is False
    assert result["adjudication"]["profitability_claim"] is False


def test_inventory_delta_is_frozen_unchanged_and_stops_before_futures() -> None:
    contract = _load(DELTA_CONTRACT)
    result = _load(DELTA_RESULT)

    assert contract["contract_sha256"] == DELTA_CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == DELTA_CONTRACT_HASH
    assert contract["maximum_requests"] == 2
    assert contract["retry_policy"] == "one_attempt_per_exact_request_no_retry"
    assert result["result_sha256"] == DELTA_RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == DELTA_RESULT_HASH
    assert result["baseline_row_count"] == result["current_row_count"] == 68
    assert result["new_tickers"] == []
    assert result["removed_tickers"] == []
    assert result["matching_unscreened_pairs"] == []
    assert result["conditional_request_executed"] is False
    assert result["request_count"] == 1
    assert result["public_prefilter_trigger_satisfied"] is False
    assert hashlib.sha256(DELTA_RAW.read_bytes()).hexdigest() == DELTA_RAW_HASH


def test_registry_updates_existing_bstock_family_and_terminalizes_snapshot() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    ranks = [row["priority_rank"] for row in registry["prioritized_hypotheses"]]
    assert ranks == list(range(1, len(ranks) + 1))
    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"]
        == "bstock_reference_conversion_and_delta_neutral_perpetual_funding"
    )
    assert "new_official_exact_multiplier_bStock_listing" in hypothesis["retry_trigger"]
    assert any(
        "polling_the_byte_identical_68_row_bStock_inventory" in shortcut
        for shortcut in hypothesis["prohibited_shortcuts"]
    )
    assert any(
        artifact["result_sha256"] == RESULT_HASH
        for artifact in hypothesis["canonical_artifacts"]
    )
    assert any(
        artifact["result_sha256"] == DELTA_RESULT_HASH
        for artifact in hypothesis["canonical_artifacts"]
    )
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_DJTBUSDT_DJTUSDT_first_new_listing_spot_perpetual_snapshot"
    )
    assert terminal["canonical_result_sha256"] == RESULT_HASH
