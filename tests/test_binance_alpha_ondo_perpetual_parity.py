from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION_VALUE / "binance-alpha-ondo-perpetual-parity-contract-v1.json"
RESULT = ACTION_VALUE / "binance-alpha-ondo-perpetual-parity-v1-2026-08-27.json"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
IMPLEMENTATION = ROOT / "tools/screen_binance_alpha_ondo_perpetual_parity.py"
CONTRACT_HASH = "2f08c6b0a8509d9d51db7716d5dde499c3a1937b68eafe77b2970e4da8311b59"
RESULT_HASH = "a3d474e9010b92c9454a5bc04b5a7f586656c8bc5842cecc61baaa508c2d8bc3"
REGISTRY_HASH = "ec41ae27eb0699809acabc273620059516a35c09ec6f7cf33520eecbf19ea78e"


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


def test_frozen_contract_covers_the_complete_four_symbol_population() -> None:
    contract = _load(CONTRACT)

    assert contract["contract_sha256"] == CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == CONTRACT_HASH
    assert contract["capture"]["maximum_http_requests"] == 7
    assert contract["capture"]["retry_permitted"] is False
    assert [row["ticker"] for row in contract["frozen_universe"]] == [
        "CRCL",
        "TSLA",
        "COIN",
        "MSTR",
    ]
    assert len({row["contract_address"] for row in contract["frozen_universe"]}) == 4


def test_public_capture_is_hash_bound_and_action_free() -> None:
    result = _load(RESULT)

    assert result["result_sha256"] == RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == RESULT_HASH
    assert result["contract"]["sha256"] == CONTRACT_HASH
    assert result["authority"] == {
        "public_unauthenticated_GET_requests": 7,
        "authenticated_requests": 0,
        "account_state_accessed": False,
        "orders_quotes_transfers_or_wallet_actions": 0,
        "paper_or_live_trading_authority": False,
    }
    capture = result["capture"]
    assert capture["request_count"] == len(capture["sources"]) == 7
    assert capture["all_status_codes_200"] is True
    assert all(source["method"] == "GET" for source in capture["sources"])
    assert (
        result["implementation"]["sha256"]
        == hashlib.sha256(IMPLEMENTATION.read_bytes()).hexdigest()
    )


def test_every_exact_book_row_fails_the_frozen_stress() -> None:
    result = _load(RESULT)
    economics = result["economics"]

    assert economics["row_count"] == 4
    assert economics["gross_positive_count"] == 3
    assert economics["after_labeled_stress_positive_count"] == 0
    assert economics["best_ticker"] == "MSTR"
    assert Decimal(economics["best_gross_entry_headroom_bps"]) < Decimal("20")
    for row in economics["rows"]:
        quantity = Decimal(row["minimum_common_quantity"])
        alpha_ask = Decimal(row["alpha_best_ask"])
        perpetual_bid = Decimal(row["perpetual_best_bid"])
        alpha_cost = alpha_ask * quantity
        gross = (perpetual_bid - alpha_ask) * quantity
        stress = alpha_cost * Decimal("20") / Decimal("10000")
        assert row["top_level_capacity_passes"] is True
        assert Decimal(row["alpha_cost_usdt"]) == alpha_cost
        assert Decimal(row["gross_entry_headroom_usdt"]) == gross
        assert Decimal(row["labeled_20_bps_stress_usdt"]) == stress
        assert Decimal(row["after_labeled_stress_usdt"]) == gross - stress
        assert row["after_labeled_stress_positive"] is False

    adjudication = result["adjudication"]
    assert adjudication["accepted_edge"] is False
    assert adjudication["profitability_claim"] is False
    assert adjudication["public_after_cost_profit_floor_usdt"] == "0"


def test_registry_terminalizes_only_the_current_snapshot() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 19
    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"]
        == "binance_Ondo_bStock_stock_perpetual_exact_multiplier_wrapper_parity"
    )
    assert hypothesis["retry_trigger"] == (
        "material_Binance_Alpha_fee_execution_or_book_architecture_change_"
        "capable_of_exceeding_the_frozen_20_bips_pre_account_stress"
    )
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_alpha_ondo_perpetual_current_four_contract_book_snapshot"
    )
    assert terminal["canonical_result_sha256"] == RESULT_HASH
