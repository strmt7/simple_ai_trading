from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
CONTRACT = (
    ACTION_VALUE / "binance-commodity-option-perpetual-lower-bound-contract-v1.json"
)
RESULT = (
    ACTION_VALUE
    / "binance-commodity-option-perpetual-lower-bound-v1-2026-08-27.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
IMPLEMENTATION = (
    ROOT / "tools/screen_binance_commodity_option_perpetual_lower_bound.py"
)
CONTRACT_HASH = "a1ecde2ac379d40fba81840cc9adf10dd731f29bd8b4eba030a6e71521158b94"
RESULT_HASH = "3cbc79050473b456e4175239b687b0329bc1c7a66d3530842e524ac4200a0905"
REGISTRY_HASH = "2f3b1dfbe64d7f9ea1787a08ed059a49564f9f24f37aed18519e88355a9713d2"


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


def test_contract_freezes_complete_population_and_costs_before_tickers() -> None:
    contract = _load(CONTRACT)

    assert contract["contract_sha256"] == CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == CONTRACT_HASH
    discovery = contract["discovery_boundary_already_observed"]
    assert discovery["active_TRADFI_OPTIONS_COMMODITY_XAU_XAG_row_count"] == 92
    assert discovery["sorted_symbol_population_sha256"] == (
        "da4c880101f75e987edc143c991d0589aaf72343c3f25e9fb7b4de63b4c1e1dc"
    )
    assert contract["capture"]["maximum_http_requests"] == 7
    assert contract["capture"]["retry_permitted"] is False
    assert contract["economics"]["fixed_nonfunding_stress_bps_of_underlying_notional"] == "33.5"


def test_capture_is_complete_hash_bound_and_action_free() -> None:
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
    assert result["population"]["active_commodity_option_count"] == 92
    assert (
        result["implementation"]["sha256"]
        == hashlib.sha256(IMPLEMENTATION.read_bytes()).hexdigest()
    )


def test_every_quoted_option_has_a_nonpositive_indicative_lower_bound() -> None:
    result = _load(RESULT)
    economics = result["economics"]

    assert economics["row_count"] == 92
    assert economics["positive_ask_count"] == 73
    assert economics["option_ask_quantity_available_count"] == 0
    assert economics["top_level_capacity_pass_count"] == 0
    assert economics["indicative_gross_positive_count_without_option_quantity"] == 0
    assert economics["indicative_after_stress_positive_count_without_option_quantity"] == 0
    assert economics["best_indicative_symbol_without_option_quantity"] == (
        "XAU-260827-4580-C"
    )
    assert Decimal(
        economics["best_indicative_gross_terminal_payoff_floor_bps"]
    ) < 0

    quoted_rows = [row for row in economics["rows"] if row["quoted_positive_ask"]]
    assert len(quoted_rows) == 73
    for row in quoted_rows:
        quantity = Decimal(row["minimum_common_quantity"])
        strike = Decimal(row["strike_price"])
        option_ask = Decimal(row["option_best_ask"])
        perpetual_entry = Decimal(row["perpetual_entry_price"])
        gross = (
            perpetual_entry - strike - option_ask
            if row["option_side"] == "CALL"
            else strike - perpetual_entry - option_ask
        ) * quantity
        assert row["option_ask_quantity_available"] is False
        assert row["top_level_capacity_passes"] is False
        assert Decimal(row["indicative_gross_terminal_payoff_floor_usdt"]) == gross
        assert gross <= 0
        assert row["indicative_after_all_frozen_stress_positive"] is False

    adjudication = result["adjudication"]
    assert adjudication["accepted_edge"] is False
    assert adjudication["profitability_claim"] is False
    assert adjudication["public_stressed_candidate_count"] == 0


def test_registry_terminalizes_only_this_expiry_population() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 17
    assert [row["priority_rank"] for row in registry["prioritized_hypotheses"]] == list(
        range(1, 41)
    )
    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"]
        == "binance_long_commodity_option_opposite_perpetual_terminal_payoff_lower_bound"
    )
    assert hypothesis["retry_trigger"] == (
        "new_listed_XAU_or_XAG_option_expiry_or_material_option_fee_funding_basis_book_or_product_access_change"
    )
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_commodity_option_perpetual_260827_260828_terminal_lower_bound_snapshot"
    )
    assert terminal["canonical_result_sha256"] == RESULT_HASH
