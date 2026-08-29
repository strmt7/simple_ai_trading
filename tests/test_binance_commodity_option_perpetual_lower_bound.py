from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

import pytest

from tools.check_binance_commodity_option_expiry_delta import main as run_delta_main


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
CONTRACT = (
    ACTION_VALUE / "binance-commodity-option-perpetual-lower-bound-contract-v1.json"
)
RESULT = (
    ACTION_VALUE / "binance-commodity-option-perpetual-lower-bound-v1-2026-08-27.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
IMPLEMENTATION = ROOT / "tools/screen_binance_commodity_option_perpetual_lower_bound.py"
DELTA_IMPLEMENTATION = ROOT / "tools/check_binance_commodity_option_expiry_delta.py"
DELTA_CONTRACT = (
    ACTION_VALUE / "binance-commodity-option-expiry-delta-contract-v1-2026-08-29.json"
)
DELTA_RESULT = (
    ACTION_VALUE / "binance-commodity-option-expiry-delta-result-v1-2026-08-29.json"
)
DELTA_ADJUDICATION = (
    ACTION_VALUE
    / "binance-commodity-option-expiry-delta-timestamp-adjudication-v1-2026-08-29.json"
)
DELTA_JOURNAL = (
    ROOT / "data/binance-commodity-option-expiry-delta-v1/request-journal.jsonl"
)
DELTA_RAW = (
    ROOT / "data/binance-commodity-option-expiry-delta-v1/raw/options-exchange-info.raw"
)
CONTRACT_HASH = "a1ecde2ac379d40fba81840cc9adf10dd731f29bd8b4eba030a6e71521158b94"
RESULT_HASH = "3cbc79050473b456e4175239b687b0329bc1c7a66d3530842e524ac4200a0905"
REGISTRY_HASH = "6062ef4cb774983d86d7edd5dad7adcaafa31a8202d37ec777e12fc33028d157"


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
    assert (
        contract["economics"]["fixed_nonfunding_stress_bps_of_underlying_notional"]
        == "33.5"
    )


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
    assert (
        economics["indicative_after_stress_positive_count_without_option_quantity"] == 0
    )
    assert economics["best_indicative_symbol_without_option_quantity"] == (
        "XAU-260827-4580-C"
    )
    assert Decimal(economics["best_indicative_gross_terminal_payoff_floor_bps"]) < 0

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
    assert registry["accepted_edge_count"] == 19
    assert [row["priority_rank"] for row in registry["prioritized_hypotheses"]] == list(
        range(1, 43)
    )
    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"]
        == "binance_long_commodity_option_opposite_perpetual_terminal_payoff_lower_bound"
    )
    assert hypothesis["retry_trigger"] == (
        "new_official_XAU_or_XAG_commodity_option_listing_or_relisting_announcement_after_2026_08_29T13_58_50_782Z_or_material_option_fee_funding_basis_book_or_product_access_change"
    )
    assert hypothesis["canonical_artifacts"][-3:] == [
        {
            "path": DELTA_CONTRACT.relative_to(ROOT).as_posix(),
            "result_sha256": "a56adbe108d92f8c63dac92825305e17f421ef4a2fddc4232a49fa07b336fc37",
        },
        {
            "path": DELTA_RESULT.relative_to(ROOT).as_posix(),
            "result_sha256": "42e7fc2fb8e999f948b2c219d61462ee2af7dbc1477cc4acc68e324d4dea5f1d",
        },
        {
            "path": DELTA_ADJUDICATION.relative_to(ROOT).as_posix(),
            "result_sha256": "072d6b83c90a71a50bcf36cb310f0c933f4dfdc1b5e50bb3b7672a97f960f5ba",
        },
    ]
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_commodity_option_perpetual_260827_260828_terminal_lower_bound_snapshot"
    )
    assert terminal["canonical_result_sha256"] == RESULT_HASH


def test_one_request_expiry_delta_closes_trigger_without_downstream_market_data() -> (
    None
):
    contract = _load(DELTA_CONTRACT)
    result = _load(DELTA_RESULT)
    receipt = json.loads(DELTA_JOURNAL.read_text(encoding="ascii"))

    assert hashlib.sha256(DELTA_CONTRACT.read_bytes()).hexdigest() == (
        "a56adbe108d92f8c63dac92825305e17f421ef4a2fddc4232a49fa07b336fc37"
    )
    assert result["result_sha256"] == (
        "42e7fc2fb8e999f948b2c219d61462ee2af7dbc1477cc4acc68e324d4dea5f1d"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert result["authority"]["public_unauthenticated_GET_requests"] == 1
    assert result["authority"]["authenticated_requests"] == 0
    assert result["authority"]["orders_quotes_transfers_or_wallet_actions"] == 0
    assert result["population"]["active_commodity_option_count"] == 0
    assert result["population"]["new_expiry_date_ms"] == []
    assert result["population"]["removed_expiry_date_ms"] == [
        1787864400000,
        1787950800000,
    ]
    assert result["adjudication"]["new_listed_expiry_trigger_satisfied"] is False
    assert (
        receipt["response_sha256"] == hashlib.sha256(DELTA_RAW.read_bytes()).hexdigest()
    )
    assert receipt["response_sha256"] == (
        "abe8ead861500779a43c501a883d1ce5d2827a8f671161de2d422abf89489841"
    )
    assert contract["capture"]["maximum_http_requests"] == 1
    assert contract["capture"]["retry_permitted"] is False


def test_timestamp_metadata_error_is_preserved_and_future_contracts_fail_pre_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adjudication = _load(DELTA_ADJUDICATION)
    assert hashlib.sha256(DELTA_ADJUDICATION.read_bytes()).hexdigest() == (
        "072d6b83c90a71a50bcf36cb310f0c933f4dfdc1b5e50bb3b7672a97f960f5ba"
    )
    assert adjudication["adjudication"]["literal_timestamp_valid"] is False
    assert adjudication["adjudication"]["contract_mutated_after_access"] is False
    assert adjudication["adjudication"]["public_request_repeated"] is False

    future_contract = tmp_path / "contract.json"
    future_contract.write_text(
        json.dumps(
            {
                "schema_version": "binance-commodity-option-expiry-delta-contract-v1",
                "status": "frozen_before_one_public_inventory_request",
                "frozen_at_utc": "2999-01-01T00:00:00Z",
                "capture": {"endpoint": "https://example.invalid/exchangeInfo"},
            }
        ),
        encoding="ascii",
    )
    monkeypatch.setattr(
        "tools.check_binance_commodity_option_expiry_delta.requests.get",
        lambda *args, **kwargs: pytest.fail("HTTP must not run for future freeze time"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(DELTA_IMPLEMENTATION),
            "--contract",
            str(future_contract),
            "--raw-dir",
            str(tmp_path / "raw"),
            "--journal",
            str(tmp_path / "journal.jsonl"),
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    with pytest.raises(ValueError, match="later than request start"):
        run_delta_main()
