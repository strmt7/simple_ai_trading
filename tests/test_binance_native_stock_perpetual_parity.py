from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION_VALUE / "binance-native-stock-perpetual-parity-contract-v1.json"
RESULT = ACTION_VALUE / "binance-native-stock-perpetual-parity-v1-2026-08-27.json"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
IMPLEMENTATION = ROOT / "tools/screen_binance_native_stock_perpetual_parity.py"
NEW_TEMPLATE = (
    ACTION_VALUE
    / "binance-native-stock-new-tradfi-perpetual-template-v1-2026-08-29.json"
)
NEW_CONTRACT = (
    ACTION_VALUE
    / "binance-native-stock-new-tradfi-perpetual-contract-v1-2026-08-29.json"
)
NEW_RESULT = (
    ACTION_VALUE / "binance-native-stock-new-tradfi-perpetual-result-v1-2026-08-29.json"
)
NEW_IMPLEMENTATION = ROOT / "tools/screen_binance_native_stock_new_tradfi_perpetuals.py"
NEW_RAW = ROOT / "data/binance-native-stock-new-tradfi-perpetual-v1/raw"
NEW_JOURNAL = (
    ROOT / "data/binance-native-stock-new-tradfi-perpetual-v1/request-journal.jsonl"
)
CONTRACT_HASH = "ec5d4855c69d3afa461838b674530936a07e646394540a1a2b30ae3ddaf77db1"
RESULT_HASH = "2776ff86fddf78e7e87860c6b9500cb237fce5af908a4840d351ae0cc2eff930"
REGISTRY_HASH = "98714cb8665d2132cb53670f09e73d11816cdbb7a9c3bc221dce4db4f865f98d"
NEW_TEMPLATE_HASH = "0b187e4db1a4d3cdb654da83ef83c61505c0822d92cee897cca90898cfe7c5f9"
NEW_CONTRACT_HASH = "37a3424645103a351c232ec7bf7c6e2cb4912be1e60bf136bc8cc170644f9adf"
NEW_RESULT_HASH = "d8b87863ea750386f1074daef988443a12390f0a36cfecc538765e00bded9a9f"


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


def test_contract_freezes_complete_same_usdt_quote_overlap_before_capture() -> None:
    contract = _load(CONTRACT)

    assert contract["contract_sha256"] == CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == CONTRACT_HASH
    universe = contract["frozen_universe"]
    assert len(universe) == 14
    assert len({row["ticker"] for row in universe}) == 14
    assert all(row["perpetual_symbol"] == f"{row['ticker']}USDT" for row in universe)
    assert contract["capture"]["maximum_public_GET_requests"] == 28
    assert contract["capture"]["maximum_public_websocket_connections"] == 14
    assert contract["capture"]["retry_permitted"] is False
    assert contract["discovery_boundary"]["excluded_cross_quote"] == {
        "ticker": "SPCX",
        "perpetual_symbol": "SPCXUSD1",
        "reason": (
            "requires_a_distinct_USD1_to_USDC_executable_conversion_path_and_is_"
            "not_part_of_the_frozen_same_USDT_quote_identity"
        ),
    }


def test_capture_is_hash_bound_public_only_and_preserves_incompleteness() -> None:
    result = _load(RESULT)

    assert result["result_sha256"] == RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == RESULT_HASH
    assert result["contract"]["sha256"] == CONTRACT_HASH
    assert (
        result["implementation"]["sha256"]
        == hashlib.sha256(IMPLEMENTATION.read_bytes()).hexdigest()
    )
    assert result["authority"] == {
        "public_unauthenticated_GET_requests": 26,
        "public_unauthenticated_websocket_connections": 14,
        "authenticated_requests": 0,
        "account_state_accessed": False,
        "orders_quotes_transfers_disclaimer_or_wallet_actions": 0,
        "paper_or_live_trading_authority": False,
    }
    capture = result["capture"]
    assert capture["complete_population"] is False
    assert capture["complete_row_count"] == 13
    assert capture["incomplete_row_count"] == 1
    assert capture["errors"] == [
        {
            "ticker": "KLAC",
            "perpetual_symbol": "KLACUSDT",
            "error_type": "TimeoutError",
            "error": "",
        }
    ]
    assert capture["retained_source_count"] == 39


def test_observed_rows_fail_stress_or_one_share_capacity() -> None:
    result = _load(RESULT)
    economics = result["economics"]

    assert economics["row_count"] == 13
    assert economics["after_labeled_stress_positive_count"] == 0
    assert economics["best_ticker"] == "NBIS"
    assert Decimal(economics["best_gross_entry_headroom_bps"]) < Decimal("30")
    rows = economics["rows"]
    best_capacity_valid = max(
        (row for row in rows if row["all_three_top_level_capacities_pass"]),
        key=lambda row: Decimal(row["gross_entry_headroom_bps"]),
    )
    assert best_capacity_valid["ticker"] == "SNDK"
    assert Decimal(best_capacity_valid["gross_entry_headroom_bps"]) == Decimal(
        "14.64029978610898480329251573"
    )
    for row in rows:
        stock_ask = Decimal(row["native_stock_best_ask_USD"])
        perpetual_bid = Decimal(row["perpetual_best_bid_USDT"])
        fx_ask = Decimal(row["USDCUSDT_best_ask"])
        gross = perpetual_bid / fx_ask - stock_ask
        stress = stock_ask * Decimal("30") / Decimal("10000")
        assert Decimal(row["gross_entry_headroom_USDC"]) == gross
        assert Decimal(row["labeled_30_bps_stress_USDC"]) == stress
        assert Decimal(row["after_labeled_stress_USDC"]) == gross - stress
        assert row["after_labeled_stress_positive"] is False

    adjudication = result["adjudication"]
    assert adjudication["status"] == (
        "incomplete_public_quote_population_no_economic_adjudication"
    )
    assert adjudication["accepted_edge"] is False
    assert adjudication["public_after_cost_profit_floor_USDC"] == "0"


def test_new_five_ticker_delta_was_frozen_before_public_access() -> None:
    template = _load(NEW_TEMPLATE)
    contract = _load(NEW_CONTRACT)

    assert hashlib.sha256(NEW_TEMPLATE.read_bytes()).hexdigest() == NEW_TEMPLATE_HASH
    assert contract["contract_sha256"] == NEW_CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == NEW_CONTRACT_HASH
    assert contract["template"] == {
        "path": (
            "docs/model-research/action-value/"
            "binance-native-stock-new-tradfi-perpetual-template-v1-2026-08-29.json"
        ),
        "sha256": NEW_TEMPLATE_HASH,
    }
    assert contract["frozen_universe"] == template["frozen_universe"]
    assert [row["ticker"] for row in contract["frozen_universe"]] == [
        "TEM",
        "MRK",
        "IONQ",
        "MARA",
        "PDD",
    ]
    frozen_at = datetime.fromisoformat(contract["frozen_at_utc"].replace("Z", "+00:00"))
    assert frozen_at.tzinfo is not None
    assert frozen_at <= datetime.now(timezone.utc)
    assert contract["capture"]["retry_permitted"] is False
    assert (
        contract["capture"]["downstream_requests_for_absent_or_invalid_stock_quote"]
        == 0
    )


def test_new_five_ticker_delta_stops_before_downstream_books() -> None:
    result = _load(NEW_RESULT)

    assert result["result_sha256"] == NEW_RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == NEW_RESULT_HASH
    assert result["contract"]["sha256"] == NEW_CONTRACT_HASH
    assert (
        result["implementation"]["sha256"]
        == hashlib.sha256(NEW_IMPLEMENTATION.read_bytes()).hexdigest()
    )
    assert (
        result["implementation"]["capture_helper_sha256"]
        == hashlib.sha256(IMPLEMENTATION.read_bytes()).hexdigest()
    )
    assert result["authority"] == {
        "public_unauthenticated_GET_requests": 0,
        "public_unauthenticated_websocket_connections": 5,
        "authenticated_requests": 0,
        "account_state_accessed": False,
        "orders_quotes_transfers_disclaimer_or_wallet_actions": 0,
        "paper_or_live_trading_authority": False,
    }
    capture = result["capture"]
    assert capture["complete_row_count"] == 0
    assert capture["incomplete_row_count"] == 5
    assert [row["ticker"] for row in capture["errors"]] == [
        "TEM",
        "MRK",
        "IONQ",
        "MARA",
        "PDD",
    ]
    assert all(row["error_type"] == "TimeoutError" for row in capture["errors"])
    assert capture["retained_source_count"] == 0
    assert capture["raw_response_bytes"] == 0
    assert not NEW_RAW.exists() or list(NEW_RAW.iterdir()) == []
    assert NEW_JOURNAL.read_bytes() == b""
    assert result["adjudication"]["status"] == (
        "zero_new_perpetual_tickers_have_a_live_native_stock_quote"
    )
    assert result["adjudication"]["accepted_edge"] is False
    assert result["adjudication"]["public_after_cost_profit_floor_USDC"] == "0"


def test_registry_retires_stale_recoveries_and_consumed_delta() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 21
    hypotheses = registry["prioritized_hypotheses"]
    assert [row["priority_rank"] for row in hypotheses] == list(range(1, 45))
    hypothesis = next(
        row
        for row in hypotheses
        if row["mechanism"] == "binance_native_stock_USDC_TradFi_perpetual_USDT_parity"
    )
    assert hypothesis["retry_trigger"].startswith(
        "new_official_native_stock_or_TradFi_perpetual_listing"
    )
    assert "do_not_rerun_the_consumed_five_ticker_delta" in hypothesis["next_action"]
    assert "or_KLAC_recovery" in hypothesis["next_action"]
    assert hypothesis["canonical_artifacts"][-2:] == [
        {
            "path": (
                "docs/model-research/action-value/"
                "binance-native-stock-new-tradfi-perpetual-contract-v1-2026-08-29.json"
            ),
            "result_sha256": NEW_CONTRACT_HASH,
        },
        {
            "path": (
                "docs/model-research/action-value/"
                "binance-native-stock-new-tradfi-perpetual-result-v1-2026-08-29.json"
            ),
            "result_sha256": NEW_RESULT_HASH,
        },
    ]
