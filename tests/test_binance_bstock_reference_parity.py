from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-bstock-reference-parity-v1-2026-08-26.json"
)
TOOL_PATH = ROOT / "tools" / "screen_binance_bstock_reference_parity.py"
EXPECTED_RESULT_HASH = (
    "73fec22cc61fc8be0c792a78c0340fcb163b9bf7862708796d50521a9c44a8ac"
)
EXPECTED_TOOL_HASH = "93e8f3364629eb99ff04dceabedfa0111ca57692c7cd921aada8300be656ed36"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _load() -> dict[str, object]:
    return json.loads(ARTIFACT_PATH.read_bytes())


def test_bstock_artifact_and_tool_are_source_bound() -> None:
    artifact = _load()
    body = dict(artifact)
    assert body.pop("result_sha256") == EXPECTED_RESULT_HASH
    assert hashlib.sha256(_canonical_json(body).encode("ascii")).hexdigest() == (
        EXPECTED_RESULT_HASH
    )
    assert artifact["implementation"]["sha256"] == EXPECTED_TOOL_HASH
    assert hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest() == EXPECTED_TOOL_HASH
    assert artifact["authority"] == {
        "conversions_requested": False,
        "credentials_used": False,
        "funds_used": False,
        "orders_placed": False,
        "trading_authority": False,
    }


def test_snxx_is_a_positive_reference_candidate_but_not_execution_evidence() -> None:
    artifact = _load()
    assert artifact["scope"]["screened_trading_bstock_count"] == 67
    assert (
        artifact["scope"]["research_only_outside_current_btc_eth_sol_execution_scope"]
        is True
    )
    selected = artifact["selected_candidate"]
    assert selected["symbol"] == "SNXXBUSDT"
    assert Decimal(selected["multiplier"]) == Decimal(1)
    assert selected["reference_calculation_type"] == "EXTERNAL"
    assert selected["external_calculation_id"] == 2
    assert selected["reference_price"] == selected["reference_price_after"]
    assert [row["target_usdt"] for row in selected["size_results"]] == [
        "1000",
        "5000",
    ]
    assert all(
        row["after_labeled_cost_sensitivity_positive"] is True
        for row in selected["size_results"]
    )
    assert selected["historical_diagnostic"]["role"] == (
        "non_executable_persistence_diagnostic_only"
    )
    assert artifact["verdict"]["accepted_edge"] is False
    assert artifact["verdict"]["deployment_ready"] is False


def test_four_direction_neutral_funding_candidates_clear_complete_month_gate() -> None:
    diagnostic = _load()["delta_neutral_funding_carry_diagnostic"]
    assert diagnostic["market_direction_forecast_required"] is False
    assert diagnostic["candidate_count"] == 4
    assert diagnostic["all_candidates_clear_every_complete_inner_month"] is True
    cases = {row["ticker"]: row for row in diagnostic["candidates"]}
    assert set(cases) == {"DRAM", "MRVL", "MU", "SNDK"}
    expected_worst_month = {
        "DRAM": Decimal("86.2101"),
        "MRVL": Decimal("61.1628"),
        "MU": Decimal("79.3878"),
        "SNDK": Decimal("140.3408"),
    }
    for ticker, row in cases.items():
        months = row["complete_inner_months"]
        assert len(months) >= 2
        assert all(
            month["clears_labeled_round_trip_cost_sensitivity"] is True
            for month in months
        )
        assert (
            min(Decimal(month["short_funding_received_bips"]) for month in months)
            == expected_worst_month[ticker]
        )
        assert Decimal(row["total_short_funding_received_bips"]) > 300


def test_bstock_candidate_remains_account_cost_and_scope_gated() -> None:
    artifact = _load()
    assert (
        "synchronous executable Binance Stocks sale quote rather than an external reference price"
        in artifact["blocking_evidence"]
    )
    assert (
        "explicit scope expansion beyond BTC ETH and SOL before any execution work"
        in artifact["blocking_evidence"]
    )
    assert artifact["delta_neutral_funding_carry_diagnostic"]["adjudication"] == (
        "promising_persistent_gross_candidate_not_after_cost_or_in_scope"
    )
