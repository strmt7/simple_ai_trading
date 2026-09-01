from decimal import Decimal
import json
from pathlib import Path

import pytest

from tools.adjudicate_binance_paradex_funding import (
    HOUR_MS,
    _paradex_series,
    _role_result,
    _validate_inventory,
)
from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import _canonical_hash


ROOT = Path(__file__).resolve().parents[1]


def test_paradex_series_normalizes_eight_hour_rate_to_hourly(tmp_path) -> None:
    path = tmp_path / "funding.json"
    path.write_text(
        json.dumps(
            {
                "next": None,
                "results": [
                    {
                        "created_at": index * HOUR_MS,
                        "funding_period_hours": 8,
                        "funding_rate_8h": "0.0008",
                        "market": "BTC-USD-PERP",
                    }
                    for index in range(8)
                ],
            }
        ),
        encoding="utf-8",
    )

    series = _paradex_series(path, market="BTC-USD-PERP")

    assert len(series) == 8
    assert sum(series.values(), Decimal(0)) == Decimal("0.0008")


def test_paradex_series_rejects_pagination_or_schedule_gap(tmp_path) -> None:
    path = tmp_path / "funding.json"
    path.write_text(
        json.dumps(
            {
                "next": "cursor",
                "results": [
                    {
                        "created_at": 0,
                        "funding_period_hours": 8,
                        "funding_rate_8h": "0",
                        "market": "BTC-USD-PERP",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="envelope"):
        _paradex_series(path, market="BTC-USD-PERP")

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["next"] = None
    payload["results"].append({**payload["results"][0], "created_at": 2 * HOUR_MS})
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete"):
        _paradex_series(path, market="BTC-USD-PERP")


def test_inventory_requires_exact_usdc_settled_perpetual(tmp_path) -> None:
    row = {
        "asset_kind": "PERP",
        "base_currency": "BTC",
        "expiry_at": 0,
        "funding_period_hours": 8,
        "open_at": 0,
        "quote_currency": "USD",
        "settlement_currency": "USDC",
        "symbol": "BTC-USD-PERP",
        "trading_mode": "STANDARD",
    }
    path = tmp_path / "markets.json"
    path.write_text(json.dumps({"results": [row]}), encoding="utf-8")
    _validate_inventory(path, assets=["BTC"], first_bucket_ms=1)

    row["settlement_currency"] = "USDT"
    path.write_text(json.dumps({"results": [row]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="instrument identity"):
        _validate_inventory(path, assets=["BTC"], first_bucket_ms=1)


def test_role_result_charges_every_frozen_hurdle() -> None:
    contract = {
        "economics": {
            "round_trip_execution_bips": "20",
            "annual_two_leg_capital_hurdle_percent": "10",
            "usdc_usdt_quote_unit_stress_bips": "25",
            "custody_latency_failure_stress_bips": "25",
            "continuous_sampling_stress_bips": "10",
        },
        "gates": {"minimum_positive_interval_fraction": "0.5"},
    }
    result = _role_result([Decimal("0.01"), Decimal("0.01")], contract)

    assert result["gross_funding_spread_bips"] == "200.00"
    assert Decimal(result["net_after_all_frozen_hurdles_bips"]) < Decimal(120)
    assert result["passes"] is True


def test_terminal_source_gate_is_hash_bound_and_registered() -> None:
    terminal_path = (
        ROOT
        / "docs/model-research/action-value/binance-paradex-btc-eth-sol-funding-terminal-v1-2026-09-01.json"
    )
    terminal = json.loads(terminal_path.read_bytes())
    assert _canonical_hash(terminal, "result_sha256") == terminal["result_sha256"]
    assert terminal["adjudication"]["funding_values_viewed_or_used"] is False
    assert all(
        source.get("continuation_cursor_present") is True
        for source in terminal["source_captures"]
        if source["name"].endswith("_funding")
    )

    registry = json.loads(
        (
            ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
        ).read_bytes()
    )
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    assert len(registry["prioritized_hypotheses"]) == 59
    assert len(registry["terminal_do_not_repeat"]) == 177
    assert (
        registry["prioritized_hypotheses"][-1]["canonical_artifacts"][0][
            "result_sha256"
        ]
        == terminal["result_sha256"]
    )
