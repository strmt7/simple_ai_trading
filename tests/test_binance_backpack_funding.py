from decimal import Decimal
import json

import pytest

from tools.adjudicate_binance_backpack_funding import (
    HOUR_MS,
    _backpack_series,
    _role_result,
    _timestamp_ms,
    _validate_inventory,
)
from tools.adjudicate_binance_backpack_funding_v2 import (
    _timestamp_ms as _timestamp_ms_v2,
)


def test_timestamp_parser_accepts_documented_string_encodings() -> None:
    assert _timestamp_ms("2026-09-01T08:00:00Z") == 1788249600000
    assert _timestamp_ms("1788249600000") == 1788249600000
    assert _timestamp_ms("1788249600") == 1788249600000


def test_v2_timestamp_parser_accepts_only_strict_naive_utc_hour() -> None:
    assert _timestamp_ms_v2("2026-09-01T08:00:00") == 1788249600000
    with pytest.raises(RuntimeError, match="strict UTC whole-hour"):
        _timestamp_ms_v2("2026-09-01T08:30:00")


def test_backpack_series_requires_complete_hourly_schedule(tmp_path) -> None:
    start_ms = 1788249600000
    path = tmp_path / "funding.json"
    path.write_text(
        json.dumps(
            [
                {
                    "symbol": "BTC_USDC_PERP",
                    "intervalEndTimestamp": str(start_ms + index * HOUR_MS),
                    "fundingRate": "0.0001",
                }
                for index in range(8)
            ]
        ),
        encoding="utf-8",
    )
    series = _backpack_series(path, symbol="BTC_USDC_PERP")
    assert len(series) == 8
    assert sum(series.values(), Decimal(0)) == Decimal("0.0008")

    rows = json.loads(path.read_text(encoding="utf-8"))
    rows.pop(3)
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete"):
        _backpack_series(path, symbol="BTC_USDC_PERP")


def test_inventory_requires_open_visible_usdc_perpetual(tmp_path) -> None:
    row = {
        "symbol": "BTC_USDC_PERP",
        "baseSymbol": "BTC",
        "quoteSymbol": "USDC",
        "marketType": "PERP",
        "orderBookState": "Open",
        "visible": True,
    }
    path = tmp_path / "markets.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    _validate_inventory(path, assets=["BTC"])

    row["orderBookState"] = "Closed"
    path.write_text(json.dumps([row]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="instrument identity"):
        _validate_inventory(path, assets=["BTC"])


def test_role_result_charges_every_frozen_hurdle() -> None:
    contract = {
        "economics": {
            "round_trip_execution_bips": "20",
            "annual_two_leg_capital_hurdle_percent": "10",
            "usdc_usdt_quote_unit_stress_bips": "25",
            "custody_latency_failure_stress_bips": "25",
        },
        "gates": {"minimum_positive_interval_fraction": "0.5"},
    }
    result = _role_result([Decimal("0.01"), Decimal("0.01")], contract)
    assert result["gross_funding_spread_bips"] == "200.00"
    assert Decimal(result["net_after_all_frozen_hurdles_bips"]) < Decimal(130)
    assert result["passes"] is True
