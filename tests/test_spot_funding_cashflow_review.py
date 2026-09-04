from decimal import Decimal as D
import json
from pathlib import Path

import pytest

from tools.verify_spot_funding_cashflow_publication import published_bytes, verify

from tools.review_spot_funding_cashflows import (
    canonical,
    cashflow_metrics,
    digest,
    normalize_history,
    run,
)


def test_weighted_cash_can_reverse_unweighted_rate_sign():
    rows = [
        {"rate": D("0.0011"), "mark": D(100)},
        {"rate": D("-0.001"), "mark": D(120)},
    ]
    result = cashflow_metrics(rows, D(100), D(2), D(0), D(2), D(0))
    assert D(result["original_unit_weight_funding_bps"]) == 1
    assert D(result["fixed_base_funding_bps_at_reference"]) == -1
    assert D(result["funding_cash_quote_per_base"]) == D("-0.01")


def test_fee_and_capital_break_even_are_separate_from_cash():
    rows = [{"rate": D("0.01"), "mark": D(100)}]
    result = cashflow_metrics(rows, D(100), D(365), D(32), D(2), D(1000))
    assert D(result["funding_cash_quote_per_base"]) == 1
    assert D(result["after_execution_sensitivity_before_capital_bps"]) == 68
    assert D(result["after_execution_and_original_capital_stress_bps"]) == -1932
    assert D(result["break_even_annual_capital_cost_bps_per_reference_leg"]) == 34


def test_rescaling_marks_does_not_manufacture_relative_return():
    a = cashflow_metrics(
        [{"rate": D("0.01"), "mark": D(100)}], D(80), D(30), D(32), D(2), D(1000)
    )
    b = cashflow_metrics(
        [{"rate": D("0.01"), "mark": D(200)}], D(160), D(30), D(32), D(2), D(1000)
    )
    assert D(a["funding_cash_quote_per_base"]) * 2 == D(
        b["funding_cash_quote_per_base"]
    )
    assert (
        a["fixed_base_funding_bps_at_reference"]
        == b["fixed_base_funding_bps_at_reference"]
    )


def test_history_identity_and_duplicate_timestamp_rejected():
    row = {
        "symbol": "BTCUSDT",
        "fundingTime": 1,
        "fundingRate": "0.001",
        "markPrice": "100",
    }
    assert len(normalize_history([row], "BTCUSDT")) == 1
    for values, symbol in (
        ([row, row], "BTCUSDT"),
        ([row], "ETHUSDT"),
        ([], "BTCUSDT"),
    ):
        with pytest.raises(ValueError):
            normalize_history(values, symbol)


@pytest.mark.parametrize(
    "field,value",
    [
        ("fundingTime", True),
        ("fundingTime", 0),
        ("fundingRate", "NaN"),
        ("markPrice", "Infinity"),
        ("markPrice", "0"),
    ],
)
def test_invalid_history_stops(field, value):
    row = {
        "symbol": "BTCUSDT",
        "fundingTime": 1,
        "fundingRate": "0.001",
        "markPrice": "100",
    }
    row[field] = value
    with pytest.raises(ValueError):
        normalize_history([row], "BTCUSDT")


def test_retained_full_population_and_source_bindings():
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "docs/review/2026-09-04/spot-funding-cashflow-review.json").read_bytes()
    )
    assert canonical(result, "result_sha256") == result["result_sha256"]
    assert result["symbol_count"] == 17
    assert result["role_count"] == len(result["rows"]) == 51
    assert len({(row["symbol"], row["role"]) for row in result["rows"]}) == 51
    assert all(
        row["reference_time_ms"] < row["first_time_ms"] for row in result["rows"]
    )
    for binding in result["bindings"]:
        assert digest(published_bytes(binding)) == binding["sha256"]
    assert result["accepted_edge"] is False
    assert result["reference_is_executable_entry"] is False
    assert result["old_acceptance_rerun"] is False
    assert verify() == 51


def test_published_source_tamper_rejected_without_local_fallback(tmp_path):
    path = tmp_path / "docs/review/2026-09-04/spot-funding-sources/a.raw"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"changed")
    binding = {
        "path": "data/binance-broad-crypto-funding-carry-preflight-v1/raw/a.raw",
        "sha256": digest(b"original"),
    }
    with pytest.raises(ValueError, match="hash differs"):
        published_bytes(binding, tmp_path)


@pytest.mark.parametrize(
    "path",
    [
        "../outside.raw",
        "data/binance-broad-crypto-funding-carry-preflight-v1/raw/nested/a.raw",
    ],
)
def test_publication_mapping_rejects_outside_or_nested_paths(tmp_path, path):
    with pytest.raises(ValueError):
        published_bytes({"path": path, "sha256": digest(b"unused")}, tmp_path)


def test_consumed_output_or_journal_never_reruns(tmp_path):
    existing = tmp_path / "exists.json"
    existing.touch()
    with pytest.raises(RuntimeError, match="already exists"):
        run(existing)
    other = tmp_path / "has-journal.json"
    other.with_suffix(".journal.jsonl").touch()
    with pytest.raises(RuntimeError, match="already exists"):
        run(other)


def test_capital_stress_is_not_confused_with_funding_cash_loss():
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "docs/review/2026-09-04/spot-funding-cashflow-review.json").read_bytes()
    )
    btc_test = next(
        row
        for row in result["rows"]
        if (row["symbol"], row["role"]) == ("BTCUSDT", "test")
    )
    metrics = btc_test["metrics"]
    assert D(metrics["funding_cash_quote_per_base"]) > 0
    assert D(metrics["after_execution_sensitivity_before_capital_bps"]) > 0
    assert D(metrics["after_execution_and_original_capital_stress_bps"]) < 0
