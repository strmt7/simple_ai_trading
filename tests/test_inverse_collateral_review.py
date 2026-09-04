from dataclasses import replace
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path

import pytest

from tools.review_inverse_collateral import (
    InverseShortScenario,
    illustrative_scenarios,
    write_review,
)


@pytest.fixture
def matched():
    return InverseShortScenario(
        10, D("10"), D("100"), D("1"), D("100"), D("100"), D("0"), D("0"), D("0.005")
    )


@pytest.mark.parametrize("price", ["0.001", "25", "100", "400", "100000000"])
def test_matched_common_price_is_constant_not_double_counted(matched, price):
    result = replace(matched, mark_future=D(price), spot_conversion=D(price)).evaluate()
    assert result["total_wealth_quote"] == D("100")
    assert result["common_price_residual_delta_coin"] == 0
    assert result["scenario_margin_surplus_coin"] > 0


def test_basis_and_coin_debit_destroy_unconditional_dollar_floor(matched):
    basis = replace(matched, spot_conversion=D("90")).evaluate()
    assert basis["total_wealth_quote"] == 90
    fee = replace(
        matched,
        net_coin_cashflow=D("-0.01"),
        mark_future=D("10000"),
        spot_conversion=D("10000"),
    ).evaluate()
    assert fee["total_wealth_quote"] == 0
    assert fee["scenario_margin_surplus_coin"] < 0


def test_retained_funding_differs_from_converted_funding(matched):
    market = replace(matched, mark_future=D("400"), spot_conversion=D("400"))
    retained = replace(market, net_coin_cashflow=D("0.001")).evaluate()
    converted = replace(market, outside_quote_cash=D("0.1")).evaluate()
    assert retained["total_wealth_quote"] == D("100.4")
    assert converted["total_wealth_quote"] == D("100.1")
    assert converted["common_price_residual_delta_coin"] == 0
    assert converted["equity_coin"] == D("0.25")


def test_contract_rounding_and_excess_collateral_leave_delta(matched):
    rounded = replace(matched, contract_count=9).evaluate()
    assert rounded["common_price_residual_delta_coin"] == D("0.1")
    excess = replace(matched, collateral_coin=D("1.1")).evaluate()
    assert excess["common_price_residual_delta_coin"] == D("0.1")


def test_decomposition_matches_direct_cash_ledger(matched):
    scenario = replace(
        matched,
        collateral_coin=D("1.03"),
        mark_future=D("250"),
        spot_conversion=D("240"),
        net_coin_cashflow=D("-0.02"),
        outside_quote_cash=D("-1"),
    )
    result = scenario.evaluate()
    coin_ledger = D("1.03") + D("100") * (1 / D("250") - 1 / D("100")) - D("0.02")
    assert result["equity_coin"] == coin_ledger
    assert result["total_wealth_quote"] == coin_ledger * 240 - 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("contract_count", True),
        ("contract_count", 0),
        ("contract_count", 1.5),
        ("contract_size_quote", D("NaN")),
        ("entry_future", D("0")),
        ("mark_future", D("-1")),
        ("spot_conversion", D("Infinity")),
        ("collateral_coin", D("-1")),
        ("net_coin_cashflow", D("NaN")),
        ("outside_quote_cash", D("Infinity")),
        ("maintenance_rate", D("1")),
        ("maintenance_rate", D("-0.01")),
        ("collateral_coin", 1.0),
    ],
)
def test_invalid_scenario_rejected(matched, field, value):
    with pytest.raises(ValueError):
        replace(matched, **{field: value}).evaluate()


def test_fixed_scenario_matrix_is_not_market_evidence():
    rows = illustrative_scenarios()
    assert len(rows) == 12
    assert {row["label"] for row in rows} == {
        "matched_no_cashflow",
        "coin_fee_debit",
        "coin_funding_retained",
        "coin_funding_converted_at_100",
    }
    assert all("accepted_edge" not in row["outputs"] for row in rows)


def test_artifact_bindings_and_exclusive_output(tmp_path):
    output = tmp_path / "review.json"
    write_review(output)
    result = json.loads(output.read_text(encoding="utf-8"))
    digest = result.pop("result_sha256")
    assert (
        digest
        == hashlib.sha256(
            json.dumps(
                result,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    assert result["accepted_edge"] is False
    assert result["official_mechanics_verified"] is False
    root = Path(__file__).resolve().parents[1]
    for binding in result["bindings"]:
        assert (
            hashlib.sha256((root / binding["path"]).read_bytes()).hexdigest()
            == binding["sha256"]
        )
    with pytest.raises(FileExistsError):
        write_review(output)


def test_retained_artifact_reconstructs_scenarios_and_hashes():
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "docs/review/2026-09-04/inverse-collateral-scenarios.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["scenarios"] == illustrative_scenarios()
    for binding in result["bindings"]:
        assert (
            hashlib.sha256((root / binding["path"]).read_bytes()).hexdigest()
            == binding["sha256"]
        )
    digest = result.pop("result_sha256")
    assert (
        hashlib.sha256(
            json.dumps(
                result, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        == digest
    )
