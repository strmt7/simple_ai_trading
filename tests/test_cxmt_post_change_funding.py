from copy import deepcopy
from decimal import Decimal

import pytest

from tools.review_cxmt_post_change_funding import ENDS, HOURS, evaluate


def _inputs():
    return {
        "more": False,
        "data": [{"timestamp": t, "funding_rate": "0.00001"} for t in HOURS],
    }, [
        {
            "fundingTime": t,
            "fundingRate": "0.009",
            "symbol": "CXMTUSDT",
            "rateType": "Regular",
        }
        for t in ENDS
    ]


def test_fixed_orientation_and_exact_roles_pass_only_synthetic_high_carry():
    result = evaluate(*_inputs())
    assert result["fixed_orientation"] == "long_polymarket_short_binance"
    assert [row["count"] for row in result["roles"].values()] == [6, 3, 3]
    assert result["history_survivor"] is True and result["qualified_edge"] is False
    assert Decimal(result["rows"][0]["gross_equal_notional_carry_bips"]) == Decimal(
        "89.6"
    )


def test_cheap_funding_proxy_fails_before_price_or_account_escalation():
    poly, binance = _inputs()
    for row in binance:
        row["fundingRate"] = "0.0001"
    result = evaluate(poly, binance)
    assert not result["history_survivor"]
    assert all(not role["passes"] for role in result["roles"].values())


@pytest.mark.parametrize(
    "case",
    [
        "missing_poly",
        "duplicate_poly",
        "more",
        "outside_poly",
        "bad_phase",
        "missing_binance",
        "duplicate_binance",
        "special",
        "wrong_symbol",
        "nan",
        "cap",
        "numeric_rate",
    ],
)
def test_incomplete_or_incompatible_evidence_rejects(case):
    poly, binance = deepcopy(_inputs())
    if case == "missing_poly":
        poly["data"].pop()
    elif case == "duplicate_poly":
        poly["data"][1] = poly["data"][0]
    elif case == "more":
        poly["more"] = True
    elif case == "outside_poly":
        poly["data"][0]["timestamp"] -= 3_600_000
    elif case == "bad_phase":
        poly["data"][0]["timestamp"] += 60_001
    elif case == "missing_binance":
        binance.pop()
    elif case == "duplicate_binance":
        binance[1] = binance[0]
    elif case == "special":
        binance[0]["rateType"] = "Special"
    elif case == "wrong_symbol":
        binance[0]["symbol"] = "BTCUSDT"
    elif case == "nan":
        binance[0]["fundingRate"] = "NaN"
    elif case == "cap":
        binance[0]["fundingRate"] = "0.02"
    else:
        binance[0]["fundingRate"] = 0.01
    with pytest.raises(ValueError):
        evaluate(poly, binance)


def test_validation_failure_cannot_be_hidden_by_training_profit():
    poly, binance = _inputs()
    for row in binance[6:9]:
        row["fundingRate"] = "-0.009"
    result = evaluate(poly, binance)
    assert result["roles"]["training"]["passes"]
    assert not result["roles"]["validation"]["passes"]
    assert not result["history_survivor"]
