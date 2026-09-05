from decimal import Decimal as D
import json

import pytest

from tools import review_carry_turnover_liquidity as review
from tools.review_spot_funding_cashflows import canonical, digest


@pytest.mark.parametrize(
    "cash,net,prefund,drawdown,negative",
    [
        ([5, -8, 4], 1, 3, 8, 1),
        ([5, -2, 1], 4, 0, 2, 1),
        ([-2, -3], -5, 5, 5, 2),
        ([0, 0], 0, 0, 0, 0),
    ],
)
def test_funding_path(cash, net, prefund, drawdown, negative):
    result = review.funding_path(list(map(D, cash)))
    assert result["net_funding_quote_per_base"] == net
    assert result["minimum_retained_funding_prefix_quote_per_base"] == -prefund
    assert (
        result["funding_only_prefund_if_all_receipts_retained_quote_per_base"]
        == prefund
    )
    assert result["largest_funding_peak_to_trough_quote_per_base"] == drawdown
    assert result["negative_settlement_count"] == negative


@pytest.mark.parametrize("cash", [[], [D("NaN")], [D("Infinity")]])
def test_invalid_cash_rejected(cash):
    with pytest.raises(ValueError):
        review.funding_path(cash)


def test_published_result_reconstructs_complete_population():
    plan_bytes = (review.BASE / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    result = json.loads((review.BASE / "result.json").read_bytes())
    assert result["plan_sha256"] == digest(plan_bytes)
    assert result["result_sha256"] == canonical(result, "result_sha256")
    expected = review.calculate(plan)
    assert all(result[key] == value for key, value in expected.items())
    assert len(result["rows"]) == result["symbol_count"] == 17
    assert [row["symbol"] for row in result["rows"]] == plan["symbols"]
    assert result["new_requests"] == 0
    assert not result["accepted_edge"]
    assert not result["profitability_claim"]
    assert not result["old_acceptance_changed"]
    for row in result["rows"]:
        m = row["metrics"]
        one = D(m["net_funding_less_one_allowance_bps_at_first_reference"])
        three = D(m["net_funding_less_three_allowances_bps_at_first_reference"])
        saving = D(m["avoided_two_round_trip_allowances_quote_per_base"])
        reference = D(m["first_reference_mark_quote_per_base"])
        assert abs((one - three) - saving / reference * 10000) < D("1e-20")
        assert row["reference_time_ms"] < row["first_funding_time_ms"]


def test_source_tamper_rejected():
    plan = json.loads((review.BASE / "plan.json").read_bytes())
    path = next(iter(plan["source_sha256"]))
    plan["source_sha256"][path] = "0" * 64
    with pytest.raises(ValueError, match="frozen source differs"):
        review.calculate(plan)


@pytest.mark.parametrize("existing", ["result.json", "journal.jsonl"])
def test_consumed_run_never_rewrites(monkeypatch, tmp_path, existing):
    target = tmp_path / existing
    target.write_bytes(b"original")
    monkeypatch.setattr(review, "BASE", tmp_path)
    with pytest.raises(FileExistsError):
        review.main()
    assert target.read_bytes() == b"original"
