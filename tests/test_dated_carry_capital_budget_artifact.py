import hashlib
import json
from decimal import Decimal as D

import pytest

from tools import review_dated_carry_capital_budget as budget


def test_complete_retained_budget_reconstructs_and_independent_identity():
    recorded = json.loads((budget.BASE / "result.json").read_bytes())
    assert budget.review() == recorded
    assert len(recorded["rows"]) == 12
    positive = 0
    for row in recorded["rows"]:
        assert len(row["scenarios"]) == 9
        years = D(row["delivery_time_ms"] - row["capture_time_ms"]) / D(31557600000)
        for scenario in row["scenarios"]:
            capital_cost = (
                D(scenario["annual_rate"])
                * D(scenario["capital_multiple"])
                * years
                * 10000
            )
            remaining = D(scenario["remaining_noncapital_cost_budget_bips"])
            assert abs(
                remaining + capital_cost - D(row["original_gross_basis_bips"])
            ) < D("1e-20")
            assert abs(
                remaining
                - 35
                - D(scenario["headroom_after_separate_noncapital_reserve_bips"])
            ) < D("1e-20")
            positive += (
                D(scenario["headroom_after_separate_noncapital_reserve_bips"]) > 0
            )
    assert positive == 17
    assert not recorded["accepted_edge"]
    assert not recorded["capital_feasibility_proved"]


@pytest.mark.parametrize("failure", ["binding", "reserve", "population"])
def test_retained_input_contract_guards(tmp_path, monkeypatch, failure):
    plan = json.loads((budget.BASE / "plan.json").read_bytes())
    source = json.loads((budget.ROOT / plan["snapshot_path"]).read_bytes())
    if failure == "reserve":
        source["screens"][0]["quantity_results"][0]["all_in_cost_hurdle_bips"] = "34"
    if failure == "population":
        source["screens"].pop()
    raw = json.dumps(source).encode()
    (tmp_path / "source.json").write_bytes(raw)
    plan["snapshot_path"] = "source.json"
    plan["source_sha256"] = {"source.json": hashlib.sha256(raw).hexdigest()}
    if failure == "binding":
        plan["source_sha256"]["source.json"] = "0" * 64
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="ascii")
    monkeypatch.setattr(budget, "ROOT", tmp_path)
    monkeypatch.setattr(budget, "BASE", tmp_path)
    with pytest.raises(ValueError):
        budget.review()
