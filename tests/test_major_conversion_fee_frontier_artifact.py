import hashlib
import json
from decimal import Decimal as D

from tools.review_major_conversion_fee_frontier import BASE, ROOT, review


def test_full_retained_route_frontier_reconstructs_without_promotion():
    result = json.loads((BASE / "result.json").read_bytes())
    assert review() == result
    plan = json.loads((BASE / "plan.json").read_bytes())
    assert {x["sample_index"] for x in plan["inputs"]} == set(range(12))
    for entry in plan["inputs"]:
        assert (
            hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest()
            == entry["sha256"]
        )
    assert len(result["rows"]) == 288
    assert len(result["route_summaries"]) == 24
    assert sum(D(x["gross_incremental_bips"]) > 0 for x in result["rows"]) == 113
    assert (
        sum(
            x["positive_samples_by_uniform_fee"]["0"] == 12
            for x in result["route_summaries"]
        )
        == 3
    )
    assert all(
        x["positive_samples_by_uniform_fee"]["1.2"] < 12
        for x in result["route_summaries"]
    )
    assert not any(
        x["all_samples_above_three_bip_stress_by_uniform_fee"]["0"]
        for x in result["route_summaries"]
    )
    assert result["network_requests"] == 0
    assert result["accepted_edge"] is False
    assert result["new_independent_validation"] is False
