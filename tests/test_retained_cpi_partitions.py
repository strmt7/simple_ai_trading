"""Offline source reconstruction and within-event payoff identities."""

from decimal import Decimal
import json

from tools.screen_retained_cpi_partitions import BASE, adjudicate, canonical, sha


def test_frozen_result_reconstructs_and_no_cross_event_claim_is_made():
    plan = json.loads((BASE / "plan.json").read_bytes())
    result = json.loads((BASE / "result.json").read_bytes())
    expected = result.pop("result_sha256")
    assert canonical(result) == expected
    assert result.pop("plan_sha256") == sha((BASE / "plan.json").read_bytes())
    result.pop("schema_version")
    result.pop("created_at_utc")
    assert result == adjudicate(plan)
    assert result["cross_event_packages"] == result["network_requests"] == 0
    assert len(result["rows"]) == 42
    assert result["gross_positive_rows"] == result["fee_tick_positive_rows"] == 0
    assert not result["promotion_eligible"] and not result["accepted_edge"]


def test_reported_one_decimal_partition_has_no_gap_or_overlap_including_tails():
    plan = json.loads((BASE / "plan.json").read_bytes())
    for event, lower, upper in zip(plan["events"], (29, -3), (40, 5), strict=True):
        labels = event["labels"]
        assert len(labels) == upper - lower + 1
        for point in range(lower - 1, upper + 2):
            value = Decimal(point) / 10
            matched = []
            for label in labels:
                if label.startswith("≤"):
                    matched.append(value <= Decimal(label[1:-1]))
                elif label.startswith("≥"):
                    matched.append(value >= Decimal(label[1:-1]))
                else:
                    matched.append(value == Decimal(label[:-1]))
            assert sum(matched) == 1
        assert sum(label.startswith("≤") for label in labels) == 1
        assert sum(label.startswith("≥") for label in labels) == 1


def test_every_published_package_has_its_claimed_floor_in_each_valid_event_state():
    plan = json.loads((BASE / "plan.json").read_bytes())
    result = json.loads((BASE / "result.json").read_bytes())
    from tools.screen_retained_cpi_partitions import ROOT

    for event in plan["events"]:
        raw = json.loads((ROOT / event["raw_path"]).read_bytes())
        ids = [str(m["id"]) for m in raw["markets"]]
        for row in (r for r in result["rows"] if r["event_id"] == event["id"]):
            payouts = [
                sum(
                    (leg["market_id"] == winner) == (leg["outcome"] == "Yes")
                    for leg in row["legs"]
                )
                for winner in ids
            ]
            assert Decimal(min(payouts)) == Decimal(
                row["guaranteed_payout_floor_pUSD_per_share"]
            )
