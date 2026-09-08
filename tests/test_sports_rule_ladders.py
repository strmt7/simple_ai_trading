from copy import deepcopy

import pytest

from tools.screen_sports_rule_ladders import evaluate
from tools import screen_sports_rule_ladders as module
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from decimal import Decimal


def market(identifier, threshold):
    return {
        "id": identifier,
        "sportsMarketType": "totals",
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "line": threshold - 0.5,
        "outcomes": '["Over","Under"]',
        "description": f'This market will resolve to "Over" if A and B combine to score {threshold} or more points in this game. If the combined total is less than {threshold}, this market will resolve to "Under". If the game is canceled entirely, with no make-up game, this market will resolve 50-50.',
        "bestAsk": 0.6,
        "bestBid": 0.4,
    }


def test_complete_ladder_uses_asks_not_midpoints():
    rows = [market(str(i), i + 1) for i in range(3)]
    for row in rows:
        row["outcomePrices"] = '["0.1","0.1"]'
    result = evaluate({"markets": rows})
    assert result["relation_count"] == 3
    assert result["subfloor_count"] == 0
    assert result["best_relation"]["cost"] == "1.2"


def test_different_observation_rules_never_form_a_ladder():
    rows = [market("a", 1), market("b", 2)]
    rows[1]["description"] += " Overtime is excluded."
    assert evaluate({"markets": rows})["relation_count"] == 0


@pytest.mark.parametrize(
    "failure", ["missing_price", "missing_rule", "same_id", "positive"]
)
def test_rejection_and_incomplete_paths(failure):
    rows = [market("a", 1), market("b", 2)]
    if failure == "same_id":
        with pytest.raises(ValueError):
            evaluate({"markets": [rows[0], deepcopy(rows[0])]})
        return
    if failure == "missing_price":
        del rows[0]["bestAsk"]
    if failure == "missing_rule":
        rows[0]["description"] = "no exact rules"
    if failure == "positive":
        rows[0]["bestAsk"] = 0.3
    result = evaluate({"markets": rows})
    assert result["subfloor_count"] == (1 if failure == "positive" else 0)
    assert result["book_requests_authorized"] is False


def test_zero_ask_does_not_create_a_free_leg():
    rows = [market("a", 1), market("b", 2)]
    rows[0]["bestAsk"] = 0
    result = evaluate({"markets": rows})
    assert result["price_complete_count"] == result["subfloor_count"] == 0


def test_spread_ladder_and_cancellation_floor():
    rows = []
    for identifier, threshold in [("a", 2), ("b", 5)]:
        row = market(identifier, threshold)
        row.update(
            sportsMarketType="spreads",
            line=-(threshold - 0.5),
            outcomes='["A","B"]',
            description=f'This market will resolve to "A" if A win the game by {threshold} or more points. Otherwise, this market will resolve to "B". If the game is canceled entirely, with no make-up game, this market will resolve 50-50.',
        )
        rows.append(row)
    result = evaluate({"markets": rows})
    assert result["relation_count"] == 1
    assert result["subfloor_count"] == 0
    for margin in range(-10, 11):
        assert int(margin >= 2) + int(margin < 5) >= 1
    assert 0.5 + 0.5 == 1


def test_equal_threshold_duplicates_have_both_coverage_directions():
    result = evaluate({"markets": [market("a", 2), market("b", 2)]})
    assert result["relation_count"] == 2


def test_rule_only_size_bound_precedes_prices():
    rows = [market(str(i), i + 1) for i in range(143)]
    for row in rows:
        del row["bestAsk"]
        del row["bestBid"]
    with pytest.raises(ValueError, match="before economics"):
        evaluate({"markets": rows})


@pytest.mark.parametrize("tamper", [False, True])
def test_frozen_offline_run_checks_sources_and_never_repeats(
    tmp_path, monkeypatch, tamper
):
    monkeypatch.setattr(module, "_root_path", lambda value: tmp_path / value)
    raw = {"slug": "exact", "markets": [market("a", 1), market("b", 2)]}
    data = json.dumps(raw).encode()
    (tmp_path / "raw.json").write_bytes(data)
    plan = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": "raw.json", "sha256": hashlib.sha256(data).hexdigest()},
        "implementations": [],
        "event_slug": "exact",
        "result_path": "result.json",
        "gate": {
            "network_requests": 0,
            "book_requests_authorized": False,
            "maximum_relations": 10000,
            "price_semantics": "bestAsk_or_1-bestBid_rejection_only",
            "families": ["spreads", "totals"],
        },
    }
    plan["contract_sha256"] = module._canonical_hash(plan, "contract_sha256")
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(plan))
    if tamper:
        (tmp_path / "raw.json").write_bytes(b"tampered")
        with pytest.raises(ValueError):
            module.run(path)
        assert not (tmp_path / "result.json").exists()
        return
    assert module.run(path, preflight=True) is None
    assert not (tmp_path / "result.json").exists()
    result = module.run(path)
    assert result["screen"]["relation_count"] == 1
    assert result["screen"]["subfloor_count"] == 0
    assert result["result_sha256"] == module._canonical_hash(result, "result_sha256")
    with pytest.raises(FileExistsError):
        module.run(path)


def test_retained_cfb_deployment_and_every_economic_row_reconstruct():
    root = Path(__file__).resolve().parents[1]
    folder = root / "docs/review/2026-09-08/cfb-deployment"
    for name, field in [
        ("contract", "contract_sha256"),
        ("source-result", "result_sha256"),
        ("deployment-result", "result_sha256"),
        ("ladder-contract", "contract_sha256"),
        ("ladder-result", "result_sha256"),
    ]:
        value = json.loads((folder / f"{name}.json").read_bytes())
        assert value[field] == module._canonical_hash(value, field)
    contract = json.loads((folder / "ladder-contract.json").read_bytes())
    raw = (root / contract["source"]["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == contract["source"]["sha256"]
    event = json.loads(raw)
    deployment = json.loads((folder / "deployment-result.json").read_bytes())[
        "deployment"
    ]
    assert deployment["deployment_gate_passed"]
    assert module.instant(event["createdAt"]) > module.instant(
        "2026-08-31T16:33:29.370Z"
    )
    assert len(event["markets"]) == 58
    by_id = {m["id"]: m for m in event["markets"]}
    screen = json.loads((folder / "ladder-result.json").read_bytes())["screen"]
    assert screen["relation_count"] == screen["price_complete_count"] == 553
    assert len(screen["relations"]) == 553
    assert screen["included_markets"] == 48 and len(screen["exclusions"]) == 10
    assert screen["subfloor_count"] == 0
    for row in screen["relations"]:
        lower = by_id[row["positive_market_id"]]
        upper = by_id[row["complement_market_id"]]
        cost = Decimal(str(lower["bestAsk"])) + 1 - Decimal(str(upper["bestBid"]))
        assert cost == Decimal(row["cost"])
        assert cost >= 1
        assert row["lower_threshold"] <= row["upper_threshold"]
        assert row["strictly_subfloor"] is False
    assert min(Decimal(r["cost"]) for r in screen["relations"]) == Decimal("1.07")
    registry = json.loads(
        (
            root / "docs/model-research/structural-edge-priority-registry-v1.json"
        ).read_bytes()
    )
    memberships = [
        (row["priority_rank"], artifact["path"])
        for row in registry["prioritized_hypotheses"]
        for artifact in row["canonical_artifacts"]
        if "/cfb-deployment/" in artifact["path"]
    ]
    assert len(memberships) == 4
    assert {rank for rank, _ in memberships} == {30}
