"""Rule admission is price blind; consumed offline experiments cannot rerun."""

from dataclasses import asdict
from decimal import Decimal
from fractions import Fraction
from importlib.metadata import version
import json
from pathlib import Path
import sys

import pytest

from tools import screen_retained_score_cover as module


APPOINTMENT = "November 9 at 8:15PM ET"


def fixture_event():
    ending = module.POSTPONE + " " + module.CANCEL
    rows = []

    def add(kind, question, line, outcomes, description):
        identifier = len(rows) + 1
        rows.append(
            dict(
                id=str(identifier),
                conditionId="0x" + f"{identifier:064x}",
                clobTokenIds=json.dumps([str(identifier * 2), str(identifier * 2 + 1)]),
                question=question,
                line=line,
                sportsMarketType=kind,
                outcomes=json.dumps(outcomes),
                description=description,
                active=True,
                closed=False,
                acceptingOrders=True,
                enableOrderBook=True,
            )
        )

    add(
        "moneyline",
        "Alpha vs. Beta",
        None,
        ["Alpha", "Beta"],
        f"In the upcoming NFL football game between Alpha and Beta, scheduled for {APPOINTMENT}: "
        'This market will resolve to "Alpha" if Alpha win the game. '
        'This market will resolve to "Beta" if Beta win the game. '
        "Overtime is included if played. If the game ends in a tie, this market will resolve 50-50. "
        + ending,
    )
    add(
        "spreads",
        "Spread: Beta (-1.5)",
        -1.5,
        ["Beta", "Alpha"],
        f"In the upcoming NFL football game between Alpha and Beta, scheduled for {APPOINTMENT}: "
        'This market will resolve to "Beta" if the Beta win the game by 2 or more points. '
        'Otherwise, this market will resolve to "Alpha". Overtime is included if played. '
        + ending,
    )
    add(
        "totals",
        "Alpha vs. Beta: O/U 1.5",
        1.5,
        ["Over", "Under"],
        f"In the upcoming NFL game, scheduled for {APPOINTMENT}: "
        'This market will resolve to "Over" if the Alpha and Beta combine to score 2 or more points in this game. '
        'If the combined total is less than 2, this market will resolve to "Under". '
        + ending,
    )
    for team in ("Alpha", "Beta"):
        add(
            "team_totals",
            f"{team} Team Total: O/U 0.5",
            0.5,
            ["Over", "Under"],
            f"In the upcoming NFL game between Alpha and Beta, scheduled for {APPOINTMENT}: "
            f'This market will resolve to "Over" if the {team} score 1 or more points in this game. '
            f'If the {team} score less than 1 points, this market will resolve to "Under". '
            f"This market is based solely on the {team}'s score, not the combined score of both teams. "
            + ending,
        )
    return dict(
        id="fixture",
        slug="fixture",
        title="Alpha vs. Beta",
        teams=[dict(alias="Alpha"), dict(alias="Beta")],
        active=True,
        closed=False,
        markets=rows,
    )


def test_complete_rules_map_favored_team_and_ties_without_reading_prices():
    system, bindings = module.compile_event(fixture_event(), appointment=APPOINTMENT)
    assert len(bindings) == 5
    assert [asdict(rule) for rule in system.markets][1] == dict(
        identifier="2", kind="margin", threshold=2, team=1
    )
    assert system.markets[0].first_payout_twice(3, 3) == 1
    assert system.markets[1].first_payout_twice(1, 3) == 2


@pytest.mark.parametrize(
    "failure",
    [
        "overtime",
        "cancellation",
        "opposite-team",
        "threshold",
        "outcome",
        "extra-text",
        "quote",
        "duplicate-token",
    ],
)
def test_complete_rule_and_scope_mismatches_reject_before_economics(failure):
    event = fixture_event()
    row = event["markets"][1]
    if failure == "overtime":
        row["description"] = row["description"].replace("included", "excluded")
    if failure == "cancellation":
        row["description"] = row["description"].replace("50-50", "zero")
    if failure == "opposite-team":
        row["description"] = row["description"].replace("if the Beta", "if the Alpha")
    if failure == "threshold":
        row["line"] = -2.5
    if failure == "outcome":
        row["outcomes"] = '["Alpha","Beta"]'
    if failure == "extra-text":
        row["description"] += " The second half is excluded."
    if failure == "quote":
        row["bestAsk"] = "not-to-be-consulted"
    if failure == "duplicate-token":
        row["clobTokenIds"] = event["markets"][0]["clobTokenIds"]
    with pytest.raises(ValueError):
        module.compile_event(event, appointment=APPOINTMENT)


def test_zero_and_absent_quotes_remain_missing_and_outcome_prices_are_ignored():
    event = fixture_event()
    _, bindings = module.compile_event(event, appointment=APPOINTMENT)
    for market in event["markets"]:
        market.update(
            bestAsk=Decimal("0.55"),
            bestBid=Decimal("0.45"),
            outcomePrices="not-a-price-source",
        )
    event["markets"][0]["bestAsk"] = 0
    event["markets"][1].pop("bestBid")
    prices, _ = module.price_legs(event["markets"], bindings)
    assert prices[0] is None and prices[3] is None
    assert prices[1] == Fraction(11, 20)


def test_positive_crossed_quotes_reject():
    event = fixture_event()
    _, bindings = module.compile_event(event, appointment=APPOINTMENT)
    for market in event["markets"]:
        market.update(bestAsk=Decimal("0.4"), bestBid=Decimal("0.6"))
    with pytest.raises(ValueError, match="crossed"):
        module.price_legs(event["markets"], bindings)


def test_real_entrypoint_preflight_has_no_price_access_and_actual_run_is_one_use(
    tmp_path, monkeypatch, capsys
):
    root = Path(__file__).resolve().parents[1]
    implementations = {}
    for name in [
        "src/simple_ai_trading/score_cover.py",
        "tools/screen_retained_score_cover.py",
    ]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
        implementations[name] = module.digest(target.read_bytes())
    event = fixture_event()
    for market in event["markets"]:
        market.update(bestAsk=0.55, bestBid=0.45)
    source = tmp_path / "raw.json"
    source.write_text(json.dumps(dict(events=[event])))
    contract = dict(
        contract_path="contract.json",
        source_path="raw.json",
        source_sha256=module.digest(source.read_bytes()),
        output_path="result.json",
        journal_path="journal.jsonl",
        implementations=implementations,
        numerical_runtime={name: version(name) for name in ("numpy", "scipy")},
        event_id="fixture",
        event_slug="fixture",
        appointment=APPOINTMENT,
        policy=dict(
            eligible_kinds=sorted(module.KINDS),
            solver="highs-ds",
            solver_seconds=20,
            rational_denominator_limit=1000000,
            common_cancellation_payoff="1/2",
            interpretation="consumed-response exploratory proxy system only",
            book_fee_account_or_order_authority=False,
        ),
    )
    contract["contract_sha256"] = module.digest(module.canonical(contract))
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["screen", "--contract", "contract.json", "--preflight"]
    )
    original = module.price_legs
    monkeypatch.setattr(
        module, "price_legs", lambda *a: pytest.fail("preflight consulted prices")
    )
    module.main()
    assert json.loads(capsys.readouterr().out)["economic_fields_consulted"] is False
    assert not (tmp_path / "journal.jsonl").exists()
    monkeypatch.setattr(module, "price_legs", original)
    monkeypatch.setattr(sys, "argv", ["screen", "--contract", "contract.json"])
    module.main()
    result = json.loads((tmp_path / "result.json").read_bytes())
    assert result["cost_lower"] == result["cost_upper"] == "11/10"
    assert result["book_fee_account_or_order_authority"] is False
    assert [
        json.loads(line)["phase"]
        for line in (tmp_path / "journal.jsonl").read_text().splitlines()
    ] == ["intent", "completed"]
    with pytest.raises(ValueError, match="consumed"):
        module.main()
