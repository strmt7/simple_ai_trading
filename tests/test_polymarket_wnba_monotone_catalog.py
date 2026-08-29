from __future__ import annotations

import json
from decimal import Decimal

from tools.screen_polymarket_wnba_monotone_catalog import _screen_event


def _market(
    market_id: str,
    market_type: str,
    outcomes: list[str],
    prices: list[str],
    description: str,
    *,
    line: float | None,
) -> dict[str, object]:
    return {
        "id": market_id,
        "slug": f"market-{market_id}",
        "conditionId": f"0x{market_id}",
        "question": market_id,
        "description": description,
        "sportsMarketType": market_type,
        "line": line,
        "outcomes": json.dumps(outcomes),
        "outcomePrices": json.dumps(prices),
        "clobTokenIds": json.dumps([f"{market_id}-yes", f"{market_id}-no"]),
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "enableOrderBook": True,
        "negRisk": False,
        "feesEnabled": True,
        "feeSchedule": {
            "exponent": 1,
            "rate": 0.05,
            "rebateRate": 0.15,
            "takerOnly": True,
        },
        "takerBaseFee": 1000,
        "secondsDelay": 1,
        "orderMinSize": 5,
        "orderPriceMinTickSize": 0.01,
    }


def test_screen_event_proves_wnba_moneyline_spread_floor() -> None:
    team_a = "Alpha"
    team_b = "Beta"
    moneyline = _market(
        "1",
        "moneyline",
        [team_a, team_b],
        ["0.60", "0.40"],
        (
            f'If the {team_a} win, the market will resolve to "{team_a}". '
            f'If the {team_b} win, the market will resolve to "{team_b}". '
            "If the game is canceled entirely, with no make-up game, this market will resolve 50-50. "
            "The result will be determined based on the final score including any overtime periods."
        ),
        line=None,
    )
    spread = _market(
        "2",
        "spreads",
        [team_a, team_b],
        ["0.45", "0.55"],
        (
            f'This market will resolve to "{team_a}" if the {team_a} win the game by 3 or more points. '
            f'Otherwise it resolves to "{team_b}". If canceled, this market will resolve 50-50.'
        ),
        line=-2.5,
    )
    relations, summary = _screen_event(
        {
            "id": "event-1",
            "slug": "wnba-alpha-beta-2026-08-31",
            "title": "Alpha vs. Beta",
            "startTime": "2026-08-31T20:00:00Z",
            "markets": [moneyline, spread],
        }
    )

    assert summary["margin_threshold_count"] == 2
    assert len(relations) == 1
    relation = relations[0]
    assert relation["minimum_terminal_payout_per_share_pUSD"] == Decimal("1")
    assert relation["displayed_price_sum_per_share_pUSD"] == Decimal("1.15")
    assert relation["passes_strictly_below_payout_gate"] is False
