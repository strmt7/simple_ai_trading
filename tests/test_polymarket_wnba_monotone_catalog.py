from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

from tools.screen_polymarket_wnba_monotone_catalog import _screen_event


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
CONTRACT_HASH = "36faeee7464832f335739ec8d1fc5609c98e1cdc9b6f267901934fbd8277f831"
RESULT_HASH = "fd0a9e844a7ad7d1a6eb5372c961ff82ea52d3c72a8c558ba191a53bace02cef"
REGISTRY_HASH = json.loads(
    (ROOT / "docs/model-research/structural-edge-priority-registry-v1.json").read_text(
        encoding="utf-8"
    )
)["result_sha256"]


def _canonical_hash(value: dict[str, object], field: str) -> str:
    body = dict(value)
    body.pop(field)
    payload = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


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


def test_complete_future_window_stops_before_books_and_updates_existing_family() -> (
    None
):
    contract = json.loads(
        (
            ACTION_VALUE
            / "polymarket-future-wnba-monotone-catalog-contract-v1-2026-08-29.json"
        ).read_text(encoding="utf-8")
    )
    result = json.loads(
        (
            ACTION_VALUE
            / "polymarket-future-wnba-monotone-catalog-result-v1-2026-08-29.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["contract_sha256"] == CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == CONTRACT_HASH
    assert result["result_sha256"] == RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == RESULT_HASH
    assert result["capture"]["returned_event_count"] == 3
    assert result["capture"]["population_complete_under_frozen_filter"] is True
    assert result["screen"]["complete_relation_count"] == 2
    assert result["screen"]["candidate_count_strictly_below_payout_floor"] == 0
    assert result["screen"]["depth_candidate"] is None
    assert result["authority"]["book_requests"] == 0
    assert result["authority"]["fee_requests"] == 0
    assert result["adjudication"]["accepted_edge"] is False

    raw = ROOT / "data/polymarket-future-wnba-monotone-catalog-v1/raw/events.json"
    journal = ROOT / (
        "data/polymarket-future-wnba-monotone-catalog-v1/request-journal.jsonl"
    )
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == (
        "1b989cf0b2edf79445c73ce616149c2696cff96c6f6dd7d89c310f770d4004e3"
    )
    assert hashlib.sha256(journal.read_bytes()).hexdigest() == (
        "40a7edc61fcf5c3a42a8ab7aef65cf04703cde30989b7b23f23a915291620368"
    )

    registry = json.loads(
        (
            ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 30
    )
    assert family["canonical_artifacts"][-1]["result_sha256"] == RESULT_HASH
    assert any(
        row["canonical_result_sha256"] == RESULT_HASH
        for row in registry["terminal_do_not_repeat"]
    )
