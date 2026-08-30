"""Adjudicate one retained exact CFB event without network access."""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from tools import adjudicate_polymarket_exact_nfl_monotone_prefilter as football
from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    HALF,
    _canonical_hash,
    _common_market_gate,
    _json_pair,
    _market_ref,
    _root_path,
    _sha256,
)


SCHEMA = "polymarket-exact-cfb-monotone-prefilter-adjudication-v1"


def _margin_markets(
    markets: list[dict[str, Any]], moneyline: dict[str, Any]
) -> tuple[str, str, list[dict[str, Any]]]:
    """Parse exact CFB margin rules without assuming an article before a team name."""
    _common_market_gate(moneyline)
    team_a, team_b = _json_pair(moneyline.get("outcomes"), "moneyline outcomes")
    description = str(moneyline["description"])
    for team in (team_a, team_b):
        required = f'If {team} wins, the market will resolve to "{team}".'
        if required not in description:
            raise RuntimeError("CFB moneyline lacks exact team-win semantics")
    if (
        "If the game is canceled entirely or ends in a tie, with no make-up game, "
        "this market will resolve 50-50."
    ) not in description:
        raise RuntimeError("CFB moneyline lacks exact half-half tie semantics")

    rows = [
        _market_ref(
            moneyline,
            threshold=1,
            positive_outcome=team_a,
            complement_outcome=team_b,
            resolver="moneyline_with_half_half_tie",
        )
    ]
    for market in markets:
        if market.get("sportsMarketType") != "spreads":
            continue
        _common_market_gate(market)
        outcomes = _json_pair(market.get("outcomes"), "spread outcomes")
        if set(outcomes) != {team_a, team_b}:
            raise RuntimeError("spread team identity mismatch")
        line = Decimal(str(market.get("line")))
        if line >= 0 or (-line % 1) != HALF:
            raise RuntimeError("CFB spread line is not a negative half-point")
        required_margin = int(-line + HALF)
        favored = outcomes[0]
        spread_description = str(market["description"])
        exact_win = (
            f'This market will resolve to "{favored}" if {favored} win the game by '
            f"{required_margin} or more points."
        )
        other = team_b if favored == team_a else team_a
        exact_other = f'Otherwise, this market will resolve to "{other}".'
        exact_cancel = (
            "If the game is canceled entirely, with no make-up game, this market "
            "will resolve 50-50."
        )
        if not all(
            fragment in spread_description
            for fragment in (exact_win, exact_other, exact_cancel)
        ):
            raise RuntimeError("CFB spread description does not bind its exact line")
        threshold = (
            required_margin if favored == team_a else -(required_margin - 1)
        )
        prices = dict(
            zip(
                _json_pair(market.get("outcomes"), "spread outcomes"),
                _json_pair(market.get("outcomePrices"), "spread prices"),
            )
        )
        rows.append(
            {
                **_market_ref(
                    market,
                    threshold=threshold,
                    positive_outcome=team_a,
                    complement_outcome=team_b,
                    resolver="integer_margin_threshold",
                ),
                "positive_price_pUSD": Decimal(prices[team_a]),
                "complement_price_pUSD": Decimal(prices[team_b]),
                "favored_team": favored,
                "line": line,
            }
        )
    thresholds = [int(row["threshold"]) for row in rows]
    if len(thresholds) != len(set(thresholds)):
        raise RuntimeError("CFB margin lattice contains duplicate logical thresholds")
    return team_a, team_b, sorted(rows, key=lambda row: int(row["threshold"]))


def adjudicate(metadata: dict[str, Any]) -> dict[str, Any]:
    original = football._margin_markets
    football._margin_markets = _margin_markets
    try:
        result = football.adjudicate(metadata)
    finally:
        football._margin_markets = original
    result["schema_version"] = SCHEMA
    result["implementation"] = {
        "path": "tools/adjudicate_polymarket_exact_cfb_monotone_prefilter.py",
        "sha256": _sha256(Path(__file__).read_bytes()),
        "football_algebra_path": (
            "tools/adjudicate_polymarket_exact_nfl_monotone_prefilter.py"
        ),
        "football_algebra_sha256": _sha256(Path(football.__file__).read_bytes()),
        "shared_algebra_path": (
            "tools/adjudicate_polymarket_exact_mlb_monotone_prefilter.py"
        ),
        "shared_algebra_sha256": _sha256(
            (
                Path(__file__).resolve().parents[1]
                / "tools/adjudicate_polymarket_exact_mlb_monotone_prefilter.py"
            ).read_bytes()
        ),
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    contract_path = _root_path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    metadata = football._validate_contract(contract, contract_path)
    output_path = _root_path(str(contract["output_path"]))
    if output_path.exists():
        raise RuntimeError("adjudication output already exists")
    result = adjudicate(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    best = result["rejection_only_gamma_prefilter"]["best_relation"]
    print(
        json.dumps(
            {
                "candidate_count": result["rejection_only_gamma_prefilter"][
                    "candidate_count_strictly_below_payout_floor"
                ],
                "complete_relation_count": result["payoff_proof"][
                    "complete_relation_count"
                ],
                "best_displayed_price_sum_pUSD": best[
                    "displayed_price_sum_per_share_pUSD"
                ],
                "best_family": best["family"],
                "network_requests": 0,
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
