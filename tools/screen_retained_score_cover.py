"""One-use exploratory score-cover certificate from an exact retained response."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal
from fractions import Fraction
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
from datetime import datetime, timezone

from simple_ai_trading.score_cover import ScoreMarket, score_states, solve_score_cover


ROOT = Path(__file__).resolve().parents[1]
KINDS = {"moneyline", "spreads", "totals", "team_totals"}
FIELDS = {
    "id",
    "question",
    "description",
    "sportsMarketType",
    "line",
    "outcomes",
    "conditionId",
    "clobTokenIds",
    "active",
    "closed",
    "acceptingOrders",
    "enableOrderBook",
    "resolutionSource",
}
POSTPONE = "If the game is postponed, this market will remain open until the game has been completed."
CANCEL = "If the game is canceled entirely, with no make-up game, this market will resolve 50-50."


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True
    ).encode("ascii")


def flat(value):
    if not isinstance(value, str):
        raise ValueError("score rule text is missing")
    return re.sub(r"(?<=\d) (?=AM|PM)", "", " ".join(value.split()))


def compile_market(market, *, teams, title, appointment):
    """Validate the complete declared rule template without consulting prices."""
    if any(
        market.get(key) is not value
        for key, value in {
            "active": True,
            "closed": False,
            "acceptingOrders": True,
            "enableOrderBook": True,
        }.items()
    ):
        raise ValueError(
            "score market is not active and accepting orders in retained metadata"
        )
    identifier = market.get("id")
    if not isinstance(identifier, str) or not re.fullmatch(r"[0-9]{1,30}", identifier):
        raise ValueError("score market identity is invalid")
    condition = market.get("conditionId")
    if not isinstance(condition, str) or not re.fullmatch(r"0x[0-9a-f]{64}", condition):
        raise ValueError("score condition identity is invalid")
    try:
        outcomes = json.loads(market["outcomes"])
        tokens = json.loads(market["clobTokenIds"])
    except (ValueError, TypeError, KeyError):
        raise ValueError("score outcome representation differs") from None
    if (
        not isinstance(tokens, list)
        or len(tokens) != 2
        or len(set(tokens)) != 2
        or any(
            not isinstance(token, str)
            or not re.fullmatch(r"[1-9][0-9]{0,77}", token)
            or int(token) >= 2**256
            for token in tokens
        )
    ):
        raise ValueError("score token identity is invalid")
    question, kind = market.get("question"), market.get("sportsMarketType")
    first, second = teams
    ending = POSTPONE + " " + CANCEL
    if kind == "moneyline":
        if question != title or market.get("line") is not None:
            raise ValueError("score moneyline identity differs")
        expected_outcomes = [first, second]
        description = (
            f"In the upcoming NFL football game between {first} and {second}, scheduled for {appointment}: "
            f'This market will resolve to "{first}" if {first} win the game. '
            f'This market will resolve to "{second}" if {second} win the game. '
            "Overtime is included if played. If the game ends in a tie, this market will resolve 50-50. "
            + ending
        )
        rule = ScoreMarket(identifier, "winner")
    elif kind == "spreads":
        match = re.fullmatch(
            r"Spread: ("
            + "|".join(re.escape(team) for team in teams)
            + r") \(-([0-9]{1,3})\.5\)",
            str(question),
        )
        if match is None:
            raise ValueError("score spread grammar differs")
        favorite, integer = match.group(1), int(match.group(2))
        threshold, team = integer + 1, teams.index(favorite)
        other = teams[1 - team]
        if Decimal(str(market.get("line"))) != -(Decimal(integer) + Decimal("0.5")):
            raise ValueError("score spread line conflicts with its question")
        expected_outcomes = [favorite, other]
        description = (
            f"In the upcoming NFL football game between {first} and {second}, scheduled for {appointment}: "
            f'This market will resolve to "{favorite}" if the {favorite} win the game by {threshold} or more points. '
            f'Otherwise, this market will resolve to "{other}". Overtime is included if played. '
            + ending
        )
        rule = ScoreMarket(identifier, "margin", threshold, team)
    elif kind in {"totals", "team_totals"}:
        prefixes = (
            {title: None}
            if kind == "totals"
            else {f"{team} Team Total": index for index, team in enumerate(teams)}
        )
        matches = [
            (
                prefix,
                index,
                re.fullmatch(
                    re.escape(prefix) + r": O/U ([0-9]{1,3})\.5", str(question)
                ),
            )
            for prefix, index in prefixes.items()
        ]
        matches = [(index, match) for _, index, match in matches if match is not None]
        if len(matches) != 1:
            raise ValueError("score total grammar differs")
        team, match = matches[0]
        integer, threshold = int(match.group(1)), int(match.group(1)) + 1
        if Decimal(str(market.get("line"))) != Decimal(integer) + Decimal("0.5"):
            raise ValueError("score total line conflicts with its question")
        expected_outcomes = ["Over", "Under"]
        if team is None:
            description = (
                f"In the upcoming NFL game, scheduled for {appointment}: "
                f'This market will resolve to "Over" if the {first} and {second} combine to score {threshold} or more points in this game. '
                f'If the combined total is less than {threshold}, this market will resolve to "Under". '
                + ending
            )
            rule = ScoreMarket(identifier, "total", threshold)
        else:
            alias = teams[team]
            description = (
                f"In the upcoming NFL game between {first} and {second}, scheduled for {appointment}: "
                f'This market will resolve to "Over" if the {alias} score {threshold} or more points in this game. '
                f'If the {alias} score less than {threshold} points, this market will resolve to "Under". '
                f"This market is based solely on the {alias}'s score, not the combined score of both teams. "
                + ending
            )
            rule = ScoreMarket(identifier, "team_total", threshold, team)
    else:
        raise ValueError("unsupported score market kind")
    if outcomes != expected_outcomes or flat(market.get("description")) != flat(
        description
    ):
        raise ValueError("complete score outcome or settlement rule differs")
    return rule, {
        "market_id": identifier,
        "condition_id": condition,
        "tokens": tokens,
        "outcomes": outcomes,
        "rule": asdict(rule),
        "description_sha256": digest(market["description"].encode("utf-8")),
        "resolution_source": market.get("resolutionSource"),
    }


def compile_event(event, *, appointment):
    """Consume only price-blind metadata selected before this function is called."""
    if event.get("active") is not True or event.get("closed") is not False:
        raise ValueError("score event deployment differs")
    teams = tuple(team["alias"] for team in event["teams"])
    if (
        len(teams) != 2
        or any(not isinstance(team, str) or not 0 < len(team) <= 80 for team in teams)
        or len(set(teams)) != 2
    ):
        raise ValueError("score event team identity differs")
    title = f"{teams[0]} vs. {teams[1]}"
    if event.get("title") != title:
        raise ValueError("score event title differs from team mapping")
    if (
        not isinstance(event.get("markets"), list)
        or not 1 <= len(event["markets"]) <= 160
    ):
        raise ValueError("score market population is missing or oversized")
    rules, bindings = [], []
    for market in sorted(event["markets"], key=lambda market: str(market["id"])):
        if set(market) - FIELDS:
            raise ValueError("price-blind compiler received undeclared fields")
        rule, binding = compile_market(
            market, teams=teams, title=title, appointment=appointment
        )
        rules.append(rule)
        bindings.append(binding)
    if len({binding["condition_id"] for binding in bindings}) != len(bindings) or len(
        {token for binding in bindings for token in binding["tokens"]}
    ) != 2 * len(bindings):
        raise ValueError("duplicate score conditions or tokens")
    return score_states(tuple(rules)), bindings


def _price(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (Decimal, int, str)):
        raise ValueError("score price representation differs")
    try:
        number = Decimal(value)
    except (ValueError, TypeError, ArithmeticError):
        raise ValueError("score price is not a finite unit quote") from None
    if (
        not number.is_finite()
        or not 0 <= number <= 1
        or not -12 <= number.as_tuple().exponent <= 0
        or len(number.as_tuple().digits) > 16
    ):
        raise ValueError("score price exceeds its finite precision scope")
    return Fraction(number)


def price_legs(markets, bindings):
    by_id = {str(market["id"]): market for market in markets}
    prices, rows = [], []
    for binding in bindings:
        market = by_id[binding["market_id"]]
        ask, bid = _price(market.get("bestAsk")), _price(market.get("bestBid"))
        ask = ask if ask else None
        if ask is not None and bid is not None and bid > ask:
            raise ValueError("score quote is crossed")
        # Zero ask is missing available inventory, never a free contract.
        values = (
            ask,
            1 - bid if bid is not None and bid < 1 else None,
        )
        for side, price in enumerate(values):
            prices.append(price)
            rows.append(
                {
                    "market_id": binding["market_id"],
                    "token_id": binding["tokens"][side],
                    "outcome": binding["outcomes"][side],
                    "proxy_price": None if price is None else str(price),
                    "price_source": "bestAsk" if side == 0 else "1-bestBid",
                }
            )
    return tuple(prices), rows


def _path(value):
    if not isinstance(value, str) or Path(value).is_absolute():
        raise ValueError("score paths must be repository relative")
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def _write(path, value):
    with path.open("x", encoding="ascii") as output:
        output.write(canonical(value).decode("ascii") + "\n")
        output.flush()
        os.fsync(output.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    contract_path = _path(args.contract)
    contract = json.loads(contract_path.read_bytes())
    body = dict(contract)
    expected = body.pop("contract_sha256")
    if (
        digest(canonical(body)) != expected
        or contract["contract_path"] != args.contract
    ):
        raise ValueError("score contract identity differs")
    if set(contract["implementations"]) != {
        "src/simple_ai_trading/score_cover.py",
        "tools/screen_retained_score_cover.py",
    }:
        raise ValueError("score implementation binding is incomplete")
    for name, expected_hash in contract["implementations"].items():
        if digest(_path(name).read_bytes()) != expected_hash:
            raise ValueError("score implementation changed after freeze")
    if contract["numerical_runtime"] != {
        name: version(name) for name in ("numpy", "scipy")
    }:
        raise ValueError("score numerical runtime differs from its frozen identity")
    if contract["policy"] != {
        "eligible_kinds": sorted(KINDS),
        "solver": "highs-ds",
        "solver_seconds": 20,
        "rational_denominator_limit": 1000000,
        "common_cancellation_payoff": "1/2",
        "interpretation": "consumed-response exploratory proxy system only",
        "book_fee_account_or_order_authority": False,
    }:
        raise ValueError("score policy differs from the reviewed implementation")
    source = _path(contract["source_path"])
    with source.open("rb") as stream:
        raw = stream.read(3 * 1024 * 1024 + 1)
    if len(raw) > 3 * 1024 * 1024 or digest(raw) != contract["source_sha256"]:
        raise ValueError("score retained source identity differs")
    output, journal = _path(contract["output_path"]), _path(contract["journal_path"])
    if (
        output.exists()
        or journal.exists()
        or not output.parent.is_dir()
        or not journal.parent.is_dir()
    ):
        raise ValueError("score one-use output is consumed or unavailable")
    record = {
        "phase": "intent",
        "contract_sha256": expected,
        "at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not args.preflight:
        _write(journal, record)
    try:
        payload = json.loads(raw, parse_float=Decimal)
        selected = [
            event
            for event in payload["events"]
            if str(event.get("id")) == contract["event_id"]
        ]
        if len(selected) != 1 or selected[0]["slug"] != contract["event_slug"]:
            raise ValueError("score selected event identity differs")
        event = selected[0]
        native = [
            market
            for market in event["markets"]
            if market.get("sportsMarketType") in KINDS
        ]
        metadata = {
            key: event[key] for key in ["id", "title", "teams", "active", "closed"]
        }
        metadata["markets"] = [
            {key: value for key, value in market.items() if key in FIELDS}
            for market in native
        ]
        system, bindings = compile_event(metadata, appointment=contract["appointment"])
        preflight = {
            "market_count": len(bindings),
            "payoff_state_count": len(system.rows),
            "score_bound": system.bound,
            "matrix_sha256": system.matrix_sha256,
            "economic_fields_consulted": False,
        }
        if args.preflight:
            print(json.dumps(preflight))
            return
        prices, legs = price_legs(native, bindings)
        result = solve_score_cover(system, prices)
        report = {
            "schema_version": "retained-score-cover-v1",
            "contract_sha256": expected,
            "source_sha256": contract["source_sha256"],
            "event_id": contract["event_id"],
            "event_slug": contract["event_slug"],
            "rule_bindings": bindings,
            "state_system": {
                **preflight,
                "economic_fields_consulted": True,
                "scalar_limit": system.scalar_limit,
                "margin_limit": system.margin_limit,
            },
            "legs": legs,
            "prices_complete": result.complete_prices,
            "cost_lower": str(result.cost_lower),
            "cost_upper": str(result.cost_upper),
            "primal": [
                {"leg_index": index, "quantity": str(weight)}
                for index, weight in result.primal
            ],
            "dual": [
                {
                    "state_index": index,
                    "representative_score": system.representatives[index],
                    "weight": str(weight),
                    "payoff_row_sha256": digest(system.rows[index]),
                }
                for index, weight in result.dual
            ],
            "exact_certificate_verified": True,
            "proxy_system_no_strict_cover": result.complete_prices
            and result.cost_lower >= 1,
            "exploratory_subfloor_cover": result.cost_upper < 1,
            "new_network_requests": 0,
            "new_independent_validation": False,
            "accepted_edge": False,
            "profitability_claim": False,
            "book_fee_account_or_order_authority": False,
        }
        report["result_sha256"] = digest(canonical(report))
        _write(output, report)
        terminal = {
            "phase": "completed",
            "contract_sha256": expected,
            "result_sha256": report["result_sha256"],
            "at_utc": datetime.now(timezone.utc).isoformat(),
        }
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in [
                        "cost_lower",
                        "cost_upper",
                        "prices_complete",
                        "proxy_system_no_strict_cover",
                        "exploratory_subfloor_cover",
                        "result_sha256",
                    ]
                }
            )
        )
    except BaseException:
        terminal = {
            "phase": "failed",
            "contract_sha256": expected,
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "retry_permitted": False,
        }
        raise
    finally:
        if not args.preflight:
            with journal.open("a", encoding="ascii") as stream:
                stream.write(canonical(terminal).decode("ascii") + "\n")
                stream.flush()
                os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
