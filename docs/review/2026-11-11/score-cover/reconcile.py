"""Audit the retained certificate with standard-library rational arithmetic only."""

from fractions import Fraction
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def read(path, maximum):
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, "audit input exceeds byte budget")
    return raw


def relative(value):
    require(
        isinstance(value, str) and not Path(value).is_absolute(),
        "audit path is not relative",
    )
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def fraction(value):
    require(
        isinstance(value, str) and len(value) <= 4096,
        "audit fraction is missing or oversized",
    )
    result = Fraction(value)
    require(
        max(result.numerator.bit_length(), result.denominator.bit_length()) <= 4096,
        "audit fraction exceeds integer budget",
    )
    return result


def native_fraction(value):
    require(
        not isinstance(value, bool) and isinstance(value, (Decimal, int, str)),
        "native numeric type differs",
    )
    number = Decimal(value)
    require(
        number.is_finite()
        and abs(number) <= 128
        and -12 <= number.as_tuple().exponent <= 0
        and len(number.as_tuple().digits) <= 16,
        "native numeric value exceeds audit scope",
    )
    return Fraction(number)


def audit(result, contract):
    """Verify the fixed admitted rule program; this is not new settlement research."""
    require(
        hashlib.sha256(
            canonical({k: v for k, v in contract.items() if k != "contract_sha256"})
        ).hexdigest()
        == contract["contract_sha256"],
        "contract seal differs",
    )
    seal = result["result_sha256"]
    require(
        hashlib.sha256(
            canonical({k: v for k, v in result.items() if k != "result_sha256"})
        ).hexdigest()
        == seal,
        "result seal differs",
    )
    require(
        result["contract_sha256"] == contract["contract_sha256"],
        "contract binding differs",
    )
    raw = read(relative(contract["source_path"]), 3 * 1024 * 1024)
    require(
        hashlib.sha256(raw).hexdigest()
        == contract["source_sha256"]
        == result["source_sha256"],
        "source binding differs",
    )
    payload = json.loads(raw, parse_float=Decimal)
    event = [row for row in payload["events"] if str(row["id"]) == contract["event_id"]]
    require(
        len(event) == 1 and event[0]["slug"] == contract["event_slug"],
        "event selection differs",
    )
    native = {
        str(row["id"]): row
        for row in event[0]["markets"]
        if row.get("sportsMarketType")
        in {"moneyline", "spreads", "totals", "team_totals"}
    }
    bindings = result["rule_bindings"]
    require(
        [row["market_id"] for row in bindings] == sorted(native),
        "admitted population differs",
    )
    prices = []
    teams = [team["alias"] for team in event[0]["teams"]]
    for binding in bindings:
        market = native[binding["market_id"]]
        require(
            binding["description_sha256"]
            == hashlib.sha256(market["description"].encode()).hexdigest(),
            "rule text differs",
        )
        require(
            binding["condition_id"] == market["conditionId"]
            and binding["outcomes"] == json.loads(market["outcomes"])
            and binding["tokens"] == json.loads(market["clobTokenIds"]),
            "condition or outcome binding differs",
        )
        native_kind = market["sportsMarketType"]
        if native_kind == "moneyline":
            expected_rule = dict(
                identifier=binding["market_id"], kind="winner", threshold=0, team=0
            )
        else:
            line = native_fraction(market["line"])
            threshold = abs(line) + Fraction(1, 2)
            require(threshold.denominator == 1, "native threshold is not a half point")
            if native_kind == "spreads":
                kind, team = "margin", teams.index(binding["outcomes"][0])
            elif native_kind == "team_totals":
                kind = "team_total"
                candidates = [
                    index
                    for index, name in enumerate(teams)
                    if market["question"].startswith(name + " Team Total:")
                ]
                require(len(candidates) == 1, "native team total cannot be assigned")
                team = candidates[0]
            else:
                kind, team = "total", 0
            expected_rule = dict(
                identifier=binding["market_id"],
                kind=kind,
                threshold=int(threshold),
                team=team,
            )
        require(
            binding["rule"] == expected_rule,
            "compiled rule differs from native market type, line or team",
        )
        require(
            binding["resolution_source"] == market.get("resolutionSource"),
            "resolution-source binding differs",
        )
        ask, bid = market.get("bestAsk"), market.get("bestBid")
        ask = None if ask in (None, "", 0, "0") else native_fraction(ask)
        bid = None if bid in (None, "") else native_fraction(bid)
        require(
            (ask is None or 0 < ask <= 1) and (bid is None or 0 <= bid <= 1),
            "native quote is outside unit interval",
        )
        prices.extend((ask, 1 - bid if bid is not None and bid < 1 else None))
    require(len(prices) == len(result["legs"]), "leg population differs")
    for index, (price, leg) in enumerate(zip(prices, result["legs"], strict=True)):
        binding = bindings[index // 2]
        require(
            leg["market_id"] == binding["market_id"]
            and leg["token_id"] == binding["tokens"][index % 2]
            and leg["outcome"] == binding["outcomes"][index % 2],
            "leg identity differs",
        )
        require(
            leg["proxy_price"] == (None if price is None else str(price)),
            "retained proxy differs",
        )

    rules = [binding["rule"] for binding in bindings]
    scalar = max(
        (
            rule["threshold"]
            for rule in rules
            if rule["kind"] in {"total", "team_total"}
        ),
        default=0,
    )
    margin = max(
        (rule["threshold"] for rule in rules if rule["kind"] == "margin"), default=1
    )
    bound = scalar + margin + 2
    require(
        0 <= scalar <= 128 and 1 <= margin <= 64 and len(rules) <= 160,
        "audit rule budget differs",
    )
    require(result["state_system"]["score_bound"] == bound, "finite projection differs")
    states = {}
    for first in range(bound + 1):
        for second in range(bound + 1):
            row = []
            for rule in rules:
                own, other = (first, second) if rule["team"] == 0 else (second, first)
                kind, threshold = rule["kind"], rule["threshold"]
                if kind == "winner":
                    payout = 1 if own == other else 2 if own > other else 0
                elif kind == "margin":
                    payout = 2 if own - other >= threshold else 0
                elif kind == "total":
                    payout = 2 if first + second >= threshold else 0
                elif kind == "team_total":
                    payout = 2 if own >= threshold else 0
                else:
                    raise ValueError("audit rule kind differs")
                row.extend((payout, 2 - payout))
            states.setdefault(bytes(row), (first, second))
    states.setdefault(bytes([1] * len(prices)), None)
    rows = sorted(states)
    require(
        len(rows) == result["state_system"]["payoff_state_count"]
        and hashlib.sha256(b"".join(rows)).hexdigest()
        == result["state_system"]["matrix_sha256"],
        "complete payoff matrix differs",
    )
    primal = {
        item["leg_index"]: fraction(item["quantity"]) for item in result["primal"]
    }
    dual = {item["state_index"]: fraction(item["weight"]) for item in result["dual"]}
    require(
        len(primal) == len(result["primal"]) and len(dual) == len(result["dual"]),
        "duplicate witness index",
    )
    require(
        all(
            type(index) is int
            and 0 <= index < len(prices)
            and value > 0
            and prices[index] is not None
            for index, value in primal.items()
        ),
        "invalid primal witness",
    )
    require(
        all(
            type(index) is int and 0 <= index < len(rows) and value > 0
            for index, value in dual.items()
        ),
        "invalid dual witness",
    )
    payoffs = [
        sum((weight * row[index] for index, weight in primal.items()), Fraction()) / 2
        for row in rows
    ]
    require(min(payoffs) >= 1, "portfolio misses a score or common-void state")
    for item in result["dual"]:
        row = rows[item["state_index"]]
        require(
            hashlib.sha256(row).hexdigest() == item["payoff_row_sha256"],
            "dual state digest differs",
        )
        representative = states[row]
        require(
            item["representative_score"]
            == (None if representative is None else list(representative)),
            "dual representative differs",
        )
    for column, price in enumerate(prices):
        if price is not None:
            require(
                sum(
                    (weight * rows[index][column] for index, weight in dual.items()),
                    Fraction(),
                )
                / 2
                <= price,
                "dual exceeds a priced leg",
            )
    lower = sum(dual.values(), Fraction())
    upper = sum(
        (prices[index] * weight for index, weight in primal.items()), Fraction()
    )
    require(
        lower <= upper
        and str(lower) == result["cost_lower"]
        and str(upper) == result["cost_upper"],
        "exact objective bounds differ",
    )
    require(
        type(result["prices_complete"]) is bool
        and result["prices_complete"] == all(price is not None for price in prices),
        "price coverage differs",
    )
    require(
        result["proxy_system_no_strict_cover"]
        is (result["prices_complete"] and lower >= 1)
        and result["exploratory_subfloor_cover"] is (upper < 1),
        "bound classification differs",
    )
    require(
        result["accepted_edge"] is False
        and result["profitability_claim"] is False
        and result["book_fee_account_or_order_authority"] is False
        and result["new_network_requests"] == 0
        and result["new_independent_validation"] is False,
        "exploratory authority boundary differs",
    )
    return {
        "source_sha256": result["source_sha256"],
        "result_sha256": seal,
        "market_count": len(bindings),
        "leg_count": len(prices),
        "state_count": len(rows),
        "dual_support_count": len(dual),
        "cost_lower": str(lower),
        "cost_upper": str(upper),
        "payoff_minimum": str(min(payoffs)),
        "payoff_maximum": str(max(payoffs)),
        "all_primal_state_and_dual_leg_inequalities_verified_exactly": True,
        "numpy_or_scipy_imported": False,
        "new_network_requests": 0,
        "new_independent_validation": False,
    }


if __name__ == "__main__":
    result = json.loads(read(relative(sys.argv[1]), 4 * 1024 * 1024))
    contract = json.loads(
        read(ROOT / "docs/review/2026-11-11/score-cover/contract.json", 64 * 1024)
    )
    print(json.dumps(audit(result, contract), sort_keys=True))
