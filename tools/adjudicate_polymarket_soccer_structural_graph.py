"""Enumerate retained soccer implication and monotone-payoff relations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
SCORE_RE = re.compile(r"^(?P<home>.+) (?P<h>\d+) - (?P<a>\d+) (?P<away>.+)$")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: Mapping[str, object], *, field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(_canonical_json(body).encode("ascii"))


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _decoded_list(value: object, *, name: str) -> list[Any]:
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, list):
        raise ValueError(f"{name} must be a list")
    return decoded


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if contract.get("schema_version") != "polymarket-soccer-structural-graph-contract-v1":
        raise ValueError("unexpected contract schema")
    if contract_path != _root_path(contract["contract_path"]):
        raise ValueError("contract path mismatch")
    if _canonical_hash(contract, field="contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise ValueError("contract hash mismatch")
    frozen = datetime.fromisoformat(
        str(contract["frozen_at_utc"]).replace("Z", "+00:00")
    )
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("contract timestamp is invalid or future")
    implementation = _root_path(contract["implementation"]["path"])
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise ValueError("implementation hash mismatch")


def _description(market: Mapping[str, Any]) -> str:
    return " ".join(str(market.get("description") or "").split())


def _require_common_horizon_and_source(*markets: Mapping[str, Any]) -> None:
    for market in markets:
        description = _description(market)
        if not (
            (
                "90 minutes of regular play plus stoppage time" in description
                or "90 minutes of regulation plus stoppage time" in description
            )
            and "The primary resolution source for this market is the official statistics of the event"
            in description
        ):
            raise ValueError(f"full-game horizon or source changed: {market.get('id')}")


def _require_active(market: Mapping[str, Any]) -> None:
    if not (
        market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
    ):
        raise ValueError(f"retained market was not active and accepting: {market.get('id')}")


def _yes_ask(market: Mapping[str, Any]) -> Decimal | None:
    value = market.get("bestAsk")
    return None if value is None else Decimal(str(value))


def _no_ask_proxy(market: Mapping[str, Any]) -> Decimal | None:
    value = market.get("bestBid")
    return None if value is None else Decimal("1") - Decimal(str(value))


def _market_identity(market: Mapping[str, Any], *, side: str) -> dict[str, Any]:
    outcomes = _decoded_list(market.get("outcomes"), name="outcomes")
    tokens = _decoded_list(market.get("clobTokenIds"), name="tokens")
    if len(outcomes) != 2 or len(tokens) != 2:
        raise ValueError(f"market is not binary: {market.get('id')}")
    index = 0 if side == "first_outcome" else 1
    return {
        "market_id": str(market["id"]),
        "condition_id": str(market["conditionId"]),
        "question": str(market["question"]),
        "outcome": str(outcomes[index]),
        "token_id": str(tokens[index]),
        "side_specific_price_kind": (
            "bestAsk" if index == 0 else "one_minus_first_outcome_bestBid"
        ),
    }


def _relation(
    *,
    family: str,
    base_slug: str,
    first_market: Mapping[str, Any],
    first_side: str,
    second_market: Mapping[str, Any],
    second_side: str,
    proof: str,
    cancellation_floor_pUSD: str,
) -> dict[str, Any]:
    _require_active(first_market)
    _require_active(second_market)
    first_price = (
        _yes_ask(first_market)
        if first_side == "first_outcome"
        else _no_ask_proxy(first_market)
    )
    second_price = (
        _yes_ask(second_market)
        if second_side == "first_outcome"
        else _no_ask_proxy(second_market)
    )
    prices_complete = first_price is not None and second_price is not None
    proxy_sum = first_price + second_price if prices_complete else None
    return {
        "family": family,
        "base_slug": base_slug,
        "legs": [
            _market_identity(first_market, side=first_side),
            _market_identity(second_market, side=second_side),
        ],
        "payoff_proof": proof,
        "common_rule_floor_pUSD": "1",
        "cancellation_floor_pUSD": cancellation_floor_pUSD,
        "side_specific_prices_complete": prices_complete,
        "rejection_proxy_sum_pUSD": (
            _decimal_text(proxy_sum) if proxy_sum is not None else None
        ),
        "optimistic_proxy_headroom_pUSD": (
            _decimal_text(Decimal("1") - proxy_sum)
            if proxy_sum is not None
            else None
        ),
        "passes_strict_side_specific_rejection_gate": bool(
            proxy_sum is not None and proxy_sum < 1
        ),
    }


def _main_team_markets(event: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    title = str(event["title"])
    parts = title.split(" vs. ", 1)
    if len(parts) != 2:
        raise ValueError(f"unexpected soccer match title: {title}")
    home, away = parts
    markets = event.get("markets")
    if not isinstance(markets, list) or len(markets) != 3:
        raise ValueError(f"main soccer event cardinality changed: {event.get('slug')}")
    by_label = {str(row.get("groupItemTitle")): _mapping(row, name="main market") for row in markets}
    expected = {home, away, f"Draw ({title})"}
    if set(by_label) != expected:
        raise ValueError(f"main soccer labels changed: {event.get('slug')}")
    return home, away, by_label


def _enumerate(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("retained events must be a list")
    by_slug = {str(event["slug"]): _mapping(event, name="event") for event in events}
    relations: list[dict[str, Any]] = []
    counts = {
        "main_exact_score_pairs": 0,
        "main_first_to_score_pairs": 0,
        "main_more_markets_pairs": 0,
    }

    for slug in sorted(by_slug):
        if not slug.endswith("-exact-score"):
            continue
        base = slug[: -len("-exact-score")]
        main = by_slug.get(base)
        if main is None:
            continue
        counts["main_exact_score_pairs"] += 1
        home, away, result_markets = _main_team_markets(main)
        draw_label = f"Draw ({main['title']})"
        exact_markets = by_slug[slug].get("markets")
        if not isinstance(exact_markets, list) or len(exact_markets) != 17:
            raise ValueError(f"exact-score cardinality changed: {slug}")
        for exact_value in exact_markets:
            exact = _mapping(exact_value, name="exact-score market")
            label = str(exact.get("groupItemTitle"))
            if label == "Any Other Score":
                continue
            match = SCORE_RE.fullmatch(label)
            if match is None or match.group("home") != home or match.group("away") != away:
                raise ValueError(f"unparseable exact score: {label}")
            home_score = int(match.group("h"))
            away_score = int(match.group("a"))
            implied = (
                home
                if home_score > away_score
                else away
                if away_score > home_score
                else draw_label
            )
            result_market = result_markets[implied]
            _require_common_horizon_and_source(exact, result_market)
            exact_description = _description(exact)
            result_description = _description(result_market)
            if not (
                'canceled with no make-up game, the market resolves to "0-0."'
                in exact_description
                and (
                    'this market will resolve to "Yes".' in result_description
                    if implied == draw_label
                    else 'this market will resolve "No".' in result_description
                )
            ):
                raise ValueError(f"exact-result cancellation rules changed: {label}")
            relations.append(
                _relation(
                    family="exact_score_implies_full_game_result",
                    base_slug=base,
                    first_market=exact,
                    first_side="second_outcome",
                    second_market=result_market,
                    second_side="first_outcome",
                    proof=f"score {home_score}-{away_score} implies {implied}",
                    cancellation_floor_pUSD="1",
                )
            )

    for slug in sorted(by_slug):
        if not slug.endswith("-first-to-score"):
            continue
        base = slug[: -len("-first-to-score")]
        main = by_slug.get(base)
        if main is None:
            continue
        counts["main_first_to_score_pairs"] += 1
        _, _, result_markets = _main_team_markets(main)
        draw = next(
            market
            for label, market in result_markets.items()
            if label.startswith("Draw (")
        )
        first_markets = by_slug[slug].get("markets")
        if not isinstance(first_markets, list) or len(first_markets) != 3:
            raise ValueError(f"first-to-score cardinality changed: {slug}")
        neither = next(
            _mapping(row, name="first-to-score market")
            for row in first_markets
            if row.get("groupItemTitle") == "Neither"
        )
        _require_common_horizon_and_source(neither, draw)
        if not (
            'canceled entirely, with no make-up game, this market will resolve to "Neither"'
            in _description(neither)
            and 'canceled entirely, with no make-up game, this market will resolve to "Yes"'
            in _description(draw)
        ):
            raise ValueError(f"neither-draw cancellation rules changed: {base}")
        relations.append(
            _relation(
                family="neither_first_to_score_implies_full_game_draw",
                base_slug=base,
                first_market=neither,
                first_side="second_outcome",
                second_market=draw,
                second_side="first_outcome",
                proof="if neither team scores first the full-game score is 0-0 and therefore a draw",
                cancellation_floor_pUSD="1",
            )
        )

    for slug in sorted(by_slug):
        if not slug.endswith("-more-markets"):
            continue
        base = slug[: -len("-more-markets")]
        main = by_slug.get(base)
        if main is None:
            continue
        counts["main_more_markets_pairs"] += 1
        _, _, result_markets = _main_team_markets(main)
        draw = next(
            market
            for label, market in result_markets.items()
            if label.startswith("Draw (")
        )
        markets = by_slug[slug].get("markets")
        if not isinstance(markets, list):
            raise ValueError(f"more-markets inventory changed: {slug}")
        totals = sorted(
            (
                _mapping(row, name="total market")
                for row in markets
                if row.get("sportsMarketType") == "totals"
            ),
            key=lambda row: Decimal(str(row["line"])),
        )
        if len(totals) < 2:
            raise ValueError(f"full-game total ladder is incomplete: {slug}")
        for total in totals:
            _require_common_horizon_and_source(total)
            description = _description(total)
            if not (
                "canceled entirely, with no make-up game" in description
                and "resolve 50" in description
            ):
                raise ValueError(f"total cancellation rule changed: {total.get('id')}")
        for lower_index, lower in enumerate(totals):
            for higher in totals[lower_index + 1 :]:
                relations.append(
                    _relation(
                        family="full_game_total_monotone",
                        base_slug=base,
                        first_market=lower,
                        first_side="first_outcome",
                        second_market=higher,
                        second_side="second_outcome",
                        proof=f"Over {lower['line']} plus Under {higher['line']} covers every integer full-game total",
                        cancellation_floor_pUSD="1",
                    )
                )
        btts = next(
            (
                _mapping(row, name="BTTS market")
                for row in markets
                if row.get("sportsMarketType") == "both_teams_to_score"
            ),
            None,
        )
        over_15 = next(
            (row for row in totals if Decimal(str(row["line"])) == Decimal("1.5")),
            None,
        )
        over_05 = next(
            (row for row in totals if Decimal(str(row["line"])) == Decimal("0.5")),
            None,
        )
        if btts is None or over_15 is None or over_05 is None:
            raise ValueError(f"BTTS or required total market is absent: {slug}")
        _require_common_horizon_and_source(btts, over_15, over_05, draw)
        if not (
            "canceled entirely, with no make-up game" in _description(btts)
            and "resolve 50" in _description(btts)
        ):
            raise ValueError(f"BTTS cancellation rule changed: {btts.get('id')}")
        relations.append(
            _relation(
                family="both_teams_to_score_implies_over_1_5",
                base_slug=base,
                first_market=btts,
                first_side="second_outcome",
                second_market=over_15,
                second_side="first_outcome",
                proof="both teams scoring implies at least two full-game goals",
                cancellation_floor_pUSD="1",
            )
        )
        relations.append(
            _relation(
                family="under_0_5_implies_full_game_draw",
                base_slug=base,
                first_market=over_05,
                first_side="first_outcome",
                second_market=draw,
                second_side="first_outcome",
                proof="the complement of Under 0.5 is Over 0.5; if Under 0.5 occurs the score is 0-0 and draw pays",
                cancellation_floor_pUSD="1.5",
            )
        )
    return relations, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    contract_path = _root_path(args.contract)
    contract = _mapping(json.loads(contract_path.read_bytes()), name="contract")
    _validate_contract(contract, contract_path)
    source = _mapping(contract["retained_source"], name="retained source")
    source_path = _root_path(source["path"])
    source_bytes = source_path.read_bytes()
    if _sha256(source_bytes) != source["sha256"]:
        raise ValueError("retained source hash mismatch")
    payload = _mapping(json.loads(source_bytes), name="retained payload")
    relations, population_counts = _enumerate(payload)
    relations.sort(
        key=lambda row: (
            row["rejection_proxy_sum_pUSD"] is None,
            Decimal(row["rejection_proxy_sum_pUSD"])
            if row["rejection_proxy_sum_pUSD"] is not None
            else Decimal("Infinity"),
            row["family"],
            row["base_slug"],
            row["legs"][0]["market_id"],
            row["legs"][1]["market_id"],
        )
    )
    price_complete = [row for row in relations if row["side_specific_prices_complete"]]
    candidates = [
        row for row in relations if row["passes_strict_side_specific_rejection_gate"]
    ]
    best = price_complete[0] if price_complete else None
    output_path = _root_path(contract["output_path"])
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite result: {output_path}")
    result: dict[str, Any] = {
        "schema_version": "polymarket-soccer-structural-graph-result-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {"path": args.contract, "sha256": contract["contract_sha256"]},
        "retained_source": source,
        "population": {
            **population_counts,
            "tested_relation_count": len(relations),
            "side_specific_price_complete_count": len(price_complete),
            "missing_side_specific_price_count": len(relations) - len(price_complete),
        },
        "screen": {
            "relation_families": sorted({row["family"] for row in relations}),
            "strict_side_specific_candidate_count": len(candidates),
            "best_price_complete_relation": best,
            "relations": relations,
            "price_gate": "first outcome bestAsk plus second outcome conservative proxy 1 minus first outcome bestBid; outcomePrices are never used",
            "gamma_role": "rejection_only_never_acceptance_or_promotion_evidence",
        },
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "next_action": (
                "freeze_one_current_book_batch_for_the_deterministic_best_candidate"
                if candidates
                else "stop_without_any_venue_request"
            ),
        },
        "authority": {
            "network_requests": 0,
            "credentials_used": False,
            "accounts_orders_funds_or_transactions": 0,
            "trading_authority": False,
        },
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result, field="result_sha256")
    write_bytes_atomic(output_path, (_canonical_json(result) + "\n").encode("ascii"))
    print(
        json.dumps(
            {
                "population": result["population"],
                "strict_side_specific_candidate_count": len(candidates),
                "best_price_complete_relation": best,
                "network_requests": 0,
                "result_sha256": result["result_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
