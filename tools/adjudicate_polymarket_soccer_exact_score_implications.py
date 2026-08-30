"""Enumerate retained soccer exact-score to match-result payoff implications."""

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
    if contract.get("schema_version") not in {
        "polymarket-soccer-exact-score-implication-contract-v1",
        "polymarket-soccer-exact-score-implication-contract-v2",
        "polymarket-soccer-exact-score-implication-contract-v3",
    }:
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


def _market_summary(market: dict[str, Any]) -> dict[str, Any]:
    return {
        key: market.get(key)
        for key in (
            "id",
            "question",
            "description",
            "active",
            "closed",
            "acceptingOrders",
            "sportsMarketType",
            "line",
            "outcomes",
            "outcomePrices",
            "clobTokenIds",
            "conditionId",
            "enableOrderBook",
            "negRisk",
            "feesEnabled",
            "feeSchedule",
            "takerBaseFee",
            "orderMinSize",
            "orderPriceMinTickSize",
        )
    }


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
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("retained payload events must be a list")

    event_contract = _mapping(contract["event_contract"], name="event contract")
    main_event = next(
        (row for row in events if row.get("slug") == event_contract["match_slug"]),
        None,
    )
    exact_event = next(
        (row for row in events if row.get("slug") == event_contract["exact_score_slug"]),
        None,
    )
    if not isinstance(main_event, Mapping) or not isinstance(exact_event, Mapping):
        raise ValueError("exact retained event pair is absent")
    main_markets = main_event.get("markets")
    exact_markets = exact_event.get("markets")
    if not isinstance(main_markets, list) or not isinstance(exact_markets, list):
        raise ValueError("event markets must be lists")
    if len(main_markets) != 3 or len(exact_markets) != 17:
        raise ValueError("retained event market cardinality changed")

    result_market_by_label = {
        str(row.get("groupItemTitle")): _mapping(row, name="result market")
        for row in main_markets
    }
    home = str(event_contract["home_team"])
    away = str(event_contract["away_team"])
    draw_label = f"Draw ({home} vs. {away})"
    if set(result_market_by_label) != {home, away, draw_label}:
        raise ValueError("full-game result labels changed")

    required_common = event_contract["required_common_rule_fragments"]
    if not isinstance(required_common, list):
        raise ValueError("required common fragments must be a list")
    relations: list[dict[str, Any]] = []
    for exact_market_value in exact_markets:
        exact_market = _mapping(exact_market_value, name="exact-score market")
        label = str(exact_market.get("groupItemTitle"))
        if label == "Any Other Score":
            continue
        match = SCORE_RE.fullmatch(label)
        if match is None or match.group("home") != home or match.group("away") != away:
            raise ValueError(f"unparseable exact score label: {label}")
        home_score = int(match.group("h"))
        away_score = int(match.group("a"))
        implied_label = (
            home
            if home_score > away_score
            else away
            if away_score > home_score
            else draw_label
        )
        result_market = result_market_by_label[implied_label]
        descriptions = [
            str(exact_market.get("description") or ""),
            str(result_market.get("description") or ""),
        ]
        if not all(fragment in description for fragment in required_common for description in descriptions):
            raise ValueError("common resolution horizon or source fragment changed")
        if home_score == away_score == 0:
            cancellation_fragments = event_contract["required_zero_zero_cancellation_fragments"]
            if not all(
                fragment in description
                for description, fragment in zip(descriptions, cancellation_fragments, strict=True)
            ):
                raise ValueError("zero-zero cancellation alignment changed")
        exact_prices = [
            Decimal(str(value))
            for value in _decoded_list(
                exact_market.get("outcomePrices"), name="exact prices"
            )
        ]
        result_prices = [
            Decimal(str(value))
            for value in _decoded_list(
                result_market.get("outcomePrices"), name="result prices"
            )
        ]
        exact_tokens = _decoded_list(exact_market.get("clobTokenIds"), name="exact tokens")
        result_tokens = _decoded_list(result_market.get("clobTokenIds"), name="result tokens")
        if _decoded_list(exact_market.get("outcomes"), name="exact outcomes") != ["Yes", "No"]:
            raise ValueError("exact-score outcomes changed")
        if _decoded_list(result_market.get("outcomes"), name="result outcomes") != ["Yes", "No"]:
            raise ValueError("match-result outcomes changed")
        midpoint_diagnostic_sum = exact_prices[1] + result_prices[0]
        exact_yes_best_bid = Decimal(str(exact_market.get("bestBid")))
        result_yes_best_ask = Decimal(str(result_market.get("bestAsk")))
        if not (
            Decimal("0") <= exact_yes_best_bid <= Decimal("1")
            and Decimal("0") <= result_yes_best_ask <= Decimal("1")
        ):
            raise ValueError("retained side-specific price is outside [0, 1]")
        exact_no_ask_proxy = Decimal("1") - exact_yes_best_bid
        rejection_proxy_sum = exact_no_ask_proxy + result_yes_best_ask
        relations.append(
            {
                "exact_score": f"{home_score}-{away_score}",
                "implied_result": implied_label,
                "exact_score_market_id": str(exact_market["id"]),
                "exact_score_no_token_id": str(exact_tokens[1]),
                "exact_score_no_price_pUSD": _decimal_text(exact_prices[1]),
                "result_market_id": str(result_market["id"]),
                "result_yes_token_id": str(result_tokens[0]),
                "result_yes_price_pUSD": _decimal_text(result_prices[0]),
                "midpoint_diagnostic_sum_pUSD": _decimal_text(
                    midpoint_diagnostic_sum
                ),
                "exact_score_yes_best_bid_pUSD": _decimal_text(
                    exact_yes_best_bid
                ),
                "exact_score_no_ask_rejection_proxy_pUSD": _decimal_text(
                    exact_no_ask_proxy
                ),
                "result_yes_best_ask_pUSD": _decimal_text(result_yes_best_ask),
                "rejection_proxy_sum_pUSD": _decimal_text(rejection_proxy_sum),
                "guaranteed_common_rule_floor_pUSD": "1",
                "optimistic_proxy_headroom_pUSD": _decimal_text(
                    Decimal("1") - rejection_proxy_sum
                ),
                "passes_strict_side_specific_rejection_gate": (
                    rejection_proxy_sum < 1
                ),
            }
        )
    if len(relations) != 16:
        raise ValueError("exact implication count changed")
    relations.sort(
        key=lambda row: (
            Decimal(row["rejection_proxy_sum_pUSD"]),
            int(row["exact_score_market_id"]),
        )
    )
    candidates = [
        row
        for row in relations
        if row["passes_strict_side_specific_rejection_gate"]
    ]
    best = candidates[0] if candidates else None
    active_accepting_markets: list[dict[str, Any]] = []
    if best is not None:
        selected_ids = {best["exact_score_market_id"], best["result_market_id"]}
        active_accepting_markets = [
            _market_summary(_mapping(row, name="selected market"))
            for row in [*main_markets, *exact_markets]
            if str(row.get("id")) in selected_ids
        ]

    output_path = _root_path(contract["output_path"])
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite result: {output_path}")
    result: dict[str, Any] = {
        "schema_version": "polymarket-soccer-exact-score-implication-result-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {"path": args.contract, "sha256": contract["contract_sha256"]},
        "retained_source": source,
        "screen": {
            "event_pair": [event_contract["match_slug"], event_contract["exact_score_slug"]],
            "tested_relation_count": len(relations),
            "strict_displayed_candidate_count": len(candidates),
            "relations": relations,
            "best_candidate": best,
            "gamma_role": "rejection_only_never_acceptance_or_promotion_evidence",
            "price_gate": "YES bestAsk plus conservative NO ask proxy 1 minus YES bestBid; outcomePrices are midpoint diagnostics only",
        },
        "discovery": {"active_accepting_markets": active_accepting_markets},
        "payoff_proof": {
            "market_direction_forecast_required": False,
            "invariant": "For each explicit score S implying full-game result R, NO(S) plus YES(R) pays at least one pUSD under the retained common regulation, source, postponement, and cancellation rules.",
            "best_candidate_states": [
                {"state": "zero_zero", "exact_score_no": "0", "draw_yes": "1", "package_payout_pUSD": "1"},
                {"state": "other_draw", "exact_score_no": "1", "draw_yes": "1", "package_payout_pUSD": "2"},
                {"state": "non_draw", "exact_score_no": "1", "draw_yes": "0", "package_payout_pUSD": "1"},
                {"state": "canceled_no_makeup", "exact_score_no": "0", "draw_yes": "1", "package_payout_pUSD": "1"},
            ],
            "unqualified_risk": "Independent condition disputes, inconsistent resolutions, oracle error, settlement delay, and non-atomic execution remain unqualified.",
        },
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "next_action": (
                "freeze_one_exact_two_token_current_book_batch_for_the_deterministic_best_candidate"
                if best is not None
                else "stop_without_book_or_fee_requests"
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
    write_bytes_atomic(
        output_path,
        (_canonical_json(result) + "\n").encode("ascii"),
    )
    print(
        json.dumps(
            {
                "tested_relation_count": len(relations),
                "strict_displayed_candidate_count": len(candidates),
                "best_candidate": best,
                "network_requests": 0,
                "result_sha256": result["result_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
