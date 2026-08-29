"""Screen one current MLB moneyline, spread, and totals payoff lattice."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Mapping

import requests

from simple_ai_trading.paper_execution import BookLevel
from simple_ai_trading.polymarket_fees import PolymarketFeeModel
from simple_ai_trading.storage import write_bytes_atomic


CONTRACT_SCHEMA = "polymarket-current-mlb-monotone-parity-contract-v1"
RESULT_SCHEMA = "polymarket-current-mlb-monotone-parity-result-v1"
EVENT_RESULT_HASH = "e274e3b05227022eb8c021fecdfa1a42e369ba30175ba906b24d6fd8459da80d"
EVENT_RAW_HASH = "ac40ee9716f14123bbeb8904cfaaf2e0e86d2b88ff0555f5a79f1df9afa91fd3"
EVENT_SLUG = "mlb-bos-nyy-2026-06-06"
BOOKS_URL = "https://clob.polymarket.com/books"
QUANTITY = Decimal("5")
ADVERSE_TICKS_PER_LEG = 2
EXPECTED_THRESHOLDS = {
    "BOS_margin": [1, 2, 3, 4],
    "NYY_margin": [1, 2, 3, 4, 5],
    "total_runs": [6, 7, 8, 9, 10, 11, 12],
}
TEAM_ABBREVIATIONS = {
    "Boston Red Sox": "BOS",
    "New York Yankees": "NYY",
}
_MARGIN_RE = re.compile(
    r'will resolve to "(?P<team>[^"]+)" if .* win .* by (?P<n>\d+) or more runs'
)
_TOTAL_RE = re.compile(
    r'will resolve to "Over" if .* combine to score (?P<n>\d+) or more runs'
)


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


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _json_list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be encoded JSON")
    return _list(json.loads(value), name=name)


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _parse_market(raw: object) -> dict[str, object] | None:
    market = _mapping(raw, name="embedded market")
    market_type = market.get("sportsMarketType")
    if market_type not in {"moneyline", "spreads", "totals"}:
        return None
    if not (
        market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
        and market.get("enableOrderBook") is True
        and market.get("negRisk") is False
    ):
        raise ValueError("scoped market is not active accepting standard CLOB")
    market_id = str(market.get("id") or "")
    condition_id = str(market.get("conditionId") or "").lower()
    outcomes = [
        str(value)
        for value in _json_list(market.get("outcomes"), name="market outcomes")
    ]
    tokens = [
        str(value)
        for value in _json_list(market.get("clobTokenIds"), name="market tokens")
    ]
    schedule = _mapping(market.get("feeSchedule"), name="fee schedule")
    if (
        not market_id.isdigit()
        or len(condition_id) != 66
        or not condition_id.startswith("0x")
        or len(outcomes) != 2
        or len(tokens) != 2
        or len(set(tokens)) != 2
        or market.get("feesEnabled") is not True
        or schedule
        != {"exponent": 1, "rate": 0.03, "takerOnly": True, "rebateRate": 0.25}
        or Decimal(str(market.get("orderMinSize"))) > QUANTITY
    ):
        raise ValueError(f"market {market_id} identity, fee, or size contract differs")
    tick = Decimal(str(market.get("orderPriceMinTickSize")))
    if tick not in {Decimal("0.01"), Decimal("0.001")}:
        raise ValueError(f"market {market_id} tick size differs")
    description = str(market.get("description") or "")
    if market_type == "moneyline":
        if outcomes != ["Boston Red Sox", "New York Yankees"]:
            raise ValueError("moneyline outcomes differ")
        return {
            "market_id": market_id,
            "condition_id": condition_id,
            "slug": market.get("slug"),
            "market_type": market_type,
            "line": None,
            "tick_size": _decimal_text(tick),
            "fee_schedule": schedule,
            "lattice_nodes": [
                {
                    "family": "BOS_margin",
                    "threshold": 1,
                    "positive_outcome": outcomes[0],
                    "positive_token": tokens[0],
                    "complement_outcome": outcomes[1],
                    "complement_token": tokens[1],
                },
                {
                    "family": "NYY_margin",
                    "threshold": 1,
                    "positive_outcome": outcomes[1],
                    "positive_token": tokens[1],
                    "complement_outcome": outcomes[0],
                    "complement_token": tokens[0],
                },
            ],
        }
    if market_type == "spreads":
        if outcomes[0] not in TEAM_ABBREVIATIONS or outcomes[1] in {
            outcomes[0],
            "Over",
            "Under",
        }:
            raise ValueError(f"spread market {market_id} outcomes differ")
        match = _MARGIN_RE.search(" ".join(description.split()))
        threshold = int(abs(Decimal(str(market.get("line")))) + Decimal("0.5"))
        if (
            match is None
            or match.group("team") != outcomes[0]
            or int(match.group("n")) != threshold
        ):
            raise ValueError(f"spread market {market_id} rule threshold differs")
        return {
            "market_id": market_id,
            "condition_id": condition_id,
            "slug": market.get("slug"),
            "market_type": market_type,
            "line": market.get("line"),
            "tick_size": _decimal_text(tick),
            "fee_schedule": schedule,
            "lattice_nodes": [
                {
                    "family": f"{TEAM_ABBREVIATIONS[outcomes[0]]}_margin",
                    "threshold": threshold,
                    "positive_outcome": outcomes[0],
                    "positive_token": tokens[0],
                    "complement_outcome": outcomes[1],
                    "complement_token": tokens[1],
                }
            ],
        }
    if outcomes != ["Over", "Under"]:
        raise ValueError(f"total market {market_id} outcomes differ")
    match = _TOTAL_RE.search(" ".join(description.split()))
    threshold = int(Decimal(str(market.get("line"))) + Decimal("0.5"))
    if match is None or int(match.group("n")) != threshold:
        raise ValueError(f"total market {market_id} rule threshold differs")
    return {
        "market_id": market_id,
        "condition_id": condition_id,
        "slug": market.get("slug"),
        "market_type": market_type,
        "line": market.get("line"),
        "tick_size": _decimal_text(tick),
        "fee_schedule": schedule,
        "lattice_nodes": [
            {
                "family": "total_runs",
                "threshold": threshold,
                "positive_outcome": outcomes[0],
                "positive_token": tokens[0],
                "complement_outcome": outcomes[1],
                "complement_token": tokens[1],
            }
        ],
    }


def _load_lattice(
    *, event_result_path: Path, event_raw_path: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    result = _mapping(json.loads(event_result_path.read_bytes()), name="event result")
    if (
        result.get("result_sha256") != EVENT_RESULT_HASH
        or _canonical_hash(result, field="result_sha256") != EVENT_RESULT_HASH
        or _sha256(event_raw_path.read_bytes()) != EVENT_RAW_HASH
    ):
        raise ValueError("event discovery evidence hash differs")
    event = _mapping(json.loads(event_raw_path.read_bytes()), name="raw event")
    if (
        event.get("slug") != EVENT_SLUG
        or event.get("active") is not True
        or event.get("closed") is not False
    ):
        raise ValueError("event state differs from frozen discovery")
    markets = [
        parsed
        for raw in _list(event.get("markets"), name="event markets")
        if (parsed := _parse_market(raw)) is not None
    ]
    if len(markets) != 15:
        raise ValueError("scoped moneyline, spread, and total market count differs")
    nodes = [
        dict(node)
        for market in markets
        for node in _list(market["lattice_nodes"], name="lattice nodes")
    ]
    relations: list[dict[str, object]] = []
    for family, expected in EXPECTED_THRESHOLDS.items():
        family_nodes = sorted(
            [node for node in nodes if node["family"] == family],
            key=lambda node: int(node["threshold"]),
        )
        if [int(node["threshold"]) for node in family_nodes] != expected:
            raise ValueError(f"{family} threshold lattice differs")
        for superset_index, superset in enumerate(family_nodes[:-1]):
            for subset in family_nodes[superset_index + 1 :]:
                relations.append(
                    {
                        "family": family,
                        "superset_threshold": superset["threshold"],
                        "subset_threshold": subset["threshold"],
                        "superset_positive_outcome": superset["positive_outcome"],
                        "superset_positive_token": superset["positive_token"],
                        "subset_complement_outcome": subset["complement_outcome"],
                        "subset_complement_token": subset["complement_token"],
                    }
                )
    if len(relations) != 37:
        raise ValueError("monotone relation count differs")
    return markets, relations


def _freeze_contract(
    *,
    event_result_path: Path,
    event_raw_path: Path,
    contract_path: Path,
    markets: list[dict[str, object]],
    relations: list[dict[str, object]],
) -> dict[str, object]:
    if contract_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen contract: {contract_path}")
    tokens = sorted(
        {
            str(node[key])
            for market in markets
            for node in _list(market["lattice_nodes"], name="lattice nodes")
            for key in ("positive_token", "complement_token")
        },
        key=int,
    )
    if len(tokens) != 30:
        raise ValueError("exact token population differs")
    contract: dict[str, object] = {
        "schema_version": CONTRACT_SCHEMA,
        "status": "frozen_before_one_public_CLOB_batch_book_request",
        "purpose": "test 37 exact same-game monotone payoff relations at five-share displayed depth after current per-market taker fees and two adverse ticks per leg",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_evidence": {
            "event_result_path": event_result_path.as_posix(),
            "event_result_sha256": EVENT_RESULT_HASH,
            "event_raw_path": event_raw_path.as_posix(),
            "event_raw_sha256": EVENT_RAW_HASH,
            "event_slug": EVENT_SLUG,
            "active_scoped_market_count": len(markets),
        },
        "payoff_proof": {
            "relation": "subset event A is contained in superset event B",
            "package": "buy B positive token plus buy A complement token",
            "terminal_states_per_share": [
                {
                    "state": "A_true",
                    "B_positive": "1",
                    "A_complement": "0",
                    "package": "1",
                },
                {
                    "state": "B_true_A_false",
                    "B_positive": "1",
                    "A_complement": "1",
                    "package": "2",
                },
                {
                    "state": "B_false",
                    "B_positive": "0",
                    "A_complement": "1",
                    "package": "1",
                },
                {
                    "state": "canceled_both_half",
                    "B_positive": "0.5",
                    "A_complement": "0.5",
                    "package": "1",
                },
            ],
            "minimum_terminal_payout_pUSD": "5",
            "relations": relations,
        },
        "capture": {
            "documentation_url": "https://docs.polymarket.com/api-reference/market-data/get-order-books-request-body",
            "method": "POST",
            "operation": "public unauthenticated read-only batch order-book retrieval",
            "url": BOOKS_URL,
            "request_count": 1,
            "retry_permitted": False,
            "token_count": len(tokens),
            "tokens": tokens,
            "raw_response_retained_before_parsing": True,
        },
        "economics": {
            "quantity_shares_each_leg": "5",
            "displayed_depth_only": True,
            "fee_formula": "sum_per_fill_quantity_times_0_03_times_price_times_one_minus_price_rounded_up_to_0_00001_pUSD",
            "fee_source": "each retained current market feeSchedule rate 0.03 exponent 1 takerOnly true",
            "adverse_ticks_per_leg": ADVERSE_TICKS_PER_LEG,
            "candidate_gate": "strictly positive guaranteed terminal floor after full displayed depth current taker fees and two adverse ticks on both legs",
            "accepted_edge": "never from one snapshot; separately frozen recurrence across games timing and liquidity states plus owned fills and unwind are required",
        },
        "authority": {
            "public_unauthenticated_market_data_only": True,
            "credentials_accounts_orders_or_wallets_permitted": False,
            "paper_or_live_trading_authority": False,
        },
        "one_use_outcomes": {
            "zero_candidates": "terminalize this exact event snapshot without retry",
            "positive_candidates": "freeze a separate prospective multi-game recurrence study without trading",
            "all_outcomes": "accepted edge count unchanged and no account order wallet or funded action authorized",
        },
        "implementation": {
            "path": "tools/screen_polymarket_current_mlb_monotone_parity.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    contract["contract_sha256"] = ""
    contract["contract_sha256"] = _canonical_hash(contract, field="contract_sha256")
    write_bytes_atomic(
        contract_path,
        (json.dumps(contract, indent=2, ensure_ascii=True) + "\n").encode("ascii"),
    )
    return contract


def _capture_books(
    *, contract: Mapping[str, object], raw_path: Path, journal_path: Path
) -> tuple[object, dict[str, object]]:
    capture = _mapping(contract.get("capture"), name="capture contract")
    body = [{"token_id": token} for token in _list(capture["tokens"], name="tokens")]
    started_ms = time.time_ns() // 1_000_000
    response = requests.post(
        BOOKS_URL,
        json=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "simple-ai-trading-public-mlb-parity/1.0",
        },
        timeout=30,
    )
    completed_ms = time.time_ns() // 1_000_000
    raw = response.content
    write_bytes_atomic(raw_path, raw)
    receipt = {
        "name": "polymarket-current-mlb-batch-books",
        "transport": "HTTPS",
        "method": "POST",
        "operation": "public unauthenticated read-only batch order-book retrieval",
        "url": response.url,
        "request_body_sha256": _sha256(_canonical_json(body).encode("ascii")),
        "request_token_count": len(body),
        "status_code": response.status_code,
        "requested_at_ms": started_ms,
        "completed_at_ms": completed_ms,
        "response_bytes": len(raw),
        "response_sha256": _sha256(raw),
        "raw_path": raw_path.as_posix(),
    }
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("xb") as stream:
        stream.write((_canonical_json(receipt) + "\n").encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())
    response.raise_for_status()
    return response.json(), receipt


def _ask_levels(book: Mapping[str, object]) -> tuple[BookLevel, ...]:
    levels = tuple(
        BookLevel(
            price=Decimal(str(_mapping(value, name="ask level").get("price"))),
            quantity=Decimal(str(_mapping(value, name="ask level").get("size"))),
        ).validated()
        for value in _list(book.get("asks"), name="book asks")
    )
    if tuple(sorted(levels, key=lambda level: level.price)) != levels:
        raise ValueError("CLOB asks are not sorted ascending")
    if len({level.price for level in levels}) != len(levels):
        raise ValueError("CLOB asks contain duplicate prices")
    return levels


def _fill(
    *, levels: tuple[BookLevel, ...], tick: Decimal, adverse_ticks: int
) -> dict[str, object] | None:
    remaining = QUANTITY
    actual_cost = Decimal("0")
    stressed_cost = Decimal("0")
    actual_fee = Decimal("0")
    stressed_fee = Decimal("0")
    fee_model = PolymarketFeeModel(True, Decimal("0.03"), 1, True)
    fills: list[dict[str, object]] = []
    for level in levels:
        consumed = min(remaining, level.quantity)
        if consumed <= 0:
            continue
        stressed_price = level.price + tick * adverse_ticks
        if stressed_price >= 1:
            return None
        level_actual_fee = fee_model(level.price, consumed, "taker")
        level_stressed_fee = fee_model(stressed_price, consumed, "taker")
        actual_cost += consumed * level.price
        stressed_cost += consumed * stressed_price
        actual_fee += level_actual_fee
        stressed_fee += level_stressed_fee
        fills.append(
            {
                "quantity": _decimal_text(consumed),
                "ask_price": _decimal_text(level.price),
                "stressed_price": _decimal_text(stressed_price),
                "actual_fee_pUSD": _decimal_text(level_actual_fee),
                "stressed_fee_pUSD": _decimal_text(level_stressed_fee),
            }
        )
        remaining -= consumed
        if remaining == 0:
            return {
                "actual_cost_pUSD": _decimal_text(actual_cost),
                "stressed_cost_pUSD": _decimal_text(stressed_cost),
                "actual_fee_pUSD": _decimal_text(actual_fee),
                "stressed_fee_pUSD": _decimal_text(stressed_fee),
                "fills": fills,
            }
    return None


def _evaluate(
    *,
    contract_path: Path,
    contract: Mapping[str, object],
    markets: list[dict[str, object]],
    payload: object,
    receipt: Mapping[str, object],
) -> dict[str, object]:
    raw_books = [
        _mapping(value, name="CLOB book") for value in _list(payload, name="books")
    ]
    expected_tokens = {
        str(token)
        for token in _list(
            _mapping(contract["capture"], name="capture")["tokens"], name="tokens"
        )
    }
    books = {str(book.get("asset_id")): book for book in raw_books}
    if len(raw_books) != 30 or set(books) != expected_tokens:
        raise ValueError("CLOB batch book population differs")
    token_terms: dict[str, dict[str, object]] = {}
    timestamps: list[int] = []
    for market in markets:
        tick = Decimal(str(market["tick_size"]))
        for node in _list(market["lattice_nodes"], name="lattice nodes"):
            for key in ("positive_token", "complement_token"):
                token = str(_mapping(node, name="lattice node")[key])
                book = books[token]
                if (
                    str(book.get("market") or "").lower() != market["condition_id"]
                    or book.get("neg_risk") is not False
                    or Decimal(str(book.get("min_order_size"))) > QUANTITY
                    or Decimal(str(book.get("tick_size"))) != tick
                ):
                    raise ValueError(f"CLOB identity differs for token {token}")
                token_terms[token] = {
                    "tick": tick,
                    "levels": _ask_levels(book),
                }
                timestamps.append(int(book.get("timestamp", -1)))
    rows: list[dict[str, object]] = []
    relations = _list(
        _mapping(contract["payoff_proof"], name="payoff proof")["relations"],
        name="relations",
    )
    payout = QUANTITY
    for raw_relation in relations:
        relation = _mapping(raw_relation, name="relation")
        token_a = str(relation["superset_positive_token"])
        token_b = str(relation["subset_complement_token"])
        term_a = token_terms[token_a]
        term_b = token_terms[token_b]
        fill_a = _fill(
            levels=term_a["levels"],  # type: ignore[arg-type]
            tick=term_a["tick"],  # type: ignore[arg-type]
            adverse_ticks=ADVERSE_TICKS_PER_LEG,
        )
        fill_b = _fill(
            levels=term_b["levels"],  # type: ignore[arg-type]
            tick=term_b["tick"],  # type: ignore[arg-type]
            adverse_ticks=ADVERSE_TICKS_PER_LEG,
        )
        if fill_a is None or fill_b is None:
            actual_net = None
            stressed_net = None
        else:
            actual_net = payout - sum(
                Decimal(str(fill[key]))
                for fill in (fill_a, fill_b)
                for key in ("actual_cost_pUSD", "actual_fee_pUSD")
            )
            stressed_net = payout - sum(
                Decimal(str(fill[key]))
                for fill in (fill_a, fill_b)
                for key in ("stressed_cost_pUSD", "stressed_fee_pUSD")
            )
        rows.append(
            {
                **relation,
                "quantity_shares_each_leg": _decimal_text(QUANTITY),
                "minimum_terminal_payout_pUSD": _decimal_text(payout),
                "superset_positive_fill": fill_a,
                "subset_complement_fill": fill_b,
                "actual_after_fee_profit_floor_pUSD": _decimal_text(actual_net),
                "stressed_after_fee_profit_floor_pUSD": _decimal_text(stressed_net),
                "passes_frozen_candidate_gate": stressed_net is not None
                and stressed_net > 0,
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: Decimal(
            str(row["stressed_after_fee_profit_floor_pUSD"] or "-Infinity")
        ),
        reverse=True,
    )
    candidates = [row for row in ranked if row["passes_frozen_candidate_gate"]]
    result: dict[str, object] = {
        "schema_version": RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract_path.as_posix(),
            "sha256": contract["contract_sha256"],
        },
        "authority": {
            "public_unauthenticated_read_only_POST_requests": 1,
            "authenticated_requests": 0,
            "account_state_accessed": False,
            "orders_quotes_wallets_or_transactions": 0,
            "paper_or_live_trading_authority": False,
        },
        "capture": {
            "receipt": dict(receipt),
            "book_count": len(raw_books),
            "minimum_book_timestamp_ms": min(timestamps),
            "maximum_book_timestamp_ms": max(timestamps),
        },
        "economics": {
            "relation_count": len(ranked),
            "complete_depth_relation_count": sum(
                row["stressed_after_fee_profit_floor_pUSD"] is not None
                for row in ranked
            ),
            "frozen_candidate_count": len(candidates),
            "best_relation": ranked[0],
            "candidates": candidates,
            "relations": ranked,
        },
        "adjudication": {
            "accepted_edge": False,
            "profitability_claim": False,
            "candidate_for_prospective_recurrence": bool(candidates),
            "deployment_ready": False,
            "trading_authority": False,
            "status": (
                "current_exact_event_has_positive_stressed_monotone_parity_candidate"
                if candidates
                else "current_exact_event_has_no_positive_stressed_monotone_parity_candidate"
            ),
            "next_action": (
                "freeze_a_separate_prospective_multi_game_recurrence_capture_without_trading"
                if candidates
                else "terminalize_this_exact_event_snapshot_without_retry"
            ),
        },
        "implementation": dict(contract["implementation"]),
    }
    result["result_sha256"] = _canonical_hash(result, field="result_sha256")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-result", type=Path, required=True)
    parser.add_argument("--event-raw", type=Path, required=True)
    parser.add_argument("--contract-output", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.contract_output, args.raw, args.journal, args.output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite one-use output: {path}")
    markets, relations = _load_lattice(
        event_result_path=args.event_result,
        event_raw_path=args.event_raw,
    )
    contract = _freeze_contract(
        event_result_path=args.event_result,
        event_raw_path=args.event_raw,
        contract_path=args.contract_output,
        markets=markets,
        relations=relations,
    )
    payload, receipt = _capture_books(
        contract=contract,
        raw_path=args.raw,
        journal_path=args.journal,
    )
    result = _evaluate(
        contract_path=args.contract_output,
        contract=contract,
        markets=markets,
        payload=payload,
        receipt=receipt,
    )
    write_bytes_atomic(args.output, (_canonical_json(result) + "\n").encode("ascii"))
    print(json.dumps(result["capture"], indent=2))
    print(json.dumps(result["economics"]["best_relation"], indent=2))
    print(json.dumps(result["adjudication"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
