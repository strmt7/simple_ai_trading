"""Screen one exact NegRisk event; run as ``python -m tools.<module>``."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from simple_ai_trading.paper_execution import BookLevel
from simple_ai_trading.polymarket_fees import PolymarketFeeModel
from simple_ai_trading.structural_parity import (
    MAX_EXACT_NEGATIVE_RISK_CONVERSION_VARIABLES,
    NegativeRiskOutcome,
    NegativeRiskParityPath,
    NegativeRiskParityScreen,
    screen_negative_risk_parity,
)
from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)
from tools.screen_polymarket_exact_two_leg_package import _request
from tools.screen_polymarket_negrisk_complete_set_catalog import (
    _eligible_event,
    _json_array,
)
from tools.screen_polymarket_structural_parity import _levels


ZERO_FEE = PolymarketFeeModel(False, Decimal("0"), 1, True)
SCHEMA = "polymarket-exact-negrisk-books-result-v1"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain an object")
    return value


def _validate_instant(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError(f"{name} must be an explicit UTC instant")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed > datetime.now(timezone.utc):
        raise RuntimeError(f"{name} is invalid or in the future")
    return parsed


def _retained_event(contract: dict[str, Any]) -> dict[str, Any]:
    retained = contract["retained_sources"]
    source_contract_path = _root_path(retained["prefilter_contract_path"])
    source_result_path = _root_path(retained["prefilter_result_path"])
    source_raw_path = _root_path(retained["event_raw_path"])
    source_contract = _load_object(source_contract_path)
    source_result = _load_object(source_result_path)
    if (
        _canonical_hash(source_contract, "contract_sha256")
        != retained["prefilter_contract_sha256"]
        or source_contract["contract_sha256"] != retained["prefilter_contract_sha256"]
        or _canonical_hash(source_result, "result_sha256")
        != retained["prefilter_result_sha256"]
        or source_result["result_sha256"] != retained["prefilter_result_sha256"]
        or _sha256(source_raw_path.read_bytes()) != retained["event_raw_sha256"]
    ):
        raise RuntimeError("retained prefilter lineage differs")
    if source_result["screen"]["source_only_candidate"] is not True:
        raise RuntimeError("retained prefilter did not authorize a book proof")
    event = _load_object(source_raw_path)
    if not _eligible_event(event) or event.get("slug") != contract["event_slug"]:
        raise RuntimeError("retained exact event identity or fixed state differs")
    return event


def _tokens(event: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for market in event["markets"]:
        values.extend(
            str(value) for value in _json_array(market["clobTokenIds"], "tokens")
        )
    expected = len(event["markets"]) * 2
    if (
        not 4 <= expected <= 200
        or len(values) != expected
        or len(set(values)) != expected
        or any(not value.isdigit() for value in values)
    ):
        raise RuntimeError("exact token population differs")
    return values


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(contract["contract_path"]):
        raise RuntimeError("contract path mismatch")
    _validate_instant(contract.get("frozen_at_utc"), "frozen_at_utc")
    if (
        contract.get("quantity_shares") != "5"
        or contract.get("adverse_ticks_per_leg") != 1
    ):
        raise RuntimeError("quantity or adverse-tick stress changed")
    if contract.get("conversion_fee_bips") != 0:
        raise RuntimeError("unproved conversion fee changed")
    if contract.get("authority") != {
        "account_requests": 0,
        "book_batch_requests": 1,
        "credentials_used": False,
        "fee_requests": 0,
        "funds_used": False,
        "gamma_requests": 0,
        "onchain_requests": 0,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "public_unauthenticated_read_only_requests": 1,
        "signed_requests": 0,
        "trading_authority": False,
    }:
        raise RuntimeError("authority boundary changed")
    event = _retained_event(contract)
    if len(event.get("markets", ())) > MAX_EXACT_NEGATIVE_RISK_CONVERSION_VARIABLES:
        raise RuntimeError(
            "exact event exceeds the bounded negative-risk conversion ceiling"
        )
    tokens = _tokens(event)
    body = json.dumps(
        [{"token_id": token} for token in tokens],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    request = contract["request"]
    if request != {
        "body_sha256": _sha256(body),
        "count": 1,
        "method": "POST",
        "url": "https://clob.polymarket.com/books",
    }:
        raise RuntimeError("book request boundary changed")
    if contract.get("token_ids") != tokens:
        raise RuntimeError("frozen token order differs")
    for implementation in contract["implementations"]:
        path = _root_path(implementation["path"])
        if _sha256(path.read_bytes()) != implementation["sha256"]:
            raise RuntimeError(f"implementation hash mismatch: {path.name}")
    return event


LEGACY_FEE_SCHEDULE = {
    "exponent": 1,
    "rate": 0.05,
    "rebateRate": 0.25,
    "takerOnly": True,
}


def _fee_model(
    market: dict[str, Any], expected_fee_schedule: dict[str, Any] | None = None
) -> PolymarketFeeModel:
    expected = expected_fee_schedule or LEGACY_FEE_SCHEDULE
    schedule = market.get("feeSchedule")
    if (
        market.get("feesEnabled") is not True
        or not isinstance(schedule, dict)
        or schedule != expected
        or set(expected) != {"exponent", "rate", "rebateRate", "takerOnly"}
        or isinstance(expected.get("exponent"), bool)
        or not isinstance(expected.get("exponent"), int)
        or expected["exponent"] < 0
        or isinstance(expected.get("rate"), bool)
        or not isinstance(expected.get("rate"), (int, float))
        or not 0 < Decimal(str(expected["rate"])) < 1
        or isinstance(expected.get("rebateRate"), bool)
        or not isinstance(expected.get("rebateRate"), (int, float))
        or not 0 <= Decimal(str(expected["rebateRate"])) <= 1
        or expected.get("takerOnly") is not True
    ):
        raise RuntimeError("exact current fee schedule differs")
    return PolymarketFeeModel(
        True,
        Decimal(str(expected["rate"])),
        int(expected["exponent"]),
        True,
    )


def _outcomes(
    event: dict[str, Any],
    books: dict[str, dict[str, Any]],
    expected_fee_schedule: dict[str, Any] | None = None,
) -> tuple[NegativeRiskOutcome, ...]:
    parsed: list[NegativeRiskOutcome] = []
    market_id = str(event["negRiskMarketID"]).lower()
    for market in event["markets"]:
        tokens = [str(value) for value in _json_array(market["clobTokenIds"], "tokens")]
        yes, no = (books[token] for token in tokens)
        condition_id = str(market["conditionId"]).lower()
        minimum = Decimal(str(market["orderMinSize"]))
        tick = Decimal(str(market["orderPriceMinTickSize"]))
        if (
            str(market["negRiskMarketID"]).lower() != market_id
            or any(
                str(book.get("market") or "").lower() != condition_id
                for book in (yes, no)
            )
            or any(book.get("neg_risk") is not True for book in (yes, no))
            or any(
                Decimal(str(book.get("min_order_size"))) != minimum
                for book in (yes, no)
            )
            or any(Decimal(str(book.get("tick_size"))) != tick for book in (yes, no))
        ):
            raise RuntimeError("book and retained event identities differ")
        parsed.append(
            NegativeRiskOutcome(
                label=str(market.get("groupItemTitle") or market["question"]),
                yes_bids=_levels(yes, "bids"),
                yes_asks=_levels(yes, "asks"),
                no_asks=_levels(no, "asks"),
                fee_model=_fee_model(market, expected_fee_schedule),
            ).validated()
        )
    return tuple(parsed)


def _stress_levels(
    levels: tuple[BookLevel, ...], delta: Decimal
) -> tuple[BookLevel, ...]:
    stressed = tuple(
        BookLevel(price=level.price + delta, quantity=level.quantity)
        for level in levels
        if Decimal("0") < level.price + delta < Decimal("1")
    )
    return tuple(sorted(stressed, key=lambda level: level.price, reverse=delta < 0))


def _stressed_outcomes(
    event: dict[str, Any], outcomes: tuple[NegativeRiskOutcome, ...]
) -> tuple[NegativeRiskOutcome, ...]:
    stressed: list[NegativeRiskOutcome] = []
    for market, outcome in zip(event["markets"], outcomes, strict=True):
        tick = Decimal(str(market["orderPriceMinTickSize"]))
        stressed.append(
            NegativeRiskOutcome(
                label=outcome.label,
                yes_bids=_stress_levels(outcome.yes_bids, -tick),
                yes_asks=_stress_levels(outcome.yes_asks, tick),
                no_asks=_stress_levels(outcome.no_asks, tick),
                fee_model=outcome.fee_model,
            ).validated()
        )
    return tuple(stressed)


def _path(path: NegativeRiskParityPath | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "mechanism": path.mechanism,
        "selected_no_outcomes": list(path.selected_no_outcomes),
        "net_quote": format(path.net_quote, "f"),
        "taker_fees_quote": format(path.taker_fees_quote, "f"),
        "initial_outlay_quote": format(path.initial_outlay_quote, "f"),
    }


def _screen(screen: NegativeRiskParityScreen) -> dict[str, Any]:
    return {
        "quantity": format(screen.quantity, "f"),
        "evaluated_path_count": screen.evaluated_path_count,
        "executable_path_count": screen.executable_path_count,
        "profitable_path_count": screen.profitable_path_count,
        "buy_all_yes_hold": _path(screen.buy_all_yes_hold),
        "mint_all_yes_sell": _path(screen.mint_all_yes_sell),
        "best_no_conversion": _path(screen.best_no_conversion),
        "best_path": _path(screen.best_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = _load_object(contract_path)
    event = _validate_contract(contract, contract_path)
    paths = {name: _root_path(path) for name, path in contract["outputs"].items()}
    for path in paths.values():
        if path.exists():
            raise RuntimeError(f"one-use output already exists: {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        [{"token_id": token} for token in contract["token_ids"]],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    raw, receipt = _request(
        method="POST",
        url=contract["request"]["url"],
        body=body,
        name="exact-negrisk-complete-token-books",
        raw_path=paths["raw_path"],
        raw_relative_path=contract["outputs"]["raw_path"],
        journal_path=paths["journal_path"],
    )
    rows = json.loads(raw)
    expected_book_count = len(contract["token_ids"])
    if not isinstance(rows, list) or len(rows) != expected_book_count:
        raise RuntimeError("book batch response count differs")
    books = {
        str(row.get("asset_id") or ""): row for row in rows if isinstance(row, dict)
    }
    if set(books) != set(contract["token_ids"]) or len(books) != expected_book_count:
        raise RuntimeError("book batch token identities differ")
    timestamps = [int(str(book.get("timestamp"))) for book in books.values()]
    completed_ms = int(receipt["completed_at_ms"])
    elapsed_ms = completed_ms - int(receipt["requested_at_ms"])
    age_ms = completed_ms - min(timestamps)
    skew_ms = max(timestamps) - min(timestamps)
    freshness = contract["freshness"]
    freshness_passed = (
        elapsed_ms <= freshness["request_max_elapsed_ms"]
        and 0 <= age_ms <= freshness["book_max_event_age_ms"]
        and skew_ms <= freshness["book_max_timestamp_skew_ms"]
    )
    expected_fee_schedule = contract.get("expected_fee_schedule")
    if expected_fee_schedule is not None and not isinstance(
        expected_fee_schedule, dict
    ):
        raise RuntimeError("expected fee schedule must be an object")
    outcomes = _outcomes(event, books, expected_fee_schedule)
    quantity = Decimal(contract["quantity_shares"])
    gross = screen_negative_risk_parity(
        tuple(replace(outcome, fee_model=ZERO_FEE) for outcome in outcomes),
        quantity=quantity,
        conversion_fee_bips=0,
    )
    after_fee = screen_negative_risk_parity(
        outcomes, quantity=quantity, conversion_fee_bips=0
    )
    stressed = screen_negative_risk_parity(
        _stressed_outcomes(event, outcomes),
        quantity=quantity,
        conversion_fee_bips=0,
    )
    candidate = bool(
        freshness_passed
        and after_fee.best_path is not None
        and after_fee.best_path.net_quote > 0
        and stressed.best_path is not None
        and stressed.best_path.net_quote > 0
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "capture": {
            "receipt": receipt,
            "request_elapsed_ms": elapsed_ms,
            "oldest_book_event_age_ms": age_ms,
            "book_timestamp_skew_ms": skew_ms,
            "freshness_passed": freshness_passed,
            "book_count": len(books),
        },
        "screen": {
            "zero_fee_no_stress": _screen(gross),
            "gamma_fee_no_stress": _screen(after_fee),
            "gamma_fee_one_adverse_tick_each_leg": _screen(stressed),
            "candidate_after_all_frozen_gates": candidate,
        },
        "adjudication": {
            "status": (
                "source_freshness_failure_no_promotion"
                if not freshness_passed
                else (
                    "candidate_requires_onchain_adapter_fee_gas_latency_atomicity_and_owned_execution_proof"
                    if candidate
                    else "rejected_after_exact_depth_and_current_gamma_fees_before_onchain_requests"
                )
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
        },
        "authority": contract["authority"],
        "implementation": {
            "path": "tools/screen_polymarket_exact_negrisk_books.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    paths["result_path"].write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "freshness_passed": freshness_passed,
                "gross_best_net": gross.best_path.net_quote
                if gross.best_path
                else None,
                "after_fee_best_net": after_fee.best_path.net_quote
                if after_fee.best_path
                else None,
                "stressed_best_net": stressed.best_path.net_quote
                if stressed.best_path
                else None,
                "candidate": candidate,
                "payloads_printed": 0,
            },
            default=str,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
