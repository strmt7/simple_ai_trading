from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from simple_ai_trading.polymarket_fees import PolymarketFeeModel


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-shutdown-house-identity-parity-contract-v1-2026-08-29.json"
)
RESULT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-shutdown-house-identity-parity-result-v1-2026-08-29.json"
)
DATA_ROOT = ROOT / "data/polymarket-shutdown-house-identity-parity-v1"
JOURNAL_PATH = DATA_ROOT / "request-journal.jsonl"
QUANTITY = Decimal("5")
FEE_MODEL = PolymarketFeeModel(True, Decimal("0.04"), 1, True)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: dict[str, Any], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return _sha256(encoded)


def _decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value, "f")


def _journal(payload: dict[str, Any]) -> None:
    with JOURNAL_PATH.open("a", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _request(
    *, method: str, url: str, body: bytes, name: str, raw_path: Path
) -> tuple[bytes, dict[str, Any]]:
    requested_at_ms = time.time_ns() // 1_000_000
    intent = {
        "method": method,
        "name": name,
        "phase": "intent",
        "request_body_sha256": _sha256(body),
        "requested_at_ms": requested_at_ms,
        "url": url,
    }
    _journal(intent)
    headers = {"User-Agent": "simple-ai-trading-public-research/1"}
    if body:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body or None, method=method, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            response_bytes = response.read()
            status_code = response.status
    except HTTPError as exc:
        response_bytes = exc.read()
        status_code = exc.code
        raw_path.write_bytes(response_bytes)
        receipt = {
            **intent,
            "completed_at_ms": time.time_ns() // 1_000_000,
            "phase": "completed",
            "raw_path": raw_path.relative_to(ROOT).as_posix(),
            "response_bytes": len(response_bytes),
            "response_sha256": _sha256(response_bytes),
            "status_code": status_code,
        }
        _journal(receipt)
        raise
    raw_path.write_bytes(response_bytes)
    receipt = {
        **intent,
        "completed_at_ms": time.time_ns() // 1_000_000,
        "phase": "completed",
        "raw_path": raw_path.relative_to(ROOT).as_posix(),
        "response_bytes": len(response_bytes),
        "response_sha256": _sha256(response_bytes),
        "status_code": status_code,
    }
    _journal(receipt)
    if status_code != 200:
        raise RuntimeError(f"unexpected HTTP status {status_code}")
    return response_bytes, receipt


def _load_metadata(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for source in contract["metadata_sources"]:
        path = ROOT / source["path"]
        raw = path.read_bytes()
        if _sha256(raw) != source["sha256"]:
            raise RuntimeError(f"metadata hash mismatch: {path}")
        payloads[source["name"]] = json.loads(raw)

    shutdown = payloads["shutdown"]
    shutdown_market = shutdown["markets"][0]
    if not (
        shutdown["slug"] == contract["payoff_proof"]["shutdown_event_slug"]
        and shutdown["closed"] is True
        and shutdown_market["id"] == "680392"
        and shutdown_market["closed"] is True
        and shutdown_market["acceptingOrders"] is False
        and json.loads(shutdown_market["outcomePrices"]) == ["1", "0"]
        and shutdown_market["umaResolutionStatus"] == "resolved"
    ):
        raise RuntimeError("incorporated shutdown is not final Yes")

    combined = payloads["combined"]
    house = payloads["house"]
    combined_description = str(combined.get("description") or "")
    for required_link in contract["payoff_proof"]["required_combined_rule_links"]:
        if required_link not in combined_description:
            raise RuntimeError("combined rule does not link an exact source event")

    markets: dict[str, dict[str, Any]] = {}
    for definition in contract["markets"]:
        event = combined if definition["event"] == "combined" else house
        market = next(
            (row for row in event["markets"] if row.get("id") == definition["id"]),
            None,
        )
        if market is None:
            raise RuntimeError(f"market absent: {definition['id']}")
        tokens = json.loads(market["clobTokenIds"])
        if not (
            market["question"] == definition["question"]
            and market["groupItemTitle"] == definition["group_item_title"]
            and market["conditionId"] == definition["condition_id"]
            and tokens == definition["tokens"]
            and market["active"] is True
            and market["closed"] is False
            and market["acceptingOrders"] is True
            and market["negRisk"] is True
            and json.loads(market["outcomes"]) == ["Yes", "No"]
            and market["feeSchedule"] == contract["execution"]["fee_schedule"]
            and market["takerBaseFee"] == 1000
            and Decimal(str(market["orderMinSize"])) <= QUANTITY
            and Decimal(str(market["orderPriceMinTickSize"]))
            == Decimal(definition["tick_size"])
        ):
            raise RuntimeError(f"market terms changed: {definition['id']}")
        markets[definition["name"]] = market

    truth_table = []
    for house_state in ("Democratic", "Republican", "Other"):
        row: dict[str, Any] = {"house_state": house_state, "shutdown_yes": True}
        for party in ("Democratic", "Republican"):
            combined_payoff = int(True and house_state == party)
            standalone_payoff = int(house_state == party)
            row[f"combined_{party}_payoff"] = combined_payoff
            row[f"standalone_{party}_payoff"] = standalone_payoff
            if combined_payoff != standalone_payoff:
                raise RuntimeError("payoff identity failed")
        truth_table.append(row)
    markets["truth_table"] = {"rows": truth_table}  # type: ignore[assignment]
    return markets


def _asks(book: dict[str, Any]) -> list[tuple[Decimal, Decimal]]:
    levels = [
        (Decimal(str(row["price"])), Decimal(str(row["size"])))
        for row in book.get("asks", [])
    ]
    prices = [price for price, _ in levels]
    ascending = all(left < right for left, right in zip(prices, prices[1:]))
    descending = all(left > right for left, right in zip(prices, prices[1:]))
    if len(levels) > 1 and not (ascending or descending):
        raise RuntimeError("ask ordering is not strictly monotone")
    if descending:
        levels.reverse()
    return levels


def _fill(
    book: dict[str, Any], *, tick: Decimal, adverse_ticks: int
) -> dict[str, Any] | None:
    remaining = QUANTITY
    actual_cost = Decimal("0")
    stressed_cost = Decimal("0")
    actual_fee = Decimal("0")
    stressed_fee = Decimal("0")
    fills: list[dict[str, str]] = []
    for price, available in _asks(book):
        consumed = min(remaining, available)
        if consumed <= 0:
            continue
        stressed_price = price + tick * adverse_ticks
        if stressed_price >= 1:
            return None
        actual_cost += consumed * price
        stressed_cost += consumed * stressed_price
        actual_fee += FEE_MODEL(price, consumed, "taker")
        stressed_fee += FEE_MODEL(stressed_price, consumed, "taker")
        fills.append(
            {
                "ask_price": _decimal_text(price) or "",
                "quantity": _decimal_text(consumed) or "",
                "stressed_price": _decimal_text(stressed_price) or "",
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


def main() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if _canonical_hash(contract, "contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("contract hash mismatch")
    implementation = ROOT / contract["implementation"]["path"]
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    if DATA_ROOT.exists() or RESULT_PATH.exists():
        raise RuntimeError("one-use output already exists")
    (DATA_ROOT / "raw/fees").mkdir(parents=True)
    markets = _load_metadata(contract)

    tokens = contract["execution"]["tokens"]
    request_body = json.dumps(
        [{"token_id": token} for token in tokens], separators=(",", ":")
    ).encode("ascii")
    books_raw, book_receipt = _request(
        method="POST",
        url="https://clob.polymarket.com/books",
        body=request_body,
        name="exact-eight-token-book-batch",
        raw_path=DATA_ROOT / "raw/books.json",
    )
    raw_books = json.loads(books_raw)
    books = {str(book["asset_id"]): book for book in raw_books}
    if len(raw_books) != len(tokens) or set(books) != set(tokens):
        raise RuntimeError("book population differs from contract")
    timestamps = []
    definitions = {row["name"]: row for row in contract["markets"]}
    for definition in contract["markets"]:
        for token in definition["tokens"]:
            book = books[token]
            if not (
                str(book["market"]).lower() == definition["condition_id"]
                and book["neg_risk"] is True
                and Decimal(str(book["min_order_size"])) <= QUANTITY
                and Decimal(str(book["tick_size"])) == Decimal(definition["tick_size"])
            ):
                raise RuntimeError(f"book identity changed: {token}")
            timestamps.append(int(book["timestamp"]))
    source_skew_ms = max(timestamps) - min(timestamps)
    if source_skew_ms > contract["execution"]["maximum_book_timestamp_skew_ms"]:
        raise RuntimeError("book timestamp skew exceeds contract")

    package_fills: dict[str, tuple[dict[str, Any] | None, dict[str, Any] | None]] = {}
    fee_tokens: set[str] = set()
    for package in contract["packages"]:
        left = _fill(
            books[package["tokens"][0]],
            tick=Decimal(definitions[package["markets"][0]]["tick_size"]),
            adverse_ticks=contract["execution"]["adverse_ticks_per_leg"],
        )
        right = _fill(
            books[package["tokens"][1]],
            tick=Decimal(definitions[package["markets"][1]]["tick_size"]),
            adverse_ticks=contract["execution"]["adverse_ticks_per_leg"],
        )
        package_fills[package["name"]] = (left, right)
        if left is not None and right is not None:
            gross_cost = Decimal(left["actual_cost_pUSD"]) + Decimal(
                right["actual_cost_pUSD"]
            )
            if gross_cost < QUANTITY:
                fee_tokens.update(package["tokens"])

    fee_receipts: dict[str, dict[str, Any]] = {}
    for token in tokens:
        if token not in fee_tokens:
            continue
        fee_raw, fee_receipt = _request(
            method="GET",
            url=f"https://clob.polymarket.com/fee-rate/{token}",
            body=b"",
            name=f"fee-rate-{token}",
            raw_path=DATA_ROOT / f"raw/fees/{token}.json",
        )
        fee_payload = json.loads(fee_raw)
        if fee_payload != {"base_fee": 1000}:
            raise RuntimeError(f"fee rate changed: {token}")
        fee_receipts[token] = fee_receipt

    rows: list[dict[str, Any]] = []
    for package in contract["packages"]:
        left, right = package_fills[package["name"]]
        if left is None or right is None:
            actual_net = None
            stressed_net = None
        else:
            actual_gross_cost = Decimal(left["actual_cost_pUSD"]) + Decimal(
                right["actual_cost_pUSD"]
            )
            stressed_gross_cost = Decimal(left["stressed_cost_pUSD"]) + Decimal(
                right["stressed_cost_pUSD"]
            )
            if actual_gross_cost < QUANTITY and not set(package["tokens"]) <= fee_tokens:
                raise RuntimeError("positive gross package lacks fee audit")
            actual_fees = Decimal(left["actual_fee_pUSD"]) + Decimal(
                right["actual_fee_pUSD"]
            )
            stressed_fees = Decimal(left["stressed_fee_pUSD"]) + Decimal(
                right["stressed_fee_pUSD"]
            )
            actual_net = QUANTITY - actual_gross_cost - actual_fees
            stressed_net = QUANTITY - stressed_gross_cost - stressed_fees
        rows.append(
            {
                **package,
                "left_fill": left,
                "right_fill": right,
                "actual_after_fee_profit_floor_pUSD": _decimal_text(actual_net),
                "stressed_after_fee_profit_floor_pUSD": _decimal_text(stressed_net),
                "passes_frozen_candidate_gate": stressed_net is not None
                and stressed_net > 0,
            }
        )
    best = max(
        rows,
        key=lambda row: Decimal(row["stressed_after_fee_profit_floor_pUSD"] or "-Infinity"),
    )
    candidate_count = sum(row["passes_frozen_candidate_gate"] for row in rows)
    result: dict[str, Any] = {
        "schema_version": "polymarket-shutdown-house-identity-parity-result-v1",
        "contract": {
            "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "sha256": contract["contract_sha256"],
        },
        "payoff_proof": {
            "shutdown_final_yes": True,
            "truth_table": markets["truth_table"]["rows"],
            "duplicate_payoff_identity_count": 2,
        },
        "capture": {
            "book_receipt": book_receipt,
            "fee_receipts": fee_receipts,
            "book_timestamp_min_ms": min(timestamps),
            "book_timestamp_max_ms": max(timestamps),
            "book_timestamp_skew_ms": source_skew_ms,
        },
        "rows": rows,
        "adjudication": {
            "current_stressed_positive_package_count": candidate_count,
            "best_package": best["name"],
            "best_stressed_profit_floor_pUSD": best[
                "stressed_after_fee_profit_floor_pUSD"
            ],
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "next_action": (
                "freeze_one_short_bounded_public_recurrence_capture_before_any_"
                "account_or_order_work"
                if candidate_count
                else "terminalize_this_exact_snapshot_and_wait_for_a_material_"
                "price_fee_rule_or_book_change"
            ),
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    RESULT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(json.dumps(result["adjudication"], sort_keys=True))


if __name__ == "__main__":
    main()
