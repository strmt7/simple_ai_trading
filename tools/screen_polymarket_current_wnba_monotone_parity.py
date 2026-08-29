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
    "polymarket-current-wnba-monotone-parity-contract-v1-2026-08-29.json"
)
RESULT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-current-wnba-monotone-parity-result-v1-2026-08-29.json"
)
DATA_ROOT = ROOT / "data/polymarket-current-wnba-monotone-parity-v1"
JOURNAL_PATH = DATA_ROOT / "request-journal.jsonl"
QUANTITY = Decimal("5")
FEE_MODEL = PolymarketFeeModel(True, Decimal("0.05"), 1, True)


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
    return None if value is None else format(value, "f")


def _journal(payload: dict[str, Any]) -> None:
    with JOURNAL_PATH.open("a", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _request(
    *, method: str, url: str, body: bytes, name: str, raw_path: Path
) -> tuple[bytes, dict[str, Any]]:
    intent = {
        "method": method,
        "name": name,
        "phase": "intent",
        "request_body_sha256": _sha256(body),
        "requested_at_ms": time.time_ns() // 1_000_000,
        "url": url,
    }
    _journal(intent)
    headers = {"User-Agent": "simple-ai-trading-public-research/1"}
    if body:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body or None, method=method, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read()
            status_code = response.status
    except HTTPError as exc:
        raw = exc.read()
        status_code = exc.code
        raw_path.write_bytes(raw)
        _journal(
            {
                **intent,
                "completed_at_ms": time.time_ns() // 1_000_000,
                "phase": "completed",
                "raw_path": raw_path.relative_to(ROOT).as_posix(),
                "response_bytes": len(raw),
                "response_sha256": _sha256(raw),
                "status_code": status_code,
            }
        )
        raise
    raw_path.write_bytes(raw)
    receipt = {
        **intent,
        "completed_at_ms": time.time_ns() // 1_000_000,
        "phase": "completed",
        "raw_path": raw_path.relative_to(ROOT).as_posix(),
        "response_bytes": len(raw),
        "response_sha256": _sha256(raw),
        "status_code": status_code,
    }
    _journal(receipt)
    if status_code != 200:
        raise RuntimeError(f"unexpected HTTP status {status_code}")
    return raw, receipt


def _validate_metadata(contract: dict[str, Any]) -> None:
    source = contract["metadata_source"]
    raw = (ROOT / source["path"]).read_bytes()
    if _sha256(raw) != source["sha256"]:
        raise RuntimeError("metadata hash mismatch")
    event = json.loads(raw)
    if not (
        event["slug"] == contract["event"]["slug"]
        and event["active"] is True
        and event["closed"] is False
    ):
        raise RuntimeError("exact event is not active and open")

    for definition in contract["markets"]:
        market = next(
            (row for row in event["markets"] if row.get("id") == definition["id"]),
            None,
        )
        if market is None:
            raise RuntimeError(f"market absent: {definition['id']}")
        description = str(market.get("description") or "")
        if not all(
            fragment in description for fragment in definition["required_rule_fragments"]
        ):
            raise RuntimeError(f"market rules changed: {definition['id']}")
        if not (
            market["question"] == definition["question"]
            and market["sportsMarketType"] == definition["sports_market_type"]
            and market.get("line") == definition["line"]
            and market["conditionId"] == definition["condition_id"]
            and json.loads(market["outcomes"]) == definition["outcomes"]
            and json.loads(market["clobTokenIds"]) == definition["tokens"]
            and market["active"] is True
            and market["closed"] is False
            and market["acceptingOrders"] is True
            and market["enableOrderBook"] is True
            and market["negRisk"] is False
            and market["feeSchedule"] == contract["execution"]["fee_schedule"]
            and market["takerBaseFee"] == 1000
            and market["secondsDelay"] == 1
            and Decimal(str(market["orderMinSize"])) <= QUANTITY
            and Decimal(str(market["orderPriceMinTickSize"])) == Decimal("0.01")
        ):
            raise RuntimeError(f"market identity changed: {definition['id']}")

    for package in contract["packages"]:
        payouts = [
            sum(Decimal(str(state["payouts"][token])) for token in package["token_names"])
            for state in contract["payoff_proof"]["states"]
        ]
        if min(payouts) != Decimal("1"):
            raise RuntimeError(f"package lacks unit payout floor: {package['name']}")


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
    book: dict[str, Any], *, adverse_ticks: int
) -> dict[str, str] | None:
    remaining = QUANTITY
    cost = Decimal("0")
    fee = Decimal("0")
    for price, available in _asks(book):
        consumed = min(remaining, available)
        if consumed <= 0:
            continue
        stressed_price = price + Decimal("0.01") * adverse_ticks
        if stressed_price >= 1:
            return None
        cost += consumed * stressed_price
        fee += FEE_MODEL(stressed_price, consumed, "taker")
        remaining -= consumed
        if remaining == 0:
            return {
                "cost_pUSD": _decimal_text(cost) or "",
                "fee_pUSD": _decimal_text(fee) or "",
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
    _validate_metadata(contract)

    token_ids = list(contract["execution"]["token_ids"])
    request_body = json.dumps(
        [{"token_id": token} for token in token_ids], separators=(",", ":")
    ).encode("ascii")
    raw, book_receipt = _request(
        method="POST",
        url="https://clob.polymarket.com/books",
        body=request_body,
        name="exact-four-token-book-batch",
        raw_path=DATA_ROOT / "raw/books.json",
    )
    raw_books = json.loads(raw)
    books = {str(row["asset_id"]): row for row in raw_books}
    if len(raw_books) != 4 or set(books) != set(token_ids):
        raise RuntimeError("book population differs from contract")

    token_definitions = contract["tokens"]
    timestamps: list[int] = []
    for name, definition in token_definitions.items():
        book = books[definition["token_id"]]
        if not (
            str(book["market"]).lower() == definition["condition_id"]
            and book["neg_risk"] is False
            and Decimal(str(book["min_order_size"])) <= QUANTITY
            and Decimal(str(book["tick_size"])) == Decimal("0.01")
        ):
            raise RuntimeError(f"book identity changed: {name}")
        timestamps.append(int(book["timestamp"]))
    skew_ms = max(timestamps) - min(timestamps)
    synchronized = skew_ms <= contract["execution"]["maximum_book_timestamp_skew_ms"]

    package_fills: dict[str, dict[str, list[dict[str, str] | None]]] = {}
    fee_tokens: set[str] = set()
    for package in contract["packages"]:
        fills_by_stress: dict[str, list[dict[str, str] | None]] = {}
        for stress_name, adverse_ticks in contract["execution"]["stress_ticks"].items():
            fills_by_stress[stress_name] = [
                _fill(
                    books[token_definitions[token_name]["token_id"]],
                    adverse_ticks=adverse_ticks,
                )
                for token_name in package["token_names"]
            ]
        package_fills[package["name"]] = fills_by_stress
        actual = fills_by_stress["actual"]
        if all(fill is not None for fill in actual):
            gross_cost = sum(Decimal(fill["cost_pUSD"]) for fill in actual if fill)
            if gross_cost < QUANTITY:
                fee_tokens.update(
                    token_definitions[name]["token_id"]
                    for name in package["token_names"]
                )

    fee_receipts: dict[str, dict[str, Any]] = {}
    for token in token_ids:
        if token not in fee_tokens:
            continue
        fee_raw, receipt = _request(
            method="GET",
            url=f"https://clob.polymarket.com/fee-rate/{token}",
            body=b"",
            name=f"fee-rate-{token}",
            raw_path=DATA_ROOT / f"raw/fees/{token}.json",
        )
        if json.loads(fee_raw) != {"base_fee": 1000}:
            raise RuntimeError(f"fee rate changed: {token}")
        fee_receipts[token] = receipt

    rows: list[dict[str, Any]] = []
    for package in contract["packages"]:
        economics: dict[str, Any] = {}
        for stress_name, fills in package_fills[package["name"]].items():
            if any(fill is None for fill in fills):
                economics[stress_name] = None
                continue
            cost = sum(Decimal(fill["cost_pUSD"]) for fill in fills if fill)
            fee = sum(Decimal(fill["fee_pUSD"]) for fill in fills if fill)
            if stress_name == "actual" and cost < QUANTITY:
                package_tokens = {
                    token_definitions[name]["token_id"]
                    for name in package["token_names"]
                }
                if not package_tokens <= fee_tokens:
                    raise RuntimeError("gross-positive package lacks fee audit")
            economics[stress_name] = {
                "cost_pUSD": _decimal_text(cost),
                "fee_pUSD": _decimal_text(fee),
                "after_fee_profit_floor_pUSD": _decimal_text(QUANTITY - cost - fee),
            }
        delay_3s = economics["delay_3s_sensitivity"]
        rows.append(
            {
                **package,
                "economics": economics,
                "passes_frozen_candidate_gate": synchronized
                and delay_3s is not None
                and Decimal(delay_3s["after_fee_profit_floor_pUSD"]) > 0,
            }
        )

    best = max(
        rows,
        key=lambda row: Decimal(
            (row["economics"]["delay_3s_sensitivity"] or {}).get(
                "after_fee_profit_floor_pUSD", "-Infinity"
            )
        ),
    )
    candidate_count = sum(row["passes_frozen_candidate_gate"] for row in rows)
    result: dict[str, Any] = {
        "schema_version": "polymarket-current-wnba-monotone-parity-result-v1",
        "contract": {
            "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "sha256": contract["contract_sha256"],
        },
        "payoff_proof": contract["payoff_proof"],
        "capture": {
            "book_receipt": book_receipt,
            "fee_receipts": fee_receipts,
            "book_timestamp_skew_ms": skew_ms,
            "passes_synchronization_gate": synchronized,
        },
        "rows": rows,
        "adjudication": {
            "current_positive_package_count": candidate_count,
            "best_package": best["name"],
            "best_delay_3s_profit_floor_pUSD": best["economics"][
                "delay_3s_sensitivity"
            ]["after_fee_profit_floor_pUSD"],
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "next_action": (
                "require_independent_positive_recurrence_before_any_order_work"
                if candidate_count
                else "terminalize_this_exact_event_without_resampling"
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
