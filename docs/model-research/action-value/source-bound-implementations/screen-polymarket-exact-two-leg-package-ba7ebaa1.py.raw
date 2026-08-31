from __future__ import annotations

import argparse
from datetime import datetime, timezone
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


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def _frozen_instant(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError("frozen_at_utc must be an exact UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RuntimeError("frozen_at_utc is unparsable") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RuntimeError("frozen_at_utc must carry the UTC offset")
    if parsed > datetime.now(timezone.utc):
        raise RuntimeError("frozen_at_utc is in the future")
    return parsed


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _journal(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _request(
    *,
    method: str,
    url: str,
    body: bytes,
    name: str,
    raw_path: Path,
    raw_relative_path: str,
    journal_path: Path,
) -> tuple[bytes, dict[str, Any]]:
    intent = {
        "method": method,
        "name": name,
        "phase": "intent",
        "request_body_sha256": _sha256(body),
        "requested_at_ms": time.time_ns() // 1_000_000,
        "url": url,
    }
    _journal(journal_path, intent)
    request = Request(
        url,
        data=body or None,
        method=method,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "simple-ai-trading-public-research/1",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read()
            status_code = response.status
    except HTTPError as exc:
        raw = exc.read()
        status_code = exc.code
        raw_path.write_bytes(raw)
        _journal(
            journal_path,
            {
                **intent,
                "completed_at_ms": time.time_ns() // 1_000_000,
                "phase": "completed",
                "raw_path": raw_relative_path,
                "response_bytes": len(raw),
                "response_sha256": _sha256(raw),
                "status_code": status_code,
            },
        )
        raise
    raw_path.write_bytes(raw)
    receipt = {
        **intent,
        "completed_at_ms": time.time_ns() // 1_000_000,
        "phase": "completed",
        "raw_path": raw_relative_path,
        "response_bytes": len(raw),
        "response_sha256": _sha256(raw),
        "status_code": status_code,
    }
    _journal(journal_path, receipt)
    if status_code != 200:
        raise RuntimeError(f"unexpected HTTP status {status_code}")
    return raw, receipt


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
    book: dict[str, Any],
    *,
    quantity: Decimal,
    tick_size: Decimal,
    adverse_ticks: int,
    fee_model: PolymarketFeeModel | None,
) -> dict[str, Any] | None:
    remaining = quantity
    cost = Decimal("0")
    fee = Decimal("0")
    fills: list[dict[str, str]] = []
    for price, available in _asks(book):
        consumed = min(remaining, available)
        if consumed <= 0:
            continue
        stressed_price = price + tick_size * adverse_ticks
        if stressed_price >= 1:
            return None
        cost += consumed * stressed_price
        if fee_model is not None:
            fee += fee_model(stressed_price, consumed, "taker")
        fills.append(
            {
                "price_pUSD": _decimal_text(stressed_price),
                "quantity_shares": _decimal_text(consumed),
            }
        )
        remaining -= consumed
        if remaining == 0:
            return {
                "cost_pUSD": cost,
                "fee_pUSD": fee if fee_model is not None else None,
                "fills": fills,
            }
    return None


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("contract hash mismatch")
    _frozen_instant(contract.get("frozen_at_utc"))
    if contract_path != _root_path(contract["contract_path"]):
        raise RuntimeError("contract path mismatch")
    implementation = _root_path(contract["implementation"]["path"])
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    if contract["execution"]["book_request_count"] != 1:
        raise RuntimeError("exactly one book batch must be frozen")
    if contract["execution"]["maximum_fee_requests"] != 2:
        raise RuntimeError("fee request ceiling must equal two")
    if contract["authority"]["public_unauthenticated_read_only_requests_maximum"] != 3:
        raise RuntimeError("public request ceiling must equal three")


def _validate_metadata(contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["metadata_source"]
    path = _root_path(source["path"])
    raw = path.read_bytes()
    if _sha256(raw) != source["file_sha256"]:
        raise RuntimeError("metadata file hash mismatch")
    metadata = json.loads(raw)
    if _canonical_hash(metadata, "result_sha256") != source["result_sha256"]:
        raise RuntimeError("metadata canonical hash mismatch")
    markets = metadata["discovery"]["active_accepting_markets"]
    for definition in contract["markets"]:
        market = next(
            (row for row in markets if str(row.get("id")) == definition["id"]),
            None,
        )
        if market is None:
            raise RuntimeError(f"market absent: {definition['id']}")
        description = str(market.get("description") or "")
        if not all(
            fragment in description for fragment in definition["required_rule_fragments"]
        ):
            raise RuntimeError(f"market rules changed: {definition['id']}")
        outcomes = json.loads(market["outcomes"])
        prices = json.loads(market["outcomePrices"])
        tokens = json.loads(market["clobTokenIds"])
        if not (
            market["question"] == definition["question"]
            and market["sportsMarketType"] == definition["sports_market_type"]
            and Decimal(str(market["line"])) == Decimal(definition["line"])
            and market["conditionId"] == definition["condition_id"]
            and outcomes == definition["outcomes"]
            and prices == definition["outcome_prices"]
            and tokens == definition["tokens"]
            and market["active"] is True
            and market["closed"] is False
            and market["acceptingOrders"] is True
            and market["enableOrderBook"] is True
            and market["negRisk"] is False
            and market["feeSchedule"] == contract["execution"]["fee_schedule"]
            and market["takerBaseFee"] == 1000
            and Decimal(str(market["orderMinSize"]))
            <= Decimal(contract["execution"]["quantity_shares_each_leg"])
            and Decimal(str(market["orderPriceMinTickSize"]))
            == Decimal(contract["execution"]["tick_size"])
        ):
            raise RuntimeError(f"market identity changed: {definition['id']}")
    payouts = [
        sum(Decimal(str(value)) for value in state["payouts"].values())
        for state in contract["payoff_proof"]["states"]
    ]
    floor = Decimal(contract["payoff_proof"]["guaranteed_floor_per_share_pUSD"])
    if min(payouts) != floor:
        raise RuntimeError("package lacks its frozen payout floor")
    prices = [
        Decimal(contract["tokens"][name]["gamma_price_pUSD"])
        for name in contract["package"]["token_names"]
    ]
    if sum(prices) != Decimal(contract["gamma_prefilter"]["displayed_price_sum_pUSD"]):
        raise RuntimeError("Gamma prefilter sum mismatch")
    if sum(prices) >= floor:
        raise RuntimeError("Gamma prefilter did not clear the strict rejection gate")
    return metadata


def _json_ready(value: Any) -> Any:
    if isinstance(value, Decimal):
        return _decimal_text(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen one frozen exact two-leg Polymarket payoff package."
    )
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    contract_path = _root_path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(contract, contract_path)
    _validate_metadata(contract)

    result_path = _root_path(contract["outputs"]["result_path"])
    journal_path = _root_path(contract["outputs"]["journal_path"])
    data_root = journal_path.parent
    if data_root.exists() or result_path.exists():
        raise RuntimeError("one-use output already exists")
    fee_root = data_root / "raw/fees"
    fee_root.mkdir(parents=True)

    token_names = contract["package"]["token_names"]
    token_ids = [contract["tokens"][name]["token_id"] for name in token_names]
    request_body = json.dumps(
        [{"token_id": token} for token in token_ids], separators=(",", ":")
    ).encode("ascii")
    books_relative = contract["outputs"]["books_raw_path"]
    books_raw, book_receipt = _request(
        method="POST",
        url="https://clob.polymarket.com/books",
        body=request_body,
        name="exact-two-token-book-batch",
        raw_path=_root_path(books_relative),
        raw_relative_path=books_relative,
        journal_path=journal_path,
    )
    raw_books = json.loads(books_raw)
    books = {str(row["asset_id"]): row for row in raw_books}
    if len(raw_books) != 2 or set(books) != set(token_ids):
        raise RuntimeError("book population differs from contract")

    tick_size = Decimal(contract["execution"]["tick_size"])
    quantity = Decimal(contract["execution"]["quantity_shares_each_leg"])
    timestamps: list[int] = []
    for name in token_names:
        definition = contract["tokens"][name]
        book = books[definition["token_id"]]
        if not (
            str(book["market"]).lower() == definition["condition_id"]
            and book["neg_risk"] is False
            and Decimal(str(book["min_order_size"])) <= quantity
            and Decimal(str(book["tick_size"])) == tick_size
        ):
            raise RuntimeError(f"book identity changed: {name}")
        timestamps.append(int(book["timestamp"]))
    skew_ms = max(timestamps) - min(timestamps)

    zero_fee_fills: dict[str, list[dict[str, Any] | None]] = {}
    for stress_name, adverse_ticks in contract["execution"]["stress_ticks"].items():
        zero_fee_fills[stress_name] = [
            _fill(
                books[contract["tokens"][name]["token_id"]],
                quantity=quantity,
                tick_size=tick_size,
                adverse_ticks=int(adverse_ticks),
                fee_model=None,
            )
            for name in token_names
        ]

    floor = quantity * Decimal(
        contract["payoff_proof"]["guaranteed_floor_per_share_pUSD"]
    )
    final_stress_name = contract["execution"]["candidate_stress_name"]
    final_zero_fee = zero_fee_fills[final_stress_name]
    final_gross_positive = all(final_zero_fee) and (
        floor
        - sum(fill["cost_pUSD"] for fill in final_zero_fee if fill is not None)
        > 0
    )

    fee_receipts: dict[str, dict[str, Any]] = {}
    fee_model: PolymarketFeeModel | None = None
    if final_gross_positive:
        for name in token_names:
            token = contract["tokens"][name]["token_id"]
            relative = f"{contract['outputs']['fee_raw_root']}/{token}.json"
            fee_raw, receipt = _request(
                method="GET",
                url=f"https://clob.polymarket.com/fee-rate/{token}",
                body=b"",
                name=f"fee-rate-{name}",
                raw_path=_root_path(relative),
                raw_relative_path=relative,
                journal_path=journal_path,
            )
            if json.loads(fee_raw) != {"base_fee": 1000}:
                raise RuntimeError(f"fee rate changed: {name}")
            fee_receipts[name] = receipt
        schedule = contract["execution"]["fee_schedule"]
        fee_model = PolymarketFeeModel(
            enabled=True,
            rate=Decimal(str(schedule["rate"])),
            exponent=int(schedule["exponent"]),
            taker_only=bool(schedule["takerOnly"]),
        )

    economics: dict[str, Any] = {}
    for stress_name, adverse_ticks in contract["execution"]["stress_ticks"].items():
        fills = [
            _fill(
                books[contract["tokens"][name]["token_id"]],
                quantity=quantity,
                tick_size=tick_size,
                adverse_ticks=int(adverse_ticks),
                fee_model=fee_model,
            )
            for name in token_names
        ]
        if not all(fills):
            economics[stress_name] = None
            continue
        cost = sum(fill["cost_pUSD"] for fill in fills if fill is not None)
        fee = (
            sum(fill["fee_pUSD"] for fill in fills if fill is not None)
            if fee_model is not None
            else None
        )
        economics[stress_name] = {
            "adverse_ticks_per_leg": adverse_ticks,
            "cost_pUSD": cost,
            "optimistic_zero_fee_profit_floor_pUSD": floor - cost,
            "current_fee_pUSD": fee,
            "after_current_fee_profit_floor_pUSD": (
                floor - cost - fee if fee is not None else None
            ),
            "fills": fills,
        }

    synchronized = skew_ms <= contract["execution"]["maximum_book_timestamp_skew_ms"]
    final = economics[final_stress_name]
    passes = bool(
        synchronized
        and final is not None
        and final["after_current_fee_profit_floor_pUSD"] is not None
        and final["after_current_fee_profit_floor_pUSD"] > 0
    )
    result: dict[str, Any] = {
        "schema_version": "polymarket-exact-two-leg-package-result-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "capture": {
            "book_receipt": book_receipt,
            "fee_receipts": fee_receipts,
            "book_timestamp_skew_ms": skew_ms,
            "within_frozen_skew_gate": synchronized,
        },
        "payoff_proof": contract["payoff_proof"],
        "gamma_prefilter": contract["gamma_prefilter"],
        "economics": economics,
        "adjudication": {
            "passes_frozen_candidate_gate": passes,
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "next_action": (
                "require_one_independent_positive_recurrence_before_any_order_capable_work"
                if passes
                else "terminalize_this_exact_event_without_refetch_or_retry"
            ),
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    serializable = _json_ready(result)
    serializable["result_sha256"] = _canonical_hash(serializable, "result_sha256")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "book_request_count": 1,
                "fee_request_count": len(fee_receipts),
                "passes_candidate_gate": passes,
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
