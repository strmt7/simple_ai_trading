"""Screen Binance Spot Smart Order Routing as an organic taker-cost overlay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Mapping

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)
from tools.screen_polymarket_exact_two_leg_package import _request
from tools.screen_polymarket_mlb_cross_period_catalog import _frozen_instant


SCHEMA = "binance-spot-sor-liquidity-overlay-result-v1"
TEN_THOUSAND = Decimal("10000")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be an object")
    return dict(value)


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{name} must be a list")
    return value


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    _frozen_instant(contract.get("frozen_at_utc"))
    if contract["capture"] != {
        "book_ticker_url": "https://api.binance.com/api/v3/ticker/bookTicker",
        "exchange_info_url": "https://api.binance.com/api/v3/exchangeInfo",
        "maximum_request_count": 2,
    }:
        raise RuntimeError("capture boundary changed")
    if contract["screen"] != {
        "candidate_threshold_bips_strictly_greater_than": "1",
        "notional_sizes_in_submitted_quote": ["100", "1000"],
        "scoped_base_assets": ["BTC", "ETH", "SOL"],
        "sides": ["BUY", "SELL"],
        "top_level_only": True,
    }:
        raise RuntimeError("screen boundary changed")
    if contract["authority"] != {
        "account_requests": 0,
        "credentials_used": False,
        "funds_used": False,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "public_unauthenticated_read_only_requests_maximum": 2,
        "signed_requests": 0,
        "trading_authority": False,
    }:
        raise RuntimeError("authority boundary changed")
    for implementation in contract["implementations"]:
        path = _root_path(str(implementation["path"]))
        if _sha256(path.read_bytes()) != implementation["sha256"]:
            raise RuntimeError(f"implementation hash mismatch: {path.name}")


def _parse_exchange_info(
    payload: object, scoped_bases: set[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    root = _mapping(payload, "exchange info")
    symbols: dict[str, dict[str, Any]] = {}
    for raw in _list(root.get("symbols"), "exchange symbols"):
        item = _mapping(raw, "exchange symbol")
        symbol = str(item.get("symbol") or "")
        if not symbol or symbol in symbols:
            raise RuntimeError("exchange symbol identity is missing or duplicated")
        symbols[symbol] = item

    groups: list[dict[str, Any]] = []
    raw_sors = root.get("sors", [])
    for raw in _list(raw_sors, "SOR configurations"):
        item = _mapping(raw, "SOR configuration")
        base = str(item.get("baseAsset") or "")
        configured_symbols = [
            str(value) for value in _list(item.get("symbols"), "SOR symbols")
        ]
        if base not in scoped_bases:
            continue
        if len(configured_symbols) < 2 or len(set(configured_symbols)) != len(
            configured_symbols
        ):
            raise RuntimeError("scoped SOR group lacks distinct alternative books")
        quotes: list[str] = []
        for symbol in configured_symbols:
            info = symbols.get(symbol)
            if info is None:
                raise RuntimeError("SOR symbol is absent from exchange information")
            if (
                info.get("baseAsset") != base
                or info.get("status") != "TRADING"
                or info.get("isSpotTradingAllowed") is not True
                or "MARKET" not in _list(info.get("orderTypes"), "order types")
            ):
                raise RuntimeError("SOR symbol is not a compatible trading market")
            quotes.append(str(info.get("quoteAsset") or ""))
        if any(not quote for quote in quotes) or len(set(quotes)) != len(quotes):
            raise RuntimeError("SOR quote assets are missing or duplicated")
        groups.append(
            {"base_asset": base, "symbols": configured_symbols, "quote_assets": quotes}
        )
    groups.sort(key=lambda row: (row["base_asset"], row["symbols"]))
    return groups, symbols


def _parse_books(payload: object) -> dict[str, dict[str, Decimal]]:
    books: dict[str, dict[str, Decimal]] = {}
    for raw in _list(payload, "book ticker"):
        item = _mapping(raw, "book ticker row")
        symbol = str(item.get("symbol") or "")
        if not symbol or symbol in books:
            raise RuntimeError("book ticker symbol is missing or duplicated")
        values = {
            key: Decimal(str(item.get(key) or "0"))
            for key in ("bidPrice", "bidQty", "askPrice", "askQty")
        }
        if all(value.is_finite() and value > 0 for value in values.values()):
            books[symbol] = values
    return books


def _fill_top_levels(
    books: list[tuple[str, dict[str, Decimal]]], quantity: Decimal, side: str
) -> tuple[Decimal, list[dict[str, str]]] | None:
    if side == "BUY":
        ordered = sorted(books, key=lambda row: (row[1]["askPrice"], row[0]))
        price_key, quantity_key = "askPrice", "askQty"
    elif side == "SELL":
        ordered = sorted(books, key=lambda row: (-row[1]["bidPrice"], row[0]))
        price_key, quantity_key = "bidPrice", "bidQty"
    else:
        raise RuntimeError("unsupported side")
    remaining = quantity
    total = Decimal("0")
    fills: list[dict[str, str]] = []
    for symbol, book in ordered:
        consumed = min(remaining, book[quantity_key])
        if consumed <= 0:
            continue
        total += consumed * book[price_key]
        fills.append(
            {
                "symbol": symbol,
                "price": _decimal_text(book[price_key]),
                "base_quantity": _decimal_text(consumed),
            }
        )
        remaining -= consumed
        if remaining == 0:
            return total, fills
    return None


def _evaluate_group(
    group: dict[str, Any],
    books: dict[str, dict[str, Decimal]],
    sizes: list[Decimal],
    threshold_bips: Decimal,
) -> list[dict[str, Any]]:
    symbols = [str(value) for value in group["symbols"]]
    if not set(symbols).issubset(books):
        return []
    group_books = [(symbol, books[symbol]) for symbol in symbols]
    rows: list[dict[str, Any]] = []
    for submitted_symbol in symbols:
        direct = books[submitted_symbol]
        for side in ("BUY", "SELL"):
            direct_price = direct["askPrice" if side == "BUY" else "bidPrice"]
            direct_qty = direct["askQty" if side == "BUY" else "bidQty"]
            for notional in sizes:
                base_quantity = notional / direct_price
                direct_capacity = direct_qty >= base_quantity
                sor_fill = _fill_top_levels(group_books, base_quantity, side)
                if not direct_capacity or sor_fill is None:
                    rows.append(
                        {
                            "base_asset": group["base_asset"],
                            "submitted_symbol": submitted_symbol,
                            "side": side,
                            "submitted_quote_notional": _decimal_text(notional),
                            "base_quantity": _decimal_text(base_quantity),
                            "top_level_comparison_complete": False,
                            "gross_improvement_bips": None,
                            "public_candidate": False,
                        }
                    )
                    continue
                sor_total, fills = sor_fill
                direct_total = base_quantity * direct_price
                if side == "BUY":
                    improvement = (
                        (direct_total - sor_total) / direct_total * TEN_THOUSAND
                    )
                else:
                    improvement = (
                        (sor_total - direct_total) / direct_total * TEN_THOUSAND
                    )
                rows.append(
                    {
                        "base_asset": group["base_asset"],
                        "submitted_symbol": submitted_symbol,
                        "side": side,
                        "submitted_quote_notional": _decimal_text(notional),
                        "base_quantity": _decimal_text(base_quantity),
                        "direct_total_in_submitted_quote": _decimal_text(direct_total),
                        "sor_total_in_submitted_quote": _decimal_text(sor_total),
                        "sor_top_level_fills": fills,
                        "top_level_comparison_complete": True,
                        "gross_improvement_bips": _decimal_text(improvement),
                        "public_candidate": improvement > threshold_bips,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text(encoding="ascii"))
    _validate_contract(contract, contract_path)
    paths = {
        name: _root_path(str(value)) for name, value in contract["outputs"].items()
    }
    for path in paths.values():
        if path.exists():
            raise RuntimeError(f"one-use output already exists: {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)

    exchange_raw, exchange_receipt = _request(
        method="GET",
        url=contract["capture"]["exchange_info_url"],
        body=b"",
        name="binance-spot-sor-exchange-info",
        raw_path=paths["exchange_info_raw_path"],
        raw_relative_path=contract["outputs"]["exchange_info_raw_path"],
        journal_path=paths["journal_path"],
    )
    scoped_bases = set(contract["screen"]["scoped_base_assets"])
    groups, _symbols = _parse_exchange_info(json.loads(exchange_raw), scoped_bases)
    book_receipt: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    if groups:
        book_raw, book_receipt = _request(
            method="GET",
            url=contract["capture"]["book_ticker_url"],
            body=b"",
            name="binance-spot-sor-book-ticker",
            raw_path=paths["book_ticker_raw_path"],
            raw_relative_path=contract["outputs"]["book_ticker_raw_path"],
            journal_path=paths["journal_path"],
        )
        books = _parse_books(json.loads(book_raw))
        sizes = [
            Decimal(value)
            for value in contract["screen"]["notional_sizes_in_submitted_quote"]
        ]
        threshold = Decimal(
            contract["screen"]["candidate_threshold_bips_strictly_greater_than"]
        )
        for group in groups:
            rows.extend(_evaluate_group(group, books, sizes, threshold))

    candidates = sorted(
        (row for row in rows if row["public_candidate"]),
        key=lambda row: (
            -Decimal(row["gross_improvement_bips"]),
            row["submitted_symbol"],
            row["side"],
        ),
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "capture": {
            "exchange_info_receipt": exchange_receipt,
            "book_ticker_receipt": book_receipt,
        },
        "screen": {
            "sor_groups": groups,
            "scoped_group_count": len(groups),
            "rows": rows,
            "complete_top_level_row_count": sum(
                row["top_level_comparison_complete"] for row in rows
            ),
            "public_candidate_count": len(candidates),
            "best_public_candidate": candidates[0] if candidates else None,
        },
        "adjudication": {
            "status": "public_gross_candidate_requires_exact_account_commission_and_owned_fill_reconciliation"
            if candidates
            else "current_public_snapshot_has_no_gross_candidate",
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "stable_edge_claim": False,
            "next_action": "with_both_designated_ephemeral_credentials_and_explicit_signed_test_only_authority_run_one_sor_order_test_with_computeCommissionRates_for_an_independently_required_organic_order_then_require_separate_order_authority_and_owned_allocation_reconciliation"
            if candidates
            else "do_not_repeat_without_a_material_SOR_configuration_quote_fee_or_execution_change",
        },
        "authority": {
            "public_unauthenticated_read_only_requests": 1
            + int(book_receipt is not None),
            "credentials_used": False,
            "signed_requests": 0,
            "account_requests": 0,
            "orders_or_transactions": 0,
            "funds_used": False,
            "trading_authority": False,
            "protected_capture_touched": False,
        },
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    paths["result_path"].write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "scoped_group_count": len(groups),
                "evaluated_rows": len(rows),
                "candidate_count": len(candidates),
                "best_improvement_bips": None
                if not candidates
                else candidates[0]["gross_improvement_bips"],
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
