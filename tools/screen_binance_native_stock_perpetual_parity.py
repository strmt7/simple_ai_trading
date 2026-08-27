"""Run one frozen public Binance native-stock versus TradFi-perpetual screen."""

from __future__ import annotations

import argparse
import asyncio
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping

import requests
import websockets

from simple_ai_trading.storage import write_bytes_atomic


STOCK_STREAM_TEMPLATE = "wss://nbstream.binance.com/equity/ws/{ticker}@quote"
FUTURES_BOOK_URL = "https://fapi.binance.com/fapi/v1/ticker/bookTicker"
SPOT_BOOK_URL = "https://api.binance.com/api/v3/ticker/bookTicker"
SCHEMA_VERSION = "binance-native-stock-perpetual-parity-v1"
STRESS_BPS = Decimal("30")
QUANTITY = Decimal("1")


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


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _canonical_hash(payload: Mapping[str, object], *, field: str) -> str:
    body = dict(payload)
    body.pop(field)
    return _sha256(_canonical_json(body).encode("ascii"))


def _retain(
    *, raw_path: Path, payload: bytes, receipt: dict[str, object]
) -> tuple[object, dict[str, object]]:
    if raw_path.exists():
        raise FileExistsError(f"refusing to overwrite retained response: {raw_path}")
    write_bytes_atomic(raw_path, payload)
    receipt.update(
        {
            "response_bytes": len(payload),
            "response_sha256": _sha256(payload),
            "raw_path": str(raw_path.as_posix()),
        }
    )
    return json.loads(payload), receipt


def _get_json(
    *, url: str, params: Mapping[str, object], name: str, raw_dir: Path
) -> tuple[object, dict[str, object]]:
    started_ms = time.time_ns() // 1_000_000
    response = requests.get(
        url,
        params=params,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "simple-ai-trading-public-edge-research/1.0",
        },
        timeout=30,
    )
    completed_ms = time.time_ns() // 1_000_000
    payload, receipt = _retain(
        raw_path=raw_dir / f"{name}.raw",
        payload=response.content,
        receipt={
            "name": name,
            "transport": "HTTPS",
            "method": "GET",
            "url": response.url,
            "status_code": response.status_code,
            "requested_at_ms": started_ms,
            "completed_at_ms": completed_ms,
        },
    )
    return payload, receipt


async def _capture_row(
    *, frozen: dict[str, object], raw_dir: Path, timeout_seconds: int
) -> dict[str, object]:
    ticker = str(frozen["ticker"])
    perpetual_symbol = str(frozen["perpetual_symbol"])
    stream_url = STOCK_STREAM_TEMPLATE.format(ticker=ticker)
    started_ms = time.time_ns() // 1_000_000
    sources: list[dict[str, object]] = []
    try:
        async with websockets.connect(
            stream_url,
            open_timeout=timeout_seconds,
            close_timeout=5,
        ) as stream:
            message = await asyncio.wait_for(stream.recv(), timeout_seconds)
        completed_ms = time.time_ns() // 1_000_000
        if not isinstance(message, str):
            raise ValueError(f"{ticker} quote was not a text JSON message")
        stock_raw, stock_source = _retain(
            raw_path=raw_dir / f"stock-{ticker.lower()}.raw",
            payload=message.encode("utf-8"),
            receipt={
                "name": f"stock-{ticker.lower()}",
                "transport": "WEBSOCKET",
                "method": "RECEIVE_FIRST_MESSAGE",
                "url": stream_url,
                "status_code": 101,
                "requested_at_ms": started_ms,
                "completed_at_ms": completed_ms,
            },
        )
        stock = _mapping(stock_raw, name=f"{ticker} stock quote")
        sources.append(stock_source)
        if stock.get("e") != "quote" or stock.get("s") != ticker:
            raise ValueError(f"{ticker} stock quote identity differs")

        results = await asyncio.gather(
            asyncio.to_thread(
                _get_json,
                url=FUTURES_BOOK_URL,
                params={"symbol": perpetual_symbol},
                name=f"perpetual-{ticker.lower()}",
                raw_dir=raw_dir,
            ),
            asyncio.to_thread(
                _get_json,
                url=SPOT_BOOK_URL,
                params={"symbol": "USDCUSDT"},
                name=f"fx-{ticker.lower()}",
                raw_dir=raw_dir,
            ),
            return_exceptions=True,
        )
        for result in results:
            if not isinstance(result, Exception):
                sources.append(result[1])
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise failures[0]
        (perpetual_raw, perpetual_source), (fx_raw, fx_source) = results
        if perpetual_source["status_code"] != 200 or fx_source["status_code"] != 200:
            raise ValueError(f"{ticker} paired GET returned non-200 status")
        perpetual = _mapping(perpetual_raw, name=f"{ticker} perpetual book")
        fx = _mapping(fx_raw, name=f"{ticker} FX book")
        if perpetual.get("symbol") != perpetual_symbol:
            raise ValueError(f"{ticker} perpetual book identity differs")
        if fx.get("symbol") != "USDCUSDT":
            raise ValueError(f"{ticker} FX book identity differs")
        return {
            "status": "complete",
            "frozen": frozen,
            "stock": stock,
            "perpetual": perpetual,
            "fx": fx,
            "sources": sources,
        }
    except Exception as exc:
        return {
            "status": "incomplete",
            "frozen": frozen,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "sources": sources,
        }


async def _run_async(
    *, contract_path: Path, raw_dir: Path
) -> tuple[dict[str, object], list[dict[str, object]]]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_hash = _canonical_hash(contract, field="contract_sha256")
    if contract_hash != contract.get("contract_sha256"):
        raise ValueError("contract hash does not match canonical contents")
    universe = [
        _mapping(value, name="frozen universe row")
        for value in _list(contract.get("frozen_universe"), name="frozen universe")
    ]
    if len(universe) != 14:
        raise ValueError("the frozen universe must contain exactly fourteen tickers")
    timeout_seconds = int(contract["capture"]["websocket_timeout_seconds"])
    captured = await asyncio.gather(
        *(
            _capture_row(
                frozen=frozen,
                raw_dir=raw_dir,
                timeout_seconds=timeout_seconds,
            )
            for frozen in universe
        )
    )
    return contract, captured


def run(*, contract_path: Path, raw_dir: Path, journal_path: Path) -> dict[str, object]:
    contract, captured = asyncio.run(
        _run_async(contract_path=contract_path, raw_dir=raw_dir)
    )
    sources = sorted(
        (source for item in captured for source in item["sources"]),
        key=lambda source: (int(source["requested_at_ms"]), str(source["name"])),
    )
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("xb") as stream:
        for source in sources:
            stream.write((_canonical_json(source) + "\n").encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())

    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for item in captured:
        frozen = _mapping(item["frozen"], name="captured frozen row")
        if item["status"] != "complete":
            errors.append(
                {
                    **frozen,
                    "error_type": item["error_type"],
                    "error": item["error"],
                }
            )
            continue
        stock = _mapping(item["stock"], name="stock quote")
        perpetual = _mapping(item["perpetual"], name="perpetual book")
        fx = _mapping(item["fx"], name="FX book")
        stock_ask = Decimal(str(stock["ap"]))
        stock_ask_qty = Decimal(str(stock["as"]))
        perpetual_bid = Decimal(str(perpetual["bidPrice"]))
        perpetual_bid_qty = Decimal(str(perpetual["bidQty"]))
        fx_ask = Decimal(str(fx["askPrice"]))
        fx_ask_qty = Decimal(str(fx["askQty"]))
        if (
            min(
                stock_ask,
                stock_ask_qty,
                perpetual_bid,
                perpetual_bid_qty,
                fx_ask,
                fx_ask_qty,
            )
            <= 0
        ):
            raise ValueError(f"{frozen['ticker']} contains a non-positive book field")
        perpetual_bid_usdc = perpetual_bid / fx_ask
        needed_fx_usdc = perpetual_bid_usdc * QUANTITY
        capacity_ok = (
            stock_ask_qty >= QUANTITY
            and perpetual_bid_qty >= QUANTITY
            and fx_ask_qty >= needed_fx_usdc
        )
        stock_cost = stock_ask * QUANTITY
        gross = (perpetual_bid_usdc - stock_ask) * QUANTITY
        gross_bps = gross / stock_cost * Decimal(10_000)
        stress = stock_cost * STRESS_BPS / Decimal(10_000)
        after_stress = gross - stress
        source_times = [
            int(source["completed_at_ms"])
            for source in item["sources"]
            if source["transport"] == "HTTPS"
        ]
        rows.append(
            {
                **frozen,
                "quantity_shares": _decimal_text(QUANTITY),
                "native_stock_best_ask_USD": _decimal_text(stock_ask),
                "native_stock_best_ask_quantity": _decimal_text(stock_ask_qty),
                "perpetual_best_bid_USDT": _decimal_text(perpetual_bid),
                "perpetual_best_bid_quantity": _decimal_text(perpetual_bid_qty),
                "USDCUSDT_best_ask": _decimal_text(fx_ask),
                "USDCUSDT_best_ask_quantity_USDC": _decimal_text(fx_ask_qty),
                "perpetual_bid_USDC_equivalent": _decimal_text(perpetual_bid_usdc),
                "all_three_top_level_capacities_pass": capacity_ok,
                "gross_entry_headroom_USDC": _decimal_text(gross),
                "gross_entry_headroom_bps": _decimal_text(gross_bps),
                "labeled_30_bps_stress_USDC": _decimal_text(stress),
                "after_labeled_stress_USDC": _decimal_text(after_stress),
                "after_labeled_stress_positive": capacity_ok and after_stress > 0,
                "native_stock_event_time_ms": stock.get("E"),
                "native_stock_transaction_time_ms": stock.get("T"),
                "perpetual_book_time_ms": perpetual.get("time"),
                "row_local_HTTPS_completion_window_ms": max(source_times)
                - min(source_times),
            }
        )

    complete_population = len(rows) == 14
    stressed_positive = sum(
        row["after_labeled_stress_positive"] is True for row in rows
    )
    best = (
        max(rows, key=lambda row: Decimal(str(row["gross_entry_headroom_bps"])))
        if rows
        else None
    )
    if not complete_population:
        status = "incomplete_public_quote_population_no_economic_adjudication"
        retry_trigger = (
            "new_separately_frozen_capture_during_active_native_stock_quote_state"
        )
    elif stressed_positive:
        status = "unaccepted_candidate_survives_public_stress_requires_exact_account_cost_and_recurrence_evidence"
        retry_trigger = (
            "exact_read_only_stock_and_perpetual_account_cost_eligibility_evidence"
        )
    else:
        status = "current_fourteen_ticker_native_stock_perpetual_snapshot_fails_labeled_pre_account_cost_stress"
        retry_trigger = (
            "material_native_stock_fee_execution_or_book_architecture_change"
        )
    contract_hash = _canonical_hash(contract, field="contract_sha256")
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract": {"path": str(contract_path.as_posix()), "sha256": contract_hash},
        "authority": {
            "public_unauthenticated_GET_requests": sum(
                source["transport"] == "HTTPS" for source in sources
            ),
            "public_unauthenticated_websocket_connections": 14,
            "authenticated_requests": 0,
            "account_state_accessed": False,
            "orders_quotes_transfers_disclaimer_or_wallet_actions": 0,
            "paper_or_live_trading_authority": False,
        },
        "capture": {
            "complete_population": complete_population,
            "complete_row_count": len(rows),
            "incomplete_row_count": len(errors),
            "errors": errors,
            "retained_source_count": len(sources),
            "raw_response_bytes": sum(
                int(source["response_bytes"]) for source in sources
            ),
            "journal_path": str(journal_path.as_posix()),
            "sources": sources,
        },
        "economics": {
            "direction": "buy_one_native_stock_share_and_short_one_matching_TradFi_perpetual_share",
            "labeled_pre_account_cost_stress_bps": _decimal_text(STRESS_BPS),
            "row_count": len(rows),
            "after_labeled_stress_positive_count": stressed_positive,
            "best_ticker": best["ticker"] if best else None,
            "best_gross_entry_headroom_bps": (
                best["gross_entry_headroom_bps"] if best else None
            ),
            "rows": rows,
        },
        "adjudication": {
            "status": status,
            "accepted_edge": False,
            "profitability_claim": False,
            "public_after_cost_profit_floor_USDC": "0",
            "deployment_ready": False,
            "trading_authority": False,
            "retry_trigger": retry_trigger,
        },
        "limitations": [
            "native_stock_and_perpetual_legs_are_not_atomic",
            "exact_account_stock_and_perpetual_commissions_are_unbound",
            "short_eligibility_margin_funding_exit_basis_settlement_and_orphan_risk_are_unbound",
            "the_30_bps_stress_is_an_escalation_gate_not_after_cost_profit_evidence",
        ],
        "implementation": {
            "path": "tools/screen_binance_native_stock_perpetual_parity.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.journal.exists():
        raise FileExistsError(f"refusing to append to prior journal: {args.journal}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite prior result: {args.output}")
    if args.raw_dir.exists() and any(args.raw_dir.iterdir()):
        raise FileExistsError(
            f"refusing to reuse non-empty raw directory: {args.raw_dir}"
        )
    result = run(
        contract_path=args.contract,
        raw_dir=args.raw_dir,
        journal_path=args.journal,
    )
    write_bytes_atomic(args.output, (_canonical_json(result) + "\n").encode("ascii"))
    print(json.dumps(result["capture"], indent=2))
    print(json.dumps(result["adjudication"], indent=2))
    print(json.dumps(result["economics"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
