"""Run one frozen public Binance Alpha/Ondo versus stock-perpetual screen."""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


ALPHA_BASE_URL = "https://www.binance.com"
FUTURES_BASE_URL = "https://fapi.binance.com"
SCHEMA_VERSION = "binance-alpha-ondo-perpetual-parity-v1"
STRESS_BPS = Decimal("20")


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


def _filter(symbol: Mapping[str, object], filter_type: str) -> dict[str, object]:
    for value in _list(symbol.get("filters"), name="symbol filters"):
        row = _mapping(value, name="symbol filter")
        if row.get("filterType") == filter_type:
            return row
    raise ValueError(f"{symbol.get('symbol')} lacks {filter_type}")


def _minimum_notional(symbol: Mapping[str, object]) -> Decimal:
    for name in ("MIN_NOTIONAL", "NOTIONAL"):
        try:
            row = _filter(symbol, name)
        except ValueError:
            continue
        value = row.get("minNotional") or row.get("notional")
        if value is not None:
            return Decimal(str(value))
    raise ValueError(f"{symbol.get('symbol')} lacks minimum notional")


def _common_step(left: Decimal, right: Decimal) -> Decimal:
    step = max(left, right)
    smaller = min(left, right)
    if step <= 0 or smaller <= 0 or step % smaller != 0:
        raise ValueError("quantity steps do not have an exact common coarser step")
    return step


def _round_up(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_CEILING) * step


class _Client:
    def __init__(self, raw_dir: Path, journal_path: Path) -> None:
        self.raw_dir = raw_dir
        self.journal_path = journal_path
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept-Encoding": "identity",
                "User-Agent": "simple-ai-trading-public-edge-research/1.0",
            }
        )

    def get(
        self,
        url: str,
        *,
        name: str,
        params: Mapping[str, object] | None = None,
    ) -> tuple[object, dict[str, object]]:
        started_ms = time.time_ns() // 1_000_000
        response = self.session.get(url, params=params, timeout=30)
        finished_ms = time.time_ns() // 1_000_000
        payload = response.content
        raw_path = self.raw_dir / f"{name}.raw"
        if raw_path.exists():
            raise FileExistsError(
                f"refusing to overwrite retained response: {raw_path}"
            )
        write_bytes_atomic(raw_path, payload)
        receipt = {
            "name": name,
            "method": "GET",
            "url": response.url,
            "status_code": response.status_code,
            "requested_at_ms": started_ms,
            "completed_at_ms": finished_ms,
            "response_bytes": len(payload),
            "response_sha256": _sha256(payload),
            "raw_path": str(raw_path.as_posix()),
        }
        line = (_canonical_json(receipt) + "\n").encode("ascii")
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        response.raise_for_status()
        try:
            return response.json(), receipt
        except requests.JSONDecodeError as exc:
            raise ValueError(f"{name} did not return JSON") from exc


def run(*, contract_path: Path, raw_dir: Path, journal_path: Path) -> dict[str, object]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_hash = _canonical_hash(contract, field="contract_sha256")
    if contract_hash != contract.get("contract_sha256"):
        raise ValueError("contract hash does not match canonical contents")
    universe = [
        _mapping(value, name="frozen universe row")
        for value in _list(contract.get("frozen_universe"), name="frozen universe")
    ]
    if len(universe) != 4:
        raise ValueError("the frozen universe must contain exactly four contracts")

    client = _Client(raw_dir, journal_path)
    alpha_exchange_raw, alpha_exchange_source = client.get(
        f"{ALPHA_BASE_URL}/bapi/defi/v1/public/alpha-trade/get-exchange-info",
        name="alpha-exchange-info",
    )
    futures_exchange_raw, futures_exchange_source = client.get(
        f"{FUTURES_BASE_URL}/fapi/v1/exchangeInfo",
        name="futures-exchange-info",
    )
    futures_books_raw, futures_books_source = client.get(
        f"{FUTURES_BASE_URL}/fapi/v1/ticker/bookTicker",
        name="futures-book-tickers",
    )

    alpha_envelope = _mapping(alpha_exchange_raw, name="Alpha exchange envelope")
    alpha_exchange = _mapping(alpha_envelope.get("data"), name="Alpha exchange data")
    alpha_symbols = {
        str(row["symbol"]): row
        for row in (
            _mapping(value, name="Alpha symbol")
            for value in _list(alpha_exchange.get("symbols"), name="Alpha symbols")
        )
    }
    futures_exchange = _mapping(futures_exchange_raw, name="Futures exchange info")
    futures_symbols = {
        str(row["symbol"]): row
        for row in (
            _mapping(value, name="Futures symbol")
            for value in _list(futures_exchange.get("symbols"), name="Futures symbols")
        )
    }
    futures_books = {
        str(row["symbol"]): row
        for row in (
            _mapping(value, name="Futures book")
            for value in _list(futures_books_raw, name="Futures books")
        )
    }

    rows: list[dict[str, object]] = []
    depth_sources: list[dict[str, object]] = []
    for frozen in universe:
        alpha_symbol = str(frozen["alpha_symbol"])
        perpetual_symbol = str(frozen["perpetual_symbol"])
        if alpha_symbol not in alpha_symbols:
            raise ValueError(f"{alpha_symbol} is absent from Alpha exchange info")
        if (
            perpetual_symbol not in futures_symbols
            or perpetual_symbol not in futures_books
        ):
            raise ValueError(f"{perpetual_symbol} is absent from Futures public data")
        alpha_info = alpha_symbols[alpha_symbol]
        futures_info = futures_symbols[perpetual_symbol]
        if (
            alpha_info.get("status") != "TRADING"
            or futures_info.get("status") != "TRADING"
        ):
            raise ValueError(f"{alpha_symbol} or {perpetual_symbol} is not trading")
        if alpha_info.get("orderTypes") != ["LIMIT"]:
            raise ValueError(f"{alpha_symbol} order type contract changed")

        depth_raw, depth_source = client.get(
            f"{ALPHA_BASE_URL}/bapi/defi/v1/public/alpha-trade/fullDepth",
            name=f"alpha-depth-{str(frozen['ticker']).lower()}",
            params={"symbol": alpha_symbol},
        )
        depth_sources.append(depth_source)
        depth_envelope = _mapping(depth_raw, name=f"{alpha_symbol} depth envelope")
        depth = _mapping(depth_envelope.get("data"), name=f"{alpha_symbol} depth")
        if depth.get("symbol") != alpha_symbol:
            raise ValueError(f"{alpha_symbol} depth identity differs")
        asks = sorted(
            (
                (Decimal(str(level[0])), Decimal(str(level[1])))
                for level in (
                    _list(value, name="Alpha ask level")
                    for value in _list(depth.get("asks"), name="Alpha asks")
                )
            ),
            key=lambda level: level[0],
        )
        if not asks or min(asks[0]) <= 0:
            raise ValueError(f"{alpha_symbol} has no positive ask")
        alpha_ask, alpha_ask_qty = asks[0]
        futures_book = futures_books[perpetual_symbol]
        perpetual_bid = Decimal(str(futures_book["bidPrice"]))
        perpetual_bid_qty = Decimal(str(futures_book["bidQty"]))
        if min(perpetual_bid, perpetual_bid_qty) <= 0:
            raise ValueError(f"{perpetual_symbol} has no positive bid")

        alpha_lot = _filter(alpha_info, "LOT_SIZE")
        futures_lot = _filter(futures_info, "LOT_SIZE")
        alpha_step = Decimal(str(alpha_lot["stepSize"]))
        futures_step = Decimal(str(futures_lot["stepSize"]))
        step = _common_step(alpha_step, futures_step)
        minimum_quantity = max(
            Decimal(str(alpha_lot["minQty"])),
            Decimal(str(futures_lot["minQty"])),
            _minimum_notional(alpha_info) / alpha_ask,
            _minimum_notional(futures_info) / perpetual_bid,
        )
        quantity = _round_up(minimum_quantity, step)
        capacity_ok = quantity <= min(alpha_ask_qty, perpetual_bid_qty)
        if not capacity_ok:
            gross = Decimal(0)
            alpha_cost = Decimal(0)
            stress_cost = Decimal(0)
            after_stress = Decimal(0)
            gross_bps = Decimal(0)
        else:
            alpha_cost = alpha_ask * quantity
            gross = (perpetual_bid - alpha_ask) * quantity
            gross_bps = gross / alpha_cost * Decimal(10_000)
            stress_cost = alpha_cost * STRESS_BPS / Decimal(10_000)
            after_stress = gross - stress_cost
        rows.append(
            {
                **frozen,
                "alpha_order_types": alpha_info["orderTypes"],
                "alpha_best_ask": _decimal_text(alpha_ask),
                "alpha_best_ask_quantity": _decimal_text(alpha_ask_qty),
                "perpetual_best_bid": _decimal_text(perpetual_bid),
                "perpetual_best_bid_quantity": _decimal_text(perpetual_bid_qty),
                "common_quantity_step": _decimal_text(step),
                "minimum_common_quantity": _decimal_text(quantity),
                "top_level_capacity_passes": capacity_ok,
                "alpha_cost_usdt": _decimal_text(alpha_cost),
                "gross_entry_headroom_usdt": _decimal_text(gross),
                "gross_entry_headroom_bps": _decimal_text(gross_bps),
                "labeled_20_bps_stress_usdt": _decimal_text(stress_cost),
                "after_labeled_stress_usdt": _decimal_text(after_stress),
                "after_labeled_stress_positive": capacity_ok and after_stress > 0,
                "alpha_event_time_ms": depth.get("E"),
                "alpha_transaction_time_ms": depth.get("T"),
                "futures_book_time_ms": futures_book.get("time"),
            }
        )

    sources = [
        alpha_exchange_source,
        futures_exchange_source,
        futures_books_source,
        *depth_sources,
    ]
    started_ms = min(int(source["requested_at_ms"]) for source in sources)
    completed_ms = max(int(source["completed_at_ms"]) for source in sources)
    positive_gross = sum(
        Decimal(str(row["gross_entry_headroom_usdt"])) > 0 for row in rows
    )
    positive_stressed = sum(
        row["after_labeled_stress_positive"] is True for row in rows
    )
    best = max(rows, key=lambda row: Decimal(str(row["gross_entry_headroom_bps"])))
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract": {
            "path": str(contract_path.as_posix()),
            "sha256": contract_hash,
        },
        "authority": {
            "public_unauthenticated_GET_requests": len(sources),
            "authenticated_requests": 0,
            "account_state_accessed": False,
            "orders_quotes_transfers_or_wallet_actions": 0,
            "paper_or_live_trading_authority": False,
        },
        "capture": {
            "started_at_ms": started_ms,
            "completed_at_ms": completed_ms,
            "window_ms": completed_ms - started_ms,
            "request_count": len(sources),
            "all_status_codes_200": all(
                source["status_code"] == 200 for source in sources
            ),
            "raw_response_bytes": sum(
                int(source["response_bytes"]) for source in sources
            ),
            "journal_path": str(journal_path.as_posix()),
            "sources": sources,
        },
        "economics": {
            "direction": "buy_Alpha_Ondo_wrapper_and_short_equal_stock_perpetual_quantity",
            "labeled_pre_account_cost_stress_bps": _decimal_text(STRESS_BPS),
            "row_count": len(rows),
            "gross_positive_count": positive_gross,
            "after_labeled_stress_positive_count": positive_stressed,
            "best_ticker": best["ticker"],
            "best_gross_entry_headroom_bps": best["gross_entry_headroom_bps"],
            "rows": rows,
        },
        "adjudication": {
            "status": (
                "unaccepted_candidate_survives_public_stress_but_requires_exact_account_cost_and_recurrence_evidence"
                if positive_stressed
                else "current_four_contract_Alpha_Ondo_perpetual_snapshot_fails_labeled_pre_account_cost_stress"
            ),
            "accepted_edge": False,
            "profitability_claim": False,
            "public_after_cost_profit_floor_usdt": "0",
            "deployment_ready": False,
            "trading_authority": False,
            "retry_trigger": (
                "exact_read_only_account_fee_and_Alpha_settlement_cost_evidence_then_separately_frozen_recurrence_capture"
                if positive_stressed
                else "material_Alpha_fee_execution_or_book_architecture_change"
            ),
        },
        "limitations": [
            "Alpha_full_depth_is_public_executable_looking_depth_not_an_owned_fill_or_atomic_quote",
            "exact_account_perpetual_commission_and_Alpha_network_or_settlement_charges_are_unbound",
            "short_eligibility_margin_funding_exit_basis_and_orphan_risk_are_unbound",
            "the_20_bps_stress_is_an_escalation_gate_not_after_cost_profit_evidence",
        ],
        "implementation": {
            "path": "tools/screen_binance_alpha_ondo_perpetual_parity.py",
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
    print(json.dumps(result["adjudication"], indent=2))
    print(json.dumps(result["economics"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
