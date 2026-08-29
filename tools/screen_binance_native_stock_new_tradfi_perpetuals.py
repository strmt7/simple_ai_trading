"""Freeze and run one five-symbol Binance stock/perpetual parity screen."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path

from simple_ai_trading.storage import write_bytes_atomic
from tools.screen_binance_native_stock_perpetual_parity import (
    QUANTITY,
    STRESS_BPS,
    _canonical_hash,
    _canonical_json,
    _capture_row,
    _decimal_text,
    _list,
    _mapping,
    _sha256,
)


TEMPLATE_SCHEMA = "binance-native-stock-new-tradfi-perpetual-template-v1"
CONTRACT_SCHEMA = "binance-native-stock-new-tradfi-perpetual-contract-v1"
RESULT_SCHEMA = "binance-native-stock-new-tradfi-perpetual-result-v1"
EXPECTED_TICKERS = ("TEM", "MRK", "IONQ", "MARA", "PDD")


def _freeze_contract(*, template_path: Path, contract_path: Path) -> dict[str, object]:
    if contract_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen contract: {contract_path}")
    template_bytes = template_path.read_bytes()
    template = _mapping(json.loads(template_bytes), name="contract template")
    if (
        template.get("schema_version") != TEMPLATE_SCHEMA
        or template.get("status") != "prefreeze_template_no_market_access_yet"
    ):
        raise ValueError("unexpected or already-consumed contract template")
    universe = [
        _mapping(value, name="frozen universe row")
        for value in _list(template.get("frozen_universe"), name="frozen universe")
    ]
    if tuple(str(row.get("ticker")) for row in universe) != EXPECTED_TICKERS:
        raise ValueError("template must contain the exact five official new tickers")
    contract = dict(template)
    contract["schema_version"] = CONTRACT_SCHEMA
    contract["status"] = "frozen_before_exact_five_ticker_public_capture"
    contract["frozen_at_utc"] = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    contract["template"] = {
        "path": template_path.as_posix(),
        "sha256": _sha256(template_bytes),
    }
    contract["contract_sha256"] = ""
    contract["contract_sha256"] = _canonical_hash(contract, field="contract_sha256")
    write_bytes_atomic(
        contract_path,
        (json.dumps(contract, indent=2, ensure_ascii=True) + "\n").encode("ascii"),
    )
    retained = _mapping(json.loads(contract_path.read_bytes()), name="frozen contract")
    if _canonical_hash(retained, field="contract_sha256") != retained.get(
        "contract_sha256"
    ):
        raise ValueError("persisted frozen contract fails its canonical hash")
    frozen = datetime.fromisoformat(
        str(retained["frozen_at_utc"]).replace("Z", "+00:00")
    )
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("persisted frozen contract time is invalid or future")
    return retained


async def _capture(
    *, contract: dict[str, object], raw_dir: Path
) -> list[dict[str, object]]:
    universe = [
        _mapping(value, name="frozen universe row")
        for value in _list(contract.get("frozen_universe"), name="frozen universe")
    ]
    timeout_seconds = int(
        _mapping(contract["capture"], name="capture")["websocket_timeout_seconds"]
    )
    return list(
        await asyncio.gather(
            *(
                _capture_row(
                    frozen=row,
                    raw_dir=raw_dir,
                    timeout_seconds=timeout_seconds,
                )
                for row in universe
            )
        )
    )


def _evaluate(
    *,
    contract_path: Path,
    contract: dict[str, object],
    captured: list[dict[str, object]],
    journal_path: Path,
) -> dict[str, object]:
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
        https_times = [
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
                "row_local_HTTPS_completion_window_ms": max(https_times)
                - min(https_times),
            }
        )

    complete_population = len(rows) == len(EXPECTED_TICKERS)
    stressed_positive = sum(
        row["after_labeled_stress_positive"] is True for row in rows
    )
    best = (
        max(rows, key=lambda row: Decimal(str(row["gross_entry_headroom_bps"])))
        if rows
        else None
    )
    if not rows:
        status = "zero_new_perpetual_tickers_have_a_live_native_stock_quote"
        retry_trigger = "new_official_native_stock_or_TradFi_perpetual_listing_or_stream_architecture_change"
    elif not complete_population:
        status = "partial_new_perpetual_native_stock_overlap_no_complete_population_rejection"
        retry_trigger = "new_official_native_stock_listing_for_an_unmatched_frozen_ticker_or_material_stream_change"
    elif stressed_positive:
        status = "unaccepted_public_candidate_survives_frozen_stress"
        retry_trigger = (
            "exact_read_only_stock_and_perpetual_account_cost_eligibility_evidence"
        )
    else:
        status = "complete_five_ticker_snapshot_fails_frozen_pre_account_stress"
        retry_trigger = (
            "material_native_stock_fee_execution_or_book_architecture_change"
        )

    result: dict[str, object] = {
        "schema_version": RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract_path.as_posix(),
            "sha256": contract["contract_sha256"],
        },
        "authority": {
            "public_unauthenticated_GET_requests": sum(
                source["transport"] == "HTTPS" for source in sources
            ),
            "public_unauthenticated_websocket_connections": len(EXPECTED_TICKERS),
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
            "journal_path": journal_path.as_posix(),
            "sources": sources,
        },
        "economics": {
            "direction": "buy_one_native_stock_share_and_short_one_matching_TradFi_perpetual_share",
            "labeled_pre_account_cost_stress_bps": _decimal_text(STRESS_BPS),
            "row_count": len(rows),
            "after_labeled_stress_positive_count": stressed_positive,
            "best_ticker": best["ticker"] if best else None,
            "best_gross_entry_headroom_bps": best["gross_entry_headroom_bps"]
            if best
            else None,
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
            "path": "tools/screen_binance_native_stock_new_tradfi_perpetuals.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
            "capture_helper_path": "tools/screen_binance_native_stock_perpetual_parity.py",
            "capture_helper_sha256": _sha256(
                Path(
                    "tools/screen_binance_native_stock_perpetual_parity.py"
                ).read_bytes()
            ),
        },
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--contract-output", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.journal.exists() or args.output.exists():
        raise FileExistsError("refusing to overwrite retained capture evidence")
    if args.raw_dir.exists() and any(args.raw_dir.iterdir()):
        raise FileExistsError("refusing to reuse a non-empty raw directory")
    args.raw_dir.mkdir(parents=True, exist_ok=False)
    contract = _freeze_contract(
        template_path=args.template,
        contract_path=args.contract_output,
    )
    captured = asyncio.run(_capture(contract=contract, raw_dir=args.raw_dir))
    result = _evaluate(
        contract_path=args.contract_output,
        contract=contract,
        captured=captured,
        journal_path=args.journal,
    )
    write_bytes_atomic(args.output, (_canonical_json(result) + "\n").encode("ascii"))
    print(json.dumps(result["capture"], indent=2))
    print(json.dumps(result["economics"], indent=2))
    print(json.dumps(result["adjudication"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
