"""Audit Binance PRIMARY_PEG order-acceptance value from retained public tickers."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _decimal(value: object, *, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a decimal") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _retained(source: Mapping[str, object]) -> bytes:
    path = ROOT / str(source["path"])
    raw = path.read_bytes()
    if _sha256(raw) != source["sha256"]:
        raise ValueError(f"retained source hash changed: {source['path']}")
    return raw


def _contract(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    contract = _mapping(json.loads(raw), name="contract")
    claimed = str(contract.pop("result_sha256", ""))
    if claimed != _sha256(_canonical(contract)):
        raise ValueError("contract embedded hash does not reconstruct")
    if contract.get("schema_version") != "binance-primary-peg-retained-contract-v1":
        raise ValueError("unsupported contract schema")
    implementation = _mapping(contract.get("implementation"), name="implementation")
    if implementation.get("tool_sha256") != _sha256(Path(__file__).read_bytes()):
        raise ValueError("contract tool hash does not match implementation")
    frozen = datetime.fromisoformat(
        str(contract["frozen_at_utc"]).replace("Z", "+00:00")
    )
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("frozen_at_utc is missing an offset or is in the future")
    return {**contract, "result_sha256": claimed}, _sha256(raw)


def _validate_official_sources(contract: Mapping[str, object]) -> dict[str, object]:
    sources = _mapping(contract["official_sources"], name="official sources")
    faq_source = _mapping(sources["pegged_order_faq"], name="FAQ source")
    faq = _retained(faq_source).decode("utf-8")
    required = [str(value) for value in faq_source["required_phrases"]]
    missing = [phrase for phrase in required if phrase not in faq]
    if missing:
        raise ValueError("official pegged-order FAQ phrase gate failed")
    exchange_source = _mapping(sources["exchange_info"], name="exchange source")
    exchange = _mapping(json.loads(_retained(exchange_source)), name="exchange info")
    rows = {
        str(row.get("symbol")): row
        for row in (
            _mapping(value, name="exchange symbol")
            for value in exchange.get("symbols", [])
        )
    }
    symbols = [str(value) for value in contract["symbols"]]
    for symbol in symbols:
        row = rows.get(symbol)
        if not row or not (
            row.get("status") == "TRADING"
            and row.get("pegInstructionsAllowed") is True
            and "LIMIT_MAKER" in row.get("orderTypes", [])
        ):
            raise ValueError(f"PRIMARY_PEG LIMIT_MAKER unavailable for {symbol}")
    return {
        "faq_required_phrase_count": len(required),
        "exchange_server_time_ms": exchange.get("serverTime"),
        "eligible_symbols": symbols,
    }


def _ticker_rows(
    raw: bytes,
    *,
    symbols: list[str],
) -> dict[str, list[tuple[int, Decimal, Decimal]]]:
    result: dict[str, list[tuple[int, Decimal, Decimal]]] = {
        symbol: [] for symbol in symbols
    }
    for line_number, line in enumerate(raw.splitlines(), start=1):
        record = _mapping(json.loads(line), name=f"ticker line {line_number}")
        payload = _mapping(record.get("data"), name="ticker payload")
        symbol = str(payload.get("s") or "")
        if symbol not in result or not (
            record.get("stream") == f"{symbol.lower()}@ticker"
            and payload.get("e") == "24hrTicker"
        ):
            raise ValueError("unexpected retained ticker identity")
        received_ns = int(record["received_monotonic_ns"])
        bid = _decimal(payload.get("b"), name="best bid")
        ask = _decimal(payload.get("a"), name="best ask")
        if bid >= ask:
            raise ValueError("retained ticker is locked or crossed")
        if result[symbol] and received_ns <= result[symbol][-1][0]:
            raise ValueError("retained receipt monotonic time is not increasing")
        result[symbol].append((received_ns, bid, ask))
    return result


def _quantile(values: list[Decimal], numerator: int, denominator: int) -> Decimal:
    ordered = sorted(values)
    index = ((len(ordered) - 1) * numerator) // denominator
    return ordered[index]


def _analyze_symbol(
    rows: list[tuple[int, Decimal, Decimal]],
    *,
    lag_ms: int,
    maximum_pair_gap_ms: int,
) -> dict[str, object]:
    timestamps = [row[0] for row in rows]
    comparisons = 0
    buy_rejections = 0
    sell_rejections = 0
    either_rejections = 0
    state_changes = 0
    pair_gaps: list[Decimal] = []
    buy_abs_bps: list[Decimal] = []
    sell_abs_bps: list[Decimal] = []
    for index, (timestamp, bid, ask) in enumerate(rows):
        later = bisect_left(
            timestamps,
            timestamp + lag_ms * 1_000_000,
            index + 1,
        )
        if later >= len(rows):
            continue
        later_timestamp, later_bid, later_ask = rows[later]
        gap_ms = Decimal(later_timestamp - timestamp) / Decimal(1_000_000)
        if gap_ms > maximum_pair_gap_ms:
            raise ValueError("retained observation pair exceeded maximum gap")
        buy_rejected = bid >= later_ask
        sell_rejected = ask <= later_bid
        comparisons += 1
        buy_rejections += buy_rejected
        sell_rejections += sell_rejected
        either_rejections += buy_rejected or sell_rejected
        state_changes += bid != later_bid or ask != later_ask
        pair_gaps.append(gap_ms)
        buy_abs_bps.append(abs(later_bid - bid) / bid * Decimal(10_000))
        sell_abs_bps.append(abs(later_ask - ask) / ask * Decimal(10_000))
    if comparisons == 0:
        raise ValueError("retained window has no eligible comparisons")
    denominator = Decimal(comparisons)
    return {
        "ticker_rows": len(rows),
        "comparison_rows": comparisons,
        "fixed_buy_limit_maker_rejection_count": buy_rejections,
        "fixed_sell_limit_maker_rejection_count": sell_rejections,
        "either_side_rejection_count": either_rejections,
        "either_side_rejection_rate": str(Decimal(either_rejections) / denominator),
        "state_change_count": state_changes,
        "state_change_rate": str(Decimal(state_changes) / denominator),
        "mean_absolute_primary_peg_bid_reprice_bps": str(
            sum(buy_abs_bps, Decimal()) / denominator
        ),
        "mean_absolute_primary_peg_ask_reprice_bps": str(
            sum(sell_abs_bps, Decimal()) / denominator
        ),
        "pair_gap_ms": {
            "minimum": str(min(pair_gaps)),
            "median": str(_quantile(pair_gaps, 1, 2)),
            "p95": str(_quantile(pair_gaps, 95, 100)),
            "maximum": str(max(pair_gaps)),
        },
    }


def run(*, contract_path: Path, output: Path) -> dict[str, object]:
    contract, contract_file_sha = _contract(contract_path)
    official = _validate_official_sources(contract)
    symbols = [str(value) for value in contract["symbols"]]
    lag_ms = int(contract["observation_lag_ms"])
    maximum_gap = int(contract["maximum_observation_pair_gap_ms"])
    windows: list[dict[str, object]] = []
    recurrence = True
    for value in contract["windows"]:
        source = _mapping(value, name="window source")
        rows = _ticker_rows(_retained(source), symbols=symbols)
        results = {
            symbol: _analyze_symbol(
                rows[symbol],
                lag_ms=lag_ms,
                maximum_pair_gap_ms=maximum_gap,
            )
            for symbol in symbols
        }
        recurrence = recurrence and all(
            int(result["either_side_rejection_count"]) > 0
            for result in results.values()
        )
        windows.append(
            {
                "role": source["role"],
                "source_path": source["path"],
                "source_sha256": source["sha256"],
                "symbols": results,
            }
        )
    artifact: dict[str, object] = {
        "schema_version": "binance-primary-peg-retained-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mechanism": "PRIMARY_PEG_LIMIT_MAKER_matching_engine_arrival_price_overlay",
        "market_direction_forecast_required": False,
        "official_contract": official,
        "method": {
            "observation_lag_ms": lag_ms,
            "maximum_observation_pair_gap_ms": maximum_gap,
            "fixed_buy_rejection_rule": "origin_best_bid_greater_than_or_equal_to_later_best_ask",
            "fixed_sell_rejection_rule": "origin_best_ask_less_than_or_equal_to_later_best_bid",
            "primary_peg_counterfactual": "matching_engine_selects_later_same_side_best_and_queues_after_existing_best_price_orders",
        },
        "windows": windows,
        "verdict": {
            "status": "recurrent_order_acceptance_overlay_candidate_not_profit_evidence",
            "recurrent_in_every_window_and_symbol": recurrence,
            "accepted_edge": False,
            "profitability_claim": False,
            "deployment_ready": False,
            "trading_authority": False,
        },
        "authority": {
            "network_requests": 0,
            "credentials_used": False,
            "orders_or_cancellations": 0,
            "account_or_funded_actions": 0,
        },
        "sources": {
            "contract_path": contract_path.resolve().relative_to(ROOT).as_posix(),
            "contract_file_sha256": contract_file_sha,
            "contract_result_sha256": contract["result_sha256"],
            "official_sources": contract["official_sources"],
            "tool_sha256": _sha256(Path(__file__).read_bytes()),
        },
        "limitations": [
            "The public ticker streams sample state at roughly one-second cadence; this audit does not infer subsecond matching-engine state or a realistic colocated latency distribution.",
            "A fixed LIMIT_MAKER crossing the later opposite quote is a deterministic rejection counterfactual, not a missed-fill or profit amount.",
            "PRIMARY_PEG joins behind existing orders at the selected best price and does not prove queue priority, fills, spread capture, adverse-selection reduction, or realized PnL.",
            "The two windows are same-day discovery and validation observations, not full bull, bear, sideways, choppy, volatility, liquidity, or latency-stress coverage.",
        ],
        "next_trigger": "explicit separate Binance Spot testnet or paper order authority for one minimum-size PRIMARY_PEG LIMIT_MAKER acknowledgement/cancel comparison against a frozen fixed-price counterfactual; no mainnet order authority",
    }
    artifact["result_sha256"] = _sha256(_canonical(artifact))
    write_bytes_atomic(output, _canonical(artifact) + b"\n")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(contract_path=args.contract, output=args.output)
    print(
        json.dumps(
            {
                "status": result["verdict"]["status"],
                "recurrent": result["verdict"]["recurrent_in_every_window_and_symbol"],
                "result_sha256": result["result_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
