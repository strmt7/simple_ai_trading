"""Run the frozen public Binance zero-fee stablecoin-cycle screen."""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_DOWN
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

import requests

from simple_ai_trading.storage import write_bytes_atomic


BASE_URL = "https://api.binance.com"
SCHEMA_VERSION = "binance-zero-fee-stablecoin-cycle-v1"
CONTRACT_PATH = Path(
    "docs/model-research/action-value/"
    "binance-zero-fee-stablecoin-cycle-recovery-contract-v2.json"
)
DEFAULT_OUTPUT = Path(
    "docs/model-research/action-value/"
    "binance-zero-fee-stablecoin-cycle-recovery-v2-2026-08-26.json"
)
DEFAULT_JOURNAL = Path(
    "data/binance-zero-fee-stablecoin-cycle-recovery-v2/raw/journal.jsonl"
)
IMPLEMENTATION_PATH = Path("tools/screen_binance_zero_fee_stablecoin_cycles.py")
TEN_THOUSAND = Decimal(10_000)


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


def _contract() -> tuple[dict[str, object], str]:
    contract = _mapping(json.loads(CONTRACT_PATH.read_text()), name="contract")
    declared = str(contract.pop("contract_sha256"))
    actual = _sha256(_canonical_json(contract).encode("ascii"))
    if declared != actual:
        raise ValueError(f"contract hash differs: declared={declared} actual={actual}")
    contract["contract_sha256"] = declared
    if contract.get("status") != "frozen_before_recovery_book_outcome_access":
        raise ValueError("contract is not frozen")
    return contract, actual


def _floor_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        raise ValueError("step must be positive")
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def _market_step(symbol: Mapping[str, object]) -> Decimal:
    fallback: Decimal | None = None
    for raw_filter in _list(symbol.get("filters"), name="symbol filters"):
        item = _mapping(raw_filter, name="symbol filter")
        if item.get("filterType") == "MARKET_LOT_SIZE":
            step = Decimal(str(item["stepSize"]))
            if step > 0:
                return step
        if item.get("filterType") == "LOT_SIZE":
            step = Decimal(str(item["stepSize"]))
            if step > 0:
                fallback = step
    if fallback is None:
        raise ValueError(f"{symbol.get('symbol')} lacks a positive lot step")
    return fallback


def _validate_exchange_info(
    raw: object, required_symbols: Sequence[str]
) -> dict[str, dict[str, object]]:
    info = _mapping(raw, name="exchange info")
    selected = {
        str(item.get("symbol")): item
        for value in _list(info.get("symbols"), name="exchange symbols")
        for item in [_mapping(value, name="exchange symbol")]
        if item.get("symbol") in required_symbols
    }
    if set(selected) != set(required_symbols):
        raise ValueError("exchange info lacks a required symbol")
    for symbol, item in selected.items():
        if (
            item.get("status") != "TRADING"
            or item.get("isSpotTradingAllowed") is not True
        ):
            raise ValueError(f"{symbol} is not spot TRADING")
        if item.get("quoteOrderQtyMarketAllowed") is not True:
            raise ValueError(f"{symbol} lacks quote-order market support")
        if "MARKET" not in _list(item.get("orderTypes"), name=f"{symbol} order types"):
            raise ValueError(f"{symbol} lacks MARKET orders")
        _market_step(item)
    return selected


def _book_map(
    raw: object, required_symbols: Sequence[str]
) -> dict[str, dict[str, Decimal]]:
    rows = _list(raw, name="book ticker")
    result: dict[str, dict[str, Decimal]] = {}
    for value in rows:
        item = _mapping(value, name="book row")
        symbol = str(item.get("symbol"))
        if symbol not in required_symbols:
            raise ValueError(f"unexpected symbol {symbol}")
        parsed = {
            key: Decimal(str(item[key]))
            for key in ("bidPrice", "bidQty", "askPrice", "askQty")
        }
        if any(number <= 0 for number in parsed.values()):
            raise ValueError(f"{symbol} contains a nonpositive book value")
        result[symbol] = parsed
    if set(result) != set(required_symbols):
        raise ValueError("book ticker lacks a required symbol")
    return result


def _leg(
    amount: Decimal,
    *,
    symbol: str,
    side: str,
    book: Mapping[str, Decimal],
    market_step: Decimal,
) -> tuple[Decimal, bool]:
    if side == "BUY":
        base_qty = _floor_step(amount / book["askPrice"], market_step)
        capacity_ok = base_qty <= book["askQty"]
        return base_qty, capacity_ok
    if side == "SELL":
        base_qty = _floor_step(amount, market_step)
        capacity_ok = base_qty <= book["bidQty"]
        return base_qty * book["bidPrice"], capacity_ok
    raise ValueError(f"unknown side {side} for {symbol}")


def _evaluate_orientation(
    start: Decimal,
    *,
    legs: Sequence[tuple[str, str]],
    books: Mapping[str, Mapping[str, Decimal]],
    steps: Mapping[str, Decimal],
    stress_bips: Decimal,
) -> dict[str, object]:
    amount = start
    capacity_ok = True
    for symbol, side in legs:
        amount, leg_capacity = _leg(
            amount,
            symbol=symbol,
            side=side,
            book=books[symbol],
            market_step=steps[symbol],
        )
        capacity_ok = capacity_ok and leg_capacity
    gross_bips = (amount / start - Decimal(1)) * TEN_THOUSAND
    stressed_bips = gross_bips - stress_bips
    return {
        "ending_amount": _decimal_text(amount),
        "gross_bips": _decimal_text(gross_bips),
        "stressed_bips": _decimal_text(stressed_bips),
        "top_level_capacity_ok": capacity_ok,
        "positive_after_stress": capacity_ok and stressed_bips > 0,
    }


def _orientations() -> dict[str, tuple[tuple[str, str], ...]]:
    return {
        "USDT_U_RLUSD_forward": (
            ("UUSDT", "BUY"),
            ("RLUSDU", "BUY"),
            ("RLUSDUSDT", "SELL"),
        ),
        "USDT_U_RLUSD_reverse": (
            ("RLUSDUSDT", "BUY"),
            ("RLUSDU", "SELL"),
            ("UUSDT", "SELL"),
        ),
        "USDC_U_USD1_forward": (
            ("UUSDC", "BUY"),
            ("USD1U", "BUY"),
            ("USD1USDC", "SELL"),
        ),
        "USDC_U_USD1_reverse": (
            ("USD1USDC", "BUY"),
            ("USD1U", "SELL"),
            ("UUSDC", "SELL"),
        ),
    }


def _append_journal(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write((_canonical_json(value) + "\n").encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())


def _get_json(
    session: requests.Session, path: str, *, params: Mapping[str, object] | None = None
) -> tuple[object, dict[str, object]]:
    started_ns = time.time_ns()
    response = session.get(f"{BASE_URL}{path}", params=params, timeout=10)
    ended_ns = time.time_ns()
    response.raise_for_status()
    payload = response.content
    try:
        body = response.json()
    except requests.JSONDecodeError as exc:
        raise ValueError(f"{path} did not return JSON") from exc
    return body, {
        "request_started_ns": started_ns,
        "response_ended_ns": ended_ns,
        "request_elapsed_ms": (ended_ns - started_ns) // 1_000_000,
        "status_code": response.status_code,
        "url": response.url,
        "response_bytes": len(payload),
        "response_sha256": _sha256(payload),
    }


def _median(values: Sequence[Decimal]) -> Decimal:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def _summaries(
    samples: Sequence[Mapping[str, object]], *, sizes: Sequence[Decimal]
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for orientation in _orientations():
        for size in sizes:
            rows = [
                _mapping(
                    _mapping(sample["evaluations"], name="evaluations")[orientation],
                    name="orientation evaluation",
                )[str(size)]
                for sample in samples
            ]
            typed = [_mapping(row, name="size evaluation") for row in rows]
            positives = [
                Decimal(str(row["stressed_bips"]))
                for row in typed
                if row["positive_after_stress"] is True
            ]
            block_counts = [
                sum(
                    row["positive_after_stress"] is True
                    for row in typed[index : index + 120]
                )
                for index in range(0, 600, 120)
            ]
            empirical_candidate = (
                len(typed) == 600
                and len(positives) >= 30
                and all(count > 0 for count in block_counts)
                and _median(positives) > 0
            )
            summaries.append(
                {
                    "orientation": orientation,
                    "starting_size": _decimal_text(size),
                    "positive_sample_count": len(positives),
                    "positive_sample_fraction": _decimal_text(
                        Decimal(len(positives)) / Decimal(len(typed))
                    ),
                    "positive_counts_by_120_sample_block": block_counts,
                    "maximum_stressed_bips": _decimal_text(
                        max(Decimal(str(row["stressed_bips"])) for row in typed)
                    ),
                    "median_positive_stressed_bips": (
                        _decimal_text(_median(positives)) if positives else None
                    ),
                    "empirical_candidate": empirical_candidate,
                }
            )
    return summaries


def run(*, output: Path, journal: Path) -> dict[str, object]:
    contract, contract_hash = _contract()
    capture = _mapping(contract["capture"], name="capture")
    universe = _mapping(contract["universe"], name="universe")
    sample_count = int(capture["sample_count"])
    interval_ms = int(capture["interval_ms"])
    sizes = [
        Decimal(str(value))
        for value in _list(universe["starting_quote_sizes"], name="sizes")
    ]
    stress_bips = Decimal(
        str(
            _mapping(contract["execution_model"], name="execution model")[
                "operational_stress_bips"
            ]
        )
    )
    required_symbols = sorted(
        {
            str(symbol)
            for raw_cycle in _list(universe["cycles"], name="cycles")
            for symbol in _list(
                _mapping(raw_cycle, name="cycle")["symbols"], name="cycle symbols"
            )
        }
    )
    if journal.exists():
        raise ValueError(f"one-shot journal already exists: {journal}")

    session = requests.Session()
    session.headers.update(
        {
            "Accept-Encoding": "identity",
            "User-Agent": "simple-ai-trading-public-edge-research/1.0",
        }
    )
    exchange_raw, exchange_receipt = _get_json(session, "/api/v3/exchangeInfo")
    selected = _validate_exchange_info(exchange_raw, required_symbols)
    steps = {symbol: _market_step(item) for symbol, item in selected.items()}
    exchange_record = {
        "record_type": "exchange_info",
        "receipt": exchange_receipt,
        "selected_symbols": selected,
    }
    _append_journal(journal, exchange_record)

    samples: list[dict[str, object]] = []
    capture_started_ns = time.time_ns()
    next_sample_ns = time.monotonic_ns()
    encoded_symbols = json.dumps(required_symbols, separators=(",", ":"))
    for sample_index in range(sample_count):
        remaining_ns = next_sample_ns - time.monotonic_ns()
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1_000_000_000)
        final_error: Exception | None = None
        for attempt in range(1, 3):
            try:
                raw_book, receipt = _get_json(
                    session,
                    "/api/v3/ticker/bookTicker",
                    params={"symbols": encoded_symbols},
                )
                _append_journal(
                    journal,
                    {
                        "record_type": "book_response",
                        "sample_index": sample_index,
                        "attempt": attempt,
                        "receipt": receipt,
                        "book": raw_book,
                    },
                )
                books = _book_map(raw_book, required_symbols)
                evaluations: dict[str, object] = {}
                for name, legs in _orientations().items():
                    evaluations[name] = {
                        str(size): _evaluate_orientation(
                            size,
                            legs=legs,
                            books=books,
                            steps=steps,
                            stress_bips=stress_bips,
                        )
                        for size in sizes
                    }
                sample = {
                    "record_type": "book_evaluation",
                    "sample_index": sample_index,
                    "attempt": attempt,
                    "receipt": receipt,
                    "evaluations": evaluations,
                }
                _append_journal(journal, sample)
                samples.append(sample)
                final_error = None
                break
            except (requests.RequestException, ValueError) as exc:
                final_error = exc
        if final_error is not None:
            raise ValueError(f"sample {sample_index} failed") from final_error
        next_sample_ns += interval_ms * 1_000_000
    capture_ended_ns = time.time_ns()

    elapsed_ms = (capture_ended_ns - capture_started_ns) // 1_000_000
    maximum_request_ms = max(
        int(_mapping(sample["receipt"], name="receipt")["request_elapsed_ms"])
        for sample in samples
    )
    summaries = _summaries(samples, sizes=sizes)
    capture_valid = (
        len(samples) == sample_count
        and elapsed_ms >= int(capture["minimum_elapsed_ms"])
        and maximum_request_ms <= int(capture["maximum_request_elapsed_ms"])
    )
    empirical_candidates = [
        row for row in summaries if capture_valid and row["empirical_candidate"] is True
    ]
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_ns": time.time_ns(),
        "contract": {
            "path": str(CONTRACT_PATH.as_posix()),
            "contract_sha256": contract_hash,
        },
        "implementation": {
            "path": str(IMPLEMENTATION_PATH.as_posix()),
            "sha256": _sha256(IMPLEMENTATION_PATH.read_bytes()),
        },
        "raw_evidence": {
            "journal_path": str(journal.as_posix()),
            "journal_sha256": _sha256(journal.read_bytes()),
            "record_count": sum(1 for _ in journal.open("rb")),
        },
        "capture": {
            "sample_count": len(samples),
            "elapsed_ms": elapsed_ms,
            "maximum_request_elapsed_ms": maximum_request_ms,
            "capture_valid": capture_valid,
        },
        "summaries": summaries,
        "verdict": {
            "empirical_candidate_count": len(empirical_candidates),
            "empirical_candidates": empirical_candidates,
            "accepted_edge": False,
            "deployment_ready": False,
            "reason": (
                "public_five_minute_screen_only_account_eligibility_fee_confirmation_"
                "paper_fills_reconciliation_and_cross_window_persistence_absent"
            ),
        },
        "authority": {
            "orders_submitted": 0,
            "credentials_used": False,
            "paper_authority": False,
            "live_authority": False,
        },
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    write_bytes_atomic(output, (_canonical_json(result) + "\n").encode("ascii"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    args = parser.parse_args()
    result = run(output=args.output, journal=args.journal)
    print(f"result_sha256={result['result_sha256']}")
    print(_canonical_json(result["verdict"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
