"""Run one frozen public BTC/ETH USD1-versus-USDT funding screen."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
BASE_TOOL_PATH = ROOT / "tools/screen_binance_u_usdt_funding_differential.py"
CONTRACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-usd1-usdt-funding-differential-contract-v1.json"
)
DEFAULT_OUTPUT = ROOT / (
    "docs/model-research/action-value/"
    "binance-usd1-usdt-funding-differential-v1-2026-08-26.json"
)
DEFAULT_JOURNAL = ROOT / ("data/binance-usd1-usdt-funding-differential-v1/journal.json")
IMPLEMENTATION_PATH = Path(__file__).resolve()
FUTURES_BASE_URL = "https://fapi.binance.com"
SPOT_BASE_URL = "https://api.binance.com"
PAIRS = (
    {"base": "BTC", "usd1_symbol": "BTCUSD1", "usdt_symbol": "BTCUSDT"},
    {"base": "ETH", "usd1_symbol": "ETHUSD1", "usdt_symbol": "ETHUSDT"},
)
SCHEMA_VERSION = "binance-usd1-usdt-funding-differential-v1"
TEN_THOUSAND = Decimal(10_000)


def _load_module(path: Path) -> object:
    spec = importlib.util.spec_from_file_location("u_usdt_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("U-USDT helper cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_module(BASE_TOOL_PATH)


def _canonical_json(value: object) -> str:
    return BASE._canonical_json(value)  # noqa: SLF001


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, name: str) -> dict[str, object]:
    return BASE._mapping(value, name=name)  # noqa: SLF001


def _list(value: object, *, name: str) -> list[object]:
    return BASE._list(value, name=name)  # noqa: SLF001


def _write_journal(path: Path, journal: Mapping[str, object]) -> None:
    write_bytes_atomic(path, (_canonical_json(journal) + "\n").encode("ascii"))


def _load_contract() -> tuple[dict[str, object], str]:
    contract = _mapping(json.loads(CONTRACT_PATH.read_bytes()), name="contract")
    claimed = str(contract.pop("result_sha256"))
    actual = _sha256(_canonical_json(contract).encode("ascii"))
    if claimed != actual:
        raise ValueError("contract canonical hash differs")
    contract["result_sha256"] = claimed
    if contract.get("status") != "frozen_before_any_usd1_funding_history_access":
        raise ValueError("contract is not frozen")
    if contract["implementation_sha256"] != _sha256(IMPLEMENTATION_PATH.read_bytes()):
        raise ValueError("implementation hash differs from frozen contract")
    return contract, actual


def _get(
    session: requests.Session,
    *,
    base_url: str,
    path: str,
    name: str,
    params: Mapping[str, object] | None,
    journal_path: Path,
    journal: dict[str, object],
) -> object:
    response = session.get(f"{base_url}{path}", params=params, timeout=30)
    receipt = {
        "name": name,
        "method": "GET",
        "url": response.url,
        "status_code": response.status_code,
        "payload_bytes": len(response.content),
        "payload_sha256": _sha256(response.content),
    }
    if response.status_code == 429:
        raise RuntimeError("Binance rate limit reached; stopped without retry")
    response.raise_for_status()
    try:
        decoded = response.json()
    except requests.JSONDecodeError as exc:
        raise ValueError(f"{name} did not return JSON") from exc
    _list(journal["responses"], name="journal responses").append(
        {"receipt": receipt, "payload": decoded}
    )
    _write_journal(journal_path, journal)
    return decoded


def _selected_contracts(raw: object) -> dict[str, dict[str, object]]:
    exchange = _mapping(raw, name="exchange info")
    required = {
        str(pair[key]) for pair in PAIRS for key in ("usd1_symbol", "usdt_symbol")
    }
    selected = {
        str(item.get("symbol")): item
        for value in _list(exchange["symbols"], name="exchange symbols")
        for item in [_mapping(value, name="exchange symbol")]
        if item.get("symbol") in required
    }
    if set(selected) != required:
        raise ValueError("exchange info lacks a required perpetual")
    for pair in PAIRS:
        for key, quote in (("usd1_symbol", "USD1"), ("usdt_symbol", "USDT")):
            symbol = str(pair[key])
            item = selected[symbol]
            if (
                item.get("baseAsset") != pair["base"]
                or item.get("quoteAsset") != quote
                or item.get("marginAsset") != quote
                or item.get("contractType") != "PERPETUAL"
                or item.get("status") != "TRADING"
            ):
                raise ValueError(f"{symbol} contract identity differs")
    return dict(sorted(selected.items()))


def _book_rows(raw: object) -> dict[str, dict[str, object]]:
    required = {
        str(pair[key]) for pair in PAIRS for key in ("usd1_symbol", "usdt_symbol")
    }
    rows = {
        str(row.get("symbol")): row
        for value in _list(raw, name="futures books")
        for row in [_mapping(value, name="futures book")]
        if row.get("symbol") in required
    }
    if set(rows) != required:
        raise ValueError("futures books lack a required symbol")
    return rows


def _spread_bips(book: Mapping[str, object]) -> Decimal:
    ask = Decimal(str(book["askPrice"]))
    bid = Decimal(str(book["bidPrice"]))
    if ask <= 0 or bid <= 0 or ask < bid:
        raise ValueError("book prices are invalid")
    return (ask - bid) / ((ask + bid) / 2) * TEN_THOUSAND


def _fx_evidence(raw_klines: object, raw_book: object) -> dict[str, object]:
    closes: list[tuple[int, Decimal]] = []
    for value in _list(raw_klines, name="USD1USDT daily klines"):
        row = _list(value, name="USD1USDT kline")
        if len(row) < 7:
            raise ValueError("USD1USDT kline is malformed")
        closes.append((int(row[0]), Decimal(str(row[4]))))
    if len(closes) < 31 or any(
        closes[index][0] <= closes[index - 1][0] for index in range(1, len(closes))
    ):
        raise ValueError("USD1USDT daily history is insufficient or unordered")
    declines = [
        (closes[index][1] / closes[index - 30][1] - 1) * TEN_THOUSAND
        for index in range(30, len(closes))
    ]
    book = _mapping(raw_book, name="USD1USDT book")
    if book.get("symbol") != "USD1USDT":
        raise ValueError("USD1USDT book identity differs")
    return {
        "daily_row_count": len(closes),
        "first_open_time_ms": closes[0][0],
        "last_open_time_ms": closes[-1][0],
        "worst_30_day_close_change_bips": str(min(declines)),
        "worst_30_day_decline_stress_bips": str(max(Decimal(0), -min(declines))),
        "current_book": book,
        "current_spread_bips": str(_spread_bips(book)),
    }


def _evaluate_pair(
    pair: Mapping[str, object],
    histories: Mapping[str, object],
    books: Mapping[str, Mapping[str, object]],
    fx: Mapping[str, object],
    contract: Mapping[str, object],
) -> dict[str, object]:
    usd1_symbol = str(pair["usd1_symbol"])
    usdt_symbol = str(pair["usdt_symbol"])
    usd1_rows = BASE._history(histories[usd1_symbol], symbol=usd1_symbol)  # noqa: SLF001
    usdt_rows = BASE._history(histories[usdt_symbol], symbol=usdt_symbol)  # noqa: SLF001
    aligned = sorted(set(usd1_rows) & set(usdt_rows))
    causal = _mapping(contract["causal_roles"], name="causal roles")
    if len(aligned) < int(causal["minimum_aligned_rows"]):
        return {
            "base": pair["base"],
            "aligned_row_count": len(aligned),
            "public_persistence_candidate": False,
            "rejection_reasons": ["insufficient_aligned_rows"],
        }
    training_end = len(aligned) // 2
    validation_end = training_end + (len(aligned) - training_end) // 2
    role_times = {
        "training": aligned[:training_end],
        "validation": aligned[training_end:validation_end],
        "test": aligned[validation_end:],
    }
    if any(
        len(values) < int(causal["minimum_rows_per_role"])
        for values in role_times.values()
    ):
        raise ValueError("causal role has too few rows")
    training = role_times["training"]
    usd1_mean = sum(
        (usd1_rows[epoch]["rate"] for epoch in training), Decimal(0)
    ) / Decimal(len(training))
    usdt_mean = sum(
        (usdt_rows[epoch]["rate"] for epoch in training), Decimal(0)
    ) / Decimal(len(training))
    short_symbol, long_symbol = (
        (usd1_symbol, usdt_symbol)
        if usd1_mean > usdt_mean
        else (usdt_symbol, usd1_symbol)
    )
    parsed = {usd1_symbol: usd1_rows, usdt_symbol: usdt_rows}
    economic = _mapping(contract["economic_hurdles"], name="economic hurdles")
    execution = Decimal(str(economic["round_trip_execution_stress_bips"]))
    annual = Decimal(
        str(economic["annual_opportunity_hurdle_bips_per_capital_leg"])
    ) * Decimal(str(economic["gross_capital_legs"]))
    depeg_stress = Decimal(str(fx["worst_30_day_decline_stress_bips"]))
    roles: dict[str, object] = {}
    rejection_reasons: list[str] = []
    for name, times in role_times.items():
        differences = [
            parsed[short_symbol][epoch]["rate"] - parsed[long_symbol][epoch]["rate"]
            for epoch in times
        ]
        role = BASE._role(  # noqa: SLF001
            differences,
            times,
            execution_bips=execution,
            annual_hurdle_bips=annual,
        )
        role["net_after_observed_fx_stress_bips"] = str(
            Decimal(str(role["net_after_frozen_hurdles_bips"])) - depeg_stress
        )
        roles[name] = role
        if Decimal(str(role["net_after_observed_fx_stress_bips"])) <= 0:
            rejection_reasons.append(f"{name}_net_after_fx_stress_nonpositive")
        if name != "training" and (
            Decimal(str(role["first_half_gross_bips"])) <= 0
            or Decimal(str(role["second_half_gross_bips"])) <= 0
        ):
            rejection_reasons.append(f"{name}_chronological_half_nonpositive")
        if name != "training" and Decimal(str(role["positive_fraction"])) < Decimal(
            "0.60"
        ):
            rejection_reasons.append(f"{name}_positive_fraction_below_60_percent")
        if Decimal(str(role["maximum_cumulative_drawdown_bips"])) > Decimal("50"):
            rejection_reasons.append(f"{name}_drawdown_above_50_bips")
    return {
        "base": pair["base"],
        "symbols": {"short": short_symbol, "long": long_symbol},
        "aligned_row_count": len(aligned),
        "training_means": {"usd1": str(usd1_mean), "usdt": str(usdt_mean)},
        "current_round_trip_book_spread_bips": str(
            _spread_bips(books[usd1_symbol]) + _spread_bips(books[usdt_symbol])
        ),
        "roles": roles,
        "rejection_reasons": rejection_reasons,
        "public_persistence_candidate": not rejection_reasons,
    }


def run(*, output: Path, journal_path: Path) -> dict[str, object]:
    contract, contract_hash = _load_contract()
    if output.exists() or journal_path.exists():
        raise ValueError("one-shot output or journal already exists")
    journal: dict[str, object] = {
        "schema_version": f"{SCHEMA_VERSION}-journal",
        "contract_sha256": contract_hash,
        "responses": [],
    }
    session = requests.Session()
    session.headers.update(
        {
            "Accept-Encoding": "identity",
            "User-Agent": "simple-ai-trading-public-edge-research/1.0",
        }
    )
    exchange = _get(
        session,
        base_url=FUTURES_BASE_URL,
        path="/fapi/v1/exchangeInfo",
        name="futures_exchange_info",
        params=None,
        journal_path=journal_path,
        journal=journal,
    )
    selected = _selected_contracts(exchange)
    histories: dict[str, object] = {}
    for pair in PAIRS:
        for key in ("usd1_symbol", "usdt_symbol"):
            symbol = str(pair[key])
            histories[symbol] = _get(
                session,
                base_url=FUTURES_BASE_URL,
                path="/fapi/v1/fundingRate",
                name=f"funding_{symbol}",
                params={"symbol": symbol, "limit": 500},
                journal_path=journal_path,
                journal=journal,
            )
    books = _book_rows(
        _get(
            session,
            base_url=FUTURES_BASE_URL,
            path="/fapi/v1/ticker/bookTicker",
            name="futures_books",
            params=None,
            journal_path=journal_path,
            journal=journal,
        )
    )
    fx_klines = _get(
        session,
        base_url=SPOT_BASE_URL,
        path="/api/v3/klines",
        name="usd1usdt_daily_klines",
        params={"symbol": "USD1USDT", "interval": "1d", "limit": 500},
        journal_path=journal_path,
        journal=journal,
    )
    fx_book = _get(
        session,
        base_url=SPOT_BASE_URL,
        path="/api/v3/ticker/bookTicker",
        name="usd1usdt_book",
        params={"symbol": "USD1USDT"},
        journal_path=journal_path,
        journal=journal,
    )
    if len(_list(journal["responses"], name="responses")) != 8:
        raise ValueError("frozen request count differs")
    fx = _fx_evidence(fx_klines, fx_book)
    evaluations = [
        _evaluate_pair(pair, histories, books, fx, contract) for pair in PAIRS
    ]
    candidates = [row for row in evaluations if row["public_persistence_candidate"]]
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "contract": {
            "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "result_sha256": contract_hash,
        },
        "implementation": {
            "path": IMPLEMENTATION_PATH.relative_to(ROOT).as_posix(),
            "sha256": _sha256(IMPLEMENTATION_PATH.read_bytes()),
        },
        "raw_evidence": {
            "journal_path": journal_path.relative_to(ROOT).as_posix(),
            "journal_sha256": _sha256(journal_path.read_bytes()),
            "response_count": 8,
        },
        "selected_contracts": selected,
        "fx_evidence": fx,
        "evaluations": evaluations,
        "verdict": {
            "public_persistence_candidate_count": len(candidates),
            "accepted_edge": False,
            "profitability_claim": False,
            "credentials_used": False,
            "orders_submitted": 0,
            "reason": "public_history_and_books_only_exact_account_cost_collateral_conversion_borrow_and_paper_fill_evidence_remain_unresolved",
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
    result = run(output=args.output.resolve(), journal_path=args.journal.resolve())
    print(f"result_sha256={result['result_sha256']}")
    print(_canonical_json(result["verdict"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
