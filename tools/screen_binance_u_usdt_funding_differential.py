"""Run the frozen public BTCU/ETHU versus USDT funding screen."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import requests

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
BASE_TOOL_PATH = ROOT / "tools" / "screen_binance_cross_stablecoin_funding.py"
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-u-usdt-funding-differential-contract-v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-u-usdt-funding-differential-v1-2026-08-26.json"
)
DEFAULT_JOURNAL = (
    ROOT / "data" / "binance-u-usdt-funding-differential-v1" / "journal.json"
)
IMPLEMENTATION_PATH = ROOT / "tools" / "screen_binance_u_usdt_funding_differential.py"
BASE_URL = "https://fapi.binance.com"
DAY_MS = Decimal(86_400_000)
YEAR_DAYS = Decimal(365)
TEN_THOUSAND = Decimal(10_000)
SCHEMA_VERSION = "binance-u-usdt-funding-differential-v1"


def _load_module(path: Path) -> object:
    spec = importlib.util.spec_from_file_location("cross_stablecoin_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cross-stablecoin helper cannot be loaded")
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


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _load_contract() -> tuple[dict[str, object], str]:
    contract = _mapping(json.loads(CONTRACT_PATH.read_text()), name="contract")
    declared = str(contract.pop("contract_sha256"))
    actual = _sha256(_canonical_json(contract).encode("ascii"))
    if declared != actual:
        raise ValueError(f"contract hash differs: declared={declared} actual={actual}")
    contract["contract_sha256"] = declared
    if contract.get("status") != "frozen_before_any_u_usdt_funding_history_access":
        raise ValueError("contract is not frozen")
    return contract, actual


def _write_journal(path: Path, journal: Mapping[str, object]) -> None:
    write_bytes_atomic(path, (_canonical_json(journal) + "\n").encode("ascii"))


def _get(
    session: requests.Session,
    *,
    path: str,
    name: str,
    params: Mapping[str, object] | None,
    journal_path: Path,
    journal: dict[str, object],
) -> object:
    started_ms = time.time_ns() // 1_000_000
    response = session.get(f"{BASE_URL}{path}", params=params, timeout=30)
    ended_ms = time.time_ns() // 1_000_000
    response.raise_for_status()
    payload = response.content
    try:
        decoded = response.json()
    except requests.JSONDecodeError as exc:
        raise ValueError(f"{name} did not return JSON") from exc
    _list(journal["responses"], name="journal responses").append(
        {
            "name": name,
            "receipt": {
                "url": response.url,
                "status_code": response.status_code,
                "requested_before_ms": started_ms,
                "received_after_ms": ended_ms,
                "request_elapsed_ms": ended_ms - started_ms,
                "payload_bytes": len(payload),
                "payload_sha256": _sha256(payload),
            },
            "payload": decoded,
        }
    )
    _write_journal(journal_path, journal)
    return decoded


def _selected_contracts(
    raw: object, pairs: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    exchange = _mapping(raw, name="exchange info")
    required = {str(pair[key]) for pair in pairs for key in ("u_symbol", "usdt_symbol")}
    selected = {
        str(item.get("symbol")): item
        for value in _list(exchange["symbols"], name="exchange symbols")
        for item in [_mapping(value, name="exchange symbol")]
        if item.get("symbol") in required
    }
    if set(selected) != required:
        raise ValueError("exchange info lacks a required perpetual")
    for pair in pairs:
        for suffix, quote in (("u_symbol", "U"), ("usdt_symbol", "USDT")):
            symbol = str(pair[suffix])
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


def _history(raw: object, *, symbol: str) -> dict[int, dict[str, Decimal]]:
    result: dict[int, dict[str, Decimal]] = {}
    for value in _list(raw, name=f"{symbol} history"):
        row = _mapping(value, name=f"{symbol} row")
        if row.get("symbol") != symbol:
            raise ValueError(f"{symbol} row identity differs")
        epoch = int(row["fundingTime"])
        if epoch in result:
            raise ValueError(f"{symbol} duplicate funding time")
        result[epoch] = {
            "rate": Decimal(str(row["fundingRate"])),
            "mark": Decimal(str(row["markPrice"])),
        }
    return dict(sorted(result.items()))


def _role(
    values: Sequence[Decimal],
    times: Sequence[int],
    *,
    execution_bips: Decimal,
    annual_hurdle_bips: Decimal,
) -> dict[str, object]:
    if len(values) != len(times) or not values:
        raise ValueError("role values and times must be nonempty and aligned")
    duration_days = Decimal(times[-1] - times[0] + 28_800_000) / DAY_MS
    capital_hurdle = annual_hurdle_bips * duration_days / YEAR_DAYS
    gross_bips = sum(values, Decimal(0)) * TEN_THOUSAND
    net_bips = gross_bips - execution_bips - capital_hurdle
    midpoint = len(values) // 2
    halves = (values[:midpoint], values[midpoint:])
    return {
        "row_count": len(values),
        "start_time_ms": times[0],
        "end_time_ms": times[-1],
        "duration_days": _decimal_text(duration_days),
        "statistics": BASE._statistics(values),  # noqa: SLF001
        "gross_bips": _decimal_text(gross_bips),
        "execution_hurdle_bips": _decimal_text(execution_bips),
        "capital_hurdle_bips": _decimal_text(capital_hurdle),
        "net_after_frozen_hurdles_bips": _decimal_text(net_bips),
        "positive_fraction": _decimal_text(
            Decimal(sum(value > 0 for value in values)) / Decimal(len(values))
        ),
        "first_half_gross_bips": _decimal_text(
            sum(halves[0], Decimal(0)) * TEN_THOUSAND
        ),
        "second_half_gross_bips": _decimal_text(
            sum(halves[1], Decimal(0)) * TEN_THOUSAND
        ),
        "maximum_cumulative_drawdown_bips": _decimal_text(
            BASE._drawdown_bips(values)  # noqa: SLF001
        ),
    }


def _evaluate_pair(
    pair: Mapping[str, object],
    histories: Mapping[str, object],
    contract: Mapping[str, object],
) -> dict[str, object]:
    u_symbol = str(pair["u_symbol"])
    usdt_symbol = str(pair["usdt_symbol"])
    u_rows = _history(histories[u_symbol], symbol=u_symbol)
    usdt_rows = _history(histories[usdt_symbol], symbol=usdt_symbol)
    aligned = sorted(set(u_rows) & set(usdt_rows))
    causal = _mapping(contract["causal_roles"], name="causal roles")
    minimum = int(causal["minimum_aligned_rows"])
    if len(aligned) < minimum:
        return {
            "base": pair["base"],
            "symbols": [u_symbol, usdt_symbol],
            "aligned_row_count": len(aligned),
            "public_persistence_candidate": False,
            "rejection_reasons": ["fewer_than_120_aligned_rows"],
        }
    train_end = len(aligned) // 2
    validation_end = train_end + (len(aligned) - train_end) // 2
    role_times = {
        "training": aligned[:train_end],
        "validation": aligned[train_end:validation_end],
        "test": aligned[validation_end:],
    }
    if any(
        len(values) < int(causal["minimum_rows_per_role"])
        for values in role_times.values()
    ):
        return {
            "base": pair["base"],
            "symbols": [u_symbol, usdt_symbol],
            "aligned_row_count": len(aligned),
            "public_persistence_candidate": False,
            "rejection_reasons": ["one_or_more_roles_below_30_rows"],
        }
    training = role_times["training"]
    u_mean = sum((u_rows[epoch]["rate"] for epoch in training), Decimal(0)) / Decimal(
        len(training)
    )
    usdt_mean = sum(
        (usdt_rows[epoch]["rate"] for epoch in training), Decimal(0)
    ) / Decimal(len(training))
    short_symbol, long_symbol = (
        (u_symbol, usdt_symbol) if u_mean > usdt_mean else (usdt_symbol, u_symbol)
    )
    parsed = {u_symbol: u_rows, usdt_symbol: usdt_rows}
    economic = _mapping(contract["economic_hurdles"], name="economic hurdles")
    execution = Decimal(str(economic["round_trip_execution_stress_bips"]))
    annual = Decimal(
        str(economic["annual_opportunity_hurdle_bips_per_capital_leg"])
    ) * Decimal(str(economic["gross_capital_legs"]))
    roles: dict[str, object] = {}
    rejection_reasons: list[str] = []
    for name, times in role_times.items():
        differences = [
            parsed[short_symbol][epoch]["rate"] - parsed[long_symbol][epoch]["rate"]
            for epoch in times
        ]
        role = _role(
            differences, times, execution_bips=execution, annual_hurdle_bips=annual
        )
        roles[name] = role
        if Decimal(str(role["net_after_frozen_hurdles_bips"])) <= 0:
            rejection_reasons.append(f"{name}_net_nonpositive")
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
        "training_means": {
            "u": _decimal_text(u_mean),
            "usdt": _decimal_text(usdt_mean),
        },
        "roles": roles,
        "rejection_reasons": rejection_reasons,
        "public_persistence_candidate": not rejection_reasons,
    }


def run(*, output: Path, journal_path: Path) -> dict[str, object]:
    contract, contract_hash = _load_contract()
    if journal_path.exists():
        raise ValueError(f"one-shot journal already exists: {journal_path}")
    pairs = [
        _mapping(value, name="pair")
        for value in _list(
            _mapping(contract["universe"], name="universe")["pairs"], name="pairs"
        )
    ]
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
        path="/fapi/v1/exchangeInfo",
        name="exchange_info",
        params=None,
        journal_path=journal_path,
        journal=journal,
    )
    selected = _selected_contracts(exchange, pairs)
    histories: dict[str, object] = {}
    for pair in pairs:
        for key in ("u_symbol", "usdt_symbol"):
            symbol = str(pair[key])
            histories[symbol] = _get(
                session,
                path="/fapi/v1/fundingRate",
                name=f"funding_{symbol}",
                params={"symbol": symbol, "limit": 500},
                journal_path=journal_path,
                journal=journal,
            )
    if len(_list(journal["responses"], name="responses")) != 5:
        raise ValueError("frozen request count differs")
    evaluations = [_evaluate_pair(pair, histories, contract) for pair in pairs]
    candidates = [
        row for row in evaluations if row["public_persistence_candidate"] is True
    ]
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "contract": {
            "path": str(CONTRACT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "contract_sha256": contract_hash,
        },
        "implementation": {
            "path": str(IMPLEMENTATION_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(IMPLEMENTATION_PATH.read_bytes()),
        },
        "base_helper": {
            "path": str(BASE_TOOL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(BASE_TOOL_PATH.read_bytes()),
        },
        "raw_evidence": {
            "journal_path": str(journal_path.relative_to(ROOT)).replace("\\", "/"),
            "journal_sha256": _sha256(journal_path.read_bytes()),
            "response_count": 5,
        },
        "selected_contracts": selected,
        "evaluations": evaluations,
        "verdict": {
            "public_persistence_candidate_count": len(candidates),
            "book_escalation_permitted": bool(candidates),
            "accepted_edge": False,
            "profitability_claim": False,
            "credentials_used": False,
            "orders_submitted": 0,
            "reason": "public_history_gate_only_u_fx_account_cost_collateral_regimes_and_paper_fills_unresolved",
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
    result = run(output=args.output, journal_path=args.journal)
    print(f"result_sha256={result['result_sha256']}")
    print(_canonical_json(result["verdict"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
