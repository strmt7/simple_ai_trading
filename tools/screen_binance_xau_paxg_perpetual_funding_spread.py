"""Freeze and run one XAU/PAXG perpetual funding-spread preflight."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


TEMPLATE_SCHEMA = "binance-xau-paxg-perpetual-funding-spread-template-v1"
CONTRACT_SCHEMA = "binance-xau-paxg-perpetual-funding-spread-contract-v1"
RESULT_SCHEMA = "binance-xau-paxg-perpetual-funding-spread-result-v1"
SYMBOLS = ("XAUUSDT", "PAXGUSDT")
CUTOFF_MS = 1_786_665_600_000
END_MS = CUTOFF_MS - 1
EXPECTED_INTERVAL_MS = 8 * 60 * 60 * 1_000
EXECUTION_STRESS = Decimal("0.004")
ANNUAL_OPPORTUNITY_COST = Decimal("0.10")
CAPITAL_LEGS = Decimal(2)
MILLISECONDS_PER_YEAR = Decimal(365 * 24 * 60 * 60 * 1_000)


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


def _canonical_hash(payload: Mapping[str, object], *, field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(_canonical_json(body).encode("ascii"))


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _decimal(value: object, *, name: str) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _validate_inventory(template: Mapping[str, object]) -> dict[str, object]:
    evidence = _mapping(template.get("mechanism_evidence"), name="mechanism evidence")
    path = Path(str(evidence.get("current_exchange_info_path")))
    raw = path.read_bytes()
    if _sha256(raw) != evidence.get("current_exchange_info_sha256"):
        raise ValueError("retained current exchange-info hash differs")
    payload = _mapping(json.loads(raw), name="retained current exchange info")
    rows = {
        str(row.get("symbol")): row
        for value in _list(payload.get("symbols"), name="exchange-info symbols")
        for row in (_mapping(value, name="exchange-info row"),)
        if str(row.get("symbol")) in SYMBOLS
    }
    required = [
        _mapping(value, name="required inventory row")
        for value in _list(evidence.get("required_rows"), name="required rows")
    ]
    if set(rows) != set(SYMBOLS) or len(required) != len(SYMBOLS):
        raise ValueError("retained current inventory lacks the exact pair")
    for expected in required:
        symbol = str(expected.get("symbol"))
        if symbol not in rows or any(
            rows[symbol].get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"retained inventory identity differs for {symbol}")
    return {
        "path": path.as_posix(),
        "sha256": _sha256(raw),
        "symbols": [
            {key: rows[symbol].get(key) for key in required[index]}
            for index, symbol in enumerate(SYMBOLS)
        ],
    }


def _freeze_contract(*, template_path: Path, contract_path: Path) -> dict[str, object]:
    if contract_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen contract: {contract_path}")
    template_bytes = template_path.read_bytes()
    template = _mapping(json.loads(template_bytes), name="contract template")
    if (
        template.get("schema_version") != TEMPLATE_SCHEMA
        or template.get("status") != "prefreeze_template_no_funding_access_yet"
    ):
        raise ValueError("unexpected or already-consumed contract template")
    historical = _mapping(
        template.get("historical_contract"), name="historical contract"
    )
    requests_plan = [
        _mapping(value, name="request plan")
        for value in _list(historical.get("requests"), name="request plans")
    ]
    if (
        [row.get("symbol") for row in requests_plan] != list(SYMBOLS)
        or any(
            row.get("endTime") != str(END_MS) or row.get("limit") != "1000"
            for row in requests_plan
        )
        or historical.get("maximum_requests") != 2
        or historical.get("maximum_cumulative_request_weight") != 2
        or historical.get("retry_permitted") is not False
    ):
        raise ValueError("template differs from the exact two-request contract")
    inventory = _validate_inventory(template)
    contract = dict(template)
    contract["schema_version"] = CONTRACT_SCHEMA
    contract["status"] = "frozen_before_two_public_funding_history_requests"
    contract["frozen_at_utc"] = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    contract["validated_current_inventory"] = inventory
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
        raise ValueError("persisted frozen contract hash differs")
    frozen = datetime.fromisoformat(
        str(retained["frozen_at_utc"]).replace("Z", "+00:00")
    )
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("persisted frozen contract timestamp is invalid or future")
    return retained


def _retain_response(
    *,
    symbol: str,
    endpoint: str,
    raw_dir: Path,
) -> tuple[bytes, dict[str, object]]:
    started_ms = time.time_ns() // 1_000_000
    response = requests.get(
        endpoint,
        params={"symbol": symbol, "endTime": END_MS, "limit": 1000},
        headers={
            "Accept": "application/json",
            "User-Agent": "simple-ai-trading-public-gold-funding-research/1.0",
        },
        timeout=30,
    )
    completed_ms = time.time_ns() // 1_000_000
    raw = response.content
    raw_path = raw_dir / f"funding-{symbol.lower()}.json"
    write_bytes_atomic(raw_path, raw)
    receipt = {
        "name": f"funding-{symbol.lower()}",
        "transport": "HTTPS",
        "method": "GET",
        "url": response.url,
        "status_code": response.status_code,
        "requested_at_ms": started_ms,
        "completed_at_ms": completed_ms,
        "response_bytes": len(raw),
        "response_sha256": _sha256(raw),
        "raw_path": raw_path.as_posix(),
        "request_weight": 1,
    }
    return raw, receipt


def _capture(
    *,
    contract: Mapping[str, object],
    raw_dir: Path,
    journal_path: Path,
) -> dict[str, list[dict[str, object]]]:
    historical = _mapping(
        contract.get("historical_contract"), name="historical contract"
    )
    endpoint = str(historical["endpoint"])
    minimum_gap_ms = int(historical["minimum_milliseconds_between_request_starts"])
    rows_by_symbol: dict[str, list[dict[str, object]]] = {}
    previous_started_ms: int | None = None
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("xb") as stream:
        for symbol in SYMBOLS:
            if previous_started_ms is not None:
                elapsed = time.time_ns() // 1_000_000 - previous_started_ms
                if elapsed < minimum_gap_ms:
                    time.sleep((minimum_gap_ms - elapsed) / 1_000)
            raw, receipt = _retain_response(
                symbol=symbol,
                endpoint=endpoint,
                raw_dir=raw_dir,
            )
            previous_started_ms = int(receipt["requested_at_ms"])
            stream.write((_canonical_json(receipt) + "\n").encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
            if receipt["status_code"] != 200:
                raise ValueError(f"{symbol} funding request returned non-200")
            payload = json.loads(raw)
            rows_by_symbol[symbol] = [
                _mapping(value, name=f"{symbol} funding row")
                for value in _list(payload, name=f"{symbol} funding response")
            ]
    return rows_by_symbol


def _validated_rows(
    raw_rows: list[dict[str, object]], *, symbol: str
) -> dict[int, dict[str, Decimal | int | str]]:
    result: dict[int, dict[str, Decimal | int | str]] = {}
    for row in raw_rows:
        if row.get("symbol") != symbol:
            raise ValueError(f"{symbol} response contains another symbol")
        timestamp = int(row["fundingTime"])
        if timestamp >= CUTOFF_MS or timestamp in result:
            raise ValueError(f"{symbol} funding time is post-cutoff or duplicated")
        rate_type = row.get("rateType")
        if rate_type is None:
            raise ValueError(f"{symbol} funding row lacks current rateType identity")
        if rate_type != "Regular":
            continue
        funding_rate = _decimal(row.get("fundingRate"), name=f"{symbol} funding rate")
        mark_price = _decimal(row.get("markPrice"), name=f"{symbol} mark price")
        if mark_price <= 0:
            raise ValueError(f"{symbol} mark price must be positive")
        result[timestamp] = {
            "funding_time_ms": timestamp,
            "funding_rate": funding_rate,
            "mark_price": mark_price,
            "rate_type": str(rate_type),
        }
    return result


def _role_metrics(
    *,
    timestamps: list[int],
    xau: Mapping[int, Mapping[str, Decimal | int | str]],
    paxg: Mapping[int, Mapping[str, Decimal | int | str]],
    direction: str,
) -> dict[str, object]:
    if len(timestamps) < 2:
        raise ValueError("role requires at least two aligned funding marks")
    start = timestamps[0]
    end = timestamps[-1]
    xau_start = Decimal(xau[start]["mark_price"])
    paxg_start = Decimal(paxg[start]["mark_price"])
    xau_quantity = Decimal(1) / xau_start
    paxg_quantity = Decimal(1) / paxg_start
    sign = Decimal(1) if direction == "long_XAU_short_PAXG" else Decimal(-1)
    relative_mark_pnl = sign * (
        (Decimal(xau[end]["mark_price"]) - xau_start) * xau_quantity
        - (Decimal(paxg[end]["mark_price"]) - paxg_start) * paxg_quantity
    )
    funding_pnl = Decimal(0)
    for timestamp in timestamps[1:]:
        xau_notional = Decimal(xau[timestamp]["mark_price"]) * xau_quantity
        paxg_notional = Decimal(paxg[timestamp]["mark_price"]) * paxg_quantity
        base = (
            -Decimal(xau[timestamp]["funding_rate"]) * xau_notional
            + Decimal(paxg[timestamp]["funding_rate"]) * paxg_notional
        )
        funding_pnl += sign * base
    duration_ms = end - start
    opportunity = (
        ANNUAL_OPPORTUNITY_COST
        * CAPITAL_LEGS
        * Decimal(duration_ms)
        / MILLISECONDS_PER_YEAR
    )
    net = funding_pnl + relative_mark_pnl - EXECUTION_STRESS - opportunity
    return {
        "direction": direction,
        "start_funding_time_ms": start,
        "end_funding_time_ms": end,
        "aligned_mark_count": len(timestamps),
        "duration_days": str(Decimal(duration_ms) / Decimal(86_400_000)),
        "gross_funding_spread_bips": str(funding_pnl * Decimal(10_000)),
        "relative_mark_basis_PnL_bips": str(relative_mark_pnl * Decimal(10_000)),
        "round_trip_execution_stress_bips": str(EXECUTION_STRESS * Decimal(10_000)),
        "two_leg_opportunity_cost_bips": str(opportunity * Decimal(10_000)),
        "net_after_frozen_hurdles_bips": str(net * Decimal(10_000)),
        "passes": net > 0,
    }


def _evaluate(
    *,
    contract_path: Path,
    contract: Mapping[str, object],
    rows_by_symbol: Mapping[str, list[dict[str, object]]],
    journal_path: Path,
) -> dict[str, object]:
    xau = _validated_rows(rows_by_symbol["XAUUSDT"], symbol="XAUUSDT")
    paxg = _validated_rows(rows_by_symbol["PAXGUSDT"], symbol="PAXGUSDT")
    timestamps = sorted(set(xau) & set(paxg))
    if len(timestamps) < 10:
        raise ValueError("insufficient exact common funding history")
    unexpected_gaps = [
        {"left_ms": left, "right_ms": right, "gap_ms": right - left}
        for left, right in zip(timestamps, timestamps[1:])
        if right - left != EXPECTED_INTERVAL_MS
    ]
    interval_count = len(timestamps) - 1
    training_end = int(interval_count * 0.60)
    validation_end = int(interval_count * 0.80)
    if (
        training_end < 1
        or validation_end <= training_end
        or validation_end >= interval_count
    ):
        raise ValueError("causal split cannot allocate all three roles")
    role_times = {
        "training": timestamps[: training_end + 1],
        "validation": timestamps[training_end : validation_end + 1],
        "test": timestamps[validation_end:],
    }
    directions = ("long_XAU_short_PAXG", "long_PAXG_short_XAU")
    training_by_direction = {
        direction: _role_metrics(
            timestamps=role_times["training"], xau=xau, paxg=paxg, direction=direction
        )
        for direction in directions
    }
    selected = max(
        directions,
        key=lambda direction: Decimal(
            str(training_by_direction[direction]["net_after_frozen_hurdles_bips"])
        ),
    )
    roles = {
        role: (
            training_by_direction[selected]
            if role == "training"
            else _role_metrics(
                timestamps=role_timestamps,
                xau=xau,
                paxg=paxg,
                direction=selected,
            )
        )
        for role, role_timestamps in role_times.items()
    }
    first_after_later_onboard = (
        timestamps[0] <= 1_765_440_300_000 + 24 * 60 * 60 * 1_000
    )
    tail_reaches_cutoff = CUTOFF_MS - timestamps[-1] <= 12 * 60 * 60 * 1_000
    complete_common_history = (
        first_after_later_onboard and tail_reaches_cutoff and not unexpected_gaps
    )
    all_roles_positive = all(bool(role["passes"]) for role in roles.values())
    candidate = complete_common_history and all_roles_positive
    result: dict[str, object] = {
        "schema_version": RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract_path.as_posix(),
            "sha256": contract["contract_sha256"],
        },
        "authority": {
            "public_unauthenticated_GET_requests": 2,
            "cumulative_request_weight": 2,
            "authenticated_requests": 0,
            "account_state_accessed": False,
            "positions_or_orders": 0,
            "paper_or_live_trading_authority": False,
        },
        "capture": {
            "journal_path": journal_path.as_posix(),
            "XAUUSDT_returned_row_count": len(rows_by_symbol["XAUUSDT"]),
            "PAXGUSDT_returned_row_count": len(rows_by_symbol["PAXGUSDT"]),
            "exact_common_regular_row_count": len(timestamps),
            "first_common_funding_time_ms": timestamps[0],
            "last_common_funding_time_ms": timestamps[-1],
            "unexpected_gap_count": len(unexpected_gaps),
            "unexpected_gaps": unexpected_gaps,
            "first_common_within_24h_of_later_onboard": first_after_later_onboard,
            "last_common_within_12h_of_cutoff": tail_reaches_cutoff,
            "complete_common_history": complete_common_history,
        },
        "analysis": {
            "training_direction_comparison": training_by_direction,
            "selected_direction_from_training_only": selected,
            "roles": roles,
            "all_roles_positive": all_roles_positive,
        },
        "adjudication": {
            "status": (
                "unaccepted_positive_historical_preflight"
                if candidate
                else "historical_preflight_rejected"
            ),
            "candidate_for_prospective_study": candidate,
            "accepted_edge": False,
            "profitability_claim": False,
            "deployment_ready": False,
            "trading_authority": False,
            "retry_trigger": (
                "separately_frozen_prospective_synchronized_basis_funding_liquidity_and_cross_regime_study"
                if candidate
                else "material_funding_index_fee_or_product_architecture_change"
            ),
        },
        "limitations": [
            "XAU_and_PAXG_indices_are_related_gold_exposures_not_proven_exactly_redeemable",
            "historical_mark_prices_do_not_prove_executable_entry_exit_or_capacity",
            "exact_account_fees_margin_liquidation_and_orphan_costs_are_unbound",
            "historical_funding_and_basis_do_not_guarantee_future_recurrence",
        ],
        "implementation": {
            "path": "tools/screen_binance_xau_paxg_perpetual_funding_spread.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    result["result_sha256"] = _canonical_hash(result, field="result_sha256")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--contract-output", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(path.exists() for path in (args.contract_output, args.journal, args.output)):
        raise FileExistsError("refusing to overwrite one-use evidence")
    if args.raw_dir.exists():
        raise FileExistsError("refusing to reuse one-use raw directory")
    args.raw_dir.mkdir(parents=True)
    contract = _freeze_contract(
        template_path=args.template,
        contract_path=args.contract_output,
    )
    rows = _capture(
        contract=contract,
        raw_dir=args.raw_dir,
        journal_path=args.journal,
    )
    result = _evaluate(
        contract_path=args.contract_output,
        contract=contract,
        rows_by_symbol=rows,
        journal_path=args.journal,
    )
    write_bytes_atomic(args.output, (_canonical_json(result) + "\n").encode("ascii"))
    print(json.dumps(result["capture"], indent=2))
    print(json.dumps(result["analysis"], indent=2))
    print(json.dumps(result["adjudication"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
