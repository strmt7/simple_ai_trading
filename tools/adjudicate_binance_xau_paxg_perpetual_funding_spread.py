"""Adjudicate retained XAU/PAXG funding rows after timestamp-jitter failure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from simple_ai_trading.storage import write_bytes_atomic
from tools.screen_binance_xau_paxg_perpetual_funding_spread import (
    ANNUAL_OPPORTUNITY_COST,
    CAPITAL_LEGS,
    EXECUTION_STRESS,
    MILLISECONDS_PER_YEAR,
    _canonical_hash,
    _canonical_json,
    _decimal,
    _list,
    _mapping,
)


SCHEMA_VERSION = "binance-xau-paxg-perpetual-funding-spread-failure-adjudication-v1"
MAXIMUM_PAIR_TIMESTAMP_SKEW_MS = 1_000
EXPECTED_GAPS_MS = (4 * 60 * 60 * 1_000, 8 * 60 * 60 * 1_000)
GAP_TOLERANCE_MS = 1_000
DIRECTIONS = ("long_XAU_short_PAXG", "long_PAXG_short_XAU")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_hash_bound(path: Path, *, field: str) -> dict[str, object]:
    payload = _mapping(json.loads(path.read_bytes()), name=path.name)
    if _canonical_hash(payload, field=field) != payload.get(field):
        raise ValueError(f"{path} canonical hash differs")
    return payload


def _rows(path: Path, *, symbol: str) -> list[dict[str, Decimal | int | str]]:
    result = []
    for value in _list(json.loads(path.read_bytes()), name=f"{symbol} raw rows"):
        row = _mapping(value, name=f"{symbol} row")
        if row.get("symbol") != symbol or row.get("rateType") != "Regular":
            raise ValueError(f"{symbol} retained row identity differs")
        result.append(
            {
                "funding_time_ms": int(row["fundingTime"]),
                "funding_rate": _decimal(
                    row.get("fundingRate"), name=f"{symbol} funding rate"
                ),
                "mark_price": _decimal(
                    row.get("markPrice"), name=f"{symbol} mark price"
                ),
                "rate_type": str(row["rateType"]),
            }
        )
    result.sort(key=lambda row: int(row["funding_time_ms"]))
    if len({int(row["funding_time_ms"]) for row in result}) != len(result):
        raise ValueError(f"{symbol} retained funding times duplicate")
    return result


def _gap_audit(rows: list[dict[str, Decimal | int | str]]) -> dict[str, object]:
    gaps = [
        int(right["funding_time_ms"]) - int(left["funding_time_ms"])
        for left, right in zip(rows, rows[1:])
    ]
    invalid = [
        gap
        for gap in gaps
        if min(abs(gap - expected) for expected in EXPECTED_GAPS_MS) > GAP_TOLERANCE_MS
    ]
    return {
        "minimum_gap_ms": min(gaps),
        "maximum_gap_ms": max(gaps),
        "four_hour_equivalent_count": sum(
            abs(gap - EXPECTED_GAPS_MS[0]) <= GAP_TOLERANCE_MS for gap in gaps
        ),
        "eight_hour_equivalent_count": sum(
            abs(gap - EXPECTED_GAPS_MS[1]) <= GAP_TOLERANCE_MS for gap in gaps
        ),
        "invalid_gap_count": len(invalid),
        "invalid_gaps_ms": invalid,
    }


def _metrics(
    *,
    pairs: list[tuple[dict[str, Decimal | int | str], dict[str, Decimal | int | str]]],
    direction: str,
) -> dict[str, object]:
    xau_start, paxg_start = pairs[0]
    xau_end, paxg_end = pairs[-1]
    xau_start_price = Decimal(xau_start["mark_price"])
    paxg_start_price = Decimal(paxg_start["mark_price"])
    xau_quantity = Decimal(1) / xau_start_price
    paxg_quantity = Decimal(1) / paxg_start_price
    sign = Decimal(1) if direction == DIRECTIONS[0] else Decimal(-1)
    relative_mark_pnl = sign * (
        (Decimal(xau_end["mark_price"]) - xau_start_price) * xau_quantity
        - (Decimal(paxg_end["mark_price"]) - paxg_start_price) * paxg_quantity
    )
    funding_pnl = Decimal(0)
    for xau, paxg in pairs[1:]:
        base = (
            -Decimal(xau["funding_rate"]) * Decimal(xau["mark_price"]) * xau_quantity
            + Decimal(paxg["funding_rate"])
            * Decimal(paxg["mark_price"])
            * paxg_quantity
        )
        funding_pnl += sign * base
    start_ms = min(
        int(xau_start["funding_time_ms"]), int(paxg_start["funding_time_ms"])
    )
    end_ms = max(int(xau_end["funding_time_ms"]), int(paxg_end["funding_time_ms"]))
    duration_ms = end_ms - start_ms
    opportunity = (
        ANNUAL_OPPORTUNITY_COST
        * CAPITAL_LEGS
        * Decimal(duration_ms)
        / MILLISECONDS_PER_YEAR
    )
    net = funding_pnl + relative_mark_pnl - EXECUTION_STRESS - opportunity
    return {
        "direction": direction,
        "paired_settlement_count": len(pairs),
        "start_time_ms": start_ms,
        "end_time_ms": end_ms,
        "duration_days": str(Decimal(duration_ms) / Decimal(86_400_000)),
        "gross_funding_spread_bips": str(funding_pnl * Decimal(10_000)),
        "relative_mark_basis_PnL_bips": str(relative_mark_pnl * Decimal(10_000)),
        "round_trip_execution_stress_bips": str(EXECUTION_STRESS * Decimal(10_000)),
        "two_leg_opportunity_cost_bips": str(opportunity * Decimal(10_000)),
        "net_after_frozen_hurdles_bips": str(net * Decimal(10_000)),
        "passes": net > 0,
    }


def run(
    *,
    contract_path: Path,
    result_path: Path,
    xau_path: Path,
    paxg_path: Path,
) -> dict[str, object]:
    contract = _load_hash_bound(contract_path, field="contract_sha256")
    original = _load_hash_bound(result_path, field="result_sha256")
    if original["contract"]["sha256"] != contract["contract_sha256"]:
        raise ValueError("original result does not bind the supplied contract")
    xau = _rows(xau_path, symbol="XAUUSDT")
    paxg = _rows(paxg_path, symbol="PAXGUSDT")
    if len(xau) != 500 or len(paxg) != 500:
        raise ValueError("adjudication is bound to the retained 500-row responses")
    pairs = list(zip(xau, paxg, strict=True))
    skews = [
        abs(int(left["funding_time_ms"]) - int(right["funding_time_ms"]))
        for left, right in pairs
    ]
    if max(skews) > MAXIMUM_PAIR_TIMESTAMP_SKEW_MS:
        raise ValueError("ordinal settlement pairs exceed the labeled sensitivity")
    xau_gaps = _gap_audit(xau)
    paxg_gaps = _gap_audit(paxg)
    if xau_gaps["invalid_gap_count"] or paxg_gaps["invalid_gap_count"]:
        raise ValueError("retained individual funding histories contain a gap")
    interval_count = len(pairs) - 1
    training_end = int(interval_count * 0.60)
    validation_end = int(interval_count * 0.80)
    role_pairs = {
        "training": pairs[: training_end + 1],
        "validation": pairs[training_end : validation_end + 1],
        "test": pairs[validation_end:],
    }
    by_role_and_direction = {
        role: {
            direction: _metrics(pairs=rows, direction=direction)
            for direction in DIRECTIONS
        }
        for role, rows in role_pairs.items()
    }
    selected = max(
        DIRECTIONS,
        key=lambda direction: Decimal(
            str(
                by_role_and_direction["training"][direction][
                    "net_after_frozen_hurdles_bips"
                ]
            )
        ),
    )
    selected_roles = {
        role: values[selected] for role, values in by_role_and_direction.items()
    }
    all_directions_fail_every_role = all(
        not bool(metrics["passes"])
        for values in by_role_and_direction.values()
        for metrics in values.values()
    )
    adjudication: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "original_contract": {
            "path": contract_path.as_posix(),
            "sha256": contract["contract_sha256"],
            "consumed_and_immutable": True,
        },
        "original_result": {
            "path": result_path.as_posix(),
            "sha256": original["result_sha256"],
            "preserved": True,
            "status": original["adjudication"]["status"],
        },
        "retained_sources": {
            "XAUUSDT": {
                "path": xau_path.as_posix(),
                "sha256": _sha256(xau_path.read_bytes()),
                "row_count": len(xau),
                "gap_audit": xau_gaps,
            },
            "PAXGUSDT": {
                "path": paxg_path.as_posix(),
                "sha256": _sha256(paxg_path.read_bytes()),
                "row_count": len(paxg),
                "gap_audit": paxg_gaps,
            },
        },
        "failure_diagnosis": {
            "exact_millisecond_intersection_count": original["capture"][
                "exact_common_regular_row_count"
            ],
            "ordinal_pair_count": len(pairs),
            "maximum_ordinal_timestamp_skew_ms": max(skews),
            "ordinal_pair_tolerance_ms": MAXIMUM_PAIR_TIMESTAMP_SKEW_MS,
            "diagnosis": (
                "both official histories contain the same 500 consecutive four-or-eight-hour "
                "settlement slots but fundingTime differs by up to 13 ms; exact timestamp "
                "intersection discarded 417 valid corresponding cash-flow rows"
            ),
            "outcome_aware_sensitivity_can_support_promotion": False,
        },
        "retained_evidence_sensitivity": {
            "directions": by_role_and_direction,
            "selected_direction_from_training_only": selected,
            "selected_direction_roles": selected_roles,
            "all_directions_fail_every_role": all_directions_fail_every_role,
        },
        "adjudication": {
            "status": "rejected_under_retained_complete_500_slot_tail",
            "accepted_edge": False,
            "profitability_claim": False,
            "candidate_for_prospective_study": False,
            "deployment_ready": False,
            "trading_authority": False,
            "rerun_or_pagination_justified": False,
            "retry_trigger": "material_funding_index_fee_or_product_architecture_change",
        },
        "implementation": {
            "path": "tools/adjudicate_binance_xau_paxg_perpetual_funding_spread.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    if not all_directions_fail_every_role:
        raise ValueError("retained sensitivity does not support terminal rejection")
    adjudication["result_sha256"] = _canonical_hash(adjudication, field="result_sha256")
    return adjudication


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--xau", type=Path, required=True)
    parser.add_argument("--paxg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite adjudication: {args.output}")
    adjudication = run(
        contract_path=args.contract,
        result_path=args.result,
        xau_path=args.xau,
        paxg_path=args.paxg,
    )
    write_bytes_atomic(
        args.output, (_canonical_json(adjudication) + "\n").encode("ascii")
    )
    print(json.dumps(adjudication["failure_diagnosis"], indent=2))
    print(json.dumps(adjudication["retained_evidence_sensitivity"], indent=2))
    print(json.dumps(adjudication["adjudication"], indent=2))
    print(f"result_sha256={adjudication['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
