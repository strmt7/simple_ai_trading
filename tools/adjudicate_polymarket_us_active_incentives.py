"""Adjudicate a frozen Polymarket US active-incentive inventory without network."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: dict[str, Any], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def _instant(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _decimal(value: object) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() else None


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(contract["contract_path"]):
        raise RuntimeError("contract path mismatch")
    for implementation in contract["implementations"]:
        path = _root_path(implementation["path"])
        if _sha256(path.read_bytes()) != implementation["sha256"]:
            raise RuntimeError(f"implementation hash mismatch: {path.name}")


def _capture_receipt(contract: dict[str, Any]) -> dict[str, Any]:
    journal_path = _root_path(contract["outputs"]["journal_path"])
    records = [
        json.loads(line)
        for line in journal_path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]
    if len(records) != 2 or records[0].get("phase") != "intent":
        raise RuntimeError("request journal is not one intent plus one completion")
    receipt = records[1]
    if receipt.get("phase") != "completed" or receipt.get("status_code") != 200:
        raise RuntimeError("request did not complete with HTTP 200")
    raw = _root_path(contract["outputs"]["raw_path"]).read_bytes()
    if receipt.get("response_sha256") != _sha256(raw):
        raise RuntimeError("raw response hash differs from request journal")
    if receipt.get("url") != contract["request"]["url"]:
        raise RuntimeError("journal URL differs from contract")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()

    contract_path = _root_path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(contract, contract_path)
    output_path = _root_path(contract["outputs"]["adjudication_path"])
    if output_path.exists():
        raise RuntimeError("one-use adjudication output already exists")
    receipt = _capture_receipt(contract)
    captured = datetime.fromtimestamp(
        receipt["completed_at_ms"] / 1000, tz=timezone.utc
    )
    payload = json.loads(
        _root_path(contract["outputs"]["raw_path"]).read_text(encoding="utf-8")
    )
    programs = payload.get("programs") if isinstance(payload, dict) else None
    if not isinstance(programs, list):
        raise RuntimeError("response does not contain a programs list")
    pagination_field_present = "nextPageToken" in payload
    next_page_token = payload.get("nextPageToken")
    complete = pagination_field_present and next_page_token in (None, "")

    rows: list[dict[str, Any]] = []
    malformed_period_count = 0
    observed_period_count = 0
    for program in programs:
        if not isinstance(program, dict):
            malformed_period_count += 1
            continue
        slug = program.get("marketSlug")
        periods = program.get("timePeriods")
        if not isinstance(slug, str) or not slug or not isinstance(periods, list):
            malformed_period_count += 1
            continue
        for period in periods:
            observed_period_count += 1
            if not isinstance(period, dict):
                malformed_period_count += 1
                continue
            start = _instant(period.get("start"))
            end = _instant(period.get("end"))
            reward = _decimal(period.get("rewardPool"))
            target = _decimal(period.get("targetSize"))
            discount = _decimal(period.get("discountFactor"))
            valid = bool(
                period.get("status") == "active"
                and period.get("programType") == "liquidityProgram"
                and start
                and end
                and start <= captured < end
                and reward is not None
                and reward > 0
                and target is not None
                and target > 0
                and discount is not None
                and 0 < discount <= 1
                and isinstance(period.get("programId"), str)
                and period.get("programId")
            )
            if not valid:
                continue
            assert start is not None and end is not None
            assert reward is not None and target is not None and discount is not None
            duration_hours = Decimal(str((end - start).total_seconds())) / Decimal(3600)
            if duration_hours <= 0:
                malformed_period_count += 1
                continue
            normalized = reward / duration_hours / target
            rows.append(
                {
                    "market_slug": slug,
                    "program_id": period["programId"],
                    "period": period.get("period"),
                    "start": start.isoformat().replace("+00:00", "Z"),
                    "end": end.isoformat().replace("+00:00", "Z"),
                    "duration_hours": _decimal_text(duration_hours),
                    "reward_pool_USD": _decimal_text(reward),
                    "discount_factor": _decimal_text(discount),
                    "target_size_contracts_per_side": _decimal_text(target),
                    "reward_pool_per_hour_per_target_contract_USD": _decimal_text(
                        normalized
                    ),
                }
            )
    rows.sort(
        key=lambda row: (
            -Decimal(row["reward_pool_per_hour_per_target_contract_USD"]),
            row["market_slug"],
            row["program_id"],
        )
    )
    source_gate = complete and malformed_period_count == 0 and bool(rows)
    result: dict[str, Any] = {
        "schema_version": "polymarket-us-active-incentives-adjudication-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "capture": {
            "receipt": receipt,
            "captured_at_utc": captured.isoformat().replace("+00:00", "Z"),
            "returned_market_group_count": len(programs),
            "observed_period_count": observed_period_count,
            "pagination_field_present": pagination_field_present,
            "next_page_token_empty": next_page_token in (None, ""),
            "complete_single_page": complete,
        },
        "active_liquidity_programs": {
            "qualifying_period_count": len(rows),
            "malformed_or_unadmitted_period_count": malformed_period_count,
            "deterministic_ranking": contract["ranking_rule"],
            "rows": rows,
            "top_public_book_candidate": rows[0] if source_gate else None,
        },
        "adjudication": {
            "source_gate_passed": source_gate,
            "accepted_scoped_structural_edge": source_gate,
            "market_direction_forecast_required": False,
            "deployment_ready": False,
            "stable_account_qualified_after_all_cost_edge": False,
            "profitability_claim": False,
            "public_forward_profit_floor_USD": "0",
            "next_action": (
                "freeze_one_public_book_competition_and_adverse_selection_gate_for_the_deterministic_top_candidate"
                if source_gate
                else "stop_without_pagination_book_account_credential_order_or_fund_access"
            ),
        },
        "limitations": contract["limitations"],
        "authority": contract["authority"],
        "implementation": {
            "path": "tools/adjudicate_polymarket_us_active_incentives.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "accepted_scoped_structural_edge": source_gate,
                "complete_single_page": complete,
                "qualifying_period_count": len(rows),
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
