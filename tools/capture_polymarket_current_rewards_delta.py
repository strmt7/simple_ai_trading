"""Capture one frozen public Polymarket current-rewards population delta."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs/model-research/polymarket/current-rewards-population-delta-contract-v1-2026-08-30.json"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_contract(now: datetime) -> tuple[dict[str, Any], str]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="ascii"))
    claimed = str(contract.get("result_sha256") or "")
    body = dict(contract)
    body.pop("result_sha256", None)
    if claimed != _sha256(_canonical(body)):
        raise ValueError("contract embedded hash does not reconstruct")
    frozen = datetime.fromisoformat(str(contract["frozen_at_utc"]))
    if frozen.tzinfo is None or frozen.utcoffset() is None:
        raise ValueError("frozen_at_utc must include an offset")
    frozen = frozen.astimezone(timezone.utc)
    if frozen > now:
        raise ValueError("frozen_at_utc is in the future")
    if now - frozen > timedelta(minutes=30):
        raise ValueError("frozen contract activation window expired")
    return contract, _sha256(CONTRACT_PATH.read_bytes())


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _rows_by_condition(rows: list[object]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in rows:
        row = _mapping(value, name="reward row")
        condition_id = str(row.get("condition_id") or "").lower()
        if not condition_id.startswith("0x") or len(condition_id) != 66:
            raise ValueError("reward row has invalid condition_id")
        if condition_id in result:
            raise ValueError("duplicate condition_id in reward population")
        result[condition_id] = row
    return result


def _journaled_get(
    session: requests.Session,
    *,
    endpoint: str,
    cursor: str,
    page_number: int,
    journal_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = {"next_cursor": cursor}
    url = f"{endpoint}?{urlencode(params)}"
    prefix = journal_dir / f"{page_number:02d}-current-rewards"
    intent = {
        "method": "GET",
        "url": url,
        "request_body_sha256": _sha256(b""),
        "journaled_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_bytes_atomic(prefix.with_suffix(".intent.json"), _canonical(intent) + b"\n")
    requested = datetime.now(timezone.utc)
    response = session.get(endpoint, params=params, timeout=20)
    received = datetime.now(timezone.utc)
    raw = response.content
    write_bytes_atomic(prefix.with_suffix(".raw"), raw)
    receipt = {
        "method": "GET",
        "url": response.url,
        "status_code": response.status_code,
        "response_bytes": len(raw),
        "response_sha256": _sha256(raw),
        "requested_at_utc": requested.isoformat(),
        "received_at_utc": received.isoformat(),
        "elapsed_ms": int((received - requested).total_seconds() * 1000),
    }
    write_bytes_atomic(prefix.with_suffix(".receipt.json"), _canonical(receipt) + b"\n")
    response.raise_for_status()
    payload = response.json()
    return _mapping(payload, name="current rewards response"), receipt


def run(*, output: Path, journal_dir: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    contract, contract_file_sha = _load_contract(started)
    request_contract = _mapping(contract["request_contract"], name="request contract")
    baseline_contract = _mapping(contract["baseline"], name="baseline")
    baseline_path = ROOT / str(baseline_contract["path"])
    baseline_raw = baseline_path.read_bytes()
    if _sha256(baseline_raw) != baseline_contract["response_sha256"]:
        raise ValueError("baseline response hash mismatch")
    baseline_payload = _mapping(json.loads(baseline_raw), name="baseline response")
    baseline_rows = baseline_payload.get("data")
    if not isinstance(baseline_rows, list):
        raise ValueError("baseline data must be an array")
    if len(baseline_rows) != int(baseline_contract["row_count"]):
        raise ValueError("baseline row count mismatch")
    baseline = _rows_by_condition(baseline_rows)

    journal_dir.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    cursor = str(request_contract["initial_cursor"])
    terminal_cursor = str(request_contract["terminal_cursor"])
    seen_cursors: set[str] = set()
    current_rows: list[object] = []
    receipts: list[dict[str, Any]] = []
    page_summaries: list[dict[str, Any]] = []
    complete = False
    for page_number in range(1, int(request_contract["maximum_pages"]) + 1):
        if cursor in seen_cursors:
            raise ValueError("current rewards cursor repeated")
        seen_cursors.add(cursor)
        payload, receipt = _journaled_get(
            session,
            endpoint=str(request_contract["endpoint"]),
            cursor=cursor,
            page_number=page_number,
            journal_dir=journal_dir,
        )
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise ValueError("current rewards data must be an array")
        next_cursor = str(payload.get("next_cursor") or "")
        current_rows.extend(rows)
        receipts.append(receipt)
        page_summaries.append(
            {
                "page": page_number,
                "row_count": len(rows),
                "next_cursor": next_cursor,
            }
        )
        if next_cursor == terminal_cursor:
            complete = True
            break
        if not next_cursor:
            raise ValueError("current rewards cursor missing")
        cursor = next_cursor
    if not complete:
        raise ValueError("page ceiling reached before terminal cursor")

    current = _rows_by_condition(current_rows)
    added = sorted(set(current) - set(baseline))
    removed = sorted(set(baseline) - set(current))
    changed = sorted(
        condition_id
        for condition_id in set(current) & set(baseline)
        if _canonical(current[condition_id]) != _canonical(baseline[condition_id])
    )
    artifact: dict[str, Any] = {
        "schema_version": "polymarket-current-rewards-population-delta-v1",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "row_count": len(baseline),
            "response_sha256": _sha256(baseline_raw),
        },
        "current": {
            "row_count": len(current),
            "page_summaries": page_summaries,
            "population_sha256": _sha256(_canonical(current)),
        },
        "delta": {
            "added_condition_ids": added,
            "removed_condition_ids": removed,
            "changed_condition_ids": changed,
            "added_count": len(added),
            "removed_count": len(removed),
            "changed_count": len(changed),
            "material_delta": bool(added or removed or changed),
        },
        "verdict": {
            "status": (
                "material_population_delta_requires_separate_economic_gate"
                if added or removed or changed
                else "terminal_no_population_delta"
            ),
            "accepted_edge": False,
            "profitability_claim": False,
            "market_metadata_requested": False,
            "books_requested": False,
            "publicly_proven_reward_payout_floor_pUSD": "0",
        },
        "authority": {
            "public_unauthenticated_get_only": True,
            "credentials_used": False,
            "orders_or_cancellations": 0,
            "funded_actions": 0,
        },
        "sources": {
            "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "contract_file_sha256": contract_file_sha,
            "contract_result_sha256": contract["result_sha256"],
            "requests": receipts,
            "tool_sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    artifact["result_sha256"] = _sha256(_canonical(artifact))
    write_bytes_atomic(output, _canonical(artifact) + b"\n")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(output=args.output, journal_dir=args.journal_dir)
    verdict = _mapping(result["verdict"], name="verdict")
    delta = _mapping(result["delta"], name="delta")
    print(
        json.dumps(
            {
                "status": verdict["status"],
                "added_count": delta["added_count"],
                "removed_count": delta["removed_count"],
                "changed_count": delta["changed_count"],
                "result_sha256": result["result_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
