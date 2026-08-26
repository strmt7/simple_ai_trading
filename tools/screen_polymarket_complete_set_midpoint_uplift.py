"""Run the frozen public Polymarket complete-set midpoint-uplift screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / "src"))

from simple_ai_trading.storage import write_bytes_atomic  # noqa: E402


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    body = dict(contract)
    claimed = body.pop("contract_sha256")
    actual = _sha256_bytes(_canonical_json(body).encode("ascii"))
    if claimed != actual:
        raise RuntimeError(f"contract hash mismatch: claimed={claimed} actual={actual}")
    return contract


def _as_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        decoded = json.loads(value)
        if isinstance(decoded, list):
            return decoded
    raise ValueError("expected a JSON list or encoded JSON list")


class Journal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[dict[str, object]] = []

    def add(self, record: dict[str, object]) -> None:
        self.records.append(record)
        payload = {
            "schema_version": "polymarket-complete-set-midpoint-uplift-journal-v1",
            "records": self.records,
        }
        write_bytes_atomic(self.path, (_canonical_json(payload) + "\n").encode("ascii"))


def _request(
    *,
    url: str,
    raw_path: Path,
    journal: Journal,
    method: str = "GET",
    body: bytes | None = None,
) -> bytes:
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "simple-ai-trading-public-research/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
        payload = response.read()
        status = response.status
    write_bytes_atomic(raw_path, payload)
    journal.add(
        {
            "method": method,
            "raw_path": _repo_relative(raw_path),
            "response_bytes": len(payload),
            "response_sha256": _sha256_bytes(payload),
            "status_code": status,
            "url": url,
        }
    )
    if status != 200:
        raise RuntimeError(f"unexpected HTTP status {status} for {url}")
    return payload


def _eligible_market(market: dict[str, Any]) -> bool:
    try:
        outcomes = _as_list(market["outcomes"])
        token_ids = _as_list(market["clobTokenIds"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
        and market.get("enableOrderBook") is True
        and market.get("holdingRewardsEnabled") is True
        and len(outcomes) == 2
        and len(token_ids) == 2
    )


def _market_row(
    *,
    event_slug: str,
    market: dict[str, Any],
    midpoints: dict[str, object],
    threshold: Decimal,
) -> dict[str, object]:
    outcomes = [str(value) for value in _as_list(market["outcomes"])]
    token_ids = [str(value) for value in _as_list(market["clobTokenIds"])]
    midpoint_rows: list[dict[str, str]] = []
    total = Decimal(0)
    complete = True
    for outcome, token_id in zip(outcomes, token_ids, strict=True):
        raw = midpoints.get(token_id)
        try:
            midpoint = Decimal(str(raw))
        except (InvalidOperation, TypeError):
            complete = False
            midpoint_rows.append(
                {"outcome": outcome, "token_id": token_id, "midpoint": "missing"}
            )
            continue
        total += midpoint
        midpoint_rows.append(
            {"outcome": outcome, "token_id": token_id, "midpoint": str(midpoint)}
        )
    return {
        "event_slug": event_slug,
        "market_id": str(market.get("id", "")),
        "condition_id": str(market.get("conditionId", "")),
        "question": str(market.get("question", "")),
        "midpoints": midpoint_rows,
        "midpoint_sum": str(total) if complete else "missing",
        "public_uplift_candidate": complete and total >= threshold,
    }


def _write_result(path: Path, result: dict[str, object]) -> None:
    result["result_sha256"] = _sha256_bytes(_canonical_json(result).encode("ascii"))
    write_bytes_atomic(path, (_canonical_json(result) + "\n").encode("ascii"))


def run(
    *, contract_path: Path, output: Path, raw_dir: Path, journal_path: Path
) -> None:
    if output.exists() or journal_path.exists() or raw_dir.exists():
        raise RuntimeError(
            "output journal or raw directory already exists; rerun prohibited"
        )
    contract = _load_contract(contract_path)
    raw_dir.mkdir(parents=True, exist_ok=False)
    journal = Journal(journal_path)
    sources: list[dict[str, object]] = []
    admitted: list[tuple[str, dict[str, Any]]] = []
    try:
        for index, slug in enumerate(contract["universe"]["event_slugs"]):
            url = f"https://gamma-api.polymarket.com/events/slug/{slug}"
            raw_path = raw_dir / f"event-{index:02d}.raw"
            payload = _request(url=url, raw_path=raw_path, journal=journal)
            event = json.loads(payload)
            if not isinstance(event, dict):
                raise ValueError(f"Gamma event {slug} was not an object")
            markets = event.get("markets")
            if not isinstance(markets, list):
                raise ValueError(f"Gamma event {slug} did not contain a market list")
            eligible = [
                row
                for row in markets
                if isinstance(row, dict) and _eligible_market(row)
            ]
            admitted.extend((str(slug), row) for row in eligible)
            sources.append(
                {
                    "event_slug": slug,
                    "event_id": str(event.get("id", "")),
                    "market_count": len(markets),
                    "eligible_market_count": len(eligible),
                    "raw_path": _repo_relative(raw_path),
                    "response_sha256": journal.records[-1]["response_sha256"],
                    "url": url,
                }
            )

        token_ids = sorted(
            {
                str(token_id)
                for _, market in admitted
                for token_id in _as_list(market["clobTokenIds"])
            }
        )
        midpoint_body = _canonical_json(
            [{"token_id": token_id} for token_id in token_ids]
        ).encode("ascii")
        midpoint_url = "https://clob.polymarket.com/midpoints"
        midpoint_raw_path = raw_dir / "midpoints.raw"
        midpoint_payload = _request(
            url=midpoint_url,
            raw_path=midpoint_raw_path,
            journal=journal,
            method="POST",
            body=midpoint_body,
        )
        midpoints = json.loads(midpoint_payload)
        if not isinstance(midpoints, dict):
            raise ValueError("CLOB midpoint response was not an object")

        threshold = Decimal(
            contract["economic_identity"]["minimum_midpoint_sum_for_escalation"]
        )
        markets = sorted(
            (
                _market_row(
                    event_slug=slug,
                    market=market,
                    midpoints=midpoints,
                    threshold=threshold,
                )
                for slug, market in admitted
            ),
            key=lambda row: (
                Decimal(str(row["midpoint_sum"]))
                if row["midpoint_sum"] != "missing"
                else Decimal("-1")
            ),
            reverse=True,
        )
        complete = [row for row in markets if row["midpoint_sum"] != "missing"]
        candidates = [row for row in complete if row["public_uplift_candidate"]]
        result: dict[str, object] = {
            "schema_version": "polymarket-complete-set-midpoint-uplift-v1",
            "contract": {
                "path": _repo_relative(contract_path),
                "contract_sha256": contract["contract_sha256"],
            },
            "implementation": {
                "path": _repo_relative(TOOL_PATH),
                "sha256": _sha256_bytes(TOOL_PATH.read_bytes()),
            },
            "authority": {
                "credentials_used": False,
                "orders_or_transactions_submitted": 0,
                "accepted_edge": False,
                "profitability_claim": False,
            },
            "sources": sources,
            "midpoint_source": {
                "url": midpoint_url,
                "raw_path": _repo_relative(midpoint_raw_path),
                "response_sha256": journal.records[-1]["response_sha256"],
                "requested_token_count": len(token_ids),
                "returned_token_count": len(midpoints),
            },
            "markets": markets,
            "summary": {
                "eligible_market_count": len(admitted),
                "complete_midpoint_market_count": len(complete),
                "public_uplift_candidate_count": len(candidates),
                "maximum_midpoint_sum": complete[0]["midpoint_sum"]
                if complete
                else "missing",
                "minimum_escalation_midpoint_sum": str(threshold),
                "history_or_book_escalation_permitted": bool(candidates),
            },
            "raw_evidence": {
                "journal_path": _repo_relative(journal_path),
                "journal_sha256": _sha256_bytes(journal_path.read_bytes()),
                "response_count": len(journal.records),
            },
            "verdict": {
                "accepted_edge": False,
                "public_uplift_candidate": bool(candidates),
                "reason": (
                    "at_least_one_current_complete_set_midpoint_sum_cleared_the_frozen_economic_threshold"
                    if candidates
                    else "no_current_complete_set_midpoint_sum_cleared_the_frozen_economic_threshold"
                ),
            },
        }
        _write_result(output, result)
    except Exception as exc:
        failure: dict[str, object] = {
            "schema_version": "polymarket-complete-set-midpoint-uplift-v1",
            "contract": {
                "path": _repo_relative(contract_path),
                "contract_sha256": contract["contract_sha256"],
            },
            "implementation": {
                "path": _repo_relative(TOOL_PATH),
                "sha256": _sha256_bytes(TOOL_PATH.read_bytes()),
            },
            "authority": {
                "credentials_used": False,
                "orders_or_transactions_submitted": 0,
                "accepted_edge": False,
                "profitability_claim": False,
            },
            "terminal_failure": {"type": type(exc).__name__, "message": str(exc)},
            "raw_evidence": {
                "journal_path": _repo_relative(journal_path),
                "journal_sha256": (
                    _sha256_bytes(journal_path.read_bytes())
                    if journal_path.exists()
                    else None
                ),
                "response_count": len(journal.records),
            },
        }
        _write_result(output, failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--journal", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.contract, args.output, args.raw_dir, args.journal):
        _repo_relative(path)
    run(
        contract_path=args.contract,
        output=args.output,
        raw_dir=args.raw_dir,
        journal_path=args.journal,
    )


if __name__ == "__main__":
    main()
