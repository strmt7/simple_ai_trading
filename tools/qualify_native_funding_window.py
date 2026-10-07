"""One-use value qualification of a frozen native funding window, without cash fits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from simple_ai_trading.funding_cash import parse_binance_usdm_funding_history
from tools.capture_public_source_contract import (
    _canonical_hash,
    _root_path,
    _validate_contract,
)


def qualify_window(
    raw: bytes, *, sha256: str, symbol: str, start_ms: int, end_ms: int, limit: int
) -> dict[str, object]:
    """Validate supplied native values and bounds, not independent event completeness."""
    if (
        type(start_ms) is not int
        or type(end_ms) is not int
        or start_ms < 0
        or end_ms < start_ms
        or type(limit) is not int
        or not 1 < limit <= 1000
    ):
        raise ValueError("funding window bounds or limit are invalid")
    events = parse_binance_usdm_funding_history(
        raw, expected_sha256=sha256, expected_symbol=symbol
    )
    if not 0 < len(events) < limit:
        raise ValueError("empty or limit-saturated funding window is not admitted")
    if any(not start_ms <= event.funding_time_ms <= end_ms for event in events):
        raise ValueError("native funding event lies outside frozen window")
    return {
        "native_rows": len(events),
        "first_funding_time_ms": events[0].funding_time_ms,
        "last_funding_time_ms": events[-1].funding_time_ms,
        "native_value_gate_passed": True,
        "api_page_limit_saturated": False,
        "independent_event_population_qualified": False,
        "owned_entitlement_qualified": False,
        "cash_labels_admitted": False,
        "economic_metrics_computed": False,
    }


def adjudicate(contract_path: Path) -> dict[str, object]:
    """Consume one captured response once, retaining terminal failure without retry."""
    contract_path = contract_path.resolve()
    plan = json.loads(contract_path.read_bytes())
    _validate_contract(plan, contract_path)
    window = plan["funding_window"]
    query = parse_qs(urlsplit(plan["request"]["url"]).query)
    expected_query = {
        "symbol": [window["symbol"]],
        "startTime": [str(window["start_ms"])],
        "endTime": [str(window["end_ms"])],
        "limit": [str(window["limit"])],
    }
    parsed = urlsplit(plan["request"]["url"])
    if (
        parsed.netloc != "fapi.binance.com"
        or parsed.path != "/fapi/v1/fundingRate"
        or query != expected_query
    ):
        raise ValueError("native funding window differs from frozen API request")
    result_path = _root_path(plan["adjudication_output"])
    journal_path = _root_path(plan["adjudication_journal"])
    if result_path.exists() or journal_path.exists():
        raise FileExistsError("funding qualification is already consumed")
    receipt_result = json.loads(_root_path(plan["outputs"]["result_path"]).read_bytes())
    if not (
        _canonical_hash(receipt_result, "result_sha256")
        == receipt_result["result_sha256"]
        and receipt_result["contract"]["sha256"] == plan["contract_sha256"]
    ):
        raise ValueError("native funding capture identity differs")
    receipt = receipt_result["capture"]["receipt"]
    records = [
        json.loads(line)
        for line in _root_path(plan["outputs"]["journal_path"])
        .read_text(encoding="ascii")
        .splitlines()
    ]
    if (
        len(records) != 2
        or records[0]["phase"] != "intent"
        or records[0]["contract_sha256"] != plan["contract_sha256"]
        or records[0]["request"] != plan["request"]
        or records[1] != receipt
    ):
        raise ValueError("funding request journal differs from captured receipt")
    raw = _root_path(plan["outputs"]["raw_path"]).read_bytes()
    if (
        hashlib.sha256(raw).hexdigest() != receipt["response_sha256"]
        or len(raw) != receipt["response_bytes"]
    ):
        raise ValueError("funding captured bytes differ from receipt")
    with journal_path.open("x", encoding="ascii", newline="\n") as journal:
        journal.write(
            json.dumps(
                {"phase": "intent", "contract_sha256": plan["contract_sha256"]},
                sort_keys=True,
            )
            + "\n"
        )
        journal.flush()
        os.fsync(journal.fileno())
        qualification = None
        error = None
        try:
            if not receipt_result["source_gate"]["passed"]:
                raise ValueError("bounded native source gate failed")
            if (
                receipt["status_code"] != 200
                or receipt["error_type"] is not None
                or receipt["oversize_body_is_truncated"]
                or len(raw) > plan["response_byte_ceiling"]
            ):
                raise ValueError("native funding HTTP/byte boundary failed")
            qualification = qualify_window(
                raw,
                sha256=receipt["response_sha256"],
                symbol=window["symbol"],
                start_ms=window["start_ms"],
                end_ms=window["end_ms"],
                limit=window["limit"],
            )
        except (ValueError, UnicodeDecodeError) as failure:
            error = {"type": type(failure).__name__, "reason": str(failure)}
        result = {
            "schema_version": "native-funding-window-qualification-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "contract_sha256": plan["contract_sha256"],
            "capture_result_sha256": receipt_result["result_sha256"],
            "source_body_sha256": receipt["response_sha256"],
            "window": window,
            "passed": qualification is not None,
            "qualification": qualification,
            "error": error,
            "profitability_claim": False,
            "accepted_edge": False,
        }
        result["result_sha256"] = _canonical_hash(result, "result_sha256")
        with result_path.open("x", encoding="ascii", newline="\n") as output:
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        journal.write(
            json.dumps(
                {
                    "phase": "completed",
                    "result_sha256": result["result_sha256"],
                    "passed": result["passed"],
                },
                sort_keys=True,
            )
            + "\n"
        )
        journal.flush()
        os.fsync(journal.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    result = adjudicate(args.contract)
    print(
        json.dumps(
            {"passed": result["passed"], "qualification": result["qualification"]}
        )
    )
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
