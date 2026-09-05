"""Forward-only bounded transport for an existing frozen public-source contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from tools.capture_public_source_contract import (
    _canonical_hash,
    _inspect_utf8_source,
    _load_object,
    _root_path,
    _validate_contract,
)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _record(journal, payload: dict) -> None:
    journal.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")
    journal.flush()
    os.fsync(journal.fileno())


def capture(contract_path: Path, *, preflight: bool = False) -> dict | None:
    """Retain one bounded response, including HTTP failures, without redirects/retries."""
    contract_path = contract_path.resolve()
    plan = _load_object(contract_path)
    _validate_contract(plan, contract_path)
    if plan.get("transport") != {
        "socket_timeout_seconds": 10,
        "read_budget_seconds": 30,
        "redirects": False,
        "retries": 0,
        "proxies": False,
    }:
        raise ValueError("exact bounded transport configuration required")
    paths = {key: _root_path(value) for key, value in plan["outputs"].items()}
    if len(set(paths.values())) != 3 or any(path.exists() for path in paths.values()):
        raise FileExistsError("one-use outputs exist or are not distinct")
    for path in paths.values():
        if not path.parent.is_dir() or not os.access(path.parent, os.W_OK):
            raise ValueError("output parents must exist and be writable before access")
    if preflight:
        return None
    ceiling = plan["response_byte_ceiling"]
    status, error = None, None
    started_ms = time.time_ns() // 1_000_000
    with paths["journal_path"].open("x", encoding="ascii", newline="\n") as journal:
        _record(
            journal,
            {
                "phase": "intent",
                "requested_at_ms": started_ms,
                "request": plan["request"],
                "contract_sha256": plan["contract_sha256"],
            },
        )
        with paths["raw_path"].open("xb") as output:
            try:
                opener = build_opener(ProxyHandler({}), NoRedirect())
                request = Request(
                    plan["request"]["url"],
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "identity",
                        "User-Agent": "simple-ai-trading-public-research/1",
                    },
                )
                try:
                    response = opener.open(request, timeout=10)
                except HTTPError as failure:
                    response = failure
                with response:
                    status = response.code
                    deadline = time.monotonic() + 30
                    count = 0
                    while count <= ceiling:
                        if time.monotonic() > deadline:
                            raise TimeoutError("read budget exceeded")
                        chunk = response.read(min(65536, ceiling + 1 - count))
                        if not chunk:
                            break
                        output.write(chunk)
                        count += len(chunk)
            except Exception as failure:
                # Public transport failures are terminal evidence, never retried.
                error = type(failure).__name__
            finally:
                output.flush()
                os.fsync(output.fileno())
        raw = paths["raw_path"].read_bytes()
        receipt = {
            "phase": "completed",
            "requested_at_ms": started_ms,
            "completed_at_ms": time.time_ns() // 1_000_000,
            "status_code": status,
            "error_type": error,
            "response_bytes": len(raw),
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_path": plan["outputs"]["raw_path"],
            "oversize_body_is_truncated": len(raw) > ceiling,
        }
        _record(journal, receipt)
    phrases, representation = _inspect_utf8_source(raw, plan["required_utf8_phrases"])
    passed = (
        status == 200
        and error is None
        and len(raw) <= ceiling
        and all(phrases.values())
    )
    result = {
        "schema_version": "bounded-public-source-result-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {"path": plan["contract_path"], "sha256": plan["contract_sha256"]},
        "capture": {
            "receipt": receipt,
            "journal_path": plan["outputs"]["journal_path"],
        },
        "source_gate": {
            "passed": passed,
            "phrases": phrases,
            "representation": representation,
        },
        "authority": plan["authority"],
        "accepted_edge": False,
        "profitability_claim": False,
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    with paths["result_path"].open("x", encoding="ascii", newline="\n") as output:
        output.write(json.dumps(result, sort_keys=True, ensure_ascii=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = capture(args.contract, preflight=args.preflight)
    print(
        json.dumps(
            {
                "preflight": args.preflight,
                "source_gate_passed": None
                if result is None
                else result["source_gate"]["passed"],
            }
        )
    )


if __name__ == "__main__":
    main()
