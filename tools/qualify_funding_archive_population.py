"""One-use archive population qualification, never a funding-profit evaluator."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
import zipfile

from simple_ai_trading.derivatives_archive import derivatives_archive_file_url
from simple_ai_trading.funding_cash import parse_binance_usdm_funding_history
from tools.capture_public_source_bounded import NoRedirect, _record
from tools.capture_public_source_contract import _canonical_hash, _root_path


def archive_events(
    raw: bytes, checksum: bytes, filename: str
) -> list[tuple[int, Fraction]]:
    """Verify published digest and bounded CSV before retaining exact event units."""
    match = re.fullmatch(
        rb"([0-9a-fA-F]{64})[ \t]+\*?" + re.escape(filename.encode("ascii")) + rb"\s*",
        checksum,
    )
    if match is None or match[1].decode().lower() != hashlib.sha256(raw).hexdigest():
        raise ValueError("archive checksum is missing, malformed or mismatched")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if len(members) != 1:
            raise ValueError("archive must contain exactly its named CSV")
        member = members[0]
        if (
            member.filename != filename.removesuffix(".zip") + ".csv"
            or not 0 < member.file_size <= 1_048_576
            or member.file_size > 250 * max(1, member.compress_size)
            or member.flag_bits & 1
            or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
        ):
            raise ValueError("archive member identity or resource boundary differs")
        text = archive.read(member).decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or rows[0] != [
        "calc_time",
        "funding_interval_hours",
        "last_funding_rate",
    ]:
        raise ValueError("archive header differs")
    events: list[tuple[int, Fraction]] = []
    for row in rows[1:]:
        if (
            len(row) != 3
            or not row[0].isdigit()
            or row[1] not in map(str, range(1, 9))
            or re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", row[2])
            is None
        ):
            raise ValueError("archive event schema differs")
        event = (int(row[0]), Fraction(row[2]))
        if event[0] < 0 or abs(event[1]) > Fraction(1, 10):
            raise ValueError("archive event value is invalid")
        if events and event[0] <= events[-1][0]:
            raise ValueError("archive clocks are not strictly increasing")
        events.append(event)
    if not events or len(events) > 744:
        raise ValueError("archive event population is empty or oversized")
    return events


def capture_bytes(url: str, path: Path, journal, ceiling: int) -> bytes:
    """Capture one public response durably; every transport failure is terminal."""
    _record(
        journal,
        {"phase": "request_intent", "url": url, "at_ms": time.time_ns() // 1_000_000},
    )
    status, error = None, None
    with path.open("xb") as output:
        try:
            opener = build_opener(ProxyHandler({}), NoRedirect())
            request = Request(url, headers={"Accept-Encoding": "identity"})
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
                        raise TimeoutError("archive read budget exceeded")
                    chunk = response.read(min(65_536, ceiling + 1 - count))
                    if not chunk:
                        break
                    output.write(chunk)
                    count += len(chunk)
        except Exception as failure:
            # Preserve partial bytes and terminalize; never retry a consumed request.
            error = type(failure).__name__
        finally:
            output.flush()
            os.fsync(output.fileno())
    raw = path.read_bytes()
    _record(
        journal,
        {
            "phase": "request_completed",
            "url": url,
            "at_ms": time.time_ns() // 1_000_000,
            "status_code": status,
            "error_type": error,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    )
    if status != 200 or error is not None or not 0 < len(raw) <= ceiling:
        raise ValueError("public archive response failed its frozen transport gate")
    return raw


def run(plan_path: Path, *, preflight: bool = False) -> dict[str, object] | None:
    """Qualify only the frozen retained-native intersection; never infer owned income."""
    plan = json.loads(plan_path.read_bytes())
    month_start = datetime.strptime(plan["period"], "%Y-%m").replace(
        tzinfo=timezone.utc
    )
    next_month = month_start.replace(
        year=month_start.year + (month_start.month == 12),
        month=month_start.month % 12 + 1,
    )
    if (
        _canonical_hash(plan, "contract_sha256") != plan["contract_sha256"]
        or plan_path.resolve() != _root_path(plan["contract_path"])
        or plan["schema"] != "funding-archive-population-contract-v1"
        or plan["symbols"] != ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        or not plan["archive_start_ms"]
        <= plan["intersection_start_ms"]
        < plan["intersection_end_ms"]
        == plan["archive_end_ms"]
        or datetime.fromisoformat(plan["frozen_at_utc"].replace("Z", "+00:00"))
        > datetime.now(timezone.utc)
        or plan["max_requests"] != 6
        or plan["archive_start_ms"] != int(month_start.timestamp() * 1000)
        or plan["archive_end_ms"] != int(next_month.timestamp() * 1000)
        or plan["capture_boundaries_sha256"]
        != hashlib.sha256(
            _root_path("docs/RESEARCH_CAPTURE_BOUNDARIES.md").read_bytes()
        ).hexdigest()
    ):
        raise ValueError("frozen population contract differs")
    for source in plan["implementations"]:
        if (
            hashlib.sha256(_root_path(source["path"]).read_bytes()).hexdigest()
            != source["sha256"]
        ):
            raise ValueError("implementation binding differs")
    certificate = json.loads(
        _root_path(plan["native_certificate"]["path"]).read_bytes()
    )
    if (
        _canonical_hash(certificate, "result_sha256")
        != plan["native_certificate"]["result_sha256"]
        or certificate["result_sha256"] != plan["native_certificate"]["result_sha256"]
        or certificate["status"]
        != "all_four_native_value_gates_passed_not_cash_label_admission"
        or {(item["raw_path"], item["raw_sha256"]) for item in certificate["inputs"]}
        != {
            (item["path"], item["sha256"])
            for inputs in plan["native_inputs"].values()
            for item in inputs
        }
    ):
        raise ValueError("native origin certificate or population differs")
    native: dict[str, list[tuple[int, Fraction]]] = {}
    for symbol in plan["symbols"]:
        events = []
        for source in plan["native_inputs"][symbol]:
            raw = _root_path(source["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != source["sha256"]:
                raise ValueError("retained native binding differs")
            events.extend(
                parse_binance_usdm_funding_history(
                    raw, expected_symbol=symbol, expected_sha256=source["sha256"]
                )
            )
        native[symbol] = [
            (event.funding_time_ms, event.rate)
            for event in events
            if plan["intersection_start_ms"]
            <= event.funding_time_ms
            < plan["intersection_end_ms"]
        ]
        if len(native[symbol]) != plan["expected_intersection_events_per_symbol"]:
            raise ValueError("retained intersection count differs")
        if any(
            right[0] <= left[0]
            for left, right in zip(native[symbol], native[symbol][1:])
        ):
            raise ValueError("retained intersection clocks differ")
    base = _root_path(plan["output_directory"])
    paths = [
        base / name
        for symbol in plan["symbols"]
        for name in (symbol + ".zip.raw", symbol + ".checksum.raw")
    ]
    paths += [base / "journal.jsonl", base / "result.json"]
    if not base.is_dir() or any(path.exists() for path in paths):
        raise FileExistsError("one-use output is consumed or parent is absent")
    if preflight:
        return None
    result: dict[str, object] = {
        "schema": "funding-archive-population-result-v1",
        "contract_sha256": plan["contract_sha256"],
        "symbols": [],
        "qualified_intersection": False,
        "accepted_edge": False,
        "training_admitted": False,
        "owned_entitlement_qualified": False,
        "independent_publisher": False,
        "error_type": None,
        "failure_stage": None,
    }
    with (base / "journal.jsonl").open("x", encoding="ascii", newline="\n") as journal:
        _record(
            journal,
            {"phase": "study_intent", "contract_sha256": plan["contract_sha256"]},
        )
        try:
            for symbol in plan["symbols"]:
                url = derivatives_archive_file_url(
                    symbol=symbol,
                    data_type="fundingRate",
                    period=plan["period"],
                    interval="",
                )
                result["failure_stage"] = "archive_transport"
                raw = capture_bytes(
                    url, base / (symbol + ".zip.raw"), journal, 1_048_576
                )
                result["failure_stage"] = "checksum_transport"
                checksum = capture_bytes(
                    url + ".CHECKSUM", base / (symbol + ".checksum.raw"), journal, 4096
                )
                result["failure_stage"] = "archive_validation"
                archive = archive_events(raw, checksum, url.rsplit("/", 1)[1])
                if any(
                    not plan["archive_start_ms"] <= event[0] < plan["archive_end_ms"]
                    for event in archive
                ):
                    raise ValueError("archive event is outside its calendar month")
                selected = [
                    event
                    for event in archive
                    if event[0] >= plan["intersection_start_ms"]
                ]
                result["failure_stage"] = "exact_population_join"
                if selected != native[symbol]:
                    raise ValueError(
                        "archive/native exact timestamp and decimal rate population differs"
                    )
                result["symbols"].append(
                    {
                        "symbol": symbol,
                        "archive_rows": len(archive),
                        "intersection_rows": len(selected),
                        "exact_join": True,
                    }
                )
            result["qualified_intersection"] = True
            result["failure_stage"] = None
        except (ValueError, OSError, zipfile.BadZipFile, UnicodeError) as failure:
            result["error_type"] = type(failure).__name__
        result["result_sha256"] = _canonical_hash(result, "result_sha256")
        with (base / "result.json").open("x", encoding="ascii", newline="\n") as output:
            output.write(json.dumps(result, sort_keys=True, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        _record(
            journal,
            {
                "phase": "study_completed",
                "result_sha256": result["result_sha256"],
                "qualified_intersection": result["qualified_intersection"],
            },
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = run(args.contract, preflight=args.preflight)
    print(
        json.dumps(
            {
                "preflight": args.preflight,
                "qualified_intersection": None
                if result is None
                else result["qualified_intersection"],
            }
        )
    )


if __name__ == "__main__":
    main()
