"""Frozen public fixed-base funding-index prefilter; no trading capability."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
STEP = 8 * 3600 * 1000
ASSETS = ("BTC", "ETH", "SOL")


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def decimal(value: object, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise ValueError("numeric field must be an explicit decimal")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("invalid numeric field") from error
    if not result.is_finite() or (positive and result <= 0):
        raise ValueError("numeric field is nonfinite or not positive")
    return result


def requests_for(start: int, end: int) -> list[dict]:
    requests = []
    for asset in ASSETS:
        for index, boundary in enumerate(range(start, end + 1, STEP)):
            lower, upper = boundary + 1001, boundary + 301000
            requests.append(
                {
                    "name": f"{asset.lower()}-index-{index:02d}",
                    "kind": "index",
                    "asset": asset,
                    "boundary": boundary,
                    "start": lower,
                    "end": upper,
                    "url": f"https://api.prod.paradex.trade/v1/funding/data?market={asset}-USD-PERP&start_at={lower}&end_at={upper}&page_size=5000",
                }
            )
        requests.append(
            {
                "name": f"{asset.lower()}-binance",
                "kind": "binance",
                "asset": asset,
                "start": start,
                "end": end + 1000,
                "url": f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={asset}USDT&startTime={start}&endTime={end + 1000}&limit=1000",
            }
        )
    return requests


def index_row(payload: object, request: dict) -> dict:
    if not isinstance(payload, dict) or payload.get("next") not in (None, ""):
        raise ValueError("index envelope is incomplete or paginated")
    rows = payload.get("results")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 5000:
        raise ValueError("index window is empty or exceeds row ceiling")
    timestamps = set()
    for row in rows:
        stamp = row.get("created_at") if isinstance(row, dict) else None
        if (
            type(stamp) is not int
            or not request["start"] <= stamp <= request["end"]
            or stamp in timestamps
            or row.get("market") != request["asset"] + "-USD-PERP"
        ):
            raise ValueError("index window timestamps or market differ")
        timestamps.add(stamp)
        decimal(row.get("funding_index"))
        if decimal(row.get("funding_period_hours")) != 8:
            raise ValueError("funding period differs")
    # Selection uses time only, never index value or API row order.
    selected = min(rows, key=lambda row: row["created_at"])
    return {
        "time": selected["created_at"],
        "index": str(decimal(selected["funding_index"])),
    }


def binance_rows(payload: object, *, asset: str, start: int, end: int) -> list[dict]:
    if not isinstance(payload, list) or len(payload) != (end - start) // STEP + 1:
        raise ValueError("Binance population count differs")
    mapped = {}
    for row in payload:
        stamp = row.get("fundingTime") if isinstance(row, dict) else None
        if type(stamp) is not int or row.get("symbol") != asset + "USDT":
            raise ValueError("Binance timestamp or symbol differs")
        bucket = (stamp // STEP) * STEP
        if not 0 <= stamp - bucket <= 1000 or bucket in mapped:
            raise ValueError("Binance schedule differs")
        mapped[bucket] = {
            "time": stamp,
            "rate": str(decimal(row.get("fundingRate"))),
            "mark": str(decimal(row.get("markPrice"), positive=True)),
        }
    if set(mapped) != set(range(start, end + 1, STEP)):
        raise ValueError("Binance endpoints differ")
    return [mapped[bucket] for bucket in range(start, end + 1, STEP)]


def economic_rows(indices: list[dict], funding: list[dict]) -> list[dict]:
    if len(indices) != len(funding) or len(indices) < 2:
        raise ValueError("unequal source population")
    rows = []
    for i in range(1, len(indices)):
        left, right, event = indices[i - 1], indices[i], funding[i]
        if not left["time"] < event["time"] < right["time"]:
            raise ValueError("discrete settlement not inside actual index interval")
        rows.append(
            {
                "start": left["time"],
                "end": right["time"],
                "paradex_short_cash_usdc_per_base": str(
                    decimal(right["index"]) - decimal(left["index"])
                ),
                "binance_long_cash_usdt_per_base": str(
                    -decimal(event["rate"]) * decimal(event["mark"], positive=True)
                ),
                "reference_mark": funding[i - 1]["mark"],
            }
        )
    return rows


def role_result(rows: list[dict], *, sign: int) -> dict:
    if not rows or sign not in (-1, 1):
        raise ValueError("empty role or invalid orientation")
    p_cash = [sign * decimal(row["paradex_short_cash_usdc_per_base"]) for row in rows]
    b_cash = [sign * decimal(row["binance_long_cash_usdt_per_base"]) for row in rows]
    par_spreads = [p + b for p, b in zip(p_cash, b_cash, strict=True)]
    reference = decimal(rows[0]["reference_mark"], positive=True)
    duration_days = Decimal(rows[-1]["end"] - rows[0]["start"]) / Decimal(86400000)
    gross = sum(par_spreads) / reference * 10000
    capital = Decimal("0.10") * duration_days / 365 * 10000
    split = len(rows) // 2
    halves = [sum(par_spreads[:split]), sum(par_spreads[split:])]
    positive_fraction = Decimal(sum(value > 0 for value in par_spreads)) / len(rows)
    net = gross - Decimal(80) - capital
    return {
        "rows": len(rows),
        "actual_duration_days": str(duration_days),
        "paradex_cash_usdc_per_base": str(sum(p_cash)),
        "binance_cash_usdt_per_base": str(sum(b_cash)),
        "par_valuation_gross_bips": str(gross),
        "reference_binance_mark_not_executable_entry": str(reference),
        "execution_stress_bips": "20",
        "quote_unit_stress_bips": "25",
        "custody_latency_stress_bips": "25",
        "retained_extra_reserve_bips": "10",
        "two_leg_annual_10_percent_capital_stress_bips": str(capital),
        "net_after_frozen_hurdles_bips": str(net),
        "positive_interval_fraction": str(positive_fraction),
        "half_gross_cash_at_par": [str(value) for value in halves],
        "passes_prefilter": net > 0
        and positive_fraction >= Decimal("0.5")
        and min(halves) > 0,
    }


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def record(journal, value: dict) -> None:
    journal.write(
        json.dumps(
            {"at_utc": datetime.now(timezone.utc).isoformat(), **value}, sort_keys=True
        )
        + "\n"
    )
    journal.flush()
    os.fsync(journal.fileno())


def captured_get(request: dict, *, raw: Path, journal) -> dict:
    ceiling = 1048576
    record(
        journal,
        {
            "phase": "request_started",
            **request,
            "method": "GET",
            "byte_ceiling": ceiling,
        },
    )
    count, status, started = 0, None, time.monotonic()
    try:
        with raw.open("xb") as output:
            try:
                response = build_opener(NoRedirect()).open(
                    Request(request["url"], headers={"Accept": "application/json"}),
                    timeout=20,
                )
            except HTTPError as error:
                response = error
            try:
                status = response.status
                while count <= ceiling:
                    if time.monotonic() - started > 30:
                        raise TimeoutError("elapsed response budget exceeded")
                    chunk = response.read(min(65536, ceiling + 1 - count))
                    if not chunk:
                        break
                    output.write(chunk)
                    count += len(chunk)
            finally:
                response.close()
                output.flush()
                os.fsync(output.fileno())
        receipt = {
            "phase": "request_completed",
            "name": request["name"],
            "status": status,
            "bytes": count,
            "raw_sha256": digest(raw.read_bytes()),
        }
        record(journal, receipt)
        if status != 200 or count > ceiling:
            raise ValueError("HTTP or byte-ceiling failure")
        return receipt
    except Exception as error:
        record(
            journal,
            {
                "phase": "request_failed",
                "name": request["name"],
                "error_type": type(error).__name__,
                "status": status,
                "raw_sha256": digest(raw.read_bytes()) if raw.exists() else None,
            },
        )
        raise


def run(contract_path: Path, *, preflight: bool = False) -> dict | None:
    contract = json.loads(contract_path.read_bytes())
    expected = contract.pop("contract_sha256")
    if digest(canonical(contract)) != expected:
        raise ValueError("contract self-hash differs")
    for path, source_hash in contract["source_sha256"].items():
        if digest((ROOT / path).read_bytes()) != source_hash:
            raise ValueError("implementation or primary-source hash differs")
    start, end = contract["start_boundary_ms"], contract["end_boundary_ms"]
    prior = json.loads((ROOT / contract["prior_preregistration"]).read_bytes())
    if start <= prior["population"]["paradex_query_end_ms"]:
        raise ValueError("new population overlaps the consumed source window")
    if end + 301000 >= int(datetime.now(timezone.utc).timestamp() * 1000):
        raise ValueError("population has not fully elapsed")
    requests = requests_for(start, end)
    if requests != contract["requests"] or (end - start) // STEP != 26:
        raise ValueError("exact frozen population differs")
    output_root = ROOT / contract["output_root"]
    if output_root.exists():
        raise FileExistsError("study already has outputs; do not retry")
    if preflight:
        print(
            json.dumps(
                {
                    "preflight": True,
                    "public_requests": len(requests),
                    "intervals_per_asset": 26,
                }
            )
        )
        return None
    output_root.mkdir(parents=True)
    receipts, selected = [], {}
    result = {
        "schema_version": "paradex-index-boundary-prefilter-v1",
        "contract_sha256": expected,
        "source_sha256": contract["source_sha256"],
        "receipts": receipts,
        "accepted_edge": False,
        "profitability_claim": False,
        "trading_authority": False,
    }
    with (output_root / "requests.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as journal:
        try:
            for request in requests:
                raw = output_root / (request["name"] + ".json")
                receipt = captured_get(request, raw=raw, journal=journal)
                receipts.append(
                    {**receipt, "raw_path": raw.relative_to(ROOT).as_posix()}
                )
                payload = json.loads(raw.read_bytes(), parse_float=Decimal)
                selected[request["name"]] = (
                    index_row(payload, request)
                    if request["kind"] == "index"
                    else binance_rows(
                        payload, asset=request["asset"], start=start, end=end
                    )
                )
                if len(receipts) % 10 == 0:
                    print(
                        f"Source gates completed: {len(receipts)}/{len(requests)}",
                        flush=True,
                    )
            assets = {}
            with localcontext() as context:
                context.prec = 50
                for asset in ASSETS:
                    indices = [
                        selected[f"{asset.lower()}-index-{i:02d}"] for i in range(27)
                    ]
                    rows = economic_rows(indices, selected[f"{asset.lower()}-binance"])
                    training = sum(
                        decimal(r["paradex_short_cash_usdc_per_base"])
                        + decimal(r["binance_long_cash_usdt_per_base"])
                        for r in rows[:13]
                    )
                    sign = 1 if training >= 0 else -1
                    roles = {
                        name: role_result(rows[left:right], sign=sign)
                        for name, left, right in (
                            ("training", 0, 13),
                            ("validation", 13, 19),
                            ("test", 19, 26),
                        )
                    }
                    assets[asset] = {
                        "sign": sign,
                        "rows": rows,
                        "roles": roles,
                        "passes_prefilter": all(
                            r["passes_prefilter"] for r in roles.values()
                        ),
                    }
            result.update(
                status="complete_funding_only_prefilter",
                assets=assets,
                survivors=[a for a, v in assets.items() if v["passes_prefilter"]],
            )
            record(
                journal, {"phase": "study_complete", "survivors": result["survivors"]}
            )
        except Exception as error:
            result.update(
                status="terminal_source_or_alignment_failure",
                failure_type=type(error).__name__,
                failure_reason=str(error),
                completed_requests=len(receipts),
                economics_evaluated=False,
            )
            record(
                journal,
                {"phase": "study_terminal_failure", "error_type": type(error).__name__},
            )
    result["result_sha256"] = digest(canonical(result))
    with (output_root / "result.json").open("xb") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False).encode() + b"\n")
        output.flush()
        os.fsync(output.fileno())
    print(
        json.dumps(
            {
                "status": result["status"],
                "requests_completed": len(receipts),
                "result_sha256": result["result_sha256"],
            }
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    run(args.contract, preflight=args.preflight)
