"""One-use fixed-base cross-venue funding blocks; no account or order capability."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import json
import os
from pathlib import Path

from tools import screen_paradex_index_boundaries as native


ROOT = Path(__file__).resolve().parents[1]
WEEK = 7 * 86_400_000
ECONOMICS = {
    "execution_stress_bips": "20",
    "quote_unit_stress_bips": "25",
    "custody_latency_stress_bips": "25",
    "retained_extra_stress_bips": "10",
    "annual_total_two_leg_capital_percent": "10",
}


def requests_for(start: int, end: int) -> list[dict]:
    requests = [
        {
            "name": "inventory",
            "kind": "inventory",
            "url": "https://api.prod.paradex.trade/v1/markets",
        }
    ]
    for asset in native.ASSETS:
        for i, boundary in enumerate(range(start, end + 1, WEEK)):
            lower, upper = boundary + 1001, boundary + 301000
            requests.append(
                {
                    "name": f"{asset.lower()}-index-{i:02d}",
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
    for request in requests:
        request["request_body_sha256"] = native.digest(b"")
    return requests


def validate_inventory(payload: object, start: int) -> None:
    if not isinstance(payload, dict) or payload.get("next") not in (None, ""):
        raise ValueError("instrument inventory envelope is incomplete")
    rows = payload.get("results")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 5000:
        raise ValueError("instrument inventory count differs")
    for asset in native.ASSETS:
        matches = [
            r
            for r in rows
            if isinstance(r, dict) and r.get("symbol") == asset + "-USD-PERP"
        ]
        if len(matches) != 1:
            raise ValueError("instrument identity is absent or duplicated")
        row = matches[0]
        expected = {
            "asset_kind": "PERP",
            "base_currency": asset,
            "quote_currency": "USD",
            "settlement_currency": "USDC",
            "trading_mode": "STANDARD",
        }
        if any(row.get(k) != v for k, v in expected.items()) or (
            native.decimal(row.get("funding_period_hours")) != 8
            or type(row.get("open_at")) is not int
            or not 0 <= row["open_at"] <= start
            or type(row.get("expiry_at")) is not int
            or row["expiry_at"] != 0
        ):
            raise ValueError("instrument metadata differs from the frozen cash model")


def block_rows(indices: list[dict], funding: list[dict]) -> list[dict]:
    """Telescope continuous indices and retain every actual discrete settlement."""
    if len(indices) < 3 or len(funding) != (len(indices) - 1) * WEEK // native.STEP + 1:
        raise ValueError("source populations do not cover whole funding blocks")
    rows = []
    used = []
    for i, (left, right) in enumerate(zip(indices, indices[1:])):
        events = [r for r in funding if left["time"] < r["time"] < right["time"]]
        if len(events) != WEEK // native.STEP:
            raise ValueError(
                "actual index endpoints do not enclose every block settlement"
            )
        used.extend(r["time"] for r in events)
        rows.append(
            {
                "start": left["time"],
                "end": right["time"],
                "paradex_short_cash_usdc_per_base": str(
                    native.decimal(right["index"]) - native.decimal(left["index"])
                ),
                "binance_long_cash_usdt_per_base": str(
                    -sum(
                        (
                            native.decimal(r["rate"])
                            * native.decimal(r["mark"], positive=True)
                            for r in events
                        ),
                        Decimal(0),
                    )
                ),
                "reference_mark": funding[i * WEEK // native.STEP]["mark"],
                "binance_settlements": len(events),
            }
        )
    expected = [
        r["time"]
        for r in funding
        if indices[0]["time"] < r["time"] < indices[-1]["time"]
    ]
    if used != expected or len(set(used)) != len(used):
        raise ValueError("discrete settlements were omitted or counted twice")
    return rows


def adjudicate(rows: list[dict], roles: dict[str, list[int]]) -> dict:
    a, b = roles["training"]
    calibration = sum(
        native.decimal(r["paradex_short_cash_usdc_per_base"])
        + native.decimal(r["binance_long_cash_usdt_per_base"])
        for r in rows[a:b]
    )
    sign = 1 if calibration >= 0 else -1
    outputs = {}
    for name, (a, b) in roles.items():
        subset = rows[a:b]
        result = native.role_result(subset, sign=sign)
        ideal = sum(
            abs(
                native.decimal(r["paradex_short_cash_usdc_per_base"])
                + native.decimal(r["binance_long_cash_usdt_per_base"])
            )
            for r in subset
        )
        ideal = (
            ideal / native.decimal(subset[0]["reference_mark"], positive=True) * 10000
        )
        ideal -= native.decimal(result["two_leg_annual_10_percent_capital_stress_bips"])
        result["perfect_foresight_weekly_sign_zero_execution_net_of_capital_bips"] = (
            str(ideal)
        )
        outputs[name] = result
    survives = all(r["passes_prefilter"] for r in outputs.values())
    return {
        "orientation": sign,
        "roles": outputs,
        "passes_prefilter": survives,
        "perfect_foresight_bound_positive_in_every_role": all(
            native.decimal(
                r["perfect_foresight_weekly_sign_zero_execution_net_of_capital_bips"]
            )
            > 0
            for r in outputs.values()
        ),
    }


def validate_contract(path: Path) -> dict:
    contract = json.loads(path.read_bytes())
    expected = contract["contract_sha256"]
    if (
        native.digest(
            native.canonical(
                {k: v for k, v in contract.items() if k != "contract_sha256"}
            )
        )
        != expected
    ):
        raise ValueError("contract digest differs")
    for name, digest in contract["source_sha256"].items():
        if native.digest((ROOT / name).read_bytes()) != digest:
            raise ValueError("frozen implementation or primary source changed")
    start, end = contract["start_boundary_ms"], contract["end_boundary_ms"]
    if type(start) is not int or type(end) is not int or (end - start) != 9 * WEEK:
        raise ValueError("the frozen nine-week population differs")
    if start % WEEK != 4 * 86_400_000 or end + 301000 >= int(
        datetime.now(timezone.utc).timestamp() * 1000
    ):
        raise ValueError("Monday population is misaligned or not fully elapsed")
    prior = json.loads((ROOT / contract["prior_consumed_contract"]).read_bytes())
    if start <= prior["end_boundary_ms"] + 301000:
        raise ValueError("population overlaps consumed source windows")
    if (
        contract["requests"] != requests_for(start, end)
        or len(contract["requests"]) != 34
    ):
        raise ValueError("frozen exact requests differ")
    if (
        contract["roles"] != {"training": [0, 5], "validation": [5, 7], "test": [7, 9]}
        or contract["economics"] != ECONOMICS
    ):
        raise ValueError("frozen roles or unchanged economic hurdles differ")
    output = ROOT / contract["output_root"]
    if (
        not output.resolve().is_relative_to((ROOT / "data/research").resolve())
        or output.exists()
    ):
        raise ValueError("output must be a new owned private study directory")
    if not output.parent.is_dir():
        raise ValueError("create the verified study parent before access")
    return contract


def run(path: Path, *, preflight: bool = False) -> dict | None:
    contract = validate_contract(path)
    output = ROOT / contract["output_root"]
    if preflight:
        print(
            json.dumps(
                {"preflight": True, "requests": 34, "funding_blocks_per_asset": 9}
            )
        )
        return None
    output.mkdir()
    receipts, selected = [], {}
    result = {
        "schema": "hedged-weekly-funding-prefilter-v1",
        "contract_sha256": contract["contract_sha256"],
        "source_sha256": contract["source_sha256"],
        "receipts": receipts,
        "accepted_edge": False,
        "profitability_claim": False,
        "trading_authority": False,
    }
    with (output / "requests.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as journal:
        native.record(
            journal,
            {"phase": "study_started", "contract_sha256": contract["contract_sha256"]},
        )
        try:
            for request in contract["requests"]:
                raw = output / (request["name"] + ".json")
                with raw.with_suffix(".preflight").open("xb") as probe:
                    probe.write(b"{}\n")
                    probe.flush()
                    os.fsync(probe.fileno())
                receipt = native.captured_get(request, raw=raw, journal=journal)
                receipts.append(
                    {**receipt, "raw_path": raw.relative_to(ROOT).as_posix()}
                )
                payload = json.loads(raw.read_bytes(), parse_float=Decimal)
                if request["kind"] == "inventory":
                    validate_inventory(payload, contract["start_boundary_ms"])
                elif request["kind"] == "index":
                    selected[request["name"]] = native.index_row(payload, request)
                else:
                    selected[request["name"]] = native.binance_rows(
                        payload,
                        asset=request["asset"],
                        start=contract["start_boundary_ms"],
                        end=contract["end_boundary_ms"],
                    )
                native.record(
                    journal, {"phase": "source_gate_passed", "name": request["name"]}
                )
                print(f"Source gates completed: {len(receipts)}/34", flush=True)
            with localcontext() as context:
                context.prec = 50
                assets = {}
                for asset in native.ASSETS:
                    indices = [
                        selected[f"{asset.lower()}-index-{i:02d}"] for i in range(10)
                    ]
                    rows = block_rows(indices, selected[f"{asset.lower()}-binance"])
                    assets[asset] = {
                        "rows": rows,
                        **adjudicate(rows, contract["roles"]),
                    }
            result.update(
                status="complete_funding_only_prefilter",
                assets=assets,
                survivors=[a for a, r in assets.items() if r["passes_prefilter"]],
            )
            native.record(
                journal, {"phase": "study_completed", "survivors": result["survivors"]}
            )
        except Exception as error:
            result.update(
                status="terminal_source_or_alignment_failure",
                failure_type=type(error).__name__,
                failure_detail=str(error),
            )
            native.record(
                journal,
                {
                    "phase": "study_terminal_failure",
                    "error_type": type(error).__name__,
                    "detail": str(error),
                },
            )
    result["result_sha256"] = native.digest(native.canonical(result))
    with (output / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({k: result[k] for k in ("status", "result_sha256")}))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    run(args.contract, preflight=args.preflight)


if __name__ == "__main__":
    main()
