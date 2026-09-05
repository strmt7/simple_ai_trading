"""One frozen public major-asset quote window; diagnostic, never trading."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal as D
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {
    "BTCUSDT": ("BTC", "USDT"),
    "ETHUSDT": ("ETH", "USDT"),
    "SOLUSDT": ("SOL", "USDT"),
    "ETHBTC": ("ETH", "BTC"),
    "SOLBTC": ("SOL", "BTC"),
    "SOLETH": ("SOL", "ETH"),
}
URL = "https://data-api.binance.vision/api/v3/ticker/bookTicker?" + urlencode(
    {"symbols": json.dumps(list(PAIRS), separators=(",", ":"))}
)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="ascii", newline="\n") as out:
        out.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        out.flush()
        os.fsync(out.fileno())


def screen(payload: bytes) -> list[dict]:
    """Exhaust six USDT-start directed cycles without asserting simultaneous fills."""
    rows = json.loads(payload)
    if (
        not isinstance(rows, list)
        or len(rows) != len(PAIRS)
        or {x.get("symbol") for x in rows} != set(PAIRS)
    ):
        raise ValueError("exact six-symbol population required")
    rates = {}
    for row in rows:
        values = {k: D(row[k]) for k in ("bidPrice", "askPrice", "bidQty", "askQty")}
        if (
            any(not x.is_finite() or x <= 0 for x in values.values())
            or values["askPrice"] < values["bidPrice"]
        ):
            raise ValueError("invalid or crossed book")
        if type(row.get("bidPrice")) is not str or type(row.get("askPrice")) is not str:
            raise ValueError("exact decimal string prices required")
        base, quote = PAIRS[row["symbol"]]
        rates[base, quote] = values["bidPrice"]
        rates[quote, base] = D(1) / values["askPrice"]
    result = []
    for first, second in itertools.permutations(("BTC", "ETH", "SOL"), 2):
        gross = (
            rates["USDT", first] * rates[first, second] * rates[second, "USDT"] - 1
        ) * 10000
        result.append(
            {
                "cycle": f"USDT->{first}->{second}->USDT",
                "ideal_zero_fee_bips": str(gross),
                "after_three_bip_stress": str(gross - 3),
            }
        )
    return result


def capture(path: Path, url: str, journal: Path, index: int) -> tuple[bytes, dict]:
    if url != URL:
        raise ValueError("only exact frozen public book URL allowed")
    started = time.time_ns() // 1_000_000
    append(
        journal,
        {
            "phase": "request_intent",
            "index": index,
            "url": url,
            "method": "GET",
            "started_ms": started,
        },
    )
    status = None
    error = None
    with path.open("xb") as out:
        try:
            request = Request(
                url,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                    "User-Agent": "simple-ai-trading-public-research/1",
                },
            )
            try:
                response = build_opener(ProxyHandler({}), NoRedirect()).open(
                    request, timeout=3
                )
            except HTTPError as exc:
                response = exc
            with response:
                status = response.code
                deadline = time.monotonic() + 3
                total = 0
                while total < 65537:
                    if time.monotonic() > deadline:
                        raise TimeoutError("elapsed read budget")
                    chunk = response.read(min(4096, 65537 - total))
                    if not chunk:
                        break
                    out.write(chunk)
                    total += len(chunk)
        except Exception as exc:
            error = type(exc).__name__
        finally:
            out.flush()
            os.fsync(out.fileno())
    data = path.read_bytes()
    receipt = {
        "phase": "request_completed",
        "index": index,
        "status": status,
        "error_type": error,
        "started_ms": started,
        "finished_ms": time.time_ns() // 1_000_000,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    append(journal, receipt)
    if error or status != 200 or len(data) > 65536:
        raise ValueError("source request failed; consumed without retry")
    return data, receipt


def run(contract_path: Path) -> dict:
    plan = json.loads(contract_path.read_bytes())
    if (
        plan["url"] != URL
        or plan["sample_count"] != 12
        or plan["interval_seconds"] != 5
    ):
        raise ValueError("frozen request scope mismatch")
    source = Path(__file__)
    if hashlib.sha256(source.read_bytes()).hexdigest() != plan["implementation_sha256"]:
        raise ValueError("implementation mismatch")
    now = datetime.now(timezone.utc)
    if (
        not datetime.fromisoformat(plan["not_before"])
        <= now
        <= datetime.fromisoformat(plan["start_deadline"])
    ):
        raise ValueError("outside frozen start window")
    base = ROOT / plan["output_directory"]
    if base.resolve() != contract_path.parent.resolve() or not base.is_dir():
        raise ValueError("outputs must stay with frozen contract")
    journal = base / "journal.jsonl"
    if (base / "result.json").exists():
        raise ValueError("consumed result destination")
    with journal.open("x"):
        pass
    for i in range(12):
        if (base / f"book-{i:02}.json").exists():
            raise ValueError("consumed raw destination")
    start = time.monotonic()
    observations = []
    failure = None
    try:
        for index in range(12):
            due = start + index * 5
            time.sleep(max(0, due - time.monotonic()))
            if time.monotonic() - due > 1:
                raise ValueError("missed fixed sample slot")
            payload, receipt = capture(
                base / f"book-{index:02}.json", URL, journal, index
            )
            if receipt["finished_ms"] - receipt["started_ms"] > 2000:
                raise ValueError("request receipt span exceeds two seconds")
            observations.append(
                {"index": index, "receipt": receipt, "cycles": screen(payload)}
            )
    except Exception as exc:
        failure = type(exc).__name__
    result = {
        "schema_version": "prospective-triangle-rate-window-v1",
        "observations": observations,
        "complete": len(observations) == 12 and failure is None,
        "failure_type": failure,
        "accepted_edge": False,
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "scope": "Rate-product diagnostic; no size, rounding, fee, account or simultaneous-fill qualification",
    }
    if result["complete"]:
        result["cycles"] = [
            {
                "cycle": observations[0]["cycles"][j]["cycle"],
                "positive_after_stress_observations": sum(
                    D(o["cycles"][j]["after_three_bip_stress"]) > 0
                    for o in observations
                ),
                "all_twelve_positive": all(
                    D(o["cycles"][j]["after_three_bip_stress"]) > 0
                    for o in observations
                ),
            }
            for j in range(6)
        ]
    append(
        journal,
        {"phase": "terminal", "complete": result["complete"], "failure_type": failure},
    )
    with (base / "result.json").open("x", encoding="ascii", newline="\n") as out:
        json.dump(result, out, sort_keys=True, indent=2)
        out.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    args = parser.parse_args()
    result = run(args.contract.resolve())
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "observations"}, sort_keys=True
        )
    )
