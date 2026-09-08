"""Run the separately authorized September 8 MRNA rejection-only observation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import re

from tools.capture_public_source_bounded import capture
from tools.capture_public_source_contract import _canonical_hash, _load_object

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/review/2026-09-08/mrna-topbook"
NOT_BEFORE = datetime(2026, 9, 8, 13, 35, tzinfo=timezone.utc)
INVENTORY = "docs/model-research/action-value/binance-bstock-sep2-listing-inventory-result-v2-2026-09-04.json"
INVENTORY_HASH = "ba5ebb29ce89c8cc09bde5066bbe4a0cfc7dc11fa926f5a00de6321054caba26"
IMPLEMENTATIONS = (
    "tools/screen_mrna_bstock_topbook.py",
    "tools/capture_public_source_bounded.py",
    "tools/capture_public_source_contract.py",
    "tools/adjudicate_polymarket_exact_mlb_monotone_prefilter.py",
    "tools/screen_polymarket_exact_two_leg_package.py",
)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_new(path: Path, value: dict, field: str) -> None:
    value[field] = _canonical_hash(value, field)
    with path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + "\n")


def load_bound(path: Path, field: str, expected: str | None = None) -> dict:
    value = _load_object(path)
    if _canonical_hash(value, field) != value.get(field):
        raise ValueError("canonical evidence hash mismatch")
    if expected is not None and value[field] != expected:
        raise ValueError("expected evidence hash mismatch")
    return value


def parse_book(raw: bytes, symbol: str) -> dict[str, Decimal]:
    """Admit exact finite top-book numbers without treating missing sides as free."""
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("symbol") != symbol:
        raise ValueError("book symbol mismatch")
    result = {}
    for field in ("askPrice", "askQty", "bidPrice", "bidQty"):
        value = payload.get(field)
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[0-9]{1,30}(?:\.[0-9]{1,30})?", value) is None
        ):
            raise ValueError("book number must be a finite nonnegative decimal string")
        result[field] = Decimal(value)
    return result


def economics(spot: dict, future: dict, skew_ms: int) -> dict:
    with localcontext() as context:
        context.prec = 100
        ask, bid = spot["askPrice"], future["bidPrice"]
        gross = bid - ask
        stress = ask * Decimal("0.005")
        positive = all(
            value > 0 for value in (ask, bid, spot["askQty"], future["bidQty"])
        )
        return {
            "spot_ask_USDT_per_share": str(ask),
            "perpetual_bid_USDT_per_share": str(bid),
            "spot_ask_quantity_shares": str(spot["askQty"]),
            "perpetual_bid_quantity_shares": str(future["bidQty"]),
            "common_displayed_quantity_shares": str(
                min(spot["askQty"], future["bidQty"])
            ),
            "gross_entry_headroom_USDT_per_share": str(gross),
            "fixed_50_bip_stress_USDT_per_share": str(stress),
            "stressed_headroom_USDT_per_share": str(gross - stress),
            "stressed_headroom_bps": str((gross - stress) / ask * 10000)
            if ask > 0
            else None,
            "positive_entry_sides": positive,
            "request_start_skew_ms": skew_ms,
            "request_start_skew_passed": 0 <= skew_ms <= 10000,
            "passes_fixed_rejection_gate": positive
            and 0 <= skew_ms <= 10000
            and gross > stress,
        }


def freeze() -> None:
    now = datetime.now(timezone.utc)
    if now < NOT_BEFORE:
        raise ValueError("MRNA not-before gate has not elapsed")
    inventory = load_bound(ROOT / INVENTORY, "result_sha256", INVENTORY_HASH)
    tickers = [row["ticker"] for row in inventory["matching_unscreened_pairs"]]
    if tickers != ["CRWD", "MRNA", "SQQQ"]:
        raise ValueError("retained deterministic population changed")
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    (DIRECTORY / "raw").mkdir()
    timestamp = now.isoformat().replace("+00:00", "Z")
    bindings = [
        {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
        for path in IMPLEMENTATIONS
    ]
    sources = []
    for leg, old_symbol, symbol, host, route in (
        ("spot", "crwdb", "MRNABUSDT", "api", "api/v3"),
        ("futures", "crwd", "MRNAUSDT", "fapi", "fapi/v1"),
    ):
        template = (
            ROOT
            / f"docs/model-research/action-value/binance-{old_symbol}-{leg}-book-source-contract-v1-2026-09-04.json"
        )
        plan = _load_object(template)
        path = DIRECTORY / f"{leg}-contract.json"
        plan.update(
            frozen_at_utc=timestamp,
            contract_path=relative(path),
            request_name=f"mrna_{leg}_2026_09_08",
            exact_url_dedupe={
                "retained_match_count_before_access": 0,
                "alias_retry_forbidden": True,
            },
            required_utf8_phrases=[
                '"symbol"',
                f'"{symbol}"',
                '"askPrice"',
                '"askQty"',
                '"bidPrice"',
                '"bidQty"',
            ],
            outputs={
                "raw_path": relative(DIRECTORY / "raw" / f"{leg}.json"),
                "journal_path": relative(DIRECTORY / f"{leg}-journal.jsonl"),
                "result_path": relative(DIRECTORY / f"{leg}-result.json"),
            },
            implementations=bindings,
            transport={
                "socket_timeout_seconds": 10,
                "read_budget_seconds": 30,
                "redirects": False,
                "retries": 0,
                "proxies": False,
            },
        )
        plan["request"]["url"] = (
            f"https://{host}.binance.com/{route}/ticker/bookTicker?symbol={symbol}"
        )
        write_new(path, plan, "contract_sha256")
        sources.append(
            {
                "leg": leg,
                "symbol": symbol,
                "path": relative(path),
                "sha256": plan["contract_sha256"],
            }
        )
    contract = {
        "schema_version": "mrna-bstock-topbook-contract-v1",
        "frozen_at_utc": timestamp,
        "not_before_utc": NOT_BEFORE.isoformat(),
        "trigger": "Rank 12 next_action at Git c0dfb1d99260b8dd9cb3d5dfe98575e73326ac8c authorizes exactly the next lexicographic MRNA observation after September 8 13:35 UTC; CRWD remains consumed and SQQQ is not a fallback.",
        "inventory": {"path": INVENTORY, "result_sha256": INVENTORY_HASH},
        "sources": sources,
        "implementations": bindings,
        "economic_gate": "Buy MRNABUSDT at ask and short equal MRNAUSDT at bid. Exact-one multiplier from retained inventory; require positive displayed entry prices/quantities, request-start skew <=10000 ms, and bid-ask minus 50 bips of spot ask strictly positive. No favorable funding credited.",
        "limitations": "Screening stress is not actual fees or an economic guarantee. Request receipt/start times and futures transaction time do not prove quote-update freshness. No depth, fill, atomicity, conversion, exit basis, funding persistence, account, tax or capital-cost qualification.",
        "request_budget": {
            "maximum_GETs": 2,
            "weight_each": 2,
            "sequential_no_polling_or_retry": True,
            "stop_after_source_failure": True,
        },
        "consequences": "A failed exact observation stops without any downstream requests or adaptive SQQQ selection. A survivor permits only a separately frozen depth/adverse-funding/exit-basis/cost study, never orders or edge acceptance.",
        "source_documentation": "docs/review/2026-09-08/mrna-source-classification.md",
        "authority": {
            "public_only": True,
            "credentials_used": False,
            "account_requests": 0,
            "orders_or_funds": 0,
            "protected_capture_touched": False,
        },
    }
    write_new(DIRECTORY / "contract.json", contract, "contract_sha256")
    validate()


def validate() -> dict:
    contract = load_bound(DIRECTORY / "contract.json", "contract_sha256")
    frozen = datetime.fromisoformat(contract["frozen_at_utc"].replace("Z", "+00:00"))
    if not NOT_BEFORE <= frozen <= datetime.now(timezone.utc):
        raise ValueError("invalid freeze time")
    load_bound(ROOT / INVENTORY, "result_sha256", INVENTORY_HASH)
    for binding in contract["implementations"]:
        if (
            hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest()
            != binding["sha256"]
        ):
            raise ValueError("implementation changed")
    for source in contract["sources"]:
        load_bound(ROOT / source["path"], "contract_sha256", source["sha256"])
    return contract


def run(*, offline: bool) -> dict:
    contract = validate()
    result_path = DIRECTORY / "result.json"
    if not offline and result_path.exists():
        raise FileExistsError("observation already consumed")
    sources = contract["sources"]
    if not offline:
        for source in sources:
            capture(ROOT / source["path"], preflight=True)
    receipts, books, bindings = [], [], []
    for source in sources:
        plan = load_bound(ROOT / source["path"], "contract_sha256", source["sha256"])
        path = ROOT / plan["outputs"]["result_path"]
        if not offline:
            capture(ROOT / source["path"])
        result = load_bound(path, "result_sha256")
        if result["contract"] != {"path": source["path"], "sha256": source["sha256"]}:
            raise ValueError("source contract mismatch")
        bindings.append(
            {"path": relative(path), "result_sha256": result["result_sha256"]}
        )
        receipt = result["capture"]["receipt"]
        raw = (ROOT / receipt["raw_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt["response_sha256"]:
            raise ValueError("raw response changed")
        receipts.append(receipt)
        if not result["source_gate"]["passed"]:
            break
        try:
            books.append(parse_book(raw, source["symbol"]))
        except (ValueError, TypeError):
            break
    values = (
        economics(
            *books, abs(receipts[0]["requested_at_ms"] - receipts[1]["requested_at_ms"])
        )
        if len(books) == 2
        else None
    )
    outcome = {
        "schema_version": "mrna-bstock-topbook-result-v1",
        "contract_sha256": contract["contract_sha256"],
        "source_results": bindings,
        "request_count": len(receipts),
        "economics": values,
        "status": "source_failed_closed"
        if values is None
        else (
            "prefilter_survivor_not_an_edge"
            if values["passes_fixed_rejection_gate"]
            else "exact_observation_rejected"
        ),
        "accepted_edge": False,
        "profitability_claim": False,
        "deployment_ready": False,
        "authority": contract["authority"],
    }
    if not offline:
        write_new(result_path, outcome, "result_sha256")
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("freeze", "preflight", "capture", "reconstruct")
    )
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze()
    elif args.mode == "preflight":
        for source in validate()["sources"]:
            capture(ROOT / source["path"], preflight=True)
    elif args.mode == "capture":
        print(json.dumps(run(offline=False), sort_keys=True))
    else:
        actual = load_bound(DIRECTORY / "result.json", "result_sha256")
        actual.pop("result_sha256")
        if actual != run(offline=True):
            raise ValueError("reconstruction differs")
        print("Exact zero-network reconstruction passed")


if __name__ == "__main__":
    main()
