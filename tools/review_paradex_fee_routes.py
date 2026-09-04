"""Parse two retained official fee sources; no account or execution adapter."""

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/review/2026-09-04/paradex-fee-source"


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def one(pattern: str, text: str) -> tuple:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    if len(matches) != 1:
        raise ValueError("source field absent or ambiguous")
    return matches[0].groups()


def parse_terms(fees: str, profiles: str) -> dict:
    if "<html" in (fees + profiles).lower() or "<!doctype" in (fees + profiles).lower():
        raise ValueError("HTML is not the frozen Markdown format")
    for marker in (
        "# Trading Fees",
        "Fees are charged on trade notional (Size × Trade Price).",
        "When a market is delisted, a settlement fee applies",
    ):
        if marker not in fees:
            raise ValueError("fee semantics absent")
    for marker in (
        "not on a permanent user profile",
        "token_usage=interactive",
        "Orders submitted without an interactive token are classified as Pro Orders by default.",
        "account automatically switches to Pro Order behavior and loses access to RPI liquidity.",
    ):
        if marker not in profiles:
            raise ValueError("order classification semantics absent")
    for role in ("Taker", "Maker"):
        one(r"^\| " + role + r"\s+\| 0%\s+\| 0%\s+\|$", fees)
    maker_percent, maker_bps = one(r"Flat ([0-9.]+)% \(([0-9.]+) bps\) fee", fees)
    if Decimal(maker_percent) * 100 != Decimal(maker_bps):
        raise ValueError("maker percent and basis points disagree")
    floor_bps, floor_percent = one(
        r"Pro taker fees cannot go below ([0-9.]+) bps \(([0-9.]+)%\)", fees
    )
    if Decimal(floor_percent) * 100 != Decimal(floor_bps):
        raise ValueError("taker floor units disagree")
    tiers = {}
    for tier in range(5):
        _, value = one(
            r"^\| Pro " + str(tier) + r"\s*\| ([^|]+)\| ([0-9.]+)%\s*\|$", fees
        )
        tiers[str(tier)] = str(Decimal(value) * 100)
    bumps = {}
    for action in ("submission", "cancellation"):
        (value,) = one(
            r"^\| Order " + action + r" speed bump\s*\| ([0-9]+)ms\s*\| None\s*\|$",
            profiles,
        )
        bumps[action] = int(value)
    limits = {}
    for period in ("second", "minute", "hour", "24 hours"):
        (value,) = one(
            r"^\| Rate limit per " + period + r"\s*\| ([0-9,]+) orders[^|]*\|[^\n]+$",
            profiles,
        )
        limits[period] = int(value.replace(",", ""))
    (settlement_percent,) = one(
        r"^\| Spot and Perpetual Futures\s*\| ([0-9.]+)%\s*\|$", fees
    )
    return {
        "retail_perps_spot_maker_fee_bps": "0",
        "retail_perps_spot_taker_fee_bps": "0",
        "pro_perps_spot_maker_fee_bps": maker_bps,
        "pro_perps_spot_base_taker_fee_bps_by_tier": tiers,
        "pro_perps_spot_minimum_taker_fee_bps": floor_bps,
        "retail_documented_speed_bump_ms": bumps,
        "retail_documented_order_limits": limits,
        "delisted_spot_perpetual_open_position_settlement_fee_bps": str(
            Decimal(settlement_percent) * 100
        ),
        "classification_basis": "per order submission, not permanent user profile",
        "standard_api_default": "Pro",
        "documented_retail_api_route": "interactive token; token request not authorized or made",
        "retail_limit_breach": "Pro behavior and loss of RPI access",
        "fastfills_discount_is_fill_conditioned": True,
        "historical_effective_year_of_july3_maker_notice": None,
        "account_qualification": None,
        "intended_automated_workflow_eligibility_confirmed": False,
    }


def build_review() -> dict:
    raw = {name: (BASE / (name + ".md")).read_bytes() for name in ("fees", "profiles")}
    records = [
        json.loads(line) for line in (BASE / "requests.jsonl").read_text().splitlines()
    ]
    if len(records) != 4:
        raise ValueError("exact two-source journal differs")
    for i, name in enumerate(raw):
        completed = records[2 * i + 1]
        if (
            completed.get("phase") != "completed"
            or completed.get("passed") is not True
            or completed.get("status") != 200
            or completed.get("raw_sha256") != sha(raw[name])
            or completed.get("bytes") != len(raw[name])
        ):
            raise ValueError("source receipt differs")
    terms = parse_terms(raw["fees"].decode("utf-8"), raw["profiles"].decode("utf-8"))
    paths = [
        BASE / "fees.md",
        BASE / "profiles.md",
        BASE / "requests.jsonl",
        ROOT / "docs/review/2026-09-04/paradex-fee-source-plan.json",
        ROOT / "tools/capture_paradex_trading_markdown.ps1",
        Path(__file__),
    ]
    return {
        "schema_version": "paradex-fee-route-source-review-v1",
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in paths
        },
        "terms": terms,
        "decision": "Qualify exact order classification and all incremental latency, fill, currency, venue and counterparty costs before freezing a future execution route. No public fee is credited to an unqualified account. Preserve the consumed funding study and its rejection.",
        "fee_cash_identity": "sum over actual executions of abs(base quantity) * execution price * applicable fee fraction, retaining each payment asset; not twice an entry notional unless quantities and prices justify it",
        "retail_vs_pro_savings_condition": "The nominal rate difference is not execution improvement unless lower fees exceed changed spread, hedge latency, adverse selection, cancellation-race and other incremental costs.",
        "independently_existing_token_discount_state_required": True,
        "no_volume_manufacture_or_rate_limit_evasion": True,
        "market_requests": 0,
        "account_requests": 0,
        "token_requests": 0,
        "orders": 0,
        "credentials_used": False,
        "protected_capture_access": False,
        "historical_results_changed": False,
        "accepted_edge": False,
        "profitability_claim": False,
    }


def main():
    output = BASE / "review.json"
    if output.exists():
        raise FileExistsError("review already exists")
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **build_review(),
    }
    result["result_sha256"] = sha(
        json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )
    with output.open("xb") as stream:
        stream.write(json.dumps(result, indent=2).encode() + b"\n")
    print(
        json.dumps(
            {"terms": result["terms"], "result_sha256": result["result_sha256"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
