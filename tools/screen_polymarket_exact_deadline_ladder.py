"""Capture and rejection-screen one exact two-deadline Polymarket ladder."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)

SCHEMA = "polymarket-exact-deadline-ladder-prefilter-result-v1"


def _list(value: object, *, name: str) -> list[Any]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise RuntimeError(f"{name} must be a list")
    return parsed


def _journal(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".atomic-write.tmp")
    if temporary.exists():
        raise RuntimeError("atomic-write temporary path already exists")
    temporary.write_bytes(payload)
    if temporary.read_bytes() != payload:
        raise RuntimeError("durable-path byte preflight failed")
    temporary.replace(path)


def _preflight_durable_paths(outputs: dict[str, Path]) -> None:
    for parent in sorted({path.parent for path in outputs.values()}):
        parent.mkdir(parents=True, exist_ok=True)
        probe = parent / ".deadline-ladder-write-preflight.tmp"
        if probe.exists():
            raise RuntimeError("durable-path preflight file already exists")
        probe.write_bytes(b"synthetic-preflight\n")
        if probe.read_bytes() != b"synthetic-preflight\n":
            raise RuntimeError("durable-path preflight readback failed")
        probe.unlink()


def _request(
    *, url: str, raw_path: Path, raw_relative_path: str, journal_path: Path
) -> tuple[bytes, dict[str, Any]]:
    intent = {
        "method": "GET",
        "name": "exact-deadline-ladder-event",
        "phase": "intent",
        "request_body_sha256": _sha256(b""),
        "requested_at_ms": time.time_ns() // 1_000_000,
        "url": url,
    }
    _journal(journal_path, intent)
    request = Request(
        url,
        method="GET",
        headers={"User-Agent": "simple-ai-trading-public-research/1"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read()
            status_code = response.status
    except HTTPError as exc:
        raw = exc.read()
        status_code = exc.code
        _atomic_write(raw_path, raw)
        _journal(
            journal_path,
            {
                **intent,
                "completed_at_ms": time.time_ns() // 1_000_000,
                "phase": "completed",
                "raw_path": raw_relative_path,
                "response_bytes": len(raw),
                "response_sha256": _sha256(raw),
                "status_code": status_code,
            },
        )
        raise
    _atomic_write(raw_path, raw)
    receipt = {
        **intent,
        "completed_at_ms": time.time_ns() // 1_000_000,
        "phase": "completed",
        "raw_path": raw_relative_path,
        "response_bytes": len(raw),
        "response_sha256": _sha256(raw),
        "status_code": status_code,
    }
    _journal(journal_path, receipt)
    if status_code != 200:
        raise RuntimeError(f"unexpected HTTP status {status_code}")
    return raw, receipt


def _validate_contract(contract: dict[str, Any], path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    frozen_text = contract.get("frozen_at_utc")
    if not isinstance(frozen_text, str) or not frozen_text.endswith("Z"):
        raise RuntimeError("frozen_at_utc must be an exact UTC instant")
    frozen = datetime.fromisoformat(frozen_text.replace("Z", "+00:00"))
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise RuntimeError("frozen_at_utc is invalid or in the future")
    slug = contract.get("event_slug")
    if contract.get("request") != {
        "body_sha256": _sha256(b""),
        "count": 1,
        "method": "GET",
        "url": f"https://gamma-api.polymarket.com/events/slug/{slug}",
    }:
        raise RuntimeError("request boundary changed")
    groups = contract.get("expected_deadline_groups")
    if not (
        isinstance(groups, list)
        and len(groups) == 2
        and all(isinstance(value, str) and value for value in groups)
        and len(set(groups)) == 2
    ):
        raise RuntimeError("exact two-deadline population is invalid")
    if contract.get("authority") != {
        "account_requests": 0,
        "book_requests": 0,
        "credentials_used": False,
        "fee_requests": 0,
        "funds_used": False,
        "onchain_requests": 0,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "public_unauthenticated_read_only_requests": 1,
        "signed_requests": 0,
        "trading_authority": False,
    }:
        raise RuntimeError("authority boundary changed")
    implementation = _root_path(str(contract["implementation"]["path"]))
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")


def _market_row(
    market: dict[str, Any],
    *,
    expected_group: str,
    required_rule_fragments: list[str],
) -> dict[str, Any]:
    if market.get("groupItemTitle") != expected_group:
        raise RuntimeError("deadline group changed")
    description = str(market.get("description") or "")
    if not all(fragment in description for fragment in required_rule_fragments):
        raise RuntimeError("exact deadline resolution rules changed")
    outcomes = _list(market.get("outcomes"), name="outcomes")
    prices = _list(market.get("outcomePrices"), name="outcome prices")
    tokens = _list(market.get("clobTokenIds"), name="CLOB token IDs")
    if not (
        outcomes == ["Yes", "No"]
        and len(prices) == 2
        and len(tokens) == 2
        and all(isinstance(token, str) and token for token in tokens)
        and market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
        and market.get("enableOrderBook") is True
    ):
        raise RuntimeError("deadline market is not active and order-book enabled")
    yes = Decimal(str(prices[0]))
    no = Decimal(str(prices[1]))
    if not (
        yes.is_finite()
        and no.is_finite()
        and Decimal("0") <= yes <= Decimal("1")
        and Decimal("0") <= no <= Decimal("1")
    ):
        raise RuntimeError("displayed outcome price is invalid")
    return {
        "deadline_group": expected_group,
        "gamma_market_id": str(market["id"]),
        "condition_id": str(market["conditionId"]),
        "question": str(market["question"]),
        "yes_price_pUSD": format(yes, "f"),
        "no_price_pUSD": format(no, "f"),
        "yes_token_id": tokens[0],
        "no_token_id": tokens[1],
        "fees_enabled": bool(market.get("feesEnabled")),
        "fee_schedule": market.get("feeSchedule"),
        "minimum_order_size": str(market.get("orderMinSize")),
        "minimum_tick_size": str(market.get("orderPriceMinTickSize")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text(encoding="ascii"))
    _validate_contract(contract, contract_path)
    outputs = {key: _root_path(value) for key, value in contract["outputs"].items()}
    for output in outputs.values():
        if output.exists():
            raise RuntimeError(f"one-use output already exists: {output.name}")
    _preflight_durable_paths(outputs)

    raw, receipt = _request(
        url=contract["request"]["url"],
        raw_path=outputs["raw_path"],
        raw_relative_path=contract["outputs"]["raw_path"],
        journal_path=outputs["journal_path"],
    )
    event = json.loads(raw)
    if not isinstance(event, dict):
        raise RuntimeError("exact event response must be an object")
    if not (
        event.get("slug") == contract["event_slug"]
        and event.get("title") == contract["event_title"]
        and event.get("active") is True
        and event.get("closed") is False
    ):
        raise RuntimeError("exact deadline event identity or state changed")
    markets = event.get("markets")
    if not isinstance(markets, list) or len(markets) != 2:
        raise RuntimeError("exact deadline market count changed")
    by_group: dict[str, dict[str, Any]] = {}
    for market in markets:
        if not isinstance(market, dict):
            raise RuntimeError("deadline market must be an object")
        group = str(market.get("groupItemTitle") or "")
        if group in by_group:
            raise RuntimeError("duplicate deadline group")
        by_group[group] = market
    legs = [
        _market_row(
            by_group[group],
            expected_group=group,
            required_rule_fragments=contract["required_rule_fragments"],
        )
        for group in contract["expected_deadline_groups"]
    ]
    earlier, later = legs
    displayed_sum = Decimal(earlier["no_price_pUSD"]) + Decimal(later["yes_price_pUSD"])
    passes = displayed_sum < Decimal("1")
    package = {
        "earlier_deadline_group": earlier["deadline_group"],
        "earlier_no_market_id": earlier["gamma_market_id"],
        "earlier_no_token_id": earlier["no_token_id"],
        "later_deadline_group": later["deadline_group"],
        "later_yes_market_id": later["gamma_market_id"],
        "later_yes_token_id": later["yes_token_id"],
        "displayed_price_sum_pUSD": format(displayed_sum, "f"),
        "optimistic_displayed_headroom_pUSD": format(Decimal("1") - displayed_sum, "f"),
        "guaranteed_payout_floor_pUSD_per_share": "1",
        "passes_strict_displayed_gross_gate": passes,
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "capture": {"receipt": receipt},
        "payoff_identity": contract["payoff_identity"],
        "screen": {
            "event_slug": contract["event_slug"],
            "market_count": len(legs),
            "legs": legs,
            "package": package,
            "gamma_role": "rejection_only_never_acceptance_or_promotion_evidence",
        },
        "adjudication": {
            "status": (
                "source_only_candidate_requires_separate_exact_depth_and_fee_proof"
                if passes
                else "rejected_before_books_fees_and_onchain_requests"
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    outputs["result_path"].write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "best_displayed_sum_pUSD": package["displayed_price_sum_pUSD"],
                "market_count": len(legs),
                "payloads_printed": 0,
                "strict_displayed_candidate_count": int(passes),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
