"""Screen one frozen Polymarket book from retained exact reward sources."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any

import requests

from simple_ai_trading.storage import write_bytes_atomic
from tools.screen_polymarket_exact_paired_maker_reward import (
    _canonical,
    _decimal,
    _diagnostic,
    _json_list,
    _levels,
    _list,
    _mapping,
    _request,
    _sha256,
    _utc_datetime,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs/model-research/polymarket/elon-posts-40-64-retained-source-book-contract-v1-2026-08-30.json"
)


def _contract(now: datetime) -> tuple[dict[str, Any], str]:
    contract = _mapping(json.loads(CONTRACT_PATH.read_text("ascii")), name="contract")
    claimed = str(contract.get("result_sha256") or "")
    body = dict(contract)
    body.pop("result_sha256", None)
    if claimed != _sha256(_canonical(body)):
        raise ValueError("contract embedded hash does not reconstruct")
    frozen = _utc_datetime(contract["frozen_at_utc"])
    capture = _mapping(contract["capture"], name="capture")
    if frozen > now or now - frozen > timedelta(
        minutes=int(capture["activation_window_minutes"])
    ):
        raise ValueError("frozen contract activation window expired")
    return contract, _sha256(CONTRACT_PATH.read_bytes())


def _retained_json(source: dict[str, Any]) -> object:
    path = ROOT / str(source["path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError(f"retained source hash changed: {source['path']}")
    return json.loads(raw)


def _retained_market(
    raw: object,
    *,
    candidate: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, object]:
    rows = [_mapping(value, name="Gamma row") for value in _list(raw, name="Gamma")]
    if len(rows) != 1:
        raise ValueError("retained Gamma source is not exactly one market")
    row = rows[0]
    condition_id = str(row.get("conditionId") or "").lower()
    tokens = [str(value) for value in _json_list(row.get("clobTokenIds"), name="tokens")]
    if not (
        condition_id == candidate["condition_id"]
        and row.get("slug") == candidate["market_slug"]
        and row.get("question") == candidate["question"]
        and tokens == [candidate["yes_token_id"], candidate["no_token_id"]]
        and row.get("active") is True
        and row.get("closed") is False
        and row.get("acceptingOrders") is True
        and row.get("enableOrderBook") is True
    ):
        raise ValueError("retained Gamma identity or execution state changed")
    fee = _mapping(row.get("feeSchedule"), name="fee schedule")
    maker_fee_zero = row.get("feesEnabled") is True and fee.get("takerOnly") is True
    if maker_fee_zero is not expected["expected_maker_fee_zero"]:
        raise ValueError("retained maker fee gate changed")
    checks = (
        (row.get("orderPriceMinTickSize"), expected["expected_tick_size"], "tick"),
        (
            row.get("orderMinSize"),
            expected["expected_minimum_order_size_shares"],
            "order minimum",
        ),
        (
            row.get("rewardsMinSize"),
            expected["expected_reward_minimum_size_shares"],
            "reward minimum",
        ),
        (
            row.get("rewardsMaxSpread"),
            expected["expected_reward_maximum_spread_cents"],
            "reward spread",
        ),
    )
    for actual, frozen, name in checks:
        if _decimal(actual, name=name) != _decimal(frozen, name=f"expected {name}"):
            raise ValueError(f"retained Gamma {name} changed")
    event_end = _utc_datetime(row.get("endDate"))
    if event_end != _utc_datetime(candidate["event_end_utc"]):
        raise ValueError("retained event end changed")
    return {
        "condition_id": condition_id,
        "tokens": tokens,
        "tick_size": _decimal(row["orderPriceMinTickSize"], name="tick"),
        "minimum_order_size": _decimal(row["orderMinSize"], name="minimum"),
        "event_end": event_end.isoformat(),
        "fee_schedule": fee,
        "maker_fee_zero": maker_fee_zero,
    }


def _retained_reward(
    raw: object,
    *,
    market: dict[str, object],
    expected: dict[str, Any],
    now: datetime,
) -> dict[str, object]:
    payload = _mapping(raw, name="reward response")
    if payload.get("next_cursor") != expected["expected_terminal_cursor"]:
        raise ValueError("retained exact reward cursor changed")
    rows = [_mapping(value, name="reward row") for value in _list(payload.get("data"), name="reward data")]
    if len(rows) != 1 or str(rows[0].get("condition_id") or "").lower() != market["condition_id"]:
        raise ValueError("retained exact reward identity changed")
    row = rows[0]
    size = _decimal(row.get("rewards_min_size"), name="reward minimum", positive=True)
    spread = _decimal(row.get("rewards_max_spread"), name="reward spread", positive=True)
    if size != _decimal(expected["expected_minimum_size_shares"], name="expected size"):
        raise ValueError("retained exact reward minimum changed")
    if spread != _decimal(expected["expected_maximum_spread_cents"], name="expected spread"):
        raise ValueError("retained exact reward spread changed")
    configurations = [_mapping(value, name="reward configuration") for value in _list(row.get("rewards_config"), name="reward configurations")]
    active = []
    for config in configurations:
        start = _utc_datetime(config.get("start_date"))
        end = _utc_datetime(config.get("end_date"), end_of_date=True)
        if start <= now < end:
            active.append((config, end))
    if len(active) != int(expected["expected_active_configuration_count"]):
        raise ValueError("retained active reward configuration count changed")
    rate = sum(
        (_decimal(config.get("rate_per_day"), name="daily rate", positive=True) for config, _end in active),
        Decimal("0"),
    )
    if rate != _decimal(expected["expected_daily_rate_pUSD"], name="expected rate"):
        raise ValueError("retained active daily reward rate changed")
    return {
        "minimum_size": size,
        "maximum_spread_cents": spread,
        "daily_rate": rate,
        "active_end": min(end for _config, end in active).isoformat(),
        "active_configurations": [config for config, _end in active],
    }


def run(*, output: Path, journal_dir: Path) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    contract, contract_file_sha = _contract(started)
    journal_dir.mkdir(parents=True, exist_ok=False)
    candidate = _mapping(contract["candidate"], name="candidate")
    retained = _mapping(contract["retained_exact_sources"], name="retained sources")
    gamma_source = _mapping(retained["gamma"], name="retained Gamma")
    reward_source = _mapping(retained["reward"], name="retained reward")
    market = _retained_market(
        _retained_json(gamma_source), candidate=candidate, expected=gamma_source
    )
    reward = _retained_reward(
        _retained_json(reward_source),
        market=market,
        expected=reward_source,
        now=started,
    )
    if reward["minimum_size"] != _decimal(
        gamma_source["expected_reward_minimum_size_shares"], name="Gamma size"
    ) or reward["maximum_spread_cents"] != _decimal(
        gamma_source["expected_reward_maximum_spread_cents"], name="Gamma spread"
    ):
        raise ValueError("retained exact Gamma and reward settings disagree")
    request = _mapping(contract["request_contract"], name="request contract")
    books_body = [{"token_id": token} for token in market["tokens"]]
    books_raw, books_source = _request(
        requests.Session(),
        method="POST",
        url=str(request["books_endpoint"]),
        params=None,
        json_body=books_body,
        journal_dir=journal_dir,
        source_name="01-two-token-books",
        byte_ceiling=int(request["response_byte_ceiling"]),
    )
    rows = [_mapping(value, name="book") for value in _list(books_raw, name="books")]
    books = {str(row.get("asset_id") or ""): row for row in rows}
    if len(rows) != 2 or set(books) != set(market["tokens"]):
        raise ValueError("books response did not contain the two exact tokens")
    timestamps: list[int] = []
    for row in rows:
        if str(row.get("market") or "").lower() != market["condition_id"]:
            raise ValueError("book condition identity changed")
        if _decimal(row.get("tick_size"), name="book tick") != market["tick_size"]:
            raise ValueError("book and retained Gamma tick sizes disagree")
        if _decimal(row.get("min_order_size"), name="book minimum") != market["minimum_order_size"]:
            raise ValueError("book and retained Gamma order minimums disagree")
        _levels(row, side="bids")
        _levels(row, side="asks")
        timestamps.append(int(str(row.get("timestamp"))))
    capture = _mapping(contract["capture"], name="capture")
    received_ms = int(books_source["received_after_ms"])
    event_age = received_ms - min(timestamps)
    skew = max(timestamps) - min(timestamps)
    fresh = (
        int(books_source["elapsed_ms"]) <= int(capture["request_max_elapsed_ms"])
        and skew <= int(capture["book_max_timestamp_skew_ms"])
        and 0 <= event_age <= int(capture["book_max_event_age_ms"])
    )
    if not fresh:
        raise ValueError("book capture failed frozen freshness gates")
    observed_at = datetime.fromtimestamp(received_ms / 1000, timezone.utc)
    decision = _mapping(contract["decision"], name="decision")
    diagnostic = _diagnostic(
        market=market,
        reward=reward,
        books=books,
        stress_multipliers=_list(decision["stress_multipliers"], name="stress multipliers"),
        observed_at=observed_at,
    )
    artifact: dict[str, object] = {
        "schema_version": "polymarket-retained-exact-source-book-screen-v1",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            **candidate,
            "maker_fee_zero": market["maker_fee_zero"],
            "fee_schedule": market["fee_schedule"],
            "tick_size": str(market["tick_size"]),
            "minimum_order_size_shares": str(market["minimum_order_size"]),
            "reward_minimum_size_shares": str(reward["minimum_size"]),
            "reward_maximum_spread_cents": str(reward["maximum_spread_cents"]),
            "reward_daily_rate_pUSD": str(reward["daily_rate"]),
            "active_reward_configurations": reward["active_configurations"],
        },
        "capture": {
            "request_elapsed_ms": books_source["elapsed_ms"],
            "oldest_book_event_age_ms": event_age,
            "book_timestamp_skew_ms": skew,
            "freshness_passed": fresh,
        },
        "diagnostic": diagnostic,
        "verdict": {
            "status": diagnostic["status"],
            "accepted_edge": False,
            "profitability_claim": False,
            "trading_authority": False,
            "publicly_proven_reward_payout_floor_pUSD": "0",
        },
        "authority": {
            "public_unauthenticated_read_only": True,
            "credentials_used": False,
            "orders_or_cancellations": 0,
            "funded_actions": 0,
        },
        "sources": {
            "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "contract_file_sha256": contract_file_sha,
            "contract_result_sha256": contract["result_sha256"],
            "retained_gamma": gamma_source,
            "retained_reward": reward_source,
            "books_request": books_source,
            "tool_sha256": _sha256(Path(__file__).read_bytes()),
        },
        "limitations": [
            "Public books do not expose maker identity, queue position, reward persistence, random sampling, final epoch normalization, fills, or realized rewards.",
            "The score uses a conditional post-quote top midpoint because the public size-cutoff-adjusted midpoint construction is not fully specified.",
            "A one-sided fill is directional inventory and the public snapshot proves no positive reward payout floor.",
            "No adverse-selection, cancellation-latency, capacity, outage, tax, custody, or realized after-cost evidence is available from this snapshot.",
        ],
    }
    artifact["result_sha256"] = _sha256(_canonical(artifact))
    write_bytes_atomic(output, _canonical(artifact) + b"\n")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(output=args.output, journal_dir=args.journal_dir)
    except Exception as exc:
        args.journal_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "raw_responses_retained_before_validation": True,
            "retry_permitted": False,
        }
        write_bytes_atomic(
            args.journal_dir / "terminal-failure.json", _canonical(failure) + b"\n"
        )
        raise
    print(
        json.dumps(
            {
                "status": result["verdict"]["status"],
                "result_sha256": result["result_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
