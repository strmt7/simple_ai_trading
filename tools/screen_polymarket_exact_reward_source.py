"""Reconcile one frozen Polymarket market with its exact sponsored reward source."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

import requests

from simple_ai_trading.storage import write_bytes_atomic
from tools.screen_polymarket_exact_paired_maker_reward import (
    _canonical,
    _decimal,
    _json_list,
    _list,
    _mapping,
    _request,
    _sha256,
    _utc_datetime,
)


ROOT = Path(__file__).resolve().parents[1]


def _contract(
    path: Path,
    now: datetime,
    *,
    schema_version: str = "polymarket-exact-reward-source-contract-v1",
) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    contract = _mapping(json.loads(raw), name="contract")
    claimed = str(contract.get("result_sha256") or "")
    body = dict(contract)
    body.pop("result_sha256", None)
    if claimed != _sha256(_canonical(body)):
        raise ValueError("contract embedded hash does not reconstruct")
    if contract.get("schema_version") != schema_version:
        raise ValueError("unsupported contract schema")
    frozen = _utc_datetime(contract.get("frozen_at_utc"))
    capture = _mapping(contract.get("capture"), name="capture")
    if frozen > now or now - frozen > timedelta(
        minutes=int(capture["activation_window_minutes"])
    ):
        raise ValueError("frozen contract activation window expired")
    return contract, _sha256(raw)


def _preflight_destination(path: Path, *, name: str) -> None:
    parent = path.parent
    if not parent.is_dir():
        raise ValueError(f"{name} parent does not exist")
    probe = parent / f".{path.name}.write-probe"
    try:
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        raise ValueError(f"{name} parent is not writable") from exc


def _gamma(raw: object, *, candidate: dict[str, Any]) -> dict[str, Any]:
    rows = [_mapping(value, name="Gamma row") for value in _list(raw, name="Gamma")]
    if len(rows) != 1:
        raise ValueError("Gamma did not return exactly one frozen market")
    row = rows[0]
    if not (
        row.get("slug") == candidate["market_slug"]
        and row.get("question") == candidate["question"]
        and row.get("active") is True
        and row.get("closed") is False
        and row.get("acceptingOrders") is True
        and row.get("enableOrderBook") is True
    ):
        raise ValueError("Gamma market identity or execution state changed")
    condition_id = str(row.get("conditionId") or "").lower()
    if not condition_id.startswith("0x") or len(condition_id) != 66:
        raise ValueError("Gamma condition ID is invalid")
    outcomes = [
        str(value) for value in _json_list(row.get("outcomes"), name="outcomes")
    ]
    tokens = [
        str(value) for value in _json_list(row.get("clobTokenIds"), name="tokens")
    ]
    if outcomes != candidate["outcomes"] or len(tokens) != 2 or len(set(tokens)) != 2:
        raise ValueError("Gamma binary outcome or token identity changed")
    event_end = _utc_datetime(row.get("endDate"), end_of_date=True)
    if "event_end_utc" in candidate:
        if event_end != _utc_datetime(candidate["event_end_utc"]):
            raise ValueError("Gamma event end changed")
    elif "event_end_date_utc" in candidate:
        expected_end_date = datetime.fromisoformat(
            str(candidate["event_end_date_utc"])
        ).date()
        if event_end.date() != expected_end_date:
            raise ValueError("Gamma event end date changed")
    else:
        raise ValueError("candidate event end gate is missing")
    order_minimum = _decimal(
        row.get("orderMinSize"), name="order minimum", positive=True
    )
    reward_minimum = _decimal(
        row.get("rewardsMinSize"), name="Gamma reward minimum", positive=True
    )
    reward_spread = _decimal(
        row.get("rewardsMaxSpread"), name="Gamma reward spread", positive=True
    )
    fee_schedule = row.get("feeSchedule")
    fee = {} if fee_schedule is None else _mapping(fee_schedule, name="fee schedule")
    maker_fee_zero = row.get("feesEnabled") is False or (
        row.get("feesEnabled") is True and fee.get("takerOnly") is True
    )
    if not maker_fee_zero:
        raise ValueError("Gamma does not establish zero maker fee")
    return {
        "condition_id": condition_id,
        "tokens": tokens,
        "outcomes": outcomes,
        "event_end": event_end,
        "order_minimum": order_minimum,
        "reward_minimum": reward_minimum,
        "reward_spread": reward_spread,
        "tick_size": _decimal(
            row.get("orderPriceMinTickSize"), name="tick", positive=True
        ),
        "maker_fee_zero": maker_fee_zero,
        "fee_schedule": fee,
    }


def _reward(
    raw: object,
    *,
    market: dict[str, Any],
    candidate: dict[str, Any],
    now: datetime,
    terminal_cursor: str,
) -> dict[str, Any]:
    payload = _mapping(raw, name="reward response")
    if payload.get("next_cursor") != terminal_cursor:
        raise ValueError("exact reward response is not a complete one-page population")
    rows = [
        _mapping(value, name="reward row")
        for value in _list(payload.get("data"), name="reward data")
    ]
    if len(rows) != 1:
        raise ValueError("exact reward response did not contain exactly one row")
    row = rows[0]
    if not (
        str(row.get("condition_id") or "").lower() == market["condition_id"]
        and row.get("market_slug") == candidate["market_slug"]
        and row.get("event_slug") == candidate["event_slug"]
        and row.get("question") == candidate["question"]
    ):
        raise ValueError("exact reward identity changed")
    reward_tokens = [
        _mapping(value, name="reward token")
        for value in _list(row.get("tokens"), name="reward tokens")
    ]
    if [str(value.get("token_id") or "") for value in reward_tokens] != market[
        "tokens"
    ] or [str(value.get("outcome") or "").casefold() for value in reward_tokens] != [
        str(value).casefold() for value in market["outcomes"]
    ]:
        raise ValueError("exact Gamma and sponsored reward token identity disagree")
    minimum = _decimal(
        row.get("rewards_min_size"), name="reward minimum", positive=True
    )
    spread = _decimal(
        row.get("rewards_max_spread"), name="reward spread", positive=True
    )
    if minimum != market["reward_minimum"] or spread != market["reward_spread"]:
        raise ValueError("exact Gamma and sponsored reward settings disagree")
    if minimum < market["order_minimum"]:
        raise ValueError("reward minimum is below the executable order minimum")
    active: list[dict[str, Any]] = []
    for value in _list(row.get("rewards_config"), name="reward configurations"):
        config = _mapping(value, name="reward configuration")
        start = _utc_datetime(config.get("start_date"))
        end = _utc_datetime(config.get("end_date"), end_of_date=True)
        if start <= now < end:
            active.append({**config, "_end_utc": end})
    if len(active) != 1:
        raise ValueError("expected exactly one active dated reward configuration")
    rate = _decimal(active[0].get("rate_per_day"), name="daily rate", positive=True)
    active_end = active[0]["_end_utc"]
    remaining_days = Decimal(
        str(max((min(active_end, market["event_end"]) - now).total_seconds(), 0))
    ) / Decimal("86400")
    if remaining_days <= 0:
        raise ValueError("reward or market horizon has ended")
    config = dict(active[0])
    config.pop("_end_utc", None)
    return {
        "minimum_size": minimum,
        "maximum_spread_cents": spread,
        "daily_rate": rate,
        "active_end": active_end,
        "remaining_days": remaining_days,
        "active_configuration": config,
        "market_competitiveness": row.get("market_competitiveness"),
    }


def run(*, contract_path: Path, output: Path, journal_dir: Path) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    contract, contract_file_sha = _contract(contract_path, started)
    _preflight_destination(output, name="output")
    if journal_dir.exists():
        raise ValueError("journal directory already exists")
    _preflight_destination(journal_dir, name="journal")
    journal_dir.mkdir()
    candidate = _mapping(contract.get("candidate"), name="candidate")
    request = _mapping(contract.get("request_contract"), name="request contract")
    byte_ceiling = int(request["response_byte_ceiling"])
    http = requests.Session()
    gamma_raw, gamma_source = _request(
        http,
        method="GET",
        url=str(request["gamma_endpoint"]),
        params=_mapping(request["gamma_params"], name="Gamma params"),
        json_body=None,
        journal_dir=journal_dir,
        source_name="01-exact-gamma-market",
        byte_ceiling=byte_ceiling,
    )
    market = _gamma(gamma_raw, candidate=candidate)
    reward_raw, reward_source = _request(
        http,
        method="GET",
        url=str(request["exact_reward_endpoint_pattern"]).format(
            condition_id=market["condition_id"]
        ),
        params=_mapping(request["reward_params"], name="reward params"),
        json_body=None,
        journal_dir=journal_dir,
        source_name="02-exact-sponsored-reward",
        byte_ceiling=byte_ceiling,
    )
    reward = _reward(
        reward_raw,
        market=market,
        candidate=candidate,
        now=started,
        terminal_cursor=str(request["terminal_cursor"]),
    )
    artifact: dict[str, object] = {
        "schema_version": "polymarket-exact-reward-source-prefilter-v1",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "event_slug": candidate["event_slug"],
            "market_slug": candidate["market_slug"],
            "question": candidate["question"],
            "condition_id": market["condition_id"],
            "tokens": market["tokens"],
            "event_end_utc": market["event_end"].isoformat(),
            "tick_size": str(market["tick_size"]),
            "minimum_order_size_shares": str(market["order_minimum"]),
            "maker_fee_zero": market["maker_fee_zero"],
            "fee_schedule": market["fee_schedule"],
        },
        "exact_reward": {
            "minimum_size_shares": str(reward["minimum_size"]),
            "maximum_spread_cents": str(reward["maximum_spread_cents"]),
            "daily_rate_pUSD": str(reward["daily_rate"]),
            "active_end_utc": reward["active_end"].isoformat(),
            "remaining_reward_days": str(reward["remaining_days"]),
            "active_configuration": reward["active_configuration"],
            "market_competitiveness": reward["market_competitiveness"],
        },
        "verdict": {
            "status": "exact_sources_reconciled_book_screen_permitted",
            "books_requested": False,
            "accepted_edge": False,
            "profitability_claim": False,
            "trading_authority": False,
            "retry_permitted": False,
        },
        "authority": {
            "public_unauthenticated_read_only": True,
            "credentials_used": False,
            "orders_or_cancellations": 0,
            "funded_actions": 0,
        },
        "sources": {
            "contract_path": contract_path.resolve().relative_to(ROOT).as_posix(),
            "contract_file_sha256": contract_file_sha,
            "contract_result_sha256": contract["result_sha256"],
            "gamma_request": gamma_source,
            "reward_request": reward_source,
            "tool_sha256": _sha256(Path(__file__).read_bytes()),
        },
        "limitations": [
            "Discovery-page prices, spread, size, and reward values are excluded from exact economics.",
            "This source-only prefilter requests no books and proves no executable spread, fill, reward payout, or profit.",
            "Any book screen must be separately frozen from these retained exact identities and settings.",
        ],
    }
    artifact["result_sha256"] = _sha256(_canonical(artifact))
    write_bytes_atomic(output, _canonical(artifact) + b"\n")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(
            contract_path=args.contract,
            output=args.output,
            journal_dir=args.journal_dir,
        )
    except Exception as exc:
        if args.journal_dir.exists():
            failure = {
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "raw_responses_retained_before_validation": True,
                "retry_permitted": False,
            }
            write_bytes_atomic(
                args.journal_dir / "terminal-failure.json",
                _canonical(failure) + b"\n",
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
