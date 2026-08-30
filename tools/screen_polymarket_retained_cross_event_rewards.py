"""Reconcile exact sponsored rewards for one retained cross-event maker package."""

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
SCHEMA = "polymarket-retained-cross-event-rewards-result-v1"


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def _canonical_hash(payload: dict[str, Any], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(_canonical(body))


def _json_ready(value: Any) -> Any:
    """Convert exact arithmetic values without losing their decimal representation."""
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _contract(path: Path, now: datetime) -> dict[str, Any]:
    contract = _mapping(json.loads(path.read_bytes()), name="contract")
    if _canonical_hash(contract, "contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise ValueError("contract embedded hash does not reconstruct")
    if contract.get("schema_version") != (
        "polymarket-retained-cross-event-rewards-contract-v1"
    ):
        raise ValueError("unsupported contract schema")
    frozen = _utc_datetime(contract.get("frozen_at_utc"))
    window = int(contract["capture"]["activation_window_minutes"])
    if frozen > now or now - frozen > timedelta(minutes=window):
        raise ValueError("frozen contract activation window expired")
    if path.resolve() != _root_path(str(contract["contract_path"])):
        raise ValueError("contract path mismatch")
    implementation = _root_path(str(contract["implementation"]["path"]))
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise ValueError("implementation hash mismatch")
    return contract


def _retained_market(
    contract: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    source = contract["retained_sources"][candidate["source_name"]]
    path = _root_path(str(source["path"]))
    raw = path.read_bytes()
    if _sha256(raw) != source["sha256"]:
        raise ValueError("retained Gamma source hash mismatch")
    event = _mapping(json.loads(raw), name="retained Gamma event")
    if event.get("slug") != candidate["event_slug"]:
        raise ValueError("retained event identity changed")
    rows = [
        _mapping(value, name="Gamma market")
        for value in _list(event.get("markets"), name="Gamma markets")
        if str(_mapping(value, name="Gamma market").get("id"))
        == candidate["gamma_market_id"]
    ]
    if len(rows) != 1:
        raise ValueError("exact retained Gamma market is absent or duplicated")
    row = rows[0]
    outcomes = [
        str(value) for value in _json_list(row.get("outcomes"), name="outcomes")
    ]
    tokens = [
        str(value) for value in _json_list(row.get("clobTokenIds"), name="tokens")
    ]
    condition_id = str(row.get("conditionId") or "").lower()
    if not (
        row.get("slug") == candidate["market_slug"]
        and row.get("question") == candidate["question"]
        and condition_id == candidate["condition_id"]
        and outcomes == candidate["outcomes"]
        and tokens == candidate["tokens"]
        and row.get("active") is True
        and row.get("closed") is False
        and row.get("acceptingOrders") is True
        and row.get("enableOrderBook") is True
    ):
        raise ValueError("retained exact market identity changed")
    minimum = _decimal(
        row.get("rewardsMinSize"), name="Gamma reward minimum", positive=True
    )
    spread = _decimal(
        row.get("rewardsMaxSpread"), name="Gamma reward spread", positive=True
    )
    if minimum != Decimal(candidate["reward_minimum_size_shares"]) or spread != Decimal(
        candidate["reward_maximum_spread_cents"]
    ):
        raise ValueError("retained reward settings changed")
    return {
        "condition_id": condition_id,
        "outcomes": outcomes,
        "tokens": tokens,
        "event_end": _utc_datetime(row.get("endDate"), end_of_date=True),
        "minimum": minimum,
        "spread": spread,
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
    if not rows:
        return {
            "exact_row_count": 0,
            "active_configuration_count": 0,
            "daily_rate_pUSD": Decimal("0"),
            "remaining_days": Decimal("0"),
            "maximum_remaining_pool_pUSD": Decimal("0"),
            "funded_active_configuration": False,
        }
    if len(rows) != 1:
        raise ValueError("exact reward response contains multiple rows")
    row = rows[0]
    if not (
        str(row.get("condition_id") or "").lower() == market["condition_id"]
        and row.get("market_slug") == candidate["market_slug"]
        and row.get("event_slug") == candidate["event_slug"]
        and row.get("question") == candidate["question"]
    ):
        raise ValueError("exact sponsored reward identity changed")
    reward_tokens = [
        _mapping(value, name="reward token")
        for value in _list(row.get("tokens"), name="reward tokens")
    ]
    if [str(value.get("token_id") or "") for value in reward_tokens] != market[
        "tokens"
    ] or [str(value.get("outcome") or "").casefold() for value in reward_tokens] != [
        value.casefold() for value in market["outcomes"]
    ]:
        raise ValueError("Gamma and reward token identity disagree")
    if _decimal(
        row.get("rewards_min_size"), name="reward minimum", positive=True
    ) != market["minimum"] or _decimal(
        row.get("rewards_max_spread"), name="reward spread", positive=True
    ) != market["spread"]:
        raise ValueError("Gamma and reward size or spread disagree")
    active: list[tuple[dict[str, Any], datetime]] = []
    for value in _list(row.get("rewards_config"), name="reward configurations"):
        config = _mapping(value, name="reward configuration")
        start = _utc_datetime(config.get("start_date"))
        end = _utc_datetime(config.get("end_date"), end_of_date=True)
        if start <= now < end:
            active.append((config, end))
    if not active:
        return {
            "exact_row_count": 1,
            "active_configuration_count": 0,
            "daily_rate_pUSD": Decimal("0"),
            "remaining_days": Decimal("0"),
            "maximum_remaining_pool_pUSD": Decimal("0"),
            "funded_active_configuration": False,
            "market_competitiveness": row.get("market_competitiveness"),
        }
    if len(active) != 1:
        raise ValueError("multiple active dated reward configurations")
    config, active_end = active[0]
    rate = _decimal(config.get("rate_per_day"), name="daily rate", positive=True)
    remaining_days = Decimal(
        str(max((min(active_end, market["event_end"]) - now).total_seconds(), 0))
    ) / Decimal("86400")
    return {
        "exact_row_count": 1,
        "active_configuration_count": 1,
        "daily_rate_pUSD": rate,
        "remaining_days": remaining_days,
        "maximum_remaining_pool_pUSD": rate * remaining_days,
        "funded_active_configuration": remaining_days > 0,
        "active_configuration": config,
        "market_competitiveness": row.get("market_competitiveness"),
    }


def _verify_retained_books(contract: dict[str, Any]) -> dict[str, Decimal]:
    source = contract["retained_sources"]["books"]
    path = _root_path(str(source["path"]))
    raw = path.read_bytes()
    if _sha256(raw) != source["sha256"]:
        raise ValueError("retained books hash mismatch")
    books = {
        str(row["asset_id"]): row
        for row in _list(json.loads(raw), name="retained books")
    }
    economics = contract["optimistic_rejection_bound"]
    quantity = Decimal(economics["quantity_shares_each_leg"])
    quotes: dict[str, Decimal] = {}
    for name, definition in economics["maker_quotes"].items():
        book = _mapping(books[definition["token_id"]], name="retained book")
        tick = Decimal(definition["tick_size"])
        asks = sorted(Decimal(str(row["price"])) for row in book.get("asks", []))
        bids = sorted(
            (Decimal(str(row["price"])) for row in book.get("bids", [])), reverse=True
        )
        if definition["construction"] == "best_bid_plus_one_tick":
            quote = bids[0] + tick
        elif definition["construction"] == "best_ask_minus_one_tick_when_no_bid":
            if bids:
                raise ValueError("retained no-bid construction no longer holds")
            quote = asks[0] - tick
        else:
            raise ValueError("unsupported maker quote construction")
        if quote != Decimal(definition["price_pUSD"]) or not (
            Decimal("0") < quote < asks[0]
        ):
            raise ValueError("retained maker quote does not reconstruct")
        quotes[name] = quote
    costs = {name: price * quantity for name, price in quotes.items()}
    floor = quantity * Decimal(economics["common_rule_floor_per_share_pUSD"])
    if sum(costs.values()) != Decimal(economics["both_fill_cost_pUSD"]):
        raise ValueError("both-fill cost does not reconstruct")
    if floor - sum(costs.values()) != Decimal(economics["both_fill_gross_pUSD"]):
        raise ValueError("both-fill gross does not reconstruct")
    maximum_orphan = max(costs.values())
    if maximum_orphan != Decimal(economics["maximum_one_leg_orphan_loss_pUSD"]):
        raise ValueError("orphan bound does not reconstruct")
    return {
        "maximum_orphan": maximum_orphan,
        "both_fill_gross": floor - sum(costs.values()),
    }


def run(*, contract_path: Path, output: Path, journal_dir: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    contract = _contract(contract_path, started)
    if output.exists() or journal_dir.exists():
        raise ValueError("one-use output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    journal_dir.parent.mkdir(parents=True, exist_ok=True)
    journal_dir.mkdir()
    retained_bound = _verify_retained_books(contract)
    http = requests.Session()
    results: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for candidate in contract["candidates"]:
        market = _retained_market(contract, candidate)
        raw, source = _request(
            http,
            method="GET",
            url=contract["request_contract"]["endpoint_pattern"].format(
                condition_id=market["condition_id"]
            ),
            params=contract["request_contract"]["params"],
            json_body=None,
            journal_dir=journal_dir,
            source_name=f"reward-{candidate['name']}",
            byte_ceiling=int(contract["request_contract"]["response_byte_ceiling"]),
        )
        results[candidate["name"]] = _reward(
            raw,
            market=market,
            candidate=candidate,
            now=started,
            terminal_cursor=contract["request_contract"]["terminal_cursor"],
        )
        sources[candidate["name"]] = source
    maximum_pool = sum(
        (row["maximum_remaining_pool_pUSD"] for row in results.values()),
        Decimal("0"),
    )
    passes = maximum_pool > retained_bound["maximum_orphan"]
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "exact_rewards": results,
        "optimistic_rejection_bound": {
            **contract["optimistic_rejection_bound"],
            "maximum_remaining_reward_pool_pUSD": maximum_pool,
            "strictly_exceeds_maximum_orphan_loss": passes,
        },
        "adjudication": {
            "status": (
                "source_only_reward_candidate_requires_fresh_books_competition_and_payout_proof"
                if passes
                else "rejected_before_fresh_books_accounts_credentials_orders_and_funds"
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
        },
        "authority": contract["authority"],
        "sources": {
            "reward_requests": sources,
            "implementation": contract["implementation"],
        },
    }
    artifact = _json_ready(artifact)
    artifact["result_sha256"] = _canonical_hash(artifact, "result_sha256")
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
            write_bytes_atomic(
                args.journal_dir / "terminal-failure.json",
                _canonical(
                    {
                        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "raw_responses_retained_before_validation": True,
                        "retry_permitted": False,
                    }
                )
                + b"\n",
            )
        raise
    print(
        json.dumps(
            {
                "status": result["adjudication"]["status"],
                "maximum_remaining_reward_pool_pUSD": result[
                    "optimistic_rejection_bound"
                ]["maximum_remaining_reward_pool_pUSD"],
                "payloads_printed": 0,
            },
            default=str,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
