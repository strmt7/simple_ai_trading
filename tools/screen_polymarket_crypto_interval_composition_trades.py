"""Screen retained aligned crypto intervals against one public trade batch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlencode

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)
from tools.screen_polymarket_exact_two_leg_package import _request


SCHEMA = "polymarket-crypto-interval-composition-trade-screen-result-v1"
_SLUG = re.compile(r"^(btc|eth|sol)-updown-(5m|15m)-(\d+)$")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="ascii"))


def _json_list(value: object, *, name: str) -> list[Any]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise RuntimeError(f"{name} must be a JSON list")
    return parsed


def _market_row(market: dict[str, Any], *, slug: str) -> dict[str, Any]:
    match = _SLUG.fullmatch(slug)
    if match is None:
        raise RuntimeError("selected interval slug is invalid")
    outcomes = _json_list(market.get("outcomes"), name="outcomes")
    tokens = _json_list(market.get("clobTokenIds"), name="tokens")
    if not (
        outcomes == ["Up", "Down"]
        and len(tokens) == 2
        and all(isinstance(token, str) and token for token in tokens)
        and market.get("closed") is True
        and market.get("enableOrderBook") is True
        and market.get("negRisk") is False
        and market.get("automaticallyResolved") is True
    ):
        raise RuntimeError("selected historical interval market state changed")
    event_id = market.get("events", [{}])[0].get("id")
    if not isinstance(event_id, str) or not event_id.isdigit():
        raise RuntimeError("selected market event ID is invalid")
    return {
        "asset": match.group(1).upper(),
        "duration": match.group(2),
        "start_epoch_seconds": int(match.group(3)),
        "slug": slug,
        "event_id": int(event_id),
        "condition_id": str(market["conditionId"]),
        "up_token_id": tokens[0],
        "down_token_id": tokens[1],
        "minimum_order_size": str(market.get("orderMinSize")),
        "fees_enabled": bool(market.get("feesEnabled")),
        "fee_schedule": market.get("feeSchedule"),
    }


def _market_identity(market: dict[str, Any]) -> dict[str, Any]:
    events = market.get("events")
    event_id = events[0].get("id") if isinstance(events, list) and events else None
    return {
        "slug": market.get("slug"),
        "condition_id": market.get("conditionId"),
        "outcomes": _json_list(market.get("outcomes"), name="outcomes"),
        "tokens": _json_list(market.get("clobTokenIds"), name="tokens"),
        "event_id": event_id,
        "resolution_source": market.get("resolutionSource"),
        "description": market.get("description"),
        "minimum_order_size": market.get("orderMinSize"),
        "fee_schedule": market.get("feeSchedule"),
    }


def _population(contract: dict[str, Any]) -> list[dict[str, Any]]:
    by_slug: dict[str, dict[str, Any]] = {}
    for source in contract["retained_market_sources"]:
        path = _root_path(str(source["path"]))
        raw = path.read_bytes()
        if _sha256(raw) != source["sha256"]:
            raise RuntimeError("retained market source hash mismatch")
        rows = json.loads(raw)
        if not isinstance(rows, list):
            raise RuntimeError("retained market source must be an array")
        for value in rows:
            if not isinstance(value, dict):
                raise RuntimeError("retained market row must be an object")
            slug = str(value.get("slug") or "")
            if _SLUG.fullmatch(slug):
                prior = by_slug.get(slug)
                if prior is not None:
                    if _market_identity(prior) != _market_identity(value):
                        raise RuntimeError(
                            "conflicting duplicate retained market identity"
                        )
                    if str(value.get("updatedAt") or "") > str(
                        prior.get("updatedAt") or ""
                    ):
                        by_slug[slug] = value
                else:
                    by_slug[slug] = value

    selected: list[dict[str, Any]] = []
    for start in contract["selected_start_epochs"]:
        for asset in contract["assets"]:
            prefix = str(asset).lower()
            slugs = [
                f"{prefix}-updown-5m-{start}",
                f"{prefix}-updown-5m-{start + 300}",
                f"{prefix}-updown-5m-{start + 600}",
                f"{prefix}-updown-15m-{start}",
            ]
            rows = [_market_row(by_slug[slug], slug=slug) for slug in slugs]
            if [row["start_epoch_seconds"] for row in rows] != [
                start,
                start + 300,
                start + 600,
                start,
            ]:
                raise RuntimeError("selected intervals do not exactly align")
            selected.extend(rows)
    if len(selected) != contract["decision"]["expected_market_count"]:
        raise RuntimeError("selected historical market population changed")
    if len({row["condition_id"] for row in selected}) != len(selected):
        raise RuntimeError("selected condition IDs are not unique")
    return selected


def _trade(value: object, allowed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("trade row must be an object")
    condition = str(value.get("conditionId") or "")
    market = allowed.get(condition)
    if market is None:
        raise RuntimeError("trade response escaped selected conditions")
    token = str(value.get("asset") or "")
    if token not in {market["up_token_id"], market["down_token_id"]}:
        raise RuntimeError("trade response token is outside selected market")
    side = value.get("side")
    if side not in {"BUY", "SELL"}:
        raise RuntimeError("trade side is invalid")
    timestamp = int(value["timestamp"])
    price = Decimal(str(value["price"]))
    size = Decimal(str(value["size"]))
    if not (
        timestamp > 0
        and price.is_finite()
        and Decimal("0") <= price <= Decimal("1")
        and size.is_finite()
        and size > 0
    ):
        raise RuntimeError("trade numeric field is invalid")
    return {
        "condition_id": condition,
        "token_id": token,
        "side": side,
        "timestamp": timestamp,
        "price_pUSD": format(price, "f"),
        "size_shares": format(size, "f"),
        "transaction_hash": str(value.get("transactionHash") or ""),
    }


def _package_tokens(rows: list[dict[str, Any]], direction: str) -> list[str]:
    shorts, long = rows[:3], rows[3]
    if direction == "up_chain":
        return [row["down_token_id"] for row in shorts] + [long["up_token_id"]]
    if direction == "down_chain":
        return [row["up_token_id"] for row in shorts] + [long["down_token_id"]]
    raise RuntimeError("unknown package direction")


def _synchronized_windows(
    trades: list[dict[str, Any]], tokens: list[str], minimum_size: Decimal
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in trades
        if row["side"] == "BUY"
        and row["token_id"] in tokens
        and Decimal(row["size_shares"]) >= minimum_size
    ]
    timestamps = sorted({row["timestamp"] for row in eligible})
    observations: dict[tuple[tuple[str, int, str, str], ...], dict[str, Any]] = {}
    for start in timestamps:
        legs: list[dict[str, Any]] = []
        for token in tokens:
            choices = [
                row
                for row in eligible
                if row["token_id"] == token and start <= row["timestamp"] <= start + 1
            ]
            if not choices:
                break
            legs.append(
                min(
                    choices,
                    key=lambda row: (
                        Decimal(row["price_pUSD"]),
                        row["timestamp"],
                        row["transaction_hash"],
                    ),
                )
            )
        if len(legs) != len(tokens):
            continue
        key = tuple(
            sorted(
                (
                    row["token_id"],
                    row["timestamp"],
                    row["price_pUSD"],
                    row["transaction_hash"],
                )
                for row in legs
            )
        )
        total = sum((Decimal(row["price_pUSD"]) for row in legs), Decimal("0"))
        observations[key] = {
            "minimum_timestamp": min(row["timestamp"] for row in legs),
            "maximum_timestamp": max(row["timestamp"] for row in legs),
            "timestamp_skew_seconds": max(row["timestamp"] for row in legs)
            - min(row["timestamp"] for row in legs),
            "displayed_trade_price_sum_pUSD": format(total, "f"),
            "optimistic_gross_headroom_pUSD": format(Decimal("1") - total, "f"),
            "passes_strict_gross_signal_gate": total < Decimal("1"),
            "legs": legs,
        }
    return sorted(
        observations.values(),
        key=lambda row: (
            Decimal(row["displayed_trade_price_sum_pUSD"]),
            row["minimum_timestamp"],
        ),
    )


def run(contract_path: Path) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = _load(contract_path)
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    implementation = _root_path(str(contract["implementation"]["path"]))
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    population = _population(contract)
    conditions = sorted(row["condition_id"] for row in population)
    params = {
        "limit": "10000",
        "offset": "0",
        "takerOnly": "true",
        "market": ",".join(conditions),
    }
    url = f"{contract['request']['base_url']}?{urlencode(params, safe=',')}"
    request_boundary = {
        "base_url": contract["request"]["base_url"],
        "limit": params["limit"],
        "offset": params["offset"],
        "takerOnly": params["takerOnly"],
        "market_condition_count": len(conditions),
        "market_csv_sha256": _sha256(params["market"].encode("ascii")),
    }
    if request_boundary != contract["request"]["expected"]:
        raise RuntimeError("trade request boundary changed")

    outputs = {key: _root_path(value) for key, value in contract["outputs"].items()}
    for path in outputs.values():
        if path.exists():
            raise RuntimeError(f"one-use output already exists: {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
    raw, receipt = _request(
        method="GET",
        url=url,
        body=b"",
        name="crypto-interval-composition-selected-market-trades",
        raw_path=outputs["raw_path"],
        raw_relative_path=contract["outputs"]["raw_path"],
        journal_path=outputs["journal_path"],
    )
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise RuntimeError("trade response must be an array")
    allowed = {row["condition_id"]: row for row in population}
    trades = [_trade(value, allowed) for value in payload]
    response_complete = len(trades) < 10000

    package_rows: list[dict[str, Any]] = []
    if response_complete:
        by_key = {
            (row["asset"], row["start_epoch_seconds"], row["duration"]): row
            for row in population
        }
        for start in contract["selected_start_epochs"]:
            for asset in contract["assets"]:
                rows = [
                    by_key[(asset, start, "5m")],
                    by_key[(asset, start + 300, "5m")],
                    by_key[(asset, start + 600, "5m")],
                    by_key[(asset, start, "15m")],
                ]
                minimum_size = max(Decimal(row["minimum_order_size"]) for row in rows)
                for direction in ("up_chain", "down_chain"):
                    windows = _synchronized_windows(
                        trades, _package_tokens(rows, direction), minimum_size
                    )
                    package_rows.append(
                        {
                            "asset": asset,
                            "start_epoch_seconds": start,
                            "direction": direction,
                            "minimum_size_shares": format(minimum_size, "f"),
                            "synchronized_window_count": len(windows),
                            "strict_sub_floor_signal_count": sum(
                                row["passes_strict_gross_signal_gate"]
                                for row in windows
                            ),
                            "best_synchronized_window": windows[0] if windows else None,
                        }
                    )

    strict_signals = sum(row["strict_sub_floor_signal_count"] for row in package_rows)
    synchronized = sum(row["synchronized_window_count"] for row in package_rows)
    best = min(
        (
            {
                "asset": row["asset"],
                "start_epoch_seconds": row["start_epoch_seconds"],
                "direction": row["direction"],
                **row["best_synchronized_window"],
            }
            for row in package_rows
            if row["best_synchronized_window"] is not None
        ),
        key=lambda row: Decimal(row["displayed_trade_price_sum_pUSD"]),
        default=None,
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "capture": {
            "receipt": receipt,
            "response_row_count": len(trades),
            "response_complete_below_limit": response_complete,
        },
        "population": {
            "market_count": len(population),
            "set_count": len(population) // 4,
            "conditions": conditions,
        },
        "screen": {
            "package_count": len(package_rows),
            "synchronization_window_seconds": 1,
            "package_rows": package_rows,
            "synchronized_window_count": synchronized,
            "strict_sub_floor_signal_count": strict_signals,
            "best_synchronized_window": best,
        },
        "adjudication": {
            "status": (
                "incomplete_response_hit_limit_no_retry_or_claim"
                if not response_complete
                else "historical_trade_signal_requires_prospective_exact_depth_capture"
                if strict_signals
                else "complete_selected_history_has_no_strict_synchronized_trade_signal"
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "historical_trades_are_not_atomic_owned_execution": True,
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
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.contract)
    print(
        json.dumps(
            {
                "response_rows": result["capture"]["response_row_count"],
                "complete": result["capture"]["response_complete_below_limit"],
                "synchronized_windows": result["screen"]["synchronized_window_count"],
                "strict_signals": result["screen"]["strict_sub_floor_signal_count"],
                "best_sum": None
                if result["screen"]["best_synchronized_window"] is None
                else result["screen"]["best_synchronized_window"][
                    "displayed_trade_price_sum_pUSD"
                ],
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
