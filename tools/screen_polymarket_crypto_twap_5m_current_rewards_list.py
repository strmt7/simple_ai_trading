"""Join frozen Polymarket five-minute markets to the public current-rewards list."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Mapping

import requests

import screen_polymarket_crypto_twap_liquidity_rewards as prior
from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs/model-research/polymarket/crypto-twap-5m-current-rewards-list-join-contract-v1-2026-08-27.json"
)


def _canonical(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_contract(now: datetime) -> tuple[dict[str, object], str]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="ascii"))
    claimed = str(contract.get("result_sha256") or "")
    body = dict(contract)
    body.pop("result_sha256", None)
    if claimed != _sha256(_canonical(body).encode("ascii")):
        raise ValueError("contract embedded hash does not reconstruct")
    capture = prior._mapping(contract.get("capture"), name="capture")
    lower = datetime.fromisoformat(
        str(capture["activation_not_before_utc"]).replace("Z", "+00:00")
    )
    upper = datetime.fromisoformat(
        str(capture["activation_not_after_utc"]).replace("Z", "+00:00")
    )
    if not lower <= now <= upper:
        raise ValueError(f"outside frozen activation window: {now.isoformat()}")
    return contract, _sha256(CONTRACT_PATH.read_bytes())


def _epoch_ms(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be epoch milliseconds")
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        return int(value)
    if isinstance(value, str):
        try:
            return int(
                datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
                * 1000
            )
        except ValueError as exc:
            raise ValueError(f"{name} must be epoch milliseconds or ISO time") from exc
    raise ValueError(f"{name} must be epoch milliseconds or ISO time")


def _current_reward(
    row: Mapping[str, object], market: Mapping[str, object], *, observed_ms: int
) -> dict[str, object]:
    if str(row.get("condition_id") or "").lower() != market["condition_id"]:
        raise ValueError("current reward condition identity changed")
    size = prior._decimal(
        row.get("rewards_min_size"), name="reward minimum size", positive=True
    )
    spread = prior._decimal(
        row.get("rewards_max_spread"), name="reward maximum spread", positive=True
    )
    gamma = prior._mapping(market["gamma_payload"], name="Gamma payload")
    if size != prior._decimal(
        gamma.get("rewardsMinSize"), name="Gamma reward size", positive=True
    ):
        raise ValueError("Gamma and current-rewards minimum sizes disagree")
    if spread != prior._decimal(
        gamma.get("rewardsMaxSpread"), name="Gamma reward spread", positive=True
    ):
        raise ValueError("Gamma and current-rewards maximum spreads disagree")
    if size < prior._decimal(market["minimum_order_size"], name="minimum order size"):
        raise ValueError("reward size is below the executable order minimum")

    active: list[dict[str, object]] = []
    for value in prior._list(row.get("rewards_config"), name="reward configurations"):
        config = dict(prior._mapping(value, name="reward configuration"))
        start_ms = _epoch_ms(config.get("start_date"), name="reward start")
        raw_end = config.get("end_date")
        end_ms = None if raw_end in (None, "") else _epoch_ms(raw_end, name="reward end")
        rate = prior._decimal(
            config.get("rate_per_day"), name="configuration daily rate", positive=True
        )
        if start_ms <= observed_ms and (end_ms is None or observed_ms <= end_ms):
            active.append(config)
            config["validated_rate_per_day"] = str(rate)
    configured_daily_rate = sum(
        (
            prior._decimal(
                config.get("rate_per_day"),
                name="active configuration daily rate",
                positive=True,
            )
            for config in active
        ),
        Decimal("0"),
    )
    total_daily_rate = prior._decimal(
        row.get("total_daily_rate"), name="total current daily rate", positive=True
    )
    if not active or configured_daily_rate <= 0:
        raise ValueError("no positive active dated reward configuration")
    if configured_daily_rate != total_daily_rate:
        raise ValueError("active configuration sum and total current daily rate disagree")
    return {
        "minimum_size": size,
        "maximum_spread": spread,
        "daily_rate": total_daily_rate,
        "active": active,
        "current_reward_row": dict(row),
    }


def _terminal_artifact(
    *,
    started: datetime,
    contract: Mapping[str, object],
    contract_file_sha: str,
    gamma_source: Mapping[str, object],
    reward_sources: list[Mapping[str, object]],
    page_summaries: list[Mapping[str, object]],
    markets: Mapping[str, Mapping[str, object]],
    reason: str,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "schema_version": "polymarket-crypto-twap-5m-current-rewards-list-join-v1",
        "started_at_utc": started.isoformat().replace("+00:00", "Z"),
        "completed_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "market_identities": [
            {
                "slug": slug,
                "asset": market["asset"],
                "condition_id": market["condition_id"],
            }
            for slug, market in markets.items()
        ],
        "current_rewards_pages": page_summaries,
        "verdict": {
            "status": "rejected_without_resampling",
            "reason": reason,
            "accepted_edge": False,
            "profitability_claim": False,
            "trading_authority": False,
            "books_requested": False,
            "publicly_proven_reward_payout_floor_pUSD": "0",
        },
        "authority": {
            "public_read_only": True,
            "credentials_used": False,
            "orders_or_cancellations": 0,
            "funded_actions": 0,
        },
        "sources": {
            "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "contract_file_sha256": contract_file_sha,
            "contract_result_sha256": contract["result_sha256"],
            "gamma_request": gamma_source,
            "current_rewards_requests": reward_sources,
            "tool_sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    artifact["result_sha256"] = _sha256(_canonical(artifact).encode("ascii"))
    return artifact


def run(*, journal_dir: Path, session: requests.Session | None = None) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    contract, contract_file_sha = _load_contract(started)
    capture = prior._mapping(contract["capture"], name="capture")
    request_contract = prior._mapping(contract["request_contract"], name="request contract")
    expected: dict[str, tuple[str, int, int]] = {}
    for value in prior._list(capture["markets"], name="markets"):
        row = prior._mapping(value, name="market contract")
        expected[str(row["slug"])] = (
            str(row["asset"]),
            int(row["start_epoch_seconds"]) * 1000,
            int(capture["duration_ms"]),
        )

    http = session or requests.Session()
    gamma_raw, gamma_source = prior._request(
        http,
        "GET",
        "https://gamma-api.polymarket.com/markets",
        journal_dir=journal_dir,
        source_name="01-gamma-seven-markets",
        params=[*(("slug", slug) for slug in expected), ("closed", "false")],
    )
    markets = prior._market_rows(gamma_raw, expected)

    current_rows: list[dict[str, object]] = []
    reward_sources: list[Mapping[str, object]] = []
    page_summaries: list[Mapping[str, object]] = []
    cursor: str | None = None
    observed_cursors: set[str] = set()
    end_cursor = str(request_contract["current_rewards_end_cursor"])
    maximum_pages = int(request_contract["current_rewards_maximum_pages"])
    complete = False
    for page_number in range(1, maximum_pages + 1):
        params = {"sponsored": "true"}
        if cursor is not None:
            params["next_cursor"] = cursor
        raw, source = prior._request(
            http,
            "GET",
            "https://clob.polymarket.com/rewards/markets/current",
            journal_dir=journal_dir,
            source_name=f"{page_number + 1:02d}-current-rewards-page-{page_number:02d}",
            params=params,
        )
        reward_sources.append(source)
        payload = prior._mapping(raw, name="current rewards page")
        rows = [
            dict(prior._mapping(value, name="current reward row"))
            for value in prior._list(payload.get("data"), name="current rewards data")
        ]
        next_cursor = str(payload.get("next_cursor") or "")
        page_summaries.append(
            {
                "page": page_number,
                "row_count": len(rows),
                "declared_count": payload.get("count"),
                "declared_limit": payload.get("limit"),
                "next_cursor": next_cursor,
                "target_condition_matches": sum(
                    str(row.get("condition_id") or "").lower()
                    in {str(market["condition_id"]) for market in markets.values()}
                    for row in rows
                ),
            }
        )
        current_rows.extend(rows)
        if next_cursor == end_cursor:
            complete = True
            break
        if not next_cursor or next_cursor in observed_cursors:
            return _terminal_artifact(
                started=started,
                contract=contract,
                contract_file_sha=contract_file_sha,
                gamma_source=gamma_source,
                reward_sources=reward_sources,
                page_summaries=page_summaries,
                markets=markets,
                reason="current_rewards_cursor_missing_or_repeated",
            )
        observed_cursors.add(next_cursor)
        cursor = next_cursor

    if not complete:
        return _terminal_artifact(
            started=started,
            contract=contract,
            contract_file_sha=contract_file_sha,
            gamma_source=gamma_source,
            reward_sources=reward_sources,
            page_summaries=page_summaries,
            markets=markets,
            reason="current_rewards_page_ceiling_reached_before_terminal_cursor",
        )

    rows_by_condition: dict[str, list[dict[str, object]]] = {}
    for row in current_rows:
        rows_by_condition.setdefault(str(row.get("condition_id") or "").lower(), []).append(
            row
        )
    matched = {
        slug: rows_by_condition.get(str(market["condition_id"]), [])
        for slug, market in markets.items()
    }
    if any(len(rows) != 1 for rows in matched.values()):
        counts = ",".join(f"{slug}:{len(rows)}" for slug, rows in matched.items())
        return _terminal_artifact(
            started=started,
            contract=contract,
            contract_file_sha=contract_file_sha,
            gamma_source=gamma_source,
            reward_sources=reward_sources,
            page_summaries=page_summaries,
            markets=markets,
            reason=f"current_rewards_exact_join_not_one_to_one:{counts}",
        )

    observed_ms = int(gamma_source["received_after_ms"])
    rewards = {
        slug: _current_reward(rows[0], markets[slug], observed_ms=observed_ms)
        for slug, rows in matched.items()
    }
    tokens = [
        str(token)
        for market in markets.values()
        for token in prior._list(market["tokens"], name="tokens")
    ]
    books_raw, books_source = prior._request(
        http,
        "POST",
        "https://clob.polymarket.com/books",
        journal_dir=journal_dir,
        source_name=f"{len(reward_sources) + 2:02d}-fourteen-token-books",
        json_body=[{"token_id": token} for token in tokens],
    )
    received_ms = int(books_source["received_after_ms"])
    books = {
        str(prior._mapping(value, name="book")["asset_id"]): prior._mapping(
            value, name="book"
        )
        for value in prior._list(books_raw, name="books")
    }
    if set(books) != set(tokens) or len(books) != len(tokens):
        raise ValueError("batch books did not return the fourteen exact tokens")
    timestamps: list[int] = []
    for token, book in books.items():
        market = next(value for value in markets.values() if token in value["tokens"])
        if prior._decimal(
            book.get("tick_size"), name="book tick", positive=True
        ) != market["tick_size"] or prior._decimal(
            book.get("min_order_size"), name="book min", positive=True
        ) != market["minimum_order_size"]:
            raise ValueError("book execution parameters disagree with Gamma")
        prior._levels(book, "bids")
        prior._levels(book, "asks")
        timestamps.append(int(str(book["timestamp"])))
    receipt_span = received_ms - int(gamma_source["requested_before_ms"])
    age = received_ms - min(timestamps)
    skew = max(timestamps) - min(timestamps)
    remaining = min(int(market["end_ms"]) for market in markets.values()) - received_ms
    fresh = (
        0 <= age <= int(capture["book_max_event_age_ms"])
        and skew <= int(capture["book_max_timestamp_skew_ms"])
        and receipt_span <= int(capture["receipt_max_span_ms"])
        and remaining >= int(capture["minimum_remaining_market_ms"])
    )
    results: list[dict[str, object]] = []
    for slug, market in markets.items():
        reward = rewards[slug]
        results.append(
            {
                "slug": slug,
                "asset": market["asset"],
                "condition_id": market["condition_id"],
                "tokens": market["tokens"],
                "fee_schedule": market["fee_schedule"],
                "reward": {
                    "minimum_size": str(reward["minimum_size"]),
                    "maximum_spread_cents": str(reward["maximum_spread"]),
                    "daily_rate_pUSD": str(reward["daily_rate"]),
                    "active_configurations": reward["active"],
                },
                "diagnostic": prior._diagnostic(market, reward, books),
            }
        )
    candidates = (
        [
            str(row["slug"])
            for row in results
            if prior._mapping(row["diagnostic"], name="diagnostic")["status"]
            == "prospective_capture_candidate_not_an_edge"
        ]
        if fresh
        else []
    )
    artifact: dict[str, object] = {
        "schema_version": "polymarket-crypto-twap-5m-current-rewards-list-join-v1",
        "started_at_utc": started.isoformat().replace("+00:00", "Z"),
        "completed_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "capture": {
            "receipt_span_ms": receipt_span,
            "oldest_book_age_ms": age,
            "book_timestamp_skew_ms": skew,
            "remaining_market_ms": remaining,
            "freshness_passed": fresh,
        },
        "current_rewards_pages": page_summaries,
        "market_results": results,
        "verdict": {
            "status": (
                "prospective_capture_design_eligible_not_an_edge"
                if candidates
                else "rejected_without_resampling"
            ),
            "candidate_slugs": candidates,
            "accepted_edge": False,
            "profitability_claim": False,
            "trading_authority": False,
            "books_requested": True,
            "publicly_proven_reward_payout_floor_pUSD": "0",
        },
        "authority": {
            "public_read_only": True,
            "credentials_used": False,
            "orders_or_cancellations": 0,
            "funded_actions": 0,
        },
        "sources": {
            "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "contract_file_sha256": contract_file_sha,
            "contract_result_sha256": contract["result_sha256"],
            "gamma_request": gamma_source,
            "current_rewards_requests": reward_sources,
            "books_request": books_source,
            "tool_sha256": _sha256(Path(__file__).read_bytes()),
        },
        "limitations": [
            "A public top-midpoint score is conditional because maker identities queue order persistence competition random sampling and final epoch normalization are unavailable.",
            "A one-sided fill is directional orphan exposure.",
            "A one-time public snapshot cannot prove realized reward after-cost recurrence.",
        ],
    }
    artifact["result_sha256"] = _sha256(_canonical(artifact).encode("ascii"))
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(journal_dir=args.journal_dir)
    except Exception as exc:
        failure = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "failed_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "raw_responses_retained_before_validation": True,
        }
        write_bytes_atomic(
            args.journal_dir / "terminal-failure.json",
            (_canonical(failure) + "\n").encode("ascii"),
        )
        raise
    write_bytes_atomic(args.output, (_canonical(result) + "\n").encode("ascii"))
    print(json.dumps(result["verdict"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
